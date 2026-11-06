"""Korak 06 — okolje: CO2 po poteh v 30 letih (obratovalni po urnem potrošnem EF SI 2025 × trajektorija razogljičenja + goriva;
vgrajeni CO2 ukrepov in zamenjav), CO2 povračilna doba ovoja in PV, CO2 na EUR spodbude.

Vhodi: cache/03_fond_letno.parquet, tabele/03_energija_variante.csv (CO2_el_urni_kg po variantah, korak 03),
       poti in spodbude iz 04_ekonomika.py, parametri.yaml: okolje (viri v podatki/06_viri_CO2.md).
Izhodi: tabele/06_okolje_reprezentativne.csv, tabele/06_okolje_fond.csv, cache/06_co2_poti_fond.parquet, 06_povzetek.md
"""
from __future__ import annotations

import importlib.util
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("ek", HERE / "04_ekonomika.py")
EKM = importlib.util.module_from_spec(spec)
spec.loader.exec_module(EKM)
PAR = EKM.PAR
OK, EK = PAR["okolje"], PAR["ekonomika"]
VG = OK["vgrajeno"]
N, L0 = EKM.N, EK["leto0"]
TAB, CACHE = HERE / "tabele", HERE / "cache"


def traj(t):
    """Faktor EF elektrike v letu t (1..N) glede na 2025, linearna interpolacija."""
    yrs = sorted(OK["EF_trajektorija"])
    return float(np.interp(L0 + t - 1, yrs, [OK["EF_trajektorija"][y] for y in yrs]))


def op_co2(w, var, t):
    """Obratovalni CO2 [kg] variante v letu t."""
    def one(v, ef_fuel=0.0):
        x = w[f"{v}|CO2_el_urni_kg"] * traj(t) + w[f"{v}|gorivo_kWh"] * ef_fuel
        if OK["odstej_izognjene_oddaje"]:
            x = x - w[f"{v}|CO2_izogib_oddaja_kg"] * traj(t)
        return x.to_numpy()
    if var == "V0":
        out = np.where(w["V0"] == "V0b", one("V0b", OK["EF_goriva_kg_kWh"]["biomasa"]), one("V0a", OK["EF_goriva_kg_kWh"]["ELKO"]))
        if "V0c|uvoz_kWh" in w.columns:
            out = np.where(w["V0"] == "V0c", one("V0c", OK["EF_goriva_kg_kWh"]["plin"]), out)
        return out
    return one(var)


def embodied(w, item, arg):
    """Vgrajeni CO2 [kg] dogodka na poti."""
    kW = {"e": w["V1|TC_kW"].to_numpy(), "FSO": w["V4|TC_kW"].to_numpy(), "GLOB": w["V7|TC_kW"].to_numpy(), "r": w["V2r|TC_kW"].to_numpy() if "V2r|TC_kW" in w.columns else w["V1|TC_kW"].to_numpy()}
    if item in ("TC", "TCz"):
        return VG["TC_kg_enota"] + VG["TC_kg_kW"] * kW[arg] + VG["hladivo_kg_CO2e_zivljenjska_doba"]
    if item in ("PV", "PVB"):
        k = w[f"{arg}|PV_kWp"].to_numpy()
        return (VG["PV_kg_kWp"] + VG["razsmernik_kg_kW"]) * k
    if item == "RAZS":
        return VG["razsmernik_kg_kW"] * w[f"{arg}|PV_kWp"].to_numpy()
    if item in ("BAT", "BATz"):
        return VG["baterija_kg_kWh"] * w["V6|baterija_kWh"].to_numpy()
    if item in ("FSO", "GLOB"):
        A = w["V4|A"].to_numpy()
        g = np.stack([[EKM.geom(a)[k] for k in ("stena", "streha", "okna", "tla")] for a in w["arhetip"]])
        dl = EKM.ovoj_delez(w, item)                             # krog 3: le manjkajoči ukrepi (fasada, streha, okna, tla)
        e = A * (dl["stena"] * g[:, 0] * VG["fasada_kg_m2"] + dl["streha"] * g[:, 1] * VG["streha_kg_m2"] + dl["okna"] * g[:, 2] * VG["okna_kg_m2"]
                 + dl["tla"] * g[:, 3] * VG["tla_kg_m2"])
        if item == "GLOB":
            e = e + A * VG["MVHR_kg_m2"]
        return e
    if item == "MVHRz":
        return w["V4|A"].to_numpy() * VG["MVHR_kg_m2"]
    return np.zeros(len(w))


def co2_paths(w):
    rows = {}
    for p, d in EKM.POTI.items():
        op = sum(op_co2(w, EKM.leto_varianta(p, t), t) for t in range(1, N + 1))
        emb = sum(embodied(w, item, arg) for (y, item, arg) in d["ev"])
        rows[(p, "obratovalni_t")] = op / 1000
        rows[(p, "vgrajeni_t")] = emb / 1000
        rows[(p, "skupaj_t")] = (op + emb) / 1000
        rows[(p, "leto1_t")] = op_co2(w, EKM.leto_varianta(p, 1), 1) / 1000
    df = pd.DataFrame(rows, index=w.index)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def summarise(w, c, sp_nom):
    out = pd.DataFrame(index=w.index)
    for p in EKM.POTI:
        out[f"{p} skupaj [t]"] = c[(p, "skupaj_t")]
    for a, b in (("P3", "P1"), ("P4", "P1"), ("P6", "P1"), ("P1", "P0")):
        out[f"{a} − {b} [t]"] = c[(a, "skupaj_t")] - c[(b, "skupaj_t")]
    # CO2 povračilna doba: vgrajeni CO2 ukrepa / letni obratovalni prihranek (leto 1, EF 2025)
    emb_env = embodied(w, "FSO", None) / 1000
    sav_env = (w["V1|CO2_el_urni_kg"] - w["V4|CO2_el_urni_kg"]).to_numpy() / 1000
    out["CO2 povračilna doba ovoja [a]"] = np.where(sav_env > 0, emb_env / np.maximum(sav_env, 1e-9), np.inf)
    sav_pv = (w["V1|CO2_el_urni_kg"] - w["V2|CO2_el_urni_kg"]).to_numpy() / 1000
    out["CO2 povračilna doba PV (V1→V2) [a]"] = np.where(sav_pv > 0, (embodied(w, "PV", "V2") / 1000) / np.maximum(sav_pv, 1e-9), np.inf)
    for p in ("P1", "P3", "P4", "P6"):
        out[f"kg CO2 izogib / EUR spodbude {p}"] = 1000 * (c[("P0", "skupaj_t")] - c[(p, "skupaj_t")]) / sp_nom[p].replace(0, np.nan)
    out["Qr"], out["arhetip"] = w["Qr"].values, w["arhetip"].values
    return out


def main():
    t0 = time.time()
    if any(v is None for v in VG.values()) or OK["EF_goriva_kg_kWh"]["ELKO"] is None:
        raise SystemExit("parametri.yaml: okolje.vgrajeno / EF_goriva še niso izpolnjeni (podatki/06_viri_CO2.md)")
    md = [f"# 06 — Okolje: CO2 po poteh v {N} letih ({time.strftime('%d. %m. %Y %H:%M')})", "",
          f"Obratovalni CO2 = urni uvoz × potrošni EF SI 2025 (energy-charts, korak 05a) × trajektorija {OK['EF_trajektorija']} "
          f"+ goriva (ELKO {OK['EF_goriva_kg_kWh']['ELKO']} kg/kWh, biomasa {OK['EF_goriva_kg_kWh']['biomasa']} — le CH4 + N2O). Vgrajeni CO2 ukrepov in zamenjav "
          "(podatki/06_viri_CO2.md). Izognjene emisije oddaje PV " + ("odštete." if OK["odstej_izognjene_oddaje"] else "niso odštete (konzervativno)."), ""]
    for src in ("repr", "fond"):
        w = EKM.load_units(src)
        c = co2_paths(w)
        sp_nom, _ = EKM.spodbude_poti(w)
        s = summarise(w, c, sp_nom)
        if src == "repr":
            s.round(2).to_csv(TAB / "06_okolje_reprezentativne.csv", sep=";", encoding="utf-8-sig")
            md += ["## Reprezentativne hiše", "", s.round(1).reset_index().to_string(index=False), ""]
        else:
            flat = c.copy()
            flat.columns = ["|".join(x) for x in flat.columns]
            flat.reset_index().to_parquet(CACHE / "06_co2_poti_fond.parquet", index=False)
            agg = s.replace(np.inf, np.nan).groupby("Qr").median(numeric_only=True).T
            tot = pd.DataFrame({"fond skupaj [kt CO2e, 30 let]": [c[(p, "skupaj_t")].sum() / 1000 for p in EKM.POTI]}, index=list(EKM.POTI))
            agg.round(2).to_csv(TAB / "06_okolje_fond.csv", sep=";", encoding="utf-8-sig")
            share = pd.DataFrame({"delež P3 < P1 [%]": [100 * (s.loc[s["Qr"] == q, "P3 − P1 [t]"] < 0).mean() for q in ("Q1", "Q2")]}, index=["Q1", "Q2"])
            md += ["## Fond — mediane po razredu", "", agg.round(1).to_string(), "", "Vsota fonda:", "", tot.round(1).to_string(), "",
                   share.round(1).to_string(), ""]
    md.append(f"Trajanje {time.time() - t0:.0f} s.")
    (HERE / "06_povzetek.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
