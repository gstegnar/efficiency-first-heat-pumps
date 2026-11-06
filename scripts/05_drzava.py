"""Korak 05 — država: spodbude, omrežje (sistemska konica, zimski uvoz, konica hiše), javnofinančni NPV (spodbude + omrežje + CO2).

Vhodi: 04_ekonomika.py (poti, enote, spodbude), 06_okolje.py (obratovalni in vgrajeni CO2), cache/05a_sistem_SI.parquet
       (ure sistemske konice), parametri.yaml: ekonomika, okolje, drzava.
Javni NPV (diskont javni, realno) = SV(spodbude) + strošek ojačitve omrežja (Δ prispevek h konici proti P0 × EUR/kW)
                                    + SV(CO2 × senčna cena) — manj je bolje.
Izhodi: tabele/05_drzava_reprezentativne.csv, tabele/05_drzava_fond.csv, tabele/05_omrezje_fond.csv, 05_povzetek.md
"""
from __future__ import annotations

import importlib.util
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def _mod(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


EKM = _mod("ek", "04_ekonomika.py")
OKM = _mod("ok", "06_okolje.py")
PAR = EKM.PAR
EK, OK, DR = PAR["ekonomika"], PAR["okolje"], PAR["drzava"]
N, L0 = EKM.N, EK["leto0"]
r = EK["diskont_javni"]
TAB = HERE / "tabele"
POTI = ["P0", "P1", "P1b", "P2", "P3", "P3b", "P4", "P5", "P6", "P1r"]


def cena_co2(t, flat=None):
    if flat is not None:
        return flat
    yrs = sorted(OK["cena_CO2_EUR_t"])
    return float(np.interp(L0 + t - 1, yrs, [OK["cena_CO2_EUR_t"][y] for y in yrs]))


def drzava(w, flat_co2=None):
    sp_nom, sp_pv = EKM.spodbude_poti(w, r=r)
    res = {}
    for p in POTI:
        v1, v20 = EKM.leto_varianta(p, 1), EKM.leto_varianta(p, 20)
        g = lambda v, q: (w[f"{v}|{q}"] if v != "V0" else np.select([w["V0"] == "V0b", w["V0"] == "V0c"],  # noqa: E731
                                                                    [w[f"V0b|{q}"], w.get(f"V0c|{q}", w[f"V0a|{q}"])], w[f"V0a|{q}"]))
        kon20 = np.asarray(g(v20, "uvoz_konica_sis_kW"), float)
        kon0 = np.asarray(g("V0", "uvoz_konica_sis_kW"), float)
        omr = np.maximum(kon20 - kon0, 0) * DR["ojacitev_EUR_kW"]
        co2 = sum(OKM.op_co2(w, EKM.leto_varianta(p, t), t) / 1000 * cena_co2(t, flat_co2) * (1 + r) ** (-t) for t in range(1, N + 1))
        co2 += sum(OKM.embodied(w, item, arg) / 1000 * cena_co2(max(y, 1), flat_co2) * (1 + r) ** (-y) for (y, item, arg) in EKM.POTI[p]["ev"])
        el30 = sum(np.asarray(g(EKM.leto_varianta(p, t), "el_skupaj_kWh"), float) for t in range(1, N + 1))
        fu30 = sum(np.asarray(g(EKM.leto_varianta(p, t), "gorivo_kWh"), float) for t in range(1, N + 1))
        res[p] = pd.DataFrame({"spodbude_nom": sp_nom[p], "spodbude_SV": sp_pv[p], "omrezje_EUR": omr, "CO2_SV_EUR": co2,
                               "javni_NPV": sp_pv[p] + omr + co2, "konica_sis_kW_leto1": np.asarray(g(v1, "uvoz_konica_sis_kW"), float),
                               "konica_sis_kW_leto20": kon20, "konica_hise_kW_leto20": np.asarray(g(v20, "uvoz_konica_kW"), float),
                               "zimski_uvoz_kWh_leto20": np.asarray(g(v20, "uvoz_zima_kWh"), float),
                               "koncna_energija_30_MWh": (el30 + fu30) / 1000}, index=w.index)
    return res


def main():
    t0 = time.time()
    md = [f"# 05 — Država: spodbude, omrežje, javni NPV ({time.strftime('%d. %m. %Y %H:%M')})", "",
          f"Javni diskont {r:.0%} realno; senčna cena CO2 EIB {OK['cena_CO2_EUR_t']} EUR/t (občutljivost UBA {OK['cena_CO2_obcutljivost_EUR_t']}); "
          f"ojačitev omrežja {DR['ojacitev_EUR_kW']} EUR na kW prispevka h sistemski konici nad P0 (PREVERI). "
          "Sistemska konica = povprečni uvoz hiše v 100 urah najvišjega odjema SI 2025.", ""]
    for src in ("repr", "fond"):
        w = EKM.load_units(src)
        res = drzava(w)
        res_s = drzava(w, flat_co2=OK["cena_CO2_obcutljivost_EUR_t"])
        long = pd.concat({p: d for p, d in res.items()}, names=["pot", "enota"]).reset_index()
        long["javni_NPV_UBA"] = pd.concat({p: d for p, d in res_s.items()}).reset_index(drop=True)["javni_NPV"]
        long = long.merge(w[["Qr", "arhetip"]], left_on="enota", right_index=True)
        if src == "repr":
            long.round(1).to_csv(TAB / "05_drzava_reprezentativne.csv", sep=";", encoding="utf-8-sig", index=False)
            t = long.pivot_table(index="arhetip", columns="pot", values="javni_NPV")[POTI] / 1000
            k = long.pivot_table(index="arhetip", columns="pot", values="konica_sis_kW_leto20")[POTI]
            sp = long.pivot_table(index="arhetip", columns="pot", values="spodbude_nom")[POTI] / 1000
            md += ["## Reprezentativne hiše", "", "Javni NPV [kEUR] (spodbude + omrežje + CO2):", "", t.round(2).to_string(), "",
                   "Spodbude nominalno [kEUR]:", "", sp.round(2).to_string(), "",
                   "Prispevek h sistemski konici v letu 20 [kW/hišo]:", "", k.round(2).to_string(), ""]
        else:
            agg = long.groupby(["Qr", "pot"]).agg(
                stavb=("enota", "size"), javni_NPV_med=("javni_NPV", "median"), javni_NPV_UBA_med=("javni_NPV_UBA", "median"),
                spodbude_nom_vsota_mioEUR=("spodbude_nom", lambda x: x.sum() / 1e6), konica_sis_MW_leto1=("konica_sis_kW_leto1", lambda x: x.sum() / 1000),
                konica_sis_MW_leto20=("konica_sis_kW_leto20", lambda x: x.sum() / 1000), zimski_uvoz_GWh_leto20=("zimski_uvoz_kWh_leto20", lambda x: x.sum() / 1e6),
                koncna_energija_30_TWh=("koncna_energija_30_MWh", lambda x: x.sum() / 1e6)).reset_index()
            agg.round(2).to_csv(TAB / "05_drzava_fond.csv", sep=";", encoding="utf-8-sig", index=False)
            piv = long.pivot_table(index=["enota", "Qr"], columns="pot", values="javni_NPV").reset_index()
            share = piv.groupby("Qr").apply(lambda d: pd.Series({
                "P3 < P1 [%]": 100 * (d["P3"] < d["P1"]).mean(), "P4 < P1 [%]": 100 * (d["P4"] < d["P1"]).mean(),
                "P6 < P1 [%]": 100 * (d["P6"] < d["P1"]).mean(), "P1 < P0 [%]": 100 * (d["P1"] < d["P0"]).mean()}))
            # učinkovitost spodbud glede na P0 (30 let)
            e = long.pivot_table(index="enota", columns="pot", values=["spodbude_nom", "koncna_energija_30_MWh"])
            c6 = pd.read_parquet(HERE / "cache" / "06_co2_poti_fond.parquet").set_index("ID_Stavbe")
            ef = pd.DataFrame({p: {"EUR spodbud / MWh prihranjene končne energije": (e[("spodbude_nom", p)].sum()
                                                                                    / (e[("koncna_energija_30_MWh", "P0")] - e[("koncna_energija_30_MWh", p)]).sum()),
                                   "EUR spodbud / t CO2e izognjene (30 let)": (e[("spodbude_nom", p)].sum()
                                                                                / (c6["P0|skupaj_t"] - c6[f"{p}|skupaj_t"]).sum())}
                               for p in ("P1", "P1b", "P3", "P4", "P5", "P6")}).T
            ef.round(1).to_csv(TAB / "05_ucinkovitost_spodbud.csv", sep=";", encoding="utf-8-sig")
            md += ["## Fond 29.977 stavb", "", agg.round(1).to_string(index=False), "", "Deleži stavb (javni NPV):", "", share.round(1).to_string(), "",
                   "Učinkovitost spodbud glede na P0 (fond, 30 let):", "", ef.round(1).to_string(), ""]
    md.append(f"Trajanje {time.time() - t0:.0f} s.")
    (HERE / "05_povzetek.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
