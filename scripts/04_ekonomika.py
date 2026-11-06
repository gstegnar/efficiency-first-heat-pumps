"""Korak 04 — posameznik: stroški poti v 30 letih (NPV), režimi obračuna R1/R2/R3/R2d, mejna vrednost ovoja s PV in brez.

Vhodi: cache/03_fond_letno.parquet in tabele/03_energija_variante.csv (letna energija in računi po variantah, korak 03),
       parametri.yaml: ekonomika (cene, omrežnina, investicije, spodbude, življenjske dobe — viri v podatki/04_viri_*.md).
Poti (CLAUDE.md §4.2; leto 0 = 2026):
  P0  ostane pri kotlu (V0a ELKO / V0b biomasa po nadomeščenem energentu)
  P1  TČ + PV (V2) v letu 0, nikoli ovoj            P1b  + baterija (V3)
  P2  V2 v letu 0 → FSO v letu 10 (V8, obstoječa TČ in PV) → nova TČ ob zamenjavi v letu 18 (V9)
  P3  FSO + TČ v letu 0 (V4) → PV v letu 1 (V5)      P3b  + baterija v letu 1 (V6)
  P4  vse naenkrat (V5)                               P5   + baterija (V6)          P6  GLOB + TČ + PV + baterija (V7)
  PA  samo TČ (V1)          PB  TČ (V1) → FSO v letu 10 (V4)   — za mejno vrednost ovoja brez PV
  PC  FSO + TČ brez PV (V4)  — URE-najprej za hiše brez PV (korak 08)
NPV = sedanja vrednost stroškov (pozitivno = strošek): investicije po spodbudah + zamenjave + računi + gorivo + vzdrževanje − ostanek.
Mejna vrednost ovoja: s PV = NPV(P1) − NPV(P2); brez PV = NPV(PA) − NPV(PB) (pozitivno = ovoj se izplača).
Izhodi: tabele/04_posameznik_reprezentativne.csv, tabele/04_posameznik_fond.csv, tabele/04_mejna_vrednost_ovoja.csv,
        tabele/04_obcutljivost_reprezentativne.csv, cache/04_npv_fond.parquet, 04_povzetek.md
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
PAR = yaml.safe_load((HERE / "parametri.yaml").read_text(encoding="utf-8"))
EK, HV = PAR["ekonomika"], PAR["hecm_vhodi"]
_GEOM = __import__('os').environ.get("GEOM", "G13")
if _GEOM in ("G11", "G13"):            # krog 11: strošek po Eko količini (G11); krog 13: strošek = toplotni ovoj G13 × izmerjena pokritost (geometrija_G13_strosek)
    for _a, _g in HV["geometrija_G11" if _GEOM == "G11" else "geometrija_G13_strosek"].items():
        HV["arhetipi"][_a]["geometrija"] = _g
SP = EK["spodbude"]
TAB, CACHE = HERE / "tabele", HERE / "cache"
N = EK["horizont_let"]
REZ = ["R1", "R2", "R3", "R2d"]
t0 = time.time()

POTI = {
    "P0": {"seg": [(0, "V0")], "ev": []},
    "P1": {"seg": [(0, "V2")], "ev": [(0, "TC", "e"), (0, "PV", "V2"), (15, "RAZS", "V2"), (18, "TCz", "e")]},
    "P1b": {"seg": [(0, "V3")], "ev": [(0, "TC", "e"), (0, "PVB", "V3"), (0, "BAT", None), (15, "RAZS", "V3"), (15, "BATz", None), (18, "TCz", "e")]},
    "P2": {"seg": [(0, "V2"), (10, "V8"), (18, "V9")], "ev": [(0, "TC", "e"), (0, "PV", "V2"), (10, "FSO", None), (15, "RAZS", "V2"), (18, "TCz", "FSO")]},
    "P3": {"seg": [(0, "V4"), (1, "V5")], "ev": [(0, "FSO", None), (0, "TC", "FSO"), (1, "PV", "V5"), (16, "RAZS", "V5"), (18, "TCz", "FSO")]},
    "P3b": {"seg": [(0, "V4"), (1, "V6")], "ev": [(0, "FSO", None), (0, "TC", "FSO"), (1, "PVB", "V6"), (1, "BAT", None), (16, "RAZS", "V6"), (16, "BATz", None), (18, "TCz", "FSO")]},
    "P4": {"seg": [(0, "V5")], "ev": [(0, "FSO", None), (0, "TC", "FSO"), (0, "PV", "V5"), (15, "RAZS", "V5"), (18, "TCz", "FSO")]},
    "P5": {"seg": [(0, "V6")], "ev": [(0, "FSO", None), (0, "TC", "FSO"), (0, "PVB", "V6"), (0, "BAT", None), (15, "RAZS", "V6"), (15, "BATz", None), (18, "TCz", "FSO")]},
    "P6": {"seg": [(0, "V7")], "ev": [(0, "GLOB", None), (0, "TC", "GLOB"), (0, "PVB", "V7"), (0, "BAT", None), (15, "RAZS", "V7"), (15, "BATz", None), (18, "TCz", "GLOB"), (25, "MVHRz", None)]},
    "PA": {"seg": [(0, "V1")], "ev": [(0, "TC", "e"), (18, "TCz", "e")]},
    "PB": {"seg": [(0, "V1"), (10, "V4")], "ev": [(0, "TC", "e"), (10, "FSO", None), (18, "TCz", "FSO")]},
    "PC": {"seg": [(0, "V4")], "ev": [(0, "FSO", None), (0, "TC", "FSO"), (18, "TCz", "FSO")]},
    # krog 3: »OVE najprej, pravilno dimenzionirano« — TČ in PV po rabi hiše brez ovoja, ovoja nikoli (spodbude po pravilih 2026)
    "P1r": {"seg": [(0, "V2r")], "ev": [(0, "TC", "r"), (0, "PV", "V2r"), (15, "RAZS", "V2r"), (18, "TCz", "r")]},
}


def geom(arh):
    g = HV["geometrija"][HV["arhetipi"][arh]["geometrija"]]
    return {k: g[k] / g["A"] for k in ("stena", "streha", "okna", "tla")}


def load_units(src: str, tag: str = "") -> pd.DataFrame:
    """Wide table: one row per building (or representative house), columns <var>|<quantity>."""
    if src == "fond":
        fl = pd.read_parquet(CACHE / f"03_fond_letno{tag}.parquet")
        f1 = pd.read_parquet(CACHE / "01_fond.parquet", columns=["ID_Stavbe", "staro_ogrevanje"])
        if tag:                                              # sintetične stavbe (07): kotel ELKO; realne (K2_5) po ID
            ids = fl["ID_Stavbe"].unique()
            f1 = f1.set_index("ID_Stavbe").reindex(ids).fillna({"staro_ogrevanje": "ELKO"}).rename_axis("ID_Stavbe").reset_index()
        key = "ID_Stavbe"
    else:
        fl = pd.read_csv(TAB / "03_energija_variante.csv", sep=";", encoding="utf-8-sig")
        fl["ID_Stavbe"] = "REPR_" + fl["arhetip"]
        fl["Qr"] = np.where(fl["arhetip"] == "A4", "Q2", "Q1")
        f1 = pd.DataFrame({"ID_Stavbe": fl["ID_Stavbe"].unique(), "staro_ogrevanje": "ELKO"})
        key = "ID_Stavbe"
    q = ["A", "TC_kW", "PV_kWp", "baterija_kWh", "gorivo_kWh", "uvoz_kWh"] + [f"{r}_{c}" for r in ("R1", "R2", "R3") for c in ("energija", "oddaja", "omr_daj")] + ["R2d_energija"]
    q += [c for c in ("CO2_el_urni_kg", "CO2_el_povp_kg", "CO2_izogib_oddaja_kg", "uvoz_konica_sis_kW", "uvoz_konica_sis_kWh", "uvoz_zima_kWh",
                      "uvoz_konica_kW", "oddaja_kWh", "el_skupaj_kWh", "PE_kWh_m2", "Qh_nd", "R3_DOM1_kW",
                      "uvoz_konica_dist_kW", "oddaja_konica_dist_kW", "lestvica_k", "lestvica_w", "brez_ovoja_P3", "Qh_nd_cilj",
                      "PV_kWh", "PV_izgubljeno_kWh", "PV_izgubljeno_EUR") if c in fl.columns]
    q += [c for c in fl.columns if (c.startswith(("soc_",)) or c.endswith(("_omreznina", "_prispevki", "_trosarina", "_ddv"))) and c not in q]
    w = fl.pivot_table(index=key, columns="varianta", values=q, aggfunc="first")
    w.columns = [f"{v}|{c}" for c, v in w.columns]
    meta = fl.drop_duplicates(key).set_index(key)[["arhetip", "Qr"]]
    w = w.join(meta).join(f1.set_index(key))
    w["V0"] = np.select([w["staro_ogrevanje"] == "biomasa", (w["staro_ogrevanje"] == "plin") & ("V0c|uvoz_kWh" in w.columns)], ["V0b", "V0c"], "V0a")
    # opazovano stanje: dejansko izplačane spodbude (register Eko sklad) — TČ za vse stavbe fonda, PV za stavbe z dejanskim PV
    w["dej_TC_sub"], w["dej_PV_sub"] = np.nan, np.nan
    if src == "fond" and SP.get("dejanske_spodbude_opazovano", EK.get("dejanske_spodbude_opazovano")):   # sintetične: NaN -> pravila
        an = Path(PAR["poti"]["analiza"])
        an = an if an.is_absolute() else (HERE / an).resolve()
        reg = pd.read_parquet(an / "cache" / "07_stavbe_paket.parquet", columns=["ID_Stavbe", "tc_znesek", "PV_znesek", "ima_PV"]).set_index("ID_Stavbe")
        reg = reg.reindex(w.index)
        w["dej_TC_sub"] = reg["tc_znesek"].astype(float)
        w["dej_PV_sub"] = np.where(reg["ima_PV"].fillna(False).astype(bool), reg["PV_znesek"].astype(float), np.nan)
    return w


def annual_bill(w, var, rez, pf):
    """Annual electricity (+ fuel) cost [EUR] for variant var under regime rez; pf = energy price factor (scalar)."""
    r_ = "R2" if rez == "R2d" else rez
    ecol = "R2d_energija" if rez == "R2d" else f"{r_}_energija"

    def one(v, fuel_p=0.0):
        return (w[f"{v}|{ecol}"] * pf - w[f"{v}|{r_}_oddaja"] * pf + w[f"{v}|{r_}_omr_daj"]
                + w[f"{v}|gorivo_kWh"] * fuel_p * pf).to_numpy()
    if var == "V0":
        out = np.where(w["V0"] == "V0b", one("V0b", EK["biomasa_EUR_kWh"]), one("V0a", EK["ELKO_EUR_kWh"]))
        if "V0c|uvoz_kWh" in w.columns:
            out = np.where(w["V0"] == "V0c", one("V0c", EK["plin_EUR_kWh"]), out)
        return out
    return one(var)


MANJKA = {0: ("stena", "streha", "okna", "tla"), 1: ("stena", "streha", "tla"), 2: ("stena", "tla"), 3: ()}   # krog 3: manjkajoči ukrepi do cilja


def ovoj_delez(w, kind):
    """Delež površine elementa, ki ga ukrep obnovi (0–1), po hiši. Krog 3: FSO = manjkajoči ukrepi do cilja (fasada, streha, okna, tla) glede na
    položaj na lestvici stanj, utež w kot pri profilu (B2); brez ukrepa na ovoju (B3) → 0. GLOB: vsi elementi."""
    n = len(w)
    if kind == "GLOB":
        return {el: np.ones(n) for el in ("stena", "streha", "okna", "tla")}
    if "V4|lestvica_k" not in w.columns:                       # stari izhodi (krog 2): FSO = fasada, streha, okna
        return {"stena": np.ones(n), "streha": np.ones(n), "okna": np.ones(n), "tla": np.zeros(n)}
    k = w["V4|lestvica_k"].to_numpy().astype(int)
    ww = w["V4|lestvica_w"].to_numpy(float)
    brez = w["V4|brez_ovoja_P3"].to_numpy(float) > 0.5
    k1 = np.minimum(k + 1, 3)
    out = {}
    for el in ("stena", "streha", "okna", "tla"):
        a0 = np.array([el in MANJKA[x] for x in k], float)
        a1 = np.array([el in MANJKA[x] for x in k1], float)
        out[el] = np.where(brez, 0.0, (1 - ww) * a0 + ww * a1)
    return out


_G_CEL = None


def faktor_debeline(w, kind):
    """Krog 3 (občutljivost A4): strošek neprozornih elementov × (1 + α·(1/g_eff − 1)), g_eff = zaostritev ciljne prenove po položaju na lestvici."""
    global _G_CEL
    a = EK.get("ovoj_delez_debeline", 0.0)
    if not a or kind != "FSO" or "V4|lestvica_k" not in w.columns:
        return np.ones(len(w))
    if _G_CEL is None:
        v = pd.read_csv(TAB / "02_variante.csv", sep=";", encoding="utf-8-sig")
        _G_CEL = {(r.arhetip, r.varianta): float(getattr(r, "g_celovita", 1.0)) for r in v.itertuples()}
    cil = {0: "FSO", 1: "FSO_S1", 2: "FSO_S2", 3: "S3"}
    k = w["V4|lestvica_k"].to_numpy().astype(int)
    ww = w["V4|lestvica_w"].to_numpy(float)
    k1 = np.minimum(k + 1, 3)
    g0 = np.array([_G_CEL.get((ar, cil[x]), 1.0) for ar, x in zip(w["arhetip"], k)])
    g1 = np.array([_G_CEL.get((ar, cil[x]), 1.0) for ar, x in zip(w["arhetip"], k1)])
    g = (1 - ww) * g0 + ww * g1
    return 1 + a * (1 / np.maximum(g, 0.1) - 1)


def envelope_cost(w, kind, basis):
    """(gross investment, subsidy, residual-value fraction weights per element) for FSO or GLOB."""
    A = w["V4|A"].to_numpy()
    gr = np.stack([[geom(a)[k] for k in ("stena", "streha", "okna", "tla")] for a in w["arhetip"]])
    area = {k: gr[:, i] * A for i, k in enumerate(("stena", "streha", "okna", "tla"))}
    dl = ovoj_delez(w, kind)
    fd = faktor_debeline(w, kind)
    inv, sub, res = np.zeros(len(w)), np.zeros(len(w)), np.zeros(len(w))
    fs = EK.get("ovoj_strosek_faktor", 1.0)                 # krog 10: strošek ovoja × razmerje dejansko / model (po arhetipu ali skalar); osnova 1
    fs = np.array([fs.get(a, 1.0) for a in w["arhetip"]]) if isinstance(fs, dict) else np.full(len(w), float(fs))
    for el in ("stena", "streha", "okna", "tla"):
        c = dl[el] * (area[el] * EK["ovoj_EUR_enota"][el] if basis == "element" else A * EK["ovoj_EUR_m2_ogrevane"][el]) * (fd if el != "okna" else 1.0) * fs
        s = np.minimum(SP["ovoj_delez"] * c, (SP["okna_max_EUR_m2"] if el == "okna" else SP["ovoj_max_EUR_m2"]) * area[el] * dl[el])
        L = EK["ovoj_let"][el]
        inv += c
        sub += s
        res += (c - s) * L                                  # weighted by life, converted below
    if kind == "GLOB":
        mv = np.full(len(w), EK["MVHR_EUR"])
        smv = np.minimum(SP["MVHR_delez"] * mv, SP["MVHR_max"])
        zr = area["streha"] * EK["zrakotesnost_EUR_m2_strehe"]
        inv += mv + zr
        sub += smv
        res += (mv - smv) * EK["ovoj_let"]["MVHR"] + zr * EK["ovoj_let"]["streha"]
    return inv, sub, res                                    # res = Σ net_cost·life (for residual value)


def npv_paths(w, rez, r, pf_path, basis="element", subsidies=True):
    """NPV of each path for every unit; pf_path(t) = energy price factor in year t (1..N)."""
    disc = lambda t: (1 + r) ** (-t)                        # noqa: E731
    kW = {"e": w["V1|TC_kW"].to_numpy(), "FSO": w["V4|TC_kW"].to_numpy(), "GLOB": w["V7|TC_kW"].to_numpy(), "r": w["V2r|TC_kW"].to_numpy() if "V2r|TC_kW" in w.columns else w["V1|TC_kW"].to_numpy()}
    kwp = {v: w[f"{v}|PV_kWp"].to_numpy() for v in ("V2", "V3", "V5", "V6", "V7", "V2r") if f"{v}|PV_kWp" in w.columns}
    sf = 1.0 if subsidies else 0.0
    res_out = {}
    for p, d in POTI.items():
        cost = np.zeros(len(w))
        residual = np.zeros(len(w))
        maint_items = []                                    # (from_year, annual EUR array)
        for (y, item, arg) in d["ev"]:
            if item in ("TC", "TCz"):
                inv = EK["TC_a_EUR"] + EK["TC_b_EUR_kW"] * kW[arg]
                sub = np.minimum(SP["TC_delez"] * inv, SP["TC_max"]) * sf if item == "TC" else 0
                if item == "TC" and arg == "e":               # obstoječa (opazovana) TČ: dejanska spodbuda, če je znana
                    sub = np.where(w["dej_TC_sub"].notna(), w["dej_TC_sub"].to_numpy() * sf, sub)
                cost += (inv - sub) * disc(y)
                life_end = y + EK["TC_let"]
                if life_end > N:
                    residual += (inv - sub) * (life_end - N) / EK["TC_let"]
                if item == "TC":                            # vzdrževanje TČ teče do konca horizonta (tudi po zamenjavi)
                    maint_items.append((y, EK["TC_vzdrzevanje_delez"] * inv, N))
            elif item in ("PV", "PVB"):
                k = kwp[arg]
                inv = np.where(k > 0, EK["PV_a_EUR"] + EK["PV_b_EUR_kWp"] * k, 0)
                if item == "PVB":
                    invb = EK["baterija_EUR_kWh"] * w["V6|baterija_kWh"].to_numpy()
                    sub = np.minimum(SP["PV_bat_EUR_kW"] * k, SP["PV_bat_max_delez"] * (inv + invb)) * sf
                else:
                    sub = np.minimum(SP["PV_EUR_kW"] * k, SP["PV_max_delez"] * inv) * sf
                    if arg == "V2":                             # opazovani PV: dejanska spodbuda, če je znana
                        sub = np.where(w["dej_PV_sub"].notna(), w["dej_PV_sub"].to_numpy() * sf, sub)
                cost += (inv - sub) * disc(y)
                residual += (inv - sub) * max(0, y + 30 - N) / 30
                maint_items.append((y, EK["PV_vzdrzevanje_delez"] * inv, N))
            elif item == "RAZS":
                k = kwp[arg]
                cost += np.where(k > 0, EK["razsmernik_delez_PV"] * (EK["PV_a_EUR"] + EK["PV_b_EUR_kWp"] * k), 0) * disc(y)
            elif item in ("BAT", "BATz"):
                invb = EK["baterija_EUR_kWh"] * w["V6|baterija_kWh"].to_numpy()
                cost += invb * (1 if item == "BAT" else EK["baterija_zamenjava_delez"]) * disc(y)
            elif item in ("FSO", "GLOB"):
                inv, sub, lifew = envelope_cost(w, item, basis)
                net = inv - sub * sf
                cost += net * disc(y)
                # residual: each element net cost × (remaining life / life); approximated with cost-weighted mean life
                Lm = lifew / np.maximum(inv - sub, 1e-9)
                residual += net * np.clip((y + Lm - N) / Lm, 0, 1)
                if item == "GLOB":
                    maint_items.append((y, np.full(len(w), EK["MVHR_vzdrzevanje_EUR_a"]), N))
            elif item == "MVHRz":
                cost += EK["MVHR_EUR"] * disc(y)
        # annual operating costs
        segs = d["seg"] + [(N, None)]
        for (ys, var), (ye, _) in zip(segs[:-1], segs[1:]):
            for t in range(ys + 1, ye + 1):
                cost += annual_bill(w, var, rez, pf_path(t)) * disc(t)
        for (y, m, end) in maint_items:
            for t in range(y + 1, N + 1):
                cost += m * disc(t)
        if p == "P0":
            for t in range(1, N + 1):
                cost += EK["vzdrzevanje_kotla_EUR_a"] * disc(t)
        cost -= residual * disc(N)
        res_out[p] = cost
    return pd.DataFrame(res_out, index=w.index)


def leto_varianta(p: str, t: int) -> str:
    """Varianta, aktivna v letu t (1..N) na poti p."""
    v = None
    for y, var in POTI[p]["seg"]:
        if t > y:
            v = var
    return v


def spodbude_poti(w, basis="element", r=None):
    """Spodbude [EUR] po poti: (vsota nominalno, sedanja vrednost pri diskontu r) — enaka logika kot v npv_paths."""
    r = EK["diskont_javni"] if r is None else r
    kW = {"e": w["V1|TC_kW"].to_numpy(), "FSO": w["V4|TC_kW"].to_numpy(), "GLOB": w["V7|TC_kW"].to_numpy(), "r": w["V2r|TC_kW"].to_numpy() if "V2r|TC_kW" in w.columns else w["V1|TC_kW"].to_numpy()}
    kwp = {v: w[f"{v}|PV_kWp"].to_numpy() for v in ("V2", "V3", "V5", "V6", "V7", "V2r") if f"{v}|PV_kWp" in w.columns}
    nom, pv = {}, {}
    for p, d in POTI.items():
        a, b = np.zeros(len(w)), np.zeros(len(w))
        for (y, item, arg) in d["ev"]:
            s_ = np.zeros(len(w))
            if item == "TC":
                s_ = np.minimum(SP["TC_delez"] * (EK["TC_a_EUR"] + EK["TC_b_EUR_kW"] * kW[arg]), SP["TC_max"])
                if arg == "e":
                    s_ = np.where(w["dej_TC_sub"].notna(), w["dej_TC_sub"].to_numpy(), s_)
            elif item in ("PV", "PVB"):
                k = kwp[arg]
                inv = np.where(k > 0, EK["PV_a_EUR"] + EK["PV_b_EUR_kWp"] * k, 0)
                if item == "PVB":
                    invb = EK["baterija_EUR_kWh"] * w["V6|baterija_kWh"].to_numpy()
                    s_ = np.minimum(SP["PV_bat_EUR_kW"] * k, SP["PV_bat_max_delez"] * (inv + invb))
                else:
                    s_ = np.minimum(SP["PV_EUR_kW"] * k, SP["PV_max_delez"] * inv)
                    if arg == "V2":
                        s_ = np.where(w["dej_PV_sub"].notna(), w["dej_PV_sub"].to_numpy(), s_)
            elif item in ("FSO", "GLOB"):
                s_ = envelope_cost(w, item, basis)[1]
            a += s_
            b += s_ * (1 + r) ** (-y)
        nom[p], pv[p] = a, b
    return pd.DataFrame(nom, index=w.index), pd.DataFrame(pv, index=w.index)


def pf_factory(scen):
    if scen == "konstantne":
        return lambda t: 1.0
    if scen == "rast2":
        return lambda t: (1 + EK["scenariji_cen"]["rast2"]) ** t
    if scen == "sok":
        return lambda t: EK["sok_faktor"] if t in EK["sok_leta"] else 1.0
    raise ValueError(scen)


def summarise(npv, w, rez):
    d = npv.copy()
    d["Qr"], d["arhetip"] = w["Qr"].values, w["arhetip"].values
    d["mejna_ovoj_s_PV"] = d["P1"] - d["P2"]
    d["mejna_ovoj_brez_PV"] = d["PA"] - d["PB"]
    d["P3_boljse_od_P1"] = d["P3"] < d["P1"]
    d["P4_boljse_od_P1"] = d["P4"] < d["P1"]
    d["P3b_boljse_od_P1b"] = d["P3b"] < d["P1b"]
    d["rezim"] = rez
    return d


def main():
    out_md = [f"# 04 — Posameznik: NPV poti v {N} letih ({time.strftime('%d. %m. %Y %H:%M')})", "",
              f"Diskont {EK['diskont_zasebni']:.0%} realno, konstantne realne cene (osnovno), spodbude 2026 (Eko sklad 114SUB-OB24, Borzen kot SO-02 — NEPREVERJENO), "
              f"ovoj: EUR na enoto × površina elementa ({EK['ovoj_osnova']}). NPV = sedanja vrednost STROŠKOV [EUR] (manj je bolje).", ""]
    # ---- reprezentativne hiše
    wr = load_units("repr")
    rows = []
    for rez in REZ:
        n = npv_paths(wr, rez, EK["diskont_zasebni"], pf_factory("konstantne"), EK["ovoj_osnova"])
        s = summarise(n, wr, rez)
        rows.append(s)
    R = pd.concat(rows)
    R.round(0).to_csv(TAB / "04_posameznik_reprezentativne.csv", sep=";", encoding="utf-8-sig")
    show = R.reset_index()[["arhetip", "rezim", "P0", "P1", "P1b", "P2", "P3", "P3b", "P4", "P5", "P6", "mejna_ovoj_s_PV", "mejna_ovoj_brez_PV"]]
    out_md += ["## Reprezentativne hiše — NPV stroškov [kEUR]", "", (show.set_index(["arhetip", "rezim"]) / 1000).round(1).reset_index().to_string(index=False), ""]
    # ---- občutljivost (reprezentativne)
    sens = []
    for lab, kw in [("osnova", {}), ("diskont 2 %", {"r": 0.02}), ("diskont 7 %", {"r": 0.07}), ("cene +2 %/a", {"scen": "rast2"}),
                    ("šok cen leta 5–6", {"scen": "sok"}), ("brez spodbud", {"subsidies": False}), ("ovoj po EUR/m² ogrevane", {"basis": "ogrevana"})]:
        for rez in ("R1", "R2", "R3"):
            n = npv_paths(wr, rez, kw.get("r", EK["diskont_zasebni"]), pf_factory(kw.get("scen", "konstantne")),
                          kw.get("basis", EK["ovoj_osnova"]), kw.get("subsidies", True))
            s = summarise(n, wr, rez)
            s["obcutljivost"] = lab
            sens.append(s)
    S = pd.concat(sens).reset_index()
    S.round(0).to_csv(TAB / "04_obcutljivost_reprezentativne.csv", sep=";", encoding="utf-8-sig", index=False)
    S["P3−P1 [kEUR]"] = (S["P3"] - S["P1"]) / 1000
    S["P4−P1 [kEUR]"] = (S["P4"] - S["P1"]) / 1000
    S["mejna ovoja s PV / brez PV [kEUR]"] = (S["mejna_ovoj_s_PV"] / 1000).round(1).astype(str) + " / " + (S["mejna_ovoj_brez_PV"] / 1000).round(1).astype(str)
    out_md += ["## Občutljivost (reprezentativne hiše): P3 − P1 in P4 − P1 [kEUR] (negativno = URE-najprej cenejše)", "",
               S.pivot_table(index=["obcutljivost", "arhetip"], columns="rezim", values="P3−P1 [kEUR]").round(1).reset_index().to_string(index=False), "",
               "Mejna vrednost ovoja v letu 10 s PV / brez PV [kEUR] (pozitivno = ovoj se izplača):", "",
               S[S["rezim"].isin(["R1", "R2", "R3"])].pivot_table(index=["obcutljivost", "arhetip"], columns="rezim",
                                                                  values="mejna ovoja s PV / brez PV [kEUR]", aggfunc="first").reset_index().to_string(index=False), ""]
    # ---- fond
    wf = load_units("fond")
    F = []
    for rez in REZ:
        n = npv_paths(wf, rez, EK["diskont_zasebni"], pf_factory("konstantne"), EK["ovoj_osnova"])
        F.append(summarise(n, wf, rez))
    F = pd.concat(F)
    F.reset_index().to_parquet(CACHE / "04_npv_fond.parquet", index=False)
    agg = (F.reset_index().groupby(["rezim", "Qr"])
           .agg(stavb=("P1", "size"), **{f"{p} mediana": (p, "median") for p in ["P0", "P1", "P1b", "P2", "P3", "P3b", "P4", "P5", "P6"]},
                **{"delez P3 < P1 [%]": ("P3_boljse_od_P1", lambda x: 100 * x.mean()),
                   "delez P4 < P1 [%]": ("P4_boljse_od_P1", lambda x: 100 * x.mean()),
                   "delez P3b < P1b [%]": ("P3b_boljse_od_P1b", lambda x: 100 * x.mean()),
                   "mejna ovoj s PV mediana": ("mejna_ovoj_s_PV", "median"),
                   "mejna ovoj brez PV mediana": ("mejna_ovoj_brez_PV", "median"),
                   "delez ovoj s PV > 0 [%]": ("mejna_ovoj_s_PV", lambda x: 100 * (x > 0).mean()),
                   "delez ovoj brez PV > 0 [%]": ("mejna_ovoj_brez_PV", lambda x: 100 * (x > 0).mean())})
           .reset_index())
    agg.round(0).to_csv(TAB / "04_posameznik_fond.csv", sep=";", encoding="utf-8-sig", index=False)
    mv = agg[["rezim", "Qr", "mejna ovoj s PV mediana", "mejna ovoj brez PV mediana", "delez ovoj s PV > 0 [%]", "delez ovoj brez PV > 0 [%]"]]
    mv.round(0).to_csv(TAB / "04_mejna_vrednost_ovoja.csv", sep=";", encoding="utf-8-sig", index=False)
    a2 = agg.copy()
    for c in a2.columns:
        if "mediana" in c:
            a2[c] = (a2[c] / 1000).round(1)
    out_md += ["## Fond 29.977 stavb — mediane NPV [kEUR] in deleži", "", a2.round(1).to_string(index=False), "",
               f"Trajanje {time.time() - t0:.0f} s."]
    (HERE / "04_povzetek.md").write_text("\n".join(out_md) + "\n", encoding="utf-8")
    print("\n".join(out_md))


if __name__ == "__main__":
    main()
