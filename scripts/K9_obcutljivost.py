"""Krog 9, §2, §3, §5, §6 — občutljivosti podnebje, raba gospodinjstva, cena elektrike; P0 in ETS2 (NAVODILA_CLAUDE_CODE_v9.md).

Paket 5.969 (vhodi L3 iz kroga 8). Urni teki: cache/K9_obcutljivost.sh (--tag _K9pod090 / _K9pod080 / _K9g2500 / _K9g6000); osnova = _K9c10 (= L3).
  §2 podnebje: Q_H × povprečni faktor horizonta 2026–2055 (0,9417 / 0,8833; urni_model.faktor_QH)
  §3 gospodinjstvo 2.500 / 6.000 kWh/a (osnova 4.000); P3/P4 PV po rabi, P1 dejanski kWp
  §5 realna rast cene elektrike (energija IN omrežnina/dajatve) +1 / +2 %/a pri diskontu 4 %; družbena urna vrednost elektrike enako (javni diskont 3 %)
  §6 P0 (kotel ostane) z in brez ETS2 proti P1 in P3, posameznik R3, mediana po obdobju in energentu
Koraki: B → cache/17_rezultati.parquet; T → clanek1/tabele/17_obcutljivost_podnebje_gospodinjstvo_cene.csv, 17b_P0_ETS2.csv (+ tabele/)
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
PAR = yaml.safe_load((HERE / "parametri.yaml").read_text(encoding="utf-8"))
TAB, OUT_T, CACHE = HERE / "tabele", HERE / "clanek1" / "tabele", HERE / "cache"
EK = PAR["ekonomika"]
OBD = {"paket": ["A1", "A2", "A3", "A4"], "pred 1981": ["A1", "A2", "A3"], "1981–2002": ["A4"]}
# scenarij: (pripona urnega teka, realna rast cene elektrike)
SCEN = {"osnova": ("_K9c10", 0.0),
        "podnebje 0,90 v 2050 (f = 0,942)": ("_K9pod090", 0.0), "podnebje 0,80 v 2050 (f = 0,883)": ("_K9pod080", 0.0),
        "gospodinjstvo 2.500 kWh/a": ("_K9g2500", 0.0), "gospodinjstvo 6.000 kWh/a": ("_K9g6000", 0.0),
        "cena elektrike +1 %/a": ("_K9c10", 0.01), "cena elektrike +2 %/a": ("_K9c10", 0.02)}
# ETS2 (§6): 45 EUR/t od 2027, linearno do 100 EUR/t v 2040, nato konstantno (realno) — PREVERI (Direktiva 2023/959, mehanizem 45 EUR/t v cenah 2020;
# začetek ETS2 morda zamaknjen na 2028)
ETS2 = {"zacetek": 2027, "cena0": 45.0, "leto_cilj": 2040, "cena_cilj": 100.0}
EF_ETS2 = {"V0a": PAR["okolje"]["EF_goriva_kg_kWh"]["ELKO"], "V0c": PAR["okolje"]["EF_goriva_kg_kWh"]["plin"], "V0b": 0.0}   # biomasa ni v ETS2


def _moduli():
    s = importlib.util.spec_from_file_location("k21", HERE / "K2_1_drzava.py")
    K1 = importlib.util.module_from_spec(s)
    s.loader.exec_module(K1)
    return K1, K1.E, K1.DR.OKM


def ets2_cena(leto):
    if leto < ETS2["zacetek"]:
        return 0.0
    x = min(1.0, (leto - ETS2["zacetek"]) / (ETS2["leto_cilj"] - ETS2["zacetek"]))
    return ETS2["cena0"] + x * (ETS2["cena_cilj"] - ETS2["cena0"])


def korak_b():
    K1, E, OKM = _moduli()
    orig_bill = E.annual_bill

    def bill_rast(w, var, rez, pf):
        """Rast cene elektrike velja za energijo in omrežnino/dajatve (orig. pf le za energijo); gorivo se ne spreminja (P1/P3 ga nimata)."""
        r_ = "R2" if rez == "R2d" else rez
        out = orig_bill(w, var, rez, pf)
        return out if var == "V0" else out + (pf - 1) * w[f"{var}|{r_}_omr_daj"].to_numpy()

    rj = PAR["drzava_k2"]["diskont"]
    out, p0 = [], []
    for sc, (tag, g) in SCEN.items():
        w = E.load_units("fond", tag)
        pf = (lambda t: (1 + g) ** t) if g else E.pf_factory("konstantne")
        E.annual_bill = bill_rast
        n = {rez: E.npv_paths(w, rez, EK["diskont_zasebni"], pf, EK["ovoj_osnova"]) for rez in ("R1", "R3")}
        E.annual_bill = orig_bill
        rc = K1.racun(w)
        o = pd.DataFrame(index=w.index)
        o["arhetip"] = w["arhetip"]
        for p, v in (("P1", "V2"), ("P3", "V5")):
            for rez in ("R1", "R3"):
                o[f"{p}|NPV {rez}"] = n[rez][p]
            soc = rc[p]["družbeni NPV"].to_numpy()
            if g:                                            # družbena urna vrednost elektrike raste enako
                soc = soc + sum(((1 + g) ** t - 1) * (w[f"{E.leto_varianta(p, t)}|soc_uvoz_EUR"] - w[f"{E.leto_varianta(p, t)}|soc_oddaja_EUR"]).to_numpy()
                                * (1 + rj) ** (-t) for t in range(1, E.N + 1))
            o[f"{p}|družbeni NPV"] = soc
            o[f"{p}|NN kW"] = rc[p]["obremenitev NN omrežja [kW]"]
            op = sum(OKM.op_co2(w, E.leto_varianta(p, t), t) for t in range(1, E.N + 1))
            emb = sum(OKM.embodied(w, item, arg) for (y, item, arg) in E.POTI[p]["ev"])
            o[f"{p}|CO2e t"] = (op + emb) / 1000
            o[f"{p}|zimski uvoz kWh"] = w[f"{v}|uvoz_zima_kWh"]
            o[f"{p}|PV kWp"] = w[f"{v}|PV_kWp"]
        o["Qh_nd P1"] = w["V2|Qh_nd"]
        o["scenarij"] = sc
        out.append(o.reset_index())
        if sc == "osnova":                                    # §6 P0 z in brez ETS2 (posameznik R3, diskont 4 %)
            r = EK["diskont_zasebni"]
            q = pd.DataFrame(index=w.index)
            q["arhetip"], q["V0"] = w["arhetip"], w["V0"]
            for rez in ("R1", "R3"):
                for p in ("P0", "P1", "P3"):
                    q[f"{p}|NPV {rez}"] = n[rez][p]
            gor = np.select([w["V0"] == x for x in ("V0a", "V0b", "V0c")],
                            [w[f"{x}|gorivo_kWh"].to_numpy() if f"{x}|gorivo_kWh" in w.columns else 0 for x in ("V0a", "V0b", "V0c")], 0.0)
            ef = w["V0"].map(EF_ETS2).to_numpy(float)
            q["ETS2 SV"] = sum(gor * ef / 1000 * ets2_cena(EK["leto0"] + t) * (1 + EK["ddv"]) * (1 + r) ** (-t) for t in range(1, E.N + 1))
            q["P0|NPV R3 + ETS2"] = q["P0|NPV R3"] + q["ETS2 SV"]
            q["P0|NPV R1 + ETS2"] = q["P0|NPV R1"] + q["ETS2 SV"]
            p0.append(q.reset_index())
    pd.concat(out, ignore_index=True).to_parquet(CACHE / "17_rezultati.parquet", index=False)
    pd.concat(p0, ignore_index=True).to_parquet(CACHE / "17b_P0.parquet", index=False)


KAZ = [("NPV R1", "posameznik NPV R1", "kEUR", 1e3, 1e6, "mio EUR"), ("NPV R3", "posameznik NPV R3", "kEUR", 1e3, 1e6, "mio EUR"),
       ("družbeni NPV", "družbeni NPV", "kEUR", 1e3, 1e6, "mio EUR"), ("CO2e t", "CO2e 30 let", "t", 1, 1e3, "kt"),
       ("zimski uvoz kWh", "zimski uvoz", "kWh/a", 1, 1e6, "GWh/a"), ("NN kW", "obremenitev NN", "kW", 1, 1e3, "MW")]


def korak_tabela():
    r = pd.read_parquet(CACHE / "17_rezultati.parquet")
    rows = []
    for sc in SCEN:
        for obd, ars in OBD.items():
            g = r[(r["scenarij"] == sc) & r["arhetip"].isin(ars)]
            row = {"scenarij": sc, "obdobje": obd, "hiš": len(g), "Qh,nd P1 mediana [kWh/m²a]": g["Qh_nd P1"].median(),
                   "PV P1 / P3 mediana [kWp]": f"{g['P1|PV kWp'].median():.1f} / {g['P3|PV kWp'].median():.1f}"}
            for k, lab, en, sc_, vsc, ven in KAZ:
                d = g[f"P3|{k}"] - g[f"P1|{k}"]
                row[f"{lab}: P3 − P1 mediana [{en}]"] = d.median() / sc_
                row[f"{lab}: delež P3 boljši [%]"] = 100 * (d < 0).mean()
                row[f"{lab}: P3 − P1 vsota [{ven}]"] = d.sum() / vsc
            rows.append(row)
    T = pd.DataFrame(rows)
    q = pd.read_parquet(CACHE / "17b_P0.parquet")
    lab0 = {"V0a": "ELKO (V0a)", "V0b": "biomasa (V0b)", "V0c": "plin (V0c)"}
    r0 = []
    for obd, ars in list(OBD.items())[1:]:
        for v0, g in q[q["arhetip"].isin(ars)].groupby("V0"):
            row = {"obdobje": obd, "nadomeščeni energent": lab0[v0], "hiš": len(g)}
            for k in ("P0|NPV R3", "P0|NPV R3 + ETS2", "P1|NPV R3", "P3|NPV R3", "P0|NPV R1", "P0|NPV R1 + ETS2", "P1|NPV R1", "P3|NPV R1", "ETS2 SV"):
                row[f"{k} mediana [kEUR]"] = g[k].median() / 1e3
            for p in ("P1", "P3"):
                row[f"delež {p} cenejši od P0, R3 brez / z ETS2 [%]"] = f"{100 * (g[f'{p}|NPV R3'] < g['P0|NPV R3']).mean():.0f} / {100 * (g[f'{p}|NPV R3'] < g['P0|NPV R3 + ETS2']).mean():.0f}"
            r0.append(row)
    T0 = pd.DataFrame(r0)
    for o in (TAB, OUT_T):
        T.round(3).to_csv(o / "17_obcutljivost_podnebje_gospodinjstvo_cene.csv", sep=";", encoding="utf-8-sig", index=False)
        T0.round(3).to_csv(o / "17b_P0_ETS2.csv", sep=";", encoding="utf-8-sig", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    pd.set_option("display.max_colwidth", 30)
    kol = ["scenarij", "obdobje"] + [c for c in T.columns if ("mediana" in c and "P3 − P1" in c) or "delež" in c]
    print(T[kol].round(1).to_string(index=False))
    print(T0.round(1).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--korak", default="T")
    a = ap.parse_args()
    {"B": korak_b, "T": korak_tabela}[a.korak]()
