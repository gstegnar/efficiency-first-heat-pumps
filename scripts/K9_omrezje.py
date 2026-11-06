"""Krog 9, §1 — omrežje z omejitvijo oddaje PV (NAVODILA_CLAUDE_CODE_v9.md).

Paket 5.969 (vhodi L3 iz kroga 8 = osnova članka), urni model 03 z oddajo(t) ≤ c · kWp, c ∈ {1,0 (osnova), 0,7, 0,5}; presežek izgubljen.
Teki 03: cache/K9_omrezje.sh (--tag _K9c10 / _K9c07 / _K9c05; pri c < 1 drugi prehod z urami sočasne oddaje iz lastnega teka).
  B  ekonomika (iste funkcije kot 04, K2_1): NN obremenitev P1 / P3, ojačitev, izgubljena energija, družbeni NPV, R1 / R3 → cache/16_rezultati.parquet
  T  tabela clanek1/tabele/16_omrezje_omejitev_oddaje.csv (+ tabele/), vključno s spodnjo mejo »samo uvoz« (d) in izpisom 03b (e)
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
C_TAG = {1.0: "_K9c10", 0.7: "_K9c07", 0.5: "_K9c05"}
OBD = {"paket": ["A1", "A2", "A3", "A4"], "pred 1981": ["A1", "A2", "A3"], "1981–2002": ["A4"]}
GLAVNA = {"P1": "V2", "P3": "V5"}                      # varianta poti v letih 1–30 (P3: V4 le v letu 0)


def _moduli():
    s = importlib.util.spec_from_file_location("k21", HERE / "K2_1_drzava.py")
    K1 = importlib.util.module_from_spec(s)
    s.loader.exec_module(K1)
    return K1, K1.E


def korak_b():
    K1, E = _moduli()
    ojac = PAR["drzava"]["ojacitev_EUR_kW"]
    out = []
    for c, tag in C_TAG.items():
        w = E.load_units("fond", tag)
        n = {rez: E.npv_paths(w, rez, EK["diskont_zasebni"], E.pf_factory("konstantne"), EK["ovoj_osnova"]) for rez in ("R1", "R3")}
        rc = K1.racun(w)
        col = lambda v, q: w[f"{v}|{q}"].to_numpy(float)  # noqa: E731
        uv0 = K1.nn_obremenitev(w, "V0")
        o = pd.DataFrame(index=w.index)
        o["arhetip"] = w["arhetip"]
        for p in ("P1", "P3"):
            for rez in ("R1", "R3"):
                o[f"{p}|NPV {rez}"] = n[rez][p]
            o[f"{p}|družbeni NPV"] = rc[p]["družbeni NPV"]
            o[f"{p}|ojačitev"] = rc[p]["ojačitev omrežja"]
            o[f"{p}|NN kW"] = rc[p]["obremenitev NN omrežja [kW]"]
            # (d) spodnja meja: NN samo po sočasni konici uvoza (oddaja v celoti obvladljiva), najslabše leto poti nad P0
            vs = sorted({E.leto_varianta(p, t) for t in range(1, E.N + 1)})
            nn_u = np.max(np.stack([col(v, "uvoz_konica_dist_kW") for v in vs]), axis=0)
            o[f"{p}|NN uvoz kW"] = nn_u
            o[f"{p}|ojačitev uvoz"] = np.maximum(nn_u - uv0, 0) * ojac
            o[f"{p}|družbeni NPV uvoz"] = o[f"{p}|družbeni NPV"] - o[f"{p}|ojačitev"] + o[f"{p}|ojačitev uvoz"]
            v = GLAVNA[p]
            o[f"{p}|uvoz dist kW"] = col(v, "uvoz_konica_dist_kW")
            o[f"{p}|oddaja dist kW"] = col(v, "oddaja_konica_dist_kW")
            o[f"{p}|PV kWh"] = col(v, "PV_kWh") if f"{v}|PV_kWh" in w.columns else np.nan
            o[f"{p}|izgubljeno kWh"] = col(v, "PV_izgubljeno_kWh") if f"{v}|PV_izgubljeno_kWh" in w.columns else 0.0
            o[f"{p}|izgubljeno EUR"] = col(v, "PV_izgubljeno_EUR") if f"{v}|PV_izgubljeno_EUR" in w.columns else 0.0
            o[f"{p}|zimski uvoz kWh"] = col(v, "uvoz_zima_kWh")
        o["c"] = c
        out.append(o.reset_index())
    r = pd.concat(out, ignore_index=True)
    r.to_parquet(CACHE / "16_rezultati.parquet", index=False)
    print(r.groupby(["c", "arhetip"]).size().unstack())


def flota(tag):
    """Sočasna konica paketa [MW] iz vsote urnih nizov teka (P1 = V2, P3 = V5)."""
    h = pd.read_parquet(CACHE / f"03_fond_urno_vsota{tag}.parquet")
    return {p: (h[f"{v}|uvoz"].max() / 1e6, h[f"{v}|oddaja"].max() / 1e6) for p, v in GLAVNA.items()}


def korak_tabela():
    r = pd.read_parquet(CACHE / "16_rezultati.parquet")
    rows = []

    def vrstica(c, obd, kaz, enota, p1, p3, sc=1.0, vsc=None, ven=None, manj_boljse=True):
        d = (p3 - p1) / sc
        row = {"c [kW/kWp]": c, "obdobje": obd, "kazalnik": kaz, "enota": enota,
               "P1 mediana": np.median(p1) / sc, "P3 mediana": np.median(p3) / sc, "P3 − P1 mediana": np.median(d),
               "delež P3 nižji/boljši [%]": 100 * ((p3 - p1) < 0).mean() if manj_boljse else np.nan}
        if vsc:
            row["P1 vsota"], row["P3 vsota"], row["P3 − P1 vsota"], row["enota vsote"] = p1.sum() / vsc, p3.sum() / vsc, (p3 - p1).sum() / vsc, ven
        rows.append(row)

    for c in C_TAG:
        fl = flota(C_TAG[c])
        for obd, ars in OBD.items():
            g = r[(r["c"] == c) & r["arhetip"].isin(ars)]
            x = lambda p, k: g[f"{p}|{k}"].to_numpy(float)  # noqa: E731
            vrstica(c, obd, "obremenitev NN izvoda max(uvoz, oddaja)", "kW/hišo", x("P1", "NN kW"), x("P3", "NN kW"), vsc=1e3, ven="MW")
            vrstica(c, obd, "ojačitev NN (670 EUR/kW nad P0)", "kEUR/hišo", x("P1", "ojačitev"), x("P3", "ojačitev"), sc=1e3, vsc=1e6, ven="mio EUR")
            vrstica(c, obd, "(d) obremenitev NN samo po uvozu", "kW/hišo", x("P1", "NN uvoz kW"), x("P3", "NN uvoz kW"), vsc=1e3, ven="MW")
            vrstica(c, obd, "(d) ojačitev NN samo po uvozu", "kEUR/hišo", x("P1", "ojačitev uvoz"), x("P3", "ojačitev uvoz"), sc=1e3, vsc=1e6, ven="mio EUR")
            vrstica(c, obd, "sočasna konica uvoza (hiša, 10 h flote)", "kW/hišo", x("P1", "uvoz dist kW"), x("P3", "uvoz dist kW"), vsc=1e3, ven="MW")
            vrstica(c, obd, "sočasna konica oddaje (hiša, 10 h flote)", "kW/hišo", x("P1", "oddaja dist kW"), x("P3", "oddaja dist kW"), vsc=1e3, ven="MW")
            for p in ("P1", "P3"):
                odl = (x(p, "oddaja dist kW") > x(p, "uvoz dist kW")).mean() * 100
                rows.append({"c [kW/kWp]": c, "obdobje": obd, "kazalnik": f"odločilna konica {p}: delež hiš, kjer je oddaja > uvoz", "enota": "%",
                             f"{p} mediana": odl})
            vrstica(c, obd, "izgubljena energija PV (curtailment)", "kWh/a na hišo", x("P1", "izgubljeno kWh"), x("P3", "izgubljeno kWh"), vsc=1e6, ven="GWh/a",
                    manj_boljse=False)
            vrstica(c, obd, "izgubljena energija PV, delež proizvodnje", "%", 100 * x("P1", "izgubljeno kWh") / np.maximum(x("P1", "PV kWh"), 1),
                    100 * x("P3", "izgubljeno kWh") / np.maximum(x("P3", "PV kWh"), 1), manj_boljse=False)
            vrstica(c, obd, "vrednost izgubljene energije (urna cena dan vnaprej 2025)", "EUR/a na hišo", x("P1", "izgubljeno EUR"), x("P3", "izgubljeno EUR"),
                    vsc=1e6, ven="mio EUR/a", manj_boljse=False)
            vrstica(c, obd, "družbeni NPV", "kEUR/hišo", x("P1", "družbeni NPV"), x("P3", "družbeni NPV"), sc=1e3, vsc=1e6, ven="mio EUR")
            vrstica(c, obd, "(d) družbeni NPV, NN samo po uvozu", "kEUR/hišo", x("P1", "družbeni NPV uvoz"), x("P3", "družbeni NPV uvoz"), sc=1e3, vsc=1e6, ven="mio EUR")
            vrstica(c, obd, "posameznik NPV R1", "kEUR/hišo", x("P1", "NPV R1"), x("P3", "NPV R1"), sc=1e3, vsc=1e6, ven="mio EUR")
            vrstica(c, obd, "posameznik NPV R3", "kEUR/hišo", x("P1", "NPV R3"), x("P3", "NPV R3"), sc=1e3, vsc=1e6, ven="mio EUR")
            vrstica(c, obd, "zimski uvoz", "kWh/a na hišo", x("P1", "zimski uvoz kWh"), x("P3", "zimski uvoz kWh"), vsc=1e6, ven="GWh/a")
        rows.append({"c [kW/kWp]": c, "obdobje": "paket", "kazalnik": "sočasna konica paketa (vsota urnih nizov): uvoz", "enota": "MW",
                     "P1 mediana": fl["P1"][0], "P3 mediana": fl["P3"][0], "P3 − P1 mediana": fl["P3"][0] - fl["P1"][0]})
        rows.append({"c [kW/kWp]": c, "obdobje": "paket", "kazalnik": "sočasna konica paketa (vsota urnih nizov): oddaja", "enota": "MW",
                     "P1 mediana": fl["P1"][1], "P3 mediana": fl["P3"][1], "P3 − P1 mediana": fl["P3"][1] - fl["P1"][1]})
    T = pd.DataFrame(rows)
    T = pd.concat([T, izpis_03b()], ignore_index=True)
    for o in (TAB, OUT_T):
        T.round(3).to_csv(o / "16_omrezje_omejitev_oddaje.csv", sep=";", encoding="utf-8-sig", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    pd.set_option("display.max_colwidth", 60)
    print(T.round(2).to_string(index=False))
    return T


def izpis_03b():
    """(e) Krog 1 (03b, reprezentativne hiše): koliko optimirano upravljanje in hranilniki zmanjšajo zimski uvoz in konico uvoza, brez ovoja in po FSO.
    Brez novih izračunov. Mreža 03b (12,5 kWp, brez EV) obstaja le za A3 in A4; za A1–A4 baterija 10 kWh iz 03b_prihranek_optimiranja (V2/V3, V5/V6)."""
    rows = []
    m = pd.read_csv(TAB / "03b_mreza.csv", sep=";", encoding="utf-8-sig")
    m = m[(m["kWp"] == 12.5) & (~m["EV"].astype(str).eq("True")) & m["ovoj"].isin(["BREZ", "FSO"])]
    komb = [("brez hranilnikov", 0, 0), ("baterija 10 kWh", 10, 0), ("hranilnik toplote 800 L", 0, 800), ("baterija 10 kWh + hranilnik 800 L", 10, 800)]
    for a in ("A3", "A4"):
        for ov in ("BREZ", "FSO"):
            g = m[(m["arhetip"] == a) & (m["ovoj"] == ov)]
            ref = g[(g["cilj"] == "pravila") & (g["baterija_kWh"] == 0) & (g["hranilnik_L"] == 0)].iloc[0]
            for lab, b, h in komb:
                s = g[(g["baterija_kWh"] == b) & (g["hranilnik_L"] == h)]
                o1, o2 = s[s["cilj"] == "O1"].iloc[0], s[s["cilj"] == "O2"].iloc[0]
                pr = s[s["cilj"] == "pravila"].iloc[0]
                rows.append({"obdobje": "(e) 03b krog 1, 12,5 kWp", "kazalnik": f"{a} {'brez ovoja' if ov == 'BREZ' else 'FSO'}: {lab}", "enota": "kWh / kW",
                             "zimski uvoz pravila brez hranilnikov [kWh]": ref["uvoz_zima_kWh"], "zimski uvoz pravila [kWh]": pr["uvoz_zima_kWh"],
                             "zimski uvoz O1 [kWh]": o1["uvoz_zima_kWh"], "znižanje zimskega uvoza O1 [%]": 100 * (1 - o1["uvoz_zima_kWh"] / ref["uvoz_zima_kWh"]),
                             "konica uvoza pravila brez hranilnikov [kW]": ref["uvoz_konica_kW"], "konica uvoza O2 [kW]": o2["uvoz_konica_kW"],
                             "znižanje konice O2 [%]": 100 * (1 - o2["uvoz_konica_kW"] / ref["uvoz_konica_kW"])})
    po = pd.read_csv(TAB / "03b_prihranek_optimiranja.csv", sep=";", encoding="utf-8-sig")
    po = po[~po["EV"].astype(str).eq("True")]
    for a in ("A1", "A2", "A3", "A4"):
        for ov, v0, vb in (("BREZ", "V2", "V3"), ("FSO", "V5", "V6")):
            g = po[po["arhetip"] == a].set_index("varianta")
            ref = g.loc[v0]
            for lab, v in (("brez hranilnikov", v0), ("baterija 10 kWh", vb)):
                x = g.loc[v]
                rows.append({"obdobje": "(e) 03b krog 1, PV paketa", "kazalnik": f"{a} {'brez ovoja' if ov == 'BREZ' else 'FSO'}: {lab}", "enota": "kWh / kW",
                             "zimski uvoz pravila brez hranilnikov [kWh]": ref["zimski uvoz pravila [kWh]"], "zimski uvoz pravila [kWh]": x["zimski uvoz pravila [kWh]"],
                             "zimski uvoz O1 [kWh]": x["zimski uvoz O1 [kWh]"],
                             "znižanje zimskega uvoza O1 [%]": 100 * (1 - x["zimski uvoz O1 [kWh]"] / ref["zimski uvoz pravila [kWh]"]),
                             "konica uvoza pravila brez hranilnikov [kW]": ref["konica pravila [kW]"], "konica uvoza O2 [kW]": x["konica O2 [kW]"],
                             "znižanje konice O2 [%]": 100 * (1 - x["konica O2 [kW]"] / ref["konica pravila [kW]"])})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--korak", default="T")
    a = ap.parse_args()
    {"B": korak_b, "T": korak_tabela}[a.korak]()
