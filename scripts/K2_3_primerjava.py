"""Krog 2, točka 3 — enotna primerjalna tabela »OVE najprej« (P1, P2) proti »URE najprej« (P3, P4): posameznik, država, okolje.

Za vsako hišo in pot P1–P4 izračuna kazalnike treh zornih kotov iz obstoječih izračunov (brez novih predpostavk):
  posameznik: NPV stroškov 30 let R1–R3 (04, 4 %); investicije v ukrepe nominalno (brez zamenjav), spodbude, lastna udeležba;
              letni račun za elektriko (+ gorivo) v letu 2 (P3: PV vgrajen v letu 1) in letu 20 (R1, R3)
  država:     javni proračun NPV R1, R3 in družbeni NPV (K2_1, 3 %); spodbude nominalno; zimski uvoz in prispevek h sistemski
              konici v letu 20; strošek zmogljivosti in ojačitve omrežja (SV)
  okolje:     CO2e 30 let (obratovalni + vgrajeni, 06); električna raba in uvoz iz omrežja 30 let
Po vzorcih paket 5.969 / Q1 / Q2 / fond: mediana na hišo po poteh, P3−P1 in P3−P2 (mediana, P25–P75, delež P3 < P1/P2), vsote.
P3−P2 je primerjava »URE najprej« z »OVE najprej, ovoj pozneje« (kot v kalkulatorju NPS 2050), P3−P1 z »ovoja nikoli«.
Pdej = »OVE najprej, dejansko«: (1 − F)·P1 + F·P2, F = empirična verjetnost prve prenove ovoja v 20 letih po paketu TČ+PV
(K2_4_vzrocnost.py, cache/11_verjetnost_ovoja.json, po razredu Q1/Q2; P2 kot predstavnik »ovoj pozneje«).
Zadnje vrstice: kateri pristop je boljši in prelomni Qh,nd (kjer delež hiš s P3 < P1 preseže 50 %; fond, drseči kvantilni razredi).
Izhodi: clanek1/tabele/09_primerjava_pristopov.csv, clanek1/slike/Fig4_three_perspectives(_sl).png/svg (prejšnja → *_krog1)
"""
from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT_T, OUT_S, CACHE = HERE / "clanek1" / "tabele", HERE / "clanek1" / "slike", HERE / "cache"
POTI = ["P1", "P2", "P3", "P4", "P1r"]


def load_mod(name, f):
    s = importlib.util.spec_from_file_location(name, HERE / f)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


K1 = load_mod("k21", "K2_1_drzava.py")
E = K1.E
EK, N = E.EK, E.N


def investicije_nom(w, p):
    """Nominalna bruto investicija v ukrepe poti (TČ, PV, baterija, ovoj; brez zamenjav TČ, razsmernika, baterije, MVHR)."""
    kW = {"e": w["V1|TC_kW"].to_numpy(float), "FSO": w["V4|TC_kW"].to_numpy(float), "GLOB": w["V7|TC_kW"].to_numpy(float), "r": w["V2r|TC_kW"].to_numpy(float)}
    inv = np.zeros(len(w))
    for (y, item, arg) in E.POTI[p]["ev"]:
        if item == "TC":
            inv += EK["TC_a_EUR"] + EK["TC_b_EUR_kW"] * kW[arg]
        elif item in ("PV", "PVB"):
            k = w[f"{arg}|PV_kWp"].to_numpy(float)
            inv += np.where(k > 0, EK["PV_a_EUR"] + EK["PV_b_EUR_kWp"] * k, 0)
        elif item == "BAT":
            inv += EK["baterija_EUR_kWh"] * w["V6|baterija_kWh"].to_numpy(float)
        elif item in ("FSO", "GLOB"):
            inv += E.envelope_cost(w, item, EK["ovoj_osnova"])[0]
    return inv


def kazalniki(w):
    """{(perspektiva, kazalnik, enota na hišo, faktor za vsoto, enota vsote): DataFrame[P1..P4] na hišo}."""
    K = {}
    npv = pd.read_parquet(CACHE / "04_npv_fond.parquet").set_index("ID_Stavbe")
    for rez in ("R1", "R2", "R3"):
        K[("posameznik", f"NPV stroškov 30 let, {rez}", "kEUR", 1e3, 1e6, "mio EUR")] = npv[npv["rezim"] == rez][POTI].reindex(w.index)
    sp_nom, _ = E.spodbude_poti(w, basis=EK["ovoj_osnova"])
    inv = pd.DataFrame({p: investicije_nom(w, p) for p in POTI}, index=w.index)
    K[("posameznik", "investicija v ukrepe brez spodbud (nominalno)", "kEUR", 1e3, 1e6, "mio EUR")] = inv
    K[("posameznik", "lastna udeležba (investicija − spodbude)", "kEUR", 1e3, 1e6, "mio EUR")] = inv - sp_nom[POTI]
    for rez in ("R1", "R3"):
        for t in (2, 20):              # leto 2 = prvo leto, ko ima P3 vse ukrepe (PV v letu 1)
            K[("posameznik", f"letni račun za energijo, leto {t}, {rez}", "EUR/a", 1, 1e6, "mio EUR/a")] = pd.DataFrame(
                {p: E.annual_bill(w, E.leto_varianta(p, t), rez, 1.0) for p in POTI}, index=w.index)
    res = K1.racun(w)
    for k, lab in (("R1 proračun neto", "javni proračun NPV, R1"), ("R3 proračun neto", "javni proračun NPV, R3"),
                   ("družbeni NPV", "družbeni NPV"), ("družbeni NPV, zima ×f", "družbeni NPV, zimske cene ×1,5"),
                   ("družbeni NPV, CCGT", "družbeni NPV, zmogljivost CCGT")):
        K[("država", lab, "kEUR", 1e3, 1e6, "mio EUR")] = pd.DataFrame({p: res[p][k] for p in POTI})
    K[("država", "spodbude (nominalno)", "kEUR", 1e3, 1e6, "mio EUR")] = sp_nom[POTI]
    v20 = {p: E.leto_varianta(p, 20) for p in POTI}
    K[("država", "zimski uvoz (nov–feb), leto 20", "kWh/a", 1, 1e6, "GWh/a")] = pd.DataFrame({p: w[f"{v20[p]}|uvoz_zima_kWh"] for p in POTI})
    K[("država", "prispevek h sistemski konici, leto 20", "kW", 1, 1e3, "MW")] = pd.DataFrame({p: w[f"{v20[p]}|uvoz_konica_sis_kW"] for p in POTI})
    K[("država", "obremenitev NN omrežja (maks. sočasni uvoz / oddaja PV)", "kW", 1, 1e3, "MW")] = pd.DataFrame(
        {p: res[p]["obremenitev NN omrežja [kW]"] for p in POTI})
    K[("država", "strošek zmogljivosti in ojačitve omrežja (SV)", "kEUR", 1e3, 1e6, "mio EUR")] = pd.DataFrame(
        {p: res[p]["zmogljivost"] + res[p]["ojačitev omrežja"] for p in POTI})
    c6 = pd.read_parquet(CACHE / "06_co2_poti_fond.parquet").set_index("ID_Stavbe").reindex(w.index)
    for k, lab in (("skupaj_t", "CO2e 30 let (obratovalni + vgrajeni)"), ("obratovalni_t", "CO2e 30 let, obratovalni"), ("vgrajeni_t", "CO2e, vgrajeni")):
        K[("okolje", lab, "t", 1, 1e3, "kt")] = pd.DataFrame({p: c6[f"{p}|{k}"] for p in POTI})
    for q, lab in (("el_skupaj_kWh", "končna raba elektrike 30 let (TČ + gospodinjstvo)"), ("uvoz_kWh", "uvoz iz omrežja 30 let")):
        K[("okolje", lab, "MWh", 1e3, 1e6, "GWh")] = pd.DataFrame(
            {p: sum(w[f"{E.leto_varianta(p, t)}|{q}"].to_numpy(float) for t in range(1, N + 1)) for p in POTI}, index=w.index)
    return K


def prelom(qh, d):
    """Qh,nd, pri katerem delež hiš s P3 < P1 preseže 50 % (20 kvantilnih razredov Qh,nd, linearna interpolacija)."""
    df = pd.DataFrame({"q": qh, "b": (d < 0).astype(float)}).dropna()
    if len(df) < 20:                                         # krog 13: prazen ali premajhen podvzorec (npr. brez ukrepa na ovoju pri umestitvi po registru)
        return "premalo hiš"
    df["k"] = pd.qcut(df["q"], 20, duplicates="drop")
    g = df.groupby("k", observed=True).agg(q=("q", "median"), s=("b", "mean"))
    x, s = g["q"].to_numpy(), g["s"].to_numpy()
    if (s >= 0.5).all():
        return f"P3 boljši v vseh razredih (≥ {x[0]:.0f})"
    if (s < 0.5).all():
        return f"P1 boljši v vseh razredih (do {x[-1]:.0f})"
    i = np.where(np.diff(np.sign(s - 0.5)) != 0)[0]
    out = []
    for j in i:
        xc = x[j] + (0.5 - s[j]) * (x[j + 1] - x[j]) / (s[j + 1] - s[j])
        out.append(f"{xc:.0f}" + (" (P3 boljši nad)" if s[j + 1] > s[j] else " (P3 boljši pod)"))
    return "; ".join(dict.fromkeys(out))


def main():
    w = E.load_units("fond")
    meta = pd.read_parquet(CACHE / "01_fond.parquet", columns=["ID_Stavbe", "Qr", "paket_osnovni", "arhetip", "tip_tc"]).set_index("ID_Stavbe").reindex(w.index)
    pk, q = meta["paket_osnovni"].astype(bool).to_numpy(), pd.Series(np.where(meta["arhetip"].isin(["A1", "A2", "A3"]), "Q1", "Q2"), index=meta.index).to_numpy()
    VZ = {"paket": pk, "paket pred 1981": pk & (q == "Q1"), "paket 1981–2002": pk & (q == "Q2"),
          "fond pred 1981": q == "Q1", "fond 1981–2002": q == "Q2", "fond": np.ones(len(w), bool)}
    # krog 3 (C2): podvzorci paketa po simuliranem Qh,nd hiše in po prevladujočem stanju na lestvici; (D2) brez TČ zemlja-voda / voda-voda
    qh_h = w["V2|Qh_nd"].to_numpy(float)
    VZ.update({"paket Qh > 140": pk & (qh_h > 140), "paket Qh 90–140": pk & (qh_h >= 90) & (qh_h <= 140), "paket Qh < 90": pk & (qh_h < 90)})
    if "V4|lestvica_k" in w.columns:
        st_h = np.minimum(w["V4|lestvica_k"].to_numpy().astype(int) + (w["V4|lestvica_w"].to_numpy(float) >= 0.5), 3)
        VZ.update({f"paket S{j}": pk & (st_h == j) for j in range(4)})
        VZ["paket brez ukrepa na ovoju v P3"] = pk & (w["V4|brez_ovoja_P3"].to_numpy(float) > 0.5)
    VZ["paket brez TČ zemlja/voda-voda"] = pk & ~meta["tip_tc"].astype(str).str.contains("zemlja|tcvv|voda", case=False).to_numpy()
    qh = w["V2|Qh_nd"].to_numpy(float)
    K = kazalniki(w)
    # »OVE najprej, dejansko«: mešanica P1/P2 po empirični verjetnosti prenove ovoja v 20 letih (točka 4)
    fj = CACHE / "11_verjetnost_ovoja.json"
    VER = json.loads(fj.read_text(encoding="utf-8")) if fj.exists() else None
    if VER:
        F = np.where(q == "Q1", VER["Q1"]["F20"], np.where(q == "Q2", VER["Q2"]["F20"], VER["vse"]["F20"]))
        for key in K:
            df = K[key].reindex(w.index)
            df["Pdej"] = (1 - F) * df["P1"].to_numpy(float) + F * df["P2"].to_numpy(float)
            K[key] = df
    rows = []
    for (persp, kaz, en, sc, vsc, ven), df in K.items():
        df = df.reindex(w.index)
        d = (df["P3"] - df["P1"]).to_numpy(float)
        d2 = (df["P3"] - df["P2"]).to_numpy(float)
        for vz, m in VZ.items():
            row = {"perspektiva": persp, "kazalnik": kaz, "vzorec": vz, "stavb": int(m.sum()), "enota na hišo": en}
            for p in POTI:
                row[f"{p} mediana"] = np.nanmedian(df[p].to_numpy(float)[m]) / sc
            dm = d[m]
            row.update({"P3−P1 mediana": np.nanmedian(dm) / sc, "P3−P1 P25": np.nanquantile(dm, 0.25) / sc, "P3−P1 P75": np.nanquantile(dm, 0.75) / sc,
                        "P4−P1 mediana": np.nanmedian((df["P4"] - df["P1"]).to_numpy(float)[m]) / sc,
                        "delež P3 < P1 [%]": 100 * np.mean(dm < 0),
                        "P3−P2 mediana": np.nanmedian(d2[m]) / sc, "P3−P2 P25": np.nanquantile(d2[m], 0.25) / sc,
                        "P3−P2 P75": np.nanquantile(d2[m], 0.75) / sc, "delež P3 < P2 [%]": 100 * np.mean(d2[m] < 0), "enota vsote": ven})
            for p in POTI:
                row[f"{p} vsota"] = np.nansum(df[p].to_numpy(float)[m]) / vsc
            row["P3−P1 vsota"] = np.nansum(dm) / vsc
            row["P3−P2 vsota"] = np.nansum(d2[m]) / vsc
            # krog 3: P1r = »OVE najprej, pravilno dimenzionirano«; P3 − P1 = (P1r − P1) učinek dimenzioniranja + (P3 − P1r) učinek ovoja
            dr_ = (df["P1r"] - df["P1"]).to_numpy(float)[m]
            de_ = (df["P3"] - df["P1r"]).to_numpy(float)[m]
            row.update({"P1r−P1 mediana": np.nanmedian(dr_) / sc, "delež P1r < P1 [%]": 100 * np.mean(dr_ < 0), "P1r−P1 vsota": np.nansum(dr_) / vsc,
                        "P3−P1r mediana": np.nanmedian(de_) / sc, "P3−P1r P25": np.nanquantile(de_, 0.25) / sc, "P3−P1r P75": np.nanquantile(de_, 0.75) / sc,
                        "delež P3 < P1r [%]": 100 * np.mean(de_ < 0), "P3−P1r vsota": np.nansum(de_) / vsc})
            if "Pdej" in df:
                dd = (df["P3"] - df["Pdej"]).to_numpy(float)[m]
                row.update({"Pdej mediana": np.nanmedian(df["Pdej"].to_numpy(float)[m]) / sc, "P3−Pdej mediana": np.nanmedian(dd) / sc,
                            "P3−Pdej P25": np.nanquantile(dd, 0.25) / sc, "P3−Pdej P75": np.nanquantile(dd, 0.75) / sc,
                            "delež P3 < Pdej [%]": 100 * np.mean(dd < 0), "Pdej vsota": np.nansum(df["Pdej"].to_numpy(float)[m]) / vsc,
                            "P3−Pdej vsota": np.nansum(dd) / vsc})
            rows.append(row)
    T = pd.DataFrame(rows)
    # zadnja vrstica po perspektivi: boljši pristop in prelomni Qh,nd
    odl = {"posameznik": [("NPV stroškov 30 let, R1", "kEUR"), ("NPV stroškov 30 let, R3", "kEUR")],
           "država": [("javni proračun NPV, R3", "kEUR"), ("družbeni NPV", "kEUR")],
           "okolje": [("CO2e 30 let (obratovalni + vgrajeni)", "t")]}
    zad = []
    for persp, items in odl.items():
        for kaz, _ in items:
            key = next(k for k in K if k[0] == persp and k[1] == kaz)
            df = K[key].reindex(w.index)
            d = (df["P3"] - df["P1"]).to_numpy(float)
            d2 = (df["P3"] - df["P2"]).to_numpy(float)
            for vz, m in VZ.items():
                s = 100 * np.mean(d[m] < 0)
                s2 = 100 * np.mean(d2[m] < 0)
                zad.append({"perspektiva": persp, "kazalnik": f"BOLJŠI PRISTOP — {kaz}", "vzorec": vz, "stavb": int(m.sum()),
                            "enota na hišo": "besedilo", "delež P3 < P1 [%]": s,
                            "odločitev": ("URE najprej (P3)" if s >= 50 else "OVE najprej (P1)") + f" pri {max(s, 100 - s):.0f} % hiš; "
                                         f"vsota P3−P1 {np.nansum(d[m]) / key[4]:+.1f} {key[5]}",
                            "prelomni Qh,nd [kWh/m²a]": prelom(qh[m], d[m]), "delež P3 < P2 [%]": s2,
                            "odločitev proti P2": ("URE najprej (P3)" if s2 >= 50 else "OVE najprej, ovoj pozneje (P2)") + f" pri {max(s2, 100 - s2):.0f} % hiš; "
                                                  f"vsota P3−P2 {np.nansum(d2[m]) / key[4]:+.1f} {key[5]}",
                            "prelomni Qh,nd proti P2 [kWh/m²a]": prelom(qh[m], d2[m]),
                            "delež P3 < P1r [%]": 100 * np.mean((df["P3"] - df["P1r"]).to_numpy(float)[m] < 0),
                            "odločitev proti P1r": ("URE najprej (P3)" if np.mean((df["P3"] - df["P1r"]).to_numpy(float)[m] < 0) >= 0.5
                                                    else "OVE najprej, pravilno dimenzionirano (P1r)")
                            + f" pri {100 * max(np.mean((df['P3'] - df['P1r']).to_numpy(float)[m] < 0), np.mean((df['P3'] - df['P1r']).to_numpy(float)[m] >= 0)):.0f} % hiš; "
                            f"vsota P3−P1r {np.nansum((df['P3'] - df['P1r']).to_numpy(float)[m]) / key[4]:+.1f} {key[5]}",
                            "prelomni Qh,nd proti P1r [kWh/m²a]": prelom(qh[m], (df["P3"] - df["P1r"]).to_numpy(float)[m]),
                            **({} if "Pdej" not in df else {
                                "delež P3 < Pdej [%]": 100 * np.mean((df["P3"] - df["Pdej"]).to_numpy(float)[m] < 0),
                                "odločitev proti Pdej": ("URE najprej (P3)" if np.mean((df["P3"] - df["Pdej"]).to_numpy(float)[m] < 0) >= 0.5
                                                         else "OVE najprej, dejansko (Pdej)")
                                + f" pri {100 * max(np.mean((df['P3'] - df['Pdej']).to_numpy(float)[m] < 0), np.mean((df['P3'] - df['Pdej']).to_numpy(float)[m] >= 0)):.0f} % hiš; "
                                f"vsota P3−Pdej {np.nansum((df['P3'] - df['Pdej']).to_numpy(float)[m]) / key[4]:+.1f} {key[5]}",
                                "prelomni Qh,nd proti Pdej [kWh/m²a]": prelom(qh[m], (df["P3"] - df["Pdej"]).to_numpy(float)[m])})})
    T = pd.concat([T, pd.DataFrame(zad)], ignore_index=True)
    T.round(3).to_csv(OUT_T / "09_primerjava_pristopov.csv", sep=";", encoding="utf-8-sig", index=False)
    slika(T)
    pd.set_option("display.width", 260)
    pd.set_option("display.max_colwidth", 70)
    for vz in ("paket", "paket pred 1981", "paket 1981–2002", "fond"):
        print(f"\n=== {vz}")
        t = T[(T["vzorec"] == vz) & (T["enota na hišo"] != "besedilo")]
        print(t[["perspektiva", "kazalnik", "enota na hišo", "P1 mediana", "P2 mediana", "P3 mediana", "P4 mediana", "P3−P1 mediana",
                 "P3−P1 P25", "P3−P1 P75", "delež P3 < P1 [%]", "P3−P1 vsota", "P3−P2 mediana", "delež P3 < P2 [%]", "P3−P2 vsota",
                 "P3−Pdej mediana", "delež P3 < Pdej [%]", "P3−Pdej vsota", "enota vsote"]].round(2).to_string(index=False))
    print()
    print(T[T["enota na hišo"] == "besedilo"][["kazalnik", "vzorec", "odločitev", "prelomni Qh,nd [kWh/m²a]", "odločitev proti P2",
                                               "prelomni Qh,nd proti P2 [kWh/m²a]", "odločitev proti Pdej"]].to_string(index=False))


def _stevila():
    """Krog 13: število hiš za oznake slik iz 01_fond (paket, pred 1981, 1981–2002, fond) — ne na trdo."""
    f = pd.read_parquet(Path(__file__).resolve().parent / "cache" / "01_fond.parquet", columns=["paket_osnovni", "arhetip"])
    pk, pre = f["paket_osnovni"].astype(bool), f["arhetip"].isin(["A1", "A2", "A3"])
    return {"paket": int(pk.sum()), "pred": int((pk & pre).sum()), "po": int((pk & ~pre).sum()), "fond": len(f)}


def _en(n):
    return f"{n:,}"


def _sl(n):
    return f"{n:,}".replace(",", ".")


def slika(T):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
    N = _stevila()
    VZ = [("paket", INK, "s", (f"HP+PV package ({_en(N['paket'])})", f"paket TČ+PV ({_sl(N['paket'])})")),
          ("paket pred 1981", "#eb6834", "o", (f"package pre-1981 ({_en(N['pred'])})", f"paket pred 1981 ({_sl(N['pred'])})")),
          ("paket 1981–2002", "#2a78d6", "D", (f"package 1981–2002 ({_en(N['po'])})", f"paket 1981–2002 ({_sl(N['po'])})"))]
    PAN = [("posameznik", ("Household: NPV of costs, 30 y [€ thousand]", "Posameznik: NPV stroškov, 30 let [tisoč €]"),
            [("NPV stroškov 30 let, R1", ("R1 net metering", "R1 neto meritev")), ("NPV stroškov 30 let, R2", ("R2 hourly settlement", "R2 urni obračun")),
             ("NPV stroškov 30 let, R3", ("R3 5-block tariff", "R3 5 blokov")), ("lastna udeležba (investicija − spodbude)", ("own investment", "lastna udeležba"))]),
           ("država", ("State: present value [€ thousand]", "Država: sedanja vrednost [tisoč €]"),
            [("javni proračun NPV, R1", ("public budget, R1", "javni proračun, R1")), ("javni proračun NPV, R3", ("public budget, R3", "javni proračun, R3")),
             ("družbeni NPV", ("social NPV", "družbeni NPV")), ("družbeni NPV, zimske cene ×1,5", ("social, winter prices ×1.5", "družbeni, zimske cene ×1,5")),
             ("družbeni NPV, zmogljivost CCGT", ("social, capacity CCGT", "družbeni, zmogljivost CCGT"))]),
           ("okolje", ("Environment: CO$_2$e, 30 y [t]", "Okolje: CO$_2$e, 30 let [t]"),
            [("CO2e 30 let (obratovalni + vgrajeni)", ("total", "skupaj")), ("CO2e 30 let, obratovalni", ("operational", "obratovalni")),
             ("CO2e, vgrajeni", ("embodied", "vgrajeni"))])]
    for lang, sfx in (("en", ""), ("sl", "_sl")):
        L = 0 if lang == "en" else 1
        fig, axs = plt.subplots(1, 3, figsize=(11.5, 4.4), gridspec_kw={"width_ratios": [1.1, 1.1, 1.0]})
        for ax, (persp, xl, items) in zip(axs, PAN):
            ys = np.arange(len(items))[::-1]
            for yi, (kaz, lab) in zip(ys, items):
                for j, (vz, c, mk, _) in enumerate(VZ):
                    r = T[(T["perspektiva"] == persp) & (T["kazalnik"] == kaz) & (T["vzorec"] == vz)].iloc[0]
                    yy = yi + 0.22 - 0.22 * j
                    ax.plot([r["P3−P1 P25"], r["P3−P1 P75"]], [yy, yy], color=c, lw=1.6, alpha=0.55, solid_capstyle="round")
                    ax.plot(r["P3−P1 mediana"], yy, mk, color=c, ms=6, mec="white", mew=0.8, zorder=3)
                    ax.plot(r["P3−P2 mediana"], yy, mk, mfc="none", mec=c, ms=6, mew=1.2, zorder=3)
                    if "P3−P1r mediana" in r and pd.notna(r["P3−P1r mediana"]):
                        ax.plot(r["P3−P1r mediana"], yy, "x", color=c, ms=6.5, mew=1.6, zorder=4)
                    if j == 0:
                        ax.text(1.03, yi + 0.22, f"{r['delež P3 < P1 [%]']:.0f} %", transform=ax.get_yaxis_transform(), ha="left", clip_on=False, va="center",
                                fontsize=7, color=MUTED)
            ax.axvline(0, color=MUTED, lw=0.9)
            ax.set_yticks(ys)
            ax.set_yticklabels([lab[L] for _, lab in items], fontsize=7.5)
            ax.set_xlabel(xl[L] + "\nP3 − P1", fontsize=8)
            ax.grid(axis="y", visible=False)
            ax.grid(axis="x", color=GRID, lw=0.6)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            ax.text(0.0, 1.01, "← " + ("envelope first" if lang == "en" else "URE najprej"), transform=ax.transAxes, fontsize=7, color="#008300", va="bottom")
            ax.text(1.0, 1.01, ("HP+PV first" if lang == "en" else "OVE najprej") + " →", transform=ax.transAxes, fontsize=7, color="#e34948", va="bottom", ha="right")
            ax.text(1.03, 1.01, "P3 < P1", transform=ax.transAxes, fontsize=6.8, color=MUTED, va="bottom", ha="left")
            ax.set_ylim(-0.6, len(items) - 0.45)
        from matplotlib.lines import Line2D
        h = [Line2D([], [], color=c, marker=mk, ls="-", lw=1.6, ms=6, label=lab[L]) for _, c, mk, lab in VZ]
        h += [Line2D([], [], color=MUTED, marker="o", ls="", ms=6, label="P3 − P1 (HP+PV, no envelope)" if lang == "en" else "P3 − P1 (TČ+PV, ovoja nikoli)"),
              Line2D([], [], color=MUTED, marker="o", mfc="none", ls="", ms=6, mew=1.2,
                     label="P3 − P2 (envelope in year 10), median" if lang == "en" else "P3 − P2 (ovoj v letu 10), mediana"),
              Line2D([], [], color=MUTED, marker="x", ls="", ms=6.5, mew=1.6,
                     label="P3 − P1r (right-sized HP+PV, no envelope), median" if lang == "en" else "P3 − P1r (pravilno dimenzionirana TČ+PV, brez ovoja), mediana")]
        fig.legend(handles=h, loc="lower center", ncol=3, fontsize=7.2, frameon=False)
        fig.text(0.995, 0.135, "median and P25–P75 of P3 − P1; % = share of package houses with P3 < P1" if lang == "en"
                 else "mediana in P25–P75 razlike P3 − P1; % = delež hiš paketa, kjer je P3 < P1", ha="right", fontsize=6.8, color=MUTED)
        fig.tight_layout(rect=(0, 0.16, 1, 1), w_pad=2.5)
        for ext in ("png", "svg"):
            fn = OUT_S / f"Fig4_three_perspectives{sfx}.{ext}"
            bak = OUT_S / f"Fig4_three_perspectives_krog1{sfx}.{ext}"
            if fn.exists() and not bak.exists():
                shutil.copy2(fn, bak)
            fig.savefig(fn, dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    main()
