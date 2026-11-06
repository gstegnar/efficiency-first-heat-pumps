"""Korak 01 — tipična hiša Q1 in Q2 iz analize v3 (07_stavbe_paket) + populacijske uteži.

Izhodi: tabele/01_tipicna_hisa.csv (dolga oblika), tabele/01_tipicna_hisa_pregled.csv (široka),
        tabele/01_populacija.csv, tabele/01_novi_letno.csv
Zagon:  "C:/Anaconda/envs/new_env/python.exe" 01_tipicna_hisa.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
PAR = yaml.safe_load((HERE / "parametri.yaml").read_text(encoding="utf-8"))
ANALIZA = (HERE / PAR["poti"]["analiza"]).resolve()
sys.path.insert(0, str(ANALIZA))
import common as C  # noqa: E402  (analiza v3 — uvozi, ne kopiraj)

TH = PAR["tipicna_hisa"]
TAB = HERE / "tabele"
TAB.mkdir(exist_ok=True)

d = C.load_parquet("07_stavbe_paket.parquet") if hasattr(C, "load_parquet") else pd.read_parquet(C.CACHE / "07_stavbe_paket.parquet")
for c in ["brez_ovoja", "brez_ovoja_ob_tc", "paket_osnovni", "ima_PV", "ima_prezrac", "tc_kW_imputirano", "obdobje_imputirano"]:
    d[c] = d[c].astype(float).fillna(0).astype(bool)   # nullable -> bool (past iz analize: NA lomi maske)
# krog 13 (Gašper 22. 10. 2026): prenova v GURS pred 2010 ni ukrep URE → brez_ovoja, stopnja ob vgradnji, Qh,nd ob vgradnji in razred Qr na novo
# s pragom GURS ≥ fond.gurs_min_leto (analiza v3: 2003; ista logika kot 02_gurs.py in 04_energija.py, analiza ostane samo za branje).
GMIN = PAR["fond"].get("gurs_min_leto", 2003)
if GMIN != 2003:
    _y = lambda c: pd.to_numeric(d[c], errors="coerce")                                                  # noqa: E731
    _t = _y("tc_leto")
    komb = pd.DataFrame({"fasada": (_y("n_fasada") > 0) | (_y("LETO_OBN_FASADE") >= GMIN), "okna": (_y("n_okna_vrata") > 0) | (_y("okna_obn_max") >= GMIN),
                         "streha": (_y("n_streha") > 0) | (_y("LETO_OBN_STREHE") >= GMIN)})
    komb_at = pd.DataFrame({"fasada": (_y("leto_prvi_fasada") <= _t) | (_y("LETO_OBN_FASADE").between(GMIN, 3000) & (_y("LETO_OBN_FASADE") <= _t)),
                            "okna": (_y("leto_prvi_okna_vrata") <= _t) | (_y("okna_obn_min").between(GMIN, 3000) & (_y("okna_obn_min") <= _t)),
                            "streha": (_y("leto_prvi_streha") <= _t) | (_y("LETO_OBN_STREHE").between(GMIN, 3000) & (_y("LETO_OBN_STREHE") <= _t))})
    st_ob = komb_at.sum(axis=1).astype(int).clip(0, 3)
    st_ob[d["celovita_prenova"].astype(float).fillna(0).astype(bool) & (_y("leto_prvi_celovita_prenova") <= _t)] = 3
    QK = C.load_qhnd_kal()
    tipg = d["tip_stavbe"].map(C.tip_group)
    qh_ob = pd.Series([QK.get((int(o), int(s), t), C.qhnd(int(o), int(s))) if pd.notna(o) else np.nan for o, s, t in zip(d["obdobje"], st_ob, tipg)], index=d.index)
    ei_pred = d["Qh_nd_ob_vgradnji_vir"] == "EI"
    for c in ("brez_ovoja", "brez_ovoja_ob_tc", "Qr", "Qh_nd_ob_vgradnji"):
        d[f"{c}_2003"] = d[c]
    d["Qh_nd_ob_vgradnji"] = np.where(ei_pred, d["Qh_nd_ob_vgradnji"], qh_ob)
    d["Qr"] = np.where(d["obdobje_imputirano"], "neznano", d["Qh_nd_ob_vgradnji"].map(C.qhnd_razred))
    d["brez_ovoja"] = komb.sum(axis=1) == 0
    d["brez_ovoja_ob_tc"] = st_ob == 0
    print(f"krog 13, GURS ≥ {GMIN}: brez_ovoja {int(d['brez_ovoja_2003'].sum())} → {int(d['brez_ovoja'].sum())}; "
          f"Q1+Q2 brez ovoja {int((d['brez_ovoja_2003'] & d['Qr_2003'].isin(['Q1', 'Q2'])).sum())} → {int((d['brez_ovoja'] & d['Qr'].isin(['Q1', 'Q2'])).sum())}")
    (HERE / "cache").mkdir(exist_ok=True)
    _q = d[["ID_Stavbe", "Qr", "Qh_nd_ob_vgradnji", "brez_ovoja", "brez_ovoja_ob_tc"]].reset_index(drop=True)
    _q["ID_Stavbe"], _q["Qr"] = _q["ID_Stavbe"].astype(str), _q["Qr"].astype(str)
    _q.to_feather(HERE / "cache" / "01_Qr_gurs2010.feather")          # za K2_4 / K10 (feather: berljiv tudi z novejšim pyarrow)
d["A_ok"] = d["A_vir"] != "imputirano"
d["Wm2"] = np.where(d["A_ok"] & ~d["tc_kW_imputirano"], d["tc_kW"] * 1000 / d["A_ogr"], np.nan)

base = d[d["Qr"].isin(TH["razredi"]) & d[TH["pogoj_ovoja"]]]
POP = {
    "TČ brez ovoja": lambda g: g,
    "paket TČ+PV brez ovoja": lambda g: g[g["paket_osnovni"]],
}
ZNANI = ["ELKO", "biomasa", "plin", "elektrika", "premog", "poleg_obstojece"]


def q(s: pd.Series, p: float) -> float:
    s = pd.to_numeric(s, errors="coerce").dropna()
    return float(s.quantile(p)) if len(s) else np.nan


rows = []


def add(raz, pop, kaz, vr, enota="", opomba="", n=None):
    rows.append({"razred": raz, "populacija": pop, "kazalnik": kaz, "vrednost": vr, "enota": enota, "n": n, "opomba": opomba})


for raz in TH["razredi"]:
    for pop, f in POP.items():
        g = f(base[base["Qr"] == raz])
        add(raz, pop, "št. stavb", len(g), "št.")
        a = g.loc[g["A_ok"], "A_ogr"]
        for p, lab in [(0.25, "P25"), (0.5, "mediana"), (0.75, "P75")]:
            add(raz, pop, f"A_ogr {lab}", round(q(a, p), 1), "m²", "brez imputirane površine", len(a))
        k = g.loc[~g["tc_kW_imputirano"], "tc_kW"]
        for p, lab in [(0.25, "P25"), (0.5, "mediana"), (0.75, "P75")]:
            add(raz, pop, f"TČ moč {lab}", round(q(k, p), 1), "kW", "brez imputirane moči", len(k))
        add(raz, pop, "TČ specifična moč mediana", round(q(g["Wm2"], 0.5), 0), "W/m²", "oba podatka neimputirana", int(g["Wm2"].notna().sum()))
        pv = g.loc[g["ima_PV"], "PV_kWp"]
        for p, lab in [(0.25, "P25"), (0.5, "mediana"), (0.75, "P75")]:
            add(raz, pop, f"PV {lab}", round(q(pv, p), 1), "kWp", "stavbe s PV", len(pv))
        add(raz, pop, "delež s PV", round(100 * g["ima_PV"].mean(), 1), "%")
        add(raz, pop, "delež PV ≥ 11 kWp", round(100 * (pv >= 11).mean(), 1) if len(pv) else np.nan, "%", "med stavbami s PV")
        add(raz, pop, "Qh,nd ob vgradnji mediana", round(q(g["Qh_nd_ob_vgradnji"], 0.5), 1), "kWh/m²a", "kalibrirano (06) ali rEI")
        add(raz, pop, "Qh,nd ob vgradnji P25–P75", f"{q(g['Qh_nd_ob_vgradnji'], .25):.0f}–{q(g['Qh_nd_ob_vgradnji'], .75):.0f}", "kWh/m²a")
        add(raz, pop, "delež Qh,nd iz EI", round(100 * (g["Qh_nd_ob_vgradnji_vir"] == "EI").mean(), 1), "%")
        for ob in sorted(g["obdobje"].dropna().unique()):
            add(raz, pop, f"obdobje {int(ob)} ({C.OBDOBJE_OPIS[int(ob)]})", round(100 * (g["obdobje"] == ob).mean(), 1), "%")
        add(raz, pop, "leto gradnje mediana", round(q(g["LETO_IZGRA"], 0.5), 0), "leto")
        for t, n in g["tip_tc"].value_counts(normalize=True).items():
            add(raz, pop, f"tip TČ {t}", round(100 * n, 1), "%")
        so = g["staro_ogrevanje"].where(g["staro_ogrevanje"].isin(ZNANI))
        for t in ZNANI:
            add(raz, pop, f"nadomeščeno: {t}", round(100 * (so == t).sum() / so.notna().sum(), 1), "%",
                "med znanimi kodami (uradni šifrant)", int(so.notna().sum()))
        add(raz, pop, "delež z znano kodo nadomeščenega", round(100 * so.notna().mean(), 1), "%")
        add(raz, pop, "delež s prezračevanjem", round(100 * g["ima_prezrac"].mean(), 1), "%")
        add(raz, pop, "leto vgradnje TČ mediana", round(q(g["leto_TC"], 0.5), 0), "leto")
        add(raz, pop, "toplota Q_heat mediana", round(q(g["Q_heat_kWh"], 0.5), 0), "kWh/a", "analiza, mesečni pristop")
        add(raz, pop, "SCOP realni mediana", round(q(g["SCOP_real"], 0.5), 2), "–", "analiza §5")
        add(raz, pop, "E_el TČ realni mediana", round(1000 * q(g["E_el_realni_MWh"], 0.5), 0), "kWh/a", "analiza")
        add(raz, pop, "lock-in dosegljivi mediana", round(1000 * q(g["lockin_dos_MWh"], 0.5), 0), "kWh/a", "analiza")

long = pd.DataFrame(rows)
long.to_csv(TAB / "01_tipicna_hisa.csv", sep=C.SEP, encoding=C.ENC, index=False)
wide = long.pivot_table(index="kazalnik", columns=["razred", "populacija"], values="vrednost", aggfunc="first", sort=False)
wide.to_csv(TAB / "01_tipicna_hisa_pregled.csv", sep=C.SEP, encoding=C.ENC)

# ---------------------------------------------------------------- populacijske uteži (za korak 08)
dd = d[~d["obdobje_imputirano"]]
pop = (dd.assign(brez_ovoja=dd["brez_ovoja"], paket=dd["paket_osnovni"])
       .groupby(["Qr", "brez_ovoja", "paket", "obdobje"]).agg(stavb=("ID_Stavbe", "size"),
                                                           A_ogr_mediana=("A_ogr", "median"),
                                                           PV_MW=("PV_kWp", lambda s: s.sum() / 1000))
       .round(2).reset_index())
pop.to_csv(TAB / "01_populacija.csv", sep=C.SEP, encoding=C.ENC, index=False)

nov = (d[d["Qr"].isin(TH["razredi"])]
       .groupby("leto_TC").agg(TC_Q1Q2=("ID_Stavbe", "size"),
                               brez_ovoja_ob_tc=("brez_ovoja_ob_tc", "sum"),
                               s_paketom_brez_ovoja=("paket_osnovni", lambda s: int((s & d.loc[s.index, "brez_ovoja"]).sum())))
       .loc[2015:].reset_index())
nov.to_csv(TAB / "01_novi_letno.csv", sep=C.SEP, encoding=C.ENC, index=False)

# ---------------------------------------------------------------- FOND (dvoravenski pristop, 9. 10. 2026)
FD = PAR["fond"]
CACHE = HERE / "cache"
CACHE.mkdir(exist_ok=True)
f = d[d["Qr"].isin(FD["razredi"]) & d[FD["pogoj_ovoja"]]].copy()
f["arhetip"] = f["obdobje"].astype(int).map(FD["arhetip_po_obdobju"])
f["A_razred"] = pd.cut(f["A_ogr"], [-np.inf, *FD["meje_A_m2"], np.inf], labels=["A≤85", "A85–125", "A>125"]).astype(str)
pv = f["PV_kWp"].where(f["ima_PV"], 0).fillna(0)
f["PV_razred"] = np.select([pv <= FD["meje_PV_kWp"][0], pv <= FD["meje_PV_kWp"][1]], ["brez PV", "PV≤11"], "PV>11")
f["stratum"] = f["arhetip"] + "|" + f["A_razred"] + "|" + f["PV_razred"]
f["Qhnd_lastna_EI"] = np.where(f["Qh_nd_ob_vgradnji_vir"] == "EI", f["Qh_nd_ob_vgradnji"], np.nan)
keep = ["ID_Stavbe", "Qr", "obdobje", "arhetip", "stratum", "A_razred", "PV_razred", "A_ogr", "A_vir", "tc_kW", "tc_kW_imputirano",
        "tc_kW_uporabljena", "tip_tc", "PV_kWp", "ima_PV", "leto_TC", "leto_PV", "sekvenca", "paket_osnovni", "staro_ogrevanje",
        "bivalentno", "ima_prezrac", "Qh_nd_ob_vgradnji", "Qh_nd_ob_vgradnji_vir", "Qhnd_lastna_EI", "LETO_IZGRA", "STEVILO_ET",
        "ID_Regije", "ID_Obcine", "E_el_realni_MWh", "SCOP_real"]
f[keep].to_parquet(CACHE / "01_fond.parquet", index=False)

strat = (f.groupby(["arhetip", "A_razred", "PV_razred"])
         .agg(stavb=("ID_Stavbe", "size"), paket=("paket_osnovni", "sum"),
              A_ogr_mediana=("A_ogr", "median"), TC_kW_mediana=("tc_kW_uporabljena", "median"),
              PV_kWp_mediana=("PV_kWp", "median"))
         .round(1).reset_index())
strat.to_csv(TAB / "01_fond_stratumi.csv", sep=C.SEP, encoding=C.ENC, index=False)


def repr_row(g, lab):
    a, k, p = g.loc[g["A_vir"] != "imputirano", "A_ogr"], g.loc[~g["tc_kW_imputirano"], "tc_kW"], g.loc[g["ima_PV"], "PV_kWp"]
    so = g["staro_ogrevanje"].where(g["staro_ogrevanje"].isin(ZNANI))
    return {"arhetip": lab, "opis": FD["arhetip_opis"].get(lab, lab), "stavb": len(g), "od tega paket": int(g["paket_osnovni"].sum()),
            "delež fonda %": round(100 * len(g) / len(f), 1), "Q1 %": round(100 * (g["Qr"] == "Q1").mean(), 1),
            "A_ogr mediana": round(a.median(), 0), "A_ogr P25–P75": f"{a.quantile(.25):.0f}–{a.quantile(.75):.0f}",
            "TČ kW mediana": round(k.median(), 1), "TČ W/m² mediana": round(g["Wm2"].median(), 0),
            "PV kWp mediana (s PV)": round(p.median(), 1), "delež s PV %": round(100 * g["ima_PV"].mean(), 1),
            "Qh,nd celica": round(g["Qh_nd_ob_vgradnji"].median(), 1), "etaž GURS mediana": g["STEVILO_ET"].median(),
            "ELKO %": round(100 * (so == "ELKO").sum() / so.notna().sum(), 0),
            "biomasa+premog %": round(100 * so.isin(["biomasa", "premog"]).sum() / so.notna().sum(), 0)}


f["Wm2"] = d.loc[f.index, "Wm2"]
rep = pd.DataFrame([repr_row(g, a) for a, g in f.groupby("arhetip")] + [repr_row(f, "fond")])
rep.to_csv(TAB / "01_reprezentativne.csv", sep=C.SEP, encoding=C.ENC, index=False)

# rEI vzorec za Monte Carlo Qh,nd (ista definicija kot vzorec B v 06_kalibracija.py; logika ponovljena, ker 06 ni modul)
e = C.load_parquet("03_stavbe_gurs_ei.parquet") if hasattr(C, "load_parquet") else pd.read_parquet(C.CACHE / "03_stavbe_gurs_ei.parquet")
e = e[e["tc_ogrevalna"].astype(bool)].copy()
e["tipg"] = e["tip_stavbe"].map(C.tip_group)
lei = pd.to_numeric(e["leto_EI"], errors="coerce").astype(float)


def do_leta(col, gurs=False):
    y = pd.to_numeric(e[col], errors="coerce").astype(float)
    ok = (y <= lei)
    if gurs:
        ok &= (y >= GMIN)              # krog 13: prag GURS (2010)
    return ok.fillna(False)


e["st"] = pd.DataFrame({"f": do_leta("leto_prvi_fasada") | do_leta("LETO_OBN_FASADE", True),
                        "o": do_leta("leto_prvi_okna_vrata") | do_leta("okna_obn_min", True),
                        "s": do_leta("leto_prvi_streha") | do_leta("LETO_OBN_STREHE", True)}).sum(axis=1).astype(int)
cel = pd.to_numeric(e["leto_prvi_celovita_prenova"], errors="coerce").astype(float)
e.loc[(cel <= lei).fillna(False), "st"] = 3
r = e[(e["tip_EI"] == "rEI") & e["Qh_nd_EI"].between(5, 400) & e["obdobje"].notna() & lei.notna()].copy()
r["obdobje"] = r["obdobje"].astype(int)
r[["ID_Stavbe", "obdobje", "st", "tipg", "Qh_nd_EI", "leto_EI"]].to_parquet(CACHE / "01_rEI_vzorec.parquet", index=False)
qs = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
por = (r[(r["tipg"] == "eno_dvo")].groupby(["obdobje", "st"])["Qh_nd_EI"]
       .agg(["size", *[(lambda p: (lambda s: s.quantile(p)))(p) for p in qs]]))
por.columns = ["n"] + [f"P{int(p * 100)}" for p in qs]
por = por.round(1).reset_index()
por.to_csv(TAB / "01_rEI_porazdelitev.csv", sep=C.SEP, encoding=C.ENC, index=False)

# ---------------------------------------------------------------- kontrolne številke
print(f"FOND: {len(f)} stavb (Q1 {int((f['Qr'] == 'Q1').sum())}, Q2 {int((f['Qr'] == 'Q2').sum())}), "
      f"paket {int(f['paket_osnovni'].sum())}, lastna rEI {int(f['Qhnd_lastna_EI'].notna().sum())}, stratumov {f['stratum'].nunique()} "
      f"(z ≥ 30 stavbami {int((strat['stavb'] >= 30).sum())})")
print(rep.to_string(index=False))
print("rEI porazdelitev eno_dvo, stopnja 0:")
print(por[por["st"] == 0].to_string(index=False))
kal = pd.read_csv(ANALIZA / "06_Qhnd_kalibrirano.csv", sep=";", encoding="utf-8-sig")
kal = kal[(kal["tip"] == "eno_dvo") & (kal["stopnja"] == 0)].set_index("obdobje")["uporabljena_vrednost"]
pk = d["paket_osnovni"] & d["brez_ovoja"]
print("Kontrola proti analizi v3:")
print(f"  paket brez ovoja Q1: {int((pk & (d['Qr'] == 'Q1')).sum())} (07_T7e: 2751)")
print(f"  paket brez ovoja Q1+Q2: {int((pk & d['Qr'].isin(['Q1', 'Q2'])).sum())} (07_T7e: 5969)")
print(f"  TČ brez ovoja Q1+Q2: {len(base)} (11_T11g: 29.977)")
ln = nov.set_index("leto_TC").loc[TH["leta_novih"]]
print(f"  novi letno, povprečje {TH['leta_novih']}: vse TČ v Q1+Q2 {ln['TC_Q1Q2'].mean():.0f} (CLAUDE.md ≈ 8.500), "
      f"od tega brez ovoja ob TČ {ln['brez_ovoja_ob_tc'].mean():.0f}, s paketom brez ovoja {ln['s_paketom_brez_ovoja'].mean():.0f}")
print(f"  kalibrirana Qh,nd brez ovoja (06): {kal.round(1).to_dict()}")
print()
print(wide.to_string())
