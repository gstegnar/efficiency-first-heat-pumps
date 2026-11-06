"""Krog 4, K4_2 — ciljna skupina »PV brez TČ« (hiše, pri katerih odločitev o vrstnem redu še ni padla).

Korak A (`--korak A`): opredelitev in štetje. Vir: ../Interno/2026 analiza/cache/03_stavbe_gurs_ei.parquet (vse stavbe z ukrepom Eko sklada 2011–2025, GURS, EI),
eko_long.parquet (moč PV), izvorna datoteka energentov izkaznic (common.EI_ENERGENTI; samo branje).
Opredelitev: eno-/dvostanovanjske (GURS), obdobje gradnje 1–4, brez novogradenj (Eko ukrep novogradnja ali GURS leto gradnje ≥ leto PV − 2),
subvencionirana PV, **brez ogrevalne TČ do konca 2025** (nobene kategorije TČ razen sanitarne), **brez ukrepa na ovoju** (Eko sklad ∪ GURS ≥ 2003).
Dodatno štetje: PV brez TČ **z** ukrepom na ovoju.
Opomba: PV prek Borzena (od 2024) v registru ni → skupina je spodnja meja.
Izhodi: tabele/14_K4_2_stetje.csv (+ clanek1/tabele/), cache/14_K4_2_skupina.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
PAR = yaml.safe_load((HERE / "parametri.yaml").read_text(encoding="utf-8"))
AN = (HERE / PAR["poti"]["analiza"]).resolve()
sys.path.insert(0, str(AN))
import common as C  # noqa: E402

TAB, OUT_T, CACHE = HERE / "tabele", HERE / "clanek1" / "tabele", HERE / "cache"
ARH = {1: "A1", 2: "A2", 3: "A3", 4: "A4"}
TC_KAT = ["tc_neznano", "tci", "tcvv", "tczemlja", "tczv"]


def leta(d, cols, gurs=False):
    cs = [pd.to_numeric(d[c], errors="coerce").astype(float) for c in cols if c in d]
    if not cs:
        return pd.DataFrame(index=d.index)
    y = pd.concat(cs, axis=1)
    return y.where(y >= 2010) if gurs else y          # krog 13: ukrep v GURS pred 2010 ni ukrep URE (prej 2003)


def energent_ei(ids: pd.Series) -> pd.Series:
    """Glavni energent ogrevanja iz izkaznice (prednost: ELKO, plin, biomasa, daljinsko, elektrika)."""
    pot = Path(PAR["poti"]["ei_energenti"]) if Path(PAR["poti"]["ei_energenti"]).exists() else C.EI_ENERGENTI
    en = C.read_csv(pot, sep=";", usecols=["EIEI_ID", "ENERGENT", "ENERGENT_SIFRA"])
    en = en[en["EIEI_ID"].isin(set(ids.dropna()))]
    e = (en["ENERGENT"].fillna("") + " " + en["ENERGENT_SIFRA"].fillna("")).map(C.norm_text)
    kat = np.select([e.str.contains("kurilno olje|elko|oil"), e.str.contains("plin|gas|unp|lpg"),
                     e.str.contains("les|biomas|pelet|sekanc|wood"), e.str.contains("daljin|district"),
                     e.str.contains("toplota okolja|geoterm|energy_to|energy_ge"), e.str.contains("elektri|electr")],
                    ["ELKO", "plin", "biomasa", "daljinsko", "TČ (izkaznica)", "elektrika"], "drugo")
    pri = {"ELKO": 0, "plin": 1, "biomasa": 2, "daljinsko": 3, "TČ (izkaznica)": 4, "elektrika": 5, "drugo": 6}
    t = pd.DataFrame({"EIEI_ID": en["EIEI_ID"].values, "kat": kat})
    t["p"] = t["kat"].map(pri)
    t = t.sort_values("p").drop_duplicates("EIEI_ID")
    return ids.map(t.set_index("EIEI_ID")["kat"])


def skupina():
    d = pd.read_parquet(AN / "cache" / "03_stavbe_gurs_ei.parquet")
    d = d[d["tip_stavbe"].isin(["enostanovanjska", "dvostanovanjska"])].copy()
    lpv = pd.to_numeric(d["leto_prvi_fotovoltaika"], errors="coerce")
    tc = d["tc_ogrevalna"].astype(bool) | leta(d, [f"leto_prvi_{k}" for k in TC_KAT]).notna().any(axis=1)
    ovoj = pd.concat([leta(d, ["leto_prvi_fasada", "leto_prvi_streha", "leto_prvi_okna_vrata", "leto_prvi_talna_izolacija",
                                "leto_prvi_kletna_izolacija", "leto_prvi_celovita_prenova"]),
                      leta(d, ["LETO_OBN_FASADE", "LETO_OBN_STREHE", "okna_obn_min"], gurs=True)], axis=1).notna().any(axis=1)
    iz = pd.to_numeric(d["LETO_IZGRA"], errors="coerce")
    novo = d["leto_prvi_novogradnja"].notna() | (iz >= lpv - 2)
    d["ima_PV"], d["ima_TC"], d["ima_ovoj"], d["novogradnja_k4"], d["leto_PV"] = lpv.notna(), tc, ovoj, novo, lpv
    d["arhetip"] = d["obdobje"].map(ARH)
    e = pd.read_parquet(AN / "cache" / "eko_long.parquet", columns=["ID_Stavbe", "kat", "Kolicina_n", "Leto"])
    kwp = e[e["kat"] == "fotovoltaika"].groupby("ID_Stavbe")["Kolicina_n"].sum()
    d["PV_kWp"] = d["ID_Stavbe"].map(kwp)
    return d


def korak_a():
    d = skupina()
    osn = d["ima_PV"] & ~d["ima_TC"] & d["obdobje"].between(1, 4) & ~d["novogradnja_k4"]
    k = d[osn & ~d["ima_ovoj"]].copy()
    k["EIEI_ID"] = k["EIEI_ID"].astype(str).str.strip().replace({"None": np.nan, "nan": np.nan, "": np.nan})
    k["energent"] = energent_ei(k["EIEI_ID"]).fillna("ni znan (brez EI)")
    n_tc_ei = int((k["energent"] == "TČ (izkaznica)").sum())
    k = k[k["energent"] != "TČ (izkaznica)"].copy()                     # izkaznica navaja TČ (nesubvencionirano) — izločeno
    k["A_ogr"] = pd.to_numeric(k.get("A_stan"), errors="coerce").where(lambda x: x > 0, pd.to_numeric(k.get("A_uporabna"), errors="coerce"))
    vse = {"eno-/dvostanovanjske s subvencionirano PV": int(d["ima_PV"].sum()),
           "  brez ogrevalne TČ do 2025": int((d["ima_PV"] & ~d["ima_TC"]).sum()),
           "  … obdobje 1–4, brez novogradenj": int(osn.sum()),
           "K4_2: … brez ukrepa na ovoju": len(k) + n_tc_ei,
           "  izločeno: izkaznica navaja TČ kot energent": n_tc_ei,
           "K4_2 (končno)": len(k),
           "dodatno: PV brez TČ, z ukrepom na ovoju (obdobje 1–4)": int((osn & d["ima_ovoj"]).sum()),
           "K4_2 z izkaznico (rEI ali mEI)": int(k["EIEI_ID"].notna().sum()),
           "K4_2 z računsko izkaznico (Qh,nd)": int(k["Qh_nd_EI"].notna().sum())}
    print("\n".join(f"{a}: {b}" for a, b in vse.items()))
    k["kWp_razred"] = pd.cut(k["PV_kWp"], [0, 6, 9, 11, 14, 1e9], labels=["≤ 6", "6–9", "9–11", "11–14", "> 14"])
    t_obd = k.pivot_table(index="arhetip", values="ID_Stavbe", aggfunc="count").rename(columns={"ID_Stavbe": "stavb"})
    t_obd["kWp mediana"] = k.groupby("arhetip")["PV_kWp"].median().round(1)
    t_obd["A_ogr mediana (GURS)"] = k.groupby("arhetip")["A_ogr"].median().round(0)
    NMP = PAR["krog4"]["net_metering_priklop_do"]
    t_obd[f"delež PV do {NMP} (letni net metering) [%]"] = k.groupby("arhetip")["leto_PV"].apply(lambda x: round(100 * (x <= NMP).mean(), 1))
    t_obd.loc["skupaj"] = [len(k), round(k["PV_kWp"].median(), 1), round(k["A_ogr"].median(), 0), round(100 * (k["leto_PV"] <= NMP).mean(), 1)]
    t_leto = k.pivot_table(index="leto_PV", columns="arhetip", values="ID_Stavbe", aggfunc="count", fill_value=0)
    t_kwp = k.pivot_table(index="kWp_razred", columns="arhetip", values="ID_Stavbe", aggfunc="count", fill_value=0)
    t_reg = k.pivot_table(index="ID_Regije", columns="arhetip", values="ID_Stavbe", aggfunc="count", fill_value=0)
    t_en = k.pivot_table(index="energent", columns="arhetip", values="ID_Stavbe", aggfunc="count", fill_value=0)
    for t, f in ((t_obd, "14_K4_2_stetje.csv"), (t_leto, "14_K4_2_po_letu.csv"), (t_kwp, "14_K4_2_po_kWp.csv"), (t_reg, "14_K4_2_po_regiji.csv"),
                 (t_en, "14_K4_2_energent.csv")):
        for o in (TAB, OUT_T):
            t.to_csv(o / f, sep=";", encoding="utf-8-sig")
    pd.Series(vse).to_csv(TAB / "14_K4_2_povzetek.csv", sep=";", encoding="utf-8-sig", header=["vrednost"])
    k.drop(columns=["kWp_razred"]).to_parquet(CACHE / "14_K4_2_skupina.parquet", index=False)
    pd.set_option("display.width", 220)
    for lab, t in (("po obdobju", t_obd), ("po letu PV", t_leto), ("po kWp", t_kwp), ("energent (izkaznica)", t_en), ("po regiji", t_reg)):
        print(f"\n{lab}:\n", t.to_string())


K4 = PAR["krog4"]
TAG = "_K42"
EK = PAR["ekonomika"]
SKUP = {"pred 1981": ["A1", "A2", "A3"], "1981–2002": ["A4"], "skupaj": ["A1", "A2", "A3", "A4"]}


# ---------------------------------------------------------------- korak B1: pričakovana vgradnja TČ (Kaplan–Meier prve TČ po PV)
def km(d=None):
    """Diskretni Kaplan–Meier: čas od leta PV do prve ogrevalne TČ (leta), cenzura konec 2025. Kohorta: eno-/dvostanovanjske s PV, obdobje 1–4,
    brez novogradenj in brez ukrepa na ovoju (kot K4_2), brez TČ v letu PV ali prej (paket). Vrne {arhetip|vse: S(t), t = 0..40}."""
    d = skupina() if d is None else d
    g = d[d["ima_PV"] & d["obdobje"].between(1, 4) & ~d["novogradnja_k4"] & ~d["ima_ovoj"]].copy()
    lt = leta(g, [f"leto_prvi_{k}" for k in TC_KAT]).min(axis=1)
    g["leto_PV"] = g["leto_PV"].astype(float)                      # Float64 (NA) → float: primerjave z NA bi izločile hiše brez TČ
    g = g[~(lt.notna() & (lt <= g["leto_PV"])) & ~(g["tc_ogrevalna"].astype(bool) & lt.isna())].copy()
    lt = lt.reindex(g.index)
    ev = lt.notna() & (lt <= 2025)
    T = np.where(ev, lt - g["leto_PV"], 2025 - g["leto_PV"]).astype(int)
    g["T"], g["E"] = T, ev.to_numpy()
    out, rows = {}, []
    for lab, x in [("vse", g)] + [(a, x) for a, x in g.groupby("arhetip")]:
        tmax = int(x["T"].max())
        h = []
        for t in range(1, tmax + 1):
            n = int((x["T"] >= t).sum())
            e = int(((x["T"] == t) & x["E"]).sum())
            h.append((t, n, e, e / n if n else np.nan))
            rows.append({"skupina": lab, "t [leta po PV]": t, "v tveganju": n, "dogodkov": e, "hazard": e / n if n else np.nan})
        hz = [x_[3] for x_ in h if x_[1] >= K4["km_min_tveganje"]]
        rep = float(np.mean(hz[-K4["km_rep_let"]:]))
        haz = np.array([x_[3] if x_[1] >= K4["km_min_tveganje"] else rep for x_ in h] + [rep] * (40 - len(h)))
        out[lab] = np.concatenate([[1.0], np.cumprod(1 - haz)])
        out[lab + "_rep"] = rep
    KM = pd.DataFrame(rows)
    KM.to_csv(TAB / "14_K4_2_kaplan_meier.csv", sep=";", encoding="utf-8-sig", index=False)
    print(f"KM kohorta {len(g)}, dogodkov {int(g['E'].sum())}; hazard v repu: " + ", ".join(f"{k} {out[k + '_rep']:.3f}" for k in ["vse", "A1", "A2", "A3", "A4"]))
    return out


def verjetnost(k, S):
    """P(TČ v naslednjih h letih | brez TČ do konca 2025) = 1 − S(d + h) / S(d), d = 2025 − leto PV."""
    dd = (2025 - k["leto_PV"]).astype(int).clip(0, 25).to_numpy()
    for h in K4["km_horizonti"]:
        k[f"p_TC_{h}"] = [1 - S[a][x + h] / S[a][x] for a, x in zip(k["arhetip"], dd)]
    return k


# ---------------------------------------------------------------- korak B2: vhod za urni model (03 --fond … --tag _K42)
def korak_fond():
    k = pd.read_parquet(CACHE / "14_K4_2_skupina.parquet")
    f = pd.read_parquet(CACHE / "01_fond.parquet")
    pk = f[f["paket_osnovni"].astype(float).fillna(0).astype(bool)]
    wm2 = (pk["tc_kW_uporabljena"] * 1000 / pk["A_ogr"]).groupby(pk["arhetip"]).median()
    rng = np.random.default_rng(K4["seme_k42"])
    k["arhetip"] = k["obdobje"].astype(int).map(ARH)
    k["A_ogr"] = k["A_ogr"].fillna(k.groupby("arhetip")["A_ogr"].transform("median"))
    k["PV_kWp"] = k["PV_kWp"].fillna(k.groupby("arhetip")["PV_kWp"].transform("median"))
    en = k["energent"].where(k["energent"].isin(["ELKO", "plin", "biomasa"]))
    for a, g in k[en.isna()].groupby("arhetip"):                 # brez EI: porazdelitev nadomeščenih energentov paketa istega obdobja
        p = pk.loc[(pk["arhetip"] == a) & pk["staro_ogrevanje"].isin(["ELKO", "plin", "biomasa"]), "staro_ogrevanje"].value_counts(normalize=True)
        en.loc[g.index] = rng.choice(p.index.to_numpy(), size=len(g), p=p.to_numpy())
    k["staro_ogrevanje"], k["energent_vir"] = en, np.where(k["energent"].isin(["ELKO", "plin", "biomasa"]), "izkaznica", "porazdelitev paketa")
    d = pd.DataFrame({"ID_Stavbe": k["ID_Stavbe"].astype(str), "obdobje": k["obdobje"].astype(int), "arhetip": k["arhetip"], "A_ogr": k["A_ogr"],
                      "A_vir": "GURS", "tc_kW": k["A_ogr"] * k["arhetip"].map(wm2) / 1000, "tc_kW_imputirano": False,
                      "ima_PV": True, "PV_kWp": k["PV_kWp"], "paket_osnovni": True, "pv_fiksen": True,
                      "Qhnd_lastna_EI": k["Qh_nd_EI"].where(k["tip_EI"] == "rEI"), "tip_stavbe": k["tip_stavbe"],
                      "Qr": np.where(k["arhetip"] == "A4", "Q2", "Q1"), "stratum": k["arhetip"], "staro_ogrevanje": k["staro_ogrevanje"],
                      "energent_vir": k["energent_vir"], "leto_PV": k["leto_PV"], "ID_Regije": k["ID_Regije"]})
    d["tc_kW_uporabljena"] = d["tc_kW"]
    d.to_parquet(CACHE / "14_K4_2_fond.parquet", index=False)
    print(f"{len(d)} stavb; W/m² paketa {wm2.round(1).to_dict()}; energent: {d['staro_ogrevanje'].value_counts().to_dict()}; "
          f"lastna rEI {int(d['Qhnd_lastna_EI'].notna().sum())}")


# ---------------------------------------------------------------- korak B3: tri prihodnosti, tri perspektive
POTI_K42 = {"F1": {"seg": [(0, "V2")], "ev": [(0, "TC", "e"), (18, "TCz", "e")]},
            "F1r": {"seg": [(0, "V2r")], "ev": [(0, "TC", "r"), (18, "TCz", "r")]},
            "F3": {"seg": [(0, "V9")], "ev": [(0, "FSO", None), (0, "TC", "FSO"), (18, "TCz", "FSO")]}}


def korak_b():
    import importlib.util
    s = importlib.util.spec_from_file_location("k21", HERE / "K2_1_drzava.py")
    K1 = importlib.util.module_from_spec(s)
    s.loader.exec_module(K1)
    E, DR, OKM = K1.E, K1.DR, K1.DR.OKM
    for mod in (E, OKM.EKM):
        mod.POTI = POTI_K42
    DR.POTI = K1.POTI = list(POTI_K42)
    fd = pd.read_parquet(CACHE / "14_K4_2_fond.parquet").set_index("ID_Stavbe")
    w = E.load_units("fond", TAG).drop(columns=["staro_ogrevanje"]).join(fd[["staro_ogrevanje", "energent_vir", "leto_PV", "obdobje", "A_ogr"]])
    w["V0"] = np.select([w["staro_ogrevanje"] == "biomasa", w["staro_ogrevanje"] == "plin"], ["V0b", "V0c"], "V0a")
    o = pd.DataFrame(index=w.index)
    nm = (w["leto_PV"] <= K4["net_metering_priklop_do"]).to_numpy()
    # dejanski režim (krog 7): PV do net_metering_priklop_do ima R1 v letih t ≤ tn (leto t = leto0 + t), nato R3; poznejši PV R3.
    # osnova: tn = konec življenjske dobe PV (leto PV + zivljenjska_doba_PV − 1); občutljivost: konec leta net_metering_obcutljivost
    konec = {"": (w["leto_PV"].to_numpy(float) + K4["zivljenjska_doba_PV"] - 1)}
    konec.update({f" {y}": np.full(len(w), float(y)) for y in K4["net_metering_obcutljivost"]})
    for rez in ("R1", "R3"):
        n = E.npv_paths(w, rez, EK["diskont_zasebni"], E.pf_factory("konstantne"), EK["ovoj_osnova"])
        for p in POTI_K42:
            o[f"{p}|NPV {rez}"] = n[p]
    for p in POTI_K42:
        razl = {t: (E.annual_bill(w, E.leto_varianta(p, t), "R1", 1.0) - E.annual_bill(w, E.leto_varianta(p, t), "R3", 1.0)) * (1 + EK["diskont_zasebni"]) ** (-t)
                for t in range(1, E.N + 1)}
        for sfx, ky in konec.items():
            tn = ky - EK["leto0"]
            corr = sum(razl[t] * (t <= tn) for t in razl)
            o[f"{p}|NPV dejanski{sfx}"] = o[f"{p}|NPV R3"] + np.where(nm, corr, 0.0)
    rc = K1.racun(w)
    sp_nom, _ = E.spodbude_poti(w)
    v20 = {p: E.leto_varianta(p, 20) for p in POTI_K42}
    for p in POTI_K42:
        o[f"{p}|proračun R3"] = rc[p]["R3 proračun neto"]
        o[f"{p}|družbeni NPV"] = rc[p]["družbeni NPV"]
        o[f"{p}|NN kW"] = rc[p]["obremenitev NN omrežja [kW]"]
        o[f"{p}|spodbude"] = sp_nom[p]
        o[f"{p}|zima"] = w[f"{v20[p]}|uvoz_zima_kWh"]
        o[f"{p}|konica kW"] = w[f"{v20[p]}|uvoz_konica_sis_kW"]
        op = sum(OKM.op_co2(w, E.leto_varianta(p, t), t) for t in range(1, E.N + 1))
        emb = sum(OKM.embodied(w, item, arg) for (y, item, arg) in POTI_K42[p]["ev"])
        o[f"{p}|CO2e t"] = (op + emb) / 1000
        o[f"{p}|TČ kW"] = w[f"{v20[p]}|TC_kW"]
    o["Qh_nd"] = w["V2|Qh_nd"]
    o["Qh_nd_cilj"] = w["V9|Qh_nd"]
    o["brez_ovoja_F3"] = w["V4|brez_ovoja_P3"] > 0.5
    w = w[["arhetip", "staro_ogrevanje", "energent_vir", "leto_PV", "A_ogr", "V2|PV_kWp"]].join(o)
    w = verjetnost(w, km())
    w.reset_index().to_parquet(CACHE / "14_K4_2_rezultati.parquet", index=False)
    print(f"rezultati: {len(w)} stavb → cache/14_K4_2_rezultati.parquet")


# ---------------------------------------------------------------- korak B4: glavna tabela 14_pv_brez_tc.csv in Fig10
KAZ = {"NPV R1": ("posameznik", "NPV stroškov 30 let, R1 (letni net metering)", "kEUR", 1e3),
       "NPV R3": ("posameznik", "NPV stroškov 30 let, R3", "kEUR", 1e3),
       "NPV dejanski": ("posameznik", f"NPV stroškov 30 let, dejanski režim (PV ≤ {K4['net_metering_priklop_do']}: R1 do konca življenjske dobe PV, nato R3)", "kEUR", 1e3),
       **{f"NPV dejanski {y}": ("posameznik", f"NPV stroškov 30 let, dejanski režim — občutljivost: R1 do {y}, nato R3", "kEUR", 1e3)
          for y in PAR["krog4"]["net_metering_obcutljivost"]},
       "proračun R3": ("država", "javni proračun NPV, R3", "kEUR", 1e3), "družbeni NPV": ("država", "družbeni NPV", "kEUR", 1e3),
       "spodbude": ("država", "spodbude (nominalno)", "kEUR", 1e3), "zima": ("država", "zimski uvoz (nov–feb), leto 20", "kWh/a", 1),
       "konica kW": ("država", "prispevek h sistemski konici, leto 20", "kW", 1), "NN kW": ("država", "obremenitev NN omrežja", "kW", 1),
       "CO2e t": ("okolje", "CO2e 30 let (obratovalni + vgrajeni)", "t", 1), "TČ kW": ("posameznik", "moč TČ (leto 20)", "kW", 1)}
VSOTA = {"kEUR": (1e6, "mio EUR"), "kWh/a": (1e6, "GWh/a"), "kW": (1e3, "MW"), "t": (1e3, "kt")}


def korak_tabela():
    w = pd.read_parquet(CACHE / "14_K4_2_rezultati.parquet")
    rows = []

    def add(sk, kaz, en, fn, st="mediana"):
        rows.append({"sklop": sk, "kazalnik": kaz, "enota": en, "statistika": st, **{c: fn(w[w["arhetip"].isin(a)]) for c, a in SKUP.items()}})
    add("skupina", "stavb", "št.", len, "število")
    add("skupina", "PV", "kWp", lambda g: g["V2|PV_kWp"].median())
    NMP = K4["net_metering_priklop_do"]
    add("skupina", f"delež PV do {NMP} (letni net metering)", "%", lambda g: 100 * (g["leto_PV"] <= NMP).mean(), "delež")
    add("skupina", "ogrevana površina", "m²", lambda g: g["A_ogr"].median())
    add("skupina", "Qh,nd (model, položaj kot pri paketu)", "kWh/m²a", lambda g: g["Qh_nd"].median())
    add("skupina", "energent iz izkaznice", "%", lambda g: 100 * (g["energent_vir"] == "izkaznica").mean(), "delež")
    add("skupina", "≤ ciljna prenova + 5 (F3 brez ukrepa na ovoju)", "%", lambda g: 100 * g["brez_ovoja_F3"].mean(), "delež")
    for h in K4["km_horizonti"]:
        add("pričakovana vgradnja TČ (Kaplan–Meier)", f"v {h} letih", "%", lambda g, h=h: 100 * g[f"p_TC_{h}"].mean(), "delež")
        add("pričakovana vgradnja TČ (Kaplan–Meier)", f"v {h} letih", "št.", lambda g, h=h: g[f"p_TC_{h}"].sum(), "pričakovano število")
    for key, (sk, lab, en, sc) in KAZ.items():
        for p in POTI_K42:
            add(f"prihodnosti ({sk})", f"{lab} — {p}", en, lambda g, p=p, key=key, sc=sc: g[f"{p}|{key}"].median() / sc)
    for ref in ("F1", "F1r"):
        for key, (sk, lab, en, sc) in KAZ.items():
            dd = lambda g, key=key, ref=ref: g[f"F3|{key}"] - g[f"{ref}|{key}"]   # noqa: E731
            add(f"F3 − {ref}: {sk}", lab, en, lambda g, dd=dd, sc=sc: dd(g).median() / sc)
            add(f"F3 − {ref}: {sk}", lab, "%", lambda g, dd=dd: 100 * (dd(g) < 0).mean(), f"delež F3 < {ref}")
    add(f"F1r − F1", "NPV stroškov 30 let, dejanski režim", "kEUR", lambda g: (g["F1r|NPV dejanski"] - g["F1|NPV dejanski"]).median() / 1e3)
    # ukrep A: pričakovano število TČ × razlika F3 − F1 (F3 − F1r)
    for ref in ("F1", "F1r"):
        for h in K4["km_horizonti"]:
            for key in ("zima", "konica kW", "NN kW", "CO2e t", "spodbude", "družbeni NPV", "proračun R3"):
                sk, lab, en, sc = KAZ[key]
                v, ve = VSOTA[en] if en in VSOTA else VSOTA["kEUR"]
                add(f"ukrep A: pričakovane TČ v {h} letih × (F3 − {ref})", lab, ve,
                    lambda g, key=key, ref=ref, h=h, v=v: float((g[f"p_TC_{h}"] * (g[f"F3|{key}"] - g[f"{ref}|{key}"])).sum() / v), "vsota")
    # krog 15: pravilo s pragom — ovoj (F3) le nad prag_ovoj_Qhnd, med ukrep_A_prag_Qhnd in pragom le dimenzioniranje (F1r); proti F1
    PO = PAR["populacija"]
    p2, p1 = PO["prag_ovoj_Qhnd"], PO["ukrep_A_prag_Qhnd"]
    add(f"pravilo s pragom {p2}", f"Qh,nd > {p2}", "%", lambda g: 100 * (g["Qh_nd"] > p2).mean(), "delež hiš")
    for h in K4["km_horizonti"]:
        add(f"pravilo s pragom {p2}", f"pričakovane TČ v {h} letih, Qh,nd > {p2}", "št.", lambda g, h=h: g[f"p_TC_{h}"].where(g["Qh_nd"] > p2, 0).sum(), "pričakovano število")
        for key in ("zima", "konica kW", "NN kW", "CO2e t", "spodbude", "družbeni NPV", "proračun R3"):
            sk, lab, en, sc = KAZ[key]
            v, ve = VSOTA[en] if en in VSOTA else VSOTA["kEUR"]

            def pravilo(g, key=key, h=h, v=v, samo_ovoj=False):
                q = g["Qh_nd"]
                ref = np.where(q > p2, g[f"F3|{key}"], np.where((q > p1) & (not samo_ovoj), g[f"F1r|{key}"], g[f"F1|{key}"]))
                return float((g[f"p_TC_{h}"] * (ref - g[f"F1|{key}"])).sum() / v)
            add(f"pravilo s pragom {p2}: pričakovane TČ v {h} letih × (F3 nad {p2} − F1)", lab, ve, lambda g, f=pravilo: f(g, samo_ovoj=True), "vsota")
            add(f"pravilo s pragom {p2}: pričakovane TČ v {h} letih × (F3 nad {p2}, F1r {p1}–{p2} − F1)", lab, ve, pravilo, "vsota")
    # net metering: delitev po roku priklopa odstranjena (krog 5) — po roku je le 51 hiš (PV 2025); R1 / R3 / dejanski režim so v sklopu F3 − F1
    T = pd.DataFrame(rows)
    for o in (TAB, OUT_T):
        T.to_csv(o / "14_pv_brez_tc.csv", sep=";", encoding="utf-8-sig", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 75)
    pd.set_option("display.max_rows", 500)
    print(T.drop(columns=["sklop"]).round(2).to_string(index=False))
    slika10(w)


def slika10(w):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
    plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "font.family": "DejaVu Sans"})
    SEQ = ["#08519c", "#4292c6", "#9ecae1", "#e6e5e0"]
    per = [("A1", ("pre-1946", "pred 1946")), ("A2", ("1946–1970", "1946–1970")), ("A3", ("1971–1980", "1971–1980")), ("A4", ("1981–2002", "1981–2002"))]
    pp = [("NPV R3", ("household,\nR3", "posameznik,\nR3")), ("NPV dejanski", ("household,\nactual tariff", "posameznik,\ndejanski\nrežim")),
          ("proračun R3", ("public\nbudget", "javni\nproračun")), ("družbeni NPV", ("social\nCBA", "družbena\nCBA")), ("CO2e t", ("CO₂e", "CO₂e"))]
    for lang, sfx in (("en", ""), ("sl", "_sl")):
        i = 0 if lang == "en" else 1
        fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [1, 1.25]})
        for y, (ar, lab) in enumerate(per):
            g = w[w["arhetip"] == ar]
            n = len(g)
            p3, p5, p10 = (g[f"p_TC_{h}"].sum() for h in (3, 5, 10))
            left = 0
            for v, c in zip((p3, p5 - p3, p10 - p5, n - p10), SEQ):
                a.barh(y, v, left=left, color=c, height=0.62, edgecolor="white", linewidth=1.2)
                left += v
            a.text(n + 40, y, f"{n:,}".replace(",", "." if lang == "sl" else ","), va="center", fontsize=7, color=MUTED)
        a.set_yticks(range(len(per)))
        a.set_yticklabels([l_[i] for _, l_ in per])
        a.invert_yaxis()
        a.set_xlabel("houses with PV, no heat pump" if lang == "en" else "hiše s PV brez TČ")
        a.grid(axis="x", color=GRID, lw=0.6)
        a.set_axisbelow(True)
        a.set_xlim(0, w["arhetip"].value_counts().max() * 1.15)
        lab = (["HP within 3 years", "3–5 years", "5–10 years", "no HP in 10 years"] if lang == "en"
               else ["TČ v 3 letih", "3–5 let", "5–10 let", "brez TČ v 10 letih"])
        a.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in SEQ], labels=lab, ncol=2, fontsize=6.5, loc="upper center",
                 bbox_to_anchor=(0.45, -0.22), handlelength=1)
        a.set_title("(a) " + ("expected heat-pump uptake" if lang == "en" else "pričakovana vgradnja TČ"), loc="left", fontsize=8.5, color=INK)
        x = np.arange(len(pp))
        for j, (gr, c) in enumerate((("pred 1981", "#eb6834"), ("1981–2002", "#2a78d6"))):
            g = w[w["arhetip"].isin(SKUP[gr])]
            vals = [100 * ((g[f"F3|{k}"] - g[f"F1|{k}"]) < 0).mean() for k, _ in pp]
            b.bar(x + (j - 0.5) * 0.36, vals, width=0.34, color=c, label=(("pre-1981", "1981–2002")[j] if lang == "en" else gr))
            for xx, v in zip(x + (j - 0.5) * 0.36, vals):
                b.text(xx, v + 1.5, f"{v:.0f}", ha="center", fontsize=6.5, color=MUTED)
        b.axhline(50, color=MUTED, lw=0.8, ls="--")
        b.set_ylim(0, 110)
        b.set_xticks(x)
        b.set_xticklabels([l_[i].replace(", ", ",\n") for _, l_ in pp], fontsize=6.8)
        b.set_ylabel("houses where F3 beats F1 [%]" if lang == "en" else "hiše, kjer je F3 boljši od F1 [%]")
        b.grid(axis="y", color=GRID, lw=0.6)
        b.set_axisbelow(True)
        b.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, handlelength=1)
        b.set_title("(b) " + ("envelope first (F3) vs HP as practised (F1)" if lang == "en" else "ovoj najprej (F3) proti TČ kot v praksi (F1)"),
                    loc="left", fontsize=8.5, color=INK)
        fig.tight_layout()
        for ext in ("png", "svg"):
            fig.savefig(HERE / "clanek1" / "slike" / f"Fig10_pv_without_hp{sfx}.{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--korak", default="A")
    a = ap.parse_args()
    {"A": korak_a, "KM": km, "F": korak_fond, "B": korak_b, "T": korak_tabela}[a.korak]()
