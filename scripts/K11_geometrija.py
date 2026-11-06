"""Krog 11, točka 1 — površine elementov ovoja iz projektov Eko sklada proti geometriji modela (TABULA) (NAVODILA_CLAUDE_CODE_v11.md).

Vir: eko_eur_m2.xlsx, list Nalozbe (SFH, V statistiki = DA, 2023–2025; P: samo branje — K11_NALOZBE = feather kopija lista).
a) količina ukrepa / površina po elementu (fasada, streha/strop, okna/vrata, tla, kletni strop), obdobje gradnje (A1–A4, pred 1981 / 1981–2002),
   razred ogrevane površine (≤ 100, 100–150, 150–200, > 200 m²); normirano na tri površine: (i) ogrevana uporabna iz vloge Eko sklada,
   (ii) neto tlorisna (vsi deli) iz vloge, (iii) A_ogr po pravilu modela (GURS A_stan × 0,85, sicer glavna / uporabna; kot 04_energija analize).
b) sestava ovoja iste hiše: stavbe s fasado + streho + okni (ločene vloge iste stavbe).
c) geometrija modela (TABULA TAB03 / TAB04, linearno z A_ogr) in neodvisna ocena bruto stene iz GURS: obseg kvadratnega tlorisa
   (BRUTO_TLOR / STEVILO_ET) × višina etaže (hecm_vhodi.visina_etaze_m) × STEVILO_ET; neto = bruto − okna iste hiše (če jih je menjala).
Izhod: tabele/21a_povrsine_eko_proti_modelu.csv (+ clanek1/tabele/).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
PAR = yaml.safe_load((HERE / "parametri.yaml").read_text(encoding="utf-8"))
AN = (HERE / PAR["poti"]["analiza"]).resolve()
TAB, OUT_T = HERE / "tabele", HERE / "clanek1" / "tabele"
HV = PAR["hecm_vhodi"]
EL = {"fasada": "stena", "streha": "streha", "okna_vrata": "okna", "talna_izolacija": "tla", "kletna_izolacija": "tla (kletni strop)"}
COST, A_EKO, A_NETO, Q = "Priznani stroški [EUR]", "Ogrevana uporabna površina [m²]", "Neto tlorisna površina (vsi deli) [m²]", "Količina ukrepa"
DELEZ_OGR = 0.85


def arhetip(leto):
    return pd.Series(np.select([leto <= 1945, leto <= 1970, leto <= 1980, leto <= 2002], ["A1", "A2", "A3", "A4"], ""), index=leto.index)


def podatki():
    n = pd.read_feather(os.environ["K11_NALOZBE"]) if os.environ.get("K11_NALOZBE") else pd.read_excel(PAR["poti"]["eko_eur_m2"], sheet_name="Nalozbe")
    n = n[(n["Tip"] == "SFH") & (n["V statistiki"] == "DA") & n["Ukrep (koda)"].isin(list(EL)) & (n["Enota"] == "m2")].copy()
    n["stavba"] = n["KO-stavba|del"].str.split("|").str[0]
    n["leto_izgr"] = pd.to_numeric(n["Leto izgradnje"], errors="coerce")
    n["arhetip"] = arhetip(n["leto_izgr"])
    n["obdobje"] = np.where(n["arhetip"].isin(["A1", "A2", "A3"]), "pred 1981", np.where(n["arhetip"] == "A4", "1981–2002", ""))
    n["A_eko"] = pd.to_numeric(n[A_EKO], errors="coerce").where(lambda x: x >= 20)
    n["A_neto"] = pd.to_numeric(n[A_NETO], errors="coerce").where(lambda x: x >= 20)
    g = pd.read_parquet(AN / "cache" / "02_stavbe_gurs.parquet", columns=["ID_Stavbe", "A_stan", "A_glavna", "A_uporabna", "BRUTO_TLOR", "STEVILO_ET"])
    g["ID_Stavbe"] = g["ID_Stavbe"].astype(str)
    g = g.drop_duplicates("ID_Stavbe").set_index("ID_Stavbe")
    a_m = (g["A_stan"].where(g["A_stan"] >= 20)).fillna(g["A_glavna"].where(g["A_glavna"] >= 20)).fillna(g["A_uporabna"].where(g["A_uporabna"] >= 20)) * DELEZ_OGR
    n["A_model"] = n["stavba"].map(a_m)
    et = pd.to_numeric(g["STEVILO_ET"], errors="coerce").where(lambda x: x.between(1, 5))
    fp = pd.to_numeric(g["BRUTO_TLOR"], errors="coerce").where(lambda x: x >= 30) / et
    n["stena_GURS_bruto"] = n["stavba"].map(4 * np.sqrt(fp) * HV["visina_etaze_m"] * et)
    n["ET"] = n["stavba"].map(et)
    n["A_razred"] = pd.cut(n["A_eko"], [0, 100, 150, 200, 1e9], labels=["≤ 100", "100–150", "150–200", "> 200"]).astype(str)
    # ena vrstica na stavbo in element (vsota količin, če je več vlog)
    s = (n.groupby(["stavba", "Ukrep (koda)"]).agg(Q=(Q, "sum"), A_eko=("A_eko", "median"), A_neto=("A_neto", "median"), A_model=("A_model", "first"),
                                                  arhetip=("arhetip", "first"), obdobje=("obdobje", "first"), A_razred=("A_razred", "first"),
                                                  stena_GURS_bruto=("stena_GURS_bruto", "first"), ET=("ET", "first"), leto=("Leto", "min"))
         .reset_index())
    return s


def stat(x):
    x = x.dropna()
    x = x[np.isfinite(x)]
    return {"n": len(x), "P25": x.quantile(.25) if len(x) else np.nan, "mediana": x.median() if len(x) else np.nan, "P75": x.quantile(.75) if len(x) else np.nan}


def geom_model():
    out = {}
    for a, d in HV["arhetipi"].items():
        g = HV["geometrija"][d["geometrija"]]
        out[a] = {"stena": g["stena"] / g["A"], "streha": g["streha"] / g["A"], "okna": g["okna"] / g["A"], "tla": g["tla"] / g["A"], "A": g["A"], "vir": g["vir"]}
    return out


def main():
    s = podatki()
    s = s[s["arhetip"] != ""]
    GM = geom_model()
    rows = []
    # a) po elementu × obdobje / arhetip × razred A; tri normiranja
    for kod, el in EL.items():
        e = s[s["Ukrep (koda)"] == kod]
        for skup, col in (("obdobje", "obdobje"), ("arhetip", "arhetip")):
            for key, g in e.groupby(col):
                for raz in ["vse"] + ["≤ 100", "100–150", "150–200", "> 200"]:
                    h = g if raz == "vse" else g[g["A_razred"] == raz]
                    if len(h) < 5:
                        continue
                    for nor, ac in (("ogrevana (Eko)", "A_eko"), ("neto tlorisna (Eko)", "A_neto"), ("A_ogr modela (GURS)", "A_model")):
                        st = stat(h["Q"] / h[ac])
                        mdl = np.nan
                        if el in ("stena", "streha", "okna", "tla", "tla (kletni strop)"):
                            ars = ["A1", "A2", "A3"] if key == "pred 1981" else (["A4"] if key == "1981–2002" else [key])
                            mdl = np.mean([GM[a][el.split(" ")[0]] for a in ars])
                        rows.append({"sklop": "a) količina / površina", "element": kod, "skupina": f"{skup}: {key}", "razred A (Eko ogrevana)": raz,
                                     "normirano na": nor, **st, "model TABULA [m²/m²]": mdl,
                                     "mediana / model": st["mediana"] / mdl if mdl == mdl else np.nan,
                                     "mediana A [m²]": h[ac].median()})
    # b) sestava ovoja ene hiše: fasada + streha + okna
    p = s.pivot_table(index="stavba", columns="Ukrep (koda)", values="Q", aggfunc="sum")
    meta = s.groupby("stavba").agg(A_eko=("A_eko", "median"), A_model=("A_model", "first"), obdobje=("obdobje", "first"), ET=("ET", "first"),
                                   stena_GURS_bruto=("stena_GURS_bruto", "first"))
    k = p.dropna(subset=["fasada", "streha", "okna_vrata"]).join(meta)
    for obd, g in [("vse", k)] + list(k.groupby("obdobje")):
        for kod in ("fasada", "streha", "okna_vrata", "talna_izolacija", "kletna_izolacija"):
            if kod not in g:
                continue
            for nor, ac in (("ogrevana (Eko)", "A_eko"), ("A_ogr modela (GURS)", "A_model")):
                rows.append({"sklop": "b) sestava ene hiše (fasada + streha + okna)", "element": kod, "skupina": obd, "razred A (Eko ogrevana)": "vse",
                             "normirano na": nor, **stat(g[kod] / g[ac]), "mediana A [m²]": g[ac].median()})
        rows.append({"sklop": "b) sestava ene hiše (fasada + streha + okna)", "element": "okna / fasada", "skupina": obd, "razred A (Eko ogrevana)": "vse",
                     "normirano na": "razmerje", **stat(g["okna_vrata"] / g["fasada"])})
        rows.append({"sklop": "b) sestava ene hiše (fasada + streha + okna)", "element": "streha / A_eko", "skupina": obd, "razred A (Eko ogrevana)": "vse",
                     "normirano na": "razmerje", **stat(g["streha"] / g["A_eko"])})
    # c) GURS ocena bruto stene in neto / bruto pri fasadi
    f = s[s["Ukrep (koda)"] == "fasada"].set_index("stavba")
    ok = p["okna_vrata"].reindex(f.index)
    for obd, g in [("vse", f)] + list(f.groupby("obdobje")):
        o = ok.reindex(g.index)
        rows.append({"sklop": "c) GURS: bruto stena (kvadratni tloris × etaže × višina)", "element": "stena GURS bruto / A_ogr modela", "skupina": obd,
                     "razred A (Eko ogrevana)": "vse", "normirano na": "A_ogr modela (GURS)", **stat(g["stena_GURS_bruto"] / g["A_model"])})
        rows.append({"sklop": "c) GURS: bruto stena (kvadratni tloris × etaže × višina)", "element": "fasada Eko / stena GURS bruto", "skupina": obd,
                     "razred A (Eko ogrevana)": "vse", "normirano na": "razmerje", **stat(g["Q"] / g["stena_GURS_bruto"])})
        rows.append({"sklop": "c) GURS: bruto stena (kvadratni tloris × etaže × višina)", "element": "fasada Eko / (stena GURS bruto − okna Eko), hiše z okni",
                     "skupina": obd, "razred A (Eko ogrevana)": "vse", "normirano na": "razmerje", **stat(g["Q"] / (g["stena_GURS_bruto"] - o))})
        rows.append({"sklop": "c) GURS: bruto stena (kvadratni tloris × etaže × višina)", "element": "A_eko / A_ogr modela", "skupina": obd,
                     "razred A (Eko ogrevana)": "vse", "normirano na": "razmerje", **stat(g["A_eko"] / g["A_model"])})
        rows.append({"sklop": "c) GURS: bruto stena (kvadratni tloris × etaže × višina)", "element": "A_neto / A_eko", "skupina": obd,
                     "razred A (Eko ogrevana)": "vse", "normirano na": "razmerje", **stat(g["A_neto"] / g["A_eko"])})
        rows.append({"sklop": "c) GURS: bruto stena (kvadratni tloris × etaže × višina)", "element": "število etaž (GURS)", "skupina": obd,
                     "razred A (Eko ogrevana)": "vse", "normirano na": "", **stat(g["ET"])})
    for a, d in GM.items():
        for el in ("stena", "streha", "okna", "tla"):
            rows.append({"sklop": "c) model (TABULA, linearno z A_ogr)", "element": el, "skupina": f"arhetip: {a}", "razred A (Eko ogrevana)": "vse",
                         "normirano na": f"{d['vir']}, A = {d['A']:.0f} m²", "mediana": d[el]})
    T = pd.DataFrame(rows)
    num = T.select_dtypes("number").columns
    T[num] = T[num].round(3)
    for o in (TAB, OUT_T):
        T.to_csv(o / "21a_povrsine_eko_proti_modelu.csv", sep=";", encoding="utf-8-sig", index=False)
    pd.set_option("display.width", 260)
    pd.set_option("display.max_rows", 400)
    pd.set_option("display.max_colwidth", 60)
    v = T[(T["sklop"].str.startswith("a")) & (T["razred A (Eko ogrevana)"] == "vse") & T["skupina"].str.startswith("obdobje")]
    print(v.pivot_table(index=["element", "skupina"], columns="normirano na", values=["mediana", "n"], aggfunc="first").round(2).to_string())
    v = T[(T["sklop"].str.startswith("a")) & (T["normirano na"] == "ogrevana (Eko)") & T["skupina"].str.startswith("obdobje") & (T["razred A (Eko ogrevana)"] != "vse")]
    print(v.pivot_table(index=["element", "skupina"], columns="razred A (Eko ogrevana)", values="mediana").round(2).to_string())
    v = T[(T["sklop"].str.startswith("a")) & (T["normirano na"] == "ogrevana (Eko)") & T["skupina"].str.startswith("arhetip") & (T["razred A (Eko ogrevana)"] == "vse")]
    print(v[["element", "skupina", "n", "P25", "mediana", "P75", "model TABULA [m²/m²]", "mediana / model", "mediana A [m²]"]].to_string(index=False))
    print(T[T["sklop"].str.startswith(("b", "c) GURS"))].drop(columns=["model TABULA [m²/m²]", "mediana / model"]).to_string(index=False))



def kontrola_stroska():
    """2c: modelski strošek celovitega paketa (in po elementu) z geometrijo G11 proti dejanskim stroškom (stavbe kroga 10, 20a_strosek_ovoja_stavbe.csv).
    Model: EUR/enoto (parametri ekonomika.ovoj_EUR_enota) × razmerje elementa × A_ogr modela (0,85 × A_stan, kot v verigi)."""
    d = pd.read_csv(TAB / "20a_strosek_ovoja_stavbe.csv", sep=";", encoding="utf-8-sig")
    EKo = PAR["ekonomika"]["ovoj_EUR_enota"]
    rows = []
    for geo in ("TABULA", "G11"):
        def razm(a, el):
            g = HV["geometrija"][HV["geometrija_G11"][a] if geo == "G11" else HV["arhetipi"][a]["geometrija"]]
            return g[el] / g["A"]
        d[f"model {geo}"] = [sum(razm(a, e) * DELEZ_OGR * A * EKo[e] for e in el.split("+")) for a, A, el in zip(d["arhetip"], d["A_ogr"], d["elementi"])]
        d[f"dejansko / {geo}"] = d["dejansko [EUR]"] / d[f"model {geo}"]
    for (vz, ob), g in list(d.groupby(["vzorec", "obdobje"])) + [((vz, "vse"), g) for vz, g in d.groupby("vzorec")]:
        rows.append({"sklop": "d) strošek: dejansko / model", "element": vz, "skupina": ob, "razred A (Eko ogrevana)": "vse", "normirano na": "A_ogr modela (GURS)",
                     "n": len(g), **{f"{geo} {q}": g[f"dejansko / {geo}"].quantile(p) for geo in ("TABULA", "G11") for q, p in (("P25", .25), ("mediana", .5), ("P75", .75))}})
    R = pd.DataFrame(rows)
    T = pd.read_csv(TAB / "21a_povrsine_eko_proti_modelu.csv", sep=";", encoding="utf-8-sig")
    T = pd.concat([T[~T["sklop"].str.startswith("d)")], R.round(3)], ignore_index=True)
    for o in (TAB, OUT_T):
        T.to_csv(o / "21a_povrsine_eko_proti_modelu.csv", sep=";", encoding="utf-8-sig", index=False)
    pd.set_option("display.width", 250)
    print(R.round(2).drop(columns=["sklop", "razred A (Eko ogrevana)", "normirano na"]).to_string(index=False))



def kontrola_qhnd():
    """2b: Qh,nd stanj S0–S3 — model TABULA (02_variante), G11 (02_variante_G11), različica C (stena na neto tlorisno; K11 hitri tek, vpisano ročno)
    proti izkaznicam (rEI, eno-/dvostanovanjske, stopnja 0–3 kot v 02_lestvica_kontrola) na dveh osnovah: (i) kot v izkaznici (na kondicionirano površino),
    (ii) preračunano na A_ogr modela: Qh,nd_EI × A_kond / (0,85 × A_stan) po hiši. Izhod: tabele/21b_qhnd_g11_proti_izkaznicam.csv."""
    import pyarrow  # noqa: F401
    vT = pd.read_csv(TAB / "02_variante.csv", sep=";", encoding="utf-8-sig").set_index(["arhetip", "varianta"])["Qh_nd"]
    vG = pd.read_csv(TAB / "02_variante_G11.csv", sep=";", encoding="utf-8-sig").set_index(["arhetip", "varianta"])["Qh_nd"]
    vC = {("A1", "BREZ"): 320.3, ("A1", "S1"): 291.0, ("A1", "S2"): 217.3, ("A1", "S3"): 58.9, ("A2", "BREZ"): 252.4, ("A2", "S1"): 236.0, ("A2", "S2"): 183.3,
          ("A2", "S3"): 58.9, ("A3", "BREZ"): 196.9, ("A3", "S1"): 186.4, ("A3", "S2"): 149.5, ("A3", "S3"): 58.9, ("A4", "BREZ"): 108.3, ("A4", "S1"): 101.9,
          ("A4", "S2"): 89.8, ("A4", "S3"): 54.6}
    r = pd.read_parquet(HERE / "cache" / "01_rEI_vzorec.parquet")
    r = r[r["tipg"] == "eno_dvo"].copy()
    g = pd.read_parquet(AN / "cache" / "03_stavbe_gurs_ei.parquet", columns=["ID_Stavbe", "KONDICIONIRANA_POVRSINA", "A_stan"]).drop_duplicates("ID_Stavbe")
    g["f"] = pd.to_numeric(g["KONDICIONIRANA_POVRSINA"], errors="coerce") / (DELEZ_OGR * pd.to_numeric(g["A_stan"], errors="coerce").where(lambda x: x >= 20))
    r["f"] = r["ID_Stavbe"].map(g.set_index("ID_Stavbe")["f"])
    r["Qh_Amodel"] = r["Qh_nd_EI"] * r["f"]
    rows = []
    for a, o in (("A1", 1), ("A2", 2), ("A3", 3), ("A4", 4)):
        for st, v in ((0, "BREZ"), (1, "S1"), (2, "S2"), (3, "S3")):
            x = r[(r["obdobje"] == o) & (r["st"] == st)]
            y = x["Qh_Amodel"].dropna()
            rows.append({"arhetip": a, "stanje": {0: "S0", 1: "S1", 2: "S2", 3: "S3"}[st], "model TABULA": vT[(a, v)], "model G11 (Eko geometrija, B)": vG[(a, v)],
                         "model C (stena na neto tlorisno)": vC[(a, v)], "rEI n": len(x), "rEI P50 (na A_kond)": x["Qh_nd_EI"].median(),
                         "rEI P75 (na A_kond)": x["Qh_nd_EI"].quantile(.75), "rEI P90 (na A_kond)": x["Qh_nd_EI"].quantile(.9),
                         "A_kond / A_ogr modela mediana": x["f"].median(), "rEI na A_ogr modela: n": len(y), "P50": y.median(), "P75": y.quantile(.75), "P90": y.quantile(.9),
                         "G11 / rEI P50 (na A_ogr modela)": vG[(a, v)] / y.median(), "TABULA / rEI P50 (na A_kond)": vT[(a, v)] / x["Qh_nd_EI"].median()})
    T = pd.DataFrame(rows).round(2)
    for o in (TAB, OUT_T):
        T.to_csv(o / "21b_qhnd_g11_proti_izkaznicam.csv", sep=";", encoding="utf-8-sig", index=False)
    pd.set_option("display.width", 260)
    print(T.round(1).to_string(index=False))



POTI3 = ["P1", "P1r", "P2", "P3", "P4"]


def vrednotenje(tag):
    """3: kazalniki po hiši za fond 29.977 (paket je podmnožica) — iste funkcije kot članek (04 npv_paths, K2_1 racun, 06 op_co2/embodied),
    stroški iz parametri.yaml (krog 11: nove 04a). tag "" = osnova TABULA (03_fond_letno), "_G11" = G11 (zaženi z GEOM=G11).
    Izhod: cache/21c_hise{tag or '_TABULA'}.parquet."""
    import importlib.util
    sp = importlib.util.spec_from_file_location("k21", HERE / "K2_1_drzava.py")
    K1 = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(K1)
    E, OKM = K1.E, K1.DR.OKM
    w = E.load_units("fond", tag)
    EKp = PAR["ekonomika"]
    o = pd.DataFrame(index=w.index)
    for rez in ("R1", "R3"):
        n = E.npv_paths(w, rez, EKp["diskont_zasebni"], E.pf_factory("konstantne"), EKp["ovoj_osnova"])
        for p in POTI3:
            o[f"{p}|NPV {rez}"] = n[p]
    rc = K1.racun(w)
    for p in POTI3:
        o[f"{p}|proračun R3"] = rc[p]["R3 proračun neto"]
        o[f"{p}|družbeni NPV"] = rc[p]["družbeni NPV"]
        o[f"{p}|NN kW"] = rc[p]["obremenitev NN omrežja [kW]"]
        v20 = E.leto_varianta(p, 20)
        o[f"{p}|zima"] = w[f"{v20}|uvoz_zima_kWh"]
        op = sum(OKM.op_co2(w, E.leto_varianta(p, t), t) for t in range(1, E.N + 1))
        emb = sum(OKM.embodied(w, item, arg) for (y, item, arg) in E.POTI[p]["ev"])
        o[f"{p}|CO2e t"] = (op + emb) / 1000
    o["Qh_nd"] = w["V2|Qh_nd"]
    o["A"] = w["V4|A"]
    o["lestvica_k"], o["lestvica_w"], o["brez_P3"] = w["V4|lestvica_k"], w["V4|lestvica_w"], w["V4|brez_ovoja_P3"]
    o["TC_kW_P3"] = w["V4|TC_kW"]
    f1 = pd.read_parquet(HERE / "cache" / "01_fond.parquet", columns=["ID_Stavbe", "arhetip", "paket_osnovni"]).set_index("ID_Stavbe")
    o = o.join(f1)
    o.reset_index().to_parquet(HERE / "cache" / f"21c_hise{tag or '_TABULA'}.parquet", index=False)
    print(f"{tag or 'TABULA'}: {len(o)} hiš")


def tabela3():
    KZ = [("NPV R1", "posameznik R1", "kEUR", 1e3, 1e6, "mio EUR"), ("NPV R3", "posameznik R3", "kEUR", 1e3, 1e6, "mio EUR"),
          ("družbeni NPV", "družbeni NPV", "kEUR", 1e3, 1e6, "mio EUR"), ("proračun R3", "javni proračun R3", "kEUR", 1e3, 1e6, "mio EUR"),
          ("CO2e t", "CO2e 30 let", "t", 1, 1e3, "kt"), ("NN kW", "obremenitev NN", "kW", 1, 1e3, "MW"), ("zima", "zimski uvoz, leto 20", "kWh/a", 1, 1e6, "GWh/a")]
    R = {lab: pd.read_parquet(HERE / "cache" / f"21c_hise_{lab}.parquet") for lab in ("TABULA", "G11")}
    k10 = pd.read_csv(TAB / "20b_obcutljivost_strosek_ovoja.csv", sep=";", encoding="utf-8-sig")
    k10 = k10[k10["scenarij stroška ovoja"].str.contains("mediana")].set_index("obdobje")
    rows = []
    for vz, mvz in (("paket", lambda d: d["paket_osnovni"].astype(float).fillna(0).astype(bool)), ("fond", lambda d: pd.Series(True, index=d.index))):
        for obd, ars in (("pred 1981", ["A1", "A2", "A3"]), ("1981–2002", ["A4"])):
            for ref in ("P1", "P1r"):
                for k, lab, en, sc, vsc, ven in KZ:
                    row = {"vzorec": vz, "obdobje": obd, "primerjava": f"P3 − {ref}", "kazalnik": lab, "enota": en}
                    for g, d in R.items():
                        x = d[mvz(d) & d["arhetip"].isin(ars)]
                        dd = x[f"P3|{k}"] - x[f"{ref}|{k}"]
                        row[f"{g}: mediana"] = dd.median() / sc
                        row[f"{g}: delež P3 boljši [%]"] = 100 * (dd < 0).mean()
                        row[f"{g}: vsota [{ven}]"] = dd.sum() / vsc
                        row["hiš"] = len(x)
                    kk = {"posameznik R1": "posameznik R1", "posameznik R3": "posameznik R3", "družbeni NPV": "družbeni NPV", "javni proračun R3": "proračun R3"}.get(lab)
                    if vz == "paket" and ref == "P1" and kk and obd in k10.index:
                        row["krog 10 × mediana: mediana"] = k10.loc[obd, f"{kk}: P3 − P1 mediana [kEUR]"]
                        row["krog 10 × mediana: delež P3 boljši [%]"] = k10.loc[obd, f"{kk}: delež P3 boljši [%]"]
                    rows.append(row)
            for g, d in R.items():
                x = d[mvz(d) & d["arhetip"].isin(ars)]
                st = np.minimum(x["lestvica_k"].astype(int) + (x["lestvica_w"] >= 0.5), 3)
                rows.append({"vzorec": vz, "obdobje": obd, "primerjava": "opis", "kazalnik": f"{g}: Qh,nd mediana [kWh/m²a]; A mediana [m²]; TČ P3 [kW]; S0/S1/S2/S3 [%]; brez ukrepa [%]",
                             "enota": "", "hiš": len(x),
                             f"{g}: mediana": f"{x['Qh_nd'].median():.0f}; {x['A'].median():.0f}; {x['TC_kW_P3'].median():.0f}; "
                                              + "/".join(f"{100 * (st == j).mean():.0f}" for j in range(4)) + f"; {100 * (x['brez_P3'] > 0.5).mean():.1f}"})
    T = pd.DataFrame(rows)
    num = T.select_dtypes("number").columns
    T[num] = T[num].round(2)
    for o in (TAB, OUT_T):
        T.to_csv(o / "21c_primerjava_G11.csv", sep=";", encoding="utf-8-sig", index=False)
    pd.set_option("display.width", 260)
    pd.set_option("display.max_rows", 300)
    pd.set_option("display.max_colwidth", 70)
    kol = ["vzorec", "obdobje", "primerjava", "kazalnik", "TABULA: mediana", "TABULA: delež P3 boljši [%]", "G11: mediana", "G11: delež P3 boljši [%]",
           "krog 10 × mediana: mediana", "krog 10 × mediana: delež P3 boljši [%]"]
    print(T[[c for c in kol if c in T]].to_string(index=False))



def kontrola_k41():
    """3: K4_1 — model (stanje po registru) proti izkaznici po zadnjem ukrepu ovoja, TABULA (02_variante_K41) in G11 (02_variante_K41G11);
    G11 je na ogrevani površini G11 (≈ kondicionirana površina izkaznice), TABULA na A_ogr modela (izkaznica tam preračunana × A_kond / A_ogr).
    Izhod: tabele/21d_k41_g11.csv."""
    import importlib.util
    sp = importlib.util.spec_from_file_location("k41", HERE / "K4_1_primerjalna.py")
    K = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(K)
    k = pd.read_parquet(HERE / "cache" / "13_K4_1_skupina.parquet")
    k = k[k["obdobje"].between(1, 4)].copy()
    k["arhetip"] = k["obdobje"].astype(int).map(K.ARH)
    k["stanje"] = K.stanje_iz(k)
    ok = (k["tip_EI"] == "rEI") & (pd.to_numeric(k["leto_EI"], errors="coerce") >= k["leto_ovoj_zadnji"])
    k = k[ok & k["Qh_nd_EI"].notna()].copy()
    g = pd.read_parquet(AN / "cache" / "03_stavbe_gurs_ei.parquet", columns=["ID_Stavbe", "KONDICIONIRANA_POVRSINA", "A_stan"]).drop_duplicates("ID_Stavbe")
    g["f"] = pd.to_numeric(g["KONDICIONIRANA_POVRSINA"], errors="coerce") / (DELEZ_OGR * pd.to_numeric(g["A_stan"], errors="coerce").where(lambda x: x >= 20))
    k["f"] = k["ID_Stavbe"].map(g.set_index("ID_Stavbe")["f"])
    vT = pd.read_csv(TAB / "02_variante_K41.csv", sep=";", encoding="utf-8-sig").set_index(["arhetip", "varianta"])["Qh_nd"]
    vG = pd.read_csv(TAB / "02_variante_K41G11.csv", sep=";", encoding="utf-8-sig").set_index(["arhetip", "varianta"])["Qh_nd"]
    k["model TABULA"] = [vT[(a, s_)] for a, s_ in zip(k["arhetip"], k["stanje"])]
    k["model G11"] = [vG[(a, s_)] for a, s_ in zip(k["arhetip"], k["stanje"])]
    k["rEI na A_ogr modela"] = k["Qh_nd_EI"] * k["f"]
    rows = []
    for (lab, x) in [("K4_1 vse", k)] + [(f"zaporedje: {z}", x) for z, x in k.groupby("zaporedje")] + [(f"stanje: {s_}", x) for s_, x in k.groupby("stanje")]             + [(f"obdobje: {o}", x) for o, x in k.groupby(np.where(k["arhetip"] == "A4", "1981–2002", "pred 1981"))]:
        rows.append({"skupina": lab, "hiš": len(x), "izkaznica (na A_kond) mediana": x["Qh_nd_EI"].median(), "model G11 mediana": x["model G11"].median(),
                     "G11 / izkaznica (mediana razmerij)": (x["model G11"] / x["Qh_nd_EI"]).median(),
                     "izkaznica na A_ogr modela mediana": x["rEI na A_ogr modela"].median(), "model TABULA mediana": x["model TABULA"].median(),
                     "TABULA / izkaznica na A_ogr modela (mediana razmerij)": (x["model TABULA"] / x["rEI na A_ogr modela"]).median()})
    T = pd.DataFrame(rows).round(2)
    for o in (TAB, OUT_T):
        T.to_csv(o / "21d_k41_g11.csv", sep=";", encoding="utf-8-sig", index=False)
    pd.set_option("display.width", 250)
    print(T.round(1).to_string(index=False))


if __name__ == "__main__":
    import sys
    a = sys.argv[1] if len(sys.argv) > 1 else "1"
    if a == "3":
        vrednotenje(sys.argv[2] if len(sys.argv) > 2 else "")
    else:
        {"1": main, "2b": kontrola_qhnd, "2c": kontrola_stroska, "3t": tabela3, "k41": kontrola_k41}[a]()
