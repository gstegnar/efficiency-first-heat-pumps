"""Krog 13, točka 1 — toplotni ovoj ločeno od izolirane površine (G13), diagnostika (NAVODILA_CLAUDE_CODE_v13.md).

a/b) Toplotni ovoj na hišo iz GURS (../Interno/2026 analiza/cache/02_stavbe_gurs.parquet, samo branje) za hiše fonda 29.977 (in paket 5.969):
     tloris etaže F = BRUTO_TLOR / STEVILO_ET; ogrevana površina G11 A = A_ogr modela (0,85 × A_stan) × povrsina_G11_faktor (obdobje);
     ogrevanih etaž n_ogr = A / (0,85 · F), omejeno na [1, ET]; stena = 4·√F × visina_etaze_m (2,8) × n_ogr − okna; okna = okna G11 na m² × A;
     streha/strop = F; tla/kletni strop = F. Brez BRUTO_TLOR ali ET (1–5) hiša ni ocenjena (poročano n). Mediane in P25–P75 na m² ogrevane G11 po
     obdobju; razmerje toplotni ovoj / izolirana površina (Eko količina = geometrija G11 na m²). → tabele/22a_toplotni_ovoj_gurs.csv
c)   S0–S3 z G13 (02_lestvica_stanj.py, GEOM=G13, K3_TAG=_G13; izvirni U GreenRenoV8 brez faktorja) proti G11 brez faktorja (cache/02_lestvica_G11.log),
     G11 s faktorjem (osnova, tabele/02_variante.csv) in izkaznicam (rEI eno-/dvostanovanjske, stopnja 0–3, na kondicionirano površino ≈ ogrevana G11).
d)   K4_1: model G13 (02_variante_K41G13) / izkaznica po zadnjem ukrepu, kot 21d. → tabele/22b_S0_G13_proti_izkaznicam.csv
Zagon: pybe (pyarrow 13); --korak ab | cd
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
PAR = yaml.safe_load((HERE / "parametri.yaml").read_text(encoding="utf-8"))
AN = (HERE / PAR["poti"]["analiza"]).resolve()
TAB, OUT_T, CACHE = HERE / "tabele", HERE / "clanek1" / "tabele", HERE / "cache"
HV = PAR["hecm_vhodi"]
ARH = {1: "A1", 2: "A2", 3: "A3", 4: "A4"}
H_ET = HV["visina_etaze_m"]


def g11(a):
    g = HV["geometrija"][HV["geometrija_G11"][a]]
    return {k: g[k] / g["A"] for k in ("stena", "streha", "okna", "tla")}


def ovoj():
    f = pd.read_parquet(CACHE / "01_fond.parquet", columns=["ID_Stavbe", "obdobje", "arhetip", "A_ogr", "A_vir", "paket_osnovni"])
    f["ID_Stavbe"] = f["ID_Stavbe"].astype(str)
    g = pd.read_parquet(AN / "cache" / "02_stavbe_gurs.parquet", columns=["ID_Stavbe", "BRUTO_TLOR", "STEVILO_ET", "A_stan", "A_uporabna", "n_delov", "n_stan_delov"])
    g["ID_Stavbe"] = g["ID_Stavbe"].astype(str)
    d = f.merge(g.drop_duplicates("ID_Stavbe"), on="ID_Stavbe", how="left")
    d["A"] = d["A_ogr"].astype(float) * d["obdobje"].map(HV["povrsina_G11_faktor"]).astype(float)
    et = pd.to_numeric(d["STEVILO_ET"], errors="coerce").where(lambda x: x.between(1, 5))
    bt = pd.to_numeric(d["BRUTO_TLOR"], errors="coerce").where(lambda x: x >= 30)
    d["ET"], d["F"] = et, bt / et
    d["n_ogr"] = np.clip(d["A"] / (0.85 * d["F"]), 1, d["ET"])
    ok = np.array([g11(a)["okna"] for a in d["arhetip"]])
    d["okna_G13"] = ok * d["A"]
    d["stena_G13"] = 4 * np.sqrt(d["F"]) * H_ET * d["n_ogr"] - d["okna_G13"]
    d["streha_G13"], d["tla_G13"] = d["F"], d["F"]
    d["ocenjeno"] = d["F"].notna() & d["A"].notna() & (d["stena_G13"] > 0)
    return d


def korak_ab():
    d = ovoj()
    rows = []
    for vz, m in (("fond", pd.Series(True, index=d.index)), ("paket", d["paket_osnovni"].astype(float).fillna(0).astype(bool))):
        for a in ("A1", "A2", "A3", "A4"):
            x = d[m & (d["arhetip"] == a)]
            y = x[x["ocenjeno"]]
            G = g11(a)
            row = {"vzorec": vz, "arhetip": a, "hiš": len(x), "z GURS tlorisom in ET": len(y), "delež ocenjenih [%]": 100 * len(y) / max(len(x), 1),
                   "ET mediana": y["ET"].median(), "n_ogr mediana": y["n_ogr"].median(), "delež n_ogr = ET [%]": 100 * (y["n_ogr"] >= y["ET"] - 1e-9).mean(),
                   "delež n_ogr = 1 (omejeno) [%]": 100 * (y["n_ogr"] <= 1 + 1e-9).mean(),
                   "A ogrevana G11 mediana [m²]": y["A"].median(), "F tloris etaže mediana [m²]": y["F"].median(),
                   "neto tlorisna / A_stan (GURS) mediana": (pd.to_numeric(y["A_uporabna"], errors="coerce") / pd.to_numeric(y["A_stan"], errors="coerce")).median()}
            for el in ("stena", "streha", "tla", "okna"):
                r_ = y[f"{el}_G13"] / y["A"]
                row[f"{el} G13 na m² ogrevane: P25"] = r_.quantile(.25)
                row[f"{el} G13: mediana"] = r_.median()
                row[f"{el} G13: P75"] = r_.quantile(.75)
                row[f"{el} Eko (G11) na m²"] = G[el]
                if el != "okna":
                    row[f"{el}: toplotni ovoj / izolirana (mediana)"] = r_.median() / G[el]
            rows.append(row)
    T = pd.DataFrame(rows)
    for o in (TAB, OUT_T):
        T.round(3).to_csv(o / "22a_toplotni_ovoj_gurs.csv", sep=";", encoding="utf-8-sig", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 60)
    print(T.round(2).T.to_string())
    f = T[T["vzorec"] == "fond"].set_index("arhetip")
    print("\npredlog parametri.yaml (G13, A = 100, mediane fonda na m² ogrevane G11):")
    for a in ("A1", "A2", "A3", "A4"):
        print(f"    G13_{a}: {{vir: \"GURS toplotni ovoj (K13), mediana fonda {a}\", A: 100.0, stena: {100 * f.loc[a, 'stena G13: mediana']:.1f}, "
              f"streha: {100 * f.loc[a, 'streha G13: mediana']:.1f}, okna: {100 * f.loc[a, 'okna G13: mediana']:.1f}, tla: {100 * f.loc[a, 'tla G13: mediana']:.1f}}}")


def g11_brez_faktorja():
    """S0–S3 G11 z izvirnimi U (prvi tek kroga 11, pred kalibracijo A1/A2) iz cache/02_lestvica_G11.log."""
    out = {}
    for line in (CACHE / "02_lestvica_G11.log").read_text(encoding="utf-8").splitlines():
        m = re.match(r"\s+(A[1-4])\s+(BREZ|S1|S2|S3)\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+([\d.]+)\s+[\d.]+$", line)
        if m and (m.group(1), m.group(2)) not in out:
            out[(m.group(1), m.group(2))] = float(m.group(3))
    return out


def korak_cd():
    v0 = g11_brez_faktorja()
    vG = pd.read_csv(TAB / "02_variante.csv", sep=";", encoding="utf-8-sig").set_index(["arhetip", "varianta"])["Qh_nd"]
    v13 = pd.read_csv(TAB / "02_variante_G13.csv", sep=";", encoding="utf-8-sig").set_index(["arhetip", "varianta"])["Qh_nd"]
    r = pd.read_parquet(CACHE / "01_rEI_vzorec.parquet")
    r = r[r["tipg"] == "eno_dvo"]
    rows = []
    for o, a in ARH.items():
        for st, v in ((0, "BREZ"), (1, "S1"), (2, "S2"), (3, "S3")):
            x = r[(r["obdobje"] == o) & (r["st"] == st)]["Qh_nd_EI"]
            p50, p75, p90 = x.median(), x.quantile(.75), x.quantile(.9)
            rows.append({"del": "c lestvica", "arhetip": a, "stanje": {0: "S0", 1: "S1", 2: "S2", 3: "S3"}[st],
                         "G11 brez faktorja": v0.get((a, v), np.nan), "G11 s faktorjem (osnova)": vG[(a, v)], "G13": v13[(a, v)],
                         "izkaznice n": len(x), "izkaznice P50": p50, "P75": p75, "P90": p90,
                         "G13 / P90" if st == 0 else "G13 / P50": v13[(a, v)] / (p90 if st == 0 else p50),
                         "G13 v pasu": ("P75–P90" if p75 <= v13[(a, v)] <= p90 else ("nad P90" if v13[(a, v)] > p90 else "pod P75")) if st == 0
                         else ("P25–P75" if x.quantile(.25) <= v13[(a, v)] <= p75 else ("nad P75" if v13[(a, v)] > p75 else "pod P25"))})
        for v in ("FSO", "GLOB"):
            rows.append({"del": "c lestvica", "arhetip": a, "stanje": v, "G11 s faktorjem (osnova)": vG[(a, v)], "G13": v13[(a, v)]})
    T = pd.DataFrame(rows)
    # d) K4_1
    f41 = TAB / "02_variante_K41G13.csv"
    if f41.exists():
        import importlib.util
        sp = importlib.util.spec_from_file_location("k41", HERE / "K4_1_primerjalna.py")
        K = importlib.util.module_from_spec(sp)
        sp.loader.exec_module(K)
        k = pd.read_parquet(CACHE / "13_K4_1_skupina.parquet")
        k = k[k["obdobje"].between(1, 4)].copy()
        k["arhetip"] = k["obdobje"].astype(int).map(K.ARH)
        k["stanje"] = K.stanje_iz(k)
        k = k[(k["tip_EI"] == "rEI") & (pd.to_numeric(k["leto_EI"], errors="coerce") >= k["leto_ovoj_zadnji"]) & k["Qh_nd_EI"].notna()].copy()
        v41 = pd.read_csv(f41, sep=";", encoding="utf-8-sig").set_index(["arhetip", "varianta"])["Qh_nd"]
        vG41 = pd.read_csv(TAB / "02_variante_K41.csv", sep=";", encoding="utf-8-sig").set_index(["arhetip", "varianta"])["Qh_nd"]
        k["G13"] = [v41[(a, s_)] for a, s_ in zip(k["arhetip"], k["stanje"])]
        k["G11"] = [vG41[(a, s_)] for a, s_ in zip(k["arhetip"], k["stanje"])]
        sk = [("K4_1 vse", k)] + [(f"stanje: {s_}", x) for s_, x in k.groupby("stanje")] + \
             [(f"obdobje: {o}", x) for o, x in k.groupby(np.where(k["arhetip"] == "A4", "1981–2002", "pred 1981"))]
        for lab, x in sk:
            T = pd.concat([T, pd.DataFrame([{"del": "d K4_1", "arhetip": lab, "stanje": f"{len(x)} hiš", "izkaznice P50": x["Qh_nd_EI"].median(),
                                             "G11 s faktorjem (osnova)": x["G11"].median(), "G13": x["G13"].median(),
                                             "G11 / izkaznica (mediana razmerij)": (x["G11"] / x["Qh_nd_EI"]).median(),
                                             "G13 / izkaznica (mediana razmerij)": (x["G13"] / x["Qh_nd_EI"]).median()}])], ignore_index=True)
    for o in (TAB, OUT_T):
        T.round(3).to_csv(o / "22b_S0_G13_proti_izkaznicam.csv", sep=";", encoding="utf-8-sig", index=False)
    pd.set_option("display.width", 260)
    print(T.round(2).to_string(index=False))


# ---------------------------------------------------------------- G13 (končna, Gašper 22. 10. 2026): enotna geometrija, popravljena na velikost fonda
def korak_geo():
    """Toplotni ovoj na m² ogrevane površine G11 pri MEDIANI fonda (podvzorec z GURS tlorisom je večji: log-log regresija na A po arhetipu),
    pokritost izolirane površine na hišo (Eko količina / ovoj iste hiše, tabela 22e) → stroškovna geometrija = ovoj × pokritost.
    Okna = G11 (Eko). Izhod: tabele/22a_toplotni_ovoj_gurs.csv + predlog parametri.yaml."""
    d = ovoj()
    u = pd.read_csv(TAB / "22e_uskladitev_B2_ETICS.csv", sep=";", encoding="utf-8-sig")
    pok = {(r.arhetip, r.skupina): r.P50 for r in u[u["del"] == "2 ETICS"].itertuples()}
    rows, yml = [], []
    for a in ("A1", "A2", "A3", "A4"):
        x = d[d["arhetip"] == a]
        y = x[x["ocenjeno"]].copy()
        Am = x["A"].median()
        okna = g11(a)["okna"]
        bruto = (4 * np.sqrt(y["F"]) * H_ET * y["n_ogr"]) / y["A"]
        b1, c1 = np.polyfit(np.log(y["A"]), np.log(bruto), 1)
        b2, c2 = np.polyfit(np.log(y["A"]), np.log(y["F"] / y["A"]), 1)
        stena = np.exp(c1) * Am ** b1 - okna
        strop = np.exp(c2) * Am ** b2
        ps = pok[(a, "Eko fasada / stena toplotnega ovoja (G13)")]
        pr = pok[(a, "Eko streha / tloris etaže F")]
        G = g11(a)
        rows.append({"arhetip": a, "A fond mediana [m²]": Am, "A podvzorec GURS mediana [m²]": y["A"].median(), "n podvzorca": len(y),
                     "eksponent stena ~ A^b": b1, "eksponent streha ~ A^b": b2,
                     "ovoj: stena": stena, "ovoj: streha/strop": strop, "ovoj: tla/kletni strop": strop, "ovoj: okna": okna,
                     "pokritost stena (Eko / ovoj iste hiše)": ps, "pokritost streha (Eko / F iste hiše)": pr,
                     "strošek: stena": stena * ps, "strošek: streha": strop * pr, "strošek: tla": strop, "strošek: okna": okna,
                     "G11 (Eko mediane): stena / streha / tla / okna": f"{G['stena']:.2f} / {G['streha']:.2f} / {G['tla']:.2f} / {G['okna']:.3f}",
                     "G13a (podvzorec, brez popravka): stena / streha": f"{y['stena_G13'].div(y['A']).median():.2f} / {y['streha_G13'].div(y['A']).median():.2f}"})
        yml.append(f"    G13_{a}: {{vir: \"GURS toplotni ovoj pri mediani fonda (K13 geo, 22a)\", A: 100.0, stena: {100 * stena:.1f}, streha: {100 * strop:.1f}, "
                   f"okna: {100 * okna:.1f}, tla: {100 * strop:.1f}}}")
        yml.append(f"    G13S_{a}: {{vir: \"strošek: ovoj × pokritost Eko (stena {ps:.2f}, streha {pr:.2f}; 22e)\", A: 100.0, stena: {100 * stena * ps:.1f}, "
                   f"streha: {100 * strop * pr:.1f}, okna: {100 * okna:.1f}, tla: {100 * strop:.1f}}}")
    T = pd.DataFrame(rows)
    for o in (TAB, OUT_T):
        T.round(3).to_csv(o / "22a_toplotni_ovoj_gurs.csv", sep=";", encoding="utf-8-sig", index=False)
    pd.set_option("display.width", 250)
    print(T.round(2).T.to_string())
    print("\n".join(yml))


# ---------------------------------------------------------------- uskladitev (Gašper 22. 10. 2026): B2 in pokritost ETICS
def korak_uskladitev():
    """(1) B2: izkaznice hiš brez zabeleženega ukrepa ob izkaznici (st 0) razdeljene na: (a) hiše, ki tudi do 2026 nimajo nobenega ukrepa na ovoju
       (Eko ∪ GURS vseh let), (b) hiše, ki ukrep izvedejo PO izkaznici (izkaznica pred prenovo), (c) hiše z obnovo v GURS pred 2003 (filter ≥ 2003 jo izpusti).
       Če je (a) ≈ (b), razpršenost izkaznic pri st 0 ni posledica nezabeleženih prenov (B2 ne velja), ampak raznolikosti neprenovljenih hiš.
    (2) Po hišah z dejansko količino fasade (Eko 2023–2025) in GURS tlorisom: Eko fasada / stena toplotnega ovoja G13 (ista formula kot 22a).
    Izhod: tabele/22e_uskladitev_B2_ETICS.csv."""
    import os
    e = pd.read_parquet(AN / "cache" / "03_stavbe_gurs_ei.parquet")
    e = e[e["tc_ogrevalna"].astype(bool) & e["tip_stavbe"].isin(["enostanovanjska", "dvostanovanjska"]) & e["obdobje"].between(1, 4)].copy()
    lei = pd.to_numeric(e["leto_EI"], errors="coerce")
    yr = lambda c: pd.to_numeric(e[c], errors="coerce")                                                     # noqa: E731
    eko = pd.concat([yr(c) for c in ("leto_prvi_fasada", "leto_prvi_streha", "leto_prvi_okna_vrata", "leto_prvi_celovita_prenova",
                                        "leto_prvi_talna_izolacija", "leto_prvi_kletna_izolacija")], axis=1)
    gurs = pd.concat([yr(c) for c in ("LETO_OBN_FASADE", "LETO_OBN_STREHE", "okna_obn_min")], axis=1)
    prvi_vsi = pd.concat([eko, gurs.where(gurs >= 2003)], axis=1).min(axis=1)            # kot register modela (GURS ≥ 2003)
    gurs_pred03 = (gurs < 2003).any(axis=1).to_numpy()
    ok = (e["tip_EI"] == "rEI") & pd.to_numeric(e["Qh_nd_EI"], errors="coerce").between(5, 400) & lei.notna()
    st0 = ok & ~(prvi_vsi <= lei).fillna(False)                                          # brez zabeleženega ukrepa do leta izkaznice
    st0, brez = st0.to_numpy(bool), prvi_vsi.isna().to_numpy()
    po = (prvi_vsi > lei).fillna(False).to_numpy(bool)
    grp = np.select([st0 & brez & ~gurs_pred03, st0 & brez & gurs_pred03, st0 & po],
                    ["(a) brez ukrepa tudi do 2026", "(c) brez ukrepa ≥ 2003, a GURS obnova pred 2003", "(b) ukrep PO izkaznici (izkaznica pred prenovo)"], "")
    e["skupina"] = grp
    q = pd.to_numeric(e["Qh_nd_EI"], errors="coerce")
    v13 = pd.read_csv(TAB / "02_variante_G13.csv", sep=";", encoding="utf-8-sig").set_index(["arhetip", "varianta"])["Qh_nd"]
    vG = pd.read_csv(TAB / "02_variante.csv", sep=";", encoding="utf-8-sig").set_index(["arhetip", "varianta"])["Qh_nd"]
    rows = []
    for o, a in ARH.items():
        for s in ["(a) brez ukrepa tudi do 2026", "(b) ukrep PO izkaznici (izkaznica pred prenovo)", "(c) brez ukrepa ≥ 2003, a GURS obnova pred 2003", "vse st 0"]:
            m = (e["obdobje"] == o) & ((e["skupina"] == s) if s != "vse st 0" else (e["skupina"] != ""))
            x = q[m]
            rows.append({"del": "1 B2", "arhetip": a, "skupina": s, "n": int(m.sum()), "P25": x.quantile(.25), "P50": x.median(), "P75": x.quantile(.75),
                         "P90": x.quantile(.9), "model S0 G13": v13[(a, "BREZ")], "model S0 G11": vG[(a, "BREZ")],
                         "leto izkaznice mediana": lei[m].median()})
    # (2) ETICS / toplotni ovoj po hišah
    n = pd.read_feather(os.environ["K13_NALOZBE"])
    n = n[(n["Tip"] == "SFH") & (n["V statistiki"] == "DA") & (n["Enota"] == "m2") & n["Ukrep (koda)"].isin(["fasada", "streha"])].copy()
    n["stavba"] = n["KO-stavba|del"].str.split("|").str[0]
    n["Q"] = pd.to_numeric(n["Količina ukrepa"], errors="coerce")
    kol = n.pivot_table(index="stavba", columns="Ukrep (koda)", values="Q", aggfunc="sum")
    ly = n.groupby("stavba")["Leto izgradnje"].first().astype(float)
    g = pd.read_parquet(AN / "cache" / "02_stavbe_gurs.parquet", columns=["ID_Stavbe", "BRUTO_TLOR", "STEVILO_ET", "A_stan"]).drop_duplicates("ID_Stavbe")
    g["ID_Stavbe"] = g["ID_Stavbe"].astype(str)
    g = g.set_index("ID_Stavbe").reindex(kol.index)
    o_ = pd.cut(ly.reindex(kol.index), [0, 1945, 1970, 1980, 2002], labels=[1, 2, 3, 4]).astype(float)
    A = 0.85 * pd.to_numeric(g["A_stan"], errors="coerce").where(lambda x: x >= 20) * o_.map(HV["povrsina_G11_faktor"])
    et = pd.to_numeric(g["STEVILO_ET"], errors="coerce").where(lambda x: x.between(1, 5))
    F = pd.to_numeric(g["BRUTO_TLOR"], errors="coerce").where(lambda x: x >= 30) / et
    nog = np.clip(A / (0.85 * F), 1, et)
    okna = o_.map({1: .138, 2: .123, 3: .114, 4: .117}) * A
    stena = 4 * np.sqrt(F) * H_ET * nog - okna
    bruto_vse = 4 * np.sqrt(F) * H_ET * et
    d = pd.DataFrame({"o": o_, "fas": kol.get("fasada"), "str": kol.get("streha"), "stena": stena, "bruto": bruto_vse, "F": F, "A": A, "nog": nog, "et": et})
    for o, a in ARH.items():
        for lab, num, den in (("Eko fasada / stena toplotnega ovoja (G13)", "fas", "stena"), ("Eko fasada / bruto stena vseh etaž GURS", "fas", "bruto"),
                              ("Eko streha / tloris etaže F", "str", "F")):
            x = (d[num] / d[den])[(d["o"] == o)].replace([np.inf, -np.inf], np.nan).dropna()
            x = x[x.between(x.quantile(.01), x.quantile(.99))] if len(x) > 20 else x
            rows.append({"del": "2 ETICS", "arhetip": a, "skupina": lab, "n": len(x), "P25": x.quantile(.25), "P50": x.median(), "P75": x.quantile(.75),
                         "n_ogr / ET mediana": (d["nog"] / d["et"])[(d["o"] == o) & d[den].notna() & d[num].notna()].median()})
    T = pd.DataFrame(rows)
    for o_t in (TAB, OUT_T):
        T.round(3).to_csv(o_t / "22e_uskladitev_B2_ETICS.csv", sep=";", encoding="utf-8-sig", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 50)
    print(T.round(2).to_string(index=False))


# ---------------------------------------------------------------- točka 2: primerjalne tabele (zagon v delovni kopiji G13 po verigi)
KAZ22 = [("posameznik", "NPV stroškov 30 let, R1", "posameznik R1 [kEUR]"), ("posameznik", "NPV stroškov 30 let, R3", "posameznik R3 [kEUR]"),
         ("država", "družbeni NPV", "družbeni NPV [kEUR]"), ("država", "javni proračun NPV, R3", "proračun R3 [kEUR]"),
         ("okolje", "CO2e 30 let (obratovalni + vgrajeni)", "CO2e 30 let [t]"), ("država", "zimski uvoz (nov–feb), leto 20", "zimski uvoz [kWh/a]"),
         ("država", "obremenitev NN omrežja (maks. sočasni uvoz / oddaja PV)", "obremenitev NN [kW]")]
VZ22 = ["paket pred 1981", "paket 1981–2002", "paket"]


def korak_22c(g11, g13, tab, g13a=None):
    """22c: P3 − P1 in P3 − P1r — G11 | G13 | TABULA (| G13a, prvotna različica) iz tabel 09_primerjava_pristopov posameznih verig."""
    src = {"G11": g11, "G13": g13, "TABULA": tab, **({"G13a": g13a} if g13a else {})}
    T = {k: pd.read_csv(v, sep=";", encoding="utf-8-sig").set_index(["perspektiva", "kazalnik", "vzorec"]) for k, v in src.items()}
    rows = []
    for p, k, lab in KAZ22:
        for vz in VZ22:
            row = {"kazalnik": lab, "vzorec": vz}
            for g, t in T.items():
                r = t.loc[(p, k, vz)]
                row[f"{g}: P3 − P1 mediana"] = r["P3−P1 mediana"]
                row[f"{g}: delež P3 < P1 [%]"] = r["delež P3 < P1 [%]"]
                row[f"{g}: P3 − P1 vsota"] = r["P3−P1 vsota"]
                row[f"{g}: P3 − P1r mediana"] = r["P3−P1r mediana"]
                row[f"{g}: delež P3 < P1r [%]"] = r["delež P3 < P1r [%]"]
            row["enota vsote"] = T["G13"].loc[(p, k, vz), "enota vsote"]
            row["G13 − G11, delež P3 < P1 [o.t.]"] = row["G13: delež P3 < P1 [%]"] - row["G11: delež P3 < P1 [%]"]
            row["G13 − G11, delež P3 < P1r [o.t.]"] = row["G13: delež P3 < P1r [%]"] - row["G11: delež P3 < P1r [%]"]
            row["OPOZORILO > 10 o.t. (pred 1981)"] = vz == "paket pred 1981" and max(abs(row["G13 − G11, delež P3 < P1 [o.t.]"]), abs(row["G13 − G11, delež P3 < P1r [o.t.]"])) > 10
            rows.append(row)
    R = pd.DataFrame(rows)
    for o in (TAB, OUT_T):
        R.round(3).to_csv(o / "22c_primerjava_G11_G13_TABULA.csv", sep=";", encoding="utf-8-sig", index=False)
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 40)
    kol = ["kazalnik", "vzorec"] + [c for c in R.columns if "mediana" in c or "delež" in c] + ["OPOZORILO > 10 o.t. (pred 1981)"]
    print(R[kol].round(1).to_string(index=False))


def korak_22d_21e():
    """22d: razstavitev P3 − P1 = (P1r − P1) + (P3 − P1r) z deleži P3 < / = / > P1r (Tabela 7; »=« = hiše, kjer P3 nima ukrepa na ovoju in je enak P1r).
    21e: površina in deleži koristi po obdobju (fond, paket). Iste funkcije kot K2_3 (kazalniki); zagon v delovni kopiji G13."""
    import importlib.util
    sp = importlib.util.spec_from_file_location("k23", HERE / "K2_3_primerjava.py")
    K3 = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(K3)
    w = K3.E.load_units("fond")
    meta = pd.read_parquet(CACHE / "01_fond.parquet", columns=["ID_Stavbe", "paket_osnovni", "arhetip"]).set_index("ID_Stavbe").reindex(w.index)
    pk = meta["paket_osnovni"].astype(bool).to_numpy()
    pre = meta["arhetip"].isin(["A1", "A2", "A3"]).to_numpy()
    K = K3.kazalniki(w)
    brez = w["V4|brez_ovoja_P3"].to_numpy(float) > 0.5
    rows = []
    kaz = {lab: key for key in K for p, k, lab in KAZ22 if key[0] == p and key[1] == k}
    for lab, key in kaz.items():
        df = K[key].reindex(w.index)
        for vz, m in (("paket", pk), ("paket pred 1981", pk & pre), ("paket 1981–2002", pk & ~pre)):
            d = (df["P3"] - df["P1"]).to_numpy(float)[m]
            dr = (df["P1r"] - df["P1"]).to_numpy(float)[m]
            de = (df["P3"] - df["P1r"]).to_numpy(float)[m]
            vsc = key[4]
            tie = np.isclose(de, 0, atol=1e-6 * max(1.0, np.nanmax(np.abs(de))))
            rows.append({"kazalnik": lab, "vzorec": vz, "hiš": int(m.sum()), "enota vsote": key[5],
                         "P3 − P1 vsota": np.nansum(d) / vsc, "P1r − P1 vsota (dimenzioniranje)": np.nansum(dr) / vsc, "P3 − P1r vsota (ovoj)": np.nansum(de) / vsc,
                         "P3 boljši od P1r [%]": 100 * np.mean((de < 0) & ~tie), "P3 = P1r [%]": 100 * np.mean(tie), "P3 slabši od P1r [%]": 100 * np.mean((de > 0) & ~tie),
                         "hiš brez ukrepa na ovoju v P3 [%]": 100 * brez[m].mean(),
                         "P3 boljši od P1r med hišami z ukrepom na ovoju [%]": 100 * np.mean(((de < 0) & ~tie)[~brez[m]]) if (~brez[m]).any() else np.nan})
    D = pd.DataFrame(rows)
    # 21e
    A = w["V2|A"].to_numpy(float)
    r2 = []
    ke = {"zimski uvoz [GWh/a]": "zimski uvoz [kWh/a]", "CO2e [kt]": "CO2e 30 let [t]", "družbeni NPV [mio EUR]": "družbeni NPV [kEUR]",
          "posameznik R3 [mio EUR]": "posameznik R3 [kEUR]", "NN [MW]": "obremenitev NN [kW]"}
    for vz, base in (("fond", np.ones(len(w), bool)), ("paket", pk)):
        for ob, m in (("pred 1981", base & pre), ("1981–2002", base & ~pre)):
            row = {"vzorec": vz, "obdobje": ob, "hiš": int(m.sum()), "delež hiš [%]": 100 * m.sum() / base.sum(),
                   "ogrevana površina [1000 m²]": A[m].sum() / 1e3, "delež površine [%]": 100 * A[m].sum() / A[base].sum(), "A mediana [m²]": np.median(A[m])}
            for col, lab in ke.items():
                key = kaz[lab]
                df = K[key].reindex(w.index)
                row[f"{col.split(' [')[0]} P3−P1 [{col.split('[')[1]}"] = np.nansum((df["P3"] - df["P1"]).to_numpy(float)[m]) / key[4]
            r2.append(row)
    E21 = pd.DataFrame(r2)
    for o in (TAB, OUT_T):
        D.round(3).to_csv(o / "22d_razstavitev_P1r.csv", sep=";", encoding="utf-8-sig", index=False)
    E21.round(1).to_csv(TAB / "21e_povrsina_po_obdobju.csv", sep=";", encoding="utf-8-sig", index=False)
    pd.set_option("display.width", 260)
    print(D.round(1).to_string(index=False))
    print(E21.round(1).to_string(index=False))


def korak_b2():
    """Občutljivost umestitve: register (osnova, paket = tag _L3 iz K8) proti B2 (umestitev po Qh,nd, tag _B2); iste funkcije kot K8/K9.
    Izhod: tabele/22f_obcutljivost_B2.csv. Zagon v delovni kopiji G13 po verigi."""
    import importlib.util
    sp = importlib.util.spec_from_file_location("k21", HERE / "K2_1_drzava.py")
    K1 = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(K1)
    E, OKM = K1.E, K1.DR.OKM
    EK = PAR["ekonomika"]
    rows = []
    for lab, tag in (("register (osnova)", "_L3"), ("B2 (umestitev po Qh,nd)", "_B2")):
        w = E.load_units("fond", tag)
        n = {rez: E.npv_paths(w, rez, EK["diskont_zasebni"], E.pf_factory("konstantne"), EK["ovoj_osnova"]) for rez in ("R1", "R3")}
        rc = K1.racun(w)
        co2 = {p: (sum(OKM.op_co2(w, E.leto_varianta(p, t), t) for t in range(1, E.N + 1)) + sum(OKM.embodied(w, it, ar) for (y, it, ar) in E.POTI[p]["ev"])) / 1e3
               for p in ("P1", "P3", "P1r")}
        K = {"posameznik R1 [kEUR]": ({p: n["R1"][p] for p in ("P1", "P3", "P1r")}, 1e3), "posameznik R3 [kEUR]": ({p: n["R3"][p] for p in ("P1", "P3", "P1r")}, 1e3),
             "družbeni NPV [kEUR]": ({p: rc[p]["družbeni NPV"] for p in ("P1", "P3", "P1r")}, 1e3),
             "proračun R3 [kEUR]": ({p: rc[p]["R3 proračun neto"] for p in ("P1", "P3", "P1r")}, 1e3), "CO2e 30 let [t]": (co2, 1),
             "obremenitev NN [kW]": ({p: rc[p]["obremenitev NN omrežja [kW]"] for p in ("P1", "P3", "P1r")}, 1)}
        pre = w["arhetip"].isin(["A1", "A2", "A3"]).to_numpy()
        for kaz, (d, sc) in K.items():
            for vz, m in (("pred 1981", pre), ("1981–2002", ~pre), ("paket", np.ones(len(w), bool))):
                x = (np.asarray(d["P3"]) - np.asarray(d["P1"]))[m]
                y = (np.asarray(d["P3"]) - np.asarray(d["P1r"]))[m]
                rows.append({"umestitev": lab, "kazalnik": kaz, "vzorec": vz, "P3 − P1 mediana": np.median(x) / sc, "delež P3 < P1 [%]": 100 * (x < 0).mean(),
                             "P3 − P1r mediana": np.median(y) / sc, "delež P3 < P1r [%]": 100 * (y < 0).mean(),
                             "brez ukrepa na ovoju v P3 [%]": 100 * (w["V4|brez_ovoja_P3"].to_numpy(float)[m] > 0.5).mean()})
    T = pd.DataFrame(rows)
    for o in (TAB, OUT_T):
        T.round(3).to_csv(o / "22f_obcutljivost_B2.csv", sep=";", encoding="utf-8-sig", index=False)
    pd.set_option("display.width", 250)
    print(T.round(1).to_string(index=False))


if __name__ == "__main__":
    import sys
    ap = argparse.ArgumentParser()
    ap.add_argument("--korak", default="ab")
    ap.add_argument("poti", nargs="*")
    a = ap.parse_args()
    if a.korak == "22c":
        korak_22c(*a.poti)
    else:
        {"ab": korak_ab, "cd": korak_cd, "22d": korak_22d_21e, "usk": korak_uskladitev, "geo": korak_geo, "b2": korak_b2}[a.korak]()
