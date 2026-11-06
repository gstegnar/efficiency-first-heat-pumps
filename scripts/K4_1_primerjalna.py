"""Krog 4, K4_1 — primerjalna skupina »ovoj je že narejen« (naravni eksperiment ob kontrafaktičnem modelu).

Korak A (`--korak A`): opredelitev in štetje. Vir: ../Interno/2026 analiza/cache/07_stavbe_paket.parquet (eno-/dvostanovanjske s TČ, Eko sklad 2011–2025,
GURS, rEI; samo branje).
Opredelitev: ogrevalna TČ, obdobje gradnje 1–4 (A1–A4), brez novogradenj, vsaj en ukrep na ovoju (fasada, streha/strop, okna/vrata, tla/klet,
celovita prenova; Eko sklad ∪ GURS leta obnov ≥ 2003).
Novogradnja: staro_ogrevanje = novogradnja (07) ALI (šifra stare naprave −1 ali prazna, vloga od 2018, GURS leto gradnje ≥ leto vloge − 2).
Zaporedje po zadnjem ukrepu ovoja L glede na leto TČ T: ovoj_prej L ≤ T − 2; hkrati T − 1 ≤ L ≤ T + 1; ovoj_pozneje L ≥ T + 2.
Sestava (kot lestvica): okna → S1; streha + okna → S2; fasada + streha + okna (+ tla) → S3; ostalo po kombinaciji.
Izhodi: tabele/13_K4_1_stetje.csv (+ clanek1/tabele/), 13_K4_1_sestava.csv; cache/13_K4_1_skupina.parquet
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
TAB, OUT_T, CACHE = HERE / "tabele", HERE / "clanek1" / "tabele", HERE / "cache"
GURS_MIN = 2010                                              # krog 13 (Gašper 22. 10. 2026): ukrep v GURS pred 2010 ni ukrep URE (prej 2003)
ARH = {1: "A1", 2: "A2", 3: "A3", 4: "A4"}
K4 = PAR["krog4"]
TAG = __import__("os").environ.get("K41_TAG", "_K41")       # krog 11: K41_TAG / K41_BAZA (npr. _K41G11 / _G11) za geometrijo G11
BAZA = __import__("os").environ.get("K41_BAZA", "")                                                      # pripona vhodov 02 in izhodov 03 kroga 4 (osnova kroga 3 ostane)
ZAP = ["ovoj_prej", "hkrati", "ovoj_pozneje"]
ELEM = ("stena", "streha", "okna", "tla")
STANJE_EL = {"BREZ": (), "S1": ("okna",), "S2": ("streha", "okna"), "S3": ELEM, "SF": ("stena",), "SS": ("streha",),
             "SFO": ("stena", "okna"), "SFS": ("stena", "streha")}
CILJ = {"BREZ": "FSO", "S1": "FSO_S1", "S2": "FSO_S2", "S3": "S3", "SF": "FSO_SF", "SS": "FSO_SS", "SFO": "FSO_SFO", "SFS": "FSO_SFS"}
MANJKA = {s: tuple(e for e in ELEM if e not in el) for s, el in STANJE_EL.items()}     # manjkajoči ukrepi do ciljne prenove
STANJE_K = {s: i for i, s in enumerate(STANJE_EL)}


def leta(d, cols, gurs=False):
    cs = [pd.to_numeric(d[c], errors="coerce").astype(float) for c in cols if c in d]
    if not cs:
        return pd.DataFrame(index=d.index)
    y = pd.concat(cs, axis=1)
    if gurs:
        y = y.where(y >= GURS_MIN)
    return y


def novogradnja(d):
    lt = pd.to_numeric(d["leto_TC"], errors="coerce")
    sk = pd.to_numeric(d["tc_stara_kurilna"], errors="coerce")
    iz = pd.to_numeric(d["LETO_IZGRA"], errors="coerce")
    pravilo = ((sk == -1) | sk.isna()) & (lt >= 2018) & (iz >= lt - 2)
    return (d["staro_ogrevanje"] == "novogradnja") | pravilo.fillna(False)


def ukrepi_ovoja(d):
    """Po elementu: ali je ukrep (Eko ∪ GURS ≥ 2003) in leto zadnjega ukrepa."""
    el = {"fasada": (["leto_prvi_fasada", "leto_zadnji_fasada"], ["LETO_OBN_FASADE"]),
          "streha": (["leto_prvi_streha", "leto_zadnji_streha"], ["LETO_OBN_STREHE"]),
          "okna": (["leto_prvi_okna_vrata", "leto_zadnji_okna_vrata"], ["okna_obn_min", "okna_obn_max"]),
          "tla": (["leto_prvi_talna_izolacija", "leto_zadnji_talna_izolacija", "leto_prvi_kletna_izolacija", "leto_zadnji_kletna_izolacija"], []),
          "celovita": (["leto_prvi_celovita_prenova", "leto_zadnji_celovita_prenova"], [])}
    ima, zadnje = {}, {}
    for k, (eko, gurs) in el.items():
        y = pd.concat([leta(d, eko), leta(d, gurs, gurs=True)], axis=1)
        ima[k] = y.notna().any(axis=1)
        zadnje[k] = y.max(axis=1)
    return pd.DataFrame(ima), pd.DataFrame(zadnje)


def sestava(I):
    f, s, o, t, c = (I[k].to_numpy() for k in ("fasada", "streha", "okna", "tla", "celovita"))
    lab = np.select([c | (f & s & o), s & o & ~f, o & ~f & ~s, f & ~s & ~o, f & o & ~s, f & s & ~o, s & ~f & ~o],
                    ["S3 fasada+streha+okna (+tla) / celovita", "S2 streha+okna", "S1 okna", "samo fasada", "fasada+okna", "fasada+streha", "samo streha"],
                    "samo tla / drugo")
    return lab


def skupina():
    d = pd.read_parquet(AN / "cache" / "07_stavbe_paket.parquet")
    I, Z = ukrepi_ovoja(d)
    d["ima_ovoj"] = I.any(axis=1)
    d["leto_ovoj_zadnji"] = Z.max(axis=1)
    d["novogradnja_k4"] = novogradnja(d)
    T = pd.to_numeric(d["leto_TC"], errors="coerce")
    L = d["leto_ovoj_zadnji"]
    d["zaporedje"] = np.select([L <= T - 2, (L >= T - 1) & (L <= T + 1), L >= T + 2], ["ovoj_prej", "hkrati", "ovoj_pozneje"], "brez")
    d["sestava"] = sestava(I)
    for k in I:
        d[f"ovoj_{k}"] = I[k]
    d["arhetip"] = d["obdobje"].map(ARH)
    return d


def korak_a():
    d = skupina()
    osnova = d["obdobje"].between(1, 4) & ~d["novogradnja_k4"]
    k = d[osnova & d["ima_ovoj"]].copy()
    vse = {
        "eno-/dvostanovanjske s TČ (07)": len(d),
        "obdobje 1–4, brez novogradenj": int(osnova.sum()),
        "  od tega obdobje imputirano": int((osnova & d["obdobje_imputirano"].astype(bool)).sum()),
        "K4_1: z ukrepom na ovoju (obdobje 1–4)": len(k),
        "  od tega obdobje znano (neimputirano)": int((~k["obdobje_imputirano"].astype(bool)).sum()),
        "  brez ovoja (za primerjavo; obdobje 1–4)": int((osnova & ~d["ima_ovoj"]).sum()),
        "kontrola: Q1+Q2 z ovojem (07: brez_ovoja = False)": int((d["Qr"].isin(["Q1", "Q2"]) & ~d["brez_ovoja"].astype(bool)).sum()),
        "kontrola: Q1+Q2 z ovojem, ta opredelitev": int((d["Qr"].isin(["Q1", "Q2"]) & d["ima_ovoj"] & ~d["novogradnja_k4"]).sum()),
        "K4_1 po razredu Qh,nd ob TČ: Q1 / Q2 / Q3 / Q4 / neznano": " / ".join(str(int((k["Qr"] == q).sum())) for q in ("Q1", "Q2", "Q3", "Q4", "neznano")),
    }
    print("\n".join(f"{a}: {b}" for a, b in vse.items()))
    tab = (k.assign(PV=np.where(k["ima_PV"].astype(bool), "s PV", "brez PV"))
           .pivot_table(index=["arhetip", "sestava"], columns=["zaporedje", "PV"], values="ID_Stavbe", aggfunc="count", fill_value=0))
    tab["skupaj"] = tab.sum(axis=1)
    z = k.pivot_table(index="arhetip", columns="zaporedje", values="ID_Stavbe", aggfunc="count", fill_value=0)
    z["skupaj"] = z.sum(axis=1)
    z.loc["skupaj"] = z.sum()
    s = k.pivot_table(index="sestava", columns="arhetip", values="ID_Stavbe", aggfunc="count", fill_value=0)
    s["skupaj"] = s.sum(axis=1)
    s["delež [%]"] = (100 * s["skupaj"] / s["skupaj"].sum()).round(1)
    s = s.sort_values("skupaj", ascending=False)
    pv = k.pivot_table(index="arhetip", columns="zaporedje", values="ima_PV", aggfunc=lambda x: round(100 * x.astype(bool).mean(), 1))
    for t, f in ((z, "13_K4_1_stetje.csv"), (s, "13_K4_1_sestava.csv"), (tab, "13_K4_1_stetje_podrobno.csv")):
        for o in (TAB, OUT_T):
            t.to_csv(o / f, sep=";", encoding="utf-8-sig")
    pd.Series(vse).to_csv(TAB / "13_K4_1_povzetek.csv", sep=";", encoding="utf-8-sig", header=["vrednost"])
    k.to_parquet(CACHE / "13_K4_1_skupina.parquet", index=False)
    pd.set_option("display.width", 250)
    print("\nPo obdobju in zaporedju:\n", z.to_string())
    print("\nSestava ukrepov (vse zaporedje):\n", s.to_string())
    print("\nDelež s PV [%] po obdobju in zaporedju:\n", pv.to_string())


# ---------------------------------------------------------------- korak B1: dodatna stanja ovoja (HECM, ista koda kot lestvica)
def korak_stanja():
    """Dodatna stanja (krog4.stanja_dodatna) in ciljne prenove iz njih; celovita ≤ 50 (zaostritev g kot v 02_lestvica_stanj).
    Urni nizi cache/02_urno_<A>_<var>_K41.parquet; tabele/02_variante_K41.csv = osnovna lestvica (kopija, nespremenjena) + nova stanja."""
    import importlib
    import shutil
    from multiprocessing import Pool
    sys.path.insert(0, str(HERE))
    L = importlib.import_module("02_lestvica_stanj")              # nastavi preglasitve HECM (prezračevanje) kot v krogu 3
    M = L.M
    L.nastavi_S0()
    V = {}
    for s, uk in K4["stanja_dodatna"].items():
        V[s] = dict(uk)
        t = dict(uk)
        for el in L.ELEM:
            t.setdefault(el, L.FSO[el])
        t["n50_max"] = uk.get("n50_max", L.FSO["n50_max"]) if "okna" in uk else L.FSO["n50_max"]
        V[CILJ[s]] = t
    tasks = [(f"CLK-SI-SFH-{a}-{v}-K41", L.vhod(a, v, uk), a, v) for a in L.ARH for v, uk in V.items()]
    with Pool(M.HV["delavcev"]) as pool:
        res = pool.map(M.simulate, [(n, b) for n, b, _, _ in tasks])
    QMAX, MREZA = L.K3["celovita_max_Qhnd"], L.K3["celovita_g_mreza"]
    gC = {}
    pre = [(i, t) for i, t in enumerate(tasks) if t[3].startswith("FSO_") and res[i][0]["Qh_nd"] > QMAX]
    if pre:
        mt = []
        for i, (name, b, a, v) in pre:
            for g in MREZA[1:]:
                u2 = {k: (x * g if k in ("stena", "streha", "tla") else x) for k, x in V[v].items()}
                mt.append((i, g, (f"{name}-g{int(round(100 * g))}", L.vhod(a, v, u2))))
        with Pool(M.HV["delavcev"]) as pool:
            mres = pool.map(M.simulate, [x[2] for x in mt])
        for i, (name, b, a, v) in pre:
            cand = [(g, r, x) for (j, g, x), r in zip(mt, mres) if j == i]
            ok = [c for c in cand if c[1][0]["Qh_nd"] <= QMAX]
            g, r, (n2, b2) = max(ok, key=lambda c: c[0]) if ok else min(cand, key=lambda c: c[1][0]["Qh_nd"])
            res[i], tasks[i], gC[(a, v)] = r, (n2, b2, a, v), g
    rows = []
    for (r, hourly), (name, bui, a, v) in zip(res, tasks):
        h = hourly[["T_ext", "T_op", "Q_H", "Q_C", "Q_W", "G_h"]].copy()
        h.to_parquet(CACHE / f"02_urno_{a}_{v}{TAG}.parquet")
        u = bui["transmittance_U_elements"]
        rows.append({"arhetip": a, "varianta": v, "f_U": 1.0, "g_celovita": gC.get((a, v), 1.0), "U_stena": round(u[1], 3), "U_streha": round(u[9], 3),
                     "U_tla": round(u[0], 3), "U_okna": round(u[5], 2), "n50": bui["airtightness_n50"], "rekuperacija": bui["heat_recovery_efficiency"],
                     **{k: (round(x, 2) if isinstance(x, float) else x) for k, x in r.items() if k != "ime"}})
    nov = pd.DataFrame(rows)
    base = pd.read_csv(TAB / f"02_variante{BAZA}.csv", sep=";", encoding="utf-8-sig")
    base = base[~base["varianta"].isin(nov["varianta"].unique())]          # krog 13: osnovna lestvica G13 že vsebuje dodatna stanja — brez podvajanja
    for r in base.itertuples():
        f_ = CACHE / f"02_urno_{r.arhetip}_{r.varianta}{BAZA}.parquet"
        if f_.exists():
            shutil.copyfile(f_, CACHE / f"02_urno_{r.arhetip}_{r.varianta}{TAG}.parquet")
    pd.concat([base, nov], ignore_index=True).to_csv(TAB / f"02_variante{TAG}.csv", sep=";", encoding="utf-8-sig", index=False)
    q = pd.concat([base, nov]).pivot_table(index="arhetip", columns="varianta", values="Qh_nd")
    cols = ["BREZ", "S1", "SS", "SF", "S2", "SFO", "SFS", "S3", "FSO", "FSO_S1", "FSO_SS", "FSO_SF", "FSO_S2", "FSO_SFO", "FSO_SFS"]
    pd.set_option("display.width", 250)
    print(nov[["arhetip", "varianta", "U_stena", "U_streha", "U_tla", "U_okna", "n50", "g_celovita", "Qh_nd", "QH_P999_W_m2"]].to_string(index=False))
    print("\nQh,nd po stanju in cilju [kWh/m²a]:\n", q[cols].round(1).to_string())


# ---------------------------------------------------------------- korak B2: vhod za urni model (03 --fond … --tag _K41)
def stanje_iz(k, streha_eko=False):
    """Stanje na lestvici po sestavi ukrepov iz registra (krog4.sestava_stanje); streha_eko: streha šteje le iz Eko sklada (občutljivost 1c)."""
    if not streha_eko:
        return k["sestava"].map(K4["sestava_stanje"])
    I = pd.DataFrame({e: k[f"ovoj_{e}"].astype(bool) for e in ("fasada", "streha", "okna", "tla", "celovita")})
    I["streha"] = leta(k, ["leto_prvi_streha", "leto_zadnji_streha"]).notna().any(axis=1) | (I["celovita"])
    return pd.Series(sestava(I), index=k.index).map(K4["sestava_stanje"])


def korak_fond():
    """Tabela stavb za 03: (1) K4_1 po registru; (2) K4_1, kjer se stanje spremeni, če streha šteje le iz Eko sklada (ID|streha_eko);
    (3) zaklenjeni fond 29.977 po registru = S0 (ID); (4) zaklenjeni fond z Qh,nd kot v modelu članka (rEI, položaj na lestvici; ID|rEI).
    Opazovani PV: hiše brez PV dobijo 0 kWp (ima_PV = True, PV_kWp = 0 — varianta V2 je tako opazovano stanje)."""
    k = pd.read_parquet(CACHE / "13_K4_1_skupina.parquet")
    f = pd.read_parquet(CACHE / "01_fond.parquet")
    st07 = pd.read_parquet(AN / "cache" / "07_stavbe_paket.parquet", columns=["ID_Stavbe", "tip_stavbe"])
    kol = ["ID_Stavbe", "obdobje", "arhetip", "A_ogr", "A_vir", "tc_kW", "tc_kW_imputirano", "tc_kW_uporabljena", "tip_tc", "PV_kWp", "ima_PV",
           "leto_TC", "staro_ogrevanje", "ID_Regije", "paket_osnovni"]
    k = k[k["obdobje"].between(1, 4)].copy()
    for c in ("ima_PV", "tc_kW_imputirano", "paket_osnovni"):
        k[c] = k[c].astype(float).fillna(0).astype(bool)
        f[c] = f[c].astype(float).fillna(0).astype(bool)
    k["arhetip"] = k["obdobje"].astype(int).map(ARH)
    k["stanje"] = stanje_iz(k)
    k["stanje_obc"] = stanje_iz(k, streha_eko=True)
    k["skupina"] = k["zaporedje"]
    med_k = k[k["ima_PV"]].groupby("arhetip")["PV_kWp"].median()
    med_f = f[f["paket_osnovni"]].groupby("arhetip")["PV_kWp"].median()
    T = []
    a = k[kol + ["stanje", "skupina"]].copy()
    a["PV_kWp"] = np.where(a["ima_PV"], a["PV_kWp"].fillna(a["arhetip"].map(med_k)), 0.0)
    T.append(a)
    b = k.loc[k["stanje_obc"] != k["stanje"], kol + ["stanje_obc", "skupina"]].rename(columns={"stanje_obc": "stanje"})
    b["PV_kWp"] = np.where(b["ima_PV"], b["PV_kWp"].fillna(b["arhetip"].map(med_k)), 0.0)
    b["ID_Stavbe"] = b["ID_Stavbe"].astype(str) + "|streha_eko"
    T.append(b)
    z = f[kol].copy()
    z["PV_kWp"] = np.where(z["ima_PV"], z["PV_kWp"].fillna(z["arhetip"].map(med_f)), 0.0)
    z["stanje"], z["skupina"] = "BREZ", "zaklenjene"
    T.append(z)
    q = pd.read_parquet(CACHE / "03_fond_letno.parquet", columns=["ID_Stavbe", "varianta", "qh"])
    q = q[q["varianta"] == "V2"].set_index("ID_Stavbe")["qh"]
    z2 = z.copy()
    z2["stanje"], z2["skupina"] = np.nan, "zaklenjene_rEI"
    z2["Qhnd_lastna_EI"] = z2["ID_Stavbe"].map(q)
    z2["ID_Stavbe"] = z2["ID_Stavbe"].astype(str) + "|rEI"
    T.append(z2)
    d = pd.concat(T, ignore_index=True)
    d["tip_stavbe"] = d["ID_Stavbe"].astype(str).str.split("|").str[0].map(st07.assign(ID_Stavbe=st07["ID_Stavbe"].astype(str)).set_index("ID_Stavbe")["tip_stavbe"])
    d["ima_PV_dej"] = d["ima_PV"]
    d["ima_PV"] = True
    d["stanje_var"] = d["stanje"]
    d["cilj_var"] = d["stanje"].map(CILJ)
    d["stanje_k"] = d["stanje"].map(STANJE_K)
    d["Qhnd_lastna_EI"] = np.where(d["stanje"].notna(), np.nan, d.get("Qhnd_lastna_EI"))
    v = pd.read_csv(TAB / f"02_variante{TAG}.csv", sep=";", encoding="utf-8-sig").set_index(["arhetip", "varianta"])["Qh_nd"]
    m = d["stanje"].notna()
    d.loc[m, "Qhnd_lastna_EI"] = [v[(x, s)] for x, s in zip(d.loc[m, "arhetip"], d.loc[m, "stanje"])]
    d["Qr"] = np.where(d["arhetip"].isin(["A1", "A2", "A3"]), "Q1", "Q2")
    d["stratum"] = d["arhetip"] + "|" + d["skupina"]
    d["ID_Stavbe"] = d["ID_Stavbe"].astype(str)
    d.to_parquet(CACHE / "13_K4_1_fond.parquet", index=False)
    print(d.groupby(["skupina", "stanje"], dropna=False).size().unstack(fill_value=0).to_string())
    print(f"skupaj {len(d)} vrstic")


# ---------------------------------------------------------------- korak B3: ohranitev stanja proti dokončanju (tri perspektive)
def _moduli():
    import importlib.util
    s = importlib.util.spec_from_file_location("k21", HERE / "K2_1_drzava.py")
    K1 = importlib.util.module_from_spec(s)
    s.loader.exec_module(K1)
    return K1, K1.E, K1.DR, K1.DR.OKM


def poti_k4(R):
    """K0 ohranitev: opazovano stanje (V2: obstoječa TČ in PV), zamenjava TČ enake moči v letu R.
    K1 dokončanje: manjkajoči ukrepi do ciljne prenove v letu 0, obstoječa TČ in PV na prenovljeni hiši (V8), v letu R nova, manjša TČ (V9).
    TČ in PV, ki že obstajata, sta potopljena stroška (v obeh poteh enaka)."""
    return {"K0": {"seg": [(0, "V2")], "ev": [(R, "TCz", "e")]},
            "K1": {"seg": [(0, "V8"), (R, "V9")], "ev": [(0, "FSO", None), (R, "TCz", "FSO")]}}


def korak_b():
    K1, E, DR, OKM = _moduli()
    fd = pd.read_parquet(CACHE / "13_K4_1_fond.parquet").set_index("ID_Stavbe")
    w = E.load_units("fond", TAG)
    w = w.drop(columns=["staro_ogrevanje"]).join(fd[["stanje", "skupina", "staro_ogrevanje", "leto_TC", "ID_Regije", "obdobje", "ima_PV_dej", "tc_kW_uporabljena", "A_ogr"]])
    w["staro_ogrevanje"] = w["staro_ogrevanje"].fillna("")
    w["V0"] = np.select([w["staro_ogrevanje"] == "biomasa", w["staro_ogrevanje"] == "plin"], ["V0b", "V0c"], "V0a")
    orig = E.ovoj_delez

    def ovoj_delez_k4(ww, kind):
        """K4_1: manjkajoči ukrepi po stanju iz registra (MANJKA); vrstice brez stanja (zaklenjene, model članka) po lestvici kroga 3."""
        n = len(ww)
        out = {el: np.zeros(n) for el in ELEM}
        st = ww["stanje"].to_numpy(object) if "stanje" in ww else np.full(n, None, object)
        ima = pd.notna(st)
        brez = ww["V4|brez_ovoja_P3"].to_numpy(float) > 0.5
        for el in ELEM:
            out[el][ima] = [float(el in MANJKA[s]) for s in st[ima]]
            out[el][brez] = 0.0
        if (~ima).any():
            o = orig(ww[~ima], kind)
            for el in ELEM:
                out[el][~ima] = o[el]
        return out
    for mod in (E, OKM.EKM):
        mod.ovoj_delez = ovoj_delez_k4
    L = PAR["ekonomika"]["TC_let"]
    w["R"] = np.clip(pd.to_numeric(w["leto_TC"], errors="coerce").fillna(2020).astype(int) + L - PAR["ekonomika"]["leto0"], 1, L)
    res = []
    for R, g in w.groupby("R"):
        P = poti_k4(int(R))
        for mod in (E, OKM.EKM):
            mod.POTI = P
        DR.POTI = K1.POTI = list(P)
        o = pd.DataFrame(index=g.index)
        for rez in ("R1", "R3"):
            n = E.npv_paths(g, rez, PAR["ekonomika"]["diskont_zasebni"], E.pf_factory("konstantne"), PAR["ekonomika"]["ovoj_osnova"])
            for p in P:
                o[f"{p}|NPV {rez}"] = n[p]
        rc = K1.racun(g)
        sp_nom, _ = E.spodbude_poti(g)
        for p in P:
            o[f"{p}|proračun R3"] = rc[p]["R3 proračun neto"]
            o[f"{p}|družbeni NPV"] = rc[p]["družbeni NPV"]
            o[f"{p}|NN kW"] = rc[p]["obremenitev NN omrežja [kW]"]
            o[f"{p}|spodbude"] = sp_nom[p]
            op = sum(OKM.op_co2(g, E.leto_varianta(p, t), t) for t in range(1, E.N + 1))
            emb = sum(OKM.embodied(g, item, arg) for (y, item, arg) in P[p]["ev"])
            o[f"{p}|CO2e t"] = (op + emb) / 1000
            o[f"{p}|investicija"] = E.envelope_cost(g, "FSO", PAR["ekonomika"]["ovoj_osnova"])[0] if p == "K1" else 0.0
        res.append(o)
    o = pd.concat(res)
    w = w.join(o)
    w.reset_index().to_parquet(CACHE / "13_K4_1_rezultati.parquet", index=False)
    print(f"rezultati: {len(w)} vrstic → cache/13_K4_1_rezultati.parquet")


# ---------------------------------------------------------------- korak B4: glavna tabela 13_primerjalna.csv in Fig9
def wq(x, w, q=0.5):
    """Utežen kvantil (uteži CEM)."""
    x, w = np.asarray(x, float), np.asarray(w, float)
    m = ~np.isnan(x) & (w > 0)
    if not m.any():
        return np.nan
    o = np.argsort(x[m])
    c = np.cumsum(w[m][o])
    return float(x[m][o][np.searchsorted(c, q * c[-1])])


def cem_utezi(t, c):
    """Uteži zaklenjenih (c) za ujemanje s K4_1 (t): celice obdobje × kvartil površine × regija × razred leta TČ × energent (kot K2_4);
    utež = n_t / n_c v celici (ATT); vrne uteži in delež K4_1 v celicah z zaklenjeno hišo."""
    A = pd.concat([t["A_ogr"], c["A_ogr"]])
    qa = A.quantile([.25, .5, .75]).to_numpy()
    mj = K4["cem_leta_TC_meje"]

    def cel(d):
        en = d["staro_ogrevanje"].where(d["staro_ogrevanje"].isin(K4["cem_energenti"]), "drugo")
        lt = pd.to_numeric(d["leto_TC"], errors="coerce")
        return (d["obdobje"].astype(int).astype(str) + "|" + np.searchsorted(qa, d["A_ogr"].to_numpy()).astype(str) + "|"
                + d["ID_Regije"].astype(str) + "|" + np.searchsorted(mj, lt.fillna(2020).to_numpy()).astype(str) + "|" + en.astype(str))
    ct, cc = cel(t), cel(c)
    nt, nc = ct.value_counts(), cc.value_counts()
    w = cc.map(nt).fillna(0) / cc.map(nc)
    return w.to_numpy(float), float(ct.isin(nc.index).mean()), int((w > 0).sum())


def korak_tabela():
    r = pd.read_parquet(CACHE / "13_K4_1_rezultati.parquet")
    fl = pd.read_parquet(CACHE / f"03_fond_letno{TAG}.parquet", columns=["ID_Stavbe", "varianta", "el_TC_kWh", "el_grelnik_kWh"])
    fl = fl[fl["varianta"] == "V2"].set_index("ID_Stavbe")
    dup = set(fl.index[fl.index.duplicated()])                    # hiši v K4_1 (le tla) in hkrati v fondu zaklenjenih (07 brez_ovoja) — izločeni
    fl = fl[~fl.index.isin(dup)]
    r = r[~r["ID_Stavbe"].str.split("|").str[0].isin(dup)].copy()
    print(f"izločeni podvojeni ID (K4_1 in fond hkrati): {sorted(dup)}")
    r["el_TC"] = r["ID_Stavbe"].map(fl["el_TC_kWh"] + fl["el_grelnik_kWh"])
    r["baza"] = r["ID_Stavbe"].str.split("|").str[0]
    # rEI (empirična kontrola): K4_1 — izkaznica po zadnjem ukrepu ovoja; zaklenjeni — lastna rEI (01_fond)
    k = pd.read_parquet(CACHE / "13_K4_1_skupina.parquet")
    k["ID_Stavbe"] = k["ID_Stavbe"].astype(str)
    okk = (k["tip_EI"] == "rEI") & (pd.to_numeric(k["leto_EI"], errors="coerce") >= k["leto_ovoj_zadnji"])
    rei = pd.concat([k.loc[okk].set_index("ID_Stavbe")["Qh_nd_EI"],
                     pd.read_parquet(CACHE / "01_fond.parquet", columns=["ID_Stavbe", "Qhnd_lastna_EI"]).astype({"ID_Stavbe": str}).set_index("ID_Stavbe")["Qhnd_lastna_EI"].dropna()])
    r["rEI"] = r["baza"].map(rei[~rei.index.duplicated()])
    for p in ("K0", "K1"):
        r[f"{p}|zima20"] = r[f"{'V2' if p == 'K0' else 'V9'}|uvoz_zima_kWh"]
    K4b = r[r["skupina"].isin(ZAP) & ~r["ID_Stavbe"].str.contains(r"\|")].copy()
    obc = K4b.set_index("ID_Stavbe")
    sub = r[r["ID_Stavbe"].str.endswith("|streha_eko")].set_index("baza")
    obc = pd.concat([obc.drop(sub.index), sub.drop(columns=["ID_Stavbe"])]).reset_index(drop=True)
    Z = r[r["skupina"] == "zaklenjene"].copy()
    Zr = r[r["skupina"] == "zaklenjene_rEI"].copy()
    wz, pokr, nz = cem_utezi(K4b, Z)
    Zr = Zr.set_index("baza").loc[Z["baza"]].reset_index()
    G = {"zaklenjene, ujemanje (S0 po registru)": (Z, wz), "zaklenjene, ujemanje (Qh,nd po rEI, model članka)": (Zr, wz)}
    for z in ZAP:
        g = K4b[K4b["skupina"] == z]
        G[z] = (g, np.ones(len(g)))
    G["K4_1 skupaj"] = (K4b, np.ones(len(K4b)))
    G["K4_1 skupaj, streha samo Eko sklad (obč.)"] = (obc, np.ones(len(obc)))
    D = {"NPV R1": ("posameznik", "kEUR", 1e3), "NPV R3": ("posameznik", "kEUR", 1e3), "proračun R3": ("država", "kEUR", 1e3),
         "družbeni NPV": ("država", "kEUR", 1e3), "CO2e t": ("okolje", "t", 1), "zima20": ("država", "kWh/a", 1), "NN kW": ("država", "kW", 1)}
    lab = {"NPV R1": "NPV stroškov 30 let posameznika, R1", "NPV R3": "NPV stroškov 30 let posameznika, R3", "proračun R3": "javni proračun NPV, R3",
           "družbeni NPV": "družbeni NPV", "CO2e t": "CO2e 30 let (obratovalni + vgrajeni)", "zima20": "zimski uvoz (nov–feb), leto 20",
           "NN kW": "obremenitev NN omrežja (maks. sočasni uvoz / oddaja PV)"}
    st_lab = {"BREZ": "S0 (brez ukrepa)", "S1": "S1 okna", "SS": "samo streha", "SF": "samo fasada", "S2": "S2 streha + okna", "SFO": "fasada + okna",
              "SFS": "fasada + streha", "S3": "S3 celovita"}
    rows = []

    def add(sk, kaz, en, fn, vrsta="mediana"):
        row = {"sklop": sk, "kazalnik": kaz, "enota": en, "statistika": vrsta}
        for c, (g, w) in G.items():
            try:
                row[c] = fn(g, w)
            except (KeyError, ValueError):
                row[c] = np.nan
        rows.append(row)
    add("register", "stavb (zaklenjene: v ujemanju / utežene na K4_1)", "št.", lambda g, w: len(g) if w.min() == 1 else f"{int((w > 0).sum())} / {w.sum():.0f}", "število")
    add("register", "moč TČ", "kW", lambda g, w: wq(g["tc_kW_uporabljena"], w))
    add("register", "moč TČ na m² ogrevane površine", "W/m²", lambda g, w: wq(1000 * g["tc_kW_uporabljena"] / g["A_ogr"], w))
    add("register", "delež s PV", "%", lambda g, w: 100 * np.average(g["ima_PV_dej"].astype(float), weights=w), "delež")
    add("register", "PV (hiše s PV)", "kWp", lambda g, w: wq(g["V2|PV_kWp"].where(g["ima_PV_dej"].astype(bool)), w))
    add("register", "ogrevana površina", "m²", lambda g, w: wq(g["A_ogr"], w))
    add("register", "leto vgradnje TČ", "leto", lambda g, w: wq(pd.to_numeric(g["leto_TC"], errors="coerce"), w))
    add("empirična kontrola (rEI)", "Qh,nd iz rEI (K4_1: izkaznica po zadnjem ukrepu)", "kWh/m²a", lambda g, w: wq(g["rEI"], w))
    add("empirična kontrola (rEI)", "hiš z rEI", "št.", lambda g, w: int(g["rEI"].notna().sum()), "število")
    add("empirična kontrola (rEI)", "Qh,nd model (hiše z rEI)", "kWh/m²a", lambda g, w: wq(g["V2|Qh_nd"].where(g["rEI"].notna()), w))
    for s, l_ in st_lab.items():
        add("lestvica stanj", f"delež v stanju {l_}", "%", lambda g, w, s=s: 100 * np.average((g["stanje"] == s).astype(float), weights=w)
            if g["stanje"].notna().any() else np.nan, "delež")
    add("lestvica stanj", "položaj po Qh,nd (model članka): S0–S1 / S1–S2 / S2–S3 / S3", "%",
        lambda g, w: " / ".join(f"{100 * np.average((g['V4|lestvica_k'].astype(int) == j).astype(float), weights=w):.0f}" for j in range(4))
        if g["stanje"].isna().all() else "", "delež")
    add("lestvica stanj", "delež ≤ ciljna prenova + 5 kWh/m²a (brez ukrepa na ovoju)", "%",
        lambda g, w: 100 * np.average((g["V4|brez_ovoja_P3"] > 0.5).astype(float), weights=w), "delež")
    for el, l_ in (("stena", "fasada"), ("streha", "streha"), ("okna", "okna"), ("tla", "tla/strop kleti")):
        add("lestvica stanj", f"manjka do ciljne prenove: {l_}", "%", lambda g, w, el=el: 100 * np.average(
            np.array([float(el in MANJKA[s]) for s in g["stanje"]]) * (g["V4|brez_ovoja_P3"].to_numpy() <= 0.5), weights=w)
            if g["stanje"].notna().all() else np.nan, "delež")
    add("opazovano stanje (simulirano)", "Qh,nd", "kWh/m²a", lambda g, w: wq(g["V2|Qh_nd"], w))
    add("opazovano stanje (simulirano)", "elektrika TČ + grelnik", "kWh/a", lambda g, w: wq(g["el_TC"], w))
    add("opazovano stanje (simulirano)", "zimski uvoz (nov–feb)", "kWh/a", lambda g, w: wq(g["V2|uvoz_zima_kWh"], w))
    add("opazovano stanje (simulirano)", "prispevek h sistemski konici", "kW", lambda g, w: wq(g["V2|uvoz_konica_sis_kW"], w))
    add("opazovano stanje (simulirano)", "obremenitev NN omrežja", "kW", lambda g, w: wq(g["K0|NN kW"], w))
    for key in ("CO2e t", "NPV R1", "NPV R3", "proračun R3", "družbeni NPV"):
        sk, en, sc = D[key]
        add("opazovano stanje (simulirano)", f"{lab[key]} — ohranitev stanja (K0)", en, lambda g, w, key=key, sc=sc: wq(g[f"K0|{key}"] / sc, w))
    add("dokončanje (K1) − ohranitev (K0)", "investicija v manjkajoče ukrepe (bruto)", "kEUR", lambda g, w: wq(g["K1|investicija"] / 1e3, w))
    add("dokončanje (K1) − ohranitev (K0)", "spodbude za ovoj", "kEUR", lambda g, w: wq(g["K1|spodbude"] / 1e3, w))
    for key in ("NPV R1", "NPV R3", "proračun R3", "družbeni NPV", "CO2e t", "zima20", "NN kW"):
        sk, en, sc = D[key]
        dd = lambda g, key=key, sc=sc: (g[f"K1|{key}"] - g[f"K0|{key}"]) / sc   # noqa: E731
        add(f"dokončanje (K1) − ohranitev (K0): {sk}", lab[key], en, lambda g, w, dd=dd: wq(dd(g), w))
        add(f"dokončanje (K1) − ohranitev (K0): {sk}", lab[key], "%", lambda g, w, dd=dd: 100 * np.average((dd(g) < 0).astype(float), weights=w), "delež K1 < K0")
        vs = {"kEUR": (1e6, "mio EUR"), "t": (1e3, "kt"), "kWh/a": (1e6, "GWh/a"), "kW": (1e3, "MW")}[en]
        add(f"dokončanje (K1) − ohranitev (K0): {sk}", lab[key], vs[1], lambda g, w, dd=dd, v=vs[0], sc=sc: float(np.sum(dd(g) * sc / v)),
            "vsota (zaklenjene: ves fond v ujemanju, neuteženo)")
    T = pd.DataFrame(rows)
    for o in (TAB, OUT_T):
        T.to_csv(o / "13_primerjalna.csv", sep=";", encoding="utf-8-sig", index=False)
    pd.set_option("display.width", 300)
    pd.set_option("display.max_colwidth", 60)
    num = T.copy()
    for c in G:
        num[c] = [(f"{x:,.1f}".replace(",", " ") if isinstance(x, (float, np.floating)) and not np.isnan(x) else ("" if isinstance(x, float) else str(x))) for x in num[c]]
    print(f"CEM: K4_1 v celicah z zaklenjeno hišo {100 * pokr:.1f} %; zaklenjenih v ujemanju {nz}")
    print(num.drop(columns=["sklop"]).to_string(index=False))
    slika9(G, st_lab)
    return T


def slika9(G, st_lab):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
    plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "font.family": "DejaVu Sans"})
    SEQ = {"BREZ": "#d9d8d3", "S1": "#c6dbef", "SS": "#9ecae1", "SF": "#6baed6", "S2": "#4292c6", "SFO": "#2171b5", "SFS": "#08519c", "S3": "#08306b"}
    en_st = {"BREZ": "S0 no measure", "S1": "S1 windows", "SS": "roof only", "SF": "façade only", "S2": "S2 roof + windows", "SFO": "façade + windows",
             "SFS": "façade + roof", "S3": "S3 comprehensive"}
    grp = [("zaklenjene, ujemanje (Qh,nd po rEI, model članka)", ("locked-in, matched\n(model position by Qh,nd)", "zaklenjene, ujemanje\n(položaj po Qh,nd v modelu)")),
           ("ovoj_prej", ("envelope before HP", "ovoj pred TČ")), ("hkrati", ("envelope with HP (±1 a)", "ovoj hkrati s TČ (±1 leto)")),
           ("ovoj_pozneje", ("envelope after HP", "ovoj po TČ"))]
    kaz = [("V2|uvoz_zima_kWh", ("winter\ngrid import", "zimski\nuvoz")), ("K0|NN kW", ("LV grid\nload", "obremenitev\nNN omrežja")),
           ("K0|CO2e t", ("CO₂e,\n30 years", "CO₂e,\n30 let"))]
    for lang, sfx in (("en", ""), ("sl", "_sl")):
        i = 0 if lang == "en" else 1
        fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [1.35, 1]})
        for y, (gk, gl) in enumerate(grp):
            g, w = G[gk]
            if g["stanje"].isna().all():                    # zaklenjene: položaj po Qh,nd na lestvici (zaokroženo na najbližje stanje)
                kk = np.minimum(g["V4|lestvica_k"].astype(int) + (g["V4|lestvica_w"] >= 0.5), 3)
                sh = {s: np.average((kk == j).astype(float), weights=w) for j, s in enumerate(["BREZ", "S1", "S2", "S3"])}
            else:
                sh = {s: np.average((g["stanje"] == s).astype(float), weights=w) for s in SEQ}
            left = 0.0
            for s, c in SEQ.items():
                v = 100 * sh.get(s, 0.0)
                a.barh(y, v, left=left, color=c, height=0.62, edgecolor="white", linewidth=1.2)
                if v >= 9:
                    a.text(left + v / 2, y, f"{v:.0f}", ha="center", va="center", fontsize=6.5, color=INK if s in ("BREZ", "S1", "SS") else "white")
                left += v
        a.set_yticks(range(len(grp)))
        a.set_yticklabels([gl[i] for _, gl in grp], fontsize=7)
        a.invert_yaxis()
        a.set_xlim(0, 100)
        a.set_xlabel("share of houses [%]" if lang == "en" else "delež hiš [%]")
        a.grid(axis="x", color=GRID, lw=0.6)
        a.set_axisbelow(True)
        a.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in SEQ.values()], labels=[(en_st if lang == "en" else st_lab)[s] for s in SEQ],
                 ncol=4, fontsize=6, loc="upper center", bbox_to_anchor=(0.45, -0.3), handlelength=1)
        a.set_title("(a) " + ("envelope state" if lang == "en" else "stanje ovoja"), loc="left", fontsize=8.5, color=INK)
        ref = G["zaklenjene, ujemanje (Qh,nd po rEI, model članka)"]
        bc = ["#9a9893", "#2a78d6", "#4a3aa7", "#eb6834"]
        for j, (gk, gl) in enumerate(grp):
            g, w = G[gk]
            vals = [100 * wq(g[c], w) / wq(ref[0][c], ref[1]) for c, _ in kaz]
            b.bar(np.arange(len(kaz)) + (j - 1.5) * 0.2, vals, width=0.18, color=bc[j], label=gl[i].replace("\n", " "))
        b.axhline(100, color=MUTED, lw=0.8, ls="--")
        b.set_ylim(0, 115)
        b.legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1, handlelength=1)
        b.set_xticks(range(len(kaz)))
        b.set_xticklabels([l_[i] for _, l_ in kaz], fontsize=7)
        b.set_ylabel("median, locked-in = 100" if lang == "en" else "mediana, zaklenjene = 100")
        b.grid(axis="y", color=GRID, lw=0.6)
        b.set_axisbelow(True)
        b.set_title("(b) " + ("observed state, per house" if lang == "en" else "opazovano stanje, na hišo"), loc="left", fontsize=8.5, color=INK)
        fig.tight_layout()
        for ext in ("png", "svg"):
            fig.savefig(HERE / "clanek1" / "slike" / f"Fig9_comparison_group{sfx}.{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--korak", default="A")
    a = ap.parse_args()
    if a.korak == "A":
        korak_a()
    elif a.korak == "S":
        korak_stanja()
    elif a.korak == "F":
        korak_fond()
    elif a.korak == "B":
        korak_b()
    elif a.korak == "T":
        korak_tabela()
    else:
        sys.exit(f"neznan korak {a.korak}")
