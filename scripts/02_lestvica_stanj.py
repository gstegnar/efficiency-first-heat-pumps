"""Korak 02 (krog 3) — lestvica stanj ovoja po obdobju namesto kalibracije fU.

S0 = izvirno stanje (U, n50 iz GreenRenoV8 izhodišč, parametri.yaml: hecm_vhodi.krog3.S0), geometrija TABULA kot doslej, brez množitelja fU.
S1–S3 = obstoječe delne / celovite prenove (fizika samo ukrepi URE, kakovost »starejšega ukrepa«: hecm_vhodi.krog3.stanja).
Ciljna prenova iz vsakega stanja (hecm_vhodi.krog3.cilj: fasada + streha + okna + tla): FSO (iz S0), FSO_S1, FSO_S2 (iz S3 = S3); GLOB kot doslej;
Celovita prenova (S3 in cilj) ima Qh,nd ≤ krog3.celovita_max_Qhnd: sicer se U prenovljenih neprozornih elementov zaostrijo s faktorjem g (mreža);
F in FS iz S0 (za slike). Prezračevanje (krog3.prezracevanje, krog3.cfg_preglasitve): 0,3 h⁻¹ + n50/20 na neto prostornino a_use × 2,5 m.
Sestava stanj iz registra: Eko sklad ∪ GURS (≥ 2003), eno-/dvostanovanjske, obdobja 1–4, brez novogradenj (../Interno/2026 analiza, samo branje).
Kontrola namesto kalibracije: monotonost in primerjava z rEI (S0 proti P75–P90 stopnje 0; S1–S3 proti medianam in P25–P75 stopenj 1–3).
Izhodi (pripona TAG, privzeto _K3, dokler Gašper ne potrdi; nato prazna): cache/02_urno_<arhetip>_<varianta>{TAG}.parquet,
tabele/02_variante{TAG}.csv, 02_lestvica_sestava{TAG}.csv, 02_lestvica_kontrola{TAG}.csv; vhodi hecm_vhodi/CLK-SI-SFH-<A>-<var>{-K3}.txt
"""
from __future__ import annotations

import importlib
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

import json  # noqa: E402

import yaml  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
_K3 = yaml.safe_load((HERE / "parametri.yaml").read_text(encoding="utf-8"))["hecm_vhodi"]["krog3"]
if os.environ.get("K3_PAR"):                                     # občutljivost (K2_5): npr. {"cilj": {"n50_max": 3.0}}
    for _k, _v in json.loads(os.environ["K3_PAR"]).items():
        _K3[_k] = {**_K3[_k], **_v} if isinstance(_v, dict) else _v
if os.environ.get("GEOM", "G13") == "G11":                             # krog 11: S0 A1/A2 po kalibraciji na izkaznice, cilj po U (brez pravila ≤ 50)
    _K3["S0"] = {**_K3["S0"], **_K3.get("S0_G11", {})}
    _K3["celovita_max_Qhnd"] = 1e9
elif os.environ.get("GEOM", "G13") == "G13":                           # krog 13 (osnova od 22. 10. 2026): toplotni ovoj iz GURS, izvirni U GreenRenoV8 brez faktorja, cilj po U
    if os.environ.get("S0_FAKTOR", "0") == "1":                         # krog 15: občutljivost »S0 previsok« (S0 A1/A2 × izkaznica / S0 iz 23d); osnova brez faktorja (dvojno štetje z B2)
        _K3["S0"] = {**_K3["S0"], **_K3.get("S0_G13", {})}
    _K3["celovita_max_Qhnd"] = 1e9
# preglasitev konfiguracije HECM tudi v delavcih (okoljska spremenljivka K2_PAR, ki jo 02 združi ob uvozu); krog 13: obstoječi K2_PAR (npr. pot HECM) se ohrani
_k2 = json.loads(os.environ["K2_PAR"]) if os.environ.get("K2_PAR") else {}
_k2.setdefault("hecm_vhodi", {})["cfg_preglasitve"] = _K3["cfg_preglasitve"]
os.environ["K2_PAR"] = json.dumps(_k2)
M = importlib.import_module("02_hecm_vhodi")                  # gradniki koraka 02 (make_bui, simulate, calibrate_glob)
HV = M.HV
K3 = HV["krog3"]
K3.update(_K3)
TAG = os.environ.get("K3_TAG", "_K3")
ARH = list(HV["arhetipi"])
OBD = {"A1": 1, "A2": 2, "A3": 3, "A4": 4}
FSO = dict(K3["cilj"])
ELEM = ("stena", "streha", "okna", "tla")
ACH = K3["prezracevanje"]["osnovna_ach_1_h"]


def nastavi_S0(var=None):
    """S0 v pomnilniku modula 02 (U, n50 GreenRenoV8; datoteka GreenRenoV8 se ne spreminja)."""
    for a in ARH:
        s0 = K3["S0"][a] if not (var and a == "A1") else K3[var]
        HV["arhetipi"][a]["U"] = dict(s0["U"])
        HV["arhetipi"][a]["n50"] = float(s0["n50"])


def stanje(s: str) -> dict:
    return {} if s == "S0" else dict(K3["stanja"][s])


def fso_iz(s: str) -> dict:
    """Ciljna prenova iz stanja s: manjkajoči ukrepi fasada/streha/okna/tla po ciljnem U; izvedeni ostanejo."""
    uk = stanje(s)
    for el in ELEM:
        if el not in uk:
            uk[el] = FSO[el]
    uk["n50_max"] = uk.get("n50_max", FSO["n50_max"]) if "okna" in stanje(s) else FSO["n50_max"]
    return uk


def vhod(a, v, uk):
    b = M.make_bui(a, v, 1.0, uk)
    b["air_change_rate_base_value"] = ACH
    return b


def variante():
    out = {"BREZ": stanje("S0"), "S1": stanje("S1"), "S2": stanje("S2"), "S3": stanje("S3"),
           "F": {"stena": FSO["stena"]}, "FS": {"stena": FSO["stena"], "streha": FSO["streha"]},
           "FSO": fso_iz("S0"), "FSO_S1": fso_iz("S1"), "FSO_S2": fso_iz("S2")}
    if os.environ.get("GEOM", "G13") == "G13":          # krog 13: umestitev po registru — dodatna stanja kroga 4 (samo fasada, samo streha, …) in cilji iz njih
        for s, uk in M.PAR["krog4"]["stanja_dodatna"].items():
            out[s] = dict(uk)
            t = dict(uk)
            for el in ELEM:
                t.setdefault(el, FSO[el])
            t["n50_max"] = uk.get("n50_max", FSO["n50_max"]) if "okna" in uk else FSO["n50_max"]
            out[f"FSO_{s}"] = t
    return out


def sestava():
    an = (HERE / M.PAR["poti"]["analiza"]).resolve()
    d = pd.read_parquet(an / "cache" / "03_stavbe_gurs_ei.parquet")
    d = d[d["tip_stavbe"].isin(["enostanovanjska", "dvostanovanjska"]) & d["obdobje"].between(1, 4) & d["leto_prvi_novogradnja"].isna()].copy()
    y = lambda c: pd.to_numeric(d[c], errors="coerce").notna()                     # noqa: E731
    g = lambda c: (pd.to_numeric(d[c], errors="coerce") >= 2003).fillna(False)     # noqa: E731
    U = pd.DataFrame({"fasada": y("leto_prvi_fasada") | g("LETO_OBN_FASADE"), "streha": y("leto_prvi_streha") | g("LETO_OBN_STREHE"),
                      "okna": y("leto_prvi_okna_vrata") | g("okna_obn_min"), "tla": y("leto_prvi_talna_izolacija") | y("leto_prvi_kletna_izolacija"),
                      "prezrac": y("leto_prvi_prezrac")})
    O = pd.DataFrame({"TČ": d["tc_ogrevalna"].astype(bool) | y("leto_prvi_tc_neznano"), "PV": y("leto_prvi_fotovoltaika"),
                      "solar": y("leto_prvi_kolektorji"), "biomasa": y("leto_prvi_biomasa")})
    n = U.sum(axis=1) + O.sum(axis=1)
    cel = y("leto_prvi_celovita_prenova")
    d["stanje"] = np.select([cel | (n >= 4), n == 3, n >= 1], ["S3", "S2", "S1"], "S0")
    d["nU"] = U.sum(axis=1)
    d["URE"] = U.apply(lambda r: "+".join(k for k in U.columns if r[k]) or "(samo OVE)", axis=1)
    rows = []
    for (o, s), x in d.groupby(["obdobje", "stanje"]):
        vc = x["URE"].value_counts(normalize=True).head(4)
        rows.append({"obdobje": int(o), "stanje": s, "stavb": len(x), "delež samo OVE [%]": round(100 * (x["nU"] == 0).mean(), 1),
                     "najpogostejše kombinacije URE": "; ".join(f"{k} {100 * v:.0f} %" for k, v in vc.items()),
                     "delež z oknami [%]": round(100 * x["URE"].str.contains("okna").mean(), 1),
                     "delež s streho [%]": round(100 * x["URE"].str.contains("streha").mean(), 1),
                     "delež s fasado [%]": round(100 * x["URE"].str.contains("fasada").mean(), 1)})
    return pd.DataFrame(rows)


def main():
    nastavi_S0()
    V = variante()
    tasks = [(f"CLK-SI-SFH-{a}-{v}{TAG.replace('_', '-')}", vhod(a, v, uk), a, v) for a in ARH for v, uk in V.items()]
    # občutljivost A1 (povprečje 1896–1920 in 1921–1945)
    nastavi_S0("S0_A1_obcutljivost")
    tasks.append((f"CLK-SI-SFH-A1-BREZobc{TAG.replace('_', '-')}", vhod("A1", "BREZobc", {}), "A1", "BREZ_obc_A1"))
    nastavi_S0()
    with Pool(HV["delavcev"]) as pool:
        res = pool.map(M.simulate, [(n, b) for n, b, _, _ in tasks])
        glogs = pool.map(M.calibrate_glob, [(a, 1.0) for a in ARH])
    gkal = pd.DataFrame([row for lg in glogs for row in lg])
    tol = HV["glob_kalibracija"]["toleranca_kWh_m2a"]
    gG = {}
    for a, g in gkal.groupby("arhetip"):
        ok = g[g["Qh_nd"] <= g["cilj_max"] + tol]
        gG[a] = float(ok["g"].max()) if len(ok) else float(g.loc[g["Qh_nd"].idxmin(), "g"])
    # celovita prenova: Qh,nd ≤ meja — zaostri U prenovljenih neprozornih elementov s faktorjem g (najvišji g iz mreže, ki doseže mejo)
    QMAX, MREZA = K3["celovita_max_Qhnd"], K3["celovita_g_mreza"]
    gC = {}
    pre = [(i, t) for i, t in enumerate(tasks) if t[3] in ("S3", "FSO", "FSO_S1", "FSO_S2") and res[i][0]["Qh_nd"] > QMAX]
    if pre:
        mt = []
        for i, (name, b, a, v) in pre:
            uk = V[v]
            for g in MREZA[1:]:
                u2 = {k: (x * g if k in ("stena", "streha", "tla") else x) for k, x in uk.items()}
                mt.append((i, g, (f"{name}-g{int(round(100 * g))}", vhod(a, v, u2)), u2))
        with Pool(HV["delavcev"]) as pool:
            mres = pool.map(M.simulate, [x[2] for x in mt])
        for i, (name, b, a, v) in pre:
            cand = [(g, r, x, u2) for (j, g, x, u2), r in zip(mt, mres) if j == i]
            ok = [c for c in cand if c[1][0]["Qh_nd"] <= QMAX]
            g, r, (n2, b2), u2 = max(ok, key=lambda c: c[0]) if ok else min(cand, key=lambda c: c[1][0]["Qh_nd"])
            res[i] = r
            tasks[i] = (n2, b2, a, v)
            gC[(a, v)] = g
        print("celovita prenova, zaostritev g:", gC)
    gtasks = [(f"CLK-SI-SFH-{a}-GLOB{TAG.replace('_', '-')}", vhod(a, "GLOB", M.glob_ukrepi(gG[a])), a, "GLOB") for a in ARH]
    with Pool(len(ARH)) as pool:
        res += pool.map(M.simulate, [(n, b) for n, b, _, _ in gtasks])
    tasks += gtasks
    rows = []
    for (r, hourly), (name, bui, a, v) in zip(res, tasks):
        h = hourly[["T_ext", "T_op", "Q_H", "Q_C", "Q_W", "G_h"]].copy()
        h.attrs["a_use_m2"] = r["a_use"]
        if v != "BREZ_obc_A1":
            h.to_parquet(M.CACHE / f"02_urno_{a}_{v}{TAG}.parquet")
        u = bui["transmittance_U_elements"]
        rows.append({"arhetip": a, "varianta": v, "f_U": 1.0, "g_celovita": gC.get((a, v), 1.0), "U_stena": round(u[1], 3), "U_streha": round(u[9], 3), "U_tla": round(u[0], 3),
                     "U_okna": round(u[5], 2), "n50": bui["airtightness_n50"], "rekuperacija": bui["heat_recovery_efficiency"],
                     **{k: (round(x, 2) if isinstance(x, float) else x) for k, x in r.items() if k != "ime"}})
    var = pd.DataFrame(rows)
    var.to_csv(M.TAB / f"02_variante{TAG}.csv", sep=";", encoding="utf-8-sig", index=False)

    # ---- kontrola: monotonost in primerjava z rEI
    sest = sestava()
    sest.to_csv(M.TAB / f"02_lestvica_sestava{TAG}.csv", sep=";", encoding="utf-8-sig", index=False)
    r = pd.read_parquet(M.CACHE / "01_rEI_vzorec.parquet")
    r = r[r["tipg"] == "eno_dvo"]
    q = r.groupby(["obdobje", "st"])["Qh_nd_EI"].describe(percentiles=[.25, .5, .75, .9])
    K = []
    for a in ARH:
        o = OBD[a]
        for s, v, st in (("S0", "BREZ", 0), ("S1", "S1", 1), ("S2", "S2", 2), ("S3", "S3", 3)):
            m = var[(var["arhetip"] == a) & (var["varianta"] == v)].iloc[0]
            fz = var[(var["arhetip"] == a) & (var["varianta"] == ("FSO" if s == "S0" else (f"FSO_{s}" if s != "S3" else "S3")))].iloc[0]
            qq = q.loc[(o, st)]
            ref = (qq["75%"], qq["90%"]) if s == "S0" else (qq["50%"],)
            K.append({"arhetip": a, "obdobje": o, "stanje": s, "U stena/streha/tla/okna": f"{m['U_stena']} / {m['U_streha']} / {m['U_tla']} / {m['U_okna']}",
                      "n50": m["n50"], "Qh,nd model": m["Qh_nd"], "P99,9 [W/m²]": m["QH_P999_W_m2"], "Qh,nd FSO iz stanja": fz["Qh_nd"],
                      "rEI n": int(qq["count"]), "rEI P25": qq["25%"], "rEI P50": qq["50%"], "rEI P75": qq["75%"], "rEI P90": qq["90%"],
                      "primerjava": "S0 proti P75–P90" if s == "S0" else f"S{st} proti P50 (P25–P75)",
                      "odstopanje od P50 / P75 [kWh/m²a]": round(m["Qh_nd"] - (qq["75%"] if s == "S0" else qq["50%"]), 1),
                      "v pasu": bool(qq["75%"] <= m["Qh_nd"] <= qq["90%"]) if s == "S0" else bool(qq["25%"] <= m["Qh_nd"] <= qq["75%"])})
    K = pd.DataFrame(K)
    K.round(1).to_csv(M.TAB / f"02_lestvica_kontrola{TAG}.csv", sep=";", encoding="utf-8-sig", index=False)
    s0 = K[K["stanje"] == "S0"].set_index("arhetip")["Qh,nd model"]
    mono_obd = bool(s0["A1"] >= s0["A2"] >= s0["A3"] > s0["A4"])
    mono_st = {a: bool(np.all(np.diff(K[K["arhetip"] == a]["Qh,nd model"].to_numpy()) < 0)) for a in ARH}
    pd.set_option("display.width", 260)
    pd.set_option("display.max_colwidth", 90)
    print(sest.to_string(index=False))
    print()
    print(var[["arhetip", "varianta", "U_stena", "U_streha", "U_tla", "U_okna", "n50", "rekuperacija", "Qh_nd", "QH_P999_W_m2"]].to_string(index=False))
    print()
    print(K.drop(columns=["primerjava"]).round(1).to_string(index=False))
    print(f"\nmonotonost S0 po obdobjih A1 ≥ A2 ≥ A3 > A4: {mono_obd} ({s0.round(1).to_dict()}); S0 > S1 > S2 > S3: {mono_st}; GLOB g: {gG}")
    # položaj hiš (Qh,nd kot v krogu 2) na lestvici obdobja
    v_ = var.set_index(["arhetip", "varianta"])["Qh_nd"]
    f1 = pd.read_parquet(M.CACHE / "01_fond.parquet", columns=["ID_Stavbe", "arhetip", "paket_osnovni"])
    qh = pd.read_parquet(M.CACHE / "03_fond_letno.parquet", columns=["ID_Stavbe", "qh"]).drop_duplicates("ID_Stavbe")
    h = f1.merge(qh, on="ID_Stavbe")
    lab = ["nad S0", "S0–S1", "S1–S2", "S2–S3", "pod S3"]
    h["polozaj"] = [lab[int(np.searchsorted(-np.array([v_[(a, x)] for x in ("BREZ", "S1", "S2", "S3")]), -q))] for a, q in zip(h["arhetip"], h["qh"])]
    h["ob_FSO"] = [q <= v_[(a, "FSO")] + 5 for a, q in zip(h["arhetip"], h["qh"])]
    rows = []
    for vz, m in (("paket", h["paket_osnovni"].astype(bool)), ("fond", pd.Series(True, index=h.index))):
        for a, g in h[m].groupby("arhetip"):
            rows.append({"vzorec": vz, "arhetip": a, "stavb": len(g), **{l_: round(100 * (g["polozaj"] == l_).mean(), 1) for l_ in lab},
                         "≤ FSO + 5 [%]": round(100 * g["ob_FSO"].mean(), 1)})
        g = h[m]
        rows.append({"vzorec": vz, "arhetip": "skupaj", "stavb": len(g), **{l_: round(100 * (g["polozaj"] == l_).mean(), 1) for l_ in lab},
                     "≤ FSO + 5 [%]": round(100 * g["ob_FSO"].mean(), 1)})
    Pz = pd.DataFrame(rows)
    Pz.to_csv(M.TAB / f"02_lestvica_polozaj{TAG}.csv", sep=";", encoding="utf-8-sig", index=False)
    print(Pz.to_string(index=False))
    obc = var[var["varianta"] == "BREZ_obc_A1"]["Qh_nd"].iloc[0]
    print(f"občutljivost A1 S0 (povprečje 1896–1920 in 1921–1945): Qh,nd {obc:.1f} (osnovno {s0['A1']:.1f})")


if __name__ == "__main__":
    main()
