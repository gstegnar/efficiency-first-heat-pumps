"""Krog 10, §5 — strošek ovoja iz dejanskih projektov Eko sklada proti modelu (NAVODILA_CLAUDE_CODE_v10.md).

a) eko_eur_m2.xlsx, list Nalozbe (SFH, V statistiki = DA, 2023–2025; starejših let s stroški v datoteki ni): stavbe (KO-stavba), ki so izvedle
   fasado + streho/strop + okna ali celovito prenovo; vsota priznanih stroškov, na m² ogrevane površine, po obdobju gradnje (Leto izgradnje, GURS).
   Dodatno po elementu (vse stavbe z ukrepom): dejanski strošek elementa proti modelskemu.
b) modelski strošek istih stavb: (i) na element = EUR/enoto (Eko mediane, parametri ekonomika.ovoj_EUR_enota) × površina elementa TABULA (geom × A_ogr);
   (ii) na m² ogrevane = A_ogr × ovoj_EUR_m2_ogrevane. Razmerje dejansko / model (mediana, IQR) po obdobju.
   Stroški so nominalni 2023–2025 kot enotni stroški modela (iste naložbe) — primerjava je konsistentna brez deflacije.
c) paket 5.969 (L3 / _K9c10): P3 − P1 posameznik R1, R3, družba, proračun R3 s stroškom ovoja = model (i) × razmerje (mediana, P25, P75) po obdobju.
Izhodi: clanek1/tabele/20a_strosek_ovoja_dejanski.csv, 20b_obcutljivost_strosek_ovoja.csv (+ tabele/). Vhod: K10_NALOZBE (feather kopija lista Nalozbe).
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
PAR = yaml.safe_load((HERE / "parametri.yaml").read_text(encoding="utf-8"))
TAB, OUT_T = HERE / "tabele", HERE / "clanek1" / "tabele"
EK, HV = PAR["ekonomika"], PAR["hecm_vhodi"]
EL = {"fasada": "stena", "streha": "streha", "okna_vrata": "okna", "talna_izolacija": "tla", "kletna_izolacija": "tla"}
COST, AREA = "Priznani stroški [EUR]", "Ogrevana uporabna površina [m²]"


GEO = {"G11": HV.get("geometrija_G11"), "G13": HV.get("geometrija_G13_strosek")}.get(os.environ.get("GEOM", "G13"))   # krog 11/13: stroškovna geometrija kot v 04


def geom(a):
    g = HV["geometrija"][GEO[a] if GEO else HV["arhetipi"][a]["geometrija"]]
    return {k: g[k] / g["A"] for k in ("stena", "streha", "okna", "tla")}


def arhetip(leto):
    return np.select([leto <= 1945, leto <= 1970, leto <= 1980, leto <= 2002], ["A1", "A2", "A3", "A4"], "")


def model(a, A, elementi):
    g = geom(a)
    i = sum(g[e] * A * EK["ovoj_EUR_enota"][e] for e in elementi)
    ii = sum(A * EK["ovoj_EUR_m2_ogrevane"][e] for e in elementi)
    return i, ii


def korak_a():
    n = pd.read_feather(os.environ["K10_NALOZBE"])
    n = n[(n["Tip"] == "SFH") & (n["V statistiki"] == "DA")].copy()
    n["stavba"] = n["KO-stavba|del"].str.split("|").str[0]
    n["leto_izgr"] = pd.to_numeric(n["Leto izgradnje"], errors="coerce")
    n["A"] = pd.to_numeric(n[AREA], errors="coerce")
    ov = n[n["Ukrep (koda)"].isin(list(EL) + ["celovita_prenova"])]
    rows = []
    # celoviti paketi
    for st, g in ov.groupby("stavba"):
        u = set(g["Ukrep (koda)"])
        if not ({"fasada", "streha", "okna_vrata"} <= u or "celovita_prenova" in u):
            continue
        A, ly = g["A"].median(), g["leto_izgr"].median()
        a = str(arhetip(np.array([ly]))[0]) if np.isfinite(ly) else ""
        if not a or not np.isfinite(A):
            continue
        el = sorted({EL[x] for x in u if x in EL}) if "celovita_prenova" not in u else ["stena", "streha", "okna", "tla"]
        mi, mii = model(a, A, el)
        rows.append({"vzorec": "celovit paket (fasada + streha + okna ali celovita prenova)", "stavba": st, "arhetip": a, "leto izgradnje": ly, "A_ogr": A,
                     "elementi": "+".join(el), "dejansko [EUR]": g[COST].sum(), "model element [EUR]": mi, "model m² ogrevane [EUR]": mii, "leta": "+".join(sorted(g["Leto"].astype(str).unique()))})
    # po elementu (vse naložbe z ukrepom)
    for kod, e in (("fasada", "stena"), ("streha", "streha"), ("okna_vrata", "okna")):
        s = ov[(ov["Ukrep (koda)"] == kod) & ov["A"].notna() & ov["leto_izgr"].notna()]
        s = s.assign(arhetip=arhetip(s["leto_izgr"].to_numpy()))
        s = s[s["arhetip"] != ""]
        for r in s.itertuples():
            mi, mii = model(r.arhetip, r.A, [e])
            rows.append({"vzorec": f"element: {kod}", "stavba": r.stavba, "arhetip": r.arhetip, "leto izgradnje": r.leto_izgr, "A_ogr": r.A, "elementi": e,
                         "dejansko [EUR]": getattr(r, "_6"), "model element [EUR]": mi, "model m² ogrevane [EUR]": mii, "leta": str(r.Leto)})
    d = pd.DataFrame(rows)
    d["razmerje dejansko / model element"] = d["dejansko [EUR]"] / d["model element [EUR]"]
    d["razmerje dejansko / model m² ogrevane"] = d["dejansko [EUR]"] / d["model m² ogrevane [EUR]"]
    d["obdobje"] = np.where(d["arhetip"] == "A4", "1981–2002", "pred 1981")
    # odreži skrajnosti (1./99. percentil razmerja) po vzorcu
    keep = d.groupby("vzorec")["razmerje dejansko / model element"].transform(lambda x: x.between(x.quantile(0.01), x.quantile(0.99)))
    d = d[keep]
    T = []
    for (vz, ob), g in list(d.groupby(["vzorec", "obdobje"])) + [((vz, "vse"), g) for vz, g in d.groupby("vzorec")]:
        T.append({"vzorec": vz, "obdobje": ob, "stavb": len(g), "A_ogr mediana [m²]": g["A_ogr"].median(),
                  "dejansko mediana [EUR]": g["dejansko [EUR]"].median(), "dejansko na m² ogrevane mediana [EUR/m²]": (g["dejansko [EUR]"] / g["A_ogr"]).median(),
                  **{f"{lab}: {q}": g[c].quantile(p) for lab, c in (("dejansko / model element", "razmerje dejansko / model element"),
                                                                   ("dejansko / model m² ogrevane", "razmerje dejansko / model m² ogrevane"))
                     for q, p in (("P25", 0.25), ("mediana", 0.5), ("P75", 0.75))}})
    T = pd.DataFrame(T)
    T["bližja osnova"] = np.where((np.log(T["dejansko / model element: mediana"])).abs() < np.log(T["dejansko / model m² ogrevane: mediana"]).abs(), "element", "m² ogrevane")
    for o in (TAB, OUT_T):
        T.round(3).to_csv(o / "20a_strosek_ovoja_dejanski.csv", sep=";", encoding="utf-8-sig", index=False)
    d.to_csv(TAB / "20a_strosek_ovoja_stavbe.csv", sep=";", encoding="utf-8-sig", index=False)
    pd.set_option("display.width", 250)
    print(T.round(2).to_string(index=False))
    return T


def korak_c(T):
    s = importlib.util.spec_from_file_location("k21", HERE / "K2_1_drzava.py")
    K1 = importlib.util.module_from_spec(s)
    s.loader.exec_module(K1)
    E = K1.E
    w = E.load_units("fond", "_K9c10")
    pk = T[T["vzorec"].str.startswith("celovit")].set_index("obdobje")
    obd = {"pred 1981": ["A1", "A2", "A3"], "1981–2002": ["A4"]}
    scen = [("osnova (model × 1)", {})]
    for q in ("P25", "mediana", "P75"):
        f = {a: float(pk.loc[o, f"dejansko / model element: {q}"]) for o, ars in obd.items() for a in ars}
        scen.append((f"dejansko: model × razmerje {q} ({f['A1']:.2f} / {f['A4']:.2f})", f))
    scen.append(("osnova m² ogrevane (obstoječa občutljivost)", "ogrevana"))
    rows = []
    for lab, f in scen:
        basis = "ogrevana" if f == "ogrevana" else EK["ovoj_osnova"]
        E.EK["ovoj_strosek_faktor"] = f if isinstance(f, dict) else 1.0
        K1.EK = E.EK
        old = E.EK["ovoj_osnova"]
        E.EK["ovoj_osnova"] = basis
        n = {rez: E.npv_paths(w, rez, EK["diskont_zasebni"], E.pf_factory("konstantne"), basis) for rez in ("R1", "R3")}
        rc = K1.racun(w)
        E.EK["ovoj_osnova"] = old
        for o, ars in [("paket", ["A1", "A2", "A3", "A4"])] + list(obd.items()):
            m = w["arhetip"].isin(ars).to_numpy()
            row = {"scenarij stroška ovoja": lab, "obdobje": o, "hiš": int(m.sum())}
            for k, x1, x3 in (("posameznik R1", n["R1"]["P1"], n["R1"]["P3"]), ("posameznik R3", n["R3"]["P1"], n["R3"]["P3"]),
                              ("družbeni NPV", rc["P1"]["družbeni NPV"], rc["P3"]["družbeni NPV"]), ("proračun R3", rc["P1"]["R3 proračun neto"], rc["P3"]["R3 proračun neto"])):
                d = (np.asarray(x3) - np.asarray(x1))[m]
                row[f"{k}: P3 − P1 mediana [kEUR]"] = np.median(d) / 1e3
                row[f"{k}: delež P3 boljši [%]"] = 100 * (d < 0).mean()
                row[f"{k}: vsota [mio EUR]"] = d.sum() / 1e6
            rows.append(row)
    E.EK["ovoj_strosek_faktor"] = 1.0
    R = pd.DataFrame(rows)
    for o in (TAB, OUT_T):
        R.round(3).to_csv(o / "20b_obcutljivost_strosek_ovoja.csv", sep=";", encoding="utf-8-sig", index=False)
    kol = ["scenarij stroška ovoja", "obdobje"] + [c for c in R.columns if "mediana" in c or "delež" in c]
    print(R[kol].round(2).to_string(index=False))


if __name__ == "__main__":
    korak_c(korak_a())
