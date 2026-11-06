"""Krog 14 — ali je pravilo za politiko robustno na umestitev hiš (register proti B2)? (NAVODILA_CLAUDE_CODE_v14.md)

Brez nove verige: obstoječa teka G13 za paket 8.134 hiš — register (osnova, tag _L3 iz K8) in B2 (umestitev po Qh,nd, tag _B2); iste funkcije kot 22f.
Potrebna toplota ob vgradnji = Qh,nd hiše v P1 (V2|Qh_nd: lastna izkaznica ali vzorec iz izkaznic obdobja, na kondicionirano površino; enaka v obeh tekih).
  1  P3 − P1 po razredu Qh,nd (> 140 / 90–140 / < 90) × obdobje × umestitev → tabele/23a_po_razredu_umestitev.csv
  2  empirična prelomna točka: mediana in delež P3 boljši po razredih 20 kWh/m²a; Qh,nd, pri katerem mediana prečka 0 in delež 50 %
     → tabele/23b_prelomna_umestitev.csv, clanek1/slike/Fig23_break_even_umestitev(.png, _sl)
  3  ekonomika ovoja pod registrom: bruto strošek P3 ovoja, spodbuda, prihranek elektrike in računa R3 → tabele/23c_ovoj_ekonomika_register.csv
  4  hiše fonda z lastno izkaznico pred TČ: izkaznica / S0 obdobja (G13) → tabele/23d_izkaznica_proti_S0.csv (izkaznice v fondu so ≥ 90 kWh/m²a)
  5  kandidatna pravila po pragu Qh,nd (oba načina umestitve) → tabele/23e_pravilo_prag.csv
Zagon: C:/Anaconda/envs/new_env/python.exe K14_umestitev.py
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
TAB, OUT_T, OUT_S, CACHE = HERE / "tabele", HERE / "clanek1" / "tabele", HERE / "clanek1" / "slike", HERE / "cache"
sp = importlib.util.spec_from_file_location("k21", HERE / "K2_1_drzava.py")
K1 = importlib.util.module_from_spec(sp)
sp.loader.exec_module(K1)
E, OKM, PAR = K1.E, K1.DR.OKM, K1.PAR
EK = PAR["ekonomika"]
UMES = {"register": "_L3", "B2": "_B2"}
OBD = {"pred 1981": ["A1", "A2", "A3"], "1981–2002": ["A4"]}
RAZ = [("> 140", 140, 1e9), ("90–140", 90, 140), ("< 90", 0, 90)]
# kazalnik: (enota mediane, delitelj mediane, enota vsote, delitelj vsote)
KAZ = {"posameznik R1": ("tisoč €", 1e3, "mio €", 1e6), "posameznik R3": ("tisoč €", 1e3, "mio €", 1e6), "družba": ("tisoč €", 1e3, "mio €", 1e6),
       "proračun R3": ("tisoč €", 1e3, "mio €", 1e6), "CO2e 30 let": ("t", 1, "kt", 1e3), "zimski uvoz (leto 20)": ("kWh/a", 1, "GWh/a", 1e6),
       "obremenitev NN": ("kW", 1, "MW", 1e3)}


def izracun(tag):
    w = E.load_units("fond", tag)
    n = {rez: E.npv_paths(w, rez, EK["diskont_zasebni"], E.pf_factory("konstantne"), EK["ovoj_osnova"]) for rez in ("R1", "R3")}
    rc = K1.racun(w)
    co2 = {p: (sum(OKM.op_co2(w, E.leto_varianta(p, t), t) for t in range(1, E.N + 1)) + sum(OKM.embodied(w, it, ar) for (y, it, ar) in E.POTI[p]["ev"]))
           for p in ("P1", "P3")}                                                       # kg
    d = pd.DataFrame(index=w.index)
    d["arhetip"], d["Qh"] = w["arhetip"], w["V2|Qh_nd"].astype(float)
    d["posameznik R1"] = n["R1"]["P3"] - n["R1"]["P1"]
    d["posameznik R3"] = n["R3"]["P3"] - n["R3"]["P1"]
    d["družba"] = rc["P3"]["družbeni NPV"].to_numpy() - rc["P1"]["družbeni NPV"].to_numpy()
    d["proračun R3"] = rc["P3"]["R3 proračun neto"].to_numpy() - rc["P1"]["R3 proračun neto"].to_numpy()
    d["CO2e 30 let"] = (np.asarray(co2["P3"]) - np.asarray(co2["P1"])) / 1e3
    d["zimski uvoz (leto 20)"] = w["V5|uvoz_zima_kWh"].astype(float) - w["V2|uvoz_zima_kWh"].astype(float)
    d["obremenitev NN"] = rc["P3"]["obremenitev NN omrežja [kW]"].to_numpy() - rc["P1"]["obremenitev NN omrežja [kW]"].to_numpy()
    d["obdobje"] = np.where(d["arhetip"].isin(OBD["pred 1981"]), "pred 1981", "1981–2002")
    return w, d


def razred(q):
    return np.select([q > 140, q >= 90], ["> 140", "90–140"], "< 90")


def k23a(D):
    rows = []
    for um, d in D.items():
        d = d.assign(razred=razred(d["Qh"]))
        for obd in ("pred 1981", "1981–2002", "paket"):
            g0 = d if obd == "paket" else d[d["obdobje"] == obd]
            for rz in [r[0] for r in RAZ] + ["vse"]:
                g = g0 if rz == "vse" else g0[g0["razred"] == rz]
                for k, (eu, dv, es, ds) in KAZ.items():
                    x = g[k]
                    rows.append({"umestitev": um, "obdobje": obd, "razred Qh,nd [kWh/m²a]": rz, "hiš": len(g), "Qh,nd mediana": g["Qh"].median(),
                                 "kazalnik": k, "enota": eu, "P3 − P1 mediana": x.median() / dv, "P25": x.quantile(.25) / dv, "P75": x.quantile(.75) / dv,
                                 "delež P3 boljši [%]": 100 * (x < 0).mean(), "vsota": x.sum() / ds, "enota vsote": es})
    return pd.DataFrame(rows)


def k23b(D):
    rows, pre = [], []
    edges = np.arange(40, 321, 20)
    for um, d in D.items():
        for obd in ("pred 1981", "1981–2002", "paket"):
            g0 = d if obd == "paket" else d[d["obdobje"] == obd]
            b = pd.cut(g0["Qh"], edges)
            for k in ("posameznik R3", "družba", "CO2e 30 let"):
                eu, dv = KAZ[k][:2]
                s = g0.groupby(b, observed=True)[k].agg(["size", "median", lambda x: 100 * (x < 0).mean()])
                s.columns = ["hiš", "med", "del"]
                s = s[s["hiš"] >= 30]
                mid = np.array([iv.mid for iv in s.index])
                for m_, (_, r_) in zip(mid, s.iterrows()):
                    rows.append({"umestitev": um, "obdobje": obd, "kazalnik": k, "Qh,nd razred sredina": m_, "hiš": int(r_["hiš"]),
                                 "P3 − P1 mediana": r_["med"] / dv, "enota": eu, "delež P3 boljši [%]": r_["del"]})
                pre.append({"umestitev": um, "obdobje": obd, "kazalnik": k, "prelom mediana = 0 [kWh/m²a]": prelom(mid, s["med"].to_numpy()),
                            "prelom delež = 50 % [kWh/m²a]": prelom(mid, 50 - s["del"].to_numpy())})
    return pd.DataFrame(rows), pd.DataFrame(pre)


def prelom(x, y):
    """Najvišji Qh,nd, pri katerem y preide iz > 0 (P3 slabši) v ≤ 0 (P3 boljši) ob naraščajočem Qh,nd; linearna interpolacija.
    NaN = ni prehoda v podprtem razponu (‹ = povsod boljši, › = povsod slabši)."""
    if len(y) == 0:
        return np.nan
    if (y <= 0).all():
        return -np.inf
    if (y > 0).all():
        return np.inf
    i = np.where((y[:-1] > 0) & (y[1:] <= 0))[0]
    if len(i) == 0:
        return np.nan
    i = i[-1]
    return float(x[i] + (x[i + 1] - x[i]) * y[i] / (y[i] - y[i + 1]))


def k23c(w, d):
    inv, sub, _ = E.envelope_cost(w, "FSO", EK["ovoj_osnova"])
    b2 = E.annual_bill(w, "V2", "R3", 1.0)
    b5 = E.annual_bill(w, "V5", "R3", 1.0)
    r = EK["diskont_zasebni"]
    af = sum((1 + r) ** -t for t in range(1, E.N + 1))
    x = pd.DataFrame({"obdobje": d["obdobje"], "Qh": d["Qh"], "bruto": inv, "spodbuda": sub, "neto": inv - sub,
                      "el_prihranek_MWh": (w["V2|el_skupaj_kWh"].astype(float) - w["V5|el_skupaj_kWh"].astype(float)) / 1e3,
                      "uvoz_prihranek_MWh": (w["V2|uvoz_kWh"].astype(float) - w["V5|uvoz_kWh"].astype(float)) / 1e3,
                      "racun_prihranek": np.asarray(b2) - np.asarray(b5), "TC_kW_P1": w["V1|TC_kW"].astype(float), "TC_kW_P3": w["V4|TC_kW"].astype(float),
                      "PV_P1": w["V2|PV_kWp"].astype(float), "PV_P3": w["V5|PV_kWp"].astype(float), "R3": d["posameznik R3"]}, index=w.index)
    x["SV prihranka računa 30 let"] = x["racun_prihranek"] * af
    rows = []
    edges = [0, 70, 90, 110, 140, 180, 1e9]
    lab = ["< 70", "70–90", "90–110", "110–140", "140–180", "> 180"]
    for obd in ("pred 1981", "1981–2002", "paket"):
        g0 = x if obd == "paket" else x[x["obdobje"] == obd]
        for rz, g in g0.groupby(pd.cut(g0["Qh"], edges, labels=lab), observed=True):
            if len(g) < 20:
                continue
            rows.append({"obdobje": obd, "razred Qh,nd [kWh/m²a]": rz, "hiš": len(g), "Qh,nd mediana": g["Qh"].median(),
                         "bruto strošek ovoja P3 [tisoč €]": g["bruto"].median() / 1e3, "spodbuda ovoja [tisoč €]": g["spodbuda"].median() / 1e3,
                         "neto strošek ovoja [tisoč €]": g["neto"].median() / 1e3,
                         "prihranek elektrike [MWh/a]": g["el_prihranek_MWh"].median(), "prihranek uvoza [MWh/a]": g["uvoz_prihranek_MWh"].median(),
                         "prihranek računa R3 [€/a]": g["racun_prihranek"].median(),
                         "SV prihranka računa R3, 30 let [tisoč €]": g["SV prihranka računa 30 let"].median() / 1e3,
                         "neto ovoj / SV prihranka računa": (g["neto"] / g["SV prihranka računa 30 let"]).median(),
                         "TČ P1 / P3 [kW]": f"{g['TC_kW_P1'].median():.1f} / {g['TC_kW_P3'].median():.1f}",
                         "PV P1 / P3 [kWp]": f"{g['PV_P1'].median():.1f} / {g['PV_P3'].median():.1f}",
                         "P3 − P1 R3 mediana [tisoč €]": g["R3"].median() / 1e3, "delež P3 boljši R3 [%]": 100 * (g["R3"] < 0).mean()})
    return pd.DataFrame(rows)


def k23d():
    f = pd.read_parquet(CACHE / "01_fond.parquet", columns=["ID_Stavbe", "arhetip", "Qh_nd_ob_vgradnji_vir", "Qhnd_lastna_EI", "paket_osnovni"])
    f = f[(f["Qh_nd_ob_vgradnji_vir"] == "EI") & f["Qhnd_lastna_EI"].notna()]
    v = pd.read_csv(TAB / "02_variante.csv", sep=";", encoding="utf-8-sig")
    s0 = v[v["varianta"] == "BREZ"].set_index("arhetip")["Qh_nd"]
    s3 = v[v["varianta"] == "S3"].set_index("arhetip")["Qh_nd"]
    f["S0"], f["S3"] = f["arhetip"].map(s0), f["arhetip"].map(s3)
    f["razmerje"] = f["Qhnd_lastna_EI"] / f["S0"]
    f["obdobje"] = np.where(f["arhetip"].isin(OBD["pred 1981"]), "pred 1981", "1981–2002")
    rows = []
    for key, g in list(f.groupby("arhetip")) + list(f.groupby("obdobje")) + [("fond", f)]:
        q = g["razmerje"].quantile([.1, .25, .5, .75, .9])
        rows.append({"skupina": key, "hiš": len(g), "od tega paket": int(g["paket_osnovni"].astype(float).fillna(0).astype(bool).sum()),
                     "S0 G13 [kWh/m²a]": g["S0"].iloc[0] if g["arhetip"].nunique() == 1 else np.nan,
                     "izkaznica P50 [kWh/m²a]": g["Qhnd_lastna_EI"].median(), **{f"izkaznica / S0 P{int(p * 100)}": q[p] for p in q.index},
                     "delež < 0,5 × S0 [%]": 100 * (g["razmerje"] < 0.5).mean(), "izkaznica min [kWh/m²a]": g["Qhnd_lastna_EI"].min(),
                     "delež ≥ S0 [%]": 100 * (g["razmerje"] >= 1).mean()})
    return pd.DataFrame(rows)


def k23e(D):
    """Kandidatna pravila: URE najprej (pogoj za spodbudo TČ) le pri Qh,nd ob vgradnji > prag; P3 − P1 za hiše nad pragom in vsota (pravilo proti P1 za vse)."""
    rows = []
    for prag in (0, 120, 140, 160, 180, 200, 220):
        for um, d in D.items():
            for obd in ("pred 1981", "1981–2002", "paket"):
                g0 = d if obd == "paket" else d[d["obdobje"] == obd]
                g = g0[g0["Qh"] > prag]
                r = {"prag Qh,nd [kWh/m²a]": prag, "umestitev": um, "obdobje": obd, "hiš nad pragom": len(g), "delež hiš [%]": 100 * len(g) / len(g0)}
                for k in ("posameznik R3", "družba", "proračun R3", "CO2e 30 let", "zimski uvoz (leto 20)"):
                    eu, dv, es, ds = KAZ[k]
                    r[f"{k} mediana [{eu}]"] = g[k].median() / dv
                    r[f"{k} delež boljši [%]"] = 100 * (g[k] < 0).mean()
                    r[f"{k} vsota [{es}]"] = g[k].sum() / ds
                rows.append(r)
    return pd.DataFrame(rows)


def slika(B):
    C = {"register": "#c0392b", "B2": "#2a78d6"}
    LS = {"pred 1981": "-", "1981–2002": "--"}
    for lang in ("en", "sl"):
        fig, axs = plt.subplots(1, 3, figsize=(10, 3.4))
        for ax, (k, yl) in zip(axs, (("posameznik R3", ("Household R3, P3 − P1 [€ thousand]", "Posameznik R3, P3 − P1 [tisoč €]")),
                                     ("družba", ("Social CBA, P3 − P1 [€ thousand]", "Družbeni NPV, P3 − P1 [tisoč €]")),
                                     ("CO2e 30 let", ("CO$_2$e 30 years, P3 − P1 [t]", "CO$_2$e 30 let, P3 − P1 [t]")))):
            for um in ("register", "B2"):
                for obd in ("pred 1981", "1981–2002"):
                    g = B[(B["umestitev"] == um) & (B["obdobje"] == obd) & (B["kazalnik"] == k)]
                    lu = {"register": ("register", "register"), "B2": ("B2 ladder", "B2 lestvica")}[um][lang == "sl"]
                    lo = {"pred 1981": ("pre-1981", "pred 1981"), "1981–2002": ("1981–2002", "1981–2002")}[obd][lang == "sl"]
                    ax.plot(g["Qh,nd razred sredina"], g["P3 − P1 mediana"], color=C[um], ls=LS[obd], marker="o", ms=3, label=f"{lu}, {lo}")
            ax.axhline(0, color="#52514e", lw=0.8)
            ax.set_xlabel("Heat need at HP installation [kWh/m²a]" if lang == "en" else "Potrebna toplota ob vgradnji TČ [kWh/m²a]", fontsize=8)
            ax.set_ylabel(yl[lang == "sl"], fontsize=8)
            ax.grid(color="#e6e5e0", lw=0.6)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
        axs[0].legend(fontsize=7, frameon=False)
        fig.text(0.01, 0.01, ("Median per 20 kWh/m²a bin (≥ 30 houses); below 0 = efficiency first better. HP + PV package, 8,134 houses, G13." if lang == "en" else
                              "Mediana po razredih 20 kWh/m²a (≥ 30 hiš); pod 0 = URE najprej boljši. Paket TČ + PV, 8.134 hiš, G13."), fontsize=7, color="#52514e")
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        for ext in ("png", "svg"):
            fig.savefig(OUT_S / f"Fig23_break_even_umestitev{'' if lang == 'en' else '_sl'}.{ext}", dpi=300)
        plt.close(fig)


def zapis(T, ime):
    for o in (TAB, OUT_T):
        T.round(3).to_csv(o / ime, sep=";", encoding="utf-8-sig", index=False)


def main():
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    W, D = {}, {}
    for um, tag in UMES.items():
        W[um], D[um] = izracun(tag)
        print(um, len(D[um]), "hiš")
    A = k23a(D)
    zapis(A, "23a_po_razredu_umestitev.csv")
    B, P = k23b(D)
    zapis(B, "23b_prelomna_umestitev.csv")
    zapis(P, "23b_prelomna_umestitev_tocke.csv")
    slika(B)
    Cc = k23c(W["register"], D["register"])
    zapis(Cc, "23c_ovoj_ekonomika_register.csv")
    Ee = k23e(D)
    zapis(Ee, "23e_pravilo_prag.csv")
    print(Ee[["prag Qh,nd [kWh/m²a]", "umestitev", "obdobje", "hiš nad pragom", "delež hiš [%]", "posameznik R3 mediana [tisoč €]", "posameznik R3 delež boljši [%]",
              "posameznik R3 vsota [mio €]", "družba mediana [tisoč €]", "družba delež boljši [%]", "družba vsota [mio €]", "CO2e 30 let vsota [kt]",
              "zimski uvoz (leto 20) vsota [GWh/a]", "proračun R3 vsota [mio €]"]].round(1).to_string(index=False))
    Dd = k23d()
    zapis(Dd, "23d_izkaznica_proti_S0.csv")
    s = A[A["kazalnik"].isin(["posameznik R3", "družba", "CO2e 30 let"]) & (A["obdobje"] != "paket")]
    print(s.pivot_table(index=["obdobje", "razred Qh,nd [kWh/m²a]", "kazalnik"], columns="umestitev",
                        values=["P3 − P1 mediana", "delež P3 boljši [%]", "hiš"]).round(1).to_string())
    print(P.round(0).to_string(index=False))
    print(Cc.round(2).to_string(index=False))
    print(Dd.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
