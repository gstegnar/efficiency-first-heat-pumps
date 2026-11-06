"""Krog 8 — kaj bi pokazal en arhetip namesto dejanskih hiš (NAVODILA_CLAUDE_CODE_v8.md).

Isti izračun P3 − P1 (paket 5.969 hiš, pred 1981 / 1981–2002) z vedno bolj poenostavljenim opisom hiš; isti urni model (03), cene, spodbude, režimi.
  L0 en tipični objekt na arhetip: referenčna hiša TABULA v S0, mediana ogrevane površine, moči TČ in PV paketa arhetipa, kotel ELKO, spodbude po pravilih
     2026; en rezultat na arhetip velja za vse hiše arhetipa.
  L1 arhetip z dejanskimi napravami: površina, TČ, PV, plačane spodbude in nadomeščeni energent hiše; Qh,nd = mediana izkaznic obdobja (rEI, stopnja 0,
     P5–P95 kot v 03); profil S0 skaliran na Qh,nd (pristop fU), ciljna prenova = vsi štirje ukrepi iz S0 (brez lestvice).
  L2 kot L1, Qh,nd hiše iz porazdelitve izkaznic obdobja (isti vzorec kot v članku), brez umestitve na lestvico (S0 skaliran, vsi štirje ukrepi).
  L3 osnova članka (lestvica stanj, umestitev po hiši, le manjkajoči ukrepi).
Koraki: F (vhodi cache/15_L*_fond.parquet) → 03 --fond … --tag _L* (4 ×) → B (ekonomika, cache/15_rezultati.parquet) → T (tabele/15_arhetip_proti_hisam.csv, Fig. S4).
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
TAB, OUT_T, CACHE, SL = HERE / "tabele", HERE / "clanek1" / "tabele", HERE / "cache", HERE / "clanek1" / "slike"
EK = PAR["ekonomika"]
RAVNI = ["L0", "L1", "L2", "L3"]
OBD = {"pred 1981": ["A1", "A2", "A3"], "1981–2002": ["A4"]}


def korak_fond():
    f = pd.read_parquet(CACHE / "01_fond.parquet")
    f["paket_osnovni"] = f["paket_osnovni"].astype(float).fillna(0).astype(bool)
    pk = f[f["paket_osnovni"]].copy()
    v = pd.read_csv(TAB / "02_variante.csv", sep=";", encoding="utf-8-sig").set_index(["arhetip", "varianta"])["Qh_nd"]
    q = pd.read_parquet(CACHE / "03_fond_letno.parquet", columns=["ID_Stavbe", "varianta", "qh"])
    q = q[q["varianta"] == "V2"].set_index("ID_Stavbe")["qh"]
    rei = pd.read_parquet(CACHE / "01_rEI_vzorec.parquet")
    rei = rei[(rei["st"] == 0) & (rei["tipg"] == "eno_dvo")]
    lo, hi = PAR["urni_model"]["rEI_percentili_meje"]
    med = {o: g["Qh_nd_EI"][(g["Qh_nd_EI"] >= g["Qh_nd_EI"].quantile(lo / 100)) & (g["Qh_nd_EI"] <= g["Qh_nd_EI"].quantile(hi / 100))].median()
           for o, g in rei.groupby("obdobje")}
    kol = ["ID_Stavbe", "Qr", "obdobje", "arhetip", "stratum", "A_ogr", "A_vir", "tc_kW", "tc_kW_imputirano", "tc_kW_uporabljena", "PV_kWp", "ima_PV",
           "paket_osnovni", "staro_ogrevanje"]
    # L3: osnova (Qh,nd kot v članku, položaj na lestvici)
    L3 = pk[kol].copy()
    L3["Qhnd_lastna_EI"] = L3["ID_Stavbe"].map(q)
    # L2: isti Qh,nd, brez lestvice — S0 skaliran, vsi štirje ukrepi
    L2 = L3.copy()
    L2["stanje_var"], L2["cilj_var"], L2["stanje_k"] = "BREZ", "FSO", 0
    L2["stanje_skala"] = L2["Qhnd_lastna_EI"] / [v[(a, "BREZ")] for a in L2["arhetip"]]
    # L1: Qh,nd = mediana izkaznic obdobja
    L1 = L2.copy()
    L1["Qhnd_lastna_EI"] = L1["obdobje"].map(med)
    L1["stanje_skala"] = L1["Qhnd_lastna_EI"] / [v[(a, "BREZ")] for a in L1["arhetip"]]
    # L0: tipična hiša na arhetip (TABULA S0, mediane paketa)
    rows = []
    for a, g in pk.groupby("arhetip"):
        rows.append({"ID_Stavbe": f"TIP_{a}", "Qr": "Q2" if a == "A4" else "Q1", "obdobje": int(g["obdobje"].mode()[0]), "arhetip": a, "stratum": a,
                     "A_ogr": float(g.loc[g["A_vir"] != "imputirano", "A_ogr"].median()), "A_vir": "mediana",
                     "tc_kW": float(g.loc[~g["tc_kW_imputirano"].astype(bool), "tc_kW"].median()), "tc_kW_imputirano": False,
                     "PV_kWp": float(g["PV_kWp"].median()), "ima_PV": True, "paket_osnovni": True, "staro_ogrevanje": "ELKO",
                     "Qhnd_lastna_EI": v[(a, "BREZ")], "stanje_var": "BREZ", "cilj_var": "FSO", "stanje_k": 0, "stanje_skala": 1.0,
                     "tip_stavbe": "enostanovanjska"})
    L0 = pd.DataFrame(rows)
    L0["tc_kW_uporabljena"] = L0["tc_kW"]
    for lab, d in (("L0", L0), ("L1", L1), ("L2", L2), ("L3", L3)):
        d["ID_Stavbe"] = d["ID_Stavbe"].astype(str)
        d.to_parquet(CACHE / f"15_{lab}_fond.parquet", index=False)
    print(f"paket {len(pk)}; mediane izkaznic obdobja {({k: round(x, 1) for k, x in med.items()})}")
    print(L0[["ID_Stavbe", "A_ogr", "tc_kW", "PV_kWp", "Qhnd_lastna_EI"]].round(1).to_string(index=False))
    print("L2 skala fU (mediana po arhetipu):", L2.groupby("arhetip")["stanje_skala"].median().round(2).to_dict())


def _moduli():
    s = importlib.util.spec_from_file_location("k21", HERE / "K2_1_drzava.py")
    K1 = importlib.util.module_from_spec(s)
    s.loader.exec_module(K1)
    return K1, K1.E, K1.DR.OKM


def korak_b():
    K1, E, OKM = _moduli()
    out = []
    for lab in RAVNI:
        w = E.load_units("fond", f"_{lab}")
        n = {rez: E.npv_paths(w, rez, EK["diskont_zasebni"], E.pf_factory("konstantne"), EK["ovoj_osnova"]) for rez in ("R1", "R3")}
        rc = K1.racun(w)
        o = pd.DataFrame(index=w.index)
        o["arhetip"] = w["arhetip"]
        for p in ("P1", "P2", "P3"):
            for rez in ("R1", "R3"):
                o[f"{p}|NPV {rez}"] = n[rez][p]
            if p != "P2":
                o[f"{p}|družbeni NPV"] = rc[p]["družbeni NPV"]
                o[f"{p}|NN kW"] = rc[p]["obremenitev NN omrežja [kW]"]
                op = sum(OKM.op_co2(w, E.leto_varianta(p, t), t) for t in range(1, E.N + 1))
                emb = sum(OKM.embodied(w, item, arg) for (y, item, arg) in E.POTI[p]["ev"])
                o[f"{p}|CO2e t"] = (op + emb) / 1000
        o["Qh_nd"] = w["V2|Qh_nd"]
        o["raven"] = lab
        out.append(o.reset_index())
    r = pd.concat(out, ignore_index=True)
    r.to_parquet(CACHE / "15_rezultati.parquet", index=False)
    print(r.groupby(["raven", "arhetip"]).size().unstack())


KAZ = [("NPV R3", "posameznik R3", "kEUR", 1e3, 1e6, "mio EUR"), ("NPV R1", "posameznik R1", "kEUR", 1e3, 1e6, "mio EUR"),
       ("družbeni NPV", "družbeni NPV", "kEUR", 1e3, 1e6, "mio EUR"), ("CO2e t", "CO2e 30 let", "t", 1, 1e3, "kt"),
       ("NN kW", "obremenitev NN omrežja", "kW", 1, 1e3, "MW")]


def razsiri(r):
    """L0: rezultat tipične hiše arhetipa za vsako hišo paketa (ID iz L3)."""
    L3 = r[r["raven"] == "L3"][["ID_Stavbe", "arhetip"]]
    t = r[r["raven"] == "L0"].drop(columns=["ID_Stavbe"]).set_index("arhetip")
    L0 = L3.join(t.drop(columns=["raven"]), on="arhetip")
    L0["raven"] = "L0"
    return pd.concat([L0, r[r["raven"] != "L0"]], ignore_index=True)


def korak_tabela():
    r = razsiri(pd.read_parquet(CACHE / "15_rezultati.parquet"))
    for k, *_ in KAZ:
        r[f"d|{k}"] = r[f"P3|{k}"] - r[f"P1|{k}"]
    for rez in ("R1", "R3"):
        r[f"mejna {rez}"] = r[f"P1|NPV {rez}"] - r[f"P2|NPV {rez}"]       # vrednost ovoja v letu 10 s PV (Fig. 5: P2 proti P1)
    ref = r[r["raven"] == "L3"].set_index("ID_Stavbe")
    rows = []
    for lab in RAVNI:
        x = r[r["raven"] == lab].set_index("ID_Stavbe")
        for obd, ars in OBD.items():
            g = x[x["arhetip"].isin(ars)]
            gr = ref.loc[g.index]
            row = {"raven": lab, "obdobje": obd, "hiš": len(g)}
            for k, lab_k, en, sc, vsc, ven in KAZ:
                d = g[f"d|{k}"]
                row[f"{lab_k}: mediana P3 − P1 [{en}]"] = d.median() / sc
                row[f"{lab_k}: delež P3 boljši [%]"] = 100 * (d < 0).mean()
                row[f"{lab_k}: vsota P3 − P1 [{ven}]"] = d.sum() / vsc
            for k, lab_k in (("NPV R3", "posameznik R3"), ("družbeni NPV", "družbeni NPV")):
                row[f"{lab_k}: napačno razvrščenih proti L3 [%]"] = 100 * ((g[f"d|{k}"] < 0) != (gr[f"d|{k}"] < 0)).mean()
                s_l, s_3 = (g[f"d|{k}"] < 0).mean() >= 0.5, (gr[f"d|{k}"] < 0).mean() >= 0.5
                row[f"{lab_k}: sklep (zmaga)"] = ("URE najprej" if s_l else "OVE najprej") + ("" if s_l == s_3 else " — DRUGAČE kot L3")
            for rez in ("R1", "R3"):
                row[f"vrednost ovoja v letu 10 s PV, {rez}: mediana [kEUR]"] = g[f"mejna {rez}"].median() / 1e3
            row["Qh,nd mediana [kWh/m²a]"] = g["Qh_nd"].median()
            rows.append(row)
    T = pd.DataFrame(rows)
    for o in (TAB, OUT_T):
        T.to_csv(o / "15_arhetip_proti_hisam.csv", sep=";", encoding="utf-8-sig", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 40)
    print(T.set_index(["obdobje", "raven"]).sort_index().T.round(1).to_string())
    slika(r)
    return T


def slika(r):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
    plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "font.family": "DejaVu Sans"})
    kaz = [("NPV R3", ("household R3 [€ thousand]", "posameznik R3 [tisoč €]"), 1e3), ("družbeni NPV", ("social NPV [€ thousand]", "družbeni NPV [tisoč €]"), 1e3)]
    ob = [("pred 1981", ("pre-1981", "pred 1981")), ("1981–2002", ("1981–2002", "1981–2002"))]
    for lang, sfx in (("en", ""), ("sl", "_sl")):
        i = 0 if lang == "en" else 1
        fig, axs = plt.subplots(2, 2, figsize=(7.2, 4.6))
        for c, (k, kl, sc) in enumerate(kaz):
            for rr, (obd, ol) in enumerate(ob):
                ax = axs[rr, c]
                g = r[r["arhetip"].isin(OBD[obd])]
                x3 = (g.loc[g["raven"] == "L3", f"d|{k}"] / sc).to_numpy()
                lo, hi = np.percentile(x3, [0.5, 99.5])
                ax.hist(np.clip(x3, lo, hi), bins=60, color="#9ecae1", edgecolor="white", linewidth=0.3, label="L3 houses" if lang == "en" else "L3 hiše")
                ax.axvline(0, color=MUTED, lw=0.8)
                for lab, col, ls in (("L0", "#e34948", "-"), ("L1", "#eda100", "--"), ("L2", "#4a3aa7", ":")):
                    xx = g.loc[g["raven"] == lab, f"d|{k}"] / sc
                    m = np.median(xx)
                    mis = 100 * ((g.loc[g["raven"] == lab, f"d|{k}"].to_numpy() < 0) != (g.loc[g["raven"] == "L3", f"d|{k}"].to_numpy() < 0)).mean()
                    ax.axvline(m, color=col, lw=1.6, ls=ls, label=f"{lab} {'median' if lang == 'en' else 'mediana'} ({mis:.0f} % {'misclassified' if lang == 'en' else 'napačno'})")
                ax.set_title(f"{ol[i]} — {kl[i]}", loc="left", fontsize=8, color=INK)
                ax.grid(axis="x", color=GRID, lw=0.6)
                ax.set_yticks([])
                ax.legend(fontsize=5.8, loc="upper left" if rr == 0 else "upper right", bbox_to_anchor=(0.0, 1.0) if rr == 0 else (1.0, 1.0), handlelength=1.6,
                          framealpha=0.85, frameon=True, edgecolor="none")
                ax.set_ylim(0, ax.get_ylim()[1] * 1.45)
        for ax in axs[1]:
            ax.set_xlabel("P3 − P1 per house (negative = efficiency first better)" if lang == "en" else "P3 − P1 na hišo (negativno = URE najprej boljši)", fontsize=7.5)
        fig.tight_layout()
        for ext in ("png", "svg"):
            fig.savefig(SL / f"FigS4_archetype_vs_houses{sfx}.{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--korak", default="F")
    a = ap.parse_args()
    {"F": korak_fond, "B": korak_b, "T": korak_tabela}[a.korak]()
