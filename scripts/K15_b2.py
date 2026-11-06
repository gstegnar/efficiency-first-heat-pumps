"""Krog 15 — osnova B2 (umestitev po izkaznici) na G13 s faktorjem S0 A1/A2 iz 23d; register (krog 13) kot spodnja meja (NAVODILA_CLAUDE_CODE_v15.md).

--korak s0   S0–S3 po umeritvi proti izkaznicam (kot 22b): G11 (osnova kroga 12) | G13 register (krog 13, brez faktorja) | G13 + faktor (krog 15)
             → tabele/24f_S0_faktor_obcutljivost.csv (občutljivost S0F)
--korak 24c  prag potrebne toplote ob vgradnji (> 180 / 140–180 / 90–140 / < 90) × obdobje, paket in fond, pod B2 (osnova, brez faktorja S0),
             B2 + faktor S0 (_S0F) in registrom (_REG) → tabele/24c_prag_180.csv
--korak 24ab  register kot spodnja meja ob glavni tabeli iz tabel 09 (B2 / S0F / register) → tabele/24a_glavna_B2_register.csv, 24b_obcutljivost_umestitev.csv
--korak 24g   Monte Carlo 12b pod B2 / S0F / register → tabele/24g_monte_carlo_umestitev.csv
--korak 24d   populacija s pragom iz 08 (štirje scenariji) → tabele/24d_populacija_prag.csv
--korak fig23 Fig23_break_even_umestitev (B2 / S0F / register, navpičnici 90 in 180) → tabele/24h_prelomna_umestitev*.csv
--korak 24e   razstavitev (P1r − P1) + (P3 − P1r) pod B2, po razredih praga → tabele/24e_razstavitev_B2.csv
--korak kontrola  paket pred 1981 / 1981–2002: P3 − P1 (R3, družba) iz osnovnih tabel proti 22f
Zagon: C:/Anaconda/envs/new_env/python.exe K15_b2.py --korak s0
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TAB, OUT_T, CACHE = HERE / "tabele", HERE / "clanek1" / "tabele", HERE / "cache"
ARH = {1: "A1", 2: "A2", 3: "A3", 4: "A4"}


def rd(f):
    return pd.read_csv(f, sep=";", encoding="utf-8-sig")


def zapis(T, ime):
    for o in (TAB, OUT_T):
        T.round(3).to_csv(o / ime, sep=";", encoding="utf-8-sig", index=False)


def korak_s0(reg_dat):
    """reg_dat: 02_variante kroga 13 (G13 register, brez faktorja)."""
    vN = rd(TAB / "02_variante.csv").set_index(["arhetip", "varianta"])["Qh_nd"]
    vR = rd(reg_dat).set_index(["arhetip", "varianta"])["Qh_nd"]
    v11 = rd(TAB / "02_variante_G11.csv").set_index(["arhetip", "varianta"])["Qh_nd"]
    r = pd.read_parquet(CACHE / "01_rEI_vzorec.parquet")
    r = r[r["tipg"] == "eno_dvo"]
    rows = []
    for o, a in ARH.items():
        for st, v in ((0, "BREZ"), (1, "S1"), (2, "S2"), (3, "S3")):
            x = r[(r["obdobje"] == o) & (r["st"] == st)]["Qh_nd_EI"]
            p25, p50, p75, p90 = x.quantile(.25), x.median(), x.quantile(.75), x.quantile(.9)
            q = vN[(a, v)]
            if st == 0:
                pas = "P75–P90" if p75 <= q <= p90 else ("nad P90" if q > p90 else ("P50–P75" if q >= p50 else "pod P50"))
            else:
                pas = "P25–P75" if p25 <= q <= p75 else ("nad P75" if q > p75 else "pod P25")
            rows.append({"arhetip": a, "stanje": {0: "S0", 1: "S1", 2: "S2", 3: "S3"}[st], "G11 (krog 12)": v11.get((a, v), np.nan),
                         "G13 register (krog 13)": vR[(a, v)], "G13 + faktor (krog 15)": q, "izkaznice n": len(x), "P25": p25, "P50": p50, "P75": p75, "P90": p90,
                         "krog 15 / P50": q / p50, "krog 15 / P90": q / p90 if st == 0 else np.nan, "krog 15 v pasu": pas})
        for v in ("FSO", "GLOB"):
            rows.append({"arhetip": a, "stanje": v, "G11 (krog 12)": v11.get((a, v), np.nan), "G13 register (krog 13)": vR[(a, v)], "G13 + faktor (krog 15)": vN[(a, v)]})
    T = pd.DataFrame(rows)
    mono = {a: bool(vN[(a, "BREZ")] > vN[(a, "S1")] > vN[(a, "S2")] > vN[(a, "S3")]) for a in ARH.values()}
    zapis(T, "24f_S0_faktor_obcutljivost.csv")
    pd.set_option("display.width", 250)
    print(T.round(1).to_string(index=False))
    print("monotonost S0 > S1 > S2 > S3:", mono)
    print("monotonost S0 A1 > A2 > A3 > A4:", bool(vN[("A1", "BREZ")] > vN[("A2", "BREZ")] > vN[("A3", "BREZ")] > vN[("A4", "BREZ")]))


RAZ24 = [("> 180", 180, 1e9), ("140–180", 140, 180), ("90–140", 90, 140), ("< 90", -1, 90)]
TEKI = {"B2 (osnova)": "", "B2 + faktor S0 (občutljivost)": "_S0F", "register (spodnja meja)": "_REG"}


def korak_24c():
    """Prag potrebne toplote ob vgradnji: fond in paket po razredih × obdobje × umestitev; P3 − P1 in delež prihranka, ki ga zajame razred / prag.
    → tabele/24c_prag_180.csv"""
    import importlib.util
    sp = importlib.util.spec_from_file_location("k14", HERE / "K14_umestitev.py")
    K = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(K)
    pak = pd.read_parquet(CACHE / "01_fond.parquet", columns=["ID_Stavbe", "paket_osnovni"]).set_index("ID_Stavbe")["paket_osnovni"]
    kaz = ["posameznik R3", "posameznik R1", "družba", "proračun R3", "CO2e 30 let", "zimski uvoz (leto 20)", "obremenitev NN"]
    rows = []
    for um, tag in TEKI.items():
        _, d = K.izracun(tag)
        d["paket"] = pak.reindex(d.index).astype(float).fillna(0).astype(bool).to_numpy()
        print(um, len(d), "hiš, paket", int(d["paket"].sum()))
        for vz in ("paket", "fond"):
            dv = d[d["paket"]] if vz == "paket" else d
            for obd in ("pred 1981", "1981–2002", "vse"):
                g0 = dv if obd == "vse" else dv[dv["obdobje"] == obd]
                for rz, lo, hi in RAZ24 + [("≥ 180 (prag)", 180, 1e9), ("vse", -1, 1e9)]:
                    g = g0[(g0["Qh"] > lo) & (g0["Qh"] <= hi)]
                    r = {"umestitev": um, "vzorec": vz, "obdobje": obd, "razred Qh,nd [kWh/m²a]": rz, "hiš": len(g), "delež hiš [%]": 100 * len(g) / max(len(g0), 1)}
                    for k in kaz:
                        eu, dvz, es, ds = K.KAZ[k]
                        x = g[k]
                        r[f"{k} mediana [{eu}]"] = x.median() / dvz
                        r[f"{k} delež P3 boljši [%]"] = 100 * (x < 0).mean() if len(x) else np.nan
                        r[f"{k} vsota [{es}]"] = x.sum() / ds
                    for k in ("CO2e 30 let", "zimski uvoz (leto 20)", "obremenitev NN"):
                        tot = g0[k].sum()
                        r[f"{k}: delež prihranka obdobja [%]"] = 100 * g[k].sum() / tot if tot else np.nan
                    rows.append(r)
    T = pd.DataFrame(rows)
    zapis(T, "24c_prag_180.csv")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    c = ["umestitev", "obdobje", "razred Qh,nd [kWh/m²a]", "hiš", "delež hiš [%]", "posameznik R3 mediana [tisoč €]", "posameznik R3 delež P3 boljši [%]",
         "družba mediana [tisoč €]", "družba delež P3 boljši [%]", "CO2e 30 let delež P3 boljši [%]", "CO2e 30 let: delež prihranka obdobja [%]",
         "zimski uvoz (leto 20): delež prihranka obdobja [%]", "obremenitev NN: delež prihranka obdobja [%]"]
    for vz in ("paket", "fond"):
        print(f"\n=== {vz}")
        print(T[T["vzorec"] == vz][c].round(1).to_string(index=False))


K24A = {"posameznik R1 [tisoč €]": "NPV stroškov 30 let, R1", "posameznik R3 [tisoč €]": "NPV stroškov 30 let, R3", "proračun R3 [tisoč €]": "javni proračun NPV, R3",
        "družba [tisoč €]": "družbeni NPV", "CO2e 30 let [t]": "CO2e 30 let (obratovalni + vgrajeni)", "zimski uvoz, leto 20 [kWh/a]": "zimski uvoz (nov–feb), leto 20",
        "obremenitev NN [kW]": "obremenitev NN omrežja (maks. sočasni uvoz / oddaja PV)"}
VZ24 = ["paket", "paket pred 1981", "paket 1981–2002", "fond", "fond pred 1981", "fond 1981–2002"]
KR24 = {"B2 (osnova)": "B2", "B2 + faktor S0 (občutljivost)": "S0F", "register (spodnja meja)": "register"}


def korak_24ab():
    """Register kot spodnja meja ob glavni tabeli (korak 2) iz tabel 09 treh tekov (B2 osnova / B2 + S0F / register); brez novega izračuna.
    → tabele/24a_glavna_B2_register.csv (mediana, delež P3 < P1, vsota; umestitve v stolpcih), tabele/24b_obcutljivost_umestitev.csv (kot 22f, tudi P3 − P1r)"""
    T = {um: rd(OUT_T / f"09_primerjava_pristopov{tag}.csv") for um, tag in TEKI.items()}
    a, b = [], []
    for k, kz in K24A.items():
        for vz in VZ24:
            r = {"kazalnik": k, "vzorec": vz}
            for um, t in T.items():
                x = t[(t["kazalnik"] == kz) & (t["vzorec"] == vz)].iloc[0]
                kr = KR24[um]
                r["hiš"], r["enota vsote"] = int(x["stavb"]), x["enota vsote"]
                r[f"{kr}: P3 − P1 mediana"], r[f"{kr}: delež P3 < P1 [%]"], r[f"{kr}: P3 − P1 vsota"] = x["P3−P1 mediana"], x["delež P3 < P1 [%]"], x["P3−P1 vsota"]
                pk = t[(t["kazalnik"] == kz) & (t["vzorec"] == "paket")]["stavb"].iloc[0]
                bu = t[(t["kazalnik"] == kz) & (t["vzorec"] == "paket brez ukrepa na ovoju v P3")]["stavb"]
                b.append({"umestitev": um, "kazalnik": k, "vzorec": vz, "hiš": int(x["stavb"]), "P3 − P1 mediana": x["P3−P1 mediana"],
                          "delež P3 < P1 [%]": x["delež P3 < P1 [%]"], "P3 − P1 vsota": x["P3−P1 vsota"], "enota vsote": x["enota vsote"],
                          "P3 − P1r mediana": x["P3−P1r mediana"], "delež P3 < P1r [%]": x["delež P3 < P1r [%]"],
                          "paket brez ukrepa na ovoju v P3 [%]": 100 * bu.iloc[0] / pk if (vz == "paket" and len(bu)) else (0.0 if vz == "paket" else np.nan)})
            r["razpon mediane register − B2"] = r["register: P3 − P1 mediana"] - r["B2: P3 − P1 mediana"]
            a.append(r)
    A, B = pd.DataFrame(a), pd.DataFrame(b)
    zapis(A, "24a_glavna_B2_register.csv")
    zapis(B, "24b_obcutljivost_umestitev.csv")
    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", 30)
    print(A.drop(columns=["enota vsote"]).round(1).to_string(index=False))


def korak_24g():
    """Monte Carlo (12b, 200 vzorcev parametrov) pod B2 / S0F / register: razpon mediane in deleža P3 boljši po vzorcih → tabele/24g_monte_carlo_umestitev.csv"""
    rows = []
    for um, tag in TEKI.items():
        t = rd(OUT_T / f"12b_monte_carlo{tag}.csv")
        for (pr, k, vz), g in t.groupby(["primerjava", "kazalnik", "vzorec"], sort=False):
            m, d = g["mediana"], g["delež P3 boljši [%]"]
            rows.append({"umestitev": um, "primerjava": pr, "kazalnik": k, "vzorec": vz, "vzorcev MC": len(g), "enota": g["enota"].iloc[0],
                         "mediana P5": m.quantile(.05), "mediana P50": m.median(), "mediana P95": m.quantile(.95),
                         "delež P3 boljši P5 [%]": d.quantile(.05), "delež P3 boljši P50 [%]": d.median(), "delež P3 boljši P95 [%]": d.quantile(.95),
                         "vzorcev z mediano P3 boljši [%]": 100 * (m < 0).mean()})
    T = pd.DataFrame(rows)
    zapis(T, "24g_monte_carlo_umestitev.csv")
    pd.set_option("display.width", 250)
    print(T[T["primerjava"] == "P3 − P1"].drop(columns=["primerjava", "vzorcev MC", "enota"]).round(1).to_string(index=False))


def korak_24d():
    """Populacija s pragom (korak 3.2) iz izhodov 08 (B2): (a) obstoječi fond — dejansko / samo dimenzioniranje / pravilo s pragom 180 / URE najprej;
    (b) tok novih 2026–2035 — nadaljevanje / ukrep A prag 90 / samo dimenzioniranje / prag 180 + dimenzioniranje (od 2028) → tabele/24d_populacija_prag.csv"""
    a = rd(TAB / "08_populacija.csv")
    kp = [c for c in a.columns if c.startswith("pravilo s pragom")][0]
    a = a.rename(columns={kp: "pravilo s pragom 180 (P3/PC nad, P1r/PA pod)"})
    A = a.melt(id_vars=["skupina", "stavb", "kazalnik"], var_name="scenarij", value_name="vrednost").assign(del_="(a) obstoječi fond")
    b = rd(TAB / "08_tok_novih_povzetek.csv")
    B = b.melt(id_vars=["kazalnik"], var_name="scenarij", value_name="vrednost").assign(del_="(b) tok novih vgradenj 2026–2035, 5.633 hiš/leto",
                                                                                        skupina="nove TČ Q1+Q2 brez ovoja", stavb=np.nan)
    T = pd.concat([A, B], ignore_index=True).rename(columns={"del_": "del"})[["del", "skupina", "stavb", "kazalnik", "scenarij", "vrednost"]]
    zapis(T, "24d_populacija_prag.csv")
    for f in ("08_populacija.csv", "08_tok_novih.csv", "08_tok_novih_povzetek.csv"):
        rd(TAB / f).to_csv(OUT_T / f, sep=";", encoding="utf-8-sig", index=False)
    pd.set_option("display.width", 250)
    print(b.round(1).to_string(index=False))


def korak_fig23():
    """Fig23_break_even_umestitev (korak 3.3): mediana P3 − P1 po razredih 20 kWh/m²a (K14.k23b) za paket pod B2 / S0F / register, navpičnici pri 90 in 180.
    Tabele 23b ne prepiše (ostane krog 14)."""
    import importlib.util
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sp = importlib.util.spec_from_file_location("k14", HERE / "K14_umestitev.py")
    K = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(K)
    D = {um: K.izracun(tag)[1] for um, tag in (("B2", "_L3"), ("S0F", "_L3_S0F"), ("register", "_L3_REG"))}
    B, P = K.k23b(D)
    zapis(B, "24h_prelomna_umestitev.csv")
    zapis(P, "24h_prelomna_umestitev_tocke.csv")
    C = {"B2": "#2a78d6", "S0F": "#8e6bbf", "register": "#c0392b"}
    LU = {"B2": ("B2 (base)", "B2 (osnova)"), "S0F": ("B2 + S0 factor", "B2 + faktor S0"), "register": ("register (lower bound)", "register (spodnja meja)")}
    LS = {"pred 1981": "-", "1981–2002": "--"}
    pak = pd.read_parquet(CACHE / "01_fond.parquet", columns=["paket_osnovni"])["paket_osnovni"].astype(float).fillna(0).astype(bool).sum()
    for lang in ("en", "sl"):
        sl = lang == "sl"
        fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.7))
        for ax, (k, yl) in zip(axs, (("posameznik R3", ("Household R3, P3 − P1 [€ thousand]", "Posameznik R3, P3 − P1 [tisoč €]")),
                                     ("družba", ("Social CBA, P3 − P1 [€ thousand]", "Družbeni NPV, P3 − P1 [tisoč €]")),
                                     ("CO2e 30 let", ("CO$_2$e 30 years, P3 − P1 [t]", "CO$_2$e 30 let, P3 − P1 [t]")))):
            for um in ("B2", "S0F", "register"):
                for obd in ("pred 1981", "1981–2002"):
                    g = B[(B["umestitev"] == um) & (B["obdobje"] == obd) & (B["kazalnik"] == k)]
                    lo = {"pred 1981": ("pre-1981", "pred 1981"), "1981–2002": ("1981–2002", "1981–2002")}[obd][sl]
                    ax.plot(g["Qh,nd razred sredina"], g["P3 − P1 mediana"], color=C[um], ls=LS[obd], marker="o", ms=2.5, lw=1.3, label=f"{LU[um][sl]}, {lo}")
            ax.axhline(0, color="#52514e", lw=0.8)
            for xv, t in ((90, "90"), (180, "180")):
                ax.axvline(xv, color="#52514e", lw=0.9, ls=":")
                ax.text(xv, 1.0, t, transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=7, color="#52514e")
            ax.set_xlabel("Heat need at HP installation [kWh/m²a]" if not sl else "Potrebna toplota ob vgradnji TČ [kWh/m²a]", fontsize=8)
            ax.set_ylabel(yl[sl], fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(color="#e6e5e0", lw=0.6)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
        h, l = axs[0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=3, fontsize=6.8, frameon=False, bbox_to_anchor=(0.5, 0.06))
        n = f"{pak:,}".replace(",", "." if sl else ",")
        fig.text(0.01, 0.01, (f"Median per 20 kWh/m²a bin (≥ 30 houses); below 0 = efficiency first better. Dotted: 90 (current measure A) and 180 kWh/m²a "
                              f"(proposed threshold). B2 and B2 + S0 factor coincide for 1981–2002 (factor only A1/A2). HP + PV package, {n} houses, G13." if not sl else
                              f"Mediana po razredih 20 kWh/m²a (≥ 30 hiš); pod 0 = URE najprej boljši. Pikčasto: 90 (sedanji ukrep A) in 180 kWh/m²a "
                              f"(predlagani prag). B2 in B2 + faktor S0 se pri 1981–2002 prekrivata (faktor le A1/A2). Paket TČ + PV, {n} hiš, G13."), fontsize=6.8, color="#52514e")
        fig.tight_layout(rect=(0, 0.17, 1, 0.97))
        for ext in ("png", "svg"):
            fig.savefig(OUT_T.parent / "slike" / f"Fig23_break_even_umestitev{'_sl' if sl else ''}.{ext}", dpi=300)
        plt.close(fig)
    pd.set_option("display.width", 250)
    print(P.round(0).to_string(index=False))


def korak_24e():
    """Razstavitev P3 − P1 = (P1r − P1) + (P3 − P1r) pod B2 (korak 4; kot K13 --korak 22d), paket in fond, tudi po razredih praga potrebne toplote ob vgradnji.
    → tabele/24e_razstavitev_B2.csv (22d ostane krog 13)"""
    import importlib.util
    mods = {}
    for ime, f in (("k23", "K2_3_primerjava.py"), ("k13", "K13_toplotni_ovoj.py")):
        sp = importlib.util.spec_from_file_location(ime, HERE / f)
        mods[ime] = importlib.util.module_from_spec(sp)
        sp.loader.exec_module(mods[ime])
    K3, K13 = mods["k23"], mods["k13"]
    w = K3.E.load_units("fond")
    meta = pd.read_parquet(CACHE / "01_fond.parquet", columns=["ID_Stavbe", "paket_osnovni", "arhetip"]).set_index("ID_Stavbe").reindex(w.index)
    pk = meta["paket_osnovni"].astype(float).fillna(0).astype(bool).to_numpy()
    pre = meta["arhetip"].isin(["A1", "A2", "A3"]).to_numpy()
    qh = w["V2|Qh_nd"].to_numpy(float)
    K = K3.kazalniki(w)
    brez = w["V4|brez_ovoja_P3"].to_numpy(float) > 0.5
    kaz = {lab: key for key in K for p, k, lab in K13.KAZ22 if key[0] == p and key[1] == k}
    vz = {}
    for vn, base in (("paket", pk), ("fond", np.ones(len(w), bool))):
        for on, om in (("vse", np.ones(len(w), bool)), ("pred 1981", pre), ("1981–2002", ~pre)):
            vz[(vn, on, "vse")] = base & om
            for rz, lo, hi in RAZ24:
                vz[(vn, on, rz)] = base & om & (qh > lo) & (qh <= hi)
    rows = []
    for lab, key in kaz.items():
        df = K[key].reindex(w.index)
        d0, dr0, de0 = ((df[a] - df[b]).to_numpy(float) for a, b in (("P3", "P1"), ("P1r", "P1"), ("P3", "P1r")))
        tol = 1e-6 * max(1.0, np.nanmax(np.abs(de0)))
        for (vn, ob, rz), m in vz.items():
            if m.sum() < 20:
                continue
            d, dr, de = d0[m], dr0[m], de0[m]
            tie = np.isclose(de, 0, atol=tol)
            rows.append({"kazalnik": lab, "vzorec": vn, "obdobje": ob, "razred Qh,nd [kWh/m²a]": rz, "hiš": int(m.sum()), "enota vsote": key[5],
                         "P3 − P1 mediana": np.nanmedian(d) / key[3], "P1r − P1 mediana": np.nanmedian(dr) / key[3], "P3 − P1r mediana": np.nanmedian(de) / key[3],
                         "P3 − P1 vsota": np.nansum(d) / key[4], "P1r − P1 vsota (dimenzioniranje)": np.nansum(dr) / key[4], "P3 − P1r vsota (ovoj)": np.nansum(de) / key[4],
                         "P3 boljši od P1r [%]": 100 * np.mean((de < 0) & ~tie), "P3 = P1r [%]": 100 * np.mean(tie), "P3 slabši od P1r [%]": 100 * np.mean((de > 0) & ~tie),
                         "hiš brez ukrepa na ovoju v P3 [%]": 100 * brez[m].mean(),
                         "P3 boljši od P1r med hišami z ukrepom na ovoju [%]": 100 * np.mean(((de < 0) & ~tie)[~brez[m]]) if (~brez[m]).any() else np.nan})
    D = pd.DataFrame(rows)
    zapis(D, "24e_razstavitev_B2.csv")
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 30)
    s = D[(D["vzorec"] == "paket") & D["kazalnik"].str.contains("R3 |družbeni|CO2e")]
    print(s.drop(columns=["vzorec", "enota vsote", "P3 = P1r [%]"]).round(1).to_string(index=False))


def korak_kontrola():
    t = rd(TAB / "09_primerjava_pristopov.csv")
    print(t.head(40).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--korak", default="s0")
    ap.add_argument("--reg", default=str(TAB / "02_variante_REG.csv"))
    a = ap.parse_args()
    {"s0": lambda: korak_s0(a.reg), "kontrola": korak_kontrola, "24c": korak_24c, "24ab": korak_24ab, "24g": korak_24g, "24d": korak_24d, "fig23": korak_fig23, "24e": korak_24e}[a.korak]()
