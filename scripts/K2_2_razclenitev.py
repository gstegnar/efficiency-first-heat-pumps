"""Krog 2, točka 2 — razčlenitev razlike v zimskem uvozu (nov–feb) in prispevku h sistemski konici med P1 (V2) in P3 (V5).

Shapleyjeva razčlenitev s tremi dejavniki; za vsako hišo urna simulacija v vseh 2³ kombinacijah:
  (i)  ovoj: potreba po toploti brez ovoja → FSO, skupaj z dimenzioniranjem TČ, ki ga ovoj omogoči (empirična moč → nova)
  (ii) COP: krivulja temperature dovoda brez ovoja (55 °C pri −13) → FSO (45 °C)
  (iii) PV: dejanski / empirični kWp (V2) → kWp na letno rabo po prenovi (V5)
Prispevek dejavnika = povprečje mejnih učinkov po vseh vrstnih redih (vsota prispevkov = celotna razlika).
(iv) gospodinjstvo je v P1 in P3 enako → prispevek k razliki 0; poročan je njegov delež v ravni zimske rabe.
Izhodi: clanek1/tabele/05e_razclenitev_zima_konica.csv, 05f_zimska_raba_po_rabah.csv, clanek1/slike/Fig_k2_razclenitev(_sl)
"""
from __future__ import annotations

import itertools
import sys
from math import factorial
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from urni_lib import (  # noqa: E402
    C, CACHE, ANALIZA, PROF, PV1, UM, ZIMA, heat_brez, heat_pump, size_hp, KONICA_SIS, cilj,
)

OUT_T, OUT_S = HERE / "clanek1" / "tabele", HERE / "clanek1" / "slike"
PROFILI = pd.read_parquet(CACHE / "03a_profili.parquet")
HH1 = PROFILI["H0"].to_numpy(np.float32)
F = ["ovoj", "COP", "PV"]


def sim_chunk(b: pd.DataFrame) -> dict:
    """b: ena skupina arhetipa. Vrne {kombinacija (e,c,p): (zimski uvoz kWh, konica kW, komponente)}."""
    a = b["arhetip"].iloc[0]
    A = b["A"].to_numpy(np.float32)
    obd = b["obdobje"].to_numpy()
    tsv = UM["tsv_izgube_faktor"]
    hh = HH1[:, None] * (UM["gospodinjstvo"]["raba_kWh_a"] * 1000 * b["faktor_gosp"].to_numpy(np.float32))[None, :]
    heat = {}
    fr = {ov: np.array([C.f_razvod(o, UM["stopnja_po_varianti"][ov]) for o in obd], np.float32) for ov in ("BREZ", "FSO")}
    qc_prof, _, _, _, brez = cilj(a, b["qh"].to_numpy())             # krog 3: ciljna prenova po lestvici stanj (kot v 03)
    heat["BREZ"] = (heat_brez(a, b["qh"].to_numpy()) * A / fr["BREZ"], PROF[(a, "BREZ")]["QW"][:, None].astype(np.float32) * A * tsv)
    heat["FSO"] = (qc_prof * A / np.where(brez, fr["BREZ"], fr["FSO"]), PROF[(a, "FSO")]["QW"][:, None].astype(np.float32) * A * tsv)
    sink_fso = np.where(brez, "BREZ", "FSO")
    kW = {"BREZ": b["tc_kW"].to_numpy(np.float32), "FSO": size_hp(*heat["FSO"])}
    kwp = {0: b["kwp_V2"].to_numpy(np.float32), 1: b["kwp_V5"].to_numpy(np.float32)}
    out = {}
    for e, c, p in itertools.product((0, 1), repeat=3):
        ov = "FSO" if e else "BREZ"
        QH, QW = heat[ov]
        sink = sink_fso if c else "BREZ"
        # heat_pump(QH, QW, kW, ovoj): ovoj določa le krivuljo dovoda (COP) — tu ločeno od potrebe po toploti
        el, bu, _ = heat_pump(QH, QW, kW[ov], sink)
        load = hh + el + bu
        imp = np.maximum(load - PV1[:, None] * kwp[p][None, :], 0)
        out[(e, c, p)] = {"zima": imp[ZIMA].sum(0) / 1000, "konica": imp[KONICA_SIS].mean(0) / 1000,
                          "el_TC_zima": el[ZIMA].sum(0) / 1000, "grelnik_zima": bu[ZIMA].sum(0) / 1000,
                          "gosp_zima": hh[ZIMA].sum(0) / 1000, "raba_zima": load[ZIMA].sum(0) / 1000}
    return out


def shapley(res: dict, key: str) -> dict:
    """Shapleyjevi prispevki k razliki V2 (0,0,0) → V5 (1,1,1) za kazalnik key (na hišo)."""
    n = 3
    phi = {f: 0.0 for f in F}
    for i, f in enumerate(F):
        others = [j for j in range(n) if j != i]
        for k in range(n):
            for S in itertools.combinations(others, k):
                w = factorial(k) * factorial(n - k - 1) / factorial(n)
                s0 = [0, 0, 0]
                for j in S:
                    s0[j] = 1
                s1 = list(s0)
                s1[i] = 1
                phi[f] = phi[f] + w * (res[tuple(s1)][key] - res[tuple(s0)][key])
    return phi


def _stevila():
    """Krog 13: število hiš za oznake slik iz 01_fond (paket, pred 1981, 1981–2002, fond) — ne na trdo."""
    f = pd.read_parquet(Path(__file__).resolve().parent / "cache" / "01_fond.parquet", columns=["paket_osnovni", "arhetip"])
    pk, pre = f["paket_osnovni"].astype(bool), f["arhetip"].isin(["A1", "A2", "A3"])
    return {"paket": int(pk.sum()), "pred": int((pk & pre).sum()), "po": int((pk & ~pre).sum()), "fond": len(f)}


def _en(n):
    return f"{n:,}"


def _sl(n):
    return f"{n:,}".replace(",", ".")


def main():
    fl = pd.read_parquet(CACHE / "03_fond_letno.parquet", columns=["ID_Stavbe", "varianta", "arhetip", "obdobje", "A", "qh", "TC_kW", "PV_kWp", "Qr", "paket_osnovni", "uvoz_zima_kWh", "uvoz_konica_sis_kW"])
    base = fl[fl["varianta"] == "V2"].set_index("ID_Stavbe")
    v5 = fl[fl["varianta"] == "V5"].set_index("ID_Stavbe").reindex(base.index)
    st07 = pd.read_parquet(ANALIZA / "cache" / "07_stavbe_paket.parquet", columns=["ID_Stavbe", "tip_stavbe"]).drop_duplicates("ID_Stavbe").set_index("ID_Stavbe")
    b = pd.DataFrame({"arhetip": base["arhetip"], "obdobje": base["obdobje"], "A": base["A"], "qh": base["qh"],
                      "tc_kW": base["TC_kW"], "kwp_V2": base["PV_kWp"], "kwp_V5": v5["PV_kWp"],
                      "faktor_gosp": np.where(st07["tip_stavbe"].reindex(base.index) == "dvostanovanjska", UM["gospodinjstvo"]["dvostanovanjska_faktor"], 1.0),
                      "Qr": np.where(base["arhetip"].isin(["A1", "A2", "A3"]), "Q1", "Q2"), "paket": base["paket_osnovni"].astype(bool)})   # krog 3 (C1)
    rows = []
    for a, g in b.groupby("arhetip"):
        for i in range(0, len(g), 1000):
            ch = g.iloc[i:i + 1000]
            res = sim_chunk(ch)
            phz, phk = shapley(res, "zima"), shapley(res, "konica")
            d = pd.DataFrame({"ID_Stavbe": ch.index, "Qr": ch["Qr"].values, "paket": ch["paket"].values,
                              "zima_P1": res[(0, 0, 0)]["zima"], "zima_P3": res[(1, 1, 1)]["zima"],
                              "konica_P1": res[(0, 0, 0)]["konica"], "konica_P3": res[(1, 1, 1)]["konica"],
                              **{f"zima_{f}": phz[f] for f in F}, **{f"konica_{f}": phk[f] for f in F},
                              **{f"{k}_P1": res[(0, 0, 0)][k] for k in ("el_TC_zima", "grelnik_zima", "gosp_zima", "raba_zima")},
                              **{f"{k}_P3": res[(1, 1, 1)][k] for k in ("el_TC_zima", "grelnik_zima", "gosp_zima", "raba_zima")}})
            rows.append(d)
        print("  ", a, len(g))
    D = pd.concat(rows, ignore_index=True)
    # kontrola: V2/V5 iz 03 (zimski uvoz)
    kontrola = tuple(np.abs(D[f"{k}_{p}"].to_numpy() - src.reindex(D["ID_Stavbe"])[c].to_numpy()).max()
                     for k, c in (("zima", "uvoz_zima_kWh"), ("konica", "uvoz_konica_sis_kW")) for p, src in (("P1", base), ("P3", v5)))

    vz = {"paket": D["paket"], "paket pred 1981": D["paket"] & (D["Qr"] == "Q1"), "paket 1981–2002": D["paket"] & (D["Qr"] == "Q2"),
          "fond pred 1981": D["Qr"] == "Q1", "fond 1981–2002": D["Qr"] == "Q2", "fond": pd.Series(True, index=D.index)}
    out = []
    for lab, m in vz.items():
        d = D[m]
        for kaz, en, sc in (("zima", "GWh/a", 1e6), ("konica", "MW", 1e3)):
            dif = d[f"{kaz}_P3"] - d[f"{kaz}_P1"]
            row = {"vzorec": lab, "stavb": len(d), "kazalnik": "zimski uvoz" if kaz == "zima" else "prispevek h sistemski konici", "enota vsote": en,
                   "P1 vsota": d[f"{kaz}_P1"].sum() / sc, "P3 vsota": d[f"{kaz}_P3"].sum() / sc,
                   "P3−P1 vsota": dif.sum() / sc}
            for f in F:
                row[f"{f} vsota"] = d[f"{kaz}_{f}"].sum() / sc
                row[f"{f} delež [%]"] = 100 * d[f"{kaz}_{f}"].sum() / dif.sum()
                row[f"{f} mediana na hišo"] = d[f"{kaz}_{f}"].median()
            row["gospodinjstvo vsota"] = 0.0
            out.append(row)
    T = pd.DataFrame(out)
    T.round(3).to_csv(OUT_T / "05e_razclenitev_zima_konica.csv", sep=";", encoding="utf-8-sig", index=False)
    # zimska raba po rabah (raven) za P1 in P3
    lv = []
    for lab, m in vz.items():
        d = D[m]
        for p in ("P1", "P3"):
            raba = d[f"raba_zima_{p}"].sum()
            lv.append({"vzorec": lab, "pot": p, "TČ [GWh]": d[f"el_TC_zima_{p}"].sum() / 1e6, "grelnik [GWh]": d[f"grelnik_zima_{p}"].sum() / 1e6,
                       "gospodinjstvo [GWh]": d[f"gosp_zima_{p}"].sum() / 1e6, "raba skupaj [GWh]": raba / 1e6,
                       "uvoz [GWh]": d[f"zima_{p}"].sum() / 1e6, "PV krije pozimi [GWh]": (raba - d[f"zima_{p}"].sum()) / 1e6,
                       "delež gospodinjstva v zimski rabi [%]": 100 * d[f"gosp_zima_{p}"].sum() / raba})
    L = pd.DataFrame(lv)
    L.round(3).to_csv(OUT_T / "05f_zimska_raba_po_rabah.csv", sep=";", encoding="utf-8-sig", index=False)

    # slika: vodoravni stolpci prispevkov (paket pred 1981, paket 1981–2002, paket skupaj), zimski uvoz in konica
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = {"ovoj": "#2a78d6", "COP": "#eb6834", "PV": "#eda100"}
    lab_f = {"en": {"ovoj": "envelope (lower heat demand, smaller HP)", "COP": "lower flow temperature (COP)", "PV": "smaller PV"},
             "sl": {"ovoj": "ovoj (nižja potreba po toploti, manjša TČ)", "COP": "nižja temperatura dovoda (COP)", "PV": "manjši PV"}}
    for lang, sfx in (("en", ""), ("sl", "_sl")):
        fig, axs = plt.subplots(1, 2, figsize=(10, 3.4))
        for ax, kaz, xl in zip(axs, ("zimski uvoz", "prispevek h sistemski konici"),
                               (("Change in winter grid import P3 − P1 [GWh/a]", "Sprememba zimskega uvoza P3 − P1 [GWh/a]"),
                                ("Change in contribution to system peak P3 − P1 [MW]", "Sprememba prispevka h sistemski konici P3 − P1 [MW]"))):
            sub = T[(T["kazalnik"] == kaz) & T["vzorec"].isin(["paket 1981–2002", "paket pred 1981", "paket"])].set_index("vzorec").loc[["paket 1981–2002", "paket pred 1981", "paket"]]
            for yi, (vlab, rr) in enumerate(sub.iterrows()):
                left_neg, left_pos = 0.0, 0.0
                for f in F:
                    v = rr[f"{f} vsota"]
                    if v < 0:
                        ax.barh(yi, v, left=left_neg, color=cols[f], height=0.6, label=lab_f[lang][f] if yi == 0 else None)
                        left_neg += v
                    else:
                        ax.barh(yi, v, left=left_pos, color=cols[f], height=0.6, label=lab_f[lang][f] if yi == 0 else None)
                        left_pos += v
                ax.plot(rr["P3−P1 vsota"], yi, "k|", ms=16, mew=2)
                ax.text(left_neg, yi + 0.36, f"{rr['P3−P1 vsota']:.1f}", fontsize=7, color="#0b0b0b")
            ax.set_yticks(range(3))
            N = _stevila()
            ax.set_yticklabels({"en": [f"package 1981–2002 ({_en(N['po'])})", f"package pre-1981 ({_en(N['pred'])})", f"package ({_en(N['paket'])})"],
                                "sl": [f"paket 1981–2002 ({_sl(N['po'])})", f"paket pred 1981 ({_sl(N['pred'])})", f"paket ({_sl(N['paket'])})"]}[lang], fontsize=7.5)
            ax.axvline(0, color="#52514e", lw=0.8)
            ax.set_xlabel(xl[0 if lang == "en" else 1], fontsize=8)
            ax.grid(axis="y", visible=False)
        h, l = axs[0].get_legend_handles_labels()
        h.append(plt.Line2D([], [], color="k", marker="|", ls="", ms=10, mew=2)); l.append({"en": "net P3 − P1", "sl": "neto P3 − P1"}[lang])
        fig.legend(h, l, fontsize=7.5, loc="lower center", ncol=4, frameon=False)
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        for ext in ("png", "svg"):
            fig.savefig(OUT_S / f"Fig_k2_razclenitev{sfx}.{ext}", dpi=300)
        plt.close(fig)

    pd.set_option("display.width", 250)
    print("kontrola proti 03 (največja abs. razlika; zima kWh P1,P3; konica kW P1,P3):", np.round(kontrola, 3))
    print(T[["vzorec", "kazalnik", "enota vsote", "P1 vsota", "P3 vsota", "P3−P1 vsota", "ovoj vsota", "COP vsota", "PV vsota", "ovoj delež [%]", "COP delež [%]", "PV delež [%]"]].round(2).to_string(index=False))
    print()
    print(L.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
