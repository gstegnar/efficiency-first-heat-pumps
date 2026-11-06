"""Korak 08 — preračun na populacijo.

(a) Obstoječi fond 29.977 eno-/dvostanovanjskih stavb s TČ v Q1+Q2 brez ovoja: dejansko stanje (P1, če ima stavba PV, sicer PA)
    proti protidejstvu »URE-najprej« (P3 oz. PC). Razlika = cena zaklepanja: zimski uvoz, prispevek h sistemski konici (leto 20),
    CO2 v 30 letih, spodbude, NPV posameznika (R3). Podmnožice: Q1, paket TČ+PV v Q1 (2.751) in Q1+Q2 (5.969).
(b) Tok novih vgradenj (parametri.yaml: populacija): scenarij »nadaljevanje« (vse kohorte kot dejansko stanje) proti »URE-najprej
    od 2028« (kohorte od leta ukrepa A: stavbe s Qh,nd nad pragom gredo po P3/PC). Letne serije do horizonta.
Na hišo se uporabijo rezultati stavb fonda (koraki 03–06); fond je reprezentativni vzorec tokov (mešanica PV / brez PV, Qh,nd, A).
Izhodi: tabele/08_populacija.csv (a), tabele/08_tok_novih.csv (b, letno), 08_povzetek.md, slike/08_tok_novih(.png, _sl)
"""
from __future__ import annotations

import importlib.util
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TAB, CACHE, SLIKE = HERE / "tabele", HERE / "cache", HERE / "slike"
spec = importlib.util.spec_from_file_location("dr", HERE / "05_drzava.py")
DR = importlib.util.module_from_spec(spec)
spec.loader.exec_module(DR)
E, O = DR.EKM, DR.OKM
PAR = E.PAR
PO, OK = PAR["populacija"], PAR["okolje"]
N = E.N


def traj_year(Y):
    yrs = sorted(OK["EF_trajektorija"])
    return float(np.interp(Y, yrs, [OK["EF_trajektorija"][y] for y in yrs]))


def var_col(w, v, q):
    return w[f"{v}|{q}"].to_numpy(float)


def main():
    t0 = time.time()
    w = E.load_units("fond")
    f1 = pd.read_parquet(CACHE / "01_fond.parquet", columns=["ID_Stavbe", "ima_PV", "paket_osnovni"]).set_index("ID_Stavbe").reindex(w.index)
    qh = pd.read_parquet(CACHE / "03_fond_letno.parquet", columns=["ID_Stavbe", "qh"]).drop_duplicates("ID_Stavbe").set_index("ID_Stavbe")["qh"].reindex(w.index)
    ima_pv = f1["ima_PV"].fillna(False).astype(bool).to_numpy()
    paket = f1["paket_osnovni"].fillna(False).astype(bool).to_numpy()
    qr = w["Qr"].to_numpy()
    asis = np.where(ima_pv, "P1", "PA")
    ure = np.where(ima_pv, "P3", "PC")
    dim = np.where(ima_pv, "P1r", "PA")                       # krog 3: »OVE najprej, pravilno dimenzionirano« (brez PV ostane PA)
    prag = qh.to_numpy() > PO["ukrep_A_prag_Qhnd"]
    ure_A = np.where(prag, ure, asis)                         # ukrep A: samo nad pragom
    dim_A = np.where(prag, dim, asis)                         # primerjava: pravilno dimenzionirano nad pragom, brez ovoja
    prag2 = qh.to_numpy() > PO["prag_ovoj_Qhnd"]
    pr_A = np.where(prag2, ure, dim_A)                        # krog 15: ovoj nad 180, pod tem (nad 90) le dimenzioniranje

    paths = ["P1", "PA", "P3", "PC", "P1r"]
    # na stavbo: zimski uvoz in konica (leto t), obratovalni CO2 brez trajektorije (osnova 2025), dogodki: spodbude, vgrajeni CO2
    zima = {p: np.stack([var_col(w, E.leto_varianta(p, t), "uvoz_zima_kWh") for t in range(1, N + 1)], 1) for p in paths}
    kon = {p: np.stack([var_col(w, E.leto_varianta(p, t), "uvoz_konica_sis_kW") for t in range(1, N + 1)], 1) for p in paths}
    co2b = {p: np.stack([var_col(w, E.leto_varianta(p, t), "CO2_el_urni_kg") for t in range(1, N + 1)], 1) for p in paths}
    sp_nom, _ = E.spodbude_poti(w)
    emb = {p: np.zeros((len(w), N + 1)) for p in paths}
    for p in paths:
        for (y, item, arg) in E.POTI[p]["ev"]:
            emb[p][:, min(y, N)] += O.embodied(w, item, arg)
    npv = pd.read_parquet(CACHE / "04_npv_fond.parquet")
    npv = npv[npv["rezim"] == "R3"].set_index("ID_Stavbe").reindex(w.index)
    c6 = pd.read_parquet(CACHE / "06_co2_poti_fond.parquet").set_index("ID_Stavbe").reindex(w.index)

    def pick(dct, sel):
        out = np.zeros_like(next(iter(dct.values())))
        for p in np.unique(sel):
            m = sel == p
            out[m] = dct[p][m]
        return out

    # ---------------- (a) obstoječi fond
    def summ(sel, mask):
        z20 = pick(zima, sel)[mask, 19].sum() / 1e6
        k20 = pick(kon, sel)[mask, 19].sum() / 1e3
        co2 = sum(c6.loc[mask & (sel == p), f"{p}|skupaj_t"].sum() for p in np.unique(sel)) / 1e3
        sp = sum(sp_nom.loc[mask & (sel == p), p].sum() for p in np.unique(sel)) / 1e6
        nv = sum(npv.loc[mask & (sel == p), p].sum() for p in np.unique(sel)) / 1e6
        return {"stavb": int(mask.sum()), "zimski uvoz [GWh/a, leto 20]": z20, "sistemska konica [MW, leto 20]": k20,
                "CO2e 30 let [kt]": co2, "spodbude [mio EUR]": sp, "NPV posameznikov R3 [mio EUR]": nv}

    sets = {"fond Q1+Q2 brez ovoja": np.ones(len(w), bool), "Q1": qr == "Q1", "Q2": qr == "Q2",
            "paket TČ+PV, Q1": paket & (qr == "Q1"), "paket TČ+PV, Q1+Q2": paket}
    rows = []
    for lab, m in sets.items():
        a, u, r, q = summ(asis, m), summ(ure, m), summ(dim, m), summ(np.where(prag2, ure, dim), m)
        for k in a:
            if k == "stavb":
                continue
            rows.append({"skupina": lab, "stavb": a["stavb"], "kazalnik": k, "dejansko (P1/PA)": a[k], "URE-najprej (P3/PC)": u[k],
                         "razlika (cena zaklepanja)": a[k] - u[k], "OVE pravilno dimenzionirano (P1r/PA)": r[k],
                         "od tega dimenzioniranje (P1−P1r)": a[k] - r[k], "od tega ovoj (P1r−P3)": r[k] - u[k],
                         f"pravilo s pragom {PO['prag_ovoj_Qhnd']} (P3/PC nad, P1r/PA pod)": q[k], "pravilo: delež razlike [%]": 100 * (a[k] - q[k]) / (a[k] - u[k])})
    T8a = pd.DataFrame(rows)
    T8a.round(2).to_csv(TAB / "08_populacija.csv", sep=";", encoding="utf-8-sig", index=False)

    # ---------------- (b) tok novih vgradenj
    Y0, Y1 = PO["leta_novih"]
    YE = PO["horizont_do"]
    years = np.arange(E.EK["leto0"], YE + 1)
    per = {}
    for scen, sel_fn in (("nadaljevanje", lambda c: asis), ("URE-najprej od " + str(PO["ukrep_A_od"]), lambda c: ure_A if c >= PO["ukrep_A_od"] else asis),
                         ("OVE pravilno dimenzionirano od " + str(PO["ukrep_A_od"]), lambda c: dim_A if c >= PO["ukrep_A_od"] else asis),
                         (f"prag {PO['prag_ovoj_Qhnd']} + dimenzioniranje od {PO['ukrep_A_od']}", lambda c: pr_A if c >= PO["ukrep_A_od"] else asis)):
        acc = {k: np.zeros(len(years)) for k in ("zimski uvoz [GWh]", "sistemska konica [MW]", "CO2e [kt]", "spodbude [mio EUR]")}
        for c in range(Y0, Y1 + 1):
            sel = sel_fn(c)
            z, k_, cb, em = pick(zima, sel).mean(0), pick(kon, sel).mean(0), pick(co2b, sel).mean(0), pick(emb, sel).mean(0)
            sp = sum(sp_nom[p].to_numpy()[sel == p].sum() for p in np.unique(sel)) / len(sel)
            for i, Y in enumerate(years):
                t = Y - c + 1
                if 1 <= t <= N:
                    n = PO["novi_letno"]
                    acc["zimski uvoz [GWh]"][i] += n * z[t - 1] / 1e6
                    acc["sistemska konica [MW]"][i] += n * k_[t - 1] / 1e3
                    acc["CO2e [kt]"][i] += n * (cb[t - 1] * traj_year(Y) + em[t - 1]) / 1e6
                if t == 1:                                   # vgrajeni CO2 dogodka v letu y poti je v em[y] (dodan zgoraj pri t = y + 1)
                    acc["spodbude [mio EUR]"][i] += PO["novi_letno"] * sp / 1e6
        per[scen] = pd.DataFrame(acc, index=years)
    T8b = pd.concat(per, axis=1)
    T8b.index.name = "leto"
    T8b.round(2).to_csv(TAB / "08_tok_novih.csv", sep=";", encoding="utf-8-sig")
    sc = list(per)
    cum = pd.DataFrame({s: {"zimski uvoz 2026–" + str(YE) + " [GWh]": per[s]["zimski uvoz [GWh]"].sum(),
                            "sistemska konica " + str(YE) + " [MW]": per[s]["sistemska konica [MW]"].iloc[-1],
                            "sistemska konica " + str(Y1) + " [MW]": per[s].loc[Y1, "sistemska konica [MW]"],
                            "zimski uvoz " + str(Y1) + " [GWh/a]": per[s].loc[Y1, "zimski uvoz [GWh]"],
                            "CO2e 2026–" + str(YE) + " [kt]": per[s]["CO2e [kt]"].sum(),
                            "spodbude 2026–" + str(Y1) + " [mio EUR]": per[s]["spodbude [mio EUR]"].sum()} for s in sc})
    cum["razlika"] = cum[sc[0]] - cum[sc[1]]
    cum["od tega dimenzioniranje"] = cum[sc[0]] - cum[sc[2]]
    cum["od tega ovoj"] = cum[sc[2]] - cum[sc[1]]
    cum[f"razlika prag {PO['prag_ovoj_Qhnd']}"] = cum[sc[0]] - cum[sc[3]]
    cum[f"prag {PO['prag_ovoj_Qhnd']}: delež razlike [%]"] = 100 * cum[f"razlika prag {PO['prag_ovoj_Qhnd']}"] / cum["razlika"]
    cum.index.name = "kazalnik"
    cum.round(2).to_csv(TAB / "08_tok_novih_povzetek.csv", sep=";", encoding="utf-8-sig")

    # slika
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for lang, sfx in (("en", ""), ("sl", "_sl")):
        fig, axs = plt.subplots(1, 3, figsize=(10, 3.2))
        lab = {"en": ["Continuation", f"Envelope-first from {PO['ukrep_A_od']}", "Winter grid import [GWh/a]", "Contribution to system peak [MW]", "CO$_2$e [kt/a]"],
               "sl": ["Nadaljevanje", f"URE-najprej od {PO['ukrep_A_od']}", "Zimski uvoz [GWh/a]", "Prispevek h sistemski konici [MW]", "CO$_2$e [kt/a]"]}[lang]
        for ax, k, t in zip(axs, ["zimski uvoz [GWh]", "sistemska konica [MW]", "CO2e [kt]"], lab[2:]):
            ax.plot(years, per[sc[0]][k], color="#B2182B", lw=2, label=lab[0])
            ax.plot(years, per[sc[1]][k], color="#1B7837", lw=2, label=lab[1])
            ax.set_title(t, fontsize=9)
            ax.grid(alpha=0.25)
        axs[0].legend(fontsize=8, frameon=False)
        fig.tight_layout()
        for ext in ("png", "svg"):
            fig.savefig(SLIKE / f"08_tok_novih{sfx}.{ext}", dpi=300)
        plt.close(fig)

    md = [f"# 08 — Populacija ({time.strftime('%d. %m. %Y %H:%M')})", "",
          "## (a) Obstoječi fond: dejansko stanje (P1 s PV / PA brez PV) proti URE-najprej (P3 / PC)", "",
          T8a.round(1).to_string(index=False), "",
          f"## (b) Tok novih vgradenj: {PO['novi_letno']} hiš/leto (Q1+Q2 brez ovoja), kohorte {Y0}–{Y1}; ukrep A od {PO['ukrep_A_od']} "
          f"(prag Qh,nd > {PO['ukrep_A_prag_Qhnd']}; delež fonda nad pragom {100 * prag.mean():.0f} %; nad {PO['prag_ovoj_Qhnd']}: {100 * prag2.mean():.0f} %)", "",
          cum.round(1).to_string(), "", "Letno (izsek):", "", T8b.loc[[2026, 2028, 2030, 2035, 2040, 2045, 2050]].round(1).to_string(), "",
          f"Trajanje {time.time() - t0:.0f} s."]
    (HERE / "08_povzetek.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
