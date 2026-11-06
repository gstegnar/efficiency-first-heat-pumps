"""Korak 07 — prelomna točka Qh,nd, tornado in Monte Carlo (P3 »URE-najprej« proti P1 »TČ + PV, nikoli ovoj«).

A) Prelomna točka: sintetične reprezentativne hiše A1–A4 × Qh,nd 40–220 kWh/m²a (03 z --fond, pripona _Q07), za vsako
   razlika P3 − P1 za posameznika (R1, R2, R3; NPV stroškov, diskont zasebni), državo (javni NPV) in okolje (t CO2e, 30 let).
   Prelomna točka = Qh,nd, kjer razlika preide iz + (P1 cenejši) v − (P3 cenejši); linearna interpolacija.
B) Fond: P3 − P1 po razredih Qh,nd stavb (dejanska heterogenost, korak 04/06).
C) Tornado (reprezentativna hiša A3; posameznik R2, R3; država): nizka/visoka vrednost vsakega parametra.
D) Monte Carlo (trikotne porazdelitve, n iz parametri.yaml): verjetnost P3 < P1 po arhetipu, perspektivi in režimu.
Izhodi: tabele/07_*.csv, slike/07_prelomna_tocka(.png/.svg, _sl), slike/07_tornado(.png, _sl), 07_povzetek.md
"""
from __future__ import annotations

import copy
import importlib.util
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TAB, CACHE, SLIKE = HERE / "tabele", HERE / "cache", HERE / "slike"
TAG = "_Q07"
REZ = ["R1", "R2", "R3"]
_M = {}


def mods():
    """Load 05 (which loads 04 and 06) once per process and unify all parameter references into one PAR."""
    if "DR" in _M:
        return _M["DR"]
    spec = importlib.util.spec_from_file_location("dr", HERE / "05_drzava.py")
    DR = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(DR)
    E, O = DR.EKM, DR.OKM
    O.EKM, O.PAR = E, E.PAR
    O.OK, O.EK, O.VG = E.PAR["okolje"], E.PAR["ekonomika"], E.PAR["okolje"]["vgrajeno"]
    DR.PAR, DR.EK, DR.OK, DR.DR = E.PAR, E.PAR["ekonomika"], E.PAR["okolje"], E.PAR["drzava"]
    _M["DR"], _M["orig"], _M["poti"] = DR, copy.deepcopy(E.PAR), copy.deepcopy(E.POTI)
    return DR


def _restore(dst, src):
    for k, v in src.items():
        if isinstance(v, dict):
            _restore(dst[k], v)
        else:
            dst[k] = copy.deepcopy(v)


def apply(ch: dict):
    """Apply parameter changes in place (keeps object identity so module-level aliases stay valid)."""
    DR = mods()
    E = DR.EKM
    P = E.PAR
    _restore(P, _M["orig"])
    for p, d in _M["poti"].items():
        E.POTI[p]["seg"], E.POTI[p]["ev"] = list(d["seg"]), list(d["ev"])
    ek, ok = P["ekonomika"], P["okolje"]
    if "ovoj_strosek_faktor" in ch:
        for k in ek["ovoj_EUR_enota"]:
            ek["ovoj_EUR_enota"][k] *= ch["ovoj_strosek_faktor"]
            ek["ovoj_EUR_m2_ogrevane"][k] *= ch["ovoj_strosek_faktor"]
        ek["MVHR_EUR"] *= ch["ovoj_strosek_faktor"]
    if "TC_strosek_faktor" in ch:
        ek["TC_a_EUR"] *= ch["TC_strosek_faktor"]
        ek["TC_b_EUR_kW"] *= ch["TC_strosek_faktor"]
    if "PV_strosek_faktor" in ch:
        ek["PV_a_EUR"] *= ch["PV_strosek_faktor"]
        ek["PV_b_EUR_kWp"] *= ch["PV_strosek_faktor"]
    if "PV_vzdrzevanje_delez" in ch:
        ek["PV_vzdrzevanje_delez"] = ch["PV_vzdrzevanje_delez"]
    if "spodbude_faktor" in ch:
        for k in ek["spodbude"]:
            ek["spodbude"][k] *= ch["spodbude_faktor"]
    if "TC_let" in ch:
        L = int(round(ch["TC_let"]))
        ek["TC_let"] = L
        for p in E.POTI:
            E.POTI[p]["seg"] = [(L if y == 18 else y, v) for y, v in E.POTI[p]["seg"]]
            E.POTI[p]["ev"] = [(L if (y == 18 and it == "TCz") else y, it, a) for y, it, a in E.POTI[p]["ev"]]
    if "cena_CO2_faktor" in ch:
        for k in ok["cena_CO2_EUR_t"]:
            ok["cena_CO2_EUR_t"][k] *= ch["cena_CO2_faktor"]
    if "ojacitev_EUR_kW" in ch:
        P["drzava"]["ojacitev_EUR_kW"] = ch["ojacitev_EUR_kW"]
    if "EF_2050" in ch:
        ok["EF_trajektorija"][2050] = ch["EF_2050"]
        ok["EF_trajektorija"][2040] = (ok["EF_trajektorija"][2030] + ch["EF_2050"]) / 2
    if "vgrajeno_faktor" in ch:
        for k in ok["vgrajeno"]:
            ok["vgrajeno"][k] *= ch["vgrajeno_faktor"]


def evaluate(w, ch: dict) -> pd.DataFrame:
    """P3 − P1 (and P4 − P1) per unit for individual (R1–R3), state and environment under parameter changes ch."""
    DR = mods()
    apply(ch)
    E = DR.EKM
    r = ch.get("diskont_zasebni", E.EK["diskont_zasebni"])
    f = ch.get("cena_energije_faktor", 1.0)
    out = pd.DataFrame(index=w.index)
    for rez in REZ:
        n = E.npv_paths(w, rez, r, lambda t: f, E.EK["ovoj_osnova"])
        out[f"posameznik_{rez}"] = n["P3"] - n["P1"]
        out[f"posameznik_{rez}_P4"] = n["P4"] - n["P1"]
    st = DR.drzava(w)
    out["drzava"] = st["P3"]["javni_NPV"] - st["P1"]["javni_NPV"]
    c = DR.OKM.co2_paths(w)
    out["okolje_t"] = c[("P3", "skupaj_t")] - c[("P1", "skupaj_t")]
    apply({})
    return out


def _mc_worker(args):
    i, ch = args
    DR = mods()
    w = DR.EKM.load_units("repr")
    e = evaluate(w, ch)
    e["vzorec"] = i
    e["arhetip"] = w["arhetip"].values
    return e.reset_index()


def crossing(q, d):
    """Qh,nd where d changes sign from + to − (first such crossing); None if none."""
    q, d = np.asarray(q, float), np.asarray(d, float)
    for k in range(len(q) - 1):
        if d[k] > 0 >= d[k + 1]:
            return q[k] + (q[k + 1] - q[k]) * d[k] / (d[k] - d[k + 1])
    return None


def fmt_cross(q, d):
    x = crossing(q, d)
    if x is not None:
        return f"{x:.0f}"
    return f"vedno P3 (< {q.min():.0f})" if np.all(d <= 0) else (f"vedno P1 (> {q.max():.0f})" if np.all(d > 0) else "ni enoznačno")


def main():
    t0 = time.time()
    DR = mods()
    E = DR.EKM
    PAR = E.PAR
    OB = PAR["obcutljivost"]
    SLIKE.mkdir(exist_ok=True)
    md = [f"# 07 — Prelomna točka, tornado, Monte Carlo ({time.strftime('%d. %m. %Y %H:%M')})", "",
          "Razlika = P3 (URE-najprej) − P1 (TČ + PV, nikoli ovoj). Posameznik in država: NPV stroškov [EUR] (negativno = P3 cenejši); "
          "okolje: t CO2e v 30 letih (negativno = P3 manj emisij).", ""]

    # ---------------- A) sintetična mreža Qh,nd
    fond = pd.read_parquet(CACHE / "01_fond.parquet")
    rows = []
    for a, obd in (("A1", 1), ("A2", 2), ("A3", 3), ("A4", 4)):
        g = fond[fond["arhetip"] == a]
        A = float(g.loc[g["A_vir"] != "imputirano", "A_ogr"].median())
        kW = float(g.loc[~g["tc_kW_imputirano"].astype(bool), "tc_kW"].median())
        kwp = float(g.loc[g["paket_osnovni"].astype(bool), "PV_kWp"].median())
        for q in OB["Qhnd_mreza"]:
            rows.append({"ID_Stavbe": f"SIN_{a}_{q}", "Qr": "Q1" if q > 140 else ("Q2" if q >= 90 else ("Q3" if q >= 60 else "Q4")),
                         "obdobje": obd, "arhetip": a, "stratum": f"{a}|sint", "A_ogr": A, "A_vir": "GURS", "tc_kW": kW,
                         "tc_kW_imputirano": False, "tc_kW_uporabljena": kW, "tip_tc": "tczv", "PV_kWp": kwp, "ima_PV": True,
                         "leto_TC": 2026, "leto_PV": 2026, "sekvenca": "hkrati", "paket_osnovni": True, "staro_ogrevanje": "ELKO",
                         "bivalentno": False, "ima_prezrac": False, "Qh_nd_ob_vgradnji": float(q), "Qh_nd_ob_vgradnji_vir": "EI",
                         "Qhnd_lastna_EI": float(q), "LETO_IZGRA": np.nan, "STEVILO_ET": np.nan, "ID_Regije": np.nan, "ID_Obcine": np.nan,
                         "E_el_realni_MWh": np.nan, "SCOP_real": np.nan})
    sin = pd.DataFrame(rows)
    sin.to_parquet(CACHE / "07_sinteticni.parquet", index=False)
    r = subprocess.run([sys.executable, str(HERE / "03_urni_model.py"), "--fond", str(CACHE / "07_sinteticni.parquet"), "--tag", TAG],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise SystemExit("03 (sintetični) ni uspel:\n" + r.stderr[-3000:])
    ws = E.load_units("fond", tag=TAG)
    ev = evaluate(ws, {})
    ev["arhetip"] = ws["arhetip"].values
    ev["Qhnd"] = [float(i.split("_")[-1]) for i in ev.index]
    ev = ev.sort_values(["arhetip", "Qhnd"])
    ev.round(1).to_csv(TAB / "07_krivulje.csv", sep=";", encoding="utf-8-sig")
    pt = []
    for a, g in ev.groupby("arhetip"):
        row = {"arhetip": a}
        for c in ["posameznik_R1", "posameznik_R2", "posameznik_R3", "drzava", "okolje_t"]:
            row[c] = fmt_cross(g["Qhnd"].to_numpy(), g[c].to_numpy())
        pt.append(row)
    pt = pd.DataFrame(pt)
    pt.to_csv(TAB / "07_prelomna_tocka.csv", sep=";", encoding="utf-8-sig", index=False)
    md += ["## A) Prelomna točka Qh,nd [kWh/m²a] (P3 postane cenejši/boljši od P1 nad to vrednostjo)", "", pt.to_string(index=False), "",
           "Krivulje (izsek, A3, kEUR oz. t):", "",
           (ev[ev["arhetip"] == "A3"].set_index("Qhnd")[["posameznik_R1", "posameznik_R2", "posameznik_R3", "drzava"]] / 1000).round(1)
           .join(ev[ev["arhetip"] == "A3"].set_index("Qhnd")["okolje_t"].round(1)).to_string(), ""]

    # slika: posameznik R1/R2/R3 + država (A3 polno, ostali arhetipi tanko)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"posameznik_R1": "#B2182B", "posameznik_R2": "#EF8A62", "posameznik_R3": "#2166AC", "drzava": "#1B7837"}
    lab = {"en": {"posameznik_R1": "Household, R1 annual net metering", "posameznik_R2": "Household, R2 hourly self-consumption",
                  "posameznik_R3": "Household, R3 = R2 + capacity-based grid tariff", "drzava": "Public (subsidies + grid + CO$_2$)",
                  "x": "Space heating need before renovation, $Q_{h,nd}$ [kWh/m²a]", "y": "NPV(P3 envelope-first) − NPV(P1 HP+PV) [kEUR]",
                  "y2": "Δ CO$_2$e over 30 years [t]", "env": "Environment (right axis)"},
           "sl": {"posameznik_R1": "Posameznik, R1 letni net metering", "posameznik_R2": "Posameznik, R2 urna samooskrba",
                  "posameznik_R3": "Posameznik, R3 = R2 + omrežnina z dogovorjeno močjo", "drzava": "Država (spodbude + omrežje + CO$_2$)",
                  "x": "Potrebna toplota pred prenovo, $Q_{h,nd}$ [kWh/m²a]", "y": "NPV(P3 URE-najprej) − NPV(P1 TČ+PV) [kEUR]",
                  "y2": "Δ CO$_2$e v 30 letih [t]", "env": "Okolje (desna os)"}}
    for lang, sfx in (("en", ""), ("sl", "_sl")):
        L = lab[lang]
        fig, ax = plt.subplots(figsize=(7.5, 4.6))
        for a, g in ev.groupby("arhetip"):
            main_ = a == "A3"
            for c in col:
                ax.plot(g["Qhnd"], g[c] / 1000, color=col[c], lw=2.2 if main_ else 0.7, alpha=1 if main_ else 0.35,
                        ls="-." if c == "drzava" else "-", label=L[c] if main_ else None)
        ax2 = ax.twinx()
        g3 = ev[ev["arhetip"] == "A3"]
        ax2.plot(g3["Qhnd"], g3["okolje_t"], color="0.35", lw=1.6, ls=":", label=L["env"])
        ax2.set_ylabel(L["y2"])
        # poravnaj ničli obeh osi (prelomna točka okolja = presečišče s črno črto)
        l1, h1 = ax.get_ylim()
        l2, h2 = ax2.get_ylim()
        fz = -l1 / (h1 - l1)
        span = max(-l2 / fz if l2 < 0 else 0, h2 / (1 - fz) if h2 > 0 else 0)
        ax2.set_ylim(-fz * span, (1 - fz) * span)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xlabel(L["x"])
        ax.set_ylabel(L["y"])
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=7.5, loc="lower left", frameon=False)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        for ext in ("png", "svg"):
            fig.savefig(SLIKE / f"07_prelomna_tocka{sfx}.{ext}", dpi=300)
        plt.close(fig)

    # ---------------- B) fond po razredih Qh,nd
    npv = pd.read_parquet(CACHE / "04_npv_fond.parquet")
    qh = pd.read_parquet(CACHE / "03_fond_letno.parquet", columns=["ID_Stavbe", "qh"]).drop_duplicates("ID_Stavbe")
    npv = npv.merge(qh, on="ID_Stavbe", how="left")
    npv["razred_Qhnd"] = pd.cut(npv["qh"], [0, 60, 80, 100, 120, 140, 160, 180, 200, 400])
    npv["d"] = npv["P3"] - npv["P1"]
    fb = npv[npv["rezim"].isin(REZ)].groupby(["razred_Qhnd", "rezim"]).agg(
        stavb=("d", "size"), mediana_kEUR=("d", lambda x: x.median() / 1000), delez_P3_cenejsi=("d", lambda x: 100 * (x < 0).mean())).reset_index()
    c6 = pd.read_parquet(CACHE / "06_co2_poti_fond.parquet").merge(qh, on="ID_Stavbe")
    c6["razred_Qhnd"] = pd.cut(c6["qh"], [0, 60, 80, 100, 120, 140, 160, 180, 200, 400])
    c6["d"] = c6["P3|skupaj_t"] - c6["P1|skupaj_t"]
    fe = c6.groupby("razred_Qhnd").agg(stavb=("d", "size"), mediana_t=("d", "median"), delez_P3_manj=("d", lambda x: 100 * (x < 0).mean())).reset_index()
    fb.round(1).to_csv(TAB / "07_fond_po_Qhnd.csv", sep=";", encoding="utf-8-sig", index=False)
    fe.round(1).to_csv(TAB / "07_fond_po_Qhnd_okolje.csv", sep=";", encoding="utf-8-sig", index=False)
    md += ["## B) Fond po razredih Qh,nd stavb (vse arhetipi skupaj)", "",
           fb.pivot_table(index="razred_Qhnd", columns="rezim", values=["mediana_kEUR", "delez_P3_cenejsi"]).round(1).to_string(), "",
           "Okolje:", "", fe.round(1).to_string(index=False), ""]

    # ---------------- C) tornado (A3)
    wr = E.load_units("repr")
    base = evaluate(wr, {})
    tor = []
    for par, (lo, hi) in OB["tornado"].items():
        for lab_, v in (("nizko", lo), ("visoko", hi)):
            e = evaluate(wr, {par: v})
            for c in ("posameznik_R2", "posameznik_R3", "drzava", "okolje_t"):
                tor.append({"parameter": par, "vrednost": lab_, "x": v, "kazalnik": c,
                            "A3": float(e.loc["REPR_A3", c]), "A4": float(e.loc["REPR_A4", c]),
                            "osnova_A3": float(base.loc["REPR_A3", c]), "osnova_A4": float(base.loc["REPR_A4", c])})
    tor = pd.DataFrame(tor)
    tor.round(0).to_csv(TAB / "07_tornado.csv", sep=";", encoding="utf-8-sig", index=False)
    t3 = tor[tor["kazalnik"] == "posameznik_R3"].pivot_table(index="parameter", columns="vrednost", values="A3")
    t3["razpon"] = (t3["visoko"] - t3["nizko"]).abs()
    t3 = t3.sort_values("razpon")
    md += ["## C) Tornado — A3, posameznik R3, P3 − P1 [kEUR] (osnova "
           f"{base.loc['REPR_A3', 'posameznik_R3'] / 1000:.1f})", "", (t3 / 1000).round(2).to_string(), ""]
    td = tor[tor["kazalnik"] == "drzava"].pivot_table(index="parameter", columns="vrednost", values="A3")
    md += [f"Država (A3, osnova {base.loc['REPR_A3', 'drzava'] / 1000:.1f} kEUR):", "", (td / 1000).round(2).to_string(), ""]
    for lang, sfx in (("en", ""), ("sl", "_sl")):
        fig, ax = plt.subplots(figsize=(6.5, 3.8))
        b0 = base.loc["REPR_A3", "posameznik_R3"] / 1000
        y = np.arange(len(t3))
        ax.barh(y, t3["nizko"] / 1000 - b0, left=b0, color="#67A9CF", label="low" if lang == "en" else "nizko")
        ax.barh(y, t3["visoko"] / 1000 - b0, left=b0, color="#EF8A62", label="high" if lang == "en" else "visoko")
        ax.set_yticks(y)
        ax.set_yticklabels(t3.index, fontsize=8)
        ax.axvline(0, color="k", lw=0.8)
        ax.axvline(b0, color="0.4", lw=0.8, ls="--")
        ax.set_xlabel("NPV(P3) − NPV(P1), A3, R3 [kEUR]" if lang == "en" else "NPV(P3) − NPV(P1), A3, R3 [kEUR]")
        ax.legend(fontsize=8, frameon=False)
        fig.tight_layout()
        for ext in ("png", "svg"):
            fig.savefig(SLIKE / f"07_tornado{sfx}.{ext}", dpi=300)
        plt.close(fig)

    # ---------------- D) Monte Carlo
    rng = np.random.default_rng(PAR["urni_model"]["seme"] + 7)
    samples = []
    for i in range(OB["mc_n"]):
        ch = {k: float(rng.triangular(*v)) for k, v in OB["mc"].items()}
        samples.append((i, ch))
    with Pool(PAR["optimizacija"]["delavcev"]) as pool:
        res = pool.map(_mc_worker, samples, chunksize=10)
    mc = pd.concat(res, ignore_index=True)
    pars = pd.DataFrame([dict(vzorec=i, **ch) for i, ch in samples])
    mc = mc.merge(pars, on="vzorec")
    mc.to_parquet(CACHE / "07_monte_carlo.parquet", index=False)
    cols = ["posameznik_R1", "posameznik_R2", "posameznik_R3", "drzava", "okolje_t"]
    pr = mc.groupby("arhetip")[cols].apply(lambda d: 100 * (d < 0).mean()).round(1)
    pr.columns = [f"P(P3 < P1) {c} [%]" for c in cols]
    q = mc.groupby("arhetip")[cols].quantile([0.05, 0.5, 0.95]).unstack()
    pr.to_csv(TAB / "07_monte_carlo.csv", sep=";", encoding="utf-8-sig")
    md += [f"## D) Monte Carlo (n = {OB['mc_n']}, trikotne porazdelitve iz parametri.yaml: obcutljivost.mc)", "", pr.to_string(), "",
           "P5 / mediana / P95 razlike (EUR oz. t):", "", q.round(0).to_string(), "", f"Trajanje {time.time() - t0:.0f} s."]
    (HERE / "07_povzetek.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
