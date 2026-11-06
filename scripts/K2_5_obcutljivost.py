"""Krog 2, točka 5 — občutljivost za vse tri perspektive na paketu TČ+PV (5.969 hiš): tornado, Monte Carlo, prelomna točka.

Kazalniki (na hišo, P3 − P1; negativno = »URE najprej« boljši): posameznik NPV R1 in R3 (04, diskont zasebni), javni proračun R3 in
družbeni NPV (K2_1, 3 %), CO2e 30 let (06). Poročano: mediana na hišo, delež hiš s P3 < P1, vsota — za paket, Q1 in Q2.

A) Ekonomski parametri (brez nove urne simulacije), nizko / visoko:
   diskont zasebni 2 / 7 %; »kot kalkulator« (brez diskonta, cene +2 %/a); cene energije ×0,8 / ×1,3; strošek ovoja ×0,85 / ×1,2;
   osnova stroška ovoja EUR/m² ogrevane površine (Eko sklad) namesto EUR/enoto elementa; strošek TČ ×0,8 / ×1,2 in 1.000 EUR/kW;
   strošek PV ×0,85 / ×1,15; življenjska doba TČ 15 / 20 let; spodbude ×0 / ×1,5; strošek zmogljivosti 0 / 113 EUR/kW-leto (CCGT);
   zimske cene ×1,5 / ×2,0 (družbena CBA); ojačitev omrežja 300 / 1.200 EUR/kW; cena CO2 ×0,5 / ×1,5; EF elektrike 2050 0,3 / 0,7;
   mejni EF 0,367 kg/kWh (plinska PPE: 0,202 kg/kWh goriva ÷ izkoristek 0,55; vse leto, s trajektorijo) — PREVERI; vgrajeni CO2 ×0,7 / ×1,3;
   diskont javni 2 / 5 %.
B) Urni parametri (03 na podvzorcu paketa, K2_PAR; primerjava z osnovnim tekom na istem podvzorcu _K25osn):
   COP A7/W35 4,0 / 5,0 (osnova 4,5); temperatura dovoda brez ovoja 50 / 60 °C (osnova 55; FSO 45 nespremenjeno); n50 pri FSO ≤ 3,0 (osnova 4,0).
C) Monte Carlo (trikotne porazdelitve, ekonomski parametri), n = MC_N: porazdelitev deleža hiš s P3 < P1 in mediane po perspektivi.
D) Prelomna točka Qh,nd: sintetične hiše A1–A4 × Qh,nd 40–220 (03 --tag _Q07) za vse tri perspektive; + prelom na dejanskem paketu.
E) Zaklepanje: P3 − Pdej (Pdej = (1 − F20)·P1 + F20·P2) pri F20 iz Coxa M1, CEM in hipotetično 50 %.
Izhodi: clanek1/tabele/12a_obcutljivost_tornado.csv, 12b_monte_carlo.csv, 12c_prelomna_tocka.csv, 12d_zaklepanje_F.csv,
        clanek1/slike/Fig6_break_even(_sl), Fig7_tornado_monte_carlo(_sl) (prejšnji → *_krog1)
"""
from __future__ import annotations

import copy
import importlib.util
import json
import shutil
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT_T, OUT_S, CACHE = HERE / "clanek1" / "tabele", HERE / "clanek1" / "slike", HERE / "cache"
MC_N = 200
EF_MEJNI = 0.202 / 0.55                       # kg CO2/kWh el: zemeljski plin 0,202 kg/kWh goriva, PPE izkoristek 0,55 (PREVERI)
KAZ = {"posameznik R1": "kEUR", "posameznik R3": "kEUR", "proračun R3": "kEUR", "družbeni NPV": "kEUR", "CO2e 30 let": "t"}
_M = {}


def mods():
    if "K1" in _M:
        return _M["K1"]
    s = importlib.util.spec_from_file_location("k21", HERE / "K2_1_drzava.py")
    K1 = importlib.util.module_from_spec(s)
    s.loader.exec_module(K1)
    DR, E, O = K1.DR, K1.E, K1.O
    O.EKM, O.PAR = E, E.PAR                   # ena kopija parametrov za vse module (kot 07.mods)
    O.OK, O.EK, O.VG = E.PAR["okolje"], E.PAR["ekonomika"], E.PAR["okolje"]["vgrajeno"]
    DR.PAR, DR.EK, DR.OK, DR.DR = E.PAR, E.PAR["ekonomika"], E.PAR["okolje"], E.PAR["drzava"]
    K1.PAR, K1.EK, K1.K2 = E.PAR, E.PAR["ekonomika"], E.PAR["drzava_k2"]
    _M.update(K1=K1, orig=copy.deepcopy(E.PAR), poti=copy.deepcopy(E.POTI), r_k1=K1.r, r_dr=DR.r)
    return K1


def _restore(dst, src):
    for k, v in src.items():
        if isinstance(v, dict):
            _restore(dst[k], v)
        else:
            dst[k] = copy.deepcopy(v)


def apply(ch: dict):
    K1 = mods()
    E, DR = K1.E, K1.DR
    P = E.PAR
    _restore(P, _M["orig"])
    for p, d in _M["poti"].items():
        E.POTI[p]["seg"], E.POTI[p]["ev"] = list(d["seg"]), list(d["ev"])
    K1.r, DR.r = _M["r_k1"], _M["r_dr"]
    ek, ok, k2 = P["ekonomika"], P["okolje"], P["drzava_k2"]
    k2["zimska_cena_faktor_scenarij"] = ch.get("zima_f", 1.0)     # osnova = urne cene 2025 (faktor 1)
    if "ovoj_strosek_faktor" in ch:
        for k in ek["ovoj_EUR_enota"]:
            ek["ovoj_EUR_enota"][k] *= ch["ovoj_strosek_faktor"]
            ek["ovoj_EUR_m2_ogrevane"][k] *= ch["ovoj_strosek_faktor"]
        ek["MVHR_EUR"] *= ch["ovoj_strosek_faktor"]
    if "ovoj_delez_debeline" in ch:
        ek["ovoj_delez_debeline"] = ch["ovoj_delez_debeline"]
    if ch.get("ovoj_osnova"):
        ek["ovoj_osnova"] = ch["ovoj_osnova"]
    if "TC_strosek_faktor" in ch:
        ek["TC_a_EUR"] *= ch["TC_strosek_faktor"]
        ek["TC_b_EUR_kW"] *= ch["TC_strosek_faktor"]
    if ch.get("TC_1000"):
        ek["TC_a_EUR"], ek["TC_b_EUR_kW"] = 0.0, 1000.0
    if "PV_strosek_faktor" in ch:
        ek["PV_a_EUR"] *= ch["PV_strosek_faktor"]
        ek["PV_b_EUR_kWp"] *= ch["PV_strosek_faktor"]
    if "spodbude_faktor" in ch:
        for k, v in ek["spodbude"].items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                ek["spodbude"][k] = v * ch["spodbude_faktor"]
    if "TC_let" in ch:
        L = int(round(ch["TC_let"]))
        ek["TC_let"] = L
        for p in E.POTI:
            E.POTI[p]["seg"] = [(L if y == 18 else y, v) for y, v in E.POTI[p]["seg"]]
            E.POTI[p]["ev"] = [(L if (y == 18 and it == "TCz") else y, it, a) for y, it, a in E.POTI[p]["ev"]]
    if "zmogljivost" in ch:
        k2["zmogljivost_EUR_kW_leto"] = ch["zmogljivost"]
    if "ojacitev_EUR_kW" in ch:
        P["drzava"]["ojacitev_EUR_kW"] = ch["ojacitev_EUR_kW"]
    if "cena_CO2_faktor" in ch:
        for k in ok["cena_CO2_EUR_t"]:
            ok["cena_CO2_EUR_t"][k] *= ch["cena_CO2_faktor"]
    if "EF_2050" in ch:
        ok["EF_trajektorija"][2050] = ch["EF_2050"]
        ok["EF_trajektorija"][2040] = (ok["EF_trajektorija"][2030] + ch["EF_2050"]) / 2
    if "vgrajeno_faktor" in ch:
        for k in ok["vgrajeno"]:
            ok["vgrajeno"][k] *= ch["vgrajeno_faktor"]
    if "diskont_javni" in ch:
        K1.r = DR.r = ch["diskont_javni"]


def evaluate(w0, ch: dict) -> pd.DataFrame:
    """Kazalniki P3 − P1 (in P1, P2, P3 za zaklepanje) na hišo pri spremembah ch."""
    K1 = mods()
    apply(ch)
    E, O = K1.E, K1.O
    w = w0
    if "spodbude_faktor" in ch or ch.get("EF_mejni"):
        w = w0.copy()
        if "spodbude_faktor" in ch:                                  # opazovane (dejanske) spodbude enako
            w["dej_TC_sub"], w["dej_PV_sub"] = w["dej_TC_sub"] * ch["spodbude_faktor"], w["dej_PV_sub"] * ch["spodbude_faktor"]
        if ch.get("EF_mejni"):
            for c in [c for c in w.columns if c.endswith("|uvoz_kWh")]:
                v = c.split("|")[0]
                w[f"{v}|CO2_el_urni_kg"] = w[c] * EF_MEJNI
                w[f"{v}|CO2_izogib_oddaja_kg"] = w[f"{v}|oddaja_kWh"] * EF_MEJNI
    r = ch.get("diskont_zasebni", E.EK["diskont_zasebni"])
    g = ch.get("rast_cen", 0.0)
    f = ch.get("cena_energije_faktor", 1.0)
    pf = lambda t: f * (1 + g) ** t                                  # noqa: E731
    out = {}
    for rez in ("R1", "R3"):
        n = E.npv_paths(w, rez, r, pf, E.EK["ovoj_osnova"])
        for p in ("P1", "P2", "P3", "P1r"):
            out[(f"posameznik {rez}", p)] = n[p].to_numpy()
    res = K1.racun(w)
    for p in ("P1", "P2", "P3", "P1r"):
        out[("proračun R3", p)] = res[p]["R3 proračun neto"].to_numpy()
        out[("družbeni NPV", p)] = res[p]["družbeni NPV, zima ×f"].to_numpy()
    c = O.co2_paths(w)
    for p in ("P1", "P2", "P3", "P1r"):
        out[("CO2e 30 let", p)] = c[(p, "skupaj_t")].to_numpy()
    apply({})
    df = pd.DataFrame(out, index=w.index)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def povzetek(ev: pd.DataFrame, VZ: dict, F=None) -> list:
    """Vrstice za P3 − P1 in P3 − P1r (krog 3: »OVE najprej, pravilno dimenzionirano«); pri F (zaklepanje) P3 − Pdej."""
    rows = []
    refs = [("P3 − P1", None), ("P3 − P1r", "P1r")] if F is None else [("P3 − Pdej", None)]
    for prim, ref in refs:
        for k, en in KAZ.items():
            sc = 1000 if en == "kEUR" else 1
            if ref is not None:
                b = ev[(k, ref)]
            else:
                b = ev[(k, "P1")] if F is None else (1 - F) * ev[(k, "P1")] + F * ev[(k, "P2")]
            d = (ev[(k, "P3")] - b).to_numpy()
            for vz, m in VZ.items():
                m_ = m.reindex(ev.index).fillna(False).to_numpy()
                rows.append({"primerjava": prim, "kazalnik": k, "vzorec": vz, "mediana": np.median(d[m_]) / sc, "delež P3 boljši [%]": 100 * np.mean(d[m_] < 0),
                             "vsota": d[m_].sum() / (1e6 if en == "kEUR" else 1e3), "enota": en, "enota vsote": "mio EUR" if en == "kEUR" else "kt"})
    return rows


def vzorci(idx):
    meta = pd.read_parquet(CACHE / "01_fond.parquet", columns=["ID_Stavbe", "Qr", "paket_osnovni", "arhetip"]).set_index("ID_Stavbe").reindex(idx)
    meta["Qr"] = pd.Series(np.where(meta["arhetip"].isin(["A1", "A2", "A3"]), "Q1", "Q2"), index=meta.index)                                            # krog 3 (C1): razred po obdobju
    pk = meta["paket_osnovni"].astype(bool)
    return {"paket": pk, "paket pred 1981": pk & (meta["Qr"] == "Q1"), "paket 1981–2002": pk & (meta["Qr"] == "Q2")}, meta


EKON = [  # (parameter, oznaka, nizko, visoko) — ch slovarji
    ("diskont zasebni 2 / 7 %", {"diskont_zasebni": 0.02}, {"diskont_zasebni": 0.07}),
    ("kot kalkulator: brez diskonta, cene +2 %/a", {"diskont_zasebni": 0.0, "rast_cen": 0.02}, None),
    ("cene energije ×0,8 / ×1,3", {"cena_energije_faktor": 0.8}, {"cena_energije_faktor": 1.3}),
    ("strošek ovoja ×0,85 / ×1,2", {"ovoj_strosek_faktor": 0.85}, {"ovoj_strosek_faktor": 1.2}),
    ("osnova ovoja EUR/m² ogrevane (Eko)", {"ovoj_osnova": "ogrevana"}, None),
    ("strošek ovoja sorazmeren debelini (A4) α 0,5 / 1,0", {"ovoj_delez_debeline": 0.5}, {"ovoj_delez_debeline": 1.0}),
    ("strošek TČ ×0,8 / ×1,2", {"TC_strosek_faktor": 0.8}, {"TC_strosek_faktor": 1.2}),
    ("TČ 1.000 EUR/kW (kalkulator)", {"TC_1000": True}, None),
    ("strošek PV ×0,85 / ×1,15", {"PV_strosek_faktor": 0.85}, {"PV_strosek_faktor": 1.15}),
    ("življenjska doba TČ 15 / 20 let", {"TC_let": 15}, {"TC_let": 20}),
    ("spodbude ×0 / ×1,5", {"spodbude_faktor": 0.0}, {"spodbude_faktor": 1.5}),
    ("zmogljivost 0 / 113 EUR/kW-leto", {"zmogljivost": 0.0}, {"zmogljivost": 113.0}),
    ("zimske cene ×1,5 / ×2,0", {"zima_f": 1.5}, {"zima_f": 2.0}),
    ("ojačitev omrežja 300 / 1.200 EUR/kW", {"ojacitev_EUR_kW": 300.0}, {"ojacitev_EUR_kW": 1200.0}),
    ("cena CO2 ×0,5 / ×1,5", {"cena_CO2_faktor": 0.5}, {"cena_CO2_faktor": 1.5}),
    ("EF elektrike 2050: 0,3 / 0,7", {"EF_2050": 0.3}, {"EF_2050": 0.7}),
    ("mejni EF 0,367 kg/kWh (plinska PPE)", {"EF_mejni": True}, None),
    ("vgrajeni CO2 ×0,7 / ×1,3", {"vgrajeno_faktor": 0.7}, {"vgrajeno_faktor": 1.3}),
    ("diskont javni 2 / 5 %", {"diskont_javni": 0.02}, {"diskont_javni": 0.05}),
]
URNI = [("COP A7/W35 4,0 / 5,0", "_K25cop40", "_K25cop50"), ("temperatura dovoda brez ovoja 50 / 60 °C", "_K25T50", "_K25T60"),
        ("n50 pri ciljni prenovi ≤ 3,0 (osnova 4,0)", "_K25n50", None)]
MC = {"diskont_zasebni": (0.02, 0.04, 0.07), "cena_energije_faktor": (0.8, 1.0, 1.3), "ovoj_strosek_faktor": (0.8, 1.0, 1.35),
      "TC_strosek_faktor": (0.7, 1.0, 1.3), "PV_strosek_faktor": (0.8, 1.0, 1.2), "TC_let": (15, 18, 20), "zmogljivost": (0.0, 62.0, 113.0),
      "zima_f": (1.0, 1.0, 2.0), "ojacitev_EUR_kW": (300.0, 670.0, 1200.0), "cena_CO2_faktor": (0.5, 1.0, 1.5), "EF_2050": (0.3, 0.5, 0.7),
      "vgrajeno_faktor": (0.7, 1.0, 1.3)}


def _w_paket():
    K1 = mods()
    w = K1.E.load_units("fond")
    VZ, _ = vzorci(w.index)
    return w[VZ["paket"].to_numpy()]


def _mc_worker(args):
    i, ch = args
    if "w" not in _M:
        _M["w"] = _w_paket()
    w = _M["w"]
    VZ, _ = vzorci(w.index)
    rows = povzetek(evaluate(w, ch), VZ)
    return [dict(r, vzorec_mc=i, **{f"par {k}": v for k, v in ch.items()}) for r in rows]


def tornado(w, VZ):
    base = pd.DataFrame(povzetek(evaluate(w, {}), VZ)).assign(parameter="osnova", smer="osnova")
    rows = [base]
    for lab, lo, hi in EKON:
        for smer, ch in (("nizko", lo), ("visoko", hi)):
            if ch is None:
                continue
            rows.append(pd.DataFrame(povzetek(evaluate(w, ch), VZ)).assign(parameter=lab, smer=smer, vrsta="ekonomski"))
            print(f"  {lab} {smer}", flush=True)
    # urni parametri: razlika glede na osnovni tek na istem podvzorcu
    K1 = mods()
    wb = K1.E.load_units("fond", tag="_K25osn")
    VZb, _ = vzorci(wb.index)
    bb = pd.DataFrame(povzetek(evaluate(wb, {}), VZb)).set_index(["primerjava", "kazalnik", "vzorec"])
    b0 = base.set_index(["primerjava", "kazalnik", "vzorec"])
    for lab, lo, hi in URNI:
        for smer, tag in (("nizko", lo), ("visoko", hi)):
            if tag is None or not (CACHE / f"03_fond_letno{tag}.parquet").exists():
                continue
            wa = K1.E.load_units("fond", tag=tag)
            VZa, _ = vzorci(wa.index)
            a = pd.DataFrame(povzetek(evaluate(wa, {}), VZa)).set_index(["primerjava", "kazalnik", "vzorec"])
            # prenesi učinek na osnovo glavnega teka (osnovni podvzorec in glavni tek se ujemata; kontrola spodaj)
            adj = b0[["mediana", "delež P3 boljši [%]", "vsota"]] + (a[["mediana", "delež P3 boljši [%]", "vsota"]] - bb[["mediana", "delež P3 boljši [%]", "vsota"]])
            rows.append(adj.join(b0[["enota", "enota vsote"]]).reset_index().assign(parameter=lab, smer=smer, vrsta="urni"))
            print(f"  {lab} {smer}", flush=True)
    kontrola = (bb["mediana"] - b0["mediana"]).abs().max()
    T = pd.concat(rows, ignore_index=True)
    return T, kontrola


EN_PAR = {"diskont zasebni 2 / 7 %": "private discount rate 2 / 7 %", "kot kalkulator: brez diskonta, cene +2 %/a": "no discounting, prices +2 %/yr",
          "cene energije ×0,8 / ×1,3": "energy prices ×0.8 / ×1.3", "strošek ovoja ×0,85 / ×1,2": "envelope cost ×0.85 / ×1.2",
          "osnova ovoja EUR/m² ogrevane (Eko)": "envelope cost per m² heated area",
          "strošek ovoja sorazmeren debelini (A4) α 0,5 / 1,0": "envelope cost ∝ insulation thickness (1981–2002) α 0.5 / 1.0", "strošek TČ ×0,8 / ×1,2": "HP cost ×0.8 / ×1.2",
          "TČ 1.000 EUR/kW (kalkulator)": "HP cost 1,000 EUR/kW", "strošek PV ×0,85 / ×1,15": "PV cost ×0.85 / ×1.15",
          "življenjska doba TČ 15 / 20 let": "HP lifetime 15 / 20 yr", "spodbude ×0 / ×1,5": "subsidies ×0 / ×1.5",
          "zmogljivost 0 / 113 EUR/kW-leto": "capacity cost 0 / 113 EUR/kW-yr", "zimske cene ×1,5 / ×2,0": "winter prices ×1.5 / ×2.0",
          "ojačitev omrežja 300 / 1.200 EUR/kW": "grid reinforcement 300 / 1,200 EUR/kW", "cena CO2 ×0,5 / ×1,5": "CO2 price ×0.5 / ×1.5",
          "EF elektrike 2050: 0,3 / 0,7": "grid EF in 2050: 0.3 / 0.7", "mejni EF 0,367 kg/kWh (plinska PPE)": "marginal EF 0.367 kg/kWh (CCGT)",
          "vgrajeni CO2 ×0,7 / ×1,3": "embodied CO2 ×0.7 / ×1.3", "diskont javni 2 / 5 %": "social discount rate 2 / 5 %",
          "COP A7/W35 4,0 / 5,0": "COP A7/W35 4.0 / 5.0", "temperatura dovoda brez ovoja 50 / 60 °C": "flow temp. unrenovated 50 / 60 °C",
          "n50 pri ciljni prenovi ≤ 3,0 (osnova 4,0)": "n50 after target renovation ≤ 3.0 (base 4.0)"}


def slika_tornado(T, MCdf):
    T = T[T["primerjava"] == "P3 − P1"] if "primerjava" in T else T            # slika: P3 − P1 (P3 − P1r v tabeli)
    MCdf = MCdf[MCdf["primerjava"] == "P3 − P1"] if "primerjava" in MCdf else MCdf
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
    PAN = [("posameznik R3", ("Household R3", "Posameznik R3"), ("NPV P3 − P1 [€ thousand]", "NPV P3 − P1 [tisoč €]")),
           ("družbeni NPV", ("Social CBA", "Družbena CBA"), ("NPV P3 − P1 [€ thousand]", "NPV P3 − P1 [tisoč €]")),
           ("CO2e 30 let", ("Environment", "Okolje"), ("CO$_2$e P3 − P1, 30 y [t]", "CO$_2$e P3 − P1, 30 let [t]"))]
    for lang, sfx in (("en", ""), ("sl", "_sl")):
        L = 0 if lang == "en" else 1
        fig = plt.figure(figsize=(13.5, 8.4))
        gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.62], hspace=0.55, wspace=1.25)
        for j, (kaz, tt, xl) in enumerate(PAN):
            for i, vz in enumerate(("paket pred 1981", "paket 1981–2002")):
                ax = fig.add_subplot(gs[i, j])
                t = T[(T["kazalnik"] == kaz) & (T["vzorec"] == vz)]
                b = float(t[t["parameter"] == "osnova"]["mediana"].iloc[0])
                pv = t[t["parameter"] != "osnova"].pivot_table(index="parameter", columns="smer", values="mediana")
                for c in ("nizko", "visoko"):
                    if c not in pv:
                        pv[c] = np.nan
                pv["razpon"] = (pv[["nizko", "visoko"]].max(axis=1) - pv[["nizko", "visoko"]].min(axis=1)).fillna(0)
                pv["razpon"] = np.maximum(pv["razpon"], (pv[["nizko", "visoko"]] - b).abs().max(axis=1))
                pv = pv[pv["razpon"] > 1e-6 * max(1, abs(b))].sort_values("razpon").tail(9)
                y = np.arange(len(pv))
                ax.barh(y, pv["nizko"] - b, left=b, color="#2a78d6", height=0.62)
                ax.barh(y, pv["visoko"] - b, left=b, color="#eb6834", height=0.62)
                ax.axvline(0, color=INK, lw=0.9)
                ax.axvline(b, color=MUTED, lw=0.8, ls="--")
                ax.set_yticks(y)
                ax.set_yticklabels([EN_PAR.get(x, x) if lang == "en" else x for x in pv.index], fontsize=6.6)
                ax.set_xlabel(xl[L], fontsize=7.2)
                ax.tick_params(axis="x", labelsize=7)
                ax.set_title(f"{tt[L]} — {vz.replace('paket pred 1981', 'package pre-1981').replace('paket 1981–2002', 'package 1981–2002') if lang == 'en' else vz}", fontsize=8, loc="left")
                ax.grid(axis="x", color=GRID, lw=0.6)
                ax.grid(axis="y", visible=False)
                for s_ in ("top", "right"):
                    ax.spines[s_].set_visible(False)
        # Monte Carlo: delež hiš s P3 < P1 po perspektivi in razredu
        ax = fig.add_subplot(gs[2, :])
        kk = [("posameznik R1", ("household R1", "posameznik R1")), ("posameznik R3", ("household R3", "posameznik R3")),
              ("proračun R3", ("public budget R3", "proračun R3")), ("družbeni NPV", ("social NPV", "družbeni NPV")),
              ("CO2e 30 let", ("CO$_2$e", "CO$_2$e"))]
        pos, data, cols = [], [], []
        for i, (kaz, _) in enumerate(kk):
            for j, (vz, c) in enumerate((("paket pred 1981", "#eb6834"), ("paket 1981–2002", "#2a78d6"))):
                d = MCdf[(MCdf["kazalnik"] == kaz) & (MCdf["vzorec"] == vz)]["delež P3 boljši [%]"].to_numpy()
                pos.append(i * 3 + j)
                data.append(d)
                cols.append(c)
        bp = ax.boxplot(data, positions=pos, widths=0.75, whis=(5, 95), showfliers=False, patch_artist=True, medianprops={"color": INK})
        for patch, c in zip(bp["boxes"], cols):
            patch.set_facecolor(c)
            patch.set_alpha(0.75)
        ax.axhline(50, color=MUTED, lw=0.8, ls="--")
        ax.set_xticks([i * 3 + 0.5 for i in range(len(kk))])
        ax.set_xticklabels([x[1][L] for x in kk], fontsize=7.5)
        ax.set_ylim(0, 100)
        ax.set_ylabel(("Monte Carlo: share of houses\nwith P3 better [%]" if lang == "en" else "Monte Carlo: delež hiš,\nkjer je P3 boljši [%]"), fontsize=7.5)
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color="#eb6834", alpha=0.75, label="package pre-1981" if lang == "en" else "paket pred 1981"),
                           Patch(color="#2a78d6", alpha=0.75, label="package 1981–2002" if lang == "en" else "paket 1981–2002")], fontsize=7, loc="upper left",
                  frameon=False)
        ax.grid(axis="y", color=GRID, lw=0.6)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        fig.text(0.5, 0.005, ("Tornado: median P3 − P1, blue = low, orange = high value; dashed = base; left of 0 = envelope first better. "
                              f"Box: P5–P95 over {MC_N} draws.") if lang == "en" else
                 ("Tornado: mediana P3 − P1, modro = nizka, oranžno = visoka vrednost; črtkano = osnova; levo od 0 = URE najprej boljši. "
                  f"Škatla: P5–P95 v {MC_N} vzorcih."), ha="center", fontsize=7, color=MUTED)
        fig.subplots_adjust(left=0.17, right=0.985, top=0.96, bottom=0.07)
        for ext in ("png", "svg"):
            fn = OUT_S / f"Fig7_tornado_monte_carlo{sfx}.{ext}"
            bak = OUT_S / f"Fig7_tornado_monte_carlo_krog1{sfx}.{ext}"
            if fn.exists() and not bak.exists():
                shutil.copy2(fn, bak)
            fig.savefig(fn, dpi=300)
        plt.close(fig)


def prelomna(K1):
    """D) sintetične hiše A1–A4 × Qh,nd (03 --tag _Q07): P3 − P1 za vse perspektive in prelomna točka."""
    ws = K1.E.load_units("fond", tag="_Q07")
    ev = evaluate(ws, {})
    rows = []
    for i in ws.index:
        a, q = ws.loc[i, "arhetip"], float(i.split("_")[-1])
        rows.append({"arhetip": a, "Qhnd": q, **{k: float(ev.loc[i, (k, "P3")] - ev.loc[i, (k, "P1")]) / (1000 if KAZ[k] == "kEUR" else 1) for k in KAZ}})
    C = pd.DataFrame(rows).sort_values(["arhetip", "Qhnd"])

    def cross(q, d):
        q, d = np.asarray(q, float), np.asarray(d, float)
        for k in range(len(q) - 1):
            if d[k] > 0 >= d[k + 1]:
                return round(q[k] + (q[k + 1] - q[k]) * d[k] / (d[k] - d[k + 1]))
        return f"vedno P3 (≥ {q.min():.0f})" if np.all(d <= 0) else (f"vedno P1 (≤ {q.max():.0f})" if np.all(d > 0) else "ni enoznačno")
    PT = pd.DataFrame([{"arhetip": a, **{k: cross(g["Qhnd"], g[k]) for k in KAZ}} for a, g in C.groupby("arhetip")])
    return C, PT


def slika_prelom(C):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
    PAN = [(("posameznik R1", "posameznik R3"), ("Household: NPV P3 − P1 [€ thousand]", "Posameznik: NPV P3 − P1 [tisoč €]"),
            {"posameznik R1": ("#4a3aa7", ("R1 net metering", "R1 neto meritev")), "posameznik R3": ("#2a78d6", ("R3 5-block tariff", "R3 5 blokov"))}),
           (("proračun R3", "družbeni NPV"), ("State: present value P3 − P1 [€ thousand]", "Država: sedanja vrednost P3 − P1 [tisoč €]"),
            {"proračun R3": ("#eda100", ("public budget (R3)", "javni proračun (R3)")), "družbeni NPV": ("#008300", ("social CBA", "družbena CBA"))}),
           (("CO2e 30 let",), ("Environment: CO$_2$e P3 − P1 [t]", "Okolje: CO$_2$e P3 − P1 [t]"),
            {"CO2e 30 let": ("#52514e", ("CO$_2$e, 30 years", "CO$_2$e, 30 let"))})]
    ls = {"A1": ":", "A2": "-.", "A3": "-", "A4": "--"}
    for lang, sfx in (("en", ""), ("sl", "_sl")):
        L = 0 if lang == "en" else 1
        fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.6))
        for ax, (ks, yl, cc) in zip(axs, PAN):
            for k in ks:
                for a, g in C.groupby("arhetip"):
                    main_ = a in ("A3", "A4")
                    ax.plot(g["Qhnd"], g[k], color=cc[k][0], ls=ls[a], lw=2.0 if main_ else 1.0, alpha=1 if main_ else 0.55)
                g3 = C[C["arhetip"] == "A3"]
                ax.annotate(cc[k][1][L], (g3["Qhnd"].iloc[-1], g3[k].iloc[-1]), xytext=(4, 0), textcoords="offset points", fontsize=7,
                            color=cc[k][0], va="center")
            ax.axhline(0, color=INK, lw=0.9)
            ax.set_xlabel("$Q_{h,nd}$ before renovation [kWh/m²a]" if lang == "en" else "$Q_{h,nd}$ pred prenovo [kWh/m²a]", fontsize=8)
            ax.set_ylabel(yl[L], fontsize=8)
            ax.set_xlim(40, 255)
            ax.grid(color=GRID, lw=0.6)
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
        from matplotlib.lines import Line2D
        h = [Line2D([], [], color=MUTED, ls=ls[a], lw=1.6, label=lab[L]) for a, lab in
             (("A1", ("A1 ≤ 1945", "A1 ≤ 1945")), ("A2", ("A2 1946–1970", "A2 1946–1970")), ("A3", ("A3 1971–1980", "A3 1971–1980")),
              ("A4", ("A4 1981–2002", "A4 1981–2002")))]
        fig.legend(handles=h, loc="lower center", ncol=4, fontsize=7.5, frameon=False)
        fig.text(0.99, 0.02, "below 0 = envelope first (P3) better" if lang == "en" else "pod 0 = URE najprej (P3) boljši", ha="right", fontsize=7, color=MUTED)
        fig.tight_layout(rect=(0, 0.07, 1, 1))
        for ext in ("png", "svg"):
            fn = OUT_S / f"Fig6_break_even{sfx}.{ext}"
            bak = OUT_S / f"Fig6_break_even_krog1{sfx}.{ext}"
            if fn.exists() and not bak.exists():
                shutil.copy2(fn, bak)
            fig.savefig(fn, dpi=300)
        plt.close(fig)


def main():
    t0 = time.time()
    K1 = mods()
    w = _w_paket()
    VZ, meta = vzorci(w.index)
    print(f"paket: {len(w)} hiš", flush=True)
    # E) zaklepanje
    ver = json.loads((CACHE / "11_verjetnost_ovoja.json").read_text(encoding="utf-8"))
    ev0 = evaluate(w, {})
    q = meta["Qr"].to_numpy()
    Z = []
    for lab, src in (("F20 Cox M1", ver), ("F20 CEM", ver["obcutljivost_M4"]), ("F = 50 % (hipotetično)", None), ("F = 100 % (= P2)", None)):
        F = (np.where(q == "Q1", src["Q1"]["F20"], src["Q2"]["F20"]) if src is not None else np.full(len(q), 0.5 if "50" in lab else 1.0))
        Z += [dict(r, scenarij=lab) for r in povzetek(ev0, VZ, F=pd.Series(F, index=w.index).to_numpy())]
    Z = pd.DataFrame(Z)
    Z.round(3).to_csv(OUT_T / "12d_zaklepanje_F.csv", sep=";", encoding="utf-8-sig", index=False)
    # A + B) tornado
    T, kontrola = tornado(w, VZ)
    T.round(3).to_csv(OUT_T / "12a_obcutljivost_tornado.csv", sep=";", encoding="utf-8-sig", index=False)
    print(f"tornado {time.time() - t0:.0f} s; kontrola podvzorca (maks. razlika median osnove): {kontrola:.4f}", flush=True)
    # C) Monte Carlo
    rng = np.random.default_rng(20261013)
    samples = [(i, {k: float(rng.triangular(*v)) for k, v in MC.items()}) for i in range(MC_N)]
    with Pool(6) as pool:
        res = pool.map(_mc_worker, samples, chunksize=4)
    MCdf = pd.DataFrame([r for rr in res for r in rr])
    MCdf.round(4).to_csv(OUT_T / "12b_monte_carlo.csv", sep=";", encoding="utf-8-sig", index=False)
    print(f"Monte Carlo {time.time() - t0:.0f} s", flush=True)
    slika_tornado(T, MCdf)
    # D) prelomna točka
    C, PT = prelomna(K1)
    C.round(3).to_csv(OUT_T / "12c_prelomna_krivulje.csv", sep=";", encoding="utf-8-sig", index=False)
    PT.to_csv(OUT_T / "12c_prelomna_tocka.csv", sep=";", encoding="utf-8-sig", index=False)
    slika_prelom(C)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    for kaz in KAZ:
        t = T[(T["kazalnik"] == kaz) & (T["primerjava"] == "P3 − P1")].pivot_table(index=["parameter", "smer"], columns="vzorec", values=["mediana", "delež P3 boljši [%]"], sort=False)
        print(f"\n=== {kaz}\n", t.round(1).to_string())
    for kaz in ("posameznik R3", "družbeni NPV", "CO2e 30 let"):
        t = T[(T["kazalnik"] == kaz) & (T["primerjava"] == "P3 − P1r")].pivot_table(index=["parameter", "smer"], columns="vzorec",
                                                                                   values=["mediana", "delež P3 boljši [%]"], sort=False)
        print(f"\n=== P3 − P1r: {kaz}\n", t.round(1).to_string())
    print("\nMonte Carlo (delež hiš s P3 boljši: P5 / mediana / P95; P(mediana < 0)):")
    print(MCdf.groupby(["primerjava", "kazalnik", "vzorec"]).apply(lambda d: pd.Series({
        "P5": d["delež P3 boljši [%]"].quantile(0.05), "mediana": d["delež P3 boljši [%]"].median(), "P95": d["delež P3 boljši [%]"].quantile(0.95),
        "P(mediana P3−P1 < 0) [%]": 100 * (d["mediana"] < 0).mean()})).round(1).to_string())
    print("\nZaklepanje:\n", Z.pivot_table(index=["kazalnik", "vzorec"], columns="scenarij", values="delež P3 boljši [%]", sort=False).round(1).to_string())
    print("\nPrelomna točka (sintetične hiše):\n", PT.to_string(index=False))
    print(f"\nTrajanje {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
