"""Korak 03b — optimirano obratovanje (LP, 8.760 h) in mreža velikosti PV × baterija × hranilnik toplote po ovoju; EV vključen.

Model na hišo (enote kWh na uro):
  stavba kot hranilnik (1R1C ekvivalent): Eb(t) = (1 − a)·Eb(t−1) + qd + sb + bu − QH(t),  0 ≤ Eb ≤ C·ΔT_pas,  a = H/C
  TSV:      wd + sw + bw = QW(t)
  hranilnik: Es(t) = (1 − l)·Es(t−1) + qt − sb − sw,  0 ≤ Es ≤ V·1,163·ΔT
  TČ:       qd + qt + wd ≤ P_TČ·kapaciteta(t);  el = qd/COP_ogr(t) + (qt + wd)/COP_55(t)
  baterija: soc(t) = soc(t−1) + ηc·ch − dis/ηd;  ch, dis ≤ C_rate·E
  EV:       Σ polnjenje v oknu dneva = dnevni obrok;  polnjenje ≤ P_polnilnik v oknu, 0 izven
  elektrika: uvoz − oddaja = gospodinjstvo + prezr. + el_TČ + grelnik + ch − dis + EV − PV;  oddaja ≤ PV
  periodični robni pogoji (t = 0 ↔ t = 8759)
Cilji: O1 = min (c_uvoz·uvoz − c_oddaja·oddaja) (R2, fiksni ceni); O2 = min maks. urni uvoz (+ utež × O1).
Neoptimirana referenca za isto konfiguracijo = pravila iz koraka 03 (TČ sledi potrebi, baterija pohlepno, EV ob prihodu,
hranilnik se ne uporablja).

Izhodi: tabele/03b_mreza.csv (A3, A4 × ovoj × kWp × bat × hranilnik × cilj × EV), tabele/03b_prihranek_optimiranja.csv
        (V1–V7 × arhetip × cilj × EV: neoptimirano proti optimirano), tabele/03b_H_C.csv, 03b_povzetek.md
Zagon:  "C:/Anaconda/envs/new_env/python.exe" 03b_optimizacija.py [--test]
"""
from __future__ import annotations

import argparse
import itertools
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import linprog

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from urni_lib import (  # noqa: E402
    PAR, UM, HV, C, CACHE, TAB, ARH, OBD_ARH, CAL, ZIMA, PROF, T_EXT, PV1, PV_SPEC, TC, CAPREL, COP_W, COP_S,
    heat_pump, size_hp, BAT, battery, round_kwp,
)

OP = PAR["optimizacija"]
T = 8760
PROFILI = pd.read_parquet(CACHE / "03a_profili.parquet")
HH = PROFILI["H0"].to_numpy() * UM["gospodinjstvo"]["raba_kWh_a"]                  # kWh/h
EV_DAY = UM["ev"]["raba_kWh_a"] / 365
EV_P = UM["ev"]["moc_polnilnika_kW"]
EV_UNC = PROFILI["EV"].to_numpy() * UM["ev"]["raba_kWh_a"]                          # nepametno polnjenje (03)
HOUR = CAL.hour.to_numpy()
EV_WIN = (HOUR >= OP["ev_okno"]["od"]) | (HOUR < OP["ev_okno"]["do"])
# day index of the charging window: hours before 'do' belong to the previous evening's window
EV_DAYIDX = np.where(HOUR < OP["ev_okno"]["do"], np.arange(T) // 24 - 1, np.arange(T) // 24) % 365
STOP = {"BREZ": 0, "F": 1, "FS": 2, "FSO": 3, "GLOB": 3}


# ---------------------------------------------------------------- H in C po arhetipu × ovoju
def h_eff(a: str, ov: str) -> float:
    """Heat-loss coefficient [W/(m²K)] from the HECM hourly need: slope of Q_H vs (20 − T_ext) for T_ext < 0 °C."""
    q = PROF[(a, ov)]["QH"]
    m = T_EXT < 0
    x, y = 20 - T_EXT[m], q[m]
    return float(np.polyfit(x, y, 1)[0])


HC = pd.DataFrame([{"arhetip": a, "ovoj": ov, "H_W_m2K": h_eff(a, ov)} for a in ARH for ov in OP["mreza"]["ovoji"]])
HC["C_Wh_m2K"] = OP["C_eff_J_m2K"] / 3600
HC["tau_h"] = HC["C_Wh_m2K"] / HC["H_W_m2K"]
HC["hranilnik_stavbe_kWh_na_100m2"] = HC["C_Wh_m2K"] * OP["komfort_pas_K"] * 100 / 1000


# ---------------------------------------------------------------- hiša
def house(a: str, ov: str, A: float, kW_brez: float, faktor=1.0):
    """Hourly inputs [kWh] for archetype a, envelope ov, floor area A."""
    obd = OBD_ARH[a]
    fr = C.f_razvod(obd, STOP[ov])
    QH = PROF[(a, ov)]["QH"] * A / 1000 / fr
    QW = PROF[(a, ov)]["QW"] * A / 1000 * UM["tsv_izgube_faktor"]
    kW = kW_brez if ov == "BREZ" else float(size_hp((QH * 1000)[:, None], (QW * 1000)[:, None])[0])
    fans = np.zeros(T)
    if ov == "GLOB":
        fans[:] = UM["prezracevanje_SFP_kW_m3s"] * 0.5 * A * HV["visina_etaze_m"] / 3600.0
    H = float(HC.set_index(["arhetip", "ovoj"]).loc[(a, ov), "H_W_m2K"]) * A
    Cb = OP["C_eff_J_m2K"] / 3600 * A / 1000                                          # kWh/K
    return {"a": a, "ov": ov, "A": A, "QH": QH, "QW": QW, "kW": kW, "fans": fans, "H": H, "Cb": Cb, "hh": HH * faktor}


# ---------------------------------------------------------------- LP
VARS = ["qd", "qt", "wd", "bu", "bw", "sb", "sw", "Eb", "Es", "ch", "dis", "soc", "imp", "exp", "evc"]
IX = {v: i for i, v in enumerate(VARS)}


def solve(h: dict, kwp: float, bat: float, tank_L: float, ev: bool, cilj: str):
    nv = len(VARS) * T + 1                                         # + Pmax
    def col(v):
        return IX[v] * T + np.arange(T)
    pv = PV1 / 1000 * kwp
    cop_s, cop_w = COP_S[h["ov"]].astype(float), COP_W.astype(float)
    cap = CAPREL.astype(float) * h["kW"]
    a_b = (h["H"] / 1000) / h["Cb"]                                 # 1/h
    Eb_max = h["Cb"] * OP["komfort_pas_K"]
    Es_max = tank_L * 1.163 * OP["hranilnik"]["dT_uporabna_K"] / 1000
    l_s = OP["hranilnik"]["izgube_na_uro"]
    Ebat = bat * BAT["DoD"]
    Pbat = bat * BAT["C_rate"]
    ec = ed = BAT["izkoristek_krozni"] ** 0.5
    r = np.arange(T)
    prev = (r - 1) % T
    rows, cols, vals, beq = [], [], [], []
    k = 0

    def add(rr, cc, vv):
        rows.append(rr), cols.append(cc), vals.append(np.broadcast_to(vv, rr.shape).astype(float))

    # 1 building
    add(k + r, col("Eb"), 1.0); add(k + r, col("Eb")[prev], -(1 - a_b))
    add(k + r, col("qd"), -1.0); add(k + r, col("sb"), -1.0); add(k + r, col("bu"), -1.0)
    beq.append(-h["QH"]); k += T
    # 2 DHW
    add(k + r, col("wd"), 1.0); add(k + r, col("sw"), 1.0); add(k + r, col("bw"), 1.0)
    beq.append(h["QW"]); k += T
    # 3 tank
    add(k + r, col("Es"), 1.0); add(k + r, col("Es")[prev], -(1 - l_s))
    add(k + r, col("qt"), -1.0); add(k + r, col("sb"), 1.0); add(k + r, col("sw"), 1.0)
    beq.append(np.zeros(T)); k += T
    # 4 battery
    add(k + r, col("soc"), 1.0); add(k + r, col("soc")[prev], -1.0)
    add(k + r, col("ch"), -ec); add(k + r, col("dis"), 1 / ed)
    beq.append(np.zeros(T)); k += T
    # 5 electricity
    add(k + r, col("imp"), 1.0); add(k + r, col("exp"), -1.0)
    add(k + r, col("qd"), -1 / cop_s); add(k + r, col("qt"), -1 / cop_w); add(k + r, col("wd"), -1 / cop_w)
    add(k + r, col("bu"), -1 / TC["grelnik_eta"]); add(k + r, col("bw"), -1 / TC["grelnik_eta"])
    add(k + r, col("ch"), -1.0); add(k + r, col("dis"), 1.0); add(k + r, col("evc"), -1.0)
    beq.append(h["hh"] + h["fans"] - pv); k += T
    # 6 EV daily energy
    if ev:
        w = np.where(EV_WIN)[0]
        add(k + EV_DAYIDX[w], col("evc")[w], 1.0)
        beq.append(np.full(365, EV_DAY)); k += 365
    Aeq = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(k, nv))
    beq = np.concatenate(beq)
    # inequalities: HP capacity; imp ≤ Pmax
    ur, uc, uv, bub = [], [], [], []
    for v in ("qd", "qt", "wd"):
        ur.append(r); uc.append(col(v)); uv.append(np.ones(T))
    bub.append(cap)
    if cilj == "O2":
        ur += [T + r, T + r]; uc += [col("imp"), np.full(T, nv - 1)]; uv += [np.ones(T), -np.ones(T)]
        bub.append(np.zeros(T))
    nub = 2 * T if cilj == "O2" else T
    Aub = sp.csr_matrix((np.concatenate(uv), (np.concatenate(ur), np.concatenate(uc))), shape=(nub, nv))
    bub = np.concatenate(bub)
    # bounds
    lb, ub = np.zeros(nv), np.full(nv, np.inf)
    ub[col("Eb")] = Eb_max
    for v in ("qt", "sb", "sw", "Es"):
        ub[col(v)] = Es_max if v == "Es" else (np.inf if Es_max > 0 else 0)
    ub[col("soc")] = Ebat
    ub[col("ch")] = ub[col("dis")] = Pbat
    ub[col("exp")] = pv
    ub[col("evc")] = np.where(EV_WIN, EV_P, 0) if ev else 0
    # objective
    c = np.zeros(nv)
    w1 = 1.0 if cilj == "O1" else OP["O2_utez_energije"]
    c[col("imp")] = w1 * OP["cena_uvoz_EUR_kWh"]
    c[col("exp")] = -w1 * OP["cena_oddaja_EUR_kWh"]
    if cilj == "O2":
        c[-1] = 1.0
    res = linprog(c, A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=beq, bounds=np.c_[lb, ub], method="highs")
    if res.status != 0:
        return {"status": res.message}
    x = res.x
    g = {v: x[col(v)] for v in VARS}
    el_tc = g["qd"] / cop_s + (g["qt"] + g["wd"]) / cop_w
    el_bu = (g["bu"] + g["bw"]) / TC["grelnik_eta"]
    return metrics(h, kwp, bat, tank_L, ev, g["imp"], g["exp"], el_tc, el_bu, g["evc"], Tin=20 + g["Eb"] / h["Cb"])


def metrics(h, kwp, bat, tank_L, ev, imp, exp_, el_tc, el_bu, evc, Tin=None):
    pv = PV1 / 1000 * kwp
    el_epbd = el_tc + el_bu + h["fans"]
    load = h["hh"] + el_epbd + evc
    share = np.where(load > 0, el_epbd / np.maximum(load, 1e-12), 0)
    pe = (imp * share).sum() * PAR["zeb"]["f_P_tot"]["elektrika"] / h["A"]
    return {"uvoz_kWh": imp.sum(), "oddaja_kWh": exp_.sum(), "uvoz_zima_kWh": imp[ZIMA].sum(), "uvoz_konica_kW": imp.max(),
            "uvoz_top10_kW": np.sort(imp)[-10:].mean(), "el_TC_kWh": el_tc.sum(), "el_grelnik_kWh": el_bu.sum(),
            "el_EV_kWh": evc.sum(), "el_skupaj_kWh": load.sum(), "PV_kWh": pv.sum(),
            "samozadostnost": 1 - imp.sum() / max(load.sum(), 1e-9),
            "samoporaba": (pv.sum() - exp_.sum()) / pv.sum() if pv.sum() > 0 else np.nan,
            "racun_R2_EUR": imp.sum() * OP["cena_uvoz_EUR_kWh"] - exp_.sum() * OP["cena_oddaja_EUR_kWh"],
            "PE_kWh_m2": pe, "T_not_povp": float(np.mean(Tin)) if Tin is not None else 20.0,
            "T_not_maks": float(np.max(Tin)) if Tin is not None else 20.0, "status": "ok"}


def rule_based(h, kwp, bat, tank_L, ev):
    """Unoptimised reference: HP follows demand, greedy battery, EV on arrival, tank unused (as in step 03)."""
    el, bu, _ = heat_pump((h["QH"] * 1000)[:, None].astype(np.float32), (h["QW"] * 1000)[:, None].astype(np.float32),
                          np.array([h["kW"]]), h["ov"])
    el, bu = el[:, 0] / 1000, bu[:, 0] / 1000
    evc = EV_UNC if ev else np.zeros(T)
    load = h["hh"] + h["fans"] + el + bu + evc
    net = (load - PV1 / 1000 * kwp)[:, None].astype(np.float32) * 1000
    if bat:
        imp, exp_ = battery(net, bat)
    else:
        imp, exp_ = np.maximum(net, 0), np.maximum(-net, 0)
    return metrics(h, kwp, bat, tank_L, ev, imp[:, 0] / 1000, exp_[:, 0] / 1000, el, bu, evc)


def task(args):
    key, h, kwp, bat, tank, ev, cilj, nacin = args
    t = time.time()
    out = solve(h, kwp, bat, tank, ev, cilj) if nacin == "opt" else rule_based(h, kwp, bat, tank, ev)
    return {**key, "kWp": kwp, "baterija_kWh": bat, "hranilnik_L": tank, "EV": ev, "cilj": cilj if nacin == "opt" else "pravila",
            "nacin": nacin, "TC_kW": h["kW"], "A": h["A"], "s": round(time.time() - t, 1), **out}


# ---------------------------------------------------------------- reprezentativne hiše (kot v 03)
fond = pd.read_parquet(CACHE / "01_fond.parquet")
fond["A_ogr"] = fond["A_ogr"].astype(float)
if __import__("os").environ.get("GEOM", "G13") in ("G11", "G13"):   # krog 12/13: ogrevana površina kot v 03 (faktor kondicionirane površine izkaznic)
    fond["A_ogr"] = fond["A_ogr"] * fond["obdobje"].map(HV["povrsina_G11_faktor"]).fillna(1.0)
REP = {}
med_pv = fond[fond["paket_osnovni"].astype(bool)].groupby("arhetip")["PV_kWp"].median()
for a in ARH:
    g = fond[fond["arhetip"] == a]
    REP[a] = {"A": float(g.loc[g["A_vir"] != "imputirano", "A_ogr"].median()),
              "kW": float(g.loc[~g["tc_kW_imputirano"].astype(bool), "tc_kW"].median()), "kWp_V2": float(med_pv[a])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    HC.round(4).to_csv(TAB / "03b_H_C.csv", sep=";", encoding="utf-8-sig", index=False)
    H = {(a, ov): house(a, ov, REP[a]["A"], REP[a]["kW"]) for a in ARH for ov in OP["mreza"]["ovoji"]}
    if args.test:
        for cilj in ("O1", "O2"):
            r = task(({"arhetip": "A3", "ovoj": "FSO"}, H[("A3", "FSO")], 8, 10, 800, True, cilj, "opt"))
            print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()})
        print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in
               task(({"arhetip": "A3", "ovoj": "FSO"}, H[("A3", "FSO")], 8, 10, 800, True, "-", "pravila")).items()})
        return

    # (1) prihranek optimiranja za variante V1–V7 (reprezentativne hiše, vsi arhetipi)
    vt = []
    for a in ARH:
        kwp5 = {}
        for ov in ("FSO", "GLOB"):
            rb = rule_based(H[(a, ov)], 0, 0, 0, False)
            kwp5[ov] = float(round_kwp((rb["el_skupaj_kWh"]) / PV_SPEC))
        conf = {"V1": ("BREZ", 0, 0), "V2": ("BREZ", REP[a]["kWp_V2"], 0), "V3": ("BREZ", REP[a]["kWp_V2"], BAT["kWh"]),
                "V4": ("FSO", 0, 0), "V5": ("FSO", kwp5["FSO"], 0), "V6": ("FSO", kwp5["FSO"], BAT["kWh"]),
                "V7": ("GLOB", kwp5["GLOB"], BAT["kWh"])}
        for v, (ov, kwp, bat) in conf.items():
            for ev in (False, True):
                key = {"arhetip": a, "varianta": v, "ovoj": ov}
                vt.append((key, H[(a, ov)], kwp, bat, 0, ev, "-", "pravila"))
                for cilj in ("O1", "O2"):
                    vt.append((key, H[(a, ov)], kwp, bat, 0, ev, cilj, "opt"))
    # (2) mreža velikosti
    mg = OP["mreza"]
    gt = []
    for a, ov, kwp, bat, tank, ev in itertools.product(mg["arhetipi"], mg["ovoji"], mg["kWp"], mg["baterija_kWh"], mg["hranilnik_L"], (False, True)):
        key = {"arhetip": a, "ovoj": ov}
        gt.append((key, H[(a, ov)], kwp, bat, tank, ev, "-", "pravila"))
        for cilj in ("O1", "O2"):
            gt.append((key, H[(a, ov)], kwp, bat, tank, ev, cilj, "opt"))
    print(f"LP + pravila: variante {len(vt)}, mreža {len(gt)}")
    with Pool(OP["delavcev"]) as pool:
        rv = pool.map(task, vt, chunksize=2)
        pd.DataFrame(rv).to_csv(TAB / "03b_prihranek_optimiranja_surovo.csv", sep=";", encoding="utf-8-sig", index=False)
        print(f"variante: {time.time() - t0:.0f} s")
        rg = pool.map(task, gt, chunksize=4)
    pd.DataFrame(rg).to_csv(TAB / "03b_mreza.csv", sep=";", encoding="utf-8-sig", index=False)
    print(f"mreža: {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
