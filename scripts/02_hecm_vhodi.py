"""Korak 02 — HECM vhodi CLK-*, kalibracija na rEI, ovojna lestvica, os Qh,nd, izvoz urnih nizov.

Geometrija in izhodiščni U: TABULA SI (parametri.yaml: hecm_vhodi). Kalibracija: f_U (množitelj U neprenovljenih
neprozornih elementov) tako, da letna Qh,nd neprenovljene hiše zadene kalibrirano rEI (06_Qhnd_kalibrirano.csv).
HECM se kliče prek API (hecm.simulate.simulate_building) z lastnim predpomnilnikom (cache/hecm) — v GreenRenoV8
in v P:/.../.hecm_cache se nič ne zapiše. PV v vhodih = 0 (CLAUDE.md §2.2).

Izhodi: hecm_vhodi/CLK-*.txt, cache/02_urno_<arhetip>_<varianta>.parquet (Wh, °C, W/m²),
        tabele/02_variante.csv, 02_kalibracija.md
Zagon:  "C:/Anaconda/envs/new_env/python.exe" 02_hecm_vhodi.py
"""
from __future__ import annotations

import copy
import json
import os
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
def _k2_preglasitev(par: dict) -> dict:
    """Občutljivost (krog 2, K2_5): okoljska spremenljivka K2_PAR = JSON, ki se globoko združi v parametri.yaml (samo v tem procesu)."""
    import json
    import os

    def merge(a, b):
        for k, v in b.items():
            if isinstance(v, dict) and isinstance(a.get(k), dict):
                merge(a[k], v)
            else:
                a[k] = v
    if os.environ.get("K2_PAR"):
        merge(par, json.loads(os.environ["K2_PAR"]))
    return par


PAR = _k2_preglasitev(yaml.safe_load((HERE / "parametri.yaml").read_text(encoding="utf-8")))
TAG02 = os.environ.get("K2_TAG02", "")   # občutljivost K2_5: izhodi s pripono, osnovni ostanejo nedotaknjeni
HV = PAR["hecm_vhodi"]
if os.environ.get("GEOM", "G13") == "G11":                     # krog 11: geometrija ovoja iz projektov Eko sklada (parametri.yaml: geometrija_G11)
    for _a, _g in HV["geometrija_G11"].items():
        HV["arhetipi"][_a]["geometrija"] = _g
elif os.environ.get("GEOM", "G13") == "G13":                    # krog 13 (osnova od 22. 10. 2026): toplotni ovoj ogrevanega dela iz GURS (izgube); strošek ostane po G11 (04)
    for _a, _g in HV["geometrija_G13"].items():
        HV["arhetipi"][_a]["geometrija"] = _g
HECM = Path(PAR["poti"]["hecm"])
sys.path.insert(0, str(HECM))

from hecm.config import load_config  # noqa: E402
from hecm.io_txt import read_building, write_txt  # noqa: E402
from hecm.runner import _load_cached, _store_cached, thermal_signature  # noqa: E402

VHODI = HERE / "hecm_vhodi"
CACHE = HERE / "cache"
TAB = HERE / "tabele"
for p in (VHODI, CACHE / "hecm", TAB):
    p.mkdir(parents=True, exist_ok=True)

# element order in the HECM SFH template: GR, OP N/E/S/W, W N/E/S/W, OP HOR (roof)
OPQ = {"tla": [0], "stena": [1, 2, 3, 4], "streha": [9]}
WIN = [5, 6, 7, 8]


def cfg():
    return load_config("SI", overrides={"cache_dir": str(CACHE / "hecm"), "out_dir": str(CACHE / "hecm_out"), **HV["cfg_preglasitve"]})


def make_bui(arh: str, var: str, f_u: float, ukrepi: dict | None = None) -> dict:
    """HECM input dict for archetype `arh`; `ukrepi` = new U per element (+ n50, rekuperacija) from the ladder."""
    a = HV["arhetipi"][arh]
    g = HV["geometrija"][a["geometrija"]]
    ukrepi = ukrepi or {}
    b = copy.deepcopy(read_building(str(HECM / HV["predloga"])))
    area = np.zeros(10)
    area[OPQ["tla"]] = g["tla"]
    area[OPQ["stena"]] = g["stena"] / 4
    area[OPQ["streha"]] = g["streha"]
    area[WIN] = g["okna"] / 4
    u = np.zeros(10)
    for el, idx in OPQ.items():
        u[idx] = ukrepi.get(el, a["U"][el] * f_u)          # f_U only on unrenovated opaque elements
    u[WIN] = ukrepi.get("okna", a["U"]["okna"])
    b.update({
        "house_id": f"CLK-SI-SFH-{arh}-{var}",
        "a_use": g["A"], "slab_on_ground": g["tla"], "exposed_perimeter": round(4 * g["tla"] ** 0.5, 2),
        "building_height": HV["visina_etaze_m"],
        "surface_envelope": g["tla"] + g["stena"] + g["streha"] + g["okna"],
        "surface_envelope_model": g["tla"] + g["stena"] + g["streha"] + g["okna"],
        "area_elements": area, "transmittance_U_elements": u,
        "airtightness_n50": ukrepi.get("n50", min(a["n50"], ukrepi.get("n50_max", a["n50"]))),   # FSO: n50_max ob zamenjavi oken
        "heat_recovery_efficiency": 0.0, "Ventilation HR efficiency": 0.0,
        "ventilation_type": "natural", "Ventilation system": "natural",
        "Annual Energy Production": 0,
    })
    if ukrepi.get("rekuperacija"):
        hr = ukrepi["rekuperacija"]
        b.update({"heat_recovery_efficiency": hr, "Ventilation HR efficiency": hr,
                  "ventilation_type": "mechanical_with_HR", "Ventilation system": "mechanical_with_HR"})
    return b


def simulate(args):
    """Worker: (name, bui) -> (name, annual dict, path of hourly parquet). Uses the project-local cache."""
    name, bui = args
    from hecm.simulate import silence_progress_bars, simulate_building
    silence_progress_bars()
    c = cfg()
    write_txt(str(VHODI / f"{name}.txt"), bui)
    bui = read_building(str(VHODI / f"{name}.txt"))     # simulate exactly what is on disk
    sig = thermal_signature(bui, c)
    hit = _load_cached(c, sig)
    t0 = time.time()
    if hit is None:
        hourly, annual = simulate_building(bui, c)
        _store_cached(c, sig, hourly, annual)
    else:
        hourly, annual = hit
    q = hourly["Q_H"]
    a_use = annual["a_use_m2"]
    out = {"ime": name, "signature": sig, "s": round(time.time() - t0),
           "Qh_nd": annual["Q_H_kWh_m2a"], "Qw": annual["Q_W_kWh_m2a"], "Qc": annual["Q_C_kWh_m2a"],
           "a_use": a_use, "ach_skupaj": annual["ach_total_1_h"], "ach_infiltracija": annual["ach_infiltration_1_h"],
           "QH_maks_W_m2": q.max() / a_use, "QH_P999_W_m2": q.quantile(0.999) / a_use,
           "QH_Text_le_m10_W_m2": q[hourly["T_ext"] <= -10].mean() / a_use,
           "ur_ogrevanja": int((q > 0).sum()), "ur_Top_pod_19": int((hourly["T_op"] < 19).sum())}
    return out, hourly


def calibrate(arh):
    """Secant on f_U until |Qh_nd - target| <= tolerance. Returns iteration log."""
    k = HV["kalibracija"]
    tgt = HV["arhetipi"][arh]["cilj_Qhnd"]
    lo, hi = k["meje"]
    log, pts = [], []
    f = 1.0
    for it in range(k["max_iteracij"]):
        r, _ = simulate((f"CLK-SI-SFH-{arh}-kal{it}", make_bui(arh, f"kal{it}", f)))
        log.append({"arhetip": arh, "iteracija": it, "f_U": round(f, 4), "Qh_nd": round(r["Qh_nd"], 2), "cilj": tgt})
        pts.append((f, r["Qh_nd"]))
        if abs(r["Qh_nd"] - tgt) <= k["toleranca_kWh_m2a"]:
            break
        if len(pts) == 1:
            # first step: opaque share of losses unknown -> assume Qh ~ f_U
            f_new = f * tgt / r["Qh_nd"]
        else:
            (f0, q0), (f1, q1) = pts[-2], pts[-1]
            f_new = f1 + (tgt - q1) * (f1 - f0) / (q1 - q0) if q1 != q0 else f1
        f = float(np.clip(f_new, lo, hi))
    return log


def glob_ukrepi(g: float) -> dict:
    """GLOB package with opaque U tightened by g (windows, n50, heat recovery unchanged)."""
    uk = dict(HV["lestvica"]["GLOB"])
    for el in ("stena", "streha", "tla"):
        uk[el] = uk[el] * g
    return uk


def calibrate_glob(args):
    """Largest g in [lo, 1] with Qh_nd <= target (secant towards Qh_nd = target). Returns iteration log."""
    arh, f_u = args
    k = HV["glob_kalibracija"]
    tgt, (lo, hi) = k["cilj_Qhnd_max"], k["meje"]
    log, pts, g = [], [], hi
    for it in range(k["max_iteracij"]):
        r, _ = simulate((f"CLK-SI-SFH-{arh}-glob{it}", make_bui(arh, f"glob{it}", f_u, glob_ukrepi(g))))
        log.append({"arhetip": arh, "iteracija": it, "g": round(g, 4), "Qh_nd": round(r["Qh_nd"], 2), "cilj_max": tgt})
        pts.append((g, r["Qh_nd"]))
        if (it == 0 and r["Qh_nd"] <= tgt) or abs(r["Qh_nd"] - tgt) <= k["toleranca_kWh_m2a"]:
            break
        if len(pts) == 1:
            g_new = g * tgt / r["Qh_nd"]
        else:
            (g0, q0), (g1, q1) = pts[-2], pts[-1]
            g_new = g1 + (tgt - q1) * (g1 - g0) / (q1 - q0) if q1 != q0 else g1
        g = float(np.clip(g_new, lo, hi))
    return log


def main():
    t0 = time.time()
    arhs = list(HV["arhetipi"])
    with Pool(len(arhs)) as pool:
        logs = pool.map(calibrate, arhs)
    kal = pd.DataFrame([row for lg in logs for row in lg])
    fU = {a: float(g.iloc[(g["Qh_nd"] - g["cilj"]).abs().argmin()]["f_U"]) for a, g in kal.groupby("arhetip")}
    print("kalibracija:", fU, f"({time.time() - t0:.0f} s)")
    with Pool(len(arhs)) as pool:
        glogs = pool.map(calibrate_glob, [(a, fU[a]) for a in arhs])
    gkal = pd.DataFrame([row for lg in glogs for row in lg])
    # choose the loosest g that still meets Qh_nd <= target + tolerance
    tol = HV["glob_kalibracija"]["toleranca_kWh_m2a"]
    gG = {}
    for a, g in gkal.groupby("arhetip"):
        ok = g[g["Qh_nd"] <= g["cilj_max"] + tol]
        gG[a] = float(ok["g"].max()) if len(ok) else float(g.loc[g["Qh_nd"].idxmin(), "g"])
    print("GLOB g:", gG)

    tasks = []
    for a in arhs:
        tasks.append((f"CLK-SI-SFH-{a}-BREZ", make_bui(a, "BREZ", fU[a])))
        for var, uk in HV["lestvica"].items():
            if var == "GLOB":
                uk = glob_ukrepi(gG[a])
            tasks.append((f"CLK-SI-SFH-{a}-{var}", make_bui(a, var, fU[a], uk)))
        for m in HV["os_Qhnd_faktorji_fU"]:
            v = f"OS{m:g}".replace(".", "p")
            tasks.append((f"CLK-SI-SFH-{a}-{v}", make_bui(a, v, fU[a] * m)))
    with Pool(HV["delavcev"]) as pool:
        res = pool.map(simulate, tasks)

    rows = []
    for (r, hourly), (name, bui) in zip(res, tasks):
        arh, var = name.split("-")[3], name.split("-")[4]
        cols = ["T_ext", "T_op", "Q_H", "Q_C", "Q_W", "G_h"]
        h = hourly[cols].copy()
        h.attrs["a_use_m2"] = r["a_use"]
        h.to_parquet(CACHE / f"02_urno_{arh}_{var}{TAG02}.parquet")
        u = bui["transmittance_U_elements"]
        rows.append({"arhetip": arh, "varianta": var, "f_U": fU[arh] * (float(var[2:].replace("p", ".")) if var.startswith("OS") else 1.0),
                     "U_stena": round(u[1], 3), "U_streha": round(u[9], 3), "U_tla": round(u[0], 3), "U_okna": round(u[5], 2),
                     "n50": bui["airtightness_n50"], "rekuperacija": bui["heat_recovery_efficiency"],
                     **{k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items() if k not in ("ime",)}})
    var = pd.DataFrame(rows)
    var.to_csv(TAB / f"02_variante{TAG02}.csv", sep=";", encoding="utf-8-sig", index=False)
    if TAG02:
        return
    kal.to_csv(TAB / "02_kalibracija_iteracije.csv", sep=";", encoding="utf-8-sig", index=False)
    gkal.to_csv(TAB / "02_glob_iteracije.csv", sep=";", encoding="utf-8-sig", index=False)
    (CACHE / "02_kalibracija.json").write_text(json.dumps({"f_U": fU, "g_GLOB": gG}, indent=1), encoding="utf-8")

    # ---------------------------------------------------------------- 02_kalibracija.md
    md = ["# 02 — Kalibracija HECM vhodov CLK-* in ovojna lestvica", "",
          f"Izvedeno {time.strftime('%d. %m. %Y %H:%M')}, `02_hecm_vhodi.py`, parametri `parametri.yaml: hecm_vhodi`. "
          f"Trajanje {time.time() - t0:.0f} s.", "",
          "**Pristop.** Geometrija in izhodiščni U iz TABULA SI (GI ZRMK 2014): kompaktna 2-etažna hiša SI.N.SFH.03 za A1–A3, "
          "SI.N.SFH.04 za A4; U neprenovljenih elementov iz lista TABULA istega obdobja; n50 iz `izhodisca_ovoja_SI.xlsx`. "
          "Naravno prezračevanje (HECM tipologija: 0,5 h⁻¹) + infiltracija n50/20. Ena kalibrirana veličina: "
          "**f_U** = množitelj U neprenovljenih neprozornih elementov (stena, streha, tla), cilj = kalibrirana rEI (06, stopnja 0).", "",
          "## Kalibracija", ""]
    s = var[var["varianta"] == "BREZ"].set_index("arhetip")
    md += ["| arhetip | obdobje | U vir | f_U | Qh,nd model | cilj rEI | iteracij | U stena / streha / tla / okna (po kalibraciji) |",
           "|---|---|---|---|---|---|---|---|"]
    for a in arhs:
        A = HV["arhetipi"][a]
        md.append(f"| {a} | {PAR['fond']['arhetip_opis'][a]} | {A['U_vir']} | {fU[a]:.3f} | {s.loc[a, 'Qh_nd']:.1f} | {A['cilj_Qhnd']} | "
                  f"{int((kal['arhetip'] == a).sum())} | {s.loc[a, 'U_stena']} / {s.loc[a, 'U_streha']} / {s.loc[a, 'U_tla']} / {s.loc[a, 'U_okna']} |")
    md += ["", "Iteracije: `tabele/02_kalibracija_iteracije.csv`.", "", "## Ovojna lestvica (vse z f_U iz kalibracije na neprenovljenih elementih)", ""]
    lv = var[~var["varianta"].str.startswith("OS")].copy()
    lv["varianta"] = pd.Categorical(lv["varianta"], ["BREZ", "F", "FS", "FSO", "GLOB"], ordered=True)
    lv = lv.sort_values(["arhetip", "varianta"])
    md += ["| arhetip | varianta | Qh,nd | Qw | maks Q_H [W/m²] | P99,9 [W/m²] | Q_H pri T_ext ≤ −10 [W/m²] | ach | ur T_op < 19 |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r in lv.itertuples():
        md.append(f"| {r.arhetip} | {r.varianta} | {r.Qh_nd:.1f} | {r.Qw:.1f} | {r.QH_maks_W_m2:.0f} | {r.QH_P999_W_m2:.0f} | "
                  f"{r.QH_Text_le_m10_W_m2:.0f} | {r.ach_skupaj:.2f} | {r.ur_Top_pod_19} |")
    md += ["", "## Kontrola: FSO (dosegljiva prenova) proti rEI s tremi ukrepi (06, stopnja 3)", "",
           "| arhetip | FSO model | rEI stopnja 3 | razlika |", "|---|---|---|---|"]
    fso = lv[lv["varianta"] == "FSO"].set_index("arhetip")
    for a in arhs:
        c = HV["arhetipi"][a]["cilj_FSO"]
        md.append(f"| {a} | {fso.loc[a, 'Qh_nd']:.1f} | {c} | {fso.loc[a, 'Qh_nd'] - c:+.1f} |")
    gl = lv[lv["varianta"] == "GLOB"].set_index("arhetip")
    md += ["", f"## GLOB = bodoči SI standard ZEB za prenovo (Qh,nd ≤ {HV['glob_kalibracija']['cilj_Qhnd_max']:g} kWh/m²a)", "",
           "Izhodišče FSO + izolacija tal (U 0,30) + n50 1,5 + rekuperacija 0,8; če Qh,nd > 25, se U stene, strehe in tal zaostrijo "
           "z enim faktorjem g ≤ 1. Pogoj na primarni energiji (EPBD čl. 11(3): ≥ 10 % pod nZEB) se preveri v koraku 03 "
           "(prag nZEB in faktor primarne energije sta v `parametri.yaml: zeb` označena PREVERI).", "",
           "| arhetip | g | U stena / streha / tla / okna | n50 | rekuperacija | Qh,nd | P99,9 [W/m²] |", "|---|---|---|---|---|---|---|"]
    for a in arhs:
        r = gl.loc[a]
        md.append(f"| {a} | {gG[a]:.3f} | {r['U_stena']} / {r['U_streha']} / {r['U_tla']} / {r['U_okna']} | {r['n50']} | "
                  f"{r['rekuperacija']} | {r['Qh_nd']:.1f} | {r['QH_P999_W_m2']:.0f} |")
    md += ["", "Iteracije: `tabele/02_glob_iteracije.csv`.", "", "## Os Qh,nd (za Monte Carlo v koraku 03)", "",
           "Neprenovljena hiša z f_U × faktor; oblika urnega profila ostane fizikalna (ne linearno skaliranje).", "",
           "| arhetip | " + " | ".join(f"f_U×{m:g}" for m in HV["os_Qhnd_faktorji_fU"]) + " |",
           "|---|" + "---|" * len(HV["os_Qhnd_faktorji_fU"])]
    for a in arhs:
        vals = []
        for m in HV["os_Qhnd_faktorji_fU"]:
            v = f"OS{m:g}".replace(".", "p")
            vals.append(f"{var[(var['arhetip'] == a) & (var['varianta'] == v)]['Qh_nd'].iloc[0]:.0f}")
        md.append(f"| {a} | " + " | ".join(vals) + " |")
    (HERE / "02_kalibracija.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
