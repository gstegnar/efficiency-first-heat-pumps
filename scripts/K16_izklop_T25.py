"""Članek 1 — občutljivost na izklop TČ (NAVODILA_CLAUDE_CODE_v3.md članka 2, točka D). Osnova članka 1 se NE spreminja.

Osnova: TČ pod −20 °C izklopljena (parametri.yaml; IWEC 3 ure → grelnik). Občutljivost: obratovanje do −25 °C (K2_PAR, tek 03 --tag _T25 na paketu 8.134, B2).
Iste funkcije kot 22f / K14 (P3 − P1 po hiši): posameznik R3, družbeni NPV (vključuje NN ojačitev 670 €/kW in zmogljivost na sistemski konici), proračun,
CO₂e, zimski uvoz, NN obremenitev, prispevek k 100 uram sistemske konice. Izhod: clanek1/tabele/25_izklop_TC_obcutljivost.csv (+ tabele/).
Merilo (navodilo): relativna sprememba deleža »P3 boljši« ali vsote < 10 % → stavek v metode/omejitve; ≥ 10 % → označi (popravek NE v8 / EP v17).
Zagon: GEOM=G13 UMESTITEV=B2 C:/Anaconda/envs/new_env/python.exe K16_izklop_T25.py
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sp = importlib.util.spec_from_file_location("k14", HERE / "K14_umestitev.py")
K = importlib.util.module_from_spec(sp)
sp.loader.exec_module(K)
TEKI = {"−20 °C (osnova)": "_L3", "−25 °C (občutljivost)": "_T25"}
rows = []
for lab, tag in TEKI.items():
    w, d = K.izracun(tag)
    rc = K.K1.racun(w)
    lev = {"obremenitev NN [kW]": (rc["P1"]["obremenitev NN omrežja [kW]"].to_numpy(), rc["P3"]["obremenitev NN omrežja [kW]"].to_numpy(), 1e3, "MW"),
           "prispevek k 100 uram sistemske konice [kW]": (w["V2|uvoz_konica_sis_kW"].to_numpy(float), w["V5|uvoz_konica_sis_kW"].to_numpy(float), 1e3, "MW"),
           "konica uvoza hiše [kW]": (w["V2|uvoz_konica_kW"].to_numpy(float), w["V5|uvoz_konica_kW"].to_numpy(float), 1e3, "MW (vsota)"),
           "zimski uvoz (leto 20) [kWh]": (w["V2|uvoz_zima_kWh"].to_numpy(float), w["V5|uvoz_zima_kWh"].to_numpy(float), 1e6, "GWh/a")}
    for obd in ("pred 1981", "1981–2002", "paket"):
        m = np.ones(len(d), bool) if obd == "paket" else (d["obdobje"] == obd).to_numpy()
        for k, (p1, p3, ds, es) in lev.items():
            x = (p3 - p1)[m]
            rows.append({"izklop TČ": lab, "vzorec": obd, "kazalnik": k, "P1 mediana": np.median(p1[m]), "P3 mediana": np.median(p3[m]),
                         "P3 − P1 mediana": np.median(x), "delež P3 boljši [%]": 100 * (x < 0).mean(), "vsota P3 − P1": x.sum() / ds, "enota vsote": es})
        for k in ("posameznik R3", "družba", "proračun R3", "CO2e 30 let"):
            eu, dv, es, ds = K.KAZ[k]
            x = d[k].to_numpy()[m]
            rows.append({"izklop TČ": lab, "vzorec": obd, "kazalnik": f"{k} [{eu}]", "P3 − P1 mediana": np.median(x) / dv,
                         "delež P3 boljši [%]": 100 * (x < 0).mean(), "vsota P3 − P1": x.sum() / ds, "enota vsote": es})
T = pd.DataFrame(rows)
a, b = (T[T["izklop TČ"] == l].set_index(["vzorec", "kazalnik"]) for l in TEKI)
C = a[["P1 mediana", "P3 mediana", "P3 − P1 mediana", "delež P3 boljši [%]", "vsota P3 − P1", "enota vsote"]].join(
    b[["P1 mediana", "P3 mediana", "P3 − P1 mediana", "delež P3 boljši [%]", "vsota P3 − P1"]], lsuffix=" −20", rsuffix=" −25")
C["sprememba deleža [%, rel.]"] = 100 * (C["delež P3 boljši [%] −25"] / C["delež P3 boljši [%] −20"] - 1)
C["sprememba vsote [%, rel.]"] = 100 * (C["vsota P3 − P1 −25"] / C["vsota P3 − P1 −20"] - 1)
C["≥ 10 %"] = (C["sprememba deleža [%, rel.]"].abs() >= 10) | (C["sprememba vsote [%, rel.]"].abs() >= 10)
C = C.reset_index()
for o in (HERE / "clanek1" / "tabele", HERE / "tabele"):
    C.round(3).to_csv(o / "25_izklop_TC_obcutljivost.csv", sep=";", encoding="utf-8-sig", index=False)
pd.set_option("display.width", 260)
pd.set_option("display.max_columns", 20)
print(C.round(2).to_string(index=False))
