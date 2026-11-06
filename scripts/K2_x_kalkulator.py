"""Krog 2, preverjanje: zakaj članek (P1 proti P3) daje drugačen rezultat kot kalkulator NPS 2050 »Se mi splača?«.

Posameznik, paket TČ+PV (Q1, Q2): mediana NPV P3 − P1 in P3 − P2 ter delež hiš, kjer je P3 cenejši, ko predpostavke članka eno za drugo
zamenjamo s predpostavkami kalkulatorja (C:\\dev\\nps2050\\src\\data\\povracilne_dobe.json):
  (spodbude poziva 2026 126SUB-OB26 so od 12. 10. 2026 v osnovi članka — Sprememba 2 v spremembe_krog2.md)
  b) cena TČ sorazmerna moči 1.000 EUR/kW (članek: Theil–Sen 7.225 + 293·kW)
  c) brez diskontiranja, rast cen energije 2 %/a (članek: 4 % realno, konstantne cene)
  d) oddana elektrika 0 EUR/kWh (nova shema samooskrbe v kalkulatorju; članek R3: SPOT − 11,99 EUR/MWh)
  e) obdobje 20 let (članek 30)
  vse skupaj
Ni zajeto (zahteva novo urno simulacijo): globina ovoja (kalkulator 150 → 50, 120 → 45 kWh/m²a; članek FSO po izkaznicah ~75),
SCOP (kalkulator 2,6 / 3,4; članek 3,15 / 3,32), ovoj brez oken.
Izhod: clanek1/tabele/10_primerjava_kalkulator.csv
"""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
s = importlib.util.spec_from_file_location("ek", HERE / "04_ekonomika.py")
E = importlib.util.module_from_spec(s)
s.loader.exec_module(E)
EK0 = copy.deepcopy(E.EK)

KALK = {
    "b": ("TČ 1.000 EUR/kW", lambda: E.EK.update(TC_a_EUR=0.0, TC_b_EUR_kW=1000.0), {}),
    "c": ("brez diskonta, cene +2 %/a", lambda: None, {"r": 0.0, "pf": "rast2"}),
    "d": ("oddaja 0 EUR/kWh", lambda: None, {"oddaja0": True}),
    "e": ("obdobje 20 let", lambda: setattr(E, "N", 20), {}),
}


def reset():
    E.EK.clear()
    E.EK.update(copy.deepcopy(EK0))
    E.SP = E.EK["spodbude"]
    E.N = EK0["horizont_let"]


def run(w0, keys, rez):
    reset()
    kw = {}
    for k in keys:
        KALK[k][1]()
        kw.update(KALK[k][2])
    w = w0.copy()
    if kw.get("oddaja0"):
        for c in [c for c in w.columns if c.endswith(f"|{rez}_oddaja")]:
            w[c] = 0.0
    n = E.npv_paths(w, rez, kw.get("r", E.EK["diskont_zasebni"]), E.pf_factory(kw.get("pf", "konstantne")), E.EK["ovoj_osnova"])
    reset()
    return n


def main():
    w = E.load_units("fond")
    meta = pd.read_parquet(HERE / "cache" / "01_fond.parquet", columns=["ID_Stavbe", "Qr", "paket_osnovni", "arhetip"]).set_index("ID_Stavbe").reindex(w.index)
    meta["Qr"] = pd.Series(np.where(meta["arhetip"].isin(["A1", "A2", "A3"]), "Q1", "Q2"), index=meta.index)                                            # krog 3 (C1): razred po obdobju
    pk = meta["paket_osnovni"].astype(bool).to_numpy()
    VZ = {"paket": pk, "paket pred 1981": pk & (meta["Qr"] == "Q1").to_numpy(), "paket 1981–2002": pk & (meta["Qr"] == "Q2").to_numpy()}
    scen = [("osnova (članek)", [])] + [(KALK[k][0], [k]) for k in KALK] + [("vse skupaj (kot kalkulator)", list(KALK))]
    rows = []
    for rez in ("R3", "R1"):
        for lab, keys in scen:
            if rez == "R1" and "d" in keys and len(keys) == 1:
                continue                                    # pri letnem net meteringu oddaja ni plačana posebej
            n = run(w, keys, rez)
            for vz, m in VZ.items():
                d31, d32 = (n["P3"] - n["P1"]).to_numpy()[m], (n["P3"] - n["P2"]).to_numpy()[m]
                rows.append({"režim": rez, "scenarij": lab, "vzorec": vz,
                             "P3−P1 mediana [kEUR]": np.median(d31) / 1e3, "delež P3 < P1 [%]": 100 * (d31 < 0).mean(),
                             "P3−P2 mediana [kEUR]": np.median(d32) / 1e3, "delež P3 < P2 [%]": 100 * (d32 < 0).mean()})
    T = pd.DataFrame(rows)
    T.round(2).to_csv(HERE / "clanek1" / "tabele" / "10_primerjava_kalkulator.csv", sep=";", encoding="utf-8-sig", index=False)
    pd.set_option("display.width", 220)
    for rez in ("R3", "R1"):
        t = T[T["režim"] == rez].pivot_table(index="scenarij", columns="vzorec", values=["P3−P1 mediana [kEUR]", "delež P3 < P1 [%]"], sort=False)
        print(f"\n=== {rez}: P3 − P1\n", t.round(1).to_string())
        t = T[T["režim"] == rez].pivot_table(index="scenarij", columns="vzorec", values=["P3−P2 mediana [kEUR]", "delež P3 < P2 [%]"], sort=False)
        print(f"\n=== {rez}: P3 − P2 (»OVE najprej, ovoj pozneje« kot v kalkulatorju)\n", t.round(1).to_string())


if __name__ == "__main__":
    main()
