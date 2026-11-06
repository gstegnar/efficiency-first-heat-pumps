"""Krog 9, §4 — hlajenje po stanju ovoja (samo izpis za razpravo, ne v poti; NAVODILA_CLAUDE_CODE_v9.md).

Iz urnih nizov HECM (korak 02, cache/02_urno_<arhetip>_<stanje>.parquet: Q_C [Wh/h na stavbo], T_op [°C]) za vsako obdobje × stanje ovoja S0–S3, FSO, GLOB:
potreba po hlajenju [kWh/m²a], P99,9 urne moči hlajenja [W/m²], ure T_op > 26 °C, električna moč hlajenja pri EER 3,5 v 10 urah najvišje
odjemne obremenitve sistema SI poleti (jun–avg, cache/05a_sistem_SI.parquet) [W/m² in kW za mediano površino paketa].
Nastavitve HECM (vhodi 02, hecm/configs/defaults.yaml, hecm/config.py) se izpišejo v dnevnik.
Izhod: clanek1/tabele/18_hlajenje_po_stanju.csv (+ tabele/)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
PAR = yaml.safe_load((HERE / "parametri.yaml").read_text(encoding="utf-8"))
TAB, OUT_T, CACHE = HERE / "tabele", HERE / "clanek1" / "tabele", HERE / "cache"
EER = 3.5
OBD = {"A1": "≤1945", "A2": "1946–1970", "A3": "1971–1980", "A4": "1981–2002"}
STANJA = [("BREZ", "S0 (brez ukrepa)"), ("S1", "S1 okna"), ("S2", "S2 streha + okna"), ("S3", "S3 celovita"),
          ("FSO", "FSO (ciljna prenova iz S0)"), ("GLOB", "GLOB (ZEB)")]
CAL = pd.date_range("2023-01-01", periods=8760, freq="h")


def main():
    var = pd.read_csv(TAB / "02_variante.csv", sep=";", encoding="utf-8-sig").set_index(["arhetip", "varianta"])
    s = pd.read_parquet(CACHE / "05a_sistem_SI.parquet")
    odj = s["odjem_MW"].to_numpy()[:8760]
    poletje = np.isin(CAL.month, [6, 7, 8])
    ure = np.where(poletje)[0][np.argsort(-odj[poletje])[:10]]
    fond = pd.read_parquet(CACHE / "01_fond.parquet", columns=["arhetip", "obdobje", "A_ogr", "A_vir", "paket_osnovni"])
    fond = fond[fond["paket_osnovni"].astype(float).fillna(0).astype(bool) & (fond["A_vir"] != "imputirano")]
    fond = fond.assign(A_ogr=fond["A_ogr"].astype(float))
    if __import__("os").environ.get("GEOM", "G13") in ("G11", "G13"):   # krog 12/13: ogrevana površina kot v 03
        fond["A_ogr"] = fond["A_ogr"] * fond["obdobje"].map(PAR["hecm_vhodi"]["povrsina_G11_faktor"]).fillna(1.0)
    A_med = fond.groupby("arhetip")["A_ogr"].median()
    rows = []
    for a, obd in OBD.items():
        for v, lab in STANJA:
            f = CACHE / f"02_urno_{a}_{v}.parquet"
            if not f.exists():
                continue
            h = pd.read_parquet(f)
            A = float(var.loc[(a, v), "a_use"])
            qc = h["Q_C"].to_numpy(float) / A                          # Wh/m² na uro = W/m²
            el = qc[ure].mean() / EER
            rows.append({"arhetip": a, "obdobje": obd, "stanje": lab, "Qh,nd [kWh/m²a]": float(var.loc[(a, v), "Qh_nd"]),
                         "hlajenje Q_C [kWh/m²a]": qc.sum() / 1000, "P99,9 moč hlajenja [W/m²]": np.quantile(qc, 0.999),
                         "maks. moč hlajenja [W/m²]": qc.max(), "ure s hlajenjem": int((qc > 0).sum()),
                         "ure T_op > 26 °C": int((h["T_op"] > 26).sum()), "maks. T_op [°C]": h["T_op"].max(),
                         f"el. moč hlajenja EER {EER}, 10 h poletne konice SI [W/m²]": el,
                         "mediana A paketa [m²]": A_med[a], "el. moč hlajenja na hišo paketa [kW]": el * A_med[a] / 1000,
                         "el. raba hlajenja na hišo paketa [kWh/a]": qc.sum() / 1000 / EER * A_med[a]})
    T = pd.DataFrame(rows)
    for o in (TAB, OUT_T):
        T.round(3).to_csv(o / "18_hlajenje_po_stanju.csv", sep=";", encoding="utf-8-sig", index=False)
    pd.set_option("display.width", 250)
    print("ure poletne konice SI:", [str(CAL[i]) for i in sorted(ure)])
    print(T.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
