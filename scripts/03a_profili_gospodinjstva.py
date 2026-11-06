"""Korak 03a — urni profili gospodinjske rabe elektrike (brez ogrevanja/TSV) in EV.

- H0: BDEW standardni profil gospodinjstva, dinamiziran (demandlib), SI prazniki, leto 2023 (8.760 h)
- stoh: N stohastičnih profilov posameznih gospodinjstev (CREST / richardsonpy), 1–5 članov, brez električnih grelnih naprav,
        sevanje iz istega EPW (vpliva na razsvetljavo)
- EV: nepametno polnjenje ob prihodu domov, enak dnevni obrok
Vsi profili so normirani na vsoto 1 (delež letne rabe na uro).
Izhodi: cache/03a_profili.parquet (stolpci H0, nadomestek_HECM, EV, stoh_000 … stoh_099), cache/03a_stoh_clani.csv,
        tabele/03a_profili_kontrola.csv
Zagon:  "C:/Anaconda/envs/new_env/python.exe" 03a_profili_gospodinjstva.py
"""
from __future__ import annotations

import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
PAR = yaml.safe_load((HERE / "parametri.yaml").read_text(encoding="utf-8"))
G, EV, HV = PAR["urni_model"]["gospodinjstvo"], PAR["urni_model"]["ev"], PAR["hecm_vhodi"]
CAL = pd.date_range("2023-01-01", periods=8760, freq="h")
CACHE, TAB = HERE / "cache", HERE / "tabele"


def h0() -> np.ndarray:
    from demandlib import bdew
    hol = {pd.Timestamp(d).date(): "praznik" for d in G["prazniki_SI_2023"]}
    e = bdew.ElecSlp(2023, holidays=hol)
    s = e.create_dynamic_h0_profile()
    hourly = s.resample("h").sum().to_numpy()[:8760]
    return hourly / hourly.sum()


def nadomestek() -> np.ndarray:
    import sys
    sys.path.insert(0, PAR["poti"]["hecm"])
    from hecm.io_txt import read_building
    b = read_building(str(Path(PAR["poti"]["hecm"]) / HV["predloga"]))
    wd, we = (np.asarray(b[k], float) for k in ("Lighting hourly fractions workday", "Lighting hourly fractions weekend"))
    shape = np.where(CAL.dayofweek.to_numpy() >= 5, we[CAL.hour], wd[CAL.hour])
    shape = G["osnovna_raba_delez"] + (1 - G["osnovna_raba_delez"]) * shape / shape.mean()
    shape *= 1 + G["sezonska_amplituda"] * np.cos(2 * np.pi * (CAL.dayofyear.to_numpy() - 15) / 365)
    return shape / shape.sum()


def ev() -> np.ndarray:
    daily = EV["raba_kWh_a"] / 365
    p = np.zeros(8760)
    for d in range(365):
        rest, h = daily, EV["prihod_ura"]
        while rest > 1e-9:
            idx = d * 24 + h
            e = min(rest, EV["moc_polnilnika_kW"])
            p[idx % 8760] += e
            rest -= e
            h += 1
    return p / p.sum()


def stoh_one(args):
    """One CREST household, hourly kWh shape (sum 1)."""
    i, n_occ, seed = args
    import random
    import pvlib
    import richardsonpy.classes.electric_load as eload
    import richardsonpy.classes.occupancy as occ
    import richardsonpy.functions.change_resolution as cr
    random.seed(seed)
    np.random.seed(seed)
    d, _ = pvlib.iotools.read_epw(PAR["poti"]["epw"])
    qd = cr.change_resolution(((d["ghi"] - d["dhi"]).clip(lower=0) / 1000).to_numpy(), old_res=3600, new_res=60)
    qf = cr.change_resolution((d["dhi"] / 1000).to_numpy(), old_res=3600, new_res=60)
    o = occ.Occupancy(number_occupants=n_occ)
    e = eload.ElectricLoad(occ_profile=o.occupancy, total_nb_occ=n_occ, q_direct=qd, q_diffuse=qf, annual_demand=1000,
                           is_sfh=True, prev_heat_dev=True, do_normalization=True, timestep=60)
    hourly = np.asarray(e.loadcurve, float)[:8760 * 60].reshape(8760, 60).mean(1)
    return i, n_occ, hourly / hourly.sum()


def main():
    t0 = time.time()
    rng = np.random.default_rng(PAR["urni_model"]["seme"])
    k = list(G["stoh_clani_delezi"])
    clani = rng.choice(k, size=G["stoh_n_profilov"], p=[G["stoh_clani_delezi"][x] for x in k])
    tasks = [(i, int(c), int(PAR["urni_model"]["seme"] + i)) for i, c in enumerate(clani)]
    with Pool(6) as pool:
        res = pool.map(stoh_one, tasks)
    df = pd.DataFrame({"H0": h0(), "nadomestek_HECM": nadomestek(), "EV": ev()}, index=CAL)
    for i, c, prof in sorted(res):
        df[f"stoh_{i:03d}"] = prof
    df.to_parquet(CACHE / "03a_profili.parquet")
    pd.DataFrame({"profil": [f"stoh_{i:03d}" for i, _, _ in sorted(res)], "clani": [c for _, c, _ in sorted(res)]}).to_csv(
        CACHE / "03a_stoh_clani.csv", sep=";", index=False)

    # kontrola: pri 4.000 kWh/a — maks. urna moč, P99, delež zima, delež 10–16 h (sončne ure), večerna konica
    rows = []
    for c in ["H0", "nadomestek_HECM", "EV"] + [x for x in df.columns if x.startswith("stoh_")]:
        p = df[c].to_numpy() * (EV["raba_kWh_a"] if c == "EV" else G["raba_kWh_a"])
        rows.append({"profil": c, "maks_kW": p.max(), "P99_kW": np.quantile(p, 0.99), "povp_kW": p.mean(),
                     "delez_zima_%": 100 * p[np.isin(CAL.month, PAR["urni_model"]["zimski_meseci"])].sum() / p.sum(),
                     "delez_10_16h_%": 100 * p[(CAL.hour >= 10) & (CAL.hour < 16)].sum() / p.sum()})
    k = pd.DataFrame(rows)
    st = k[k["profil"].str.startswith("stoh_")]
    summ = pd.concat([k[~k["profil"].str.startswith("stoh_")],
                      pd.DataFrame([{"profil": "stoh mediana (100)", **st.drop(columns="profil").median().to_dict()},
                                    {"profil": "stoh vsota 100 (agregat)", "maks_kW": (df[[c for c in df if c.startswith('stoh_')]].sum(1) * G['raba_kWh_a'] / 100).max(),
                                     "P99_kW": np.nan, "povp_kW": G["raba_kWh_a"] / 8760, "delez_zima_%": np.nan, "delez_10_16h_%": np.nan}])])
    summ.round(3).to_csv(TAB / "03a_profili_kontrola.csv", sep=";", encoding="utf-8-sig", index=False)
    print(summ.round(2).to_string(index=False))
    print(f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
