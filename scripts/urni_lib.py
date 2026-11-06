"""Skupni gradniki urnega modela (koraki 03, 03b): HECM profili na m², PV (pvlib + PVGIS), TČ (urni COP, kapaciteta,
dimenzioniranje), baterija (pravilo). Parametri iz parametri.yaml; nobena številka ni trdo kodirana."""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")


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
HERE = Path(__file__).resolve().parent
PAR = _k2_preglasitev(yaml.safe_load((HERE / "parametri.yaml").read_text(encoding="utf-8")))
TAG02 = __import__("os").environ.get("K2_TAG02", "")   # pripona vhodov iz 02 (občutljivost K2_5)
UM, ZEB, HV = PAR["urni_model"], PAR["zeb"], PAR["hecm_vhodi"]
ANALIZA = (HERE / PAR["poti"]["analiza"]).resolve()
sys.path.insert(0, str(ANALIZA))
sys.path.insert(0, PAR["poti"]["hecm"])
import common as C  # noqa: E402
from hecm.heatpump import carnot_cop  # noqa: E402

CACHE, TAB = HERE / "cache", HERE / "tabele"
ARH = list(HV["arhetipi"])
OBD_ARH = {"A1": 1, "A2": 2, "A3": 3, "A4": 4}
VARIANTE = ["V0a", "V0b", "V0c", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V2r"]
OVOJ = {"V0a": "BREZ", "V0b": "BREZ", "V0c": "BREZ", "V1": "BREZ", "V2": "BREZ", "V3": "BREZ", "V4": "FSO", "V5": "FSO", "V6": "FSO", "V7": "GLOB", "V8": "FSO", "V9": "FSO", "V2r": "BREZ"}
# V8 = FSO pozneje ob obstoječi (predimenzionirani) TČ in obstoječem PV (pot P2 po prenovi); V9 = isto po zamenjavi TČ (nova, manjša)
CAL = pd.date_range("2023-01-01", periods=8760, freq="h")     # non-leap, aligned row by row with the EPW / HECM series
ZIMA = np.isin(CAL.month, UM["zimski_meseci"])
t0 = time.time()

# ---------------------------------------------------------------- HECM urni profili na m²
var02 = pd.read_csv(TAB / f"02_variante{TAG02}.csv", sep=";", encoding="utf-8-sig")
PROF, OS = {}, {}
for r in var02.itertuples():
    f_ = CACHE / f"02_urno_{r.arhetip}_{r.varianta}{TAG02}.parquet"
    if not f_.exists():                                    # npr. občutljivost A1 v tabeli variant (krog 3) brez urnega niza
        continue
    h = pd.read_parquet(f_)
    PROF[(r.arhetip, r.varianta)] = {"QH": h["Q_H"].to_numpy() / r.a_use, "QW": h["Q_W"].to_numpy() / r.a_use, "Qh_nd": r.Qh_nd}
T_EXT = pd.read_parquet(CACHE / "02_urno_A1_BREZ.parquet")["T_ext"].to_numpy()
# krog 3: os Qh,nd = lestvica stanj ovoja S0 (BREZ) … S3 (02_lestvica_stanj.py); prej os fU (OS*)
LESTVICA = ["BREZ", "S1", "S2", "S3"]
CILJ_IZ = {"BREZ": "FSO", "S1": "FSO_S1", "S2": "FSO_S2", "S3": "S3"}    # ciljna prenova iz stanja (manjkajoči ukrepi)
for a in ARH:
    if all((a, v) in PROF for v in LESTVICA):
        axis = [(PROF[(a, v)]["Qh_nd"], PROF[(a, v)]["QH"]) for v in LESTVICA]
    else:
        axis = [(PROF[(a, v)]["Qh_nd"], PROF[(a, v)]["QH"]) for v in ["BREZ"] + [x for x in var02["varianta"].unique() if x.startswith("OS")]]
    axis.sort(key=lambda x: x[0])
    OS[a] = (np.array([q for q, _ in axis]), np.stack([p for _, p in axis], 1))


def heat_brez(a: str, q: np.ndarray) -> np.ndarray:
    """Hourly space-heating need [Wh/m²] (8760, n) for an un-renovated house with annual need q [kWh/m²a].
    Linear interpolation between neighbouring physical profiles on the f_U axis (02); outside the axis the end
    profile is scaled."""
    qs, P = OS[a]
    q = np.asarray(q, float)
    hi = np.clip(np.searchsorted(qs, q), 1, len(qs) - 1)
    lo = hi - 1
    w = np.clip((q - qs[lo]) / (qs[hi] - qs[lo]), 0, 1)
    out = P[:, lo] * (1 - w) + P[:, hi] * w
    for mask, j in ((q < qs[0], 0), (q > qs[-1], -1)):
        if mask.any():
            out[:, mask] = P[:, [j]] * (q[mask] / qs[j])
    return out.astype(np.float32)


def polozaj(a: str, q: np.ndarray):
    """Položaj hiše na lestvici obdobja (krog 3): k = zgornje stanje (0 = S0 … 3 = S3), w = utež proti stanju k + 1.
    Nad S0: k = 0, w = 0 (skaliran S0); pod S3: k = 3, w = 0."""
    Q = np.array([PROF[(a, v)]["Qh_nd"] for v in LESTVICA], float)
    q = np.asarray(q, float)
    k, w = np.zeros(len(q), int), np.zeros(len(q))
    for j in range(3):
        m = (q < Q[j]) & (q >= Q[j + 1])
        k[m], w[m] = j, (Q[j] - q[m]) / (Q[j] - Q[j + 1])
    k[q < Q[3]], w[q < Q[3]] = 3, 0.0
    return k, w


def cilj(a: str, q: np.ndarray):
    """Ciljna prenova »URE najprej« (krog 3, B1–B3): urni profil QH [Wh/m²] (8760, n) = interpolacija ciljnih prenov sosednjih stanj z isto utežjo,
    Qh,nd cilja, k, w, brez (Qh,nd hiše ≤ Qh,nd cilja + prag → brez ukrepa na ovoju, profil hiše ostane)."""
    k, w = polozaj(a, q)
    C = [CILJ_IZ[v] for v in LESTVICA]
    P = np.stack([PROF[(a, c)]["QH"] for c in C], 1).astype(np.float32)
    Qc = np.array([PROF[(a, c)]["Qh_nd"] for c in C], float)
    k1 = np.minimum(k + 1, 3)
    prof = P[:, k] * (1 - w).astype(np.float32) + P[:, k1] * w.astype(np.float32)
    qc = Qc[k] * (1 - w) + Qc[k1] * w
    brez = np.asarray(q, float) <= qc + UM["brez_ovoja_prag_kWh_m2a"]
    if brez.any():
        prof[:, brez] = heat_brez(a, np.asarray(q, float)[brez])
    return prof, np.where(brez, q, qc), k, w, brez


# ---------------------------------------------------------------- PV (pvlib, isti EPW)
def pv_profile() -> np.ndarray:
    import pvlib
    pv = UM["pv"]
    data, meta = pvlib.iotools.read_epw(PAR["poti"]["epw"])
    loc = pvlib.location.Location(meta["latitude"], meta["longitude"], tz=f"Etc/GMT{-int(meta['TZ']):+d}", altitude=meta["altitude"])
    times = pd.date_range("2023-01-01 00:30", periods=8760, freq="h", tz=loc.tz)    # mid-hour, EPW hour-ending
    sp = loc.get_solarposition(times)
    dni_extra = pvlib.irradiance.get_extra_radiation(times)
    poa = pvlib.irradiance.get_total_irradiance(pv["nagib"], pv["azimut"], sp["apparent_zenith"], sp["azimuth"],
                                                data["dni"].to_numpy(), data["ghi"].to_numpy(), data["dhi"].to_numpy(),
                                                dni_extra=dni_extra, model="perez",
                                                airmass=pvlib.atmosphere.get_relative_airmass(sp["apparent_zenith"]))
    prm = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["sapm"][pv["temp_model"]]
    tc = pvlib.temperature.sapm_cell(poa["poa_global"].fillna(0), data["temp_air"].to_numpy(), data["wind_speed"].to_numpy(), **prm)
    dc = pvlib.pvsystem.pvwatts_dc(poa["poa_global"].fillna(0), tc, 1000.0, pv["gamma_pdc"])
    prof = np.clip(dc.to_numpy(), 0, None) * (1 - pv["sistemske_izgube"])          # Wh per kWp per hour (IWEC)
    if pv.get("skaliraj_na_PVGIS"):
        m = CAL.month.to_numpy()
        iwec_m = np.array([prof[m == k].sum() / 1000 for k in range(1, 13)])
        f = np.asarray(pv["PVGIS_E_m_kWh_kWp"]) / iwec_m
        print("PV: IWEC mesečno [kWh/kWp]", iwec_m.round(1), f"letno {iwec_m.sum():.0f}; faktorji PVGIS/IWEC", f.round(2))
        prof = prof * f[m - 1]
    return prof.astype(np.float32)


PV1 = pv_profile()
PV_SPEC = PV1.sum() / 1000

# ---------------------------------------------------------------- TČ
TC = UM["tc"]
ETA_TC = TC["COP_A7W35"] * (35.0 - 7.0) / (35.0 + 273.15)


def sink_curve(ovoj: str) -> np.ndarray:
    td, tmin = TC["T_dovod_projektna"][ovoj], TC["T_dovod_min"]
    x = (TC["T_zun_brez_ogrevanja"] - T_EXT) / (TC["T_zun_brez_ogrevanja"] - TC["T_zun_projektna"])
    return np.clip(tmin + (td - tmin) * x, min(tmin, td), td)


def capacity_rel() -> np.ndarray:
    c = np.clip(1 + TC["kapaciteta_k_na_K"] * (T_EXT - 7.0), TC["kapaciteta_min"], 1.0)
    off = T_EXT < (TC["T_zun_min_obratovanja"] if TC["nacin"] == "vzporedno" else TC["T_bivalentna_hecm"])
    return np.where(off, 0.0, c).astype(np.float32)


CAPREL = capacity_rel()
COP_W = carnot_cop(T_EXT, TC["T_dovod_TSV"], ETA_TC, TC["cop_min"], TC["cop_max"]).astype(np.float32)
COP_S = {ov: carnot_cop(T_EXT, sink_curve(ov), ETA_TC, TC["cop_min"], TC["cop_max"]).astype(np.float32) for ov in TC["T_dovod_projektna"]}
CAP_DESIGN = float(np.clip(1 + TC["kapaciteta_k_na_K"] * (TC["T_zun_projektna"] - 7.0), TC["kapaciteta_min"], 1.0))


def heat_pump(QH, QW, kW, ovoj):
    """QH, QW (8760, n) Wh to deliver; kW (n,) nominal at A7. Returns HP electricity, backup electricity, HP heat."""
    cap = CAPREL[:, None] * (np.asarray(kW, np.float32)[None, :] * 1000)
    tot = QH + QW
    served = np.minimum(tot, cap)
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(tot > 0, served / tot, 0).astype(np.float32)
    if isinstance(ovoj, str):
        cop = COP_S[ovoj][:, None]
    else:                                                  # krivulja dovoda po hiši (krog 3: hiše brez ukrepa na ovoju v P3 ostanejo pri BREZ)
        ov = np.asarray(ovoj)
        cop = np.where(ov[None, :] == "BREZ", COP_S["BREZ"][:, None], COP_S["FSO"][:, None])
    el = frac * (QH / cop + QW / COP_W[:, None])
    backup = (tot - served) / TC["grelnik_eta"]
    return el, backup, served


def size_hp(QH, QW) -> np.ndarray:
    need = np.quantile(QH, 0.999, axis=0) + QW.mean(axis=0)
    req = need / 1000 / CAP_DESIGN
    sizes = np.asarray(TC["velikosti_kW"], float)
    idx = np.clip(np.searchsorted(sizes, req), 0, len(sizes) - 1)
    return sizes[idx]


# ---------------------------------------------------------------- baterija (pravilo: polni iz presežka, prazni ob uvozu)
BAT = UM["baterija"]


def battery(net: np.ndarray, kWh: float):
    """net = load − PV (8760, n) Wh. Returns import, export (8760, n)."""
    n = net.shape[1]
    E = kWh * 1000 * BAT["DoD"]
    pmax = kWh * 1000 * BAT["C_rate"]
    ec = ed = BAT["izkoristek_krozni"] ** 0.5
    soc = np.zeros(n, np.float32)
    imp, exp_ = np.empty_like(net), np.empty_like(net)
    for t in range(net.shape[0]):
        sur = np.maximum(-net[t], 0)
        dfc = np.maximum(net[t], 0)
        ch = np.minimum(np.minimum(sur, pmax), (E - soc) / ec)
        soc += ch * ec
        dis = np.minimum(np.minimum(dfc, pmax), soc * ed)
        soc -= dis / ed
        imp[t] = dfc - dis
        exp_[t] = sur - ch
    return imp, exp_


def round_kwp(x):
    s = UM["pv"]["korak_kWp"]
    return np.maximum(np.ceil(np.asarray(x) / s) * s, UM["pv"]["min_kWp"])


# ---------------------------------------------------------------- računi za elektriko (korak 04; parametri: ekonomika)
EK = PAR["ekonomika"]


def _bloki() -> np.ndarray:
    """Časovni blok (1–5) za vsako uro koledarja modela; pravila do 31. 12. 2026 (Ur. l. 146/22, Priloga 2)."""
    vs = np.isin(CAL.month, [11, 12, 1, 2])
    praz = np.isin(CAL.normalize(), pd.to_datetime(UM["gospodinjstvo"]["prazniki_SI_2023"]))
    prost = (CAL.dayofweek >= 5) | praz
    h = CAL.hour.to_numpy()
    base = np.select([h <= 5, h == 6, h <= 13, h <= 15, h <= 19, h <= 21], [3, 2, 1, 2, 1, 2], 3)   # VS delovni dan
    return (base + np.where(vs, 0, 1) + np.where(prost, 1, 0)).astype(int)                         # NS +1, prost dan +1


BLOK = _bloki()
_c = pd.read_csv(HERE / EK["cene_datoteka"], sep=";", encoding="utf-8-sig", index_col=0)
_c.index = pd.to_datetime(_c.index, utc=True).tz_convert("Europe/Ljubljana")
SPOT = (_c.loc[_c.index.year == EK["cene_leto"], "cena_EUR_MWh"].to_numpy()[:8760] / 1000).astype(np.float64)   # EUR/kWh
MESECEV_BLOKA = np.array([len(np.unique(CAL.month[BLOK == b])) for b in range(1, 6)])


def racuni(imp_Wh: np.ndarray, exp_Wh: np.ndarray, kwp: np.ndarray) -> dict:
    """Letni računi [EUR, z DDV] po režimih; komponente ločeno (energija ∝ cenovni scenarij, omrežnina in dajatve fiksno).
    imp, exp (8760, n) Wh; kwp (n,). R1 letni net metering; R2 urni obračun energije, omrežnina brez 15-min (fiksna moč);
    R3 = R2 + omrežnina 5 blokov z DOM; R2d = R2 z dinamično ceno uvoza."""
    imp, exp_ = imp_Wh.astype(np.float64) / 1000, exp_Wh.astype(np.float64) / 1000
    v = 1 + EK["ddv"]
    Ei, Ee = imp.sum(0), exp_.sum(0)
    netto = np.maximum(Ei - Ee, 0)
    ima_pv = np.asarray(kwp) > 0
    P_obr = EK["prikljucna_moc_kW"] * EK["obracunska_moc_delez_brez15"]
    ove_fix = np.maximum(P_obr - np.asarray(kwp, float), 0) * EK["prispevek_OVE_EUR_kW_mes"] * 12
    fix = EK["nadomestilo_dobavitelja_mes"] * 12
    moc_fix = P_obr * EK["omreznina_moc_brez15_EUR_kW_mes"] * 12
    trosarina = netto * EK["trosarina_EUR_kWh"]
    ure = np.where(ima_pv, 0.0, Ei * EK["prispevek_URE_EUR_kWh"])
    out = {}
    # R1 — letni net metering
    out["R1_energija"] = v * netto * EK["energija_ET_netmetering"]
    out["R1_oddaja"] = np.zeros_like(Ei)
    out["R1_omr_daj"] = v * (netto * (EK["omreznina_energija_ET"] + EK["prispevek_operater_trga_EUR_kWh"] + EK["prispevek_URE_EUR_kWh"])
                            + moc_fix + ove_fix + trosarina + fix)
    # R2 — urni obračun, fiksna moč
    odd = (exp_ * (SPOT[:, None] - EK["oddaja_odbitek"])).sum(0)
    out["R2_energija"] = v * Ei * EK["energija_ET"]
    out["R2_oddaja"] = odd                                  # prihodek gospodinjstva (brez DDV)
    omr_daj2 = v * (Ei * (EK["omreznina_energija_ET"] + EK["prispevek_operater_trga_EUR_kWh"]) + ure + moc_fix + ove_fix + trosarina + fix)
    out["R2_omr_daj"] = omr_daj2
    out["R2d_energija"] = v * (imp * (SPOT[:, None] + EK["dinamicna_pribitek_uvoz"])).sum(0)
    # R3 — 5 blokov: energija po blokih, moč po DOM (povprečje 5 najvišjih urnih uvozov bloka), monotonost, min. blok 1
    ce = np.asarray(EK["omreznina_energija_blok_EUR_kWh"])[BLOK - 1]
    om_e = (imp * ce[:, None]).sum(0)
    dom = []
    for b in range(1, 6):
        x = np.sort(imp[BLOK == b], axis=0)[-EK["DOM_konic"]:]
        dom.append(x.mean(0))
    dom = np.stack(dom, 0)
    dom[0] = np.maximum(dom[0], EK["min_DOM_blok1_kW"])
    dom = np.maximum.accumulate(dom, axis=0)
    om_m = (dom * (np.asarray(EK["omreznina_moc_blok_EUR_kW_mes"]) * MESECEV_BLOKA)[:, None]).sum(0)
    ove3 = (np.maximum(dom[0] - kwp, 0) * 4 + np.maximum(dom[1] - kwp, 0) * 8) * EK["prispevek_OVE_EUR_kW_mes"]
    out["R3_energija"] = out["R2_energija"]
    out["R3_oddaja"] = odd
    out["R3_omr_daj"] = v * (Ei * EK["prispevek_operater_trga_EUR_kWh"] + om_e + om_m + ure + ove3 + trosarina + fix)
    out["R3_DOM1_kW"] = dom[0]
    out["R3_DOM3_kW"] = dom[2]
    # krog 2 — komponente za javni proračun (brez DDV; DDV posebej) in urna vrednost elektrike za družbeno CBA
    d = EK["ddv"]
    op = EK["prispevek_operater_trga_EUR_kWh"]
    for rez, base_e, omr, prisp in (
            ("R1", netto * EK["energija_ET_netmetering"], netto * EK["omreznina_energija_ET"] + moc_fix, netto * (op + EK["prispevek_URE_EUR_kWh"]) + ove_fix),
            ("R2", Ei * EK["energija_ET"], Ei * EK["omreznina_energija_ET"] + moc_fix, Ei * op + ure + ove_fix),
            ("R3", Ei * EK["energija_ET"], om_e + om_m, Ei * op + ure + ove3)):
        out[f"{rez}_omreznina"] = omr
        out[f"{rez}_prispevki"] = prisp
        out[f"{rez}_trosarina"] = trosarina
        out[f"{rez}_ddv"] = d * (base_e + omr + prisp + trosarina + fix)
    zima = np.isin(CAL.month, UM["zimski_meseci"])
    out["soc_uvoz_EUR"] = (imp * SPOT[:, None]).sum(0)
    out["soc_uvoz_zima_EUR"] = (imp[zima] * SPOT[zima, None]).sum(0)
    out["soc_oddaja_EUR"] = (exp_ * SPOT[:, None]).sum(0)
    out["soc_oddaja_zima_EUR"] = (exp_[zima] * SPOT[zima, None]).sum(0)
    return out


# ---------------------------------------------------------------- sistem SI 2025 (05a): urni potrošni EF in ure sistemske konice
_s = pd.read_parquet(CACHE / "05a_sistem_SI.parquet")
EF_URNI = _s["EF_potrosni_SI"].to_numpy()[:8760].astype(np.float64)            # t/MWh = kg/kWh, položajna preslikava
EF_POVP = float((_s["EF_potrosni_SI"] * _s["odjem_MW"]).sum() / _s["odjem_MW"].sum())
KONICA_SIS = _s["sistemska_konica"].to_numpy()[:8760].astype(bool)


def sistem_kazalniki(imp_Wh: np.ndarray, exp_Wh: np.ndarray) -> dict:
    """CO2 elektrike (urni in povprečni EF), izognjene emisije oddaje, uvoz v urah sistemske konice."""
    imp, exp_ = imp_Wh.astype(np.float64) / 1000, exp_Wh.astype(np.float64) / 1000
    return {"CO2_el_urni_kg": (imp * EF_URNI[:, None]).sum(0), "CO2_el_povp_kg": imp.sum(0) * EF_POVP,
            "CO2_izogib_oddaja_kg": (exp_ * EF_URNI[:, None]).sum(0),
            "uvoz_konica_sis_kWh": imp[KONICA_SIS].sum(0), "uvoz_konica_sis_kW": imp[KONICA_SIS].mean(0)}
