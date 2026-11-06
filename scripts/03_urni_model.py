"""Korak 03 — urni energijski model: toplota (HECM, korak 02), TČ (urni COP), PV (pvlib), baterija, gospodinjstvo, bilanca.

Dve ravni (odločitev 9. 10. 2026):
  (A) reprezentativne hiše po arhetipu (mediane fonda, Qh,nd = kalibrirana vrednost celice) — urni nizi za slike;
  (B) vsaka stavba fonda (29.977) z lastno A_ogr, močjo TČ, kWp in Qh,nd (lastna rEI ali vzorec iz porazdelitve rEI celice).
Variante (CLAUDE.md §4.1): V0a ELKO, V0b biomasa, V1 TČ, V2 TČ+PV, V3 TČ+PV+bat, V4 FSO+TČ, V5 FSO+TČ+PV, V6 +bat, V7 GLOB+TČ+PV+bat;
za poti (§4.2): V8 FSO + obstoječa TČ + obstoječi PV (P2 po prenovi), V9 FSO + nova TČ + obstoječi PV (P2 po zamenjavi TČ).
Obratovanje je neoptimirano (TČ sledi potrebi, baterija polni iz presežka, prazni ob uvozu); optimirano v 03b.
PV in elektrika iz HECM se ne uporabljata (CLAUDE.md §2.2).

Izhodi: tabele/03_energija_variante.csv (reprezentativne), tabele/03_fond_variante.csv (fond), 03_povzetek.md,
        cache/03_bilanca_reprezentativne.parquet (urno), cache/03_fond_letno.parquet (stavba × varianta),
        cache/03_fond_urno_vsota.parquet (vsota fonda po urah)
Zagon:  "C:/Anaconda/envs/new_env/python.exe" 03_urni_model.py [--gosp H0|stoh|nadomestek_HECM] [--ev]
        Osnovni izračun: H0 brez EV (brez pripone). Občutljivosti dobijo pripono, npr. _stoh, _H0_EV, _stoh_EV.
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from urni_lib import (  # noqa: E402
    HERE, PAR, UM, ZEB, HV, ANALIZA, C, CACHE,
    TAB, ARH, OBD_ARH, VARIANTE, OVOJ, CAL, ZIMA, t0,
    var02, PROF, OS, T_EXT, heat_brez, pv_profile, PV1, PV_SPEC,
    TC, ETA_TC, sink_curve, capacity_rel, CAPREL, COP_W, COP_S, CAP_DESIGN,
    heat_pump, size_hp, BAT, battery, round_kwp, carnot_cop, racuni, sistem_kazalniki, cilj, SPOT,
)

# ---------------------------------------------------------------- gospodinjstvo (nadomestek, PREVERI)
ap = argparse.ArgumentParser()
ap.add_argument("--gosp", default=UM["gospodinjstvo"]["profil"], choices=["H0", "stoh", "nadomestek_HECM"])
ap.add_argument("--ev", action="store_true")
ap.add_argument("--fond", default=None, help="alternativna tabela stavb (parquet, stolpci kot 01_fond); npr. sintetična mreža Qh,nd v koraku 07")
ap.add_argument("--tag", default=None, help="pripona izhodov (preglasi samodejno)")
ap.add_argument("--dist-oddaja", default=None, help="krog 9: pripona vsote, iz katere se vzamejo ure sočasne konice ODDAJE (uvoz ostane iz osnove)")
ARGS = ap.parse_args()
TAG = "" if (ARGS.gosp == UM["gospodinjstvo"]["profil"] and not ARGS.ev) else f"_{ARGS.gosp}" + ("_EV" if ARGS.ev else "")
if ARGS.tag is not None:
    TAG = ARGS.tag
PROFILI = pd.read_parquet(CACHE / "03a_profili.parquet")          # 03a_profili_gospodinjstva.py; vsak stolpec vsota 1
STOH = PROFILI[[c for c in PROFILI if c.startswith("stoh_")]].to_numpy(np.float32)
HH1 = PROFILI["H0" if ARGS.gosp == "stoh" else ARGS.gosp].to_numpy(np.float32)   # za stoh: H0 le kot rezerva
EV1 = PROFILI["EV"].to_numpy(np.float32) * (UM["ev"]["raba_kWh_a"] * 1000 if ARGS.ev else 0.0)
print(f"gospodinjski profil: {ARGS.gosp}, EV: {ARGS.ev}, pripona: '{TAG}'")




# ---------------------------------------------------------------- en sklop stavb, vse variante
# ure sočasne konice flote po varianti (distribucijsko omrežje, krog 2): iz osnovnega teka (cache/03_fond_urno_vsota.parquet)
_NDIST = PAR["drzava_k2"]["distribucija_ur_konice"]
_vs = CACHE / "03_fond_urno_vsota.parquet"
if _vs.exists():
    _h = pd.read_parquet(_vs)
    DIST_URE = {v: (np.argsort(-_h[f"{v}|uvoz"].to_numpy())[:_NDIST], np.argsort(-_h[f"{v}|oddaja"].to_numpy())[:_NDIST])
                for v in VARIANTE if f"{v}|uvoz" in _h}
    del _h
    _vo = CACHE / f"03_fond_urno_vsota{ARGS.dist_oddaja}.parquet" if ARGS.dist_oddaja else None
    if _vo is not None and _vo.exists():                  # krog 9: pri omejitvi oddaje so ure sočasne oddaje flote drugačne (2. prehod)
        _h = pd.read_parquet(_vo)
        DIST_URE = {v: (u, np.argsort(-_h[f"{v}|oddaja"].to_numpy())[:_NDIST]) for v, (u, _) in DIST_URE.items()}
        del _h
else:
    DIST_URE = {}
    print("OPOZORILO: ni osnovnega 03_fond_urno_vsota — sočasna konica za NN omrežje ni izračunana (poženi osnovni 03 dvakrat)")


def simulate(b: pd.DataFrame, keep_hourly=False):
    """b: columns arhetip, obdobje, A, qh, tc_kW, pv_kWp_V2, faktor_gosp (one archetype per call)."""
    a = b["arhetip"].iloc[0]
    n = len(b)
    A = b["A"].to_numpy(np.float32)
    obd = b["obdobje"].to_numpy()
    raba = (UM["gospodinjstvo"]["raba_kWh_a"] * 1000 * b["faktor_gosp"].to_numpy(np.float32))[None, :]
    base = STOH[:, b["gosp_idx"].to_numpy()] if ARGS.gosp == "stoh" else HH1[:, None]
    hh = base * raba + EV1[:, None]          # gospodinjstvo (+ EV pri občutljivosti) — ni del rab EPBD
    tsv = UM["tsv_izgube_faktor"]
    rows, hourly, sums = [], {}, {}

    def frazvod(ov):
        st = UM["stopnja_po_varianti"][ov]
        return np.array([C.f_razvod(o, st) for o in obd], np.float32)

    heat = {}
    qh_b = heat_brez(a, b["qh"].to_numpy())
    # krog 3: ciljna prenova po lestvici stanj (interpolacija, B1–B3); hiše brez ukrepa na ovoju ostanejo pri svojem profilu in razvodu
    qc_prof, qc_nd, lest_k, lest_w, brez_P3 = cilj(a, b["qh"].to_numpy())
    # krog 4 (K4_1): stanje in ciljna prenova po registru — profil stanja in cilja po hiši (stolpca stanje_var, cilj_var; kjer sta podana)
    if "stanje_var" in b.columns:
        m4 = b["stanje_var"].notna().to_numpy()
        for j in np.where(m4)[0]:
            sv, cv = b["stanje_var"].iat[j], b["cilj_var"].iat[j]
            sk = float(b["stanje_skala"].iat[j]) if "stanje_skala" in b.columns and pd.notna(b["stanje_skala"].iat[j]) else 1.0   # krog 8: skaliran profil stanja (fU)
            cs = float(b["cilj_skala"].iat[j]) if "cilj_skala" in b.columns and pd.notna(b["cilj_skala"].iat[j]) else 1.0      # krog 13: sorazmeren cilj
            qs, qc = PROF[(a, sv)]["Qh_nd"] * sk, PROF[(a, cv)]["Qh_nd"] * cs
            qh_b[:, j] = PROF[(a, sv)]["QH"] * sk
            brez_P3[j] = qs <= qc + UM["brez_ovoja_prag_kWh_m2a"]
            qc_prof[:, j] = qh_b[:, j] if brez_P3[j] else PROF[(a, cv)]["QH"] * cs
            qc_nd[j] = qs if brez_P3[j] else qc
            lest_k[j], lest_w[j] = int(b["stanje_k"].iat[j]), 0.0
    fq = np.float32(UM.get("faktor_QH", 1.0))                 # krog 9: toplejše podnebje — urna potreba po toploti × povprečni faktor horizonta
    qh_b, qc_prof = qh_b * fq, qc_prof * fq
    heat["BREZ"] = (qh_b * A, PROF[(a, "BREZ")]["QW"][:, None].astype(np.float32) * A * tsv, frazvod("BREZ"))
    heat["FSO"] = (qc_prof * A, PROF[(a, "FSO")]["QW"][:, None].astype(np.float32) * A * tsv, np.where(brez_P3, frazvod("BREZ"), frazvod("FSO")))
    sink_fso = np.where(brez_P3, "BREZ", "FSO")
    heat["GLOB"] = (PROF[(a, "GLOB")]["QH"][:, None].astype(np.float32) * A * fq, PROF[(a, "GLOB")]["QW"][:, None].astype(np.float32) * A * tsv, frazvod("GLOB"))
    del qh_b

    fans = np.zeros((8760, n), np.float32)
    q_vent = 0.5 * A * HV["visina_etaze_m"] / 3600.0                      # m3/s (HECM typology 0.5 1/h)
    fans_glob = np.repeat((UM["prezracevanje_SFP_kW_m3s"] * q_vent * 1000)[None, :], 8760, 0).astype(np.float32)

    def finish(v, ov, QHn, QH_del, QW_del, el_hp, el_bu, served, fuel, fuel_f, el_dhw_bojler, kwp, bat_kWh, kW, fan):
        el_epbd = el_hp + el_bu + el_dhw_bojler + fan
        load = hh + el_epbd
        pv = PV1[:, None] * np.asarray(kwp, np.float32)[None, :]
        net = load - pv
        if bat_kWh:
            imp, exp_ = battery(net, bat_kWh)
        else:
            imp, exp_ = np.maximum(net, 0), np.maximum(-net, 0)
        izg = np.zeros_like(exp_)
        if UM["pv"].get("omejitev_oddaje_kW_kWp") is not None:     # krog 9: oddaja ≤ c · kWp, presežek izgubljen (po bateriji)
            lim = (UM["pv"]["omejitev_oddaje_kW_kWp"] * 1000 * np.asarray(kwp, np.float32))[None, :]
            izg = np.maximum(exp_ - lim, 0)
            exp_ = exp_ - izg
        with np.errstate(divide="ignore", invalid="ignore"):
            share = np.where(load > 0, el_epbd / load, 0)
        imp_epbd = (imp * share).sum(0) / 1000
        amb = np.maximum(served - el_hp, 0).sum(0) / 1000 if served is not None else np.zeros(n)
        fp = ZEB["f_P_tot"]
        pe = imp_epbd * fp["elektrika"] + fuel / 1000 * fuel_f + (amb * fp["toplota_okolice"] if ZEB["vkljuci_toploto_okolice"] else 0)
        el_epbd_a = el_epbd.sum(0) / 1000
        kexp = ZEB["k_exp"]["z_baterijo" if bat_kWh else "brez_baterije"]
        pe_kred = pe - (exp_.sum(0) / 1000) * fp["elektrika"] * kexp * np.where(load.sum(0) > 0, el_epbd.sum(0) / load.sum(0), 0)
        top10 = -np.sort(-imp, axis=0)[:10].mean(0) / 1000
        pv_a = pv.sum(0) / 1000
        r = pd.DataFrame({
            "varianta": v, "ovoj": ov, "A": A, "Qh_nd": QHn.sum(0) / 1000 / A,
            "toplota_ogrevanje_kWh": QH_del.sum(0) / 1000, "toplota_TSV_kWh": QW_del.sum(0) / 1000,
            "TC_kW": kW, "el_TC_kWh": el_hp.sum(0) / 1000, "el_grelnik_kWh": el_bu.sum(0) / 1000,
            "el_TSV_bojler_kWh": el_dhw_bojler.sum(0) / 1000, "el_prezr_kWh": fan.sum(0) / 1000,
            "el_gosp_kWh": hh.sum(0) / 1000, "el_EPBD_kWh": el_epbd_a, "el_skupaj_kWh": load.sum(0) / 1000,
            "gorivo_kWh": fuel / 1000, "PV_kWp": kwp, "PV_kWh": pv_a, "baterija_kWh": bat_kWh,
            "uvoz_kWh": imp.sum(0) / 1000, "oddaja_kWh": exp_.sum(0) / 1000, "uvoz_zima_kWh": imp[ZIMA].sum(0) / 1000,
            "uvoz_konica_kW": imp.max(0) / 1000, "uvoz_top10_kW": top10,
            "uvoz_konica_dist_kW": imp[DIST_URE[v][0]].mean(0) / 1000 if v in DIST_URE else np.nan,
            "oddaja_konica_dist_kW": exp_[DIST_URE[v][1]].mean(0) / 1000 if v in DIST_URE else np.nan,
            "el_konica_brez_PV_kW": load.max(0) / 1000,
            "samozadostnost": 1 - imp.sum(0) / np.maximum(load.sum(0), 1),
            "samoporaba": np.where(pv_a > 0, (pv_a - exp_.sum(0) / 1000) / np.maximum(pv_a, 1e-9), np.nan),
            "PE_kWh_m2": pe / A, "PE_z_okolico_kWh_m2": (pe + (0 if ZEB["vkljuci_toploto_okolice"] else amb * fp["toplota_okolice"])) / A,
            "PE_z_oddajo_kWh_m2": pe_kred / A,
            "PV_izgubljeno_kWh": izg.sum(0) / 1000, "PV_izgubljeno_EUR": (izg.astype(np.float64) / 1000 * SPOT[:, None]).sum(0),
        })
        for k_, x_ in racuni(imp, exp_, np.asarray(kwp, float)).items():   # letni računi R1/R2/R3 (korak 04)
            r[k_] = x_
        for k_, x_ in sistem_kazalniki(imp, exp_).items():                   # CO2 in sistemska konica (koraka 05, 06)
            r[k_] = x_
        heat_tc = served.sum(0) / 1000 if served is not None else np.nan
        r["SCOP_TC"] = np.where(el_hp.sum(0) > 0, heat_tc / np.maximum(el_hp.sum(0) / 1000, 1e-9), np.nan)
        r["SPF_sistem"] = np.where((el_hp + el_bu).sum(0) > 0, (QH_del + QW_del).sum(0) / 1000 / np.maximum((el_hp + el_bu).sum(0) / 1000, 1e-9), np.nan)
        r["grelnik_delez_toplote"] = np.where((QH_del + QW_del).sum(0) > 0, el_bu.sum(0) * TC["grelnik_eta"] / (QH_del + QW_del).sum(0), 0)
        r["ZEB"] = (r["Qh_nd"] <= ZEB["Qhnd_max_kWh_m2a"] + 0.5) & (r["PE_kWh_m2"] <= ZEB["EP_tot_max_SFH_kWh_m2a"])
        r["lestvica_k"], r["lestvica_w"], r["brez_ovoja_P3"], r["Qh_nd_cilj"] = lest_k, lest_w, brez_P3.astype(float), qc_nd
        rows.append(r)
        sums[v] = {"uvoz": imp.sum(1), "oddaja": exp_.sum(1), "el_skupaj": load.sum(1), "el_TC": (el_hp + el_bu).sum(1), "PV": pv.sum(1)}
        if keep_hourly:
            hourly[v] = {"uvoz": imp, "oddaja": exp_, "PV": pv, "el_TC": el_hp, "el_grelnik": el_bu, "gosp": hh, "el_EPBD": el_epbd,
                         "toplota": QH_del + QW_del}
        return r

    # V0a, V0b — kotli (TSV: V0a iz kotla, V0b električni bojler)
    QHn, QW, fr = heat["BREZ"]
    QHd = QHn / fr
    k = UM["kotli"]
    zero = np.zeros((8760, n), np.float32)
    fuel_a = (QHd + QW).sum(0) / k["ELKO"]["eta"]
    finish("V0a", "BREZ", QHn, QHd, QW, zero, zero, None, fuel_a, ZEB["f_P_tot"]["ELKO"], zero, np.zeros(n), 0, np.zeros(n), zero)
    fuel_c = (QHd + QW).sum(0) / k["plin"]["eta"]                 # krog 3 (D1): plinski kotel (TSV iz kotla)
    finish("V0c", "BREZ", QHn, QHd, QW, zero, zero, None, fuel_c, ZEB["f_P_tot"]["plin"], zero, np.zeros(n), 0, np.zeros(n), zero)
    fuel_b = QHd.sum(0) / k["biomasa"]["eta"]
    finish("V0b", "BREZ", QHn, QHd, QW, zero, zero, None, fuel_b, ZEB["f_P_tot"]["lesna_biomasa"], QW / k["tsv_V0b_elektricni_bojler_eta"],
           np.zeros(n), 0, np.zeros(n), zero)
    # V1–V3 — TČ brez ovoja (empirična moč TČ)
    kW_e = b["tc_kW"].to_numpy(np.float32)
    el, bu, sv = heat_pump(QHd, QW, kW_e, "BREZ")
    for v, kwp, bat in (("V1", np.zeros(n), 0), ("V2", b["pv_kWp_V2"].to_numpy(), 0), ("V3", b["pv_kWp_V2"].to_numpy(), BAT["kWh"])):
        finish(v, "BREZ", QHn, QHd, QW, el, bu, sv, np.zeros(n), 0, zero, kwp, bat, kW_e, zero)
    # krog 3: V2r = »OVE najprej, pravilno dimenzionirano« — TČ po urni obremenitvi hiše brez ovoja, PV po letni rabi (pravilo kot V5), ovoja nikoli
    kW_r = size_hp(QHd, QW)
    el_r, bu_r, sv_r = heat_pump(QHd, QW, kW_r, "BREZ")
    kwp_r = round_kwp(((el_r + bu_r).sum(0) / 1000 + hh.sum(0) / 1000) / PV_SPEC)
    if "pv_fiksen" in b.columns:                           # krog 4 (K4_2): PV je že nameščen in ostane, kakršen je
        kwp_r = np.where(b["pv_fiksen"].to_numpy(bool), b["pv_kWp_V2"].to_numpy(), kwp_r)
    finish("V2r", "BREZ", QHn, QHd, QW, el_r, bu_r, sv_r, np.zeros(n), 0, zero, kwp_r, 0, kW_r, zero)
    # V4–V6 — FSO, nova (manjša) TČ, PV na letno rabo po prenovi
    for ov, vs in (("FSO", ("V4", "V5", "V6")), ("GLOB", ("V7",))):
        QHn, QW, fr = heat[ov]
        QHd = QHn / fr
        kW_n = size_hp(QHd, QW)
        el, bu, sv = heat_pump(QHd, QW, kW_n, sink_fso if ov == "FSO" else ov)
        fan = fans_glob if ov == "GLOB" else zero
        kwp = round_kwp(((el + bu + fan).sum(0) / 1000 + hh.sum(0) / 1000) / PV_SPEC)
        if ov == "FSO":
            finish("V4", ov, QHn, QHd, QW, el, bu, sv, np.zeros(n), 0, zero, np.zeros(n), 0, kW_n, fan)
            finish("V5", ov, QHn, QHd, QW, el, bu, sv, np.zeros(n), 0, zero, kwp, 0, kW_n, fan)
            finish("V6", ov, QHn, QHd, QW, el, bu, sv, np.zeros(n), 0, zero, kwp, BAT["kWh"], kW_n, fan)
            # pot P2: ovoj pozneje — obstoječa TČ (empirična moč) in obstoječi PV (kWp iz V2), nato nova TČ ob zamenjavi
            kwp2 = b["pv_kWp_V2"].to_numpy()
            el8, bu8, sv8 = heat_pump(QHd, QW, kW_e, sink_fso)
            finish("V8", ov, QHn, QHd, QW, el8, bu8, sv8, np.zeros(n), 0, zero, kwp2, 0, kW_e, fan)
            finish("V9", ov, QHn, QHd, QW, el, bu, sv, np.zeros(n), 0, zero, kwp2, 0, kW_n, fan)
        else:
            finish("V7", ov, QHn, QHd, QW, el, bu, sv, np.zeros(n), 0, zero, kwp, BAT["kWh"], kW_n, fan)
    out = pd.concat(rows, ignore_index=True)
    return out, sums, hourly


# ---------------------------------------------------------------- vhodni podatki fonda
fond = pd.read_parquet(ARGS.fond if ARGS.fond else CACHE / "01_fond.parquet")
st07 = pd.read_parquet(ANALIZA / "cache" / "07_stavbe_paket.parquet", columns=["ID_Stavbe", "tip_stavbe"])
if "tip_stavbe" not in fond.columns:                              # krog 4: tabela K4_1 ima tip stavbe že (podvojeni ID s pripono)
    fond = fond.merge(st07, on="ID_Stavbe", how="left")
fond["faktor_gosp"] = np.where(fond["tip_stavbe"] == "dvostanovanjska", UM["gospodinjstvo"]["dvostanovanjska_faktor"], 1.0)
fond["gosp_idx"] = np.random.default_rng(UM["seme"] + 1).integers(0, STOH.shape[1], len(fond))   # stohastični profil stavbe
fond["A"] = fond["A_ogr"].astype(float)
if __import__("os").environ.get("GEOM", "G13") in ("G11", "G13"):   # krog 11/13: ogrevana površina = A_ogr × faktor po obdobju (kondicionirana površina izkaznic)
    fond["A"] = fond["A"] * fond["obdobje"].map(HV["povrsina_G11_faktor"]).fillna(1.0)
fond["tc_kW"] = fond["tc_kW_uporabljena"].astype(float)
med_pv = fond[fond["paket_osnovni"].astype(bool)].groupby("arhetip")["PV_kWp"].median()
fond["pv_kWp_V2"] = np.where(fond["ima_PV"].astype(bool) & fond["PV_kWp"].notna(), fond["PV_kWp"], fond["arhetip"].map(med_pv))

# Qh,nd: lastna rEI ali vzorec iz porazdelitve rEI celice (obdobje × stopnja 0 × eno_dvo), obrezano na P5–P95
# krog 13 (Gašper 22. 10. 2026): ukrep v GURS pred 2010 ni ukrep URE → umestitev po registru (Eko ∪ GURS ≥ 2010) namesto po Qh,nd (B2);
# UMESTITEV=B2 = prejšnja logika (občutljivost). Stopnja 0 izkaznice: brez ukrepa Eko ali GURS ≥ 2010 do leta izkaznice.
UMES = __import__("os").environ.get("UMESTITEV", "register")
GURS_MIN_URE = 2010
rng = np.random.default_rng(UM["seme"])
rei = pd.read_parquet(CACHE / "01_rEI_vzorec.parquet")
if UMES == "register":
    _e = pd.read_parquet(ANALIZA / "cache" / "03_stavbe_gurs_ei.parquet",
                         columns=["ID_Stavbe", "leto_prvi_fasada", "leto_prvi_streha", "leto_prvi_okna_vrata", "leto_prvi_celovita_prenova",
                                  "LETO_OBN_FASADE", "LETO_OBN_STREHE", "okna_obn_min"]).drop_duplicates("ID_Stavbe").set_index("ID_Stavbe").reindex(rei["ID_Stavbe"])
    _lei = pd.to_numeric(rei["leto_EI"], errors="coerce").to_numpy(float)
    _y = lambda c, g=False: pd.to_numeric(_e[c], errors="coerce").to_numpy(float)   # noqa: E731
    _uk = np.zeros(len(rei), bool)
    for c in ("leto_prvi_fasada", "leto_prvi_streha", "leto_prvi_okna_vrata", "leto_prvi_celovita_prenova"):
        _uk |= _y(c) <= _lei
    for c in ("LETO_OBN_FASADE", "LETO_OBN_STREHE", "okna_obn_min"):
        _uk |= (_y(c) >= GURS_MIN_URE) & (_y(c) <= _lei)
    rei = rei[~_uk & (rei["tipg"] == "eno_dvo").to_numpy()]
else:
    rei = rei[(rei["st"] == 0) & (rei["tipg"] == "eno_dvo")]
lo_p, hi_p = UM["rEI_percentili_meje"]
fond["qh"] = fond["Qhnd_lastna_EI"]
for o, g in fond[fond["qh"].isna()].groupby("obdobje"):
    v = rei.loc[rei["obdobje"] == o, "Qh_nd_EI"]
    v = v[(v >= v.quantile(lo_p / 100)) & (v <= v.quantile(hi_p / 100))].to_numpy()
    fond.loc[g.index, "qh"] = rng.choice(v, size=len(g), replace=True)
fond["qh_vir"] = np.where(fond["Qhnd_lastna_EI"].notna(), "lastna rEI", "vzorec rEI celice")
S0_QH = {a: PROF[(a, "BREZ")]["Qh_nd"] for a in ARH}


def po_registru(d):
    """Krog 13: hiše brez ukrepa po registru (fond, K4_2, sintetične) so v S0; profil S0 in cilj FSO sorazmerno skalirana na Qh,nd hiše."""
    if UMES != "register" or "stanje_var" in d.columns:
        return d
    d = d.copy()
    sk = d["qh"].astype(float).to_numpy() / d["arhetip"].map(S0_QH).astype(float).to_numpy()
    d["stanje_var"], d["cilj_var"], d["stanje_k"], d["stanje_skala"], d["cilj_skala"] = "BREZ", "FSO", 0, sk, sk
    return d


fond = po_registru(fond)
print(f"umestitev: {UMES}; vzorec izkaznic st 0: {len(rei)}")

# ---------------------------------------------------------------- (A) reprezentativne hiše
rep = []
for a in ARH:
    g = fond[fond["arhetip"] == a]
    rep.append({"ID_Stavbe": f"REPR_{a}", "arhetip": a, "obdobje": OBD_ARH[a],
                "A": float(g.loc[g["A_vir"] != "imputirano", "A_ogr"].median()),
                "qh": float(HV["arhetipi"][a]["cilj_Qhnd"]),
                "tc_kW": float(g.loc[~g["tc_kW_imputirano"].astype(bool), "tc_kW"].median()),
                "pv_kWp_V2": float(med_pv[a]), "faktor_gosp": 1.0, "gosp_idx": 0})
rep = po_registru(pd.DataFrame(rep))
rep_rows, rep_hourly = [], {}
for a in ARH:
    r, _, h = simulate(rep[rep["arhetip"] == a].reset_index(drop=True), keep_hourly=True)
    r.insert(0, "arhetip", a)
    rep_rows.append(r)
    for v, d in h.items():
        rep_hourly[(a, v)] = pd.DataFrame({k: x[:, 0] / 1000 for k, x in d.items()}, index=CAL)   # kWh per hour
repr_tab = pd.concat(rep_rows, ignore_index=True)
hr = pd.concat({f"{a}|{v}": d for (a, v), d in rep_hourly.items()}, names=["hisa_varianta", "ura"]).reset_index()
hr.to_parquet(CACHE / f"03_bilanca_reprezentativne{TAG}.parquet", index=False)
print(f"reprezentativne hiše: {time.time() - t0:.0f} s; PV donos {PV_SPEC:.0f} kWh/kWp")

# ---------------------------------------------------------------- (B) fond
CHUNK = 1000
fond_rows, fsum = [], {v: {k: np.zeros(8760) for k in ("uvoz", "oddaja", "el_skupaj", "el_TC", "PV")} for v in VARIANTE}
fsum_q = {(q, v): np.zeros(8760) for q in ("Q1", "Q2") for v in VARIANTE}
for a in ARH:
    ga = fond[fond["arhetip"] == a]
    for i in range(0, len(ga), CHUNK):
        b = ga.iloc[i:i + CHUNK].reset_index(drop=True)
        r, s, _ = simulate(b)
        r.insert(0, "ID_Stavbe", np.tile(b["ID_Stavbe"].to_numpy(), len(VARIANTE)))
        fond_rows.append(r)
        for v in VARIANTE:
            for k in fsum[v]:
                fsum[v][k] += s[v][k]
    print(f"  {a}: {len(ga)} stavb, {time.time() - t0:.0f} s")
fl = pd.concat(fond_rows, ignore_index=True)
fl = fl.merge(fond[["ID_Stavbe", "arhetip", "Qr", "obdobje", "paket_osnovni", "qh", "qh_vir", "stratum"]], on="ID_Stavbe", how="left")
fl.to_parquet(CACHE / f"03_fond_letno{TAG}.parquet", index=False)
pd.DataFrame({f"{v}|{k}": x for v, d in fsum.items() for k, x in d.items()}, index=CAL).to_parquet(CACHE / f"03_fond_urno_vsota{TAG}.parquet")

# ---------------------------------------------------------------- tabele
KOL = ["Qh_nd", "A", "TC_kW", "SCOP_TC", "SPF_sistem", "grelnik_delez_toplote", "el_TC_kWh", "el_grelnik_kWh", "el_EPBD_kWh",
       "el_skupaj_kWh", "gorivo_kWh", "PV_kWp", "PV_kWh", "uvoz_kWh", "oddaja_kWh", "uvoz_zima_kWh", "uvoz_konica_kW",
       "uvoz_top10_kW", "samozadostnost", "samoporaba", "PE_kWh_m2", "PE_z_okolico_kWh_m2", "PE_z_oddajo_kWh_m2", "ZEB",
       "baterija_kWh"] + [f"{r_}_{c_}" for r_ in ("R1", "R2", "R3") for c_ in ("energija", "oddaja", "omr_daj")] + ["R2d_energija", "R3_DOM1_kW", "R3_DOM3_kW", "CO2_el_urni_kg", "CO2_el_povp_kg", "CO2_izogib_oddaja_kg", "uvoz_konica_sis_kWh", "uvoz_konica_sis_kW"]
KOL += [f"{r_}_{c_}" for r_ in ("R1", "R2", "R3") for c_ in ("omreznina", "prispevki", "trosarina", "ddv")] + ["soc_uvoz_EUR", "soc_uvoz_zima_EUR", "soc_oddaja_EUR", "soc_oddaja_zima_EUR"]
repr_tab[["arhetip", "varianta", "ovoj"] + KOL].round(3).to_csv(TAB / f"03_energija_variante{TAG}.csv", sep=";", encoding="utf-8-sig", index=False)


def agg(g):
    return pd.Series({
        "stavb": len(g), "Qh_nd mediana": g["Qh_nd"].median(), "TČ kW mediana": g["TC_kW"].median(),
        "SCOP_TC mediana": g["SCOP_TC"].median(), "grelnik % toplote": 100 * g["grelnik_delez_toplote"].mean(),
        "el TČ+grelnik [GWh]": (g["el_TC_kWh"] + g["el_grelnik_kWh"]).sum() / 1e6, "el skupaj [GWh]": g["el_skupaj_kWh"].sum() / 1e6,
        "gorivo [GWh]": g["gorivo_kWh"].sum() / 1e6, "PV [MWp]": g["PV_kWp"].sum() / 1e3,
        "uvoz [GWh]": g["uvoz_kWh"].sum() / 1e6, "oddaja [GWh]": g["oddaja_kWh"].sum() / 1e6, "zimski uvoz [GWh]": g["uvoz_zima_kWh"].sum() / 1e6,
        "zimski uvoz mediana [kWh]": g["uvoz_zima_kWh"].median(), "konica uvoza mediana [kW]": g["uvoz_konica_kW"].median(),
        "PE mediana [kWh/m²]": g["PE_kWh_m2"].median(), "ZEB %": 100 * g["ZEB"].mean()})


fv = fl.groupby(["varianta"]).apply(agg).reindex(VARIANTE)
fq = fl.groupby(["Qr", "varianta"]).apply(agg)
fv.reset_index().assign(Qr="fond").pipe(lambda x: pd.concat([x, fq.reset_index()])).round(2).to_csv(
    TAB / f"03_fond_variante{TAG}.csv", sep=";", encoding="utf-8-sig", index=False)
sum_fleet = pd.DataFrame({v: {"konica fonda uvoz [MW]": fsum[v]["uvoz"].max() / 1e6, "konica fonda el. rabe [MW]": fsum[v]["el_skupaj"].max() / 1e6,
                              "maks. oddaja fonda [MW]": fsum[v]["oddaja"].max() / 1e6} for v in VARIANTE}).T

# ---------------------------------------------------------------- kontrola proti analizi v3 (mesečni pristop)
t10 = pd.read_csv(ANALIZA / "10_T10a_bilanca_po_Qhnd.csv", sep=";", encoding="utf-8-sig")
zcol = [c for c in t10.columns if "mediana zimski uvoz" in c]
kontrola = []
pk = fl[(fl["varianta"] == "V2") & fl["paket_osnovni"].astype(bool)]
for q in ("Q1", "Q2"):
    an = t10.loc[t10.iloc[:, 0] == q, zcol[0]].iloc[0] if zcol else np.nan
    kontrola.append({"razred": q, "zimski uvoz paketa mediana – urni model [kWh]": round(pk.loc[pk["Qr"] == q, "uvoz_zima_kWh"].median()),
                     "– analiza v3 mesečno [kWh]": an})

# ---------------------------------------------------------------- 03_povzetek.md
def mdt(df):
    df = df.copy()
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        out.append("| " + " | ".join("" if (isinstance(x, float) and np.isnan(x)) else (f"{x:,.1f}".replace(",", " ") if isinstance(x, float) else str(x)) for x in r) + " |")
    return "\n".join(out)


rt = repr_tab.copy()
rt["el TČ+grelnik"] = rt["el_TC_kWh"] + rt["el_grelnik_kWh"]
show = rt[["arhetip", "varianta", "Qh_nd", "TC_kW", "SCOP_TC", "grelnik_delez_toplote", "el TČ+grelnik", "el_skupaj_kWh", "gorivo_kWh",
           "PV_kWp", "uvoz_kWh", "oddaja_kWh", "uvoz_zima_kWh", "uvoz_konica_kW", "samozadostnost", "PE_kWh_m2", "ZEB"]].copy()
show["grelnik_delez_toplote"] *= 100
show["samozadostnost"] *= 100
show.columns = ["arh", "var", "Qh,nd", "TČ kW", "SCOP TČ", "grelnik %", "el TČ [kWh]", "el skupaj [kWh]", "gorivo [kWh]", "PV kWp",
                "uvoz [kWh]", "oddaja [kWh]", "zimski uvoz [kWh]", "konica uvoza [kW]", "samozadostnost %", "PE [kWh/m²]", "ZEB"]
md = [f"# 03 — Urni energijski model{TAG} ({time.strftime('%d. %m. %Y %H:%M')})", "",
      f"`03_urni_model.py`, parametri `parametri.yaml: urni_model, zeb`. Trajanje {time.time() - t0:.0f} s. Neoptimirano obratovanje (03b sledi).", "",
      f"- PV (pvlib, IWEC Ljubljana, 30° jug, izgube 14 %, mesečno skalirano na PVGIS-SARAH2): **{PV_SPEC:.0f} kWh/kWp** (IWEC brez skaliranja 963; analiza v3: {UM['pv']['kontrola_kWh_kWp']}).",
      f"- TČ: η = {ETA_TC:.3f} iz COP A7/W35 = {TC['COP_A7W35']}; krivulja dovoda {TC['T_dovod_projektna']} pri −13 °C → {TC['T_dovod_min']} °C pri +15 °C; "
      f"TSV {TC['T_dovod_TSV']} °C; način `{TC['nacin']}`, kapaciteta pri −13 °C {CAP_DESIGN:.0%} nazivne.",
      f"- Gospodinjska raba {UM['gospodinjstvo']['raba_kWh_a']:.0f} kWh/a, profil **{ARGS.gosp}** (H0 = BDEW dinamiziran, stoh = CREST posamezna gospodinjstva); "
      f"EV: {'da, ' + str(UM['ev']['raba_kWh_a']) + ' kWh/a, nepametno polnjenje od ' + str(UM['ev']['prihod_ura']) + 'h' if ARGS.ev else 'ne'}.",
      f"- Primarna energija: f_P,tot elektrika {ZEB['f_P_tot']['elektrika']}, okoliška toplota {'vključena' if ZEB['vkljuci_toploto_okolice'] else 'izključena'} "
      f"(občutljivost: stolpec PE_z_okolico, f = {ZEB['f_P_tot']['toplota_okolice']}); oddaja PV ni odšteta (občutljivost PE_z_oddajo, k_exp PURES); ZEB: Qh,nd ≤ {ZEB['Qhnd_max_kWh_m2a']:g} in PE ≤ {ZEB['EP_tot_max_SFH_kWh_m2a']:g} kWh/m²a.",
      "", "## (A) Reprezentativne hiše — energija po variantah", "",
      "Mediane fonda po arhetipu (A_ogr, moč TČ, kWp paketa), Qh,nd = kalibrirana vrednost celice.", "", mdt(show.round(1)), "",
      "## (B) Fond 29.977 stavb — vsote in mediane", "", mdt(fv.reset_index().round(1)), "",
      "Po razredih:", "", mdt(fq.reset_index().round(1)), "",
      "Hkratne konice fonda (vsota urnih nizov):", "", mdt(sum_fleet.reset_index().rename(columns={"index": "varianta"}).round(1)), "",
      "## Kontrola proti analizi v3 (mesečni pristop, 10_T10a)", "", mdt(pd.DataFrame(kontrola)), "",
      f"Qh,nd stavb: lastna rEI {int((fond['qh_vir'] == 'lastna rEI').sum())}, vzorec {int((fond['qh_vir'] != 'lastna rEI').sum())}; "
      f"mediana {fond['qh'].median():.0f} (Q1 {fond.loc[fond['Qr'] == 'Q1', 'qh'].median():.0f}, Q2 {fond.loc[fond['Qr'] == 'Q2', 'qh'].median():.0f}), "
      f"P10–P90 {fond['qh'].quantile(.1):.0f}–{fond['qh'].quantile(.9):.0f} kWh/m²a."]
(HERE / f"03_povzetek{TAG}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print("\n".join(md))
