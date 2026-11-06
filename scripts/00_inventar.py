"""Korak 00 — inventar: poti, HECM okolje, urni predpomnilnik, kodiranje scenarijev,
arhetipi s PV != 0 v HECM, Eko sklad EUR/m2.  Izhod: 00_inventar.md

Zagon:  "C:/Anaconda/envs/new_env/python.exe" 00_inventar.py
"""
from __future__ import annotations

import inspect
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
PAR = yaml.safe_load((HERE / "parametri.yaml").read_text(encoding="utf-8"))
P = PAR["poti"]


def pot(key: str) -> Path:
    p = Path(P[key])
    return p if p.is_absolute() else (HERE / p).resolve()


ANALIZA, HECM, CACHE = pot("analiza"), pot("hecm"), pot("hecm_cache")
md: list[str] = [f"# 00 — Inventar ({datetime.now():%d. %m. %Y %H:%M})", "",
                 "Skripta `00_inventar.py`, parametri v `parametri.yaml` (`poti`).", ""]


def ok(b) -> str:
    return "✅" if b else "❌"


def table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        out.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |")
    return "\n".join(out)


# ------------------------------------------------------------------ 1. poti
md += ["## 1. Poti", ""]
rows = []
checks = {
    "analiza v3": ANALIZA,
    "analiza common.py": ANALIZA / "common.py",
    "06_Qhnd_kalibrirano.csv": ANALIZA / "06_Qhnd_kalibrirano.csv",
    "cache/07_stavbe_paket.parquet": ANALIZA / "cache" / "07_stavbe_paket.parquet",
    "cache/04_energija_stavbe_kal.parquet": ANALIZA / "cache" / "04_energija_stavbe_kal.parquet",
    "08_T8a_RR_skupaj.csv": ANALIZA / "08_T8a_RR_skupaj.csv",
    "08_T8d_obratni_test_PV_TC.csv": ANALIZA / "08_T8d_obratni_test_PV_TC.csv",
    "09_T9a_PV_kWp_po_Qhnd.csv": ANALIZA / "09_T9a_PV_kWp_po_Qhnd.csv",
    "09_T9d_TC_Wm2_Qhnd_x_obdobje.csv": ANALIZA / "09_T9d_TC_Wm2_Qhnd_x_obdobje.csv",
    "10_T10a_bilanca_po_Qhnd.csv": ANALIZA / "10_T10a_bilanca_po_Qhnd.csv",
    "11_T11c_nadomescen_energent_po_Qhnd_koda.csv": ANALIZA / "11_T11c_nadomescen_energent_po_Qhnd_koda.csv",
    "HECM (Claude code - the method)": HECM,
    "HECM README.md": HECM / "README.md",
    "HECM docs/CHANGELOG.md": HECM / "docs" / "CHANGELOG.md",
    "HECM PREGLED_HECM_2026-08-15.md": HECM / "PREGLED_HECM_2026-08-15.md",
    "HECM hecm/heatpump.py": HECM / "hecm" / "heatpump.py",
    "HECM hecm/configs/SI.yaml": HECM / "hecm" / "configs" / "SI.yaml",
    "HECM Inputs/SI": HECM / "Inputs" / "SI",
    "HECM Results/SI/annual_SI.parquet": HECM / "Results" / "SI" / "annual_SI.parquet",
    "HECM docs/izhodisca_ovoja_SI.xlsx": HECM / "docs" / "izhodisca_ovoja_SI.xlsx",
    "conda new_env python": pot("hecm_python"),
    "urni predpomnilnik .hecm_cache/SI": CACHE,
    "vreme EPW (IWEC Ljubljana)": pot("epw"),
    "Eko EUR/m2 (P:, trajna)": pot("eko_eur_m2"),
    "Eko EUR/m2 (kopija GreenRenoV8)": pot("eko_eur_m2_kopija"),
}
for k, p in checks.items():
    rows.append({"kaj": k, "obstaja": ok(p.exists()), "pot": f"`{p}`"})
md += [table(pd.DataFrame(rows)), "",
       "Opomba: CLAUDE.md §2.2 navaja HECM pod `D:\\Users\\gaspers\\...`; na tem računalniku te poti ni, "
       "HECM je pod `C:\\Users\\Gašper\\...` (zapisano v `parametri.yaml`).", ""]

# ------------------------------------------------------------------ 2. HECM okolje
md += ["## 2. HECM okolje (`new_env`)", ""]
py = str(pot("hecm_python"))
env_rows = []
for args in (["-m", "hecm", "--help"], ["-m", "hecm", "selftest"]):
    r = subprocess.run([py, *args], cwd=HECM, capture_output=True, text=True, encoding="utf-8", errors="replace")
    last = (r.stdout.strip().splitlines() or r.stderr.strip().splitlines() or [""])[-1]
    env_rows.append({"ukaz": "`python " + " ".join(args) + "`", "izhodna koda": r.returncode, "zadnja vrstica": last[:120]})
r = subprocess.run([py, "-c", "import sys; print(sys.version.split()[0])"], capture_output=True, text=True)
md += [f"Python v `new_env`: {r.stdout.strip()}", "", table(pd.DataFrame(env_rows)), ""]

pk = subprocess.run([py, "-c",
                     "import importlib\n"
                     "for m in ['pandas','numpy','pyarrow','openpyxl','yaml','scipy','matplotlib','pvlib','linopy','pyomo','highspy','statsmodels']:\n"
                     "    try:\n        mod=importlib.import_module(m); print(m, getattr(mod,'__version__','?'))\n"
                     "    except Exception: print(m, 'MANJKA')"],
                    capture_output=True, text=True).stdout.split("\n")
pk = [l.split(" ", 1) for l in pk if l.strip()]
md += ["Knjižnice v `new_env`:", "", table(pd.DataFrame(pk, columns=["paket", "verzija"])), "",
       "Za korak 03b (LP) manjkata `linopy`/`pyomo` in `highspy` — pred korakom 03b jih je treba namestiti "
       "(npr. `pip install highspy linopy` v `new_env`); scipy `linprog(method='highs')` je rezerva.", ""]

sys.path.insert(0, str(HECM))
from hecm import heatpump  # noqa: E402

sig = inspect.signature(heatpump.electricity_and_backup)
hp_cfg = yaml.safe_load((HECM / "hecm" / "configs" / "defaults.yaml").read_text(encoding="utf-8")).get("heat_pump", {})
md += [f"`hecm.heatpump.electricity_and_backup{sig}` — uvoz uspešen. Privzete nastavitve TČ (`defaults.yaml: heat_pump`):", "",
       "```yaml", yaml.safe_dump(hp_cfg, allow_unicode=True, sort_keys=False).strip(), "```", ""]

# ------------------------------------------------------------------ 3. urni predpomnilnik
md += ["## 3. Urni predpomnilnik", ""]
dflt = yaml.safe_load((HECM / "hecm" / "configs" / "defaults.yaml").read_text(encoding="utf-8"))
pq = sorted(CACHE.glob("*.parquet"))
js = sorted(CACHE.glob("*.json"))
ann = pd.read_parquet(HECM / "Results" / "SI" / "annual_SI.parquet")
ann["v_cache"] = ann["signature"].map(lambda s: (CACHE / f"{s}.parquet").exists())
sfh = ann[ann["building_type"] == "SFH"].copy()
smp = pd.read_parquet(CACHE / f"{sfh.iloc[0]['signature']}.parquet")
md += [f"- `cache_dir` v `defaults.yaml`: `{dflt['cache_dir']}` → `{CACHE}`",
       f"- datotek: {len(pq)} parquet + {len(js)} json (json = letni povzetek; imena = podpis `signature` iz `annual_SI.parquet`; "
       "ime hiše v json ni — povezava hiša → podpis gre prek `annual_SI.parquet`).",
       f"- SI vse stavbe: {len(ann)} vrstic, {ann['signature'].nunique()} različnih podpisov, v predpomnilniku {ann['v_cache'].mean():.0%}.",
       f"- SI SFH: {len(sfh)} hiš, {sfh['signature'].nunique()} podpisov, v predpomnilniku {sfh['v_cache'].sum()} / {len(sfh)}.",
       f"- urni niz: {smp.shape[0]} ur, stolpci `{', '.join(smp.columns)}`, indeks {smp.index[0]} … {smp.index[-1]} (Q v W, T v °C, G_h v W/m²).",
       f"- dostop: `python -m hecm inspect --country SI --house <ime> --csv out.csv` ali neposredno `{CACHE}/<signature>.parquet`.",
       "- `.hecm_cache/SI` se v tem projektu samo bere; novi vhodi `CLK-*` dobijo svoje podpise (CLAUDE.md §12).", ""]

# ------------------------------------------------------------------ 4. kodiranje scenarijev
md += ["## 4. Kodiranje scenarijev SI (SFH)", ""]


def read_txt(p: Path) -> dict:
    d = {}
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            d[k.strip()] = v.strip()
    return d


def first_u(d: dict, typ: str) -> float | None:
    t = d["typology_elements"].split(",")
    u = d["transmittance_U_elements"].split(",")
    o = d["orientation_elements"].split(",")
    for ti, ui, oi in zip(t, u, o):
        if ti == typ and (typ != "OP" or oi != "HOR"):
            return float(ui)
    return None


def roof_u(d: dict) -> float | None:
    t, u, o = (d[k].split(",") for k in ("typology_elements", "transmittance_U_elements", "orientation_elements"))
    for ti, ui, oi in zip(t, u, o):
        if ti == "OP" and oi == "HOR":
            return float(ui)
    return None


inp = HECM / "Inputs" / "SI"
recs = []
for p in sorted(inp.glob("*.txt")):
    d = read_txt(p)
    recs.append({"house": p.stem, "PV_kWh_vhod": float(d.get("Annual Energy Production", "0") or 0),
                 "U_stena": first_u(d, "OP"), "U_okno": first_u(d, "W"), "U_strop": roof_u(d), "U_tla": first_u(d, "GR"),
                 "n50": float(d.get("airtightness_n50", "nan")), "HR": float(d.get("heat_recovery_efficiency", "nan")),
                 "a_use": float(d.get("a_use", "nan")), "P_ogr_max_W": float(d.get("power_heating_max", "nan")),
                 "ogrevanje": d.get("Heating system – Technology", "")})
inp_df = pd.DataFrame(recs)
sfh = sfh.merge(inp_df, on="house", how="left")
sfh["arhetip"] = sfh["archetype"].str.replace(r"^\d+-", "", regex=True).str.replace("-EXB", "", regex=False)
sfh["paket"] = sfh["scenario"].str.split("-").str[0]
sfh["razred_koda"] = sfh["scenario"].str.split("-").str[1]
sfh["sistem"] = sfh["scenario"].str.split("-").str[2]
RAZ = PAR["hecm"]["razredi_scenarijev"]
sfh["razred"] = sfh["razred_koda"].map(RAZ)

s01 = sfh[sfh["sistem"] == "02"]  # isti ovoj v vseh sistemih; prikaži -02 (TČ zrak-voda)
piv = s01.pivot_table(index="razred_koda", columns="arhetip", values="Q_H_kWh_m2a").round(0)
piv.insert(0, "razred", piv.index.map(RAZ))
mono = {a: bool(piv[a].iloc[1:].is_monotonic_increasing) for a in piv.columns if a != "razred"}
u_tab = (s01[s01["arhetip"] == "SI-SFH-1966-1981"]
         .set_index("razred_koda")[["razred", "U_stena", "U_okno", "U_strop", "U_tla", "n50", "HR", "ach_total_1_h", "Q_H_kWh_m2a"]]
         .round(2))
md += ["Ime: `<id>-SI-SFH-<obdobje>-EXB-<paket>-<razred>-<sistem>`; v `annual_SI.parquet` stolpec `scenario` = `<paket>-<razred>-<sistem>`.",
       f"- `<paket>`: v vseh SFH vhodih samo {sorted(sfh['paket'].unique())}.",
       "- `<razred>`: `hecm/configs/SI.yaml` — »00 → EPC class 0 (the un-renovated state), 10 → A1 … 90 → G«. "
       "Vmesne kode po vrstnem redu SI razredov EI: 10 A1, 20 A2, 30 B1, 40 B2, 50 C, 60 D, 70 E, 80 F, 90 G.",
       f"- `<sistem>`: {', '.join(f'-{k}: ' + ', '.join(sorted(set(v))) for k, v in sfh.groupby('sistem')['ogrevanje'])}.",
       "",
       "**Empirična potrditev** — Qh,nd [kWh/m²a] iz `annual_SI.parquet` (sistem -02; ovoj je enak v vseh sistemih):", "",
       table(piv.reset_index()), "",
       f"Qh,nd narašča monotono od 10 (A1) do 90 (G) pri vseh arhetipih: {all(mono.values())} ({mono}). "
       "Kodiranje 10–90 → A1 … G je potrjeno.", "",
       "U-vrednosti in zrak po razredih, `1966-1981` (osnova za `CLK-SI-SFH-Q1-*`):", "",
       table(u_tab.reset_index()), "",
       "- Scenarij `00` ima mehansko prezračevanje z rekuperacijo in hkrati najvišjo Qh,nd — nekonsistenten, **ne uporabljamo** (CLAUDE.md §3.1).",
       "- Razred `90` (G) arhetipa 1966-1981 ima U-vrednosti, ki so blizu realni neprenovljeni hiši (stena 1,1, okna 3,0) — "
       "dober izhodiščni vhod za korak 02; Qh,nd pa je precej nad kalibriranim rEI (146).", ""]

cap = s01[s01["arhetip"].isin(["SI-SFH-1966-1981", "SI-SFH-1982-2008"]) & s01["razred_koda"].isin(["50", "60", "70", "80", "90"])].copy()
pk_rows = []
for r in cap.itertuples():
    x = pd.read_parquet(CACHE / f"{r.signature}.parquet")
    qh = x["Q_H"]
    pk_rows.append({"arhetip": r.arhetip, "razred": r.razred, "Qh_nd": round(r.Q_H_kWh_m2a),
                    "meja_W (200 W/m² × a_use)": round(r.heating_power_max_W),
                    "Q_H maks [W]": round(qh.max()), "P99,9 [W]": round(qh.quantile(0.999)),
                    "maks [W/m²]": round(qh.max() / r.a_use_m2), "T_ext ob maks": x.loc[qh.idxmax(), "T_ext"],
                    "ur na meji": int((qh >= 0.999 * r.heating_power_max_W).sum()), "ur T_op < 19 °C": int((x["T_op"] < 19).sum())})
md += ["**Moč ogrevanja v HECM (za korak 02):** `power_heating_max` iz vhoda (10 kW) se ne uporablja; HECM omeji moč s "
       "`power_limit_W_m2` × `a_use` = 200 W/m². Stolpec `heating_power_max_W` v `annual` je **ta meja, ne dejanska konica** "
       "(`heating_hours_at_power_cap` = 0). Dejanske konice iz urnih nizov (sistem -02):", "",
       table(pd.DataFrame(pk_rows)), "",
       "Konice nastopijo pri T_ext ≈ −20 °C (najhladnejša ura IWEC), ne ob jutranjem dvigu po setbacku; meja ne reže, udobje je "
       "ves čas zagotovljeno. V koraku 02 za dimenzioniranje TČ uporabi P99,9 urne Q_H (in preveri, da maks < meja).", ""]

# ------------------------------------------------------------------ 5. PV v HECM
md += ["## 5. Arhetipi, katerim HECM odšteva PV (`Annual Energy Production` ≠ 0)", ""]
allpv = ann.merge(inp_df[["house", "PV_kWh_vhod"]], on="house", how="left")
pvnz = allpv[allpv["PV_kWh_vhod"] > 0]
sfh_pv = (sfh[sfh["PV_kWh_vhod"] > 0]
          .groupby(["arhetip", "sistem"])
          .agg(scenariji=("razred_koda", lambda s: ",".join(sorted(s))), PV_kWh_a=("PV_kWh_vhod", "first"),
               PV_annual_kWh_HECM=("PV_annual_kWh", "first"),
               PV_izvoz_kWh_mediana=("PV_exported_kWh", "median"),
               el_pred_PV_kWh_mediana=("TOTAL_electricity_before_PV_kWh", "median"),
               el_po_PV_kWh_mediana=("TOTAL_electricity_kWh", "median"))
          .round(0).reset_index())
by_type = (pvnz.groupby("building_type").agg(stavb=("house", "size"),
                                              PV_kWh_min=("PV_kWh_vhod", "min"), PV_kWh_max=("PV_kWh_vhod", "max"))
           .reset_index())
md += [f"Vseh SI vhodov s PV ≠ 0: {len(pvnz)} od {len(ann)}; po tipih:", "", table(by_type), "",
       "**SI SFH s PV ≠ 0** (arhetip × sistem; scenariji = razredi s PV):", "",
       table(sfh_pv) if len(sfh_pv) else "_nobeden_", "",
       "HECM ta PV **letno** odšteje od elektrike (`systems.py` ~v. 580–663: "
       "`TOTAL_electricity_kWh = max(TOTAL_electricity_before_PV_kWh − PV, 0)`), višek gre v `PV_exported_kWh` (`indicators.py`). "
       "To je implicitni letni net metering. **Za WP3 (samo zapis, brez popravka):** pri teh arhetipih so deleži OVE precenjeni in "
       "raba elektrike (`TOTAL_electricity_kWh`) podcenjena. V tem projektu se PV iz HECM ne uporablja; vhodi `CLK-*` imajo "
       "`Annual Energy Production=0`.", ""]

# ------------------------------------------------------------------ 6. Eko sklad EUR/m2
md += ["## 6. Eko sklad EUR/m² (`eko_eur_m2.xlsx`)", ""]
eko_rows = []
for key in ("eko_eur_m2", "eko_eur_m2_kopija"):
    p = pot(key)
    rec = {"datoteka": key, "obstaja": ok(p.exists())}
    if p.exists():
        rec["spremenjena"] = f"{datetime.fromtimestamp(p.stat().st_mtime):%d. %m. %Y %H:%M}"
        try:
            xl = pd.ExcelFile(p)
            rec["listi"] = ", ".join(xl.sheet_names)
            nal = pd.read_excel(p, sheet_name="Nalozbe")
            rec["Nalozbe vrstic"] = len(nal)
            rec["berljiv"] = ok(True)
        except Exception as e:  # noqa: BLE001
            rec["berljiv"] = f"❌ {e}"
    eko_rows.append(rec)
md += [table(pd.DataFrame(eko_rows)), ""]
p = pot("eko_eur_m2")
nal = pd.read_excel(p, sheet_name="Nalozbe")
med = pd.read_excel(p, sheet_name="Mediana EUR_m2")
md += [f"Uporabljamo trajno verzijo na P: (memory: kopija v GreenRenoV8 je zastarela; razlika v velikosti "
       f"{pot('eko_eur_m2').stat().st_size - pot('eko_eur_m2_kopija').stat().st_size:+d} B).", "",
       f"List `Nalozbe`: {len(nal):,} vrstic; stolpci: `{', '.join(map(str, nal.columns))}`.".replace(",", ".", 1), "",
       f"`V statistiki` = True: {int(nal['V statistiki'].astype(bool).sum()):,} vrstic "
       f"(CLAUDE.md §2.3 navaja 78.483 naložb s stroškom, površino in količino); po tipih: "
       f"{nal.loc[nal['V statistiki'].astype(bool), 'Tip'].value_counts().to_dict()}; s stroškom, površino in količino hkrati: "
       f"{int(nal[['Priznani stroški [EUR]', 'Ogrevana uporabna površina [m²]', 'Količina ukrepa']].notna().all(axis=1).sum()):,}.", "",
       "List `Mediana EUR_m2` (prvih 15 vrstic):", "", table(med.head(15).round(1)), "",
       "**Past:** stolpec »Povprečje 2023–2025« je Excelova formula brez shranjene vrednosti (pandas/openpyxl jo bere kot 0). "
       "Uporabljaj stolpec »2023–2025 skupaj (vse naložbe)« (to so vrednosti v CLAUDE.md §2.3: fasada 158, streha 97, okna 90, "
       "TČ zrak-voda 84) ali računaj neposredno iz lista `Nalozbe`.", ""]

# ------------------------------------------------------------------ 7. ugotovitve za WP3 (samo zapis)
md += ["## 7. Ugotovitve za WP3 (samo zapis, v GreenRenoV8 se nič ne popravlja)", ""]
TABG = PAR["hecm_vhodi"]["geometrija"]
geo = []
for p in sorted(inp.glob("*-SI-SFH-*-EXB-00-90-01.txt")):
    dd = read_txt(p)
    ar = [float(x) for x in dd["area_elements"].split(",")]
    tla, streha, okna = ar[0], ar[9], sum(ar[5:9])
    best = min(TABG, key=lambda k: abs(TABG[k]["tla"] - tla) + abs(TABG[k]["streha"] - streha) + abs(TABG[k]["okna"] - okna))
    geo.append({"HECM arhetip": p.stem.split("-SI-SFH-")[1].split("-EXB")[0], "a_use": dd["a_use"], "tla": tla, "streha": streha,
                "okna": okna, "najbližja TABULA": f"{best} {TABG[best]['vir']}"})
hr = ann[(ann["building_type"] == "SFH") & (ann["hr_factor"] < 1)]
from hecm.config import load_config  # noqa: E402
_air = load_config("SI").airflow
si_air = {"min_total_ach": _air.min_total_ach, "storey_height_source": f"{_air.storey_height_source} ({_air.storey_height_default} m)"}
md += ["1. **Geometrija SI SFH je zamaknjena za eno obdobje.** Površine elementov HECM arhetipov (razred G, sistem -01) ujemajo "
       "s primeri TABULA SI drugega obdobja:", "", table(pd.DataFrame(geo)), "",
       "   1896-1920 ima geometrijo TABULA 1946–70 (a_use 190 = polovica 380 m²), 1946-1965 geometrijo TABULA 1971–80, "
       "od 1966-1981 naprej pa vsi pritlično hišo TABULA ≤1945 (streha in tla 226 m² pri 151 m²; TABULA Qh,nd 516). Zato je "
       "Qh,nd razreda G pri 1966-1981 391 in pri 1946-1965 le 210 kWh/m²a. Ta projekt uporablja TABULA geometrijo po obdobju "
       "(glej 02_kalibracija.md).", "",
       f"2. **Rekuperacija brez učinka:** `airflow.min_total_ach` = {si_air['min_total_ach']} (efektivno v SI) se v `hecm/airflow.py` "
       "uporabi na izmenjavi PO rekuperaciji (`q_base = hr_factor·q_vent + q_infil`, nato `max(q_base, floor·V)`). "
       f"Pri {len(hr)} SFH vhodih z rekuperacijo (hr_factor < 1) je efektivna izmenjava "
       f"{', '.join(f'{v:.2f}' for v in sorted(hr['ach_total_1_h'].round(2).unique()))} h⁻¹ — meja preglasi rekuperacijo, "
       "zato razredi s HR (A1, A2, B1) nimajo prihranka prezračevalnih izgub. Spodnja meja bi morala veljati za pretok zraka, ne za toplotno izgubo. "
       "V tem projektu je za CLK-* preglašena na 0 (`parametri.yaml: hecm_vhodi.cfg_preglasitve`).", "",
       f"3. **Višina etaže:** `airflow.storey_height_source` = {si_air['storey_height_source']}, "
       "volumen = a_use × 2,8 m ne glede na `building_height` v vhodu.", ""]

# ------------------------------------------------------------------ 8. kontrolne številke
md += ["## 8. Kontrolne številke", ""]
st = pd.read_parquet(ANALIZA / "cache" / "07_stavbe_paket.parquet")
md += [f"- `07_stavbe_paket.parquet`: {len(st):,} eno-/dvostanovanjskih stavb s TČ, {st.shape[1]} stolpcev.",
       f"- HECM SI: {len(ann)} stavb, SFH {len(sfh)}, SFH s PV ≠ 0: {int((sfh['PV_kWh_vhod'] > 0).sum())}.",
       f"- urni predpomnilnik: {len(pq)} parquet."]
out = HERE / "00_inventar.md"
out.write_text("\n".join(md) + "\n", encoding="utf-8")
print("\n".join(md[-4:]))
print(f"zapisano: {out}")
