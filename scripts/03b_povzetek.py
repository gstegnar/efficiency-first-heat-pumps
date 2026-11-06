"""Korak 03b — povzetek: prihranek optimiranja (V1–V7) in mreža velikosti po ovoju.

Vhodi: tabele/03b_prihranek_optimiranja_surovo.csv, tabele/03b_mreza.csv, tabele/03b_H_C.csv
Izhodi: tabele/03b_prihranek_optimiranja.csv, tabele/03b_konica_po_ovoju.csv, tabele/03b_mejna_vrednost.csv, 03b_povzetek.md
Optimalne velikosti po NPV se izberejo v koraku 04 (stroški); tu je energijska in računska stran (R2 s fiksnima cenama).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
TAB = HERE / "tabele"
PAR = yaml.safe_load((HERE / "parametri.yaml").read_text(encoding="utf-8"))
OP = PAR["optimizacija"]
OV = ["BREZ", "F", "FS", "FSO", "GLOB"]
rd = lambda f: pd.read_csv(TAB / f, sep=";", encoding="utf-8-sig")  # noqa: E731
v = rd("03b_prihranek_optimiranja_surovo.csv")
g = rd("03b_mreza.csv")
hc = rd("03b_H_C.csv")
bad = pd.concat([v, g])
bad = bad[bad["status"] != "ok"]


def mdt(df):
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        out.append("| " + " | ".join("" if (isinstance(x, float) and np.isnan(x)) else (f"{x:,.1f}".replace(",", " ") if isinstance(x, float) else str(x)) for x in r) + " |")
    return "\n".join(out)


# ---------------------------------------------------------------- 1. prihranek optimiranja po variantah
K = ["uvoz_kWh", "uvoz_zima_kWh", "uvoz_konica_kW", "racun_R2_EUR", "PE_kWh_m2", "samozadostnost"]
p = v.pivot_table(index=["arhetip", "varianta", "ovoj", "EV"], columns="cilj", values=K, aggfunc="first")
rows = []
for idx, r in p.iterrows():
    rows.append({"arhetip": idx[0], "varianta": idx[1], "ovoj": idx[2], "EV": idx[3],
                 "račun pravila [EUR/a]": r[("racun_R2_EUR", "pravila")], "račun O1 [EUR/a]": r[("racun_R2_EUR", "O1")],
                 "prihranek O1 [EUR/a]": r[("racun_R2_EUR", "pravila")] - r[("racun_R2_EUR", "O1")],
                 "prihranek O1 [%]": 100 * (1 - r[("racun_R2_EUR", "O1")] / r[("racun_R2_EUR", "pravila")]),
                 "konica pravila [kW]": r[("uvoz_konica_kW", "pravila")], "konica O1 [kW]": r[("uvoz_konica_kW", "O1")],
                 "konica O2 [kW]": r[("uvoz_konica_kW", "O2")],
                 "znižanje konice O2 [%]": 100 * (1 - r[("uvoz_konica_kW", "O2")] / r[("uvoz_konica_kW", "pravila")]),
                 "zimski uvoz pravila [kWh]": r[("uvoz_zima_kWh", "pravila")], "zimski uvoz O1 [kWh]": r[("uvoz_zima_kWh", "O1")],
                 "PE pravila": r[("PE_kWh_m2", "pravila")], "PE O1": r[("PE_kWh_m2", "O1")]})
po = pd.DataFrame(rows)
po.round(2).to_csv(TAB / "03b_prihranek_optimiranja.csv", sep=";", encoding="utf-8-sig", index=False)

# ---------------------------------------------------------------- 2. konica po ovoju (O2) in vloga baterije / hranilnika
k8 = g[(g["kWp"] == 8) & (g["cilj"].isin(["O2", "pravila"]))]
kon = (k8.pivot_table(index=["arhetip", "EV", "ovoj", "baterija_kWh", "hranilnik_L"], columns="cilj", values="uvoz_konica_kW")
       .reset_index())
kon.round(2).to_csv(TAB / "03b_konica_po_ovoju.csv", sep=";", encoding="utf-8-sig", index=False)

# ---------------------------------------------------------------- 3. mejna letna vrednost baterije, hranilnika in PV (O1, R2)
o1 = g[g["cilj"] == "O1"].set_index(["arhetip", "EV", "ovoj", "kWp", "baterija_kWh", "hranilnik_L"])["racun_R2_EUR"]
mv = []
for (a, ev, ov), _ in o1.groupby(level=[0, 1, 2]):
    s = o1.loc[(a, ev, ov)]
    base = s.loc[(8, 0, 0)]
    mv.append({"arhetip": a, "EV": ev, "ovoj": ov,
               "račun 8 kWp brez bat/hran [EUR/a]": base,
               "PV 0 → 8 kWp [EUR/a]": s.loc[(0, 0, 0)] - base,
               "PV 8 → 12,5 kWp [EUR/a]": base - s.loc[(12.5, 0, 0)],
               "baterija 10 kWh pri 8 kWp [EUR/a]": base - s.loc[(8, 10, 0)],
               "hranilnik 800 L pri 8 kWp [EUR/a]": base - s.loc[(8, 0, 800)],
               "hranilnik 800 L brez PV [EUR/a]": s.loc[(0, 0, 0)] - s.loc[(0, 0, 800)],
               "bat 10 + hran 800 pri 8 kWp [EUR/a]": base - s.loc[(8, 10, 800)]})
mv = pd.DataFrame(mv)
mv["ovoj"] = pd.Categorical(mv["ovoj"], OV, ordered=True)
mv = mv.sort_values(["arhetip", "EV", "ovoj"])
mv.round(1).to_csv(TAB / "03b_mejna_vrednost.csv", sep=";", encoding="utf-8-sig", index=False)

# zimski uvoz: koliko ga optimiranje + baterija + hranilnik sploh zmanjšajo (O1, 8 kWp)
zi = g[(g["kWp"] == 8) & (g["cilj"].isin(["O1", "pravila"]))].pivot_table(
    index=["arhetip", "EV", "ovoj"], columns=["cilj", "baterija_kWh", "hranilnik_L"], values="uvoz_zima_kWh")
zt = pd.DataFrame({"zimski uvoz pravila, brez bat/hran": zi[("pravila", 0, 0)], "O1 brez bat/hran": zi[("O1", 0, 0)],
                   "O1 + bat 15 + hran 1500": zi[("O1", 15, 1500)]}).reset_index()
zt["znižanje [%]"] = 100 * (1 - zt["O1 + bat 15 + hran 1500"] / zt["zimski uvoz pravila, brez bat/hran"])
zt["ovoj"] = pd.Categorical(zt["ovoj"], OV, ordered=True)
zt = zt.sort_values(["arhetip", "EV", "ovoj"])

# ---------------------------------------------------------------- md
pk = po[~po["EV"]][["arhetip", "varianta", "ovoj", "račun pravila [EUR/a]", "prihranek O1 [EUR/a]", "prihranek O1 [%]",
                     "konica pravila [kW]", "konica O1 [kW]", "konica O2 [kW]", "znižanje konice O2 [%]", "PE pravila", "PE O1"]]
pe = po[po["EV"]][["arhetip", "varianta", "račun pravila [EUR/a]", "prihranek O1 [EUR/a]", "konica pravila [kW]", "konica O2 [kW]",
                    "znižanje konice O2 [%]"]]
kk = kon[(kon["baterija_kWh"].isin([0, 10])) & (kon["hranilnik_L"].isin([0, 800]))].copy()
kk["ovoj"] = pd.Categorical(kk["ovoj"], OV, ordered=True)
kk = kk.sort_values(["arhetip", "EV", "ovoj", "baterija_kWh", "hranilnik_L"])
md = [f"# 03b — Optimirano obratovanje in mreža velikosti ({time.strftime('%d. %m. %Y %H:%M')})", "",
      "LP 8.760 h (scipy HiGHS), stavba kot hranilnik (1R1C, C = "
      f"{OP['C_eff_J_m2K'] / 1000:.0f} kJ/m²K, pas {OP['komfort_pas_K']:g} K), hranilnik TSV/ogrevanja, baterija, EV (okno "
      f"{OP['ev_okno']['od']}–{OP['ev_okno']['do']} h). O1 = račun R2 s fiksnima cenama {OP['cena_uvoz_EUR_kWh']} / "
      f"{OP['cena_oddaja_EUR_kWh']} EUR/kWh (urne cene in R3 v koraku 04); O2 = min. maksimalni urni uvoz. "
      "»Pravila« = neoptimirano kot v koraku 03.", "",
      f"Neuspešnih rešitev: {len(bad)}.", "",
      "## Časovna konstanta stavbe τ = C/H [h] (hranilnik v stavbi pri pasu 2 K: "
      f"{hc['hranilnik_stavbe_kWh_na_100m2'].iloc[0]:.1f} kWh_th na 100 m²)", "",
      mdt(hc.pivot_table(index="arhetip", columns="ovoj", values="tau_h")[OV].round(0).reset_index()), "",
      "## 1. Prihranek optimiranja po variantah (reprezentativne hiše, brez EV)", "", mdt(pk.round(1)), "",
      "Z EV:", "", mdt(pe.round(1)), "",
      "## 2. Konica uvoza hiše po ovoju — O2 proti pravilom (8 kWp)", "", mdt(kk.round(2)), "",
      "## 3. Mejna letna vrednost (O1, R2) po ovoju — vhod za NPV v koraku 04", "", mdt(mv.round(0)), "",
      "## 4. Zimski uvoz (nov–feb): koliko ga zmanjšajo optimiranje, baterija in hranilnik (8 kWp)", "", mdt(zt.round(0))]
(HERE / "03b_povzetek.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print("\n".join(md))
