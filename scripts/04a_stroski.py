"""Korak 04a — stroškovne funkcije iz Eko skladovih naložb (SFH, 2023–2025, nominalno z DDV).

- TČ zrak-voda (tczv), TČ zemlja-voda (tczemlja), PV (fotovoltaika): strošek = a + b·kW, robustno (Theil–Sen na naključnem
  vzorcu, seme iz parametri.yaml), 95 % IZ naklona; samo enota »kw«, kW in strošek v razumnem razponu.
- Ovoj: mediana EUR na enoto (m² fasade / strehe-stropa / oken / tal) in EUR na m² ogrevane površine (občutljivost).
- Prezračevanje centralno: mediana EUR na napravo.
- Delež spodbude: mediana spodbuda / priznani stroški.
Izhod: tabele/04a_stroskovne_funkcije.csv, tabele/04a_ovoj_enote.csv, 04a_stroski.md
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import theilslopes

HERE = Path(__file__).resolve().parent
PAR = yaml.safe_load((HERE / "parametri.yaml").read_text(encoding="utf-8"))
TAB = HERE / "tabele"
n = pd.read_excel(PAR["poti"]["eko_eur_m2"], sheet_name="Nalozbe")
n = n[(n["Tip"] == "SFH") & (n["V statistiki"] == "DA")]          # krog 11: »DA«/»NE« — astype(bool) je bil vedno True
rng = np.random.default_rng(PAR["urni_model"]["seme"])
COST, SUB, AREA, Q = "Priznani stroški [EUR]", "Spodbuda [EUR]", "Ogrevana uporabna površina [m²]", "Količina ukrepa"

rows = []
for kod, kw_lo, kw_hi in (("tczv", 3, 25), ("tczemlja", 4, 30), ("fotovoltaika", 2, 25)):
    s = n[(n["Ukrep (koda)"] == kod) & (n["Enota"] == "kw") & n[Q].between(kw_lo, kw_hi) & (n[COST] > 0)]
    s = s[s[COST].between(s[COST].quantile(0.01), s[COST].quantile(0.99))]
    smp = s.sample(min(4000, len(s)), random_state=int(rng.integers(1e9)))
    b, a, lo, hi = theilslopes(smp[COST], smp[Q])
    for leto, g in [("2023–2025", s)] + list(s.groupby("Leto")):
        rows.append({"ukrep": kod, "leto": leto, "n": len(g), "a_EUR": round(a), "b_EUR_kW": round(b), "b_IZ95": f"{lo:.0f}–{hi:.0f}" if leto == "2023–2025" else "",
                     "kW_mediana": g[Q].median(), "strosek_mediana": round(g[COST].median()),
                     "EUR_kW_mediana": round((g[COST] / g[Q]).median()), "delez_spodbude_mediana": round((g[SUB] / g[COST]).median(), 3)})
sf = pd.DataFrame(rows)
sf.to_csv(TAB / "04a_stroskovne_funkcije.csv", sep=";", encoding="utf-8-sig", index=False)

ov = []
for kod, enota in (("fasada", "m2"), ("streha", "m2"), ("okna_vrata", "m2"), ("talna_izolacija", "m2"), ("kletna_izolacija", "m2"),
                   ("prezrac_centr", "kos"), ("prezrac_centr", "kom")):
    s = n[(n["Ukrep (koda)"] == kod) & (n["Enota"] == enota) & (n[COST] > 0)]
    if not len(s):
        continue
    ov.append({"ukrep": kod, "enota": enota, "n": len(s),
               "EUR_na_enoto_P25": (s[COST] / s[Q]).quantile(0.25), "EUR_na_enoto_mediana": (s[COST] / s[Q]).median(),
               "EUR_na_enoto_P75": (s[COST] / s[Q]).quantile(0.75),
               "EUR_m2_ogrevane_mediana": (s[COST] / s[AREA]).median(), "kolicina_na_m2_ogrevane_mediana": (s[Q] / s[AREA]).median(),
               "strosek_mediana": s[COST].median(), "delez_spodbude_mediana": (s[SUB] / s[COST]).median()})
ov = pd.DataFrame(ov).round(3)
ov.to_csv(TAB / "04a_ovoj_enote.csv", sep=";", encoding="utf-8-sig", index=False)

md = ["# 04a — Stroškovne funkcije (Eko sklad, SFH, 2023–2025, nominalno z DDV)", "",
      "Vir: `eko_eur_m2.xlsx` (list Nalozbe, `V statistiki`), P: trajna verzija. Theil–Sen na vzorcu ≤ 4.000 naložb, odrezan 1. in 99. percentil stroška.", "",
      "## TČ in PV: strošek = a + b·kW", "", sf.to_string(index=False), "",
      "## Ovoj in prezračevanje", "", ov.to_string(index=False), "",
      "Opomba: količina ukrepa na m² ogrevane površine (stolpec `kolicina_na_m2_ogrevane_mediana`) pri fasadi kaže, koliko m² fasade "
      "ima tipična SI hiša na m² ogrevane površine; TABULA SI.N.SFH.03 (geometrija modela) ima 0,82."]
(HERE / "04a_stroski.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print("\n".join(md))
