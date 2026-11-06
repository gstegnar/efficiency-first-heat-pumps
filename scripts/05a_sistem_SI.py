"""Korak 05a — urne serije elektroenergetskega sistema SI 2025 (energy-charts.info): odjem, proizvodnja po virih, uvoz po mejah,
proizvodni EF SI in sosed, potrošni EF SI, ure sistemske konice. Preslikava na koledar modela po položaju (8.760 h).
Izhod: cache/05a_sistem_SI.parquet, tabele/05a_sistem_SI_povzetek.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
PAR = yaml.safe_load((HERE / "parametri.yaml").read_text(encoding="utf-8"))
S = PAR["sistem"]
D = HERE / "podatki" / "energy_charts"
NON_GEN = {"Cross border electricity trading", "Load", "Residual load", "Renewable share of load", "Renewable share of generation",
           "Hydro pumped storage consumption", "Battery Consumption"}


def load(files):
    fr = []
    for f in files:
        d = json.load(open(D / f, encoding="utf-8"))
        idx = pd.to_datetime(d["unix_seconds"], unit="s", utc=True)
        fr.append(pd.DataFrame({x["name"]: pd.to_numeric(pd.Series(x["data"]), errors="coerce").values for x in d["production_types"]}, index=idx))
    x = pd.concat(fr).sort_index()
    x = x[~x.index.duplicated()].tz_convert("Europe/Ljubljana")
    return x.resample("h").mean()


def ef_prod(x):
    gen = [c for c in x.columns if c not in NON_GEN]
    g = x[gen].clip(lower=0).fillna(0)
    em = sum(g[c] * S["EF_tehnologija"].get(c, 0.0) for c in gen)
    return em / g.sum(1).replace(0, np.nan), g.sum(1), em


files = {"si": ["public_power_si_2025.json"], "at": ["public_power_at_2025.json"], "hr": ["public_power_hr_2025.json"],
         "it": ["public_power_it_2025.json"], "hu": ["hu1.json", "hu2.json"]}
P = {c: load(f) for c, f in files.items()}
ef = {c: ef_prod(x)[0] for c, x in P.items()}
si = P["si"]
ef_si, gen_si, em_si = ef_prod(si)
d = json.load(open(D / "cbpf_si_2025.json", encoding="utf-8"))
cb = pd.DataFrame({x["name"]: x["data"] for x in d["countries"]},
                  index=pd.to_datetime(d["unix_seconds"], unit="s", utc=True)).astype(float).tz_convert("Europe/Ljubljana").resample("h").mean()
naziv = {"Austria": "at", "Croatia": "hr", "Hungary": "hu", "Italy": "it"}
imp_c = {naziv[k]: cb[k].clip(lower=0) * 1000 for k in naziv}            # GW -> MW, uvoz v SI
imp_tot = sum(imp_c.values())
em_imp = sum(imp_c[c] * ef[c].reindex(imp_c[c].index).fillna(ef[c].mean()) for c in imp_c)
ef_cons = (em_si + em_imp) / (gen_si + imp_tot)
out = pd.DataFrame({"odjem_MW": si["Load"], "proizvodnja_MW": gen_si, "uvoz_MW": imp_tot, "neto_trgovanje_MW": si["Cross border electricity trading"],
                    "EF_proizvodni_SI": ef_si, "EF_potrosni_SI": ef_cons,
                    **{f"EF_{c}": ef[c].reindex(si.index) for c in ("at", "hr", "it", "hu")}}).iloc[:8760]
out = out.interpolate(limit_direction="both")
thr = out["odjem_MW"].nlargest(S["konica_ur"]).min()
out["sistemska_konica"] = out["odjem_MW"] >= thr
out.index.name = "ura_lokalno"
out.to_parquet(HERE / "cache" / "05a_sistem_SI.parquet")
w = out["odjem_MW"]
summ = pd.DataFrame([
    {"kazalnik": "odjem SI [TWh]", "vrednost": w.sum() / 1e6},
    {"kazalnik": "proizvodnja SI [TWh]", "vrednost": out["proizvodnja_MW"].sum() / 1e6},
    {"kazalnik": "uvoz (fizični, vsota mej) [TWh]", "vrednost": out["uvoz_MW"].sum() / 1e6},
    {"kazalnik": "maks. odjem [MW]", "vrednost": w.max()},
    {"kazalnik": "prag 100 ur konice [MW]", "vrednost": thr},
    {"kazalnik": "EF proizvodni SI, uteženo z odjemom [t/MWh]", "vrednost": (out["EF_proizvodni_SI"] * w).sum() / w.sum()},
    {"kazalnik": "EF potrošni SI, uteženo z odjemom [t/MWh]", "vrednost": (out["EF_potrosni_SI"] * w).sum() / w.sum()},
    {"kazalnik": "EF potrošni SI, zima nov–feb [t/MWh]", "vrednost": out.loc[out.index.month.isin([11, 12, 1, 2]), "EF_potrosni_SI"].mean()},
    {"kazalnik": "EF potrošni SI, poletje jun–avg [t/MWh]", "vrednost": out.loc[out.index.month.isin([6, 7, 8]), "EF_potrosni_SI"].mean()},
    {"kazalnik": "EF potrošni SI v urah konice [t/MWh]", "vrednost": out.loc[out["sistemska_konica"], "EF_potrosni_SI"].mean()},
    {"kazalnik": "ure konice: delež v nov–feb [%]", "vrednost": 100 * out.loc[out["sistemska_konica"]].index.month.isin([11, 12, 1, 2]).mean()},
    {"kazalnik": "ure konice: najpogostejša ura dneva", "vrednost": out.loc[out["sistemska_konica"]].index.hour.value_counts().idxmax()},
] + [{"kazalnik": f"EF proizvodni {c.upper()} povprečje [t/MWh]", "vrednost": out[f"EF_{c}"].mean()} for c in ("at", "hr", "it", "hu")]).round(3)
summ.to_csv(HERE / "tabele" / "05a_sistem_SI_povzetek.csv", sep=";", encoding="utf-8-sig", index=False)
print(summ.to_string(index=False))
