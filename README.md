# Efficiency first or heat pump and PV first? — analysis code and aggregated results

Code and aggregated results for the paper

> Stegnar, G., Potočar, E. *Envelope first or heat pump and PV first? Sequencing lock-in in inefficient single-family houses from the household, public and climate perspectives.* Submitted to Applied Energy, November 2026.

The study links the Slovenian heat-pump and PV subsidy register (Eko sklad) to the building cadastre (GURS) and energy performance certificates, estimates how PV changes the hazard of a later envelope renovation (Cox model, coarsened exact matching, falsification tests), and simulates 8,134 actual heat pump + PV houses hourly over 30 years on a ladder of envelope states, comparing "renewables first" with "efficiency first" from the household, public-budget, societal and climate perspectives.

## What is here

| Folder | Content |
|---|---|
| `scripts/` | All analysis scripts, numbered in the order of the pipeline (`00_` … `09_`) plus the paper-specific scripts (`K2_*` … `K16_*`). `urni_lib.py` holds the shared hourly model, tariffs and bills. |
| `parametri.yaml` | All parameters (costs, tariffs, subsidies, emission factors, heat-pump and PV model, envelope states, discount rates). Local paths are replaced by `<set local path>`. |
| `inputs/hecm/` | Building-model input files (archetypes A1–A4 × envelope states S0–S3 and target, geometry G13 and the G11/TABULA checks). |
| `inputs/sources/` | Source notes for costs, tariffs and CO₂ factors. |
| `results/paper_tables/` | Aggregated results behind every table and figure of the paper (`09_` main comparison, `11_` lock-in, `12_` sensitivity, `13_`–`14_` register groups, `15_` archetype check, `16_`–`18_` grid, sensitivities and cooling, `19_` causal checks, `22_`–`25_` geometry, placement, threshold rule, heat-pump cut-out). Suffixes: none = base case (geometry G13, placement by certificate), `_REG` = lower bound (placement by the register), `_S0F`, `_G11`, `_TABULA` = sensitivities. |
| `results/tables/` | Intermediate aggregated tables of the pipeline. |
| `results/figures/` | Figures of the paper and of the Supplementary material (PNG, 300 dpi) with captions. |

## What is not here, and why

Building-level data — the subsidy register, the certificates and the cadastre extract — were obtained under data-sharing agreements with Eko sklad, the Ministry of Infrastructure and Energy and the Surveying and Mapping Authority and cannot be redistributed. The scripts that read them (`00_inventar.py`, `01_tipicna_hisa.py`, `K2_4_vzrocnost.py`, `K10_vzrocnost_testi.py`, `K4_1_*`, `K4_2_*`, `K13_*`) therefore cannot be re-run without access to these data; everything downstream of the cached house-level results (`cache/`, not included) is reproducible from the aggregated tables.

The hourly building model is the HECM model of the LIFE23-CET-GreenRenoV8 project (ISO 52016-1 through `pyBuildingEnergy` 2.1.0), called by `02_hecm_vhodi.py` / `02_lestvica_stanj.py`; its input files are in `inputs/hecm/`. Weather: IWEC Ljubljana (EnergyPlus). PV: `pvlib`, scaled to PVGIS-SARAH2. Household load: BDEW H0 profile via `demandlib`. Prices: Slovenian day-ahead prices 2025 (ENTSO-E / Energy-Charts), not redistributed.

## Running

Python 3.8+ (`requirements.txt`). Set the paths in `parametri.yaml`, then run the scripts in numerical order; each script writes its tables to `tabele/` or `clanek1/tabele/` and a Markdown summary. Scripts take `--tag` to produce the sensitivity variants (`_REG`, `_S0F`, `_G11`, `_TABULA`).

## Licence and citation

Code: MIT licence. Aggregated results: CC BY 4.0. Please cite the paper above.

Contact: gasper.stegnar@ijs.si
