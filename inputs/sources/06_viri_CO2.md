# Viri: vgrajeni CO2, emisijski faktorji, PM10 (raziskava 9. 10. 2026)

Poročilo raziskovalnega podagenta; [IZPELJANO] = lasten preračun, NEPREVERJENO označeno. Osrednji vir za gradbene izdelke: KBOB/ecobau-Liste 2022 v9.0 (2026), https://www.kbob.admin.ch/dam/de/sd-web/GUOl1trUNEVH/%C3%96kobilanzdaten_im_Baubereich_V9.0.xlsx (»THG Total« = proizvodnja + odstranjevanje).

| postavka | vrednost | enota | obseg | vir | izbrano v parametri.yaml |
|---|---|---|---|---|---|
| PV mono-Si streha | 35,8 g/kWh (moduli 20,2 / razsmernik 9,9 / BOS 5,8) | g CO2e/kWh pri 976 kWh/kWp, 30 a | cel cikel | IEA PVPS Task 12 Fact Sheet (2024), https://iea-pvps.org/wp-content/uploads/2024/05/Task-12-Fact-Sheet-v2-1.pdf | moduli + BOS 680 kg/kWp [IZPELJANO: 530 + 150] |
| PV modul Kitajska | 810 (steklo–folija) | kg/kWp | samo modul | Fraunhofer ISE (2021), https://www.ise.fraunhofer.de/content/dam/ise/de/documents/presseinformationen/2021/2221_ISE_e_PR_CO2%20Footprint%20of%20PV%20modules.pdf | občutljivost |
| PV Marktmix / mono-Si mala | 1.000 / 1.260 | kg/kWp | proizvodnja + EoL | KBOB 34.025 / 34.028 | – |
| razsmernik 5 kW | 115 | kg/kWp | proizvodnja + EoL | KBOB 34.033 | 115 kg/kW na enoto razsmernika |
| Li-ion baterija | 110 | kg/kWh zmogljivosti | proizvodnja | Peters et al. 2017, RSER 67:491 | 110 |
| hišna LFP 14,4 kWh | 84 g/kWh dobavljene | – | cradle-to-grave | Jasper et al. 2022, doi:10.1016/j.jclepro.2022.132899 | – |
| baterija 5 / 20 kWh | 730 / 332 | kg/kWh (30 a, z zamenjavami) | | KBOB 34.037 / 34.038 | občutljivost |
| TČ R290 8,5 kW | 666 (865 s faktorjem TM65 1,3) | kg/napravo, A1–C4 brez puščanja | | Mitsubishi TM65, https://library.mitsubishielectric.co.uk/pdf/download_full/5461 | 102 kg/kW |
| TČ R32 8,5 kW | 1.362 skupaj; puščanje ≈ 460 [IZPELJANO: 2,2 kg × 675 × (2 %·15 + 1 %)] | kg | TM65 | …/download_full/4534 | hladivo 460 kg na napravo |
| TČ zrak–voda 7 kW | 4.000 | kg/napravo | proizvodnja + EoL | KBOB 31.019 | občutljivost |
| puščanje hladiva | 1–10 %/a | % polnitve | | IPCC 2006 Vol. 3 pogl. 7, Tab. 7.9 | – |
| ETICS EPS 160 mm | 21,7 (A1–A3) / 36,1 (A–C) | kg/m² | EPD VDPM (IBU 2025) EPD-WDV-20240371-IBP2-DE | 32 kg/m² [IZPELJANO na ~12 cm] |
| ETICS mineralna volna 160 mm | 30,7 / 45,3 | kg/m² | EPD-WDV-20240370-IBP2-DE | občutljivost |
| podstrešje mineralna volna 20 cm | 3,3–6,6 | kg/m² | KBOB 10.001/10.008 [IZPELJANO] | 5 |
| tla EPS 10 cm | 22,8 | kg/m² | KBOB 10.004 [IZPELJANO] | 22,8 |
| okno PVC / les, trojna zasteklitev | ≈ 111 / ≈ 84 | kg/m² okna | KBOB 05.032/05.030 + 05.012 [IZPELJANO, 70 % stekla] | 111 |
| MVHR + PE kanali | 8,25 | kg/m² ogrevane površine | KBOB 32.002 | 8,25 |
| hranilnik TV 300 L | 380–514 | kg/kos | PEP Atlantic (prek One Click LCA) | (ni v poteh) |
| neposredno: lignit / črni premog / plin | 1,119 / 0,860 / 0,392 | t CO2/MWh_el | UBA Climate Change 13/2025, Tab. 4 | sistem.EF_tehnologija |
| neposredno: olje / odpadki | 0,70 / ≈ 0,60 | t CO2/MWh_el | IPCC 2006 Vol. 2 Tab. 2.2 + izkoristki [IZPELJANO] | sistem.EF_tehnologija |
| življenjski cikel (mediana): premog / plin CC / biomasa / jedrska / hidro / veter / PV | 820 / 490 / 230 / 12 / 24 / 11 / 41 | g/kWh | IPCC AR5 WGIII Annex III Tab. A.III.2 | (občutljivost) |
| ELKO / plin / les (ogrevanje) | 0,267 / 0,202 / 0,403 (biogeni → 0) | kg CO2/kWh goriva | IPCC 2006 Vol. 2 Tab. 2.5 | ELKO 0,267; biomasa 0,034 (CH4 + N2O, [IZPELJANO] AR5 GWP) |
| PM10: klasična peč / kotel < 50 kW / peleti / plin / TČ | 760 / 480 / 60 / 1,2 / 0 | g/GJ goriva | EMEP/EEA Guidebook 2023, 1.A.4, Tab. 3-40/3-43/3-44/3-4 | kvalitativno (06) |

Opozorila: KBOB = švicarsko ozadje, »Entsorgung« ≠ EN 15804 C1–C4, brez modula D; skupinski EPD VDPM sta »worst case«; TM65 vključuje varnostni faktor.
