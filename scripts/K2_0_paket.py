"""Krog 2, točka 0 — rezultati kroga 1 na ravni posamezne hiše za podvzorec paketa TČ+PV (5.969) in fond (29.977).

»OVE najprej« = P1 (in P2), »URE najprej« = P3 (in P4); za vsako hišo opazovano (P1, dejanski kWp, moč TČ, A, Qh,nd)
proti kontrafaktičnemu (P3) stanju iste hiše. Porazdelitve (mediana, P25–P75), delež hiš, kjer je P3 boljši, vsote na vzorec.
Vzorci: paket (5.969), paket pred 1981 (2.751), paket 1981–2002 (3.218), fond pred 1981, fond 1981–2002, fond 29.977.
Vhodi: cache/04_npv_fond.parquet (posameznik R1–R3, R2d), 06_co2_poti_fond.parquet (okolje), 05_drzava.drzava() (država, krog 1).
Izhodi: clanek1/tabele/04_posameznik_paket.csv, 05_drzava_paket.csv, 06_okolje_paket.csv; clanek1/spremembe_krog2.md
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CACHE, OUT = HERE / "cache", HERE / "clanek1" / "tabele"
POTI = ["P1", "P2", "P3", "P4"]


def load_mod(f):
    s = importlib.util.spec_from_file_location(f[:2], HERE / f)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def vzorci(meta: pd.DataFrame) -> dict:
    pk, q = meta["paket_osnovni"].astype(bool), pd.Series(np.where(meta["arhetip"].isin(["A1", "A2", "A3"]), "Q1", "Q2"), index=meta.index)
    return {"paket": pk, "paket pred 1981": pk & (q == "Q1"), "paket 1981–2002": pk & (q == "Q2"),
            "fond pred 1981": q == "Q1", "fond 1981–2002": q == "Q2", "fond": pd.Series(True, index=meta.index)}


def stat(x: pd.Series, scale=1.0) -> dict:
    x = x.dropna() / scale
    return {"mediana": x.median(), "P25": x.quantile(0.25), "P75": x.quantile(0.75)}


def tabela(values: dict, meta, perspektiva, enota, scale, vsota_scale, vsota_enota, manj_je_bolje=True):
    """values: {(rezim): DataFrame index=ID, columns=POTI}. Vrne dolgo tabelo porazdelitev."""
    rows = []
    for rez, df in values.items():
        for vz, m in vzorci(meta).items():
            d = df[m.reindex(df.index).fillna(False).to_numpy()]
            diff = d["P3"] - d["P1"]
            row = {"perspektiva": perspektiva, "rezim": rez, "vzorec": vz, "stavb": len(d), "enota": enota}
            for p in POTI:
                s = stat(d[p], scale)
                row.update({f"{p} mediana": s["mediana"], f"{p} P25": s["P25"], f"{p} P75": s["P75"]})
            s = stat(diff, scale)
            row.update({"P3−P1 mediana": s["mediana"], "P3−P1 P25": s["P25"], "P3−P1 P75": s["P75"],
                        "P4−P1 mediana": stat(d["P4"] - d["P1"], scale)["mediana"],
                        "P2−P1 mediana": stat(d["P2"] - d["P1"], scale)["mediana"],
                        "delež P3 boljši [%]": 100 * ((diff < 0) if manj_je_bolje else (diff > 0)).mean(),
                        f"vsota P1 [{vsota_enota}]": d["P1"].sum() / vsota_scale, f"vsota P3 [{vsota_enota}]": d["P3"].sum() / vsota_scale,
                        f"vsota P3−P1 [{vsota_enota}]": diff.sum() / vsota_scale})
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fond = pd.read_parquet(CACHE / "01_fond.parquet", columns=["ID_Stavbe", "Qr", "paket_osnovni", "ima_PV", "PV_kWp", "arhetip"]).set_index("ID_Stavbe")
    # --- posameznik (NPV stroškov 30 let, diskont 4 %)
    npv = pd.read_parquet(CACHE / "04_npv_fond.parquet").set_index("ID_Stavbe")
    ind = {rez: npv[npv["rezim"] == rez][POTI] for rez in ("R1", "R2", "R3", "R2d")}
    T4 = tabela(ind, fond, "posameznik: NPV stroškov 30 let", "kEUR/hišo", 1000, 1e6, "mio EUR")
    # --- okolje (t CO2e 30 let, obratovalni + vgrajeni)
    c6 = pd.read_parquet(CACHE / "06_co2_poti_fond.parquet").set_index("ID_Stavbe")
    env = {"–": pd.DataFrame({p: c6[f"{p}|skupaj_t"] for p in POTI})}
    T6 = tabela(env, fond, "okolje: CO2e 30 let", "t/hišo", 1, 1000, "kt")
    # --- država (krog 1: spodbude + ojačitev omrežja + CO2 po senčni ceni, 3 %)
    DR = load_mod("05_drzava.py")
    w = DR.EKM.load_units("fond")
    st = DR.drzava(w)
    dr = {"krog 1 (spodbude + omrežje + CO2)": pd.DataFrame({p: st[p]["javni_NPV"] for p in POTI}),
          "spodbude (nominalno)": pd.DataFrame({p: st[p]["spodbude_nom"] for p in POTI}),
          "prispevek h sistemski konici, leto 20 [kW]": pd.DataFrame({p: st[p]["konica_sis_kW_leto20"] for p in POTI}),
          "zimski uvoz, leto 20 [kWh]": pd.DataFrame({p: st[p]["zimski_uvoz_kWh_leto20"] for p in POTI})}
    T5 = []
    for k, df in dr.items():
        sc, vs, ve, en = (1000, 1e6, "mio EUR", "kEUR/hišo") if "kW]" not in k and "kWh]" not in k else \
            ((1, 1000, "MW", "kW/hišo") if "kW]" in k else (1, 1e6, "GWh/a", "kWh/hišo"))
        T5.append(tabela({k: df}, fond, "država", en, sc, vs, ve))
    T5 = pd.concat(T5)
    for t, f in ((T4, "04_posameznik_paket.csv"), (T5, "05_drzava_paket.csv"), (T6, "06_okolje_paket.csv")):
        t.round(2).to_csv(OUT / f, sep=";", encoding="utf-8-sig", index=False)

    # --- kontrola vzorca
    pk = fond["paket_osnovni"].astype(bool)
    kontrola = {"paket": int(pk.sum()), "paket pred 1981": int((pk & fond["arhetip"].isin(["A1", "A2", "A3"])).sum()), "paket 1981–2002": int((pk & (fond["arhetip"] == "A4")).sum()),
                "paket z dejanskim kWp": int((pk & fond["PV_kWp"].notna()).sum()),
                "paket kWp mediana (P25–P75)": f"{fond.loc[pk, 'PV_kWp'].median():.1f} ({fond.loc[pk, 'PV_kWp'].quantile(.25):.1f}–{fond.loc[pk, 'PV_kWp'].quantile(.75):.1f})"}
    # V2 kWp v modelu = dejanski kWp stavbe? (03: pv_kWp_V2 = PV_kWp, če ima_PV)
    fl = pd.read_parquet(CACHE / "03_fond_letno.parquet", columns=["ID_Stavbe", "varianta", "PV_kWp"])
    v2 = fl[fl["varianta"] == "V2"].set_index("ID_Stavbe")["PV_kWp"]
    ujema = np.isclose(v2.reindex(fond.index[pk]).to_numpy(), fond.loc[pk, "PV_kWp"].to_numpy(), atol=0.01).mean()
    kontrola["V2 v modelu = dejanski kWp (delež paketa)"] = f"{100 * ujema:.1f} %"
    md = ["<!-- TOCKA0 -->", "## Točka 0 (12. 10. 2026): rezultati kroga 1 na ravni hiše za paket 5.969", "",
          "Rezultati kroga 1 so bili na voljo na ravni posamezne hiše za posameznika (`cache/04_npv_fond.parquet`) in okolje "
          "(`cache/06_co2_poti_fond.parquet`), za državo pa le agregirano (05) — zdaj izpisani na ravni hiše prek `05_drzava.drzava()`. "
          "Številke kroga 1 niso spremenjene; nove so le porazdelitve za podvzorec paketa.", "",
          "Kontrola vzorca:", ""] + [f"- {k}: {v}" for k, v in kontrola.items()] + [
          "", "Izhodi: `tabele/04_posameznik_paket.csv`, `tabele/05_drzava_paket.csv`, `tabele/06_okolje_paket.csv` "
          "(skript `K2_0_paket.py` v nadrejeni mapi).", "<!-- /TOCKA0 -->"]
    # posodobi le svoj razdelek; ostale zapise v dnevniku sprememb ohrani
    f = HERE / "clanek1" / "spremembe_krog2.md"
    txt = f.read_text(encoding="utf-8") if f.exists() else "# Spremembe in kontrole — krog 2\n\n<!-- TOCKA0 --><!-- /TOCKA0 -->\n"
    a, b = txt.index("<!-- TOCKA0 -->"), txt.index("<!-- /TOCKA0 -->") + len("<!-- /TOCKA0 -->")
    f.write_text(txt[:a] + "\n".join(md) + txt[b:], encoding="utf-8")
    pd.set_option("display.width", 250)
    print("\n".join(md))
    show = ["perspektiva", "rezim", "vzorec", "stavb", "P1 mediana", "P3 mediana", "P3−P1 mediana", "P3−P1 P25", "P3−P1 P75", "delež P3 boljši [%]"]
    for t in (T4, T5, T6):
        print(t[show].round(1).to_string(index=False))
        print()


if __name__ == "__main__":
    main()
