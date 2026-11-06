"""Krog 2, točka 1 — država v dveh ločenih računih: (a) javni proračun, (b) družbena analiza stroškov in koristi (CBA).

Na hišo in pot (P1, P2 = »OVE najprej«; P3, P4 = »URE najprej«), sedanja vrednost pri diskontu drzava_k2.diskont (3 % realno):
(a) javni proračun [neto strošek za proračun, manj je bolje] po režimu R1/R2/R3:
      + spodbude (opazovano: dejansko izplačane; kontrafaktično: pravila 2026, PV 104SUB-SO22)
      − DDV na investicije in vzdrževanje (Eko skladovi stroški so z DDV)
      − DDV, trošarina in prispevki (OVE+SPTE, URE, operater trga) na elektriko
      omrežnina (regulirani prihodki ELES/SODO) je prikazana ločeno in ni del proračuna
(b) družbena CBA [družbeni strošek, manj je bolje] — samo stroški virov; spodbude, davki in računi so transferji:
      + investicije in zamenjave brez DDV − ostanek vrednosti
      + vzdrževanje brez DDV
      + elektrika po urni vrednosti dan vnaprej SI 2025: uvoz − oddaja (scenarij: zimske ure × faktor)
      + zmogljivost: prispevek h 100 uram sistemske konice [kW] × letni strošek nove konične enote (ERAA 2025, OCGT)
      + ojačitev NN omrežja: max(sočasna konica uvoza, sočasna konica oddaje PV) v najslabšem letu poti, nad P0, × EUR/kW
        (krog 2, 12. 10. 2026; prej prispevek k nacionalni sistemski konici kot v 05 — ostane v tabeli za primerjavo)
      + CO2 po senčni ceni EIB (05/06; občutljivost UBA)
Izhodi: clanek1/tabele/05b_proracun_paket.csv, 05c_druzbena_cba_paket.csv, 05d_razclenitev_P1_P3.csv,
        clanek1/slike/Fig_k2_slap_drzava(.png/.svg, _sl), dodatek v clanek1/spremembe_krog2.md
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT_T, OUT_S = HERE / "clanek1" / "tabele", HERE / "clanek1" / "slike"
spec = importlib.util.spec_from_file_location("dr", HERE / "05_drzava.py")
DR = importlib.util.module_from_spec(spec)
spec.loader.exec_module(DR)
E, O = DR.EKM, DR.OKM
PAR = E.PAR
EK, K2 = PAR["ekonomika"], PAR["drzava_k2"]
N, r, ddv = E.N, PAR["drzava_k2"]["diskont"], PAR["ekonomika"]["ddv"]
POTI = ["P1", "P2", "P3", "P4", "P1r"]
REZ = ["R1", "R2", "R3"]
disc = lambda t: (1 + r) ** (-t)  # noqa: E731


def col(w, v, q):
    return w[f"{v}|{q}"].to_numpy(float)


def investicije(w, p):
    """SV izdatkov za investicije in zamenjave (z DDV), SV ostanka vrednosti, SV vzdrževanja (z DDV)."""
    kW = {"e": col(w, "V1", "TC_kW"), "FSO": col(w, "V4", "TC_kW"), "GLOB": col(w, "V7", "TC_kW"), "r": col(w, "V2r", "TC_kW") if "V2r|TC_kW" in w.columns else col(w, "V1", "TC_kW")}
    kwp = {v: col(w, v, "PV_kWp") for v in ("V2", "V3", "V5", "V6", "V7", "V2r") if f"{v}|PV_kWp" in w.columns}
    n = len(w)
    izd, ost, vzd = np.zeros(n), np.zeros(n), np.zeros(n)
    for (y, item, arg) in E.POTI[p]["ev"]:
        if item in ("TC", "TCz"):
            inv = EK["TC_a_EUR"] + EK["TC_b_EUR_kW"] * kW[arg]
            izd += inv * disc(y)
            L = EK["TC_let"]
            ost += inv * max(0, y + L - N) / L * disc(N)
            if item == "TC":
                vzd += EK["TC_vzdrzevanje_delez"] * inv * sum(disc(t) for t in range(y + 1, N + 1))
        elif item in ("PV", "PVB"):
            inv = np.where(kwp[arg] > 0, EK["PV_a_EUR"] + EK["PV_b_EUR_kWp"] * kwp[arg], 0)
            izd += inv * disc(y)
            ost += inv * max(0, y + 30 - N) / 30 * disc(N)
            vzd += EK["PV_vzdrzevanje_delez"] * inv * sum(disc(t) for t in range(y + 1, N + 1))
        elif item == "RAZS":
            izd += np.where(kwp[arg] > 0, EK["razsmernik_delez_PV"] * (EK["PV_a_EUR"] + EK["PV_b_EUR_kWp"] * kwp[arg]), 0) * disc(y)
        elif item in ("BAT", "BATz"):
            izd += EK["baterija_EUR_kWh"] * col(w, "V6", "baterija_kWh") * (1 if item == "BAT" else EK["baterija_zamenjava_delez"]) * disc(y)
        elif item in ("FSO", "GLOB"):
            inv, sub, lifew = E.envelope_cost(w, item, EK["ovoj_osnova"])
            izd += inv * disc(y)
            Lm = lifew / np.maximum(inv - sub, 1e-9)
            ost += inv * np.clip((y + Lm - N) / Lm, 0, 1) * disc(N)
            if item == "GLOB":
                vzd += EK["MVHR_vzdrzevanje_EUR_a"] * sum(disc(t) for t in range(y + 1, N + 1))
        elif item == "MVHRz":
            izd += EK["MVHR_EUR"] * disc(y)
    return izd, ost, vzd


def nn_obremenitev(w, v):
    """Obremenitev NN izvoda varianta v [kW na hišo]: večja od sočasne konice uvoza in oddaje."""
    if v == "V0":
        out = np.where(w["V0"] == "V0b", col(w, "V0b", "uvoz_konica_dist_kW"), col(w, "V0a", "uvoz_konica_dist_kW"))
        return np.where(w["V0"] == "V0c", col(w, "V0c", "uvoz_konica_dist_kW"), out) if "V0c|uvoz_konica_dist_kW" in w.columns else out
    return np.maximum(col(w, v, "uvoz_konica_dist_kW"), col(w, v, "oddaja_konica_dist_kW"))


def nn_ojacitev(w, p):
    """Ojačitev NN omrežja [EUR]: najslabše leto poti nad izhodiščem P0 (kotel), enkratno kot v 05."""
    vs = sorted({E.leto_varianta(p, t) for t in range(1, N + 1)})
    potreba = np.max(np.stack([nn_obremenitev(w, v) for v in vs]), axis=0)
    return np.maximum(potreba - nn_obremenitev(w, "V0"), 0) * PAR["drzava"]["ojacitev_EUR_kW"], potreba


def letno(w, p, fn):
    """SV letnih tokov: fn(varianta) -> array na hišo."""
    return sum(fn(E.leto_varianta(p, t)) * disc(t) for t in range(1, N + 1))


def racun(w):
    sp_nom, sp_sv = E.spodbude_poti(w, basis=EK["ovoj_osnova"], r=r)
    st = DR.drzava(w)                                   # CO2 (EIB) in ojačitev iz 05
    st_uba = DR.drzava(w, flat_co2=PAR["okolje"]["cena_CO2_obcutljivost_EUR_t"])
    f_z = K2["zimska_cena_faktor_scenarij"]
    out = {}
    for p in POTI:
        izd, ost, vzd = investicije(w, p)
        d = {"spodbude": sp_sv[p].to_numpy(), "DDV investicije in vzdrževanje": ddv / (1 + ddv) * (izd + vzd)}
        for rez in REZ:
            d[f"{rez} DDV elektrika"] = letno(w, p, lambda v: col(w, v, f"{rez}_ddv"))
            d[f"{rez} trošarina in prispevki"] = letno(w, p, lambda v: col(w, v, f"{rez}_trosarina") + col(w, v, f"{rez}_prispevki"))
            d[f"{rez} omrežnina"] = letno(w, p, lambda v: col(w, v, f"{rez}_omreznina"))
            d[f"{rez} proračun neto"] = d["spodbude"] - d["DDV investicije in vzdrževanje"] - d[f"{rez} DDV elektrika"] - d[f"{rez} trošarina in prispevki"]
        d["investicije brez DDV (− ostanek)"] = (izd - ost) / (1 + ddv)
        d["vzdrževanje brez DDV"] = vzd / (1 + ddv)
        d["elektrika (urna vrednost)"] = letno(w, p, lambda v: col(w, v, "soc_uvoz_EUR") - col(w, v, "soc_oddaja_EUR"))
        d["elektrika, dodatek zima ×f"] = (f_z - 1) * letno(w, p, lambda v: col(w, v, "soc_uvoz_zima_EUR") - col(w, v, "soc_oddaja_zima_EUR"))
        d["zmogljivost"] = letno(w, p, lambda v: col(w, v, "uvoz_konica_sis_kW") * K2["zmogljivost_EUR_kW_leto"])
        d["zmogljivost (CCGT)"] = d["zmogljivost"] * K2["zmogljivost_obcutljivost_EUR_kW_leto"] / K2["zmogljivost_EUR_kW_leto"]
        d["ojačitev omrežja"], d["obremenitev NN omrežja [kW]"] = nn_ojacitev(w, p)
        d["ojačitev omrežja (sistemska konica, krog 1)"] = st[p]["omrezje_EUR"].to_numpy()
        d["CO2 (EIB)"] = st[p]["CO2_SV_EUR"].to_numpy()
        d["CO2 (UBA 350)"] = st_uba[p]["CO2_SV_EUR"].to_numpy()
        soc = ["investicije brez DDV (− ostanek)", "vzdrževanje brez DDV", "elektrika (urna vrednost)", "zmogljivost", "ojačitev omrežja", "CO2 (EIB)"]
        d["družbeni NPV"] = sum(d[k] for k in soc)
        d["družbeni NPV, zima ×f"] = d["družbeni NPV"] + d["elektrika, dodatek zima ×f"]
        d["družbeni NPV, CCGT"] = d["družbeni NPV"] - d["zmogljivost"] + d["zmogljivost (CCGT)"]
        d["družbeni NPV, UBA"] = d["družbeni NPV"] - d["CO2 (EIB)"] + d["CO2 (UBA 350)"]
        out[p] = pd.DataFrame(d, index=w.index)
    return out


def _stevila():
    """Krog 13: število hiš za oznake slik iz 01_fond (paket, pred 1981, 1981–2002, fond) — ne na trdo."""
    f = pd.read_parquet(Path(__file__).resolve().parent / "cache" / "01_fond.parquet", columns=["paket_osnovni", "arhetip"])
    pk, pre = f["paket_osnovni"].astype(bool), f["arhetip"].isin(["A1", "A2", "A3"])
    return {"paket": int(pk.sum()), "pred": int((pk & pre).sum()), "po": int((pk & ~pre).sum()), "fond": len(f)}


def _en(n):
    return f"{n:,}"


def _sl(n):
    return f"{n:,}".replace(",", ".")


def main():
    w = E.load_units("fond")
    meta = pd.read_parquet(HERE / "cache" / "01_fond.parquet", columns=["ID_Stavbe", "Qr", "paket_osnovni", "arhetip"]).set_index("ID_Stavbe").reindex(w.index)
    res = racun(w)
    pk, q = meta["paket_osnovni"].astype(bool), pd.Series(np.where(meta["arhetip"].isin(["A1", "A2", "A3"]), "Q1", "Q2"), index=meta.index)
    vz = {"paket": pk, "paket pred 1981": pk & (q == "Q1"), "paket 1981–2002": pk & (q == "Q2"), "fond pred 1981": q == "Q1", "fond 1981–2002": q == "Q2",
          "fond": pd.Series(True, index=w.index)}
    kaz_b = ["spodbude", "DDV investicije in vzdrževanje"] + [f"{rz} {k}" for rz in REZ for k in ("DDV elektrika", "trošarina in prispevki", "proračun neto", "omrežnina")]
    kaz_s = ["investicije brez DDV (− ostanek)", "vzdrževanje brez DDV", "elektrika (urna vrednost)", "zmogljivost", "ojačitev omrežja", "CO2 (EIB)",
             "družbeni NPV", "družbeni NPV, zima ×f", "družbeni NPV, CCGT", "družbeni NPV, UBA", "ojačitev omrežja (sistemska konica, krog 1)"]
    rows = []
    for vlab, m in vz.items():
        mm = m.to_numpy()
        for k in kaz_b + kaz_s:
            row = {"vzorec": vlab, "stavb": int(mm.sum()), "kazalnik": k, "račun": "proračun" if k in kaz_b else "družbena CBA"}
            for p in POTI:
                x = res[p][k].to_numpy()[mm]
                row[f"{p} mediana [kEUR]"] = np.median(x) / 1000
                row[f"{p} vsota [mio EUR]"] = x.sum() / 1e6
            dd = (res["P3"][k] - res["P1"][k]).to_numpy()[mm]
            row["P3−P1 mediana [kEUR]"] = np.median(dd) / 1000
            row["P3−P1 P25 [kEUR]"] = np.quantile(dd, 0.25) / 1000
            row["P3−P1 P75 [kEUR]"] = np.quantile(dd, 0.75) / 1000
            row["P3−P1 vsota [mio EUR]"] = dd.sum() / 1e6
            row["delež P3 boljši [%]"] = 100 * (dd < 0).mean() if ("neto" in k or "NPV" in k) else np.nan
            rows.append(row)
    T = pd.DataFrame(rows)
    T[T["račun"] == "proračun"].round(3).to_csv(OUT_T / "05b_proracun_paket.csv", sep=";", encoding="utf-8-sig", index=False)
    T[T["račun"] == "družbena CBA"].round(3).to_csv(OUT_T / "05c_druzbena_cba_paket.csv", sep=";", encoding="utf-8-sig", index=False)

    # razčlenitev P1 − P3 (pozitivno = URE najprej cenejši za državo/družbo) — vsota na vzorec
    razc_b = [("spodbude", -1), ("DDV investicije in vzdrževanje", +1), ("R3 DDV elektrika", +1), ("R3 trošarina in prispevki", +1)]
    razc_s = [("investicije brez DDV (− ostanek)", -1), ("vzdrževanje brez DDV", -1), ("elektrika (urna vrednost)", -1),
              ("zmogljivost", -1), ("ojačitev omrežja", -1), ("CO2 (EIB)", -1)]
    rz = []
    for vlab in ("paket", "paket pred 1981", "paket 1981–2002", "fond"):
        mm = vz[vlab].to_numpy()
        for racun_, items in (("proračun (R3)", razc_b), ("družbena CBA", razc_s)):
            for k, sgn in items:
                # P1 − P3 v smislu »koristi URE najprej«: strošek P1 − strošek P3
                if racun_.startswith("proračun") and k != "spodbude":
                    val = (res["P3"][k] - res["P1"][k]).to_numpy()[mm].sum()          # več davkov pri P3 = korist proračunu
                else:
                    val = (res["P1"][k] - res["P3"][k]).to_numpy()[mm].sum()
                rz.append({"vzorec": vlab, "račun": racun_, "postavka": k, "P1 − P3 [mio EUR]": val / 1e6})
    RZ = pd.DataFrame(rz)
    RZ.round(3).to_csv(OUT_T / "05d_razclenitev_P1_P3.csv", sep=";", encoding="utf-8-sig", index=False)

    # slika: slap (waterfall) P1 − P3 za paket, Q1, Q2 — proračun (R3) in družbena CBA
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = {"spodbude": ("subsidies", "spodbude"), "DDV investicije in vzdrževanje": ("VAT on investment", "DDV na investicije"),
             "R3 DDV elektrika": ("VAT on electricity", "DDV na elektriko"), "R3 trošarina in prispevki": ("excise and levies", "trošarina, prispevki"),
             "investicije brez DDV (− ostanek)": ("investment", "investicije"), "vzdrževanje brez DDV": ("maintenance", "vzdrževanje"),
             "elektrika (urna vrednost)": ("electricity (hourly value)", "elektrika (urna vrednost)"), "zmogljivost": ("capacity", "zmogljivost"),
             "ojačitev omrežja": ("grid reinforcement", "ojačitev omrežja"), "CO2 (EIB)": ("CO$_2$ (EIB)", "CO$_2$ (EIB)")}
    for lang, sfx in (("en", ""), ("sl", "_sl")):
        fig, axs = plt.subplots(2, 3, figsize=(11, 6.2))
        for j, vlab in enumerate(("paket", "paket pred 1981", "paket 1981–2002")):
            for i, racun_ in enumerate(("proračun (R3)", "družbena CBA")):
                ax = axs[i, j]
                d = RZ[(RZ["vzorec"] == vlab) & (RZ["račun"] == racun_)]
                cum = 0.0
                for k_, (_, rr) in enumerate(d.iterrows()):
                    v = rr["P1 − P3 [mio EUR]"]
                    ax.bar(k_, v, bottom=cum, color="#008300" if v >= 0 else "#e34948", width=0.7)
                    cum += v
                ax.bar(len(d), cum, color="#52514e", width=0.7)
                ax.text(len(d), cum + (0.5 if cum >= 0 else -0.5), f"{cum:+.1f}", ha="center", va="bottom" if cum >= 0 else "top", fontsize=7.5)
                ax.axhline(0, color="#52514e", lw=0.8)
                ax.set_xticks(range(len(d) + 1))
                ax.set_xticklabels([names[k][0 if lang == "en" else 1] for k in d["postavka"]] + (["net"] if lang == "en" else ["skupaj"]),
                                   rotation=35, ha="right", fontsize=6.8)
                ax.grid(axis="x", visible=False)
                if j == 0:
                    ax.set_ylabel(("Public budget, R3\n" if i == 0 else "Social CBA\n") + "P1 − P3 [€ million]" if lang == "en"
                                  else ("Javni proračun, R3\n" if i == 0 else "Družbena CBA\n") + "P1 − P3 [mio €]", fontsize=8)
                if i == 0:
                    N = _stevila()
                    lab = {"paket": (f"HP+PV package, {_en(N['paket'])} houses", f"paket TČ+PV, {_sl(N['paket'])} hiš"),
                           "paket pred 1981": (f"package pre-1981, {_en(N['pred'])}", f"paket pred 1981, {_sl(N['pred'])}"),
                           "paket 1981–2002": (f"package 1981–2002, {_en(N['po'])}", f"paket 1981–2002, {_sl(N['po'])}")}[vlab]
                    ax.set_title(lab[0 if lang == "en" else 1], fontsize=8.5)
        fig.text(0.5, 0.005, "Positive (green) = envelope-first (P3) is better for the state / society" if lang == "en"
                 else "Pozitivno (zeleno) = URE najprej (P3) je boljši za državo / družbo", ha="center", fontsize=8, color="#52514e")
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        for ext in ("png", "svg"):
            fig.savefig(OUT_S / f"Fig_k2_slap_drzava{sfx}.{ext}", dpi=300)
        plt.close(fig)

    pd.set_option("display.width", 250)
    show = ["vzorec", "kazalnik", "P1 mediana [kEUR]", "P3 mediana [kEUR]", "P3−P1 mediana [kEUR]", "P3−P1 vsota [mio EUR]", "delež P3 boljši [%]"]
    print(T[T["kazalnik"].isin(["R1 proračun neto", "R3 proračun neto", "R3 omrežnina", "družbeni NPV", "družbeni NPV, zima ×f", "družbeni NPV, CCGT", "družbeni NPV, UBA"])]
          [show].round(2).to_string(index=False))
    print()
    print(RZ.pivot_table(index=["račun", "postavka"], columns="vzorec", values="P1 − P3 [mio EUR]", sort=False).round(2).to_string())
    return T, RZ


if __name__ == "__main__":
    main()
