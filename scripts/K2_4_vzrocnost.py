"""Krog 2, točka 4 — vzročnost zaklepanja: čas do prve prenove ovoja po vgradnji TČ (Coxov model), PV kot časovno odvisna spremenljivka.

Kohorta: eno-/dvostanovanjske stavbe s subvencionirano TČ (Eko sklad 2011–2025), ovoj ob vgradnji neprenovljen (stopnja_ob_vgradnji = 0,
enako kot fond članka), znano obdobje gradnje 1–4, brez novogradenj, leto TČ ≤ 2024 (vsaj eno leto opazovanja).
Pozor: fond članka (29.977) je izbran na izidu (do 2025 brez ovoja) in za analizo preživetja ni primeren — tu je širša kohorta.
Čas: leta od vgradnje TČ. Dogodek (osnovno): prvi Eko skladov ukrep na ovoju (fasada, streha, okna/vrata, celovita prenova) po letu TČ.
Cenzura: konec podatkov (zadnje leto v bazi spodbud).
Modeli:
  M1  Cox, PV časovno odvisna (preklop v letu PV; ukrep v istem letu kot PV se pripiše obdobju brez PV), kovariate: obdobje, log A_ogr,
      regija, razred Qh,nd ob vgradnji, nadomeščeni energent; strata = leto TČ (koledarski učinki razpisov); robustne SE po stavbi
  M2  kot M1, PV preklopi leto prej (ukrep v letu PV se pripiše PV)
  M3  kot M1, dogodek = Eko sklad ALI GURS leto obnove fasade/strehe/oken (občutljivost na GURS)
  M4  ujemanje (CEM): paket = PV v [leto TČ − 3, leto TČ], kontrola = brez PV do leta TČ (cenzura ob poznejši PV); celice =
      obdobje × četrtina A_ogr × regija × leto TČ × razred Qh,nd × energent; Cox, stratificiran po celici
Kumulativna incidenca prenove ovoja (Kaplan–Meier, 1 − S) za paket (PV v [TČ − 3, TČ + 1]) in samo TČ (brez PV do TČ + 1) po razredu Qh,nd;
po zadnjem zanesljivem letu (≥ 200 stavb v tveganju) podaljšana s konstantno nevarnostjo iz let 3 … zadnje (PREVERI — ekstrapolacija).
Izhodi: clanek1/tabele/11a_cox_modeli.csv, 11b_kumulativna_incidenca.csv, 11c_verjetnost_ovoja_paket.csv, cache/11_verjetnost_ovoja.json,
        clanek1/slike/Fig_k2_lockin(_sl).png/svg. Teče s Python 3.12 (statsmodels).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.duration.hazard_regression import PHReg
from statsmodels.duration.survfunc import SurvfuncRight

HERE = Path(__file__).resolve().parent
AN = (HERE / ".." / "Interno" / "2026 analiza").resolve()
OUT_T, OUT_S, CACHE = HERE / "clanek1" / "tabele", HERE / "clanek1" / "slike", HERE / "cache"
OVOJ_EKO = ["fasada", "okna_vrata", "streha", "celovita_prenova"]
GURS = ["LETO_OBN_FASADE", "LETO_OBN_STREHE", "okna_obn_min"]
MIN_RISK = 200
GMIN = __import__("yaml").safe_load((Path(__file__).resolve().parent / "parametri.yaml").read_text(encoding="utf-8"))["fond"].get("gurs_min_leto", 2003)
T_EXT = (10, 20, 30)


def kohorta():
    cols = (["ID_Stavbe", "leto_TC", "leto_PV", "ima_PV", "obdobje", "obdobje_imputirano", "stopnja_ob_vgradnji", "staro_ogrevanje",
             "A_ogr", "ID_Regije", "Qr"] + [f"leto_prvi_{k}" for k in OVOJ_EKO] + GURS)
    d = pd.read_parquet(AN / "cache" / "07_stavbe_paket.parquet", columns=cols)
    yrs = pd.concat([pd.to_numeric(d[f"leto_prvi_{k}"], errors="coerce") for k in OVOJ_EKO] + [d["leto_TC"]], axis=1)
    end = int(np.nanmax(yrs.to_numpy(float)))
    # krog 13 (Gašper 22. 10. 2026): prenova v GURS pred 2010 ni ukrep URE → ovoj ob vgradnji neprenovljen po pravilu Eko ∪ GURS ≥ GMIN (analiza: 2003)
    tc0 = pd.to_numeric(d["leto_TC"], errors="coerce").astype(float)
    _y = lambda c: pd.to_numeric(d[c], errors="coerce").astype(float)                          # noqa: E731
    ob_tc = pd.concat([_y(f"leto_prvi_{c}") <= tc0 for c in OVOJ_EKO]
                      + [(_y(c) >= GMIN) & (_y(c) <= tc0) for c in GURS], axis=1).any(axis=1)
    k = d[~ob_tc & ~d["obdobje_imputirano"].astype(bool) & d["obdobje"].between(1, 4)
          & (d["staro_ogrevanje"] != "novogradnja") & (d["leto_TC"] <= end - 1)].copy()
    tc = k["leto_TC"].astype(float)

    def prvi_po(cols_):
        y = pd.concat([pd.to_numeric(k[c], errors="coerce").astype(float).where(lambda s: (s > tc) & (s <= end)) for c in cols_], axis=1)
        return y.min(axis=1)
    k["leto_ovoj_eko"] = prvi_po([f"leto_prvi_{c}" for c in OVOJ_EKO])
    k["leto_ovoj_gurs"] = prvi_po(GURS)
    k["leto_ovoj_oba"] = k[["leto_ovoj_eko", "leto_ovoj_gurs"]].min(axis=1)
    # GURS obnova ≤ leto TČ (po 2002) pomeni, da ovoj ob TČ ni bil neprenovljen — izloči le v občutljivosti z GURS
    g_pred = pd.concat([pd.to_numeric(k[c], errors="coerce").astype(float).where(lambda s: (s >= GMIN) & (s <= tc)) for c in GURS], axis=1).notna().any(axis=1)
    k["gurs_pred_tc"] = g_pred
    k["tpv"] = pd.to_numeric(k["leto_PV"], errors="coerce").astype(float) - tc
    k["regija"] = k["ID_Regije"].astype(str).replace({"": "neznano", "nan": "neznano", "<NA>": "neznano", "None": "neznano"})
    k["energent"] = k["staro_ogrevanje"].replace({"": "neznano"}).fillna("neznano")
    k["logA"] = np.log(k["A_ogr"].astype(float).clip(40, 600).fillna(k["A_ogr"].median()))
    fq = HERE / "cache" / "01_Qr_gurs2010.feather"                     # krog 13: razred Qh,nd ob vgradnji po pravilu GURS ≥ 2010 (01_tipicna_hisa)
    if GMIN != 2003 and fq.exists():
        k["Qr"] = k["ID_Stavbe"].astype(str).map(pd.read_feather(fq).set_index("ID_Stavbe")["Qr"]).fillna(k["Qr"])
    k["Qr"] = k["Qr"].fillna("neznano")
    k["obdobje"] = k["obdobje"].astype(int).astype(str)
    k["leto_TC"] = tc.astype(int)
    return k, end


def cas(k, end, ev_col):
    ev = k[ev_col]
    T = np.where(ev.notna(), ev - k["leto_TC"], end - k["leto_TC"]).astype(float)
    return T, ev.notna().to_numpy().astype(int)


def epizode(k, T, st, zamik=0):
    """Štetni proces (start, stop] s PV kot časovno odvisno spremenljivko; preklop v času tpv − zamik."""
    sw = k["tpv"].to_numpy() - zamik
    base = pd.DataFrame({"i": np.arange(len(k)), "T": T, "st": st, "sw": sw})
    pv0 = base["sw"].notna() & (base["sw"] <= 0)
    split = base["sw"].notna() & (base["sw"] > 0) & (base["sw"] < base["T"])
    a = base[~split].assign(start=0.0, stop=lambda x: x["T"], status=lambda x: x["st"], PV=pv0[~split].astype(int))
    b1 = base[split].assign(start=0.0, stop=lambda x: x["sw"], status=0, PV=0)
    b2 = base[split].assign(start=lambda x: x["sw"], stop=lambda x: x["T"], status=lambda x: x["st"], PV=1)
    return pd.concat([a, b1, b2], ignore_index=True)[["i", "start", "stop", "status", "PV"]]


def X(k, idx, pv):
    cov = pd.get_dummies(k[["obdobje", "regija", "Qr", "energent"]].iloc[idx].reset_index(drop=True), drop_first=True, dtype=float)
    cov = cov.loc[:, cov.sum() >= 30]                                      # redke kategorije v referenco
    return pd.concat([pd.Series(pv, name="PV"), k["logA"].iloc[idx].reset_index(drop=True).rename("log A_ogr"), cov], axis=1)


def cox(ep, k, strata_col="leto_TC", label=""):
    """Krog 10 (popravek): statsmodels PHReg šteje v tvegano množico tudi epizode z vstopom, enakim času dogodka; pri celoletnih časih je bila zato hiša
    v letu preklopa PV dvakrat v tvegani množici. Epizode z začetkom > 0 zdaj vstopijo 1e-6 pozneje (entry < t). Robustne SE = sandwich iz score
    ostankov po stavbi (statsmodels jih pri vstopu za zadnjim dogodkom stratuma ne izračuna — te epizode niso v nobeni tvegani množici, prispevek 0;
    pri vstopu 0 se ujema s statsmodels)."""
    idx = ep["i"].to_numpy()
    ex = X(k, idx, ep["PV"].to_numpy())
    s0 = ep["start"].to_numpy(float)
    m = PHReg(ep["stop"].to_numpy(), ex, status=ep["status"].to_numpy(), entry=np.where(s0 > 0, s0 + 1e-6, s0),
              strata=k[strata_col].to_numpy()[idx] if isinstance(strata_col, str) else strata_col, ties="efron")
    r = m.fit(method="newton", maxiter=200)
    if not np.all(np.isfinite(r.params)):                    # Newton se razide pri kategoriji brez dogodkov (npr. regija_14 v Q1) -> BFGS
        r = m.fit(method="bfgs", maxiter=2000)
    j = list(ex.columns).index("PV")
    H = np.asarray(r.cov_params())
    g = pd.DataFrame(np.nan_to_num(np.asarray(r.score_residuals))).groupby(idx).sum().to_numpy()
    b, se, se_vrsta = r.params[j], float(np.sqrt((H @ (g.T @ g) @ H)[j, j])), "robustne (po stavbi)"
    if not np.isfinite(se):                                   # robustna matrika ni izračunljiva (redke kategorije) -> naivne SE
        se, se_vrsta = r.bse[j], "naivne (robustne niso izračunljive)"
    return {"model": label, "SE": se_vrsta, "HR PV": np.exp(b), "IZ spodaj": np.exp(b - 1.96 * se), "IZ zgoraj": np.exp(b + 1.96 * se),
            "p": math.erfc(abs(b / se) / math.sqrt(2)), "stavb": int(ep["i"].nunique()), "dogodkov": int(ep["status"].sum()),
            "HR log A_ogr": float(np.exp(r.params[list(ex.columns).index("log A_ogr")])), "max |gradient|": float(np.abs(m.score(r.params)).max())}


def cem(k, end):
    tpv = k["tpv"]
    treat = tpv.between(-3, 0)
    ctrl = tpv.isna() | (tpv > 0)
    g = k[treat | ctrl].copy()
    g["paket"] = treat[treat | ctrl].astype(int)
    T, st = cas(g, end, "leto_ovoj_eko")
    # kontrole, ki pozneje dobijo PV: cenzura ob PV (ukrep v letu PV se še šteje)
    c_pv = (g["paket"] == 0) & g["tpv"].notna() & (g["tpv"] < T)
    T = np.where(c_pv, g["tpv"], T)
    st = np.where(c_pv & (g["leto_ovoj_eko"] - g["leto_TC"] > g["tpv"]), 0, st)
    st = np.where(c_pv & g["leto_ovoj_eko"].isna(), 0, st)
    g["Abin"] = pd.qcut(g["logA"], 4, labels=False, duplicates="drop").astype(str)
    g["celica"] = g[["obdobje", "Abin", "regija", "leto_TC", "Qr", "energent"]].astype(str).agg("|".join, axis=1)
    both = g.groupby("celica")["paket"].transform(lambda s: s.nunique() == 2)
    g, T, st = g[both.to_numpy()], T[both.to_numpy()], st[both.to_numpy()]
    m = PHReg(T, g[["paket"]].to_numpy(float), status=st, strata=g["celica"].to_numpy(), ties="efron").fit(method="newton", maxiter=200)
    b, se = m.params[0], m.bse[0]
    return {"model": "M4 ujemanje (CEM), paket PV v [TČ − 3, TČ] proti brez PV, Cox po celici", "HR PV": np.exp(b),
            "IZ spodaj": np.exp(b - 1.96 * se), "IZ zgoraj": np.exp(b + 1.96 * se), "p": m.pvalues[0], "stavb": len(g),
            "dogodkov": int(st.sum()), "celic": g["celica"].nunique(), "paket v ujemanju": int(g["paket"].sum()),
            "max |gradient|": float(np.abs(m.model.score(m.params)).max())}


def incidenca(k, end, hr):
    """KM (1 − S) za paket in samo TČ (opazovano). Za 10–30 let: S_samoTČ(t) s konstantno nevarnostjo po zadnjem zanesljivem letu
    (iz zadnjih 5 let), paket po sorazmernih nevarnostih S_paket = S_samoTČ^HR (hr = {oznaka: HR}). Kratka opazovalna doba paketa
    (novejše kohorte) ne omogoča lastne ekstrapolacije."""
    tpv = k["tpv"]
    grp = np.where(tpv.between(-3, 1), "paket TČ+PV", np.where(tpv.isna() | (tpv > 1), "samo TČ", None))
    T, st = cas(k, end, "leto_ovoj_eko")
    rows, ext = [], []
    for q in ("Q1", "Q2", "vse"):
        mq = (k["Qr"] == q).to_numpy() if q != "vse" else k["Qr"].isin(["Q1", "Q2"]).to_numpy()
        S_only = None
        for g in ("samo TČ", "paket TČ+PV"):
            m = (grp == g) & mq
            sf = SurvfuncRight(T[m], st[m])
            t_, S_ = np.asarray(sf.surv_times), np.asarray(sf.surv_prob)

            def S_at(x, t_=t_, S_=S_):
                return float(S_[t_ <= x][-1]) if (t_ <= x).any() else 1.0

            def n_at(x, m=m):
                return int((T[m] >= x).sum())
            t_last = max([x for x in range(1, 16) if n_at(x) >= MIN_RISK] or [1])
            for x in range(1, t_last + 1):
                rows.append({"skupina": g, "razred": q, "leto po TČ": x, "kumulativna incidenca ovoja": 1 - S_at(x), "v tveganju": n_at(x),
                             "vir": "Kaplan–Meier"})
            if g == "samo TČ":
                x0 = max(1, t_last - 5)
                h = -np.log(S_at(t_last) / S_at(x0)) / (t_last - x0)

                def S_only(x, S_at=S_at, t_last=t_last, h=h):
                    return S_at(x) if x <= t_last else S_at(t_last) * np.exp(-h * (x - t_last))
                for x in range(t_last + 1, 31):
                    rows.append({"skupina": g, "razred": q, "leto po TČ": x, "kumulativna incidenca ovoja": 1 - S_only(x), "v tveganju": n_at(x),
                                 "vir": "ekstrapolacija (konstantna nevarnost)"})
                ext.append({"skupina": g, "razred": q, "osnova": "KM + ekstrapolacija", "stavb": int(m.sum()), "dogodkov": int(st[m].sum()),
                            "zadnje zanesljivo leto": t_last, "letna nevarnost po zadnjem letu": h,
                            **{f"F({x})": 1 - S_only(x) for x in (3, 5) + T_EXT}})
            else:
                ext.append({"skupina": g, "razred": q, "osnova": "KM (opazovano)", "stavb": int(m.sum()), "dogodkov": int(st[m].sum()),
                            "zadnje zanesljivo leto": t_last, **{f"F({x})": (1 - S_at(x)) if x <= t_last else np.nan for x in (3, 5) + T_EXT}})
                for lab, h_ in hr.items():
                    for x in range(1, 31):
                        rows.append({"skupina": g, "razred": q, "leto po TČ": x, "kumulativna incidenca ovoja": 1 - S_only(x) ** h_,
                                     "v tveganju": np.nan, "vir": f"S_samoTČ^HR, {lab}"})
                    ext.append({"skupina": g, "razred": q, "osnova": f"S_samoTČ^HR, {lab} (HR {h_:.2f})", "stavb": int(m.sum()),
                                **{f"F({x})": 1 - S_only(x) ** h_ for x in (3, 5) + T_EXT}})
    return pd.DataFrame(rows), pd.DataFrame(ext)


def slika(I):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"paket TČ+PV": "#e34948", "samo TČ": "#2a78d6"}
    mk = {"paket TČ+PV": "o", "samo TČ": "s"}
    lab = {"paket TČ+PV": ("HP+PV package", "paket TČ+PV"), "samo TČ": ("HP only", "samo TČ")}
    ttl = {"Q1": ("Q1: $Q_{h,nd}$ > 140", "Q1: $Q_{h,nd}$ > 140"), "Q2": ("Q2: 90–140", "Q2: 90–140")}
    for lang, sfx in (("en", ""), ("sl", "_sl")):
        L = 0 if lang == "en" else 1
        fig, axs = plt.subplots(1, 2, figsize=(9, 3.3), sharey=True)
        for ax, q in zip(axs, ("Q1", "Q2")):
            for g in ("paket TČ+PV", "samo TČ"):
                o = I[(I["skupina"] == g) & (I["razred"] == q) & (I["vir"] == "Kaplan–Meier")]
                ax.step(np.r_[0, o["leto po TČ"]], 100 * np.r_[0, o["kumulativna incidenca ovoja"]], where="post", color=col[g], lw=2)
                src = "ekstrapolacija (konstantna nevarnost)" if g == "samo TČ" else "S_samoTČ^HR, M1"
                e = I[(I["skupina"] == g) & (I["razred"] == q) & (I["vir"] == src) & (I["leto po TČ"] <= 20)]
                if g == "samo TČ":
                    e = pd.concat([o.tail(1), e])
                ax.plot(e["leto po TČ"], 100 * e["kumulativna incidenca ovoja"], color=col[g], lw=1.3, ls=(0, (3, 2)))
                y20 = 100 * e[e["leto po TČ"] == 20]["kumulativna incidenca ovoja"].iloc[0]
                ax.text(20.4, y20, f"{lab[g][L]}\n{y20:.0f} %", color=col[g], fontsize=7.5, va="center")
            ax.set_title(ttl[q][L], fontsize=9, loc="left")
            ax.set_xlim(0, 25)
            ax.set_xlabel("Years after heat pump installation" if lang == "en" else "Leta po vgradnji TČ")
            ax.grid(color="#e6e5e0", lw=0.6)
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
        axs[0].set_ylabel("Envelope renovation, cumulative [%]" if lang == "en" else "Prenova ovoja, kumulativno [%]")
        fig.text(0.5, 0.005, "solid: Kaplan–Meier (Eko sklad, censored at end of data); dashed: HP only extrapolated (constant hazard), "
                 "package = S_HP(t)^HR with HR from Cox M1" if lang == "en"
                 else "polna črta: Kaplan–Meier (Eko sklad, cenzura na koncu podatkov); črtkano: samo TČ ekstrapolirano (konstantna nevarnost), "
                 "paket = S_TČ(t)^HR, HR iz Coxa M1",
                 ha="center", fontsize=7, color="#52514e")
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        for ext in ("png", "svg"):
            fig.savefig(OUT_S / f"Fig_k2_lockin{sfx}.{ext}", dpi=300)
        plt.close(fig)


def main():
    k, end = kohorta()
    print(f"kohorta: {len(k)} stavb, konec podatkov {end}, leto TČ {k['leto_TC'].min()}–{k['leto_TC'].max()}")
    res = []
    T, st = cas(k, end, "leto_ovoj_eko")
    res.append(cox(epizode(k, T, st), k, label="M1 Cox, PV časovno odvisna, dogodek Eko sklad"))
    res.append(cox(epizode(k, T, st, zamik=1), k, label="M2 kot M1, PV preklopi leto prej"))
    kg = k[~k["gurs_pred_tc"]].reset_index(drop=True)
    Tg, sg = cas(kg, end, "leto_ovoj_oba")
    res.append(cox(epizode(kg, Tg, sg), kg, label="M3 kot M1, dogodek Eko sklad ali GURS obnova (brez GURS obnove ≤ leto TČ)"))
    for q in ("Q1", "Q2"):
        kq = k[k["Qr"] == q].reset_index(drop=True)
        Tq, sq = cas(kq, end, "leto_ovoj_eko")
        res.append(cox(epizode(kq, Tq, sq), kq, label=f"M1 samo {q}"))
    res.append(cem(k, end))
    R = pd.DataFrame(res)
    hr = {"M1": float(R.loc[0, "HR PV"]), "M4 CEM": float(R.loc[R["model"].str.startswith("M4"), "HR PV"].iloc[0])}
    I, E = incidenca(k, end, hr)
    R.round(4).to_csv(OUT_T / "11a_cox_modeli.csv", sep=";", encoding="utf-8-sig", index=False)
    I.round(4).to_csv(OUT_T / "11b_kumulativna_incidenca.csv", sep=";", encoding="utf-8-sig", index=False)
    E.round(4).to_csv(OUT_T / "11c_verjetnost_ovoja_paket.csv", sep=";", encoding="utf-8-sig", index=False)
    pk = E[(E["skupina"] == "paket TČ+PV") & E["osnova"].str.contains("M1")].set_index("razred")
    pk4 = E[(E["skupina"] == "paket TČ+PV") & E["osnova"].str.contains("M4")].set_index("razred")
    (CACHE / "11_verjetnost_ovoja.json").write_text(json.dumps(
        {"opis": "kumulativna verjetnost prve prenove ovoja po vgradnji paketa TČ+PV (KM + ekstrapolacija), K2_4_vzrocnost.py",
         "konec_podatkov": end, "HR": hr, **{q: {f"F{x}": float(pk.loc[q, f"F({x})"]) for x in T_EXT} for q in ("Q1", "Q2", "vse")},
         "obcutljivost_M4": {q: {f"F{x}": float(pk4.loc[q, f"F({x})"]) for x in T_EXT} for q in ("Q1", "Q2", "vse")}}, indent=1), encoding="utf-8")
    slika(I)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 90)
    print(R.round(3).to_string(index=False))
    print()
    print(E.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
