"""Krog 10 — dodatni testi vzročnosti zaklepanja (NAVODILA_CLAUDE_CODE_v10.md); osnovni izhodi 11a–11c ostanejo.

Ista kohorta kot K2_4 (28.371 stavb, 1.219 dogodkov), funkcije kohorta, cas, epizode, X, cox iz K2_4_vzrocnost.py.
Konvencija časa (kot M1): čas = leta od TČ (cela leta), dogodek v letu b; PV »vklopljen« v epizodi (sw, …], ukrep v letu PV (b = sw) se pripiše obdobju brez PV.
  1  sorazmerne nevarnosti: Schoenfeldovi ostanki M1, Grambsch–Therneau (rang časa) → 19a_ph_test.csv
     HR po letih od PV (leto PV = 0; pasovi 1–2, 3–5, 6+) za M1 in CEM (M4) → 19b_hr_po_letih_od_pv.csv
  2  placebo »bodoči PV«: vodilni kazalniki pred PV (leto PV, 1–2 leti, 3+ let pred PV) + PV; znotraj hiš, ki PV kadarkoli dobijo → 19c_placebo_bodoci_pv.csv
  3–6 (po pregledu 19a–19c)
Zagon: osnovna anaconda (statsmodels); 07_stavbe_paket se bere iz feather kopije (K10_FEATHER), ker pyarrow 19 te parquet datoteke ne prebere.
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.duration.hazard_regression import PHReg

HERE = Path(__file__).resolve().parent
OUT_T = HERE / "clanek1" / "tabele"
_rp = pd.read_parquet
if os.environ.get("K10_FEATHER"):                      # 07_stavbe_paket iz feather kopije (glej docstring)
    pd.read_parquet = lambda p, columns=None, **kw: (pd.read_feather(os.environ["K10_FEATHER"], columns=columns)
                                                     if str(p).endswith("07_stavbe_paket.parquet") else _rp(p, columns=columns, **kw))
_s = importlib.util.spec_from_file_location("k24", HERE / "K2_4_vzrocnost.py")
K = importlib.util.module_from_spec(_s)
_s.loader.exec_module(K)


def _hr(b, se):
    return {"HR": np.exp(b), "IZ spodaj": np.exp(b - 1.96 * se), "IZ zgoraj": np.exp(b + 1.96 * se), "p": math.erfc(abs(b / se) / math.sqrt(2))}


def vstop(ep):
    """Vstopni čas za PHReg. statsmodels uvrsti v tvegano množico tudi epizode z vstopom, enakim času dogodka (konvencija Stata),
    zato je pri celoletnih časih hiša v letu preklopa PV v tvegani množici dvakrat (epizoda brez PV do sw IN epizoda s PV od sw).
    Popravek (krog 10): epizode z začetkom > 0 vstopijo 1e-6 pozneje (konvencija R: entry < t). K2_4 (11a–11c) tega popravka nima."""
    s_ = ep["start"].to_numpy(float)
    return np.where(s_ > 0, s_ + 1e-6, s_) if not os.environ.get("K10_STARA_TVEGANA") else s_


def fit(ep, ex, strata, robust=True):
    """Cox (Efron). Robustne SE = sandwich iz score ostankov, grozdenih po ep["i"] (stavba ali drug grozd); statsmodels jih pri vstopu za zadnjim
    dogodkom v stratumu ne izračuna (NaN ostanki) — te epizode niso v nobeni tvegani množici, njihov prispevek je 0. Pri vstopu 0 se ujema s statsmodels."""
    m = PHReg(ep["stop"].to_numpy(), ex, status=ep["status"].to_numpy(), entry=vstop(ep), strata=strata, ties="efron")
    r = m.fit(method="newton", maxiter=200)
    H = np.asarray(r.cov_params())
    se = np.sqrt(np.diag(H))
    if robust:
        g = pd.DataFrame(np.nan_to_num(np.asarray(r.score_residuals))).groupby(ep["i"].to_numpy()).sum().to_numpy()
        se = np.sqrt(np.diag(H @ (g.T @ g) @ H))
    return m, r, pd.Series(r.params, index=ex.columns), pd.Series(se, index=ex.columns)


def cox_m(ep, k, groups=None):
    """M1-specifikacija (K2_4.cox) s popravljeno tvegano množico in robustnimi SE."""
    i = ep["i"].to_numpy()
    e = ep if groups is None else ep.assign(i=groups)
    _, _, b, se = fit(e, K.X(k, i, ep["PV"].to_numpy()), k["leto_TC"].to_numpy()[i])
    return {**_hr(b["PV"], se["PV"]), "dogodkov": int(ep["status"].sum())}


def razrez(k, T, st, cuts):
    """Epizode (start, stop] po stavbi, razrezane v točkah cuts (n, m) (NaN = brez reza); dogodek le v zadnji epizodi."""
    rows = []
    for i in range(len(k)):
        c = sorted({float(x) for x in cuts[i] if np.isfinite(x) and 0 < x < T[i]})
        pts = [0.0] + c + [float(T[i])]
        for a, b in zip(pts[:-1], pts[1:]):
            rows.append((i, a, b, st[i] if b == T[i] else 0))
    return pd.DataFrame(rows, columns=["i", "start", "stop", "status"])


# ---------------------------------------------------------------- 1a Grambsch–Therneau
def ph_test(k, end):
    T, st = K.cas(k, end, "leto_ovoj_eko")
    ep = K.epizode(k, T, st)
    idx = ep["i"].to_numpy()
    ex = K.X(k, idx, ep["PV"].to_numpy())
    m = PHReg(ep["stop"].to_numpy(), ex, status=ep["status"].to_numpy(), entry=vstop(ep),
              strata=k["leto_TC"].to_numpy()[idx], ties="efron")
    r = m.fit(method="newton", maxiter=200)
    sr = np.asarray(r.schoenfeld_residuals)
    ev = (ep["status"].to_numpy() == 1) & np.all(np.isfinite(sr), axis=1)
    R, t = sr[ev], ep["stop"].to_numpy()[ev]
    d, V, beta = len(R), np.asarray(r.cov_params()), np.asarray(r.params)
    g = stats.rankdata(t)
    gc = g - g.mean()
    sstar = beta + d * R @ V                                  # skalirani ostanki (Grambsch & Therneau 1994)
    u = gc @ R
    rows = []
    for j, c in enumerate(ex.columns):
        chi = d * (V[j] @ u) ** 2 / (V[j, j] * (gc ** 2).sum())
        rows.append({"kovariata": c, "rho (rang časa)": np.corrcoef(gc, sstar[:, j])[0, 1], "chi2": chi, "df": 1, "p": stats.chi2.sf(chi, 1)})
    chi = d * u @ V @ u / (gc ** 2).sum()
    rows.append({"kovariata": "GLOBAL", "rho (rang časa)": np.nan, "chi2": chi, "df": len(beta), "p": stats.chi2.sf(chi, len(beta))})
    out = pd.DataFrame(rows)
    out.insert(0, "model", "M1 (naivna kovarianca, transformacija časa = rang)")
    out["dogodkov"] = d
    hr = []
    for lab, zam in (("M1", 0), ("M2", 1)):
        e = K.epizode(k, T, st, zamik=zam)
        r0 = K.cox(e, k)
        for opis, r_ in (("kot 11a (vstop = čas dogodka v tvegani množici)", {**_hr(np.log(r0["HR PV"]), (np.log(r0["IZ zgoraj"]) - np.log(r0["HR PV"])) / 1.96)}),
                         ("popravljena tvegana množica (vstop < t)", cox_m(e, k))):
            hr.append({"model": f"{lab} HR PV, {opis}", "kovariata": "PV", "rho (rang časa)": np.nan, "chi2": np.nan, "df": np.nan, "p": r_["p"],
                       "HR": r_["HR"], "IZ spodaj": r_["IZ spodaj"], "IZ zgoraj": r_["IZ zgoraj"], "dogodkov": int(st.sum())})
    return pd.concat([pd.DataFrame(hr), out], ignore_index=True)


# ---------------------------------------------------------------- 1b HR po letih od PV
PASOVI = [(1, 2, "1–2"), (3, 5, "3–5"), (6, 99, "6+")]


def pas_dummies(y, pasovi):
    return pd.DataFrame({f"PV {lab} let od PV": ((y >= a) & (y <= b)).astype(float) for a, b, lab in pasovi})


def hr_po_letih(k, end):
    T, st = K.cas(k, end, "leto_ovoj_eko")
    sw = k["tpv"].to_numpy(float)
    ep = razrez(k, T, st, np.c_[sw, sw + 2, sw + 5])
    i = ep["i"].to_numpy()
    y = np.where(np.isfinite(sw[i]) & (ep["stop"].to_numpy() > sw[i]), ep["stop"].to_numpy() - sw[i], -1)   # leta od leta PV
    out = []
    ev_pas = {lab: int(((y >= a) & (y <= b) & (ep["status"].to_numpy() == 1)).sum()) for a, b, lab in PASOVI}
    pas = PASOVI if ev_pas["6+"] >= 30 else [(1, 2, "1–2"), (3, 99, "3+")]
    D = pas_dummies(y, pas)
    ex = pd.concat([D, K.X(k, i, np.zeros(len(i))).drop(columns="PV")], axis=1)
    _, _, b, se = fit(ep, ex, k["leto_TC"].to_numpy()[i])
    for a_, b_, lab in pas:
        c = f"PV {lab} let od PV"
        out.append({"model": "M1 s pasovi let od PV", "pas (leta od leta PV)": lab, **_hr(b[c], se[c]),
                    "dogodkov v pasu": int((D[c].to_numpy() * ep["status"].to_numpy()).sum()), "osebo-let v pasu": float((D[c] * (ep["stop"] - ep["start"])).sum())})
    # trend: linearni člen let od PV (omejeno na 10)
    ex2 = pd.concat([pd.Series((y > 0).astype(float), name="PV"), pd.Series(np.clip(np.where(y > 0, y, 0), 0, 10), name="PV × leta od PV"),
                     K.X(k, i, np.zeros(len(i))).drop(columns="PV")], axis=1)
    _, _, b2, se2 = fit(ep, ex2, k["leto_TC"].to_numpy()[i])
    out.append({"model": "M1 s trendom", "pas (leta od leta PV)": "PV (leto 0)", **_hr(b2["PV"], se2["PV"])})
    out.append({"model": "M1 s trendom", "pas (leta od leta PV)": "× na leto od PV", **_hr(b2["PV × leta od PV"], se2["PV × leta od PV"])})
    out += cem_po_letih(k, end)
    return pd.DataFrame(out), ev_pas


def cem_po_letih(k, end):
    """M4 (CEM) kot v K2_4.cem, učinek paketa razrezan po letih od PV (čas od TČ; paket ima PV v [TČ − 3, TČ])."""
    tpv = k["tpv"]
    treat, ctrl = tpv.between(-3, 0), tpv.isna() | (tpv > 0)
    g = k[treat | ctrl].copy()
    g["paket"] = treat[treat | ctrl].astype(int)
    T, st = K.cas(g, end, "leto_ovoj_eko")
    c_pv = (g["paket"] == 0) & g["tpv"].notna() & (g["tpv"] < T)
    T = np.where(c_pv, g["tpv"], T)
    st = np.where(c_pv & (g["leto_ovoj_eko"] - g["leto_TC"] > g["tpv"]), 0, st)
    st = np.where(c_pv & g["leto_ovoj_eko"].isna(), 0, st)
    g["Abin"] = pd.qcut(g["logA"], 4, labels=False, duplicates="drop").astype(str)
    g["celica"] = g[["obdobje", "Abin", "regija", "leto_TC", "Qr", "energent"]].astype(str).agg("|".join, axis=1)
    both = g.groupby("celica")["paket"].transform(lambda s: s.nunique() == 2).to_numpy()
    g, T, st = g[both].reset_index(drop=True), T[both].astype(float), st[both].astype(int)
    sw = np.where(g["paket"] == 1, g["tpv"], np.nan).astype(float)
    # leta od PV na koncu epizode = stop − tpv (≥ stop); rezi pri časih, ko leta od PV dosežejo 2 / 5
    ep = razrez(g, T, st, np.c_[sw + 2, sw + 5])
    i = ep["i"].to_numpy()
    y = np.where(g["paket"].to_numpy()[i] == 1, ep["stop"].to_numpy() - sw[i], -1)
    ev = {lab: int(((y >= a) & (y <= b) & (ep["status"].to_numpy() == 1)).sum()) for a, b, lab in PASOVI}
    pas = PASOVI if ev["6+"] >= 30 else [(1, 2, "1–2"), (3, 99, "3+")]
    D = pas_dummies(y, pas)
    D = D.loc[:, D.sum() > 0]
    _, _, b, se = fit(ep, D, g["celica"].to_numpy()[i], robust=True)
    return [{"model": "M4 CEM s pasovi let od PV", "pas (leta od leta PV)": c.replace("PV ", "").replace(" let od PV", ""), **_hr(b[c], se[c]),
             "dogodkov v pasu": int((D[c].to_numpy() * ep["status"].to_numpy()).sum()), "osebo-let v pasu": float((D[c] * (ep["stop"] - ep["start"])).sum())}
            for c in D.columns]


# ---------------------------------------------------------------- 2 placebo »bodoči PV«
def placebo(k, end):
    T, st = K.cas(k, end, "leto_ovoj_eko")
    sw = k["tpv"].to_numpy(float)
    rez = np.c_[sw - 3, sw - 1, sw]                           # (…, sw−3] 3+ let pred PV, (sw−3, sw−1] 1–2 leti, (sw−1, sw] leto PV, (sw, …] PV
    ep = razrez(k, T, st, rez)
    i = ep["i"].to_numpy()
    b_ = ep["stop"].to_numpy()
    s = sw[i]
    do_pv = np.where(np.isfinite(s), s - b_, np.nan)          # leta do leta PV na koncu epizode
    L = pd.DataFrame({"PV": (do_pv < 0).astype(float),
                      "pred PV: leto PV": (do_pv == 0).astype(float),
                      "pred PV: 1–2 leti": ((do_pv >= 1) & (do_pv <= 2)).astype(float),
                      "pred PV: 3+ let": (do_pv >= 3).astype(float)})
    status = ep["status"].to_numpy()
    out = []

    def zapis(model, ex, strata, ep_=ep, opis=""):
        _, _, b, se = fit(ep_, ex, strata)
        st_ = ep_["status"].to_numpy()
        for c in [c for c in ex.columns if c.startswith(("PV", "pred PV"))]:
            out.append({"model": model, "spremenljivka": c, **_hr(b[c], se[c]), "dogodkov": int((ex[c].to_numpy() * st_).sum()),
                        "osebo-let": float((ex[c] * (ep_["stop"] - ep_["start"])).sum()), "opomba": opis})

    cov = K.X(k, i, np.zeros(len(i))).drop(columns="PV")
    zapis("2a M1 + vodilni kazalniki (vse stavbe)", pd.concat([L, cov], axis=1), k["leto_TC"].to_numpy()[i],
          opis="referenca = epizode brez PV pri hišah, ki PV do konca podatkov ne dobijo")
    L2 = L.copy()
    L2["PV"] = L2["PV"] + L2.pop("pred PV: leto PV")           # kot M2: ukrep v letu PV se pripiše PV
    zapis("2a varianta: leto PV pripisano PV (kot M2)", pd.concat([L2, cov], axis=1), k["leto_TC"].to_numpy()[i])
    # 2b znotraj hiš, ki PV kadarkoli dobijo (do konca podatkov): pred PV proti po PV; strata = leto TČ × obdobje
    for lab, m_ in (("2b hiše s PV kadarkoli (vključno PV pred TČ)", np.isfinite(s)),
                    ("2b hiše s PV po TČ (obe obdobji v isti hiši)", np.isfinite(s) & (sw[i] > 0))):
        e = ep[m_].reset_index(drop=True)
        ii = e["i"].to_numpy()
        dd = sw[ii] - e["stop"].to_numpy()
        ex = pd.DataFrame({"PV": (dd < 0).astype(float), "pred PV: leto PV": (dd == 0).astype(float)})
        cv = K.X(k, ii, np.zeros(len(ii))).drop(columns="PV")
        cv = cv[[c for c in cv.columns if not c.startswith("obdobje")]]
        strata = (k["leto_TC"].astype(str) + "|" + k["obdobje"].astype(str)).to_numpy()[ii]
        zapis(lab, pd.concat([ex, cv], axis=1), strata, ep_=e, opis=f"{len(np.unique(ii))} stavb; referenca = leta ≥ 1 pred PV")
    # obratna smer: ali prenova ovoja zmanjša poznejši prevzem PV? (če da, so vodilni kazalniki mehansko < 1 — razvrstitev po prihodnjem PV
    # je odvisna od izida); dogodek = PV po TČ, časovno odvisna »ovoj narejen« (ukrep v letu < leto PV), cenzura na koncu podatkov
    Tp = np.where(np.isfinite(sw) & (sw > 0), sw, end - k["leto_TC"].to_numpy()).astype(float)
    sp = (np.isfinite(sw) & (sw > 0)).astype(int)
    pv0 = np.isfinite(sw) & (sw <= 0)                       # PV že ob TČ — niso v tveganju
    kk, Tp, sp = k[~pv0].reset_index(drop=True), Tp[~pv0], sp[~pv0]
    tov = (kk["leto_ovoj_eko"] - kk["leto_TC"]).to_numpy(float)
    ep2 = razrez(kk, Tp, sp, np.c_[tov])
    i2 = ep2["i"].to_numpy()
    ov = (np.isfinite(tov[i2]) & (ep2["start"].to_numpy() >= tov[i2])).astype(float)
    ex = pd.concat([pd.Series(ov, name="PV prevzem: ovoj že narejen"), K.X(kk, i2, np.zeros(len(i2))).drop(columns="PV")], axis=1)
    _, _, b, se = fit(ep2, ex, kk["leto_TC"].to_numpy()[i2])
    c = "PV prevzem: ovoj že narejen"
    out.append({"model": "obratna smer: nevarnost prevzema PV po TČ", "spremenljivka": c, **_hr(b[c], se[c]),
                "dogodkov": int((ex[c].to_numpy() * ep2["status"].to_numpy()).sum()), "osebo-let": float((ex[c] * (ep2["stop"] - ep2["start"])).sum()),
                "opomba": f"{int(sp.sum())} prevzemov PV po TČ; ovoj šteje od leta po ukrepu"})
    # opis: pri koliko hišah je leto PV znano po dogodku / cenzuri (za razlago immortal time)
    n_pv = np.isfinite(sw) & (sw > 0)
    po_dogodku = n_pv & (st == 1) & (sw >= T)
    info = {"hiš s PV po TČ": int(n_pv.sum()), "od tega PV v letu ukrepa ali pozneje (ukrep pred PV)": int(po_dogodku.sum()),
            "dogodkov skupaj": int(st.sum())}
    return pd.DataFrame(out), info


# ---------------------------------------------------------------- 3 pokritost PV (odziv na odmerek)
PV_SPEC = 1154.75          # kWh/kWp, urni model (PVGIS-SARAH2 Ljubljana; urni model nima regionalnih donosov — enak za vse)


def dodatni_stolpci(k, cols):
    f = pd.read_feather(os.environ["K10_FEATHER"], columns=["ID_Stavbe"] + cols)
    return k[["ID_Stavbe"]].merge(f, on="ID_Stavbe", how="left")


def pokritost(k):
    d = dodatni_stolpci(k, ["PV_kWp"])
    el = pd.read_feather(os.environ["K10_EL"])
    el["obdobje"] = el["obdobje"].astype(int).astype(str)
    m = k[["ID_Stavbe", "Qr", "obdobje"]].merge(el[["ID_Stavbe", "el_skupaj_kWh"]], on="ID_Stavbe", how="left")
    med_q = el.groupby(["Qr", "obdobje"])["el_skupaj_kWh"].median()
    med_o = el.groupby("obdobje")["el_skupaj_kWh"].median()
    rab = m["el_skupaj_kWh"].to_numpy(float)
    vir = np.where(np.isfinite(rab), "urni model (hiša)", "")
    for j in np.where(~np.isfinite(rab))[0]:
        key = (m["Qr"].iat[j], m["obdobje"].iat[j])
        rab[j] = med_q.get(key, med_o.get(m["obdobje"].iat[j], np.nan))
        vir[j] = "razred Qh,nd × obdobje" if key in med_q.index else "obdobje"
    kwp = pd.to_numeric(d["PV_kWp"], errors="coerce").to_numpy(float)
    return kwp * PV_SPEC / rab, kwp, vir


def odmerek(k, end):
    T, st = K.cas(k, end, "leto_ovoj_eko")
    ep = K.epizode(k, T, st)
    i = ep["i"].to_numpy()
    pok, kwp, vir = pokritost(k)
    pv = ep["PV"].to_numpy(float)
    imap = np.isfinite(k["tpv"].to_numpy(float))
    out = []
    terc = np.nanquantile(kwp[imap & np.isfinite(kwp)], [1 / 3, 2 / 3])
    for lab, x, meje, imena in (("pokritost PV / raba po TČ", pok, [0.8, 1.2], ["< 0,8", "0,8–1,2", "> 1,2"]),
                                ("kWp v tercilih", kwp, list(terc), [f"≤ {terc[0]:.1f} kWp", f"{terc[0]:.1f}–{terc[1]:.1f} kWp", f"> {terc[1]:.1f} kWp"])):
        cls = np.where(np.isfinite(x), np.digitize(x, meje), -1)          # 0, 1, 2; −1 = neznano
        koda = {f"PV {n_}": c for c, n_ in enumerate(imena)}
        koda["PV razred neznan"] = -1
        D = pd.DataFrame({c: pv * (cls[i] == v) for c, v in koda.items()})
        D = D.loc[:, D.sum() > 0]
        cov = K.X(k, i, np.zeros(len(i))).drop(columns="PV")
        _, _, b, se = fit(ep, pd.concat([D, cov], axis=1), k["leto_TC"].to_numpy()[i])
        st_ = ep["status"].to_numpy()
        for c in D.columns:
            sel = imap & (cls == koda[c])
            out.append({"razvrstitev": lab, "razred": c.replace("PV ", ""), **_hr(b[c], se[c]), "dogodkov s PV": int((D[c].to_numpy() * st_).sum()),
                        "hiš s PV": int(sel.sum()), "mediana razreda": float(np.nanmedian(x[sel])) if sel.any() else np.nan})
        # test trenda: PV + PV × indeks razreda (0, 1, 2), hiše z znanim razredom
        ok = (cls[i] >= 0) | (pv == 0)
        e2 = ep[ok].reset_index(drop=True)
        i2 = e2["i"].to_numpy()
        ex = pd.concat([pd.Series(e2["PV"].to_numpy(float), name="PV"), pd.Series(e2["PV"].to_numpy(float) * np.maximum(cls[i2], 0), name="PV × razred"),
                        K.X(k, i2, np.zeros(len(i2))).drop(columns="PV")], axis=1)
        _, _, b2, se2 = fit(e2, ex, k["leto_TC"].to_numpy()[i2])
        out.append({"razvrstitev": lab, "razred": "trend: × na razred", **_hr(b2["PV × razred"], se2["PV × razred"])})
    vv = pd.Series(vir[imap]).value_counts().to_dict()
    return pd.DataFrame(out), vv


# ---------------------------------------------------------------- 4 negativni kontrolni izid
NK = {"prezračevanje (prezrac: centralno + lokalno)": ["prezrac"],
      "vsi ne-ovojni brez kolektorjev in ogrevanja (prezrac, razsvetljava, energ_pregled, vozila)": ["prezrac", "razsvetljava", "energ_pregled", "vozila"],
      "sončni kolektorji (PV jih neposredno nadomesti)": ["kolektorji"]}


def negativna(k, end):
    cols = sorted({f"leto_{x}_{u}" for us in NK.values() for u in us for x in ("prvi", "zadnji")} | {"prezrac_leto"})
    d = dodatni_stolpci(k, cols)
    tc = k["leto_TC"].to_numpy(float)
    out = []
    for lab, us in NK.items():
        c_ = [f"leto_{x}_{u}" for u in us for x in ("prvi", "zadnji")] + (["prezrac_leto"] if "prezrac" in us else [])
        y = np.column_stack([pd.to_numeric(d[c], errors="coerce").to_numpy(float) for c in c_])
        y = np.where((y > tc[:, None]) & (y <= end), y, np.nan)
        kk = k.copy()
        kk["leto_nk"] = np.nanmin(np.where(np.isnan(y), np.inf, y), axis=1)
        kk.loc[~np.isfinite(kk["leto_nk"]), "leto_nk"] = np.nan
        T, st = K.cas(kk, end, "leto_nk")
        ep = K.epizode(kk, T, st)
        ii = ep["i"].to_numpy()
        ex = K.X(kk, ii, ep["PV"].to_numpy())
        ev = ep["status"].to_numpy() == 1
        ex = ex[["PV"] + [c for c in ex.columns[1:] if c == "log A_ogr" or ex.loc[ev, c].sum() >= 5]]   # redke kategorije (< 5 dogodkov) v referenco
        try:
            _, _, b, se = fit(ep, ex, kk["leto_TC"].to_numpy()[ii])
            vrsta = "robustne (po stavbi)"
        except np.linalg.LinAlgError:
            try:
                _, _, b, se = fit(ep, ex, kk["leto_TC"].to_numpy()[ii], robust=False)
                vrsta = "naivne"
            except np.linalg.LinAlgError:                     # krog 13: premalo dogodkov (npr. kolektorji) — ni ocenljivo
                b, se, vrsta = pd.Series({"PV": np.nan}), pd.Series({"PV": np.nan}), "ni ocenljivo (premalo dogodkov)"
        out.append({"izid": lab, **_hr(b["PV"], se["PV"]), "dogodkov": int(st.sum()), "dogodkov s PV": int((ep["PV"].to_numpy() * ep["status"].to_numpy()).sum()),
                    "SE": vrsta, "kode ukrepov": ", ".join(us)})
    T, st = K.cas(k, end, "leto_ovoj_eko")
    r = cox_m(K.epizode(k, T, st), k)
    out.append({"izid": "primerjava: ovoj (M1)", **r, "SE": "robustne (po stavbi)",
                "kode ukrepov": "fasada, okna_vrata, streha, celovita_prenova"})
    return pd.DataFrame(out)


# ---------------------------------------------------------------- 6 grozdenje po občini, IPW
def cox_utezeni(ep, x, w, strata, groups):
    """Utežen Cox (Breslow), ena kovariata, štetni proces, strata; robustna (sandwich) SE po skupinah."""
    t0, t1, s, x, w = (ep["start"].to_numpy(float), ep["stop"].to_numpy(float), ep["status"].to_numpy(), np.asarray(x, float), np.asarray(w, float))
    kl = [(np.where(strata == z)[0]) for z in np.unique(strata)]
    times = [(ix, np.unique(t1[ix][s[ix] == 1])) for ix in kl]

    def score_info(b):
        U, I = 0.0, 0.0
        for ix, ts in times:
            for t in ts:
                R = ix[(t0[ix] < t) & (t1[ix] >= t)]
                Dm = ix[(t1[ix] == t) & (s[ix] == 1)]
                e = w[R] * np.exp(b * x[R])
                S0, S1, S2 = e.sum(), (e * x[R]).sum(), (e * x[R] ** 2).sum()
                dw = w[Dm].sum()
                U += (w[Dm] * x[Dm]).sum() - dw * S1 / S0
                I += dw * (S2 / S0 - (S1 / S0) ** 2)
        return U, I
    b = 0.0
    for _ in range(30):
        U, I = score_info(b)
        b += U / I
        if abs(U / I) < 1e-9:
            break
    _, I = score_info(b)
    res = np.zeros(len(x))
    for ix, ts in times:
        for t in ts:
            R = ix[(t0[ix] < t) & (t1[ix] >= t)]
            Dm = ix[(t1[ix] == t) & (s[ix] == 1)]
            e = w[R] * np.exp(b * x[R])
            xb = (e * x[R]).sum() / e.sum()
            dw = w[Dm].sum()
            res[Dm] += w[Dm] * (x[Dm] - xb)
            res[R] -= e * (x[R] - xb) * dw / e.sum()
    g = pd.Series(res).groupby(groups).sum().to_numpy()
    se = math.sqrt((g ** 2).sum()) / I
    return b, se, math.sqrt(1 / I)


def grozdi_ipw(k, end):
    import statsmodels.api as sm
    T, st = K.cas(k, end, "leto_ovoj_eko")
    ep = K.epizode(k, T, st)
    i = ep["i"].to_numpy()
    d = dodatni_stolpci(k, ["ID_Obcine"])
    obc = d["ID_Obcine"].astype(str).fillna("neznano").to_numpy()
    out = []
    ex = K.X(k, i, ep["PV"].to_numpy())
    _, _, b, se = fit(ep.assign(i=pd.factorize(obc[i])[0]), ex, k["leto_TC"].to_numpy()[i])
    out.append({"model": "M1, robustne SE grozdene po občini", **_hr(b["PV"], se["PV"]), "grozdov": int(len(np.unique(obc)))})
    # IPW: logit P(PV kadarkoli do 2025 | kovariate M1 + leto TČ), stabilizirane uteži, odrezane na 1./99. percentilu; Cox s PV (časovno odvisna), utežen
    Z = pd.concat([K.X(k, np.arange(len(k)), np.zeros(len(k))).drop(columns="PV"),
                   pd.get_dummies(k["leto_TC"].astype(str), prefix="leto", drop_first=True, dtype=float).reset_index(drop=True)], axis=1)
    a = np.isfinite(k["tpv"].to_numpy(float)).astype(int)
    lg = sm.Logit(a, sm.add_constant(Z)).fit(disp=0, maxiter=200)
    p = lg.predict(sm.add_constant(Z))
    sw = np.where(a == 1, a.mean() / p, (1 - a.mean()) / (1 - p))
    lo, hi = np.percentile(sw, [1, 99])
    sw = np.clip(sw, lo, hi)
    b_u, se_u, _ = cox_utezeni(ep, ep["PV"].to_numpy(), np.ones(len(ep)), k["leto_TC"].to_numpy()[i], i)
    b_w, se_w, _ = cox_utezeni(ep, ep["PV"].to_numpy(), sw[i], k["leto_TC"].to_numpy()[i], i)
    out.append({"model": "Cox s PV samo (Breslow, strata leto TČ), neutežen — kontrola", **_hr(b_u, se_u)})
    out.append({"model": "IPW: stabilizirane uteži P(PV kadarkoli), odrezane 1./99. p., Cox s PV samo, robustne SE po stavbi", **_hr(b_w, se_w),
                "uteži min / mediana / maks": f"{sw.min():.2f} / {np.median(sw):.2f} / {sw.max():.2f}", "AUC logit": float(_auc(a, p))})
    return pd.DataFrame(out)


def _auc(y, p):
    r = stats.rankdata(p)
    n1 = y.sum()
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))


def main(korak):
    k, end = K.kohorta()
    T, st = K.cas(k, end, "leto_ovoj_eko")
    ep0 = K.epizode(k, T, st)
    print(f"kohorta {len(k)} stavb, konec {end}; kontrola M1 kot 11a (HR 0,5221):", K.cox(ep0, k, label="M1")["HR PV"],
          "; M1 s popravljeno tvegano množico:", cox_m(ep0, k))
    if korak in ("1", "12"):
        A = ph_test(k, end)
        A.round(4).to_csv(OUT_T / "19a_ph_test.csv", sep=";", encoding="utf-8-sig", index=False)
        B, evp = hr_po_letih(k, end)
        B.round(4).to_csv(OUT_T / "19b_hr_po_letih_od_pv.csv", sep=";", encoding="utf-8-sig", index=False)
        pd.set_option("display.width", 250)
        print(A.round(3).to_string(index=False))
        print("dogodki po pasovih:", evp)
        print(B.round(3).to_string(index=False))
    if korak in ("2", "12"):
        Cc, info = placebo(k, end)
        Cc.round(4).to_csv(OUT_T / "19c_placebo_bodoci_pv.csv", sep=";", encoding="utf-8-sig", index=False)
        print(Cc.round(3).to_string(index=False))
        print(info)
    if korak == "3":
        Dd, vv = odmerek(k, end)
        Dd.round(4).to_csv(OUT_T / "19d_hr_po_pokritosti_pv.csv", sep=";", encoding="utf-8-sig", index=False)
        print(Dd.round(3).to_string(index=False))
        print("vir rabe za hiše s PV:", vv)
    if korak == "4":
        E = negativna(k, end)
        E.round(4).to_csv(OUT_T / "19e_negativna_kontrola.csv", sep=";", encoding="utf-8-sig", index=False)
        print(E.round(3).to_string(index=False))
    if korak == "6":
        F = grozdi_ipw(k, end)
        F.round(4).to_csv(OUT_T / "19f_grozdenje_ipw.csv", sep=";", encoding="utf-8-sig", index=False)
        print(F.round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--korak", default="12")
    main(ap.parse_args().korak)
