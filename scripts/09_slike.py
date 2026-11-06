"""Korak 09 — slike za članek (CLAUDE.md §9). Angleški napisi (osnovne datoteke) + slovenske različice s pripono _sl.

Pravila: brez naslova v sliki; ena y-os na panel (okolje v svojem panelu); enotna paleta (preverjena z validatorjem dataviz):
poti P1 rdeča #e34948, P2 rumena #eda100, P3 zelena #008300, P6 vijolična #4a3aa7, P0 nevtralno siva; režimi R1 vijolična,
R2 oranžna, R3 modra, država rumena. Rdeča/zelena sta v pasu CVD 6–8 → vedno z neposrednimi oznakami in različnimi markerji.
Izhodi: slike/09_F1 … 09_F8 (.png 300 dpi, .svg; _sl)
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

HERE = Path(__file__).resolve().parent
PAR = yaml.safe_load((HERE / "parametri.yaml").read_text(encoding="utf-8"))
TAB, CACHE, SL = HERE / "tabele", HERE / "cache", HERE / "slike"
ANALIZA = (HERE / PAR["poti"]["analiza"]).resolve()
SL.mkdir(exist_ok=True)

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
PATH_C = {"P0": "#9a9893", "P1": "#e34948", "P1b": "#e34948", "P2": "#eda100", "P3": "#008300", "P3b": "#008300",
          "P4": "#008300", "P5": "#008300", "P6": "#4a3aa7", "PA": "#9a9893", "PC": "#008300"}
PATH_M = {"P0": "o", "P1": "o", "P1b": "o", "P2": "D", "P3": "s", "P3b": "s", "P4": "s", "P5": "s", "P6": "^", "PA": "o", "PC": "s"}
REG_C = {"R1": "#4a3aa7", "R2": "#eb6834", "R3": "#2a78d6", "drzava": "#eda100"}
SEQ = ["#c6dbef", "#9ecae1", "#6baed6", "#3182bd", "#08519c"]          # ena barva, svetlo → temno (stopnje prilagodljivosti)
ENV_C = {"BREZ": "#d9d8d3", "F": "#c6dbef", "FS": "#9ecae1", "FSO": "#4292c6", "GLOB": "#08519c"}
plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "legend.frameon": False, "lines.linewidth": 2, "savefig.dpi": 300, "font.family": "DejaVu Sans"})

T = {  # napisi: (en, sl)
    "paths": {"P0": ("P0 keep boiler", "P0 ostane kotel"), "P1": ("P1 HP+PV, no envelope", "P1 TČ+PV, brez ovoja"),
              "P1b": ("P1b P1 + battery", "P1b P1 + baterija"), "P2": ("P2 PV first, envelope yr 10", "P2 PV prej, ovoj v letu 10"),
              "P3": ("P3 envelope first, PV yr 1", "P3 URE-najprej, PV v letu 1"), "P3b": ("P3b P3 + battery", "P3b P3 + baterija"),
              "P4": ("P4 all at once", "P4 vse naenkrat"), "P5": ("P5 P4 + battery", "P5 P4 + baterija"), "P6": ("P6 deep (ZEB) + battery", "P6 globoka (ZEB) + baterija")},
    "env": {"BREZ": ("no envelope", "brez ovoja"), "F": ("façade", "fasada"), "FS": ("façade + roof", "fasada + streha"),
            "FSO": ("façade + roof + windows", "fasada + streha + okna"), "GLOB": ("deep (ZEB)", "globoka (ZEB)")},
}


def tr(key, lang):
    return key[0] if lang == "en" else key[1]


def save(fig, name, lang):
    sfx = "" if lang == "en" else "_sl"
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(SL / f"{name}{sfx}.{ext}")
    plt.close(fig)


def mod(file):
    spec = importlib.util.spec_from_file_location(file[:2], HERE / file)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ---------------------------------------------------------------- F1 empirija
def f1(lang):
    d = pd.read_parquet(ANALIZA / "cache" / "07_stavbe_paket.parquet", columns=["Qr", "brez_ovoja", "ima_PV"])
    d = d[d["Qr"].isin(["Q1", "Q2", "Q3", "Q4"])]
    d["brez_ovoja"], d["ima_PV"] = d["brez_ovoja"].astype(bool), d["ima_PV"].astype(bool)
    cats = [("brez_PV", "#eb6834", ("no envelope measure, no PV", "brez ukrepa na ovoju, brez PV")),
            ("s_PV", "#4a3aa7", ("no envelope measure, with PV", "brez ukrepa na ovoju, s PV")),
            ("ovoj", "#2a78d6", ("with envelope measure", "z ukrepom na ovoju"))]
    cnt = pd.DataFrame({"brez_PV": d[d.brez_ovoja & ~d.ima_PV].groupby("Qr").size(), "s_PV": d[d.brez_ovoja & d.ima_PV].groupby("Qr").size(),
                        "ovoj": d[~d.brez_ovoja].groupby("Qr").size()}).reindex(["Q4", "Q3", "Q2", "Q1"]).fillna(0) / 1000
    rr = pd.read_csv(ANALIZA / "08_T8a_RR_skupaj.csv", sep=";", encoding="utf-8-sig")
    st = pd.read_csv(ANALIZA / "08_T8b_RR_po_stratumih.csv", sep=";", encoding="utf-8-sig").dropna(subset=["RR"])
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.2), gridspec_kw={"width_ratios": [1.1, 1]})
    left = np.zeros(len(cnt))
    ql = {"en": {"Q1": "Q1 > 140", "Q2": "Q2 90–140", "Q3": "Q3 60–90", "Q4": "Q4 < 60"},
          "sl": {"Q1": "Q1 > 140", "Q2": "Q2 90–140", "Q3": "Q3 60–90", "Q4": "Q4 < 60"}}[lang]
    for k, c, lab in cats:
        a.barh([ql[q] for q in cnt.index], cnt[k], left=left, color=c, height=0.62, edgecolor="white", linewidth=1.5, label=tr(lab, lang))
        left += cnt[k].to_numpy()
    for i, v in enumerate(left):
        a.text(v + 0.3, i, f"{v:.1f} k", va="center", color=MUTED, fontsize=7.5)
    a.set_xlabel("Single/two-family houses with subsidised HP [thousand]" if lang == "en" else "Eno-/dvostanovanjske hiše s subvencionirano TČ [tisoč]")
    a.set_ylabel("$Q_{h,nd}$ class at HP installation [kWh/m²a]" if lang == "en" else "Razred $Q_{h,nd}$ ob vgradnji TČ [kWh/m²a]")
    a.legend(fontsize=7, loc="lower right")
    a.grid(axis="y", visible=False)
    rows = [("all (crude)" if lang == "en" else "vse (surovo)", rr.iloc[0]), ("MH by period×envelope" if lang == "en" else "MH po obdobju×ovoju", rr.iloc[1]),
            ("MH by cohort year" if lang == "en" else "MH po letu kohorte", rr.iloc[2]), ("logit OR, adjusted" if lang == "en" else "logit OR, prilagojeno", rr.iloc[3])]
    rows += [(("period " if lang == "en" else "obdobje ") + r["stratum"].split("/")[0] + (", env. " if lang == "en" else ", ovoj ") + r["stratum"].split("/")[1], r)
             for _, r in st.iterrows()]
    y = np.arange(len(rows))[::-1]
    for yi, (lab, r) in zip(y, rows):
        bold = yi >= len(rows) - 4
        b.plot([r["IZ spodaj"], r["IZ zgoraj"]], [yi, yi], color=INK if bold else MUTED, lw=1.4)
        b.plot(r["RR"], yi, "s" if bold else "o", color=INK if bold else MUTED, ms=6 if bold else 4)
    b.set_yticks(y)
    b.set_yticklabels([r[0] for r in rows], fontsize=7)
    b.axvline(1, color=MUTED, lw=0.8, ls="--")
    b.set_xscale("log")
    b.set_xticks([0.25, 0.5, 1, 2, 4])
    b.set_xticklabels(["0.25", "0.5", "1", "2", "4"])
    b.set_xlabel("Relative risk of envelope renovation within 3 years:\nHP+PV package vs HP only (95 % CI)" if lang == "en"
                  else "Relativno tveganje prenove ovoja v 3 letih:\npaket TČ+PV proti samo TČ (95 % IZ)")
    b.grid(axis="y", visible=False)
    save(fig, "09_F1_empirija", lang)


# ---------------------------------------------------------------- F2 poti (časovnica)
def f2(lang):
    E = mod("04_ekonomika.py")
    paths = ["P0", "P1", "P1b", "P2", "P3", "P3b", "P4", "P6"]
    ov = {"V0": "BREZ", "V1": "BREZ", "V2": "BREZ", "V3": "BREZ", "V4": "FSO", "V5": "FSO", "V6": "FSO", "V7": "GLOB", "V8": "FSO", "V9": "FSO"}
    mk = {"TC": ("v", INK), "TCz": ("v", MUTED), "PV": ("s", "#eda100"), "PVB": ("s", "#eda100"), "BAT": ("D", "#4a3aa7"), "BATz": ("D", "#9085e9"),
          "FSO": ("|", INK), "GLOB": ("|", INK), "RAZS": ("x", MUTED), "MVHRz": ("x", MUTED)}
    fig, ax = plt.subplots(figsize=(8.2, 3.4))
    for i, p in enumerate(paths[::-1]):
        segs = E.POTI[p]["seg"] + [(30, None)]
        for (y0, v), (y1, _) in zip(segs[:-1], segs[1:]):
            ax.barh(i, y1 - y0, left=y0, color=ENV_C[ov[v]], height=0.55, edgecolor="white", linewidth=1.5)
        for (y, it, _a) in E.POTI[p]["ev"]:
            m, c = mk[it]
            off = {"FSO": 0.05, "GLOB": 0.05, "TC": 0.35, "TCz": 0.35, "PV": 0.75, "PVB": 0.75, "BAT": 1.15, "BATz": 0.75, "RAZS": 0.35, "MVHRz": 0.35}[it]
            ax.plot(y + off, i, m, color=c, ms=7 if m != "|" else 14, mew=2)
    ax.set_yticks(range(len(paths)))
    ax.set_yticklabels([tr(T["paths"][p], lang) for p in paths[::-1]], fontsize=7.5)
    ax.set_xlim(0, 30)
    ax.set_xlabel("Year (0 = 2026)" if lang == "en" else "Leto (0 = 2026)")
    ax.grid(axis="y", visible=False)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    h = [Patch(color=ENV_C[k], label=tr(T["env"][k], lang)) for k in ("BREZ", "FSO", "GLOB")]
    lab = {"en": ["heat pump (replacement)", "PV", "battery", "envelope", "inverter"], "sl": ["TČ (zamenjava)", "PV", "baterija", "ovoj", "razsmernik"]}[lang]
    h += [Line2D([], [], ls="", marker=m, color=c, label=l, ms=7) for (m, c), l in zip([mk["TC"], mk["PV"], mk["BAT"], ("|", INK), mk["RAZS"]], lab)]
    ax.legend(handles=h, fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.45, -0.22))
    save(fig, "09_F2_poti", lang)


# ---------------------------------------------------------------- F3 mesečna bilanca in krivulja trajanja uvoza
def f3(lang):
    h = pd.read_parquet(CACHE / "03_bilanca_reprezentativne.parquet")
    h["ura"] = pd.to_datetime(h["ura"])
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.1), gridspec_kw={"width_ratios": [1, 1, 1.1]})
    mon = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
    for ax, v, p in ((axs[0], "V2", "P1"), (axs[1], "V5", "P3")):
        d = h[h["hisa_varianta"] == f"A3|{v}"].set_index("ura")
        m = d.resample("MS").sum()
        rab = (m["el_EPBD"] + m["gosp"]).to_numpy() / 1000
        ax.bar(range(12), rab, color="#9ecae1", width=0.8, label="consumption (HP + household)" if lang == "en" else "raba (TČ + gospodinjstvo)")
        ax.plot(range(12), m["PV"].to_numpy() / 1000, color="#eda100", marker="o", ms=4, label="PV yield" if lang == "en" else "proizvodnja PV")
        ax.plot(range(12), m["uvoz"].to_numpy() / 1000, color=PATH_C[p], marker=PATH_M[p], ms=4, label="grid import" if lang == "en" else "uvoz iz omrežja")
        for k in (0, 1, 10, 11):
            ax.axvspan(k - 0.5, k + 0.5, color=GRID, alpha=0.6, zorder=0)
        ax.set_xticks(range(12))
        ax.set_xticklabels(mon)
        ax.set_ylabel("MWh/month" if lang == "en" else "MWh/mesec")
        ax.set_ylim(0, 2.6)
        lab = {"V2": ("no envelope, HP 11 kW, PV 12.6 kWp (P1)", "brez ovoja, TČ 11 kW, PV 12,6 kWp (P1)"),
               "V5": ("envelope FSO, HP 6 kW, PV 6.5 kWp (P3)", "ovoj FSO, TČ 6 kW, PV 6,5 kWp (P3)")}[v]
        ax.text(0.02, 0.97, tr(lab, lang), transform=ax.transAxes, va="top", fontsize=7.5, color=INK)
        ax.grid(axis="x", visible=False)
    axs[0].legend(fontsize=7, loc="upper right", bbox_to_anchor=(1.0, 0.88))
    ax = axs[2]
    for v, p in (("V2", "P1"), ("V5", "P3"), ("V7", "P6")):
        d = h[h["hisa_varianta"] == f"A3|{v}"]
        x = np.sort(d["uvoz"].to_numpy())[::-1]
        ax.plot(np.arange(len(x)), x, color=PATH_C[p], lw=1.8)
        ax.annotate({"V2": "P1", "V5": "P3", "V7": "P6 (ZEB)"}[v], (60, x[60]), xytext=(250, x[60] + 0.4), fontsize=7.5, color=PATH_C[p],
                    arrowprops={"arrowstyle": "-", "color": PATH_C[p], "lw": 0.8})
    ax.set_xscale("symlog", linthresh=100)
    ax.set_xlim(0, 8760)
    ax.set_xlabel("Hours of the year, sorted" if lang == "en" else "Ure v letu, razvrščeno")
    ax.set_ylabel("Hourly grid import [kW]" if lang == "en" else "Urni uvoz iz omrežja [kW]")
    save(fig, "09_F3_bilanca", lang)


def f3a(lang):
    """Poenostavljena F3 za članek 1: samo mesečna bilanca P1 proti P3 (krivulja trajanja ostane za članek 2)."""
    h = pd.read_parquet(CACHE / "03_bilanca_reprezentativne.parquet")
    h["ura"] = pd.to_datetime(h["ura"])
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.1), sharey=True)
    mon = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
    for ax, v, p in ((axs[0], "V2", "P1"), (axs[1], "V5", "P3")):
        m = h[h["hisa_varianta"] == f"A3|{v}"].set_index("ura").resample("MS").sum()
        ax.bar(range(12), (m["el_EPBD"] + m["gosp"]).to_numpy() / 1000, color="#9ecae1", width=0.8,
               label="consumption (HP + household)" if lang == "en" else "raba (TČ + gospodinjstvo)")
        ax.plot(range(12), m["PV"].to_numpy() / 1000, color="#eda100", marker="o", ms=4, label="PV yield" if lang == "en" else "proizvodnja PV")
        ax.plot(range(12), m["uvoz"].to_numpy() / 1000, color=PATH_C[p], marker=PATH_M[p], ms=4, label="grid import" if lang == "en" else "uvoz iz omrežja")
        for k in (0, 1, 10, 11):
            ax.axvspan(k - 0.5, k + 0.5, color=GRID, alpha=0.6, zorder=0)
        ax.set_xticks(range(12))
        ax.set_xticklabels(mon)
        ax.set_ylim(0, 2.6)
        e3 = pd.read_csv(TAB / "03_energija_variante.csv", sep=";", encoding="utf-8-sig").set_index(["arhetip", "varianta"])   # krog 3: iz podatkov
        kw, kp, qh = e3.loc[("A3", v), "TC_kW"], e3.loc[("A3", v), "PV_kWp"], e3.loc[("A3", v), "Qh_nd"]
        lab = {"V2": (f"P1: no envelope ({qh:.0f} kWh/m²a), HP {kw:.0f} kW, PV {kp:.1f} kWp",
                      f"P1: brez ovoja ({qh:.0f} kWh/m²a), TČ {kw:.0f} kW, PV {kp:.1f} kWp".replace(".", ",")),
               "V5": (f"P3: envelope ({qh:.0f} kWh/m²a), HP {kw:.0f} kW, PV {kp:.1f} kWp",
                      f"P3: ovoj ({qh:.0f} kWh/m²a), TČ {kw:.0f} kW, PV {kp:.1f} kWp".replace(".", ","))}[v]
        ax.text(0.02, 0.97, tr(lab, lang), transform=ax.transAxes, va="top", fontsize=7.5, color=INK)
        ax.grid(axis="x", visible=False)
    axs[0].set_ylabel("MWh/month" if lang == "en" else "MWh/mesec")
    axs[0].legend(fontsize=7, loc="upper right", bbox_to_anchor=(1.0, 0.88))
    save(fig, "09_F3a_mesecna_bilanca", lang)


# ---------------------------------------------------------------- F4 trije zorni koti (Q1, Q2; razlika glede na P1)
def f4(lang):
    npv = pd.read_parquet(CACHE / "04_npv_fond.parquet")
    c6 = pd.read_parquet(CACHE / "06_co2_poti_fond.parquet")
    qr = npv.drop_duplicates("ID_Stavbe").set_index("ID_Stavbe")["Qr"]
    DR = mod("05_drzava.py")
    w = DR.EKM.load_units("fond")
    st = DR.drzava(w)
    paths = ["P1b", "P2", "P3", "P3b", "P4", "P6"]
    panels = [("R1", "Household R1 [€ thousand]", "Posameznik R1 [tisoč €]"), ("R2", "Household R2 [€ thousand]", "Posameznik R2 [tisoč €]"),
              ("R3", "Household R3 [€ thousand]", "Posameznik R3 [tisoč €]"), ("drzava", "Public NPV [€ thousand]", "Javni NPV [tisoč €]"),
              ("okolje", "CO$_2$e, 30 years [t]", "CO$_2$e, 30 let [t]")]
    fig, axs = plt.subplots(2, 5, figsize=(11, 4.4), sharey=True)
    for r_, q in enumerate(["Q1", "Q2"]):
        for c_, (k, en, sl) in enumerate(panels):
            ax = axs[r_, c_]
            vals = []
            for p in paths:
                if k in ("R1", "R2", "R3"):
                    d = npv[npv["rezim"] == k].set_index("ID_Stavbe")
                    x = ((d[p] - d["P1"])[qr.reindex(d.index) == q]).median() / 1000
                elif k == "drzava":
                    x = ((st[p]["javni_NPV"] - st["P1"]["javni_NPV"])[w["Qr"] == q]).median() / 1000
                else:
                    cc = c6.set_index("ID_Stavbe")
                    x = ((cc[f"{p}|skupaj_t"] - cc["P1|skupaj_t"])[qr.reindex(cc.index) == q]).median()
                vals.append(x)
            y = np.arange(len(paths))[::-1]
            for yi, p, x in zip(y, paths, vals):
                ax.plot([0, x], [yi, yi], color=GRID, lw=2, zorder=1)
                ax.plot(x, yi, PATH_M[p], color=PATH_C[p], ms=7, zorder=2)
            ax.axvline(0, color=MUTED, lw=0.9)
            ax.grid(axis="y", visible=False)
            if r_ == 1:
                ax.set_xlabel(en if lang == "en" else sl, fontsize=7.5)
            if c_ == 0:
                ax.set_yticks(y)
                ax.set_yticklabels([tr(T["paths"][p], lang) for p in paths], fontsize=7)
                ax.text(-0.9, 0.5, q, transform=ax.transAxes, fontsize=10, weight="bold", va="center")
    fig.text(0.5, 0.005, "Difference to P1 (HP+PV, no envelope); left of 0 = better than P1" if lang == "en"
             else "Razlika glede na P1 (TČ+PV, brez ovoja); levo od 0 = boljše od P1", ha="center", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    for ext in ("png", "svg"):
        fig.savefig(SL / f"09_F4_trije_zorni_koti{'' if lang == 'en' else '_sl'}.{ext}")
    plt.close(fig)


# ---------------------------------------------------------------- F5 mejna vrednost ovoja s PV in brez PV
def f5(lang):
    npv = pd.read_parquet(CACHE / "04_npv_fond.parquet")
    # krog 4: skupini po obdobju gradnje (kot v članku: A1–A3 pred 1981, A4 1981–2002), ne po razredu Qh,nd ob vgradnji
    npv["obd"] = np.where(npv["arhetip"].isin(["A1", "A2", "A3"]), "pre", "post")
    g = npv[npv["rezim"].isin(["R1", "R2", "R3"])].groupby(["obd", "rezim"])[["mejna_ovoj_s_PV", "mejna_ovoj_brez_PV"]].median() / 1000
    fig, axs = plt.subplots(1, 2, figsize=(7.5, 3), sharey=True)
    for ax, q in zip(axs, ["pre", "post"]):
        d = g.loc[q]
        x = np.arange(3)
        ax.bar(x - 0.19, d["mejna_ovoj_s_PV"], 0.36, color="#eb6834", label="PV already installed (P2 vs P1)" if lang == "en" else "PV že postavljen (P2 proti P1)")
        ax.bar(x + 0.19, d["mejna_ovoj_brez_PV"], 0.36, color="#2a78d6", label="no PV (PB vs PA)" if lang == "en" else "brez PV (PB proti PA)")
        for xi, (a_, b_) in enumerate(zip(d["mejna_ovoj_s_PV"], d["mejna_ovoj_brez_PV"])):
            ax.text(xi - 0.19, a_ - 0.4, f"{a_:.1f}", ha="center", va="top", fontsize=7, color=INK)
            ax.text(xi + 0.19, b_ - 0.4, f"{b_:.1f}", ha="center", va="top", fontsize=7, color=INK)
        ax.set_xticks(x)
        ax.set_xticklabels(["R1 net metering", "R2 hourly", "R3 + capacity tariff"] if lang == "en" else ["R1 net metering", "R2 urno", "R3 + omrežnina moči"], fontsize=7.5)
        ax.axhline(0, color=MUTED, lw=0.9)
        ax.set_title({"pre": ("pre-1981", "pred 1981"), "post": ("1981–2002", "1981–2002")}[q][0 if lang == "en" else 1],
                     loc="left", fontsize=9.5, weight="bold", color=INK)
        ax.grid(axis="x", visible=False)
    axs[0].set_ylabel("NPV of envelope (FSO) added in year 10\nto the household [€ thousand], median" if lang == "en" else
                      "NPV ovoja (FSO), dodanega v letu 10,\nza posameznika [tisoč €], mediana")
    lo = min(g.min().min(), 0) - 0.9
    for ax in axs:
        ax.set_ylim(lo, max(g.max().max(), 0) + 0.5)
    h, l_ = axs[0].get_legend_handles_labels()
    fig.legend(h, l_, fontsize=7, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    for ext in ("png", "svg"):
        fig.savefig(SL / f"09_F5_mejna_vrednost_ovoja{'' if lang == 'en' else '_sl'}.{ext}")
    plt.close(fig)


# ---------------------------------------------------------------- F5b razbremenitev omrežja (03b, A3, 8 kWp)
def f5b(lang):
    k = pd.read_csv(TAB / "03b_konica_po_ovoju.csv", sep=";", encoding="utf-8-sig")
    k = k[(k["arhetip"] == "A3") & (~k["EV"].astype(bool))]
    ovs = ["BREZ", "F", "FS", "FSO", "GLOB"]
    sets = [("pravila", 0, 0, ("rule-based", "pravila")), ("O2", 0, 0, ("optimised", "optimirano")),
            ("O2", 0, 800, ("opt. + 800 L tank", "opt. + hranilnik 800 L")), ("O2", 10, 0, ("opt. + 10 kWh battery", "opt. + baterija 10 kWh")),
            ("O2", 10, 800, ("opt. + tank + battery", "opt. + hranilnik + baterija"))]
    fig, ax = plt.subplots(figsize=(7.5, 3.1))
    wdt = 0.16
    for i, (col, bat, tank, lab) in enumerate(sets):
        v = [float(k[(k["ovoj"] == o) & (k["baterija_kWh"] == bat) & (k["hranilnik_L"] == tank)][col].iloc[0]) for o in ovs]
        ax.bar(np.arange(5) + (i - 2) * wdt, v, wdt * 0.92, color=(["#9a9893"] + SEQ[1:])[i], label=tr(lab, lang))
    ax.set_xticks(range(5))
    ax.set_xticklabels([tr(T["env"][o], lang) for o in ovs], fontsize=7.5)
    ax.set_ylabel("Peak hourly grid import of the house [kW]" if lang == "en" else "Največji urni uvoz hiše [kW]")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(axis="x", visible=False)
    save(fig, "09_F5b_razbremenitev_omrezja", lang)


# ---------------------------------------------------------------- F5c shranjevanje po ovoju
def f5c(lang):
    g = pd.read_csv(TAB / "03b_mreza.csv", sep=";", encoding="utf-8-sig")
    g = g[(~g["EV"].astype(bool)) & (g["kWp"] == 8)]
    ovs = ["BREZ", "F", "FS", "FSO", "GLOB"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(8.4, 3))
    for arh, c in (("A3", "#2a78d6"), ("A4", "#eb6834")):
        red, bat, tank = [], [], []
        for o in ovs:
            s = g[(g["arhetip"] == arh) & (g["ovoj"] == o)]
            base = s[(s["cilj"] == "pravila") & (s["baterija_kWh"] == 0) & (s["hranilnik_L"] == 0)]["uvoz_zima_kWh"].iloc[0]
            best = s[(s["cilj"] == "O1") & (s["baterija_kWh"] == 15) & (s["hranilnik_L"] == 1500)]["uvoz_zima_kWh"].iloc[0]
            red.append(100 * (1 - best / base))
            o1 = s[s["cilj"] == "O1"].set_index(["baterija_kWh", "hranilnik_L"])["racun_R2_EUR"]
            bat.append(o1.loc[(0, 0)] - o1.loc[(10, 0)])
            tank.append(o1.loc[(0, 0)] - o1.loc[(0, 800)])
        lab = {"A3": ("A3 1971–80", "A3 1971–80"), "A4": ("A4 1981–2002", "A4 1981–2002")}[arh]
        a.plot(range(5), red, marker="o", color=c, label=tr(lab, lang))
        b.plot(range(5), bat, marker="D", color=c, label=tr(lab, lang) + (" battery 10 kWh" if lang == "en" else " baterija 10 kWh"))
        b.plot(range(5), tank, marker="s", color=c, ls="--", label=tr(lab, lang) + (" tank 800 L" if lang == "en" else " hranilnik 800 L"))
    for ax in (a, b):
        ax.set_xticks(range(5))
        ax.set_xticklabels([tr(T["env"][o], lang) for o in ovs], fontsize=6.8, rotation=15)
    a.set_ylabel("Winter import reduction by optimised\nbattery 15 kWh + tank 1,500 L [%]" if lang == "en" else
                 "Znižanje zimskega uvoza z optimirano\nbaterijo 15 kWh + hranilnikom 1.500 L [%]")
    b.set_ylabel("Annual bill saving, R2 [€/a]" if lang == "en" else "Letni prihranek računa, R2 [€/a]")
    a.legend(fontsize=7)
    b.legend(fontsize=6.5)
    save(fig, "09_F5c_shranjevanje_po_ovoju", lang)


# ---------------------------------------------------------------- F6 prelomna točka (brez dvojne osi)
def f6(lang):
    ev = pd.read_csv(TAB / "07_krivulje.csv", sep=";", encoding="utf-8-sig", index_col=0)
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.5, 3.6), gridspec_kw={"width_ratios": [1.5, 1]})
    lab = {"posameznik_R1": ("Household R1 (net metering)", "Posameznik R1 (net metering)"), "posameznik_R2": ("Household R2", "Posameznik R2"),
           "posameznik_R3": ("Household R3", "Posameznik R3"), "drzava": ("Public", "Država")}
    cmap = {"posameznik_R1": REG_C["R1"], "posameznik_R2": REG_C["R2"], "posameznik_R3": REG_C["R3"], "drzava": REG_C["drzava"]}
    for arh, g in ev.groupby("arhetip"):
        main_ = arh == "A3"
        for c in cmap:
            a.plot(g["Qhnd"], g[c] / 1000, color=cmap[c], lw=2.2 if main_ else 0.7, alpha=1 if main_ else 0.35,
                   ls="-." if c == "drzava" else "-")
            if main_:
                a.text(222, g[c].iloc[-1] / 1000, tr(lab[c], lang), color=INK, fontsize=7, va="center")
        b.plot(g["Qhnd"], g["okolje_t"], color=INK, lw=2.2 if main_ else 0.7, alpha=1 if main_ else 0.35)
        if not main_:
            b.text(222, g["okolje_t"].iloc[-1], arh, fontsize=6.5, color=MUTED, va="center")
    b.text(222, ev[ev["arhetip"] == "A3"]["okolje_t"].iloc[-1], "A3", fontsize=7, color=INK, va="center", weight="bold")
    for ax in (a, b):
        ax.axhline(0, color=MUTED, lw=0.9)
        ax.set_xlabel("$Q_{h,nd}$ before renovation [kWh/m²a]" if lang == "en" else "$Q_{h,nd}$ pred prenovo [kWh/m²a]")
        ax.set_xlim(40, 270)
    a.set_ylabel("NPV(P3) − NPV(P1) [€ thousand]" if lang == "en" else "NPV(P3) − NPV(P1) [tisoč €]")
    b.set_ylabel("CO$_2$e(P3) − CO$_2$e(P1), 30 years [t]" if lang == "en" else "CO$_2$e(P3) − CO$_2$e(P1), 30 let [t]")
    a.text(0.02, 0.03, "A3 bold; A1, A2, A4 thin" if lang == "en" else "A3 debelo; A1, A2, A4 tanko", transform=a.transAxes, fontsize=7, color=MUTED)
    save(fig, "09_F6_prelomna_tocka", lang)


# ---------------------------------------------------------------- F7 tornado in Monte Carlo
def f7(lang):
    tor = pd.read_csv(TAB / "07_tornado.csv", sep=";", encoding="utf-8-sig")
    t3 = tor[tor["kazalnik"] == "posameznik_R3"].pivot_table(index="parameter", columns="vrednost", values="A3")
    base = tor[tor["kazalnik"] == "posameznik_R3"]["osnova_A3"].iloc[0] / 1000
    t3["r"] = (t3["visoko"] - t3["nizko"]).abs()
    t3 = t3[t3["r"] > 1].sort_values("r")
    names = {"diskont_zasebni": ("discount rate 2 / 7 %", "diskont 2 / 7 %"), "ovoj_strosek_faktor": ("envelope cost ×0.85 / ×1.2", "strošek ovoja ×0,85 / ×1,2"),
             "spodbude_faktor": ("subsidies ×0 / ×1.5", "spodbude ×0 / ×1,5"), "PV_strosek_faktor": ("PV cost ×0.85 / ×1.15", "strošek PV ×0,85 / ×1,15"),
             "PV_vzdrzevanje_delez": ("PV O&M 0.5 / 2 %/a", "vzdrževanje PV 0,5 / 2 %/a"), "TC_strosek_faktor": ("HP cost ×0.8 / ×1.2", "strošek TČ ×0,8 / ×1,2"),
             "TC_let": ("HP lifetime 15 / 20 a", "življ. doba TČ 15 / 20 a"), "cena_energije_faktor": ("energy price ×0.8 / ×1.3", "cena energije ×0,8 / ×1,3")}
    mc = pd.read_csv(TAB / "07_monte_carlo.csv", sep=";", encoding="utf-8-sig", index_col=0)
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.5, 3.2), gridspec_kw={"width_ratios": [1.2, 1]})
    y = np.arange(len(t3))
    a.barh(y, t3["nizko"] / 1000 - base, left=base, color="#9ecae1", label="low value" if lang == "en" else "nizka vrednost")
    a.barh(y, t3["visoko"] / 1000 - base, left=base, color="#08519c", label="high value" if lang == "en" else "visoka vrednost")
    a.set_yticks(y)
    a.set_yticklabels([tr(names.get(i, (i, i)), lang) for i in t3.index], fontsize=7)
    a.axvline(0, color=MUTED, lw=0.9)
    a.axvline(base, color=INK, lw=0.8, ls="--")
    a.set_xlabel("NPV(P3) − NPV(P1), A3, R3 [€ thousand]" if lang == "en" else "NPV(P3) − NPV(P1), A3, R3 [tisoč €]")
    a.legend(fontsize=7, loc="lower right")
    a.grid(axis="y", visible=False)
    cols = [("P(P3 < P1) posameznik_R3 [%]", REG_C["R3"], ("household R3", "posameznik R3")),
            ("P(P3 < P1) drzava [%]", "#eb6834", ("public", "država")), ("P(P3 < P1) okolje_t [%]", "#4a3aa7", ("environment", "okolje"))]
    x = np.arange(len(mc))
    for i, (c, col, lab) in enumerate(cols):
        b.bar(x + (i - 1) * 0.26, mc[c], 0.24, color=col, label=tr(lab, lang))
        for xi, v in zip(x, mc[c]):
            b.text(xi + (i - 1) * 0.26, v + 2, f"{v:.0f}", ha="center", fontsize=6.5, color=INK)
    b.set_xticks(x)
    b.set_xticklabels(mc.index)
    b.set_ylim(0, 135)
    b.set_ylabel("Probability that P3 beats P1 [%]\n(Monte Carlo, n = 1,000)" if lang == "en" else "Verjetnost, da je P3 boljši od P1 [%]\n(Monte Carlo, n = 1.000)")
    b.legend(fontsize=7, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0))
    b.grid(axis="x", visible=False)
    save(fig, "09_F7_tornado_mc", lang)


# ---------------------------------------------------------------- F8 populacija
def _stevila():
    """Krog 13: število hiš za oznake slik iz 01_fond (paket, pred 1981, 1981–2002, fond) — ne na trdo."""
    f = pd.read_parquet(Path(__file__).resolve().parent / "cache" / "01_fond.parquet", columns=["paket_osnovni", "arhetip"])
    pk, pre = f["paket_osnovni"].astype(bool), f["arhetip"].isin(["A1", "A2", "A3"])
    return {"paket": int(pk.sum()), "pred": int((pk & pre).sum()), "po": int((pk & ~pre).sum()), "fond": len(f)}


def _en(n):
    return f"{n:,}"


def _sl(n):
    return f"{n:,}".replace(",", ".")


def f8(lang):
    a8 = pd.read_csv(TAB / "08_populacija.csv", sep=";", encoding="utf-8-sig")
    a8 = a8[a8["skupina"] == "fond Q1+Q2 brez ovoja"].set_index("kazalnik")
    b8 = pd.read_csv(TAB / "08_tok_novih.csv", sep=";", encoding="utf-8-sig", header=[0, 1], index_col=0)
    sc = list(dict.fromkeys(b8.columns.get_level_values(0)))
    fig, axs = plt.subplots(1, 4, figsize=(11, 3.5), gridspec_kw={"width_ratios": [1.2, 1, 1, 1]})
    ks = ["zimski uvoz [GWh/a, leto 20]", "sistemska konica [MW, leto 20]", "CO2e 30 let [kt]"]
    lk = {"en": ["winter\nimport\n[GWh/a]", "system\npeak\n[MW]", "CO$_2$e\n30 a\n[100 kt]"],
          "sl": ["zimski\nuvoz\n[GWh/a]", "sistemska\nkonica\n[MW]", "CO$_2$e\n30 let\n[100 kt]"]}[lang]
    po = PAR["populacija"]
    kp = [c for c in a8.columns if c.startswith("pravilo s pragom")]
    ser = [("dejansko (P1/PA)", PATH_C["P1"], ("as built (P1/PA)", "dejansko (P1/PA)")),
           ("OVE pravilno dimenzionirano (P1r/PA)", "#2a78d6", ("right-sized, no envelope (P1r/PA)", "pravilno dimenzionirano, brez ovoja (P1r/PA)"))]
    if kp:
        ser.append((kp[0], "#e8a33d", (f"threshold rule (P3 > {po['prag_ovoj_Qhnd']}, else P1r)", f"pravilo s pragom (P3 > {po['prag_ovoj_Qhnd']}, sicer P1r)")))
    ser.append(("URE-najprej (P3/PC)", PATH_C["P3"], ("envelope-first (P3/PC)", "URE-najprej (P3/PC)")))
    ser = [z for z in ser if z[0] in a8.columns]
    x = np.arange(3)
    ax = axs[0]
    wb = 0.8 / len(ser)
    v1 = [a8.loc[k, ser[0][0]] / (100 if "CO2" in k else 1) for k in ks]
    for j, (col, c, ln) in enumerate(ser):
        v = [a8.loc[k, col] / (100 if "CO2" in k else 1) for k in ks]
        xo = x - 0.4 + wb * (j + 0.5)
        ax.bar(xo, v, wb, color=c, label=tr(ln, lang))
        for xi in range(3):
            ax.text(xo[xi], v[xi] + 3, f"{v[xi]:.0f}", ha="center", fontsize=5.5)
    ax.set_xticks(x)
    ax.set_xticklabels(lk, fontsize=6.5)
    ax.set_ylim(0, max(v1) * 1.6)
    ax.legend(fontsize=6, loc="upper right", frameon=False)
    ax.grid(axis="x", visible=False)
    nf = _stevila()["fond"]
    ax.set_xlabel(f"existing {_en(nf)} houses" if lang == "en" else f"obstoječih {_sl(nf)} hiš", fontsize=7, color=MUTED)
    lab_t = {"en": ["Winter import, new HPs [GWh/a]", "Contribution to system peak [MW]", "CO$_2$e [kt/a]"],
             "sl": ["Zimski uvoz, nove TČ [GWh/a]", "Prispevek h sistemski konici [MW]", "CO$_2$e [kt/a]"]}[lang]
    for ax, k, yl in zip(axs[1:], ["zimski uvoz [GWh]", "sistemska konica [MW]", "CO2e [kt]"], lab_t):
        po = PAR["populacija"]
        for s, c, ln in ((sc[0], PATH_C["P1"], ("continuation", "nadaljevanje")),
                         (sc[1], PATH_C["P3"], (f"measure A (envelope > {po['ukrep_A_prag_Qhnd']}) from {po['ukrep_A_od']}",
                                                f"ukrep A (ovoj > {po['ukrep_A_prag_Qhnd']}) od {po['ukrep_A_od']}"))):
            ax.plot(b8.index, b8[(s, k)], color=c, marker=PATH_M["P1" if s == sc[0] else "P3"], ms=3, markevery=4, label=tr(ln, lang))
        if len(sc) > 2:
            ax.plot(b8.index, b8[(sc[2], k)], color="#2a78d6", ls="--", lw=1.6,
                    label=tr((f"right-sizing only from {po['ukrep_A_od']}", f"samo dimenzioniranje od {po['ukrep_A_od']}"), lang))
        if len(sc) > 3:                                      # krog 15: pravilo s pragom (ovoj nad 180, 90–180 dimenzioniranje)
            ax.plot(b8.index, b8[(sc[3], k)], color="#e8a33d", ls="-.", lw=1.8,
                    label=tr((f"threshold rule: envelope > {po['prag_ovoj_Qhnd']}, right-sizing > {po['ukrep_A_prag_Qhnd']}",
                              f"pravilo s pragom: ovoj > {po['prag_ovoj_Qhnd']}, dimenzioniranje > {po['ukrep_A_prag_Qhnd']}"), lang))
        ax.set_ylabel(yl, fontsize=7.5)
    h, l = axs[1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(l), fontsize=6.5, frameon=False, bbox_to_anchor=(0.6, 0.0))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    for ext in ("png", "svg"):                               # brez save(): ta ponovi tight_layout brez prostora za legendo
        fig.savefig(SL / f"09_F8_populacija{'' if lang == 'en' else '_sl'}.{ext}")
    plt.close(fig)


def main():
    for f in (f1, f2, f3, f3a, f4, f5, f5b, f5c, f6, f7, f8):
        for lang in ("en", "sl"):
            f(lang)
        print("ok", f.__name__)


if __name__ == "__main__":
    main()
