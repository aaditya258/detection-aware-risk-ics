"""Colour result figures for the DAR manuscript. Reads the DAR outputs and alarm caches in $DAR_RESULTS (default ../dar).
Usage: DAR_RESULTS=<dir> python3 make_figures.py   (writes figures/*.pdf and figures/*.png)"""
import os, json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import beta as Beta

HERE = os.path.dirname(os.path.abspath(__file__))
DAR = os.environ.get("DAR_RESULTS", os.path.join(os.path.dirname(HERE), "dar"))
OUT = os.path.join(HERE, "figures"); os.makedirs(OUT, exist_ok=True)
__file__ = os.path.join(DAR, "dar.py")
exec(open(__file__, encoding="utf-8").read().split('if __name__ == "__main__":')[0])

C = {"P": "#2a78d6", "B2": "#eb6834", "B4": "#1baf7a", "B1": "#52514e", "ink": "#0b0b0b", "muted": "#8a8986"}
plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix", "font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#52514e", "axes.linewidth": 0.6, "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.5, "legend.frameon": False,
                     "savefig.bbox": "tight", "savefig.dpi": 300})
LBL = {"P": "P (proposed, $d_L$ on silence)", "B2": "B2 (detection-blind, $d=0.9$)", "B4": "B4 ($\\hat d$ on silence)"}
ASSET = {"H1": "Boiler", "H2": "Tank", "H3": "Turbine", "S1": "SWaT RO"}
TL = {"T0855-SP": "T0855 setpoint", "T0855-CO": "T0855 ctrl. output", "T0832": "T0832 reading", "T0836": "T0836 trip limit",
      "T0855": "T0855 actuator"}


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf")); fig.savefig(os.path.join(OUT, name + ".png")); plt.close(fig)


# ---------- Fig. trajectories: HAI 23.05 test1, boiler, first 4 h
P = {tuple(k.split("|")): v for k, v in json.load(open(os.path.join(DAR, "params.json"))).items()}
F, _ = load("2305"); f = F["test1"]
nb_show = 48
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.5), sharey=True)
for ax, (t, atk) in zip(axs, [("T0855-SP", "A101"), ("T0855-CO", "A103")]):
    M = run_models(f, "H1", t, P[("H1", t)], RHO_OP)
    x = np.arange(nb_show) * BLK / 3600
    ep = [e for e in f["eps"] if e["atk"] == atk and e["tech"] == t][0]
    ax.axvspan(ep["s"] / 3600, (ep["e"] + GRACE) / 3600, color="#e4e3df", lw=0)
    ax.axhline(1, color=C["B1"], lw=1, ls=(0, (1, 1.5)), label="B1 (static prior)")
    ax.axhline(0.5, color=C["muted"], lw=0.6, ls="--")
    for m, ls in [("B2", "--"), ("B4", ":"), ("P", "-")]:
        ax.step(x, M[m][:nb_show] / PIBAR, where="post", color=C[m], lw=1.6 if m == "P" else 1.3, ls=ls, label=LBL[m])
    al = np.flatnonzero(f["alerts"][("H1", t)][0][:nb_show])
    ax.plot(al * BLK / 3600, np.full(len(al), 2e3), "v", color=C["ink"], ms=4, label="monitor alert")
    ax.set_yscale("log"); ax.set_ylim(1e-3, 5e3); ax.set_xlim(0, nb_show * BLK / 3600)
    ax.set_xlabel("Time from start of HAI 23.05 test1 (h)")
    ax.set_title(f"({'a' if t == 'T0855-SP' else 'b'}) Boiler, {TL[t]}, attack {atk} (shaded)", fontsize=8.5, loc="left")
    ax.text(nb_show * BLK / 3600 * 0.99, 0.5, "0.5 prior", ha="right", va="bottom", fontsize=7, color=C["muted"])
axs[0].set_ylabel("Risk relative to prior, $\\pi_t/\\bar\\pi$ (log)")
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.2), fontsize=7.5)
save(fig, "fig4_trajectories")

# ---------- Fig. detectability intervals (HAI 22.04 development)
rows = [(h, t, v) for (h, t), v in P.items()]
fig, ax = plt.subplots(figsize=(3.5, 2.9))
for i, (h, t, v) in enumerate(rows[::-1]):
    lo, hi = Beta.ppf([0.05, 0.95], 1 + v["k"], 1 + v["n"] - v["k"])
    col = C["B2"] if t in ("T0855-SP", "T0836") else C["P"]
    ax.plot([lo, hi], [i, i], color=col, lw=2, solid_capstyle="round")
    ax.plot(v["d_hat"], i, "o", color=col, ms=5, mec="white", mew=0.8)
    ax.plot(lo, i, "|", color=C["ink"], ms=7)
    ax.text(1.02, i, f"{v['k']}/{v['n']}", va="center", fontsize=7, color=C["B1"])
ax.set_yticks(range(len(rows))); ax.set_yticklabels([f"{ASSET[h]}, {TL[t]}" for h, t, _ in rows[::-1]], fontsize=7.5)
ax.axvline(0.3, color=C["muted"], lw=0.6, ls="--"); ax.set_xlim(0, 1.12); ax.set_xticks([0, 0.3, 0.5, 1])
ax.set_xlabel("Detection probability $d$")
ax.grid(axis="y", visible=False)
from matplotlib.lines import Line2D
ax.legend([Line2D([], [], color=C["B2"], lw=2), Line2D([], [], color=C["P"], lw=2)],
          ["maskable / unobservable", "observable"], loc="upper center", bbox_to_anchor=(0.35, -0.2), ncol=2, fontsize=7)
save(fig, "figS_detectability")

# ---------- Fig. false reassurance, P vs B2 vs B4
m = pd.read_csv(os.path.join(DAR, "metrics.csv"))
sets = [("2305", "HAI 23.05"), ("2103", "HAI 21.03"), ("swat (LOO)", "SWaT (leave-one-out)")]
fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.9), gridspec_kw={"width_ratios": [10, 10, 2.6]}, sharey=True)
for ax, (s, title) in zip(axs, sets):
    d = m[m.set == s].pivot_table(index=["asset", "tech"], columns="model", values="FR").reset_index()
    order = {"T0855-SP": 0, "T0836": 1, "T0855": 1, "T0855-CO": 2, "T0832": 3}
    d["o"] = d.tech.map(order); d = d.sort_values(["o", "asset"]).reset_index(drop=True)
    x = np.arange(len(d)); w = 0.26
    for j, mdl in enumerate(["P", "B4", "B2"]):
        ax.bar(x + (j - 1) * w, d[mdl], w - 0.03, color=C[mdl], label=LBL[mdl], zorder=3)
    for xi, v in zip(x, d["P"]):
        if v == 0:
            ax.text(xi - w, 0.015, "0", ha="center", va="bottom", fontsize=6.5, color=C["P"], fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([f"{ASSET[a]}, {TL[t].split(' ', 1)[1]}" for a, t in zip(d.asset, d.tech)],
                                         rotation=55, ha="right", rotation_mode="anchor", fontsize=6.8)
    ax.set_title(title, fontsize=8.5, loc="left"); ax.set_ylim(0, 1.05); ax.grid(axis="x", visible=False)
    ninv = int(d.tech.isin(["T0855-SP", "T0836", "T0855"]).sum()) if s != "swat (LOO)" else 1
    ax.axvspan(-0.5, ninv - 0.5, color="#f1f0ec", zorder=0, lw=0)
axs[0].set_ylabel("False reassurance FR\n(share of attack blocks with $\\pi<0.5\\bar\\pi$)")
axs[0].text(1.5, 1.07, "poorly detected ($d_L<0.3$)", ha="center", fontsize=7, color=C["B1"])
for ax in axs: ax.set_ylim(0, 1.12)
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.3), fontsize=7.5)
save(fig, "fig3_false_reassurance")

# ---------- Fig. monitoring value
r = pd.read_csv(os.path.join(DAR, "rq3_residual_risk.csv"))
fig, ax = plt.subplots(figsize=(3.5, 2.3))
lab = {"historian": "Historian only", "historian + DCS internals": "+ DCS internals (HAIEnd)",
       "what-if: trip limits logged (d_L 0.9)": "What-if: trip limits logged"}
col = {"historian": C["P"], "historian + DCS internals": C["B4"], "what-if: trip limits logged (d_L 0.9)": C["B2"]}
ypos, yl = 0, []
for h in ["H1", "H2", "H3"]:
    rr = r[r.asset == h]
    blind = rr.residual_if_blind.iloc[0]
    for _, row in rr.iterrows():
        ax.barh(ypos, row.residual_quiet_risk, 0.7, color=col[row.source], zorder=3)
        ax.text(row.residual_quiet_risk + 0.2, ypos, f"{row.residual_quiet_risk:.2f}", va="center", fontsize=7)
        yl.append((ypos, f"{ASSET[h]}: {lab[row.source]}")); ypos += 1
    ax.plot([blind, blind], [ypos - len(rr) - 0.45, ypos - 0.55], color=C["ink"], lw=1.2, zorder=4)
    ypos += 0.6
ax.plot([], [], color=C["ink"], lw=1.2, label="fully blind (impact × techniques)")
ax.set_yticks([p for p, _ in yl]); ax.set_yticklabels([t for _, t in yl], fontsize=7); ax.invert_yaxis()
ax.set_xlabel("Residual quiet risk (impact-weighted)"); ax.grid(axis="y", visible=False)
ax.legend(loc="upper center", bbox_to_anchor=(0.3, -0.25), fontsize=7); ax.set_xlim(0, 18)
save(fig, "fig5_monitoring_value")
print("figures written to", OUT)
