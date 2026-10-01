"""Black-and-white process diagrams for the DAR manuscript (Fig. 1 value chain, Fig. 2 method steps).
Usage: python3 make_diagrams.py   (writes figures/fig1_value_chain.pdf and figures/fig2_method_flow.pdf at 600 dpi)"""
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures"); os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral"], "font.size": 8, "mathtext.fontset": "stix", "savefig.dpi": 600, "pdf.fonttype": 42})
K = "black"


def box(ax, x, y, w, h, text, fill="white", lw=1.0, bold=False, fs=8, ha="center"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.08", fc=fill, ec=K, lw=lw))
    tx = x + w / 2 if ha == "center" else x + 0.12
    ax.text(tx, y + h / 2, text, ha=ha, va="center", fontsize=fs, fontweight="bold" if bold else "normal", linespacing=1.3)
    return dict(x=x, y=y, w=w, h=h, l=(x, y + h / 2), r=(x + w, y + h / 2), t=(x + w / 2, y + h), b=(x + w / 2, y))


def arrow(ax, p, q, dashed=False, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=9, lw=1.0, color=K,
                                 linestyle="--" if dashed else "-", connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


def label(ax, x, y, t, **kw):
    ax.text(x, y, t, fontsize=7, style="italic", ha=kw.pop("ha", "center"), va=kw.pop("va", "center"), **kw)


# ---------------- Fig. 1: inputs, model steps, outputs (simple, no symbols)
fig, ax = plt.subplots(figsize=(7.2, 3.7)); ax.set_xlim(0, 15); ax.set_ylim(0, 7.6); ax.axis("off")
F = 8
def tbox(x, y, w, h, title, sub, fill="white", lw=1.2, fs=F):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.1", fc=fill, ec=K, lw=lw))
    if sub:
        ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center", fontsize=fs + 0.3, fontweight="bold")
        ax.text(x + w / 2, y + h * 0.30, sub, ha="center", va="center", fontsize=fs - 0.6, linespacing=1.15)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=fs - 0.4, linespacing=1.15)
    return dict(x=x, y=y, w=w, h=h, l=(x, y + h / 2), r=(x + w, y + h / 2))
BX, BY, BW, BH = 4.45, 0.1, 6.1, 7.4
ax.add_patch(FancyBboxPatch((BX, BY), BW, BH, boxstyle="round,pad=0,rounding_size=0.15", fc="#eeeeee", ec=K, lw=1.8))
ax.text(BX + BW / 2, BY + BH - 0.42, "Detection-aware risk model (this work)", ha="center", va="center", fontsize=F + 0.8, fontweight="bold")
steps = ["1. Estimate how often the monitor\ndetects each attack type",
         "2. Take a cautious lower bound\nof each detection rate",
         "3. Every 5 min, update the risk: an alert\nraises it, silence lowers it only as far\nas the lower bound allows",
         "4. Combine the risks of each loop\nwith its impact"]
sx, sw, sh, gap = BX + 0.3, BW - 0.6, 1.3, 0.32
S = []
for i, t in enumerate(steps):
    y = BY + BH - 0.95 - (i + 1) * sh - i * gap
    S.append(tbox(sx, y, sw, sh, t, None))
    if i:
        arrow(ax, (sx + sw / 2, S[i - 1]["y"]), (sx + sw / 2, y + sh))
W, IH = 3.4, 1.3
def row(i): return S[i]["y"] + (S[i]["h"] - IH) / 2
i_att = tbox(0.1, row(0), W, IH, "Recorded attacks", "development data")
i_mon = tbox(0.1, row(2), W, IH, "Process monitor", "alerts from plant data")
i_des = tbox(0.1, row(3) - 0.1, W, IH, "Design-time assessment", "prior likelihood, impact")
arrow(ax, i_att["r"], S[0]["l"])
arrow(ax, i_mon["r"], S[2]["l"])
arrow(ax, i_des["r"], (BX, i_des["r"][1]))
o1 = tbox(11.5, row(2), W, IH, "Live risk", "per attack type and loop")
o2 = tbox(11.5, row(3), W, IH, "Residual risk", "left after a day of silence")
arrow(ax, S[2]["r"], o1["l"])
arrow(ax, S[3]["r"], o2["l"])
fig.savefig(os.path.join(OUT, "fig1_value_chain.pdf"), bbox_inches="tight"); fig.savefig(os.path.join(OUT, "fig1_value_chain.png"), bbox_inches="tight", dpi=200)
plt.close(fig)

# ---------------- Fig. 2: method pipeline (offline and online bands)
from matplotlib.patches import Rectangle, Circle
steps = [
    ("Map attack types\nto evidence", "ATT&CK for ICS mapping", "link each attack type and\nloop to a monitor check",
     "loops, attacked tags,\nmonitor checks", "attack-to-check map,\nvisible or hidden"),
    ("Estimate\ndetectability", "Beta posterior", "count detected attacks and\nalerts in normal operation",
     "alerts on recorded\nattacks and normal data", "detection rate, lower\nbound, false-alert rate"),
    ("Update\nlive risk", "Hidden Markov filter", "alert uses the rate,\nsilence uses the lower bound",
     "alert or silence in each\nblock, prior attack rate", "live risk per attack\ntype and loop"),
    ("Report risk\nand blind spots", "Impact-weighted sum", "risk per loop and risk\nleft after a day of silence",
     "live risk and impact\nof each loop", "risk per loop,\nresidual risk"),
]
fig, ax = plt.subplots(figsize=(7.2, 3.3)); ax.set_xlim(0, 16); ax.set_ylim(0, 7.2); ax.axis("off")
ax.add_patch(Rectangle((0.05, 0.05), 7.85, 6.8, fc="#f0f0f0", ec="#9a9a9a", lw=0.8))
ax.add_patch(Rectangle((8.1, 0.05), 7.85, 6.8, fc="white", ec="#9a9a9a", lw=0.8, ls="--"))
ax.text(3.97, 6.55, "OFFLINE  |  development data, run once", ha="center", va="center", fontsize=7.5, fontweight="bold")
ax.text(12.02, 6.55, "ONLINE  |  every 5-min block", ha="center", va="center", fontsize=7.5, fontweight="bold")
bw, bh, y = 3.3, 2.75, 1.95
xs = [0.35, 4.3, 8.4, 12.35]
for i, (x, (title, meth, obj, inp, out)) in enumerate(zip(xs, steps)):
    ax.add_patch(FancyBboxPatch((x, y), bw, bh, boxstyle="round,pad=0,rounding_size=0.12", fc="white", ec=K, lw=1.3))
    ax.add_patch(Circle((x + 0.42, y + bh - 0.42), 0.27, fc=K, ec=K))
    ax.text(x + 0.42, y + bh - 0.42, str(i + 1), ha="center", va="center", fontsize=7.5, color="white", fontweight="bold")
    ax.text(x + 0.85, y + bh - 0.45, title, ha="left", va="center", fontsize=7.8, fontweight="bold", linespacing=1.15)
    ax.plot([x + 0.2, x + bw - 0.2], [y + 1.6, y + 1.6], color="#9a9a9a", lw=0.6)
    ax.text(x + bw / 2, y + 1.28, meth, ha="center", va="center", fontsize=6.9, fontweight="bold", style="italic")
    ax.text(x + bw / 2, y + 0.58, obj, ha="center", va="center", fontsize=6.6, linespacing=1.2)
    ax.text(x + bw / 2, y + bh + 0.72, "Input: " + inp, ha="center", va="center", fontsize=6.4, style="italic", linespacing=1.2)
    ax.add_patch(FancyArrowPatch((x + bw / 2, y + bh + 0.32), (x + bw / 2, y + bh), arrowstyle="-|>", mutation_scale=7, lw=0.8, color=K))
    ax.add_patch(FancyArrowPatch((x + bw / 2, y), (x + bw / 2, y - 0.32), arrowstyle="-|>", mutation_scale=7, lw=0.8, color=K))
    ax.text(x + bw / 2, y - 0.75, "Output: " + out, ha="center", va="center", fontsize=6.4, style="italic", linespacing=1.2)
    if i < 3:
        arrow(ax, (x + bw, y + bh / 2), (xs[i + 1], y + bh / 2))
fig.savefig(os.path.join(OUT, "fig2_method_flow.pdf"), bbox_inches="tight"); fig.savefig(os.path.join(OUT, "fig2_method_flow.png"), bbox_inches="tight", dpi=200)
plt.close(fig)
print("diagrams written")
