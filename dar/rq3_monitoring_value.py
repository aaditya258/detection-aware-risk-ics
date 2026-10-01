"""RQ3 monitoring value (descriptive, HAI 23.05 + HAIEnd 23.05): residual risk during silence per asset with
(a) historian-only evidence (frozen v5 monitor) and (b) historian + DCS-internal channels (haiend_check.py channels);
plus a what-if for configuration logging of trip limits (T0836). d and alpha for both sources are estimated on 23.05
itself, so the comparison is like-for-like; values are therefore descriptive, not held-out.
Residual risk multiplier m_t = filter posterior after 24 h of silence / prior (operational prior), P model."""
import os, sys, json, numpy as np, pandas as pd
from scipy.stats import beta as Beta
D = os.path.dirname(os.path.abspath(__file__)) + "/"
W = os.path.dirname(D.rstrip("/"))
sys.argv = ["dar.py"]
exec(open(D + "dar.py").read().split('if __name__ == "__main__":')[0])
src = open(W + "/step0e/haiend_check.py").read().split("rows, fa = [], {}")[0]
exec(src)                                  # defines tr, ho, te, CH, alarm(), thr
CH2TECH = {"IE_SP": "T0855-SP", "IE_CO": "T0855-CO", "IPL1": "T0832"}
F, assets = load("2305")
files = {"fp": ho, **{f: df for f, df, y in te}}
extra = {}
for name, df in files.items():
    nb = F[name]["nb"]
    for (h, L), chs in CH.items():
        on = []
        for c in chs:
            a = alarm(df, (h, L), c); on.append(np.flatnonzero(a[1:] & ~a[:-1]) + 1)
        on = np.concatenate(on) if on else np.array([], int)
        extra[(name, h, CH2TECH[L])] = on


def est(h, t, use_extra):
    k = n = 0; nor = al = 0
    for name, f in F.items():
        a, on = f["alerts"][(h, t)]
        if use_extra and (name, h, t) in extra:
            on = np.concatenate([on, extra[(name, h, t)]]); a = a.copy(); b = on // BLK; a[b[b < f["nb"]]] = True
        nor += f["normal"].sum(); al += (a & f["normal"]).sum()
        for ep in f["eps"]:
            if ep["asset"] == h and ep["tech"] == t:
                n += 1; k += bool(((on >= ep["s"]) & (on <= ep["e"] + GRACE)).any())
    return dict(k=k, n=n, d_hat=(1 + k) / (2 + n), d_L=float(Beta.ppf(0.05, 1 + k, 1 + n - k)), alpha=max(al / max(nor, 1), 1e-4),
                false_alerts_day=round(al / max(nor, 1) * 86400 / BLK, 2))


def quiet_mult(p):
    s = filt(np.zeros(288, bool), p["d_hat"], p["d_L"], p["alpha"], RHO_OP); return float(s[-1] / PIBAR)


rows = []
for h in ["H1", "H2", "H3"]:
    for t in TECH[h]:
        for src_name, ux in [("historian", False), ("historian + DCS internals", True)]:
            if ux and h == "H3":
                continue            # HAIEnd covers the boiler DCS only
            p = est(h, t, ux)
            rows.append(dict(asset=h, tech=t, source=src_name, attacks=p["n"], detected=p["k"], d_L=round(p["d_L"], 3),
                             false_alerts_per_day=p["false_alerts_day"], quiet_risk_multiplier=round(quiet_mult(p), 3)))
        if t == "T0836":
            p = dict(d_hat=0.95, d_L=0.9, alpha=1e-4)
            rows.append(dict(asset=h, tech=t, source="what-if: trip limits logged (d_L 0.9)", attacks=None, detected=None, d_L=0.9,
                             false_alerts_per_day=0.03, quiet_risk_multiplier=round(quiet_mult(p), 3)))
R = pd.DataFrame(rows); R.to_csv(D + "rq3_monitoring_value.csv", index=False)
agg = []
for h in ["H1", "H2", "H3"]:
    for src_name in R[R.asset == h].source.unique():
        base = R[(R.asset == h) & (R.source == "historian")].set_index("tech").quiet_risk_multiplier
        cur = R[(R.asset == h) & (R.source == src_name)].set_index("tech").quiet_risk_multiplier
        comb = base.copy(); comb.update(cur)
        agg.append(dict(asset=h, source=src_name, residual_quiet_risk=round(IMPACT[h] * comb.sum(), 2),
                        residual_if_blind=IMPACT[h] * len(base)))
A = pd.DataFrame(agg); A.to_csv(D + "rq3_residual_risk.csv", index=False)
pd.set_option("display.width", 200); print(R.to_string(index=False)); print(A.to_string(index=False))
