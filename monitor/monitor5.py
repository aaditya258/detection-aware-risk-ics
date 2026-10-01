"""Step 0b (v5 base, adds HAI 21.03 loading): detectability-bounded Bayesian integrity monitor for protection layers (no LENS code used).

Usage: python3 monitor.py 2204|2305 [rho_multiplier]
"""
import os as _os
REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
DATA = _os.environ.get("DAR_DATA", _os.path.join(REPO, "data")).replace("\\", "/")
OUTD = _os.environ.get("DAR_OUT", _os.path.join(REPO, "outputs")).replace("\\", "/")
REPO = REPO.replace("\\", "/")
import sys, json, numpy as np, pandas as pd
from sklearn.linear_model import Ridge

W = REPO
VER = sys.argv[1]
RHO = 30 / (365 * 86400) * (float(sys.argv[2]) if len(sys.argv) > 2 else 1.0)
STEP, WARM, LAGS, KAPPA, QALARM, GRACE, FAR, MERGE = 30, 120, (0, 10, 30, 60), 0.5, 0.5, 60, 900, 600
RENAME = {"x1001_15_ASSIGN_OUT": "P1_B2016", "x1002_07_SETPOINT_OUT": "P1_B3004", "x1002_08_SETPOINT_OUT": "P1_B3005",
          "x1001_05_SETPOINT_OUT": "P1_B2004", "x1003_18_SETPOINT_OUT": "P1_B4002", "x1003_24_SUM_OUT": "P1_B4022",
          "x1003_10_SETPOINT_OUT": "P1_B400B",
          # HAI 21.03 names
          "time": "timestamp", "attack": "Attack", "P2_ASD": "P2_AutoSD", "P2_MSD": "P2_ManualSD", "P2_CO_rpm": "P2_SCO"}

# ---------------------------------------------------------------- layers and evidence streams
# stream = (kind, target, spec)
LAYERS = {
    "H1": {"IE_SP": ("maskable", ["P1_B2016"], [("env", "P1_B2016", None), ("dyn", "P1_B2016", None)]),
           "IE_CO": ("observable", ["P1_PCV01D"], [("pid", "P1_PCV01D", ("P1_B2016", "P1_PIT01"))]),
           "IPL1": ("observable", ["P1_PIT01"], [("res", "P1_PIT01", ["P1_B2016", "P1_PCV01D"]),
                                                 ("res", "P1_PIT01", ["P1_PIT02", "P1_FT01"]), ("dyn", "P1_PIT01", None)]),
           "IPL2": ("observable", ["P1_PIT02"], [("res", "P1_PIT02", ["P1_PIT01", "P1_FT01", "P1_PCV02D"]), ("dyn", "P1_PIT02", None)])},
    "H2": {"IE_SP": ("maskable", ["P1_B3004"], [("env", "P1_B3004", None), ("dyn", "P1_B3004", None)]),
           "IE_CO": ("observable", ["P1_LCV01D"], [("pid", "P1_LCV01D", ("P1_B3004", "P1_LIT01"))]),
           "IPL1": ("observable", ["P1_LIT01"], [("res", "P1_LIT01", ["P1_B3004", "P1_LCV01D"]),
                                                 ("mass", "P1_LIT01", ["P1_FT01", "P1_FT02", "P1_FT03"]), ("dyn", "P1_LIT01", None)])},
    "H3": {"IE_SP": ("maskable", ["P2_AutoSD", "P2_ManualSD"], [("env", "P2_AutoSD", None), ("dyn", "P2_AutoSD", None),
                                                                ]),  # v2: ManualSD envelope dropped (constant per file, differs between files)
           "IE_CO": ("observable", ["P2_SCO"], [("pid", "P2_SCO", ("P2_AutoSD", "P2_SIT01"))]),
           "IPL1": ("observable", ["P2_SIT01"], [("res", "P2_SIT01", ["P2_AutoSD", "P2_SCO"]),
                                                 ("res", "P2_SIT01", ["P2_VT01"]), ("dyn", "P2_SIT01", None)]),
           "IPL1_limit": ("unobservable", ["P2_RTR"], []),
           "IPL2": ("observable", ["P2_VT01"], [("res", "P2_VT01", ["P2_SIT01"]), ("dyn", "P2_VT01", None)]),
           "IPL2_limit": ("unobservable", ["P2_VTR01", "P2_VTR02", "P2_VTR03", "P2_VTR04"], [])},
}
MARGIN = {"H1": ("P1_PIT01", "hi"), "H2": ("P1_LIT01", "both"), "H3": ("P2_SIT01", "hi")}


# ---------------------------------------------------------------- data
def rd(path):
    d = pd.read_csv(path).rename(columns=RENAME)
    d.columns = [c.strip() for c in d.columns]
    return d


def load():
    if VER == "2204":
        d = f"{DATA}/hai-22.04"
        fit = [rd(f"{d}/train{i}.csv") for i in range(1, 5)]
        cal = [rd(f"{d}/train5.csv")]
        fp = [rd(f"{d}/train6.csv")]
        gt = pd.read_csv(f"{REPO}/groundtruth/hai2204_attacks.csv")
        tests = []
        for f in ["test1", "test2", "test3", "test4"]:
            x = rd(f"{d}/{f}.csv"); tests.append((f, x, x["Attack"].values.astype(int)))
        gt = gt.rename(columns={"file": "file"})
    elif VER == "2103":
        d = f"{DATA}/hai-21.03"
        fit = [rd(f"{d}/train{i}.csv.gz") for i in (1, 2)]
        t3 = rd(f"{d}/train3.csv.gz"); cut = 2 * 86400
        cal = [t3.iloc[:cut].reset_index(drop=True)]
        fp = [t3.iloc[cut:].reset_index(drop=True)]
        gt = pd.read_csv(f"{REPO}/groundtruth/hai2103_attacks.csv")
        tests = []
        for f in ["test1", "test2", "test3", "test4", "test5"]:
            x = rd(f"{d}/{f}.csv.gz"); tests.append((f, x, x["Attack"].values.astype(int)))
    else:
        d = f"{DATA}/hai-23.05"
        fit = [rd(f"{d}/hai-train{i}.csv") for i in (1, 2)]
        cal = [rd(f"{d}/hai-train3.csv")]
        fp = [rd(f"{d}/hai-train4.csv")]
        gt = pd.read_csv(f"{REPO}/groundtruth/hai2305_attacks_DRAFT.csv")
        tests = []
        for f in ["test1", "test2"]:
            x = rd(f"{d}/hai-{f}.csv"); y = pd.read_csv(f"{d}/label-{f}.csv")["label"].values.astype(int)
            tests.append((f, x, y))
    return fit, cal, fp, tests, gt


# ---------------------------------------------------------------- stream features
def lagged(df, cols):
    return np.column_stack([df[c].shift(L).values for c in cols for L in LAGS])


def design(df, kind, target, spec):
    """Return (y, X) arrays for a residual-type stream."""
    if kind == "res":
        X = lagged(df, spec)
        if ARX:
            own = np.column_stack([df[target].shift(L).values for L in (1, 2, 5, 10)])
            X = np.hstack([X, own])
        return df[target].values, X
    if kind == "pid":
        sp, pv = spec
        e = df[sp] - df[pv]
        de = e.diff()
        tmp = pd.DataFrame({"e": e, "de": de})
        return df[target].diff().values, lagged(tmp, ["e", "de"])
    if kind == "mass":
        y = df[target].diff(30).values
        tmp = pd.DataFrame({c: df[c].rolling(30).mean() for c in spec})
        return y, lagged(tmp, spec)
    raise ValueError(kind)


ARX = bool(int(__import__('os').environ.get('ARX', '0')))
ADAPT_WIN, ADAPT_GAP, ADAPT_MIN = 2160, 180, 360   # in 10-s samples: 6 h window, 30 min gap, 1 h minimum


def adapt(z, med0, mad0):
    """|z - rolling median| / rolling MAD from the same file's past [t-6h, t-30min]; fixed values before 1 h of history."""
    z10 = pd.Series(z[::10])
    med = z10.rolling(ADAPT_WIN, min_periods=ADAPT_MIN).median().shift(ADAPT_GAP)
    mad = (z10 - med).abs().rolling(ADAPT_WIN, min_periods=ADAPT_MIN).median().shift(ADAPT_GAP)
    med = np.repeat(med.fillna(med0).values, 10)[:len(z)]
    mad = np.repeat(mad.fillna(mad0).values, 10)[:len(z)]
    return np.abs(z - med) / (mad + 1e-9)


def ordinal_entropy(x, w=60):
    a, b, c = x[:-2], x[1:-1], x[2:]
    code = (a > b) * 1 + (b > c) * 2 + (a > c) * 4          # 8 codes, 6 reachable, ties -> 0
    code = np.r_[0, 0, code]
    H = np.zeros(len(x))
    oh = np.eye(8)[code]
    cs = np.cumsum(np.vstack([np.zeros((1, 8)), oh]), 0)
    cnt = cs[w:] - cs[:-w]
    pr = cnt / w
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -(np.where(pr > 0, pr * np.log(pr), 0)).sum(1)
    H[w - 1:] = h
    return H


def dyn_stats(x):
    s = pd.Series(x); d = s.diff()
    return {"lsd": np.log(d.rolling(60).std().values + 1e-9),
            "ac1": d.rolling(60).corr(d.shift(1)).fillna(0).values,
            "ent": ordinal_entropy(x)}


class Stream:
    def __init__(self, kind, target, spec):
        self.kind, self.target, self.spec = kind, target, spec

    def fit(self, fit, cal):
        if self.kind in ("res", "pid", "mass"):
            ys, Xs = zip(*[design(d, self.kind, self.target, self.spec) for d in fit])
            y = np.concatenate(ys); X = np.vstack(Xs); ok = np.isfinite(y) & np.isfinite(X).all(1)
            self.m = Ridge(1.0).fit(X[ok][::5], y[ok][::5])
            r = y[ok] - self.m.predict(X[ok]); self.sd = r.std() + 1e-9
            self.med0 = {"r": float(np.median(r))}; self.mad0 = {"r": float(np.median(np.abs(r - np.median(r))) + 1e-9)}
        elif self.kind == "env":
            v = np.concatenate([d[self.target].values for d in fit])
            self.lo, self.hi = np.quantile(v, [0.0005, 0.9995]); self.sd = v.std() + 1e-9
        elif self.kind == "dyn":
            st = [dyn_stats(d[self.target].values) for d in cal]
            self.med, self.mad = {}, {}
            for k in st[0]:
                v = np.concatenate([s[k][WARM:] for s in st])
                self.med[k] = np.median(v); self.mad[k] = np.median(np.abs(v - self.med[k])) + 1e-9
            self.med0, self.mad0 = self.med, self.mad
        # conformal calibration scores
        self.cal = {k: np.sort(np.concatenate([v[WARM::STEP] for v in vals]))
                    for k, vals in self._scores_multi(cal).items()}
        return self

    def _scores_multi(self, dfs):
        out = {}
        for d in dfs:
            for k, v in self.scores(d).items():
                out.setdefault(k, []).append(v)
        return out

    def scores(self, df):
        """Dict of full-length score series (one per sub-statistic)."""
        if self.kind in ("res", "pid", "mass"):
            y, X = design(df, self.kind, self.target, self.spec)
            ok = np.isfinite(y) & np.isfinite(X).all(1)
            r = np.zeros(len(y)); r[ok] = y[ok] - self.m.predict(X[ok])
            a = adapt(r, self.med0["r"], self.mad0["r"])
            return {"r": pd.Series(a).rolling(30, min_periods=1).mean().values}
        if self.kind == "env":
            x = df[self.target].values
            return {"env": np.maximum(0, np.maximum(x - self.hi, self.lo - x)) / self.sd}
        st = dyn_stats(df[self.target].values)
        return {k: adapt(v, self.med0[k], self.mad0[k]) for k, v in st.items()}

    def evalues(self, df, upd):
        es = []
        for k, v in self.scores(df).items():
            if getattr(self, "keep", None) is not None and k not in self.keep:
                continue
            C = self.cal[k]; n = len(C)
            ge = n - np.searchsorted(C, v[upd], side="left")
            p = (1 + ge) / (n + 1)
            es.append(KAPPA * p ** (KAPPA - 1))
        return es


ADMIT_MAX = 5.0


def admit(stream_spec, fit, cal):
    c = pd.concat(cal).reset_index(drop=True); h = len(c) // 2
    A, B = c.iloc[:h].reset_index(drop=True), c.iloc[h:].reset_index(drop=True)
    m = Stream(*stream_spec).fit(fit, [A]); upd = np.arange(WARM, len(B), STEP)
    keep = []
    for k, e in zip(m.scores(B.iloc[:300]).keys(), m.evalues(B, upd)):
        p = (KAPPA / e) ** (1 / (1 - KAPPA)) if KAPPA != 1 else e
        keep.append(k) if (p < 0.001).mean() / 0.001 <= ADMIT_MAX else None
    return keep


def shiryaev(e):
    q = 0.0; out = np.zeros(len(e)); alarms = []
    for i, ei in enumerate(e):
        pt = q + (1 - q) * RHO
        q = pt * ei / (pt * ei + 1 - pt)
        out[i] = q
        if q >= QALARM:
            alarms.append(i); q = 0.0
    return out, alarms


def episodes(y):
    e = np.flatnonzero(np.diff(np.r_[0, y, 0])); return list(zip(e[::2], e[1::2]))


# ---------------------------------------------------------------- run
ADMITLOG = []


def main():
    fit, cal, fp, tests, gt = load()
    models = {}
    for h, layers in LAYERS.items():
        for L, (cls, tags, streams) in layers.items():
            ms = []
            for sp in streams:
                m = Stream(*sp).fit(fit, cal); m.keep = admit(sp, fit, cal)
                if m.keep:
                    ms.append(m)
                ADMITLOG.append((h, L, sp[0], sp[1], m.keep))
            models[(h, L)] = ms
    fitv = pd.concat([d[[v for v, _ in MARGIN.values()]] for d in fit])
    margin = {h: (np.quantile(fitv[v], 0.001), np.quantile(fitv[v], 0.999)) for h, (v, _) in MARGIN.items()}

    def run_file(df):
        upd = np.arange(WARM, len(df), STEP)
        res = {}
        for key, ms in models.items():
            if not ms:
                continue
            es = [e for m in ms for e in m.evalues(df, upd)]
            q, al = shiryaev(np.mean(es, 0))
            res[key] = (q, upd[al])
        return upd, res

    # false alarms
    fa = {h: [] for h in LAYERS}; days = 0.0
    for name, df, mask in [("fp", d, np.ones(len(d), bool)) for d in fp] + \
                          [(f, d, None) for f, d, y in tests]:
        if mask is None:
            y = dict((f, yy) for f, _, yy in tests)[name]; mask = np.ones(len(y), bool)
            for s, e in episodes(y): mask[max(0, s - FAR):e + FAR] = False
        upd, res = run_file(df)
        days += mask[WARM:].sum() / 86400
        for h in LAYERS:
            t = sorted(int(a) for (hh, L), (q, al) in res.items() if hh == h for a in al if mask[a])
            ev = [a for i, a in enumerate(t) if i == 0 or a - t[i - 1] > MERGE]
            fa[h].append(len(ev))
    fa_tab = {h: dict(events=int(sum(v)), per30d=round(sum(v) * 30 / days, 2)) for h, v in fa.items()}

    # detection
    rows = []
    for f, df, y in tests:
        upd, res = run_file(df)
        g = gt[gt.file == f].reset_index(drop=True); E = episodes(y); assert len(E) == len(g), (f, len(E), len(g))
        for (s, e), (_, a) in zip(E, g.iterrows()):
            pts = str(a.points_eval).split(";")
            for h, layers in LAYERS.items():
                var, side = MARGIN[h]; lo, hi = margin[h]; x = df[var].values[s:e + GRACE]
                cross = np.flatnonzero((x > hi) | ((x < lo) if side == "both" else False))
                c = s + cross[0] if len(cross) else None
                for L, (cls, tags, streams) in layers.items():
                    if not set(pts) & set(tags):
                        continue
                    if cls == "unobservable":
                        rows.append(dict(atk=a.id, file=f, scen=h, layer=L, cls=cls, crossed=c is not None, detected=None, delay=None)); continue
                    al = res[(h, L)][1]; hit = al[(al >= s) & (al <= (c if c is not None else e + GRACE))]
                    anyal = [res[(hh, LL)][1] for (hh, LL) in res if hh == h]
                    scen_hit = any(((z >= s) & (z <= e + GRACE)).any() for z in anyal)
                    rows.append(dict(atk=a.id, file=f, scen=h, layer=L, cls=cls, crossed=c is not None,
                                     detected=bool(len(hit)), delay=(int(hit[0] - s) if len(hit) else None), scen_alarm=scen_hit))
    R = pd.DataFrame(rows)
    tag = f"{VER}_rho{sys.argv[2] if len(sys.argv) > 2 else '1'}"
    R.to_csv(f"{OUTD}/detect_{tag}.csv", index=False)
    obs = R[R.cls == "observable"].groupby("atk").detected.max()   # an attack counts once, detected if any attacked observable layer alarmed
    out = dict(admitted=ADMITLOG, version=VER, rho=RHO, normal_days=round(days, 2), false_alarms=fa_tab,
               observable_attacks=int(len(obs)), detected=int(obs.sum()), rate=round(float(obs.mean()), 3),
               by_layer=R.groupby(["scen", "layer", "cls"]).agg(n=("atk", "size"), det=("detected", "sum"),
                                                                    crossed=("crossed", "sum")).reset_index().to_dict("records"),
               median_delay_s=float(R.delay.median()) if R.delay.notna().any() else None)
    json.dump(out, open(f"{OUTD}/summary_{tag}.json", "w"), indent=1, default=str)
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
