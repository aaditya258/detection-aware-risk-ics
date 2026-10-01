"""Step 0b v5: documented revision of v4 (see step0b_v5_protocol.md).

Changes from v4: (a) controller-output level check (new "level" evidence group on IE_CO);
(b) TMEL = 20 x design MEF (v4: 10 x, knife-edge); (c) H1 IPL2 re-registered as a mechanical pressure-relief valve
(non-hackable, PFD 0.01, q = 0) because P1_PIT02 is not physically coupled to P1_PIT01 (R2 = 0.01);
(d) second, co-registered action rule R2 (demand-aware): act when R1 fires, or when an initiating-event layer
and a protection layer of the same scenario raise advisory alarms within 300 s of each other.
Earlier v4 notes:

(1) setpoint-transient-aware conformal reference, (4) corroboration across independent evidence groups,
(5) leaky (window-limited) evidence accumulation, plus the action tier (MEF vs TMEL).
Usage: python3 v4.py 2204|2305 [tau_seconds]
"""
import os as _os
REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
DATA = _os.environ.get("DAR_DATA", _os.path.join(REPO, "data")).replace("\\", "/")
OUTD = _os.environ.get("DAR_OUT", _os.path.join(REPO, "outputs")).replace("\\", "/")
REPO = REPO.replace("\\", "/")
import sys, json, numpy as np, pandas as pd
VER_ARG = sys.argv[1]; TAU = float(sys.argv[2]) if len(sys.argv) > 2 else 180.0
MODE = sys.argv[3] if len(sys.argv) > 3 else "1of"          # "1of": strongest group; "2of": second strongest (corroboration)
ADV_THR = float(sys.argv[4]) if len(sys.argv) > 4 else 10.0  # advisory alarm when leaky log-evidence D >= ADV_THR
sys.argv = ["monitor.py", VER_ARG]
src = open(__file__.replace("v5.py", "monitor5.py")).read().replace('if __name__ == "__main__":\n    main()', "")
exec(src)

LAMBDA = np.exp(-STEP / TAU)
LOGPRIOR = np.log(RHO / (1 - RHO))
SPTAG = {"H1": "P1_B2016", "H2": "P1_B3004", "H3": "P2_AutoSD"}
TR_LAG = 300

# layer -> class, attacked tags, [(group, stream spec)]
L4 = {
    "H1": {"IE_SP": ("maskable", ["P1_B2016"], [("env", ("env", "P1_B2016", None)), ("dyn", ("dyn", "P1_B2016", None))]),
           "IE_CO": ("observable", ["P1_PCV01D"], [("loop", ("pid", "P1_PCV01D", ("P1_B2016", "P1_PIT01"))),
                                                    ("level", ("res", "P1_PCV01D", ["P1_B2016", "P1_PIT01"])),
                                                    ("dyn", ("dyn", "P1_PCV01D", None))]),
           "IPL1": ("observable", ["P1_PIT01"], [("loop", ("res", "P1_PIT01", ["P1_B2016", "P1_PCV01D"])),
                                                 ("coupling", ("res", "P1_PIT01", ["P1_PIT02", "P1_FT01"])),
                                                 ("dyn", ("dyn", "P1_PIT01", None))]),
           "IPL2": ("nonhackable", [], [])},   # v5: mechanical pressure-relief valve
    "H2": {"IE_SP": ("maskable", ["P1_B3004"], [("env", ("env", "P1_B3004", None)), ("dyn", ("dyn", "P1_B3004", None))]),
           "IE_CO": ("observable", ["P1_LCV01D"], [("loop", ("pid", "P1_LCV01D", ("P1_B3004", "P1_LIT01"))),
                                                    ("level", ("res", "P1_LCV01D", ["P1_B3004", "P1_LIT01"])),
                                                    ("dyn", ("dyn", "P1_LCV01D", None))]),
           "IPL1": ("observable", ["P1_LIT01"], [("loop", ("res", "P1_LIT01", ["P1_B3004", "P1_LCV01D"])),
                                                 ("coupling", ("mass", "P1_LIT01", ["P1_FT01", "P1_FT02", "P1_FT03"])),
                                                 ("dyn", ("dyn", "P1_LIT01", None))])},
    "H3": {"IE_SP": ("maskable", ["P2_AutoSD", "P2_ManualSD"], [("env", ("env", "P2_AutoSD", None)), ("dyn", ("dyn", "P2_AutoSD", None))]),
           "IE_CO": ("observable", ["P2_SCO"], [("loop", ("pid", "P2_SCO", ("P2_AutoSD", "P2_SIT01"))),
                                                 ("level", ("res", "P2_SCO", ["P2_AutoSD", "P2_SIT01"])),
                                                 ("dyn", ("dyn", "P2_SCO", None))]),
           "IPL1": ("observable", ["P2_SIT01"], [("loop", ("res", "P2_SIT01", ["P2_AutoSD", "P2_SCO"])),
                                                 ("coupling", ("res", "P2_SIT01", ["P2_VT01"])),
                                                 ("dyn", ("dyn", "P2_SIT01", None))]),
           "IPL1_limit": ("unobservable", ["P2_RTR"], []),
           "IPL2": ("observable", ["P2_VT01"], [("coupling", ("res", "P2_VT01", ["P2_SIT01"])), ("dyn", ("dyn", "P2_VT01", None))]),
           "IPL2_limit": ("unobservable", ["P2_VTR01", "P2_VTR02", "P2_VTR03", "P2_VTR04"], [])},
}
# LOPA register (constructed): f0 BPCS loop failure 0.1/yr, cyber IE 1/yr (TR84), PFDs, TMEL = 10 x design MEF
REG = {"H1": dict(ie=["IE_SP", "IE_CO"], ipl={"IPL1": (0.1, ["IPL1"]), "IPL2": (0.01, ["IPL2"])}),
       "H2": dict(ie=["IE_SP", "IE_CO"], ipl={"IPL1": (0.1, ["IPL1"])}),
       "H3": dict(ie=["IE_SP", "IE_CO"], ipl={"IPL1": (0.01, ["IPL1", "IPL1_limit"]), "IPL2": (0.1, ["IPL2", "IPL2_limit"])})}
F0, FCYB = 0.1, 1.0
for h, r in REG.items():
    r["mef0"] = F0 * np.prod([p for p, _ in r["ipl"].values()]); r["tmel"] = 20 * r["mef0"]   # v5: 20 x


def sp_state(df, h, tau_sp):
    sp = df[SPTAG[h]]
    return ((sp - sp.shift(TR_LAG)).abs() > tau_sp).fillna(False).values


class S4(Stream):
    """Stream with a separate conformal reference for steady and transient setpoint states."""
    def fit4(self, fit, cal, h, tau_sp):
        self.fit(fit, cal); self.h, self.tau_sp = h, tau_sp
        self.cal2 = {}
        for d in cal:
            st = sp_state(d, h, tau_sp)
            for k, v in self.scores(d).items():
                idx = np.arange(WARM, len(d), STEP)
                for s in (False, True):
                    self.cal2.setdefault((k, s), []).append(v[idx][st[idx] == s])
        self.cal2 = {k: np.sort(np.concatenate(v)) for k, v in self.cal2.items()}
        return self

    def evalues4(self, df, upd):
        st = sp_state(df, self.h, self.tau_sp)[upd]
        out = []
        for k, v in self.scores(df).items():
            if self.keep is not None and k not in self.keep:
                continue
            x = v[upd]; p = np.ones(len(x))
            for s in (False, True):
                C = self.cal2.get((k, s))
                if C is None or len(C) < 50:
                    C = self.cal[k]
                m = st == s; n = len(C)
                p[m] = (1 + n - np.searchsorted(C, x[m], side="left")) / (n + 1)
            out.append(KAPPA * p ** (KAPPA - 1))
        return out


def build(fit, cal):
    fv = pd.concat(fit)
    tau_sp = {h: max(float(np.quantile((fv[t] - fv[t].shift(TR_LAG)).abs().dropna(), 0.9)), 1e-6) for h, t in SPTAG.items()}
    models, log = {}, []
    for h, Ls in L4.items():
        for L, (cls, tags, gs) in Ls.items():
            groups = {}
            for g, sp in gs:
                m = S4(*sp).fit4(fit, cal, h, tau_sp[h]); m.keep = admit(sp, fit, cal)
                log.append((h, L, g, sp[0], sp[1], m.keep))
                if m.keep:
                    groups.setdefault(g, []).append(m)
            models[(h, L)] = groups
    return models, tau_sp, log


def layer_paths(models, df):
    upd = np.arange(WARM, len(df), STEP)
    ev = {}
    for key, groups in models.items():
        ev[key] = {g: np.mean([e for m in ms for e in m.evalues4(df, upd)], 0) for g, ms in groups.items()}
    return upd, ev


def run_layer(egroups):
    """Leaky log-evidence per group; layer statistic = 2nd largest group (corroboration) or the only group."""
    G = list(egroups); n = len(next(iter(egroups.values())))
    D = {g: 0.0 for g in G}; stat = np.zeros(n); stat2 = np.zeros(n)
    for i in range(n):
        for g in G:
            D[g] = max(0.0, LAMBDA * D[g] + np.log(egroups[g][i]))
        v = sorted(D.values(), reverse=True)
        stat[i] = v[0]; stat2[i] = v[1] if len(v) >= 2 else np.nan
    up = np.flatnonzero((stat[1:] >= ADV_THR) & (stat[:-1] < ADV_THR)) + 1
    # action tier: posterior from the strongest group, only for layers with >= 2 independent evidence groups;
    # single-group layers cannot be cross-checked and fall back to the aging prior (None)
    q = None if len(G) < 2 else 1 / (1 + np.exp(-(LOGPRIOR + stat)))
    return q, list(up)


def main4():
    fit, cal, fp, tests, gt = load()
    models, tau_sp, log = build(fit, cal)
    fitv = pd.concat([d[[v for v, _ in MARGIN.values()]] for d in fit])
    margin = {h: (np.quantile(fitv[v], 0.001), np.quantile(fitv[v], 0.999)) for h, (v, _) in MARGIN.items()}

    def run(df):
        upd, ev = layer_paths(models, df)
        Q, A = {}, {}
        for key, eg in ev.items():
            if eg:
                Q[key], al = run_layer(eg); A[key] = upd[al]
        n = len(upd); age = 1 - (1 - RHO) ** np.arange(1, n + 1)        # audit at file start
        for h, Ls in L4.items():
            for L, (cls, _, gs) in Ls.items():
                if cls == "nonhackable":
                    Q[(h, L)] = np.zeros(n)  # v5: mechanical layer, cannot be compromised remotely
                elif (h, L) not in Q or Q[(h, L)] is None:
                    Q[(h, L)] = age          # unobservable, or only one evidence group: aging prior only
        act, act2 = {}, {}
        for h, r in REG.items():
            qie = 1 - np.prod([1 - Q[(h, L)] for L in r["ie"]], 0)
            f = (1 - qie) * F0 + qie * FCYB
            mef = f.copy()
            for p0, Ls in r["ipl"].values():
                qa = 1 - np.prod([1 - Q[(h, L)] for L in Ls], 0)
                mef = mef * (p0 + (1 - p0) * qa)
            over = mef > r["tmel"]
            act[h] = upd[np.flatnonzero(over[1:] & ~over[:-1]) + 1] if len(over) > 1 else np.array([], int)
            if len(over) and over[0]:
                act[h] = np.r_[upd[0], act[h]]
            # R2 (demand-aware): IE advisory alarm and IPL advisory alarm of the same scenario within 300 s
            ie_al = np.concatenate([A.get((h, L), np.array([], int)) for L in r["ie"]])
            ipl_al = np.concatenate([A.get((h, L), np.array([], int)) for _, Ls in r["ipl"].values() for L in Ls])
            co = sorted(max(a, b) for a in ie_al for b in ipl_al if abs(a - b) <= 300)
            act2[h] = np.array(sorted(set(act[h].tolist()) | set(co)), int)
        return upd, A, act, act2

    def events(t):
        t = sorted(int(x) for x in t); return [a for i, a in enumerate(t) if i == 0 or a - t[i - 1] > MERGE]

    fa_adv = {h: 0 for h in L4}; fa_act = {h: 0 for h in L4}; fa_act2 = {h: 0 for h in L4}; days = 0.0
    runs = {}
    for name, df, y in [("fp", d, None) for d in fp] + tests:
        mask = np.ones(len(df), bool)
        if y is not None:
            for s, e in episodes(y): mask[max(0, s - FAR):e + FAR] = False
        upd, A, act, act2 = run(df); runs[name] = (df, y, A, act, act2)
        days += mask[WARM:].sum() / 86400
        for h in L4:
            fa_adv[h] += len(events([a for (hh, L), al in A.items() if hh == h for a in al if mask[a]]))
            fa_act[h] += len(events([a for a in act[h] if mask[a]]))
            fa_act2[h] += len(events([a for a in act2[h] if mask[a]]))
    rows = []
    for f, df, y in tests:
        _, _, A, act, act2 = runs[f]
        g = gt[gt.file == f].reset_index(drop=True)
        for (s, e), (_, a) in zip(episodes(y), g.iterrows()):
            pts = set(str(a.points_eval).split(";"))
            for h, Ls in L4.items():
                var, side = MARGIN[h]; lo, hi = margin[h]; x = df[var].values[s:e + GRACE]
                cr = np.flatnonzero((x > hi) | ((x < lo) if side == "both" else False))
                c = s + cr[0] if len(cr) else None; end = c if c is not None else e + GRACE
                for L, (cls, tags, gs) in Ls.items():
                    if not pts & set(tags):
                        continue
                    al = A.get((h, L), np.array([], int)); hit = al[(al >= s) & (al <= end)]
                    ac = act[h]; ahit = ac[(ac >= s) & (ac <= end)]
                    ac2 = act2[h]; ahit2 = ac2[(ac2 >= s) & (ac2 <= end)]
                    ipl_attacked = any(pts & set(Ls[LL][1]) for LL in Ls if LL.startswith("IPL") and Ls[LL][0] == "observable")
                    rows.append(dict(atk=a.id, file=f, scen=h, layer=L, cls=cls, crossed=c is not None,
                                     detected=bool(len(hit)) if cls not in ("unobservable", "nonhackable") else None,
                                     delay=int(hit[0] - s) if len(hit) else None, action=bool(len(ahit)),
                                     action_r2=bool(len(ahit2)), unprotected_demand=bool(c is not None and ipl_attacked)))
    R = pd.DataFrame(rows)
    tag = f"v5_{VER_ARG}"
    R.to_csv(f"{OUTD}/detect_{tag}.csv", index=False)
    obs = R[R.cls == "observable"].groupby("atk").detected.max()
    anyact = R.groupby("atk").action.max()
    ud = R[R.unprotected_demand].groupby(["atk", "scen"])[["action", "action_r2"]].max()
    multi = R[~R.cls.isin(["unobservable", "nonhackable"])].groupby(["atk", "scen"]).layer.nunique()
    multi = multi[multi >= 2].reset_index()
    macts = [bool(R[(R.atk == a) & (R.scen == h)].action.max()) for a, h in zip(multi.atk, multi.scen)]
    out = dict(version=VER_ARG, tau=TAU, normal_days=round(days, 2), tau_sp=tau_sp, admitted=log,
               advisory_false_alarms_per30d={h: round(v * 30 / days, 1) for h, v in fa_adv.items()},
               action_false_alarms={h: dict(events=v, per30d=round(v * 30 / days, 2)) for h, v in fa_act.items()},
               observable_attacks=int(len(obs)), detected=int(obs.sum()), rate=round(float(obs.mean()), 3),
               action_on_attacks=f"{int(anyact.sum())}/{len(anyact)}",
               action_r2_false_alarms={h: dict(events=v, per30d=round(v * 30 / days, 2)) for h, v in fa_act2.items()},
               unprotected_demand_attacks=len(ud), ud_acted_r1=int(ud.action.sum()) if len(ud) else 0,
               ud_acted_r2=int(ud.action_r2.sum()) if len(ud) else 0,
               action_on_multilayer_attacks=f"{sum(macts)}/{len(macts)}", multilayer=list(zip(multi.atk, multi.scen, macts)),
               median_delay_s=float(R.delay.median()) if R.delay.notna().any() else None,
               by_layer=R.groupby(["scen", "layer", "cls"]).agg(n=("atk", "size"), det=("detected", "sum"),
                                                                    act=("action", "sum")).reset_index().to_dict("records"),
               register={h: dict(mef0=r["mef0"], tmel=r["tmel"]) for h, r in REG.items()})
    json.dump(out, open(f"{OUTD}/summary_{tag}.json", "w"), indent=1, default=str)
    return out, runs


if __name__ == "__main__":
    out, _ = main4()
    print(json.dumps({k: v for k, v in out.items() if k not in ("admitted", "by_layer")}, indent=1, default=str))
    for r in out["by_layer"]:
        print(r)
