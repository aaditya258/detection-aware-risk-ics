"""Quick observability check with HAIEnd 23.05 (exploratory; not a confirmation).

Question: which boiler protection/initiating layers gain independent evidence when the DCS internal points recorded
in HAIEnd are added to the HAI 23.05 historian tags?
Channels per layer (all thresholds from normal data only: HAI/HAIEnd train1-3; false alarms on train4 + attack-free
test rows):
  cross   : HAI tag vs its DCS duplicate (30-s mean of |difference|)
  internal: DCS controller block output vs the command it issues
  actuator: DCS valve command vs measured valve position
  flags   : DCS built-in checks and alarm switches that are constant in normal operation
Detection: channel above its normal 99.9th percentile (flags: any change) in [attack start, end + 60 s].
Placebo: the same windows shifted +-1..3 h into attack-free time.
"""
import numpy as np, pandas as pd, json, os
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace("\\", "/")
DATA = os.environ.get("DAR_DATA", os.path.join(REPO, "data")).replace("\\", "/")
OUTD = os.environ.get("DAR_OUT", os.path.join(REPO, "outputs")).replace("\\", "/")
W = REPO
REN = {"x1001_15_ASSIGN_OUT": "P1_B2016", "x1002_07_SETPOINT_OUT": "P1_B3004"}
HCOLS = ["P1_PIT01", "P1_LIT01", "P1_PCV01D", "P1_LCV01D", "x1001_15_ASSIGN_OUT", "x1002_07_SETPOINT_OUT"]
ECOLS = ["DM-PIT01", "DM-LIT01", "DM-PCV01-D", "DM-LCV01-D", "DM-PCV01-Z", "DM-LCV01-Z", "1001.15-OUT", "1001.14-OUT",
         "1001.16-OUT", "1001.20-OUT", "1002.7-OUT", "1002.21-OUT", "1002.30-OUT", "1002.31-OUT",
         "DM-PCV01-DEV", "DM-PIT01-HH", "DM-LCV01-MIS", "DQ04-LCV01-DEV", "DQ03-LCV01-D", "DM-LSH-03", "DM-LSH01", "DM-LSL01"]
CH = {  # layer -> {channel: (kind, a, b)}
    ("H1", "IE_SP"): {"cross_sp": ("diff", "P1_B2016", "1001.15-OUT"), "internal_sp": ("diff", "1001.14-OUT", "1001.15-OUT")},
    ("H1", "IE_CO"): {"cross_co": ("diff", "P1_PCV01D", "DM-PCV01-D"), "internal_co": ("diff", "1001.16-OUT", "DM-PCV01-D"),
                      "actuator": ("diff", "DM-PCV01-D", "DM-PCV01-Z"), "flag_dev": ("flag", "DM-PCV01-DEV", None)},
    ("H1", "IPL1"): {"cross_pv": ("diff", "P1_PIT01", "DM-PIT01"), "flag_hh": ("flag", "DM-PIT01-HH", None)},
    ("H2", "IE_SP"): {"cross_sp": ("diff", "P1_B3004", "1002.7-OUT")},
    ("H2", "IE_CO"): {"cross_co": ("diff", "P1_LCV01D", "DM-LCV01-D"), "internal_co": ("diff", "1002.31-OUT", "DM-LCV01-D"),
                      "internal_lim": ("diff", "1002.21-OUT", "1002.30-OUT"), "actuator": ("diff", "DM-LCV01-D", "DM-LCV01-Z"),
                      "flag_mis": ("flag", "DM-LCV01-MIS", None), "flag_dev": ("flag", "DQ04-LCV01-DEV", None)},
    ("H2", "IPL1"): {"cross_pv": ("diff", "P1_LIT01", "DM-LIT01"), "flag_lsh03": ("flag", "DM-LSH-03", None),
                     "flag_lsh01": ("flag", "DM-LSH01", None), "flag_lsl01": ("flag", "DM-LSL01", None)}}
TAGS = {("H1", "IE_SP"): ["P1_B2016"], ("H1", "IE_CO"): ["P1_PCV01D"], ("H1", "IPL1"): ["P1_PIT01"],
        ("H2", "IE_SP"): ["P1_B3004"], ("H2", "IE_CO"): ["P1_LCV01D"], ("H2", "IPL1"): ["P1_LIT01"]}


def pair(hfile, efile):
    h = pd.read_csv(hfile, usecols=HCOLS).rename(columns=REN)
    e = pd.read_csv(efile, usecols=ECOLS)
    assert len(h) == len(e), (hfile, len(h), len(e))
    return pd.concat([h, e], axis=1)


def series(df, kind, a, b):
    if kind == "diff":
        return pd.Series(np.abs(df[a].values - df[b].values)).rolling(30, min_periods=1).mean().values
    return df[a].values.astype(float)


tr = [pair(f"{DATA}/hai-23.05/hai-train{i}.csv", f"{DATA}/haiend-23.05/end-train{i}.csv") for i in (1, 2, 3)]
ho = pair(f"{DATA}/hai-23.05/hai-train4.csv", f"{DATA}/haiend-23.05/end-train4.csv")
te = [(f, pair(f"{DATA}/hai-23.05/hai-{f}.csv", f"{DATA}/haiend-23.05/end-{f}.csv"), pd.read_csv(f"{DATA}/haiend-23.05/label-{f}.csv").label.values) for f in ("test1", "test2")]
gt = pd.read_csv(f"{REPO}/groundtruth/hai2305_attacks_DRAFT.csv")
AE_MAP = {"A222": ("H1", "IE_CO"), "A237": ("H1", "IE_CO"), "A220": ("H2", "IE_CO"), "A238": ("H2", "IE_CO")}


def eps(y):
    e = np.flatnonzero(np.diff(np.r_[0, y, 0])); return list(zip(e[::2], e[1::2]))


thr, normal = {}, {}
for lay, chs in CH.items():
    for c, (k, a, b) in chs.items():
        v = np.concatenate([series(d, k, a, b)[120:] for d in tr])
        if k == "diff":
            thr[(lay, c)] = ("gt", float(np.quantile(v, 0.999)))
        else:
            vals = np.unique(v); thr[(lay, c)] = ("ne", float(vals[0])) if len(vals) == 1 else ("never", None)


def alarm(df, lay, c):
    k, a, b = CH[lay][c]; v = series(df, k, a, b); t = thr[(lay, c)]
    if t[0] == "gt": return v > t[1]
    if t[0] == "ne": return v != t[1]
    return np.zeros(len(v), bool)


def events(mask):
    idx = np.flatnonzero(mask[1:] & ~mask[:-1]) + 1
    return [x for i, x in enumerate(idx) if i == 0 or x - idx[i - 1] > 600]


rows, fa = [], {}
days = (len(ho) - 120) / 86400
for lay, chs in CH.items():
    for c in chs:
        n = len(events(alarm(ho, lay, c)[120:]))
        for f, df, y in te:
            m = np.ones(len(y), bool)
            for s, e in eps(y): m[max(0, s - 900):e + 900] = False
            n += len(events(alarm(df, lay, c) & m))
        fa[(lay, c)] = n
days += sum(((lambda y: (lambda m: m)(np.ones(len(y), bool)))(y)).sum() for _, _, y in te) / 86400 * 0  # placeholder kept simple
normal_days = (len(ho) - 120) / 86400
for f, df, y in te:
    m = np.ones(len(y), bool)
    for s, e in eps(y): m[max(0, s - 900):e + 900] = False
    normal_days += m.sum() / 86400
A = {(f, lay, c): alarm(df, lay, c) for f, df, y in te for lay, chs in CH.items() for c in chs}
for f, df, y in te:
    g = gt[gt.file == f].reset_index(drop=True); E = eps(y)
    for (s, e), (_, a) in zip(E, g.iterrows()):
        pts = set(str(a.points_eval).split(";"))
        layers = [lay for lay, t in TAGS.items() if pts & set(t)]
        if a.id in AE_MAP: layers.append(AE_MAP[a.id])
        for lay in dict.fromkeys(layers):
            for shift in [0, -10800, -7200, -3600, 3600, 7200, 10800]:
                s2, e2 = s + shift, e + shift
                if shift and (s2 < 0 or e2 + 60 >= len(y) or y[max(0, s2 - 900):e2 + 900].any()): continue
                hits = {c: bool(A[(f, lay, c)][s2:e2 + 60].any()) for c in CH[lay]}
                rows.append(dict(atk=a.id, file=f, scen=lay[0], layer=lay[1], ae=a.id in AE_MAP, placebo=shift != 0, **{f"ch_{c}": v for c, v in hits.items()},
                                 any_new=any(hits.values())))
R = pd.DataFrame(rows); R.to_csv(f"{OUTD}/haiend_per_attack.csv", index=False)
v5 = pd.read_csv(f"{OUTD}/detect_v5_2305.csv")
out = []
for (h, L), chs in CH.items():
    r = R[(R.scen == h) & (R.layer == L)]; real, pl = r[~r.placebo], r[r.placebo]
    v = v5[(v5.scen == h) & (v5.layer == L)]
    out.append(dict(layer=f"{h} {L}", attacks=len(real), v5_historian_only=f"{int(v.detected.fillna(False).sum())}/{len(v)}",
                    with_haiend=f"{int(real.any_new.sum())}/{len(real)}", placebo_rate=round(pl.any_new.mean(), 3) if len(pl) else None,
                    false_alarm_events_per_day=round(sum(fa[((h, L), c)] for c in chs) / normal_days, 2),
                    channels_firing={c: int(real[f"ch_{c}"].sum()) for c in chs}))
S = pd.DataFrame(out); S.to_csv(f"{OUTD}/haiend_summary.csv", index=False)
print(f"normal days for false alarms: {normal_days:.2f}")
print(S.to_string(index=False))
print("\nThresholds:", {f"{k[0][0]} {k[0][1]} {k[1]}": v for k, v in thr.items()})
