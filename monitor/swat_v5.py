"""SWaT (A1, Dec 2015) oxidant/RO scenario with the frozen v5 monitor (code reused unchanged; only data, register
and tag lists are SWaT-specific). Scenario and layer structure follow Sim, Lim and Rui (IntechOpen 2025,
doi 10.5772/intechopen.1011612); PFD values are standard LOPA credits (constructed), not theirs.

S1  excess oxidant reaches the RO membrane.
    IE    NaOCl over-dosing (P205), evidence from AIT203 response
    IPL1  UV dechlorinator UV401 (flow interlock on FIT401)          PFD 0.1
    IPL2  NaHSO3 dosing controlled by AIT402 (P403/P404 never run)   PFD 0.1
    IPL3  AIT502 ORP interlock -> RO shutdown (P501)                PFD 0.01
    IPL4  AIT504 high -> divert (MV501/MV503)                        PFD 0.1
    IPL5  FIT502 alarm                                               PFD 0.1
Margin: AIT502 high (99.9th percentile of fitting data). TMEL = 20 x design MEF (as v5).
Data roles: SWaT normal file minus the first 6 h: first 60% fit, next 20% calibration, last 20% false-alarm time,
plus attack-file rows more than 15 min from any labelled attack. Labels: 'A ttack' typo normalised; attack 21 window
taken at 06:30 (list says 18:30); attacks 37-41 dated 2016.
"""
import os as _os
REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
DATA = _os.environ.get("DAR_DATA", _os.path.join(REPO, "data")).replace("\\", "/")
OUTD = _os.environ.get("DAR_OUT", _os.path.join(REPO, "outputs")).replace("\\", "/")
REPO = REPO.replace("\\", "/")
import sys, json
sys.argv = ["v5.py", "2204", "180", "1of", "9"]
p = REPO + "/monitor/"
W0 = DATA + "/swat/"
src = open(p + "v5.py").read().replace('\nif __name__ == "__main__":', "\nif False:")
__file__ = p + "v5.py"
exec(src)
VER_ARG = "swat"

L4 = {"S1": {
    "IE": ("observable", ["P205", "P203", "P201"], [("coupling", ("res", "AIT203", ["P205", "AIT201", "FIT201"])),
                                                  ("dyn", ("dyn", "AIT203", None))]),
    "IPL1": ("observable", ["UV401", "FIT401"], [("loop", ("res", "UV401", ["FIT401", "P401"])),
                                                 ("coupling", ("res", "FIT401", ["FIT301", "LIT401", "P401"])),
                                                 ("dyn", ("dyn", "FIT401", None))]),
    "IPL2": ("observable", ["AIT402", "P403", "P404"], [("coupling", ("res", "AIT402", ["AIT401", "AIT203", "FIT401"])),
                                                        ("dyn", ("dyn", "AIT402", None))]),
    "IPL3": ("observable", ["AIT502", "P501"], [("coupling", ("res", "AIT502", ["AIT402", "AIT501", "AIT503"])),
                                                ("dyn", ("dyn", "AIT502", None))]),
    "IPL4": ("observable", ["AIT504", "MV501", "MV503"], [("coupling", ("res", "AIT504", ["AIT503", "AIT502", "FIT504"])),
                                                          ("dyn", ("dyn", "AIT504", None))]),
    "IPL5": ("observable", ["FIT502"], [("coupling", ("res", "FIT502", ["FIT501", "FIT503", "PIT501"])),
                                        ("dyn", ("dyn", "FIT502", None))])}}
REG = {"S1": dict(ie=["IE"], ipl={k: (v, [k]) for k, v in dict(IPL1=0.1, IPL2=0.1, IPL3=0.01, IPL4=0.1, IPL5=0.1).items()})}
for h, r in REG.items():
    r["mef0"] = F0 * np.prod([q for q, _ in r["ipl"].values()]); r["tmel"] = 20 * r["mef0"]
MARGIN = {"S1": ("AIT502", "hi")}
SPTAG = {"S1": "P205"}
def sp_state(df, h, tau_sp):          # SWaT has no operator setpoint stream for this scenario
    return np.zeros(len(df), bool)


def load():
    N = pd.read_parquet(W0 + "swat_normal.parquet"); A = pd.read_parquet(W0 + "swat_attack.parquet")
    N = N[N.Timestamp >= N.Timestamp.min() + pd.Timedelta(hours=6)].reset_index(drop=True)
    n = len(N); fitd = N.iloc[:int(.6 * n)].reset_index(drop=True)
    cald = N.iloc[int(.6 * n):int(.8 * n)].reset_index(drop=True); fpd = N.iloc[int(.8 * n):].reset_index(drop=True)
    y = (A["Normal/Attack"].str.replace(" ", "") == "Attack").values.astype(int)
    Lst = pd.read_excel(DATA + "/swat/List_of_attacks_Final.xlsx", engine="calamine")
    Lst["st"] = pd.to_datetime(Lst["Start Time"].astype(str).str.replace("2015-01-02", "2016-01-02"), errors="coerce")
    Lst.loc[Lst["Attack #"] == 21, "st"] = pd.Timestamp("2015-12-29 06:30:00")
    rows = []
    for s, e in episodes(y):
        ts = A.Timestamp.iloc[s]; dt = (Lst.st - ts).abs(); i = dt.idxmin()
        pts = str(Lst.loc[i, "Attack Point"]).replace("-", "").replace(" ", "").upper().replace(",", ";") if dt[i] < pd.Timedelta(minutes=15) else ""
        rows.append(dict(id=f"SWaT{int(Lst.loc[i, 'Attack #'])}" if pts else "unmatched", file="attack", points_eval=pts))
    gt = pd.DataFrame(rows); gt.to_csv(p + "swat_attacks_mapped.csv", index=False)
    return [fitd], [cald], [fpd], [("attack", A, y)], gt


out, runs = main4()
json.dump(out, open(p + "summary_swat_v5.json", "w"), indent=1, default=str)
for k in ["normal_days", "advisory_false_alarms_per30d", "action_false_alarms", "action_r2_false_alarms", "detected",
          "observable_attacks", "rate", "median_delay_s", "unprotected_demand_attacks", "ud_acted_r1", "ud_acted_r2"]:
    print(k, out[k])
for r in out["by_layer"]:
    print(r)

# live MEF against Sim et al.'s static values (4e-6 without attack, 4e-2 under cyber-PHA)
fit, cal, fp, tests, gt = load(); models, tau_sp, log = build(fit, cal)
df = tests[0][1]; y = tests[0][2]
upd, ev = layer_paths(models, df); Q = {}
for key, eg in ev.items():
    if eg:
        Q[key], _ = run_layer(eg)
n = len(upd); age = 1 - (1 - RHO) ** np.arange(1, n + 1)
for L in L4["S1"]:
    if Q.get(("S1", L)) is None:
        Q[("S1", L)] = age
r = REG["S1"]; qie = Q[("S1", "IE")]; mef = (1 - qie) * F0 + qie * FCYB
for q0, Ls in r["ipl"].values():
    mef = mef * (q0 + (1 - q0) * Q[("S1", Ls[0])])
mask = np.ones(len(df), bool)
for s, e in episodes(y): mask[max(0, s - FAR):e + FAR] = False
norm = mef[mask[upd]]
res = dict(design_mef=r["mef0"], tmel=r["tmel"], normal_median=float(np.median(norm)), normal_p99=float(np.quantile(norm, .99)),
           sim_static_no_attack=4e-6, sim_static_cyber=4e-2)
per = []
for (s, e), (_, a) in zip(episodes(y), gt.iterrows()):
    w = (upd >= s) & (upd <= e + GRACE)
    per.append(dict(attack=a.id, points=a.points_eval, max_mef=float(mef[w].max()), ratio_to_design=float(mef[w].max() / r["mef0"])))
res["attacks"] = per
json.dump(res, open(p + "swat_mef_vs_sim.json", "w"), indent=1)
print({k: v for k, v in res.items() if k != "attacks"})
for x in per:
    if x["attack"] != "unmatched":
        print(x)
