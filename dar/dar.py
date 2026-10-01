"""Detection-aware dynamic risk (DAR): detectability table on HAI 22.04, then evaluation on HAI 23.05, 21.03 and SWaT.
Follows protocol_dar.md (with amendment A1). Input: alarms_<ver>.pkl from collect.py (frozen v5 monitor).
Usage: python3 dar.py            (writes detectability.csv, params.json, metrics_*.csv, summary.json)
       RHO_MULT=0.1 python3 dar.py  (prior sensitivity; outputs suffixed)
"""
import os, json, pickle, numpy as np, pandas as pd
from scipy.stats import beta as Beta

D = os.path.dirname(os.path.abspath(__file__)) + "/"
RHO_MULT = float(os.environ.get("RHO_MULT", "1"))
SUF = "" if RHO_MULT == 1 else f"_rho{RHO_MULT:g}"
BLK, GRACE, FAR = 300, 60, 900
R_END = 0.5
RHO_OP = BLK / (365 * 86400) * RHO_MULT
PIBAR = RHO_OP / (RHO_OP + R_END)
IMPACT = {"H1": 4, "H2": 2, "H3": 4, "S1": 3}
TECH = {"H1": {"T0855-SP": ["P1_B2016"], "T0855-CO": ["P1_PCV01D"], "T0832": ["P1_PIT01"]},
        "H2": {"T0855-SP": ["P1_B3004"], "T0855-CO": ["P1_LCV01D"], "T0832": ["P1_LIT01"]},
        "H3": {"T0855-SP": ["P2_AutoSD", "P2_ManualSD"], "T0855-CO": ["P2_SCO"], "T0832": ["P2_SIT01"],
               "T0836": ["P2_RTR", "P2_VTR01", "P2_VTR02", "P2_VTR03", "P2_VTR04"]},
        "S1": {"T0832": ["FIT401", "AIT402", "AIT502", "AIT504", "FIT502", "LIT401"],
               "T0855": ["UV401", "P501", "P201", "P203", "P205", "MV501", "MV503"]}}
EVID = {"T0855-SP": ["IE_SP"], "T0855-CO": ["IE_CO"], "T0832": ["IPL1"], "T0836": []}
EVID_S1 = ["IE", "IPL1", "IPL2", "IPL3", "IPL4", "IPL5"]


def episodes(y):
    e = np.flatnonzero(np.diff(np.r_[0, y, 0])); return list(zip(e[::2], e[1::2]))


def load(ver):
    st = pickle.load(open(D + f"alarms_{ver}.pkl", "rb")); gt = st["gt"]; F = {}
    assets = ["S1"] if ver == "swat" else ["H1", "H2", "H3"]
    for name, f in st["files"].items():
        n = f["n"]; nb = n // BLK; y = f["y"]
        normal = np.ones(nb, bool); eps = []
        if y is not None:
            g = gt[gt.file == name].reset_index(drop=True); E = episodes(y); assert len(E) == len(g), (ver, name)
            for (s, e), (_, a) in zip(E, g.iterrows()):
                normal[max(0, (s - FAR) // BLK):min(nb, (e + FAR) // BLK + 1)] = False
                pts = set(str(a.points_eval).split(";"))
                for h in assets:
                    for t, tags in TECH[h].items():
                        if pts & set(tags):
                            eps.append(dict(atk=a.id, asset=h, tech=t, s=int(s), e=int(e)))
        alerts = {}
        for h in assets:
            for t in TECH[h]:
                layers = EVID_S1 if h == "S1" else EVID[t]
                on = np.concatenate([f["A"].get((h, L), np.array([], int)) for L in layers]) if layers else np.array([], int)
                a = np.zeros(nb, bool); b = on // BLK; a[b[b < nb]] = True
                alerts[(h, t)] = (a, on)
        ytrue = {}
        for h in assets:
            for t in TECH[h]:
                yy = np.zeros(nb, bool)
                for ep in eps:
                    if ep["asset"] == h and ep["tech"] == t:
                        yy[ep["s"] // BLK:min(nb, (ep["e"] + GRACE) // BLK + 1)] = True
                ytrue[(h, t)] = yy
        F[name] = dict(nb=nb, normal=normal, eps=eps, alerts=alerts, y=ytrue)
    return F, assets


def detected(ep, on):
    return bool(((on >= ep["s"]) & (on <= ep["e"] + GRACE)).any())


def estimate(F, assets, exclude=None):
    P = {}
    for h in assets:
        for t in TECH[h]:
            k = n = 0; delays = []; nor = alarm = 0
            for name, f in F.items():
                a, on = f["alerts"][(h, t)]
                nor += f["normal"].sum(); alarm += (a & f["normal"]).sum()
                for ep in f["eps"]:
                    if ep["asset"] == h and ep["tech"] == t and (exclude is None or ep["atk"] != exclude):
                        n += 1; hit = detected(ep, on); k += hit
                        if hit:
                            delays.append(int(on[(on >= ep["s"]) & (on <= ep["e"] + GRACE)][0] - ep["s"]))
            P[(h, t)] = dict(k=k, n=n, d_hat=(1 + k) / (2 + n), d_L=float(Beta.ppf(0.05, 1 + k, 1 + n - k)),
                             alpha=max(alarm / max(nor, 1), 1e-4), delay_med=float(np.median(delays)) if delays else None,
                             normal_blocks=int(nor))
    return P


def filt(alert, d_alert, d_sil, alpha, rho):
    pi = rho / (rho + R_END); out = np.empty(len(alert))
    for i, al in enumerate(alert):
        pr = pi * (1 - R_END) + (1 - pi) * rho
        l1, l0 = (d_alert, alpha) if al else (1 - d_sil, 1 - alpha)
        pi = pr * l1 / (pr * l1 + (1 - pr) * l0); out[i] = pi
    return out


def run_models(f, h, t, p, rho):
    a = f["alerts"][(h, t)][0]
    return {"P": filt(a, p["d_hat"], p["d_L"], p["alpha"], rho),
            "B1": np.full(len(a), rho / (rho + R_END)),
            "B2": filt(a, 0.9, 0.9, p["alpha"], rho),
            "B4": filt(a, p["d_hat"], p["d_hat"], p["alpha"], rho)}


def evaluate(F, assets, P, rho_tb, per_episode_params=None):
    rows, fe, rank, brier = [], {}, {}, []
    days = sum(f["normal"].sum() for f in F.values()) * BLK / 86400
    for h in assets:
        risk_quiet = {m: [] for m in ["P", "B1", "B2", "B4"]}; elev_on = {m: 0 for m in ["P", "B1", "B2", "B4"]}
        for name, f in F.items():
            asset_elev = {m: np.zeros(f["nb"], bool) for m in risk_quiet}; asset_risk = {m: np.zeros(f["nb"]) for m in risk_quiet}
            for t in TECH[h]:
                p = P[(h, t)]; M = run_models(f, h, t, p, RHO_OP)
                Mtb = run_models(f, h, t, p, rho_tb[(h, t)])
                y = f["y"][(h, t)]
                for m in M:
                    asset_elev[m] |= M[m] > 10 * PIBAR; asset_risk[m] += IMPACT[h] * M[m] / PIBAR
                    brier.append(dict(asset=h, tech=t, model=m, file=name, sse=float(((Mtb[m] - y) ** 2).sum()), nb=int(len(y))))
                for ep in f["eps"]:
                    if ep["asset"] != h or ep["tech"] != t:
                        continue
                    pe = per_episode_params(ep, h, t) if per_episode_params else p
                    Me = run_models(f, h, t, pe, RHO_OP) if per_episode_params else M
                    b0, b1 = ep["s"] // BLK, min(f["nb"], (ep["e"] + GRACE) // BLK + 1)
                    for m, v in Me.items():
                        seg = v[b0:b1]; up = np.flatnonzero(seg > 10 * PIBAR)
                        rows.append(dict(asset=h, tech=t, atk=ep["atk"], file=name, model=m, blocks=len(seg),
                                         fr_blocks=int((seg < 0.5 * PIBAR).sum()), elevated=bool(len(up)),
                                         delay_blocks=int(up[0]) if len(up) else None,
                                         detected=detected(ep, f["alerts"][(h, t)][1])))
            for m in risk_quiet:
                ev = asset_elev[m] & f["normal"]; elev_on[m] += int((ev[1:] & ~ev[:-1]).sum() + (ev[0] if len(ev) else 0))
                risk_quiet[m].extend(asset_risk[m][f["normal"]].tolist())
        fe[h] = {m: round(v / days, 2) for m, v in elev_on.items()}
        rank[h] = {m: float(np.mean(v)) for m, v in risk_quiet.items()}
    R = pd.DataFrame(rows); Bs = pd.DataFrame(brier)
    return R, fe, rank, Bs, days


def summarise(R, P, label):
    g = R.groupby(["asset", "tech", "model"]).agg(attacks=("atk", "nunique"), blocks=("blocks", "sum"), fr=("fr_blocks", "sum"),
                                                  elevated=("elevated", "sum"), detected=("detected", "sum")).reset_index()
    g["FR"] = (g.fr / g.blocks).round(3); g["EL"] = (g.elevated / g.attacks).round(3)
    g["d_L_dev"] = [round(P[(a, t)]["d_L"], 3) for a, t in zip(g.asset, g.tech)]
    g["miss_obs"] = (1 - g.detected / g.attacks).round(3); g["miss_pred"] = [round(1 - P[(a, t)]["d_hat"], 3) for a, t in zip(g.asset, g.tech)]
    g.insert(0, "set", label)
    return g


if __name__ == "__main__":
    # ---- detectability on HAI 22.04 (development)
    F04, A_h = load("2204"); P = estimate(F04, A_h)
    tb_blocks = sum(f["nb"] for n_, f in F04.items() if n_ != "fp")
    rho_tb = {k: max(v["n"], 0.5) / tb_blocks for k, v in P.items()}
    det = pd.DataFrame([dict(asset=h, tech=t, source="historian (v5 monitor)", attacks_22_04=v["n"], detected=v["k"], d_hat=round(v["d_hat"], 3),
                             d_L=round(v["d_L"], 3), alpha_per_block=round(v["alpha"], 4),
                             false_alerts_per_day=round(v["alpha"] * 86400 / BLK, 2), median_delay_s=v["delay_med"]) for (h, t), v in P.items()])
    det.to_csv(D + f"detectability{SUF}.csv", index=False)
    json.dump({f"{h}|{t}": v for (h, t), v in P.items()}, open(D + f"params{SUF}.json", "w"), indent=1)
    out, allsum = {}, []
    for ver in ["2204", "2305", "2103"]:
        F, A = load(ver) if ver != "2204" else (F04, A_h)
        R, fe, rank, Bs, days = evaluate(F, A, P, rho_tb)
        S = summarise(R, P, ver); allsum.append(S)
        bs = Bs.groupby(["asset", "tech", "model"]).apply(lambda d: d.sse.sum() / d.nb.sum()).rename("brier").reset_index()
        out[ver] = dict(normal_days=round(days, 2), false_elevations_per_day=fe, mean_quiet_risk_rel=rank,
                        brier=bs.to_dict("records"))
        R.to_csv(D + f"episodes_{ver}{SUF}.csv", index=False)
    # ---- SWaT: leave-one-attack-out detectability
    Fs, As = load("swat"); Ps = estimate(Fs, As)
    sw_blocks = sum(f["nb"] for n_, f in Fs.items() if n_ != "fp")
    rho_sw = {k: max(v["n"], 0.5) / sw_blocks for k, v in Ps.items()}
    loo = lambda ep, h, t: estimate(Fs, As, exclude=ep["atk"])[(h, t)]
    R, fe, rank, Bs, days = evaluate(Fs, As, Ps, rho_sw, per_episode_params=loo)
    S = summarise(R, Ps, "swat (LOO)"); allsum.append(S)
    bs = Bs.groupby(["asset", "tech", "model"]).apply(lambda d: d.sse.sum() / d.nb.sum()).rename("brier").reset_index()
    out["swat"] = dict(normal_days=round(days, 2), false_elevations_per_day=fe, mean_quiet_risk_rel=rank, brier=bs.to_dict("records"),
                       detectability={f"{h}|{t}": v for (h, t), v in Ps.items()})
    R.to_csv(D + f"episodes_swat{SUF}.csv", index=False)
    ALL = pd.concat(allsum); ALL.to_csv(D + f"metrics{SUF}.csv", index=False)
    json.dump(out, open(D + f"summary{SUF}.json", "w"), indent=1, default=str)
    pd.set_option("display.width", 250)
    print("pi_bar", PIBAR); print(det.to_string(index=False))
    print(ALL[ALL.model.isin(["P", "B2", "B4"])].pivot_table(index=["set", "asset", "tech"], columns="model", values=["FR", "EL"]).round(2).to_string())
    print(ALL[ALL.model == "P"][["set", "asset", "tech", "attacks", "d_L_dev", "miss_obs", "miss_pred"]].to_string(index=False))
    for v, o in out.items():
        print(v, "false elevations/day", o["false_elevations_per_day"]); print(v, "quiet risk (x prior)", {h: {m: round(x, 2) for m, x in r.items()} for h, r in o["mean_quiet_risk_rel"].items()})
