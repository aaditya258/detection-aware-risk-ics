"""EXPLORATORY (not pre-registered; added 2026-09-30 while drafting the manuscript).
Q1: during attacks on poorly observed techniques (T0855-SP, T0836), do the OTHER layers of the same asset raise alerts?
Q2: false reassurance at the ASSET level: share of attack blocks in which the mean of pi/pi_bar over the asset's
    techniques is below 0.5, for P and B2.  Uses the frozen parameters in params.json.  Output: exploratory_asset_level.csv"""
import os, json, pandas as pd
D = os.path.dirname(os.path.abspath(__file__)) + "/"
__file__ = D + "dar.py"
exec(open(__file__).read().split('if __name__ == "__main__":')[0])
P = {tuple(k.split("|")): v for k, v in json.load(open(D + "params.json")).items()}
rows = []
for ver in ["2204", "2305", "2103"]:
    F, A = load(ver)
    for tech in ["T0855-SP", "T0836"]:
        n = other = single = 0; fr = {m: [0, 0] for m in ["P", "B2"]}
        for name, f in F.items():
            for ep in f["eps"]:
                if ep["tech"] != tech:
                    continue
                h = ep["asset"]; n += 1
                hit = any(detected(ep, f["alerts"][(h, t)][1]) for t in TECH[h] if t != tech)
                multi = any(e2["atk"] == ep["atk"] and e2["asset"] == h and e2["tech"] != tech for e2 in f["eps"])
                other += hit; single += hit and not multi
                b0, b1 = ep["s"] // BLK, min(f["nb"], (ep["e"] + GRACE) // BLK + 1)
                for m in fr:
                    tot = sum(run_models(f, h, t, P[(h, t)], RHO_OP)[m][b0:b1] for t in TECH[h]) / (len(TECH[h]) * PIBAR)
                    fr[m][0] += int((tot < 0.5).sum()); fr[m][1] += len(tot)
        rows.append(dict(set=ver, tech=tech, episodes=n, other_layer_alert=other, other_layer_alert_single_technique_attack=single,
                         asset_FR_P=round(fr["P"][0] / fr["P"][1], 2), asset_FR_B2=round(fr["B2"][0] / fr["B2"][1], 2)))
R = pd.DataFrame(rows); R.to_csv(D + "exploratory_asset_level.csv", index=False); print(R.to_string(index=False))
