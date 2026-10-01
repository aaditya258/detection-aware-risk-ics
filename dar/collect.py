"""Step 1 of the pipeline: run the frozen process monitor (monitor/v5.py) on one dataset and save the onset
times of its advisory alerts per layer to dar/alarms_<dataset>.pkl (input to dar.py).
Usage:  python dar/collect.py 2204|2305|2103|swat
Monitor settings used in the paper: tau = 180 s, strongest-group mode ("1of"), advisory threshold D >= 9."""
import os, sys, pickle
VER = TAG = sys.argv[1]
B = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "monitor") + os.sep
OUT = os.path.dirname(os.path.abspath(__file__)) + os.sep
os.makedirs(os.environ.get("DAR_OUT", os.path.join(os.path.dirname(OUT.rstrip(os.sep)), "outputs")), exist_ok=True)
if VER == "swat":
    src = open(B + "swat_v5.py", encoding="utf-8").read().split("out, runs = main4()")[0]
    sys.argv = ["swat_v5.py"]; __file__ = B + "v5.py"
    exec(src)
else:
    sys.argv = ["v5.py", VER, "180", "1of", "9"]
    src = open(B + "v5.py", encoding="utf-8").read().replace('\nif __name__ == "__main__":', "\nif False:")
    __file__ = B + "v5.py"
    exec(src)
    if VER == "2305":                      # primary attack table (8 DCS-internal attacks excluded)
        _load = load
        def load():
            a = _load(); return a[:4] + (pd.read_csv(REPO + "/groundtruth/hai2305_attacks_PRIMARY.csv"),)
out, runs = main4()
fit, cal, fp, tests, gt = load()
store = {"gt": gt, "files": {}}
for name, (df, y, A, act, act2) in runs.items():
    store["files"][name] = dict(n=len(df), y=None if y is None else y.astype("int8"), A={k: v.astype(int) for k, v in A.items()})
pickle.dump(store, open(OUT + f"alarms_{TAG}.pkl", "wb"))
print(TAG, {k: v["n"] for k, v in store["files"].items()})
