"""Reproduce every result of the paper (works on Windows, Linux and macOS).

    python run_all.py                 # data in ./data
    python run_all.py D:\\data\\dar     # or give the data folder

Data folder layout (see download_hai.py and prepare_swat.py):
    hai-22.04/  hai-23.05/  hai-21.03/  haiend-23.05/  swat/ (swat_normal.parquet, swat_attack.parquet,
    List_of_attacks_Final.xlsx)
Steps: 1 monitor alerts, 2 risk model and metrics (also with the prior x0.1 and x10), 3 monitoring value,
4 exploratory asset-level analysis, 5 independent re-implementation, 6 comparison with the published results
and figures. Results are written to ./dar, figures to ./paper_figures/figures. About 15 minutes."""
import os, sys, subprocess

REPO = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(REPO, "data")
DAR = os.path.join(REPO, "dar")
env = dict(os.environ, DAR_DATA=DATA, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")


def run(script, *args, extra=None, log=None):
    e = dict(env, **(extra or {}))
    print(">", os.path.relpath(script, REPO), *args, flush=True)
    r = subprocess.run([sys.executable, script, *args], cwd=DAR, env=e, capture_output=bool(log), text=True)
    if log:
        open(os.path.join(DAR, log), "w", encoding="utf-8").write(r.stdout + r.stderr)
    if r.returncode != 0:
        if log:
            print(r.stdout[-2000:], r.stderr[-4000:])
        sys.exit(f"step failed: {script}")


need = ["hai-22.04", "hai-23.05", "hai-21.03", "haiend-23.05", os.path.join("swat", "swat_attack.parquet"),
        os.path.join("swat", "swat_normal.parquet"), os.path.join("swat", "List_of_attacks_Final.xlsx")]
missing = [n for n in need if not os.path.exists(os.path.join(DATA, n))]
if missing:
    sys.exit("Missing in data folder " + DATA + ": " + ", ".join(missing) +
             "\nRun download_hai.py for HAI/HAIEnd and prepare_swat.py for SWaT.")

print("== 1/6 monitor alerts ==")
for v in ["2204", "2305", "2103", "swat"]:
    if not os.path.exists(os.path.join(DAR, f"alarms_{v}.pkl")):
        run(os.path.join(DAR, "collect.py"), v)
print("== 2/6 risk model, baselines, metrics ==")
run(os.path.join(DAR, "dar.py"), log="dar.log")
run(os.path.join(DAR, "dar.py"), extra={"RHO_MULT": "0.1"}, log="dar_rho0.1.log")
run(os.path.join(DAR, "dar.py"), extra={"RHO_MULT": "10"}, log="dar_rho10.log")
print("== 3/6 monitoring value (HAIEnd) ==")
run(os.path.join(DAR, "rq3_monitoring_value.py"), log="rq3.log")
print("== 4/6 exploratory asset-level analysis ==")
run(os.path.join(DAR, "exploratory_asset_level.py"), log="exploratory.log")
print("== 5/6 independent re-implementation ==")
run(os.path.join(DAR, "verify_independent.py"))
print("== 6/6 comparison with published results, figures ==")
r = subprocess.run([sys.executable, os.path.join(REPO, "tools", "compare_results.py"), DAR, os.path.join(REPO, "expected_results")], env=env)
run(os.path.join(REPO, "paper_figures", "make_figures.py"), extra={"DAR_RESULTS": DAR})
run(os.path.join(REPO, "paper_figures", "make_diagrams.py"))
print("Done. Results in", DAR, "| figures in", os.path.join(REPO, "paper_figures", "figures"))
sys.exit(r.returncode)
