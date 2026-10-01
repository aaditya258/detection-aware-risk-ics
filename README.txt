Detection-aware dynamic cyber risk for industrial control systems - code and results
====================================================================================
Code for the paper "Detection-aware dynamic cyber risk assessment for industrial control systems:
Preventing false reassurance from silent monitors" (Int. J. Critical Infrastructure Protection, submitted).

Requirements: Python 3.11 and the packages in requirements.txt
    pip install -r requirements.txt

Reproduce all results (Windows, Linux or macOS):
    1. python download_hai.py        HAI 22.04, 23.05, 21.03 and HAIEnd 23.05 into .\data (public, ~3.5 GB)
    2. SWaT is licensed by iTrust (SUTD). Put SWaT_Dataset_Normal_v1.xlsx, SWaT_Dataset_Attack_v0.xlsx and
       List_of_attacks_Final.xlsx into .\data\swat\ and run:  python prepare_swat.py
    3. python run_all.py             full pipeline, ~15 min, ends with "ALL RESULTS REPRODUCED"
A different data folder can be given as an argument to all three scripts.

Folders
    monitor\        process monitor that produces the alerts (frozen before the evaluation)
    dar\            risk model and evaluation: collect.py (runs the monitor), dar.py (detection rates,
                    hidden Markov filter, baselines, metrics), rq3_monitoring_value.py (residual risk, HAIEnd),
                    exploratory_asset_level.py, verify_independent.py (separate re-implementation)
    step0e\         DCS-internal channels from HAIEnd (used by rq3_monitoring_value.py)
    groundtruth\    attack tables with the attacked tags of every attack
    protocol\       pre-registered protocol and the SHA-256 fingerprints recorded before the evaluation
    paper_figures\  scripts for all figures
    expected_results\  published result files; tools\compare_results.py checks a re-run against them

Paper tables and figures
    Table 4  dar\detectability.csv          Table 5, Fig. 3   dar\metrics.csv
    Fig. 4   paper_figures\make_figures.py  Table 6, Fig. 5   dar\rq3_monitoring_value.csv, rq3_residual_risk.csv
    Prior sensitivity: metrics_rho0.1.csv, metrics_rho10.csv   Cost and calibration: summary*.json

Integrity: dar\dar.py and dar\rq3_monitoring_value.py are byte-identical to the versions fingerprinted in
protocol\frozen_dar.txt (protocol\protocol_dar.txt has the same content as the fingerprinted protocol file).
The monitor files differ from the run copies only in the lines that set file paths.

Data: HAI and HAIEnd https://github.com/icsdataset/hai . SWaT: iTrust, Singapore University of Technology and
Design (not redistributed here). Licence: see LICENSE.
