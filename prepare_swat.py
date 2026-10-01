"""Convert the SWaT spreadsheets to parquet. SWaT is licensed by iTrust (Singapore University of Technology and
Design) and cannot be redistributed: request it from iTrust, then put SWaT_Dataset_Normal_v1.xlsx,
SWaT_Dataset_Attack_v0.xlsx and List_of_attacks_Final.xlsx into <data>/swat/ and run
    python prepare_swat.py                # data in ./data
    python prepare_swat.py D:\\data\\dar
Writes <data>/swat/swat_normal.parquet and swat_attack.parquet (a few minutes)."""
import os, sys, time
import pandas as pd

REPO = os.path.dirname(os.path.abspath(__file__))
SW = os.path.join(os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(REPO, "data"), "swat")


def convert(fname, out, header_row=1):
    t = time.time()
    df = pd.read_excel(os.path.join(SW, fname), engine="calamine", header=header_row)
    df.columns = [str(c).strip() for c in df.columns]
    df["Timestamp"] = pd.to_datetime(df["Timestamp"].astype(str).str.strip(), format="%d/%m/%Y %I:%M:%S %p")
    df["Normal/Attack"] = df["Normal/Attack"].astype(str).str.strip()
    df.to_parquet(os.path.join(SW, out))
    print(out, df.shape, f"{time.time() - t:.0f}s")


convert("SWaT_Dataset_Normal_v1.xlsx", "swat_normal.parquet")
convert("SWaT_Dataset_Attack_v0.xlsx", "swat_attack.parquet")
