"""Compare reproduced DAR outputs with the published ones in expected_results/.
Usage: python3 tools/compare_results.py <reproduced_dir> <expected_dir>   (exit code 1 on any difference)"""
import sys, os, json, glob, math
import pandas as pd
new, ref = sys.argv[1], sys.argv[2]
bad = 0


def close(a, b):
    if isinstance(a, float) or isinstance(b, float):
        try:
            return (math.isnan(float(a)) and math.isnan(float(b))) or math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12)
        except (TypeError, ValueError):
            return a == b
    return a == b


def walk(a, b, path=""):
    global bad
    if isinstance(a, dict) and isinstance(b, dict):
        for k in set(a) | set(b):
            if k not in a or k not in b:
                print("  missing key", path + "/" + str(k)); bad += 1
            else:
                walk(a[k], b[k], path + "/" + str(k))
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            print("  length differs", path); bad += 1
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, f"{path}[{i}]")
    elif not close(a, b):
        print(f"  {path}: {a} != {b}"); bad += 1


for f in sorted(glob.glob(os.path.join(ref, "*"))):
    name = os.path.basename(f); g = os.path.join(new, name)
    if not os.path.exists(g):
        print("MISSING", name); bad += 1; continue
    before = bad
    if name.endswith(".csv"):
        A, B = pd.read_csv(g), pd.read_csv(f)
        if A.shape != B.shape or list(A.columns) != list(B.columns):
            print("  shape/columns differ", name); bad += 1
        else:
            for c in A.columns:
                for x, y in zip(A[c].tolist(), B[c].tolist()):
                    if not (close(x, y) or (pd.isna(x) and pd.isna(y))):
                        print(f"  {name}:{c}: {x} != {y}"); bad += 1; break
    else:
        walk(json.load(open(g)), json.load(open(f)), name)
    print(("OK   " if bad == before else "DIFF ") + name)
print("ALL RESULTS REPRODUCED" if bad == 0 else f"{bad} DIFFERENCES")
sys.exit(1 if bad else 0)
