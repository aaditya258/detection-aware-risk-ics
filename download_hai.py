"""Download HAI 22.04, 23.05, 21.03 and HAIEnd 23.05 from the public repository github.com/icsdataset/hai
(Git LFS files served by media.githubusercontent.com). About 3.5 GB.
    python download_hai.py                # into ./data
    python download_hai.py D:\\data\\dar    # or another folder
Files that already exist are skipped."""
import os, sys, time, urllib.request

REPO = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(REPO, "data")
M = "https://media.githubusercontent.com/media/icsdataset/hai/master"
FILES = ([("hai-22.04", f"train{i}.csv") for i in range(1, 7)] + [("hai-22.04", f"test{i}.csv") for i in range(1, 5)] +
         [("hai-23.05", f"hai-train{i}.csv") for i in range(1, 5)] +
         [("hai-23.05", f) for f in ("hai-test1.csv", "hai-test2.csv", "label-test1.csv", "label-test2.csv")] +
         [("hai-21.03", f + ".csv.gz") for f in ("train1", "train2", "train3", "test1", "test2", "test3", "test4", "test5")] +
         [("haiend-23.05", f + ".csv") for f in ("end-train1", "end-train2", "end-train3", "end-train4", "end-test1",
                                                  "end-test2", "label-test1", "label-test2")])
for folder, name in FILES:
    dst = os.path.join(DATA, folder, name)
    if os.path.exists(dst) and os.path.getsize(dst) > 1024:
        continue
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    for attempt in range(3):
        try:
            print("downloading", folder + "/" + name, flush=True)
            urllib.request.urlretrieve(f"{M}/{folder}/{name}", dst + ".part")
            os.replace(dst + ".part", dst)
            break
        except Exception as e:
            print("  retry:", e); time.sleep(5)
    else:
        sys.exit("download failed: " + name)
    if os.path.getsize(dst) < 1024:
        print("WARNING: very small file (Git LFS pointer?):", dst)
print("HAI data ready in", DATA)
