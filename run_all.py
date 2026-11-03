"""
Reproduce the whole Ocazionally Intelligence Engine with ONE command.

    python run_all.py --fast        # about 10 minutes: everything except the 3 slowest demos
    python run_all.py               # about 20 minutes: everything
    python run_all.py --list        # show the steps and their numbers
    python run_all.py --steps 2,9   # run only chosen steps
    python run_all.py --from 9      # continue from step 9

Each step is one of your scripts, run exactly as you ran it by hand. The run stops at the first
failure and shows the end of that step's output. At the end it runs tests/check_everything.py.
MongoDB steps are skipped automatically if MongoDB is not running.
"""
import argparse
import glob
import os
import subprocess
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--fast", action="store_true", help="skip the three slowest steps")
ap.add_argument("--list", action="store_true")
ap.add_argument("--steps", default="", help="comma-separated step numbers")
ap.add_argument("--from", dest="start", type=int, default=1)
ap.add_argument("--keep-going", action="store_true", help="do not stop at the first failure")
args = ap.parse_args()

# (title, script, needs, slow)
STEPS = [
    ("Clean the vendor listings",        "ingestion/clean_listings.py",        "raw",   False),
    ("Generate the synthetic marketplace", "generator/generate_data.py",       None,    False),
    ("Spark: load and explore",          "pipeline/01_explore.py",             None,    False),
    ("Spark: MapReduce-style vs DataFrame benchmark", "pipeline/02_benchmark.py", None, True),
    ("MongoDB: load vendors and bookings", "storage/01_load_mongo.py",         "mongo", False),
    ("MongoDB: indexes before/after",    "storage/02_indexes_and_queries.py",  "mongo", False),
    ("MongoDB: replica set failover demo", "storage/03_replica_set_demo.py",   "mongo", True),
    ("Shard key analysis",               "storage/04_shard_key_analysis.py",   None,    False),
    ("Model: demand forecast",           "models/01_demand_forecast.py",       None,    False),
    ("Model: vendor segments",           "models/02_vendor_segments.py",       None,    False),
    ("Model: bundles and packages",      "models/03_bundles_and_packages.py",  None,    False),
    ("Fairness audit",                   "models/04_fairness_audit.py",        None,    False),
    ("Streaming demo",                   "pipeline/03_streaming_demo.py",      None,    True),
    ("Health check of everything",       "tests/check_everything.py",          None,    False),
]

if args.list:
    for i, (t, s, n, slow) in enumerate(STEPS, 1):
        print(f"{i:>3}. {t:<48} {s}{'   (slow)' if slow else ''}{'   (needs MongoDB)' if n == 'mongo' else ''}")
    sys.exit(0)

chosen = {int(x) for x in args.steps.split(",") if x.strip()}


def mongo_up():
    try:
        from pymongo import MongoClient
        MongoClient("mongodb://127.0.0.1:27017", serverSelectionTimeoutMS=1500).server_info()
        return True
    except Exception:
        return False


have_mongo = mongo_up()
print(f"MongoDB running: {'yes' if have_mongo else 'no (its steps will be skipped)'}\n")
log, failed = [], False
t_all = time.time()
for i, (title, script, needs, slow) in enumerate(STEPS, 1):
    if chosen and i not in chosen:
        continue
    if i < args.start:
        continue
    label = f"[{i:>2}/{len(STEPS)}] {title}"
    if args.fast and slow and not chosen:
        print(f"{label}\n        skipped (--fast)")
        log.append((title, "skipped", 0))
        continue
    if needs == "mongo" and not have_mongo:
        print(f"{label}\n        skipped (MongoDB is not running)")
        log.append((title, "skipped", 0))
        continue
    if needs == "raw" and not glob.glob(os.path.join("data", "private", "raw_listings", "*.csv")):
        print(f"{label}\n        skipped (no raw Apify exports in data/private/raw_listings)")
        log.append((title, "skipped", 0))
        continue
    print(f"{label} ...", flush=True)
    t = time.time()
    r = subprocess.run([sys.executable, script], capture_output=True, text=True, encoding="utf-8", errors="replace")
    secs = time.time() - t
    if r.returncode == 0:
        print(f"        done in {secs:.0f}s")
        log.append((title, "ok", secs))
        if script.endswith("check_everything.py"):
            print(r.stdout)
    else:
        print(f"        FAILED after {secs:.0f}s. End of its output:\n")
        print("\n".join((r.stdout + "\n" + r.stderr).strip().splitlines()[-15:]))
        log.append((title, "FAILED", secs))
        failed = True
        if not args.keep_going:
            break

print("\n" + "=" * 70)
print(f"{'Step':<52}{'Result':>10}{'Time':>8}")
print("-" * 70)
for title, status, secs in log:
    print(f"{title:<52}{status:>10}{secs:>7.0f}s")
print("=" * 70)
print(f"Total {(time.time() - t_all) / 60:.1f} minutes.  "
      f"{'Something failed: see above.' if failed else 'Everything ran.'}")
sys.exit(1 if failed else 0)