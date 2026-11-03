"""
End-to-end health check for the Ocazionally Intelligence Engine.

    python tests/check_everything.py          # about 1 minute
    python tests/check_everything.py --deep   # also proves the generator is reproducible (+1 min)

It checks, in this order:
  1. DATA       every table exists, ids are unique, tables link to each other, the rules of the
                business hold (no booking before a vendor joined, cancelled_by only on cancellations ...)
  2. SCHEMA     every field documented in docs/schema.md really exists in the data
  3. SPARK      Parquet files use timestamps Spark can read
  4. RESULTS    benchmarks and models produced sensible, internally consistent numbers
  5. PRIVACY    no phone numbers outside the git-ignored private folder, nothing private is in Git
  6. MONGODB    (optional) the database holds the same counts as the Parquet files
Status: PASS / FAIL / WARN (worth reading) / SKIP (not applicable on this machine).
Exit code is 1 if anything FAILED.
"""
import argparse
import glob
import os
import re
import subprocess
import sys
import tempfile

import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

ap = argparse.ArgumentParser()
ap.add_argument("--deep", action="store_true", help="also re-run the generator twice to prove it is reproducible")
args = ap.parse_args()

BASE = os.path.join("data", "synthetic")
results = []


def check(group, name):
    """Decorator: run a check, record PASS/FAIL/WARN/SKIP with a one-line detail."""
    def wrap(fn):
        try:
            out = fn()
            status, detail = out if isinstance(out, tuple) else ("PASS", out or "")
        except AssertionError as e:
            status, detail = "FAIL", str(e)
        except FileNotFoundError as e:
            status, detail = "FAIL", f"missing file: {e.filename or e}"
        except Exception as e:                                     # a broken check is a failure too
            status, detail = "FAIL", f"{type(e).__name__}: {e}"
        results.append((group, name, status, str(detail)))
        return fn
    return wrap


def need(cond, msg):
    assert cond, msg


def load(name):
    return pd.read_parquet(os.path.join(BASE, name))


def log_files():
    return sorted(glob.glob(os.path.join(BASE, "events_log", "*.parquet")))


# ======================================================================================
# 1. DATA
# ======================================================================================
vendors = bookings = reviews = customers = None


@check("DATA", "all five tables load")
def _():
    global vendors, bookings, reviews, customers
    vendors, bookings = load("vendors.parquet"), load("bookings.parquet")
    reviews, customers = load("reviews.parquet"), load("customers.parquet")
    need(len(log_files()) >= 12, f"events_log has only {len(log_files())} monthly files")
    n = sum(pq.ParquetFile(f).metadata.num_rows for f in log_files())
    need(len(bookings) > 100_000 and n > 1_000_000, f"data too small: {len(bookings):,} bookings, {n:,} log rows")
    return f"{len(vendors):,} vendors, {len(customers):,} customers, {len(bookings):,} bookings, " \
           f"{len(reviews):,} reviews, {n:,} events_log rows"


@check("DATA", "ids are unique")
def _():
    need(vendors["vendor_id"].is_unique, "duplicate vendor_id")
    need(bookings["booking_id"].is_unique, "duplicate booking_id")
    need(reviews["review_id"].is_unique, "duplicate review_id")
    need(customers["customer_id"].is_unique, "duplicate customer_id")
    t = pq.read_table(log_files(), columns=["log_id"]).column(0)
    need(pc.count_distinct(t).as_py() == len(t), "duplicate log_id in events_log")


@check("DATA", "tables link to each other")
def _():
    v = set(vendors["vendor_id"])
    need(bookings["vendor_id"].isin(v).all(), "bookings point to unknown vendors")
    need(bookings["customer_id"].isin(set(customers["customer_id"])).all(), "bookings point to unknown customers")
    need(reviews["booking_id"].isin(set(bookings["booking_id"])).all(), "reviews point to unknown bookings")
    ids = pc.unique(pq.read_table(log_files(), columns=["vendor_id"]).column(0)).to_pylist()
    bad = [i for i in ids if i and i not in v]
    need(not bad, f"events_log mentions {len(bad)} unknown vendors")


@check("DATA", "bookings obey the business rules")
def _():
    b = bookings
    need((pd.to_datetime(b["booking_date"]) <= pd.to_datetime(b["event_date"])).all(), "a booking is dated after its event")
    need((b["amount"] > 0).all(), "non-positive amount")
    need((b["guest_count"] > 0).all() and (b["response_time_minutes"] > 0).all(), "non-positive guests / response time")
    canc = b["status"] == "cancelled"
    need(((b["cancelled_by"] != "") == canc).all(), "cancelled_by is set on non-cancelled bookings (or missing on cancelled)")
    need(set(b["status"]) <= {"completed", "cancelled", "confirmed"}, f"unknown status {set(b['status'])}")
    need(set(b["cancelled_by"]) <= {"", "vendor", "customer"}, "unknown cancelled_by value")
    ev = b.groupby("event_instance_id").size()
    need((ev >= 1).all(), "an event without bookings")
    return f"cancellation rate {canc.mean():.1%}, {len(ev):,} events, {len(b) / len(ev):.1f} services per event"


@check("DATA", "no booking before the vendor joined, none for unverified vendors")
def _():
    m = bookings.merge(vendors[["vendor_id", "onboarded_date", "verification_status"]], on="vendor_id")
    early = (pd.to_datetime(m["booking_date"]) < pd.to_datetime(m["onboarded_date"])).sum()
    need(early == 0, f"{early} bookings are dated before the vendor was onboarded")
    need((m["verification_status"] == "verified").all(), "bookings exist for unverified vendors")


@check("DATA", "reviews and ratings are valid")
def _():
    need(reviews["rating"].between(1, 5).all(), "rating outside 1-5")
    need(vendors["rating_avg"].between(0, 5).all(), "rating_avg outside 0-5")
    m = reviews.merge(bookings[["booking_id", "status", "vendor_id"]], on="booking_id", suffixes=("", "_b"))
    need((m["status"] == "completed").all(), "a review on a cancelled booking")
    need((m["vendor_id"] == m["vendor_id_b"]).all(), "review vendor differs from booking vendor")


@check("DATA", "values are from the allowed lists")
def _():
    cats = {"decor", "catering", "sound_dj", "photography", "mehendi", "venue", "lighting", "other"}
    need(set(vendors["category"]) <= cats and set(bookings["category"]) <= cats, "unknown category")
    need(set(vendors["city"]) <= {"Ahmedabad", "Surat", "Vadodara", "Rajkot", "Gandhinagar"}, "unknown city")
    et = {"wedding", "navratri_society", "diwali", "birthday", "college_fest", "corporate", "uttarayan", "other"}
    need(set(bookings["event_type"]) <= et, "unknown event_type")
    need(set(customers["customer_type"]) <= {"individual", "housing_society", "college", "corporate"}, "unknown customer_type")
    acts = pc.unique(pq.read_table(log_files(), columns=["action"]).column(0)).to_pylist()
    need(len(acts) == 10, f"expected 10 platform actions, found {len(acts)}: {sorted(acts)}")
    roles = pc.unique(pq.read_table(log_files(), columns=["actor_role"]).column(0)).to_pylist()
    need(set(roles) == {"customer", "vendor", "admin"}, f"actor roles are {roles}")


@check("DATA", "demand has the seasonal shape it should")
def _():
    b = bookings.copy()
    b["month"] = pd.to_datetime(b["event_date"]).dt.month
    wed = b[b["event_type"] == "wedding"].groupby("month").size()
    need(wed[[11, 12, 1, 2]].mean() > 2 * wed[[7, 8, 9]].mean(), "weddings do not peak in Nov-Feb")
    nav = b[b["event_type"] == "navratri_society"].groupby("month").size()
    need(nav.idxmax() in (9, 10), f"Navratri peak is in month {nav.idxmax()}")


# ======================================================================================
# 2. SCHEMA
# ======================================================================================
@check("SCHEMA", "every field in docs/schema.md exists in the data")
def _():
    text = open(os.path.join("docs", "schema.md"), encoding="utf-8").read()
    docs, cur = {}, None
    for line in text.splitlines():
        m = re.match(r"## (\w+)", line)
        if m:
            cur = m.group(1)
            docs[cur] = []
        elif cur and line.startswith("|") and not line.startswith("| field") and not line.startswith("|---"):
            docs[cur].append(line.split("|")[1].strip())
    actual = {"vendors": vendors, "customers": customers, "bookings": bookings, "reviews": reviews,
              "events_log": pd.read_parquet(log_files()[0])}
    missing, undocumented = [], []
    for t, df in actual.items():
        missing += [f"{t}.{f}" for f in docs.get(t, []) if f not in df.columns]
        undocumented += [f"{t}.{c}" for c in df.columns if c not in docs.get(t, [])]
    need(not missing, f"documented but absent from the data: {missing}")
    if undocumented:
        return "WARN", f"in the data but not in schema.md: {undocumented}"


# ======================================================================================
# 3. SPARK COMPATIBILITY
# ======================================================================================
@check("SPARK", "no nanosecond timestamps (Spark cannot read them)")
def _():
    files = [os.path.join(BASE, f) for f in ("vendors.parquet", "bookings.parquet", "reviews.parquet", "customers.parquet")]
    files += log_files()[:3]
    bad = [os.path.basename(f) for f in files if any(str(t).startswith("timestamp[ns") for t in pq.read_schema(f).types)]
    need(not bad, f"nanosecond timestamps in {bad}: re-run generator/generate_data.py")


# ======================================================================================
# 4. RESULTS
# ======================================================================================
def csv(path, **kw):
    return pd.read_csv(path, **kw)


@check("RESULTS", "MapReduce-style vs DataFrame benchmark")
def _():
    d = csv(os.path.join("benchmarks", "results.csv"))
    need(len(d) == 2, f"expected 2 jobs, found {len(d)}")
    need(d["results_match"].astype(str).str.lower().eq("true").all(), "the two methods gave different answers")
    need((d["speedup"] > 1).all(), "DataFrame was not faster")
    return " ; ".join(f"{r.job}: {r.speedup:.1f}x on {int(r.rows):,} rows" for r in d.itertuples())


@check("RESULTS", "demand forecast beats the baselines")
def _():
    d = csv(os.path.join("benchmarks", "forecast_metrics.csv")).set_index("model")
    gbt = d.loc[[i for i in d.index if "Gradient" in i][0], "WMAPE"]
    base = d.loc[[i for i in d.index if "Naive" in i], "WMAPE"].min()
    need(gbt < base, f"GBT error {gbt}% is not below the best baseline {base}%")
    need(os.path.getsize(os.path.join("report", "figures", "forecast_actual_vs_predicted.png")) > 10_000, "forecast chart missing")
    return f"GBT {gbt:.1f}% vs best naive {base:.1f}%  ({(1 - gbt / base):.0%} better)"


@check("RESULTS", "vendor segments cover every vendor")
def _():
    s = csv(os.path.join("models", "outputs", "vendor_segments.csv"))
    need(s["vendor_id"].is_unique, "a vendor is in two segments")
    need(len(s) == len(vendors), f"{len(s)} segments for {len(vendors)} vendors")
    names = set(s["segment"])
    need(any("cancel" in n.lower() for n in names), f"no cancellation segment among {names}")
    return f"{s['segment'].nunique()} segments"


@check("RESULTS", "bundles and packages exist and look sensible")
def _():
    b = csv(os.path.join("benchmarks", "bundles_by_event_type.csv"))
    need(len(b) >= 8, "too few bundles")
    p = csv(os.path.join("models", "outputs", "sample_packages.csv"))
    need(len(p) >= 6 and (p["est_cost"] > 0).all(), "packages are empty or have zero cost")
    need(not p["vendor_id"].isin(set(vendors.loc[vendors["verification_status"] != "verified", "vendor_id"])).any(),
         "a package recommends an unverified vendor")


@check("RESULTS", "fairness audit finds and fixes the cold-start gap")
def _():
    a = csv(os.path.join("benchmarks", "fairness_audit.csv"))
    need((a["ratio_new_vs_established"] < 0.8).all(), "expected every quality band to be flagged (<0.8)")
    p = csv(os.path.join("benchmarks", "fairness_policies.csv"))
    need(p.iloc[1]["share_new"] > p.iloc[0]["share_new"] * 3, "the fair policy did not give newcomers more exposure")
    cost = (p.iloc[1]["true_quality"] - p.iloc[0]["true_quality"]) / p.iloc[0]["true_quality"]
    need(cost > -0.05, f"fairness cost too high ({cost:.1%})")
    return f"new-vendor ratio {a['ratio_new_vs_established'].min():.2f}-{a['ratio_new_vs_established'].max():.2f}; " \
           f"slots to newcomers {p.iloc[0]['share_new']:.1%} -> {p.iloc[1]['share_new']:.1%}; quality cost {cost:+.1%}"


@check("RESULTS", "shard key analysis recommends a balanced key")
def _():
    d = csv(os.path.join("benchmarks", "shard_key_analysis.csv"))
    need(len(d) == 5, "expected 5 candidate keys")
    ok = d[(d["imbalance"] <= 1.10) & (d["newest_writes_on_busiest_shard"] <= 0.45)]
    need(len(ok) >= 1, "no candidate key is both balanced and free of write hot spots")
    need(len(ok) < len(d), "every key passed: the analysis is not discriminating")
    best = ok.sort_values("score", ascending=False).iloc[0]
    return f"{len(d) - len(ok)} of {len(d)} keys rejected; best: {best['shard_key']} (imbalance {best['imbalance']:.2f}x)"


@check("RESULTS", "MongoDB indexes turned COLLSCAN into IXSCAN")
def _():
    path = os.path.join("benchmarks", "mongo_index_results.csv")
    if not os.path.exists(path):
        return "SKIP", "benchmarks/mongo_index_results.csv not found (run storage/02_indexes_and_queries.py)"
    d = csv(path)
    big = d[d["collection"] == "bookings"]
    need((big["plan_before"] == "COLLSCAN").all() and (big["plan_after"] == "IXSCAN").all(), "plans are not COLLSCAN -> IXSCAN")
    need((big["docs_read_after"] < big["docs_read_before"] / 100).all(), "indexes did not cut documents read by 100x")
    return f"bookings queries {big['speedup'].min():.0f}x-{big['speedup'].max():.0f}x faster"


@check("RESULTS", "replica set demo shows failover and a refused write")
def _():
    path = os.path.join("benchmarks", "replica_set_demo_log.txt")
    if not os.path.exists(path):
        return "SKIP", "log not found (run storage/03_replica_set_demo.py)"
    t = open(path, encoding="utf-8").read()
    need("took over as PRIMARY" in t and "write ACCEPTED" in t, "failover or majority write missing from the log")
    need("write REFUSED" in t, "the CAP demonstration (refused write) is missing")
    m = re.search(r"after ([\d.]+) seconds", t)
    return f"new primary elected in {m.group(1)} s" if m else "ok"


@check("RESULTS", "streaming demo ran")
def _():
    path = os.path.join("benchmarks", "streaming_results.txt")
    if not os.path.exists(path):
        return "SKIP", "log not found (run pipeline/03_streaming_demo.py)"
    t = open(path, encoding="utf-8").read()
    need("SUMMARY FOR YOUR REPORT" in t, "the streaming demo did not finish")
    m = re.search(r"Surge alerts raised:\s+(\d+)", t)
    return f"{m.group(1)} surge alerts" if m else "ok"


# ======================================================================================
# 5. PRIVACY
# ======================================================================================
@check("PRIVACY", "no phone numbers in the shareable outputs")
def _():
    phone = re.compile(r"(?<![\d.])(\+91[\s-]?)?[6-9]\d{4}[\s-]?\d{5}(?![\d.])")
    hits = []
    for f in glob.glob(os.path.join("models", "outputs", "*.csv")) + glob.glob(os.path.join("benchmarks", "*.csv")):
        if any(phone.search(str(x)) for x in pd.read_csv(f, dtype=str).fillna("").to_numpy().ravel()):
            hits.append(f)
    for c in vendors.columns:
        need("phone" not in c.lower() and "contact" not in c.lower(), f"vendors table has a contact column: {c}")
    need(not hits, f"phone-like numbers in {hits}")


@check("PRIVACY", "real business names are paired with simulated metrics")
def _():
    s = csv(os.path.join("models", "outputs", "vendor_segments.csv"))
    real = s[s["source"] == "listed"]
    if len(real):
        return "WARN", (f"{len(real)} real business names appear next to SIMULATED ratings, cancellation rates and "
                        f"segments in vendor_segments.csv. Fine for your own report; do NOT publish them on a website")


@check("PRIVACY", "nothing private or huge is tracked by Git")
def _():
    try:
        files = subprocess.run(["git", "ls-files"], capture_output=True, text=True, check=True).stdout.splitlines()
    except Exception:
        return "SKIP", "not a Git repository (or Git not installed)"
    banned = ("data/private/", "data/synthetic/", "data/stream", "storage/rs_data", "venv/", ".env")
    bad = [f for f in files if f.replace("\\", "/").startswith(banned)
           and not f.endswith(".gitkeep")]                 # empty placeholders are meant to be tracked
    need(not bad, f"tracked by Git but should not be: {bad[:5]}")
    return f"{len(files)} files tracked, none private"


# ======================================================================================
# 6. MONGODB (optional)
# ======================================================================================
@check("MONGODB", "database counts match the Parquet files")
def _():
    try:
        from pymongo import MongoClient
        c = MongoClient("mongodb://127.0.0.1:27017", serverSelectionTimeoutMS=1500)
        c.server_info()
    except Exception:
        return "SKIP", "MongoDB is not running on this machine"
    db = c["ocazionally"]
    nv, nb = db.vendors.count_documents({}), db.bookings.count_documents({})
    need(nv == len(vendors), f"MongoDB has {nv:,} vendors, Parquet has {len(vendors):,}: re-run storage/01_load_mongo.py")
    need(nb == len(bookings), f"MongoDB has {nb:,} bookings, Parquet has {len(bookings):,}: re-run storage/01_load_mongo.py")
    return f"{nv:,} vendors and {nb:,} bookings match"


# ======================================================================================
# 7. DEEP: reproducibility
# ======================================================================================
if args.deep:
    @check("DEEP", "the generator is reproducible (same seed -> identical data)")
    def _():
        outs = []
        for i in range(2):
            d = tempfile.mkdtemp()
            r = subprocess.run([sys.executable, os.path.join("generator", "generate_data.py"),
                                "--scale", "0.05", "--out", d], capture_output=True, text=True)
            need(r.returncode == 0, "generator failed: " + r.stderr[-200:])
            outs.append(pd.read_parquet(os.path.join(d, "bookings.parquet")))
        pd.testing.assert_frame_equal(outs[0], outs[1])
        return f"two runs gave identical {len(outs[0]):,} bookings"

# ======================================================================================
# report
# ======================================================================================
icons = {"PASS": "PASS", "FAIL": "FAIL", "WARN": "WARN", "SKIP": "skip"}
print()
last = None
for group, name, status, detail in results:
    if group != last:
        print(f"\n{group}")
        last = group
    print(f"  [{icons[status]}] {name}")
    if detail:
        print(f"         {detail}")
n = {s: sum(1 for r in results if r[2] == s) for s in icons}
print("\n" + "=" * 70)
print(f"  {n['PASS']} passed   {n['FAIL']} failed   {n['WARN']} warnings   {n['SKIP']} skipped")
print("=" * 70)
if n["FAIL"]:
    print("  Something needs fixing: read the FAIL lines above.")
elif n["WARN"]:
    print("  Everything works. Read the WARN lines before you publish anything.")
else:
    print("  Everything works.")
sys.exit(1 if n["FAIL"] else 0)