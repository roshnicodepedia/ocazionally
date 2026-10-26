"""
Unit 3 (NoSQL), step 2: indexes - the same queries before and after.

Run from the project root (venv active, MongoDB running, data loaded):
    python storage/02_indexes_and_queries.py

For each query it records:
  - the plan MongoDB used (COLLSCAN = read every document, IXSCAN = used an index)
  - documents examined vs documents returned
  - time
Then it creates a matching compound index and measures again.
Results are saved to benchmarks/mongo_index_results.csv for the report.
"""
import argparse
import csv
import os
import statistics
import time
from datetime import datetime

from pymongo import MongoClient

ap = argparse.ArgumentParser()
ap.add_argument("--uri", default="mongodb://localhost:27017")
ap.add_argument("--db", default="ocazionally")
args = ap.parse_args()

client = MongoClient(args.uri, serverSelectionTimeoutMS=4000)
client.server_info()
db = client[args.db]

top_vendor = db.vendors.find_one(sort=[("stats.bookings_total", -1)])["_id"]

QUERIES = [
    dict(
        name="Q1  Top verified decor vendors in Ahmedabad by rating",
        coll="vendors",
        filter={"location.city": "Ahmedabad", "category": "decor", "verification_status": "verified"},
        sort={"rating_avg": -1}, limit=10,
        index=[("location.city", 1), ("category", 1), ("verification_status", 1), ("rating_avg", -1)],
    ),
    dict(
        name="Q2  One vendor's bookings in a 3-month window",
        coll="bookings",
        filter={"vendor_id": top_vendor,
                "event_date": {"$gte": datetime(2025, 9, 1), "$lt": datetime(2025, 12, 1)}},
        sort=None, limit=0,
        index=[("vendor_id", 1), ("event_date", 1)],
    ),
    dict(
        name="Q3  Surat photography bookings in Jan 2026",
        coll="bookings",
        filter={"city": "Surat", "category": "photography",
                "event_date": {"$gte": datetime(2026, 1, 1), "$lt": datetime(2026, 2, 1)}},
        sort=None, limit=0,
        index=[("city", 1), ("category", 1), ("event_date", 1)],
    ),
]


def stages(node, found=None):
    """Collect every plan stage name (COLLSCAN, IXSCAN, FETCH, ...) from an explain tree."""
    found = [] if found is None else found
    if isinstance(node, dict):
        if isinstance(node.get("stage"), str):
            found.append(node["stage"])
        for v in node.values():
            stages(v, found)
    elif isinstance(node, list):
        for v in node:
            stages(v, found)
    return found


def explain(q):
    cmd = {"find": q["coll"], "filter": q["filter"]}
    if q["sort"]:
        cmd["sort"] = q["sort"]
    if q["limit"]:
        cmd["limit"] = q["limit"]
    try:
        res = db.command({"explain": cmd, "verbosity": "executionStats"})
        es = res["executionStats"]
        st = stages(res["queryPlanner"]["winningPlan"])
        plan = "IXSCAN" if "IXSCAN" in st else ("COLLSCAN" if "COLLSCAN" in st else "/".join(st[:2]))
        return plan, es["totalDocsExamined"], es["nReturned"]
    except Exception as e:                                   # explain not available: still time it
        return f"(explain unavailable: {type(e).__name__})", None, None


def run_time(q, repeats=5):
    times = []
    for _ in range(repeats):
        cur = db[q["coll"]].find(q["filter"])
        if q["sort"]:
            cur = cur.sort(list(q["sort"].items()))
        if q["limit"]:
            cur = cur.limit(q["limit"])
        t = time.perf_counter()
        n = len(list(cur))
        times.append((time.perf_counter() - t) * 1000)
    return statistics.median(times), n


def measure(q):
    plan, examined, returned = explain(q)
    ms, n = run_time(q)
    return dict(plan=plan, examined=examined, returned=returned if returned is not None else n, ms=ms)


# ---- before: remove everything except the default _id index -----------------------------
for name in ("vendors", "bookings"):
    db[name].drop_indexes()
print("Indexes removed (only the default _id index remains).\n")
before = [measure(q) for q in QUERIES]

# ---- after: create one compound index per query ------------------------------------------
print("Creating compound indexes ...")
for q in QUERIES:
    t = time.time()
    name = db[q["coll"]].create_index(q["index"])
    print(f"  {q['coll']:9s} {name}   ({time.time() - t:.1f}s)")
print()
after = [measure(q) for q in QUERIES]

# ---- report ----------------------------------------------------------------------------------
print("=" * 100)
print(f"{'Query':<56}{'Plan':>10}{'Docs read':>12}{'Returned':>10}{'Time':>9}")
print("=" * 100)
rows = []
for q, b, a in zip(QUERIES, before, after):
    for tag, m in (("before", b), ("after ", a)):
        ex = f"{m['examined']:,}" if m["examined"] is not None else "n/a"
        print(f"{q['name'] if tag == 'before' else '':<56}{m['plan']:>10}{ex:>12}"
              f"{m['returned']:>10,}{m['ms']:>7.1f}ms")
    speed = b["ms"] / a["ms"] if a["ms"] else float("inf")
    print(f"{'':<56}{'-> index made it':>32} {speed:.0f}x faster\n")
    rows.append([q["name"].strip(), q["coll"], b["plan"], b["examined"], round(b["ms"], 2),
                 a["plan"], a["examined"], round(a["ms"], 2), round(speed, 1)])
print("=" * 100)

os.makedirs("benchmarks", exist_ok=True)
with open(os.path.join("benchmarks", "mongo_index_results.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["query", "collection", "plan_before", "docs_read_before", "ms_before",
                "plan_after", "docs_read_after", "ms_after", "speedup"])
    w.writerows(rows)
print("Saved benchmarks/mongo_index_results.csv")

# ---- an aggregation pipeline (MongoDB's map-reduce-style analytics) --------------------------
print("\nAggregation: top 5 vendors by revenue in Ahmedabad (completed bookings)")
pipeline = [
    {"$match": {"city": "Ahmedabad", "status": "completed"}},
    {"$group": {"_id": "$vendor_id", "revenue": {"$sum": "$amount"}, "bookings": {"$sum": 1}}},
    {"$sort": {"revenue": -1}},
    {"$limit": 5},
    {"$lookup": {"from": "vendors", "localField": "_id", "foreignField": "_id", "as": "v"}},
    {"$unwind": "$v"},
    {"$project": {"_id": 0, "vendor": "$v.name", "category": "$v.category",
                  "source": "$v.source", "bookings": 1, "revenue": 1}},
]
t = time.perf_counter()
out = list(db.bookings.aggregate(pipeline))
print(f"  ({(time.perf_counter() - t) * 1000:.0f} ms)")
for r in out:
    print(f"  {r['vendor'][:34]:<36}{r['category']:<12}{r['source']:<11}"
          f"{r['bookings']:>6,} bookings  Rs {r['revenue'] / 1e5:>8,.1f} lakh")

print("\nIndexes now on the bookings collection:")
for name, info in db.bookings.index_information().items():
    print(f"  {name:45s} {info['key']}")
client.close()