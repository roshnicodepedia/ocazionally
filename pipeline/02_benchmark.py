"""
Step 2 of the Spark pipeline: MapReduce-style vs DataFrame (Unit 2 and Unit 4).

The SAME two jobs are written twice:
  (a) MapReduce programming model: map -> shuffle -> reduce, on Spark RDDs
  (b) Spark DataFrame / SparkSQL, which uses the Catalyst optimiser and
      Parquet's columnar format

Run from the project root (venv active):
    python pipeline/02_benchmark.py            # 3 timed runs per method
    python pipeline/02_benchmark.py --runs 2   # quicker

Results are printed and saved to benchmarks/results.csv.

Honest note for the report: this compares the MapReduce *programming model* on
Spark's engine. Classic Hadoop MapReduce also writes intermediate data to disk
between stages, which makes it slower still. We could not run a Hadoop cluster
on an 8 GB laptop, so we describe that difference rather than measure it.
"""
import argparse
import csv
import os
import statistics
import sys
import time

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
if sys.platform == "win32":
    os.environ["HADOOP_HOME"] = "C:\\hadoop"
    os.environ["PATH"] = "C:\\hadoop\\bin;" + os.environ["PATH"]

from pyspark.sql import SparkSession, functions as F

ap = argparse.ArgumentParser()
ap.add_argument("--runs", type=int, default=3)
ap.add_argument("--only", type=int, choices=[1, 2], help="run just Job 1 or Job 2")
args = ap.parse_args()

spark = (SparkSession.builder.appName("ocazionally-benchmark").master("local[2]")
         .config("spark.driver.memory", "3g")
         .config("spark.sql.shuffle.partitions", "8")
         .getOrCreate())
spark.sparkContext.setLogLevel("ERROR")

BASE = os.path.join("data", "synthetic")
bookings = spark.read.parquet(os.path.join(BASE, "bookings.parquet"))
events = spark.read.parquet(os.path.join(BASE, "events_log"))
n_book, n_ev = bookings.count(), events.count()
print(f"Data: {n_book:,} bookings, {n_ev:,} platform events\n")


# --------------------------------------------------------------------------
# JOB 1: completed bookings and revenue per city x category x month
# --------------------------------------------------------------------------
def job1_mapreduce():
    return (bookings.rdd
            .filter(lambda r: r["status"] == "completed")                        # map side
            .map(lambda r: ((r["city"], r["category"], str(r["event_date"])[:7]),
                            (1, r["amount"])))                                   # emit (key, value)
            .reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1]))                # shuffle + reduce
            .collectAsMap())


def job1_dataframe():
    rows = (bookings.filter(F.col("status") == "completed")
            .groupBy("city", "category", F.date_format("event_date", "yyyy-MM").alias("month"))
            .agg(F.count("*").alias("n"), F.sum("amount").alias("revenue"))
            .collect())
    return {(r["city"], r["category"], r["month"]): (r["n"], r["revenue"]) for r in rows}


# --------------------------------------------------------------------------
# JOB 2: platform events per month x action (the 4-million-row table)
# --------------------------------------------------------------------------
def job2_mapreduce():
    return (events.rdd
            .map(lambda r: ((r["timestamp"].strftime("%Y-%m"), r["action"]), 1))
            .reduceByKey(lambda a, b: a + b)
            .collectAsMap())


def job2_dataframe():
    rows = (events.groupBy(F.date_format("timestamp", "yyyy-MM").alias("month"), "action")
            .count().collect())
    return {(r["month"], r["action"]): r["count"] for r in rows}


def attempt(fn, tries=4):
    """On Windows, Spark sometimes cannot start a Python worker in time
    ('Python worker failed to connect back'). It is not a code problem, so wait
    and try again instead of aborting a long benchmark."""
    for k in range(tries):
        try:
            return fn()
        except Exception as e:
            if k == tries - 1:
                raise
            print(f"     (Spark worker start-up hiccup, retrying in 8s: {str(e)[:60]}...)")
            time.sleep(8)


def time_it(fn, runs):
    attempt(fn)                           # warm-up run, not counted
    times, result = [], None
    for i in range(runs):
        t = time.time()
        result = attempt(fn)
        times.append(time.time() - t)
        print(f"     run {i + 1}: {times[-1]:.1f}s")
    return times, result


def save(results):
    """Write benchmarks/results.csv, keeping rows of jobs not run this time."""
    os.makedirs("benchmarks", exist_ok=True)
    path = os.path.join("benchmarks", "results.csv")
    rows = {}
    if os.path.exists(path):
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                rows[r["job"]] = r
    for job, label, n, a, b, same in results:
        rows[job] = dict(job=job, description=label, rows=n, mapreduce_seconds=round(a, 2),
                         dataframe_seconds=round(b, 2), speedup=round(a / b, 2), results_match=same)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["job", "description", "rows", "mapreduce_seconds",
                                          "dataframe_seconds", "speedup", "results_match"])
        w.writeheader()
        for k in sorted(rows):
            w.writerow(rows[k])
    return list(rows.values())


results = []
JOBS = [("Job 1", "bookings per city/category/month", n_book, job1_mapreduce, job1_dataframe),
        ("Job 2", "events per month/action", n_ev, job2_mapreduce, job2_dataframe)]
for job, label, rows, mr, df in JOBS:
    if args.only and job != f"Job {args.only}":
        continue
    print(f"{job}: {label} ({rows:,} rows)")
    print("   MapReduce style (RDD):")
    t_mr, r_mr = time_it(mr, args.runs)
    print("   DataFrame / SparkSQL:")
    t_df, r_df = time_it(df, args.runs)

    # correctness check: both methods must give the same answer
    same = (r_mr.keys() == r_df.keys()) and all(
        tuple(r_mr[k]) == tuple(r_df[k]) if isinstance(r_mr[k], (tuple, list)) else r_mr[k] == r_df[k]
        for k in r_mr)
    print(f"   Same result from both methods: {'YES' if same else 'NO  <-- investigate'}\n")
    results.append((job, label, rows, statistics.median(t_mr), statistics.median(t_df), same))
    save(results)                         # keep what we have, in case a later job fails

all_rows = save(results)
print("=" * 78)
print(f"{'Job':<7}{'Rows':>12}{'MapReduce-style':>20}{'DataFrame':>14}{'Speed-up':>12}")
print("-" * 78)
for r in sorted(all_rows, key=lambda r: r["job"]):
    print(f"{r['job']:<7}{int(r['rows']):>12,}{float(r['mapreduce_seconds']):>18.1f}s"
          f"{float(r['dataframe_seconds']):>12.1f}s{float(r['speedup']):>11.1f}x")
print("=" * 78)
print("(median of timed runs, after one warm-up run)")
print("Saved benchmarks/results.csv")
spark.stop()