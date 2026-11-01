"""
Unit 5 (streaming analytics) and Unit 1 (velocity): a live feed of Ocazionally bookings.

Spark Structured Streaming watches a folder. A producer drops one file per simulated hour of
the busiest booking day in your data (a new file every few seconds), and Spark:
  1. reads each new file the moment it appears        (stream source)
  2. joins every booking to the vendors table          (stream-static join -> category)
  3. counts bookings per 1-hour window, city, category (windowed aggregation)
  4. raises a SURGE ALERT when a city's hourly bookings go above its normal range
     (more than 3 standard deviations above, and at least 1.5x, the usual level for that hour)

Run from the project root (venv active):
    python pipeline/03_streaming_demo.py            # about 1 minute
    python pipeline/03_streaming_demo.py --speed 1  # faster

Outputs: benchmarks/streaming_results.txt
"""
import argparse
import json
import os
import shutil
import sys
import threading
import time

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
if sys.platform == "win32":
    os.environ["HADOOP_HOME"] = "C:\\hadoop"
    os.environ["PATH"] = "C:\\hadoop\\bin;" + os.environ["PATH"]

import pandas as pd
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import StringType, StructField, StructType

ap = argparse.ArgumentParser()
ap.add_argument("--speed", type=float, default=2.5, help="seconds between simulated hours")
args = ap.parse_args()

BASE = os.path.join("data", "synthetic")
STREAM_DIR = os.path.join("data", "stream_in")
CHECKPOINT = os.path.join("data", "stream_checkpoint")
for d in (STREAM_DIR, CHECKPOINT):
    shutil.rmtree(d, ignore_errors=True)
os.makedirs(STREAM_DIR)

_log = open(os.path.join("benchmarks", "streaming_results.txt"), "w", encoding="utf-8") \
    if os.path.isdir("benchmarks") else None


def say(msg=""):
    print(msg, flush=True)
    if _log:
        _log.write(msg + "\n")
        _log.flush()


# ---- history: when are bookings normally made? -----------------------------------------------------------
ev = pd.read_parquet(os.path.join(BASE, "events_log"), columns=["action", "timestamp", "city", "vendor_id"],
                     filters=[("action", "==", "booking_confirmed")])
ev["timestamp"] = pd.to_datetime(ev["timestamp"])
ev["date"] = ev["timestamp"].dt.date
ev["hour"] = ev["timestamp"].dt.hour
daily = ev.groupby("date").size()
peak_day = daily.idxmax()
day = ev[ev["date"] == peak_day].sort_values("timestamp")
hist = ev[ev["date"] != peak_day].groupby(["city", "date", "hour"]).size().rename("n").reset_index()
base = hist.groupby(["city", "hour"])["n"].agg(["mean", "std"]).fillna(0)
base["threshold"] = (base["mean"] + 3 * base["std"]).combine(base["mean"] * 1.5, max)
threshold = base["threshold"].to_dict()
normal = base["mean"].to_dict()
say("=" * 72)
say(f"LIVE BOOKING FEED: replaying {peak_day} ({len(day):,} bookings, the busiest day in the data)")
say("=" * 72)

spark = (SparkSession.builder.appName("ocazionally-streaming").master("local[2]")
         .config("spark.driver.memory", "3g").config("spark.sql.shuffle.partitions", "4")
         .config("spark.sql.streaming.forceDeleteTempCheckpointLocation", "true")
         .getOrCreate())
spark.sparkContext.setLogLevel("ERROR")

# ---- the stream ---------------------------------------------------------------------------------------------------
schema = StructType([StructField(c, StringType()) for c in ("timestamp", "city", "vendor_id")])
vendors = spark.read.parquet(os.path.join(BASE, "vendors.parquet")).select("vendor_id", "category")
stream = (spark.readStream.schema(schema).option("maxFilesPerTrigger", 1).json(STREAM_DIR)
          .withColumn("ts", F.to_timestamp("timestamp"))
          .join(F.broadcast(vendors), "vendor_id"))                      # stream-static join
agg = (stream.groupBy(F.window("ts", "1 hour").alias("w"), "city", "category").count()
       .select(F.col("w.start").alias("hour_start"), "city", "category", "count"))

state = {"alerted": set(), "alerts": [], "batches": 0}


def on_batch(batch_df, batch_id):
    pdf = batch_df.toPandas()
    if pdf.empty:
        return
    state["batches"] += 1
    pdf["hour_start"] = pd.to_datetime(pdf["hour_start"])
    total = int(pdf["count"].sum())
    latest = pdf["hour_start"].max()
    cur = pdf[pdf["hour_start"] == latest]
    by_city = cur.groupby("city")["count"].sum().sort_values(ascending=False)
    top = cur.sort_values("count", ascending=False).head(1).iloc[0]
    line = (f"  batch {state['batches']:>2} | simulated hour {latest:%H:00} | bookings so far {total:>5,} | "
            f"this hour {int(by_city.sum()):>4} | busiest: {top['city']} {top['category']} ({int(top['count'])})")
    say(line)
    city_hour = pdf.groupby(["city", "hour_start"])["count"].sum().reset_index()
    for r in city_hour.itertuples():
        key = (r.city, r.hour_start)
        thr = threshold.get((r.city, r.hour_start.hour), None)
        if thr and r.count > thr and key not in state["alerted"]:
            state["alerted"].add(key)
            msg = (f"      SURGE ALERT  {r.city}, {r.hour_start:%H:00}-{r.hour_start.hour + 1:02d}:00 : "
                   f"{r.count} bookings, {r.count / max(normal.get((r.city, r.hour_start.hour), 1), 1):.1f}x "
                   f"the usual {normal.get((r.city, r.hour_start.hour), 0):.0f}. Notify vendors in {r.city}.")
            state["alerts"].append(msg)
            say(msg)


query = (agg.writeStream.outputMode("complete").foreachBatch(on_batch)
         .option("checkpointLocation", CHECKPOINT)
         .trigger(processingTime="1 second").start())


# ---- the producer: one file per simulated hour ------------------------------------------------------------
def produce():
    for h in range(24):
        chunk = day[day["hour"] == h]
        tmp = os.path.join(STREAM_DIR, f"_tmp_{h:02d}")
        with open(tmp, "w", encoding="utf-8") as f:
            for r in chunk.itertuples():
                f.write(json.dumps({"timestamp": r.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                                    "city": r.city, "vendor_id": r.vendor_id}) + "\n")
        os.replace(tmp, os.path.join(STREAM_DIR, f"hour_{h:02d}.json"))          # appears atomically
        time.sleep(args.speed)


say("\nStreaming started. A new hour of bookings arrives every "
    f"{args.speed:g} seconds:\n")
t0 = time.time()
producer = threading.Thread(target=produce, daemon=True)
producer.start()
producer.join()
time.sleep(6)                                                              # let the last hour finish
query.stop()
elapsed = time.time() - t0

progress = [p for p in query.recentProgress if p.get("numInputRows")]
say("\n" + "=" * 72)
say("SUMMARY FOR YOUR REPORT")
say("=" * 72)
say(f"  Bookings streamed:        {len(day):,} (in {elapsed:.0f} s of wall-clock time)")
say(f"  Micro-batches processed:  {len(progress)}")
if progress:
    lat = [p["durationMs"].get("triggerExecution", 0) for p in progress]
    say(f"  Average micro-batch latency: {sum(lat) / len(lat) / 1000:.1f} s "
        f"(from a new file arriving to the results updating)")
say(f"  Surge alerts raised:      {len(state['alerts'])}")
say("  Pipeline: file source -> stream-static join (vendors) -> 1-hour window -> alert rule")
say("  Velocity (Unit 1): the same code would read from Kafka instead of a folder.")
shutil.rmtree(STREAM_DIR, ignore_errors=True)
shutil.rmtree(CHECKPOINT, ignore_errors=True)
spark.stop()