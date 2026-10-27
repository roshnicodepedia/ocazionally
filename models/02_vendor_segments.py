"""
Unit 5 (analytics), model 2: vendor segmentation with K-Means (Spark MLlib).

Groups verified vendors by behaviour (volume, price level, cancellations, response speed,
ratings) so the marketplace can treat them differently: promote the reliable, coach the
at-risk, and give newcomers a fair chance.

Run from the project root (venv active):
    python models/02_vendor_segments.py

Vendors with fewer than 10 bookings are NOT clustered: there is too little evidence.
They form their own 'New / unproven' segment (the cold-start group the fairness check
in Unit 6 will look at).
Outputs: models/outputs/vendor_segments.csv, benchmarks/vendor_segment_profile.csv
"""
import os
import sys

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
if sys.platform == "win32":
    os.environ["HADOOP_HOME"] = "C:\\hadoop"
    os.environ["PATH"] = "C:\\hadoop\\bin;" + os.environ["PATH"]

import numpy as np
import pandas as pd
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.sql import SparkSession, functions as F

spark = (SparkSession.builder.appName("ocazionally-segments").master("local[2]")
         .config("spark.driver.memory", "3g").config("spark.sql.shuffle.partitions", "8")
         .getOrCreate())
spark.sparkContext.setLogLevel("ERROR")
BASE = os.path.join("data", "synthetic")
MIN_BOOKINGS = 10

# ---- 1. one row of behaviour per vendor ---------------------------------------------------------
vendors = pd.read_parquet(os.path.join(BASE, "vendors.parquet"))
bookings = spark.read.parquet(os.path.join(BASE, "bookings.parquet"))
agg = (bookings.groupBy("vendor_id").agg(
    F.count("*").alias("n_bookings"),
    F.avg((F.col("status") == "cancelled").cast("double")).alias("cancel_rate"),
    F.avg("response_time_minutes").alias("avg_response_min"),
    F.min("event_date").alias("first_booking"), F.max("event_date").alias("last_booking"),
)).toPandas()
reviews = pd.read_parquet(os.path.join(BASE, "reviews.parquet")).groupby("vendor_id").size().rename("n_reviews")

v = vendors.merge(agg, on="vendor_id", how="left").merge(reviews, on="vendor_id", how="left")
v["n_bookings"] = v["n_bookings"].fillna(0)
v["n_reviews"] = v["n_reviews"].fillna(0)
v["cancel_rate"] = v["cancel_rate"].fillna(0)
v["first_booking"] = pd.to_datetime(v["first_booking"])
v["last_booking"] = pd.to_datetime(v["last_booking"])
months = ((v["last_booking"] - v["first_booking"]).dt.days / 30.4).clip(lower=1).fillna(1)
v["bookings_per_month"] = v["n_bookings"] / months
v["capacity_use"] = v["bookings_per_month"] / v["capacity_per_month"]
med_price = v.groupby("category")["price_min"].transform("median")
v["price_index"] = v["price_min"] / med_price                # 1.0 = typical for its category
v["log_reviews"] = np.log1p(v["n_reviews"])

rated = v["rating_avg"] > 0
v["rating_for_model"] = v["rating_avg"].where(rated, v.loc[rated, "rating_avg"].mean())   # 0 means 'no reviews yet'
verified = v[v["verification_status"] == "verified"].copy()
clustered = verified[verified["n_bookings"] >= MIN_BOOKINGS].copy()
print(f"Verified vendors: {len(verified):,}   with enough bookings to cluster: {len(clustered):,}   "
      f"(new / unproven: {len(verified) - len(clustered):,})\n")

FEATURES = ["bookings_per_month", "capacity_use", "price_index", "cancel_rate",
            "avg_response_min", "rating_for_model", "log_reviews"]
sdf = spark.createDataFrame(clustered[["vendor_id"] + FEATURES].astype({c: float for c in FEATURES}))
sdf = VectorAssembler(inputCols=FEATURES, outputCol="raw").transform(sdf)
sdf = StandardScaler(inputCol="raw", outputCol="features", withMean=True, withStd=True).fit(sdf).transform(sdf)
sdf.cache()

# ---- 2. choose k by silhouette ------------------------------------------------------------------------
print("Choosing the number of segments (silhouette: higher = cleaner separation)")
best = None
for k in range(3, 7):
    model = KMeans(k=k, seed=7, maxIter=60, featuresCol="features").fit(sdf)
    sil = ClusteringEvaluator(featuresCol="features", metricName="silhouette").evaluate(model.transform(sdf))
    print(f"   k = {k}   silhouette = {sil:.3f}")
    if best is None or sil > best[0]:
        best = (sil, k, model)
sil, k, model = best
print(f"   -> using k = {k}\n")
labelled = model.transform(sdf).select("vendor_id", F.col("prediction").alias("cluster")).toPandas()
clustered = clustered.merge(labelled, on="vendor_id")

# ---- 3. name the segments from their behaviour -------------------------------------------------------
prof = clustered.groupby("cluster")[FEATURES + ["n_bookings"]].mean()
z = (prof - prof.mean()) / prof.std(ddof=0).replace(0, 1)
names, left = {}, set(prof.index)


def pick(label, series):
    if left:
        c = series[list(left)].idxmax()
        names[c] = label
        left.discard(c)


pick("Frequent cancellers", z["cancel_rate"])
pick("Slow responders", z["avg_response_min"])
pick("Reliable performers", z["rating_for_model"] - z["cancel_rate"] - z["avg_response_min"])
pick("High-volume workhorses", z["bookings_per_month"])
pick("Premium-priced", z["price_index"])
for c in left:
    names[c] = f"Steady mid-market {c}"
clustered["segment"] = clustered["cluster"].map(names)

new = verified[~verified["vendor_id"].isin(clustered["vendor_id"])].copy()
new["segment"] = "New / unproven (under 10 bookings)"
other = v[v["verification_status"] != "verified"].copy()
other["segment"] = "Not verified"
everyone = pd.concat([clustered, new, other], ignore_index=True)

# ---- 4. profile ----------------------------------------------------------------------------------------------
show = everyone.groupby("segment").agg(
    vendors=("vendor_id", "count"), bookings=("n_bookings", "mean"), per_month=("bookings_per_month", "mean"),
    cancel=("cancel_rate", "mean"), response_min=("avg_response_min", "mean"),
    rating=("rating_avg", lambda x: x[x > 0].mean()), price_idx=("price_index", "mean")).round(2)
show["cancel"] = (show["cancel"] * 100).round(1)
show = show.rename(columns={"bookings": "avg bookings", "per_month": "per month", "cancel": "cancel %",
                            "response_min": "response min", "price_idx": "price index"})
print("SEGMENT PROFILES")
print(show.sort_values("vendors", ascending=False).to_string())

# ---- 5. sanity check against the hidden truth (only possible because the data is simulated) ---------
hid_path = os.path.join(BASE, "vendors_hidden_truth.parquet")
if os.path.exists(hid_path):
    hid = pd.read_parquet(hid_path)[["vendor_id", "reliability", "quality"]]
    chk = clustered.merge(hid, on="vendor_id").groupby("segment")[["reliability", "quality"]].mean().round(3)
    print("\nCHECK: do the segments recover the vendors' real (hidden) behaviour?")
    print("  (true reliability / quality exist only in the simulation, never given to the model)")
    print(chk.to_string())
    r = clustered.merge(hid, on="vendor_id")
    print(f"  correlation between true reliability and observed cancel rate: "
          f"{np.corrcoef(r['reliability'], r['cancel_rate'])[0, 1]:.2f}")

# ---- 6. who needs attention --------------------------------------------------------------------------------
risk = everyone[everyone["segment"].isin(["Frequent cancellers", "Slow responders"])
                ].sort_values("cancel_rate", ascending=False).head(5)
print("\nMost at-risk vendors (candidates for a call from the Ocazionally team):")
for r in risk.itertuples():
    print(f"   {r.vendor_id}  {str(r.name)[:30]:<31}{r.category:<12}{r.city:<12}"
          f"cancel {r.cancel_rate:.0%}   response {r.avg_response_min:.0f} min")

os.makedirs(os.path.join("models", "outputs"), exist_ok=True)
os.makedirs("benchmarks", exist_ok=True)
everyone[["vendor_id", "name", "category", "city", "segment", "n_bookings", "bookings_per_month",
          "cancel_rate", "avg_response_min", "rating_avg", "n_reviews", "price_index", "source",
          "onboarded_date"]].round(3).to_csv(os.path.join("models", "outputs", "vendor_segments.csv"), index=False)
show.to_csv(os.path.join("benchmarks", "vendor_segment_profile.csv"))
print("\nSaved models/outputs/vendor_segments.csv and benchmarks/vendor_segment_profile.csv")
spark.stop()