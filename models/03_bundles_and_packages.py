"""
Unit 5 (analytics), model 3: which services are booked together, and package recommendations.

Part A  FP-Growth (Spark MLlib) finds the service bundles customers actually book for each
        event type (e.g. sound + lighting + decor for a Navratri society event).
Part B  Turns the bundles into ready-to-offer PACKAGES: for a city and event type it picks
        the best vendor for each service in the bundle and estimates the cost.
        This is the engine your AI Event Package Planner agent can call.

Run from the project root (venv active; run models/02_vendor_segments.py first):
    python models/03_bundles_and_packages.py
Outputs: benchmarks/bundles_by_event_type.csv, models/outputs/sample_packages.csv
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
from pyspark.ml.fpm import FPGrowth
from pyspark.sql import SparkSession, functions as F

spark = (SparkSession.builder.appName("ocazionally-bundles").master("local[2]")
         .config("spark.driver.memory", "3g").config("spark.sql.shuffle.partitions", "8")
         .getOrCreate())
spark.sparkContext.setLogLevel("ERROR")
BASE = os.path.join("data", "synthetic")

bk = spark.read.parquet(os.path.join(BASE, "bookings.parquet")).filter(F.col("status") == "completed")
events = (bk.groupBy("event_instance_id", "event_type")
          .agg(F.collect_set("category").alias("items")))
print(f"Completed events analysed: {events.count():,}\n")

# ---- Part A: FP-Growth per event type ----------------------------------------------------------------
print("PART A  SERVICE BUNDLES THAT ARE BOOKED TOGETHER (FP-Growth)")
print("  support = share of events of that type that book the whole bundle\n")
best_bundle, rows = {}, []
for et in [r[0] for r in events.select("event_type").distinct().orderBy("event_type").collect()]:
    sub = events.filter(F.col("event_type") == et)
    n = sub.count()
    fp = FPGrowth(itemsCol="items", minSupport=0.12, minConfidence=0.6).fit(sub)
    sets = (fp.freqItemsets.withColumn("size", F.size("items"))
            .filter(F.col("size") >= 2).toPandas())
    if sets.empty:
        continue
    sets["support"] = sets["freq"] / n
    sets["items"] = sets["items"].apply(lambda x: sorted(x))
    sets = sets.sort_values(["size", "support"], ascending=False)
    top = sets[sets["support"] >= 0.25].head(1)
    top = top if len(top) else sets.sort_values("support", ascending=False).head(1)
    best_bundle[et] = top.iloc[0]["items"]
    print(f"  {et:<18}({n:,} events)   best bundle: {' + '.join(top.iloc[0]['items'])}"
          f"   (booked together in {top.iloc[0]['support']:.0%} of events)")
    for r in sets.sort_values("support", ascending=False).head(3).itertuples():
        rows.append([et, " + ".join(r.items), r.size, round(r.support, 3)])
    rules = fp.associationRules.toPandas()
    if len(rules):
        r = rules.sort_values("lift", ascending=False).iloc[0]
        print(f"      strongest rule: booking {', '.join(sorted(r['antecedent']))} -> "
              f"{', '.join(sorted(r['consequent']))}   (confidence {r['confidence']:.0%}, lift {r['lift']:.2f})")
print("""
  Note the lift values (about 1.0): inside one event type, services are booked independently,
  so the bundle is simply 'what this kind of event needs'. The real signal is the other way
  round: which services reveal WHAT KIND of event a customer is planning.
""")
print("PART A2  FROM SERVICES TO EVENT TYPE (rules across all events)")
mixed = events.select(F.concat(F.col("items"), F.array(F.concat(F.lit("event="), F.col("event_type")))).alias("items"))
fp2 = FPGrowth(itemsCol="items", minSupport=0.02, minConfidence=0.1).fit(mixed)
r2 = fp2.associationRules.toPandas()
r2 = r2[r2["consequent"].apply(lambda c: len(c) == 1 and c[0].startswith("event="))]
r2 = r2[r2["antecedent"].apply(len) <= 3]            # short, readable rules
r2["antecedent"] = r2["antecedent"].apply(lambda a: " + ".join(sorted(a)))
r2["event"] = r2["consequent"].apply(lambda c: c[0].replace("event=", ""))
good = r2[r2["lift"] > 1.05].sort_values("lift", ascending=False)
top2 = good.drop_duplicates("event").sort_values("lift", ascending=False).head(8)   # best rule per event type
print(f"  {'If the customer books':<46}{'  it is probably':<22}{'confidence':>11}{'lift':>7}")
for r in top2.itertuples():
    print(f"  {r.antecedent:<46}  {r.event:<20}{r.confidence:>11.0%}{r.lift:>7.2f}")
print("  (lift above 1 means the services point to that event type more than chance would)")
rows.extend([["services -> event", f"{r.antecedent} => {r.event}", len(r.antecedent.split(' + ')),
              round(r.confidence, 3)] for r in top2.itertuples()])
os.makedirs("benchmarks", exist_ok=True)
pd.DataFrame(rows, columns=["event_type", "bundle", "size", "support"]).to_csv(
    os.path.join("benchmarks", "bundles_by_event_type.csv"), index=False)

# ---- Part B: packages ---------------------------------------------------------------------------------------------
print("\nPART B  READY-TO-OFFER PACKAGES (best vendor per service in the bundle)")
vendors = pd.read_parquet(os.path.join(BASE, "vendors.parquet"))
seg_path = os.path.join("models", "outputs", "vendor_segments.csv")
if os.path.exists(seg_path):
    seg = pd.read_csv(seg_path)[["vendor_id", "segment", "n_bookings", "cancel_rate",
                                  "avg_response_min", "n_reviews"]]
else:
    print("  (vendor_segments.csv not found: run models/02_vendor_segments.py first for better picks)")
    seg = pd.DataFrame(columns=["vendor_id", "segment", "n_bookings", "cancel_rate",
                                "avg_response_min", "n_reviews"])
v = vendors.merge(seg, on="vendor_id", how="left")
v = v[v["verification_status"] == "verified"].copy()
v["n_bookings"] = v["n_bookings"].fillna(0)
v["cancel_rate"] = v["cancel_rate"].fillna(0.1)
v["avg_response_min"] = v["avg_response_min"].fillna(v["avg_response_min"].median())
v["n_reviews"] = v["n_reviews"].fillna(0)

bkp = bk.select("vendor_id", "event_type", "amount").toPandas()
typical = bkp.groupby(["vendor_id", "event_type"])["amount"].median().rename("typical_cost").reset_index()
overall = bkp.groupby("vendor_id")["amount"].median().rename("overall_cost").reset_index()
v = v.merge(overall, on="vendor_id", how="left")


def score(df):
    """Trust score: good rating, few cancellations, fast replies, proven track record."""
    rating = df["rating_avg"].where(df["rating_avg"] > 0, 3.5)
    return (rating - 6 * df["cancel_rate"] - df["avg_response_min"].clip(upper=1000) / 600
            + 0.15 * np.log1p(df["n_reviews"]))


v["trust"] = score(v)


def build_package(city, event_type):
    bundle = best_bundle.get(event_type)
    if not bundle:
        return None
    picks = []
    for cat in bundle:
        pool = v[(v["city"] == city) & (v["category"] == cat) & (v["n_bookings"] >= 10)]
        if pool.empty:
            continue
        top = pool.sort_values("trust", ascending=False).iloc[0]
        cost = typical[(typical["vendor_id"] == top["vendor_id"]) & (typical["event_type"] == event_type)]
        cost = cost["typical_cost"].iloc[0] if len(cost) else top["overall_cost"]
        picks.append(dict(city=city, event_type=event_type, service=cat, vendor_id=top["vendor_id"],
                          vendor=str(top["name"])[:32], segment=top.get("segment"),
                          rating=top["rating_avg"], cancel_rate=round(top["cancel_rate"], 3),
                          est_cost=int(cost)))
    return pd.DataFrame(picks)


all_pk = []
for city, et in [("Ahmedabad", "navratri_society"), ("Surat", "wedding"), ("Vadodara", "corporate")]:
    pk = build_package(city, et)
    if pk is None or pk.empty:
        continue
    all_pk.append(pk)
    print(f"\n  {et.replace('_', ' ').title()} package, {city}")
    print(f"  {'Service':<13}{'Vendor':<34}{'Segment':<26}{'Rating':>7}{'Cancel':>8}{'Est. cost':>12}")
    for r in pk.itertuples():
        print(f"  {r.service:<13}{r.vendor:<34}{str(r.segment)[:24]:<26}{r.rating:>7.1f}"
              f"{r.cancel_rate:>8.0%}{'Rs ' + format(r.est_cost, ','):>12}")
    print(f"  {'':<13}{'Estimated total':<68}{'Rs ' + format(int(pk['est_cost'].sum()), ','):>12}")

if all_pk:
    os.makedirs(os.path.join("models", "outputs"), exist_ok=True)
    pd.concat(all_pk).to_csv(os.path.join("models", "outputs", "sample_packages.csv"), index=False)
print("\nSaved benchmarks/bundles_by_event_type.csv and models/outputs/sample_packages.csv")
print("Note: catering cost is per-plate price x guests inside each booking, so totals are per event.")
spark.stop()