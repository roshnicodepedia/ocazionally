"""
Unit 5 (analytics), model 1: weekly demand forecasting with Spark MLlib.

Question: how many bookings will each city x vendor-category receive in a given
week, 4 WEEKS in advance - early enough for Ocazionally to recruit vendors?

Run from the project root (venv active):
    python models/01_demand_forecast.py

Train: weeks before 1 Oct 2025.   Test: 1 Oct 2025 - Sep 2026 (a full year the model
never saw: Diwali, the wedding season, Uttarayan).
Compared against two baselines: 'same as 4 weeks ago' and 'same week last year'.
Outputs: metrics table, shortage list, benchmarks/forecast_metrics.csv,
         models/outputs/forecast_predictions.csv, report/figures/forecast_actual_vs_predicted.png
"""
import os
import sys

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
if sys.platform == "win32":
    os.environ["HADOOP_HOME"] = "C:\\hadoop"
    os.environ["PATH"] = "C:\\hadoop\\bin;" + os.environ["PATH"]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.regression import GBTRegressor, RandomForestRegressor
from pyspark.sql import SparkSession, Window, functions as F

spark = (SparkSession.builder.appName("ocazionally-forecast").master("local[2]")
         .config("spark.driver.memory", "3g").config("spark.sql.shuffle.partitions", "8")
         .getOrCreate())
spark.sparkContext.setLogLevel("ERROR")
BASE = os.path.join("data", "synthetic")
SPLIT = "2025-10-01"
HORIZON = 4                                   # weeks ahead

# ---- 1. weekly demand per city x category (zeros included) ------------------------------------
bk = spark.read.parquet(os.path.join(BASE, "bookings.parquet"))
weekly = (bk.withColumn("week", F.to_date(F.date_trunc("week", "event_date")))
          .groupBy("city", "category", "week").agg(F.count("*").alias("y")))
w = weekly.agg(F.min("week"), F.max("week")).collect()[0]
weeks = pd.date_range(w[0], w[1], freq="7D")[1:-1]          # drop the two partial weeks
cities = [r[0] for r in bk.select("city").distinct().collect()]
cats = [r[0] for r in bk.select("category").distinct().collect()]
grid = (spark.createDataFrame([(c, k) for c in cities for k in cats], ["city", "category"])
        .crossJoin(spark.createDataFrame([(d.date(),) for d in weeks], ["week"])))
df = grid.join(weekly, ["city", "category", "week"], "left").fillna(0, subset=["y"])

# ---- 2. features known in advance: calendar + festivals --------------------------------------
NAV = [("2023-10-15", "2023-10-23"), ("2024-10-03", "2024-10-11"),
       ("2025-09-22", "2025-10-01"), ("2026-10-11", "2026-10-19")]
DIWALI = pd.to_datetime(["2023-11-12", "2024-11-01", "2025-10-20", "2026-11-08"])
UTTAR = pd.to_datetime(["2024-01-14", "2025-01-14", "2026-01-14"])
WED = [.9, 1.0, .5, .6, .7, .4, .15, .15, .2, .25, .9, 1.0]       # wedding-season strength by month

cal = []
for d in weeks:
    days = pd.date_range(d, periods=7)
    mid = d + pd.Timedelta(days=3)
    nav_days = sum(any(pd.Timestamp(a) <= x <= pd.Timestamp(b) for a, b in NAV) for x in days)
    cal.append((d.date(), int(d.isocalendar()[1]), int(mid.month),
                float(np.sin(2 * np.pi * mid.dayofyear / 365.25)),
                float(np.cos(2 * np.pi * mid.dayofyear / 365.25)),
                float(nav_days),
                float(np.exp(-0.5 * (np.min(np.abs((DIWALI - mid).days)) / 4.0) ** 2)),
                float(np.exp(-0.5 * (np.min(np.abs((UTTAR - mid).days)) / 3.0) ** 2)),
                WED[mid.month - 1], float((d - weeks[0]).days / 7)))
calendar = spark.createDataFrame(cal, ["week", "week_of_year", "month", "season_sin", "season_cos",
                                       "navratri_days", "diwali_proximity", "uttarayan_proximity",
                                       "wedding_season", "trend"])
df = df.join(calendar, "week")

# lag features: only information available HORIZON weeks earlier (no leakage)
win = Window.partitionBy("city", "category").orderBy("week")
for k in range(HORIZON, HORIZON + 4):
    df = df.withColumn(f"lag_{k}", F.lag("y", k).over(win))
df = (df.withColumn("lag_mean_4", (F.col("lag_4") + F.col("lag_5") + F.col("lag_6") + F.col("lag_7")) / 4)
        .withColumn("lag_52", F.lag("y", 52).over(win))           # baseline only, not a feature
        .dropna(subset=["lag_7"]))

# Tree models cannot extrapolate growth. So we predict the ratio to the recent level
# (demand 4+ weeks ago) and multiply it back: growth then carries into the forecast.
df = (df.withColumn("level", F.col("lag_mean_4") + 1.0)
        .withColumn("ratio", F.col("y") / F.col("level")))
train = df.filter(F.col("week") < SPLIT)
test = df.filter(F.col("week") >= SPLIT)
print(f"Weekly city x category rows: train {train.count():,}   test {test.count():,}\n")

# ---- 3. models ---------------------------------------------------------------------------------
num = ["week_of_year", "month", "season_sin", "season_cos", "navratri_days", "diwali_proximity",
       "uttarayan_proximity", "wedding_season", "trend", "lag_4", "lag_5", "lag_6", "lag_7", "lag_mean_4"]
prep = [StringIndexer(inputCol="city", outputCol="city_i"),
        StringIndexer(inputCol="category", outputCol="category_i"),
        OneHotEncoder(inputCols=["city_i", "category_i"], outputCols=["city_v", "category_v"]),
        VectorAssembler(inputCols=num + ["city_v", "category_v"], outputCol="features")]
gbt = Pipeline(stages=prep + [GBTRegressor(labelCol="ratio", weightCol="level", maxIter=80,
                                           maxDepth=5, stepSize=0.1, seed=7)])
rf = Pipeline(stages=prep + [RandomForestRegressor(labelCol="ratio", weightCol="level",
                                                   numTrees=100, maxDepth=8, seed=7)])
print("Training GBT and Random Forest ...")
m_gbt, m_rf = gbt.fit(train), rf.fit(train)

pred = (m_gbt.transform(test).withColumn("pred_gbt", F.col("prediction") * F.col("level"))
        .select("city", "category", "week", "y", "lag_4", "lag_52", "pred_gbt"))
pred_rf = (m_rf.transform(test).withColumn("pred_rf", F.col("prediction") * F.col("level"))
           .select("city", "category", "week", "pred_rf"))
res = pred.join(pred_rf, ["city", "category", "week"]).toPandas().sort_values(["city", "category", "week"])
res["pred_gbt"] = res["pred_gbt"].clip(lower=0)
res["pred_rf"] = res["pred_rf"].clip(lower=0)

# ---- 4. evaluation ---------------------------------------------------------------------------------
def scores(name, p, y):
    e = p - y
    return dict(model=name, MAE=np.abs(e).mean(), RMSE=np.sqrt((e ** 2).mean()),
                WMAPE=np.abs(e).sum() / y.sum() * 100)

ok = res.dropna(subset=["lag_52"])
table = pd.DataFrame([
    scores("Naive: same as 4 weeks ago", ok["lag_4"], ok["y"]),
    scores("Naive: same week last year", ok["lag_52"], ok["y"]),
    scores("Random Forest (MLlib)", ok["pred_rf"], ok["y"]),
    scores("Gradient-Boosted Trees (MLlib)", ok["pred_gbt"], ok["y"]),
]).round(2)
print(f"\nTest year ({len(ok):,} city x category x week forecasts, {HORIZON} weeks ahead)")
print(table.to_string(index=False))
print("  (MAE/RMSE in bookings per week; WMAPE = total absolute error as % of total bookings)")
best_base = table.iloc[:2]["WMAPE"].min()
print(f"\nGBT cuts the error by {(1 - table.iloc[3]['WMAPE'] / best_base) * 100:.0f}% "
      f"versus the best naive baseline.")

# what drives the forecast
names = [a["name"] for t in ("numeric", "binary")
         for a in m_gbt.transform(test.limit(5)).schema["features"].metadata["ml_attr"]["attrs"].get(t, [])]
imp = m_gbt.stages[-1].featureImportances.toArray()
top = sorted(zip(names, imp), key=lambda x: -x[1])[:8]
print("\nWhat the GBT model relies on most:")
for n, v in top:
    print(f"   {n:<22}{v:>6.1%}")

# ---- 5. business use: where will vendor supply run short? -------------------------------------------
vend = pd.read_parquet(os.path.join(BASE, "vendors.parquet"))
vend = vend[vend["verification_status"] == "verified"].copy()
vend["onboarded_date"] = pd.to_datetime(vend["onboarded_date"])
res["week"] = pd.to_datetime(res["week"])
cap = []
for (city, cat), g in res.groupby(["city", "category"]):
    sub = vend[(vend["city"] == city) & (vend["category"] == cat)]
    for wk in g["week"]:
        active = sub[sub["onboarded_date"] <= wk]["capacity_per_month"].sum()
        cap.append((city, cat, wk, active * 12 / 52))
cap = pd.DataFrame(cap, columns=["city", "category", "week", "weekly_capacity"])
res = res.merge(cap, on=["city", "category", "week"])
res["utilisation"] = res["pred_gbt"] / res["weekly_capacity"].replace(0, np.nan)
worst = res.sort_values("utilisation", ascending=False).head(8)
print("\nTightest forecast weeks (predicted bookings as % of all verified vendors' capacity):")
print(f"  {'Week of':<12}{'City':<13}{'Category':<13}{'Forecast':>9}{'Capacity':>10}{'Used':>7}")
for r in worst.itertuples():
    print(f"  {r.week.date()!s:<12}{r.city:<13}{r.category:<13}{r.pred_gbt:>9.0f}"
          f"{r.weekly_capacity:>10.0f}{r.utilisation:>7.0%}")
print("\nUse: a vendor-recruitment alert. Weeks like these, 4 weeks out, tell the team which "
      "categories to recruit before demand arrives.")

# ---- 6. save -------------------------------------------------------------------------------------------
os.makedirs("benchmarks", exist_ok=True)
os.makedirs(os.path.join("models", "outputs"), exist_ok=True)
os.makedirs(os.path.join("report", "figures"), exist_ok=True)
table.to_csv(os.path.join("benchmarks", "forecast_metrics.csv"), index=False)
res.drop(columns=["lag_4", "lag_52"]).round(2).to_csv(
    os.path.join("models", "outputs", "forecast_predictions.csv"), index=False)

tot = res.groupby("week")[["y", "pred_gbt", "pred_rf"]].sum()
tot["naive_last_year"] = ok.assign(week=pd.to_datetime(ok["week"])).groupby("week")["lag_52"].sum()
fig, ax = plt.subplots(2, 1, figsize=(11, 7.5))
ax[0].plot(tot.index, tot["y"], color="black", lw=2, label="Actual")
ax[0].plot(tot.index, tot["pred_gbt"], color="#1D9E75", lw=2, label="GBT forecast (4 weeks ahead)")
ax[0].plot(tot.index, tot["naive_last_year"], color="#B4B2A9", lw=1.5, ls="--", label="Same week last year")
ax[0].set_title("Weekly bookings, all cities and categories (test year)")
ax[0].legend()
focus = (res.groupby(["city", "category"])["y"].sum().sort_values(ascending=False).index[0])
g = res[(res["city"] == focus[0]) & (res["category"] == focus[1])].set_index("week")
ax[1].plot(g.index, g["y"], color="black", lw=2, label="Actual")
ax[1].plot(g.index, g["pred_gbt"], color="#1D9E75", lw=2, label="GBT forecast")
ax[1].set_title(f"Busiest segment: {focus[0]} {focus[1]}")
ax[1].legend()
for a in ax:
    a.set_ylabel("bookings per week")
    a.grid(alpha=.3)
fig.tight_layout()
fig.savefig(os.path.join("report", "figures", "forecast_actual_vs_predicted.png"), dpi=150)
print("\nSaved: benchmarks/forecast_metrics.csv, models/outputs/forecast_predictions.csv, "
      "report/figures/forecast_actual_vs_predicted.png")
spark.stop()