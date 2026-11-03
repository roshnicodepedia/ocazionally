"""
Build public_dashboard/ : a small, SAFE folder you can publish for free.

Run from the project root (venv active):
    python dashboard/export_public_data.py

It copies only aggregated results (no raw data, no phone numbers) and builds package
recommendations for every city and event type. Real Google-listed vendors are shown ONLY as
"Decor vendor in Navrangpura (listed, V0123)": their names never leave this machine, because the
ratings and cancellation rates attached to them are simulated. The script checks this itself and
stops if a real business name appears anywhere in the output.
"""
import json
import os
import re
import shutil

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

SRC, OUT = os.path.join("data", "synthetic"), "public_dashboard"
TYPICAL = dict(wedding=300, navratri_society=400, diwali=80, birthday=40, college_fest=600,
               corporate=80, uttarayan=70, other=60)
BENCH = ["results.csv", "forecast_metrics.csv", "mongo_index_results.csv", "shard_key_analysis.csv",
         "fairness_audit.csv", "fairness_policies.csv", "bundles_by_event_type.csv",
         "vendor_segment_profile.csv", "replica_set_demo_log.txt", "streaming_results.txt"]

shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(os.path.join(OUT, "data", "figures"))
os.makedirs(os.path.join(OUT, ".streamlit"))
shutil.copy(os.path.join("dashboard", "app.py"), os.path.join(OUT, "app.py"))
open(os.path.join(OUT, "requirements.txt"), "w").write("streamlit>=1.36\npandas\nnumpy\n")
open(os.path.join(OUT, ".streamlit", "config.toml"), "w").write(
    '[theme]\nprimaryColor = "#0f8a65"\n\n[browser]\ngatherUsageStats = false\n')
open(os.path.join(OUT, "README.md"), "w").write(
    "# Ocazionally Intelligence Engine: dashboard\n\nA Big Data Analytics project (Adani University). "
    "All numbers come from a simulated marketplace; vendor names are synthetic or anonymised.\n\n"
    "Run locally: `pip install -r requirements.txt` then `streamlit run app.py`.\n")

# ---- copy aggregated results -----------------------------------------------------------------------------
for f in BENCH:
    p = os.path.join("benchmarks", f)
    if os.path.exists(p):
        shutil.copy(p, os.path.join(OUT, "data", f))
    else:
        print(f"  note: benchmarks/{f} not found (that dashboard section will be empty)")
for p, dst in [(os.path.join("models", "outputs", "forecast_predictions.csv"), "forecast_predictions.csv")]:
    if os.path.exists(p):
        shutil.copy(p, os.path.join(OUT, "data", dst))
for f in ("forecast_actual_vs_predicted.png", "fairness_audit.png"):
    p = os.path.join("report", "figures", f)
    if os.path.exists(p):
        shutil.copy(p, os.path.join(OUT, "data", "figures", f))

# ---- headline numbers -----------------------------------------------------------------------------------------
v = pd.read_parquet(os.path.join(SRC, "vendors.parquet"))
b = pd.read_parquet(os.path.join(SRC, "bookings.parquet"), columns=[
    "booking_id", "event_instance_id", "vendor_id", "category", "event_type", "status", "amount", "response_time_minutes"])
r = pd.read_parquet(os.path.join(SRC, "reviews.parquet"), columns=["vendor_id"])
logs = [os.path.join(SRC, "events_log", f) for f in sorted(os.listdir(os.path.join(SRC, "events_log")))]
head = dict(vendors=len(v), customers=int(pq.ParquetFile(os.path.join(SRC, "customers.parquet")).metadata.num_rows),
            bookings=len(b), reviews=len(r), events=int(sum(pq.ParquetFile(f).metadata.num_rows for f in logs)),
            cities=sorted(v["city"].unique().tolist()))
json.dump(head, open(os.path.join(OUT, "data", "headline.json"), "w"))

# ---- vendor trust (same formula as the models) --------------------------------------------------------------
g = b.groupby("vendor_id")
s = pd.DataFrame({"n": g.size(), "cancel": (b["status"] == "cancelled").groupby(b["vendor_id"]).mean(),
                  "resp": g["response_time_minutes"].mean(), "nr": r.groupby("vendor_id").size()})
v = v.merge(s, left_on="vendor_id", right_index=True, how="left").fillna({"n": 0, "nr": 0})
v["cancel"] = v["cancel"].fillna(0.10)
v["resp"] = v["resp"].fillna(170)
K = 10
shr = v["n"] < 10
rating = v["rating_avg"].where(v["rating_avg"] > 0, 3.5)
v["trust"] = np.where(shr, ((rating * v["nr"] + K * 3.9) / (v["nr"] + K)) - 6 * ((v["cancel"] * v["n"] + K * 0.10) / (v["n"] + K))
                      - ((v["resp"] * v["n"] + K * 170) / (v["n"] + K)).clip(upper=1000) / 600 + 0.15 * np.log1p(v["nr"]),
                      rating - 6 * v["cancel"] - v["resp"].clip(upper=1000) / 600 + 0.15 * np.log1p(v["nr"]))


def label(x):
    if x.n < 10:
        return "New vendor: not yet proven"
    if x.cancel > 0.15:
        return "Cancels more often"
    if x.resp > 400:
        return "Slower to reply"
    return "Highly reliable" if x.cancel <= 0.07 and x.rating_avg >= 4.2 else "Reliable"


v["label"] = v.apply(label, axis=1)
v["shown_name"] = np.where(v["source"] == "listed",
                           v["category"].str.replace("_", " ").str.capitalize() + " vendor in "
                           + v["area"].where(v["area"].astype(str).str.len() > 0, v["city"]) + " (listed, " + v["vendor_id"] + ")",
                           v["name"])
v = v[v["verification_status"] == "verified"]

# ---- bundles mined from bookings ------------------------------------------------------------------------------
done = b[b["status"] == "completed"].drop_duplicates(["event_instance_id", "category"])
bundles = {}
for et, grp in done.groupby("event_type"):
    share = grp.groupby("category").size() / grp["event_instance_id"].nunique()
    top = share[share >= 0.40].sort_values(ascending=False).head(5)
    bundles[et] = (top if len(top) >= 2 else share.sort_values(ascending=False).head(2)).index.tolist()

rows = []
for city in sorted(v["city"].unique()):
    for et, services in bundles.items():
        for svc in services:
            pool = v[(v["city"] == city) & (v["category"] == svc)]
            est = pool[pool["n"] >= 10].sort_values("trust", ascending=False)
            new = pool[pool["n"] < 10].sort_values("trust", ascending=False)
            picks = list(est.head(2).itertuples()) + list(new.head(1).itertuples())
            if not est.empty:
                ranks = [1, 2, 3][:len(picks)] if len(est) >= 2 else [1, 3][:len(picks)]
            else:
                picks, ranks = list(new.head(2).itertuples()), [1, 2]
            for rk, p in zip(ranks, picks):
                rows.append(dict(city=city, event_type=et, service=svc, rank=rk, vendor=p.shown_name, label=p.label,
                                 rating=round(p.rating_avg, 2), cancel_rate=round(p.cancel, 3), avg_response_min=round(p.resp),
                                 n_reviews=int(p.nr), price_min=int(p.price_min), price_max=int(p.price_max),
                                 unit="per_plate" if svc == "catering" else "per_event"))
pk = pd.DataFrame(rows)
pk.to_csv(os.path.join(OUT, "data", "packages_all.csv"), index=False)
print(f"packages: {pk[['city', 'event_type']].drop_duplicates().shape[0]} city x event-type combinations, "
      f"{len(pk)} vendor rows")

# ---- safety scan: nothing real may leave --------------------------------------------------------------------------
real = [n.lower() for n in pd.read_parquet(os.path.join(SRC, "vendors.parquet"))
        .query("source == 'listed'")["name"].astype(str) if len(n) >= 5]
phone = re.compile(r"(?<![\d.])(\+91[\s-]?)?[6-9]\d{4}[\s-]?\d{5}(?![\d.])")
bad = []
for root, _, files in os.walk(OUT):
    for f in files:
        if f.endswith((".csv", ".txt", ".json", ".md", ".py")):
            t = open(os.path.join(root, f), encoding="utf-8", errors="ignore").read()
            tl = t.lower()
            bad += [f"{f}: phone-like number" for _ in [0] if phone.search(t)]
            bad += [f"{f}: real business name '{n}'" for n in real if n in tl][:3]
if bad:
    shutil.rmtree(OUT, ignore_errors=True)
    raise SystemExit("STOPPED, and public_dashboard/ was deleted. Found:\n  " + "\n  ".join(bad))
size = sum(os.path.getsize(os.path.join(rt, f)) for rt, _, fs in os.walk(OUT) for f in fs) / 1e6
print(f"Safety scan passed ({len(real)} real business names checked). public_dashboard/ is {size:.1f} MB.")
print("Next: streamlit run public_dashboard/app.py")