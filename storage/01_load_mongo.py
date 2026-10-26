"""
Unit 3 (NoSQL), step 1: load Ocazionally into MongoDB.

Run from the project root (venv active, MongoDB running):
    python storage/01_load_mongo.py

What it builds (database "ocazionally"):
  vendors   one document per vendor, with nested price / location / stats
  bookings  one document per booking (the operational, high-volume data)

Why a document database for this? A vendor profile is one self-contained
record with nested and variable parts, read as a whole by the app.
Indexes are added in the NEXT script so you can show before/after speed.
"""
import argparse
import json
import os
import time
from datetime import datetime, time as dtime

import pandas as pd
from pymongo import MongoClient

ap = argparse.ArgumentParser()
ap.add_argument("--uri", default="mongodb://localhost:27017")
ap.add_argument("--db", default="ocazionally")
args = ap.parse_args()

BASE = os.path.join("data", "synthetic")


def to_dt(d):
    """date / Timestamp -> datetime (the type MongoDB stores as a date)."""
    if d is None or pd.isna(d):
        return None
    if isinstance(d, datetime):
        return d.to_pydatetime() if hasattr(d, "to_pydatetime") else d
    return datetime.combine(d, dtime.min)


client = MongoClient(args.uri, serverSelectionTimeoutMS=4000)
client.server_info()                                   # fails fast if MongoDB is not running
db = client[args.db]
print(f"Connected to MongoDB, database '{args.db}'")

print("\nReading Parquet files ...")
vendors = pd.read_parquet(os.path.join(BASE, "vendors.parquet"))
bookings = pd.read_parquet(os.path.join(BASE, "bookings.parquet"))
reviews = pd.read_parquet(os.path.join(BASE, "reviews.parquet"))

# ---- per-vendor statistics (computed once, stored inside each vendor document) --------
done = bookings[bookings["status"] == "completed"]
st = pd.DataFrame({
    "bookings_total": bookings.groupby("vendor_id").size(),
    "bookings_completed": done.groupby("vendor_id").size(),
    "bookings_cancelled": bookings[bookings["status"] == "cancelled"].groupby("vendor_id").size(),
    "revenue_inr": done.groupby("vendor_id")["amount"].sum(),
    "avg_response_min": bookings.groupby("vendor_id")["response_time_minutes"].mean().round(1),
    "n_reviews": reviews.groupby("vendor_id").size(),
}).fillna(0)
st["cancel_rate"] = (st["bookings_cancelled"] / st["bookings_total"].replace(0, 1)).round(3)
stats = st.to_dict("index")

# ---- vendors -------------------------------------------------------------------------------
vendor_docs = []
for r in vendors.itertuples(index=False):
    s = stats.get(r.vendor_id, {})
    vendor_docs.append({
        "_id": r.vendor_id,
        "name": r.name,
        "category": r.category,
        "location": {"city": r.city, "area": r.area or None},
        "price": {"min": int(r.price_min), "max": int(r.price_max),
                  "unit": "per_plate" if r.category == "catering" else "per_event"},
        "capacity_per_month": int(r.capacity_per_month),
        "peak_months": [int(m) for m in str(r.peak_months).split(",") if m],
        "years_active": int(r.years_active),
        "rating_avg": float(r.rating_avg),
        "onboarded_date": to_dt(r.onboarded_date),
        "verification_status": r.verification_status,
        "source": r.source,
        "stats": {k: (float(v) if k in ("avg_response_min", "cancel_rate") else int(v))
                  for k, v in s.items()},
    })

# ---- bookings ------------------------------------------------------------------------------
print("Preparing booking documents ...")
booking_docs = []
for r in bookings.itertuples(index=False):
    booking_docs.append({
        "_id": r.booking_id,
        "event_instance_id": r.event_instance_id,
        "customer_id": r.customer_id,
        "vendor_id": r.vendor_id,
        "category": r.category,
        "event_type": r.event_type,
        "city": r.city,
        "guest_count": int(r.guest_count),
        "booking_date": to_dt(r.booking_date),
        "event_date": to_dt(r.event_date),
        "amount": int(r.amount),
        "status": r.status,
        "cancelled_by": r.cancelled_by or None,
        "response_time_minutes": int(r.response_time_minutes),
    })


def load(name, docs, batch=20000):
    db.drop_collection(name)                           # safe to re-run
    t = time.time()
    for i in range(0, len(docs), batch):
        db[name].insert_many(docs[i:i + batch], ordered=False)
    print(f"  {name:9s} {len(docs):>8,} documents loaded in {time.time() - t:.1f}s")


print("\nLoading into MongoDB ...")
load("vendors", vendor_docs)
load("bookings", booking_docs)

print("\nCollections now in the database:")
for name in db.list_collection_names():
    print(f"  {name:9s} {db[name].count_documents({}):>8,} documents")

print("\nSample vendor document:")
sample = db.vendors.find_one({"source": "listed", "verification_status": "verified"}) or db.vendors.find_one()
print(json.dumps(sample, indent=2, default=str))
client.close()