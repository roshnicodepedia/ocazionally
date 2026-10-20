"""
Admin review of the listed vendors (this is the 'vendor_verified' step in real life).

  python ingestion/exclude_vendors.py --top 40        show the 40 most-reviewed vendors
  python ingestion/exclude_vendors.py V0148 V0430     mark these vendors as NOT real event vendors
  python ingestion/exclude_vendors.py --undo V0148    bring one back

Excluded vendors are skipped by the data generator. Your choices are kept when
you re-run clean_listings.py.
"""
import argparse
import os

import pandas as pd

FILE = os.path.join("data", "private", "vendors_listed.csv")

ap = argparse.ArgumentParser()
ap.add_argument("ids", nargs="*", help="vendor ids to exclude, e.g. V0148")
ap.add_argument("--top", type=int, default=0, help="list the most-reviewed vendors")
ap.add_argument("--undo", action="store_true", help="re-include the given ids")
args = ap.parse_args()

d = pd.read_csv(FILE, dtype=str).fillna("")
if "manual_exclude" not in d.columns:
    d["manual_exclude"] = ""

if args.ids:
    missing = [i for i in args.ids if i not in set(d["vendor_id"])]
    if missing:
        print("Not found:", ", ".join(missing))
    d.loc[d["vendor_id"].isin(args.ids), "manual_exclude"] = "" if args.undo else "yes"
    d.to_csv(FILE, index=False, encoding="utf-8-sig")
    verb = "re-included" if args.undo else "excluded"
    print(f"{len(set(args.ids) - set(missing))} vendor(s) {verb}. "
          f"Total excluded now: {(d['manual_exclude'] == 'yes').sum()}")

if args.top:
    live = d[(d["likely_event_vendor"] != "no") & (d["manual_exclude"] != "yes")].copy()
    live["reviews"] = pd.to_numeric(live["google_review_count"], errors="coerce").fillna(0)
    top = live.sort_values("reviews", ascending=False).head(args.top)
    pd.set_option("display.width", 200, "display.max_colwidth", 38)
    print(top[["vendor_id", "name", "google_category", "category", "reviews"]]
          .to_string(index=False))
    print("\nSpot anything that is not an event vendor? Run:  "
          "python ingestion/exclude_vendors.py V0123 V0456")