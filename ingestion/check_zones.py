"""Diagnose which areas the west-Ahmedabad run really returned.
Run from the project root:  python ingestion/check_zones.py
"""
import pandas as pd

f = "data/private/raw_listings/ahmedabad_west.csv"
r = pd.read_csv(f, dtype=str)
print(f"Rows: {len(r)}   Unique searchString values: {r['searchString'].nunique()}\n")

print("Most common searchString values:")
print(r["searchString"].fillna("(blank)").value_counts().head(15).to_string())

r["zone"] = r["searchString"].str.extract(r" in (.*?)(?:, Ahmedabad)?$")[0]
print("\nRows per zone:")
print(r["zone"].fillna("(no zone in search text)").value_counts().to_string())

for c in ("location/lat", "location/lng"):
    r[c] = pd.to_numeric(r.get(c), errors="coerce")
g = r.groupby(r["zone"].fillna("(none)")).agg(
    rows=("title", "size"),
    median_lat=("location/lat", "median"),
    median_lng=("location/lng", "median"),
)
print("\nWhere the results really are (west Ahmedabad is roughly lng 72.45-72.53, old city 72.58-72.61):")
print(g.round(3).to_string())

r["area"] = r["neighborhood"].fillna("").str.split(",").str[-1].str.strip()
print("\nTop areas overall in this file:")
print(r["area"].replace("", "(blank)").value_counts().head(15).to_string())