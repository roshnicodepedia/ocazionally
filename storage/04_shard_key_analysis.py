"""
Unit 3 (NoSQL), step 4: choosing a shard key, using YOUR booking data.

A real sharded cluster needs about six MongoDB servers plus a router, which is
too heavy for one 8 GB laptop. So we do what engineers do before sharding: take the
real data and measure how each candidate shard key would place it on 3 shards.

Run from the project root (venv active):
    python storage/04_shard_key_analysis.py

For each candidate key it reports
  - balance:   share of documents per shard (ideal = 33% each)
  - hot spot:  where the NEWEST 90 days of writes would land
  - targeting: how many shards 4 typical queries must contact (1 = fast, 3 = scatter-gather)
Results are saved to benchmarks/shard_key_analysis.csv.
"""
import bisect
import os

import numpy as np
import pandas as pd

N_SHARDS = 3
BASE = os.path.join("data", "synthetic")

b = pd.read_parquet(os.path.join(BASE, "bookings.parquet"))
b["event_date"] = pd.to_datetime(b["event_date"])
b = b.rename(columns={"booking_id": "_id"})
print(f"Bookings analysed: {len(b):,}   Shards: {N_SHARDS}\n")

cutoff = b["event_date"].max() - pd.Timedelta(days=90)
is_new = b["event_date"] > cutoff                   # the 'newest writes'


# ---- placement functions ------------------------------------------------------------------
def hashed(col):
    h = pd.util.hash_array(b[col].astype(str).to_numpy())
    return (h % N_SHARDS).astype(int)


def key_strings(df, cols):
    parts = []
    for c in cols:
        v = df[c]
        parts.append(v.dt.strftime("%Y-%m-%d") if np.issubdtype(v.dtype, np.datetime64) else v.astype(str))
    out = parts[0]
    for p in parts[1:]:
        out = out + "|" + p
    return out


def ranged(cols):
    """Range sharding: contiguous key ranges holding equal numbers of documents.
    Boundaries are learned from OLD data; new documents are placed by those boundaries
    (which is exactly why an ever-increasing key creates a hot spot)."""
    ks = key_strings(b, cols)
    hist = ks[~is_new]
    counts = hist.value_counts().sort_index()
    before = counts.cumsum() - counts
    shard_of = np.minimum((before / counts.sum() * N_SHARDS).astype(int), N_SHARDS - 1)
    starts = [counts.index[np.argmax(shard_of.to_numpy() >= s)] for s in range(1, N_SHARDS)]
    return np.array([bisect.bisect_right(starts, k) for k in ks])


CANDIDATES = [
    dict(name="hashed _id (booking id)", fields=["_id"], hashed=True, shard=hashed("_id")),
    dict(name="hashed vendor_id", fields=["vendor_id"], hashed=True, shard=hashed("vendor_id")),
    dict(name="range city", fields=["city"], hashed=False, shard=ranged(["city"])),
    dict(name="range event_date", fields=["event_date"], hashed=False, shard=ranged(["event_date"])),
    dict(name="range {city, vendor_id}", fields=["city", "vendor_id"], hashed=False,
         shard=ranged(["city", "vendor_id"])),
]

# ---- typical queries: (field, kind, mask) ------------------------------------------------
top_vendor = b["vendor_id"].value_counts().index[0]
top_customer = b["customer_id"].value_counts().index[0]
one_id = b["_id"].iloc[1000]
win = (b["event_date"] >= "2025-09-01") & (b["event_date"] < "2025-12-01")
jan = (b["event_date"] >= "2026-01-01") & (b["event_date"] < "2026-02-01")

QUERIES = {
    "A  one vendor's bookings, 3 months": {
        "vendor_id": ("eq", b["vendor_id"] == top_vendor), "event_date": ("range", win)},
    "B  city + category + month": {
        "city": ("eq", b["city"] == "Surat"), "category": ("eq", b["category"] == "photography"),
        "event_date": ("range", jan)},
    "C  one customer's history": {"customer_id": ("eq", b["customer_id"] == top_customer)},
    "D  one booking by id": {"_id": ("eq", b["_id"] == one_id)},
}


def shards_touched(cand, query):
    """How many shards must this query contact for this shard key?"""
    mask = pd.Series(True, index=b.index)
    used = 0
    for f in cand["fields"]:                          # the usable prefix of the shard key
        if f not in query:
            break
        kind, m = query[f]
        if cand["hashed"] and kind != "eq":           # hashed keys only help equality lookups
            break
        mask &= m
        used += 1
    if used == 0:
        return N_SHARDS                               # scatter-gather
    return len(np.unique(cand["shard"][mask.to_numpy()]))


# ---- report ------------------------------------------------------------------------------------
rows = []
print(f"{'Shard key':<28}{'Distinct':>9}{'Shard 1':>9}{'Shard 2':>9}{'Shard 3':>9}"
      f"{'Imbal.':>8}{'New writes':>12}")
print(f"{'':<28}{'values':>9}{'docs':>9}{'docs':>9}{'docs':>9}{'':>8}{'on 1 shard':>12}")
print("-" * 84)
for c in CANDIDATES:
    sh = c["shard"]
    share = np.bincount(sh, minlength=N_SHARDS) / len(sh)
    imbalance = share.max() * N_SHARDS
    new_share = np.bincount(sh[is_new.to_numpy()], minlength=N_SHARDS)
    hot = new_share.max() / new_share.sum()
    card = b[c["fields"]].astype(str).drop_duplicates().shape[0]
    t = {q: shards_touched(c, spec) for q, spec in QUERIES.items()}
    c.update(share=share, imbalance=imbalance, hot=hot, touched=t, card=card)
    print(f"{c['name']:<28}{card:>9,}{share[0]:>9.0%}{share[1]:>9.0%}{share[2]:>9.0%}"
          f"{imbalance:>7.2f}x{hot:>12.0%}")
    rows.append([c["name"], card, *[round(x, 3) for x in share], round(imbalance, 2),
                 round(hot, 3), *t.values(), None])

WEIGHT = {q: w for q, w in zip(QUERIES, (3, 3, 2, 1))}     # how often the app runs each query
print("\nShards contacted per query (1 = targeted and fast, 3 = scatter-gather):\n")
for q in QUERIES:
    print(f"  {q}   (importance {WEIGHT[q]}/3)")
print()
print(f"{'Shard key':<28}" + "".join(f"{q[:1]:>6}" for q in QUERIES) + f"{'Score':>9}")
print("-" * 84)
for c in CANDIDATES:
    # 1.0 for a fully targeted query, 0 for scatter-gather, weighted by importance
    c["score"] = sum(WEIGHT[q] * (N_SHARDS - n) / (N_SHARDS - 1) for q, n in c["touched"].items())
    print(f"{c['name']:<28}" + "".join(f"{c['touched'][q]:>6}" for q in QUERIES) + f"{c['score']:>9.1f}")

# ---- recommendation -------------------------------------------------------------------------------
ok = [c for c in CANDIDATES if c["imbalance"] <= 1.10 and c["hot"] <= 0.45]
best = max(ok, key=lambda c: c["score"]) if ok else None
print("\nRESULT")
for c in CANDIDATES:
    flags = []
    if c["imbalance"] > 1.10:
        flags.append(f"uneven ({c['imbalance']:.2f}x the ideal load on its busiest shard)")
    if c["hot"] > 0.45:
        flags.append(f"write hot spot ({c['hot']:.0%} of new writes hit one shard)")
    if c["card"] < 20:
        flags.append(f"only {c['card']} distinct values, so it cannot split further")
    print(f"  {c['name']:<28}{'OK' if not flags else '; '.join(flags)}")
if best:
    print(f"\n  Recommended shard key for bookings: {best['name']}")
print("\nFor your report: the vendors collection (about 2,000 small documents) should stay "
      "unsharded and replicated. Only bookings and the platform event log are large enough to shard.")

for r, c in zip(rows, CANDIDATES):
    r[-1] = round(c["score"], 2)
os.makedirs("benchmarks", exist_ok=True)
pd.DataFrame(rows, columns=["shard_key", "distinct_values", "share_shard1", "share_shard2",
                            "share_shard3", "imbalance", "newest_writes_on_busiest_shard",
                            *["shards_" + q[:1] for q in QUERIES], "score"]).to_csv(
    os.path.join("benchmarks", "shard_key_analysis.csv"), index=False)
print("\nSaved benchmarks/shard_key_analysis.csv")