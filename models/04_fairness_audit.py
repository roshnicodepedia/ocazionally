"""
Unit 6 (ethics): is Ocazionally fair to NEW vendors?

A marketplace that ranks by track record can lock newcomers out: they are not shown
because they have no history, and they get no history because they are not shown.
This script (1) measures that, then (2) tests a fix.

Run from the project root (venv active):
    python models/04_fairness_audit.py

PART 1  Exposure audit: do equally good vendors get fewer bookings while they are new?
PART 2  Recommendation audit: who would the platform's top-5 lists show?
        Policy A (today): only vendors with 10+ bookings, ranked by trust score.
        Policy B (fairer): top 4 as above, 1 slot reserved for the best newcomer, ranked with
        a score that is shrunk toward the average when there is little evidence.

Fairness measure: disparate-impact ratio = (group's share of exposure) / (group's share of vendors).
1.0 is fair. Below 0.8 is flagged (the 'four-fifths rule' from employment law).
NOTE: 'true quality' exists only because the data is simulated. A real audit would use proxies
(ratings, repeat bookings) and says so.
Outputs: benchmarks/fairness_audit.csv, report/figures/fairness_audit.png
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = os.path.join("data", "synthetic")
NEW_DAYS = 180          # a vendor is 'new' for its first 180 days
MIN_BOOKINGS = 10
TOP_K = 5

vendors = pd.read_parquet(os.path.join(BASE, "vendors.parquet"))
bookings = pd.read_parquet(os.path.join(BASE, "bookings.parquet"))
reviews = pd.read_parquet(os.path.join(BASE, "reviews.parquet"))
hidden = pd.read_parquet(os.path.join(BASE, "vendors_hidden_truth.parquet"))[["vendor_id", "quality", "reliability"]]

vendors["onboarded_date"] = pd.to_datetime(vendors["onboarded_date"])
ver = vendors[vendors["verification_status"] == "verified"].merge(hidden, on="vendor_id")
bookings["m"] = pd.to_datetime(bookings["booking_date"]).dt.to_period("M")
reviews["m"] = pd.to_datetime(reviews["timestamp"]).dt.to_period("M")
months = pd.period_range("2023-10", "2026-09", freq="M")

# ---- vendor x month table of what happened -------------------------------------------------------------
b = bookings.groupby(["vendor_id", "m"]).agg(
    n_book=("booking_id", "size"), n_cancel=("status", lambda s: (s == "cancelled").sum()),
    sum_resp=("response_time_minutes", "sum")).reset_index()
r = reviews.groupby(["vendor_id", "m"]).agg(n_rev=("rating", "size"), sum_rating=("rating", "sum")).reset_index()
grid = pd.MultiIndex.from_product([ver["vendor_id"], months], names=["vendor_id", "m"]).to_frame(index=False)
g = (grid.merge(b, how="left", on=["vendor_id", "m"]).merge(r, how="left", on=["vendor_id", "m"])
     .merge(ver[["vendor_id", "city", "category", "onboarded_date", "quality", "reliability"]], on="vendor_id")
     .fillna({"n_book": 0, "n_cancel": 0, "sum_resp": 0, "n_rev": 0, "sum_rating": 0}))
g["month_start"] = g["m"].dt.to_timestamp()
g["tenure_days"] = (g["month_start"] - g["onboarded_date"]).dt.days
g = g[g["tenure_days"] >= 0].sort_values(["vendor_id", "m"]).reset_index(drop=True)      # on the platform

# ---- PART 1: exposure audit ------------------------------------------------------------------------------------
mean_grp = g.groupby(["city", "category", "m"])["n_book"].transform("mean")
g["relative_exposure"] = np.where(mean_grp > 0, g["n_book"] / mean_grp, np.nan)
g["tenure_group"] = pd.cut(g["tenure_days"], [-1, 89, 364, 10 ** 6],
                           labels=["new (<90 days)", "settling (90-365)", "established (>1 yr)"])
g["quality_band"] = pd.qcut(g["quality"], 3, labels=["lower quality", "middle quality", "higher quality"])

print("PART 1  EXPOSURE AUDIT")
print("  Bookings per vendor-month relative to peers in the same city + category (1.0 = average)\n")
tab = g.pivot_table(index="quality_band", columns="tenure_group", values="relative_exposure",
                    aggfunc="mean", observed=True).round(2)
print(tab.to_string())
ratio = (tab["new (<90 days)"] / tab["established (>1 yr)"]).round(2)
print("\n  Disparate-impact ratio, new vs established vendors of the SAME quality (1.0 = fair, <0.8 flagged):")
for band, val in ratio.items():
    print(f"     {band:<16}{val:>6.2f}   {'FLAGGED' if val < 0.8 else 'ok'}")

# ---- PART 2: recommendation audit ---------------------------------------------------------------------------
for c in ("n_book", "n_cancel", "sum_resp", "n_rev", "sum_rating"):
    g["cum_" + c] = g.groupby("vendor_id")[c].cumsum() - g[c]          # evidence available at month START

g0_rating = reviews["rating"].mean()
g0_cancel = (bookings["status"] == "cancelled").mean()
g0_resp = bookings["response_time_minutes"].mean()
K = 10                                                                      # shrinkage strength


def trust_raw(d):
    rating = np.where(d["cum_n_rev"] >= 3, d["cum_sum_rating"] / d["cum_n_rev"].clip(lower=1), 3.5)
    cancel = d["cum_n_cancel"] / d["cum_n_book"].clip(lower=1)
    resp = (d["cum_sum_resp"] / d["cum_n_book"].clip(lower=1)).clip(upper=1000)
    return rating - 6 * cancel - resp / 600 + 0.15 * np.log1p(d["cum_n_rev"])


def trust_shrunk(d):
    """Little evidence -> pulled toward the platform average (no penalty, no bonus)."""
    n, nr = d["cum_n_book"], d["cum_n_rev"]
    rating = (d["cum_sum_rating"] + K * g0_rating) / (nr + K)
    cancel = (d["cum_n_cancel"] + K * g0_cancel) / (n + K)
    resp = ((d["cum_sum_resp"] + K * g0_resp) / (n + K)).clip(upper=1000)
    return rating - 6 * cancel - resp / 600 + 0.15 * np.log1p(nr)


g["trust_a"] = trust_raw(g)
g["trust_b"] = trust_shrunk(g)
g["is_new"] = g["tenure_days"] < NEW_DAYS

picksA, picksB, eligible_new, eligible_all = [], [], 0, 0
for T in months[-12:]:
    s = g[g["m"] == T]
    for (city, cat), grp in s.groupby(["city", "category"]):
        eligible_all += len(grp)
        eligible_new += grp["is_new"].sum()
        est = grp[grp["cum_n_book"] >= MIN_BOOKINGS].sort_values("trust_a", ascending=False)
        a = est.head(TOP_K)
        picksA.append(a)
        keep = est.head(TOP_K - 1)
        rest = grp[grp["is_new"] & ~grp["vendor_id"].isin(keep["vendor_id"])]
        newcomer = rest.sort_values(["trust_b", "cum_n_book"], ascending=[False, True]).head(1)
        picksB.append(pd.concat([keep, newcomer]))
A, B = pd.concat(picksA), pd.concat(picksB)
share_vendors_new = eligible_new / eligible_all


def summarise(name, d):
    return dict(policy=name, slots=len(d), share_new=d["is_new"].mean(),
                di_ratio=d["is_new"].mean() / share_vendors_new,
                distinct_vendors=d["vendor_id"].nunique(),
                true_quality=d["quality"].mean(), true_reliability=d["reliability"].mean(),
                newcomer_quality=d.loc[d["is_new"], "quality"].mean())


res = pd.DataFrame([summarise("A  today: needs 10+ bookings", A),
                    summarise("B  fairer: 1 reserved newcomer slot", B)])
print(f"\nPART 2  RECOMMENDATION AUDIT  (top {TOP_K} per city + category, every month of the last year)")
print(f"  New vendors are {share_vendors_new:.0%} of the vendors on the platform.\n")
print(f"  {'Policy':<38}{'Slots to new':>13}{'DI ratio':>10}{'Vendors shown':>15}{'True quality':>14}{'Reliability':>13}")
for x in res.itertuples():
    print(f"  {x.policy:<38}{x.share_new:>13.1%}{x.di_ratio:>10.2f}{x.distinct_vendors:>15,}"
          f"{x.true_quality:>14.3f}{x.true_reliability:>13.3f}")
qa, qb = res.iloc[0]["true_quality"], res.iloc[1]["true_quality"]
print(f"\n  Cost of fairness: average true quality of the lists changes by {qb - qa:+.3f} "
      f"({(qb - qa) / qa * 100:+.1f}%).")
nq = res.iloc[1]["newcomer_quality"]
print(f"  The newcomers shown have an average true quality of {nq:.3f}, versus "
      f"{g.loc[g['is_new'], 'quality'].mean():.3f} for all new vendors.")

os.makedirs("benchmarks", exist_ok=True)
os.makedirs(os.path.join("report", "figures"), exist_ok=True)
out = tab.copy()
out["ratio_new_vs_established"] = ratio
out.to_csv(os.path.join("benchmarks", "fairness_audit.csv"))
res.to_csv(os.path.join("benchmarks", "fairness_policies.csv"), index=False)

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].bar(range(len(ratio)), ratio.values, color="#D85A30")
ax[0].axhline(0.8, color="black", ls="--", lw=1)
ax[0].axhline(1.0, color="#1D9E75", lw=1)
ax[0].set_xticks(range(len(ratio)))
ax[0].set_xticklabels([str(i).replace(" quality", "") for i in ratio.index])
ax[0].set_ylim(0, 1.2)
ax[0].set_title("Exposure of new vs established vendors\n(same quality; 1.0 = fair, dashed = 0.8)")
ax[0].set_ylabel("disparate-impact ratio")
ax[1].bar([0, 1], [res.iloc[0]["share_new"] * 100, res.iloc[1]["share_new"] * 100], color=["#D85A30", "#1D9E75"])
ax[1].axhline(share_vendors_new * 100, color="black", ls="--", lw=1)
ax[1].set_xticks([0, 1])
ax[1].set_xticklabels(["A: today", "B: reserved\nnewcomer slot"])
ax[1].set_title(f"Share of top-{TOP_K} slots shown to new vendors\n(dashed = their share of vendors)")
ax[1].set_ylabel("% of recommendation slots")
fig.tight_layout()
fig.savefig(os.path.join("report", "figures", "fairness_audit.png"), dpi=150)
print("\nSaved benchmarks/fairness_audit.csv, benchmarks/fairness_policies.csv, report/figures/fairness_audit.png")