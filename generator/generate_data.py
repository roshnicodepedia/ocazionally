"""
Ocazionally synthetic data generator  (schema v1, see docs/schema.md)

Simulates three years of marketplace activity (Oct 2023 - Sep 2026) for five
Gujarat cities, in the exact table formats of docs/schema.md:

    vendors, customers, bookings, reviews, events_log

What is REAL here: the Ahmedabad vendor names, categories, areas and Google
review counts (from data/private/vendors_listed.csv, if present).
What is SIMULATED: prices, capacity, demand, bookings, reviews, vendor
behaviour. Every row is marked source = synthetic (or listed for the real
vendor names) and simulated attributes are flagged. Say this plainly in the
report: the platform has no real transaction history yet.

Usage (project root, venv active):
    python generator/generate_data.py                 # default scale 1.0
    python generator/generate_data.py --scale 0.2     # quick test run
    python generator/generate_data.py --scale 3       # bigger dataset

Festival dates below are approximate. Adjust them if you want exact muhurats.
"""
import argparse
import hashlib
import os
import shutil
import time

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
START = pd.Timestamp("2023-10-01")          # first event date
END = pd.Timestamp("2026-09-30")            # last event date
ONBOARD_FROM = pd.Timestamp("2022-10-01")
ONBOARD_TO_MAIN = pd.Timestamp("2025-12-31")
ONBOARD_TO_NEW = pd.Timestamp("2026-06-30")  # recent joiners (cold-start vendors)

CITIES = ["Ahmedabad", "Surat", "Vadodara", "Rajkot", "Gandhinagar"]
CITY_DEMAND = np.array([1.0, 0.8, 0.5, 0.45, 0.3])
CITY_PRICE = np.array([1.0, 0.95, 0.85, 0.8, 0.9])
SYN_VENDOR_TARGET = {"Ahmedabad": 750, "Surat": 450, "Vadodara": 280,
                     "Rajkot": 220, "Gandhinagar": 150}

CATS = ["decor", "catering", "sound_dj", "photography",
        "mehendi", "venue", "lighting", "other"]
# synthetic vendor mix follows demand (sound/lighting are booked by most festival events)
CAT_SHARE = np.array([.17, .16, .15, .13, .05, .12, .13, .09])
# vendor price band per category (INR). catering = PER PLATE, the rest per event
PRICE_BAND = {"decor": (15000, 400000), "catering": (250, 1200),
              "sound_dj": (8000, 60000), "photography": (15000, 250000),
              "mehendi": (3000, 40000), "venue": (30000, 600000),
              "lighting": (8000, 120000), "other": (5000, 80000)}
CAPACITY = {"decor": (10, 30), "catering": (20, 60), "sound_dj": (15, 40),
            "photography": (12, 35), "mehendi": (20, 60), "venue": (15, 50),
            "lighting": (15, 40), "other": (10, 40)}
PEAK_PRESETS = {
    "wedding_cat": ["11,12,1,2", "11,12,1,2,4", "10,11,12,1,2"],
    "festival_cat": ["9,10,11,1", "9,10,11,12,1", "10,11,1,2"],
}

ETYPES = ["wedding", "navratri_society", "diwali", "birthday",
          "college_fest", "corporate", "uttarayan", "other"]
# probability that an event books each category (columns follow CATS)
BUNDLE = np.array([
    # decor catering sound photo mehendi venue light other
    [.95, .97, .75, .92, .60, .85, .60, .35],   # wedding
    [.55, .35, .97, .25, .02, .30, .85, .40],   # navratri_society
    [.70, .60, .40, .30, .10, .35, .75, .30],   # diwali
    [.80, .60, .45, .35, .03, .30, .30, .25],   # birthday
    [.50, .35, .90, .60, .02, .30, .85, .55],   # college_fest
    [.35, .80, .50, .50, .00, .55, .40, .45],   # corporate
    [.35, .60, .80, .15, .00, .20, .50, .45],   # uttarayan
    [.40, .40, .40, .40, .10, .40, .40, .40],   # other
])
BASE_RATE = dict(wedding=12, navratri_society=60, diwali=25, birthday=8,
                 college_fest=4, corporate=3, uttarayan=40, other=3)
GUESTS = dict(wedding=(300, .5), navratri_society=(400, .6), diwali=(80, .6),
              birthday=(40, .5), college_fest=(600, .6), corporate=(80, .6),
              uttarayan=(70, .6), other=(60, .6))              # median, sigma
PER_GUEST_BUDGET = dict(wedding=(3500, .4), navratri_society=(350, .4),
                        diwali=(900, .4), birthday=(1200, .4),
                        college_fest=(300, .4), corporate=(1800, .4),
                        uttarayan=(900, .4), other=(1000, .4))
LEAD_DAYS = dict(wedding=(45, 240), navratri_society=(20, 75), diwali=(10, 45),
                 birthday=(4, 30), college_fest=(15, 60), corporate=(7, 45),
                 uttarayan=(10, 40), other=(3, 30))
# customer type mix per event type: individual, housing_society, college, corporate
CTYPES = ["individual", "housing_society", "college", "corporate"]
CTYPE_MIX = dict(wedding=[1, 0, 0, 0], navratri_society=[0, .85, .10, .05],
                 diwali=[.5, .3, 0, .2], birthday=[1, 0, 0, 0],
                 college_fest=[0, 0, 1, 0], corporate=[0, 0, 0, 1],
                 uttarayan=[.4, .5, 0, .1], other=[.6, .2, 0, .2])
CUSTOMER_POOL_FACTOR = dict(individual=.8, housing_society=.35,
                            college=.25, corporate=.3)  # smaller pool = more repeat bookings
WEEKDAY = dict(  # Monday..Sunday multipliers
    wedding=[.8, .8, .8, .9, 1.3, 1.6, 1.5],
    birthday=[.6, .6, .6, .7, 1.0, 1.6, 1.5],
    college_fest=[1, 1, 1, 1, 1, .8, .5],
    corporate=[1.2, 1.2, 1.2, 1.2, 1.1, .4, .2],
    other=[.8, .8, .8, .9, 1.1, 1.3, 1.3])
MONTH_W = dict(
    wedding=[.9, 1.0, .5, .6, .7, .4, .15, .15, .2, .25, .9, 1.0],
    college_fest=[1.2, 1.4, 1.0, .3, .1, .05, .2, .8, 1.1, 1.2, .8, .6],
    corporate=[.8, .8, 1.2, .7, .7, .9, .7, .7, 1.1, .8, .9, 1.3])
NAVRATRI = [("2023-10-15", "2023-10-23"), ("2024-10-03", "2024-10-11"),
            ("2025-09-22", "2025-10-01"), ("2026-10-11", "2026-10-19")]
DIWALI = ["2023-11-12", "2024-11-01", "2025-10-20", "2026-11-08"]
UTTARAYAN = ["2024-01-14", "2025-01-14", "2026-01-14"]

CONVERSION = 0.72          # share of search intents that end in a booking
REVIEW_PROB = 0.35
ADMIN_ASSIGN_PROB = 0.12   # early-stage manual matching by the admin
TREND_PER_YEAR = 0.30      # marketplace growth


def save_parquet(df, path):
    """Spark cannot read nanosecond timestamps, so store them as microseconds."""
    df.to_parquet(path, index=False, coerce_timestamps="us", allow_truncated_timestamps=True)


def sha_id(prefix, text, n=12):
    return prefix + hashlib.sha1(text.encode()).hexdigest()[:n]


def logistic_cdf(z):
    return 1.0 / (1.0 + np.exp(-1.702 * z))


# ----------------------------------------------------------------------------
# Vendors
# ----------------------------------------------------------------------------
def build_vendors(rng, listed_path):
    real_parts = []
    if os.path.exists(listed_path):
        real = pd.read_csv(listed_path, dtype=str).fillna("")
        real = real[real["city"].isin(CITIES) & real["category"].isin(CATS)]
        real = real.drop_duplicates("vendor_id")
        if "manual_exclude" in real.columns:
            ex = (real["manual_exclude"] == "yes").sum()
            real = real[real["manual_exclude"] != "yes"]
            if ex:
                print(f"  skipped {ex} vendors you excluded by hand")
        if "likely_event_vendor" in real.columns:
            dropped = (real["likely_event_vendor"] == "no").sum()
            real = real[real["likely_event_vendor"] != "no"]
            print(f"  dropped {dropped} listings that are not event vendors (restaurants, shops, ...)")
        for _, r in real.iterrows():
            rc = r.get("google_review_count", "")
            real_parts.append(dict(
                vendor_id=r["vendor_id"], name=r["name"], category=r["category"],
                city=r["city"], area=r.get("area", ""),
                pop=min(float(rc) if rc not in ("", "nan") else 0.0, 300.0) + 3.0,
                source="listed"))
        print(f"  using {len(real_parts)} real listed vendors from {listed_path}")
    else:
        print(f"  {listed_path} not found: all vendors will be synthetic")

    rows = list(real_parts)
    real_cc = {}
    for r in real_parts:
        real_cc[(r["city"], r["category"])] = real_cc.get((r["city"], r["category"]), 0) + 1
    counter = 1
    for city, target in SYN_VENDOR_TARGET.items():
        added = 0
        for ci, cat in enumerate(CATS):
            # the Google directory is incomplete (top results only), so fill each
            # category up to the supply this city's demand needs, with synthetic vendors
            want = int(round(target * CAT_SHARE[ci]))
            need = max(0, want - real_cc.get((city, cat), 0))
            for _ in range(need):
                rows.append(dict(vendor_id=f"S{counter:04d}", name=f"Synthetic {cat} {counter}",
                                 category=cat, city=city, area="",
                                 pop=float(min(rng.pareto(1.3) + 1, 15.0) * 20), source="synthetic"))
                counter += 1
                added += 1
        print(f"  {city}: added {added} synthetic vendors to fill supply gaps")

    v = pd.DataFrame(rows).reset_index(drop=True)
    n = len(v)
    v["pop"] = v["pop"] ** 0.7 * np.exp(rng.normal(0, 0.4, n))

    # hidden behaviour traits (kept in a separate file, never in the vendors table)
    tier = rng.beta(2, 2, n)
    quality = rng.beta(5, 2.2, n)
    reliability = rng.beta(6, 1.6, n)
    resp_median = np.clip(np.exp(rng.normal(np.log(90), 0.7, n)), 5, 1500)
    resp_rate = rng.beta(7, 2, n)

    city_i = v["city"].map({c: i for i, c in enumerate(CITIES)}).to_numpy()
    pmin = np.zeros(n)
    pmax = np.zeros(n)
    cap = np.zeros(n, dtype=int)
    peak = []
    for i, cat in enumerate(v["category"]):
        lo, hi = PRICE_BAND[cat]
        base = lo + (hi - lo) * tier[i] * 0.45
        pmin[i] = base * CITY_PRICE[city_i[i]]
        pmax[i] = min(pmin[i] * (1.4 + 1.6 * rng.random()), hi * 1.2)
        cap[i] = rng.integers(*CAPACITY[cat])
        grp = "festival_cat" if cat in ("sound_dj", "lighting") else "wedding_cat"
        peak.append(rng.choice(PEAK_PRESETS[grp]))
    step = np.where(v["category"] == "catering", 10, 100)
    v["price_min"] = (np.round(pmin / step) * step).astype(int)
    v["price_max"] = (np.round(pmax / step) * step).astype(int)
    v["capacity_per_month"] = cap
    v["peak_months"] = peak
    v["years_active"] = np.clip(rng.gamma(2.0, 2.5, n).astype(int) + 1, 1, 25)

    new_vendor = rng.random(n) < 0.12
    days_main = (ONBOARD_TO_MAIN - ONBOARD_FROM).days
    days_new = (ONBOARD_TO_NEW - pd.Timestamp("2026-01-01")).days
    onboard = np.where(
        new_vendor,
        (pd.Timestamp("2026-01-01") + pd.to_timedelta(rng.integers(0, days_new, n), "D")).to_numpy(),
        (ONBOARD_FROM + pd.to_timedelta(rng.integers(0, days_main, n), "D")).to_numpy())
    v["onboarded_date"] = pd.to_datetime(onboard)
    v["verification_status"] = rng.choice(["verified", "pending", "rejected"], n,
                                          p=[.86, .09, .05])
    v["attrs_source"] = "simulated"

    hidden = pd.DataFrame(dict(vendor_id=v["vendor_id"], tier=tier, quality=quality,
                               reliability=reliability, resp_median_min=resp_median,
                               resp_rate=resp_rate, popularity=v["pop"]))
    return v, hidden


# ----------------------------------------------------------------------------
# Demand (search intents)
# ----------------------------------------------------------------------------
def season_curves(days):
    D = len(days)
    month = days.month.to_numpy()
    wd = days.weekday.to_numpy()
    S = {}
    S["wedding"] = np.array(MONTH_W["wedding"])[month - 1]
    S["college_fest"] = np.array(MONTH_W["college_fest"])[month - 1]
    S["corporate"] = np.array(MONTH_W["corporate"])[month - 1]
    nav = np.full(D, 0.01)
    for a, b in NAVRATRI:
        nav[(days >= a) & (days <= b)] = 1.0
    S["navratri_society"] = nav
    dw = np.full(D, 0.02)
    for d0 in DIWALI:
        rel = (days - pd.Timestamp(d0)).days.to_numpy()
        dw = np.maximum(dw, np.exp(-0.5 * (rel / 4.0) ** 2))
    S["diwali"] = dw
    ut = np.full(D, 0.01)
    for d0 in UTTARAYAN:
        rel = (days - pd.Timestamp(d0)).days.to_numpy()
        ut = np.maximum(ut, np.exp(-0.5 * (rel / 1.5) ** 2))
    S["uttarayan"] = ut
    S["birthday"] = np.ones(D)
    S["other"] = np.ones(D)
    for e in ETYPES:
        if e in WEEKDAY:
            S[e] = S[e] * np.array(WEEKDAY[e])[wd]
    return S


def build_intents(rng, scale):
    days = pd.date_range(START, END, freq="D")
    D, C = len(days), len(CITIES)
    S = season_curves(days)
    t_years = (days - START).days.to_numpy() / 365.25
    trend = 1.0 + TREND_PER_YEAR * t_years

    parts = []
    for ei, e in enumerate(ETYPES):
        lam = (BASE_RATE[e] * scale * S[e][:, None] * trend[:, None] * CITY_DEMAND[None, :])
        counts = rng.poisson(lam).ravel()
        idx = np.repeat(np.arange(D * C), counts)
        parts.append(pd.DataFrame(dict(day=idx // C, city=idx % C, etype=ei)))
    it = pd.concat(parts, ignore_index=True)
    it = it.sample(frac=1.0, random_state=int(rng.integers(1_000_000))).reset_index(drop=True)
    n = len(it)

    et = it["etype"].to_numpy()
    event_date = days[it["day"].to_numpy()]
    # guests, budget, lead time
    g_med = np.array([GUESTS[e][0] for e in ETYPES])[et]
    g_sig = np.array([GUESTS[e][1] for e in ETYPES])[et]
    guests = np.clip(np.round(g_med * np.exp(g_sig * rng.normal(size=n))), 10, 5000).astype(int)
    z = rng.normal(size=n)
    pg_med = np.array([PER_GUEST_BUDGET[e][0] for e in ETYPES])[et]
    pg_sig = np.array([PER_GUEST_BUDGET[e][1] for e in ETYPES])[et]
    budget = (np.round(guests * pg_med * np.exp(pg_sig * z) / 1000) * 1000).astype(int)
    btier = logistic_cdf(z)
    lo = np.array([LEAD_DAYS[e][0] for e in ETYPES])[et]
    hi = np.array([LEAD_DAYS[e][1] for e in ETYPES])[et]
    lead = (lo + (hi - lo) * rng.random(n)).astype(int)

    # customer type + hashed customer id
    ct = np.zeros(n, dtype=int)
    for ei, e in enumerate(ETYPES):
        m = et == ei
        ct[m] = rng.choice(len(CTYPES), size=m.sum(), p=CTYPE_MIX[e])
    city = it["city"].to_numpy()
    cust_id = np.empty(n, dtype=object)
    key = city * 10 + ct
    pool_rows = []
    for k in np.unique(key):
        m = np.where(key == k)[0]
        c, t = divmod(k, 10)
        pool = max(30, int(np.ceil(len(m) * CUSTOMER_POOL_FACTOR[CTYPES[t]])))
        pick = rng.integers(0, pool, len(m))
        ids = {p: sha_id("C", f"{CITIES[c]}-{CTYPES[t]}-{p}") for p in np.unique(pick)}
        cust_id[m] = [ids[p] for p in pick]
        for p, cid in ids.items():
            pool_rows.append((cid, CTYPES[t], CITIES[c]))
    customers = pd.DataFrame(pool_rows, columns=["customer_id", "customer_type", "city"])

    # timestamp of the booking decision: event date - lead days, with time of day
    hour_w = np.array([1, .5, .3, .2, .2, .4, 1, 2, 3, 4, 5, 6, 6, 5, 4, 4, 4, 5, 6, 8, 9, 9, 6, 3], float)
    hour = rng.choice(24, size=n, p=hour_w / hour_w.sum())
    tod = hour * 3600 + rng.integers(0, 3600, n)
    book_day = event_date - pd.to_timedelta(lead, "D")
    ts_ref = (book_day.to_numpy().astype("datetime64[s]").astype(np.int64) + tod)

    converted = rng.random(n) < CONVERSION
    intents = dict(n=n, etype=et, city=city, ctype=ct, cust_id=cust_id,
                   event_date=event_date.to_numpy(), guests=guests, budget=budget,
                   btier=btier, lead=lead, ts_ref=ts_ref, converted=converted)
    return intents, customers


# ----------------------------------------------------------------------------
# Leads: (intent, category) pairs + vendor choice
# ----------------------------------------------------------------------------
def build_leads(rng, I, V, hidden):
    n = I["n"]
    Pm = BUNDLE[I["etype"]]
    R = rng.random((n, len(CATS))) < Pm
    none = ~R.any(axis=1)
    R[none, np.argmax(Pm[none], axis=1)] = True
    # abandoned searches look at one category, picked in proportion to the bundle
    ab = np.where(~I["converted"])[0]
    cum = np.cumsum(Pm[ab], axis=1)
    pick = (rng.random(len(ab))[:, None] * cum[:, -1:] > cum).sum(axis=1)
    R[ab] = False
    R[ab, np.minimum(pick, len(CATS) - 1)] = True
    li, lc = np.nonzero(R)
    L = len(li)

    lead_city = I["city"][li]
    lead_booked = I["converted"][li]
    lead_ts = I["ts_ref"][li] + rng.integers(0, 4 * 3600, L)       # per-lead jitter
    lead_month = pd.to_datetime(lead_ts, unit="s")
    mkey = (lead_month.year * 12 + lead_month.month).to_numpy()

    v_city = V["city"].map({c: i for i, c in enumerate(CITIES)}).to_numpy()
    v_cat = V["category"].map({c: i for i, c in enumerate(CATS)}).to_numpy()
    v_onb = V["onboarded_date"].to_numpy().astype("datetime64[D]")
    v_ok = (V["verification_status"] == "verified").to_numpy()
    v_pop = V["pop"].to_numpy()
    v_tier = hidden["tier"].to_numpy()

    cand = np.full((L, 4), -1, dtype=np.int64)
    chosen_col = np.zeros(L, dtype=int)
    df = pd.DataFrame(dict(c=lead_city, k=lc, m=mkey))
    for (c, k, m), rows in df.groupby(["c", "k", "m"]).indices.items():
        rows = np.asarray(rows)
        month_start = np.datetime64(f"{(m - 1) // 12:04d}-{(m - 1) % 12 + 1:02d}-01")
        month_end = month_start + np.timedelta64(28, "D")
        base = np.where((v_city == c) & (v_cat == k) & v_ok)[0]
        if len(base) == 0:
            base = np.where((v_city == c) & (v_cat == k))[0]
        elig = base[v_onb[base] <= month_start]              # vendor already on the platform
        if len(elig) == 0:
            elig = base[v_onb[base] <= month_end]
        if len(elig) == 0:
            elig = base
        age = (month_end - v_onb[elig]).astype(int)
        w = v_pop[elig] * np.where(age < 90, 0.3, 1.0)        # new vendors get less exposure
        logw = np.log(w)
        mm = len(elig)
        nc = min(4, mm)
        g = -np.log(-np.log(rng.random((len(rows), mm)) + 1e-12) + 1e-12)
        score = logw[None, :] + g
        top = np.argsort(-score, axis=1)[:, :nc]
        cand[rows, :nc] = elig[top]

    c3 = cand[:, :3]
    valid = c3 >= 0
    aff = -np.abs(np.where(valid, v_tier[np.where(valid, c3, 0)], 9.0) - I["btier"][li][:, None])
    aff = aff + rng.normal(0, 0.15, aff.shape)
    aff = np.where(valid, aff, -np.inf)
    chosen_col = np.argmax(aff, axis=1)
    chosen = c3[np.arange(L), chosen_col]
    return dict(L=L, intent=li, cat=lc, booked=lead_booked, ts=lead_ts, cand=cand,
                chosen_col=chosen_col, chosen=chosen)


# ----------------------------------------------------------------------------
# Bookings, reviews, funnel events
# ----------------------------------------------------------------------------
def build_bookings(rng, I, V, hidden, LD):
    b = np.where(LD["booked"])[0]
    nb = len(b)
    it = LD["intent"][b]
    vend = LD["chosen"][b]
    cat = LD["cat"][b]
    rel = hidden["reliability"].to_numpy()[vend]
    med = hidden["resp_median_min"].to_numpy()[vend]
    tier = hidden["tier"].to_numpy()[vend]
    resp_min = np.clip(med * np.exp(rng.normal(0, 0.8, nb)), 2, 3000)

    # price: vendor band, position driven by the event's budget tier
    u = np.clip(I["btier"][it] * 0.7 + tier * 0.3 + rng.normal(0, 0.1, nb), 0, 1)
    pmin = V["price_min"].to_numpy()[vend].astype(float)
    pmax = V["price_max"].to_numpy()[vend].astype(float)
    price = pmin + (pmax - pmin) * u
    guests = I["guests"][it].astype(float)
    typical = np.array([GUESTS[e][0] for e in ETYPES])[I["etype"][it]].astype(float)
    is_cat = cat == CATS.index("catering")
    scale_f = np.where(is_cat, guests, np.clip((guests / typical) ** 0.4, 0.5, 2.0))
    amount = (np.round(price * scale_f / 100) * 100).astype(int)

    p_cancel = np.clip(0.05 + 0.22 * (1 - rel) + 0.03 * (resp_min > 360), 0, 0.6)
    cancelled = rng.random(nb) < p_cancel
    by_vendor = rng.random(nb) < (0.35 + 0.4 * (1 - rel))
    ts_book = LD["ts"][b]
    ev_ts = I["event_date"][it].astype("datetime64[s]").astype(np.int64)
    cancel_ts = ts_book + ((ev_ts - ts_book) * rng.uniform(0.1, 0.9, nb)).astype(np.int64)

    bookings = dict(n=nb, lead_row=b, intent=it, vendor=vend, cat=cat, amount=amount,
                    resp_min=resp_min, cancelled=cancelled, by_vendor=by_vendor,
                    ts_book=ts_book, cancel_ts=cancel_ts)

    # reviews on completed bookings
    done = np.where(~cancelled)[0]
    has_rev = done[rng.random(len(done)) < REVIEW_PROB]
    q = hidden["quality"].to_numpy()[vend[has_rev]]
    rating = np.clip(np.round(rng.normal(2.2 + 2.6 * q, 0.8)), 1, 5).astype(int)
    rev_ts = ev_ts[has_rev] + rng.integers(1, 6, len(has_rev)) * 86400 + rng.integers(0, 80000, len(has_rev))
    reviews = dict(booking=has_rev, rating=rating, ts=rev_ts)
    return bookings, reviews


def build_funnel(rng, I, V, hidden, LD, B):
    """Compact arrays describing every platform action."""
    L = LD["L"]
    cand = LD["cand"]
    c3 = cand[:, :3]
    valid = c3 >= 0
    safe = np.where(valid, c3, 0)
    booked = LD["booked"]
    cc = LD["chosen_col"]
    ar = np.arange(L)

    p_other = np.where(booked, 0.55, 0.35)[:, None]
    chosen_mask = np.zeros((L, 3), dtype=bool)
    chosen_mask[ar, cc] = True
    other_req = (rng.random((L, 3)) < p_other) & valid & ~chosen_mask
    chosen_req = chosen_mask & (booked[:, None] | (rng.random((L, 3)) < 0.75))
    req = (other_req | chosen_req) & valid

    rr = hidden["resp_rate"].to_numpy()[safe]
    resp = (rng.random((L, 3)) < rr) & req
    resp = np.where(booked[:, None] & chosen_mask, req, resp)
    med = hidden["resp_median_min"].to_numpy()[safe]
    resp_min = np.clip(med * np.exp(rng.normal(0, 0.8, (L, 3))), 2, 3000)
    # use the booking's own response time for the booked vendor so the tables agree
    bk = B["lead_row"]
    bcol = cc[bk]
    resp_min[bk, bcol] = B["resp_min"]

    delay = rng.exponential(600, (L, 3)) + 5            # minutes before the booking
    q_ts = LD["ts"][:, None] - ((delay + resp_min) * 60).astype(np.int64)
    r_ts = q_ts + (resp_min * 60).astype(np.int64)
    earliest = np.where(valid, q_ts, LD["ts"][:, None]).min(axis=1)
    v_ts = earliest - rng.integers(60, 5400, L)

    # --- collect rows -------------------------------------------------------
    A = {n: i for i, n in enumerate(
        ["search", "vendor_view", "quote_request", "quote_response",
         "booking_confirmed", "booking_cancelled", "review_submitted",
         "booking_assigned", "availability_update", "vendor_verified"])}
    ACTOR = {"customer": 0, "vendor": 1, "admin": 2}
    ts_l, act_l, actor_l, intent_l, vend_l = [], [], [], [], []

    def add(ts, act, actor, intent, vend):
        ts_l.append(np.asarray(ts, dtype=np.int64))
        act_l.append(np.full(len(ts), A[act], dtype=np.int8))
        actor_l.append(np.full(len(ts), ACTOR[actor], dtype=np.int8))
        intent_l.append(np.asarray(intent, dtype=np.int64))
        vend_l.append(np.asarray(vend, dtype=np.int64))

    li = LD["intent"]
    # searches: per intent, before its earliest view
    first_view = pd.Series(v_ts).groupby(li).min()
    n_search = 1 + rng.poisson(0.6, len(first_view))
    s_int = np.repeat(first_view.index.to_numpy(), n_search)
    s_ts = np.repeat(first_view.to_numpy(), n_search) - rng.integers(300, 7200, len(s_int))
    add(s_ts, "search", "customer", s_int, np.full(len(s_int), -1))
    # views of up to 4 candidates
    for j in range(4):
        m = cand[:, j] >= 0
        add(v_ts[m] + rng.integers(0, 600, m.sum()), "vendor_view", "customer",
            li[m], cand[m, j])
    for j in range(3):
        m = req[:, j]
        add(q_ts[m, j], "quote_request", "customer", li[m], c3[m, j])
        m2 = resp[:, j]
        add(r_ts[m2, j], "quote_response", "vendor", li[m2], c3[m2, j])
    # booking outcome
    bi, bv = B["intent"], B["vendor"]
    add(B["ts_book"], "booking_confirmed", "customer", bi, bv)
    cx = np.where(B["cancelled"])[0]
    cancel_actor_vendor = B["by_vendor"][cx]
    for flag, actor in ((True, "vendor"), (False, "customer")):
        m = cx[cancel_actor_vendor == flag]
        add(B["cancel_ts"][m], "booking_cancelled", actor, bi[m], bv[m])
    admin = np.where(rng.random(B["n"]) < ADMIN_ASSIGN_PROB)[0]
    add(B["ts_book"][admin] - 60, "booking_assigned", "admin", bi[admin], bv[admin])

    return dict(ts=ts_l, act=act_l, actor=actor_l, intent=intent_l, vend=vend_l,
                A=A, ACTOR=ACTOR, add=add)


# ----------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------
def ts_to_dt(a):
    return pd.to_datetime(np.asarray(a, dtype=np.int64), unit="s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=1.0, help="demand multiplier")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=os.path.join("data", "synthetic"))
    ap.add_argument("--vendors-file", default=os.path.join("data", "private", "vendors_listed.csv"))
    args = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(args.seed)

    print("1/6 building vendors")
    V, hidden = build_vendors(rng, args.vendors_file)
    print(f"  {len(V)} vendors")

    print("2/6 building demand (search intents)")
    I, customers = build_intents(rng, args.scale)
    print(f"  {I['n']:,} search intents, {len(customers):,} customers")

    print("3/6 choosing vendors")
    LD = build_leads(rng, I, V, hidden)
    print(f"  {LD['L']:,} vendor leads")

    print("4/6 bookings, reviews")
    B, RV = build_bookings(rng, I, V, hidden, LD)
    print(f"  {B['n']:,} bookings, {len(RV['rating']):,} reviews")

    # no vendor may be booked before it joined: pull the join date back if needed
    first_book = pd.Series(B["ts_book"]).groupby(B["vendor"]).min()
    fb = (pd.to_datetime(first_book.to_numpy(), unit="s").normalize() - pd.Timedelta(days=3))
    cur = pd.to_datetime(V.loc[first_book.index, "onboarded_date"].to_numpy())
    V.loc[first_book.index, "onboarded_date"] = np.minimum(cur.to_numpy("datetime64[ns]"),
                                                          fb.to_numpy("datetime64[ns]"))
    V["onboarded_date"] = pd.to_datetime(V["onboarded_date"])

    print("5/6 platform events")
    F = build_funnel(rng, I, V, hidden, LD, B)
    add = F["add"]
    # reviews
    add(RV["ts"], "review_submitted", "customer", B["intent"][RV["booking"]], B["vendor"][RV["booking"]])
    # vendor side and admin side
    ver = np.where(V["verification_status"] == "verified")[0]
    onb = V["onboarded_date"].to_numpy().astype("datetime64[s]").astype(np.int64)
    add(onb[ver] + 9 * 3600, "vendor_verified", "admin", np.full(len(ver), -1), ver)
    months = pd.date_range(ONBOARD_FROM, END, freq="MS")
    for vi in ver:
        k = rng.poisson(0.3 * len(months[months >= V["onboarded_date"].iloc[vi]]))
        if k:
            lo_t, hi_t = onb[vi], int(END.value // 10**9)
            add(rng.integers(lo_t, hi_t, k), "availability_update", "vendor",
                np.full(k, -1), np.full(k, vi))

    ts = np.concatenate(F["ts"])
    act = np.concatenate(F["act"])
    actor = np.concatenate(F["actor"])
    intent = np.concatenate(F["intent"])
    vend = np.concatenate(F["vend"])
    order = np.argsort(ts, kind="stable")
    ts, act, actor, intent, vend = ts[order], act[order], actor[order], intent[order], vend[order]
    print(f"  {len(ts):,} events_log rows")

    print("6/6 writing files to", args.out)
    if os.path.exists(args.out):
        for f in os.listdir(args.out):
            p = os.path.join(args.out, f)
            if f == ".gitkeep":
                continue
            shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
    os.makedirs(os.path.join(args.out, "events_log"), exist_ok=True)

    # --- reviews first, so vendors get their platform rating ----------------
    bk_ids = np.array([f"B{i + 1:07d}" for i in range(B["n"])])
    rv_df = pd.DataFrame(dict(
        review_id=[f"R{i + 1:07d}" for i in range(len(RV["rating"]))],
        booking_id=bk_ids[RV["booking"]],
        vendor_id=V["vendor_id"].to_numpy()[B["vendor"][RV["booking"]]],
        rating=RV["rating"], review_text="",
        timestamp=ts_to_dt(RV["ts"]), source="synthetic"))
    r_mean = rv_df.groupby("vendor_id")["rating"].mean()

    # --- vendors -------------------------------------------------------------
    vend_out = V.copy()
    vend_out["rating_avg"] = vend_out["vendor_id"].map(r_mean).fillna(0).round(2)
    vend_out["onboarded_date"] = vend_out["onboarded_date"].dt.date
    vend_out = vend_out[["vendor_id", "name", "category", "city", "area", "price_min",
                         "price_max", "capacity_per_month", "peak_months", "years_active",
                         "rating_avg", "onboarded_date", "verification_status", "source",
                         "attrs_source"]]
    save_parquet(vend_out, os.path.join(args.out, "vendors.parquet"))
    save_parquet(hidden, os.path.join(args.out, "vendors_hidden_truth.parquet"))
    customers["source"] = "synthetic"
    save_parquet(customers, os.path.join(args.out, "customers.parquet"))

    # --- bookings ------------------------------------------------------------
    it = B["intent"]
    ev_date = pd.to_datetime(I["event_date"][it])
    bk_dt = ts_to_dt(B["ts_book"])
    status = np.where(B["cancelled"], "cancelled", "completed")
    cancelled_by = np.where(B["cancelled"], np.where(B["by_vendor"], "vendor", "customer"), "")
    bk_df = pd.DataFrame(dict(
        booking_id=bk_ids,
        event_instance_id=[f"E{i + 1:07d}" for i in it],
        customer_id=I["cust_id"][it],
        vendor_id=V["vendor_id"].to_numpy()[B["vendor"]],
        event_type=np.array(ETYPES)[I["etype"][it]],
        city=np.array(CITIES)[I["city"][it]],
        guest_count=I["guests"][it],
        booking_date=bk_dt.normalize().date,
        event_date=ev_date.date,
        amount=B["amount"], status=status, cancelled_by=cancelled_by,
        response_time_minutes=np.round(B["resp_min"]).astype(int),
        category=np.array(CATS)[B["cat"]],      # extra convenience column
        source="synthetic"))
    save_parquet(bk_df, os.path.join(args.out, "bookings.parquet"))
    save_parquet(rv_df, os.path.join(args.out, "reviews.parquet"))

    # --- events_log, one part file per month --------------------------------
    action_names = np.array(list(F["A"].keys()))
    actor_names = np.array(list(F["ACTOR"].keys()))
    vid = V["vendor_id"].to_numpy()
    has_int = intent >= 0
    safe_int = np.where(has_int, intent, 0)
    session = np.where(has_int, np.char.add("S", np.char.zfill(safe_int.astype(str), 7)), "")
    cust = np.where(has_int, I["cust_id"][safe_int], "")
    city_arr = np.where(has_int, np.array(CITIES)[I["city"][safe_int]],
                        np.where(vend >= 0, V["city"].to_numpy()[np.where(vend >= 0, vend, 0)], ""))
    et_arr = np.where(has_int, np.array(ETYPES)[I["etype"][safe_int]], "")
    use_budget = np.isin(act, [F["A"]["search"], F["A"]["quote_request"]]) & has_int
    budget = np.where(use_budget, I["budget"][safe_int], 0)
    guests = np.where(has_int, I["guests"][safe_int], 0)
    ev_arr = np.where(has_int, I["event_date"][safe_int].astype("datetime64[D]"),
                      np.datetime64("NaT"))
    vendor_col = np.where(vend >= 0, vid[np.where(vend >= 0, vend, 0)], "")
    ts_dt = ts_to_dt(ts)
    month_key = (ts_dt.year * 100 + ts_dt.month).to_numpy()

    log_id_counter = 0
    total_rows = 0
    for mk in np.unique(month_key):
        m = np.where(month_key == mk)[0]
        part = pd.DataFrame(dict(
            log_id=[f"L{log_id_counter + i + 1:09d}" for i in range(len(m))],
            timestamp=ts_dt[m],
            action=action_names[act[m]],
            actor_role=actor_names[actor[m]],
            session_id=session[m], customer_id=cust[m], vendor_id=vendor_col[m],
            city=city_arr[m], event_type=et_arr[m],
            budget=budget[m].astype("int64"), guest_count=guests[m].astype("int64"),
            event_date=pd.to_datetime(ev_arr[m]), source="synthetic"))
        log_id_counter += len(m)
        total_rows += len(m)
        save_parquet(part, os.path.join(args.out, "events_log", f"part-{mk}.parquet"))

    # --- small CSV samples to eyeball in Excel -------------------------------
    bk_df.head(500).to_csv(os.path.join(args.out, "sample_bookings.csv"), index=False)
    vend_out.head(500).to_csv(os.path.join(args.out, "sample_vendors.csv"), index=False)
    pd.read_parquet(os.path.join(args.out, "events_log", f"part-{np.unique(month_key)[len(np.unique(month_key)) // 2]}.parquet")
                    ).head(1000).to_csv(os.path.join(args.out, "sample_events_log.csv"), index=False)

    # --- summary --------------------------------------------------------------
    print("\n=== SUMMARY ===")
    print(f"vendors   {len(vend_out):>10,}")
    print(f"customers {len(customers):>10,}")
    print(f"bookings  {len(bk_df):>10,}")
    print(f"reviews   {len(rv_df):>10,}")
    print(f"events_log{total_rows:>10,}")
    print(f"cancellation rate: {B['cancelled'].mean():.1%}")
    print("bookings by event type:\n" + bk_df.groupby("event_type").size().to_string())
    bk_df["month"] = pd.to_datetime(bk_df["event_date"]).dt.to_period("M")
    top = bk_df.groupby("month").size().sort_values(ascending=False).head(5)
    print("busiest event months:\n" + top.to_string())
    share = bk_df["vendor_id"].value_counts()
    k = max(1, int(0.1 * len(share)))
    print(f"top 10% of vendors take {share.head(k).sum() / share.sum():.0%} of bookings")
    print(f"vendors with zero bookings: {len(vend_out) - len(share)}")
    vm = (bk_df.groupby(["vendor_id", "month"]).size().rename("n").reset_index()
          .merge(vend_out[["vendor_id", "capacity_per_month"]], on="vendor_id"))
    print(f"busiest vendor: {share.iloc[0]:,} bookings in 36 months "
          f"(about {share.iloc[0] / 36:.0f} a month)")
    print(f"vendor-months above stated capacity: {(vm['n'] > vm['capacity_per_month']).mean():.1%}")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()