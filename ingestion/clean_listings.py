"""
Turn raw Apify "Google Maps Scraper" CSV exports into the Ocazionally vendor list.

Usage (from the project root, venv active):
    python ingestion/clean_listings.py

Input : every CSV in data/private/raw_listings/
Output: data/private/vendors_listed.csv  (kept out of Git: it holds phone numbers)

Re-running is safe: vendors already in the output keep their vendor_id and any
notes you typed; only new places are appended.
"""
import os
import re
import sys
import glob
import pandas as pd

RAW_DIR = os.path.join("data", "private", "raw_listings")
OUT_FILE = os.path.join("data", "private", "vendors_listed.csv")

# search term -> schema category
CATEGORY_BY_SEARCH = {
    "wedding decorator": "decor",
    "event decorator": "decor",
    "mandap decorator": "decor",
    "catering service": "catering",
    "wedding caterer": "catering",
    "dj sound system rental": "sound_dj",
    "dj for wedding": "sound_dj",
    "wedding photographer": "photography",
    "event photographer": "photography",
    "mehendi artist": "mehendi",
    "party plot": "venue",
    "banquet hall": "venue",
    "event lighting": "lighting",
    "tent house": "other",
    "event management company": "other",
}

# fallback: guess from Google's own category label
CATEGORY_KEYWORDS = [
    ("decor", ["decorat", "florist", "flower"]),
    ("catering", ["cater", "food"]),
    ("sound_dj", ["dj", "sound", "music", "disc jockey"]),
    ("photography", ["photo", "video"]),
    ("mehendi", ["mehendi", "mehndi", "henna"]),
    ("venue", ["banquet", "party", "wedding venue", "event venue", "hall", "plot", "resort"]),
    ("lighting", ["light"]),
]

OUT_COLUMNS = [
    "vendor_id", "name", "category", "city", "area",
    "price_min", "price_max", "capacity_per_month", "peak_months", "years_active",
    "rating_avg", "onboarded_date", "verification_status", "source",
    "contact_phone", "website", "google_maps_url", "google_place_id",
    "google_rating", "google_review_count", "has_phone",
    "outreach_status", "open_to_pilot", "how_they_get_clients",
    "would_pay_for_leads", "cancellation_habits", "notes",
]


def col(df, *names):
    """First matching column as strings, or an empty column."""
    for n in names:
        if n in df.columns:
            return df[n].fillna("").astype(str)
    return pd.Series([""] * len(df), index=df.index)


def normalize_phone(raw):
    digits = re.sub(r"\D", "", str(raw))
    if len(digits) == 10:
        return "+91" + digits
    if len(digits) == 11 and digits.startswith("0"):
        return "+91" + digits[1:]
    if len(digits) == 12 and digits.startswith("91"):
        return "+" + digits
    return ""


def guess_category(search, google_cat):
    s = str(search).strip().lower()
    if s in CATEGORY_BY_SEARCH:
        return CATEGORY_BY_SEARCH[s]
    # area searches look like "wedding decorator in Bodakdev, Ahmedabad"
    for key, cat in CATEGORY_BY_SEARCH.items():
        if s.startswith(key + " in "):
            return cat
    g = str(google_cat).lower()
    for cat, words in CATEGORY_KEYWORDS:
        if any(w in g for w in words):
            return cat
    return "other"


KNOWN_CITIES = ["Ahmedabad", "Gandhinagar", "Surat", "Vadodara", "Rajkot"]


def normalize_city(value):
    """Google sometimes returns 'chandannagar, Ahmedabad' or "A'BAD, Ahmedabad"."""
    v = str(value).strip()
    low = v.lower()
    for c in KNOWN_CITIES:
        if c.lower() in low:
            return c
    return v


def guess_area(neighborhood, address):
    if str(neighborhood).strip():
        # Google often gives "Sub-area, Area, Locality": keep the broadest (last) part
        return str(neighborhood).split(",")[-1].strip()
    parts = [p.strip() for p in str(address).split(",") if p.strip()]
    if len(parts) >= 3:
        return parts[-3]
    return parts[0] if parts else ""


def load_raw():
    files = sorted(glob.glob(os.path.join(RAW_DIR, "*.csv")))
    if not files:
        sys.exit(f"No CSV files found in {RAW_DIR}. Put your Apify exports there first.")
    frames = []
    for f in files:
        df = pd.read_csv(f, dtype=str)
        df["_file"] = os.path.basename(f)
        frames.append(df)
        print(f"Read {len(df):4d} rows from {os.path.basename(f)}")
    return pd.concat(frames, ignore_index=True)


def build_new_rows(raw):
    raw = raw.copy()

    # drop closed places and non-Gujarat results
    for flag in ("permanentlyClosed", "temporarilyClosed"):
        if flag in raw.columns:
            raw = raw[~raw[flag].fillna("").astype(str).str.lower().isin(["true", "1"])]
    if "state" in raw.columns:
        st = raw["state"].fillna("").astype(str).str.strip().str.lower()
        raw = raw[(st == "gujarat") | (st == "")]

    out = pd.DataFrame(index=raw.index)
    out["name"] = col(raw, "title", "name").str.strip()
    google_cat = col(raw, "categoryName", "category")
    search = col(raw, "searchString", "searchTerm")
    out["category"] = [guess_category(s, g) for s, g in zip(search, google_cat)]
    city = col(raw, "city").str.strip()
    # blank city (a few Google listings lack one): use the file name, e.g. ahmedabad.csv
    file_city = raw["_file"].str.replace(".csv", "", regex=False).str.split("_").str[0].str.title()
    out["city"] = city.where(city != "", file_city)
    out["area"] = [guess_area(n, a) for n, a in
                   zip(col(raw, "neighborhood"), col(raw, "address"))]
    out["contact_phone"] = col(raw, "phone", "phoneUnformatted").map(normalize_phone)
    out["website"] = col(raw, "website")
    out["google_maps_url"] = col(raw, "url")
    out["google_place_id"] = col(raw, "placeId")
    out["google_rating"] = col(raw, "totalScore")
    out["google_review_count"] = col(raw, "reviewsCount")

    out = out[out["name"] != ""]

    # de-duplicate: same Google place, then same name + phone
    has_id = out["google_place_id"] != ""
    out = pd.concat([
        out[has_id].drop_duplicates("google_place_id"),
        out[~has_id],
    ])
    out = out.drop_duplicates(["name", "contact_phone"])

    out["has_phone"] = (out["contact_phone"] != "").map({True: "yes", False: "no"})
    out["rating_avg"] = 0          # platform rating: none yet
    out["verification_status"] = "pending"
    out["source"] = "listed"
    out["outreach_status"] = "not_contacted"
    return out.reset_index(drop=True)


def main():
    raw = load_raw()
    new = build_new_rows(raw)

    if os.path.exists(OUT_FILE):
        old = pd.read_csv(OUT_FILE, dtype=str).fillna("")
        known = set(old["google_place_id"]) - {""}
        known_np = set(zip(old["name"], old["contact_phone"]))
        keep = [
            (pid == "" or pid not in known) and (n, p) not in known_np
            for pid, n, p in zip(new["google_place_id"], new["name"], new["contact_phone"])
        ]
        new = new[keep].reset_index(drop=True)
        next_num = max([int(v[1:]) for v in old["vendor_id"] if re.fullmatch(r"V\d+", v)] + [0]) + 1
        print(f"Existing list: {len(old)} vendors. New this run: {len(new)}")
    else:
        old = pd.DataFrame(columns=OUT_COLUMNS)
        next_num = 1

    new = new.sort_values(["city", "category", "name"]).reset_index(drop=True)
    new["vendor_id"] = [f"V{next_num + i:04d}" for i in range(len(new))]
    for c in OUT_COLUMNS:
        if c not in new.columns:
            new[c] = ""
    final = pd.concat([old, new[OUT_COLUMNS]], ignore_index=True)[OUT_COLUMNS]

    final["city"] = final["city"].map(normalize_city)

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    final.to_csv(OUT_FILE, index=False, encoding="utf-8-sig")

    # keep phone numbers out of Git
    gi = ".gitignore"
    line = "data/private/"
    existing = open(gi).read() if os.path.exists(gi) else ""
    if line not in existing:
        with open(gi, "a") as f:
            f.write(("\n" if existing and not existing.endswith("\n") else "") + line + "\n")
        print("Added data/private/ to .gitignore")

    print(f"\nSaved {len(final)} vendors to {OUT_FILE}\n")
    print("Vendors per city and category:")
    print(pd.crosstab(final["city"], final["category"], margins=True).to_string())
    print(f"\nWith a phone number: {(final['has_phone'] == 'yes').mean():.0%}")
    print("\nTop areas per city (check that coverage is not stuck in one part of town):")
    for city, grp in final.groupby("city"):
        top = grp["area"].replace("", "(blank)").value_counts().head(12)
        print(f"  {city}: " + ", ".join(f"{a} {n}" for a, n in top.items()))


if __name__ == "__main__":
    main()