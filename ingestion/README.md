# Ingestion: building the first vendor list

1. In Apify, open "Google Maps Scraper", switch Input to JSON, paste one file from apify_inputs/ (start with ahmedabad.json), run it.
2. Export the dataset as CSV and save it in data/private/raw_listings/ (e.g. ahmedabad.csv).
3. From the project root, with the venv active: python ingestion/clean_listings.py
4. Open data/private/vendors_listed.csv.
Repeat for the other cities. Re-running keeps existing vendor_ids and your notes.

data/private/ is git-ignored because it contains phone numbers.
