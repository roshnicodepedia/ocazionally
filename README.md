# Ocazionally

Event vendor marketplace, with a data and intelligence layer (the Ocazionally Intelligence Engine).

## Folders
- docs/        pitch material, schema, vendor pitch script
- data/real/   vendor sheet and real bookings (never mix with synthetic)
- data/synthetic/  generated Parquet data (git-ignored)
- generator/   synthetic data generator, calibrated from data/real
- pipeline/    Spark ETL and streaming jobs
- models/      demand forecasting, vendor clustering, FP-Growth, fairness check
- storage/     MongoDB schemas, sharding and replication notes
- dashboard/   Streamlit app
- benchmarks/  MapReduce-style (RDD) vs DataFrame comparison
- report/      course report, ethics section, architecture diagram

## Setup
See docs/setup_checklist.md
