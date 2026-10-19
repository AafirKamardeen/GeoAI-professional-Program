GeoAI Professional Program — Day 5 Practice Dataset
=====================================================

Project:
Automated Geospatial Data Organizer

Purpose:
This is a synthetic training dataset designed for Day 5.
It imitates environmental/satellite monitoring observations from
multiple Indian study regions.

Input files:
- sensor_a_environmental_data.csv
- sensor_b_environmental_data.csv
- sensor_c_environmental_data.csv

Columns:
record_id     Unique observation ID
date          Observation date
station       Monitoring station ID
city          Study city
latitude      Latitude in decimal degrees
longitude     Longitude in decimal degrees
ndvi          Synthetic NDVI value
temperature   Synthetic temperature in °C
rainfall      Synthetic rainfall in mm
region        North / South / East / West
sensor        Sensor source

Intentional data-quality problems:
1. Missing latitude/longitude values
2. Invalid latitude/longitude values
3. Missing NDVI/temperature values
4. Duplicate records
5. Unsorted records

Suggested Day 5 workflow:
1. Find all CSV files with pathlib
2. Read each CSV
3. Count total records
4. Detect missing coordinates
5. Validate latitude (-90 to 90)
6. Validate longitude (-180 to 180)
7. Detect duplicate records
8. Organize valid data by region
9. Optionally organize by year/month
10. Create a JSON metadata summary

Important:
This is synthetic educational data, not real satellite observations.
Do not use it as scientific ground truth.