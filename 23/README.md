# Drought Monitoring & Planning: Karnataka

**GeoAI Professional Program · Folium & Leaflet · requested by Basavaraj's batch**
Monsoon 2026 (Jun–Sep) vs 1991–2025 baseline · 27 districts

## Contents
```
Drought_Project/
├── Drought_Student.ipynb                 ← students fill in the ____ blanks
├── Drought_Solution.ipynb                ← complete working version (SPI, VCI, planning, map)
├── drought_solution.py                   ← same solution as a plain script
├── Karnataka_Drought_Monitor.html        ← interactive map (open in any browser)
├── drought_planning_table_sep2026.csv    ← planning table: priority score and action per district
└── data/
    ├── karnataka_districts.geojson       27 REAL district polygons (2011 Census boundaries)
    ├── rainfall_monthly_1991_2026.csv    monthly rainfall per district
    ├── ndvi_september_2001_2026.csv      September NDVI per district
    ├── groundwater_postmonsoon.csv       post-monsoon depth to water, 2025 & 2026
    └── reservoirs.csv                    5 major reservoirs, storage 2026 vs 2025
```

## Method
| Step | What students learn |
|---|---|
| 2 | **SPI-3**: rolling sums, gamma fit per calendar month (`scipy.stats.gamma`), CDF → z-score |
| 3 | Drought classes (McKee SPI thresholds) |
| 4 | **VCI** from NDVI: vegetation stress against the best and worst years |
| 5 | Groundwater change and a combined district table (`merge`) |
| 6 | **Priority score** (SPI 50%, VCI 30%, groundwater 20%) and planning actions |
| 7 | Export a planning table for officers |
| 8 | Folium map: choropleth, toggle layers, SVG trend charts in popups, priority markers, legend |

## Key results (synthetic 2026 scenario, check student output against these)
- **18 of 27 districts** have SPI-3 ≤ −1. Ballari (−3.1), Chitradurga (−2.8), Vijayapura (−2.6) are the worst.
- **15 districts** need a "Declare drought" response. The north interior (Ballari, Vijayapura, Koppal, Raichur, Kalaburagi, Bidar, Bagalkot) and parts of the south (Bengaluru Rural, Tumakuru, Kolar) are most affected.
- Rainfall for June–September averages about 60% of normal. The malnad and coastal districts are closest to normal.
- Post-monsoon groundwater is deeper by up to 3–4 m in the north interior.
- Tungabhadra (30%) and Almatti (35%) reservoirs are the lowest in storage.

## Important caveats (say these in class)
1. **Boundaries:** the 27 polygons are the 2011 districts. Yadgir, Chikkaballapur, Ramanagara and Vijayanagara were created later and are missing. Use the current official shapefile for real work.
2. **Data:** rainfall, NDVI, groundwater and reservoirs are **synthetic**, built to mimic a deficient monsoon. Use IMD gridded rainfall, MODIS or Sentinel-2 NDVI, CGWB and KSNDMC data for a real assessment.
3. **Small-sample extremes:** SPI can be extreme in one district by chance, especially with short or noisy records. Udupi, a coastal district that normally gets plenty of rain, shows an extreme SPI (−2.4) because of the noise in its own history. Kodagu's VCI of 100 is the opposite extreme: its vegetation is at its best on record. Discuss this in class.
4. **Planning thresholds** (score ≥ 60 = declare) are for teaching. Drought declarations in India follow state and central guidelines.

## How to run
1. Upload the `Drought_Project` folder to JupyterLab or Colab (keep `data/` beside the notebook).
2. `pip install folium geopandas scipy` if needed.
3. Run top to bottom. The last cell saves `Karnataka_Drought_Monitor.html`.

## Extension challenges
- Replace the rainfall CSV with IMD data and rerun Step 2 unchanged.
- Animate SPI-3 month by month with `plugins.TimestampedGeoJson`.
- Split the four missing districts from the current boundaries.
- Change the weights in Step 6 to 40/40/20. Which districts move up the list?
- Add a crop rule that recommends millets or pulses for severe Kharif VCI.
