# !pip install folium geopandas scipy   # uncomment in Colab if needed
import numpy as np
import pandas as pd
import geopandas as gpd
import folium
from folium import plugins
import branca
import branca.colormap as cm
from scipy import stats

DATA = "data/"
print("folium", folium.__version__, "| geopandas", gpd.__version__)

dist  = gpd.read_file(DATA + "karnataka_districts.geojson")
rain  = pd.read_csv(DATA + "rainfall_monthly_1991_2026.csv", parse_dates=["month"])
ndvi  = pd.read_csv(DATA + "ndvi_september_2001_2026.csv")
gw    = pd.read_csv(DATA + "groundwater_postmonsoon.csv")
res   = pd.read_csv(DATA + "reservoirs.csv")

print(rain.head(3))
print("\nDistricts:", dist.district.nunique(), "| months per district:", rain.groupby("district").size().iloc[0])

# 3-month rolling totals for each district
rain = rain.sort_values(["district", "month"]).reset_index(drop=True)
rain["spi_accum"] = rain.groupby("district").rainfall_mm.transform(lambda s: s.rolling(3).sum())
rain = rain.dropna(subset=["spi_accum"])
rain["cal_month"] = rain.month.dt.month
rain["year"] = rain.month.dt.year

def compute_spi(df, base_end=2025):
    out = []
    for (d, m), grp in df.groupby(["district", "cal_month"]):
        base = grp[grp.year <= base_end].spi_accum
        a, _, b = stats.gamma.fit(base, floc=0)          # shape, location (fixed at 0), scale
        test = grp[grp.year > base_end].copy()           # 2026 values
        p = stats.gamma.cdf(test.spi_accum, a, scale=b)
        test["spi"] = stats.norm.ppf(np.clip(p, 0.001, 0.999))
        out.append(test)
    return pd.concat(out).sort_values(["district", "month"])

spi = compute_spi(rain)
spi_2026 = spi[spi.month.dt.year == 2026][["district", "month", "spi"]].round(2)
spi_sep = spi_2026[spi_2026.month.dt.month == 9].set_index("district").spi
print("SPI-3 at end of September 2026:")
print(spi_sep.sort_values().round(2).to_string())

def spi_class(v):
    if v <= -2:    return "Extreme drought"
    if v <= -1.5:  return "Severe drought"
    if v <= -1:    return "Moderate drought"
    if v < 1:      return "Normal"
    return "Wet"

spi_summary = spi_sep.apply(spi_class).value_counts().reindex(
    ["Extreme drought", "Severe drought", "Moderate drought", "Normal", "Wet"]).fillna(0).astype(int)
print(spi_summary.to_string())
print(f"\nDistricts in drought (SPI ≤ -1): {(spi_sep <= -1).sum()} of {len(spi_sep)}")

hist = ndvi[ndvi.year <= 2025].groupby("district").ndvi_sep.agg(["min", "max"]).rename(
    columns={"min": "ndvi_min", "max": "ndvi_max"})
now = ndvi[ndvi.year == 2026].set_index("district").ndvi_sep.rename("ndvi_2026")
vci = hist.join(now)
vci["vci"] = ((vci.ndvi_2026 - vci.ndvi_min) / (vci.ndvi_max - vci.ndvi_min) * 100).clip(0, 100).round(1)
print(vci[["ndvi_2026", "vci"]].sort_values("vci").round(3).to_string())

gw["gw_rise_m"] = (gw.postmonsoon_depth_2026_m - gw.postmonsoon_depth_2025_m).round(2)

table = (dist[["district", "region", "centroid_lat", "centroid_lon", "geometry"]]
         .merge(spi_sep.rename("spi_sep").reset_index(), on="district")
         .merge(vci[["vci"]].reset_index(), on="district")
         .merge(gw, on="district"))
table["spi_class"] = table.spi_sep.apply(spi_class)
print(table[["district", "spi_sep", "vci", "gw_rise_m", "spi_class"]].sort_values("spi_sep").to_string(index=False))

table["sev"]   = (-table.spi_sep).clip(0, 2.5) / 2.5
table["veg"]   = (1 - table.vci / 100).clip(0, 1)
table["gwstr"] = (table.gw_rise_m / 3).clip(0, 1)
table["priority"] = (100 * (0.5 * table.sev + 0.3 * table.veg + 0.2 * table.gwstr)).round(0).astype(int)

def action(p):
    if p >= 60: return "Declare drought: tankers, fodder camps, contingency crops"
    if p >= 40: return "Intensify: borewell and tank repair, crop advisories"
    if p >= 20: return "Watch: weekly monitoring, farmer advisories"
    return "Normal: routine monitoring"
table["action"] = table.priority.apply(action)

plan = table.sort_values("priority", ascending=False)[["district", "region", "spi_sep", "vci", "gw_rise_m", "priority", "action"]]
print(plan.to_string(index=False))
print("\nDistricts by action:")
print(table.action.value_counts().to_string())

plan.round(2).to_csv("drought_planning_table_sep2026.csv", index=False)
print("Saved: drought_planning_table_sep2026.csv")

spi_cmap = cm.StepColormap(["#7F1D1D", "#DC2626", "#F97316", "#F5F5F4", "#60A5FA"],
                           index=[-4, -2, -1.5, -1, 1, 4], vmin=-4, vmax=4,
                           caption="SPI-3 at end of September 2026")
action_color = {"Declare": "#7F1D1D", "Intensify": "#F97316", "Watch": "#FACC15", "Normal": "#22C55E"}

minx, miny, maxx, maxy = dist.total_bounds
m = folium.Map(location=[(miny + maxy) / 2, (minx + maxx) / 2], zoom_start=7, tiles=None, control_scale=True)
folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri World Imagery", name="Satellite (Esri)", show=False).add_to(m)

def spi_series_svg(d):
    s = spi_2026[spi_2026.district == d].sort_values("month")
    lo, hi = -3, 2
    y = lambda v: 100 - (v - lo) / (hi - lo) * 100
    pts = " ".join(f"{20 + i * 24:.0f},{y(v):.1f}" for i, v in enumerate(s.spi))
    return (f'<svg width="230" height="128">'
            f'<line x1="0" x2="230" y1="{y(-1):.1f}" y2="{y(-1):.1f}" stroke="#F97316" stroke-dasharray="4 3"/>'
            f'<line x1="0" x2="230" y1="{y(-2):.1f}" y2="{y(-2):.1f}" stroke="#DC2626" stroke-dasharray="4 3"/>'
            f'<polyline points="{pts}" fill="none" stroke="#1E3A8A" stroke-width="2"/>'
            f'<text x="2" y="122" font-size="9">Jan</text><text x="195" y="122" font-size="9">Sep</text>'
            f'<text x="2" y="12" font-size="9">+1</text><text x="2" y="{y(-1)+3:.0f}" font-size="9">-1</text>'
            f'<text x="2" y="{y(-2)+3:.0f}" font-size="9">-2</text></svg>')

folium.GeoJson(table, name="SPI-3 (Sep 2026)",
               style_function=lambda f: {"fillColor": spi_cmap(f["properties"]["spi_sep"]), "color": "white",
                                         "weight": 1.2, "fillOpacity": 0.8},
               highlight_function=lambda f: {"weight": 3, "color": "#F59E0B"},
               tooltip=folium.GeoJsonTooltip(fields=["district", "spi_sep", "spi_class"],
                                             aliases=["District", "SPI-3", "Class"]),
               popup=folium.GeoJsonPopup(fields=["district"], labels=False)).add_to(m)
spi_cmap.add_to(m)

# VCI layer with popups containing the trend chart
vci_layer = folium.FeatureGroup(name="Vegetation VCI (Sep 2026)", show=False)
vci_cmap = cm.LinearColormap(["#7F1D1D", "#F59E0B", "#FDE68A", "#65A30D", "#14532D"], vmin=0, vmax=100)
for _, r in table.iterrows():
    folium.CircleMarker([r.centroid_lat, r.centroid_lon], radius=14, color="#1F2937", weight=1,
                        fill=True, fill_color=vci_cmap(r.vci), fill_opacity=0.95,
                        tooltip=f"{r.district}: VCI {r.vci:.0f}").add_to(vci_layer)
vci_layer.add_to(m)

# Groundwater change
gw_layer = folium.FeatureGroup(name="Groundwater change since 2025", show=False)
for _, r in table.iterrows():
    col = "#B91C1C" if r.gw_rise_m >= 2 else "#F97316" if r.gw_rise_m >= 1 else "#16A34A"
    folium.Marker([r.centroid_lat, r.centroid_lon], icon=folium.DivIcon(html=
        f'<div style="font:11px Arial;white-space:nowrap;background:{col};color:white;border-radius:4px;padding:1px 4px">'
        f'{r.gw_rise_m:+.1f} m</div>')).add_to(gw_layer)
gw_layer.add_to(m)

# Planning priority circles with popups
plan_layer = folium.FeatureGroup(name="Planning priority", show=True)
for _, r in table.iterrows():
    col = action_color[r.action.split(":")[0].split()[0]]
    html = (f"<b>{r.district}</b> ({r.region})<br>"
            f"SPI-3 Sep: <b>{r.spi_sep:.2f}</b> · {r.spi_class}<br>"
            f"VCI: <b>{r.vci:.0f}</b> · Groundwater: {r.gw_rise_m:+.1f} m<br>"
            f"Priority score: <b>{r.priority}/100</b><br>"
            f"<b style='color:{col}'>{r.action}</b><br>"
            f"<div style='font:600 11px Arial;margin-top:4px'>SPI-3, Jan to Sep 2026</div>{spi_series_svg(r.district)}")
    folium.CircleMarker([r.centroid_lat, r.centroid_lon], radius=4 + r.priority / 8, color="white", weight=2,
                        fill=True, fill_color=col, fill_opacity=0.95,
                        tooltip=f"{r.district}: priority {r.priority}",
                        popup=folium.Popup(html, max_width=260)).add_to(plan_layer)
plan_layer.add_to(m)

# Reservoirs
res_layer = folium.FeatureGroup(name="Major reservoirs (30 Sep 2026)")
for _, r in res.iterrows():
    pct, last = r.storage_2026_pct, r.storage_2025_pct
    bar = lambda p, c: (f'<div style="background:#E5E7EB;width:200px;height:10px;border-radius:5px">'
                        f'<div style="background:{c};width:{p*2:.0f}px;height:10px;border-radius:5px"></div></div>')
    html = (f"<b>{r.reservoir}</b><br>{r.district} · capacity {r.gross_capacity_mcm:,} MCM<br>"
            f"2026: <b>{pct:.0f}%</b>{bar(pct, '#2563EB')}2025: {last:.0f}%{bar(last, '#94A3B8')}")
    folium.Marker([r.lat, r.lon], tooltip=f"{r.reservoir}: {pct:.0f}%", popup=folium.Popup(html, max_width=240),
                  icon=folium.Icon(color="blue", icon="water", prefix="fa")).add_to(res_layer)
res_layer.add_to(m)

plugins.Fullscreen().add_to(m)
plugins.MiniMap(toggle_display=True, position="bottomleft").add_to(m)
plugins.MeasureControl(primary_length_unit="kilometers", primary_area_unit="sqkilometers").add_to(m)

counts = table.action.str.split(":").str[0].str.split().str[0].value_counts()
title_html = (
    '<div style="position:fixed; top:10px; left:55px; z-index:9999; background:white; padding:8px 14px;'
    'border-left:5px solid #DC2626; border-radius:4px; box-shadow:0 1px 6px rgba(0,0,0,.3); font-family:Arial">'
    '<div style="font-size:16px; font-weight:bold; color:#0F172A">Karnataka Drought Monitor · September 2026</div>'
    f'<div style="font-size:12px; color:#334155">In drought (SPI ≤ −1): <b>{(table.spi_sep <= -1).sum()}</b> of {len(table)} districts · '
    f'Declare: <b>{counts.get("Declare", 0)}</b> · Intensify: <b>{counts.get("Intensify", 0)}</b> · '
    f'Mean VCI: <b>{table.vci.mean():.0f}</b></div></div>')
m.get_root().html.add_child(folium.Element(title_html))

legend_html = (
    '<div style="position:fixed; bottom:30px; right:10px; z-index:9999; background:white; padding:10px 12px;'
    'border-radius:6px; box-shadow:0 1px 6px rgba(0,0,0,.3); font:12px Arial; line-height:19px">'
    '<b>Planning action</b><br>'
    '<span style="color:#7F1D1D">●</span> Declare drought (score ≥ 60)<br>'
    '<span style="color:#F97316">●</span> Intensify (40–60)<br>'
    '<span style="color:#FACC15">●</span> Watch (20–40)<br>'
    '<span style="color:#22C55E">●</span> Normal (&lt; 20)<br>'
    '<b>SPI-3:</b> ≤ −2 extreme · −2 to −1.5 severe · −1.5 to −1 moderate</div>')
legend = branca.element.MacroElement(); legend._template = branca.element.Template(
    "{% macro html(this, kwargs) %}" + legend_html + "{% endmacro %}")
m.get_root().add_child(legend)

folium.LayerControl(collapsed=False).add_to(m)
m.save("Karnataka_Drought_Monitor.html")
print("Saved: Karnataka_Drought_Monitor.html")
m

