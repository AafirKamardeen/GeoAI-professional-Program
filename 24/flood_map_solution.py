# !pip install folium geopandas   # uncomment in Colab if needed
import pandas as pd
import geopandas as gpd
import folium
from folium import plugins
import branca

DATA = "data/"
print("folium", folium.__version__, "| geopandas", gpd.__version__)

aoi        = gpd.read_file(DATA + "study_area.geojson")
rivers     = gpd.read_file(DATA + "rivers.geojson")
flood      = gpd.read_file(DATA + "flood_extent_2026-07-10.geojson")
normal     = gpd.read_file(DATA + "normal_water_2026-05-20.geojson")
settle     = gpd.read_file(DATA + "settlements.geojson")

rain_st    = pd.read_csv(DATA + "rainfall_stations.csv")
rain_daily = pd.read_csv(DATA + "rainfall_daily_july2026.csv")
gauges     = pd.read_csv(DATA + "river_gauges.csv")
gauge_ts   = pd.read_csv(DATA + "river_gauge_levels_july2026.csv")
camps      = pd.read_csv(DATA + "relief_camps.csv")

rain_gdf  = gpd.GeoDataFrame(rain_st, geometry=gpd.points_from_xy(rain_st.lon, rain_st.lat), crs="EPSG:4326")
gauge_gdf = gpd.GeoDataFrame(gauges,  geometry=gpd.points_from_xy(gauges.lon, gauges.lat),  crs="EPSG:4326")

for name, g in [("rivers", rivers), ("flood", flood), ("settlements", settle), ("rain", rain_gdf), ("gauges", gauge_gdf)]:
    print(f"{name:12s} {len(g):3d} features | CRS {g.crs}")

affected = gpd.sjoin(settle, flood[["depth_class", "severity", "geometry"]], how="left", predicate="within")
affected["flooded"] = affected["depth_class"].notna()
affected["depth_class"] = affected["depth_class"].fillna("Not flooded")

flood_utm = flood.to_crs("EPSG:32646")
flood["area_km2"] = (flood_utm.area / 1e6).round(1)
normal_km2 = normal.to_crs("EPSG:32646").area.sum() / 1e6
total_flood_km2 = flood["area_km2"].sum()

print(f"Total flooded area : {total_flood_km2:,.0f} km²")
print(f"Normal water area  : {normal_km2:,.0f} km²")
print(f"Settlements flooded: {affected.flooded.sum()} of {len(affected)}")
print(f"Population exposed : {affected.loc[affected.flooded, 'population'].sum():,}")
affected.loc[affected.flooded, ["name", "district", "population", "depth_class"]]

minx, miny, maxx, maxy = aoi.total_bounds
m = folium.Map(location=[(miny + maxy) / 2, (minx + maxx) / 2], zoom_start=9, tiles=None, control_scale=True)

folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
    attr="Esri Light Gray", name="Light grey (Esri)").add_to(m)
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri World Imagery", name="Satellite (Esri)").add_to(m)

folium.GeoJson(aoi, name="Study area",
               style_function=lambda f: {"color": "#0F172A", "weight": 2, "dashArray": "6 4", "fillOpacity": 0}).add_to(m)

flood_colors = {1: "#93C5FD", 2: "#3B82F6", 3: "#1E3A8A"}

folium.GeoJson(normal, name="Normal water (20 May 2026)",
               style_function=lambda f: {"fillColor": "#06B6D4", "color": "#0891B2", "weight": 0.5, "fillOpacity": 0.6}
               ).add_to(m)

folium.GeoJson(flood, name="Flood extent (10 Jul 2026)",
               style_function=lambda f: {"fillColor": flood_colors[f["properties"]["severity"]],
                                         "color": flood_colors[f["properties"]["severity"]],
                                         "weight": 0.5, "fillOpacity": 0.55},
               highlight_function=lambda f: {"weight": 2, "color": "#F59E0B"},
               tooltip=folium.GeoJsonTooltip(fields=["depth_class", "area_km2", "date"],
                                             aliases=["Depth class", "Area (km²)", "Date"])
               ).add_to(m)

width = {1: 5, 2: 3, 3: 2}
folium.GeoJson(rivers, name="River network",
               style_function=lambda f: {"color": "#0369A1", "weight": width[f["properties"]["order"]]},
               tooltip=folium.GeoJsonTooltip(fields=["river_name", "type", "length_km"],
                                             aliases=["River", "Type", "Length in AOI (km)"])
               ).add_to(m)

imd_colors = {"Light": "#A7F3D0", "Moderate": "#FDE68A", "Heavy": "#FB923C",
              "Very heavy": "#EF4444", "Extremely heavy": "#7F1D1D"}

def rain_svg(sid):
    s = rain_daily[rain_daily.station_id == sid]
    vals, mx = s.rainfall_mm.values, 250
    bars = "".join(
        f'<rect x="{i*14+2}" y="{100 - v/mx*100:.1f}" width="11" height="{v/mx*100:.1f}" fill="{"#B91C1C" if v >= 115.6 else "#2563EB"}"/>'
        for i, v in enumerate(vals))
    return (f'<svg width="214" height="118"><line x1="0" y1="100" x2="214" y2="100" stroke="#999"/>{bars}'
            f'<text x="2" y="114" font-size="9">1 Jul</text><text x="180" y="114" font-size="9">15 Jul</text></svg>')

rain_layer = folium.FeatureGroup(name="Rainfall stations")
for _, r in rain_gdf.iterrows():
    html = f"""<b>{r.station_name}</b> ({r.station_id})<br>
    Total 1–15 Jul: <b>{r.total_rain_mm:.0f} mm</b><br>
    Peak day: {r.max_daily_mm:.0f} mm — <i>{r.imd_category_peak}</i><br>{rain_svg(r.station_id)}<br>
    <small>Red bars = very heavy rain (≥115.6 mm)</small>"""
    folium.CircleMarker(
        location=[r.lat, r.lon], radius=r.total_rain_mm / 70,
        color="#1F2937", weight=1, fill=True, fill_color=imd_colors[r.imd_category_peak], fill_opacity=0.9,
        tooltip=f"{r.station_name}: {r.total_rain_mm:.0f} mm",
        popup=folium.Popup(html, max_width=240)).add_to(rain_layer)
rain_layer.add_to(m)

# Rainfall intensity heatmap (separate toggle)
plugins.HeatMap(rain_gdf[["lat", "lon", "total_rain_mm"]].values.tolist(),
                name="Rainfall heatmap", radius=45, blur=35, show=False).add_to(m)

status_color = {"Above Danger": "red", "Above Warning": "orange", "Normal": "green"}

def hydro_svg(gid, warn, danger):
    s = gauge_ts[gauge_ts.gauge_id == gid].water_level_m.values
    lo, hi = min(s.min(), warn) - 0.5, max(s.max(), danger) + 0.5
    y = lambda v: 100 - (v - lo) / (hi - lo) * 100
    pts = " ".join(f"{i * 214 / (len(s) - 1):.1f},{y(v):.1f}" for i, v in enumerate(s))
    return (f'<svg width="214" height="112">'
            f'<line x1="0" x2="214" y1="{y(danger):.1f}" y2="{y(danger):.1f}" stroke="#DC2626" stroke-dasharray="4 3"/>'
            f'<line x1="0" x2="214" y1="{y(warn):.1f}" y2="{y(warn):.1f}" stroke="#F59E0B" stroke-dasharray="4 3"/>'
            f'<polyline points="{pts}" fill="none" stroke="#1D4ED8" stroke-width="2"/>'
            f'<text x="2" y="110" font-size="9">1 Jul</text><text x="180" y="110" font-size="9">15 Jul</text></svg>')

gauge_layer = folium.FeatureGroup(name="River gauges")
for _, g in gauge_gdf.iterrows():
    diff = g.water_level_m - g.danger_level_m
    html = f"""<b>{g.site}</b> — {g.river}<br>
    Water level: <b>{g.water_level_m:.2f} m</b> ({g.trend})<br>
    Danger level: {g.danger_level_m:.2f} m &nbsp;|&nbsp; Warning: {g.warning_level_m:.2f} m<br>
    <b style="color:{status_color[g.status]}">{g.status} ({diff:+.2f} m)</b><br>
    {hydro_svg(g.gauge_id, g.warning_level_m, g.danger_level_m)}<br>
    <small>Red dashed = danger, amber = warning · obs. {g.observed_at}</small>"""
    folium.Marker([g.lat, g.lon], tooltip=f"{g.site}: {g.status}",
                  popup=folium.Popup(html, max_width=240),
                  icon=folium.Icon(color=status_color[g.status], icon="tint", prefix="fa")).add_to(gauge_layer)
gauge_layer.add_to(m)

settle_layer = folium.FeatureGroup(name="Settlements (flood exposure)")
for _, s in affected.iterrows():
    col = "#DC2626" if s.flooded else "#16A34A"
    folium.CircleMarker(
        [s.geometry.y, s.geometry.x], radius=5 if s.population < 50000 else 9,
        color="white", weight=1.5, fill=True, fill_color=col, fill_opacity=1,
        tooltip=f"{s['name']} — {s.depth_class}",
        popup=folium.Popup(f"<b>{s['name']}</b> ({s.settlement_type})<br>District: {s.district}<br>"
                           f"Population: {s.population:,}<br>Flood status: <b>{s.depth_class}</b>", max_width=220)
    ).add_to(settle_layer)
settle_layer.add_to(m)

camp_layer = folium.FeatureGroup(name="Relief camps", show=False)
for _, cp in camps.iterrows():
    folium.Marker([cp.lat, cp.lon], tooltip=f"{cp.camp_name} (capacity {cp.capacity})",
                  icon=folium.Icon(color="purple", icon="home", prefix="fa")).add_to(camp_layer)
camp_layer.add_to(m)

plugins.Fullscreen().add_to(m)
plugins.MiniMap(toggle_display=True, position="bottomleft").add_to(m)
plugins.MeasureControl(primary_length_unit="kilometers", primary_area_unit="sqkilometers").add_to(m)
plugins.MousePosition(position="bottomright", prefix="Lat/Lon:").add_to(m)

exposed_pop = affected.loc[affected.flooded, "population"].sum()
title_html = f"""
<div style="position:fixed; top:10px; left:55px; z-index:9999; background:white; padding:8px 14px;
     border-left:5px solid #F59E0B; border-radius:4px; box-shadow:0 1px 6px rgba(0,0,0,.3); font-family:Arial">
  <div style="font-size:16px; font-weight:bold; color:#0F172A">Central Assam Flood Monitoring — 10 July 2026</div>
  <div style="font-size:12px; color:#334155">Flooded: {total_flood_km2:,.0f} km² · Settlements affected:
       {affected.flooded.sum()}/{len(affected)} · Population exposed: {exposed_pop:,}</div>
</div>"""
m.get_root().html.add_child(folium.Element(title_html))

legend_html = """
{% macro html(this, kwargs) %}
<div style="position:fixed; bottom:30px; right:10px; z-index:9999; background:white; padding:10px 12px;
     border-radius:6px; box-shadow:0 1px 6px rgba(0,0,0,.3); font:12px Arial; line-height:18px">
<b>Flood depth</b><br>
<i style="background:#93C5FD;width:14px;height:10px;display:inline-block"></i> Shallow &lt;0.5 m<br>
<i style="background:#3B82F6;width:14px;height:10px;display:inline-block"></i> Moderate 0.5–1.5 m<br>
<i style="background:#1E3A8A;width:14px;height:10px;display:inline-block"></i> Deep &gt;1.5 m<br>
<i style="background:#06B6D4;width:14px;height:10px;display:inline-block"></i> Normal water<br>
<b>Rainfall (peak day, IMD)</b><br>
<i style="background:#FB923C;border-radius:50%;width:10px;height:10px;display:inline-block"></i> Heavy
<i style="background:#EF4444;border-radius:50%;width:10px;height:10px;display:inline-block"></i> Very heavy
<i style="background:#7F1D1D;border-radius:50%;width:10px;height:10px;display:inline-block"></i> Extreme<br>
<b>Gauges</b> &nbsp;<span style="color:#DC2626">●</span> Danger <span style="color:#F59E0B">●</span> Warning
<span style="color:#16A34A">●</span> Normal<br>
<b>Settlements</b> &nbsp;<span style="color:#DC2626">●</span> Flooded <span style="color:#16A34A">●</span> Safe
</div>
{% endmacro %}"""
legend = branca.element.MacroElement()
legend._template = branca.element.Template(legend_html)
m.get_root().add_child(legend)

folium.LayerControl(collapsed=False).add_to(m)

m.save("Assam_Flood_Monitoring_Map.html")
print("Saved: Assam_Flood_Monitoring_Map.html")

