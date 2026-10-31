"""
Sentinel-2 Spectral Index Explorer
==================================
GeoAI Professional Program - Final Session Project
OrbitalIQ Analytics Private Limited

A Tkinter desktop app: load a Sentinel-2 image, click an index button,
and see the map, its statistics, a classified version and a histogram.

Run:
    pip install numpy matplotlib rasterio
    python s2_index_explorer.py

Optional - create a demo GeoTIFF for students to practise "Open image":
    python s2_index_explorer.py --make-demo demo_s2.tif

Expected input: a multiband GeoTIFF (e.g. exported from Google Earth Engine
with bands B2, B3, B4, B8, B11, B12). If the band names are stored in the
file, they are detected automatically; otherwise you choose them in a dialog.
"""

import os
import re
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import matplotlib

matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

try:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import from_origin
    from affine import Affine
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


# ----------------------------------------------------------------------------
# 1. Configuration
# ----------------------------------------------------------------------------

# OrbitalIQ colours
NAVY = "#0F172A"
TEAL = "#164E63"
TEAL_DARK = "#0E3442"
AMBER = "#F59E0B"
AMBER_LIGHT = "#FDE68A"
PANEL = "#F1F5F9"
TEXT = "#1E293B"
NODATA_COLOR = "#E5E7EB"

MAX_SIDE = 2500  # larger images are downsampled on load so the app stays fast

# The six band "roles" the app works with, and their Sentinel-2 names
ROLES = ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"]
S2_NAME = {"Blue": "B2", "Green": "B3", "Red": "B4",
           "NIR": "B8", "SWIR1": "B11", "SWIR2": "B12"}


def safe_div(num, den):
    """Division that returns NaN instead of crashing when the denominator is 0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        out = num / den
    out[~np.isfinite(out)] = np.nan
    return out


def nd(a, b):
    """Normalized difference: (a - b) / (a + b)."""
    return safe_div(a - b, a + b)


# Every index in one table: formula, function, colours, and simple classes.
# edges = class boundaries; labels = one label per class (len(edges) + 1).
INDICES = {
    "NDVI": dict(
        name="Normalized Difference Vegetation Index",
        formula="(NIR - Red) / (NIR + Red)",
        needs=("NIR", "Red"),
        func=lambda b: nd(b["NIR"], b["Red"]),
        cmap="RdYlGn", vmin=-0.2, vmax=0.8,
        use="Vegetation presence, density and health.",
        tip="Healthy leaves reflect strongly in NIR and absorb red light, "
            "so dense vegetation gives high NDVI (0.6+). Water is negative.",
        edges=[0.0, 0.2, 0.4, 0.6],
        labels=["Water / non-vegetated", "Bare soil / built-up",
                "Sparse vegetation", "Moderate vegetation", "Dense vegetation"],
        colors=["#d73027", "#fc8d59", "#fee08b", "#91cf60", "#1a9850"]),
    "NDWI": dict(
        name="Normalized Difference Water Index (McFeeters)",
        formula="(Green - NIR) / (Green + NIR)",
        needs=("Green", "NIR"),
        func=lambda b: nd(b["Green"], b["NIR"]),
        cmap="RdBu", vmin=-0.6, vmax=0.6,
        use="Open water bodies - lakes, rivers, ponds.",
        tip="Water reflects some green but almost no NIR, so water is "
            "positive. Vegetation is strongly negative.",
        edges=[0.0, 0.2],
        labels=["Non-water", "Wet surface / shallow water", "Open water"],
        colors=["#c2a57a", "#9ecae1", "#08519c"]),
    "MNDWI": dict(
        name="Modified Normalized Difference Water Index",
        formula="(Green - SWIR1) / (Green + SWIR1)",
        needs=("Green", "SWIR1"),
        func=lambda b: nd(b["Green"], b["SWIR1"]),
        cmap="RdBu", vmin=-0.6, vmax=0.6,
        use="Water mapping in and around cities.",
        tip="Uses SWIR instead of NIR. Built-up areas reflect strongly in "
            "SWIR, so MNDWI separates water from urban areas better than NDWI.",
        edges=[0.0],
        labels=["Non-water", "Water"],
        colors=["#c2a57a", "#08519c"]),
    "NDBI": dict(
        name="Normalized Difference Built-up Index",
        formula="(SWIR1 - NIR) / (SWIR1 + NIR)",
        needs=("SWIR1", "NIR"),
        func=lambda b: nd(b["SWIR1"], b["NIR"]),
        cmap="RdYlBu_r", vmin=-0.5, vmax=0.4,
        use="Urban and built-up areas.",
        tip="Concrete and roofs reflect more SWIR than NIR, so built-up is "
            "positive. Note: bare soil is also positive - compare with BSI.",
        edges=[0.0, 0.1],
        labels=["Vegetation / water", "Mixed / low", "Built-up / bare"],
        colors=["#4575b4", "#fee090", "#d73027"]),
    "NDMI": dict(
        name="Normalized Difference Moisture Index",
        formula="(NIR - SWIR1) / (NIR + SWIR1)",
        needs=("NIR", "SWIR1"),
        func=lambda b: nd(b["NIR"], b["SWIR1"]),
        cmap="BrBG", vmin=-0.5, vmax=0.6,
        use="Water content of vegetation (drought stress, irrigation).",
        tip="Water inside leaves absorbs SWIR. Moist vegetation is high; "
            "dry crops, soil and urban surfaces are low.",
        edges=[-0.2, 0.0, 0.2, 0.4],
        labels=["Very dry", "Dry", "Moderate moisture",
                "High moisture", "Very high moisture"],
        colors=["#8c510a", "#d8b365", "#c7eae5", "#5ab4ac", "#01665e"]),
    "NBR": dict(
        name="Normalized Burn Ratio",
        formula="(NIR - SWIR2) / (NIR + SWIR2)",
        needs=("NIR", "SWIR2"),
        func=lambda b: nd(b["NIR"], b["SWIR2"]),
        cmap="RdYlGn", vmin=-0.5, vmax=0.8,
        use="Burned areas and fire severity.",
        tip="Burned land loses NIR and gains SWIR2, so burn scars are "
            "negative. For severity, compare before/after images (dNBR).",
        edges=[-0.1, 0.1, 0.3],
        labels=["Burned / very low", "Low (bare)", "Moderate", "Healthy vegetation"],
        colors=["#a50026", "#f46d43", "#d9ef8b", "#1a9850"]),
    "SAVI": dict(
        name="Soil Adjusted Vegetation Index",
        formula="1.5 x (NIR - Red) / (NIR + Red + 0.5)",
        needs=("NIR", "Red"),
        func=lambda b: 1.5 * safe_div(b["NIR"] - b["Red"], b["NIR"] + b["Red"] + 0.5),
        cmap="RdYlGn", vmin=-0.2, vmax=0.7,
        use="Vegetation in dry or sparsely vegetated regions.",
        tip="Adds a soil factor L = 0.5 to reduce the effect of bright soil "
            "background - useful in arid areas like Rajasthan.",
        edges=[0.0, 0.15, 0.3, 0.45],
        labels=["Water / non-vegetated", "Bare soil", "Sparse vegetation",
                "Moderate vegetation", "Dense vegetation"],
        colors=["#d73027", "#fc8d59", "#fee08b", "#91cf60", "#1a9850"]),
    "EVI": dict(
        name="Enhanced Vegetation Index",
        formula="2.5 x (NIR - Red) / (NIR + 6 x Red - 7.5 x Blue + 1)",
        needs=("NIR", "Red", "Blue"),
        func=lambda b: 2.5 * safe_div(b["NIR"] - b["Red"],
                                      b["NIR"] + 6 * b["Red"] - 7.5 * b["Blue"] + 1),
        cmap="RdYlGn", vmin=-0.2, vmax=0.8,
        use="Dense forests, where NDVI saturates.",
        tip="Uses the blue band to correct for atmosphere and does not "
            "saturate as quickly as NDVI over thick canopy.",
        edges=[0.0, 0.2, 0.4, 0.6],
        labels=["Water / non-vegetated", "Bare / built-up", "Sparse vegetation",
                "Moderate vegetation", "Dense vegetation"],
        colors=["#d73027", "#fc8d59", "#fee08b", "#91cf60", "#1a9850"]),
    "GNDVI": dict(
        name="Green Normalized Difference Vegetation Index",
        formula="(NIR - Green) / (NIR + Green)",
        needs=("NIR", "Green"),
        func=lambda b: nd(b["NIR"], b["Green"]),
        cmap="YlGn", vmin=-0.2, vmax=0.8,
        use="Chlorophyll content and crop nitrogen status.",
        tip="More sensitive to chlorophyll than NDVI - used in precision "
            "agriculture for mid/late crop stages.",
        edges=[0.2, 0.4, 0.6],
        labels=["Non-vegetated", "Low chlorophyll", "Moderate chlorophyll",
                "High chlorophyll"],
        colors=["#d9d9d9", "#d9f0a3", "#78c679", "#006837"]),
    "BSI": dict(
        name="Bare Soil Index",
        formula="((SWIR1 + Red) - (NIR + Blue)) / ((SWIR1 + Red) + (NIR + Blue))",
        needs=("SWIR1", "Red", "NIR", "Blue"),
        func=lambda b: nd(b["SWIR1"] + b["Red"], b["NIR"] + b["Blue"]),
        cmap="YlOrBr", vmin=-0.5, vmax=0.4,
        use="Exposed soil - fallow fields, mining, construction sites.",
        tip="Combines SWIR + Red (high for soil) against NIR + Blue "
            "(high for vegetation and water).",
        edges=[0.0, 0.1],
        labels=["Vegetated / water", "Mixed", "Bare soil"],
        colors=["#41ab5d", "#fed976", "#a6611a"]),
}

COMPOSITES = {
    "TCC": dict(title="True Colour (B4-B3-B2)", rgb=("Red", "Green", "Blue"),
                text="Natural colour - what the human eye would see.\n\n"
                     "Vegetation looks green, water dark blue, cities grey."),
    "FCC": dict(title="False Colour (B8-B4-B3)", rgb=("NIR", "Red", "Green"),
                text="NIR is shown as red.\n\nHealthy vegetation appears "
                     "BRIGHT RED, water is dark, cities are cyan/grey.\n\n"
                     "This is the classic image that explains why NDVI works."),
    "SWIR": dict(title="SWIR Composite (B11-B8-B4)", rgb=("SWIR1", "NIR", "Red"),
                 text="Shortwave infrared composite.\n\nVegetation is green, "
                      "bare soil and built-up areas are pink/brown, water is "
                      "dark blue, burn scars are dark red."),
}


def sentinel_formula(formula):
    """Rewrite a role-based formula with Sentinel-2 band names."""
    for role in ["SWIR1", "SWIR2", "NIR", "Green", "Blue", "Red"]:
        formula = formula.replace(role, S2_NAME[role])
    return formula


def get_cmap(name):
    try:
        cmap = matplotlib.colormaps[name]
    except AttributeError:  # older matplotlib
        cmap = matplotlib.cm.get_cmap(name)
    return cmap.copy()


# ----------------------------------------------------------------------------
# 2. Reading images
# ----------------------------------------------------------------------------

def normalise_band_name(name):
    """'B08' -> 'B8', 'b11' -> 'B11'."""
    name = (name or "").strip().upper()
    return re.sub(r"^B0*(\d+A?)$", r"B\1", name)


def guess_mapping(descriptions, count):
    """Return ({role: band_number or None}, confident)."""
    names = [normalise_band_name(d) for d in descriptions]
    mapping = {}
    for role in ROLES:
        mapping[role] = names.index(S2_NAME[role]) + 1 if S2_NAME[role] in names else None
    if all(mapping[r] for r in ROLES):
        return mapping, True

    # No names in the file - guess from the band count
    if count == 13:    # B1..B12 incl. B10
        order = {"Blue": 2, "Green": 3, "Red": 4, "NIR": 8, "SWIR1": 12, "SWIR2": 13}
    elif count == 12:  # B1..B12 without B10 (L2A)
        order = {"Blue": 2, "Green": 3, "Red": 4, "NIR": 8, "SWIR1": 11, "SWIR2": 12}
    elif count >= 6:   # assume B2, B3, B4, B8, B11, B12
        order = {r: i + 1 for i, r in enumerate(ROLES)}
    elif count >= 4:   # assume B2, B3, B4, B8
        order = {"Blue": 1, "Green": 2, "Red": 3, "NIR": 4, "SWIR1": None, "SWIR2": None}
    else:
        order = {r: None for r in ROLES}
    return order, False


def prepare_bands(stack, nodata):
    """Mask no-data pixels and convert to 0-1 reflectance if needed."""
    stack = stack.astype(np.float32)
    invalid = ~np.all(np.isfinite(stack), axis=0)
    if nodata is not None and not np.isnan(nodata):
        invalid |= np.any(stack == nodata, axis=0)
    invalid |= np.all(stack == 0, axis=0)
    stack[:, invalid] = np.nan

    scaled = False
    sample = stack[:, ::4, ::4]
    if np.isfinite(sample).any() and np.nanpercentile(sample, 99) > 2:
        stack /= 10000.0  # Sentinel-2 DN -> surface reflectance
        scaled = True
    return stack, scaled


def pixel_area_m2(transform, crs, height):
    if transform is None:
        return None
    a, e = transform.a, transform.e
    if crs is not None and crs.is_geographic:
        lat = transform.f + e * height / 2
        return abs(a * 111320 * np.cos(np.radians(lat))) * abs(e * 110574)
    return abs(a * e)


# ----------------------------------------------------------------------------
# 3. Synthetic demo scene (works even without an image or internet)
# ----------------------------------------------------------------------------

# Typical surface reflectance per land cover:  Blue Green Red  NIR  SWIR1 SWIR2
REFL = np.array([
    [0.060, 0.070, 0.045, 0.030, 0.018, 0.012],  # 0 water
    [0.030, 0.060, 0.030, 0.420, 0.210, 0.095],  # 1 forest
    [0.050, 0.085, 0.070, 0.250, 0.230, 0.140],  # 2 cropland
    [0.130, 0.135, 0.135, 0.190, 0.250, 0.220],  # 3 built-up
    [0.100, 0.135, 0.175, 0.230, 0.330, 0.290],  # 4 bare soil
    [0.040, 0.050, 0.060, 0.110, 0.190, 0.220],  # 5 burnt area
], dtype=np.float32)


def make_demo_scene(size=600, seed=42):
    rng = np.random.default_rng(seed)
    h = w = size
    y, x = np.mgrid[0:h, 0:w].astype(np.float32) / size

    def noise(sigma_px):
        n = rng.standard_normal((h, w)).astype(np.float32)
        fy = np.fft.fftfreq(h)[:, None]
        fx = np.fft.fftfreq(w)[None, :]
        g = np.exp(-2 * (np.pi * sigma_px) ** 2 * (fx ** 2 + fy ** 2))
        out = np.real(np.fft.ifft2(np.fft.fft2(n) * g))
        return ((out - out.mean()) / (out.std() + 1e-9)).astype(np.float32)

    cls = np.full((h, w), 2, np.uint8)                       # cropland
    cls[noise(30) > 0.3] = 1                                 # forest
    cls[noise(18) > 1.3] = 4                                 # bare soil
    city = ((x - 0.72) / 0.22) ** 2 + ((y - 0.28) / 0.17) ** 2 + 0.25 * noise(12) < 1
    cls[city] = 3                                            # city
    cls[city & (noise(6) > 1.4)] = 1                         # city parks
    burn = ((x - 0.22) / 0.12) ** 2 + ((y - 0.80) / 0.09) ** 2 + 0.3 * noise(10) < 1
    cls[burn] = 5                                            # burn scar
    lake = ((x - 0.20) / 0.10) ** 2 + ((y - 0.25) / 0.08) ** 2 + 0.2 * noise(8) < 1
    cls[lake] = 0                                            # lake
    centre = 0.56 + 0.07 * np.sin(x * 2 * np.pi * 1.3) + 0.02 * np.sin(x * 2 * np.pi * 4)
    cls[np.abs(y - centre) < 0.012 + 0.005 * (1 + np.sin(x * 9))] = 0  # river
    for _ in range(12):                                      # village ponds
        cx, cy = rng.uniform(0.05, 0.95, 2)
        r = rng.uniform(0.006, 0.014)
        cls[(x - cx) ** 2 + (y - cy) ** 2 < r ** 2] = 0

    bands = REFL[cls].copy()                                 # h, w, 6
    bands *= (1 + 0.08 * noise(5))[..., None]               # brightness variation
    vigour = 0.10 * noise(8)                                 # vegetation vigour
    veg = np.isin(cls, (1, 2))
    bands[..., 3] *= np.where(veg, 1 + vigour, 1)
    bands[..., 2] *= np.where(veg, 1 - vigour, 1)
    for i in range(6):                                       # small sensor noise
        bands[..., i] *= 1 + 0.02 * noise(2)
    bands += rng.normal(0, 0.004, bands.shape).astype(np.float32)
    bands = np.clip(bands, 0.001, 1)

    p = np.pad(bands, ((1, 1), (1, 1), (0, 0)), mode="edge")  # mixed pixels
    bands = sum(p[dy:dy + h, dx:dx + w] for dy in range(3) for dx in range(3)) / 9
    return {role: bands[..., i].astype(np.float32) for i, role in enumerate(ROLES)}


DEMO_TRANSFORM_ORIGIN = (376800, 2246000, 10, 10)  # UTM 45N, near Bhubaneswar
DEMO_CRS = "EPSG:32645"


def save_demo_geotiff(path):
    if not HAS_RASTERIO:
        sys.exit("rasterio is required: pip install rasterio")
    bands = make_demo_scene()
    h, w = bands["Red"].shape
    profile = dict(driver="GTiff", height=h, width=w, count=6, dtype="uint16",
                   crs=DEMO_CRS, transform=from_origin(*DEMO_TRANSFORM_ORIGIN),
                   compress="deflate")
    with rasterio.open(path, "w", **profile) as dst:
        for i, role in enumerate(ROLES, start=1):
            dst.write((bands[role] * 10000).astype("uint16"), i)
            dst.set_band_description(i, S2_NAME[role])
    print(f"Demo Sentinel-2 image written to {path}")


# ----------------------------------------------------------------------------
# 4. Band-mapping dialog
# ----------------------------------------------------------------------------

class BandMappingDialog(tk.Toplevel):
    def __init__(self, parent, descriptions, count, mapping):
        super().__init__(parent)
        self.title("Match the bands")
        self.configure(bg=PANEL)
        self.resizable(False, False)
        self.transient(parent)
        self.result = None

        self.options = ["- not available -"] + [
            f"Band {i}" + (f"  ({descriptions[i - 1]})" if descriptions[i - 1] else "")
            for i in range(1, count + 1)]

        ttk.Label(self, style="Side.TLabel", wraplength=360, justify="left",
                  text="The file does not say which band is which. Choose the "
                       "band for each role (Sentinel-2 name in brackets).").grid(
            row=0, column=0, columnspan=2, padx=14, pady=(14, 10), sticky="w")

        self.vars = {}
        for r, role in enumerate(ROLES, start=1):
            ttk.Label(self, style="Side.TLabel",
                      text=f"{role} ({S2_NAME[role]})").grid(row=r, column=0, padx=14, pady=4, sticky="w")
            var = tk.StringVar(value=self.options[mapping.get(role) or 0])
            ttk.Combobox(self, textvariable=var, values=self.options, state="readonly",
                         width=26).grid(row=r, column=1, padx=14, pady=4)
            self.vars[role] = var

        btns = ttk.Frame(self, style="Side.TFrame")
        btns.grid(row=len(ROLES) + 1, column=0, columnspan=2, pady=14)
        ttk.Button(btns, text="Cancel", command=self.destroy).pack(side="left", padx=6)
        ttk.Button(btns, text="Load image", style="Primary.TButton",
                   command=self._ok).pack(side="left", padx=6)

        self.grab_set()
        self.wait_window(self)

    def _ok(self):
        mapping = {role: self.options.index(v.get()) or None for role, v in self.vars.items()}
        missing = [r for r in ("Blue", "Green", "Red", "NIR") if not mapping[r]]
        if missing:
            messagebox.showwarning("Bands needed", "Please choose: " + ", ".join(missing), parent=self)
            return
        self.result = mapping
        self.destroy()


# ----------------------------------------------------------------------------
# 5. The application
# ----------------------------------------------------------------------------

class IndexExplorer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Sentinel-2 Spectral Index Explorer - OrbitalIQ GeoAI")
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"{min(1440, sw - 40)}x{min(860, sh - 80)}+10+10")
        self.minsize(1000, 600)
        self.configure(bg=PANEL)

        self.bands = None        # {role: 2-D array of reflectance}
        self.meta = {}           # file info, transform, crs, pixel area
        self.cache = {}          # computed indices
        self.current = None      # selected composite or index key
        self.buttons = {}
        self.mode = tk.StringVar(value="index")
        self.auto_stretch = tk.BooleanVar(value=False)

        self._style()
        self._build_header()
        self._build_body()
        self._refresh_buttons()
        self.render()
        self.protocol("WM_DELETE_WINDOW", self._quit)

    # ---------------- layout ----------------
    def _style(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure("Side.TFrame", background=PANEL)
        s.configure("Side.TLabel", background=PANEL, foreground=TEXT, font=("Segoe UI", 10))
        s.configure("Info.TLabel", background=PANEL, foreground="#475569", font=("Segoe UI", 9))
        s.configure("Section.TLabel", background=PANEL, foreground=TEAL, font=("Segoe UI", 10, "bold"))
        s.configure("TButton", font=("Segoe UI", 10), padding=6)
        s.configure("Primary.TButton", background=TEAL, foreground="white", font=("Segoe UI", 10, "bold"))
        s.map("Primary.TButton", background=[("active", TEAL_DARK), ("disabled", "#94A3B8")])
        for name, bg in (("Index.TButton", "white"), ("Active.TButton", AMBER)):
            s.configure(name, background=bg, foreground=NAVY, font=("Segoe UI", 10, "bold"), padding=5)
            s.map(name, background=[("disabled", "#E2E8F0"), ("active", AMBER_LIGHT if bg == "white" else AMBER)],
                  foreground=[("disabled", "#94A3B8")])
        s.configure("TRadiobutton", background=PANEL, font=("Segoe UI", 10))
        s.configure("TCheckbutton", background=PANEL, font=("Segoe UI", 10))

    def _build_header(self):
        bar = tk.Frame(self, bg=NAVY, height=58)
        bar.pack(fill="x")
        tk.Label(bar, text="Sentinel-2 Spectral Index Explorer", bg=NAVY, fg="white",
                 font=("Segoe UI", 16, "bold")).pack(side="left", padx=(18, 10), pady=12)
        tk.Label(bar, text="GeoAI Professional Program  |  OrbitalIQ Analytics", bg=NAVY,
                 fg=AMBER, font=("Segoe UI", 10)).pack(side="left", pady=12)
        tk.Frame(self, bg=AMBER, height=3).pack(fill="x")

    def _section(self, parent, text):
        ttk.Label(parent, text=text, style="Section.TLabel").pack(anchor="w", pady=(10, 3))

    def _build_body(self):
        body = ttk.Frame(self, style="Side.TFrame")
        body.pack(fill="both", expand=True)

        # Left: controls
        holder = ttk.Frame(body, style="Side.TFrame")
        holder.pack(side="left", fill="y", padx=(10, 4), pady=4)
        sb_canvas = tk.Canvas(holder, bg=PANEL, highlightthickness=0, width=255)
        scroll = ttk.Scrollbar(holder, orient="vertical", command=sb_canvas.yview)
        sb_canvas.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        sb_canvas.pack(side="left", fill="y")
        side = ttk.Frame(sb_canvas, style="Side.TFrame")
        win = sb_canvas.create_window((0, 0), window=side, anchor="nw", width=250)
        side.bind("<Configure>", lambda e: sb_canvas.configure(scrollregion=sb_canvas.bbox("all")))

        def _wheel(event):
            if event.num == 4 or event.delta > 0:
                sb_canvas.yview_scroll(-1, "units")
            else:
                sb_canvas.yview_scroll(1, "units")
        for w in (sb_canvas, side):
            w.bind("<Enter>", lambda e: (sb_canvas.bind_all("<MouseWheel>", _wheel),
                                         sb_canvas.bind_all("<Button-4>", _wheel),
                                         sb_canvas.bind_all("<Button-5>", _wheel)))
            w.bind("<Leave>", lambda e: (sb_canvas.unbind_all("<MouseWheel>"),
                                         sb_canvas.unbind_all("<Button-4>"),
                                         sb_canvas.unbind_all("<Button-5>")))

        self._section(side, "1  LOAD IMAGE")
        ttk.Button(side, text="Open Sentinel-2 GeoTIFF", style="Primary.TButton",
                   command=self.open_image).pack(fill="x", pady=2)
        ttk.Button(side, text="Load demo scene", command=self.load_demo).pack(fill="x", pady=2)
        self.map_btn = ttk.Button(side, text="Band mapping...", command=self.remap_bands)
        self.map_btn.pack(fill="x", pady=2)
        self.info = ttk.Label(side, style="Info.TLabel", wraplength=235, justify="left",
                              text="No image loaded.")
        self.info.pack(anchor="w", pady=(6, 0))

        self._section(side, "2  COLOUR COMPOSITES")
        for key in COMPOSITES:
            b = ttk.Button(side, text=COMPOSITES[key]["title"], style="Index.TButton",
                           command=lambda k=key: self.select(k))
            b.pack(fill="x", pady=2)
            self.buttons[key] = b

        self._section(side, "3  SPECTRAL INDICES")
        grid = ttk.Frame(side, style="Side.TFrame")
        grid.pack(fill="x")
        for i, key in enumerate(INDICES):
            b = ttk.Button(grid, text=key, style="Index.TButton", command=lambda k=key: self.select(k))
            b.grid(row=i // 2, column=i % 2, sticky="ew", padx=2, pady=2)
            self.buttons[key] = b
        grid.columnconfigure((0, 1), weight=1)

        self._section(side, "4  DISPLAY")
        for text, val in (("Index map", "index"), ("Classified map", "classified"),
                          ("Side by side with true colour", "side"), ("Map + histogram", "hist")):
            ttk.Radiobutton(side, text=text, value=val, variable=self.mode,
                            command=self.render).pack(anchor="w")
        ttk.Checkbutton(side, text="Auto contrast stretch (2-98%)", variable=self.auto_stretch,
                        command=self.render).pack(anchor="w", pady=(4, 0))

        self._section(side, "5  EXPORT")
        self.png_btn = ttk.Button(side, text="Save map as PNG", command=self.save_png)
        self.png_btn.pack(fill="x", pady=2)
        self.tif_btn = ttk.Button(side, text="Export as GeoTIFF", command=self.export_geotiff)
        self.tif_btn.pack(fill="x", pady=2)

        # Right: statistics panel
        right = ttk.Frame(body, style="Side.TFrame", width=330)
        right.pack(side="right", fill="y", padx=(8, 14), pady=4)
        right.pack_propagate(False)
        ttk.Label(right, text="ABOUT & STATISTICS", style="Section.TLabel").pack(anchor="w", pady=(12, 4))
        self.stats = tk.Text(right, wrap="word", font=("Consolas", 10), bg="white", fg=TEXT,
                             relief="flat", padx=10, pady=10, highlightthickness=1,
                             highlightbackground="#CBD5E1")
        self.stats.pack(fill="both", expand=True, pady=(0, 10))
        self.stats.tag_configure("h", font=("Segoe UI", 12, "bold"), foreground=NAVY)
        self.stats.tag_configure("s", font=("Segoe UI", 10, "bold"), foreground=TEAL)

        # Centre: matplotlib figure + toolbar
        centre = ttk.Frame(body, style="Side.TFrame")
        centre.pack(side="left", fill="both", expand=True, pady=10)
        self.fig = Figure(figsize=(8, 6), dpi=100, facecolor="white", constrained_layout=True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=centre)
        self.toolbar = NavigationToolbar2Tk(self.canvas, centre, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(side="bottom", fill="x")
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    # ---------------- loading ----------------
    def open_image(self):
        if not HAS_RASTERIO:
            messagebox.showerror("rasterio missing", "Install it first:\n\npip install rasterio")
            return
        path = filedialog.askopenfilename(
            title="Open Sentinel-2 image",
            filetypes=[("GeoTIFF", "*.tif *.tiff *.TIF"), ("All files", "*.*")])
        if not path:
            return
        try:
            with rasterio.open(path) as src:
                descriptions, count = list(src.descriptions), src.count
        except Exception as exc:
            messagebox.showerror("Cannot open file", str(exc))
            return
        mapping, confident = guess_mapping(descriptions, count)
        if not confident:
            mapping = BandMappingDialog(self, descriptions, count, mapping).result
            if mapping is None:
                return
        self._load_file(path, mapping)

    def remap_bands(self):
        path = self.meta.get("path")
        if not path:
            return
        with rasterio.open(path) as src:
            descriptions, count = list(src.descriptions), src.count
        mapping = BandMappingDialog(self, descriptions, count, self.meta["mapping"]).result
        if mapping:
            self._load_file(path, mapping)

    def _load_file(self, path, mapping):
        self.config(cursor="watch")
        self.update_idletasks()
        try:
            roles = [r for r in ROLES if mapping.get(r)]
            with rasterio.open(path) as src:
                scale = max(src.width, src.height) / MAX_SIDE
                if scale > 1:
                    oh, ow = int(round(src.height / scale)), int(round(src.width / scale))
                    resampling = Resampling.average
                else:
                    oh, ow, resampling = src.height, src.width, Resampling.nearest
                stack = src.read([mapping[r] for r in roles], out_shape=(len(roles), oh, ow),
                                 resampling=resampling)
                transform = src.transform * Affine.scale(src.width / ow, src.height / oh)
                crs, nodata, full = src.crs, src.nodata, (src.width, src.height)
            stack, scaled = prepare_bands(stack, nodata)
        except Exception as exc:
            messagebox.showerror("Could not read image", str(exc))
            return
        finally:
            self.config(cursor="")

        self.bands = {r: stack[i] for i, r in enumerate(roles)}
        self.meta = dict(path=path, mapping=mapping, transform=transform, crs=crs,
                         scaled=scaled, full=full, demo=False,
                         area=pixel_area_m2(transform, crs, oh))
        self._after_load()

    def load_demo(self):
        self.config(cursor="watch")
        self.update_idletasks()
        self.bands = make_demo_scene()
        self.config(cursor="")
        transform = from_origin(*DEMO_TRANSFORM_ORIGIN) if HAS_RASTERIO else None
        self.meta = dict(path=None, mapping={r: i + 1 for i, r in enumerate(ROLES)},
                         transform=transform, crs=DEMO_CRS, scaled=False, demo=True,
                         full=None, area=100.0)
        self._after_load()

    def _after_load(self):
        self.cache.clear()
        h, w = self.bands["Red"].shape
        m = self.meta
        lines = ["Demo scene (synthetic, 10 m)" if m["demo"] else f"File: {os.path.basename(m['path'])}",
                 f"Size: {w} x {h} px"]
        if m.get("full") and m["full"] != (w, h):
            lines[-1] += f"  (downsampled from {m['full'][0]} x {m['full'][1]})"
        if not m["demo"]:
            lines.append(f"CRS: {m['crs'].to_string() if m['crs'] else 'unknown'}")
            if m["scaled"]:
                lines.append("Values divided by 10000 -> reflectance")
            lines.append("Bands: " + ", ".join(f"{r}=#{m['mapping'][r]}" for r in ROLES if m['mapping'].get(r)))
        self.info.config(text="\n".join(lines))
        self._refresh_buttons()
        self.select("TCC")

    def _available(self, key):
        if self.bands is None:
            return False
        needs = COMPOSITES[key]["rgb"] if key in COMPOSITES else INDICES[key]["needs"]
        return all(r in self.bands for r in needs)

    def _refresh_buttons(self):
        for key, b in self.buttons.items():
            b.state(["!disabled"] if self._available(key) else ["disabled"])
            b.configure(style="Active.TButton" if key == self.current else "Index.TButton")
        loaded = ["!disabled"] if self.bands is not None else ["disabled"]
        self.png_btn.state(loaded)
        self.tif_btn.state(loaded if HAS_RASTERIO else ["disabled"])
        self.map_btn.state(["!disabled"] if self.meta.get("path") else ["disabled"])

    # ---------------- computing ----------------
    def select(self, key):
        if not self._available(key):
            return
        self.current = key
        self._refresh_buttons()
        self.render()

    def compute(self, key):
        if key not in self.cache:
            self.cache[key] = INDICES[key]["func"](self.bands).astype(np.float32)
        return self.cache[key]

    def composite(self, key):
        rgb = np.dstack([self.bands[r] for r in COMPOSITES[key]["rgb"]])
        out = np.empty_like(rgb)
        for i in range(3):
            band = rgb[..., i]
            lo, hi = np.nanpercentile(band, (2, 98))
            out[..., i] = np.clip((band - lo) / (hi - lo + 1e-9), 0, 1)
        out[np.isnan(out)] = 1.0
        return out

    def classify(self, key, arr):
        cls = np.digitize(arr, INDICES[key]["edges"]).astype(np.float32)
        cls[np.isnan(arr)] = np.nan
        return cls

    def _limits(self, key, arr):
        if self.auto_stretch.get():
            return tuple(np.nanpercentile(arr, (2, 98)))
        return INDICES[key]["vmin"], INDICES[key]["vmax"]

    # ---------------- drawing ----------------
    def render(self):
        self.fig.clear()
        key = self.current
        if self.bands is None or key is None:
            ax = self.fig.add_subplot(111)
            ax.set_axis_off()
            ax.text(0.5, 0.55, "Load a Sentinel-2 image", ha="center", fontsize=18, color=NAVY,
                    fontweight="bold", transform=ax.transAxes)
            ax.text(0.5, 0.47, "or click  'Load demo scene'  to begin", ha="center", fontsize=12,
                    color="#64748B", transform=ax.transAxes)
            self._write_stats([("h", "Welcome\n\n"),
                               ("", "1. Load an image (or the demo scene)\n"
                                    "2. Click a composite or an index\n"
                                    "3. Change the display mode\n"
                                    "4. Hover over the map to read pixel values\n"
                                    "5. Export your map\n")])
            self.canvas.draw_idle()
            return

        if key in COMPOSITES:
            ax = self.fig.add_subplot(111)
            ax.imshow(self.composite(key), interpolation="nearest")
            ax.set_title(COMPOSITES[key]["title"], fontsize=12, fontweight="bold", color=NAVY)
            ax.set_axis_off()
            self._write_stats([("h", COMPOSITES[key]["title"] + "\n\n"),
                               ("", COMPOSITES[key]["text"] + "\n")])
        else:
            arr = self.compute(key)
            vmin, vmax = self._limits(key, arr)
            mode = self.mode.get()
            if mode == "classified":
                self._draw_classes(self.fig.add_subplot(111), key, arr)
            elif mode == "side" and self._available("TCC"):
                ax1 = self.fig.add_subplot(121)
                ax2 = self.fig.add_subplot(122, sharex=ax1, sharey=ax1)
                ax1.imshow(self.composite("TCC"), interpolation="nearest")
                ax1.set_title("True Colour", fontsize=11, fontweight="bold", color=NAVY)
                ax1.set_axis_off()
                self._draw_index(ax2, key, arr, vmin, vmax, short=True)
            elif mode == "hist":
                gs = self.fig.add_gridspec(1, 2, width_ratios=[1.5, 1])
                self._draw_index(self.fig.add_subplot(gs[0]), key, arr, vmin, vmax, short=True)
                self._draw_hist(self.fig.add_subplot(gs[1]), key, arr)
            else:
                self._draw_index(self.fig.add_subplot(111), key, arr, vmin, vmax)
            self._index_stats(key, arr)

        self.canvas.draw_idle()
        self.toolbar.update()  # reset zoom history for the new map

    def _draw_index(self, ax, key, arr, vmin, vmax, short=False):
        info = INDICES[key]
        cmap = get_cmap(info["cmap"])
        cmap.set_bad(NODATA_COLOR)
        im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        cb = self.fig.colorbar(im, ax=ax, shrink=0.55 if short else 0.8, pad=0.02)
        cb.set_label(key, fontsize=10)
        ax.set_title(key if short else f"{key} - {info['name']}",
                     fontsize=11 if short else 12, fontweight="bold", color=NAVY)
        ax.set_axis_off()

    def _class_colors(self, key):
        return INDICES[key]["colors"]

    def _draw_classes(self, ax, key, arr):
        info = INDICES[key]
        n = len(info["labels"])
        colors = self._class_colors(key)
        cmap = ListedColormap(colors)
        cmap.set_bad(NODATA_COLOR)
        norm = BoundaryNorm(np.arange(-0.5, n + 0.5, 1), n)
        ax.imshow(self.classify(key, arr), cmap=cmap, norm=norm, interpolation="nearest")
        ax.set_title(f"{key} - classified", fontsize=12, fontweight="bold", color=NAVY)
        ax.set_axis_off()
        handles = [Patch(facecolor=colors[i], edgecolor="#475569", label=info["labels"][i])
                   for i in range(n)]
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.01),
                  ncol=min(n, 3), fontsize=9, frameon=False)

    def _draw_hist(self, ax, key, arr):
        v = arr[np.isfinite(arr)]
        lo, hi = np.percentile(v, (0.5, 99.5))
        ax.hist(v, bins=120, range=(lo, hi), color=TEAL, alpha=0.85)
        for e in INDICES[key]["edges"]:
            if lo <= e <= hi:
                ax.axvline(e, color=AMBER, ls="--", lw=1.4)
        ax.axvline(v.mean(), color=NAVY, lw=1.8, label=f"Mean = {v.mean():.2f}")
        ax.set_title("Histogram  (dashed = class thresholds)", fontsize=10, color=NAVY)
        ax.set_xlabel(f"{key} value")
        ax.set_ylabel("Number of pixels")
        ax.legend(fontsize=9, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)

    # ---------------- statistics panel ----------------
    def _write_stats(self, parts):
        self.stats.config(state="normal")
        self.stats.delete("1.0", "end")
        for tag, text in parts:
            self.stats.insert("end", text, tag or ())
        self.stats.config(state="disabled")

    def _index_stats(self, key, arr):
        info = INDICES[key]
        v = arr[np.isfinite(arr)]
        area = self.meta.get("area")
        parts = [("h", f"{key}\n"), ("", f"{info['name']}\n\n"),
                 ("s", "Formula\n"), ("", f"{info['formula']}\n"),
                 ("", f"Sentinel-2: {sentinel_formula(info['formula'])}\n\n"),
                 ("s", "Used for\n"), ("", info["use"] + "\n\n"),
                 ("s", "How to read it\n"), ("", info["tip"] + "\n\n"),
                 ("s", f"Statistics  ({v.size:,} pixels)\n"),
                 ("", f"  Min   {v.min():7.3f}\n  Max   {v.max():7.3f}\n"
                      f"  Mean  {v.mean():7.3f}\n  Std   {v.std():7.3f}\n\n"),
                 ("s", "Class breakdown\n")]
        counts = np.bincount(np.digitize(v, info["edges"]), minlength=len(info["labels"]))
        for label, c in zip(info["labels"], counts):
            line = f"  {label}\n      {100 * c / v.size:5.1f}%"
            if area:
                line += f"   {c * area / 1e4:,.1f} ha"
            parts.append(("", line + "\n"))
        self._write_stats(parts)

    # ---------------- export ----------------
    def save_png(self):
        if self.current is None:
            return
        path = filedialog.asksaveasfilename(defaultextension=".png", initialfile=f"{self.current}_map.png",
                                            filetypes=[("PNG image", "*.png")])
        if path:
            self.fig.savefig(path, dpi=200, facecolor="white")
            messagebox.showinfo("Saved", f"Map saved to\n{path}")

    def export_geotiff(self):
        if self.current is None:
            return
        if self.current in COMPOSITES:
            messagebox.showinfo("Export", "Select a spectral index to export as GeoTIFF.")
            return
        if self.meta.get("transform") is None:
            messagebox.showerror("Export", "This image has no georeferencing.")
            return
        key = self.current
        arr = self.compute(key)
        classified = self.mode.get() == "classified"
        name = f"{key}_classified.tif" if classified else f"{key}.tif"
        path = filedialog.asksaveasfilename(defaultextension=".tif", initialfile=name,
                                            filetypes=[("GeoTIFF", "*.tif")])
        if not path:
            return
        if classified:
            data = self.classify(key, arr)
            data = np.where(np.isnan(data), 255, data).astype("uint8")
            dtype, nodata = "uint8", 255
        else:
            data, dtype, nodata = arr.astype("float32"), "float32", np.nan
        h, w = data.shape
        profile = dict(driver="GTiff", height=h, width=w, count=1, dtype=dtype, nodata=nodata,
                       crs=self.meta["crs"], transform=self.meta["transform"], compress="deflate")
        try:
            with rasterio.open(path, "w", **profile) as dst:
                dst.write(data, 1)
                dst.set_band_description(1, key + (" classes" if classified else ""))
        except Exception as exc:
            messagebox.showerror("Export failed", str(exc))
            return
        messagebox.showinfo("Exported", f"GeoTIFF saved to\n{path}\n\nOpen it in QGIS to check.")

    def _quit(self):
        self.quit()
        self.destroy()


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--make-demo":
        save_demo_geotiff(sys.argv[2])
    else:
        IndexExplorer().mainloop()
