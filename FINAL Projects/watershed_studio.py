"""
Watershed Delineation Studio
============================
GeoAI Professional Program - Final Session Project
OrbitalIQ Analytics Private Limited

A Tkinter desktop app for hydrological analysis of a DEM:
  DEM -> Fill sinks -> Flow direction (D8) -> Flow accumulation
      -> Stream network (Strahler order) -> click a pour point -> Watershed

Every algorithm is written in plain NumPy (sections 2-3) so it can be read
and explained in class - no black-box GIS library.

Run:
    pip install numpy matplotlib rasterio
    python watershed_studio.py

Optional - create a demo DEM GeoTIFF:
    python watershed_studio.py --make-demo demo_dem.tif
"""

import os
import sys
import json
import time
import heapq
import threading
import traceback
from collections import deque
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import matplotlib

matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap, BoundaryNorm, LogNorm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

try:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import from_origin
    from rasterio.features import shapes
    from rasterio.warp import transform as warp_transform, transform_geom
    from affine import Affine
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


# ----------------------------------------------------------------------------
# 1. Configuration
# ----------------------------------------------------------------------------

NAVY, TEAL, TEAL_DARK = "#0F172A", "#164E63", "#0E3442"
AMBER, AMBER_LIGHT = "#F59E0B", "#FDE68A"
PANEL, TEXT = "#F1F5F9", "#1E293B"
STREAM_BLUE = "#1D4ED8"

MAX_SIDE = 1500   # larger DEMs are resampled on load to keep it fast in pure Python

# D8 flow directions in the ESRI code convention
#            E    SE    S    SW    W    NW    N    NE
D8_CODES = [1, 2, 4, 8, 16, 32, 64, 128]
D8_OFFSETS = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
D8_NAMES = ["E", "SE", "S", "SW", "W", "NW", "N", "NE"]
D8_ARROWS = ["→", "↘", "↓", "↙", "←", "↖", "↑", "↗"]
D8_COLORS = ["#e41a1c", "#ff7f00", "#ffd92f", "#a6d854", "#1b9e77", "#377eb8", "#7570b3", "#e7298a"]
WS_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
             "#a65628", "#f781bf", "#17becf", "#bcbd22", "#666666"]


# ----------------------------------------------------------------------------
# 2. Terrain functions
# ----------------------------------------------------------------------------

def hillshade(dem, dx, dy, azimuth=315.0, altitude=45.0):
    """Shaded relief, sun from the north-west at 45 degrees."""
    z = np.where(np.isnan(dem), np.nanmean(dem), dem)
    gr, gc = np.gradient(z, dy, dx)          # change along rows (south), columns (east)
    slope = np.pi / 2 - np.arctan(np.hypot(gr, gc))
    aspect = np.arctan2(-gr, gc)
    az, alt = np.radians(360 - azimuth + 90), np.radians(altitude)
    shade = np.sin(alt) * np.sin(slope) + np.cos(alt) * np.cos(slope) * np.cos(az - aspect)
    shade = np.clip(shade, 0, 1)
    shade[np.isnan(dem)] = np.nan
    return shade


def slope_degrees(dem, dx, dy):
    z = np.where(np.isnan(dem), np.nanmean(dem), dem)
    gr, gc = np.gradient(z, dy, dx)
    s = np.degrees(np.arctan(np.hypot(gr, gc)))
    s[np.isnan(dem)] = np.nan
    return s


# ----------------------------------------------------------------------------
# 3. Hydrology functions
# ----------------------------------------------------------------------------

def fill_depressions(dem, eps=1e-4):
    """
    Priority-Flood + epsilon (Barnes, Lehman & Mulla, 2014).

    Idea: start from the DEM edge like water rising from the sea. Always grow
    from the LOWEST cell reached so far. If a neighbour is lower than the
    current cell it is inside a pit, so raise it to the current level (+ a
    tiny epsilon so water still flows across the filled flat).
    """
    h, w = dem.shape
    W = w + 2
    zp = np.full((h + 2, W), np.nan)
    zp[1:-1, 1:-1] = dem
    closed = np.isnan(zp)                       # border + no-data are "sea"

    near = np.zeros_like(closed)                # valid cells touching the sea
    for dr, dc in D8_OFFSETS:
        near[1:-1, 1:-1] |= closed[1 + dr:h + 1 + dr, 1 + dc:w + 1 + dc]
    seeds = np.flatnonzero(near & ~closed)

    z = zp.ravel().tolist()                     # Python lists are fastest in loops
    cl = closed.ravel().tolist()
    offsets = [dr * W + dc for dr, dc in D8_OFFSETS]
    open_q = [(z[i], i) for i in seeds]
    heapq.heapify(open_q)
    for i in seeds:
        cl[i] = True
    pit_q = deque()
    push, pop = heapq.heappush, heapq.heappop

    while open_q or pit_q:
        if pit_q:
            c = pit_q.popleft()
            zc = z[c]
        else:
            zc, c = pop(open_q)
        for o in offsets:
            n = c + o
            if cl[n]:
                continue
            cl[n] = True
            if z[n] <= zc:                      # neighbour is in a pit -> raise it
                z[n] = zc + eps
                pit_q.append(n)
            else:
                push(open_q, (z[n], n))

    return np.array(z).reshape(h + 2, W)[1:-1, 1:-1]


def flow_direction(filled, dx, dy):
    """
    D8: each cell sends all its water to the ONE neighbour with the steepest
    downhill slope (drop / distance; diagonal distance is longer).
    Returns ESRI codes, the flat index of the downstream cell (-1 = outlet),
    and the direction index 0-7.
    """
    h, w = filled.shape
    zp = np.full((h + 2, w + 2), np.nan)
    zp[1:-1, 1:-1] = filled
    diag = np.hypot(dx, dy)
    drops = np.empty((8, h, w))
    for k, (dr, dc) in enumerate(D8_OFFSETS):
        dist = diag if (dr and dc) else (dx if dr == 0 else dy)
        nb = zp[1 + dr:h + 1 + dr, 1 + dc:w + 1 + dc]
        with np.errstate(invalid="ignore"):
            d = (filled - nb) / dist
        d[np.isnan(d)] = -np.inf
        drops[k] = d

    k = np.argmax(drops, axis=0)
    has_down = np.take_along_axis(drops, k[None], 0)[0] > 0
    codes = np.where(has_down, np.array(D8_CODES)[k], 0).astype(np.uint8)
    codes[np.isnan(filled)] = 0
    rows, cols = np.indices((h, w))
    dr = np.array([o[0] for o in D8_OFFSETS])[k]
    dc = np.array([o[1] for o in D8_OFFSETS])[k]
    down = np.where(has_down, (rows + dr) * w + (cols + dc), -1).ravel()
    return codes, down, k


def flow_accumulation(filled, down):
    """
    Visit cells from highest to lowest; each cell passes its total to the
    cell downstream. Result = number of cells draining through each cell
    (including itself), so area = accumulation x cell area.
    """
    valid = np.isfinite(filled).ravel()
    order = np.argsort(-filled.ravel(), kind="stable")
    order = order[valid[order]]                 # high -> low, valid cells only
    acc = valid.astype(np.float64).tolist()
    dn = down.tolist()
    for i in order.tolist():
        d = dn[i]
        if d >= 0:
            acc[d] += acc[i]
    return np.array(acc).reshape(filled.shape), order


def strahler_order(stream_flat, down, order_desc):
    """
    Strahler: headwater streams are order 1. When two streams of the same
    order meet, the order increases by one; otherwise the higher order wins.
    """
    n = len(down)
    so, maxin, cnt = [0] * n, [0] * n, [0] * n
    st = stream_flat.tolist()
    dn = down.tolist() if isinstance(down, np.ndarray) else down
    for i in order_desc[stream_flat[order_desc]].tolist():
        m = maxin[i]
        o = 1 if m == 0 else (m + 1 if cnt[i] >= 2 else m)
        so[i] = o
        d = dn[i]
        if d >= 0 and st[d]:
            if o > maxin[d]:
                maxin[d], cnt[d] = o, 1
            elif o == maxin[d]:
                cnt[d] += 1
    return np.array(so, dtype=np.uint8)


def upstream_area(order_desc, down_list, pour):
    """
    Watershed = every cell whose flow path passes through the pour point.
    Walk from low to high: a cell is inside if its downstream cell is inside.
    """
    lab = bytearray(len(down_list))
    lab[pour] = 1
    asc = order_desc[::-1]
    start = int(np.flatnonzero(asc == pour)[0]) + 1
    for i in asc[start:].tolist():
        d = down_list[i]
        if d >= 0 and lab[d]:
            lab[i] = 1
    return np.frombuffer(bytes(lab), dtype=np.uint8).astype(bool)


# ----------------------------------------------------------------------------
# 4. Synthetic demo DEM
# ----------------------------------------------------------------------------

DEMO_ORIGIN = (376800, 2246000, 30, 30)   # 30 m cells, UTM 45N near Bhubaneswar
DEMO_CRS = "EPSG:32645"


def make_demo_dem(size=500, seed=21):
    rng = np.random.default_rng(seed)
    fy = np.fft.fftfreq(size)[:, None]
    fx = np.fft.fftfreq(size)[None, :]
    f = np.hypot(fx, fy)
    f[0, 0] = 1
    spec = (rng.standard_normal((size, size)) + 1j * rng.standard_normal((size, size))) / f ** 1.6
    spec[0, 0] = 0
    noise = np.real(np.fft.ifft2(spec))
    noise = (noise - noise.mean()) / noise.std()

    y, x = np.mgrid[0:size, 0:size] / size
    dem = 60 + 25 * noise + 320 * (1 - y) ** 1.2                        # hills sloping south
    dem += 160 * np.exp(-((x - 0.25) ** 2 + (y - 0.22) ** 2) / 0.02)    # mountain
    dem += 110 * np.exp(-((x - 0.78) ** 2 + (y - 0.30) ** 2) / 0.015)   # second hill
    for _ in range(30):                                                 # small pits
        cx, cy = rng.uniform(0.08, 0.92, 2)
        dem -= rng.uniform(3, 10) * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / rng.uniform(3e-5, 1.5e-4))
    return np.round(dem, 1).astype(np.float64)


def save_demo_dem(path):
    if not HAS_RASTERIO:
        sys.exit("rasterio is required: pip install rasterio")
    dem = make_demo_dem().astype("float32")
    with rasterio.open(path, "w", driver="GTiff", height=dem.shape[0], width=dem.shape[1],
                       count=1, dtype="float32", crs=DEMO_CRS, nodata=-9999,
                       transform=from_origin(*DEMO_ORIGIN), compress="deflate") as dst:
        dst.write(dem, 1)
    print(f"Demo DEM written to {path}")


def get_cmap(name):
    try:
        cmap = matplotlib.colormaps[name]
    except AttributeError:
        cmap = matplotlib.cm.get_cmap(name)
    return cmap.copy()


# ----------------------------------------------------------------------------
# 5. The application
# ----------------------------------------------------------------------------

VIEWS = {   # view key: (button label, processing steps it needs)
    "dem":       ("DEM (elevation)", []),
    "hillshade": ("Hillshade", []),
    "slope":     ("Slope", []),
    "fill":      ("1. Fill sinks", ["fill"]),
    "fdir":      ("2. Flow direction", ["fill", "fdir"]),
    "facc":      ("3. Flow accumulation", ["fill", "fdir", "acc"]),
    "streams":   ("4. Stream network", ["fill", "fdir", "acc", "streams"]),
    "watershed": ("5. Watersheds", ["fill", "fdir", "acc", "streams"]),
}
STEP_MESSAGES = {
    "fill": "Filling sinks (Priority-Flood)...",
    "fdir": "Computing D8 flow direction...",
    "acc": "Computing flow accumulation...",
    "streams": "Extracting streams and Strahler order...",
}


class WatershedStudio(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Watershed Delineation Studio - OrbitalIQ GeoAI")
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"{min(1440, sw - 40)}x{min(860, sh - 80)}+10+10")
        self.minsize(1000, 600)
        self.configure(bg=PANEL)

        self._reset_data()
        self.view = None
        self.buttons = {}
        self.busy = False
        self.pick_mode = False
        self.map_ax = None
        self.thr_km2 = tk.DoubleVar(value=1.0)
        self.snap_px = tk.IntVar(value=5)

        self._style()
        self._build_header()
        self._build_body()
        self._refresh_buttons()
        self.render()
        self.canvas.mpl_connect("button_press_event", self._on_click)
        self.protocol("WM_DELETE_WINDOW", self._quit)

    def _reset_data(self):
        self.dem = None
        self.meta = {}
        self.hs = self.slope = None
        self.filled = self.fdir = self.down = self.dir_idx = None
        self.acc = self.order = self.down_list = self.cell_len = None
        self.stream_order = None
        self.stream_thr = None
        self.ws_labels = None
        self.ws_points = []
        self.timings = {}

    # ---------------- layout ----------------
    def _style(self):
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure("Side.TFrame", background=PANEL)
        s.configure("Side.TLabel", background=PANEL, foreground=TEXT, font=("Segoe UI", 10))
        s.configure("Info.TLabel", background=PANEL, foreground="#475569", font=("Segoe UI", 9))
        s.configure("Section.TLabel", background=PANEL, foreground=TEAL, font=("Segoe UI", 10, "bold"))
        s.configure("Status.TLabel", background=NAVY, foreground="white", font=("Segoe UI", 10))
        s.configure("TButton", font=("Segoe UI", 10), padding=6)
        s.configure("Primary.TButton", background=TEAL, foreground="white", font=("Segoe UI", 10, "bold"))
        s.map("Primary.TButton", background=[("active", TEAL_DARK), ("disabled", "#94A3B8")])
        for name, bg in (("View.TButton", "white"), ("Active.TButton", AMBER)):
            s.configure(name, background=bg, foreground=NAVY, font=("Segoe UI", 10, "bold"), padding=5)
            s.map(name, background=[("disabled", "#E2E8F0"),
                                    ("active", AMBER_LIGHT if bg == "white" else AMBER)],
                  foreground=[("disabled", "#94A3B8")])
        s.configure("Pick.TButton", background="#DC2626", foreground="white",
                    font=("Segoe UI", 10, "bold"), padding=6)
        s.map("Pick.TButton", background=[("active", "#B91C1C"), ("disabled", "#94A3B8")])
        s.configure("Amber.Horizontal.TProgressbar", background=AMBER, troughcolor=TEAL_DARK)

    def _build_header(self):
        bar = tk.Frame(self, bg=NAVY, height=58)
        bar.pack(fill="x")
        tk.Label(bar, text="Watershed Delineation Studio", bg=NAVY, fg="white",
                 font=("Segoe UI", 16, "bold")).pack(side="left", padx=(18, 10), pady=12)
        tk.Label(bar, text="GeoAI Professional Program  |  OrbitalIQ Analytics", bg=NAVY,
                 fg=AMBER, font=("Segoe UI", 10)).pack(side="left", pady=12)
        tk.Frame(self, bg=AMBER, height=3).pack(fill="x")

    def _section(self, parent, text):
        ttk.Label(parent, text=text, style="Section.TLabel").pack(anchor="w", pady=(10, 3))

    def _scrollable_sidebar(self, body):
        holder = ttk.Frame(body, style="Side.TFrame")
        holder.pack(side="left", fill="y", padx=(10, 4), pady=4)
        cv = tk.Canvas(holder, bg=PANEL, highlightthickness=0, width=255)
        bar = ttk.Scrollbar(holder, orient="vertical", command=cv.yview)
        cv.configure(yscrollcommand=bar.set)
        bar.pack(side="right", fill="y")
        cv.pack(side="left", fill="y")
        side = ttk.Frame(cv, style="Side.TFrame")
        cv.create_window((0, 0), window=side, anchor="nw", width=250)
        side.bind("<Configure>", lambda e: cv.configure(scrollregion=cv.bbox("all")))

        def wheel(event):
            cv.yview_scroll(-1 if (event.num == 4 or event.delta > 0) else 1, "units")
        for w in (cv, side):
            w.bind("<Enter>", lambda e: [cv.bind_all(s, wheel) for s in ("<MouseWheel>", "<Button-4>", "<Button-5>")])
            w.bind("<Leave>", lambda e: [cv.unbind_all(s) for s in ("<MouseWheel>", "<Button-4>", "<Button-5>")])
        return side

    def _view_button(self, parent, key):
        b = ttk.Button(parent, text=VIEWS[key][0], style="View.TButton",
                       command=lambda k=key: self.select(k))
        b.pack(fill="x", pady=2)
        self.buttons[key] = b

    def _build_body(self):
        # status bar (bottom)
        status = tk.Frame(self, bg=NAVY)
        status.pack(side="bottom", fill="x")
        self.status = ttk.Label(status, text="Ready.", style="Status.TLabel")
        self.status.pack(side="left", padx=14, pady=5)
        self.progress = ttk.Progressbar(status, mode="indeterminate", length=180,
                                        style="Amber.Horizontal.TProgressbar")
        self.progress.pack(side="right", padx=14, pady=5)

        body = ttk.Frame(self, style="Side.TFrame")
        body.pack(fill="both", expand=True)
        side = self._scrollable_sidebar(body)

        self._section(side, "LOAD DEM")
        self.open_btn = ttk.Button(side, text="Open DEM GeoTIFF", style="Primary.TButton",
                                   command=self.open_dem)
        self.open_btn.pack(fill="x", pady=2)
        self.demo_btn = ttk.Button(side, text="Load demo DEM", command=self.load_demo)
        self.demo_btn.pack(fill="x", pady=2)
        self.info = ttk.Label(side, style="Info.TLabel", wraplength=235, justify="left",
                              text="No DEM loaded.")
        self.info.pack(anchor="w", pady=(6, 0))

        self._section(side, "TERRAIN")
        for key in ("dem", "hillshade", "slope"):
            self._view_button(side, key)

        self._section(side, "HYDROLOGY  (step by step)")
        for key in ("fill", "fdir", "facc", "streams"):
            self._view_button(side, key)

        row = ttk.Frame(side, style="Side.TFrame")
        row.pack(fill="x", pady=(4, 0))
        ttk.Label(row, text="Stream threshold (km²)", style="Side.TLabel").pack(side="left")
        ttk.Spinbox(row, from_=0.05, to=500, increment=0.25, width=6, textvariable=self.thr_km2,
                    format="%.2f").pack(side="right")
        self.thr_btn = ttk.Button(side, text="Apply threshold", command=self.apply_threshold)
        self.thr_btn.pack(fill="x", pady=2)

        self._section(side, "WATERSHED")
        self._view_button(side, "watershed")
        self.pick_btn = ttk.Button(side, text="Pick pour point on map", style="Primary.TButton",
                                   command=self.toggle_pick)
        self.pick_btn.pack(fill="x", pady=2)
        row = ttk.Frame(side, style="Side.TFrame")
        row.pack(fill="x", pady=(2, 0))
        ttk.Label(row, text="Snap to stream within (px)", style="Side.TLabel").pack(side="left")
        ttk.Spinbox(row, from_=0, to=30, increment=1, width=4,
                    textvariable=self.snap_px).pack(side="right")
        self.clear_btn = ttk.Button(side, text="Clear watersheds", command=self.clear_watersheds)
        self.clear_btn.pack(fill="x", pady=2)

        self._section(side, "EXPORT")
        self.png_btn = ttk.Button(side, text="Save map as PNG", command=self.save_png)
        self.png_btn.pack(fill="x", pady=2)
        self.tif_btn = ttk.Button(side, text="Export layer as GeoTIFF", command=self.export_geotiff)
        self.tif_btn.pack(fill="x", pady=2)
        self.json_btn = ttk.Button(side, text="Export watersheds (GeoJSON)", command=self.export_geojson)
        self.json_btn.pack(fill="x", pady=2)

        right = ttk.Frame(body, style="Side.TFrame", width=340)
        right.pack(side="right", fill="y", padx=(8, 14), pady=4)
        right.pack_propagate(False)
        ttk.Label(right, text="ABOUT & STATISTICS", style="Section.TLabel").pack(anchor="w", pady=(12, 4))
        self.stats = tk.Text(right, wrap="word", font=("Consolas", 10), bg="white", fg=TEXT,
                             relief="flat", padx=10, pady=10, highlightthickness=1,
                             highlightbackground="#CBD5E1")
        self.stats.pack(fill="both", expand=True, pady=(0, 10))
        self.stats.tag_configure("h", font=("Segoe UI", 12, "bold"), foreground=NAVY)
        self.stats.tag_configure("s", font=("Segoe UI", 10, "bold"), foreground=TEAL)

        centre = ttk.Frame(body, style="Side.TFrame")
        centre.pack(side="left", fill="both", expand=True, pady=10)
        self.fig = Figure(figsize=(8, 6), dpi=100, facecolor="white", constrained_layout=True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=centre)
        self.toolbar = NavigationToolbar2Tk(self.canvas, centre, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(side="bottom", fill="x")
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    # ---------------- background work ----------------
    def _run(self, tasks, done):
        """Run [(message, function), ...] in a thread so the window stays responsive."""
        if self.busy:
            return
        self.busy = True
        self._error = None
        self._msg = tasks[0][0] if tasks else ""
        self._refresh_buttons()
        self.progress.start(12)

        def work():
            try:
                for msg, func in tasks:
                    self._msg = msg
                    func()
            except Exception:
                self._error = traceback.format_exc()

        self._thread = threading.Thread(target=work, daemon=True)
        self._thread.start()
        self._poll(done)

    def _poll(self, done):
        self.status.config(text=self._msg)
        if self._thread.is_alive():
            self.after(100, lambda: self._poll(done))
            return
        self.busy = False
        self.progress.stop()
        self._refresh_buttons()
        if self._error:
            self.status.config(text="Error - see message.")
            messagebox.showerror("Processing error", self._error[-1500:])
            return
        done()

    def _timed(self, name, func):
        def wrapper():
            t = time.perf_counter()
            func()
            self.timings[name] = time.perf_counter() - t
        return wrapper

    # ---------------- processing steps ----------------
    def _done(self, step):
        return {"fill": self.filled, "fdir": self.fdir, "acc": self.acc,
                "streams": self.stream_order}[step] is not None

    def _step_fill(self):
        self.filled = fill_depressions(self.dem)

    def _step_fdir(self):
        dx, dy = self.meta["dx"], self.meta["dy"]
        self.fdir, self.down, self.dir_idx = flow_direction(self.filled, dx, dy)
        dists = np.array([np.hypot(dx, dy) if (r and c) else (dx if r == 0 else dy)
                          for r, c in D8_OFFSETS])
        self.cell_len = np.where(self.fdir > 0, dists[self.dir_idx], dx)

    def _step_acc(self):
        self.acc, self.order = flow_accumulation(self.filled, self.down)
        self.down_list = self.down.tolist()

    def _step_streams(self):
        cells = self._pending_thr                    # read on the main thread (Tk rule)
        flat = (self.acc >= cells).ravel()
        self.stream_order = strahler_order(flat, self.down, self.order).reshape(self.dem.shape)
        self.stream_thr = cells

    def _tasks(self, steps):
        if "streams" in steps:
            self._pending_thr = self._threshold_cells()
        return [(STEP_MESSAGES[s], self._timed(s, getattr(self, "_step_" + s))) for s in steps]

    def _threshold_cells(self):
        try:
            km2 = max(float(self.thr_km2.get()), 0.001)
        except (tk.TclError, ValueError):
            km2 = 1.0
        return max(2, int(round(km2 * 1e6 / self.meta["area"])))

    # ---------------- loading ----------------
    def open_dem(self):
        if not HAS_RASTERIO:
            messagebox.showerror("rasterio missing", "Install it first:\n\npip install rasterio")
            return
        path = filedialog.askopenfilename(title="Open DEM",
                                          filetypes=[("GeoTIFF", "*.tif *.tiff *.TIF"), ("All files", "*.*")])
        if path:
            self._load_file(path)

    def _load_file(self, path):
        try:
            with rasterio.open(path) as src:
                scale = max(src.width, src.height) / MAX_SIDE
                if scale > 1:
                    oh, ow = int(round(src.height / scale)), int(round(src.width / scale))
                else:
                    oh, ow = src.height, src.width
                dem = src.read(1, out_shape=(oh, ow), masked=True,
                               resampling=Resampling.bilinear if scale > 1 else Resampling.nearest)
                transform = src.transform * Affine.scale(src.width / ow, src.height / oh)
                crs, full = src.crs, (src.width, src.height)
            dem = dem.astype("float64").filled(np.nan)
            dem[(dem < -1000) | (dem > 9000)] = np.nan          # SRTM voids etc.
            if not np.isfinite(dem).any():
                raise ValueError("The DEM has no valid elevation values.")
        except Exception as exc:
            messagebox.showerror("Could not read DEM", str(exc))
            return

        a, e = transform.a, abs(transform.e)
        if crs is not None and crs.is_geographic:
            lat = transform.f + transform.e * oh / 2
            dx, dy = a * 111320 * np.cos(np.radians(lat)), e * 110574
        else:
            dx, dy = a, e
        self._set_dem(dem, dict(path=path, transform=transform, crs=crs, full=full,
                                dx=dx, dy=dy, area=dx * dy, demo=False))

    def load_demo(self):
        transform = from_origin(*DEMO_ORIGIN) if HAS_RASTERIO else None
        self._set_dem(make_demo_dem(), dict(path=None, transform=transform, crs=DEMO_CRS,
                                            full=None, dx=30.0, dy=30.0, area=900.0, demo=True))

    def _set_dem(self, dem, meta):
        self._reset_data()
        self.dem, self.meta = dem, meta
        self.hs = hillshade(dem, meta["dx"], meta["dy"])
        self.slope = slope_degrees(dem, meta["dx"], meta["dy"])
        self.ws_labels = np.zeros(dem.shape, np.uint8)

        total_km2 = np.isfinite(dem).sum() * meta["area"] / 1e6
        self.thr_km2.set(round(max(0.1, min(1.0, total_km2 / 200)), 2))

        h, w = dem.shape
        lines = ["Demo DEM (synthetic, 30 m)" if meta["demo"] else f"File: {os.path.basename(meta['path'])}",
                 f"Size: {w} x {h} cells"]
        if meta.get("full") and meta["full"] != (w, h):
            lines[-1] += f" (resampled from {meta['full'][0]} x {meta['full'][1]})"
        if not meta["demo"]:
            lines.append(f"CRS: {meta['crs'].to_string() if meta['crs'] else 'unknown'}")
        lines += [f"Cell size: {meta['dx']:.1f} x {meta['dy']:.1f} m",
                  f"Area: {total_km2:,.1f} km²",
                  f"Elevation: {np.nanmin(dem):.0f} - {np.nanmax(dem):.0f} m"]
        self.info.config(text="\n".join(lines))
        self.pick_mode = False
        self.status.config(text="DEM loaded. Work through the hydrology steps 1 to 5.")
        self.select("dem")

    # ---------------- interaction ----------------
    def select(self, view, keep_zoom=False):
        if self.dem is None or self.busy:
            return
        missing = [s for s in VIEWS[view][1] if not self._done(s)]
        if missing:
            self._run(self._tasks(missing), lambda: self._show(view, keep_zoom))
        else:
            self._show(view, keep_zoom)

    def _show(self, view, keep_zoom=False):
        same = keep_zoom and self.view == view and self.map_ax is not None
        lims = (self.map_ax.get_xlim(), self.map_ax.get_ylim()) if same else None
        self.view = view
        if view != "watershed" and self.pick_mode:
            self.pick_mode = False
        self._refresh_buttons()
        self.render(lims)
        if not self.pick_mode:
            self.status.config(text=f"Showing: {VIEWS[view][0]}")

    def apply_threshold(self):
        if self.dem is None or self.acc is None:
            return
        self.stream_order = None
        view = self.view if self.view in ("streams", "watershed") else "streams"
        self.select(view, keep_zoom=True)

    def toggle_pick(self):
        if self.dem is None or self.busy:
            return
        self.pick_mode = not self.pick_mode
        if self.pick_mode:
            self.select("watershed", keep_zoom=self.view == "watershed")
            self.after(50, lambda: self.status.config(
                text="Click on a stream (blue line) to place a pour point. Zoom first if needed."))
        self._refresh_buttons()

    def clear_watersheds(self):
        if self.ws_labels is not None:
            self.ws_labels[:] = 0
            self.ws_points = []
            if self.view == "watershed":
                self.render((self.map_ax.get_xlim(), self.map_ax.get_ylim()))

    def _on_click(self, event):
        if (not self.pick_mode or self.busy or event.button != 1 or self.toolbar.mode
                or event.inaxes is not self.map_ax or event.xdata is None):
            return
        h, w = self.dem.shape
        r, c = int(round(event.ydata)), int(round(event.xdata))
        if not (0 <= r < h and 0 <= c < w):
            return
        if len(self.ws_points) >= len(WS_COLORS):
            messagebox.showinfo("Limit", f"Up to {len(WS_COLORS)} watersheds. Click 'Clear watersheds'.")
            return

        # Snap: move the click to the highest-accumulation cell nearby
        rad = max(0, int(self.snap_px.get()))
        r0, r1, c0, c1 = max(0, r - rad), min(h, r + rad + 1), max(0, c - rad), min(w, c + rad + 1)
        win = np.where(np.isfinite(self.acc[r0:r1, c0:c1]), self.acc[r0:r1, c0:c1], -1)
        rr, cc = np.unravel_index(np.argmax(win), win.shape)
        r, c = r0 + rr, c0 + cc
        if not np.isfinite(self.dem[r, c]):
            return

        wid = len(self.ws_points) + 1
        result = {}

        def delineate():
            result["mask"] = upstream_area(self.order, self.down_list, r * w + c).reshape(h, w)

        def done():
            self.ws_labels[result["mask"]] = wid
            self.ws_points.append(dict(id=wid, r=int(r), c=int(c), mask=result["mask"]))
            self._show("watershed", keep_zoom=True)
            area = result["mask"].sum() * self.meta["area"] / 1e6
            self.status.config(text=f"Watershed {wid}: {area:,.2f} km². Click another point, "
                                    f"or press 'Pick pour point' again to stop.")

        self._run([("Delineating watershed...", delineate)], done)

    def _refresh_buttons(self):
        loaded = self.dem is not None and not self.busy
        for key, b in self.buttons.items():
            b.state(["!disabled"] if loaded else ["disabled"])
            b.configure(style="Active.TButton" if key == self.view else "View.TButton")
        for b in (self.thr_btn, self.png_btn, self.clear_btn):
            b.state(["!disabled"] if loaded else ["disabled"])
        self.tif_btn.state(["!disabled"] if loaded and HAS_RASTERIO else ["disabled"])
        self.json_btn.state(["!disabled"] if loaded and HAS_RASTERIO else ["disabled"])
        for b in (self.open_btn, self.demo_btn):
            b.state(["disabled"] if self.busy else ["!disabled"])
        self.pick_btn.state(["!disabled"] if loaded else ["disabled"])
        self.pick_btn.configure(style="Pick.TButton" if self.pick_mode else "Primary.TButton",
                                text="Stop picking" if self.pick_mode else "Pick pour point on map")
        self.canvas.get_tk_widget().config(cursor="crosshair" if self.pick_mode else "")

    # ---------------- drawing ----------------
    def render(self, lims=None):
        self.fig.clear()
        self.map_ax = None
        if self.dem is None or self.view is None:
            ax = self.fig.add_subplot(111)
            ax.set_axis_off()
            ax.text(0.5, 0.55, "Load a DEM", ha="center", fontsize=18, color=NAVY,
                    fontweight="bold", transform=ax.transAxes)
            ax.text(0.5, 0.47, "or click  'Load demo DEM'  to begin", ha="center", fontsize=12,
                    color="#64748B", transform=ax.transAxes)
            self._write([("h", "Welcome\n\n"),
                         ("", "Hydrology workflow:\n\n"
                              "DEM\n ↓  1. Fill sinks\n ↓  2. Flow direction (D8)\n"
                              " ↓  3. Flow accumulation\n ↓  4. Stream network\n"
                              " ↓  5. Click a pour point\nWATERSHED\n\n"
                              "Each step uses the result of the one before it.")])
            self.canvas.draw_idle()
            return

        if self.view == "watershed" and self.ws_points:
            gs = self.fig.add_gridspec(1, 2, width_ratios=[1.75, 1])
            ax = self.fig.add_subplot(gs[0])
            hax = self.fig.add_subplot(gs[1])
        else:
            ax, hax = self.fig.add_subplot(111), None
        self.map_ax = ax
        getattr(self, "_draw_" + self.view)(ax)
        if hax is not None:
            self._draw_hypsometric(hax, self.ws_points[-1]["id"])
        ax.set_axis_off()
        if lims:
            ax.set_xlim(lims[0])
            ax.set_ylim(lims[1])
        self.canvas.draw_idle()
        if not lims:
            self.toolbar.update()

    def _title(self, ax, text):
        ax.set_title(text, fontsize=12, fontweight="bold", color=NAVY)

    def _base(self, ax, terrain_alpha=0.0):
        ax.imshow(self.hs, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        if terrain_alpha:
            ax.imshow(self.dem, cmap="terrain", alpha=terrain_alpha, interpolation="nearest",
                      vmin=np.nanmin(self.dem), vmax=np.nanmax(self.dem) * 1.15)

    def _draw_dem(self, ax):
        ax.imshow(self.hs, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        im = ax.imshow(self.dem, cmap="terrain", alpha=0.7, interpolation="nearest",
                       vmin=np.nanmin(self.dem), vmax=np.nanmax(self.dem) * 1.15)
        self.fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02).set_label("Elevation (m)")
        self._title(ax, "Digital Elevation Model")
        v = self.dem[np.isfinite(self.dem)]
        self._write([("h", "Digital Elevation Model\n\n"),
                     ("", "A DEM stores the ground height (metres above sea level) in every cell. "
                          "It is the only input needed for watershed analysis: water always flows downhill.\n\n"),
                     ("s", "Statistics\n"),
                     ("", f"  Min    {v.min():8.1f} m\n  Max    {v.max():8.1f} m\n"
                          f"  Mean   {v.mean():8.1f} m\n  Relief {v.max() - v.min():8.1f} m\n")])

    def _draw_hillshade(self, ax):
        ax.imshow(self.hs, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        self._title(ax, "Hillshade (sun from NW, 45°)")
        self._write([("h", "Hillshade\n\n"),
                     ("", "Simulates sunlight falling on the terrain (azimuth 315°, altitude 45°). "
                          "Slopes facing the sun are bright, slopes facing away are dark.\n\n"
                          "It does not change the data - it only makes valleys and ridges easy "
                          "to see. Look for the valleys: that is where streams will appear.\n")])

    def _draw_slope(self, ax):
        im = ax.imshow(self.slope, cmap="YlOrRd", vmin=0,
                       vmax=np.nanpercentile(self.slope, 99), interpolation="nearest")
        self.fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02).set_label("Slope (degrees)")
        self._title(ax, "Slope")
        v = self.slope[np.isfinite(self.slope)]
        self._write([("h", "Slope\n\n"),
                     ("", "Steepness of the terrain, from the elevation change between neighbouring cells.\n\n"
                          "Steep slopes produce fast runoff and erosion; flat areas store water.\n\n"),
                     ("s", "Statistics\n"),
                     ("", f"  Mean  {v.mean():6.1f}°\n  Max   {v.max():6.1f}°\n"
                          f"  Flat (<2°)    {100 * (v < 2).mean():5.1f}%\n"
                          f"  Steep (>15°)  {100 * (v > 15).mean():5.1f}%\n")])

    def _draw_fill(self, ax):
        depth = self.filled - self.dem
        shown = np.where(depth > 0.05, depth, np.nan)
        self._base(ax)
        im = ax.imshow(shown, cmap="plasma", interpolation="nearest",
                       vmin=0, vmax=max(0.1, np.nanpercentile(shown, 99)) if np.isfinite(shown).any() else 1)
        self.fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02).set_label("Fill depth (m)")
        self._title(ax, "Fill sinks - where the DEM was raised")
        d = depth[np.isfinite(depth)]
        filled = d > 0.05
        vol = d[filled].sum() * self.meta["area"]
        self._write([("h", "Step 1 - Fill sinks\n\n"),
                     ("", "A sink is a cell (or group of cells) lower than all its neighbours. "
                          "Water flowing in has no way out, so the drainage network breaks.\n\n"
                          "Most sinks are DEM errors. Filling raises each sink up to its spill "
                          "point so that every cell can drain to the edge of the map.\n\n"),
                     ("s", "Algorithm\n"),
                     ("", "Priority-Flood (Barnes et al., 2014): flood inwards from the edges, always "
                          "from the lowest cell reached so far.\n\n"),
                     ("s", "Result\n"),
                     ("", f"  Cells filled  {filled.sum():,} ({100 * filled.mean():.2f}%)\n"
                          f"  Max depth     {d.max():.2f} m\n"
                          f"  Mean depth    {d[filled].mean() if filled.any() else 0:.2f} m\n"
                          f"  Volume        {vol:,.0f} m³\n"
                          f"  Time          {self.timings.get('fill', 0):.1f} s\n")])

    def _draw_fdir(self, ax):
        cats = np.where(self.fdir > 0, self.dir_idx, np.nan).astype(float)
        cats[np.isnan(self.dem)] = np.nan
        cmap = ListedColormap(D8_COLORS)
        cmap.set_bad("#FFFFFF")
        ax.imshow(cats, cmap=cmap, norm=BoundaryNorm(np.arange(-0.5, 8.5), 8), interpolation="nearest")
        h, w = self.dem.shape
        step = max(6, max(h, w) // 45)               # arrows on a coarse grid
        rr, cc = np.mgrid[step // 2:h:step, step // 2:w:step]
        k = self.dir_idx[rr, cc]
        ok = self.fdir[rr, cc] > 0
        dr = np.array([o[0] for o in D8_OFFSETS])[k]
        dc = np.array([o[1] for o in D8_OFFSETS])[k]
        length = step * 0.6 / np.hypot(dr, dc).clip(1)
        ax.quiver(cc[ok], rr[ok], (dc * length)[ok], (dr * length)[ok], angles="xy",
                  scale_units="xy", scale=1, width=0.0025, color=NAVY)
        handles = [Patch(facecolor=D8_COLORS[i], label=f"{D8_ARROWS[i]} {D8_NAMES[i]} ({D8_CODES[i]})")
                   for i in range(8)]
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=4,
                  fontsize=9, frameon=False)
        self._title(ax, "Flow direction (D8)")
        valid = np.isfinite(self.dem)
        counts = [(self.fdir[valid] == c).mean() * 100 for c in D8_CODES]
        table = "".join(f"  {D8_ARROWS[i]} {D8_NAMES[i]:<3} code {D8_CODES[i]:>3}   {counts[i]:5.1f}%\n"
                        for i in range(8))
        self._write([("h", "Step 2 - Flow direction (D8)\n\n"),
                     ("", "Each cell looks at its 8 neighbours and sends ALL its water to the one "
                          "with the steepest downhill slope:\n\n  slope = drop / distance\n\n"
                          "Diagonal neighbours are farther away (√2 × cell size).\n\n"),
                     ("s", "Direction codes (ESRI)\n"),
                     ("", "  32  64  128\n  16   X    1\n   8   4    2\n\n"),
                     ("s", "Share of cells\n"), ("", table),
                     ("", f"\n  Outlets (flow off map): {int((self.fdir[valid] == 0).sum()):,}\n")])

    def _draw_facc(self, ax):
        cmap = get_cmap("Blues")
        cmap.set_bad("#FFFFFF")
        im = ax.imshow(self.acc, cmap=cmap, norm=LogNorm(vmin=1, vmax=np.nanmax(self.acc)),
                       interpolation="nearest")
        self.fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02).set_label("Upstream cells (log scale)")
        self._title(ax, "Flow accumulation")
        a = self.acc[np.isfinite(self.dem)]
        km2 = self.meta["area"] / 1e6
        self._write([("h", "Step 3 - Flow accumulation\n\n"),
                     ("", "Every cell starts with 1 unit of water (rain). Following the flow "
                          "directions from the highest cell to the lowest, each cell passes its "
                          "total to the next cell downstream.\n\n"
                          "The value of a cell = number of cells that drain through it. Ridges have "
                          "low values; rivers have very high values.\n\n"
                          "Contributing area = accumulation × cell area.\n\n"),
                     ("s", "Result\n"),
                     ("", f"  Max accumulation  {a.max():,.0f} cells\n"
                          f"  = drainage area   {a.max() * km2:,.2f} km²\n"
                          f"  Cells with 1      {100 * (a <= 1).mean():.1f}% (ridges)\n"
                          f"  Time              {self.timings.get('acc', 0):.1f} s\n\n"
                          "Tip: hover over the map to read the value of any cell.\n\n"
                          "Note: straight parallel lines across a filled area (lake, flat "
                          "floodplain) are a known D8 artefact - there is no slope to follow there.\n")])

    def _stream_overlay(self, ax, alpha=1.0, single_color=False):
        so = self.stream_order.astype(float)
        if max(so.shape) > 500:                       # thicken lines so they stay visible
            p = np.pad(so, 1)
            h, w = so.shape
            so = np.maximum.reduce([p[1 + dr:h + 1 + dr, 1 + dc:w + 1 + dc]
                                    for dr in (-1, 0, 1) for dc in (-1, 0, 1)])
        so[so == 0] = np.nan
        n = int(self.stream_order.max())
        if single_color:
            colors = [STREAM_BLUE] * max(n, 1)
        else:
            base = get_cmap("Blues")
            colors = [base(v) for v in np.linspace(0.55, 1.0, max(n, 1))]
        ax.imshow(so, cmap=ListedColormap(colors), norm=BoundaryNorm(np.arange(0.5, max(n, 1) + 1.5), max(n, 1)),
                  interpolation="nearest", alpha=alpha)
        return colors, n

    def _draw_streams(self, ax):
        self._base(ax, terrain_alpha=0.25)
        colors, n = self._stream_overlay(ax)
        handles = [Patch(facecolor=colors[i], label=f"Order {i + 1}") for i in range(n)]
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.01),
                  ncol=min(n, 6), fontsize=9, frameon=False, title="Strahler stream order")
        self._title(ax, "Stream network (Strahler order)")
        so = self.stream_order
        total = self.cell_len[so > 0].sum() / 1000
        area = np.isfinite(self.dem).sum() * self.meta["area"] / 1e6
        rows = "".join(f"  Order {o}   {self.cell_len[so == o].sum() / 1000:8.2f} km\n" for o in range(1, n + 1))
        self._write([("h", "Step 4 - Stream network\n\n"),
                     ("", "A cell is part of a stream when its accumulation is above a threshold "
                          "(the minimum area needed to start a channel).\n\n"
                          "Try changing the threshold on the left: a small value gives a dense "
                          "network, a large value keeps only the main rivers.\n\n"),
                     ("s", "Strahler order\n"),
                     ("", "Headwater streams are order 1. Two streams of the same order join to make "
                          "the next order (1+1→2, 2+2→3). A smaller stream joining a bigger "
                          "one does not change its order.\n\n"),
                     ("s", "Result\n"),
                     ("", f"  Threshold   {self.stream_thr:,} cells "
                          f"({self.stream_thr * self.meta['area'] / 1e6:.2f} km²)\n"
                          f"  Max order   {n}\n  Total length {total:,.1f} km\n"
                          f"  Drainage density {total / area:.2f} km/km²\n\n"), ("", rows)])

    def _draw_watershed(self, ax):
        self._base(ax, terrain_alpha=0.30)
        if self.ws_points:
            n = len(self.ws_points)
            lab = self.ws_labels.astype(float)
            lab[lab == 0] = np.nan
            ax.imshow(lab, cmap=ListedColormap(WS_COLORS[:n]), norm=BoundaryNorm(np.arange(0.5, n + 1.5), n),
                      alpha=0.42, interpolation="nearest")
        self._stream_overlay(ax, single_color=True)
        for p in self.ws_points:
            color = WS_COLORS[p["id"] - 1]
            ax.contour(self.ws_labels == p["id"], levels=[0.5], colors=[color], linewidths=2)
            ax.plot(p["c"], p["r"], marker="*", markersize=16, markerfacecolor=AMBER,
                    markeredgecolor=NAVY, markeredgewidth=1.2)
            ax.annotate(str(p["id"]), (p["c"], p["r"]), xytext=(7, 7), textcoords="offset points",
                        fontsize=11, fontweight="bold", color="white",
                        bbox=dict(boxstyle="round,pad=0.2", fc=color, ec="none"))
        self._title(ax, "Watersheds" if self.ws_points else "Click 'Pick pour point', then click a stream")
        if self.ws_points:
            ax.legend(handles=[Line2D([], [], marker="*", ls="", ms=12, mfc=AMBER, mec=NAVY, label="Pour point"),
                               Line2D([], [], color=STREAM_BLUE, lw=2, label="Streams")],
                      loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=2, fontsize=9, frameon=False)
            self._watershed_report()
        else:
            self._write([("h", "Step 5 - Watershed\n\n"),
                         ("", "A watershed (catchment / drainage basin) is all the land that drains "
                              "to one outlet, called the pour point.\n\n"),
                         ("s", "How to use\n"),
                         ("", "1. Click 'Pick pour point on map'.\n"
                              "2. Zoom in with the toolbar if needed (then switch the zoom tool off).\n"
                              "3. Click on a blue stream line.\n\n"
                              "The click snaps to the cell with the highest flow accumulation nearby, "
                              "so the outlet is exactly on the river.\n\n"
                              "Click several points - try one far downstream and one upstream to see "
                              "nested sub-watersheds.\n")])

    def _draw_hypsometric(self, ax, wid):
        m = next(p["mask"] for p in self.ws_points if p["id"] == wid)
        z = np.sort(self.dem[m & np.isfinite(self.dem)])[::-1]
        if z.size < 2 or z.max() == z.min():
            ax.set_axis_off()
            return
        rel_area = np.arange(1, z.size + 1) / z.size
        rel_h = (z - z.min()) / (z.max() - z.min())
        idx = np.linspace(0, z.size - 1, min(600, z.size)).astype(int)
        color = WS_COLORS[wid - 1]
        ax.fill_between(rel_area[idx], rel_h[idx], color=color, alpha=0.25)
        ax.plot(rel_area[idx], rel_h[idx], color=color, lw=2.2)
        ax.plot([0, 1], [1, 0], color="#94A3B8", ls="--", lw=1)
        hi = (z.mean() - z.min()) / (z.max() - z.min())
        ax.text(0.97, 0.93, f"HI = {hi:.2f}", ha="right", transform=ax.transAxes,
                fontsize=12, fontweight="bold", color=NAVY)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Relative area above (a/A)")
        ax.set_ylabel("Relative height (h/H)")
        ax.set_title(f"Hypsometric curve - watershed {wid}", fontsize=10, color=NAVY)
        ax.set_aspect("equal")
        ax.spines[["top", "right"]].set_visible(False)

    def _ws_stats(self, wid):
        p = next(pt for pt in self.ws_points if pt["id"] == wid)
        m = p["mask"]                                 # full upstream area, incl. nested basins
        km2 = self.meta["area"] / 1e6
        z = self.dem[m & np.isfinite(self.dem)]
        streams = (self.stream_order > 0) & m
        length = self.cell_len[streams].sum() / 1000
        area = m.sum() * km2
        x, y = self._xy(p["r"], p["c"])
        return dict(area=area, upstream=self.acc[p["r"], p["c"]] * km2, zmin=z.min(), zmax=z.max(),
                    zmean=z.mean(), slope=np.nanmean(self.slope[m]), length=length,
                    density=length / area if area else 0, order=int(self.stream_order[m].max()),
                    hi=(z.mean() - z.min()) / (z.max() - z.min()) if z.max() > z.min() else 0,
                    x=x, y=y, r=p["r"], c=p["c"])

    def _xy(self, r, c):
        t = self.meta.get("transform")
        if t is None:
            return None, None
        x, y = t * (c + 0.5, r + 0.5)
        crs = self.meta["crs"]
        try:
            if HAS_RASTERIO and crs is not None:
                is_geo = crs.is_geographic if hasattr(crs, "is_geographic") else False
                if not is_geo:
                    xs, ys = warp_transform(crs, "EPSG:4326", [x], [y])
                    return xs[0], ys[0]
        except Exception:
            pass
        return x, y

    def _watershed_report(self):
        parts = [("h", "Watershed report\n\n")]
        for p in reversed(self.ws_points):
            s = self._ws_stats(p["id"])
            stage = ("youthful (actively eroding)" if s["hi"] > 0.6 else
                     "mature (equilibrium)" if s["hi"] >= 0.35 else "old (well eroded)")
            loc = f"  Outlet  {s['y']:.4f}°N, {s['x']:.4f}°E\n" if s["x"] is not None else ""
            parts += [("s", f"Watershed {p['id']}\n"),
                      ("", loc +
                           f"  Area           {s['area']:,.2f} km²\n"
                           f"  Elevation      {s['zmin']:.0f} - {s['zmax']:.0f} m\n"
                           f"  Relief         {s['zmax'] - s['zmin']:.0f} m\n"
                           f"  Mean slope     {s['slope']:.1f}°\n"
                           f"  Stream length  {s['length']:,.2f} km\n"
                           f"  Drainage dens. {s['density']:.2f} km/km²\n"
                           f"  Max order      {s['order']}\n"
                           f"  Hypsometric    HI = {s['hi']:.2f}\n"
                           f"                 {stage}\n\n")]
        parts += [("s", "Reading the hypsometric curve\n"),
                  ("", "HI = (mean - min) / (max - min) elevation.\n"
                       "Convex curve, HI > 0.6: young landscape\n"
                       "S-shaped, 0.35-0.6: mature\n"
                       "Concave, HI < 0.35: old, eroded\n\n"
                       "Check: each watershed area equals the flow accumulation at its pour "
                       "point x cell area. A pour point upstream gives a nested "
                       "sub-watershed; its statistics use its full upstream area.\n")]
        self._write(parts)

    def _write(self, parts):
        self.stats.config(state="normal")
        self.stats.delete("1.0", "end")
        for tag, text in parts:
            self.stats.insert("end", text, tag or ())
        self.stats.config(state="disabled")

    # ---------------- export ----------------
    def save_png(self):
        if self.dem is None:
            return
        path = filedialog.asksaveasfilename(defaultextension=".png", initialfile=f"{self.view}_map.png",
                                            filetypes=[("PNG image", "*.png")])
        if path:
            self.fig.savefig(path, dpi=200, facecolor="white")
            messagebox.showinfo("Saved", f"Map saved to\n{path}")

    def _layer_for_export(self):
        nan = np.isnan(self.dem)
        v = self.view
        if v in ("dem", "hillshade"):
            return "dem", self.dem.astype("float32"), np.nan
        if v == "slope":
            return "slope_deg", self.slope.astype("float32"), np.nan
        if v == "fill":
            return "filled_dem", self.filled.astype("float32"), np.nan
        if v == "fdir":
            return "flow_direction_d8", np.where(nan, 255, self.fdir).astype("uint8"), 255
        if v == "facc":
            return "flow_accumulation", np.where(nan, np.nan, self.acc).astype("float32"), np.nan
        if v == "streams":
            return "stream_order", self.stream_order.astype("uint8"), 0
        return "watersheds", self.ws_labels.astype("uint8"), 0

    def export_geotiff(self):
        if self.dem is None or self.meta.get("transform") is None:
            return
        name, data, nodata = self._layer_for_export()
        path = filedialog.asksaveasfilename(defaultextension=".tif", initialfile=f"{name}.tif",
                                            filetypes=[("GeoTIFF", "*.tif")])
        if not path:
            return
        h, w = data.shape
        try:
            with rasterio.open(path, "w", driver="GTiff", height=h, width=w, count=1,
                               dtype=str(data.dtype), nodata=nodata, crs=self.meta["crs"],
                               transform=self.meta["transform"], compress="deflate") as dst:
                dst.write(data, 1)
                dst.set_band_description(1, name)
        except Exception as exc:
            messagebox.showerror("Export failed", str(exc))
            return
        messagebox.showinfo("Exported", f"{name} saved to\n{path}\n\nOpen it in QGIS to check.")

    def export_geojson(self):
        if self.dem is None or not self.ws_points:
            messagebox.showinfo("Export", "Delineate at least one watershed first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".geojson", initialfile="watersheds.geojson",
                                            filetypes=[("GeoJSON", "*.geojson")])
        if not path:
            return
        features = []
        try:
            stats = {p["id"]: self._ws_stats(p["id"]) for p in self.ws_points}
            for p in self.ws_points:
                val, s = p["id"], stats[p["id"]]
                polys = [g["coordinates"] for g, _ in shapes(p["mask"].astype("uint8"), mask=p["mask"],
                                                             transform=self.meta["transform"])]
                geom = (dict(type="Polygon", coordinates=polys[0]) if len(polys) == 1
                        else dict(type="MultiPolygon", coordinates=polys))
                geom = transform_geom(self.meta["crs"], "EPSG:4326", geom)
                features.append(dict(type="Feature", geometry=geom, properties=dict(
                    watershed_id=int(val), area_km2=round(s["area"], 3),
                    min_elev_m=round(float(s["zmin"]), 1), max_elev_m=round(float(s["zmax"]), 1),
                    mean_slope_deg=round(float(s["slope"]), 2), stream_km=round(float(s["length"]), 3),
                    drainage_density=round(float(s["density"]), 3), max_order=s["order"],
                    hypsometric_integral=round(float(s["hi"]), 3))))
            with open(path, "w") as f:
                json.dump(dict(type="FeatureCollection", features=features), f)
        except Exception as exc:
            messagebox.showerror("Export failed", str(exc))
            return
        messagebox.showinfo("Exported", f"{len(self.ws_points)} watershed(s) saved to\n{path}")

    def _quit(self):
        self.quit()
        self.destroy()


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--make-demo":
        save_demo_dem(sys.argv[2])
    else:
        WatershedStudio().mainloop()
