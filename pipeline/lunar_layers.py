#!/usr/bin/env python3
"""
lunar_layers.py  -  global lunar data layers, computed from elevation + ephemeris
and written out as vector isobands.

Layers produced
  relief   16 elevation isobands (1.25 km steps, km relative to the 1737.4 km sphere)
  slope    surface slope in degrees
  sun      sunlight, % of one year (fraction of the solar disc above the local horizon)
  earth    Earth in view, % of the same year (fraction of Earth's disc above the horizon,
           topocentric, so libration and parallax are included)
  psr      permanently shadowed regions (cells the Sun never reaches during the year)
  h        hydrogen, ONLY if you pass a global grid with --hydrogen (e.g. LPNS epithermal
           neutron counts resampled to the DEM grid); no global hydrogen data ships with this

Inputs (all public)
  DEM        equirectangular elevation grid in km, row 0 = 90N, col 0 = 180W, pixel-centred.
             Default: moon_dem.npz from the `ssa-data-lunar` package (1/8 deg, LOLA-derived).
             Any .npy/.npz/.tif with the same layout works (GeoTIFF needs rasterio).
  SPK        JPL DE440 planetary ephemeris (de440s.bsp, or the bundled 2020-2035 subset; --year must
             fall inside whichever kernel is used)
  PCK        DE440 lunar orientation, principal-axes frame (moon_pa_de440_200625.bpc, or the bundled
             2020-2035 subset)
  All three ship in this package's raw/ folder; they also come in the ssa-data-lunar wheel
  (pip install ssa-data-lunar), which the script falls back to.

Method
  1. Optional block-average of the DEM to --res pixels per degree (default 4, ~7.6 km).
  2. Slope from central differences on the sphere; the zonal stencil widens toward the
     poles so its baseline stays about one meridional cell.
  3. Horizon map: for every cell and --naz azimuths, march along the great circle out to
     --max-km and keep the maximum elevation angle, curvature included exactly.
  4. Sun and Earth positions in the lunar PA frame every --step-h hours across --year
     (DE440 via SPICE). For each cell and time step, the body's local elevation is
     compared with the horizon at its azimuth (linear between azimuth bins); the visible
     fraction of the disc is accumulated. PA vs ME frame offset (~1 km) is below the grid.
  5. Each layer is contoured into stacked "value >= threshold" polygons (holes included),
     clipped to the globe, simplified, and densified so long edges follow the sphere.

Outputs (in --out)
  moon_layers.json             vector bands + small value rasters for the 3D globe page
  svg/global_<layer>.svg       equirectangular maps
  svg/south_pole_<layer>.svg   80S-to-pole polar stereographic maps, 0 deg longitude at top
  svg/north_pole_<layer>.svg   80N-to-pole polar stereographic maps, 0 deg longitude at bottom

Usage
  python lunar_layers.py                          # defaults: 4 px/deg, 2026, 3 h steps
  python lunar_layers.py --res 8 --step-h 1       # full DEM resolution (slower, ~1 GB RAM)
  python lunar_layers.py --dem my_dem.tif --hydrogen lpns_h.npy
  python lunar_layers.py --from-grids out/moon_layer_grids.npz   # redo vectors only

Requires: numpy, scipy, numba, spiceypy, shapely, matplotlib (contouring only)
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import json
import math
import time
import zlib
from pathlib import Path

import numpy as np

R_MOON = 1737.4          # km, LOLA reference sphere
R_SUN = 696_000.0        # km
R_EARTH = 6_371.0        # km

# ---------------------------------------------------------------- layer definitions
RELIEF_THRESH = [round(-7.5 + 1.25 * k, 2) for k in range(15)]          # -7.5 ... 10.0 km
RELIEF_COLORS = ["#262a31", "#373b41", "#46494f", "#55575c", "#626568", "#707275",
                 "#7d7f81", "#8a8b8d", "#969898", "#a3a4a4", "#afb0af", "#bcbcbb",
                 "#c8c8c6", "#d4d4d1", "#e0dfdc", "#ecebe7"]
LAYERS = {
    "sun":   dict(title="Sunlight (% of year)", thresholds=[10, 25, 40, 55, 70, 85],
                  colors=["#fde8dc", "#f9c3a6", "#f29470", "#e0613f", "#b8402a", "#86281b"]),
    "earth": dict(title="Earth in view (% of time)", thresholds=[10, 25, 40, 55, 70, 85],
                  colors=["#e3ecfb", "#bccff3", "#8eaee8", "#5b85d6", "#3a5dae", "#22397a"]),
    "slope": dict(title="Slope (degrees)", thresholds=[5, 10, 15, 20, 25],
                  colors=["#f1f1ee", "#d8d6cf", "#b9b5aa", "#958f80", "#6e6858"]),
    "h":     dict(title="Hydrogen signal (relative)", thresholds=[20, 35, 50, 65, 80],
                  colors=["#ddf3f4", "#a8e0e3", "#6cc6cc", "#2fa6ae", "#00838b"]),
}
PSR_COLOR = "#EAAA00"


# ---------------------------------------------------------------- inputs
def ssa_data_file(name: str) -> Path | None:
    """Raw input file: ../raw/ next to this script first, else the installed ssa-data-lunar package."""
    raw = Path(__file__).resolve().parent.parent / "raw" / name      # bundled copy in this package
    if raw.is_file():
        return raw
    try:
        from importlib.resources import files
        p = files("ssa_data_lunar").joinpath("data", name)
        return Path(str(p)) if p.is_file() else None
    except Exception:
        return None


def load_grid(path: Path) -> np.ndarray:
    path = Path(path)
    if path.suffix == ".npz":
        z = np.load(path, allow_pickle=False)
        key = "elev_km" if "elev_km" in z.files else z.files[0]
        return z[key].astype(np.float32)
    if path.suffix == ".npy":
        return np.load(path, allow_pickle=False).astype(np.float32)
    if path.suffix.lower() in (".tif", ".tiff"):
        import rasterio  # optional dependency
        with rasterio.open(path) as src:
            return src.read(1).astype(np.float32)
    raise ValueError(f"Unsupported grid format: {path}")


def block_mean(a: np.ndarray, f: int) -> np.ndarray:
    if f == 1:
        return a
    H, W = a.shape
    return a[: H // f * f, : W // f * f].reshape(H // f, f, W // f, f).mean(axis=(1, 3)).astype(np.float32)


def resample_to(a: np.ndarray, H: int, W: int) -> np.ndarray:
    if a.shape == (H, W):
        return a.astype(np.float32)
    if a.shape[0] % H == 0 and a.shape[1] % W == 0 and a.shape[0] // H == a.shape[1] // W:
        return block_mean(a, a.shape[0] // H)
    from scipy.ndimage import zoom
    return zoom(a, (H / a.shape[0], W / a.shape[1]), order=1).astype(np.float32)


def grid_latlon(H: int, W: int):
    lat = 90.0 - (np.arange(H) + 0.5) * 180.0 / H
    lon = -180.0 + (np.arange(W) + 0.5) * 360.0 / W
    return lat, lon


# ---------------------------------------------------------------- slope
def slope_deg(elev_km: np.ndarray) -> np.ndarray:
    H, W = elev_km.shape
    lat, _ = grid_latlon(H, W)
    dphi = math.radians(180.0 / H)
    dlam = math.radians(360.0 / W)
    dy = R_MOON * dphi
    # meridional gradient (rows run north -> south)
    gy = np.empty_like(elev_km)
    gy[1:-1] = (elev_km[:-2] - elev_km[2:]) / (2 * dy)
    gy[0], gy[-1] = gy[1], gy[-2]
    # zonal gradient with a stencil whose baseline stays ~one meridional cell
    gx = np.empty_like(elev_km)
    for r in range(H):
        c = max(math.cos(math.radians(lat[r])), 1e-6)
        k = max(1, int(round(1.0 / c)))
        k = min(k, W // 4)
        row = elev_km[r]
        gx[r] = (np.roll(row, -k) - np.roll(row, k)) / (2 * k * R_MOON * c * dlam)
    return np.degrees(np.arctan(np.hypot(gx, gy))).astype(np.float32)


# ---------------------------------------------------------------- horizon + visibility (numba)
def _numba_kernels():
    from numba import njit, prange

    @njit(cache=True, fastmath=True)
    def bilinear(e, lat, lon):
        H, W = e.shape
        fr = (90.0 - lat) * H / 180.0 - 0.5
        fc = (lon + 180.0) * W / 360.0 - 0.5
        if fr < 0.0:
            fr = 0.0
        if fr > H - 1.0:
            fr = H - 1.0
        r0 = int(math.floor(fr))
        if r0 > H - 2:
            r0 = H - 2
        tr = fr - r0
        c0f = math.floor(fc)
        tc = fc - c0f
        c0 = int(c0f) % W
        c1 = (c0 + 1) % W
        a = e[r0, c0] * (1 - tc) + e[r0, c1] * tc
        b = e[r0 + 1, c0] * (1 - tc) + e[r0 + 1, c1] * tc
        return a * (1 - tr) + b * tr

    @njit(parallel=True, cache=True, fastmath=True)
    def horizon_map(e, naz, max_km):
        H, W = e.shape
        cell = R_MOON * math.pi / H
        hz = np.empty((H, W, naz), np.float32)
        for r in prange(H):
            lat0 = math.radians(90.0 - (r + 0.5) * 180.0 / H)
            s0, c0 = math.sin(lat0), math.cos(lat0)
            for c in range(W):
                lon0 = math.radians(-180.0 + (c + 0.5) * 360.0 / W)
                r_obs = R_MOON + e[r, c]
                for a in range(naz):
                    A = 2.0 * math.pi * a / naz
                    sA, cA = math.sin(A), math.cos(A)
                    best = -1.5707963
                    d = 0.6 * cell
                    while d < max_km:
                        dl = d / R_MOON
                        sd, cd = math.sin(dl), math.cos(dl)
                        slat = s0 * cd + c0 * sd * cA
                        lat = math.asin(min(1.0, max(-1.0, slat)))
                        lon = lon0 + math.atan2(sA * sd * c0, cd - s0 * slat)
                        latd = math.degrees(lat)
                        lond = (math.degrees(lon) + 540.0) % 360.0 - 180.0
                        rt = R_MOON + bilinear(e, latd, lond)
                        ang = math.atan2(rt * cd - r_obs, rt * sd)
                        if ang > best:
                            best = ang
                        step = 0.05 * d
                        d += step if step > cell else cell
                    hz[r, c, a] = best
        return hz

    @njit(parallel=True, cache=True, fastmath=True)
    def visibility(e, hz, body, rad_km, topocentric):
        """Mean visible fraction (0..1) of a body's disc over all time steps.
        body: (T,3) position of the body from the Moon's centre, km, lunar PA frame."""
        H, W, naz = hz.shape
        T = body.shape[0]
        out = np.zeros((H, W), np.float32)
        two_pi = 2.0 * math.pi
        for r in prange(H):
            lat = math.radians(90.0 - (r + 0.5) * 180.0 / H)
            sl, cl = math.sin(lat), math.cos(lat)
            for c in range(W):
                lon = math.radians(-180.0 + (c + 0.5) * 360.0 / W)
                so, co = math.sin(lon), math.cos(lon)
                ux, uy, uz = cl * co, cl * so, sl            # up
                ex, ey = -so, co                              # east (z = 0)
                nx, ny, nz = -sl * co, -sl * so, cl           # north
                rr = R_MOON + e[r, c]
                px, py, pz = rr * ux, rr * uy, rr * uz
                hmin = 10.0
                hmax = -10.0
                for a in range(naz):
                    v = hz[r, c, a]
                    if v < hmin:
                        hmin = v
                    if v > hmax:
                        hmax = v
                acc = 0.0
                for t in range(T):
                    vx, vy, vz = body[t, 0], body[t, 1], body[t, 2]
                    if topocentric:
                        vx -= px
                        vy -= py
                        vz -= pz
                    dist = math.sqrt(vx * vx + vy * vy + vz * vz)
                    ang_r = math.asin(rad_km / dist)
                    el = math.asin((vx * ux + vy * uy + vz * uz) / dist)
                    if el + ang_r <= hmin:
                        continue
                    if el - ang_r >= hmax:
                        acc += 1.0
                        continue
                    az = math.atan2(vx * ex + vy * ey, vx * nx + vy * ny + vz * nz)
                    if az < 0.0:
                        az += two_pi
                    fa = az / two_pi * naz
                    a0 = int(fa) % naz
                    a1 = (a0 + 1) % naz
                    w = fa - math.floor(fa)
                    h = hz[r, c, a0] * (1.0 - w) + hz[r, c, a1] * w
                    f = (el - h) / (2.0 * ang_r) + 0.5
                    if f > 1.0:
                        f = 1.0
                    if f > 0.0:
                        acc += f
                out[r, c] = acc / T
        return out

    return horizon_map, visibility


# ---------------------------------------------------------------- ephemeris
def ephemeris(spk: Path, pck: Path, year: int, step_h: float):
    import spiceypy as sp
    sp.kclear()
    sp.furnsh(str(spk))
    sp.furnsh(str(pck))
    # Frame definition normally supplied by moon_de440_200625.tf (PA frame only)
    sp.pdpool("FRAME_MOON_PA_DE440", [31008])
    sp.pcpool("FRAME_31008_NAME", ["MOON_PA_DE440"])
    sp.pdpool("FRAME_31008_CLASS", [2])
    sp.pdpool("FRAME_31008_CLASS_ID", [31008])
    sp.pdpool("FRAME_31008_CENTER", [301])
    # Epoch as TDB seconds past J2000 (no leap-second kernel needed)
    t0 = (dt.datetime(year, 1, 1) - dt.datetime(2000, 1, 1, 12)).total_seconds()
    t1 = (dt.datetime(year + 1, 1, 1) - dt.datetime(2000, 1, 1, 12)).total_seconds()
    ets = np.arange(t0, t1, step_h * 3600.0)
    sun = np.array([sp.spkpos("SUN", et, "MOON_PA_DE440", "LT+S", "MOON")[0] for et in ets])
    earth = np.array([sp.spkpos("EARTH", et, "MOON_PA_DE440", "LT+S", "MOON")[0] for et in ets])
    sp.kclear()
    return ets, sun, earth


# ---------------------------------------------------------------- vectorizing
def superlevel_polygons(grid: np.ndarray, threshold: float, simplify_deg: float, densify_deg: float):
    """Polygons (shapely) covering grid >= threshold, in lon/lat degrees."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from shapely.geometry import Polygon, box
    from shapely.ops import unary_union

    H, W = grid.shape
    lat, lon = grid_latlon(H, W)
    pad = 2
    g = np.concatenate([grid[:, -pad:], grid, grid[:, :pad]], axis=1)
    g = np.concatenate([np.repeat(g[:1], pad, 0), g, np.repeat(g[-1:], pad, 0)], axis=0)
    dlon, dlat = 360.0 / W, 180.0 / H
    X = np.concatenate([lon[0] - dlon * np.arange(pad, 0, -1), lon, lon[-1] + dlon * np.arange(1, pad + 1)])
    Y = np.concatenate([lat[0] + dlat * np.arange(pad, 0, -1), lat, lat[-1] - dlat * np.arange(1, pad + 1)])
    vmax = float(np.nanmax(g))
    if vmax < threshold:
        return []
    fig = plt.figure()
    cs = plt.contourf(X, Y, g, levels=[threshold, max(vmax, threshold) + 1.0])
    rings = []
    for path in cs.get_paths():
        for poly in path.to_polygons(closed_only=True):
            if len(poly) >= 4:
                rings.append(poly)
    plt.close(fig)
    if not rings:
        return []
    # even-odd assembly: nest rings by containment depth
    from shapely.strtree import STRtree
    polys = [Polygon(r).buffer(0) for r in rings]
    polys = [p for p in polys if not p.is_empty and p.area > 1e-6]
    if not polys:
        return []
    order = sorted(range(len(polys)), key=lambda i: -polys[i].area)
    polys = [polys[i] for i in order]
    tree = STRtree(polys)
    depth = []
    for i, p in enumerate(polys):
        pt = p.representative_point()
        idx = tree.query(pt, predicate="within")
        depth.append(sum(1 for j in idx if j != i and polys[j].area > p.area))
    # rings are nested, so the even-odd fill is (U0 - U1) + (U2 - U3) + ...
    # where Ud is the union of rings at nesting depth d
    levels = {}
    for i, p in enumerate(polys):
        levels.setdefault(depth[i], []).append(p)
    U = [unary_union(levels[d]) for d in sorted(levels)]
    pieces = []
    for d in range(0, len(U), 2):
        pieces.append(U[d].difference(U[d + 1]) if d + 1 < len(U) else U[d])
    region = unary_union(pieces)
    region = region.intersection(box(-180, -90, 180, 90))
    if simplify_deg > 0:
        region = region.simplify(simplify_deg, preserve_topology=True)
    out = []
    for p in getattr(region, "geoms", [region]):
        if p.geom_type == "Polygon" and p.area > (dlat * dlat) * 0.5:
            out.append(p.segmentize(densify_deg) if densify_deg > 0 else p)
    return out


def tile_polygons(polys, step: float):
    """Split polygons on a lon/lat tile grid; returns the pieces (holes kept)."""
    from shapely.geometry import box
    from shapely.strtree import STRtree
    if not polys:
        return []
    tree = STRtree(polys)
    out = []
    for lo in np.arange(-180, 180, step):
        for la in np.arange(-90, 90, step):
            cell = box(lo, la, min(lo + step, 180), min(la + step, 90))
            for i in tree.query(cell, predicate="intersects"):
                piece = polys[i].intersection(cell)
                for g in getattr(piece, "geoms", [piece]):
                    if g.geom_type == "Polygon" and g.area > 1e-6:
                        out.append(g)
    return out


# ---------------------------------------------------------------- encoders
class Packer:
    """Packs polygons as int16 lon/lat (0.01 deg) + a uint32 structure stream."""

    def __init__(self):
        self.coords, self.struct = [], []

    def band(self, polys):
        self.struct.append(len(polys))
        for p in polys:
            rings = [p.exterior] + list(p.interiors)
            rings = [r for r in rings if len(r.coords) >= 4]
            self.struct.append(len(rings))
            for r in rings:
                a = np.asarray(r.coords)[:-1]
                self.struct.append(len(a))
                self.coords.append(a)

    def payload(self):
        allc = np.concatenate(self.coords) if self.coords else np.zeros((0, 2))
        q = np.round(allc * 100).astype("<i2")
        return {"coords": base64.b64encode(q.tobytes()).decode(),
                "struct": base64.b64encode(np.asarray(self.struct, "<u4").tobytes()).decode(),
                "points": int(len(allc))}


def raster_payload(a: np.ndarray, scale: float, offset: float):
    q = np.clip(np.round((a - offset) * scale), 0, 255).astype(np.uint8)
    return {"w": a.shape[1], "h": a.shape[0], "scale": scale, "offset": offset,
            "data": base64.b64encode(zlib.compress(q.tobytes(), 9)).decode()}


# ---------------------------------------------------------------- SVG writers
def _ring_path(coords, proj):
    pts = [proj(x, y) for x, y in coords]
    pts = [p for p in pts if p is not None]
    if len(pts) < 3:
        return ""
    return "M" + "L".join(f"{x:.2f},{y:.2f}" for x, y in pts) + "Z"


def write_svgs(out_dir: Path, layers_polys: dict, relief_polys: list, psr_polys: list):
    out_dir.mkdir(parents=True, exist_ok=True)

    def eq(x, y):          # equirectangular, 4 px per degree
        return ((x + 180) * 4, (90 - y) * 4)

    def polar(hemi, cap_deg=10.0, size=606, rad=300.0):
        t = math.tan(math.radians(cap_deg) / 2)

        def f(x, y):
            col = 90 + y if hemi < 0 else 90 - y          # colatitude from the pole
            # points far outside the cap are pulled in to a ring just outside the clip disc,
            # which keeps every polygon's inside/outside correct within the disc
            col = min(col, cap_deg * 1.5)
            r = math.tan(math.radians(col) / 2) / t * rad
            a = math.radians(x)
            if hemi < 0:   # south: 0 deg at top, east to the right
                return (size / 2 + r * math.sin(a), size / 2 - r * math.cos(a))
            return (size / 2 + r * math.sin(a), size / 2 + r * math.cos(a))
        return f

    views = {
        "global": (eq, 1440, 720, None),
        "south_pole": (polar(-1), 606, 606, (303, 303, 300)),
        "north_pole": (polar(+1), 606, 606, (303, 303, 300)),
    }

    def bands_svg(polys_by_band, colors, proj, opacity=None, data_min=None):
        parts = []
        for k, polys in enumerate(polys_by_band):
            d = "".join(_ring_path(r.coords, proj) for p in polys for r in [p.exterior, *p.interiors])
            if not d:
                continue
            attr = f' data-min="{data_min[k]}"' if data_min else ""
            parts.append(f'<path fill="{colors[k]}"{attr} d="{d}"/>')
        g_attr = f' fill-opacity="{opacity}" fill-rule="evenodd"' if opacity else ' fill-rule="evenodd"'
        return f"<g{g_attr}>" + "".join(parts) + "</g>"

    for view, (proj, w, h, disc) in views.items():
        clip_open = clip_close = defs = ""
        if disc:
            cx, cy, r = disc
            defs = f'<defs><clipPath id="disc"><circle cx="{cx}" cy="{cy}" r="{r}"/></clipPath></defs>'
            clip_open, clip_close = '<g clip-path="url(#disc)">', "</g>"
        relief = bands_svg(relief_polys, RELIEF_COLORS, proj)
        psr = bands_svg([psr_polys], [PSR_COLOR], proj, opacity=".82")
        files = {"topographic": ("Lunar relief, 16 isobands (1.25 km steps)", "")}
        for key, L in layers_polys.items():
            files[f"layer_{key}"] = (L["title"], bands_svg(L["polys"], L["colors"], proj, ".62", L["thresholds"]))
        for name, (title, overlay) in files.items():
            svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">'
                   f"<title>{title}</title>{defs}{clip_open}<rect width=\"{w}\" height=\"{h}\" fill=\"#262a31\"/>"
                   f"{relief}{overlay}{psr if name != 'topographic' else ''}{clip_close}")
            if disc:
                svg += f'<circle cx="{disc[0]}" cy="{disc[1]}" r="{disc[2]}" fill="none" stroke="#5B6573" stroke-width="0.8"/>'
            svg += "</svg>"
            (out_dir / f"{view}_{name}_vector.svg").write_text(svg)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dem", type=Path, default=None, help="elevation grid in km (default: ssa-data-lunar moon_dem.npz)")
    ap.add_argument("--spk", type=Path, default=None, help="de440s.bsp")
    ap.add_argument("--pck", type=Path, default=None, help="moon_pa_de440_200625.bpc")
    ap.add_argument("--hydrogen", type=Path, default=None, help="optional global hydrogen grid, same layout as the DEM")
    ap.add_argument("--res", type=int, default=4, help="working resolution, pixels per degree (default 4)")
    ap.add_argument("--year", type=int, default=2026)
    ap.add_argument("--step-h", type=float, default=3.0, help="ephemeris step in hours (default 3)")
    ap.add_argument("--naz", type=int, default=48, help="horizon azimuth bins (default 48)")
    ap.add_argument("--max-km", type=float, default=250.0, help="horizon search distance (default 250 km)")
    ap.add_argument("--simplify", type=float, default=None, help="polygon simplification, degrees (default 0.4 cell)")
    ap.add_argument("--raster-res", type=int, default=4, help="px/deg of value rasters embedded for the globe readout")
    ap.add_argument("--out", type=Path, default=Path("lunar_layers_out"))
    ap.add_argument("--no-svg", action="store_true")
    ap.add_argument("--tile", type=float, default=10.0, help="tile size in degrees for terrace tops (default 10)")
    ap.add_argument("--from-grids", type=Path, default=None,
                    help="reuse moon_layer_grids.npz from an earlier run and only redo the vector output")
    args = ap.parse_args()

    dem_path = args.dem or ssa_data_file("moon_dem.npz")
    spk = args.spk or ssa_data_file("de440s.bsp") or ssa_data_file("de440s_2020-2035.bsp")
    pck = args.pck or ssa_data_file("moon_pa_de440_200625.bpc") or ssa_data_file("moon_pa_de440_200625_2020-2035.bpc")
    for label, p in [("DEM", dem_path), ("SPK", spk), ("PCK", pck)]:
        if p is None or not Path(p).exists():
            raise SystemExit(f"{label} not found. pip install ssa-data-lunar, or pass --{label.lower()} PATH")

    t_start = time.time()
    log = lambda m: print(f"[{time.time() - t_start:7.1f}s] {m}", flush=True)

    dem_full = load_grid(dem_path)
    native = dem_full.shape[1] / 360.0
    H, W = 180 * args.res, 360 * args.res
    dem = resample_to(dem_full, H, W)
    log(f"DEM {dem_full.shape} ({native:g} px/deg) -> working grid {dem.shape}, "
        f"{np.nanmin(dem):.2f} to {np.nanmax(dem):.2f} km")

    if args.from_grids:
        g = np.load(args.from_grids, allow_pickle=False)
        dem, sun_pct, earth_pct, slope = g["elev_km"], g["sun_pct"], g["earth_pct"], g["slope_deg"]
        H, W = dem.shape
        args.res = H // 180
        log(f"loaded computed grids from {args.from_grids} ({H}x{W}); skipping horizon and ephemeris work")
    else:
        slope = slope_deg(dem_full)
        slope = resample_to(slope, H, W)
        log(f"slope: median {np.median(slope):.1f} deg, max {slope.max():.1f} deg")

        horizon_map, visibility = _numba_kernels()
        hz = horizon_map(dem, args.naz, args.max_km)
        log(f"horizon map {hz.shape}")

        ets, sun, earth = ephemeris(spk, pck, args.year, args.step_h)
        log(f"ephemeris: {len(ets)} steps through {args.year}")

        sun_pct = visibility(dem, hz, sun, R_SUN, False) * 100.0
        log(f"sunlight: mean {sun_pct.mean():.1f}%")
        earth_pct = visibility(dem, hz, earth, R_EARTH, True) * 100.0
        log(f"earth in view: mean {earth_pct.mean():.1f}%")
        del hz

    grids = {"sun": sun_pct, "earth": earth_pct, "slope": slope}
    if args.hydrogen:
        h = resample_to(load_grid(args.hydrogen), H, W)
        lo, hi = np.nanpercentile(h, [1, 99])
        grids["h"] = np.clip((h - lo) / (hi - lo) * 100.0, 0, 100)   # relative 0-100

    simplify = args.simplify if args.simplify is not None else 0.4 / args.res
    densify = 0.5
    packer = Packer()
    out = {"meta": {"source_dem": str(Path(dem_path).name), "dem_px_per_deg": native, "res": args.res,
                    "year": args.year, "step_h": args.step_h, "naz": args.naz, "max_km": args.max_km,
                    "radius_km": R_MOON, "generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")},
           "relief": {"title": "Elevation (km)", "thresholds": RELIEF_THRESH, "colors": RELIEF_COLORS},
           "layers": {}}

    from shapely.geometry import box
    whole = box(-180, -90, 180, 90).segmentize(densify)
    relief_polys = [[whole]]              # band 0 covers the sphere (the globe page builds it as a grid)
    for t in RELIEF_THRESH:
        relief_polys.append(superlevel_polygons(dem, t, simplify, densify))
    # visible terrace tops: band k minus band k+1, cut into tiles so a triangulator never sees
    # globe-sized polygons; terrace walls follow the untiled band outlines
    from shapely.ops import unary_union
    for k in range(len(relief_polys)):
        top = unary_union(relief_polys[k])
        if k + 1 < len(relief_polys) and relief_polys[k + 1]:
            top = top.difference(unary_union(relief_polys[k + 1]))
        parts = [g for g in getattr(top, "geoms", [top]) if g.geom_type == "Polygon" and not g.is_empty]
        packer.band(tile_polygons(parts, args.tile))
    for ps in relief_polys[1:]:
        packer.band(ps)
    log(f"relief vectorized ({sum(len(p) for p in relief_polys[1:])} polygons)")

    layers_polys = {}
    for key, grid in grids.items():
        L = LAYERS[key]
        bands = []
        for t in L["thresholds"]:
            ps = superlevel_polygons(grid, t, simplify, densify)
            packer.band(ps)
            bands.append(ps)
        out["layers"][key] = {"title": L["title"], "thresholds": L["thresholds"], "colors": L["colors"]}
        layers_polys[key] = {**L, "polys": bands}
        log(f"{key} vectorized")

    psr_polys = superlevel_polygons((sun_pct <= 0.0).astype(np.float32), 0.5, simplify, densify)
    packer.band(psr_polys)
    log(f"PSR: {len(psr_polys)} regions, {(sun_pct <= 0).sum()} cells")

    out["vectors"] = packer.payload()
    rr = args.raster_res
    out["rasters"] = {
        "elev": raster_payload(resample_to(dem_full, 180 * rr, 360 * rr), 12.0, -10.0),
        "sun": raster_payload(resample_to(sun_pct, 180 * rr, 360 * rr), 2.5, 0.0),
        "earth": raster_payload(resample_to(earth_pct, 180 * rr, 360 * rr), 2.5, 0.0),
        "slope": raster_payload(resample_to(slope, 180 * rr, 360 * rr), 4.0, 0.0),
    }
    if "h" in grids:
        out["rasters"]["h"] = raster_payload(resample_to(grids["h"], 180 * rr, 360 * rr), 2.5, 0.0)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "moon_layers.json").write_text(json.dumps(out, separators=(",", ":")))
    np.savez_compressed(args.out / "moon_layer_grids.npz", elev_km=dem, sun_pct=sun_pct,
                        earth_pct=earth_pct, slope_deg=slope)
    log(f"wrote {args.out / 'moon_layers.json'} ({out['vectors']['points']} vector points)")

    if not args.no_svg:
        write_svgs(args.out / "svg", layers_polys, relief_polys, psr_polys)
        log(f"wrote SVGs to {args.out / 'svg'}")


if __name__ == "__main__":
    main()
