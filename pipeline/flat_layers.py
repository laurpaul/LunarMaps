#!/usr/bin/env python3
"""
flat_layers.py  -  smooth flat (2D) vector maps of the lunar layers, in the ESPL style.

Uses the same inputs and smoothing as the 3D animation:
  elevation   LOLA-derived 1/8 deg grid (ssa-data-lunar, or --dem)
  layers      moon_layer_grids.npz from lunar_layers.py (sunlight %, Earth in view %, slope deg;
              permanently shadowed = cells with 0% sunlight)
  sites       zone_data.json from site_zones.py (optional, for the sites map and the close-up)

Writes SVGs to --out:
  south_pole_<layer>.svg   polar stereographic, 80S to the pole, 0 deg longitude (nearside) at top
  global_<layer>.svg       equirectangular, whole Moon
  malapert_closeup.svg     local map of the Malapert sites with safety zones and their overlap
layers: relief, slope, sunlight, earth, psr, sites

Every map is drawn from filled isobands (matplotlib contourf) on smoothed grids, so edges are
smooth curves; paths are thinned with Douglas-Peucker at 0.45 px (0.9 px on the global maps). Polar maps carry 500 m contours,
the close-up 250 m. Polar maps are resampled onto the polar plane first and smoothed there (--sigma-km),
because equirectangular cells are long and thin near the pole.

Usage: python flat_layers.py --grids out/moon_layer_grids.npz --zones out/zone_data.json --out out/flat
"""
import argparse, json, math, base64
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = 1737.4
NAVY, GOLD, GOLD_INK, MUTED, BG, PAPER = "#003057", "#B3A369", "#857437", "#5B6573", "#E9E6DD", "#F7F6F2"
RELIEF_T = [-8.75 + 1.25 * k for k in range(1, 16)]
RELIEF_C = ["#262a31", "#373b41", "#46494f", "#55575c", "#626568", "#707275", "#7d7f81", "#8a8b8d",
            "#969898", "#a3a4a4", "#afb0af", "#bcbcbb", "#c8c8c6", "#d4d4d1", "#e0dfdc", "#ecebe7"]
LAYERS = {
    "slope":    dict(title="Slope (degrees)", key="slope_deg", t=[5, 10, 15, 20, 25],
                     c=["#ece6f7", "#cdbdec", "#a991dc", "#8364c6", "#5a3a9e"]),
    "sunlight": dict(title="Sunlight (% of 2026)", key="sun_pct", t=[10, 25, 40, 55, 70, 85],
                     c=["#fde8dc", "#f9c3a6", "#f29470", "#e0613f", "#b8402a", "#86281b"]),
    "earth":    dict(title="Earth in view (% of 2026)", key="earth_pct", t=[10, 25, 40, 55, 70, 85],
                     c=["#e3ecfb", "#bccff3", "#8eaee8", "#5b85d6", "#3a5dae", "#22397a"]),
}
PSR_C, CA, CB = "#EAAA00", "#2E5BBA", "#E04F39"


def ssa(name):
    """Raw input file: the package's raw/ folder first, else the installed ssa-data-lunar package."""
    raw = Path(__file__).resolve().parent.parent / "raw" / name      # bundled copy in this package
    if raw.is_file():
        return raw
    from importlib.resources import files
    return Path(str(files("ssa_data_lunar").joinpath("data", name)))


def sampler(a, pad=4):
    H, W = a.shape
    ap = np.concatenate([a[:, -pad:], a, a[:, :pad]], axis=1)
    def f(lat, lon, order=3):
        r = (90 - lat) * H / 180 - 0.5
        c = ((lon + 180) % 360) * W / 360 - 0.5 + pad
        return map_coordinates(ap, [r.ravel(), c.ravel()], order=order, mode="nearest").reshape(lat.shape)
    return f


def smooth_wrap(a, sigma):
    pad = int(4 * sigma) + 2
    ap = np.concatenate([a[:, -pad:], a, a[:, :pad]], axis=1)
    return gaussian_filter(ap, sigma, mode="nearest")[:, pad:-pad]


# ---------------------------------------------------------------- vector helpers
TOL = 0.45   # px; Douglas-Peucker tolerance for output paths


def simplify(pts, closed):
    from shapely.geometry import LineString
    if len(pts) < 4:
        return pts
    s = np.asarray(LineString(pts).simplify(TOL, preserve_topology=False).coords)
    return s if len(s) >= (4 if closed else 2) else pts


def path_d(path, sx=1.0, sy=1.0, ox=0.0, oy=0.0):
    out = []
    for poly in path.to_polygons(closed_only=True):
        if len(poly) < 3:
            continue
        poly = simplify(poly, True)
        out.append("M" + "L".join(f"{x * sx + ox:.1f},{y * sy + oy:.1f}" for x, y in poly) + "Z")
    return "".join(out)


def isobands(Z, levels, colors, opacity=1.0, stroke=None, xy=None):
    """Filled bands between consecutive levels -> SVG <path> elements (pixel coordinates)."""
    fig = plt.figure()
    X, Y = xy if xy is not None else np.meshgrid(np.arange(Z.shape[1]) + 0.5, np.arange(Z.shape[0]) + 0.5)
    cs = plt.contourf(X, Y, Z, levels=levels)
    parts = []
    for k, p in enumerate(cs.get_paths()):
        d = path_d(p)
        if d and colors[k]:
            st = f' stroke="{stroke}" stroke-width="0.5"' if stroke else ""
            parts.append(f'<path fill="{colors[k]}" fill-opacity="{opacity}" fill-rule="evenodd"{st} d="{d}"/>')
    plt.close(fig)
    return "".join(parts)


def isolines(Z, levels, color, width, opacity, xy=None):
    fig = plt.figure()
    X, Y = xy if xy is not None else np.meshgrid(np.arange(Z.shape[1]) + 0.5, np.arange(Z.shape[0]) + 0.5)
    cs = plt.contour(X, Y, Z, levels=levels)
    d = []
    for p in cs.get_paths():
        for seg in p.to_polygons(closed_only=False):
            if len(seg) > 1:
                seg = simplify(seg, False)
                d.append("M" + "L".join(f"{x:.1f},{y:.1f}" for x, y in seg))
    plt.close(fig)
    return f'<path fill="none" stroke="{color}" stroke-width="{width}" stroke-opacity="{opacity}" d="{"".join(d)}"/>' if d else ""


def legend_svg(x, y, title, labels, colors, w=34):
    items = "".join(f'<rect x="{x + 10 + i * w}" y="{y + 24}" width="{w}" height="9" fill="{c}"/>'
                    f'<text x="{x + 10 + i * w}" y="{y + 44}" font-size="9" fill="{MUTED}" font-family="Roboto Mono,monospace">{l}</text>'
                    for i, (l, c) in enumerate(zip(labels, colors)))
    width = 20 + w * len(labels)
    return (f'<g><rect x="{x}" y="{y}" width="{max(width, 150)}" height="52" rx="3" fill="#fff" fill-opacity=".92" stroke="#E2DDD0"/>'
            f'<text x="{x + 10}" y="{y + 16}" font-size="10" font-weight="700" letter-spacing=".08em" fill="{NAVY}" font-family="Roboto,Arial">{title.upper()}</text>{items}</g>')


def frame_svg(w, h, body, title, subtitle, legend="", clip_disc=None):
    clip = ""
    if clip_disc:
        cx, cy, r = clip_disc
        clip = f'<defs><clipPath id="disc"><circle cx="{cx}" cy="{cy}" r="{r}"/></clipPath></defs>'
        body = f'<g clip-path="url(#disc)"><rect width="{w}" height="{h}" fill="{BG}"/>{body}</g><circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{NAVY}" stroke-width="1"/>'
    HH = 58
    head = (f'<rect x="0" y="0" width="{w}" height="{HH}" fill="#fff"/><rect x="0" y="{HH - 3}" width="{w * 0.4:.0f}" height="3" fill="{NAVY}"/>'
            f'<rect x="{w * 0.4:.0f}" y="{HH - 3}" width="{w * 0.6:.0f}" height="3" fill="{GOLD}"/>'
            f'<text x="16" y="25" font-size="16" font-weight="700" fill="{NAVY}" font-family="Roboto,Arial">{title}</text>'
            f'<text x="16" y="43" font-size="11" font-weight="500" fill="{GOLD_INK}" font-family="Roboto,Arial">{subtitle}</text>'
            f'<text x="{w - 16}" y="25" text-anchor="end" font-size="18" font-style="italic" font-weight="900" fill="{GOLD}" font-family="Roboto,Arial">GT</text>'
            f'<text x="{w - 16}" y="43" text-anchor="end" font-size="10" fill="{NAVY}" font-family="Roboto,Arial">Engineering Space Policy Laboratory</text>')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h + HH}" width="{w}" height="{h + HH}">'
            f'<title>{title}: {subtitle}</title>{clip}<rect width="{w}" height="{h + HH}" fill="{PAPER}"/>{head}'
            f'<g transform="translate(0,{HH})">{body}{legend}</g></svg>')


# ---------------------------------------------------------------- polar maps
def polar_grid(n, cap_deg):
    t = math.tan(math.radians(cap_deg) / 2)
    u = (np.arange(n) + 0.5) / n - 0.5
    DX, DY = np.meshgrid(u, u)
    r = np.hypot(DX, DY)
    col = 2 * np.degrees(np.arctan(2 * r * t))
    lon = np.degrees(np.arctan2(DX, -DY))         # 0 deg at the top, east to the right
    return -90 + col, lon, r


def polar_project(lat, lon, n, cap_deg):
    t = math.tan(math.radians(cap_deg) / 2)
    col = math.radians(90 + lat)
    r = math.tan(col / 2) / t * 0.5 * n
    a = math.radians(lon)
    return n / 2 + r * math.sin(a), n / 2 - r * math.cos(a)


def polar_graticule(n, cap):
    parts = []
    for lat in range(-89, -90 + int(cap), 1):
        x, y = polar_project(lat, 0, n, cap); r = n / 2 - y
        parts.append(f'<circle cx="{n / 2}" cy="{n / 2}" r="{r:.1f}" fill="none" stroke="{NAVY}" stroke-opacity=".28" stroke-width=".6"/>')
    for lon in range(0, 360, 30):
        x0, y0 = polar_project(-89.8, lon, n, cap); x1, y1 = polar_project(-90 + cap, lon, n, cap)
        parts.append(f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" stroke="{NAVY}" stroke-opacity=".28" stroke-width=".6"/>')
        lx, ly = polar_project(-90 + cap * 0.965, lon, n, cap)
        lab = "0°" if lon == 0 else (f"{lon}°E" if lon <= 180 else f"{360 - lon}°W")
        parts.append(f'<text x="{lx:.1f}" y="{ly:.1f}" font-size="9" text-anchor="middle" dominant-baseline="middle" fill="{NAVY}" font-family="Roboto Mono,monospace">{lab}</text>')
    return "".join(parts)


def scale_bar(x, y, px_per_km, km):
    w = km * px_per_km
    return (f'<g><rect x="{x - 6}" y="{y - 16}" width="{w + 12}" height="26" fill="#fff" fill-opacity=".85"/>'
            f'<path d="M{x},{y - 4}V{y + 2}H{x + w:.1f}V{y - 4}" fill="none" stroke="{NAVY}" stroke-width="1.5"/>'
            f'<text x="{x + w / 2:.1f}" y="{y - 7}" text-anchor="middle" font-size="9.5" fill="{NAVY}" font-family="Roboto Mono,monospace">{km} km</text></g>')


def site_marks(sites, proj, zone_km=None, km_px=None, labels=True):
    out = []
    for s in sites:
        x, y = proj(s["lat"], s["lon"])
        c = CA if s["g"] == "a" else CB
        fill = "#fff" if s.get("approx") else c
        if zone_km and km_px and not s.get("approx"):
            out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{zone_km * km_px:.2f}" fill="{c}" fill-opacity=".18" stroke="{c}" stroke-width="1.2"/>')
        if s["g"] == "a":
            out.append(f'<rect x="{x - 4.5:.1f}" y="{y - 4.5:.1f}" width="9" height="9" fill="{fill}" stroke="{c if s.get("approx") else "#fff"}" stroke-width="1.4"/>')
        else:
            out.append(f'<path d="M{x:.1f},{y - 6:.1f}L{x + 5.5:.1f},{y + 4:.1f}L{x - 5.5:.1f},{y + 4:.1f}Z" fill="{fill}" stroke="{c if s.get("approx") else "#fff"}" stroke-width="1.3"/>')
        if labels:
            out.append(f'<text x="{x + 8:.1f}" y="{y + 3.5:.1f}" font-size="9.5" font-weight="500" fill="{c}" font-family="Roboto,Arial" '
                       f'paint-order="stroke" stroke="#fff" stroke-width="3">{s["n"]}{" (approx.)" if s.get("approx") else ""}</text>')
    return "".join(out)


def make_polar(args, dem, grids, sites):
    n, cap = args.n, args.cap
    lat, lon, r = polar_grid(n, cap)
    km_per_px = 2 * math.radians(cap) * R / n                     # near the pole
    sig = args.sigma_km / km_per_px
    h = sampler(dem)(lat, lon, 3)
    relief = isobands(h, [-99] + RELIEF_T + [99], RELIEF_C) + isolines(h, np.arange(-8.75, 10.5, 0.5), NAVY, 0.35, 0.22) \
        + isolines(h, RELIEF_T, NAVY, 0.7, 0.45)
    grat = polar_graticule(n, cap)
    disc = (n / 2, n / 2, n / 2)
    psr_m = gaussian_filter(sampler((grids["sun_pct"] <= 0).astype(np.float32))(lat, lon, 1), sig)
    psr = isobands(psr_m, [0.5, 9], [PSR_C], 0.92, stroke="#7a5800")
    sb = scale_bar(n - 130, n - 22, 1 / km_per_px, 50)
    sub = "80°S to the pole · 0° longitude (nearside) at top"
    out = {}
    out["relief"] = frame_svg(n, n, relief + grat + sb, "Lunar south pole · Relief", sub,
                              legend_svg(12, n - 64, "Elevation, km", [f"{v:.1f}" if i % 3 == 0 else "" for i, v in enumerate([-8.75] + RELIEF_T)], RELIEF_C, 14), disc)
    for key, L in LAYERS.items():
        v = gaussian_filter(sampler(grids[L["key"]].astype(np.float32))(lat, lon, 1), sig)
        bands = isobands(v, L["t"] + [1e9], L["c"], 0.7, stroke="rgba(15,34,54,.45)")
        out[key] = frame_svg(n, n, relief + bands + psr + grat + sb, f"Lunar south pole · {L['title'].split(' (')[0]}", sub,
                             legend_svg(12, n - 64, L["title"], [f"≥{t}" for t in L["t"]], L["c"]), disc)
    out["psr"] = frame_svg(n, n, relief + psr + grat + sb, "Lunar south pole · Permanently shadowed regions", sub,
                           legend_svg(12, n - 64, "Never sunlit in 2026", ["PSR"], [PSR_C], 60), disc)
    if sites:
        proj = lambda la, lo: polar_project(la, lo, n, cap)
        marks = site_marks([s for s in sites if s["lat"] < -90 + cap], proj, args.zone_km, 1 / km_per_px)
        key = ('<g><rect x="12" y="12" width="250" height="62" rx="3" fill="#fff" fill-opacity=".92" stroke="#E2DDD0"/>'
               f'<rect x="22" y="22" width="9" height="9" fill="{CA}"/><text x="38" y="30.5" font-size="10.5" fill="#0F2236" font-family="Roboto,Arial">NASA Artemis III candidate region</text>'
               f'<path d="M26.5,40L32,50L21,50Z" fill="{CB}"/><text x="38" y="49" font-size="10.5" fill="#0F2236" font-family="Roboto,Arial">ILRS or Chang’e-7 target</text>'
               f'<text x="22" y="66" font-size="9.5" fill="{MUTED}" font-family="Roboto,Arial">Zones {args.zone_km:g} km radius, true scale. Hollow = approximate.</text></g>')
        out["sites"] = frame_svg(n, n, relief + psr + grat + marks + sb, "Lunar south pole · Opposing sites", sub, key, disc)
    return out


# ---------------------------------------------------------------- global maps
def make_global(args, dem, grids):
    H, W = 720, 1440
    from importlib.util import spec_from_file_location, module_from_spec
    spec = spec_from_file_location("ll", Path(__file__).with_name("lunar_layers.py")); ll = module_from_spec(spec); spec.loader.exec_module(ll)
    h = ll.block_mean(dem, 2)
    relief_fill = isobands(h, [-99] + RELIEF_T + [99], RELIEF_C)
    relief = relief_fill + isolines(h, RELIEF_T, NAVY, 0.4, 0.35)        # contour lines on the relief map only
    grat = "".join(f'<line x1="0" y1="{(90 - la) * 4}" x2="{W}" y2="{(90 - la) * 4}" stroke="{NAVY}" stroke-opacity=".3" stroke-width=".6"/>' for la in range(-75, 90, 15)) + \
           "".join(f'<line x1="{(lo + 180) * 4}" y1="0" x2="{(lo + 180) * 4}" y2="{H}" stroke="{NAVY}" stroke-opacity=".3" stroke-width=".6"/>' for lo in range(-165, 180, 15))
    psr = isobands(smooth_wrap((grids["sun_pct"] <= 0).astype(np.float32), args.sigma), [0.5, 9], [PSR_C], 0.92, stroke="#7a5800")
    sub = "Equirectangular · 0° longitude at centre"
    out = {"relief": frame_svg(W, H, relief + grat, "The Moon · Relief", sub,
                               legend_svg(12, H - 64, "Elevation, km", [f"{v:.1f}" if i % 3 == 0 else "" for i, v in enumerate([-8.75] + RELIEF_T)], RELIEF_C, 14))}
    for key, L in LAYERS.items():
        v = smooth_wrap(grids[L["key"]].astype(np.float32), args.sigma)
        out[key] = frame_svg(W, H, relief_fill + isobands(v, L["t"] + [1e9], L["c"], 0.7, stroke="rgba(15,34,54,.45)") + psr + grat,
                             f"The Moon · {L['title'].split(' (')[0]}", sub, legend_svg(12, H - 64, L["title"], [f"≥{t}" for t in L["t"]], L["c"]))
    out["psr"] = frame_svg(W, H, relief_fill + psr + grat, "The Moon · Permanently shadowed regions", sub,
                           legend_svg(12, H - 64, "Never sunlit in 2026", ["PSR"], [PSR_C], 60))
    return out


# ---------------------------------------------------------------- Malapert close-up
def make_closeup(args, zones):
    P = zones["patches"]["mal"]; n, L = P["n"], P["half_km"]
    h = np.frombuffer(base64.b64decode(P["h_m"]), "<i2").reshape(n, n) / 1000.0     # rows = south -> north
    psr = np.frombuffer(base64.b64decode(P["psr"]), np.uint8).reshape(n, n) / 255.0
    W = 760; half = args.closeup_km; px = W / (2 * half)
    xs = np.linspace(-L, L, n)
    X, Y = np.meshgrid((xs + half) * px, (half - xs) * px)                            # SVG: y down = south
    xy = (X, Y)
    body = isobands(h, [-99] + RELIEF_T + [99], RELIEF_C, xy=xy) + isolines(h, np.arange(-8.75, 10.5, 0.25), NAVY, 0.5, 0.3, xy) \
        + isolines(h, RELIEF_T, NAVY, 1.0, 0.55, xy) + isobands(psr, [0.5, 9], [PSR_C], 0.9, stroke="#7a5800", xy=xy)
    c = np.array(P["c"]); e = np.array(P["e"]); nv = np.array(P["nv"])
    def loc(s):
        a, b = math.radians(s["lat"]), math.radians(s["lon"])
        v = np.array([math.cos(a) * math.sin(b), math.sin(a), math.cos(a) * math.cos(b)]); d = v @ c
        return R * (v @ e) / d, R * (v @ nv) / d
    A = next(s for s in zones["sites"] if s["n"] == P["a"]); B = next(s for s in zones["sites"] if s["n"] == P["b"])
    (ax, ay), (bx, by) = loc(A), loc(B)
    r = args.zone_km; d = math.hypot(bx - ax, by - ay)
    S = lambda x, y: ((x + half) * px, (half - y) * px)
    defs = (f'<defs><pattern id="hatch" width="7" height="7" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
            f'<line x1="0" y1="0" x2="0" y2="7" stroke="{NAVY}" stroke-width="2.2"/></pattern>'
            f'<clipPath id="za"><circle cx="{S(ax, ay)[0]:.1f}" cy="{S(ax, ay)[1]:.1f}" r="{r * px:.1f}"/></clipPath></defs>')
    zones_svg = ""
    for (x, y), col in [((ax, ay), CA if A["g"] == "a" else CB), ((bx, by), CA if B["g"] == "a" else CB)]:
        sx, sy = S(x, y)
        zones_svg += f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="{r * px:.1f}" fill="{col}" fill-opacity=".16" stroke="{col}" stroke-width="2.5"/>'
    sbx, sby = S(bx, by)
    zones_svg += f'<circle cx="{sbx:.1f}" cy="{sby:.1f}" r="{r * px:.1f}" fill="url(#hatch)" fill-opacity=".6" clip-path="url(#za)"/>'
    proj = lambda la, lo: S(*loc({"lat": la, "lon": lo}))
    marks = site_marks([A, B], proj, None, None)
    ov = 2 * r - d
    note = (f'<g><rect x="12" y="12" width="330" height="58" rx="3" fill="#fff" fill-opacity=".94" stroke="#E2DDD0"/>'
            f'<text x="22" y="32" font-size="12" font-weight="700" fill="{NAVY}" font-family="Roboto,Arial">Centres {d:.1f} km apart · {r:g} km zones</text>'
            f'<text x="22" y="50" font-size="11" fill="{"#9E2F1C" if ov > 0 else NAVY}" font-family="Roboto,Arial">{"Overlap " + format(ov, ".1f") + " km (hatched)" if ov > 0 else "Clear by " + format(-ov, ".1f") + " km"} · zones touch at {d / 2:.2f} km radius</text>'
            f'<text x="22" y="63" font-size="9" fill="{MUTED}" font-family="Roboto,Arial">250 m contours, bicubic from the 1/8° grid. Markers at region centres.</text></g>')
    north = f'<g><path d="M{W - 30},{86}l7,16h-14z" fill="{NAVY}"/><text x="{W - 30}" y="80" text-anchor="middle" font-size="11" font-weight="700" fill="{NAVY}" font-family="Roboto,Arial">N</text></g>'
    body = f'<g>{body}</g>{zones_svg}{marks}{north}' + scale_bar(W - 120, W - 22, px, 5)
    return defs + "" if False else frame_svg(W, W, defs + body, "Malapert · Safety zones", f"{A['n']} vs {B['n']}", note)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dem", type=Path, default=None)
    ap.add_argument("--grids", type=Path, required=True)
    ap.add_argument("--zones", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("flat_layers"))
    ap.add_argument("--n", type=int, default=800, help="polar map size in px")
    ap.add_argument("--cap", type=float, default=10.0, help="polar map extent, degrees from the pole")
    ap.add_argument("--sigma-km", type=float, default=3.0, help="smoothing of polar layers, km")
    ap.add_argument("--sigma", type=float, default=0.7, help="smoothing of global layers, cells")
    ap.add_argument("--zone-km", type=float, default=3.0)
    ap.add_argument("--tol-global", type=float, default=0.9, help="path thinning for global maps, px")
    ap.add_argument("--closeup-km", type=float, default=12.0, help="half-width of the Malapert close-up")
    args = ap.parse_args()

    from importlib.util import spec_from_file_location, module_from_spec
    spec = spec_from_file_location("ll", Path(__file__).with_name("lunar_layers.py")); ll = module_from_spec(spec); spec.loader.exec_module(ll)
    dem = ll.load_grid(args.dem or ssa("moon_dem.npz"))
    grids = dict(np.load(args.grids))
    zones = json.loads(args.zones.read_text()) if args.zones else None
    args.out.mkdir(parents=True, exist_ok=True)
    for name, svg in make_polar(args, dem, grids, zones["sites"] if zones else None).items():
        (args.out / f"south_pole_{name}.svg").write_text(svg)
    global TOL
    TOL = args.tol_global
    for name, svg in make_global(args, dem, grids).items():
        (args.out / f"global_{name}.svg").write_text(svg)
    if zones:
        (args.out / "malapert_closeup.svg").write_text(make_closeup(args, zones))
    for f in sorted(args.out.glob("*.svg")):
        print(f"{f.name:32s} {f.stat().st_size / 1e6:6.2f} MB")


if __name__ == "__main__":
    main()
