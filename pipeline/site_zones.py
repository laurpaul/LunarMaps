#!/usr/bin/env python3
"""
site_zones.py  -  site metrics and close-up terrain patches for the south pole sites animation.

Reads the elevation grid (same default as lunar_layers.py) and the computed layer grids
(moon_layer_grids.npz from lunar_layers.py), then writes zone_data.json with:
  sites    elevation (bicubic, 1/8 deg grid), slope (1/8 deg), sunlight and Earth-in-view
           (0.25 deg layer grids, bilinear), distance to the nearest permanently shadowed cell
  patches  smooth terrain patches (local tangent-plane grid, heights in m) with 250 m contour
           lines and a PSR mask, centred on chosen site pairs

Usage:  python site_zones.py --grids out/moon_layer_grids.npz --out out/zone_data.json
"""
import argparse, base64, json, math
from pathlib import Path
import numpy as np
from scipy.ndimage import map_coordinates
from skimage.measure import find_contours

R = 1737.4
SITES = [
    ("Peak near Cabeus B", -83.6, -68.8, "a", "LROC mosaic centre"),
    ("Haworth", -86.7, -23.7, "a", "LROC mosaic centre"),
    ("Malapert Massif", -86.0, 0.3, "a", "LROC mosaic centre"),
    ("Mons Mouton Plateau", -84.3, 30.6, "a", "LROC mosaic centre"),
    ("Mons Mouton", -84.6, 31.0, "a", "summit"),
    ("Nobile Rim 1", -85.4, 38.0, "a", "approximate"),
    ("Nobile Rim 2", -83.9, 58.7, "a", "LROC mosaic centre"),
    ("de Gerlache Rim 2", -88.5, -66.7, "a", "LROC mosaic centre"),
    ("Slater Plain", -88.08, 111.28, "a", "approximate"),
    ("Chang'e-7 target", -88.8, 123.4, "b", "published target"),
    ("Shackleton ridge", -89.45, -137.3, "b", "approximate"),
    ("Amundsen", -84.5, 82.8, "b", "crater centre"),
    ("Malapert summit", -85.975, 2.0, "b", "DEM high point of the massif"),
]


def ssa(name):
    """Raw input file: the package's raw/ folder first, else the installed ssa-data-lunar package."""
    raw = Path(__file__).resolve().parent.parent / "raw" / name      # bundled copy in this package
    if raw.is_file():
        return raw
    from importlib.resources import files
    return Path(str(files("ssa_data_lunar").joinpath("data", name)))


def unit(lat, lon):
    a, b = math.radians(lat), math.radians(lon)
    return np.array([math.cos(a) * math.sin(b), math.sin(a), math.cos(a) * math.cos(b)])


def latlon(v):
    v = v / np.linalg.norm(v, axis=-1, keepdims=True)
    return np.degrees(np.arcsin(np.clip(v[..., 1], -1, 1))), np.degrees(np.arctan2(v[..., 0], v[..., 2]))


class Grid:
    """Equirectangular grid sampler (row 0 = 90N, col 0 = 180W, pixel-centred) with lon wrap."""
    def __init__(self, a, pad=4):
        self.H, self.W = a.shape
        self.pad = pad
        self.a = np.concatenate([a[:, -pad:], a, a[:, :pad]], axis=1)

    def __call__(self, lat, lon, order=3):
        lat, lon = np.asarray(lat, float), np.asarray(lon, float)
        r = (90 - lat) * self.H / 180 - 0.5
        c = ((lon + 180) % 360) * self.W / 360 - 0.5 + self.pad
        return map_coordinates(self.a, [r.ravel(), c.ravel()], order=order, mode="nearest").reshape(lat.shape)


def basis(c):
    up = np.array([0, 1.0, 0]) if abs(c[1]) < 0.99 else np.array([1.0, 0, 0])
    e = np.cross(up, c); e /= np.linalg.norm(e)
    n = np.cross(c, e)
    return e, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dem", type=Path, default=None)
    ap.add_argument("--grids", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--half-km", type=float, default=60.0)
    ap.add_argument("--n", type=int, default=181)
    ap.add_argument("--contour-m", type=float, default=250.0)
    args = ap.parse_args()

    import importlib.util
    spec = importlib.util.spec_from_file_location("ll", Path(__file__).with_name("lunar_layers.py"))
    ll = importlib.util.module_from_spec(spec); spec.loader.exec_module(ll)

    dem = ll.load_grid(args.dem or ssa("moon_dem.npz"))
    elev = Grid(dem)
    slope = Grid(ll.slope_deg(dem))
    g = np.load(args.grids)
    sun, earth = Grid(g["sun_pct"]), Grid(g["earth_pct"])
    Hs, Ws = g["sun_pct"].shape
    psr_mask = (g["sun_pct"] <= 0).astype(np.float32)
    psr = Grid(psr_mask)
    rr, cc = np.nonzero(psr_mask)
    psr_lat = 90 - (rr + 0.5) * 180 / Hs
    psr_lon = -180 + (cc + 0.5) * 360 / Ws
    psr_vec = np.stack([np.cos(np.radians(psr_lat)) * np.sin(np.radians(psr_lon)), np.sin(np.radians(psr_lat)),
                        np.cos(np.radians(psr_lat)) * np.cos(np.radians(psr_lon))], -1)

    def snap(lat, lon, rmax, rmin=0.0):
        """Best-lit cell with slope < 10 deg within rmax km of a named feature (outside rmin km)."""
        c = unit(lat, lon); e, n = basis(c)
        xs = np.arange(-rmax, rmax + 0.5, 0.5)
        X, Y = np.meshgrid(xs, xs); rad = np.hypot(X, Y)
        P = c + (X[..., None] / R) * e + (Y[..., None] / R) * n
        la, lo = latlon(P)
        s_ = sun(la, lo, 1); sl = slope(la, lo, 1)
        ok = (rad <= rmax) & (rad >= rmin) & (sl < 10)
        score = np.where(ok, s_, -1)
        i = np.unravel_index(np.argmax(score), score.shape)
        return float(la[i]), float(lo[i])

    sites = []
    for name, lat, lon, grp, src in SITES:
        if src.startswith("snap"):
            parts = [float(x) for x in src.split(":")[1:]]
            lat, lon = snap(lat, lon, parts[0], parts[1] if len(parts) > 1 else 0.0)
            src = f"approximate: best-lit point under 10 deg slope within {parts[0]:g} km of the named feature"
        v = unit(lat, lon)
        d_psr = float(R * np.arccos(np.clip(psr_vec @ v, -1, 1)).min())
        sites.append({"n": name, "lat": lat, "lon": lon, "g": grp, "src": src,
                      "elev_km": round(float(elev(lat, lon)), 3),
                      "slope_deg": round(float(slope(lat, lon, 1)), 1),
                      "sun_pct": round(float(sun(lat, lon, 1)), 1),
                      "earth_pct": round(float(earth(lat, lon, 1)), 1),
                      "psr_km": round(d_psr, 1)})

    # focus pairs: the two closest Artemis-ILRS pairs that use different sites
    # approximate positions are excluded from distance analysis
    for s in sites:
        s["approx"] = s["src"].startswith("approximate")
    A_ = [s for s in sites if s["g"] == "a" and not s["approx"]]; B_ = [s for s in sites if s["g"] == "b" and not s["approx"]]
    pairs = sorted(((float(R * np.arccos(np.clip(unit(a["lat"], a["lon"]) @ unit(b["lat"], b["lon"]), -1, 1))), a["n"], b["n"])
                    for a in A_ for b in B_))
    d0, a0, b0 = pairs[0]
    PATCHES = {"mal": (a0, b0), "sha": ("Chang'e-7 target", "Chang'e-7 target")}
    print("closest published pair:", a0, b0, round(d0, 2), "km")
    patches = {}
    L, N = args.half_km, args.n
    xs = np.linspace(-L, L, N)
    for key, (a, b) in PATCHES.items():
        sa = next(s for s in sites if s["n"] == a); sb = next(s for s in sites if s["n"] == b)
        c = unit(sa["lat"], sa["lon"]) + unit(sb["lat"], sb["lon"]); c /= np.linalg.norm(c)
        e, n = basis(c)
        X, Y = np.meshgrid(xs, xs)                       # rows = y (south -> north), cols = x (west -> east)
        P = c + (X[..., None] / R) * e + (Y[..., None] / R) * n
        la, lo = latlon(P)
        h = elev(la, lo, 3)
        pm = np.clip(psr(la, lo, 1), 0, 1)
        lines = []
        lv0 = math.ceil(h.min() * 1000 / args.contour_m) * args.contour_m
        for lvm in np.arange(lv0, h.max() * 1000, args.contour_m):
            for cl in find_contours(h, lvm / 1000):
                if len(cl) < 3:
                    continue
                pts = np.stack([np.interp(cl[:, 1], np.arange(N), xs), np.interp(cl[:, 0], np.arange(N), xs)], -1)
                lines.append((float(lvm), pts))
        coords = np.concatenate([p for _, p in lines]) if lines else np.zeros((0, 2))
        patches[key] = {
            "a": a, "b": b,
            "c_lat": float(latlon(c)[0]), "c_lon": float(latlon(c)[1]), "half_km": L, "n": N,
            "e": e.tolist(), "nv": n.tolist(), "c": c.tolist(),
            "h_m": base64.b64encode(np.round(h * 1000).astype("<i2").tobytes()).decode(),
            "psr": base64.b64encode(np.round(pm * 255).astype(np.uint8).tobytes()).decode(),
            "lines": {"lv": [lv for lv, _ in lines], "len": [len(p) for _, p in lines],
                      "xy": base64.b64encode(np.round(coords * 100).astype("<i2").tobytes()).decode()},
            "h_range_m": [float(h.min() * 1000), float(h.max() * 1000)],
        }
        print(key, "contour lines", len(lines), "points", len(coords), "h", patches[key]["h_range_m"])
    for s in sites[:0]:
        print(f"{s['n']:20s} {s['elev_km']:6.2f} km  slope {s['slope_deg']:4.1f}  sun {s['sun_pct']:5.1f}  earth {s['earth_pct']:5.1f}  psr {s['psr_km']:5.1f} km")
    args.out.write_text(json.dumps({"sites": sites, "patches": patches, "contour_m": args.contour_m}, separators=(",", ":")))


if __name__ == "__main__":
    main()
