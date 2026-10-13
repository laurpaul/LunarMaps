#!/usr/bin/env python3
"""
globe_textures.py  -  smooth terrain and layer grids for the 3D globe pages.

Writes globe_textures.json with zlib-compressed, base64 grids (heights: 10 m units, row deltas,
byte planes; layers: four planes, row deltas mod 256):
  h_eq     global elevation, 0.25 deg equirectangular (row 0 = 90S, col 0 = 180W)
  h_s      south polar elevation, polar stereographic to COLAT deg from the pole,
           bicubic from the 1/8 deg DEM (smooth, interpolated)
  l_eq     global layers RGBA uint8 at 0.25 deg: R sunlight %, G Earth in view %, B slope (0-30 deg),
           A permanently-shadowed mask; each grid lightly Gaussian-smoothed (--sigma cells)
  l_s      the same layers resampled bicubic onto the south polar grid
Polar grids use uv = (0.5 + r sin(lon), 0.5 + r cos(lon)), r = tan(colat/2) / tan(COLAT/2) / 2,
array row = v, column = u.

Usage: python globe_textures.py --grids out/moon_layer_grids.npz --out out/globe_textures.json
"""
import argparse, base64, json, math, zlib
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates


def ssa(name):
    """Raw input file: the package's raw/ folder first, else the installed ssa-data-lunar package."""
    raw = Path(__file__).resolve().parent.parent / "raw" / name      # bundled copy in this package
    if raw.is_file():
        return raw
    from importlib.resources import files
    return Path(str(files("ssa_data_lunar").joinpath("data", name)))


def pack(a):
    return base64.b64encode(zlib.compress(np.ascontiguousarray(a).tobytes(), 9)).decode()


def pack_height(h_km):
    """10 m units, delta along rows, low/high bytes split into two planes, zlib."""
    q = np.round(h_km * 100).astype(np.int32)
    d = np.diff(q, axis=1, prepend=0).astype("<i2")
    b = d.view(np.uint8)
    return pack(np.concatenate([b[..., 0::2].ravel(), b[..., 1::2].ravel()]))


def pack_layers(rgba):
    """Four planes, delta along rows (mod 256), zlib."""
    pl = np.moveaxis(rgba, -1, 0).astype(np.int16)
    return pack((np.diff(pl, axis=2, prepend=0) % 256).astype(np.uint8))


def sampler(a, pad=4):
    H, W = a.shape
    ap = np.concatenate([a[:, -pad:], a, a[:, :pad]], axis=1)
    def f(lat, lon, order=3):
        r = (90 - lat) * H / 180 - 0.5
        c = ((lon + 180) % 360) * W / 360 - 0.5 + pad
        return map_coordinates(ap, [r.ravel(), c.ravel()], order=order, mode="nearest").reshape(lat.shape)
    return f


def smooth_wrap(a, sigma):
    if sigma <= 0:
        return a
    pad = int(4 * sigma) + 2
    ap = np.concatenate([a[:, -pad:], a, a[:, :pad]], axis=1)
    return gaussian_filter(ap, sigma, mode="nearest")[:, pad:-pad]


def polar_latlon(n, colat_deg):
    t = math.tan(math.radians(colat_deg) / 2)
    u = (np.arange(n) + 0.5) / n
    U, V = np.meshgrid(u, u)               # rows = v, cols = u
    dx, dy = U - 0.5, V - 0.5
    r = np.hypot(dx, dy)
    col = 2 * np.degrees(np.arctan(r * 2 * t))
    lon = np.degrees(np.arctan2(dx, dy))
    return -90 + col, lon


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dem", type=Path, default=None)
    ap.add_argument("--grids", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--colat", type=float, default=16.0)
    ap.add_argument("--n-h", type=int, default=1280)
    ap.add_argument("--n-l", type=int, default=1024)
    ap.add_argument("--sigma", type=float, default=0.7)
    ap.add_argument("--polar-sigma-km", type=float, default=3.0)
    args = ap.parse_args()

    import importlib.util
    spec = importlib.util.spec_from_file_location("ll", Path(__file__).with_name("lunar_layers.py"))
    ll = importlib.util.module_from_spec(spec); spec.loader.exec_module(ll)
    dem = ll.load_grid(args.dem or ssa("moon_dem.npz"))            # 1/8 deg, km
    g = np.load(args.grids)
    sun, earth, slope = g["sun_pct"], g["earth_pct"], g["slope_deg"]
    psr = (sun <= 0).astype(np.float32)

    # global elevation at 0.25 deg, flipped so row 0 = south
    h_eq = ll.block_mean(dem, 2)[::-1]
    # south polar elevation, bicubic from 1/8 deg
    la, lo = polar_latlon(args.n_h, args.colat)
    h_s = sampler(dem)(la, lo, 3)

    S = lambda a: smooth_wrap(a.astype(np.float32), args.sigma)
    sun_s, earth_s, slope_s, psr_s = S(sun), S(earth), S(slope), S(psr)
    def rgba(su, ea, sl, ps):
        return np.stack([np.clip(su * 2.55, 0, 255), np.clip(ea * 2.55, 0, 255),
                         np.clip(sl / 30 * 255, 0, 255), np.clip(ps * 255, 0, 255)], -1).round().astype(np.uint8)
    l_eq = rgba(sun_s, earth_s, slope_s, psr_s)[::-1]
    la2, lo2 = polar_latlon(args.n_l, args.colat)
    # polar layers: resample the unsmoothed 0.25 deg grids, then smooth in the polar plane so the
    # smoothing is the same in every direction (equirectangular cells are long and thin near the pole)
    km_per_px = 2 * math.radians(args.colat) * 1737.4 / args.n_l
    sig_px = args.polar_sigma_km / km_per_px
    pol = [gaussian_filter(sampler(a.astype(np.float32))(la2, lo2, 1), sig_px, mode="nearest") for a in (sun, earth, slope, psr)]
    l_s = rgba(*pol)

    out = {"colat": args.colat, "sigma": args.sigma,
           "h_eq": {"w": h_eq.shape[1], "h": h_eq.shape[0], "data": pack_height(h_eq)},
           "h_s": {"n": args.n_h, "data": pack_height(h_s)},
           "l_eq": {"w": l_eq.shape[1], "h": l_eq.shape[0], "data": pack_layers(l_eq)},
           "l_s": {"n": args.n_l, "data": pack_layers(l_s)}}
    args.out.write_text(json.dumps(out, separators=(",", ":")))
    print("wrote", args.out, round(args.out.stat().st_size / 1e6, 2), "MB",
          "| h_s range", float(h_s.min()), float(h_s.max()))


if __name__ == "__main__":
    main()
