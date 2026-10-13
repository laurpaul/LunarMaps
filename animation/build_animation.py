#!/usr/bin/env python3
"""Inject the pipeline outputs into the animation template and write a single self-contained HTML file.

python build_animation.py --textures ../data/globe_textures.json --zones ../data/zone_data.json --out south-pole-zones.html
The page loads three.js 0.160.0 from cdn.jsdelivr.net, so it needs an internet connection when opened.
"""
import argparse
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--template", type=Path, default=Path(__file__).with_name("animation_template.html"))
ap.add_argument("--textures", type=Path, required=True, help="globe_textures.json from globe_textures.py")
ap.add_argument("--zones", type=Path, required=True, help="zone_data.json from site_zones.py")
ap.add_argument("--out", type=Path, default=Path("south-pole-zones.html"))
a = ap.parse_args()
t = a.template.read_text()
html = t.replace("/*GT*/", "window.GLOBE_TEX=" + a.textures.read_text() + ";").replace("/*ZDATA*/", "window.ZONE_DATA=" + a.zones.read_text() + ";")
body = "<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">\n"
a.out.write_text(body + html.replace("</style>", "</style>\n</head><body>", 1) + "\n</body></html>\n")
print("wrote", a.out, round(a.out.stat().st_size / 1e6, 1), "MB")
