 Lunar south pole: layers, opposing sites, safety zones and landing plumes

![Animation: the Moon spins and tilts to the south pole, slope, sunlight, Earth-in-view and permanently shadowed layers flash on, Artemis III and ILRS sites appear with 3 km safety zones, and a lander's dust plume sprays past the zones at Malapert](media/south-pole-zones.gif)

The animation, 47 s. Blue squares: NASA Artemis III candidate regions. Red triangles: ILRS and Chang'e-7 targets.
Gold: permanently shadowed regions. Circles: 3 km safety zones, overlap hatched. Brown dots: illustrative landing-plume ejecta.
This preview is a small, text-free cut (360 px, 5 fps). The full page with captions and the zone explorer is built in
one minute (see below).

Source for the 3D animation and the flat 2D layer maps.

## Contents

```
pipeline/
  lunar_layers.py     global layers from elevation + JPL DE440: relief, slope, sunlight, Earth in view,
                      permanently shadowed regions; writes moon_layer_grids.npz, vector JSON and SVGs
  site_zones.py       Artemis III and ILRS site metrics and close-up terrain patches -> zone_data.json
  globe_textures.py   smooth terrain + layer grids for the animation -> globe_textures.json
  flat_layers.py      smooth flat SVG maps (south pole, global, Malapert close-up)
animation/
  animation_template.html   the animation page (HTML, CSS, three.js shaders and timeline)
  build_animation.py        injects data/*.json into the template
raw/
  moon_dem.npz, de440s_2020-2035.bsp, moon_pa_de440_200625_2020-2035.bpc   the raw lunar data everything is built on
  (LOLA-derived elevation grid, JPL DE440 ephemeris, DE440 lunar orientation), with licence and checksums
data/
  moon_layer_grids.npz                  0.25 deg grids computed from raw/ (sunlight, Earth in view, slope)
  zone_data.json                        site metrics and close-up terrain from site_zones.py
flat_layers/
  south_pole_*.svg, global_*.svg, malapert_closeup.svg
media/
  south-pole-zones-preview.gif   the preview shown at the top of this README
```

## Build the animation (about 1 minute)

```bash
pip install numpy scipy numba spiceypy shapely matplotlib scikit-image   # raw data is in raw/
cd pipeline
python globe_textures.py --grids ../data/moon_layer_grids.npz --out ../data/globe_textures.json
cd ../animation
python build_animation.py --textures ../data/globe_textures.json --zones ../data/zone_data.json
```

This writes animation/south-pole-zones.html. Open it in a browser (it loads three.js from cdn.jsdelivr.net).

## Run everything from the raw data

```bash
pip install numpy scipy numba spiceypy shapely matplotlib scikit-image   # raw data is in raw/
cd pipeline
python lunar_layers.py --out out                       # about 5 min at 0.25 deg; --res 8 for full DEM resolution; --year 2020-2035
python site_zones.py   --grids out/moon_layer_grids.npz --out out/zone_data.json
python globe_textures.py --grids out/moon_layer_grids.npz --out out/globe_textures.json
python flat_layers.py  --grids out/moon_layer_grids.npz --zones out/zone_data.json --out out/flat
cd ../animation
python build_animation.py --textures ../pipeline/out/globe_textures.json --zones ../pipeline/out/zone_data.json
```

`lunar_layers.py --from-grids ../data/moon_layer_grids.npz` redoes only the vector output (about 2 min), and the
other scripts can use ../data/moon_layer_grids.npz directly instead of out/.
The animation page loads three.js 0.160.0 from cdn.jsdelivr.net, so it needs internet when opened.

## Data and method

- Elevation: LOLA-derived 1/8 deg grid from the `ssa-data-lunar` package (LLNL, MIT licence); its highest
  and lowest points match LOLA's (5.4N 158.6W, +10.55 km; Antoniadi, 70.3S 172.4W, -8.73 km).
- Sun and Earth: JPL DE440 (de440s.bsp) with the DE440 lunar principal-axes orientation, every 3 hours
  through 2026. Each cell's horizon is traced in 48 azimuths out to 250 km, curvature included; the visible
  fraction of each disc is accumulated. PSR = cells the Sun never reached in 2026.
- Layers are computed on 0.25 deg cells (about 7.6 km), then lightly smoothed for display
  (0.7 cells globally, 3 km in the polar plane). South of 75S the animation terrain is bicubic-interpolated
  from the 1/8 deg grid onto a ~0.7 km polar grid: smooth, not extra real detail.
- Sites: Artemis III regions at LROC NAC mosaic centres where available (Mons Mouton at its summit);
  Chang'e-7 at 88.8S 123.4E; ILRS Amundsen and Malapert (Shi et al. 2022), Malapert placed at the massif's
  high point in the DEM. Nobile Rim 1, Slater Plain and the Shackleton ridge are approximate and excluded from
  distance analysis.
- Safety zones: circles of 1 to 5 km radius (default 3 km), the range in Paulson and Roberts (AIAA SciTech 2026).
- Plume: 2,600 illustrative ballistic particles, launch angle 0.5 to 3 deg, speed 80 to 600 m/s, lunar g = 1.62 m/s^2,
  range = v^2 sin(2 theta) / g. Not a model of any specific vehicle; landers are notional and not to scale.
