# Raw inputs

Everything in this project is computed from these three files. They come from the
`ssa-data-lunar` 0.0.1 package on PyPI (Lawrence Livermore National Laboratory, LLNL-CODE-2015996,
MIT licence; see LICENSE_ssa-data-lunar.txt and NOTICE_ssa-data-lunar.txt). The elevation grid is an unchanged copy;
the two SPICE kernels are time subsets (2020-2035) of the originals, with identical values inside that span.

| File | What it is | Size |
|---|---|---|
| moon_dem.npz | Global lunar elevation, array `elev_km`, 1440 x 2880 float32, km relative to the 1737.4 km sphere. Equirectangular, 1/8 deg pixels, row 0 = 90N, column 0 = 180W, pixel-centred. LOLA-derived: its extremes match LOLA (max +10.55 km at 5.4N 158.6W; min -8.73 km at Antoniadi, 70.3S 172.4W). | 12.6 MB |
| de440s_2020-2035.bsp | JPL DE440 planetary ephemeris (de440s.bsp), all 14 segments cut to 2020-01-01 to 2036-01-01 with SPICE SPKSUB so it fits the file-size limit. Positions are identical to the full kernel (checked: 0 km difference for Sun and Earth seen from the Moon). Full 1849-2150 file: https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440s.bsp | 1.8 MB |
| moon_pa_de440_200625_2020-2035.bpc | DE440 lunar orientation, principal-axes frame (SPICE binary PCK, frame ID 31008, NAIF/JPL), the same Chebyshev records cut to 2020-01-01 to 2036-01-01 so the package fits the file-size limit. Orientation matrices are identical to the full kernel (checked: 0 difference). Full 1550-2650 file: https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/moon_pa_de440_200625.bpc | 0.2 MB |

The scripts in ../pipeline read these from here first and fall back to the installed package.
SHA-256 checksums are in checksums.sha256.

Computed intermediate: ../data/moon_layer_grids.npz holds the 0.25 deg grids produced by
lunar_layers.py from these inputs (elev_km, sun_pct, earth_pct, slope_deg), stored to 0.01 (% and degrees)
and 1 m, with every never-sunlit cell kept at exactly 0, so the later steps can be rerun without the
5-minute horizon and ephemeris computation.
