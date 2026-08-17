"""Building the 50 km border strips.

For each partitioned group: find where the two countries touch, buffer that
line by 50 km, and clip the ethnic polygons to the strip. A side also has to
lie within 50 km of the *other* side's polygons, otherwise we'd keep bits of
the group that sit on the border somewhere far away from their counterpart.

The dissolve at the end matters: GREG splits a group into several polygons,
and zonal_stats returns one row per polygon. Without dissolving, a single
ethnicity-country side shows up two or three times in the same year and the
panel silently doubles in size.
"""

import geopandas as gpd
from pyproj import CRS
from shapely.ops import unary_union
from shapely.validation import make_valid

import config


def buffer_crs():
    """Equidistant conic for Africa, with Web Mercator as a fallback."""
    try:
        CRS.from_user_input(config.BUFFER_CRS)
        return config.BUFFER_CRS
    except Exception:
        print(f"{config.BUFFER_CRS} not available, falling back to {config.FALLBACK_CRS}")
        return config.FALLBACK_CRS


def country_name(rows):
    for col in ("NAME_EN", "NAME", "ADMIN", "SOVEREIGNT"):
        if col in rows.columns:
            return rows.iloc[0][col]
    return "?"


def find_country(countries, fips, verbose=False):
    """Look up a country in Natural Earth.

    The raw FIPS code goes against the FIPS columns and the mapped ISO code
    only against the ISO columns - mixing them up gets you Zambia for South
    Africa and Chile for Cote d'Ivoire.
    """
    iso = config.FIPS_TO_ISO.get(fips, fips)
    attempts = [(c, fips) for c in config.FIPS_COLUMNS]
    attempts += [(c, iso) for c in config.ISO_COLUMNS]

    for col, key in attempts:
        if col not in countries.columns:
            continue
        hit = countries[countries[col] == key]
        if len(hit):
            if verbose:
                print(f"    {fips} -> {key} via {col}: {country_name(hit)}")
            return hit
    return None


def shared_border(countries, fips_a, fips_b, verbose=False):
    """The line where two countries meet, or None if they don't."""
    a = find_country(countries, fips_a, verbose)
    b = find_country(countries, fips_b, verbose)
    if a is None or b is None:
        return None

    border = make_valid(unary_union(a.geometry)).intersection(
        make_valid(unary_union(b.geometry))
    )
    return None if border.is_empty else border


def clip_group(polys, border, fips_a, fips_b):
    """Clip one group's polygons to the border strip. Returns the clipped rows
    and which sides survived."""
    metres = config.BUFFER_KM * 1000
    strip = border.buffer(metres)

    side_a = make_valid(unary_union(polys[polys["FIPS_CNTRY"] == fips_a].geometry))
    side_b = make_valid(unary_union(polys[polys["FIPS_CNTRY"] == fips_b].geometry))
    near = {fips_a: side_b.buffer(metres), fips_b: side_a.buffer(metres)}

    kept, sides = [], set()
    for _, row in polys.iterrows():
        clipped = make_valid(row.geometry).intersection(strip)
        if clipped.is_empty:
            continue
        clipped = clipped.intersection(near[row["FIPS_CNTRY"]])
        if clipped.is_empty:
            continue
        new = row.copy()
        new["geometry"] = clipped
        kept.append(new)
        sides.add(row["FIPS_CNTRY"])

    return kept, sides


def build(group_ids, verbose=True):
    """Border strips for every group in group_ids.

    Returns the dissolved buffers plus a dict of what got dropped and why,
    which goes into the sample construction table in the thesis.
    """
    crs = buffer_crs()
    print(f"buffer CRS: {crs}, width {config.BUFFER_KM} km\n")

    countries = gpd.read_file(config.COUNTRIES_10M).to_crs(crs)
    greg = gpd.read_file(config.GREG_SHP).to_crs(crs)
    greg = greg[greg["G1ID"].isin(group_ids)]

    kept_rows = []
    dropped = {"no_country": [], "no_border": [], "no_overlap": []}

    for gid in sorted(group_ids):
        polys = greg[greg["G1ID"] == gid]
        name = polys["G1LONGNAM"].iloc[0] if len(polys) else "?"
        fips = sorted(polys["FIPS_CNTRY"].unique())

        if len(fips) != 2:
            dropped["no_country"].append(gid)
            continue

        fips_a, fips_b = fips
        border = shared_border(countries, fips_a, fips_b, verbose)
        if border is None:
            print(f"  {gid} {name} ({fips_a}/{fips_b}): no shared border")
            dropped["no_border"].append(gid)
            continue

        rows, sides = clip_group(polys, border, fips_a, fips_b)
        if len(sides) < 2:
            print(f"  {gid} {name} ({fips_a}/{fips_b}): only {sorted(sides)} reaches "
                  f"the strip")
            dropped["no_overlap"].append(gid)
            continue

        kept_rows.extend(rows)
        print(f"  {gid} {name} ({fips_a}/{fips_b}): ok, {len(rows)} polygons")

    raw = gpd.GeoDataFrame(kept_rows, crs=crs)
    buffers = raw.dissolve(by=["G1ID", "FIPS_CNTRY"], aggfunc="first").reset_index()
    buffers["area_km2"] = buffers.geometry.area / 1e6
    buffers["unit_id"] = buffers["G1ID"].astype(str) + "_" + buffers["FIPS_CNTRY"]

    print(f"\npolygons before dissolve: {len(raw)}, units after: {len(buffers)}")
    print(f"groups with a buffer: {buffers['G1ID'].nunique()}")
    for reason, ids in dropped.items():
        if ids:
            print(f"dropped ({reason}): {ids}")

    sides = buffers.groupby("G1ID").size()
    bad = sides[sides != 2]
    if len(bad):
        raise ValueError(f"groups without two sides after dissolve: {bad.index.tolist()}")

    return buffers.to_crs(4326), dropped


def run(group_ids):
    buffers, dropped = build(group_ids)
    buffers.to_file(config.WORK / "buffers.gpkg", driver="GPKG")
    buffers.drop(columns="geometry").to_csv(
        config.WORK / "buffer_areas.csv", index=False
    )
    return buffers, dropped
