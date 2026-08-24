import geopandas as gpd
from pyproj import CRS
from shapely.ops import unary_union
from shapely.validation import make_valid

import config


def buffer_crs():
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
    a = find_country(countries, fips_a, verbose)
    b = find_country(countries, fips_b, verbose)
    if a is None or b is None:
        return None

    border = make_valid(unary_union(a.geometry)).intersection(
        make_valid(unary_union(b.geometry))
    )
    return None if border.is_empty else border


def clip_group(polys, border, fips_a, fips_b):
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
            print(f"  {gid} {name}: {len(fips)} countries {fips}, skipped")
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

    per_group = buffers.groupby("G1ID").size()
    bad = per_group[per_group != 2]
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
