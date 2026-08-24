import re

import numpy as np
import pandas as pd
from rasterstats import zonal_stats

import config


def raster_files():
    files = sorted(
        p for p in config.VIIRS_DIR.glob("*.tif")
        if "calDMSP" in p.name or "simVIIRS" in p.name
    )
    print(f"rasters found: {len(files)}")
    return files


def parse_filename(path):
    match = re.search(r"NTL_(\d{4})_", path.name)
    year = int(match.group(1)) if match else None
    source = "calDMSP" if "calDMSP" in path.name else "simVIIRS"
    return year, source


def extract(buffers):
    rows = []
    for path in raster_files():
        year, source = parse_filename(path)
        if year is None:
            print(f"  skipping {path.name}, no year in filename")
            continue

        stats = zonal_stats(
            buffers, str(path), stats=["mean", "count"],
            nodata=-9999, all_touched=False,
        )
        for unit, stat in zip(buffers.itertuples(), stats):
            rows.append({
                "G1ID": unit.G1ID,
                "FIPS_CNTRY": unit.FIPS_CNTRY,
                "unit_id": unit.unit_id,
                "name": getattr(unit, "G1LONGNAM", ""),
                "year": year,
                "source": source,
                "ntl_mean": stat["mean"],
                "ntl_count": stat["count"],
            })
        print(f"  {year} ({source})")

    df = pd.DataFrame(rows)
    df = df[~((df["year"] == 2013) & (df["source"] == "simVIIRS"))]
    return df.sort_values(["unit_id", "year"]).reset_index(drop=True)


def add_outcome(df, offset=None):
    c = config.NTL_OFFSET if offset is None else offset
    df = df.copy()
    df["ln_ntl"] = np.log(df["ntl_mean"].clip(lower=0) + c)
    return df


def check(df):
    missing = df["ntl_mean"].isna().sum()
    print(f"missing radiance values: {missing}")

    counts = df.groupby("unit_id")["ntl_count"].agg(["min", "max"])
    counts["spread_pct"] = 100 * (counts["max"] - counts["min"]) / counts["min"].clip(lower=1)
    worst = counts.sort_values("spread_pct", ascending=False).head(5)
    print("largest pixel count variation:")
    print(worst.to_string())
    return counts


def run(buffers):
    df = add_outcome(extract(buffers))
    check(df)
    df.to_csv(config.WORK / "ntl_by_unit_year.csv", index=False)
    return df
