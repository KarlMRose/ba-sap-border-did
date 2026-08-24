import geopandas as gpd

import config


def load_greg():
    greg = gpd.read_file(config.GREG_SHP)
    print(f"GREG polygons: {len(greg)}, distinct groups: {greg['G1ID'].nunique()}")
    return greg


def group_countries(greg):
    return (
        greg.groupby("G1ID")
        .agg(
            name=("G1LONGNAM", "first"),
            n_countries=("FIPS_CNTRY", "nunique"),
            countries=("FIPS_CNTRY", lambda s: sorted(s.unique())),
        )
        .reset_index()
    )


def partitioned_in_ssa(groups):
    split = groups[groups["n_countries"] == 2].copy()
    print(f"split across two countries: {len(split)}")

    keep = split["countries"].apply(lambda cs: all(c in config.SSA_FIPS for c in cs))
    ssa = split[keep].copy()
    ssa["country_pair"] = ssa["countries"].str.join("/")
    print(f"both countries in SSA: {len(ssa)}")
    return ssa


def build_units(greg, partitioned):
    units = (
        greg[greg["G1ID"].isin(partitioned["G1ID"])]
        .groupby(["G1ID", "FIPS_CNTRY"])
        .agg(
            name=("G1LONGNAM", "first"),
            COW=("COW", "first"),
            n_polygons=("G1ID", "size"),
        )
        .reset_index()
    )
    units["unit_id"] = units["G1ID"].astype(str) + "_" + units["FIPS_CNTRY"]
    units["COW"] = units["COW"].astype(int)

    sides = units.groupby("G1ID").size()
    bad = sides[sides != 2]
    if len(bad):
        raise ValueError(f"groups without two sides: {bad.index.tolist()}")

    return units


def run():
    greg = load_greg()
    groups = group_countries(greg)
    partitioned = partitioned_in_ssa(groups)
    units = build_units(greg, partitioned)

    print(f"\ncandidate units: {len(units)} ({units['G1ID'].nunique()} groups)")
    print("\nmost common country pairs:")
    print(partitioned["country_pair"].value_counts().head(10).to_string())

    units.to_csv(config.WORK / "units.csv", index=False)
    partitioned.drop(columns="countries").to_csv(
        config.WORK / "partitioned_groups.csv", index=False
    )
    return units, partitioned
