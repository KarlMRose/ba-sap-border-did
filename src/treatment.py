import pandas as pd

import config


def load_vreeland(cow_codes):
    v = pd.read_stata(config.VREELAND_DTA)
    v["year"] = v["year"].astype(int)
    v = v[
        v["year"].between(config.YEAR_MIN, config.YEAR_MAX)
        & v["ccode_cow"].isin(cow_codes)
        & v["type"].notna()
    ]
    return v


def sap_indicators(cow_codes):
    v = load_vreeland(cow_codes)

    def flags(x):
        out = {
            name: int(x["type"].isin(types).any())
            for name, types in config.SAP_DEFINITIONS.items()
        }
        out["program_type"] = ", ".join(x["type"].dropna().unique())
        return pd.Series(out)

    sap = (
        v.groupby(["year", "ccode_cow"])
        .apply(flags, include_groups=False)
        .reset_index()
        .rename(columns={"ccode_cow": "COW"})
    )
    sap["COW"] = sap["COW"].astype(int)

    counts = {k: int(sap[k].sum()) for k in config.SAP_DEFINITIONS}
    print("country-years by definition:", counts)
    return sap


def build_skeleton(units, sap):
    years = pd.DataFrame({"year": range(config.YEAR_MIN, config.YEAR_MAX + 1)})
    skel = units.merge(years, how="cross").merge(sap, on=["COW", "year"], how="left")

    for name in config.SAP_DEFINITIONS:
        skel[name] = skel[name].fillna(0).astype(int)
    skel["program_type"] = skel["program_type"].fillna("")

    print(f"skeleton: {len(skel)} rows, {skel['G1ID'].nunique()} groups")
    return skel


def clean_did_groups(skeleton, definition="narrow"):
    ever = skeleton.groupby("unit_id")[definition].transform("max")
    skeleton = skeleton.assign(ever_treated=ever)

    by_group = skeleton.groupby("G1ID")["ever_treated"].agg(["min", "max"])
    clean = by_group[(by_group["min"] == 0) & (by_group["max"] == 1)].index.tolist()
    print(f"[{definition}] groups with one treated and one control side: {len(clean)}")

    if config.EXCLUDE_SOUTH_AFRICA:
        sa = skeleton.loc[skeleton["FIPS_CNTRY"] == "SF", "G1ID"].unique()
        dropped = sorted(set(clean) & set(sa))
        clean = [g for g in clean if g not in sa]
        print(f"dropped for South Africa: {dropped} -> {len(clean)} groups left")

    return sorted(clean), skeleton


def run(units):
    sap = sap_indicators(units["COW"].unique().tolist())
    skeleton = build_skeleton(units, sap)
    group_ids, skeleton = clean_did_groups(skeleton)

    skeleton.to_csv(config.WORK / "unit_year_treatment.csv", index=False)
    pd.Series(group_ids, name="G1ID").to_csv(
        config.WORK / "clean_did_groups.csv", index=False
    )
    return skeleton, group_ids
