"""Assembling the estimation panel.

Two things happen here. First the sample rules: one treated and one control
side per group, a first treatment year, no groups that were already treated in
1992, and a control side that stays clean around the treatment date. Then the
within-pair difference, which is what the main specification actually runs on.
"""

import pandas as pd

import config


def _keep_one_treated_side(df):
    sides = df.groupby("G1ID")["ever_treated"].agg(["min", "max"])
    keep = sides[(sides["min"] == 0) & (sides["max"] == 1)].index
    return df[df["G1ID"].isin(keep)].copy()


def _control_stays_clean(group, definition, window):
    """True if the control side has no programme of its own within the window
    around the treated side's entry. Otherwise the comparison is contaminated."""
    t0 = int(group["first_sap_year"].iloc[0])
    control = group[group["ever_treated"] == 0]
    inside = control[control["year"].between(t0 - window, t0 + window)]
    return not bool(inside[definition].any())


def build(ntl, skeleton, definition="narrow", label=None):
    """The unit-year panel for one SAP definition."""
    label = label or definition
    treat = skeleton[["unit_id", "year", definition]]
    df = ntl.merge(treat, on=["unit_id", "year"], how="left")
    df[definition] = df[definition].fillna(0).astype(int)
    df["ever_treated"] = df.groupby("unit_id")[definition].transform("max")

    df = _keep_one_treated_side(df)

    first = (
        df[(df["ever_treated"] == 1) & (df[definition] == 1)]
        .groupby("G1ID")["year"].min().rename("first_sap_year")
    )
    df = df.merge(first, on="G1ID", how="left")

    if config.DROP_ALWAYS_TREATED:
        year_min = int(df["year"].min())
        always = sorted(first[first <= year_min].index)
        if always:
            print(f"[{label}] treated from the first year on, dropped: {always}")
            df = df[~df["G1ID"].isin(always)].copy()

    dirty = [
        gid for gid, g in df.groupby("G1ID")
        if not _control_stays_clean(g, definition, config.CLEAN_WINDOW)
    ]
    if dirty:
        print(f"[{label}] control side not clean within "
              f"+/-{config.CLEAN_WINDOW} years, dropped: {dirty}")
        df = df[~df["G1ID"].isin(dirty)].copy()

    df["treated"] = df["ever_treated"].astype(int)
    df["post"] = (df["year"] >= df["first_sap_year"]).astype(int)
    df["treated_x_post"] = df["treated"] * df["post"]
    df["rel_year"] = df["year"] - df["first_sap_year"]
    df["group_year"] = df["G1ID"].astype(str) + "_" + df["year"].astype(str)

    _report(df, label)
    return df


def _report(df, label):
    duplicates = df.groupby(["unit_id", "year"]).size()
    if (duplicates > 1).any():
        raise ValueError(f"[{label}] duplicate unit-year rows - did the buffers "
                         f"get dissolved?")

    units, years = df["unit_id"].nunique(), df["year"].nunique()
    print(f"[{label}] {df['G1ID'].nunique()} groups, {units} units, "
          f"{len(df)} rows (balanced would be {units * years})")


def pair_difference(panel):
    """One row per group and year, holding the light gap between the two sides.

    With exactly two units per group this is the same model as the unit-level
    panel with ethnicity-by-year fixed effects, just written so you can see
    what is being compared.
    """
    cols = ["G1ID", "name", "year", "ln_ntl", "ntl_mean", "FIPS_CNTRY"]
    treated = (
        panel.loc[panel["treated"] == 1, cols + ["first_sap_year"]]
        .rename(columns={"ln_ntl": "ln_treated", "ntl_mean": "ntl_treated",
                         "FIPS_CNTRY": "country_treated"})
    )
    control = (
        panel.loc[panel["treated"] == 0, cols]
        .drop(columns="name")
        .rename(columns={"ln_ntl": "ln_control", "ntl_mean": "ntl_control",
                         "FIPS_CNTRY": "country_control"})
    )

    d = treated.merge(control, on=["G1ID", "year"], how="inner")
    d["gap"] = d["ln_treated"] - d["ln_control"]
    d["post"] = (d["year"] >= d["first_sap_year"]).astype(int)
    d["rel_year"] = d["year"] - d["first_sap_year"]
    d["cohort"] = d["first_sap_year"].astype(int)
    d["country_pair"] = d["country_treated"] + "/" + d["country_control"]

    print(f"pair panel: {len(d)} group-years, {d['G1ID'].nunique()} groups")
    return d.sort_values(["G1ID", "year"]).reset_index(drop=True)


def composition(pair):
    """The sample table for the appendix: who is treated, by whom controlled,
    and when."""
    return (
        pair.groupby(["G1ID", "name", "country_pair", "cohort"])
        .agg(mean_treated=("ntl_treated", "mean"),
             mean_control=("ntl_control", "mean"),
             years=("year", "size"))
        .reset_index()
        .sort_values(["cohort", "G1ID"])
    )


def run(ntl, skeleton):
    panels = {
        name: build(ntl, skeleton, definition=name)
        for name in config.SAP_DEFINITIONS
    }
    pair = pair_difference(panels["narrow"])

    panels["narrow"].to_csv(config.WORK / "panel_narrow.csv", index=False)
    pair.to_csv(config.WORK / "pair_panel.csv", index=False)
    composition(pair).to_csv(config.OUT / "sample_composition.csv", index=False)
    return panels, pair
