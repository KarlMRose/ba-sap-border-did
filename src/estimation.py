import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import config
import panelbuild as panel_mod
import statsmodels.api as sm

MAIN = "gap ~ post + C(G1ID)"


def cluster_fit(data, formula, cluster="G1ID"):
    return smf.ols(formula, data=data).fit(
        cov_type="cluster", cov_kwds={"groups": data[cluster]}
    )


def _row(model, term, label, extra=None):
    out = {
        "spec": label,
        "coef": float(model.params[term]),
        "se": float(model.bse[term]),
        "p": float(model.pvalues[term]),
        "n": int(model.nobs),
    }
    if extra:
        out.update(extra)
    return out


def main_specifications(panel, pair):
    pair = pair.copy()
    pair["trend"] = pair["year"] - pair["year"].min()

    rows = [
        _row(cluster_fit(panel, "ln_ntl ~ treated_x_post + C(unit_id) + C(year)"),
             "treated_x_post", "unit and year FE"),
        _row(cluster_fit(panel, "ln_ntl ~ treated_x_post + C(G1ID):treated + C(group_year)"),
             "treated_x_post", "ethnicity-by-year FE"),
        _row(cluster_fit(pair, MAIN), "post", "within-pair gap"),
        _row(cluster_fit(pair, "gap ~ post + C(G1ID) + C(year)"),
             "post", "within-pair gap, year FE"),
        _row(cluster_fit(pair, "gap ~ post + trend + C(G1ID)"),
             "post", "plus common linear trend"),
        _row(cluster_fit(pair, "gap ~ post + C(G1ID) + C(G1ID):trend"),
             "post", "plus group-specific trends"),
    ]
    return pd.DataFrame(rows)


def cohort_effects(pair, min_groups=2):
    model = cluster_fit(pair, "gap ~ post:C(cohort) + C(G1ID)")
    rows = []
    for term in [t for t in model.params.index if t.startswith("post:")]:
        cohort = int(term.split("[")[-1].rstrip("]").lstrip("T."))
        groups = int(pair.loc[pair["cohort"] == cohort, "G1ID"].nunique())
        identified = groups >= min_groups
        rows.append({
            "cohort": cohort,
            "coef": float(model.params[term]),
            "se": float(model.bse[term]) if identified else float("nan"),
            "p": float(model.pvalues[term]) if identified else float("nan"),
            "groups": groups,
        })
    return pd.DataFrame(rows).sort_values("cohort")


def event_study(pair, window=None, reference=-1):
    window = window or config.EVENT_WINDOW
    data = pair[pair["rel_year"].between(-window, window)].copy()

    terms = []
    for y in range(-window, window + 1):
        if y == reference:
            continue
        name = f"ym{abs(y)}" if y < 0 else f"yp{y}"
        data[name] = (data["rel_year"] == y).astype(int)
        terms.append((y, name))

    formula = "gap ~ " + " + ".join(n for _, n in terms) + " + C(G1ID)"
    model = cluster_fit(data, formula)

    rows = []
    for y in range(-window, window + 1):
        if y == reference:
            rows.append({"rel_year": y, "coef": 0.0, "lo": 0.0, "hi": 0.0})
            continue
        name = [n for yy, n in terms if yy == y][0]
        ci = model.conf_int().loc[name]
        rows.append({"rel_year": y, "coef": float(model.params[name]),
                     "lo": float(ci[0]), "hi": float(ci[1])})

    coefs = pd.DataFrame(rows)
    coefs["groups"] = coefs["rel_year"].map(
        data.groupby("rel_year")["G1ID"].nunique()
    ).fillna(0).astype(int)
    return coefs, model, terms


def pretrend_test(model, terms):
    pre = [n for y, n in terms if y < 0]
    test = model.f_test(" = 0, ".join(pre) + " = 0")
    return {
        "F": float(np.squeeze(test.fvalue)),
        "p": float(np.squeeze(test.pvalue)),
        "restrictions": len(pre),
    }


def bootstrap(pair, formula=MAIN, term="post", n=None, seed=None):
    n = n or config.N_BOOTSTRAP
    rng = np.random.default_rng(seed or config.SEED)
    groups = pair["G1ID"].unique()

    draws = []
    for _ in range(n):
        parts = []
        for k, g in enumerate(rng.choice(groups, size=len(groups), replace=True)):
            part = pair[pair["G1ID"] == g].copy()
            part["G1ID"] = f"{g}_{k}"
            parts.append(part)
        sample = pd.concat(parts, ignore_index=True)
        try:
            draws.append(float(smf.ols(formula, data=sample).fit().params[term]))
        except Exception:
            continue
    return np.array(draws)


def randomization_inference(pair, n=None, seed=None):
    n = n or config.N_PERMUTE
    rng = np.random.default_rng(seed or config.SEED)
    groups = sorted(pair["G1ID"].unique())
    years = pair.groupby("G1ID")["first_sap_year"].first().loc[groups].values

    draws = []
    for _ in range(n):
        shuffled = dict(zip(groups, rng.permutation(years)))
        data = pair.copy()
        data["post"] = (data["year"] >= data["G1ID"].map(shuffled)).astype(int)
        if data["post"].nunique() < 2:
            continue
        try:
            draws.append(float(smf.ols(MAIN, data=data).fit().params["post"]))
        except Exception:
            continue
    return np.array(draws)


def describe_draws(draws, observed, label, centre="mean"):
    lo, hi = np.percentile(draws, [2.5, 97.5])
    reference = draws.mean() if centre == "mean" else 0.0
    return {
        "spec": label,
        "coef": observed,
        "se": float(draws.std(ddof=1)),
        "lo": float(lo),
        "hi": float(hi),
        "p": float(np.mean(np.abs(draws - reference) >= abs(observed))),
        "null_distribution": centre == "zero",
        "draws": len(draws),
    }



def offset_sensitivity(panel, offsets=(0.001, 0.01, 0.1, 1.0)):
    rows = []
    for c in offsets:
        data = panel.copy()
        data["ln_ntl"] = np.log(data["ntl_mean"].clip(lower=0) + c)
        pair = panel_mod.pair_difference(data)
        rows.append(_row(cluster_fit(pair, MAIN), "post", f"offset c = {c}"))
    return pd.DataFrame(rows)


def window_sensitivity(pair, windows=(5, 8, 12, None)):
    rows = []
    for w in windows:
        data = pair if w is None else pair[pair["rel_year"].between(-w, w)]
        if data["post"].nunique() < 2:
            continue
        label = "full panel" if w is None else f"rel_year within +/-{w}"
        rows.append(_row(cluster_fit(data, MAIN), "post", label))
    return pd.DataFrame(rows)


def brightness_threshold(pair, thresholds=(0.0, 0.01, 0.05, 0.1)):
    dim = (
        pair.groupby("G1ID")[["ntl_treated", "ntl_control"]].mean()
        .min(axis=1).rename("dimmest_side")
    )
    data = pair.merge(dim, on="G1ID")

    rows = []
    for t in thresholds:
        sub = data[data["dimmest_side"] > t]
        if sub["G1ID"].nunique() < 3 or sub["post"].nunique() < 2:
            continue
        rows.append(_row(cluster_fit(sub, MAIN), "post",
                         f"dimmest side above {t}",
                         {"groups": int(sub["G1ID"].nunique())}))
    return pd.DataFrame(rows)


BIN_EDGES = [-99, -6, -3, -1, 2, 5, 10, 15, 99]
BIN_LABELS = ["≤ −6", "−5 to −3", "−2 to −1", "0 to 2", "3 to 5",
              "6 to 10", "11 to 15", "> 15"]
BIN_REFERENCE = "−2 to −1"


def event_time_bins(pair, edges=None, labels=None, reference=None):
    edges = edges or BIN_EDGES
    labels = labels or BIN_LABELS
    reference = reference or BIN_REFERENCE

    data = pair.copy()
    data["bin"] = pd.cut(data["rel_year"], bins=edges, labels=labels)
    data = data[data["bin"].notna()]

    model = cluster_fit(data, f'gap ~ C(bin, Treatment("{reference}")) + C(G1ID)')

    counts = (
        data.groupby("bin", observed=True)
        .agg(groups=("G1ID", "nunique"), n=("gap", "size"))
        .reset_index()
    )
    counts["bin"] = counts["bin"].astype(str)

    rows = []
    for label in labels:
        if label == reference:
            rows.append({"bin": label, "coef": 0.0, "lo": 0.0, "hi": 0.0, "p": float("nan")})
            continue
        term = f'C(bin, Treatment("{reference}"))[T.{label}]'
        if term not in model.params.index:
            continue
        ci = model.conf_int().loc[term]
        rows.append({"bin": label, "coef": float(model.params[term]),
                     "lo": float(ci[0]), "hi": float(ci[1]),
                     "p": float(model.pvalues[term])})

    out = pd.DataFrame(rows).merge(counts, on="bin", how="left")
    return out.reset_index(drop=True), model

def callaway_santanna(panel):
    cohorts = sorted(panel.loc[panel["treated"] == 1, "first_sap_year"].dropna().unique())
    cohorts = [int(g) for g in cohorts if g > panel["year"].min()]

    rows = []
    for g in cohorts:
        treated_units = panel.loc[(panel["treated"] == 1) &
                                  (panel["first_sap_year"] == g), "unit_id"].unique()
        control_units = panel.loc[(panel["treated"] == 0) |
                                  ((panel["treated"] == 1) &
                                   (panel["first_sap_year"] > g)), "unit_id"].unique()

        sub = panel[panel["unit_id"].isin([*treated_units, *control_units])].copy()
        sub["cohort_treat"] = sub["unit_id"].isin(treated_units).astype(int)

        sub = sub[~((sub["cohort_treat"] == 0) & (sub["treated"] == 1) &
                    (sub["year"] >= sub["first_sap_year"]))]
        sub["interaction"] = sub["cohort_treat"] * (sub["year"] >= g).astype(int)

        if sub["interaction"].sum() == 0 or (sub["cohort_treat"] == 0).sum() == 0:
            continue

        m = cluster_fit(sub, "ln_ntl ~ interaction + C(unit_id) + C(year)")
        rows.append({"cohort": g, "att": float(m.params["interaction"]),
                     "se": float(m.bse["interaction"]),
                     "n_treated": len(treated_units)})

    out = pd.DataFrame(rows)
    weights = out["n_treated"] / out["n_treated"].sum()
    return out, float((out["att"] * weights).sum())

def programme_spells(panel, definition="narrow"):
    rows = []
    for gid, g in panel[panel["treated"] == 1].sort_values("year").groupby("G1ID"):
        years = g.loc[g[definition] == 1, "year"].tolist()
        if not years:
            continue
        spells, start = [], years[0]
        for a, b in zip(years, years[1:] + [None]):
            if b is None or b != a + 1:
                spells.append(f"{start}-{a}" if a != start else str(a))
                start = b
        rows.append({"G1ID": gid, "country": g["FIPS_CNTRY"].iloc[0],
                     "name": g["name"].iloc[0], "n_spells": len(spells),
                     "spells": ", ".join(spells)})
    return pd.DataFrame(rows)


def add_active(pair, panel, definition="narrow"):
    active = (panel.loc[panel["treated"] == 1, ["G1ID", "year", definition]]
              .rename(columns={definition: "active"}))
    out = pair.merge(active, on=["G1ID", "year"], how="left").sort_values(["G1ID", "year"])
    out["d_gap"] = out.groupby("G1ID")["gap"].diff()
    out["d_active"] = out.groupby("G1ID")["active"].diff()
    out["entry"] = (out["d_active"] == 1).astype(int)
    out["exit"] = (out["d_active"] == -1).astype(int)
    return out.reset_index(drop=True)


def on_off_specifications(pair_active):
    fd = pair_active.dropna(subset=["d_gap", "d_active"])
    both = "gap ~ post + active + C(G1ID)"
    return pd.DataFrame([
        _row(cluster_fit(pair_active, "gap ~ active + C(G1ID)"), "active",
             "active programme"),
        _row(cluster_fit(pair_active, both), "post",
             "post entry, no active programme"),
        _row(cluster_fit(pair_active, both), "active",
             "additional effect while active"),
        _row(cluster_fit(fd, "d_gap ~ entry + exit + C(G1ID)"), "entry",
             f"entry ({int(fd['entry'].sum())} events)"),
        _row(cluster_fit(fd, "d_gap ~ entry + exit + C(G1ID)"), "exit",
             f"exit ({int(fd['exit'].sum())} events)"),
    ])

def robustness_table(panels, pair, buffer_areas=None):
    rows = []

    def add(data, label, formula=MAIN, term="post"):
        if data["post"].nunique() < 2 or data["G1ID"].nunique() < 3:
            print(f"  skipped: {label}")
            return
        rows.append(_row(cluster_fit(data, formula), term, label,
                         {"groups": int(data["G1ID"].nunique())}))

    add(pair, "baseline (narrow definition)")

    labels = {"eff": "narrow plus EFF", "broad": "all IMF arrangements"}
    for name, unit_panel in panels.items():
        if name == "narrow":
            continue
        add(panel_mod.pair_difference(unit_panel), f"SAP: {labels.get(name, name)}")

    dim = (pair.groupby("G1ID")[["ntl_treated", "ntl_control"]].mean()
           .min(axis=1).rename("dimmest_side"))
    bright = pair.merge(dim, on="G1ID")
    add(bright[bright["dimmest_side"] > 0.01], "dimmest side above 0.01")

    add(pair[pair["year"] <= 2013], "panel restricted to 1992-2013")

    if buffer_areas is not None:
        ar = buffer_areas.groupby("G1ID")["area_km2"].agg(["min", "max"]).reset_index()
        ar["ratio"] = ar["max"] / ar["min"]
        with_ratio = pair.merge(ar[["G1ID", "ratio"]], on="G1ID", how="left")
        for limit in (5, 3):
            add(with_ratio[with_ratio["ratio"] <= limit], f"area ratio at most {limit}:1")

    ppml = panels["narrow"].copy()
    ppml["y"] = ppml["ntl_mean"].clip(lower=0) * 1000
    try:
        m = smf.glm("y ~ treated_x_post + C(unit_id) + C(year)", data=ppml,
                    family=sm.families.Poisson()).fit(
            cov_type="cluster", cov_kwds={"groups": ppml["G1ID"]}, maxiter=300)
        rows.append({"spec": "PPML in levels (unit and year FE)",
                     "coef": float(m.params["treated_x_post"]),
                     "se": float(m.bse["treated_x_post"]),
                     "p": float(m.pvalues["treated_x_post"]),
                     "n": int(m.nobs),
                     "groups": int(ppml["G1ID"].nunique())})
    except Exception as exc:
        print(f"  PPML failed: {str(exc)[:70]}")

    return pd.DataFrame(rows)

def run(panel, pair):
    specs = main_specifications(panel, pair)
    observed = float(cluster_fit(pair, MAIN).params["post"])

    boot = bootstrap(pair)
    perm = randomization_inference(pair)
    inference = pd.DataFrame([
        describe_draws(boot, observed, "cluster bootstrap"),
        describe_draws(perm, observed, "randomization inference", centre="zero"),
    ])

    coefs, model, terms = event_study(pair)
    pretrend = pretrend_test(model, terms)
    print(f"pre-trend test: F = {pretrend['F']:.3f}, p = {pretrend['p']:.4f}")

    specs.to_csv(config.OUT / "main_specifications.csv", index=False)
    inference.to_csv(config.OUT / "inference.csv", index=False)
    coefs.to_csv(config.OUT / "event_study.csv", index=False)
    cohort_effects(pair).to_csv(config.OUT / "cohort_effects.csv", index=False)
    event_time_bins(pair)[0].to_csv(config.OUT / "event_time_bins.csv", index=False)

    return specs, inference, coefs

def did_m(data, outcome="gap", treat="active", unit="G1ID", time="year",
          placebo=False):
    d = data[[unit, time, outcome, treat]].dropna().sort_values([unit, time]).copy()
    d["lag_treat"] = d.groupby(unit)[treat].shift(1)
    d["delta"] = d.groupby(unit)[outcome].diff()
    if placebo:
        d["delta"] = d.groupby(unit)["delta"].shift(1)
    d = d.dropna(subset=["lag_treat", "delta"])

    rows = []
    for t, g in d.groupby(time):
        joiners = g.loc[(g["lag_treat"] == 0) & (g[treat] == 1), "delta"]
        stay_off = g.loc[(g["lag_treat"] == 0) & (g[treat] == 0), "delta"]
        leavers = g.loc[(g["lag_treat"] == 1) & (g[treat] == 0), "delta"]
        stay_on = g.loc[(g["lag_treat"] == 1) & (g[treat] == 1), "delta"]

        if len(joiners) and len(stay_off):
            rows.append({time: t, "type": "switch on", "switchers": len(joiners),
                         "stayers": len(stay_off),
                         "effect": float(joiners.mean() - stay_off.mean())})
        if len(leavers) and len(stay_on):
            rows.append({time: t, "type": "switch off", "switchers": len(leavers),
                         "stayers": len(stay_on),
                         "effect": float(stay_on.mean() - leavers.mean())})

    per_period = pd.DataFrame(rows)
    if per_period.empty:
        return per_period, float("nan")
    agg = float((per_period["effect"] * per_period["switchers"]).sum()
                / per_period["switchers"].sum())
    return per_period, agg


def did_m_bootstrap(data, n=None, seed=None, unit="G1ID", **kwargs):
    n = n or config.N_BOOTSTRAP
    rng = np.random.default_rng(seed or config.SEED)
    groups = data[unit].unique()

    draws = []
    for _ in range(n):
        parts = []
        for k, g in enumerate(rng.choice(groups, size=len(groups), replace=True)):
            part = data[data[unit] == g].copy()
            part[unit] = f"{g}_{k}"
            parts.append(part)
        _, agg = did_m(pd.concat(parts, ignore_index=True), unit=unit, **kwargs)
        if np.isfinite(agg):
            draws.append(agg)
    return np.array(draws)


def twfe_weights(data, treat="active", unit="G1ID", time="year"):
    d = data[[unit, time, treat]].dropna().copy()
    d["resid"] = smf.ols(f"{treat} ~ C({unit}) + C({time})", data=d).fit().resid
    w = d[d[treat] == 1].copy()
    w["weight"] = w["resid"] / w["resid"].sum()
    return w[[unit, time, "weight"]].reset_index(drop=True)