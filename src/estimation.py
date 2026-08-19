"""Estimation and inference.

The main specification is the within-pair difference: regress the light gap
between the two halves of an ethnic group on a post-treatment dummy with group
fixed effects. Everything else here either rewrites that same comparison a
different way, or checks whether it survives.

With 11 groups the usual cluster-robust standard errors can't be trusted, so
the numbers reported in the thesis come from the bootstrap and the permutation
test at the bottom.
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import config
import panelbuild as panel_mod

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
    """Four ways of writing the same comparison, then trend controls.

    The ethnicity-by-year FE model and the pair difference are algebraically
    identical, so their coefficients have to match.
    """
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
    """Separate effect per treatment cohort.

    A cohort made up of a single ethnicity is identified by one cluster only.
    The cluster-robust variance then collapses to zero and the p-value is
    meaningless, so those rows keep the point estimate but lose the standard
    error.
    """
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
    """Coefficients by year relative to treatment.

    Keep the window narrow enough that every event year has several groups in
    it. At +/-8 the far pre-periods rest on a single group, which makes the
    clustered covariance matrix rank deficient and the pre-trend F test
    meaningless.
    """
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
    """Joint test that all pre-treatment coefficients are zero."""
    pre = [n for y, n in terms if y < 0]
    test = model.f_test(" = 0, ".join(pre) + " = 0")
    return {
        "F": float(np.squeeze(test.fvalue)),
        "p": float(np.squeeze(test.pvalue)),
        "restrictions": len(pre),
    }


def bootstrap(pair, formula=MAIN, term="post", n=None, seed=None):
    """Resample whole ethnic groups with replacement.

    Groups drawn twice need distinct ids, otherwise the fixed effects merge
    them back into one.
    """
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
    """Shuffle the treatment years across groups and refit."""
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
    """Summarise a set of draws.

    centre="mean" is for the bootstrap: the draws approximate the sampling
    distribution of the estimate, so the p-value asks how far the estimate
    sits from zero given that spread.

    centre="zero" is for randomization inference: the draws are the null
    distribution itself, so the p-value is just how often a permutation
    produces something at least as large in absolute value.
    """
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


# --- robustness -----------------------------------------------------------

def offset_sensitivity(panel, offsets=(0.001, 0.01, 0.1, 1.0)):
    """The offset in ln(mean + c) is arbitrary and the estimate moves with it,
    so all four are reported rather than just the default."""
    rows = []
    for c in offsets:
        data = panel.copy()
        data["ln_ntl"] = np.log(data["ntl_mean"].clip(lower=0) + c)
        pair = panel_mod.pair_difference(data)
        rows.append(_row(cluster_fit(pair, MAIN), "post", f"offset c = {c}"))
    return pd.DataFrame(rows)


def window_sensitivity(pair, windows=(5, 8, 12, None)):
    """Restricting to event time shows whether the estimate is a jump at
    treatment or a trend picked up over long horizons."""
    rows = []
    for w in windows:
        data = pair if w is None else pair[pair["rel_year"].between(-w, w)]
        if data["post"].nunique() < 2:
            continue
        label = "full panel" if w is None else f"rel_year within +/-{w}"
        rows.append(_row(cluster_fit(data, MAIN), "post", label))
    return pd.DataFrame(rows)


def brightness_threshold(pair, thresholds=(0.0, 0.01, 0.05, 0.1)):
    """Drop the darkest pairs. Where mean radiance is around 0.007 the offset
    is larger than the signal, so those groups measure very little."""
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
    """Effect by broad bands of event time.

    The event study only reaches +/-5, but the panel runs to +31 for the
    earliest cohort. Binning shows the long horizon, which is where the pooled
    estimate actually comes from.
    """
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
    """Cohort ATTs in the spirit of Callaway & Sant'Anna (2021).

    Reported for comparison only: the estimator assumes treatment is
    absorbing, which the programme episodes show is not the case here.
    """
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

        # a not-yet-treated control only counts until its own entry
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
    """Contiguous programme episodes per treated side."""
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
    """Flag the years in which a programme is actually running, plus the
    year-on-year changes used for the entry and exit regressions."""
    active = (panel.loc[panel["treated"] == 1, ["G1ID", "year", definition]]
              .rename(columns={definition: "active"}))
    out = pair.merge(active, on=["G1ID", "year"], how="left").sort_values(["G1ID", "year"])
    out["d_gap"] = out.groupby("G1ID")["gap"].diff()
    out["d_active"] = out.groupby("G1ID")["active"].diff()
    out["entry"] = (out["d_active"] == 1).astype(int)
    out["exit"] = (out["d_active"] == -1).astype(int)
    return out.reset_index(drop=True)


def on_off_specifications(pair_active):
    """The programme as something that switches on and off, not as an
    absorbing state. C&S and the usual event-study estimators assume the
    latter, which the spells table shows does not hold here."""
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