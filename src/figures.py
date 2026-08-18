"""Figure style and the plots that go into the thesis.

Everything is sized for the text width of the document and saved as PDF, so
LaTeX includes the figures at their natural size and never rescales them.

Call setup() once at the top of a notebook.
"""

import matplotlib.pyplot as plt

import config

# Text width of the document in inches. Get the real number by putting
# \the\textwidth in the LaTeX source once and dividing by 72.27.
# article, a4paper, 12pt with default margins is about 4.8 in.
TEXT_WIDTH = 4.8
WIDE = (TEXT_WIDTH, TEXT_WIDTH * 0.55)      # time on the x axis
TALL = (TEXT_WIDTH, TEXT_WIDTH * 0.70)      # room for rotated category labels
SQUARE = (TEXT_WIDTH, TEXT_WIDTH)           # map


def setup():
    """Serif fonts at the same size as the body text, no chartjunk."""
    plt.rcParams.update({
        "font.family": "serif",
        # CMU Serif is Computer Modern, the LaTeX default. Falls back to
        # DejaVu Serif if it isn't installed.
        "font.serif": ["CMU Serif", "Latin Modern Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.3,
        "lines.markersize": 4,
        "legend.frameon": False,
        # constrained_layout and bbox_inches="tight" fight each other and end
        # up clipping axis labels, so only one of them is on.
        "figure.constrained_layout.use": True,
        "savefig.bbox": None,
        "savefig.dpi": 300,
    })


def save(fig, name):
    """PDF for the thesis, PNG for slides and for looking at on screen."""
    for suffix in ("pdf", "png"):
        fig.savefig(config.OUT / f"{name}.{suffix}")
    print(f"saved {name}.pdf and {name}.png")


def event_study(coefs, name="event_study"):
    """Coefficient path around treatment, with error bars per event year."""
    fig, ax = plt.subplots(figsize=WIDE)
    ax.axhline(0, color="black", lw=0.7, ls="--", zorder=1)
    ax.axvline(-0.5, color=config.ACCENT, lw=1.0, ls=":", zorder=1,
               label="SAP entry")

    err = [coefs["coef"] - coefs["lo"], coefs["hi"] - coefs["coef"]]
    ax.errorbar(coefs["rel_year"], coefs["coef"], yerr=err, fmt="o",
                color=config.NAVY, ecolor=config.NAVY, elinewidth=0.9,
                capsize=2, zorder=3, label="Coefficient (95% CI)")

    ax.set_xlabel("Years relative to SAP entry")
    ax.set_ylabel("Within-pair gap in ln(NTL)")
    ax.set_xticks(coefs["rel_year"])
    ax.legend(loc="lower right")
    save(fig, name)
    return fig, ax


def event_time_bins(bins, reference="−2 to −1", name="event_time_bins"):
    """Effect by band of event time, with the reference band pinned at zero."""
    bins = bins.reset_index(drop=True)
    x = list(range(len(bins)))

    fig, ax = plt.subplots(figsize=TALL)
    ax.axhline(0, color="black", lw=0.7, ls="--", zorder=1)

    ref = bins.index[bins["bin"] == reference]
    if len(ref):
        ax.axvline(int(ref[0]) + 0.5, color=config.ACCENT, lw=1.0, ls=":",
                   zorder=1, label="SAP entry")

    err = [bins["coef"] - bins["lo"], bins["hi"] - bins["coef"]]
    ax.errorbar(x, bins["coef"], yerr=err, fmt="o", color=config.NAVY,
                ecolor=config.NAVY, elinewidth=0.9, capsize=2, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(bins["bin"], rotation=30, ha="right")
    ax.set_xlim(-0.6, len(bins) - 0.4)
    ax.set_xlabel("Years relative to SAP entry")
    ax.set_ylabel("Within-pair gap in ln(NTL)")
    ax.legend(loc="upper right")
    save(fig, name)
    return fig, ax


def permutation(draws, observed, name="randomization_inference"):
    """Where the observed estimate sits in the permutation distribution."""
    fig, ax = plt.subplots(figsize=WIDE)
    ax.hist(draws, bins=40, color=config.LIGHT, edgecolor="white", lw=0.4)
    ax.axvline(0, color="black", lw=0.7, ls="--", label="No effect")
    ax.axvline(observed, color=config.ACCENT, lw=1.4,
               label=f"Observed = {observed:.3f}")

    ax.set_xlabel("Coefficient under permuted treatment timing")
    ax.set_ylabel("Permutations")
    ax.legend(loc="upper left")
    save(fig, name)
    return fig, ax


def sample_map(buffers, africa, ethnic_areas=None, name="sample_map"):
    """The border strips in context.

    The strips are 50 km wide, so on a map of Africa they are barely visible
    on their own. Passing the full ethnic territories in ethnic_areas draws
    them underneath in grey, which shows where each pair sits.
    """
    fig, ax = plt.subplots(figsize=SQUARE)
    africa.plot(ax=ax, color="#F4F2EE", edgecolor=config.GREY, lw=0.25)

    if ethnic_areas is not None:
        ethnic_areas.plot(ax=ax, color="#D8D3CC", edgecolor="none", alpha=0.9)

    buffers[buffers["treated"] == 0].plot(
        ax=ax, color=config.LIGHT, edgecolor=config.NAVY, lw=0.25)
    buffers[buffers["treated"] == 1].plot(
        ax=ax, color=config.NAVY, edgecolor="black", lw=0.25)

    # zoom to the data with a margin, rather than showing all of Africa
    minx, miny, maxx, maxy = buffers.total_bounds
    pad = 6
    ax.set_xlim(minx - pad, maxx + pad)
    ax.set_ylim(miny - pad, maxy + pad)
    ax.set_axis_off()

    handles = [
        plt.Rectangle((0, 0), 1, 1, fc=config.NAVY, ec="black", lw=0.25),
        plt.Rectangle((0, 0), 1, 1, fc=config.LIGHT, ec=config.NAVY, lw=0.25),
    ]
    labels = ["Treated side", "Control side"]
    if ethnic_areas is not None:
        handles.append(plt.Rectangle((0, 0), 1, 1, fc="#D8D3CC", ec="none"))
        labels.append("Full ethnic area")

    ax.legend(handles, labels, loc="lower left", handlelength=1.2,
              handleheight=0.9, borderpad=0.4)
    save(fig, name)
    return fig, ax
