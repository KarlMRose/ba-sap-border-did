"""Figure style and the plots that go into the thesis.

Everything is sized for the text width of the document and saved as PDF, so
LaTeX includes the figures at their natural size and never rescales them.
Rescaling is what makes fonts in one figure look bigger than in the next.

Call setup() once at the top of a notebook.
"""

import matplotlib.pyplot as plt

import config

# Text width of the document in inches. Get the real number by putting
# \the\textwidth in the LaTeX source once and dividing by 72.27.
# article, a4paper, 12pt with default margins is about 4.8 in.
TEXT_WIDTH = 4.8
FULL = (TEXT_WIDTH, TEXT_WIDTH * 0.62)      # single figure
WIDE = (TEXT_WIDTH, TEXT_WIDTH * 0.45)      # event study, time on the x axis
SQUARE = (TEXT_WIDTH, TEXT_WIDTH)           # map


def setup():
    """Serif fonts at the same size as the body text, no chartjunk."""
    plt.rcParams.update({
        "font.family": "serif",
        # CMU Serif is Computer Modern, the LaTeX default. If it is not
        # installed matplotlib falls through to DejaVu Serif, which is close
        # enough that nobody notices in print.
        "font.serif": ["CMU Serif", "Latin Modern Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.4,
        "lines.markersize": 4,
        "legend.frameon": False,
        "figure.constrained_layout.use": True,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "savefig.dpi": 300,
    })


def save(fig, name):
    """PDF for the thesis, PNG for slides and for looking at on screen."""
    for suffix in ("pdf", "png"):
        fig.savefig(config.OUT / f"{name}.{suffix}")
    print(f"saved {name}.pdf and {name}.png")


def event_study(coefs, name="event_study"):
    """Coefficient path around treatment with a confidence band."""
    fig, ax = plt.subplots(figsize=WIDE)
    ax.axhline(0, color="black", lw=0.7, ls="--", zorder=1)
    ax.axvline(-0.5, color=config.ACCENT, lw=1.0, ls=":", zorder=1,
               label="SAP entry")
    ax.fill_between(coefs["rel_year"], coefs["lo"], coefs["hi"],
                    color=config.NAVY, alpha=0.18, lw=0, label="95% CI")
    ax.plot(coefs["rel_year"], coefs["coef"], "o-", color=config.NAVY, zorder=3)

    ax.set_xlabel("Years relative to SAP entry")
    ax.set_ylabel("Within-pair gap in ln(NTL)")
    ax.set_xticks(coefs["rel_year"])
    ax.legend(loc="best")
    save(fig, name)
    return fig, ax


def permutation(draws, observed, name="randomization_inference"):
    """Where the observed estimate sits in the permutation distribution."""
    fig, ax = plt.subplots(figsize=WIDE)
    ax.hist(draws, bins=40, color=config.LIGHT, edgecolor="white", lw=0.4)
    ax.axvline(0, color="black", lw=0.7, ls="--")
    ax.axvline(observed, color=config.ACCENT, lw=1.4,
               label=f"observed = {observed:.3f}")

    ax.set_xlabel("Coefficient under permuted treatment timing")
    ax.set_ylabel("Permutations")
    ax.legend(loc="upper left")
    save(fig, name)
    return fig, ax


def sample_map(buffers, africa, name="sample_map"):
    """The border strips in context. Treated sides dark, control sides light,
    which also keeps them apart in greyscale print."""
    fig, ax = plt.subplots(figsize=SQUARE)
    africa.plot(ax=ax, color="#F2F0EC", edgecolor=config.GREY, lw=0.3)

    treated = buffers[buffers["treated"] == 1]
    control = buffers[buffers["treated"] == 0]
    control.plot(ax=ax, color=config.LIGHT, edgecolor=config.NAVY, lw=0.3)
    treated.plot(ax=ax, color=config.NAVY, edgecolor="black", lw=0.3)

    ax.set_xlim(-20, 52)
    ax.set_ylim(-35, 25)
    ax.set_axis_off()

    handles = [
        plt.Rectangle((0, 0), 1, 1, fc=config.NAVY, ec="black", lw=0.3),
        plt.Rectangle((0, 0), 1, 1, fc=config.LIGHT, ec=config.NAVY, lw=0.3),
    ]
    ax.legend(handles, ["Treated side", "Control side"], loc="lower left")
    save(fig, name)
    return fig, ax
