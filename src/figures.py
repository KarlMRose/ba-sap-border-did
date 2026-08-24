from pathlib import Path
from matplotlib import font_manager

FONT_DIR = Path(__file__).resolve().parent / "assets" / "fonts"


def _register_fonts():
    if not FONT_DIR.exists():
        return False
    for ttf in FONT_DIR.glob("*.ttf"):
        font_manager.fontManager.addfont(str(ttf))
    return "Open Sans" in {f.name for f in font_manager.fontManager.ttflist}

import matplotlib.pyplot as plt

import config

TEXT_WIDTH = 6.1
WIDE = (TEXT_WIDTH, TEXT_WIDTH * 0.55)      
TALL = (TEXT_WIDTH, TEXT_WIDTH * 0.70)      
SQUARE = (TEXT_WIDTH, TEXT_WIDTH * 0.85)    

def _register_fonts():
    if FONT_DIR.exists():
        for ttf in FONT_DIR.glob("*.ttf"):
            font_manager.fontManager.addfont(str(ttf))
    return "Open Sans" in {f.name for f in font_manager.fontManager.ttflist}

def setup():
    if not _register_fonts():
        print("Open Sans not found, falling back to DejaVu Sans")

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Open Sans", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "axes.unicode_minus": True,
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
        "figure.constrained_layout.use": True,
        "savefig.bbox": None,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
    })


def save(fig, name):
    for suffix in ("pdf", "png"):
        fig.savefig(config.OUT / f"{name}.{suffix}")
    print(f"saved {name}.pdf and {name}.png")


def event_study(coefs, name="event_study"):
    fig, ax = plt.subplots(figsize=WIDE)
    ax.axhline(0, color="black", lw=0.7, ls="--", zorder=1)
    ax.axvline(-0.5, color=config.ACCENT, lw=1.0, ls=":", zorder=1,
               label="Programme entry")

    err = [coefs["coef"] - coefs["lo"], coefs["hi"] - coefs["coef"]]
    ax.errorbar(coefs["rel_year"], coefs["coef"], yerr=err, fmt="o",
                color=config.NAVY, ecolor=config.NAVY, elinewidth=0.9,
                capsize=2, zorder=3, label="Coefficient (95% CI)")

    ax.set_xlabel("Years relative to programme entry")
    ax.set_ylabel("Within-pair gap $\\Delta_{g,t}$")
    ax.set_xticks(coefs["rel_year"])
    ax.legend(loc="lower right")
    save(fig, name)
    return fig, ax


def event_time_bins(bins, reference="−2 to −1", name="event_time_bins"):
    bins = bins.reset_index(drop=True)
    x = list(range(len(bins)))

    fig, ax = plt.subplots(figsize=TALL)
    ax.axhline(0, color="black", lw=0.7, ls="--", zorder=1)

    ref = bins.index[bins["bin"] == reference]
    if len(ref):
        ax.axvline(int(ref[0]) + 0.5, color=config.ACCENT, lw=1.0, ls=":",
                   zorder=1, label="Programme entry")

    err = [bins["coef"] - bins["lo"], bins["hi"] - bins["coef"]]
    ax.errorbar(x, bins["coef"], yerr=err, fmt="o", color=config.NAVY,
                ecolor=config.NAVY, elinewidth=0.9, capsize=2, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(bins["bin"], rotation=30, ha="right")
    ax.set_xlim(-0.6, len(bins) - 0.4)
    ax.set_xlabel("Years relative to programme entry")
    ax.set_ylabel("Within-pair gap $\\Delta_{g,t}$")
    ax.legend(loc="upper right")
    save(fig, name)
    return fig, ax


def permutation(draws, observed, name="randomization_inference"):
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
    fig, ax = plt.subplots(figsize=SQUARE)
    africa.plot(ax=ax, color="#F4F2EE", edgecolor=config.GREY, lw=0.25)

    if ethnic_areas is not None:
        ethnic_areas.plot(ax=ax, color="#D8D3CC", edgecolor="none", alpha=0.9)

    buffers[buffers["treated"] == 0].plot(
        ax=ax, color=config.LIGHT, edgecolor=config.NAVY, lw=0.25)
    buffers[buffers["treated"] == 1].plot(
        ax=ax, color=config.NAVY, edgecolor="black", lw=0.25)

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
          handleheight=0.9, borderpad=0.5,
          frameon=True, facecolor="white", edgecolor="none", framealpha=0.85)
    save(fig, name)
    return fig, ax
