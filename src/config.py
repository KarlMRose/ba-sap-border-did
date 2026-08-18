"""Paths and settings. Everything imports from here so there are no
hardcoded paths anywhere else.
"""

from pathlib import Path

# repo root, assuming this file sits in src/
ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data" / "raw"
WORK = ROOT / "data" / "work"
OUT = ROOT / "output"

GREG_SHP = DATA / "GREG" / "GREG.shp"
VIIRS_DIR = DATA / "VIIRS"
COUNTRIES_10M = DATA / "naturalearth" / "ne_10m_admin_0_countries.shp"
COUNTRIES_110M = DATA / "naturalearth" / "ne_110m_admin_0_countries.shp"
VREELAND_DTA = DATA / "vreeland" / "master_merge.dta"

for d in (WORK, OUT):
    d.mkdir(parents=True, exist_ok=True)


# --- sample ---------------------------------------------------------------

YEAR_MIN = 1992
YEAR_MAX = 2024

# FIPS codes for Sub-Saharan Africa.
SSA_FIPS = {
    "AO", "BC", "BN", "BY", "CD", "CF", "CG", "CM", "CN", "CT", "CV", "DJ",
    "EK", "ER", "ET", "GA", "GH", "GM", "GV", "IV", "KE", "LI", "LT", "ML",
    "MP", "MR", "MZ", "NG", "NI", "OD", "PU", "RW", "SE", "SF", "SG", "SH",
    "SL", "SO", "SU", "TG", "TO", "TP", "TZ", "UG", "UV", "WA", "WZ", "ZA",
    "ZI", "ZM", "ZR",
}

# Drop ethnicities with one side in South Africa - income levels there are
# way above the rest of the sample and the homelands had their own history.
EXCLUDE_SOUTH_AFRICA = True

# Groups already treated in 1992 have no pre-period, so they can't be used.
DROP_ALWAYS_TREATED = True

# Control side must not have its own SAP within +/- this many years of the
# treated side's entry.
CLEAN_WINDOW = 5


# --- treatment ------------------------------------------------------------

SAP_NARROW = ["SAF", "ESAF", "SAF/ESAF", "ESAF/SAF", "PRGF", "ECF"]
SAP_EFF = SAP_NARROW + ["EFF"]   # same classification as Callais et al. (2026)
SAP_BROAD = SAP_EFF + ["SBA"]    # all IMF arrangements

SAP_DEFINITIONS = {"narrow": SAP_NARROW, "eff": SAP_EFF, "broad": SAP_BROAD}


# --- geography ------------------------------------------------------------

BUFFER_KM = 50

# Equidistant conic for Africa. In Web Mercator a 50 km buffer is really
# 50 * cos(lat) km, which is off by 8% at the south end of the sample.
BUFFER_CRS = "ESRI:102023"
FALLBACK_CRS = "EPSG:3857"

# GREG uses FIPS, Natural Earth has both FIPS and ISO columns.
# Careful: 'ZA' is South Africa in ISO but Zambia in FIPS, so this mapping
# must only ever be applied to the ISO columns.
FIPS_TO_ISO = {"SU": "SD", "IV": "CI", "SF": "ZA"}
FIPS_COLUMNS = ["FIPS_10", "FIPS_10_"]
ISO_COLUMNS = ["ISO_A2", "ISO_A2_EH", "ADM0_A3", "SOV_A3"]


# --- estimation -----------------------------------------------------------

NTL_OFFSET = 0.01     # ln(mean + c), sensitivity to c is reported separately
EVENT_WINDOW = 5
N_BOOTSTRAP = 999
N_PERMUTE = 2000
SEED = 42


# --- plots ----------------------------------------------------------------

NAVY = "#1B3A4B"
LIGHT = "#AECBD6"
ACCENT = "#C0392B"
GREY = "#7F8C8D"
