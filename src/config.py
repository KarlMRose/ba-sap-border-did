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


YEAR_MIN = 1992
YEAR_MAX = 2024

SSA_FIPS = {
    "AO", "BC", "BN", "BY", "CD", "CF", "CG", "CM", "CN", "CT", "CV", "DJ",
    "EK", "ER", "ET", "GA", "GH", "GM", "GV", "IV", "KE", "LI", "LT", "ML",
    "MP", "MR", "MZ", "NG", "NI", "OD", "PU", "RW", "SE", "SF", "SG", "SH",
    "SL", "SO", "SU", "TG", "TO", "TP", "TZ", "UG", "UV", "WA", "WZ", "ZA",
    "ZI", "ZM", "ZR",
}


EXCLUDE_SOUTH_AFRICA = True

DROP_ALWAYS_TREATED = True

CLEAN_WINDOW = 5


SAP_NARROW = ["SAF", "ESAF", "SAF/ESAF", "ESAF/SAF", "PRGF", "ECF"]
SAP_EFF = SAP_NARROW + ["EFF"]  
SAP_BROAD = SAP_EFF + ["SBA"]   

SAP_DEFINITIONS = {"narrow": SAP_NARROW, "eff": SAP_EFF, "broad": SAP_BROAD}


BUFFER_KM = 50

BUFFER_CRS = "ESRI:102023"
FALLBACK_CRS = "EPSG:3857"

FIPS_TO_ISO = {"SU": "SD", "IV": "CI", "SF": "ZA"}
FIPS_COLUMNS = ["FIPS_10", "FIPS_10_"]
ISO_COLUMNS = ["ISO_A2", "ISO_A2_EH", "ADM0_A3", "SOV_A3"]


NTL_OFFSET = 0.01
EVENT_WINDOW = 5
N_BOOTSTRAP = 999
N_PERMUTE = 2000
SEED = 42



NAVY = "#1B3A4B"
LIGHT = "#AECBD6"
ACCENT = "#C0392B"
GREY = "#7F8C8D"
