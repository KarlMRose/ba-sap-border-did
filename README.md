# SAPs and Local Development in Sub-Saharan Africa

Code for my bachelor thesis on IMF-supported Structural Adjustment Programs and
local economic development in Sub-Saharan Africa.

The analysis uses ethnic groups that colonial borders split across two
countries, nighttime light intensity as a proxy for local economic activity,
and a border-based difference-in-differences design. Identification follows
Michalopoulos & Papaioannou (2014): both halves of a partitioned group share
culture, language and geography, so the main thing separating them is the
national policy regime — including whether the country is under an SAP.

## Structure

```
notebooks/   analysis notebooks, run in order
src/         helper functions and paths
data/        replication data (see below)
output/      generated tables and figures
```

## Data

Raw data are not in this repository. They ship separately as `data_package.zip`.

Extract the ZIP into the project folder so that this path exists:

```
ba-sap-border-did/data/raw/GREG/GREG.shp
```

Keep the folder structure unchanged — all paths are resolved relative to the
repository root in `src/config.py`.

The package contains:

| Folder | Source |
|---|---|
| `data/raw/GREG/` | GREG shapefile (Weidmann, Rød & Cederman 2010), digitised from the Atlas Narodov Mira (1964) |
| `data/raw/VIIRS/` | harmonized DMSP/VIIRS nighttime lights, Li et al. (2020), 1992–2024 |
| `data/raw/naturalearth/` | Natural Earth country boundaries, 10m and 110m |
| `data/raw/vreeland/` | Vreeland IMF programme dataset |

## Running it

1. `pip install -r requirements.txt`
2. Run the notebooks in this order:
   - `00_partitioned_groups.ipynb` — identify groups split by a border
   - `01_treatment.ipynb` — build SAP indicators from the Vreeland data
   - `02_buffer_and_ntl.ipynb` — 50 km border buffers, extract nighttime lights
   - `03_panel.ipynb` — assemble the panel and the within-pair differences
   - `04_results.ipynb` — main estimates, event study, sample map
   - `05_inference.ipynb` — bootstrap and randomization inference

Each notebook writes intermediate files to `data/work/`, so they need to run in
order the first time. After that they can be re-run individually.

## Notes on the design

The estimating equation is a within-pair difference-in-differences with
ethnicity-by-year fixed effects, which means each treated side is only ever
compared to its own control side in the same year. With two units per group
this is algebraically the same as regressing the light gap between the two
halves on a post-treatment dummy, and both are reported.

There are 11 groups in the final sample, so standard cluster-robust inference
is not reliable. Results are reported with a pairs cluster bootstrap and a
randomization inference test that permutes treatment timing across groups.
