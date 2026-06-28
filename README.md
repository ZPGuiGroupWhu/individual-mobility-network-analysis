# Individual Mobility Network Analysis Code

This repository contains the code for the paper “Scale-invariant and compressible core-periphery structure in human mobility”.

## Repository Layout

```text
paper_code_release/
  README.md
  requirements.txt
  config/
    analysis.yaml
    paths.example.yaml
  data/
    D1_YJMob100K/
      IMN/
      location_sequence/
    D3_FourSuqare/
      IMN/
      location_sequence/
  scripts/
    00_prepare_data.py
    01_feature_analysis.py
    02_multiscale_core_periphery.py
    03_network_compressibility.py
    04_summarize_results.py
    run_all.py
  src/
    paper_analysis/
      compressibility/
      data_preprocess/
      features/
      multiscale/
      utils/
  results/
    README.md
```

## Data

The public release includes example IMN and location-sequence data for:

- `D1_YJMob100K`
- `D3_FourSuqare`

The private dataset `D2_VehicleTripleg` is not included because it contains
privacy-sensitive vehicle trip information. The path configuration keeps a
placeholder for this dataset so authorized users can place private files in the
same directory structure and run the same scripts locally.

Expected public input layout:

```text
data/D1_YJMob100K/IMN/G_*.json
data/D1_YJMob100K/location_sequence/L_*.csv
data/D3_FourSuqare/IMN/G_*.json
data/D3_FourSuqare/location_sequence/L_*.csv
```

## Environment

Python 3.9 or newer is recommended. Create a fresh environment and install the
dependencies from the release root:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS or Linux, use:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The default workflow imports modules from `src/` through the script helpers, so
the scripts can be run directly from the repository root.

## Reproduction Workflow

Run the full public workflow:

```bash
python scripts/run_all.py --dataset all
```

Run individual steps:

```bash
python scripts/00_prepare_data.py --dataset all
python scripts/01_feature_analysis.py --dataset all
python scripts/02_multiscale_core_periphery.py --dataset all
python scripts/03_network_compressibility.py --dataset all
python scripts/04_summarize_results.py --dataset all
```

The `--dataset` argument accepts:

```text
D1_YJMob100K
D3_FourSuqare
all
```

Common optional arguments:

```bash
--config config/paths.example.yaml
--analysis-config config/analysis.yaml
```

## Analysis Steps

1. `00_prepare_data.py`
   Checks the public input files and writes a data manifest.

2. `01_feature_analysis.py`
   Computes topological features from `G_*.json` and spatial features from
   paired `L_*.csv` files.

3. `02_multiscale_core_periphery.py`
   Fits container-model communities, runs module-level core-periphery detection,
   and filters modules using the fixed criteria in `config/analysis.yaml`.

4. `03_network_compressibility.py`
   Computes module compressibility and random-walk entropy. The default random
   model is ER with 10 randomized networks and seed 42.

5. `04_summarize_results.py`
   Copies reproducibility tables from `results/intermediate/` into
   `results/tables/`.
