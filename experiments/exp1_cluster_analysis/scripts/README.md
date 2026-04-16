# Experiment 1: Neuron Cluster Analysis

This folder contains the clustering sub-pipeline for multilingual neuron
activations exported by the main pipeline. It groups neurons by their per-condition
mean activations and writes a CSV summary plus a companion Markdown report.

All scripts live under:

```text
experiments/exp1_cluster_analysis/scripts/
```

- `convert_to_bundle.py` — sparse-export → per-layer bundle.
- `make_neuron_summaries.py` — per-layer neuron summary CSVs.
- `cluster_analysis.py` — clusters one layer and writes outputs to `04_analysis/`.
- `delta_gap_analysis.py` — follow-up gap analysis across several `k` values.
- `run_pipeline.py` — runs all of the above end-to-end.

## Work-directory layout

Each model/language run lives in its own work directory, for example:

```text
mech_interp/models/<model>/<language>/
```

With the following subfolders:

- `01_raw/full_neuron_activations.jsonl.gz` — sparse export from the main pipeline.
- `02_bundle/` — per-layer bundles produced by `convert_to_bundle.py`.
- `03_cleaned/` — per-layer neuron summaries produced by `make_neuron_summaries.py`.
- `04_analysis/` — cluster outputs written by `cluster_analysis.py` and `delta_gap_analysis.py`.

## What `cluster_analysis.py` expects

A single-layer summary at:

```text
03_cleaned/layer_XX_neuron_summary.csv
```

That file is produced either by running `convert_to_bundle.py` followed by
`make_neuron_summaries.py`, or automatically by `run_pipeline.py`.

## Running from the repo root

The scripts use sibling imports (e.g. `from convert_to_bundle import main`).
Either `cd` into the scripts directory before running, or set `PYTHONPATH`:

```bash
export PYTHONPATH=experiments/exp1_cluster_analysis/scripts
```

All commands below assume you are at the repo root with `PYTHONPATH` set as above.

### Cluster one layer with a fixed `k`

```bash
python3 experiments/exp1_cluster_analysis/scripts/cluster_analysis.py \
  --work_dir mech_interp/models/llama/french \
  --layer_num 31 \
  --k 2
```

### Auto-select `k` by silhouette score

```bash
python3 experiments/exp1_cluster_analysis/scripts/cluster_analysis.py \
  --work_dir mech_interp/models/llama/french \
  --layer_num 31 \
  --auto_k \
  --k_min 2 \
  --k_max 10
```

### Change the linkage method

```bash
python3 experiments/exp1_cluster_analysis/scripts/cluster_analysis.py \
  --work_dir mech_interp/models/mistral/french \
  --layer_num 31 \
  --auto_k \
  --linkage_method ward
```

## Outputs

`cluster_analysis.py` writes, into `04_analysis/`:

- `cluster_summary_layerXX_kY.csv` — one row per cluster with:
  `cluster`, `mean_english`, `mean_code_switched`, `mean_target`,
  `delta_cs`, `delta_target`, `n_neurons`.
- `README_cluster_summary_layerXX_kY.md` — a human-readable report containing
  the selected layer and `k`, linkage method, total neurons clustered,
  largest-cluster share, the mean of `|delta_cs - delta_target|` across
  clusters, silhouette candidates when `--auto_k` is used, and the full
  cluster table.

## End-to-end run

To build everything from the raw sparse export and then cluster the final
layer automatically:

```bash
python3 experiments/exp1_cluster_analysis/scripts/run_pipeline.py \
  --work_dir mech_interp/models/llama/french \
  --topk_per_layer 0 \
  --auto_k
```

This runs, in order:

1. bundle creation (`convert_to_bundle.py`)
2. neuron summary generation (`make_neuron_summaries.py`)
3. cluster analysis (`cluster_analysis.py`)
4. gap analysis (`delta_gap_analysis.py`)

## Common options

- `--work_dir` — model/language directory to operate on.
- `--layer_num` — layer to cluster (defaults to the max layer present in `03_cleaned/`).
- `--k` — fixed number of clusters.
- `--auto_k` — choose `k` automatically by silhouette score.
- `--k_min`, `--k_max` — search range for auto-`k`.
- `--topk_per_layer 0` — keep all neuron indices seen in the sparse export when building the bundle.

## Typical workflows

If you already have `03_cleaned/layer_XX_neuron_summary.csv`, jump straight to clustering:

```bash
python3 experiments/exp1_cluster_analysis/scripts/cluster_analysis.py \
  --work_dir mech_interp/models/qwen/hindi \
  --layer_num 27 \
  --auto_k
```

If you only have the raw sparse export, run the full pipeline:

```bash
python3 experiments/exp1_cluster_analysis/scripts/run_pipeline.py \
  --work_dir mech_interp/models/qwen/hindi \
  --topk_per_layer 0 \
  --auto_k
```

## Dependencies

The clustering scripts use standard scientific Python packages, all covered by
the repo's top-level `requirements.txt`:

- `numpy`
- `pandas`
- `scipy`
- `scikit-learn`

## Notes

- Cluster analysis is descriptive: it groups neurons with similar multilingual
  activation patterns — it doesn't make causal claims.
- Smaller clusters are often more interpretable than the dominant background cluster.
- `delta_gap_analysis.py` is a useful follow-up for sweeping several `k` values
  and comparing gap statistics across them.