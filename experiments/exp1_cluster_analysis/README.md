# Cluster Analysis Pipeline
This repository contains details of Experiment 1 in the COMP0087 'Fixing My Hinglish' report, which 
studies neuron activations of code-switched Hindi-English and French-English text using cluster analysis.

## What cluster analysis expects
`cluster_analysis.py` reads a single layer summary from:
```text
03_cleaned/layer_XX_neuron_summary.csv
```
This file is usually produced by:
1. `convert_to_bundle.py`
2. `make_neuron_summaries.py`
or automatically by:
3. `run_pipeline.py`
## How to run cluster analysis
### Manual `k`
```bash
python3 experiments/exp1_cluster_analysis/scripts/cluster_analysis.py \
  --work_dir mech_interp/models/llama/french \
  --layer_num 31 \
  --k 2
```
### Auto-select `k` with silhouette score
```bash
python3 experiments/exp1_cluster_analysis/scripts/cluster_analysis.py \
  --work_dir mech_interp/models/llama/french \
  --layer_num 31 \
  --auto_k \
  --k_min 2 \
  --k_max 10
```
### Change linkage method
```bash
python3 experiments/exp1_cluster_analysis/scripts/cluster_analysis.py \
  --work_dir mech_interp/models/mistral/french \
  --layer_num 31 \
  --auto_k \
  --linkage_method ward
```
## Outputs
Running cluster analysis writes:
- `04_analysis/cluster_summary_layerXX_kY.csv`
- `04_analysis/README_cluster_summary_layerXX_kY.md`

The CSV contains one row per cluster with:
- `cluster`
- `mean_english`
- `mean_code_switched`
- `mean_target`
- `delta_cs`
- `delta_target`
- `n_neurons`

The README contains:
- selected layer and `k`
- linkage method
- total neurons clustered
- largest cluster share
- mean cluster `|delta_cs - delta_target|`
- silhouette candidates when `--auto_k` is used
- full cluster table
  
## Run the full pipeline
If you want to build everything from the raw sparse export and then cluster the final layer automatically:
```bash
python3 experiments/exp1_cluster_analysis/scripts/run_pipeline.py \
  --work_dir mech_interp/models/llama/french \
  --topk_per_layer 0 \
  --auto_k
```
This runs:
1. bundle creation
2. neuron summary generation
3. cluster analysis
4. gap analysis
## Common options
- `--work_dir`: model/language directory to operate on
- `--layer_num`: layer to cluster
- `--k`: fixed number of clusters
- `--auto_k`: choose `k` automatically
- `--k_min`, `--k_max`: search range for auto-`k`
- `--topk_per_layer 0`: keep all neuron indices seen in the sparse export when building the bundle
## Typical workflow
If you already have `03_cleaned/layer_XX_neuron_summary.csv`:
```bash
python3 experiments/exp1_cluster_analysis/scripts/cluster_analysis.py \
  --work_dir mech_interp/models/qwen/hindi \
  --layer_num 27 \
  --auto_k
```
If you only have the raw export:
```bash
python3 experiments/exp1_cluster_analysis/scripts/run_pipeline.py \
  --work_dir mech_interp/models/qwen/hindi \
  --topk_per_layer 0 \
  --auto_k
```
## Dependencies
The clustering scripts require standard scientific Python packages, including:
- `numpy`
- `pandas`
- `scipy`
- `scikit-learn`
## Notes
- Cluster analysis is descriptive: it groups neurons with similar multilingual activation patterns.
- Smaller clusters are often more interpretable than the dominant background cluster.
- Gap analysis in `delta_gap_analysis.py` is a useful follow-up step after clustering.
