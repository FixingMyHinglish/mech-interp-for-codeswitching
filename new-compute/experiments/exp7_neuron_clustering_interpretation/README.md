# Experiment 7: Neuron Clustering & Interpretation

This experiment:

1. Builds per-neuron activation profiles across:
   - `code_switched`
   - `english`
   - `target_language`
2. Adds switch-window event features in code-switched text.
3. Clusters neurons using KMeans in standardized feature space.
4. Produces heuristic semantic labels and token-pattern summaries per cluster.

Outputs include:

- `tables/neuron_cluster_assignments.csv.gz`
- `tables/cluster_summary.csv`
- `tables/cluster_token_patterns.csv`
- `tables/cluster_interpretations.csv`
- `figures/cluster_centroids_heatmap.html`
- `summary.json`

## Run

```bash
python new-compute/experiments/exp7_neuron_clustering_interpretation/run.py \
  --dataset_csv data/hindi.csv \
  --model_name gpt2 \
  --out_dir new-compute/experiments/exp7_neuron_clustering_interpretation/results_hindi \
  --device cpu
```

GPU-friendly mode:

```bash
python new-compute/experiments/exp7_neuron_clustering_interpretation/run.py \
  --dataset_csv data/hindi.csv \
  --model_name gpt2 \
  --device cuda \
  --gpu_friendly
```

