# Code-Switch vs Language-Confusion Mech-Interp Pipeline

This project compares internal LLM behavior across multiple text conditions within shared domains, including:

- `english`
- `target_language`
- `code_switched`
- `confused`

It uses Tuned Lens (when available) plus activation/attention/neuron-proxy metrics.

## What You Get

For each run, outputs are written under `outputs/<run_name>/`:

- `tables/layer_metrics_raw.csv`
- `tables/attention_entropy_raw.csv`
- `tables/neuron_proxy_raw.csv`
- `tables/layer_metrics_diff.csv` (each condition vs reference)
- `tables/attention_diff.csv` (each condition vs reference)
- `tables/neuron_diff.csv` (each condition vs reference)
- `tables/summary.csv` (global deltas vs reference)
- `tables/pairwise_summary.csv` (all-condition pairwise deltas)
- `tables/neuron_events.csv` (per-sample per-layer top-neuron events)
- `tables/neuron_tendency.csv` (aggregated neuron firing tendencies)
- `tables/sample_neuron_contrast.csv` (per source sample: neuron activations across conditions + deltas)
- `tables/sample_layer_condition_distance.csv` (per source sample/layer: cosine & top-neuron overlap across condition pairs)
- `text_exports/neuron_events.jsonl`
- `text_exports/neuron_tendency.jsonl`
- `text_exports/full_neuron_activations.jsonl` (optional; very large)
- `text_exports/IMPORTANT_NUMBERS.txt`
- `figures/layer_metric_deltas.html`
- `figures/attention_entropy_heatmap_<condition>_vs_<reference>.html`
- `figures/neuron_shift_top100_<condition>_vs_<reference>.html`
- `figures/neuron_layer_heatmap_absolute_<condition>.html`
- `figures/neuron_layer_3d_absolute_<condition>.html`
- `figures/neuron_layer_heatmap_<condition>_vs_<reference>.html`
- `figures/neuron_layer_3d_<condition>_vs_<reference>.html`
- `figures/domain_metric_heatmap_<condition>_vs_<reference>.html`
- `SUMMARY.md`
- `metadata.json`

## Input Format

CSV/JSON/JSONL with required columns:

- `id`: unique identifier
- `text`: input text
- `condition`: condition label (`english`, `target_language`, `code_switched`, `confused`, etc.)
- `domain`: shared-topic grouping

Example: `data/text_pairs.example.csv`

## Setup

```bash
cd "/Users/ridhi/Desktop/mech interp final"
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

## Configure

```bash
config.yaml
```

Main fields:

- `model_name`: Hugging Face CausalLM identifier
- `input_path`: dataset path
- `output_dir`: run output folder
- `device`: `auto`, `cpu`, `cuda`, or `mps`
- `max_length`: token truncation limit
- `tuned_lens_resource_id`: optional lens checkpoint override
- `topk_neurons`: how many neuron-proxy units to keep per layer
- `reference_condition`: optional baseline condition (recommended: `english`)
- `log_level`: logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`)
- `log_every_n_samples`: emit per-sample progress log every N samples
- `save_full_neuron_activations`: stream full per-layer neuron vectors per sample to JSONL
- `full_neuron_reduce_mode`: token reduction for neuron vectors (`mean_abs`, `mean`, `max_abs`)
- `full_neuron_export_gzip`: write `full_neuron_activations.jsonl.gz` instead of plain JSONL
- `full_neuron_round_decimals`: round neuron values to reduce file size
- `full_neuron_layers`: optional list of layers to keep in full activation export
- `full_neuron_min_layer_exclusive`: keep only layers strictly greater than this value
- `full_neuron_topk_per_layer`: if >0, save sparse top-k neurons per layer instead of full vectors
- `full_neuron_sample_stride`: only export every Nth sample for the full activation file

If `reference_condition` is omitted, the pipeline auto-picks a condition containing `english`, else first alphabetically.

## Run

```bash
 python scripts/run_pipeline.py --config config.yaml
```

## Notes

- Neuron stats are MLP-neuron proxies when architecture allows (`up_proj`/`c_fc`/`fc_in`/`gate_proj`), otherwise hidden-dimension salience.
- If Tuned Lens is unavailable for your model, the pipeline falls back to logit-lens-like probing so analyses still run.

- `tables/concept_selectivity.csv` (neuron-concept selectivity: diff/ratio/effect-size/KL)
- `tables/concept_purity.csv` (top-activation concept purity/entropy per neuron)
- `tables/concept_layer_density.csv` (fraction of concept-associated neurons per layer)
- `tables/concept_classifier_summary.csv` (concept prediction from neuron activations: accuracy/F1/AUC)
- `tables/concept_classifier_per_class.csv`
- `tables/concept_clustering_summary.csv` (cluster/silhouette over neuron concept profiles)
- `tables/concept_hierarchy_consistency.csv` (if hierarchy CSV provided)
- `tables/concept_functional_effects.csv` (ablation/boost/inhibit NLL deltas; optional heavy)
- `text_exports/CONCEPT_METRICS_SUMMARY.txt`
- `concept_column`: dataset column to treat as concept label (default `domain`)
- `compute_concept_metrics`: compute concept association/selectivity/classifier/structure metrics
- `concept_top_n_purity`: top-N activations used for purity/entropy
- `concept_classifier_test_size`: test split fraction for concept prediction metric
- `concept_hierarchy_path`: optional CSV (`child,parent`) for hierarchical consistency metric
- `compute_concept_functional_tests`: run functional ablation/boost/inhibit tests (expensive)
- `concept_functional_topk_neurons`: top selective neurons per concept for functional tests
- `concept_functional_max_samples`: sample cap for functional tests