# MechInterp2.0

This repository's main workflow is a code-switching mechanistic-interpretability
pipeline built around:

- `scripts/run_pipeline.py`: the main command-line entrypoint.
- `pipeline/`: the reusable analysis, modeling, IO, concept-metric, and plotting helpers used by the main script.
- `config.yaml`: the run configuration file.

The `experiments/` directory contains separate experiment-specific code. Those
folders are intentionally self-contained.

## Repository Layout

- `data/`: cleaned Hindi-English and French-English datasets.
- `pipeline/`: core Python package used by the main pipeline.
- `scripts/run_pipeline.py`: main pipeline runner.
- `scripts/`: supporting utilities for dataset checks, activation exports, comparisons,  and visualization.
- `config.yaml`: example/default run configuration.
- `experiments/`: standalone experiments, documented inside their own folders.

## Data Format

The pipeline accepts `.csv`, `.json`, or `.jsonl` input files. The required
columns are:

- `id`: unique row identifier.
- `text`: text shown to the model.
- `condition`: condition label, for example `english`, `target_language`, `code_switched`, or `confused`.
- `domain`: topic/domain label used for grouping and summary metrics.

The cleaned datasets currently kept in the repo are:

- `data/hindi.csv`
- `data/french.csv`


## Setup

From the repo root:

```bash
python3 -m pip install -r requirements.txt
```

## Configure

Edit `config.yaml` before running:

```yaml
model_name: "mistral-7b"
input_path: "data/french.csv"
output_dir: "outputs/mistral7b_french"
device: "mps"
max_length: 256
batch_size: 10
concept_column: "condition"
topk_neurons: 100
save_full_neuron_activations: true
full_neuron_export_gzip: true
full_neuron_round_decimals: 3
full_neuron_topk_per_layer: 256
full_neuron_sample_stride: 1
full_neuron_min_layer_exclusive: 19
```

Important config fields:

- `model_name`: Hugging Face causal language model identifier.
- `input_path`: dataset path.
- `output_dir`: where tables, figures, text exports, and metadata are written.
- `device`: `auto`, `cpu`, `cuda`, or `mps`.
- `max_length`: token truncation length.
- `topk_neurons`: number of neuron-proxy units kept per layer for summary outputs.
- `reference_condition`: optional baseline condition; if omitted, the pipeline prefers a condition containing `english`.
- `save_full_neuron_activations`: whether to export per-sample, per-layer neuron activation summaries.
- `full_neuron_topk_per_layer`: if greater than zero, saves sparse top-k neuron activations per layer instead of full dense vectors.
- `concept_column`: dataset column used for concept-level metrics, usually `domain` or `condition`.
- `compute_concept_metrics`: whether to run concept selectivity, purity, classifier, and clustering summaries.

## Run

Run the main pipeline:

```bash
python3 scripts/run_pipeline.py --config config.yaml
```

## Main Outputs

Each run writes outputs under the configured `output_dir`, usually
`outputs/<run_name>/`.

Common tables:

- `tables/layer_metrics_raw.csv`
- `tables/attention_entropy_raw.csv`
- `tables/neuron_proxy_raw.csv`
- `tables/layer_metrics_diff.csv`
- `tables/attention_diff.csv`
- `tables/neuron_diff.csv`
- `tables/summary.csv`
- `tables/pairwise_summary.csv`
- `tables/neuron_events.csv`
- `tables/neuron_tendency.csv`
- `tables/sample_neuron_contrast.csv`
- `tables/sample_layer_condition_distance.csv`

Concept-metric tables, when enabled:

- `tables/concept_selectivity.csv`
- `tables/concept_purity.csv`
- `tables/concept_layer_density.csv`
- `tables/concept_classifier_summary.csv`
- `tables/concept_classifier_per_class.csv`
- `tables/concept_clustering_summary.csv`
- `tables/concept_hierarchy_consistency.csv`
- `tables/concept_functional_effects.csv`

Text exports and metadata:

- `text_exports/neuron_events.jsonl`
- `text_exports/neuron_tendency.jsonl`
- `text_exports/full_neuron_activations.jsonl.gz`, if enabled.
- `text_exports/IMPORTANT_NUMBERS.txt`
- `text_exports/CONCEPT_METRICS_SUMMARY.txt`, if concept metrics are enabled.
- `SUMMARY.md`
- `metadata.json`

Figures are written under `figures/` inside the same output directory.
