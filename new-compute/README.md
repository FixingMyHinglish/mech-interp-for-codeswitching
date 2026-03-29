# New Compute

This folder contains compact post-processing for condition-specific neuron analysis
without saving full raw activation dumps.

## Script

`compute_condition_specific_neurons.py`
`ablation_condition_unique_neurons.py`
`visualize_new_compute.py`
`ab_compare_hf_vs_tl_subset.py`

## What it does

1. Finds neurons that fire (`activation > activation_cutoff`).
2. Computes condition/domain-specific importance scores:
   - `importance_score = activation_mean * coverage`
   - `coverage = fire_count / samples_in_condition_domain`
3. Selects important neurons by:
   - per condition-domain quantile (`--importance_quantile`) or
   - absolute threshold (`--importance_min`)
4. Removes English + target-language neurons from:
   - `code_switched`
   - `confused`
5. Aggregates across domains and computes consistency.
6. Compares CS vs confused sets (overlap/Jaccard/unique counts).

Selection backends:

- `transformer_lens` (default): extracts MLP neuron activations from standard TL hooks (`blocks.{l}.mlp.hook_post`)
- `pipeline_proxy`: uses existing `run_dir/tables/neuron_proxy_raw.csv`

## Example

TransformerLens backend (default):

```bash
python new-compute/compute_condition_specific_neurons.py \
  --dataset_csv data/hindi.csv \
  --run_dir outputs/gpt2_hindi_run_002 \
  --out_dir new-compute/results_gpt2_hindi_run_002 \
  --backend transformer_lens \
  --activation_cutoff 0 \
  --importance_quantile 0.9 \
  --min_domain_consistency 0.5
```

Legacy pipeline-proxy backend:

```bash
python new-compute/compute_condition_specific_neurons.py \
  --backend pipeline_proxy \
  --run_dir outputs/gpt2_hindi_run_002 \
  --out_dir new-compute/results_gpt2_hindi_run_002
```

## Outputs

- `all_fired_rows.csv.gz`
- `neuron_importance_condition_domain.csv.gz`
- `important_neurons_condition_domain.csv.gz`
- `cs_confused_after_filtering_base.csv.gz`
- `consistent_neurons.csv.gz`
- `summary.json`

## Ablation Study (unique CS vs confused neurons)

This script silences neurons unique to each condition and checks whether generated
continuations become more or less monolingual.

```bash
python new-compute/ablation_condition_unique_neurons.py \
  --run_dir outputs/gpt2_hindi_run_002 \
  --dataset_csv data/hindi.csv \
  --consistent_neurons_csv new-compute/results_gpt2_hindi_run_002.2/consistent_neurons.csv.gz \
  --out_dir new-compute/ablation_gpt2_hindi_run_002 \
  --target_script auto \
  --max_eval_per_condition 150 \
  --max_new_tokens 32
```

Ablation outputs:

- `ablation_predictions.csv.gz`: per-prompt generated text + monolinguality metrics
- `ablation_summary.json`: average monolinguality/mixedness deltas vs baseline, including `target_language_monolinguality_score`
- `cs_unique_neurons.csv`
- `confused_unique_neurons.csv`

## Visualization (5 random samples + overall)

This creates attention-style and residual-stream attribution heatmaps:

- Per sample (default 5 random rows):
  - layer attention head grid
  - single annotated head heatmap
  - layer x position normalized residual-stream logit attribution heatmap
- Overall:
  - per-condition aggregate attention head grid
  - per-condition aggregate single-head heatmap
  - per-condition aggregate residual attribution heatmap
  - optional overall new-compute importance heatmap (if `--results_dir` is provided)

Backends:

- `--backend hf` (default): Hugging Face `transformers`
- `--backend transformer_lens`: TransformerLens `HookedTransformer` (useful for some Llama runs)

If you use TransformerLens backend, install it first:

```bash
pip install transformer-lens
```

```bash
python new-compute/visualize_new_compute.py \
  --run_dir outputs/gpt2_hindi_run_002 \
  --dataset_csv data/hindi.csv \
  --results_dir new-compute/results_gpt2_hindi_run_002 \
  --out_dir new-compute/visuals_gpt2_hindi \
  --n_random_samples 5 \
  --attention_layer 0 \
  --backend transformer_lens \
  --target_mode predicted_next_token
```

Outputs are written under:

- `samples/*.html`
- `overall/*_<condition>.html`
- `manifest.json`

## Run New-Compute Directly From Main Entrypoint

You can now call `scripts/run_pipeline.py` in new-compute-only mode.

Using config-driven defaults (recommended):

- `run_dir` defaults to `output_dir` from config
- visuals dataset defaults to `input_path` from config
- if you omit output folders, they are auto-derived under `new-compute/`
- selection backend defaults to TransformerLens
- aggregate visuals default to all rows in each condition (`--new-compute-overall-sample-cap 0`)

```bash
python scripts/run_pipeline.py \
  --new-compute-only \
  --config config.yaml
```

With visuals in the same command:

```bash
python scripts/run_pipeline.py \
  --new-compute-only \
  --config config.yaml \
  --new-compute-with-visuals \
  --new-compute-visual-backend transformer_lens
```

Explicit overrides still work:

```bash
python scripts/run_pipeline.py \
  --new-compute-only \
  --new-compute-selection-backend pipeline_proxy \
  --new-compute-run-dir outputs/gpt2_hindi_run_002 \
  --new-compute-out-dir new-compute/results_gpt2_hindi_run_002 \
  --new-compute-importance-quantile 0.9 \
  --new-compute-min-domain-consistency 0.5
```

With visuals in the same run:

```bash
python scripts/run_pipeline.py \
  --new-compute-only \
  --new-compute-run-dir outputs/gpt2_hindi_run_002 \
  --new-compute-out-dir new-compute/results_gpt2_hindi_run_002 \
  --new-compute-with-visuals \
  --new-compute-dataset-csv data/hindi.csv \
  --new-compute-visual-out-dir new-compute/visuals_gpt2_hindi \
  --new-compute-visual-backend transformer_lens
```

## A/B Selection (baseline vs TransformerLens) on a subset

This compares:

- A: current selection from `tables/neuron_proxy_raw.csv`
- B: TransformerLens MLP-hook proxy on the same subset

```bash
python new-compute/ab_compare_hf_vs_tl_subset.py \
  --run_dir outputs/gpt2_french_run_001 \
  --dataset_csv data/french.csv \
  --out_dir new-compute/ab_french_subset \
  --n_source_ids 40 \
  --seed 42
```

Key outputs:

- `ab_condition_overlap.csv`
- `ab_summary.json`
- `a_consistent_baseline.csv.gz`
- `b_consistent_tl.csv.gz`

## Experiment-Specific Runners

For standalone experiment folders (each runnable on its own), see:

- `new-compute/experiments/exp4_switch_activation/`
- `new-compute/experiments/exp6_language_selectivity/`
- `new-compute/experiments/exp21_language_pair_comparison/`
- `new-compute/experiments/exp7_neuron_clustering_interpretation/`
