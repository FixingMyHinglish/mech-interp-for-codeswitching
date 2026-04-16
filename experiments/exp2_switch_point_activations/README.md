# Experiment 2: Neuron Activation Patterns at Switch Points

This script computes token-local switch-point activation patterns for:

- `code_switched` vs `english`
- `code_switched` vs `target_language`
- `english` vs `target_language` (control; enabled by default when both are present)

It aligns each pair to the first token divergence, collects MLP post activations
in a window around that point, and exports:

- per-neuron mean deltas and z-scores
- `% neurons` above a z-threshold per layer
- top-k neurons per pair/offset
- layer-neuron heatmaps
- consensus across both monolingual baselines
- per-sample three-way neuron intersections (`focus`/`english`/`target_language`)
- per-sample filtered focus-neuron sets (baseline-filtered)
- cross-sample consistency + strict intersections of filtered focus neurons

To disable the control comparison:

```bash
python experiments/exp2_switch_point_activations/run.py \
  --dataset_csv data/hindi.csv \
  --model_name gpt2 \
  --device cpu \
  --no_english_target_control
```

To tune per-sample filtering:

```bash
python experiments/exp2_switch_point_activations/run.py \
  --dataset_csv data/hindi.csv \
  --model_name gpt2 \
  --device cpu \
  --cs_active_quantile 0.9 \
  --baseline_active_quantile 0.9 \
  --baseline_filter_mode intersection \
  --triple_similarity_rel_tol 0.1
```

## Run

```bash
python experiments/exp2_switch_point_activations/run.py \
  --dataset_csv data/hindi.csv \
  --model_name gpt2 \
  --out_dir new-compute/experiments/exp4_switch_activation/results_hindi \
  --device cpu
```

GPU-friendly mode:

```bash
python experiments/exp2_switch_point_activations/run.py \
  --dataset_csv data/hindi.csv \
  --model_name gpt2 \
  --device cuda \
  --gpu_friendly
```

## Analyze outputs (metrics + top-k report)

After `run.py` completes, generate a one-page analysis report:

```bash
python3 experiments/exp2_switch_point_activations/analyze_exp4.py \
  --results_dir new-compute/experiments/exp4_switch_activation/results_french_gpt2_mps \
  --focus_offset 0 \
  --z_threshold 2.0 \
  --top_k 50
```

The analyzer now also computes a control-filtered metric for code-switch-only neurons:

- requires high signal in both `code_switched_vs_english` and `code_switched_vs_target_language`
- requires low `|z|` in `english_vs_target_language` (control; default max `|z| <= 1.0`)

You can tune control filtering with:

```bash
python3 experiments/exp2_switch_point_activations/analyze_exp4.py \
  --results_dir new-compute/experiments/exp4_switch_activation/results_french_gpt2_mps \
  --focus_offset 0 \
  --z_threshold 2.0 \
  --control_comparison english_vs_target_language \
  --control_abs_z_max 1.0 \
  --top_k 50
```
