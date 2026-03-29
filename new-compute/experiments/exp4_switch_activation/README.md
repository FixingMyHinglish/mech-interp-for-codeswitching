# Experiment 4: Neuron Activation Patterns at Switch Points

This script computes token-local switch-point activation patterns for:

- `code_switched` vs `english`
- `code_switched` vs `target_language`

It aligns each pair to the first token divergence, collects MLP post activations
in a window around that point, and exports:

- per-neuron mean deltas and z-scores
- `% neurons` above a z-threshold per layer
- top-k neurons per pair/offset
- layer-neuron heatmaps
- consensus across both monolingual baselines

## Run

```bash
python new-compute/experiments/exp4_switch_activation/run.py \
  --dataset_csv data/hindi.csv \
  --model_name gpt2 \
  --out_dir new-compute/experiments/exp4_switch_activation/results_hindi \
  --device cpu
```

GPU-friendly mode:

```bash
python new-compute/experiments/exp4_switch_activation/run.py \
  --dataset_csv data/hindi.csv \
  --model_name gpt2 \
  --device cuda \
  --gpu_friendly
```
