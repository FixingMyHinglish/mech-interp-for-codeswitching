# Experiment 6: Language-Selective Neurons

This script uses token-level language labels inside `code_switched` text and
switch-point windows to compute:

- per-neuron selectivity index  
  `(mean_target_activation - mean_english_activation) / (mean_target_activation + mean_english_activation + eps)`
- per-layer fraction of selective neurons above threshold
- top-k target-selective, english-selective, and absolute-selective neurons
- layer-neuron selectivity heatmaps

## Run

```bash
python new-compute/experiments/exp6_language_selectivity/run.py \
  --dataset_csv data/hindi.csv \
  --model_name gpt2 \
  --out_dir new-compute/experiments/exp6_language_selectivity/results_hindi \
  --device cpu
```

GPU-friendly mode:

```bash
python new-compute/experiments/exp6_language_selectivity/run.py \
  --dataset_csv data/hindi.csv \
  --model_name gpt2 \
  --device cuda \
  --gpu_friendly
```
