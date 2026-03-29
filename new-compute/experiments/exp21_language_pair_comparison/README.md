# Experiment 21: Language-Pair Comparison (Hi-En vs Fr-En)

This script compares outputs from:

- `exp4_switch_activation`
- optional `exp6_language_selectivity`

for Hindi-English and French-English runs, then reports overlap/Jaccard and
layer-profile similarity by offset.

## Run (exp4 only)

```bash
python new-compute/experiments/exp21_language_pair_comparison/run.py \
  --hindi_exp4_dir new-compute/experiments/exp4_switch_activation/results_hindi \
  --french_exp4_dir new-compute/experiments/exp4_switch_activation/results_french \
  --out_dir new-compute/experiments/exp21_language_pair_comparison/results
```

## Run (exp4 + exp6)

```bash
python new-compute/experiments/exp21_language_pair_comparison/run.py \
  --hindi_exp4_dir new-compute/experiments/exp4_switch_activation/results_hindi \
  --french_exp4_dir new-compute/experiments/exp4_switch_activation/results_french \
  --hindi_exp6_dir new-compute/experiments/exp6_language_selectivity/results_hindi \
  --french_exp6_dir new-compute/experiments/exp6_language_selectivity/results_french \
  --out_dir new-compute/experiments/exp21_language_pair_comparison/results
```

