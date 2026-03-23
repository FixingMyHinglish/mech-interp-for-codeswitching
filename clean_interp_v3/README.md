# Clean Mech-Interp Rewrite V3

This folder is the third rewrite.

It keeps the input-only, logit-style scoring from `clean_interp_v2`, but it stops treating the whole sentence or the last token as the unit of analysis.

Instead, it aligns each matched pair to the first token where the two versions diverge and scores a small token window around that event.

## What Changed From V2

`clean_interp_v2` asked:
- at the final input token, how much does each neuron change the chosen next-token logit?

`clean_interp_v3` asks:
- around the matched switch or confusion point, how much does each neuron change the chosen next-token logit?

This makes `v3`:
- token-local
- event-aligned
- easier to interpret mechanistically for code-switching and confusion

## Event Alignment Logic

For each matched focus/baseline pair:
- tokenize both texts
- find the first token position where they diverge
- treat that as the event token
- use the prediction position just before that token as the anchor

Example:
- English: `I need a car`
- Confused: `I need a coche`

If the first difference starts at `coche`, the anchor is the token position right before it.
That is the position whose next-token prediction decides whether `car` or `coche` appears next.

## Scoring Rule

For each token position in a window such as `[-1, 0, +1]` around the anchor:
- capture each MLP neuron's value before the output projection
- reconstruct that neuron's output contribution through `c_proj` or `down_proj`
- measure how much adding that neuron changes the chosen next-token logit

The target token is configurable:
- `predicted_next_token`: use the model's own top next-token prediction at that position
- `observed_next_token`: use the actual next token from the input text

## Outputs

All main tables now include `relative_offset`.

Important files:
- `event_alignments.csv`: the detected event token and anchor per matched pair
- `position_targets.csv`: which token was scored at each offset
- `comparison_stats.csv.gz`: neuron stats per comparison and offset
- `phenomenon_consensus.csv.gz`: neurons that survive both baselines per offset
- `top_neurons_<phenomenon>.csv`: top-ranked neurons, with offsets included

Figures:
- one heatmap per comparison and offset
- one consensus heatmap per phenomenon and offset
- one layer summary heatmap across comparison-offset combinations

## Notes

- This is still input-only. It does not generate text or detect a generation-time confusion point.
- The event anchor is tokenized-text divergence, not semantic annotation.
- For GPT-2 `c_proj`, the shared projection bias is not assigned to any single neuron. Per-neuron scores only use the neuron-specific output direction.

## Run

```bash
python clean_interp_v3/run.py --config clean_interp_v3/config.example.yaml
```
