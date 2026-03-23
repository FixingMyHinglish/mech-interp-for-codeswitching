from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


@dataclass(frozen=True)
class ComparisonMeta:
    name: str
    focus_condition: str
    baseline_condition: str


class ConditionMeanAccumulator:
    def __init__(self) -> None:
        self.counts: dict[tuple[str, str], dict[int, int]] = {}
        self.sums: dict[tuple[str, str], dict[int, dict[int, np.ndarray]]] = {}

    def update(self, comparison: str, condition: str, position_vectors: dict[int, dict[int, np.ndarray]]) -> None:
        key = (comparison, condition)
        cond_counts = self.counts.setdefault(key, {})
        cond_sums = self.sums.setdefault(key, {})
        for relative_offset, layer_vectors in position_vectors.items():
            cond_counts[relative_offset] = cond_counts.get(relative_offset, 0) + 1
            offset_sums = cond_sums.setdefault(relative_offset, {})
            for layer, vector in layer_vectors.items():
                if layer not in offset_sums:
                    offset_sums[layer] = np.zeros_like(vector, dtype=np.float64)
                offset_sums[layer] += vector.astype(np.float64, copy=False)

    def to_frame(self) -> pd.DataFrame:
        rows = []
        for (comparison, condition), offset_map in self.sums.items():
            for relative_offset, layer_map in offset_map.items():
                count = self.counts[(comparison, condition)][relative_offset]
                for layer, summed in layer_map.items():
                    means = summed / max(count, 1)
                    for neuron, value in enumerate(means):
                        rows.append(
                            {
                                "comparison": comparison,
                                "condition": condition,
                                "relative_offset": int(relative_offset),
                                "layer": int(layer),
                                "neuron": int(neuron),
                                "n_samples": int(count),
                                "mean_logit_effect": float(value),
                            }
                        )
        return pd.DataFrame(
            rows,
            columns=[
                "comparison",
                "condition",
                "relative_offset",
                "layer",
                "neuron",
                "n_samples",
                "mean_logit_effect",
            ],
        )


class PairedDeltaAccumulator:
    def __init__(self, meta: ComparisonMeta) -> None:
        self.meta = meta
        self.counts: dict[int, int] = {}
        self.sum_by_offset: dict[int, dict[int, np.ndarray]] = {}
        self.sumsq_by_offset: dict[int, dict[int, np.ndarray]] = {}

    def update(
        self,
        focus_offsets: dict[int, dict[int, np.ndarray]],
        baseline_offsets: dict[int, dict[int, np.ndarray]],
    ) -> None:
        common_offsets = sorted(set(focus_offsets).intersection(baseline_offsets))
        for relative_offset in common_offsets:
            focus_layers = focus_offsets[relative_offset]
            baseline_layers = baseline_offsets[relative_offset]
            common_layers = sorted(set(focus_layers).intersection(baseline_layers))
            if not common_layers:
                continue
            self.counts[relative_offset] = self.counts.get(relative_offset, 0) + 1
            offset_sum = self.sum_by_offset.setdefault(relative_offset, {})
            offset_sumsq = self.sumsq_by_offset.setdefault(relative_offset, {})
            for layer in common_layers:
                delta = focus_layers[layer].astype(np.float64, copy=False) - baseline_layers[layer].astype(
                    np.float64, copy=False
                )
                if layer not in offset_sum:
                    offset_sum[layer] = np.zeros_like(delta, dtype=np.float64)
                    offset_sumsq[layer] = np.zeros_like(delta, dtype=np.float64)
                offset_sum[layer] += delta
                offset_sumsq[layer] += delta * delta

    def to_frame(self, alpha: float) -> pd.DataFrame:
        rows = []
        for relative_offset, layer_map in self.sum_by_offset.items():
            n = self.counts.get(relative_offset, 0)
            if n <= 0:
                continue
            for layer, summed in layer_map.items():
                sumsq = self.sumsq_by_offset[relative_offset][layer]
                mean = summed / n
                if n > 1:
                    variance = np.maximum((sumsq - (summed * summed) / n) / (n - 1), 0.0)
                    std = np.sqrt(variance)
                    se = std / np.sqrt(n)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        t_stat = np.divide(mean, se, out=np.zeros_like(mean), where=se > 0)
                        dz = np.divide(mean, std, out=np.zeros_like(mean), where=std > 0)
                    p_value = 2.0 * student_t.sf(np.abs(t_stat), df=n - 1)
                else:
                    std = np.full_like(mean, np.nan)
                    t_stat = np.full_like(mean, np.nan)
                    dz = np.full_like(mean, np.nan)
                    p_value = np.full_like(mean, np.nan)

                q_value = benjamini_hochberg(p_value)

                for neuron, value in enumerate(mean):
                    rows.append(
                        {
                            "comparison": self.meta.name,
                            "focus_condition": self.meta.focus_condition,
                            "baseline_condition": self.meta.baseline_condition,
                            "relative_offset": int(relative_offset),
                            "layer": int(layer),
                            "neuron": int(neuron),
                            "n_pairs": int(n),
                            "mean_delta": float(value),
                            "std_delta": float(std[neuron]) if np.isfinite(std[neuron]) else np.nan,
                            "effect_size_dz": float(dz[neuron]) if np.isfinite(dz[neuron]) else np.nan,
                            "t_stat": float(t_stat[neuron]) if np.isfinite(t_stat[neuron]) else np.nan,
                            "p_value": float(p_value[neuron]) if np.isfinite(p_value[neuron]) else np.nan,
                            "q_value": float(q_value[neuron]) if np.isfinite(q_value[neuron]) else np.nan,
                            "significant": bool(np.isfinite(q_value[neuron]) and q_value[neuron] <= alpha),
                        }
                    )
        return pd.DataFrame(
            rows,
            columns=[
                "comparison",
                "focus_condition",
                "baseline_condition",
                "relative_offset",
                "layer",
                "neuron",
                "n_pairs",
                "mean_delta",
                "std_delta",
                "effect_size_dz",
                "t_stat",
                "p_value",
                "q_value",
                "significant",
            ],
        )


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=np.float64)
    result = np.full_like(p_values, np.nan)
    finite_mask = np.isfinite(p_values)
    if not finite_mask.any():
        return result

    finite = p_values[finite_mask]
    order = np.argsort(finite)
    ranked = finite[order]
    m = len(ranked)
    adjusted = ranked * m / (np.arange(m) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)

    restored = np.empty_like(finite)
    restored[order] = adjusted
    result[finite_mask] = restored
    return result
