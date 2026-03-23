#!/usr/bin/env python3
"""
layer_analysis.py — Statistical layer localisation for condition-specific neurons.

Four complementary methods
--------------------------
1. Importance-weighted layer centroid
   Where is the centre-of-mass of condition-specific activity?
   layer_centroid = Σ(layer × importance_score) / Σ(importance_score)

2. Per-layer Mann-Whitney U  +  Benjamini-Hochberg FDR correction
   For each layer: are CS importance scores stochastically greater than
   confused/baseline importance scores?  Effect size = rank-biserial correlation.

3. Chi-squared layer enrichment
   Are condition-specific consistent neurons non-uniformly distributed across
   layers?  Standardised residuals identify which layers are enriched / depleted.

4. Cumulative importance CDF  +  bootstrapped median layer
   At which layer does cumulative importance cross 50%?  Bootstrap CI shows
   uncertainty in that estimate.

Usage
-----
python new-compute/layer_analysis.py \\
    --results_dir new-compute/results_GPT2_french

All inputs are produced by compute_condition_specific_neurons.py and must
exist in --results_dir before running this script.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chisquare, mannwhitneyu


# ── formatting helpers ────────────────────────────────────────────────────────

W = 72

def hdr(title: str) -> None:
    print(f"\n{'═' * W}")
    print(f"  {title}")
    print(f"{'═' * W}")


def sub(title: str) -> None:
    print(f"\n  ── {title} {'─' * max(0, W - 6 - len(title))}")


def sig_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "** "
    if p < 0.05:
        return "*  "
    return "   "


# ── Benjamini-Hochberg (no statsmodels dependency) ────────────────────────────

def bh_correction(p_values: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (reject, q_values) using Benjamini-Hochberg FDR correction."""
    n = len(p_values)
    if n == 0:
        return np.array([], dtype=bool), np.array([])
    order = np.argsort(p_values)
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n + 1)
    q_values = np.minimum(1.0, p_values * n / ranks)
    # Enforce monotonicity right-to-left so q-values are non-decreasing in rank
    for i in range(n - 2, -1, -1):
        if q_values[order[i]] > q_values[order[i + 1]]:
            q_values[order[i]] = q_values[order[i + 1]]
    return q_values <= alpha, q_values


# ── Method 1 — weighted centroid ─────────────────────────────────────────────

def method_centroid(imp: pd.DataFrame, conditions: list[str], n_layers: int) -> None:
    hdr("METHOD 1 — Importance-Weighted Layer Centroid")
    print(
        "\n  The centroid is the importance-weighted mean layer for a condition.\n"
        "  Higher value → activity concentrated in later (deeper) layers.\n"
        f"  Network has {n_layers} layers (0-indexed: 0 – {n_layers - 1}).\n"
    )

    records = []
    for cond in conditions:
        sub_df = imp[imp["condition"] == cond]
        if sub_df.empty:
            continue
        total = sub_df["importance_score"].sum()
        if total == 0:
            continue
        centroid = float((sub_df["layer"] * sub_df["importance_score"]).sum() / total)
        pct = centroid / (n_layers - 1) * 100
        records.append({
            "condition": cond,
            "centroid": centroid,
            "pct_depth": pct,
            "n_rows": len(sub_df),
            "total_importance": total,
        })

    if not records:
        print("  No data.")
        return

    df_out = pd.DataFrame(records).sort_values("centroid", ascending=False)

    print(f"  {'Condition':<24}  {'Centroid':>10}  {'% depth':>9}  {'N neuron-domain rows':>22}")
    print(f"  {'-'*24}  {'-'*10}  {'-'*9}  {'-'*22}")
    for _, row in df_out.iterrows():
        print(
            f"  {row['condition']:<24}  {row['centroid']:>10.3f}  "
            f"{row['pct_depth']:>8.1f}%  {int(row['n_rows']):>22,}"
        )

    # Highlight CS vs confused gap
    cs_rows   = df_out[df_out["condition"] == conditions[0]]
    conf_rows = df_out[df_out["condition"] == conditions[1]] if len(conditions) > 1 else pd.DataFrame()
    if not cs_rows.empty and not conf_rows.empty:
        diff = float(cs_rows["centroid"].iloc[0]) - float(conf_rows["centroid"].iloc[0])
        direction = (
            "CS activity sits deeper in the network"   if diff >  0.5 else
            "confused activity sits deeper"             if diff < -0.5 else
            "both conditions have similar layer depth"
        )
        print(f"\n  CS centroid − confused centroid = {diff:+.3f} layers  →  {direction}")


# ── Method 2 — per-layer Mann-Whitney U ──────────────────────────────────────

def method_mannwhitney(
    imp: pd.DataFrame, cs_label: str, confused_label: str, alpha: float
) -> None:
    hdr("METHOD 2 — Per-Layer Mann-Whitney U  (BH-FDR corrected)")
    print(
        "\n  For each layer: are CS importance scores stochastically greater\n"
        "  than confused importance scores?\n"
        "  Effect size = rank-biserial correlation r\n"
        "    r = +1  →  CS always higher\n"
        "    r = −1  →  confused always higher\n"
        "    r =  0  →  no difference\n"
        f"  Significance threshold: q < {alpha}  (Benjamini-Hochberg FDR)\n"
    )

    cs_df   = imp[imp["condition"] == cs_label]
    conf_df = imp[imp["condition"] == confused_label]
    layers  = sorted(imp["layer"].unique())

    rows = []
    for layer in layers:
        cs_scores   = cs_df.loc[cs_df["layer"] == layer, "importance_score"].values
        conf_scores = conf_df.loc[conf_df["layer"] == layer, "importance_score"].values
        if len(cs_scores) < 3 or len(conf_scores) < 3:
            continue
        U, p = mannwhitneyu(cs_scores, conf_scores, alternative="two-sided")
        n1, n2 = len(cs_scores), len(conf_scores)
        r = float(1.0 - (2.0 * U) / (n1 * n2))   # rank-biserial correlation
        rows.append({"layer": layer, "U": U, "p_raw": p, "r": r,
                     "n_cs": n1, "n_conf": n2})

    if not rows:
        print("  Insufficient data for Mann-Whitney test.")
        return

    result = pd.DataFrame(rows).sort_values("layer")
    reject, q_vals = bh_correction(result["p_raw"].values, alpha)
    result["q_bh"]        = q_vals
    result["significant"] = reject

    n_sig = int(result["significant"].sum())
    print(f"  Layers tested: {len(result)}   Significant after FDR correction: {n_sig}\n")

    print(f"  {'Layer':>6}  {'n_CS':>7}  {'n_conf':>7}  {'p_raw':>10}  "
          f"{'q_BH':>10}  {'r (RBC)':>9}  sig")
    print(f"  {'-'*6}  {'-'*7}  {'-'*7}  {'-'*10}  {'-'*10}  {'-'*9}  ---")

    for _, row in result.iterrows():
        marker = " ◀" if row["significant"] else ""
        print(
            f"  {int(row['layer']):>6}  {int(row['n_cs']):>7,}  {int(row['n_conf']):>7,}  "
            f"{row['p_raw']:>10.4f}  {row['q_bh']:>10.4f}  {row['r']:>+9.3f}{marker}"
        )

    sig_df = result[result["significant"]].sort_values("r", key=abs, ascending=False)
    if not sig_df.empty:
        print(f"\n  Significant layers: {sorted(sig_df['layer'].astype(int).tolist())}")
        print(f"  Top effects:")
        for _, row in sig_df.head(5).iterrows():
            direction = "CS > confused" if row["r"] > 0 else "confused > CS"
            print(f"    Layer {int(row['layer'])}: r = {row['r']:+.3f}  ({direction},  q = {row['q_bh']:.4f})")
    else:
        print(f"\n  No layers survive FDR correction at q < {alpha}.")
        best = result.nsmallest(3, "p_raw")
        print(f"  Nominally smallest p-values (uncorrected, for reference):")
        for _, row in best.iterrows():
            print(f"    Layer {int(row['layer'])}: p = {row['p_raw']:.4f},  r = {row['r']:+.3f}")


# ── Method 3 — chi-squared enrichment ────────────────────────────────────────

def method_chisq(
    consistent: pd.DataFrame, cs_label: str, confused_label: str, n_layers: int
) -> None:
    hdr("METHOD 3 — Chi-Squared Layer Enrichment")
    print(
        "\n  Tests whether condition-specific consistent neurons are non-uniformly\n"
        "  distributed across layers (null = uniform).\n"
        "  Standardised residuals: >+2 = enriched,  <−2 = depleted.\n"
    )

    all_layers = list(range(n_layers))

    for cond_label in [cs_label, confused_label]:
        sub(f"Condition: {cond_label}")
        cond_df = consistent[
            (consistent["condition"] == cond_label) & consistent["passes_consistency"]
        ]
        if cond_df.empty:
            print("  No consistent neurons found for this condition.")
            continue

        counts_series = (
            cond_df.drop_duplicates(subset=["layer", "neuron"])["layer"]
            .value_counts()
            .sort_index()
        )
        observed = np.array([counts_series.get(l, 0) for l in all_layers], dtype=float)
        total = observed.sum()

        if total < 5:
            print(f"  Only {int(total)} consistent neurons — too few for a reliable test.")
            continue

        expected  = np.full(n_layers, total / n_layers)
        chi2, p   = chisquare(observed, f_exp=expected)
        std_resid = (observed - expected) / np.sqrt(expected)

        print(f"\n  Total consistent neurons : {int(total)}")
        print(f"  χ²(df={n_layers-1}) = {chi2:.2f},  p = {p:.4f}  {sig_stars(p).strip()}")

        print(f"\n  {'Layer':>6}  {'Observed':>10}  {'Expected':>10}  {'Std residual':>13}  Note")
        print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*13}  ----")
        for i, layer in enumerate(all_layers):
            note = ""
            if   std_resid[i] >  2: note = "▲ enriched"
            elif std_resid[i] < -2: note = "▼ depleted"
            print(
                f"  {layer:>6}  {int(observed[i]):>10}  {expected[i]:>10.1f}  "
                f"{std_resid[i]:>13.2f}  {note}"
            )

        enriched = [all_layers[i] for i in range(n_layers) if std_resid[i] >  2]
        depleted = [all_layers[i] for i in range(n_layers) if std_resid[i] < -2]
        print()
        if enriched:
            print(f"  ▲ Enriched layers (residual > +2): {enriched}")
        if depleted:
            print(f"  ▼ Depleted layers (residual < −2): {depleted}")
        if not enriched and not depleted:
            print(f"  No layers show large standardised residuals (|residual| > 2).")


# ── Method 4 — CDF + bootstrapped median layer ───────────────────────────────

def method_cdf(
    imp: pd.DataFrame,
    conditions: list[str],
    n_layers: int,
    n_bootstrap: int,
    seed: int,
) -> None:
    hdr("METHOD 4 — Cumulative Importance CDF  (bootstrapped median layer)")
    print(
        "\n  The 'median layer' is where cumulative importance first crosses 50% —\n"
        "  analogous to median survival time in a survival curve.\n"
        "  90% bootstrap CI built from resampling neuron-domain rows.\n"
    )

    rng = np.random.default_rng(seed)
    all_layers = list(range(n_layers))

    def _median_layer(layers_arr: np.ndarray, scores_arr: np.ndarray) -> float:
        order  = np.argsort(layers_arr, kind="stable")
        l_sort = layers_arr[order]
        s_sort = scores_arr[order]
        cum    = np.cumsum(s_sort)
        total  = cum[-1]
        if total == 0:
            return float(l_sort[0])
        idx = min(int(np.searchsorted(cum, total * 0.5)), len(l_sort) - 1)
        return float(l_sort[idx])

    def _bootstrap_ci(layers: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
        medians = np.empty(n_bootstrap)
        for i in range(n_bootstrap):
            idx = rng.integers(0, len(layers), size=len(layers))
            medians[i] = _median_layer(layers[idx], scores[idx])
        return float(np.percentile(medians, 5)), float(np.percentile(medians, 95))

    def _depth_label(median: float) -> str:
        pct = median / max(n_layers - 1, 1) * 100
        if   pct < 33: return "early layers"
        elif pct < 55: return "middle layers"
        elif pct < 75: return "upper-middle layers"
        else:          return "late layers"

    print(f"  {'Condition':<24}  {'Median layer':>13}  {'90% CI':>16}  {'% depth':>9}  Interpretation")
    print(f"  {'-'*24}  {'-'*13}  {'-'*16}  {'-'*9}  {'-'*20}")

    cdf_data: dict[str, np.ndarray] = {}

    for cond in conditions:
        sub_df = imp[imp["condition"] == cond]
        if sub_df.empty:
            continue
        layers_arr = sub_df["layer"].values.astype(float)
        scores_arr = sub_df["importance_score"].values.astype(float)

        median   = _median_layer(layers_arr, scores_arr)
        lo, hi   = _bootstrap_ci(layers_arr, scores_arr)
        pct      = median / max(n_layers - 1, 1) * 100
        label    = _depth_label(median)

        print(
            f"  {cond:<24}  {median:>13.1f}  "
            f"[{lo:>5.1f}, {hi:>5.1f}]  {pct:>8.1f}%  {label}"
        )

        layer_totals = sub_df.groupby("layer")["importance_score"].sum().reindex(all_layers, fill_value=0.0)
        cum   = layer_totals.cumsum().values
        total = cum[-1]
        if total > 0:
            cdf_data[cond] = cum / total

    # Full CDF table
    if cdf_data:
        conds_listed = list(cdf_data.keys())
        col_w = 22
        print(f"\n  Cumulative importance by layer (█ = 10%):\n")
        header = f"  {'Layer':>6}" + "".join(f"  {c[:col_w]:<{col_w}}" for c in conds_listed)
        print(header)
        print("  " + "-"*6 + ("  " + "-"*col_w) * len(conds_listed))

        for layer in all_layers:
            row_str = f"  {layer:>6}"
            for cond in conds_listed:
                val  = float(cdf_data[cond][layer]) if layer < len(cdf_data[cond]) else 1.0
                bar  = "█" * int(val * 10)
                cell = f"{val:5.1%}  {bar}"
                row_str += f"  {cell:<{col_w}}"
            print(row_str)


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(
    imp: pd.DataFrame,
    consistent: pd.DataFrame,
    cs_label: str,
    confused_label: str,
    n_layers: int,
) -> None:
    hdr("SUMMARY")

    def centroid(cond: str) -> float | None:
        s = imp[imp["condition"] == cond]
        tot = s["importance_score"].sum()
        if tot == 0:
            return None
        return float((s["layer"] * s["importance_score"]).sum() / tot)

    cs_c   = centroid(cs_label)
    conf_c = centroid(confused_label)

    print(f"\n  Network depth: {n_layers} layers  (0 = earliest, {n_layers-1} = latest)\n")

    if cs_c is not None:
        print(f"  CS centroid       : layer {cs_c:.2f}  ({cs_c/(n_layers-1)*100:.0f}% through the network)")
    if conf_c is not None:
        print(f"  Confused centroid : layer {conf_c:.2f}  ({conf_c/(n_layers-1)*100:.0f}% through the network)")
    if cs_c is not None and conf_c is not None:
        diff = cs_c - conf_c
        print(f"  Difference        : {diff:+.2f} layers  ", end="")
        if   diff >  0.5: print("→  CS activity sits deeper")
        elif diff < -0.5: print("→  confused activity sits deeper")
        else:             print("→  similar depth for both conditions")

    for cond_label, label in [(cs_label, "CS"), (confused_label, "confused")]:
        cond_df = consistent[
            (consistent["condition"] == cond_label) & consistent["passes_consistency"]
        ]
        if cond_df.empty:
            continue
        top = (
            cond_df.drop_duplicates(["layer", "neuron"])
            .groupby("layer").size()
            .nlargest(5)
            .sort_index()
        )
        print(f"\n  {label}-specific consistent neurons — top layers by count:")
        for layer, count in top.items():
            bar = "█" * count
            print(f"    Layer {int(layer):>3}:  {count:>4}  {bar}")

    print()


# ── entry point ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Statistical layer localisation for condition-specific neurons.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--results_dir", required=True,
        help="Directory produced by compute_condition_specific_neurons.py",
    )
    p.add_argument("--cs_label",       default="code_switched")
    p.add_argument("--confused_label", default="confused")
    p.add_argument("--english_label",  default="english")
    p.add_argument("--target_label",   default="target_language")
    p.add_argument(
        "--fdr_alpha", type=float, default=0.05,
        help="FDR significance threshold for method 2 (default: 0.05).",
    )
    p.add_argument(
        "--n_bootstrap", type=int, default=2000,
        help="Bootstrap iterations for method 4 median CI (default: 2000).",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)

    imp_path        = results_dir / "neuron_importance_condition_domain.csv.gz"
    consistent_path = results_dir / "consistent_neurons.csv.gz"

    for path in [imp_path, consistent_path]:
        if not path.exists():
            print(f"ERROR: required file not found: {path}", file=sys.stderr)
            sys.exit(1)

    print(f"Loading {imp_path.name} ...")
    imp = pd.read_csv(imp_path, compression="gzip")

    print(f"Loading {consistent_path.name} ...")
    consistent = pd.read_csv(consistent_path, compression="gzip")

    n_layers           = int(imp["layer"].max()) + 1
    conditions_present = sorted(imp["condition"].unique().tolist())

    print(
        f"\n  results_dir : {results_dir}\n"
        f"  layers      : {n_layers}  (0 – {n_layers - 1})\n"
        f"  conditions  : {conditions_present}\n"
        f"  imp rows    : {len(imp):,}\n"
        f"  consistent  : {len(consistent):,}"
    )

    ordered = [args.cs_label, args.confused_label, args.english_label, args.target_label]
    conditions = [c for c in ordered if c in conditions_present]

    method_centroid(imp, conditions, n_layers)
    method_mannwhitney(imp, args.cs_label, args.confused_label, args.fdr_alpha)
    method_chisq(consistent, args.cs_label, args.confused_label, n_layers)
    method_cdf(imp, conditions, n_layers, args.n_bootstrap, args.seed)
    print_summary(imp, consistent, args.cs_label, args.confused_label, n_layers)

    print(f"{'═' * W}\n  Done.\n{'═' * W}\n")


if __name__ == "__main__":
    main()
