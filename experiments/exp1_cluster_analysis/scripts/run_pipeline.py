#!/usr/bin/env python3
"""Run the full sparse-export pipeline for a given <work_dir>."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mech_interp.models.pipeline.convert_to_bundle import main as convert_main  # noqa: E402
from mech_interp.models.pipeline.make_neuron_summaries import main as summaries_main  # noqa: E402
from mech_interp.models.pipeline.cluster_analysis import main as cluster_main  # noqa: E402
from mech_interp.models.pipeline.delta_gap_analysis import main as gap_main  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--work_dir", type=Path, required=True)
    p.add_argument("--drop_condition", default="confused")
    p.add_argument("--topk_per_layer", type=int, default=256)
    p.add_argument("--write_filtered_jsonl", action="store_true")
    p.add_argument("--layer_num", type=int, default=None, help="Default: infer as max layer in 03_cleaned/")
    p.add_argument("--k", type=int, default=200)
    p.add_argument("--auto_k", action="store_true", help="Choose k by silhouette score over --k_min..--k_max.")
    p.add_argument("--k_min", type=int, default=2)
    p.add_argument("--k_max", type=int, default=10)
    p.add_argument("--ks", type=int, nargs="+", default=[10, 20, 30, 40, 50])
    return p.parse_args()


def _infer_layer_num(work_dir: Path) -> int:
    cleaned = work_dir / "03_cleaned"
    layers: list[int] = []
    for p in cleaned.glob("layer_*_neuron_summary.csv"):
        try:
            layers.append(int(p.name.split("_")[1]))
        except Exception:
            continue
    if not layers:
        raise RuntimeError(f"Could not infer layer_num from {cleaned}")
    return max(layers)


def main() -> None:
    args = parse_args()
    work_dir = args.work_dir.resolve()

    # Step 1: bundle
    sys.argv = [
        "convert_to_bundle.py",
        "--work_dir",
        str(work_dir),
        "--drop_condition",
        str(args.drop_condition),
        "--topk_per_layer",
        str(int(args.topk_per_layer)),
    ] + (["--write_filtered_jsonl"] if args.write_filtered_jsonl else [])
    convert_main()

    # Step 2: summaries
    sys.argv = ["make_neuron_summaries.py", "--work_dir", str(work_dir)]
    summaries_main()

    # Step 3: clustering
    layer_num = int(args.layer_num) if args.layer_num is not None else _infer_layer_num(work_dir)
    sys.argv = [
        "cluster_analysis.py",
        "--work_dir",
        str(work_dir),
        "--layer_num",
        str(layer_num),
    ]
    if bool(args.auto_k):
        sys.argv += ["--auto_k", "--k_min", str(int(args.k_min)), "--k_max", str(int(args.k_max))]
    else:
        sys.argv += ["--k", str(int(args.k))]
    cluster_main()

    # Step 4: gap analysis (+ README)
    sys.argv = [
        "delta_gap_analysis.py",
        "--work_dir",
        str(work_dir),
        "--layer_num",
        str(layer_num),
        "--ks",
        *[str(int(k)) for k in args.ks],
        "--write-readme",
    ]
    gap_main()


if __name__ == "__main__":
    main()

