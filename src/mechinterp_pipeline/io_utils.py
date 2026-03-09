from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml


REQUIRED_COLUMNS = {"id", "text", "condition", "domain"}


@dataclass
class PipelineConfig:
    model_name: str
    input_path: Path
    output_dir: Path
    device: str = "auto"
    max_length: int = 256
    batch_size: int = 4
    tuned_lens_resource_id: str | None = None
    topk_neurons: int = 20
    reference_condition: str | None = None
    log_level: str = "INFO"
    log_every_n_samples: int = 1
    save_full_neuron_activations: bool = False
    full_neuron_reduce_mode: str = "mean_abs"
    full_neuron_export_gzip: bool = True
    full_neuron_round_decimals: int = 4
    full_neuron_layers: list[int] | None = None
    full_neuron_min_layer_exclusive: int | None = None
    full_neuron_topk_per_layer: int = 0
    full_neuron_sample_stride: int = 1



def load_config(config_path: str | Path) -> PipelineConfig:
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    missing = [k for k in ["model_name", "input_path", "output_dir"] if k not in raw]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")

    return PipelineConfig(
        model_name=raw["model_name"],
        input_path=Path(raw["input_path"]),
        output_dir=Path(raw["output_dir"]),
        device=raw.get("device", "auto"),
        max_length=int(raw.get("max_length", 256)),
        batch_size=int(raw.get("batch_size", 4)),
        tuned_lens_resource_id=raw.get("tuned_lens_resource_id"),
        topk_neurons=int(raw.get("topk_neurons", 20)),
        reference_condition=raw.get("reference_condition"),
        log_level=str(raw.get("log_level", "INFO")),
        log_every_n_samples=int(raw.get("log_every_n_samples", 1)),
        save_full_neuron_activations=bool(raw.get("save_full_neuron_activations", False)),
        full_neuron_reduce_mode=str(raw.get("full_neuron_reduce_mode", "mean_abs")),
        full_neuron_export_gzip=bool(raw.get("full_neuron_export_gzip", True)),
        full_neuron_round_decimals=int(raw.get("full_neuron_round_decimals", 4)),
        full_neuron_layers=(
            [int(x) for x in raw.get("full_neuron_layers", [])]
            if raw.get("full_neuron_layers", None) is not None
            else None
        ),
        full_neuron_min_layer_exclusive=(
            int(raw["full_neuron_min_layer_exclusive"])
            if raw.get("full_neuron_min_layer_exclusive", None) is not None
            else None
        ),
        full_neuron_topk_per_layer=int(raw.get("full_neuron_topk_per_layer", 0)),
        full_neuron_sample_stride=max(1, int(raw.get("full_neuron_sample_stride", 1))),
    )



def ensure_dirs(paths: Iterable[Path]) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)



def load_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() in {".jsonl", ".json"}:
        df = pd.read_json(path, lines=path.suffix.lower() == ".jsonl")
    else:
        raise ValueError("Input format must be .csv, .json, or .jsonl")

    missing_cols = REQUIRED_COLUMNS.difference(df.columns)
    if missing_cols:
        raise ValueError(f"Input missing required columns: {sorted(missing_cols)}")

    if df["condition"].nunique() < 2:
        raise ValueError("Need at least two condition labels (e.g., code_switched and confused)")

    return df
