from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ComparisonConfig:
    name: str
    focus_condition: str
    baseline_condition: str


@dataclass(frozen=True)
class PhenomenonConfig:
    name: str
    focus_condition: str
    required_baselines: tuple[str, ...]
    top_k: int = 50


@dataclass(frozen=True)
class RunConfig:
    model_name: str
    input_path: Path
    output_dir: Path
    device: str = "auto"
    max_length: int = 256
    log_every: int = 25
    alpha: float = 0.05
    min_pairs: int = 8
    chunk_size: int = 512
    max_groups: int | None = None
    token_offsets: tuple[int, ...] = (-1, 0, 1)
    target_mode: str = "predicted_next_token"
    comparisons: tuple[ComparisonConfig, ...] = ()
    phenomena: tuple[PhenomenonConfig, ...] = ()


def _default_comparisons() -> tuple[ComparisonConfig, ...]:
    return (
        ComparisonConfig("code_switched_vs_english", "code_switched", "english"),
        ComparisonConfig("code_switched_vs_target", "code_switched", "target_language"),
        ComparisonConfig("confused_vs_english", "confused", "english"),
        ComparisonConfig("confused_vs_target", "confused", "target_language"),
    )


def _default_phenomena() -> tuple[PhenomenonConfig, ...]:
    return (
        PhenomenonConfig("code_switched", "code_switched", ("english", "target_language"), 50),
        PhenomenonConfig("confused", "confused", ("english", "target_language"), 50),
    )


def load_config(path: str | Path) -> RunConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    missing = [key for key in ["model_name", "input_path", "output_dir"] if key not in raw]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")

    comparisons_raw = raw.get("comparisons") or []
    if comparisons_raw:
        comparisons = tuple(
            ComparisonConfig(
                name=str(item["name"]),
                focus_condition=str(item["focus_condition"]),
                baseline_condition=str(item["baseline_condition"]),
            )
            for item in comparisons_raw
        )
    else:
        comparisons = _default_comparisons()

    phenomena_raw = raw.get("phenomena") or []
    if phenomena_raw:
        phenomena = tuple(
            PhenomenonConfig(
                name=str(item["name"]),
                focus_condition=str(item["focus_condition"]),
                required_baselines=tuple(str(x) for x in item["required_baselines"]),
                top_k=int(item.get("top_k", 50)),
            )
            for item in phenomena_raw
        )
    else:
        phenomena = _default_phenomena()

    max_groups_raw = raw.get("max_groups")
    token_offsets = tuple(int(x) for x in raw.get("token_offsets", [-1, 0, 1]))
    if not token_offsets:
        raise ValueError("token_offsets must not be empty")

    target_mode = str(raw.get("target_mode", "predicted_next_token"))
    if target_mode not in {"predicted_next_token", "observed_next_token"}:
        raise ValueError("target_mode must be 'predicted_next_token' or 'observed_next_token'")

    return RunConfig(
        model_name=str(raw["model_name"]),
        input_path=Path(raw["input_path"]),
        output_dir=Path(raw["output_dir"]),
        device=str(raw.get("device", "auto")),
        max_length=int(raw.get("max_length", 256)),
        log_every=max(1, int(raw.get("log_every", 25))),
        alpha=float(raw.get("alpha", 0.05)),
        min_pairs=max(2, int(raw.get("min_pairs", 8))),
        chunk_size=max(16, int(raw.get("chunk_size", 512))),
        max_groups=None if max_groups_raw in (None, "") else max(1, int(max_groups_raw)),
        token_offsets=token_offsets,
        target_mode=target_mode,
        comparisons=comparisons,
        phenomena=phenomena,
    )
