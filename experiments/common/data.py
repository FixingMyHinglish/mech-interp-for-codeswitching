from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {"id", "text", "condition", "domain"}


def derive_source_id(row_id: str, condition: str) -> str:
    suffix = f"_{condition}"
    if row_id.endswith(suffix):
        return row_id[: -len(suffix)]
    return row_id


def load_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix == ".jsonl":
        df = pd.read_json(path, lines=True)
    elif suffix == ".json":
        df = pd.read_json(path)
    else:
        raise ValueError("Input must be .csv, .jsonl, or .json")

    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")

    out = df.copy()
    out["id"] = out["id"].astype(str)
    out["text"] = out["text"].astype(str)
    out["condition"] = out["condition"].astype(str)
    out["domain"] = out["domain"].astype(str)
    if "source_id" not in out.columns:
        out["source_id"] = [
            derive_source_id(row_id, cond)
            for row_id, cond in zip(out["id"], out["condition"])
        ]
    else:
        out["source_id"] = out["source_id"].astype(str)
    return out

