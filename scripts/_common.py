"""Shared helpers for workflow scripts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


RELEASE_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = RELEASE_ROOT / "src"
DEFAULT_DATASETS = ("D1_YJMob100K", "D3_FourSuqare")

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add shared command-line arguments."""
    parser.add_argument(
        "--config",
        default=str(RELEASE_ROOT / "config" / "paths.example.yaml"),
        help="Path configuration file.",
    )
    parser.add_argument(
        "--analysis-config",
        default=str(RELEASE_ROOT / "config" / "analysis.yaml"),
        help="Analysis parameter configuration file.",
    )
    parser.add_argument(
        "--dataset",
        default="all",
        choices=(*DEFAULT_DATASETS, "all"),
        help="Dataset to process.",
    )
    return parser


def selected_datasets(dataset: str) -> list[str]:
    """Return selected public datasets."""
    if dataset == "all":
        return list(DEFAULT_DATASETS)
    return [dataset]


def dataset_graph_dir(dataset: str) -> Path:
    """Return the IMN graph directory for a dataset."""
    return RELEASE_ROOT / "data" / dataset / "IMN"


def dataset_location_sequence_dir(dataset: str) -> Path:
    """Return the location-sequence directory for a dataset."""
    return RELEASE_ROOT / "data" / dataset / "location_sequence"


def dataset_intermediate_dir(dataset: str) -> Path:
    """Return and create the intermediate output directory for a dataset."""
    path = RELEASE_ROOT / "results" / "intermediate" / dataset
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Any) -> None:
    """Write JSON with stable indentation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)

