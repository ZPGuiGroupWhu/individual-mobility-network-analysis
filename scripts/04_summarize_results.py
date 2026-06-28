"""Collect workflow outputs into the public results/tables directory."""

from __future__ import annotations

import argparse
import shutil

from _common import RELEASE_ROOT, add_common_args, dataset_intermediate_dir, selected_datasets
from paper_analysis.features.process_feature_data import process_spatial_features, process_topology_features


def main() -> None:
    """Copy generated CSV tables into ``results/tables`` with dataset prefixes."""
    parser = add_common_args(argparse.ArgumentParser())
    args = parser.parse_args()
    table_dir = RELEASE_ROOT / "results" / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for dataset in selected_datasets(args.dataset):
        root = dataset_intermediate_dir(dataset)

        spatial_dir = root / "spatial_features"
        if (spatial_dir / "spatial_stats.csv").exists():
            process_spatial_features(str(spatial_dir))

        topo_dir = root / "topological_features"
        if (topo_dir / "topological_stats.csv").exists():
            process_topology_features(str(topo_dir), 0.05)

        for csv_path in root.rglob("*.csv"):
            target = table_dir / f"{dataset}_{csv_path.name}"
            shutil.copy2(csv_path, target)
            copied += 1
    print(f"Copied {copied} table(s) to {table_dir}")


if __name__ == "__main__":
    main()
