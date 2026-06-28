"""Run topological feature extraction from public IMN graph files."""

from __future__ import annotations

import argparse

from _common import (
    add_common_args,
    dataset_graph_dir,
    dataset_intermediate_dir,
    dataset_location_sequence_dir,
    selected_datasets,
)
from paper_analysis.features.spatial_features import calculate_features
from paper_analysis.features.topological_features import basic_stats


def main() -> None:
    """Compute topological features for selected public datasets."""
    parser = add_common_args(argparse.ArgumentParser())
    args = parser.parse_args()

    for dataset in selected_datasets(args.dataset):
        graph_dir = dataset_graph_dir(dataset)
        output_dir = dataset_intermediate_dir(dataset) / "topological_features"
        if not graph_dir.exists():
            raise FileNotFoundError(f"Missing graph directory: {graph_dir}")
        print(f"Running topological features for {dataset}")
        basic_stats(str(graph_dir), str(output_dir), num_processes=1)

        loc_seq_dir = dataset_location_sequence_dir(dataset)
        if not loc_seq_dir.exists():
            log_path = dataset_intermediate_dir(dataset) / "spatial_features_skipped.txt"
            log_path.write_text(
                "Spatial features require L_*.csv location-sequence files. "
                "The public example workflow starts from IMN/G_*.json, so this step is skipped.\n",
                encoding="utf-8",
            )
            print(f"Skipped spatial features for {dataset}; see {log_path}")
            continue

        spatial_dir = dataset_intermediate_dir(dataset) / "spatial_features"
        distance_metric = "euclidean" if dataset == "D1_YJMob100K" else "haversine"
        print(f"Running spatial features for {dataset}")
        calculate_features(
            LOCATION_SEQUENCE_DIR=str(loc_seq_dir),
            GRAPH_DIR=str(graph_dir),
            FEATURES_DIR=str(spatial_dir),
            distance_metric=distance_metric,
            num_processes=1,
        )


if __name__ == "__main__":
    main()
