"""Run multiscale core-periphery analysis when location sequences are available."""

from __future__ import annotations

import argparse

from _common import add_common_args, dataset_graph_dir, dataset_intermediate_dir, selected_datasets


def main() -> None:
    """Run the container/core-periphery workflow or write a skip log."""
    parser = add_common_args(argparse.ArgumentParser())
    args = parser.parse_args()

    for dataset in selected_datasets(args.dataset):
        graph_dir = dataset_graph_dir(dataset)
        loc_seq_dir = graph_dir.parent / "location_sequence"
        output_dir = dataset_intermediate_dir(dataset) / "core_periphery"

        if not loc_seq_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
            log_path = output_dir / "SKIPPED.txt"
            log_path.write_text(
                "Multiscale core-periphery analysis requires L_*.csv location-sequence files. "
                "The public example workflow starts from IMN/G_*.json, so this step is skipped.\n",
                encoding="utf-8",
            )
            print(f"Skipped multiscale analysis for {dataset}; see {log_path}")
            continue

        from paper_analysis.multiscale.container_core_periphery import (
            add_module_unique_id,
            community_core_periphery_stats,
            filter_container,
            rename_df,
        )

        distance_metric = "euclidean" if dataset == "D1_YJMob100K" else "haversine"
        community_core_periphery_stats(
            str(loc_seq_dir),
            str(graph_dir),
            str(output_dir),
            distance_metric=distance_metric,
            min_nodes=2,
            min_edges=1,
            cp_method="LipWeighted",
            significance_level=0.05,
            rg_weight="visit_count",
            avg_dist_weight="movement_count",
        )
        stat_file = output_dir / "container_core_periphery_equal_weight_stat.csv"
        rename_df(str(stat_file))
        add_module_unique_id(str(stat_file))
        filter_container(
            str(stat_file),
            attr_1="module_size",
            attr_2="average_degree",
            attr_threshold_1=10,
            attr_threshold_2=2,
        )


if __name__ == "__main__":
    main()
