"""Run compressibility and random-walk entropy analysis for filtered modules."""

from __future__ import annotations

import argparse

from _common import add_common_args, dataset_graph_dir, dataset_intermediate_dir, selected_datasets
from paper_analysis.compressibility.core_periphery_compressibility import compressibility_stat


def main() -> None:
    """Run entropy-enabled compressibility when filtered module tables exist."""
    parser = add_common_args(argparse.ArgumentParser())
    args = parser.parse_args()

    for dataset in selected_datasets(args.dataset):
        graph_dir = dataset_graph_dir(dataset)
        cp_dir = dataset_intermediate_dir(dataset) / "core_periphery"
        container_file = cp_dir / "container_core_periphery_equal_weight_stat_filtered.csv"

        if not container_file.exists():
            cp_dir.mkdir(parents=True, exist_ok=True)
            log_path = cp_dir / "compressibility_SKIPPED.txt"
            log_path.write_text(
                "Compressibility requires container_core_periphery_equal_weight_stat_filtered.csv. "
                "Run script 02 with location-sequence files before this step.\n",
                encoding="utf-8",
            )
            print(f"Skipped compressibility for {dataset}; see {log_path}")
            continue

        compressibility_stat(
            GRAPH_PATH=str(graph_dir),
            CONTAINER_FILE=str(container_file),
            OUTPUT_PATH=str(cp_dir),
            weight="movement_count",
            random_models=("er",),
            num_of_rand_net=10,
            timeout_per_graph=1200,
            use_parallel=False,
        )


if __name__ == "__main__":
    main()

