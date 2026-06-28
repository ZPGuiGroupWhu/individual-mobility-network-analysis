"""Check public IMN example data and write a manifest."""

from __future__ import annotations

import argparse

from _common import add_common_args, dataset_graph_dir, selected_datasets, write_json, RELEASE_ROOT


def build_manifest(dataset: str) -> dict:
    """Return file counts and relative file names for one public dataset."""
    graph_dir = dataset_graph_dir(dataset)
    files = sorted(graph_dir.glob("G_*.json"))
    return {
        "dataset": dataset,
        "graph_dir": str(graph_dir.relative_to(RELEASE_ROOT)),
        "graph_count": len(files),
        "files": [str(path.relative_to(RELEASE_ROOT)) for path in files],
    }


def main() -> None:
    """Validate public example data and write ``results/logs/data_manifest.json``."""
    parser = add_common_args(argparse.ArgumentParser())
    args = parser.parse_args()

    manifests = []
    for dataset in selected_datasets(args.dataset):
        manifest = build_manifest(dataset)
        if manifest["graph_count"] == 0:
            raise FileNotFoundError(f"No G_*.json files found for {dataset}")
        manifests.append(manifest)

    output = RELEASE_ROOT / "results" / "logs" / "data_manifest.json"
    write_json(output, manifests)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()

