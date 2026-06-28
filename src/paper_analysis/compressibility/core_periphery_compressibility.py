"""Compressibility and random-walk entropy analysis for core-periphery modules.

The default workflow requires ``methods/compressibility_numba_with_entropy.py``,
which replaces the earlier ``compressibility_numba.py`` implementation and
provides ``graph_rate_distortion_metrics``.
"""

import os
import numpy as np
import pandas as pd
import networkx as nx
from paper_analysis.utils.io import read_graph, log_error_jsonl

try:
    from paper_analysis.compressibility.methods.compressibility_numba_with_entropy import (
        graph_compressibility,
        graph_rate_distortion_metrics,
    )
except ModuleNotFoundError as exc:
    graph_compressibility = None
    graph_rate_distortion_metrics = None
    _MISSING_ENTROPY_MODULE_ERROR = exc
else:
    _MISSING_ENTROPY_MODULE_ERROR = None
from multiprocessing import Pool
import ast
import time
import json
from datetime import datetime
from multiprocessing import TimeoutError as MPTimeoutError
import shutil
from pathlib import Path


def generate_random_graph(
        G,
        model="er",
        weight_attr="movement_count",
        seed=None
):
    """
    Generate a randomized version of G.

    Parameters
    ----------
    model : str
        - "er"              : Erdos-Renyi (same n, m)
        - "er_weight"       : ER + shuffled weights
        - "shuffle_weight"  : same edges, shuffled weights
        - "configuration"   : degree-preserving (unweighted)
    """
    rng = np.random.default_rng(seed)
    nodes = list(G.nodes())
    n = G.number_of_nodes()
    m = G.number_of_edges()

    if model == "er":
        Gr = nx.gnm_random_graph(n, m, seed=seed)
        return nx.relabel_nodes(Gr, dict(zip(range(n), nodes)))

    if model == "shuffle_weight":
        Gr = G.copy()
        weights = np.array(
            [d.get(weight_attr, 1.0) for _, _, d in G.edges(data=True)]
        )
        rng.shuffle(weights)
        for (u, v), w in zip(Gr.edges(), weights):
            Gr[u][v][weight_attr] = w
        return Gr

    if model == "er_weight":
        Gr = nx.gnm_random_graph(n, m, seed=seed)
        Gr = nx.relabel_nodes(Gr, dict(zip(range(n), nodes)))
        weights = np.array(
            [d.get(weight_attr, 1.0) for _, _, d in G.edges(data=True)]
        )
        rng.shuffle(weights)
        for (u, v), w in zip(Gr.edges(), weights):
            Gr[u][v][weight_attr] = w
        return Gr

    if model == "configuration":
        deg = [d for _, d in G.degree()]
        Gr = nx.configuration_model(deg, seed=seed)
        Gr = nx.Graph(Gr)
        Gr = nx.relabel_nodes(Gr, dict(zip(range(n), nodes)))
        return Gr

    raise ValueError(f"Unknown random model: {model}")


def graph_compressibility_stat_individual(
        G_file,
        container_df,
        weight='movement_count',
        random_models=("er",),
        num_of_rand_net=10,
        seed=42
):
    if graph_rate_distortion_metrics is None:
        raise ImportError(
            "Missing paper_analysis.compressibility.methods.compressibility_numba_with_entropy. "
            "Copy the entropy-enabled compressibility implementation into this package before "
            "running the compressibility workflow."
        ) from _MISSING_ENTROPY_MODULE_ERROR

    try:
        G = read_graph(G_file)
        user_id = G.graph['user_id']
        print(user_id)

        container = container_df[
            container_df['user_id'] == user_id
        ].copy()

        results_mean = {m: [] for m in random_models}
        results_std = {m: [] for m in random_models}
        entropy_mean = {m: [] for m in random_models}
        entropy_std = {m: [] for m in random_models}
        C_real = []
        H_real = []

        for node_list in container['module_node_list']:

            if not isinstance(node_list, (list, tuple)):
                C_real.append(np.nan)
                H_real.append(np.nan)
                for model in random_models:
                    results_mean[model].append(np.nan)
                    results_std[model].append(np.nan)
                    entropy_mean[model].append(np.nan)
                    entropy_std[model].append(np.nan)
                continue

            subG = G.subgraph(node_list).copy()

            metrics = graph_rate_distortion_metrics(
                subG,
                weight=weight,
                mode="weighted"
            )
            C_real.append(metrics["compressibility"])
            H_real.append(metrics["random_walk_entropy"])

            for model in random_models:
                Cr = []
                Hr = []

                for i in range(num_of_rand_net):
                    Gr = generate_random_graph(
                        subG,
                        model=model,
                        weight_attr=weight,
                        seed=seed + i
                    )
                    rand_metrics = graph_rate_distortion_metrics(
                        Gr,
                        weight=None if "weight" not in model else weight,
                        mode="weighted"
                    )
                    Cr.append(rand_metrics["compressibility"])
                    Hr.append(rand_metrics["random_walk_entropy"])

                Cr = np.asarray(Cr, dtype=float)
                Hr = np.asarray(Hr, dtype=float)
                results_mean[model].append(np.nanmean(Cr))
                results_std[model].append(np.nanstd(Cr, ddof=1))
                entropy_mean[model].append(np.nanmean(Hr))
                entropy_std[model].append(np.nanstd(Hr, ddof=1))

        container["C"] = C_real
        container["random_walk_entropy"] = H_real

        for model in random_models:
            container[f"C_{model}_mean"] = results_mean[model]
            container[f"C_{model}_std"] = results_std[model]
            container[f"random_walk_entropy_{model}_mean"] = entropy_mean[model]
            container[f"random_walk_entropy_{model}_std"] = entropy_std[model]

        return container

    except Exception as e:

        error_info = {
            "time": datetime.now().isoformat(),
            "G_file": G_file,
            "error_type": type(e).__name__,
            "error_message": str(e)
        }

        print(f"[ERROR] {G_file}: {e}")

        log_error_jsonl(error_info, "./logs/compressibility_errors_1.jsonl")

        return pd.DataFrame()


def parse_node_list(x):
    if isinstance(x, list):
        return x
    if pd.isna(x):
        return []
    try:
        return ast.literal_eval(x)
    except Exception:
        return []


def compressibility_stat(
        GRAPH_PATH,
        CONTAINER_FILE,
        OUTPUT_PATH,
        weight='movement_count',
        random_models=("er",),
        num_of_rand_net=10,
        timeout_per_graph=600,
        use_parallel=True,
        num_processes=None
):
    if graph_rate_distortion_metrics is None:
        raise ImportError(
            "Missing paper_analysis.compressibility.methods.compressibility_numba_with_entropy. "
            "The default workflow requires the entropy-enabled replacement for "
            "compressibility_numba.py."
        ) from _MISSING_ENTROPY_MODULE_ERROR

    container_df = pd.read_csv(
        CONTAINER_FILE,
        usecols=['user_id', 'module_unique_id', 'module_node_list'],
        dtype={
            'user_id': 'int64',
            'module_unique_id': 'int64'
        }
    )
    container_df['module_node_list'] = container_df['module_node_list'].apply(parse_node_list)
    print('read container df is ok')

    os.makedirs(OUTPUT_PATH, exist_ok=True)

    file_paths = [
        os.path.join(root, f)
        for root, _, files in os.walk(GRAPH_PATH)
        for f in files
    ]

    stats_list = []
    if use_parallel:
        if num_processes is None:
            pool_size = max(1, (os.cpu_count() or 1) - 4)
        else:
            pool_size = max(1, int(num_processes))

        with Pool(processes=pool_size) as pool:
            async_results = [
                pool.apply_async(
                    graph_compressibility_stat_individual,
                    args=(fp, container_df, weight, random_models, num_of_rand_net)
                )
                for fp in file_paths
            ]

            for fp, ar in zip(file_paths, async_results):
                try:
                    res = ar.get(timeout=timeout_per_graph)

                    if isinstance(res, pd.DataFrame) and not res.empty:
                        stats_list.append(res)
                    else:
                        print(f"[SKIP] Empty result: {fp}")

                except MPTimeoutError:
                    error_info = {
                        "time": datetime.now().isoformat(),
                        "G_file": fp,
                        "error_type": "TimeoutError",
                        "error_message": f"Exceed {timeout_per_graph} seconds"
                    }
                    print(f"[TIMEOUT] {fp}")
                    log_error_jsonl(error_info, "./logs/compressibility_errors_1.jsonl")

                except Exception as e:
                    error_info = {
                        "time": datetime.now().isoformat(),
                        "G_file": fp,
                        "error_type": type(e).__name__,
                        "error_message": str(e)
                    }
                    print(f"[ERROR] {fp}: {e}")
                    log_error_jsonl(error_info, "./logs/compressibility_errors_1.jsonl")
    else:
        for fp in file_paths:
            try:
                res = graph_compressibility_stat_individual(
                    fp,
                    container_df,
                    weight=weight,
                    random_models=random_models,
                    num_of_rand_net=num_of_rand_net
                )

                if isinstance(res, pd.DataFrame) and not res.empty:
                    stats_list.append(res)
                else:
                    print(f"[SKIP] Empty result: {fp}")

            except Exception as e:
                error_info = {
                    "time": datetime.now().isoformat(),
                    "G_file": fp,
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }
                print(f"[ERROR] {fp}: {e}")
                log_error_jsonl(error_info, "./logs/compressibility_errors_1.jsonl")

    stats_list = [df for df in stats_list if not df.empty]

    if len(stats_list) == 0:
        print("[WARNING] No valid core-periphery results.")
        return pd.DataFrame()

    stats_df = pd.concat(stats_list, ignore_index=True)

    stats_df.to_csv(
        os.path.join(OUTPUT_PATH, "compressibility_stats_with_entropy.csv"),
        index=False
    )

    return stats_df


def copy_failed_files(
    jsonl_path,
    original_path,
    target_path,
    keyword="dataset_VehicleTripleg"
):
    """
    Parse failed files from jsonl logs and copy the matching files to target_path.
    """
    jsonl_path = Path(jsonl_path)
    original_path = Path(original_path).resolve()
    target_path = Path(target_path)

    target_path.mkdir(parents=True, exist_ok=True)

    seen_files = set()
    stats = {
        "copied": 0,
        "missing": 0,
        "skipped_keyword": 0,
        "skipped_outside_root": 0,
        "duplicated": 0
    }

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            g_file = Path(record["G_file"])
            print(g_file)

            if keyword not in str(g_file):
                stats["skipped_keyword"] += 1
                continue

            try:
                g_file.resolve().relative_to(original_path)
            except ValueError:
                stats["skipped_outside_root"] += 1
                continue

            if g_file in seen_files:
                stats["duplicated"] += 1
                continue
            seen_files.add(g_file)

            if not g_file.exists():
                stats["missing"] += 1
                continue

            target_file = target_path / g_file.name
            if not target_file.exists():
                shutil.copy2(g_file, target_file)
                stats["copied"] += 1

    return stats
