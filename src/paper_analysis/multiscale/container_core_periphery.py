"""Multiscale community and core-periphery analysis for mobility networks.

This module combines the container model with module-level core-periphery
detection. It reads location sequences and IMN graph JSON files, then writes
``container_core_periphery_equal_weight_stat.csv`` and filtered module tables.
"""

import os
import numpy as np
import pandas as pd
import networkx as nx
import cpnet
from paper_analysis.utils.gislib import point_haversine_dist, point_euclidean_dist
from paper_analysis.utils.io import read_graph, print_graph, read_location_sequence
from multiprocessing import Pool
from paper_analysis.multiscale import container_community
from paper_analysis.multiscale.methods.core_periphery.LipWeight import LipWeighted
from paper_analysis.multiscale.methods.core_periphery.qtest import qtest
import json
import os
from datetime import datetime, timedelta
import random
import time


def _remove_duplicate_modules(df, keep_depth="max"):
    """
    Remove duplicate modules with identical node composition,
    keeping the one with either minimum or maximum depth.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing module information.
    keep_depth : {"min", "max"}, default "min"
        Whether to keep the module with the minimum or maximum depth
        among duplicates.

    Returns
    -------
    df : pandas.DataFrame
        Deduplicated DataFrame.
    """
    if keep_depth not in {"min", "max"}:
        raise ValueError("keep_depth must be either 'min' or 'max'")

    df = df.copy()
    df["node_tuple"] = df["node_list"].apply(lambda x: tuple(sorted(x)))

    ascending = True if keep_depth == "min" else False
    df = df.sort_values("depth", ascending=ascending)

    df = df.drop_duplicates(subset="node_tuple", keep="first")

    df = df.drop(columns="node_tuple")
    return df


def container_hierarchical_community(L, G, distance_metric='haversine'):
    df_comm = container_community.container_fit(L, distance_metric=distance_metric)
    df_comm = container_community.replace_location_id_with_node_id(df_comm, G)

    df_comm = _remove_duplicate_modules(df_comm)

    if df_comm.empty:
        raise RuntimeError("Empty container model result")

    return df_comm


# ==========================================================
# ---------------- Spatial Feature Computation -------------
# ==========================================================
def radius_of_gyration(G, distance_metric='haversine', weight='visit_count'):
    """
    Compute the radius of gyration (spatial dispersion) of a subgraph.

    Parameters
    ----------
    G : networkx.Graph
        Subgraph with node attribute 'loc' = (lon, lat).
    distance_metric : str, optional
        Type of distance ('euclidean' or 'haversine').
    weight : str or None
        Node attribute to use as weight (e.g., 'visit_count').

    Returns
    -------
    float
        Radius of gyration of the subgraph.
    """
    if G.number_of_nodes() < 2:
        return np.nan

    # Extract node coordinates
    coords = np.array([[G.nodes[n]['loc'][1], G.nodes[n]['loc'][0]] for n in G.nodes()])

    # Extract node weights if provided
    if weight is not None:
        weights = np.array(list(nx.get_node_attributes(G, name=weight, default=1).values()))
    else:
        weights = None

    # Compute center of mass (weighted or unweighted)
    if weights is not None:
        center_lon = np.average(coords[:, 0], weights=weights)
        center_lat = np.average(coords[:, 1], weights=weights)
    else:
        center_lon = np.mean(coords[:, 0])
        center_lat = np.mean(coords[:, 1])

    # Choose distance metric
    if distance_metric == 'euclidean':
        dist_func = point_euclidean_dist
    elif distance_metric == 'haversine':
        dist_func = point_haversine_dist
    else:
        raise ValueError(f"Unsupported distance metric: {distance_metric}")

    # Compute distances from each node to center
    distances = np.array([dist_func(lon, lat, center_lon, center_lat) for lon, lat in coords])

    # Compute radius of gyration
    if weights is not None:
        rg = np.sqrt(np.sum(weights * distances ** 2) / np.sum(weights))
    else:
        rg = np.sqrt(np.mean(distances ** 2))

    return rg


def average_distance(G, distance_metric='haversine', weight='movement_count'):
    """
    Compute the average spatial distance between connected nodes.

    Parameters
    ----------
    G : networkx.Graph
        Subgraph where nodes must contain 'loc' = (lat, lon).
    distance_metric : str, optional
        Type of distance ('haversine' or 'euclidean').
    weight : str, optional
        Edge weight attribute used as weighting factor.

    Returns
    -------
    float
        Weighted or unweighted average edge distance (km).
    """
    if G.number_of_edges() < 1:
        return np.nan

    # Choose distance metric
    if distance_metric == 'euclidean':
        dist_func = point_euclidean_dist
    elif distance_metric == 'haversine':
        dist_func = point_haversine_dist
    else:
        dist_func = point_euclidean_dist

    distances = []
    weights = []

    # Iterate over edges to compute distance
    for u, v, data in G.edges(data=True):
        lat1, lon1 = G.nodes[u]['loc']
        lat2, lon2 = G.nodes[v]['loc']
        dist = dist_func(lon1, lat1, lon2, lat2)
        distances.append(dist)
        if weight is not None:
            w = data.get(weight, 1.0)
            weights.append(w)

    distances = np.array(distances)

    # Compute weighted or unweighted average
    if weight is not None:
        weights = np.array(weights)
        avg_dist = np.sum(weights * distances) / np.sum(weights)
    else:
        avg_dist = np.mean(distances)

    return avg_dist


def module_spatial_size(
        G,
        distance_metric='haversine'
):
    """
    Compute module spatial size defined as the maximum distance
    between any two nodes in the module.

    Parameters
    ----------
    G : networkx.Graph
        Subgraph with node attribute 'loc' = (lat, lon).
    distance_metric : {'euclidean', 'haversine'}

    Returns
    -------
    float
        Maximum pairwise distance.
    """
    n = G.number_of_nodes()
    if n < 2:
        return np.nan

    coords = np.array([
        (G.nodes[v]['loc'][1], G.nodes[v]['loc'][0])
        for v in G.nodes()
    ])

    lon = coords[:, 0]
    lat = coords[:, 1]

    lon1 = lon[:, None]
    lat1 = lat[:, None]
    lon2 = lon[None, :]
    lat2 = lat[None, :]

    if distance_metric == 'euclidean':
        dist_mat = point_euclidean_dist(lon1, lat1, lon2, lat2)

    elif distance_metric == 'haversine':
        dist_mat = point_haversine_dist(lon1, lat1, lon2, lat2)

    else:
        raise ValueError(f"Unsupported distance metric: {distance_metric}")

    return np.max(dist_mat)


# ==========================================================
# ==========================================================
def erdos_renyi(G):
    """
    Generate an Erdos-Renyi random graph with same size and density as G.
    """
    n = G.number_of_nodes()
    p = nx.density(G)
    return nx.fast_gnp_random_graph(n, p)


def erdos_renyi_weighted(G, weight='weight', weight_type='equal'):
    """
    Generate a weighted Erdos-Renyi random graph that preserves
    the edge count and reassigns shuffled weights from the original G.
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()
    Gr = nx.gnm_random_graph(n, m)

    weights = np.array([d.get(weight, 1.0) for _, _, d in G.edges(data=True)], dtype=float)

    if weight_type == 'equal':
        w = float(weights.mean())
        nx.set_edge_attributes(
            Gr,
            {e: w for e in Gr.edges()},
            name=weight
        )
    elif weight_type == 'same':
        np.random.shuffle(weights)

        for (u, v), w in zip(Gr.edges(), weights):
            Gr[u][v][weight] = float(w)
    else:
        raise ValueError(f"Unsupported weight type: {weight_type}")

    return Gr


def core_periphery(G, cp_method='LipWeighted', significance_level=0.05):
    """
    Perform core-periphery detection and significance testing.

    Parameters
    ----------
    G : networkx.Graph
        Input weighted graph.
    cp_method : str
        Algorithm name: 'LipWeighted' or 'Lip'.
    significance_level : float
        Statistical significance threshold for q-test.

    Returns
    -------
    c, x, significant, p_values : tuple
        - c: dict of core-periphery pair IDs
        - x: dict of coreness values
        - significant: significance flag per structure
        - p_values: corresponding p-values
    """
    # Initialize algorithm
    if cp_method == 'LipWeighted':
        alg = LipWeighted()
    else:
        alg = cpnet.Lip()

    alg.detect(G)

    c = alg.get_pair_id()
    x = alg.get_coreness()

    # Use the project-local q-test implementation to avoid cpnet API drift.
    q_obs, q_rd, significant, p_values = qtest(
        c, x, G, alg,
        significance_level=significance_level,
        num_of_rand_net=500,
        null_model=erdos_renyi_weighted,
        weight='weight'
    )
    return c, x, significant, p_values


# ==========================================================
# -------- Compute Community Core-Periphery Statistics -----
# ==========================================================

def log_error_jsonl(error_info, log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(error_info, ensure_ascii=False) + "\n")


def check_module_validity(subG, min_nodes, min_edges):
    """
    Check whether a module is large enough for network analysis.
    """
    n_nodes = subG.number_of_nodes()
    n_edges = subG.number_of_edges()

    if n_nodes < min_nodes:
        # print('too_few_nodes')
        return False

    if n_edges < min_edges:
        # print('too_few_edges')
        return False

    return True


def compute_module_network_stats(subG):
    """
    Compute module-level and largest-connected-component network statistics.
    """

    # ---------- connectivity ----------
    is_connected = nx.is_connected(subG)

    if is_connected:
        lcc = subG
    else:
        lcc_nodes = max(nx.connected_components(subG), key=len)
        lcc = subG.subgraph(lcc_nodes).copy()

    # ---------- module-level weights (fast path) ----------
    module_edge_weight_sum_movement_count = sum(
        nx.get_edge_attributes(subG, "movement_count").values()
    )
    module_edge_weight_sum_movement_fraction = sum(
        nx.get_edge_attributes(subG, "movement_fraction").values()
    )

    module_node_weight_sum_visit_count = sum(
        nx.get_node_attributes(subG, "visit_count").values()
    )
    module_node_weight_sum_visit_fraction = sum(
        nx.get_node_attributes(subG, "visit_fraction").values()
    )

    # ---------- LCC-level weights ----------
    lcc_edge_weight_sum_movement_count = sum(
        nx.get_edge_attributes(lcc, "movement_count").values()
    )
    lcc_edge_weight_sum_movement_fraction = sum(
        nx.get_edge_attributes(lcc, "movement_fraction").values()
    )

    lcc_node_weight_sum_visit_count = sum(
        nx.get_node_attributes(lcc, "visit_count").values()
    )
    lcc_node_weight_sum_visit_fraction = sum(
        nx.get_node_attributes(lcc, "visit_fraction").values()
    )

    # ---------- assemble results ----------
    stats = {
        # structure
        "module_size": subG.number_of_nodes(),
        "module_edge_num": subG.number_of_edges(),
        "is_connected": is_connected,
        "lcc_size": lcc.number_of_nodes(),
        "lcc_edge_num": lcc.number_of_edges(),

        # module-level weights
        "module_edge_weight_sum_movement_count": module_edge_weight_sum_movement_count,
        "module_edge_weight_sum_movement_fraction": module_edge_weight_sum_movement_fraction,
        "module_node_weight_sum_visit_count": module_node_weight_sum_visit_count,
        "module_node_weight_sum_visit_fraction": module_node_weight_sum_visit_fraction,

        # LCC-level weights
        "lcc_edge_weight_sum_movement_count": lcc_edge_weight_sum_movement_count,
        "lcc_edge_weight_sum_movement_fraction": lcc_edge_weight_sum_movement_fraction,
        "lcc_node_weight_sum_visit_count": lcc_node_weight_sum_visit_count,
        "lcc_node_weight_sum_visit_fraction": lcc_node_weight_sum_visit_fraction,
    }

    return stats


def community_core_periphery_stats_individual(
        L_file_name,
        G_file_name,
        distance_metric='haversine',
        min_nodes=2,
        min_edges=1,
        cp_method='LipWeighted',
        significance_level=0.05,
        rg_weight='visit_count',
        avg_dist_weight='movement_count',
        error_log_path=None,
):
    try:
        L = read_location_sequence(L_file_name)
        G = read_graph(G_file_name)

        user_id = G.graph['user_id']
        print(user_id)

        movement_weight = nx.get_edge_attributes(G, 'movement_fraction')
        nx.set_edge_attributes(G, movement_weight, 'weight')

        rg = radius_of_gyration(G, distance_metric=distance_metric, weight=rg_weight)
        avg_dist = average_distance(G, distance_metric=distance_metric, weight=avg_dist_weight)

        # Hierarchical community detection
        communities_df = container_hierarchical_community(L, G, distance_metric=distance_metric)
        records = []

        for _, row in communities_df.iterrows():

            nodes = row['node_list']

            # Extract subgraph for module
            subG = G.subgraph(nodes).copy()

            is_valid = check_module_validity(subG, min_nodes, min_edges)

            # Skip small modules
            if not is_valid:
                record = {
                    "user_id": user_id,
                    "module_id": row["module_id"],
                    'module_node_list': list(nodes),
                    "module_depth": row["depth"],
                    "module_level": row["level"],

                    "radius_of_gyration": None,
                    "average_distance": None,
                    "module_spatial_size": None,
                    "total_radius_of_gyration": None,
                    "total_average_distance": None,

                    "core_num": None,
                    "coreness": None,
                    "significant": None,
                    "p_value": None,

                    # structure
                    "module_size": None,
                    "module_edge_num": None,
                    "is_connected": None,
                    "lcc_size": None,
                    "lcc_edge_num": None,

                    # module-level weights
                    "module_edge_weight_sum_movement_count": None,
                    "module_edge_weight_sum_movement_fraction": None,
                    "module_node_weight_sum_visit_count": None,
                    "module_node_weight_sum_visit_fraction": None,

                    # LCC-level weights
                    "lcc_edge_weight_sum_movement_count": None,
                    "lcc_edge_weight_sum_movement_fraction": None,
                    "lcc_node_weight_sum_visit_count": None,
                    "lcc_node_weight_sum_visit_fraction": None,
                }
                records.append(record)
                continue

            module_stats = compute_module_network_stats(subG)
            _, x, sig, p_val = core_periphery(subG, cp_method=cp_method, significance_level=significance_level)

            # Compute spatial metrics
            rg_subg = radius_of_gyration(subG, distance_metric=distance_metric, weight=rg_weight)
            avg_dist_subg = average_distance(subG, distance_metric=distance_metric, weight=avg_dist_weight)
            module_size_subg = module_spatial_size(subG, distance_metric=distance_metric
                                                   )

            # Record results
            record = {
                "user_id": user_id,
                'module_id': row['module_id'],
                'module_node_list': nodes,
                "module_depth": row['depth'],
                "module_level": row['level'],

                "radius_of_gyration": rg_subg,
                "average_distance": avg_dist_subg,
                "module_spatial_size": module_size_subg,
                "total_radius_of_gyration": rg,
                "total_average_distance": avg_dist,

                "core_num": sum(1 for v in x.values() if v > 0),
                "coreness": x,
                "significant": sig,
                "p_value": p_val,
            }
            record.update(module_stats)
            records.append(record)

        return pd.DataFrame(records)

    except Exception as e:
        error_info = {
            "time": datetime.now().isoformat(),
            "L_file": L_file_name,
            "G_file": G_file_name,
            "error_type": type(e).__name__,
            "error_message": str(e)
        }

        print(f"[ERROR] {G_file_name}: {e}")

        if error_log_path is not None:
            log_error_jsonl(error_info, error_log_path)

        return pd.DataFrame()


def community_core_periphery_stats(
        LOC_SEQ_PATH=r'./loc_dir',
        GRAPH_PATH=r'./graph_dir',
        OUTPUT_PATH=r'./output_dir',
        distance_metric='haversine',
        min_nodes=2,
        min_edges=1,
        cp_method='LipWeighted',
        significance_level=0.05,
        rg_weight='visit_count',
        avg_dist_weight='movement_count',
):
    # Create output folder if missing
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    # Collect all graph files recursively
    L_file_paths = []
    G_file_paths = []
    args = []
    for root, dirs, files in os.walk(GRAPH_PATH):
        for G_file in files:
            G_file_path = os.path.join(GRAPH_PATH, G_file)
            G_file_paths.append(G_file_path)

            user_id = G_file.split('.')[0].split('_')[-1]
            L_file = f"L_{user_id}.csv"

            L_file_path = os.path.join(LOC_SEQ_PATH, L_file)
            L_file_paths.append(L_file_path)
            args.append((L_file_path, G_file_path, distance_metric, min_nodes, min_edges, cp_method, significance_level,
                         rg_weight, avg_dist_weight, os.path.join(OUTPUT_PATH, "container_cp_errors.jsonl")))

    # random.seed(43)
    # sampled_indices = random.sample(range(len(G_file_paths)), num_sample)
    #
    # sampled_args = [
    #     (L_file_paths[i], G_file_paths[i], distance_metric, min_nodes, cp_method, significance_level, rg_weight,
    #      avg_dist_weight)
    #     for i in sampled_indices
    # ]

    # Optional: parallel execution (currently commented out)
    with Pool(processes=max(1, (os.cpu_count() or 1) - 2)) as pool:
        stats_list = pool.starmap(community_core_periphery_stats_individual, args)

    # Sequential execution (fallback)
    # stats_list = []
    # for i,file_name in enumerate(G_file_paths):
    #     stat_i = community_core_periphery_stats_individual(L_file_paths[i],file_name,
    #                                                        distance_metric=distance_metric,
    #                                                        min_nodes=min_nodes,
    #                                                        cp_method=cp_method,
    #                                                        significance_level=significance_level,
    #                                                        rg_weight=rg_weight,
    #                                                        avg_dist_weight=avg_dist_weight
    #                                                        )
    #     stats_list.append(stat_i)

    # Merge and export results
    cp_stats_result = pd.concat(stats_list)

    if 'D1_YJMob100K' in GRAPH_PATH or 'dataset_yjmob100k' in GRAPH_PATH:
        distance_ratio = 0.5
    else:
        distance_ratio = 0.001

    cp_stats_result['average_distance'] = cp_stats_result['average_distance'] * distance_ratio
    cp_stats_result['radius_of_gyration'] = cp_stats_result['radius_of_gyration'] * distance_ratio
    cp_stats_result['module_spatial_size'] = cp_stats_result['module_spatial_size'] * distance_ratio

    cp_stats_result['total_average_distance'] = cp_stats_result['total_average_distance'] * distance_ratio
    cp_stats_result['total_radius_of_gyration'] = cp_stats_result['total_radius_of_gyration'] * distance_ratio

    cp_stats_result['relative_average_distance'] = cp_stats_result['average_distance'] / cp_stats_result[
        'total_average_distance']
    cp_stats_result['relative_radius_of_gyration'] = cp_stats_result['radius_of_gyration'] / cp_stats_result[
        'total_radius_of_gyration']

    cp_stats_result.to_csv(os.path.join(OUTPUT_PATH, 'container_core_periphery_equal_weight_stat.csv'), index=False)


def rename_df(file_name):
    df=pd.read_csv(file_name)

    df=df.rename(columns={  # module-level weights
                    "module_edge_weight_sum_movement_count": "movement_count",
                    "module_edge_weight_sum_movement_fraction": "movement_fraction",
                    "module_node_weight_sum_visit_count": "visit_count",
                    "module_node_weight_sum_visit_fraction": "visit_fraction",

                    # LCC-level weights
                    "lcc_edge_weight_sum_movement_count": "lcc_movement_count",
                    "lcc_edge_weight_sum_movement_fraction": "lcc_movement_fraction",
                    "lcc_node_weight_sum_visit_count": "lcc_visit_count",
                    "lcc_node_weight_sum_visit_fraction": "lcc_visit_fraction",
               })

    df.to_csv(file_name, index=False)


def add_module_unique_id(file_name):
    df=pd.read_csv(file_name)
    df['module_unique_id']=df.groupby('user_id').cumcount()
    df.to_csv(file_name, index=False)


def filter_container(
    file_name,
    attr_1,
    attr_2,
    attr_threshold_1,
    attr_threshold_2,
    suffix="_filtered"
):
    """
    Filter dataframe by two attribute thresholds and save results.

    Returns
    -------
    out_file : str
        Path to the filtered csv file.
    summary_file : str
        Path to the summary statistics file.
    """

    df = pd.read_csv(file_name)

    # ---------------------------
    # ---------------------------
    o_module_num = len(df)
    o_node_num = df["module_size"].sum()

    df["average_degree"] = 2 * df["module_edge_num"] / df["module_size"]

    # ---------------------------
    # ---------------------------
    df_filtered = df.loc[
        (df[attr_1] >= attr_threshold_1) &
        (df[attr_2] >= attr_threshold_2)
    ]

    # ---------------------------
    # ---------------------------
    d_module_num = len(df_filtered)
    d_node_num = df_filtered["module_size"].sum()

    # ---------------------------
    # ---------------------------
    removed_module_num = o_module_num - d_module_num
    removed_module_ratio = removed_module_num / o_module_num

    removed_node_num = o_node_num - d_node_num
    removed_node_ratio = removed_node_num / o_node_num

    # ---------------------------
    # ---------------------------
    base, ext = os.path.splitext(file_name)
    out_file = f"{base}{suffix}{ext}"
    df_filtered.to_csv(out_file, index=False)

    # ---------------------------
    # ---------------------------
    summary = pd.DataFrame({
        "metric": [
            "original_module_num",
            "filtered_module_num",
            "removed_module_num",
            "removed_module_ratio",
            "original_node_num",
            "filtered_node_num",
            "removed_node_num",
            "removed_node_ratio"
        ],
        "value": [
            o_module_num,
            d_module_num,
            removed_module_num,
            removed_module_ratio,
            o_node_num,
            d_node_num,
            removed_node_num,
            removed_node_ratio
        ]
    })

    summary_file = f"{base}{suffix}_summary.csv"
    summary.to_csv(summary_file, index=False)

    # ---------------------------
    # ---------------------------
    print(f"Filtered dataframe saved to:\n  {out_file}")
    print(f"Summary saved to:\n  {summary_file}")
    print(
        f"Removed modules: {removed_module_num} ({removed_module_ratio:.2%})\n"
        f"Removed nodes: {removed_node_num} ({removed_node_ratio:.2%})"
    )

    return out_file, summary_file




