"""Container-model community detection wrappers used by multiscale analysis."""

import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict
from paper_analysis.utils.io import read_graph, print_graph, draw_graph, read_location_sequence
from paper_analysis.multiscale.methods.container_model import scale_by_scale_optim, container_utils
from paper_analysis.multiscale.methods.container_model.container_utils import *

def container_hierarchy_to_df(container_hierarchy):
    """
    Convert ``get_container_labels`` output into a module DataFrame.
    """

    records = []

    for container_key, node_set in container_hierarchy.items():
        depth = len(container_key)
        module_id = container_key[-1]

        records.append({
            "depth": depth,
            "module_id": module_id,
            "node_list": list(node_set)
        })

    df = pd.DataFrame(records,columns=["depth","module_id","node_list"])

    max_depth = df["depth"].max()
    df["level"] = max_depth - df["depth"] + 1

    return df


def replace_location_id_with_node_id(
    df_partition,
    G,
    location_attr='location_id'
):
    """
    Replace location IDs in module node lists with graph node IDs.

    Parameters
    ----------
    df_partition : pd.DataFrame
        Module table containing ``level``, ``module_id``, and ``node_list``.
    G : networkx.Graph
        Graph with a location identifier stored as a node attribute.
    location_attr : str
        Node attribute storing location IDs.

    Returns
    -------
    df_partition_new : pd.DataFrame
        Module table with graph node IDs in ``node_list``.
    """

    loc2node = {
        int(data[location_attr]): n
        for n, data in G.nodes(data=True)
        if location_attr in data
    }

    df_partition_new = df_partition.copy()
    df_partition_new['node_list'] = df_partition_new['node_list'].apply(
        lambda loc_list: [loc2node[loc] for loc in loc_list if loc in loc2node]
    )

    return df_partition_new


def container_fit(L,distance_metric='haversine'):
    loc_df = (
        L[['location_id', 'location_latitude', 'location_longitude']]
        .drop_duplicates()
        .sort_values('location_id')
        .reset_index(drop=True)
    )

    raw_labels = L['location_id']

    unique_ids = loc_df['location_id'].tolist()
    id2idx = {lid: i for i, lid in enumerate(unique_ids)}
    idx2id = {i: lid for i, lid in enumerate(unique_ids)}

    labels = raw_labels.map(id2idx).to_numpy()

    stop_locations = loc_df[
        ['location_latitude', 'location_longitude']
    ].values

    if distance_metric == 'haversine':
        dist_f = haversine
        min_dist=1.2
    elif distance_metric == 'euclidean':
        dist_f = euclidean
        min_dist = 1.2
    else:
        print('distance_metric must be "haversine" or "euclidean"')

    optim_instance = scale_by_scale_optim.ScalesOptim(
        labels=labels,
        stop_locations=stop_locations,
        distance_func=dist_f,
        min_dist=min_dist,
        min_diff=1,
        information_criterion="BIC",
        linkage_method="complete",
        bootstrap=True,
        bootstrap_iter=200,
        siglvl=0.05,
        verbose=False
    )

    (final_series,final_scales,likelihoods,criterion_s,final_sizes,final_proba_dist,final_alphas) = optim_instance.find_best_scale()

    container_hierarchy = container_utils.get_container_labels(final_series)

    df_partition = container_hierarchy_to_df(container_hierarchy)

    if df_partition.empty:
        df_all=pd.DataFrame([{
                "depth": 0,
                "module_id": 0,
                "level": 1,
                "node_list": list(set(labels))
            }])
    else:
        df_all=pd.DataFrame([{
                "depth": 0,
                "module_id": 0,
                "level": df_partition['level'].max() + 1,
                "node_list": list(set(labels))
            }])

    df_partition = pd.concat([df_partition,df_all],ignore_index=True)

    df_partition['node_list'] = df_partition['node_list'].apply(
        lambda lst: [idx2id[i] for i in lst]
    )

    return df_partition



