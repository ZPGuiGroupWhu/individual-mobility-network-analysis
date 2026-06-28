"""Input/output helpers for the paper reproducibility workflow.

This module reads and writes tabular intermediate files and individual
mobility-network JSON files. It is used by preprocessing, feature extraction,
multiscale analysis, and compressibility scripts.
"""

import pandas as pd
import networkx as nx
import json
import numpy as np
import os

def read_triplegs(file_name):
    df = pd.read_csv(file_name)
    df["origin_time"] = pd.to_datetime(df["origin_time"], format="%Y-%m-%d %H:%M:%S")
    df["destination_time"] = pd.to_datetime(df["destination_time"], format="%Y-%m-%d %H:%M:%S")
    return df


def read_location_sequence(file_name):
    df = pd.read_csv(file_name)
    df["started_at"] = pd.to_datetime(df["started_at"], format="%Y-%m-%d %H:%M:%S%z",utc=True)
    df["finished_at"] = pd.to_datetime(df["finished_at"], format="%Y-%m-%d %H:%M:%S%z",utc=True)
    return df


def read_OD(file_name):
    """
    Read a comma-separated values (csv) file into DataFrame.
    :param file_name: str
        the path and name of file.
    :return: dataframe
        A comma-separated values (csv) file is returned as two-dimensional data structure with labeled axes.
    """
    csv_data = pd.read_csv(file_name)
    return csv_data


def write_OD(df, file_name):
    """
    Write object to a comma-separated values (csv) file.
    :param file_name: str
        the path and name of outputted file.
    :param df: dataframe
        the outputted dataframe.
    """
    try:
        df.to_csv(file_name, sep=",", index=False)
    except Exception as err:
        print(err)


def read_location(file_name):
    csv_data = pd.read_csv(file_name)
    return csv_data

def write_graph(G, file_name):
    G_data = nx.node_link_data(G)

    # Serializing json
    def convert(o):
        if isinstance(o, np.int64):
            return int(o)
        raise TypeError

    G_json = json.dumps(G_data, default=convert)
    # print(G_json)
    # Writing to json
    with open(file_name, 'w') as out_file:
        out_file.write(G_json)


def read_graph(file_name):
    try:
        with open(file_name, 'r') as file:
            data = json.load(file)
        G = nx.node_link_graph(data)
        return G
    except json.JSONDecodeError as e:
        print(f"JSONDecodeError: {e} in file {file_name}")
        raise
    except Exception as e:
        print(f"Error reading file {file_name}: {e}")
        raise


def print_graph(G):
    # # node attributes
    # print("Node attributes:")
    # for node in G.nodes:
    #     print("Node", node, ":", G.nodes[node])
    #
    # # edge attributes
    # print("\nEdge attributes:")
    # for edge in G.edges:
    #     print("Edge", edge, ":", G.edges[edge])
    # graph attributes
    print("\nGraph attributes:")
    print(G.graph)

    df_n, df_e = graph_to_dataframe(G)
    print("\nNode dataframe")
    print(df_n)
    print("\nEdge dataframe")
    print(df_e)


def draw_graph(G):
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=300, font_weight='bold')
    labels = nx.get_edge_attributes(G, 'movement_count')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)


def graph_to_dataframe(G):
    node_data = {node: G.nodes[node] for node in G.nodes}
    nodes_df = pd.DataFrame.from_dict(node_data, orient='index')
    nodes_df.reset_index(inplace=True, names='id')

    edge_df = nx.to_pandas_edgelist(G)
    return nodes_df, edge_df



def log_error_jsonl(error_info, path):
    """Append one error record to a JSONL log file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(error_info, ensure_ascii=False) + '\n')
