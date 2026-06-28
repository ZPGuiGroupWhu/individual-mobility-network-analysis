"""Pairwise graph-distance calculations for individual mobility networks."""

import os
from multiprocessing import Pool
import networkx as nx
import pandas as pd
from math import comb
from paper_analysis.utils.gislib import point_haversine_dist, point_euclidean_dist
import time
import random
from paper_analysis.utils.io import read_graph, print_graph, draw_graph
import numpy as np


# def generate_graph():
#     # # Example usage
#     # # Create two graphs with more than 10 nodes using geographical coordinates
#     # G1 = nx.Graph()
#     # # Example coordinates: (latitude, longitude)
#     # coordinates_G1 = [
#     #     (34.0522, -118.2437),  # Los Angeles
#     #     (37.7749, -122.4194),  # San Francisco
#     #     (40.7128, -74.0060),  # New York
#     #     (41.8781, -87.6298),  # Chicago
#     #     (29.7604, -95.3698),  # Houston
#     #     (39.7392, -104.9903),  # Denver
#     #     (33.4484, -112.0740),  # Phoenix
#     #     (47.6062, -122.3321),  # Seattle
#     #     (25.7617, -80.1918),  # Miami
#     #     (38.9072, -77.0369),  # Washington D.C.
#     #     (37.3382, -121.8863),  # San Jose
#     #     (32.7157, -117.1611)  # San Diego
#     # ]
#     #
#     # # Add nodes with geographical coordinates
#     # for i, (lat, lon) in enumerate(coordinates_G1, start=1):
#     #     G1.add_node(i, loc=[lat, lon])
#     #
#     # # Add some edges (random connections)
#     # edges_G1 = [(1, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7), (6, 8), (7, 9), (8, 10), (9, 11), (10, 12)]
#     # G1.add_edges_from(edges_G1)
#     #
#     # G2 = nx.Graph()
#     # # Different geographical coordinates for variation
#     # coordinates_G2 = [
#     #     (34.0522, -118.2437),  # Los Angeles
#     #     (37.7749, -122.4194)  # San Francisco
#     # ]
#     #
#     # # Add nodes with geographical coordinates
#     # for i, (lat, lon) in enumerate(coordinates_G2, start=1):
#     #     G2.add_node(i, loc=[lat, lon])
#     #
#     # # Add some edges (random connections)
#     # edges_G2 = [(1, 2)]
#     # G2.add_edges_from(edges_G2)
#     # return G1, G2
#     # Create two graphs with more than 10 nodes using geographical coordinates
#     G1 = nx.Graph()
#     # Example coordinates: (latitude, longitude)
#     coordinates_G1 = [
#         (34.0522, -118.2437),  # Los Angeles
#         (37.7749, -122.4194),  # San Francisco
#         (40.7128, -74.0060),  # New York
#         (41.8781, -87.6298),  # Chicago
#         (29.7604, -95.3698),  # Houston
#         (39.7392, -104.9903),  # Denver
#         (33.4484, -112.0740),  # Phoenix
#         (47.6062, -122.3321),  # Seattle
#         (25.7617, -80.1918),  # Miami
#         (38.9072, -77.0369),  # Washington D.C.
#         (37.3382, -121.8863),  # San Jose
#         (32.7157, -117.1611)  # San Diego
#     ]
#
#     # Add nodes with geographical coordinates
#     for i, (lat, lon) in enumerate(coordinates_G1, start=1):
#         G1.add_node(i, loc=[lat, lon])
#
#     # Add some edges (random connections)
#     edges_G1 = [(1, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7), (6, 8), (7, 9), (8, 10), (9, 11), (10, 12)]
#     G1.add_edges_from(edges_G1)
#
#     G2 = nx.Graph()
#     # Different geographical coordinates for variation
#     coordinates_G2 = [
#         (34.0522, -118.2437),  # Los Angeles
#         (37.7749, -122.4194),  # San Francisco
#         (40.7128, -74.0060),  # New York
#         (41.8781, -87.6298),  # Chicago
#         (29.7604, -95.3698),  # Houston
#         (39.7392, -104.9903),  # Denver
#         (34.0522, -118.2437),  # Another Los Angeles (similar for variation)
#         (47.6062, -122.3321),  # Seattle
#         (25.7617, -80.1918),  # Miami
#         (38.9072, -77.0369),  # Washington D.C.
#         (37.3382, -121.8863),  # San Jose
#         (32.7157, -117.1611)  # San Diego
#     ]
#
#     # Add nodes with geographical coordinates
#     for i, (lat, lon) in enumerate(coordinates_G2, start=1):
#         G2.add_node(i, loc=[lat, lon])
#
#     # Add some edges (random connections)
#     edges_G2 = [(1, 2), (1, 3), (2, 5), (3, 6), (4, 7), (5, 8), (6, 9), (7, 10), (8, 11), (9, 12)]
#     G2.add_edges_from(edges_G2)
#     return G1, G2


def GED_mobility_network(G1_file_name, G2_file_name, node_cost_type='lnglat', distance_parameter=2, normalize=True):
    user1 = G1_file_name.split('\\')[-1]
    user2 = G2_file_name.split('\\')[-1]
    print(user1, user2)
    G1 = read_graph(G1_file_name)
    G2 = read_graph(G2_file_name)
    # Cache for distance calculations
    distance_cache = {}

    # Define node substitution cost function
    def node_subst_cost(n1, n2):
        # Convert the location lists to tuples for hashing
        key = (tuple(n1['loc']), tuple(n2['loc']))

        # Check if the distance is already calculated
        if key in distance_cache:
            return distance_cache[key]

        lon1, lat1 = n1['loc'][1], n1['loc'][0]  # (longitude, latitude)
        lon2, lat2 = n2['loc'][1], n2['loc'][0]

        if node_cost_type == 'lnglat':
            dis = point_haversine_dist(lon1, lat1, lon2, lat2) / 1000  # Convert to kilometers
        elif node_cost_type == 'grid':
            dis = point_euclidean_dist(lon1, lat1, lon2, lat2) * 0.5  # Convert grid distance to kilometers
        else:
            raise ValueError("Unsupported node cost type. Please use 'lnglat' or 'grid'.")

        # Distance normalization
        cost = dis / (distance_parameter + dis)
        distance_cache[key] = cost  # Cache the result
        return cost

    # Calculate edit distance
    edit_distance = nx.algorithms.similarity.graph_edit_distance(
        G1, G2, node_subst_cost=node_subst_cost,
        roots=(0, 0),
        timeout=120
    )

    # Normalize the edit distance if required
    if normalize:
        max_dist = max(G1.number_of_nodes() + G1.number_of_edges(),
                       G2.number_of_nodes() + G2.number_of_edges())
        edit_distance = edit_distance / max_dist

    return edit_distance


def NSD_mobility_network(G1_file_name, G2_file_name, loc_type='lnglat', distance_parameter=2, normalize=True):
    """
    NODE_SPATIAL_DISTANCE: Calculate the distance between two mobility networks is measured by measuring the spatial distance between nodes.
    Nodes are matched by the number of stays.

    Parameters:
    - G1_file_name (str): The file name of the first graph (G1) to be read.
    - G2_file_name (str): The file name of the second graph (G2) to be read.
    - loc_type (str): The method used to calculate distance. Options are 'haversine' or 'grid'.
                      'haversine' calculates distances based on latitude and longitude,
                      while 'grid' uses Euclidean distance.
    - distance_parameter (float): A parameter used in the normalization of distances.
                      It helps in adjusting the impact of distances in the calculations.
    - normalize (bool): If True, the total distance will be normalized by the maximum length of the graphs.
                        If False, the raw total distance will be returned.

    Returns:
    - float: The total distance between nodes in the two graphs, or the number of nodes in the non-empty graph if one is empty.
    """

    # Read the graphs from the provided file names
    G1 = read_graph(G1_file_name)
    G2 = read_graph(G2_file_name)

    # Check if either graph is empty
    if G1.number_of_nodes() == 0 or G2.number_of_nodes() == 0:
        return max(G1.number_of_nodes(), G2.number_of_nodes())  # Return the number of nodes in the non-empty graph

    # Extract node attributes into DataFrames
    df_G1 = pd.DataFrame.from_dict(dict(G1.nodes(data=True)), orient='index')
    df_G2 = pd.DataFrame.from_dict(dict(G2.nodes(data=True)), orient='index')

    # Sort by visit count
    df_G1_sorted = df_G1.sort_values(by='visit_count', ascending=False).reset_index(drop=True)
    df_G2_sorted = df_G2.sort_values(by='visit_count', ascending=False).reset_index(drop=True)

    # Calculate the minimum and maximum lengths of the sorted DataFrames
    min_length = min(len(df_G1_sorted), len(df_G2_sorted))
    max_length = max(len(df_G1_sorted), len(df_G2_sorted))
    # print(min_length, max_length)  # Print the lengths for debugging purposes

    # Extract latitude and longitude from the sorted DataFrames
    loc_G1 = df_G1_sorted.loc[:min_length - 1, 'loc'].values
    lat1, lon1 = zip(*loc_G1)

    loc_G2 = df_G2_sorted.loc[:min_length - 1, 'loc'].values
    lat2, lon2 = zip(*loc_G2)

    # Select the distance calculation method
    if loc_type == 'grid':
        distances = point_euclidean_dist(lon1, lat1, lon2, lat2) * 0.5  # Calculate Euclidean distance
    elif loc_type == 'lnglat':
        distances = point_haversine_dist(lon1, lat1, lon2, lat2) / 1000  # Calculate Haversine distance in kilometers
    else:
        raise ValueError("Invalid location type used to calculate distance. Please use 'lnglat' or 'grid'.")

    # Normalize the distances
    distances = distances / (distance_parameter + distances)

    # Handle extra nodes by creating an array of ones
    extra_distance = np.ones(abs(max_length - min_length))
    distances = np.concatenate((distances, extra_distance))

    # Calculate the total distance
    total_distance = distances.sum()

    # Normalize the total distance if required
    if normalize:
        total_distance = total_distance / max_length

    return total_distance


def random_sampling_without_replacement(lst, N):
    # Calculate the maximum number of unique pairs
    max_pairs = comb(len(lst), 2)

    # Check if the requested number of samples exceeds the maximum possible
    if N > comb(len(lst), 2):
        raise ValueError(
            f"Cannot sample {N} unique pairs from a list of length {len(lst)}. Maximum possible pairs is {max_pairs}.")

    sampled_set = set()

    while len(sampled_set) < N:
        sampled_pair = tuple(sorted(random.sample(lst, 2)))

        if sampled_pair not in sampled_set:
            sampled_set.add(sampled_pair)

    return sampled_set


def GED_networks(GRAPH_DIR=r'./GRAPH_DIR', GED_DIR=r'./GED_DIR', node_cost_type='lnglat', sample_size=100,
                 distance_parameter=2, normalize=True):
    file_names = os.listdir(GRAPH_DIR)
    sampled_file_pairs = random_sampling_without_replacement(file_names, sample_size)

    # partial_calculate_ged = partial(GED_mobility_network, node_cost_type=node_cost_type,
    #                                 distance_parameter=distance_parameter, normalize=normalize)

    # with Pool(processes=os.cpu_count() - 2) as pool:
    #     dis_list = pool.map(GED_mobility_network, sampled_file_pairs)

    ged_results = pd.DataFrame(columns=['G1', 'G2', 'GED'])

    for pair in sampled_file_pairs:
        start_time = time.time()
        user1 = pair[0].split('.')[0]
        user2 = pair[1].split('.')[0]
        print(user1, user2)
        file1 = os.path.join(GRAPH_DIR, pair[0])
        file2 = os.path.join(GRAPH_DIR, pair[1])
        ged = GED_mobility_network(file1, file2, node_cost_type, distance_parameter, normalize)
        ged_results.loc[len(ged_results)] = [user1, user2, ged]
        end_time = time.time()
        print(f"Execution time: {end_time - start_time:.10f} seconds")

    ged_results.to_csv(os.path.join(GED_DIR, 'graph_edit_distance.csv'), index=False)


def NSD_networks(GRAPH_DIR=r'./GRAPH_DIR', GED_DIR=r'./NSD_DIR', loc_type='lnglat', sample_size=100,
                 distance_parameter=2, normalize=True):
    file_names = os.listdir(GRAPH_DIR)
    sampled_file_pairs = random_sampling_without_replacement(file_names, sample_size)

    nsd_results = pd.DataFrame(columns=['G1', 'G2', 'node_spatial_distance'])

    for pair in sampled_file_pairs:
        # start_time = time.time()
        user1 = pair[0].split('.')[0]
        user2 = pair[1].split('.')[0]
        print(user1, user2)
        file1 = os.path.join(GRAPH_DIR, pair[0])
        file2 = os.path.join(GRAPH_DIR, pair[1])
        nsd = NSD_mobility_network(file1, file2, loc_type, distance_parameter, normalize)
        nsd_results.loc[len(nsd_results)] = [user1, user2, nsd]
        # end_time = time.time()
        # print(f"Execution time: {end_time - start_time:.10f} seconds")

    nsd_results.to_csv(os.path.join(GED_DIR, 'node_spatial_distance.csv'), index=False)



