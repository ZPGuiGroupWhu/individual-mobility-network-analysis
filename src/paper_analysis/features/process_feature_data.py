
"""Post-process raw spatial and topological feature tables for summaries."""

import numpy as np
import pandas as pd
import os

def process_spatial_features(SPATIAL_DIR):
    s_df = pd.read_csv(os.path.join(SPATIAL_DIR, 'spatial_stats.csv'))

    s_df['trip_length']=s_df['trip_length']/1000
    s_df['radius_of_gyration']=s_df['radius_of_gyration']/1000

    selected_columns=['user_id', 'number_of_trips', 'number_of_locations', 'trip_length', 'radius_of_gyration']

    s_df = s_df[selected_columns]
    s_df.to_csv(os.path.join(SPATIAL_DIR,'processed_spatial_stats.csv'), index=False)
    return s_df

def process_topology_features(TOPOLOGY_DIR,alpha):
    t_df = pd.read_csv(os.path.join(TOPOLOGY_DIR, 'topological_stats.csv'))

    t_df['degree_sig']=(t_df['degree_p1'] < alpha)
    t_df['degree_fraction_sig']=(t_df['degree_fraction_p1'] < alpha)
    t_df['degree_r_sig']=(t_df['degree_p'] < alpha)
    t_df['visit_count_r_sig']=(t_df['visit_count_p'] < alpha)

    selected_columns=['user_id', 'number_of_nodes', 'number_of_edges',
    'density', 'average_degree',
    'degree_distribution', 'degree_sig', 'degree_parament1', 'degree_parament2',
    'degree_fraction_distribution', 'degree_fraction_sig',  'degree_fraction_parament1','degree_fraction_parament2',
    'average_shortest_path_length', 'diameter',
    'average_clustering_coefficient',
    'degree_r', 'degree_r_sig',
    'visit_count_r', 'visit_count_r_sig']

    t_df = t_df[selected_columns]
    t_df.to_csv(os.path.join(TOPOLOGY_DIR,'processed_topological_stats.csv'), index=False)
    return t_df


