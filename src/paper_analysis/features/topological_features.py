"""Topological feature extraction for individual mobility networks.

Inputs are NetworkX node-link JSON files named ``G_*.json``. The main output is
``topological_stats.csv``, with one row per individual network and columns used
by downstream feature processing and dispersion summaries.
"""

from multiprocessing import Pool

import numpy as np
import pandas as pd
import os

import powerlaw
import networkx as nx

from paper_analysis.utils.io import read_graph
import warnings
from paper_analysis.features.assortativity_coefficient import assortativity_coefficient

warnings.filterwarnings('ignore')


def fitting_parameters(sequence, xmin=None, discrete=False):
    fit = powerlaw.Fit(sequence, xmin=xmin, discrete=discrete, verbose=False)
    ratio1, p1 = fit.distribution_compare('exponential', 'power_law')
    ratio2, p2 = fit.distribution_compare('exponential', 'truncated_power_law')
    ratio3, p3 = fit.distribution_compare('power_law', 'truncated_power_law')
    # print(ratio1, p1)
    # print(ratio2, p2)
    # print(ratio3, p3)
    # print('Exponential', fit.exponential.parameter1_name, fit.exponential.parameter2_name)
    # print('Power Law', fit.power_law.parameter1_name, fit.power_law.parameter2_name)
    # print('Truncated Power Law', fit.truncated_power_law.parameter1_name, fit.truncated_power_law.parameter2_name,
    #       fit.truncated_power_law.parameter3_name)
    if ratio1 > 0 and ratio2 > 0:
        return 'Exponential', p1, p2, fit.exponential.parameter1, None, None

    elif ratio1 < 0 and ratio3 > 0:
        return 'Power Law', p1, p3, fit.power_law.alpha, None, fit.power_law.xmin

    elif ratio2 < 0 and ratio3 < 0:
        return 'Truncated Power Law', p2, p3, fit.truncated_power_law.alpha, fit.truncated_power_law.parameter2, fit.truncated_power_law.xmin

    else:
        return None, None, None, None, None, None


def number_of_nodes_edge(G):
    E = G.number_of_edges()
    N = G.number_of_nodes()
    return E, N


def density(G):
    return nx.density(G)


def average_degree(G):
    E = G.number_of_edges()
    N = G.number_of_nodes()
    return (2 * E) / N


def degree_sequence(G):
    degree_seq = list(dict(G.degree()).values())
    return degree_seq


def fitting_degree_sequence(G):
    degree_seq = degree_sequence(G)
    try:
        best_distribution, p1, p2, parament1, parament2, parament3 = fitting_parameters(degree_seq, discrete=True)
        return best_distribution, p1, p2, parament1, parament2, parament3
    except (ValueError, TypeError, ZeroDivisionError) as e:
        print(f"error: {e}")
        return None, None, None, None, None, None


def degree_fraction_sequence(G):
    degree_seq = list(dict(G.degree()).values())
    degree_fraction_seq = [int(d_i) / sum(degree_seq) for d_i in degree_seq]
    return degree_fraction_seq


def fitting_degree_fraction_sequence(G):
    degree_fraction_seq = degree_fraction_sequence(G)
    try:
        best_distribution, p1, p2, parament1, parament2, parament3 = fitting_parameters(degree_fraction_seq)
        return best_distribution, p1, p2, parament1, parament2, parament3
    except (ValueError, TypeError, ZeroDivisionError) as e:
        print(f"error: {e}")
        return None, None, None, None, None, None


def average_shortest_path_length(G, normalize=False):
    if nx.is_connected(G):
        avg_p = nx.average_shortest_path_length(G)
    else:
        print('No connected graph,compute the average shortest path for the largest component.')
        max_connected_component = max(nx.connected_components(G), key=len)
        subgraph = G.subgraph(max_connected_component)
        avg_p = nx.average_shortest_path_length(subgraph)

    if normalize:
        avg_p = avg_p / np.log(G.number_of_nodes())
    return avg_p


def diameter(G, normalize=False):
    if nx.is_connected(G):
        dim = nx.diameter(G)
    else:
        print('No connected graph,compute the diameter of the largest component.')
        max_connected_component = max(nx.connected_components(G), key=len)
        subgraph = G.subgraph(max_connected_component)
        dim = nx.diameter(subgraph)

    if normalize:
        dim = dim / np.log(G.number_of_nodes())

    return dim


def average_clustering_coefficient(G):
    return nx.average_clustering(G)


def attributes_assortativity_coefficient(G,alternative='two-sided'):
    d_prsr, d_prsp = assortativity_coefficient(G, attribute='degree', method='pearson', alternative=alternative,edge_weight=None)
    vc_prsr, vc_prsp = assortativity_coefficient(G, attribute='visit_count', method='pearson',alternative=alternative,edge_weight=None)
    # vf_prsr, vf_prsp = assortativity_coefficient(G, attribute='visit_fraction', method='pearson', edge_weight=None)

    vc_prsr_wgt, vc_prsp_wgt = assortativity_coefficient(G, attribute='visit_count', method='pearson',alternative=alternative,
                                                         edge_weight='movement_count')
    # vf_prsr_wgt, vf_prsp_wgt = assortativity_coefficient(G, attribute='visit_fraction', method='pearson',
    #                                                      edge_weight='movement_count')
    return (d_prsr, d_prsp,
            vc_prsr, vc_prsp,
            vc_prsr_wgt, vc_prsp_wgt,
            # vf_prsr, vf_prsp,
            # vf_prsr_wgt, vf_prsp_wgt
            )


def node_attribute_sequence(G, attribute):
    return list(nx.get_node_attributes(G, attribute).values())


def average_node_attribute(G, attribute):
    attribute_seq = node_attribute_sequence(G, attribute)
    return np.mean(attribute_seq)


def fitting_node_attribute_sequence(G, attribute, discrete=False):
    node_attr_seq = node_attribute_sequence(G, attribute)
    try:
        best_distribution, p1, p2, parament1, parament2, parament3 = fitting_parameters(node_attr_seq,
                                                                                        discrete=discrete)
        return best_distribution, p1, p2, parament1, parament2, parament3
    except (ValueError, TypeError, ZeroDivisionError) as e:
        print(f"error: {e}")
        return None, None, None, None, None, None


def edge_attribute_sequence(G, attribute):
    return list(nx.get_edge_attributes(G, attribute).values())


def average_edge_attribute(G, attribute):
    attribute_seq = edge_attribute_sequence(G, attribute)
    return np.mean(attribute_seq)


def fitting_edge_attribute_sequence(G, attribute, discrete=False):
    edge_attr_seq = edge_attribute_sequence(G, attribute)
    try:
        best_distribution, p1, p2, parament1, parament2, parament3 = fitting_parameters(edge_attr_seq,
                                                                                        discrete=discrete)
        return best_distribution, p1, p2, parament1, parament2, parament3
    except (ValueError, TypeError, ZeroDivisionError) as e:
        print(f"Error: {e}")
        return None, None, None, None, None, None


def basic_stats_individual(G_name):
    print(G_name.split('\\')[-1])
    G = read_graph(G_name)
    E, N = number_of_nodes_edge(G)

    dens = density(G)
    avg_degree = average_degree(G)
    fit_degree_seq = fitting_degree_sequence(G)
    fit_degree_fraction_seq = fitting_degree_fraction_sequence(G)

    avg_shortest_path_length = average_shortest_path_length(G,normalize=False)
    dim = diameter(G,normalize=False)

    avg_cc = average_clustering_coefficient(G)

    d_prsr, d_prsp, vc_prsr, vc_prsp, vc_prsr_wgt, vc_prsp_wgt = (
        attributes_assortativity_coefficient(G))
    # assor_coef_degree = assortativity_coefficient(G, attribute='degree')
    # 
    # assor_coef_visit_count = assortativity_coefficient(G, attribute='visit_count')
    # assor_coef_visit_count_weighted = assortativity_coefficient(G, attribute='visit_count',
    #                                                             edge_weight='movement_count')
    # 
    # assor_coef_visit_fraction = assortativity_coefficient(G, attribute='visit_fraction')
    # assor_coef_visit_fraction_weighted = assortativity_coefficient(G, attribute='visit_fraction',
    #                                                                edge_weight='movement_count')

    # avg_visit_count = average_node_attribute(G, attribute='visit_count')
    # fit_visit_count_seq = fitting_node_attribute_sequence(G, attribute='visit_count', discrete=True)
    #
    # avg_visit_fraction = average_node_attribute(G, attribute='visit_fraction')
    # fit_visit_fraction_seq = fitting_node_attribute_sequence(G, attribute='visit_fraction')
    #
    # avg_movement_count = average_edge_attribute(G, attribute='movement_count')
    # fit_movement_count_seq = fitting_edge_attribute_sequence(G, attribute='movement_count', discrete=True)
    #
    # avg_movement_fraction = average_edge_attribute(G, attribute='movement_fraction')
    # fit_movement_fraction_seq = fitting_edge_attribute_sequence(G, attribute='movement_fraction')

    return {'user_id': G.graph['user_id'],
             'number_of_nodes': N,
             'number_of_edges': E,

             'density': dens,
             'average_degree': avg_degree,

             'degree_distribution': fit_degree_seq[0],
             'degree_p1': fit_degree_seq[1],
             'degree_p2': fit_degree_seq[2],
             'degree_parament1': fit_degree_seq[3],
             'degree_parament2': fit_degree_seq[4],
             'degree_parament3': fit_degree_seq[5],

             'degree_fraction_distribution': fit_degree_fraction_seq[0],
             'degree_fraction_p1': fit_degree_fraction_seq[1],
             'degree_fraction_p2': fit_degree_fraction_seq[2],
             'degree_fraction_parament1': fit_degree_fraction_seq[3],
             'degree_fraction_parament2': fit_degree_fraction_seq[4],
             'degree_fraction_parament3': fit_degree_fraction_seq[5],

             'average_shortest_path_length': avg_shortest_path_length,
             'diameter': dim,

             'average_clustering_coefficient': avg_cc,

             'degree_r': d_prsr, 'degree_p': d_prsp,
             'visit_count_r': vc_prsr, 'visit_count_p': vc_prsp,
             'visit_count_r_weighted': vc_prsr_wgt, 'visit_count_p_weighted': vc_prsp_wgt,
             # 'visit_fraction_r': vf_prsr, 'visit_fraction_p': vf_prsp,
             # 'visit_fraction_r_weighted': vf_prsr_wgt, 'visit_fraction_p_weighted': vf_prsp_wgt
             }
            # ,
            # {'user_id': G.graph['user_id'],
            #
            #  'average_visit_count': avg_visit_count,
            #  'visit_count_distribution': fit_visit_count_seq[0],
            #  'visit_count_p1': fit_visit_count_seq[1],
            #  'visit_count_p2': fit_visit_count_seq[2],
            #  'visit_count_parament1': fit_visit_count_seq[3],
            #  'visit_count_parament2': fit_visit_count_seq[4],
            #  'visit_count_parament3': fit_visit_count_seq[5],
            #
            #  'average_visit_fraction': avg_visit_fraction,
            #  'visit_fraction_distribution': fit_visit_fraction_seq[0],
            #  'visit_fraction_p1': fit_visit_fraction_seq[1],
            #  'visit_fraction_p2': fit_visit_fraction_seq[2],
            #  'visit_fraction_parament1': fit_visit_fraction_seq[3],
            #  'visit_fraction_parament2': fit_visit_fraction_seq[4],
            #  'visit_fraction_parament3': fit_visit_fraction_seq[5],
            #
            #  'average_movement_count': avg_movement_count,
            #  'movement_count_distribution': fit_movement_count_seq[0],
            #  'movement_count_p1': fit_movement_count_seq[1],
            #  'movement_count_p2': fit_movement_count_seq[2],
            #  'movement_count_parament1': fit_movement_count_seq[3],
            #  'movement_count_parament2': fit_movement_count_seq[4],
            #  'movement_count_parament3': fit_movement_count_seq[5],
            #
            #  'average_movement_fraction': avg_movement_fraction,
            #  'movement_fraction_distribution': fit_movement_fraction_seq[0],
            #  'movement_fraction_p1': fit_movement_fraction_seq[1],
            #  'movement_fraction_p2': fit_movement_fraction_seq[2],
            #  'movement_fraction_parament1': fit_movement_fraction_seq[3],
            #  'movement_fraction_parament2': fit_movement_fraction_seq[4],
            #  'movement_fraction_parament3': fit_movement_fraction_seq[5]}



def basic_stats(INPUT_PATH=r'./result/graph', OUTPUT_PATH=r'./result/basic_stats', num_processes=1):
    # If the output folder does not exist, an empty one is generated
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    # # Collect all file paths
    file_paths = []
    for root, dirs, files in os.walk(INPUT_PATH):
        for file in files:
            file_paths.append(os.path.join(root, file))

    if num_processes and num_processes > 1:
        with Pool(processes=num_processes) as pool:
            stats_list = pool.map(basic_stats_individual, file_paths)
    else:
        stats_list = [basic_stats_individual(file_path) for file_path in file_paths]

    # topological_stats_list, attribute_stats_list = zip(*stats_list)

    # topological_stats_list = []
    # attribute_stats_list = []

    # for file in file_paths:
    #     print(file)
    #     topo_stats_idv, attr_stats_idv = basic_stats_individual(file)
    #     topological_stats_list.append(topo_stats_idv)
    #     attribute_stats_list.append(attr_stats_idv)

    topological_stats_result = pd.DataFrame(stats_list,
                                            columns=['user_id', 'number_of_nodes', 'number_of_edges',

                                                     'density', 'average_degree',
                                                     'degree_distribution', 'degree_p1', 'degree_p2',
                                                     'degree_parament1', 'degree_parament2', 'degree_parament3',
                                                     'degree_fraction_distribution', 'degree_fraction_p1',
                                                     'degree_fraction_p2', 'degree_fraction_parament1',
                                                     'degree_fraction_parament2',
                                                     'degree_fraction_parament3',

                                                     'average_shortest_path_length', 'diameter',

                                                     'average_clustering_coefficient',

                                                     'degree_r', 'degree_p',
                                                     'visit_count_r', 'visit_count_p',
                                                     'visit_count_r_weighted', 'visit_count_p_weighted',
                                                     # 'visit_fraction_r', 'visit_fraction_p',
                                                     # 'visit_fraction_r_weighted', 'visit_fraction_p_weighted'
                                                     ])

    # attribute_stats_result = pd.DataFrame(attribute_stats_list, columns=['user_id',
    #
    #                                                                      'average_visit_count',
    #                                                                      'visit_count_distribution',
    #                                                                      'visit_count_p1', 'visit_count_p2',
    #                                                                      'visit_count_parament1',
    #                                                                      'visit_count_parament2',
    #                                                                      'visit_count_parament3',
    #
    #                                                                      'average_visit_fraction',
    #                                                                      'visit_fraction_distribution',
    #                                                                      'visit_fraction_p1', 'visit_fraction_p2',
    #                                                                      'visit_fraction_parament1',
    #                                                                      'visit_fraction_parament2',
    #                                                                      'visit_fraction_parament3',
    #
    #                                                                      'average_movement_count',
    #                                                                      'movement_count_distribution',
    #                                                                      'movement_count_p1', 'movement_count_p2',
    #                                                                      'movement_count_parament1',
    #                                                                      'movement_count_parament2',
    #                                                                      'movement_count_parament3',
    #
    #                                                                      'average_movement_fraction',
    #                                                                      'movement_fraction_distribution',
    #                                                                      'movement_fraction_p1', 'movement_fraction_p2',
    #                                                                      'movement_fraction_parament1',
    #                                                                      'movement_fraction_parament2',
    #                                                                      'movement_fraction_parament3'
    #                                                                      ])

    topological_stats_result.to_csv(os.path.join(OUTPUT_PATH, 'topological_stats.csv'), index=False)
    # attribute_stats_result.to_csv(os.path.join(OUTPUT_PATH, 'attribute_stats.csv'), index=False)


def assortativity_coefficient_individual(G_name):
    print(G_name.split('\\')[-1])
    G = read_graph(G_name)

    d_prsr, d_prsp, vc_prsr, vc_prsp, vc_prsr_wgt, vc_prsp_wgt = (
        attributes_assortativity_coefficient(G))

    return {'user_id': G.graph['user_id'],
             'degree_r': d_prsr, 'degree_p': d_prsp,
             'visit_count_r': vc_prsr, 'visit_count_p': vc_prsp,
             'visit_count_r_weighted': vc_prsr_wgt, 'visit_count_p_weighted': vc_prsp_wgt,
             }


def assortativity_coefficient_stats(INPUT_PATH=r'./result/graph', OUTPUT_PATH=r'./result/basic_stats'):
    # If the output folder does not exist, an empty one is generated
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    # # Collect all file paths
    file_paths = []
    for root, dirs, files in os.walk(INPUT_PATH):
        for file in files:
            file_paths.append(os.path.join(root, file))

    with Pool(processes=max(1, (os.cpu_count() or 1) - 2)) as pool:
        stats_list = pool.map(assortativity_coefficient_individual, file_paths)

    topological_stats_result = pd.DataFrame(stats_list,
                                            columns=['user_id',
                                                     'degree_r', 'degree_p',
                                                     'visit_count_r', 'visit_count_p',
                                                     'visit_count_r_weighted', 'visit_count_p_weighted'
                                                     ])
    topological_stats_result.to_csv(os.path.join(OUTPUT_PATH, 'assortativity_coefficient_stats.csv'), index=False)


def merge_topological_features_assortativity_coefficient(STAT_DIR):
    t_df=pd.read_csv(os.path.join(STAT_DIR, 'topological_stats.csv'))
    a_df=pd.read_csv(os.path.join(STAT_DIR, 'assortativity_coefficient_stats.csv'))
    t_df.drop(columns=['degree_r', 'degree_p','visit_count_r', 'visit_count_p','visit_count_r_weighted', 'visit_count_p_weighted'], inplace=True)
    new_t_df=t_df.merge(a_df, on='user_id', how='left')
    new_t_df.to_csv(os.path.join(STAT_DIR, 'topological_stats.csv'),index=False)



