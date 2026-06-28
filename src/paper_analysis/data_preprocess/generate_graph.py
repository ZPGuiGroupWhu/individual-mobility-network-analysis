import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns
import networkx as nx
import math
from paper_analysis.utils.io import read_OD, read_location, write_graph, read_graph, print_graph, draw_graph
import warnings
from multiprocessing import Pool


def _create_graph_dataframe(OD, graph_type, self_loop):
    # If no self-loop is included, delete the rows with the same OD point position.
    if not self_loop:
        OD = OD.drop(OD[OD['origin_id'] == OD['destination_id']].index)

    if OD.empty:
        print('OD dataframe is empty')
        return None, None

    # calculating the count and fraction of edges between stay points
    OD['movement_count'] = 1  # Each trip record is counted as 1
    edges = pd.DataFrame()
    if graph_type == 'undirected':
        # For undirected graphs, create pairs of origin and destination points. Undirected graphs do not need to
        # distinguish between O and d
        OD['pair'] = [tuple(sorted([s, t])) for s, t in zip(OD['origin_id'], OD['destination_id'])]
        # Group by pairs and sum the movement counts
        edges = OD.groupby(by='pair')['movement_count'].sum().reset_index()
        # Split the pair column into origin_id and destination_id
        edges[['origin_id', 'destination_id']] = edges['pair'].apply(lambda x: pd.Series(list(x)))
        # Drop the pair column
        edges.drop(columns='pair', inplace=True)
    elif graph_type == 'directed':
        # For directed graphs, group by origin and destination and sum the movement counts
        edges = OD.groupby(by=['origin_id', 'destination_id'])['movement_count'].sum().reset_index()
    else:
        # If the graph type is invalid, issue a warning
        warnings.warn("invalid graph type")
    # Calculate the movement fraction for each edge
    edges['movement_fraction'] = edges['movement_count'] / sum(edges['movement_count'])

    # calculating the count and fraction of visit to stay points
    loc = pd.DataFrame()
    loc['location_id'] = pd.concat([OD['origin_id'], OD['destination_id']])  # Combine all origin and destination points
    # Every time a stop point is visited, it will be consecutively recorded in O and D respectively. there is
    # duplication of visit count
    loc['visit_count'] = 1
    nodes = loc.groupby(by='location_id')['visit_count'].sum().to_frame()
    # Handle consecutive visits to the same point to avoid double counting
    spare = pd.DataFrame()
    spare['location_id'] = OD['origin_id']
    spare['spare'] = (OD['origin_id'] - OD['destination_id'].shift(1))  # Check for consecutive visits
    spare = spare[spare['spare'] == 0]
    if not spare.empty:
        spare_count = spare.groupby(by='location_id')['spare'].count().to_frame()  # Count consecutive visits
        nodes = pd.merge(nodes, spare_count, on='location_id', how='left').fillna(0)
        nodes['visit_count'] = nodes['visit_count'] - nodes[
            'spare']  # Subtract consecutive visit counts from total visits
        nodes.drop(columns='spare', inplace=True)
    # Reset the index
    nodes.reset_index(inplace=True)
    # Calculate the visit fraction for each location
    nodes['visit_fraction'] = nodes['visit_count'] / sum(nodes['visit_count'])
    return edges, nodes


def generate_graph_individual(OD_file_name, loc_attribute_file_name, graph_file_name, graph_type, self_loop):
    print(OD_file_name.split('\\')[-1])
    # read OD and location dataframe
    OD = read_OD(OD_file_name)
    loc_attribute = read_location(loc_attribute_file_name)

    # generate dataframe of edge and node
    global G
    # If no self-loop is included, delete the rows with the same OD point position
    edges_df, nodes_df = _create_graph_dataframe(OD, graph_type, self_loop)

    if edges_df is None:
        # print('edges_df is None')
        return None

    # generate graph by edge dataframe
    if graph_type == 'undirected':
        # Create an undirected graph from the edge list
        G = nx.from_pandas_edgelist(edges_df, source='origin_id', target='destination_id',
                                    edge_attr=['movement_count', 'movement_fraction'])
    elif graph_type == 'directed':
        # Create a directed graph from the edge list
        G = nx.from_pandas_edgelist(edges_df, source='origin_id', target='destination_id',
                                    edge_attr=['movement_count', 'movement_fraction'],
                                    create_using=nx.DiGraph())
    else:
        # Issue a warning if the graph type is invalid
        warnings.warn("invalid graph type")

    # Add node attributes from the location attribute dataframe
    nodes_df = nodes_df.merge(loc_attribute, how='left', on='location_id')
    # Combine latitude and longitude into a tuple and store it in 'loc'
    nodes_df['loc'] = nodes_df[['location_latitude', 'location_longitude']].apply(lambda x: (x.iloc[0], x.iloc[1]),
                                                                                  axis=1)
    # Drop unnecessary columns from the nodes dataframe
    nodes_df.drop(columns=['user_id', 'location_latitude', 'location_longitude'], inplace=True)
    # Set node attributes in the graph using the nodes dataframe
    nx.set_node_attributes(G, nodes_df.set_index('location_id').to_dict(orient='index'))
    # Add graph attributes, such as userid
    G.graph.update({'user_id': OD['user_id'][0]})
    # Relabel nodes to integers based on their degree (number of connections). The smaller the number, the greater
    # the degree
    G = nx.convert_node_labels_to_integers(G, ordering='decreasing degree', label_attribute='location_id')
    # # write the graph
    # print_graph(G)
    # draw_graph(G)
    # plt.show()
    write_graph(G, graph_file_name)


def generate_graph(OD_PATH=r'./data/OD', LOC_PATH=r'./data/locations', GRAPH_PATH=r'./results/graph',
                   graph_type='undirected', self_loop=True):
    # If the output folder does not exist, an empty one is generated
    output_dir = os.path.join(GRAPH_PATH, str(graph_type) + '_' + str(self_loop) + ' ' + 'self loop_graph')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with Pool(processes=os.cpu_count()) as pool:
        for root, dirs, files in os.walk(OD_PATH):
            if files:
                for file in files:
                    pool.apply_async(generate_graph_individual,
                                     args=(
                                         os.path.join(root, file),
                                         os.path.join(LOC_PATH, file.replace('OD', 'loc')),
                                         os.path.join(output_dir, file.split('.')[0].replace('OD', 'G') + '.json'),
                                         graph_type,
                                         self_loop
                                     )
                                     )
        pool.close()
        pool.join()

    # for root, dirs, files in os.walk(OD_PATH):
    #     if files:
    #         for file in files:
    #             print(file)
    #             generate_graph_individual(os.path.join(root, file),
    #                                       os.path.join(LOC_PATH, file.replace('OD', 'loc')),
    #                                       os.path.join(output_dir, file.split('.')[0].replace('OD', 'G') + '.json'),
    #                                       graph_type,
    #                                       self_loop)


def calculate_self_loop_ratio_individual(OD_file_name):
    """Calculate the number of self-loop records and their ratio."""
    OD = read_OD(OD_file_name)  # Read the OD data from the file
    user_id = OD['user_id'].iloc[0]  # Get the user ID from the first record
    total_records = len(OD)  # Total number of records
    self_loop_count = OD[OD['origin_id'] == OD['destination_id']].shape[0]  # Count self-loops
    self_loop_ratio = self_loop_count / total_records if total_records > 0 else 0  # Calculate the ratio
    return user_id, self_loop_count, self_loop_ratio  # Return user ID, self-loop count, and ratio


def calculate_self_loop_ratio(OD_PATH=r'./data/OD', OUTPUT_PATH=r'./result'):
    """Calculate self-loop ratios for multiple OD files."""
    user_self_loop_ratios = []  # List to store results

    # Ensure the output directory exists
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # Use a pool of processes to calculate self-loop ratios in parallel
    with Pool(processes=max(1, (os.cpu_count() or 1) - 2)) as pool:
        tasks = []  # List to hold file paths
        for root, dirs, files in os.walk(OD_PATH):  # Walk through the directory
            for file in files:
                OD_file = os.path.join(root, file)  # Get the full file path
                tasks.append(OD_file)  # Add file path to tasks

        # Calculate self-loop ratios for all files
        results = pool.map(calculate_self_loop_ratio_individual, tasks)  # Parallel processing

        # Create DataFrame and specify column names
        results_df = pd.DataFrame(results, columns=['user_id', 'self_loop_count', 'self_loop_ratio'])

        # Save results to a CSV file
        results_df.to_csv(os.path.join(OUTPUT_PATH, 'self_loop_ratios.csv'), index=False)



