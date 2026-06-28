"""Spatial and temporal feature extraction from location sequences.

Inputs are ``L_*.csv`` location-sequence files and the corresponding IMN graph
directory used to select included individuals. The main output is
``spatial_stats.csv``. The default release workflow starts from IMN files, so
this module is only run when location-sequence files are available.
"""

import datetime
import os
import pandas as pd
import numpy as np
from scipy import stats
from paper_analysis.utils.gislib import point_haversine_dist, point_euclidean_dist
from multiprocessing import Pool

def read_location_sequence_without_timezone(file_name):
    df = pd.read_csv(file_name)
    df["started_at"] = pd.to_datetime(df["started_at"].str[:-6], format="%Y-%m-%d %H:%M:%S")
    df["finished_at"] = pd.to_datetime(df["finished_at"].str[:-6], format="%Y-%m-%d %H:%M:%S")
    return df


def day_of_week_entropy(L, normalize=False):
    L['day_of_week'] = L['started_at'].dt.dayofweek
    probs = L['day_of_week'].value_counts(normalize=True).values
    entropy = stats.entropy(probs, base=2.0)
    if normalize:
        n_vals = 7
        entropy /= np.log2(n_vals)
    return entropy


def _frequency_time_of_day(L, name='started_at', period=3600):
    """
    Calculate the frequency of stay points in equal time slots over a day (24 hours).

    :param L: DataFrame containing stay points with a datetime column specified by `name`.
    :param name: Column name in the DataFrame representing the start or end time of stay points.
    :param period: Time interval for the slots in seconds (e.g., 3600 for 1 hour).
    :return: Frequency array of stay points in each time slot.
    """

    def time2seconds(t):
        return int(t.hour) * 3600 + int(t.minute) * 60 + int(t.second)

    time_cut = pd.DataFrame()
    time_cut['time'] = L[name].apply(lambda x: time2seconds(x.time()))

    end_time = datetime.datetime.strptime('23:59:59', '%H:%M:%S')
    end_time_seconds = time2seconds(end_time)

    time_slot = list(range(0, end_time_seconds + 1, period))

    time_cut['time_cut'] = pd.cut(time_cut['time'], time_slot, labels=range(len(time_slot) - 1))
    time_cut['time_cut'] = time_cut['time_cut'].fillna(0)
    frequency = time_cut['time_cut'].value_counts(normalize=True).values
    return frequency


def time_of_day_entropy(L, normalize=False):
    probs = _frequency_time_of_day(L)
    entropy = stats.entropy(probs, base=2.0)

    if normalize:
        n_vals = 24
        entropy /= np.log2(n_vals)

    return entropy


def radius_of_gyration(L, method='quantity', distance_metric='haversine'):
    """
    Calculate the radius of gyration for a set of geographical points.

    :param L: DataFrame containing longitude and latitude columns.
    :param method: Method to calculate radius of gyration (currently only 'quantity' is implemented).
    :param distance_metric: Distance metric to use ('euclidean' or 'haversine').
    :return: Radius of gyration as a float.
    """

    if distance_metric == 'euclidean':
        dist_func = point_euclidean_dist
    elif distance_metric == 'haversine':
        dist_func = point_haversine_dist
    else:
        print('Unsupported distance metric name. Using Euclidean distance calculation.')
        dist_func = point_euclidean_dist

    if method == 'quantity':
        lngs_lats = L[['longitude', 'latitude']].values
        center_lng = L['longitude'].mean()
        center_lat = L['latitude'].mean()
        rg = np.sqrt(np.mean([dist_func(lng, lat, center_lng, center_lat) ** 2.0 for lng, lat in lngs_lats]))
        return rg


def location_entropy(L, normalize=False):
    """
    Compute the uncorrelated entropy of a single individual given their TrajDataFrame.

    Parameters
    ----------
    traj : TrajDataFrame
        the trajectories of the individuals.

    normalize : boolean, optional
        if True, normalize the entropy in the range :math:`[0, 1]` by dividing by :math:`log_2(N_u)`, where :math:`N` is the number of distinct locations visited by individual :math:`u`. The default is False.

    Returns
    -------
    float
        the temporal-uncorrelated entropy of the individual
    """
    n = len(L)
    probs = [1.0 * len(group) / n for group in
             L.groupby(by='location_id').groups.values()]
    entropy = stats.entropy(probs, base=2.0)
    if normalize:
        n_vals = len(np.unique(L['location_id'].values, axis=0))
        if n_vals > 1:
            entropy /= np.log2(n_vals)
        else:  # to avoid NaN
            entropy = 0.0
    return entropy


def number_of_trips(L):
    return len(L)


def number_of_locations(L):
    return L['location_id'].nunique()


def trip_length(L, distance_metric='haversine'):
    """
    Calculate the average trip length based on the distance between consecutive geographical points.

    Parameters
    ----------
    L : DataFrame
        A DataFrame containing 'longitude' and 'latitude' columns.

    distance_metric : str, optional
        The distance metric to use ('euclidean' or 'haversine'). Default is 'haversine'.

    Returns
    -------
    float
        The average trip length between consecutive points.
    """
    if len(L) < 2:
        raise ValueError("At least two points are required to calculate trip length.")

    if distance_metric == 'euclidean':
        dist_func = point_euclidean_dist
    elif distance_metric == 'haversine':
        dist_func = point_haversine_dist
    else:
        dist_func = point_euclidean_dist

    lngs_lats = L[['longitude', 'latitude']]
    N = len(lngs_lats)
    total_distance = 0
    for i in range(N - 1):
        total_distance += dist_func(lngs_lats.iloc[i, 0], lngs_lats.iloc[i, 1],
                                    lngs_lats.iloc[i + 1, 0], lngs_lats.iloc[i + 1, 1])
    return total_distance / (N - 1)


def calculate_features_individual(file_name, distance_metric='haversine'):
    L = read_location_sequence_without_timezone(file_name)
    user_id = L['user_id'][0]
    trip_n = number_of_trips(L)
    location_n = number_of_locations(L)
    trip_l = trip_length(L, distance_metric=distance_metric)
    rg = radius_of_gyration(L, distance_metric=distance_metric)

    de = day_of_week_entropy(L, normalize=False)
    te = time_of_day_entropy(L, normalize=False)
    le = location_entropy(L, normalize=False)

    de_norm = day_of_week_entropy(L, normalize=True)
    te_norm = time_of_day_entropy(L, normalize=True)
    le_norm = location_entropy(L, normalize=True)

    feature_individual = {
        'user_id': user_id,

        'number_of_trips': trip_n,
        'number_of_locations': location_n,

        'trip_length': trip_l,

        'radius_of_gyration': rg,

        'day_of_week_entropy': de,
        'time_of_day_entropy': te,
        'location_entropy': le,

        'day_of_week_entropy_normalize': de_norm,
        'time_of_day_entropy_normalize': te_norm,
        'location_entropy_normalize': le_norm,
    }
    print(user_id)
    return feature_individual


def calculate_features(LOCATION_SEQUENCE_DIR=r'./location_sequence', GRAPH_DIR=r'./OD',
                       FEATURES_DIR=r'./result/spatial_features', distance_metric='haversine',
                       num_processes=1):
    # If the output folder does not exist, an empty one is generated
    if not os.path.exists(FEATURES_DIR):
        os.makedirs(FEATURES_DIR)

    selected_file_names = [
        'L_{}.csv'.format(file_name.split('.')[0].split('_')[-1])
        for file_name in os.listdir(GRAPH_DIR)
        if file_name.endswith(".json")
    ]

    # Collect all file paths
    file_paths = []
    distance_metrics = []
    for selected_file_name in selected_file_names:
        file_paths.append(os.path.join(LOCATION_SEQUENCE_DIR, selected_file_name))
        distance_metrics.append(distance_metric)

    if num_processes and num_processes > 1:
        with Pool(processes=num_processes) as pool:
            result_list = pool.starmap(calculate_features_individual, zip(file_paths, distance_metrics))
    else:
        result_list = [
            calculate_features_individual(file_path, distance_metric=metric)
            for file_path, metric in zip(file_paths, distance_metrics)
        ]

    feature_result = pd.DataFrame(data=result_list,
                                  columns=[
                                      'user_id', 'number_of_trips', 'number_of_locations', 'trip_length',
                                      'radius_of_gyration',
                                      'day_of_week_entropy', 'time_of_day_entropy',
                                      'location_entropy',
                                      'day_of_week_entropy_normalize', 'time_of_day_entropy_normalize',
                                      'location_entropy_normalize'
                                  ])

    # feature_result = pd.DataFrame(columns=[
    #     'user_id', 'number_of_trips', 'number_of_locations', 'trip_length', 'radius_of_gyration',
    #     'day_of_week_entropy', 'time_of_day_entropy', 'location_entropy'
    # ])
    #
    # for file_name in file_paths:
    #     feature_individual = calculate_features_individual(file_name, distance_metric=distance_metric)
    #     feature_result.loc[len(feature_result)] = feature_individual

    if 'D1_YJMob100K' in LOCATION_SEQUENCE_DIR or 'dataset_yjmob100k' in LOCATION_SEQUENCE_DIR:
        distance_ratio = 500
        feature_result['trip_length'] = feature_result['trip_length'] * distance_ratio
        feature_result['radius_of_gyration'] = feature_result['radius_of_gyration'] * distance_ratio

    feature_result.to_csv(os.path.join(FEATURES_DIR, 'spatial_stats.csv'), index=False)
    # print(feature_result)



