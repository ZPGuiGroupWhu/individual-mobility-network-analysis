import sys
import time
from multiprocessing import Pool
import pandas as pd
import numpy as np
import os
from paper_analysis.utils.io import read_triplegs
import pytz
import trackintel as ti
from datetime import datetime, timedelta
from paper_analysis.utils.io import read_location_sequence
import geopandas as gpd
from shapely.geometry import Point
from paper_analysis.utils.gislib import point_haversine_dist
from paper_analysis.data_preprocess.preprocess_utils.CDC import CDC
from shapely.geometry import MultiPoint
from trackintel.preprocessing.util import angle_centroid_multipoints
from trackintel.geogr import check_gdf_planar

# does it stay long enough for the activity?
stay_activity_min = 20  # (minutes)

# clustering staypoints to detect locations
eps = 100  # (meters)
# If minp is greater than 1, there are some noise points and the location id is empty. You need to write additional code to assign the location id and coordinates of the noise points.
minp = 1
is_use_CDC = True
max_distance = 500  # (meters)
k_num = 10
ratio = 0.8

# If the time interval between ODs is greater than gap_max, the OD is unreasonable and is deleted.
gap_max = 24 * 60

# filter user
# Triplegs1(shenzhen,laiwu,guiyang)
# duration_min = 180  # (days)
# trip_per_day_min = 0.5

# Triplegs2
duration_min = 180  # (days)
trip_per_day_min = 0.5

# Triplegs
# duration_min = 360  # (days)
# trip_per_day_min = 0.5

# Triplegs1_2019(shenzhen,laiwu,guiyang)
# duration_min = 300  # (days)
# trip_per_day_min = 0.2

def delete_anomalous_triplegs(TRIPLEGS_DIR, CLEANED_TRIPLEGS_DIR, speed_threshold=200 / 3.6, interval_min=60):
    """
    Remove tripleg data that may have drifted in position. If the average speed corresponding to a tripleg exceeds the threshold, the tripleg is considered to have positional drift.
    :param TRIPLEGS_DIR:
    :param CLEANED_TRIPLEGS_DIR:
    :param speed_threshold:
    :param interval_min: # Minimum time interval between the origin and destination in a triple. Delete short trips, such as moving a car. This will cause self-loops.
    :return:
    """
    for root, dirs, files in os.walk(TRIPLEGS_DIR):
        if files:
            # If the output folder does not exist, create it
            output_root = os.path.join(CLEANED_TRIPLEGS_DIR, root.split('\\')[-1])
            if not os.path.exists(output_root):
                os.makedirs(output_root)

            for file_name in files:
                triplegs = read_triplegs(os.path.join(root, file_name))
                # Remove duplicate and missing rows
                triplegs.dropna(how='any', axis=0, inplace=True)
                triplegs.drop_duplicates(subset='origin_time', keep='first', inplace=True)
                # Units in meters
                triplegs['OD_dis'] = point_haversine_dist(triplegs['origin_lng'], triplegs['origin_lat'],
                                                          triplegs['destination_lng'], triplegs['destination_lat'])
                # Units in seconds
                triplegs['OD_time_diff'] = (triplegs['destination_time'] - triplegs['origin_time']).dt.total_seconds()
                #  Delete short trips, such as moving a car. This will cause self-loops.
                triplegs = triplegs[triplegs['OD_time_diff'] > interval_min]
                triplegs['speed'] = triplegs['OD_dis'] / triplegs['OD_time_diff']
                # delete anomalous triplegs
                triplegs = triplegs[triplegs['speed'] < speed_threshold]
                print(file_name)
                # write
                triplegs.drop(columns=['OD_dis', 'OD_time_diff', 'speed'], inplace=True)
                triplegs.to_csv(os.path.join(output_root, file_name), index=False)


def triplegs_to_staypoints_individual(file_name, SP_DIR):
    triplegs = pd.read_csv(file_name)
    sp = gpd.GeoDataFrame()
    sp['id'] = range(len(triplegs))
    sp['user_id'] = triplegs['TID']
    sp['started_at'] = triplegs['destination_time']
    sp['finished_at'] = triplegs['origin_time'].shift(-1)
    sp['geom'] = gpd.points_from_xy(triplegs['destination_lng'], triplegs['destination_lat'])
    sp.dropna(inplace=True)
    user_id = file_name.split('\\')[-1].split('.')[0]
    city = file_name.split('\\')[-2]
    print(user_id)
    if len(sp) < 1:
        print('The number of staypoints is less than 1, no output will be made.')
        return None
    sp.to_csv(os.path.join(SP_DIR, 'sp_' + str(city) + '_' + str(user_id) + '.csv'), index=False)


def triplegs_to_staypoints(TRIPLEG_DIR='./triplegs', SP_DIR='./staypoints'):
    if not os.path.exists(SP_DIR):
        os.makedirs(SP_DIR)

    # for root, dirs, files in os.walk(TRIPLEG_DIR):
    #     if files:
    #         for file_name in files:
    #             print(file_name)
    #             start_time = time.time()
    #             triplegs_to_staypoints_individual(os.path.join(root, file_name), SP_DIR)
    #             print('sp took {} seconds'.format(time.time() - start_time))

    with Pool(processes=os.cpu_count()) as pool:
        for root, dirs, files in os.walk(TRIPLEG_DIR):
            if files:
                for file_name in files:
                    pool.apply_async(triplegs_to_staypoints_individual,
                                     args=(os.path.join(root, file_name), SP_DIR))
        pool.close()
        pool.join()


def dbscan_cdc(sp, max_distance=500, k_num=10, ratio=0.8):
    # Filter staypoints with max_distance greater than the specified threshold
    sp_dense = sp[sp['max_distance'] > max_distance].copy()
    # Check if the number of dense points is less than k_num
    if len(sp_dense) <= k_num:
        print('There are no clusters that need to be processed using the cdc algorithm (not enough k-neighbors.)')
        sp.drop(columns=['max_distance'], inplace=True)  # Drop max_distance column and return
        return sp

    # Filter staypoints with max_distance less than or equal to the specified threshold
    sp_sparse = sp[sp['max_distance'] <= max_distance].copy()

    # Apply CDC clustering to dense points and assign cluster IDs
    coords = pd.DataFrame()
    coords['longitude'] = sp_dense['geom'].x
    coords['latitude'] = sp_dense['geom'].y

    sp_dense['location_id_DBSCAN_CDC'] = CDC(k_num, ratio, coords.to_numpy())
    sp_dense['location_id_DBSCAN_CDC'] = sp_dense['location_id_DBSCAN_CDC'].map(int)  # Convert to integer

    # Assign unique IDs to sparse points based on existing location IDs
    sp_sparse['location_id_DBSCAN_CDC'] = pd.factorize(sp_sparse['location_id'])[0] + sp_dense[
        'location_id_DBSCAN_CDC'].max() + 1

    # Combine dense and sparse points, sorting by 'started_at'
    sp = pd.concat([sp_dense, sp_sparse]).sort_values(by='started_at')

    # Format: set location_id to the new DBSCAN cluster IDs
    sp['location_id'] = sp['location_id_DBSCAN_CDC']
    sp.drop(columns=['location_id_DBSCAN_CDC'], inplace=True)  # Drop temporary column

    # create locations as grouped staypoints
    temp_sp = sp[["user_id", "location_id", sp.geometry.name]]
    locs = temp_sp.dissolve(by=["user_id", "location_id"], as_index=False)  # Aggregate staypoints

    # Calculate centroids based on the geometry type
    if check_gdf_planar(locs):
        locs["center"] = locs.geometry.centroid  # Use built-in centroid method for planar geometries
    else:
        # error of wrapping e.g. mean([-180, +180]) -> own function needed
        locs["center"] = angle_centroid_multipoints(locs.geometry)

    # Drop unnecessary columns from locations
    locs.drop(columns=['user_id', 'geom'], inplace=True)
    sp.drop(columns=['center', 'max_distance'], inplace=True)

    # Merge centroids back into the original DataFrame
    sp = sp.merge(locs, on='location_id', how='left')

    return sp


def staypoints_to_location_sequence_individual(file_name, LOCATION_SEQUENCE_DIR):
    # timezone is beijing
    sp = ti.read_staypoints_csv(file_name, index_col='id', tz='Asia/Shanghai', crs='EPSG:4326')

    # add activity_flag column and drop non-activity staypoints
    sp = sp.create_activity_flag(method='time_threshold', time_threshold=stay_activity_min,
                                 activity_column_name='activity')
    sp = sp[sp['activity']]

    # Check if there are any active staypoints
    if sp.empty:
        print('After filtering, no active staypoints were found.')
        return None

    # clustering to create location sequence
    sp, locs = sp.generate_locations(method='dbscan', epsilon=eps, num_samples=minp, activities_only=True)

    # merge location dataframe and staypoints dataframe
    locs['location_id'] = locs.index
    locs.drop(columns=['user_id', 'extent'], inplace=True)
    sp = sp.merge(locs, on='location_id', how='left')

    if is_use_CDC:
        sp = dbscan_cdc(sp, max_distance=max_distance, k_num=k_num, ratio=ratio)

    sp = ti.analysis.location_identifier(sp, method='OSNA')
    # generate the useful columns
    sp['location_longitude'] = sp['center'].x
    sp['location_latitude'] = sp['center'].y
    sp['longitude'] = sp['geom'].x
    sp['latitude'] = sp['geom'].y

    loc_seq = sp[['user_id', 'started_at', 'finished_at', 'longitude', 'latitude',
                  'location_id', 'location_longitude', 'location_latitude','purpose']]

    # write location sequence
    output_file_name = file_name.split('\\')[-1].replace('sp', 'L')
    print(output_file_name)
    loc_seq.to_csv(os.path.join(LOCATION_SEQUENCE_DIR, output_file_name), index=False)


def staypoints_to_location_sequence(SP_DIR='./staypoints', LOCATION_SEQUENCE_DIR='./location_sequence'):
    if not os.path.exists(LOCATION_SEQUENCE_DIR):
        os.makedirs(LOCATION_SEQUENCE_DIR)

    # for file_name in os.listdir(SP_DIR):
    #     # print(file_name)
    #     start_time = time.time()
    #     staypoints_to_location_sequence_individual(os.path.join(SP_DIR, file_name), LOCATION_SEQUENCE_DIR)
    #     print('sp took {} seconds'.format(time.time() - start_time))

    with Pool(processes=os.cpu_count()) as pool:
        for file_name in os.listdir(SP_DIR):
            pool.apply_async(staypoints_to_location_sequence_individual,
                             args=(os.path.join(SP_DIR, file_name), LOCATION_SEQUENCE_DIR))
        pool.close()
        pool.join()


def select_user(OD):
    # duration
    first_stay = OD.iloc[0]['origin_time']
    last_stay = OD.iloc[-1]['origin_time']
    days_between = (last_stay - first_stay).days + 1
    # frequency
    trip_per_day = len(OD) / days_between

    if (days_between < duration_min) or (trip_per_day < trip_per_day_min):
        return False

    return True


def location_sequence_to_OD_individual(file_name, OD_DIR, LOC_DIR):
    loc_seq = read_location_sequence(file_name)
    if len(loc_seq) < 1:
        print('the length of L is less than 1')
        return None

    OD = pd.DataFrame()
    OD['user_id'] = loc_seq['user_id']
    OD['origin_time'] = loc_seq['finished_at']
    OD['destination_time'] = loc_seq['started_at'].shift(-1)
    OD['origin_id'] = loc_seq['location_id']
    OD['destination_id'] = loc_seq['location_id'].shift(-1)
    OD.dropna(inplace=True)

    # filter OD. If the travel time corresponding to the OD is greater than 24 hours, it may mean that the data is missing, and the OD is unreliable. Delete this OD.
    # OD = OD[OD['destination_time'] - OD['origin_time'] < pd.Timedelta(gap_max, unit="minutes")]

    # filter user
    is_qualified = select_user(OD)
    if not is_qualified:
        print('The user duration or frequency do not satisfy the conditions')
        return None

    output_OD_file_name = file_name.split('\\')[-1].replace('L', 'OD')
    print(output_OD_file_name)
    OD.to_csv(os.path.join(OD_DIR, output_OD_file_name), index=False)

    locs = loc_seq[['user_id', 'location_id', 'location_longitude', 'location_latitude', 'purpose']].drop_duplicates(
        ['location_id'], keep='first')
    output_locs_file_name = file_name.split('\\')[-1].replace('L', 'loc')
    locs.to_csv(os.path.join(LOC_DIR, output_locs_file_name), index=False)


def location_sequence_to_OD(LOCATION_SEQUENCE_DIR='./location_sequence', OD_DIR='./OD', LOC_DIR='./locations'):
    if not os.path.exists(OD_DIR):
        os.makedirs(OD_DIR)
    if not os.path.exists(LOC_DIR):
        os.makedirs(LOC_DIR)

    # for file_name in os.listdir(LOCATION_SEQUENCE_DIR):
    #     print(file_name)
    #     # start_time = time.time()
    #     location_sequence_to_OD_individual(os.path.join(LOCATION_SEQUENCE_DIR, file_name), OD_DIR, LOC_DIR)
    #     # print('sp took {} seconds'.format(time.time() - start_time))

    with Pool(processes=os.cpu_count()) as pool:
        for file_name in os.listdir(LOCATION_SEQUENCE_DIR):
            pool.apply_async(location_sequence_to_OD_individual,
                             args=(os.path.join(LOCATION_SEQUENCE_DIR, file_name), OD_DIR, LOC_DIR))
        pool.close()
        pool.join()


# def location_sequence_to_OD_individual(file_name, OD_DIR, LOC_DIR):
#     loc_seq = read_location_sequence(file_name)
#     if len(loc_seq) < 2:
#         print('the length of L is less than 2')
#         return 0
#
#     OD = pd.DataFrame()
#     OD['user_id'] = loc_seq['user_id']
#     OD['origin_time'] = loc_seq['finished_at']
#     OD['destination_time'] = loc_seq['started_at'].shift(-1)
#     OD['origin_id'] = loc_seq['location_id']
#     OD['destination_id'] = loc_seq['location_id'].shift(-1)
#     OD.dropna(inplace=True)
#
#     # filter OD. If the travel time corresponding to the OD is greater than 24 hours, it may mean that the data is missing, and the OD is unreliable. Delete this OD.
#     OD = OD[OD['destination_time'] - OD['origin_time'] < pd.Timedelta(gap_max, unit="minutes")]
#
#     # filter user
#     is_qualified = select_user(OD)
#     if not is_qualified:
#         # print('The user duration or frequency do not satisfy the conditions')
#         # return None
#         return 0
#
#     output_OD_file_name = file_name.split('\\')[-1].replace('L', 'OD')
#     # print(output_OD_file_name)
#     # OD.to_csv(os.path.join(OD_DIR, output_OD_file_name), index=False)
#     #
#     # locs = loc_seq[['user_id', 'location_id', 'location_longitude', 'location_latitude', 'purpose']].drop_duplicates(
#     #     ['location_id'], keep='first')
#     # output_locs_file_name = file_name.split('\\')[-1].replace('L', 'loc')
#     # locs.to_csv(os.path.join(LOC_DIR, output_locs_file_name), index=False)
#     return 1
#
#
# def location_sequence_to_OD(LOCATION_SEQUENCE_DIR='./location_sequence', OD_DIR='./OD', LOC_DIR='./locations'):
#     if not os.path.exists(OD_DIR):
#         os.makedirs(OD_DIR)
#     if not os.path.exists(LOC_DIR):
#         os.makedirs(LOC_DIR)
#     user_number = 0
#
#     for file_name in os.listdir(LOCATION_SEQUENCE_DIR):
#         # print(file_name)
#         # start_time = time.time()
#         selected = location_sequence_to_OD_individual(os.path.join(LOCATION_SEQUENCE_DIR, file_name), OD_DIR, LOC_DIR)
#         # print('sp took {} seconds'.format(time.time() - start_time))
#         user_number += selected
#     return user_number
#     # with Pool(processes=os.cpu_count()) as pool:
#     #     for file_name in os.listdir(LOCATION_SEQUENCE_DIR):
#     #         pool.apply_async(location_sequence_to_OD_individual,
#     #                          args=(os.path.join(LOCATION_SEQUENCE_DIR, file_name), OD_DIR, LOC_DIR))
#     #     pool.close()
#     #     pool.join()

def count_city_files(OD_PATH):
    """Print the number of OD files per city."""
    city_file_counts = {}

    for filename in os.listdir(OD_PATH):
        city = filename.split('_')[1]
        if city in city_file_counts:
            city_file_counts[city] += 1
        else:
            city_file_counts[city] = 1

    sum_counts = sum(city_file_counts.values())
    print(sum_counts)
    for city, count in city_file_counts.items():
        print(f"City {city} has {count} files")





