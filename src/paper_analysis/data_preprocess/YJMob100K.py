import sys
import time
from multiprocessing import Pool
import pandas as pd
import numpy as np
import os
import trackintel as ti
from datetime import datetime, timedelta
from paper_analysis.utils.io import read_location_sequence
import geopandas as gpd
from shapely.geometry import Point

# parameters
# positionfixs to staypoint
# Within the eight neighborhoods are also considered to be stays sqrt(2)=1.4142135 roaming_dis_max = 1.5
# If it is just considered to be a stay within the same grid, set roaming_dis_max = 0.5
roaming_dis_max = 1.5  # (>= meters) the maximum roaming distance allowed for two points within the same stay.
stay_min = 40  # (>= minutes) the minimum duration of a stay.
gap_max = 30*24*60  # (> minutes) the maximum time difference between two consecutive records for them to be considered within the same stay. Delete if exceeded

# does it stay long enough for the activity?
stay_activity_min = stay_min  # (minutes)

# clustering staypoints to detect locations
# eps = 100  # (meters)
# minp = 1

# filter user
duration_min = 60  # (days)
trip_per_day_min = 1


def _convert_datetime(date, time):
    base_date = datetime(2023, 1, 1)
    date_delta = timedelta(days=int(date))
    time_delta = timedelta(minutes=int(time * 30))
    return base_date + date_delta + time_delta


def split_trajectory(INPUT_FILE, POS_DIR):
    if not os.path.exists(POS_DIR):
        os.makedirs(POS_DIR)

    traj_group = pd.read_csv(INPUT_FILE)
    print('read data')

    # formatted data
    # Generate the base_date column
    base_date = pd.to_datetime('2023-01-01')
    traj_group['base_date'] = base_date

    # Generate the date_delta column
    traj_group['date_delta'] = pd.to_timedelta(traj_group['d'], unit='d')

    # Generate the time_delta column
    traj_group['time_delta'] = pd.to_timedelta(traj_group['t'] * 30, unit='m')

    # Generate the new datetime column
    traj_group['datetime'] = traj_group['base_date'] + traj_group['date_delta'] + traj_group['time_delta']

    # traj_group['tracked_at'] = traj_group.apply(lambda row: _convert_datetime(row['d'], row['t']), axis=1)
    traj_group.drop(columns=['d', 't', 'base_date', 'date_delta', 'time_delta'], inplace=True)
    # traj_group.drop(columns=['d', 't'], inplace=True)
    traj_group.rename(columns={'uid': 'user_id', 'datetime': 'tracked_at', 'x': 'longitude', 'y': 'latitude'},
                      inplace=True)

    print('formatted data')

    # split data by id
    groups = traj_group.groupby('user_id')
    for name, group in groups:
        group.to_csv(os.path.join(POS_DIR, str(name) + '.csv'), index=False)


def positionfixes_to_staypoints_individual(file_name, SP_DIR):
    # EPSG:3857 WGS 1984/Pseudo-Mercator; EPSG:4326 WGS 1984;
    # The coordinate range of WGS 1984 coordinate system is [(-180.0 -90.0),(180.0 90.0)], this dataset is a 200*200 grid, it will be out of range, the grid coordinates beyond 180 will become negative, so we change to projected coordinate system.
    pfs = ti.read_positionfixes_csv(file_name, index_col=None, tz='UTC', crs='EPSG:3857')
    # Generate new stops when the distance is greater than or equal to dist_threshold; set to 0.5 since new stops
    # should not be generated when this dataset is equal to zero; time_threshold and gap_threshold are in minutes
    pfs, sp = pfs.generate_staypoints(distance_metric='euclidean', dist_threshold=roaming_dis_max,
                                      time_threshold=stay_min,
                                      gap_threshold=gap_max,
                                      agg_method='most_frequent',
                                      n_jobs=1)
    user_id = file_name.split('\\')[-1].split('.')[0]
    print(user_id)
    sp.to_csv(os.path.join(SP_DIR, 'sp_' + user_id + '.csv'))


def positionfixes_to_staypoints(POS_DIR='./psitionfixes', SP_DIR='./staypoints'):
    if not os.path.exists(SP_DIR):
        os.makedirs(SP_DIR)

    # for file_name in os.listdir(POS_DIR):
    #     print(file_name)
    #     start_time = time.time()
    #     positionfixes_to_staypoints_individual(os.path.join(POS_DIR, file_name), SP_DIR)
    #     print('sp took {} seconds'.format(time.time() - start_time))

    with Pool(processes=os.cpu_count()) as pool:
        for file_name in os.listdir(POS_DIR):
            pool.apply_async(positionfixes_to_staypoints_individual,
                             args=(os.path.join(POS_DIR, file_name), SP_DIR))
        pool.close()
        pool.join()


def staypoints_to_location_sequence_individual(file_name, LOCATION_SEQUENCE_DIR):
    def is_file_empty(file_name):
        with open(file_name, 'r') as file:
            lines = file.readlines()
            # Check if there is only one line (header) or no data lines
            return len(lines) <= 1

    if is_file_empty(file_name):
        print('The staypoint file has no data.')
        return None

    sp = ti.read_staypoints_csv(file_name, index_col='id')

    # add activity_flag column and drop non-activity staypoints
    sp = sp.create_activity_flag(method='time_threshold', time_threshold=stay_activity_min)
    sp = sp[sp['is_activity']]

    # clustering to create location sequence
    sp['location_id'] = sp.groupby('geom').ngroup()

    # rest: 2-8; work: 8-19; leisure:19-2
    sp = ti.analysis.location_identifier(sp, method='OSNA')

    # format
    sp['location_longitude'] = sp['geom'].x
    sp['location_latitude'] = sp['geom'].y
    sp['longitude'] = sp['geom'].x
    sp['latitude'] = sp['geom'].y
    loc_seq = sp[['user_id', 'started_at', 'finished_at', 'longitude', 'latitude',
                  'location_id', 'location_longitude', 'location_latitude', 'purpose']]

    # write location sequence
    user_id = file_name.split('\\')[-1].split('.')[0].split('_')[1]
    print(user_id)
    loc_seq.to_csv(os.path.join(LOCATION_SEQUENCE_DIR, 'L_' + user_id + '.csv'), index=False)


def staypoints_to_location_sequence(SP_DIR='./staypoints', LOCATION_SEQUENCE_DIR='./location_sequence'):
    if not os.path.exists(LOCATION_SEQUENCE_DIR):
        os.makedirs(LOCATION_SEQUENCE_DIR)

    # for file_name in os.listdir(SP_DIR):
    #     print(file_name)
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
    if len(loc_seq) < 2:
        print('the length of L dataframe is less than 2.')
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
    if len(OD) < 2:
        print('the length of ODs dataframe is less than 2.')
        return None

    is_qualified = select_user(OD)
    if not is_qualified:
        print('The user duration or frequency do not satisfy the conditions.')
        return None

    output_OD_file_name = file_name.split('\\')[-1].replace('L', 'OD')
    print(output_OD_file_name)
    OD.to_csv(os.path.join(OD_DIR, output_OD_file_name), index=False)

    # The locs are not filtered and there may be locs that do not appear in the OD data.
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
    #     start_time = time.time()
    #     location_sequence_to_OD_individual(os.path.join(LOCATION_SEQUENCE_DIR, file_name),OD_DIR, LOC_DIR)
    #     print('sp took {} seconds'.format(time.time() - start_time))

    with Pool(processes=os.cpu_count()) as pool:
        for file_name in os.listdir(LOCATION_SEQUENCE_DIR):
            pool.apply_async(location_sequence_to_OD_individual,
                             args=(os.path.join(LOCATION_SEQUENCE_DIR, file_name), OD_DIR, LOC_DIR))
        pool.close()
        pool.join()



