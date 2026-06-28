from multiprocessing import Pool
import pandas as pd
import numpy as np
import os

import pytz
import trackintel as ti
from datetime import datetime, timedelta
from paper_analysis.data_preprocess.preprocess_utils.checkin_analysis import (
    create_activity_flag_checkin,
    location_identifier_checkin,
)
import time

# does it stay long enough for the activity?
stay_activity_min = 20  # (minutes)

# If the time interval between ODs is greater than gap_max, the OD is unreasonable and is deleted.
gap_max = 24*60

# filter user
duration_min = 60
trip_per_day_min = 0.2


def split_checkins(CHECKIN_FILE, SP_DIR):
    if not os.path.exists(SP_DIR):
        os.makedirs(SP_DIR)
    # Read the dataset
    traj_group = pd.read_csv(CHECKIN_FILE, sep='\t', header=None,
                             names=['user_id', 'venue_id', 'utc_time', 'timezone_offset'])
    print('Read checkin data')

    # formatted checkin data
    # errors: If 'coerce', then invalid parsing will be set as NaT.
    traj_group['utc_time'] = pd.to_datetime(traj_group['utc_time'], format='%a %b %d %H:%M:%S +0000 %Y',
                                            errors='coerce')
    traj_group['timezone_offset_delta'] = pd.to_timedelta(traj_group['timezone_offset'], unit='m', errors='coerce')

    print(f"Number of rows of checkin data: {len(traj_group)}")
    traj_group = traj_group.dropna()
    print(f"Number of rows after deletion of an exception (e.g. incorrect time format): {len(traj_group)}")

    # Calculate local time
    traj_group['tracked_at'] = traj_group['utc_time'] + traj_group['timezone_offset_delta']

    # Add time zone information to local time
    traj_group['tracked_at'] = traj_group['tracked_at'].dt.strftime('%Y-%m-%d %H:%M:%S') + traj_group[
        'timezone_offset'].apply(lambda x: f"{x // 60:+03d}:00")

    traj_group.drop(['utc_time', 'timezone_offset', 'timezone_offset_delta'], axis=1, inplace=True)
    print('Format checkin data')

    # split data by id
    groups = traj_group.groupby('user_id')
    row_num = 0
    for name, group in groups:
        # Sort the group by the timestamp column in ascending order
        group = group.sort_values(by='tracked_at', ascending=True)
        # print(len(group))
        row_num += len(group)
        group.to_csv(os.path.join(SP_DIR, 'sp_' + str(name) + '.csv'), index=False)
    print('total checkins by all user:', row_num)


def staypoints_to_location_sequence_individual(file_name, LOCATION_SEQUENCE_DIR):
    #  read file
    sp = pd.read_csv(file_name)
    sp['tracked_at'] = sp['tracked_at'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S%z'))

    # add activity_flag column and drop non-activity staypoints
    sp = create_activity_flag_checkin(staypoints=sp, method='time_threshold', time_threshold=stay_activity_min,
                                      activity_column_name='activity')
    sp = sp[sp['activity']]

    # Check if there are any active staypoints
    if sp.empty:
        print('After filtering, no active staypoints were found.')
        return None

    sp['location_id'], unique = pd.factorize(sp['venue_id'])
    # method = 'FREQ' or 'OSNA'
    sp = location_identifier_checkin(sp, method='OSNA')

    # generate the useful columns
    loc_seq = sp[['user_id', 'tracked_at', 'venue_id', 'location_id', 'purpose']]

    # write location sequence
    output_file_name = file_name.split('\\')[-1].replace('sp', 'L')
    print(output_file_name)
    loc_seq.to_csv(os.path.join(LOCATION_SEQUENCE_DIR, output_file_name), index=False)


def staypoints_to_location_sequence(SP_DIR='./staypoints', LOCATION_SEQUENCE_DIR='./location_sequence'):
    if not os.path.exists(LOCATION_SEQUENCE_DIR):
        os.makedirs(LOCATION_SEQUENCE_DIR)

    # for file_name in os.listdir(SP_DIR):
    #     print(file_name)
    #     # start_time = time.time()
    #     staypoints_to_location_sequence_individual(os.path.join(SP_DIR, file_name), LOCATION_SEQUENCE_DIR)
    #     # print('sp took {} seconds'.format(time.time() - start_time))

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


def location_sequence_to_OD_individual(file_name, OD_DIR, LOC_DIR, is_max_gap_filter=True):
    print(file_name.split('\\')[-1])
    # read the location sequence
    loc_seq = pd.read_csv(file_name)
    loc_seq['tracked_at'] = loc_seq['tracked_at'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S%z'))

    if loc_seq.empty:
        print('The length of L is less than 1, and cannot be converted to OD data.')
        return None

    OD = pd.DataFrame()
    OD['user_id'] = loc_seq['user_id']
    OD['origin_time'] = loc_seq['tracked_at']
    OD['destination_time'] = loc_seq['tracked_at'].shift(-1)
    OD['origin_id'] = loc_seq['location_id']
    OD['destination_id'] = loc_seq['location_id'].shift(-1)
    OD.dropna(inplace=True)

    # filter OD. If the travel time corresponding to the OD is greater than 24 hours, it may mean that the data is missing, and the OD is unreliable. Delete this OD.
    if is_max_gap_filter:
        OD = OD[OD['destination_time'] - OD['origin_time'] < pd.Timedelta(gap_max, unit="minutes")]

    if OD.empty:
        print('After filtering, no ODs were found.')
        return None

    # filter user
    is_qualified = select_user(OD)
    if not is_qualified:
        print('The user duration or frequency do not satisfy the conditions')
        return None

    output_OD_file_name = file_name.split('\\')[-1].replace('L', 'OD')
    OD.to_csv(os.path.join(OD_DIR, output_OD_file_name), index=False)

    locs = loc_seq[['user_id', 'venue_id', 'location_id', 'purpose']].drop_duplicates(
        ['location_id'], keep='first')
    output_locs_file_name = file_name.split('\\')[-1].replace('L', 'loc')
    locs.to_csv(os.path.join(LOC_DIR, output_locs_file_name), index=False)


def location_sequence_to_OD(LOCATION_SEQUENCE_DIR='./location_sequence',
                            OD_DIR='./OD', LOC_DIR='./locations',is_max_gap_filter=True):
    if not os.path.exists(OD_DIR):
        os.makedirs(OD_DIR)
    if not os.path.exists(LOC_DIR):
        os.makedirs(LOC_DIR)

    # for file_name in os.listdir(LOCATION_SEQUENCE_DIR):
    #     start_time = time.time()
    #     print(file_name)
    #     location_sequence_to_OD_individual(os.path.join(LOCATION_SEQUENCE_DIR, file_name), OD_DIR, LOC_DIR,is_max_gap_filter)
    #     print('sp took {} seconds'.format(time.time() - start_time))

    with Pool(processes=os.cpu_count()) as pool:
        for file_name in os.listdir(LOCATION_SEQUENCE_DIR):
            pool.apply_async(location_sequence_to_OD_individual,
                             args=(os.path.join(LOCATION_SEQUENCE_DIR, file_name), OD_DIR, LOC_DIR,is_max_gap_filter))
        pool.close()
        pool.join()


def merge_loc_attribute(LOC_DIR, VENUE_ATTRIBUTES_FILE):
    start_time = datetime.now()
    df_list = []
    for filename in os.listdir(LOC_DIR):
        file_path = os.path.join(LOC_DIR, filename)
        df = pd.read_csv(file_path)
        df_list.append(df)
    loc_all_user = pd.concat(df_list, ignore_index=True)
    print('Read location data, Time taken:', datetime.now() - start_time)

    # read venue attributes data
    venue_attr = pd.read_csv(VENUE_ATTRIBUTES_FILE, sep='\t', header=None,
                             names=['venue_id', 'location_latitude', 'location_longitude', 'category', 'country_code'])
    print('Read venue data, Time taken:', datetime.now() - start_time)

    # merge locs with venue attributes data
    locs_attributes = pd.merge(loc_all_user, venue_attr, left_on='venue_id', right_on='venue_id', how='left')
    # format
    locs_attributes = locs_attributes[['user_id', 'location_id', 'location_longitude', 'location_latitude',
                                       'purpose', 'venue_id', 'category', 'country_code']]

    print('Merge location attributes, Time taken:', datetime.now() - start_time)

    # Split data by userid and output
    for userid, group in locs_attributes.groupby('user_id'):
        output_file_path = os.path.join(LOC_DIR, f'loc_{userid}.csv')
        group.to_csv(output_file_path, index=False)
    print('Merge location attributes, Time taken:', datetime.now() - start_time)


def rename_by_adding_country(LOC_DIR, OD_DIR):
    for file_name in os.listdir(LOC_DIR):
        user_id = file_name.split('.')[0].split('_')[1]
        loc_i = pd.read_csv(os.path.join(LOC_DIR, file_name))
        country = loc_i['country_code'].mode()[0]
        os.rename(os.path.join(OD_DIR, f'OD_{user_id}.csv'), os.path.join(OD_DIR, f'OD_{country}_{user_id}.csv'))
        os.rename(os.path.join(LOC_DIR, file_name), os.path.join(LOC_DIR, f'loc_{country}_{user_id}.csv'))
        print(user_id)


def generate_L_for_spatial_features_individual(L_file_name,loc_file_name,NEW_LOCATION_SEQUENCE_DIR='./new_location_sequence'):

    loc = pd.read_csv(loc_file_name)
    L = pd.read_csv(L_file_name)
    L_formated=pd.DataFrame()
    L_formated['user_id'] = L['user_id']
    L_formated['started_at'] = L['tracked_at']
    L_formated['finished_at'] = L['tracked_at'].shift(-1)

    L_formated['location_id'] = L['location_id']
    L_formated=pd.merge(L_formated, loc[['location_id','location_longitude','location_latitude','purpose']], on='location_id')

    L_formated[['longitude','latitude']]=L_formated[['location_longitude','location_latitude']]
    L_formated=L_formated[['user_id',	'started_at',	'finished_at',	'longitude',	'latitude',	'location_id',	'location_longitude',	'location_latitude',	'purpose'
]]

    new_file_name=L_file_name.split('\\')[-1]
    # print(new_file_name)
    L_formated.to_csv(os.path.join(NEW_LOCATION_SEQUENCE_DIR,new_file_name), index=False)


def generate_L_for_spatial_features(LOCATION_SEQUENCE_DIR='./location_sequence',LOC_DIR='./loc',NEW_LOCATION_SEQUENCE_DIR='./new_location_sequence'):
    if not os.path.exists(NEW_LOCATION_SEQUENCE_DIR):
        os.makedirs(NEW_LOCATION_SEQUENCE_DIR)

    for file_name in os.listdir(LOC_DIR):
        # start_time = time.time()
        print(file_name)

        # Extract the userid correctly
        userid = file_name.split('.')[0].split('_')[-1]
        L_file_name = 'L_{}.csv'.format(userid)

        # Call the function with the correct formatted filename
        generate_L_for_spatial_features_individual(
            os.path.join(LOCATION_SEQUENCE_DIR, L_file_name),
            os.path.join(LOC_DIR, file_name),
            NEW_LOCATION_SEQUENCE_DIR
        )

        # print('sp took {} seconds'.format(time.time() - start_time))



