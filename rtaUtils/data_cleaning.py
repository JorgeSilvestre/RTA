import pandas as pd

from common import haversine_np
from paths import *

DIFF_SPEED_THRESHOLD = 0.5 # Km per second
DIFF_ALTITUDE_THRESHOLD = 120  # Feet per second

max_altitude_threshold = 46000
default_wind_degrees   = -25
default_visibility     = 7
default_wind_speed     = 0
default_temperature    = -50
default_operator       = 'none'

# Outliers

def detect_outliers(data: pd.DataFrame):
    """Flags individual vectors of a trajectory if 3D position or time values
       are deemed incorrect

       The time, position, and altitude values are compared with the latest 
       correct vector, and flagged as incorrect following simple heuristics
        Time: If timestamp is duplicated
        Altitude: If the required vertical speed to move from previous altitude 
            to current is unfeasible
        Position: If the required horizontal speed to move from previous
            position to current is unfeasible
        If time difference is higher than 60 seconds, these heuristics are not applied

    Args:
        data: Dataframe with data from one trajectory    
    """
    # The first vector is assumed to be correct
    first = data.iloc[0]
    latest = (first.latitude, first.longitude, first.timestamp, first.altitude, first.vspeed)
    flags = [False]

    for idx, row in list(data.iterrows())[1:]:
        # Time check
        diff_time = row.timestamp - latest[2]
        if diff_time == 0:
            flags.append(True)
            continue
        if diff_time > 60:
            flags.append(False)
            latest = (row.latitude, row.longitude, row.timestamp, row.altitude, row.vspeed)
            continue

        # Altitude check
        diff_altitude = abs(latest[3] - row.altitude) / diff_time
        exceeds_altitude = (diff_altitude > ((abs(latest[4]) + abs(row.vspeed)) / 120 + DIFF_ALTITUDE_THRESHOLD))
        if exceeds_altitude:
            flags.append(True)
            continue

        # Position check
        diff = haversine_np(float(row.latitude), float(row.longitude), 
                            latest[0], latest[1]) / (row.timestamp - latest[2])
        exceeds_position = diff > DIFF_SPEED_THRESHOLD
        if exceeds_position:
            flags.append(True)
            continue
        
        latest = (row.latitude, row.longitude, row.timestamp, row.altitude, row.vspeed)
        flags.append(False)
    
    if suma := sum(flags):
        print(data.iloc[0].fpId, f'{suma:3}/{len(flags):5}/{data.shape[0]:5}')

    data['is_outlier'] = flags

    return data


# Data checking

def modify_data_types(data: pd.DataFrame) -> pd.DataFrame:
    """Transform Numpy data types for numeric values (memory optimization)"""
    # Fix data types
    data['latitude']         = data.latitude.astype('float64')
    data['longitude']        = data.longitude.astype('float64')
    data['speed']            = data.speed.astype('float32')
    data['wind_dir_degrees'] = data.wind_dir_degrees.astype('float32')
    data['altitude']         = data.altitude.astype('float32')
    data['vspeed']           = data.vspeed.astype('float32')
    data['delay']            = data.delay.astype('int32')
    data['RTA']              = data.RTA.astype('int32')
    
    return data


def fill_missing_data(data: pd.DataFrame) -> pd.DataFrame:
    """Fills missing values using linear interpolation or default values"""
    # Assumes that data are sorted by flight/leg and timestamp.

    # Pending: Ensure that is only applied inside of individual trajectories
    # Only few vectors mid-trajectory have problems
    data['vspeed'] = data.vspeed.interpolate(method='linear')
    data['vspeed'] = data.vspeed.astype('int32')
    data['speed'] = data.speed.interpolate(method='linear')
    data['speed'] = data.speed.astype('int32')

    # Fill missing data using default values
    # Fill null visibility values, which indicate a perfect visibility condition (>=7 miles)
    data['visibility_statute_mi'] = data['visibility_statute_mi'].fillna(default_visibility)
    # Fill null wind_dir_degrees with "impossible" numeric value
    data['wind_dir_degrees'] = data.wind_dir_degrees.fillna(default_wind_degrees)
    # Fill null wind_speed_kt with 0 knots
    data['wind_speed_kt'] = data.wind_speed_kt.fillna(default_wind_speed)
    # Fill null wind_gust_kt with 0 knots
    data = data.rename({'wing_gust_kt':'wind_gust_kt'}, axis=1)
    data['wind_gust_kt'] = data.wind_gust_kt.fillna(default_wind_speed)
    # Fill null operators with empty strings
    # data['operator'] = data.operator.fillna(default_operator)
    data = data.dropna(subset=['operator'])

    # Fill null delay with zeros
    data['delay'] = data.delay.fillna(0)

    # Eliminar una vez integrado en el proceso previo
    # Fill null change_indicator with BASE
    data['change_indicator'] = data.change_indicator.fillna('BASE')
    data['cloud_base_ft_agl'] = data.cloud_base_ft_agl.fillna(0)
    data['cloud_type'] = data.cloud_type.fillna('')
    
    
    return data


def remove_incorrect(data: pd.DataFrame) -> pd.DataFrame:
    """Removes vectors with incorrect values

    Currently only altitude and speed null values are considered
    """
    data = data.dropna(subset=['altitude', 'speed'])

    return data


def include_additional_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Adds derived columns"""
    data = data.copy()
    # Separate min and max temperatures
    # data['min_temp'] = data.temperature.apply(lambda x: default_temperature if len(x)!=2 else x[1]['min_temp_c'])
    # data['max_temp'] = data.temperature.apply(lambda x: default_temperature if len(x)!=2 else x[0]['max_temp_c'])
    
    # Sky condition
    # data = data[~data.sky_condition.isna()]
    # data['clouds'] = data.sky_condition.apply(lambda x: x[0]['cloud_base_ft_agl'] if x[0]['cloud_base_ft_agl'] else -10)
    # data['sky_status'] = data.sky_condition.apply(lambda x: x[0]['sky_cover'] )
    
    data['day_of_week'] = pd.to_datetime(data.flightDate).dt.day_of_week.copy().astype('int16')
    # data['holiday']   = data.day_of_week.isin((5,6)).copy()
    
    data['hav_distance'] = haversine_np(data['latitude'].values, data['longitude'].values)
    
    aggs = data.groupby(['fpId']).agg(to_max=('actualTakeOffTime','max'))
    
    data['time_of_day']  = pd.to_datetime(data.join(aggs, on='fpId', how='left').to_max, unit = 's').apply(assign_time_of_day)#.copy()
    
    return data
    

def drop_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Removes unnecesary columns"""
    data = data.drop(['wx_string','sky_condition', 'temperature'], axis=1)
    return data


def drop_duplicates_sort(data: pd.DataFrame) -> pd.DataFrame:
    """Removes duplicated vectors and sorts by trajectory and timestamp"""
    data = data.drop_duplicates(subset=['fpId','latitude','longitude','timestamp'])\
               .sort_values(by=['fpId', 'timestamp']).reset_index(drop=True)
    return data


def assign_time_of_day(x):
    """Assigns time of day depending on the hour"""
    if x.hour < 7:
        return 'night'
    elif x.hour < 13:
        return 'morning'
    elif x.hour < 20:
        return 'evening'
    else:
        return 'night'