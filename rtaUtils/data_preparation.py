import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

import data_loading, paths

# Window generation
GAP_THRESHOLD       = 180
TEMP_DIST_THRESHOLD = 5*60
SHUFFLE_BUFFER_SIZE = 50000 # 25000

def sample_data(dataframe: pd.DataFrame, sampling: int):
    """Sample data based on timestamp using a sampling period
    
    Vectors are divided into buckets of 'sampling' seconds. Only the first
    vector in each bucket is kept.

    Args:
        dataframe: Input trajectory data
        sampling: Sampling period used to downsample trajectory data
    """
    dataframe['bucket']   = dataframe.timestamp // sampling
    dataframe   = (dataframe.sort_values(['fpId', 'timestamp'])
                            .drop_duplicates(subset = ['fpId', 'bucket'], keep = 'first')
                            .drop('bucket', axis=1).reset_index(drop=True))
    return dataframe


def identify_gaps(dataframe: pd.DataFrame, desc: str = None) -> np.ndarray:
    """Checks for gaps longer than a given threshold in trajectories

    Calculates the difference between timestamps of adjacent vectors

    In order to ensure time regularity inside of the generated windows,
    we effectively divide trajectories into segments if gaps longer 
    than a threshold detected between adjacent vectors. Sliding window
    is applied then on segments, and not on the whole trajectory

    Current implementation discards any window with vectors from two or
    more different segments

    Args:
        dataframe: Input trajectory data
        desc: If not None, displays a progress bar
    """
    conv = np.array([1,-1])
    diffs = []

    if desc:
        it = tqdm(dataframe[['fpId','timestamp']].groupby('fpId'), desc=desc)
    else:
        it = dataframe[['fpId','timestamp']].groupby('fpId')
    for i, group in it:     
        if group.shape[0] > 1:      
            diffs += [0] + list(np.convolve(group.timestamp.astype('int').values, conv, 'same'))[1:]
        else:
            # Directly force new segment on flights with only one vector 
            # (otherwise the convolution operation would fail)
            diffs += [GAP_THRESHOLD + 1]

    return (np.array(diffs) > GAP_THRESHOLD).astype(float).cumsum()


def get_windows(dataframe: pd.DataFrame, lookback: int, encoders, scaler, features:dict):
    """Generate sequences of size lookback of adjacent vectors

    Windows with vectors from two or more different segments are discarded.

    Args:
        dataframe: Trajectories data
        lookback: Length of the sliding window
        encoders: Scikit-learn Encoder object for the passed categoric features
        scaler: Scikit-learn Scaler object for the passed feature configuration
        features: Dictionary with list of strings identifying features of
            each type:
            { numeric:[feat1, ...], categoric:[...], objective:[...] }
    """
    numeric_feat = features.get('numeric')
    categoric_feat = features.get('categoric')
    objective = features.get('objective')

    dataframe['segment'] = identify_gaps(dataframe)

    indices = dataframe.reset_index(drop=True).reset_index().groupby(['fpId','segment'])\
                       .aggregate({'index':['min','max']}).reset_index()
    for feat in categoric_feat:
        dataframe[feat]  = encoders[feat].transform(dataframe[feat]).reshape(-1,1)
    scaled_data = scaler.transform(dataframe[numeric_feat+categoric_feat+objective])

    dataset = [tf.data.Dataset.from_tensor_slices(scaled_data[start:end+1,:])
               for f, (fpId, segment, start, end) in list(indices.iterrows())]
    dataset = tf.data.Dataset.from_tensor_slices(dataset)
    dataset = dataset.flat_map(lambda x: x.window(lookback, shift=1, stride=1, drop_remainder=True))
    dataset = dataset.flat_map(lambda window: window.batch(lookback))

    return dataset


def get_windows_at_time(dataframe: pd.DataFrame, time: int, lookback: int, 
                        encoders, scaler, features:dict):
    """Generates the last available window for each trajectory, before a given 
    cutting point (currently, by time before landing)
    
    Args:
        dataframe: Trajectories data
        time: The cutting point (in seconds before landing)
        lookback: Length of the sliding window
        encoders: Scikit-learn Encoder object for the passed categoric features
        scaler: Scikit-learn Scaler object for the passed feature configuration
        features: Dictionary with list of strings identifying features of
            each type:
            { numeric:[feat1, ...], categoric:[...], objective:[...] }
    """
    numeric_feat = features.get('numeric')
    categoric_feat = features.get('categoric')
    objective = features.get('objective')

    # Pending: Currently, if the window generated from a trajectory contains vectors 
    # from more than one segment, the trajectory is skipped. We should be able to
    # generate new candidate windows (within the defined threshold) for this trajectory
    # so it is represented in the dataset

    dataframe['segment'] = identify_gaps(dataframe)

    final_data = []

    time_data = dataframe[dataframe.RTA>=time*60].copy()          # By time
    # time_data = dataframe[dataframe.hav_distance>=time].copy()  # By distance  

    for idx, data in time_data.groupby('fpId'): # ,'segment'
        if data.shape[0] < lookback:
            continue

        if time == 0:
            d = data.iloc[:lookback].copy()
        else:
            d = data.iloc[-lookback:].copy()
            # Windows that are too far from the cutting point are removed
            if d.RTA.iloc[-1] - time * 60 > TEMP_DIST_THRESHOLD:   # By time
                continue

            # if d.hav_distance.iloc[-1] - time  > 10:   # By distance
            #     continue

        # Windows with vectors from different segments are removed
        if len(d.segment.unique()) > 1:
            continue

        for feat in categoric_feat:
            d[feat]  = encoders[feat].transform(d[feat].values).reshape(-1,1)
        scaled_data_test  = scaler.transform(d[numeric_feat+categoric_feat+objective])
        final_data.append(scaled_data_test)
    dataset = tf.data.Dataset.from_tensor_slices(final_data)
    # print(time, '==>', len(final_data))

    return dataset


def generate_save_windows(month: str, lookback: int, sampling: int, features:dict, airport: str = '*' ):
    """Persists windowed data on disk using Tensorflow Dataset API

    This function loads a month's data for all or a given airport, downsamples 
    the trajectory data and builds the examples using a sliding window of 
    length lookback by calling get_windows function. Then, the results are
    written into disk to reduce the computational burden during training 
    and evaluation processes
    
    Args:
        month: Month of final data to be loaded, in format 'YYYYMM'
        lookback: Length of the sliding window
        sampling: Sampling period used to downsample trajectory data
        airport: ICAO code of an airport, or * for all available airports
        features: Dictionary with list of strings identifying features of
            each type:
            { numeric:[feat1, ...], categoric:[...], objective:[...] }
    """
    numeric_feat = features.get('numeric')
    categoric_feat = features.get('categoric')
    objective = features.get('objective')

    encoders = joblib.load(paths.utils_path / 'encoder.joblib')
    scaler   = joblib.load(paths.utils_path / f'scaler_{len(numeric_feat+categoric_feat)}.joblib')

    for dataset in ('train', 'test', 'val'):
        dataframe = data_loading.load_final_data(month, dataset, airport, sampling)

        for ap in tqdm(sorted(dataframe.aerodromeOfDeparture.unique()), desc=f'{month} {dataset.ljust(5)}'):
            airport_data = dataframe[dataframe.aerodromeOfDeparture == ap].copy()
            ds = get_windows(airport_data, lookback, encoders, scaler, features)

            tf.data.Dataset.save(ds.shuffle(SHUFFLE_BUFFER_SIZE),
            str(paths.window_data_path / f'data{lookback}_s{sampling}/{dataset}/{month}-{ap}'))
