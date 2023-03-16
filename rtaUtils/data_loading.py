from paths import *
import pandas as pd
import datetime

# Configuration parameters
# Column projection
cols_sort = ['vectorId', 'fpId', 'flightDate', 'aerodromeOfDeparture',
             'latitude', 'longitude', 'altitude', 'timestamp', 'track', 'ground']
cols_sorted = ('vectorId', 'fixed_timestamp', 'fixed_altitude', 'ordenInicial', 'ordenFinal')


# Preprocessing
def load_raw_data_sort(date: datetime.datetime) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Loads and preprocess one day of raw trajectory data to sort individual trajectories
    
    Args:
        date: Date of raw data to be loaded
    """
    file_path = list(raw_data_path_v.glob(f'fDate={date.date()}/*.parquet'))[0]
    data = pd.read_parquet(file_path, columns = cols_sort)

    # Eliminar outliers claramente fuera del rango geográfico
    data = data[data.longitude.apply(lambda x: -10 < x < 30)
                & data.latitude.apply(lambda x: 35 < x < 57)]
    data = (data.drop_duplicates(subset=['fpId','timestamp'], keep='first')
                .drop_duplicates(subset=['latitude', 'longitude'], keep='first')
                .sort_values(by=['fpId','timestamp'])
                .reset_index(drop=True).reset_index()
                .rename({'index':'ordenInicial'}, axis=1))
    return data

def calculate_indexes(flights: pd.DataFrame) -> pd.DataFrame:
    """Calculates position index of individual trajectories

    Retrieves the position of the first and last vectors of each trajectory
    within a dataframe

    Args:
        flights: Dataframe with trajectory data
    """
    indices = (flights.reset_index()
                      .groupby(['fpId'])
                      .agg({'index': [min,max]})
                      .reset_index())
    return indices

def load_sorted_data(month: str) -> pd.DataFrame:
    """Loads one month of sorted data and joins it with raw and weather data
    
    Args:
        month: Month of sorted data to be loaded, in format 'YYYYMM'
    """
    raw_name = f'fDate={month[:4]}-{month[4:]}'

    sorted_data = pd.concat([pd.read_parquet(f, columns=cols_sorted)
                             for f in sorted_data_path.glob(f'{month}[0-9][0-9].parquet')])\
                    .rename({'fixed_timestamp':'timestamp', 'fixed_altitude':'altitude'}, axis=1)

    raw_file_paths = list(raw_data_path_v.glob(f'{raw_name}*/*.parquet'))
    raw_data = pd.concat([pd.read_parquet(path).drop(['altitude','timestamp'], axis=1)
                          for path in raw_file_paths])
    output_data = pd.merge(raw_data, sorted_data, on='vectorId', how='inner')

    # Join weather data
    weather_file_paths = list(raw_data_path_w.glob(f'{raw_name}*/*.parquet'))
    weather_data = pd.concat([pd.read_parquet(path)
                              for path in weather_file_paths])
    output_data = pd.merge(output_data, weather_data, on='reportId', how='inner')

    return output_data

# ML utils

def load_clean_data(month: str, airport: str = None, columns: list = None) -> pd.DataFrame:
    """Loads one month of cleaned data
    
    Args:
        month: Month of sorted data to be loaded, in format 'YYYYMM'
        airport: ICAO code of an airport. Only trajectories coming from this
            airport will be included (optional)
        columns: Project only a set of desired columns
    """
    flights_files = list(clean_data_path.glob(f'{month}.parquet'))
    if columns:
        data = pd.concat([pd.read_parquet(x, columns=columns) for x in flights_files])
    else:
        data = pd.concat([pd.read_parquet(x) for x in flights_files])

    if airport and airport != '*':
        data = data[data.aerodromeOfDeparture.isin(airport.split(','))]

    return data

# Model generation

def load_final_data(month: str, dataset: str, airport: str = None, sampling: int = 0) -> pd.DataFrame:
    """Loads one month of definitive data

    If sampling is provided, final data is loaded from the corresponding folder.
    Assumes that sampled data have been saved into disk before.
    
    Args:
        month: Month of sorted data to be loaded, in format 'YYYYMM'
        dataset: Set of data to be loaded (either 'train', 'test' or 'val')
        airport: ICAO code of an airport. Only trajectories coming from this
            airport will be included (optional)
        sampling: Sampling period of the data
    """
    if sampling:
        paths = sampled_data_path.glob(f's{sampling}/{month}.{dataset}.parquet')
    else:
        paths = final_data_path.glob(f'{month}.{dataset}.parquet')
    data = pd.concat([pd.read_parquet(x) for x in paths])
    if airport and airport != '*':
        data = data[data.aerodromeOfDeparture == airport]
    data = data.sort_values(by=['fpId','timestamp'])

    return data
