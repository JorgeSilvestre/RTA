import concurrent
import datetime
import time
import warnings

import numpy as np
import pandas as pd

import data_loading
from common import haversine_np, haversine_np_track
from paths import *

warnings.simplefilter("ignore", category=FutureWarning)

airports = pd.read_csv(airports_file_path, sep = ',', usecols=['id','lat','lon'])

# cols_sort = ['vectorId', 'fpId', 'flightDate', 'aerodromeOfDeparture',
#              'latitude', 'longitude', 'altitude', 'timestamp', 'track', 'ground']

TMA_AREA_MAX = 110
TMA_AREA_MIN = 30
AIRPORT_AREA = 5

MAX_ROTATION = 365
SINGLE_LOOP_ROTATION = 180

# Parámetros del proceso
overlap_fast     = 5
window_size_fast = 20
overlap_slow     = 75
window_size_slow = 101

# No se aplican
# max_kchanges = 200
max_iteraciones = 100


def calculate_distance(array: np.array, angle = False) -> np.array:
    array = array.astype('float32')

    distances = haversine_np(array[1:,0], array[1:,1],
                             array[:-1,0], array[:-1,1],
                             None, None, angle)
    
    # distances = haversine_np_track(array[1:,0], array[1:,1],
    #                          array[:-1,0], array[:-1,1],
    #                          array[1:,2], array[:-1,2], angle)

    return np.concatenate([[0], distances])


def generate_windows(data_size:int, window_size:int, overlap:int, min_index:int=0, max_index:int=0) -> tuple[tuple[int,int,int]]:
    '''Genera ventanas deslizantes a partir del tamaño de una estructura tabular

    Genera índices para las ventanas deslizantes de tamaño window_size aplicadas
    sobre una estructura de datos de longitud data_size.

    Args:
        data_size: Longitud de la estructura de datos
        window_size: Tamaño de la ventana deslizante
        overlap: Número de elementos que se solapan entre ventanas consecutivas
        min_index: Índice a partir del cual se calculan las ventanas
        max_index: Índice hasta el que se generan las ventanas

    Returns:
        Un conjunto de tripletas (num. ventana, índice inicial, índice final)
    '''
    if max_index and max_index <= data_size: #
        data_size = max_index

    windows = []
    number_of_windows = ((data_size-min_index)//(window_size-overlap))+1
    for x in range(number_of_windows):
        window_min = x*(window_size-overlap) + min_index
        window_max = x*(window_size-overlap) + min_index + window_size
        window_max = window_max if window_max <= data_size else data_size
        windows.append((x, window_min, window_max))
        if window_max == data_size:
            break

    return windows


def generate_changes(start:int, end:int, skip:int = 1) -> list[tuple[int,int]]:
    '''
        Genera reemplazos de cada elemento con cada uno de los elementos siguientes
        (sin contar el inmediatamente siguiente) en base a sus índices correlativos
        en la secuencia. Ignora el primer y último elementos.

        start
            Índice del primer elemento a considerar
        end
            Índice del último elemento a considerar
        skip
            Número de elementos al comienzo de la secuencia para los que no se generan cambios

    '''
    changes = [(v1,v2) for v1 in range(start+skip, end-start-2) for v2 in range(v1+2, end-start-1)]
    return changes


def sort_windows(array: np.array, windows: tuple, angle: bool = False) -> np.array:
    for it, start, end in windows:
        data = array[start:end].copy()
        current_dist = initial_dist = calculate_distance(data[:,[4,5,8]], angle).sum()

        # Ojo: índices relativos a la ventana, no a data entero
        changes = generate_changes(0, end-start, skip = 2 if angle else 1)
        # changes = list(np.random.permutation(changes))

        for j in range(max_iteraciones):
            improvements = 0
            for i, (v1, v2) in enumerate(changes):
                candidate = np.concatenate([data[:v1+1],
                                            data[v2:v1:-1],
                                            data[v2+1:]])
                candidate_dist = calculate_distance(candidate[:,[4,5,8]], angle).sum()

                if candidate_dist < current_dist:
                    improvements += 1
                    data, current_dist = candidate, candidate_dist
                    array[start:end] = candidate

            if not improvements:
                break

    return array


def calculate_rotation(tracks: pd.Series) -> float:
    # Rotación = Σ Variación de track entre mensajes adyacentes
    delta = tracks.astype(int).rolling(2,2).apply(lambda x: x.iloc[1]-x.iloc[0])
    delta = delta.apply(lambda x: x if abs(x) < 180 else (x - np.sign(x)*360))
    # Primer elemento es nan
    track_variation = delta.iloc[1:].sum()

    return track_variation


def sort_nearest_vector(array: np.array, stop: str = 'undefined') -> np.array:
    '''
    Args:
        array: Array of vector attributes to be sorted
        stop: ID of the last vector in the trajectory. Once this vector is reached,
            unsorted vectors are concatenated with sorted ones without being sorted.
    '''
    if array.shape[0]==0:
        return array
    # Calculate distance from each vector to each other vector
    distances = []
    for x in array:
        distances.append(haversine_np(array[:,4].astype('float'),
                                      array[:,5].astype('float'),
                                      x[4], x[5]))
    distances = np.array(distances)

    # Starting from the first vector, closest vector is recursively identified,
    # excluding those that were already visited
    mins = [0]
    # Mask used to hide visited vectors
    mask = np.zeros(shape=array.shape[0]).astype(bool)
    while len(mins) < array.shape[0]:
        mask[mins[-1]] = True
        current_vector = np.ma.array(distances[mins[-1]], mask=mask)
        mins.append(current_vector.argmin(fill_value=np.inf))
        # If we reach last vector, we stop iterating
        if array[mins[-1],0] == stop:
            break

    # Rearranged vectors
    sorted = array[mins]
    # Unsorted vectors (if any)
    mask = np.ones(array.shape[0]).astype(bool)
    mask[mins] = False
    unsorted = array[mask,:]

    return np.concatenate([sorted,unsorted], axis=0)


def sort_by_position(data: pd.DataFrame, remove_ground_vectors: bool = False) -> tuple[pd.DataFrame, float]:
    '''
        Ordena las filas de un dataframe de acuerdo a su latitud y longitud. Mantiene el
        índice original (el orden anterior queda registrado en la columna ordenInicial)

        data : pd.DataFrame
            Dataframe con los vectores ordenados por timestamp
        remove_ground_vectors : bool
            Indica si deben eliminarse los vectores en tierra en el aeropuerto de origen
    '''
    if data.shape[0] == 0:
        return data

    # Investigar estos duplicados
    # Dan lugar a error en el cálculo de las distancias
    data = data[~data.duplicated(subset=['latitude'])]
    data = data[~data.duplicated(subset=['longitude'])]

    stats = {}
    last_vector = None

    origin_airport = airports[airports.id == data.iloc[0].aerodromeOfDeparture].iloc[0]
    data['distance_org'] = haversine_np(data.latitude, data.longitude,
                                        origin_airport.lat, origin_airport.lon)
    data['distance_dst'] = haversine_np(data.latitude, data.longitude)

    if remove_ground_vectors:
        # Eliminamos mensajes en tierra en aeropuerto de origen
        st = data.shape[0]
        data = data[~(data.ground & (data.distance_org < TMA_AREA_MIN))].copy()
        stats['dropped_ground'] = st - data.shape[0]

        # min_index = data.index.min()
        # data = data.reset_index(drop=True)
        # data.index += min_index

    # Si hay vectores en origen, tomamos el primero cronológicamente
    if data[data.distance_org < AIRPORT_AREA].shape[0]:
        data['distance_org'] = haversine_np(data.latitude, data.longitude,
                                            data.iloc[0].latitude, data.iloc[0].longitude)

    # Si hay vectores en destino, tomamos como último el más cercano al punto de referencia
    # if data[data.distance_dst < AIRPORT_AREA].shape[0]:
    #     last_vector = data.loc[data[data.distance_dst < AIRPORT_AREA].distance_dst.idxmin()]

    # Segments identification
    end_origin_segment = data[data.distance_org<TMA_AREA_MAX/2].shape[0]
    end_cruise_segment = data[data.distance_dst>TMA_AREA_MAX].shape[0]
    stats['initial'] = calculate_distance(data[['latitude','longitude','track']].values).sum()

    data['ordenInicial'] = range(data.shape[0])

    # Aseguramos que los primeros mensajes son los más cercanos al aeropuerto de origen
    data.iloc[:] = data.iloc[:].sort_values(by='distance_org').values
    # Tramo de salida ordenado por timestamp
    # data.iloc[:end_origin_segment] = data.iloc[:end_origin_segment].sort_values(by='timestamp').values
    data.iloc[:end_origin_segment] = sort_nearest_vector(data.iloc[:end_origin_segment].values)
    # Tramo de crucero ordenado por distancia decreciente al aeropuerto de destino
    data.iloc[end_origin_segment:end_cruise_segment] = data.iloc[end_origin_segment:end_cruise_segment].sort_values(by='distance_dst', ascending=False).values
    # Tramo de entrada ordenado por timestamp
    # data.iloc[end_cruise_segment:] = data.iloc[end_cruise_segment:].sort_values(by='timestamp').values
    # Tramo de entrada preordenado de acuerdo al vector más cercano
    # data.iloc[end_cruise_segment:] = presort(data.iloc[end_cruise_segment:].values, last_vector.vectorId if not (last_vector is None) else data.shape[0])
    data.iloc[end_cruise_segment:] = sort_nearest_vector(data.iloc[end_cruise_segment:].values)

    # Rotación del avión en zona de maniobras
    tmp = data[(data.distance_dst.between(TMA_AREA_MIN, TMA_AREA_MAX)) & (data.altitude>0)].copy()
    track_variation = calculate_rotation(tmp.track)

    # data = data.drop(['distance_org', 'distance_dst'], axis=1)

    # Primer tramo
    windows = generate_windows(data.shape[0], window_size_slow, overlap_slow, 0, end_origin_segment)
    data.iloc[:] = sort_windows(data.values, windows, angle=False)
    # Segundo tramo
    windows = generate_windows(data.shape[0], window_size_fast, overlap_fast, end_origin_segment, end_cruise_segment)
    data.iloc[:] = sort_windows(data.values, windows, angle=False)

    # Para descartar reordenación del tercer tramo si el resultado es peor que el inicial
    # by_time = data.copy()
    # by_time.iloc[end_cruise_segment:] = by_time.iloc[end_cruise_segment:].sort_values(by='timestamp').values
    # by_nearest = data.copy()

    # Tercer tramo
    last_idx=data.shape[0]
    # last_idx = data.index.get_loc(data[data.vectorId == last_vector.vectorId].iloc[0].name) if not (last_vector is None) else data.shape[0]
    windows = generate_windows(data.shape[0], window_size_slow, overlap_slow, end_cruise_segment, last_idx) #-2
    if abs(track_variation) > MAX_ROTATION:
        # print(f'WARNING: Loop múltiple detectado en {data.fpId.iloc[0]}')
        stats['loop'] = 'D'
        data.iloc[:] = sort_windows(data.values, windows, angle=True)
    elif abs(track_variation) > SINGLE_LOOP_ROTATION:
        stats['loop'] = 'S'
        data.iloc[:] = sort_windows(data.values, windows, angle=True)
    else:
        data.iloc[:] = sort_windows(data.values, windows, angle=False)

    data['ordenFinal'] = range(data.shape[0]) # data.index #

    final_distance = calculate_distance(data[['latitude','longitude','track']].values).sum()
    stats['final'] = final_distance

    # time_distance=covered_distance(by_time[['latitude','longitude','track']].values).sum()
    # time_presort=covered_distance(by_nearest[['latitude','longitude','track']].values).sum()
    # print(f"{data.fpId.iloc[0]}     Time: {time_distance:8.2f}     \
    #       Pres: {time_presort:8.2f} ({100*(time_presort-time_distance)/time_distance:5.1f}%)     \
    #       TSP: {final_distance:8.2f} ({100*(final_distance-time_distance)/time_distance:5.1f}%)")

    # if  time_distance < final_distance:
    #     data = by_time
    #     stats['final'] = time_distance

    stats['rotation'] = track_variation

    return data, stats


def is_resorted(x, acc_list: list) -> bool:
    found, missing = acc_list
    removed = []
    added = None
    while (x.ordenFinal - (len(found) - len(missing))) in found:
        removed.append(str(x.ordenFinal - (len(found) - len(missing))))
        found.remove(x.ordenFinal - (len(found) - len(missing)))

    acc = (len(found) - len(missing))
    reordenado = False
    if (x.ordenFinal == x.ordenInicial + acc):
        reordenado = False
    elif x.ordenFinal - acc < x.ordenInicial:
        if x.ordenFinal - acc + 1 == x.ordenInicial: # Falta un vector -> ¿¿cómo generalizar a más de uno??
            added = x.ordenInicial - 1
            missing.add(added)
            reordenado = False
        else:
            added = x.ordenInicial
            found.add(added)
            reordenado = True
        # print(f'Added: {added}')
       
    elif x.ordenFinal - acc > x.ordenInicial:
        if x.ordenInicial in missing:
            missing.remove(x.ordenInicial)
            reordenado = True
    else:
        print(' AYMAMAMAMA')
    
    # print(f'({x.ordenFinal:3}   {x.ordenInicial:3} | {x.ordenFinal - x.ordenInicial:4})   {len(found) - len(missing):3}' + 
    #       f'   {x.ordenFinal - x.ordenInicial - len(found) + len(missing):4} {"-["+" ".join(removed)+"]" if removed else ""} {"+["+str(added)+"]" if added else ""} {"x" if reordenado else ""}')
    # if reordenado: print(acc_list[1])

    return reordenado


def fix_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    # acc como lista para usarlo como objeto mutable y evitar el uso de una variable global
    # [found, missing]
    acc = [set(), set()]
    
    df['reordenado'] = df[['ordenFinal','ordenInicial']].astype(int).apply(is_resorted, args=[acc], axis=1) # ,'timestamp'
    df['fixed_timestamp'] = df[~df.reordenado].timestamp

    # print(f'Found:   {acc[0]}')
    # print(f'Missing: {acc[1]}')
    
    # Usando la distancia recorrida para realizar la interpolación
    cum_sum = np.cumsum(calculate_distance(df[['latitude','longitude','track']].values)) # [::-1]
    df['fixed_timestamp'] = df.set_index(cum_sum)['fixed_timestamp']\
                              .interpolate(method='index', limit_direction='both').values # limit_area='inside', limit = 5,

    try:
        df['fixed_timestamp'] = df['fixed_timestamp'].astype(int)
        # df['fixed_timestamp'] = np.floor(df['fixed_timestamp'].values)
    except pd.errors.IntCastingNaNError:
        df = df.dropna(subset=['fixed_timestamp'],axis=0).copy()
        print('Caramba!')
        df['fixed_timestamp'] = df['fixed_timestamp'].astype(int)
    return df


def fix_altitude(df: pd.DataFrame) -> pd.DataFrame:
    df['incorrect_altitude'] = False # df.altitude.isna()
    df['filtered_altitude'] = df.altitude

    num_filters = 2
    for i in range(num_filters):
        if i == 0:
            # Primer filtro más grueso
            altitude_threshold = 2000
            win_size = 15
        else:
            # Segundo filtro más fino
            altitude_threshold = 500
            win_size = 5

        df['median_value'] = (df.filtered_altitude # .dropna()
                                .rolling(win_size, min_periods=3, center=True, closed='both')
                                .median()).values
        df['incorrect_altitude'] = ((abs(df.filtered_altitude-df.median_value) > altitude_threshold) | df['incorrect_altitude']).copy()
        df.iloc[[df.index[0], df.index().max()],'incorrect_altitude'] = False
        df['filtered_altitude']  = df[~df.incorrect_altitude].filtered_altitude

    # df['interpolated_altitude'] = df['filtered_altitude'].interpolate(method='linear', limit = 3, limit_area='inside')
    df['interpolated_altitude'] = (df.set_index('fixed_timestamp')['filtered_altitude']
                                     .interpolate(method='index', limit = 5, limit_area='inside')
                                     .values) # .reset_index(drop=True)
    df['fixed_altitude'] = df.filtered_altitude.combine_first(df.interpolated_altitude)

    return df


def fix_trajectory(data: pd.DataFrame) -> pd.DataFrame:
    data, stats = sort_by_position(data, remove_ground_vectors=True)

    initial = stats.get('initial', -1)
    final = stats.get('final', -1)
    rotation = stats.get('rotation', -1)
    dropped = stats.get('dropped_ground', -1)
    loop = stats.get('loop', " ")

    data = fix_timestamp(data)
    data = fix_altitude(data)

    print(f'{data.fpId.iloc[0]}: {dropped:3} {int(rotation):5} {loop}  {initial:9.2f} -> {final:8.2f} ({((final-initial)/initial)*100:6.2f}%)')

    with open('sort_stats.csv', 'a+', encoding='utf8') as file:
        file.write(f'{int(time.time())},{data.flightDate.iloc[0]},{data.fpId.iloc[0]},{dropped},{rotation},{initial},{final},{((final-initial)/initial)*100:.2f}\n')

    return data
        #[['vectorId', 'fpId', 'aerodromeOfDeparture', 'latitude', 'longitude',
        #  'fixed_altitude', 'fixed_timestamp', 'altitude', 'timestamp'
        #  'ordenInicial', 'ordenFinal' ]]


def main():
    date_start, date_end = '2022-09-21', '2022-09-21'

    date_start = datetime.datetime.strptime(date_start, '%Y-%m-%d')
    date_end   = datetime.datetime.strptime(date_end,   '%Y-%m-%d')
    dates      = [(date_start + datetime.timedelta(days=x))
                  for x in range((date_end - date_start).days + 1)]

    with open('bad_trays.txt', 'r', encoding='utf8') as file:
        bad_trays = [x.strip() for x in file.readlines()]

    for date in dates:
        try:
            flights = data_loading.load_raw_data_sort(date)
            flights = pd.read_parquet('../data/test.parquet')

            # flights = flights[flights.fpId=='AT05516307'] ### TEST  AT05521006
        except IndexError: # Si no hay datos para el día
            continue
        indices = data_loading.calculate_indexes(flights)
        if bad_trays:
            indices = indices[~indices.fpId.isin(bad_trays)]

        us_data = []
        for i, (fpId, start_index, end_index) in indices.iterrows():
            us_data.append(flights.loc[start_index:end_index].copy())  
            ## OJO: loc->extremo superior inclusivo, iloc->extremo superior exclusivo

        print(f' Processing: {date.strftime("%Y-%m-%d")} ({len(us_data)} tray) '.center(56,'='))

        with concurrent.futures.ProcessPoolExecutor(max_workers=7) as executor:
            result = list(executor.map(fix_trajectory, us_data))

        if len(result) > 0:
            output = pd.concat(result).sort_values(['ordenFinal'])
            output.to_parquet(sorted_data_path / f'{date.strftime("%Y%m%d")}.parquet')

        print('Procesado.')


if __name__ == '__main__':
    main()