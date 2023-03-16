import concurrent
import datetime
import time
import warnings

warnings.simplefilter("ignore", category=FutureWarning)

import numpy as np
import pandas as pd

import data_loading
from common import haversine_np
from paths import *

# Test path
# sorted_data_path = Path('../data/sorted2')

airports = pd.read_csv(airports_file_path, sep = ',')

cols_sort = ['vectorId', 'fpId', 'flightDate', 'aerodromeOfDeparture', 
             'latitude', 'longitude', 'altitude', 'timestamp', 'track', 'ground']

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

# distance_threshold_destination = TMA_AREA_MAX
# distance_threshold_origin = TMA_AREA_MAX

# No se aplican
max_kchanges = 200
max_iteraciones = 1000


def distance(array: np.array, angle = False) -> np.array:
    array = array.astype('float32')
    
    distances = haversine_np(array[1:,0], array[1:,1],
                             array[:-1,0], array[:-1,1],
                                  None, None, angle)
    
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
    else:
        data_size = data_size
        
    windows = []
    number_of_windows = ((data_size-min_index)//(window_size-overlap))+1
    for x in range(number_of_windows):
        window_min = x*(window_size-overlap) + min_index
        window_max = x*(window_size-overlap) + min_index + window_size
        window_max = data_size if window_max > data_size else window_max
        windows.append((x,window_min,window_max))
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


def sort_windows_numpy(array: np.array, windows: tuple, angle: bool = False) -> np.array:
    for it, start, end in windows:
        data = array[start:end].copy()
        # [6,5,7] en lugar de [5,4,6] según col_sort, porque añadimos el index al comienzo
        current_dist = initial_dist = distance(data[:,[5,6,7]], angle).sum()

        # Ojo: índices relativos a la ventana, no a data entero
        changes = generate_changes(0, end-start, skip = 2 if angle else 1)
        # changes = list(np.random.permutation(changes))
        
        for j in range(max_iteraciones):
            improvements = 0
            for i, (v1, v2) in enumerate(changes):
                candidate = np.concatenate([data[:v1+1],
                                            data[v2:v1:-1],
                                            data[v2+1:]])  
                candidate_dist = distance(candidate[:,[5,6,7]], angle).sum()

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
    delta = delta.apply(lambda x: x if abs(x) < 180 else (x - np.sign(x)*360))     ### REVISAR: ¿detecta loops múltiples?
    # Primer elemento es nan
    track_variation = delta.iloc[1:].sum()

    return track_variation


def sort_by_position(data: pd.DataFrame, stats: dict, remove_ground_vectors: bool = False) -> tuple[pd.DataFrame, float]:
    '''
        Ordena las filas de un dataframe de acuerdo a su latitud y longitud. Mantiene el 
        índice original (el orden anterior queda registrado en la columna ordenInicial)
        
        data : pd.DataFrame
            Dataframe con los vectores ordenados por timestamp
        distance_destination : float
            Umbral de distancia al aeropuerto de origen que limita el primer tramo
        distance_origin : float
            Umbral de distancia al aeropuerto de destino que inicia el tercer tramo
    '''
    
    # Asumimos que el primer vector está correctamente ordenado
    data['distance_org'] = haversine_np(data.latitude, data.longitude,
                                        data.latitude.iloc[0], data.longitude.iloc[0])
    data['distance_dst'] = haversine_np(data.latitude, data.longitude)

    # Eliminamos mensajes en tierra en aeropuerto de origen
    if remove_ground_vectors:
        min_index = data.ordenInicial.min()
        st = data.shape[0]
        data = data[~(data.ground & (data.distance_org < TMA_AREA_MIN))]
        stats['dropped_ground'] = st - data.shape[0]
        corrupt_altitudes_indx = data[(data.altitude.isna() & (data.distance_org < AIRPORT_AREA))].index
        data.loc[corrupt_altitudes_indx, 'altitude'] = 0
        data = data.drop('ordenInicial', axis=1).reset_index(drop=True).reset_index().rename({'index':'ordenInicial'}, axis=1)
        data.index = data.index+min_index
        data['ordenInicial'] += min_index

    # Si hay vectores en destino, tomamos el último como punto de referencia
    # Asumimos que es correcto
    if data[data.distance_dst < AIRPORT_AREA].shape[0]:
        last_vector = data.loc[data[data.distance_dst < AIRPORT_AREA].timestamp.idxmax()]
        data['distance_dst'] = haversine_np(data.latitude, data.longitude,
                                            last_vector.latitude, last_vector.longitude)
    
    end_origin_segment = data[data.distance_org<TMA_AREA_MAX].shape[0]
    end_cruise_segment = data[data.distance_dst>TMA_AREA_MAX].shape[0]
    commited = data[data.distance_dst>TMA_AREA_MIN].shape[0]
    
    stats['initial'] = distance(data[['latitude','longitude','altitude']].values).sum()

    # Rotación del avión en zona de maniobras (cerca del aeropuerto, pero no en la terminal)
    tmp = data[(data.distance_dst.between(TMA_AREA_MIN, TMA_AREA_MAX)) & (data.altitude>0)].copy()
    tmp = tmp.sort_values(by='timestamp', ascending=True)
    track_variation = calculate_rotation(tmp.track)
    
    # Orden inicial por distancia creciente al primer vector de posición
    # data.loc[:] = data.sort_values(by='distance_org', ascending=True).values

    # Tramo de salida ordenado por distancia creciente al primer vector de posición
    # Tramo de crucero ordenado por distancia decreciente al aeropuerto de destino
    # data.loc[:] = np.concatenate([
    #     data.iloc[:end_origin_segment].values,
    #     data.iloc[end_origin_segment:].sort_values(by='distance_dst', ascending=False).values
    # ])
    data.iloc[:] = data.iloc[:].sort_values(by='distance_org', ascending=False)
    data.iloc[end_origin_segment:] = data.iloc[end_origin_segment:].sort_values(by='distance_dst', ascending=False).values
    data.iloc[end_cruise_segment:-2] = data.iloc[end_cruise_segment:-2].sort_values(by='timestamp').values

    data = data.drop(['distance_org', 'distance_dst'], axis=1)

    # Primer tramo
    windows = generate_windows(data.shape[0], window_size_slow, overlap_slow, 0, end_origin_segment)
    data.iloc[:] = sort_windows_numpy(data.values, windows, angle=True)

    # Segundo tramo
    windows = generate_windows(data.shape[0], window_size_fast, overlap_fast, end_origin_segment-5, end_cruise_segment)
    data.iloc[:] = sort_windows_numpy(data.values, windows, angle=False)

    tmp_data = data.copy()

    # Tercer tramo
    if abs(track_variation) > MAX_ROTATION:
        # data.iloc[end_cruise_segment+1:-2] = data.iloc[end_cruise_segment+1:-2].sort_values(by='timestamp').values
        print(f'WARNING: Loop múltiple detectado en {data.fpId.iloc[0]}')
    elif abs(track_variation) > SINGLE_LOOP_ROTATION:
        # data.iloc[end_cruise_segment+1:-2] = data.iloc[end_cruise_segment+1:-2].sort_values(by='timestamp').values
        windows = generate_windows(data.shape[0], window_size_slow, overlap_slow, end_cruise_segment-5, data.shape[0])
        data.iloc[:] = sort_windows_numpy(data.values, windows, angle=True)
    else:
        windows = generate_windows(data.shape[0], window_size_slow, overlap_slow, end_cruise_segment-5, data.shape[0])
        data.iloc[:] = sort_windows_numpy(data.values, windows, angle=False) 
    final_distance = distance(data[['latitude','longitude','altitude']].values).sum()
    stats['final'] = final_distance

    tmp_distance = distance(tmp_data[['latitude','longitude','altitude']].values).sum()
    if  tmp_distance < final_distance:
        data = tmp_data
        stats['final'] = tmp_distance
    
    stats['rotation'] = track_variation

    return data


def is_resorted(x, acc_list) -> bool:
    acc = acc_list[0]
    if (x.ordenFinal == x.ordenInicial - acc):
        return False
    else:
        if x.ordenFinal < x.ordenInicial: # Vector posterior insertado aquí
            acc_list[0] += 1
            return True
        # if x.ordenFinal < x.ordenInicial + acc: # Falta un vector, pero este es correcto
        #     acc_list[0] -= 1
        #     return False
        if x.ordenFinal > x.ordenInicial: # Vector anterior insertado aquí
            acc_list[0] -= 1
            return True

    # def is_resorted(x, acc_list):
    # acc = acc_list[0]

    # if (x.ordenFinal == x.ordenInicial + acc):
    #     return False
    # else:
    #     if x.ordenFinal < x.ordenInicial: # Vector posterior insertado aquí
    #         acc_list[0] += 1
    #         return True
    #     elif x.ordenFinal < x.ordenInicial + acc: # Falta un vector, pero este es correcto
    #         acc_list[0] -= 1 
    #         return False
    #     elif x.ordenFinal > x.ordenInicial: # Vector anterior insertado aquí
    #         acc_list[0] += 1 
    #         return True

    return False


def fix_timestamp(df: pd.DataFrame):
    # acc como lista para usarlo como objeto mutable y pasarlo por referencia
    # Evita el uso de una variable global
    acc = [0]

    df['reordenado'] = df[['ordenFinal','ordenInicial','timestamp']].apply(is_resorted, args=[acc], axis=1)
    df['fixed_timestamp'] = df[~df.reordenado].timestamp
    # df['fixed_timestamp'] = df['fixed_timestamp'].interpolate(method='linear').astype(int)
    
    # Usando la distancia recorrida para realizar la interpolación
    df['fixed_timestamp'] = df.set_index(np.cumsum(distance(df[['latitude','longitude','altitude']].values))[::-1])['fixed_timestamp']\
                              .interpolate(method='index').values # limit_area='inside'
    # limit = 5,
    try:
        df['fixed_timestamp'] = df['fixed_timestamp'].astype(int)
    except pd.errors.IntCastingNaNError:
        df = df.dropna(subset=['fixed_timestamp'],axis=0).copy()
        print('Caramba!')
        df['fixed_timestamp'] = df['fixed_timestamp'].astype(int)
    return df


def fix_altitude(df: pd.DataFrame):
    df['incorrect_altitude'] = False
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
        df.loc[[0,df.index.max()],'incorrect_altitude'] = False
        df['filtered_altitude']  = df[~df.incorrect_altitude].filtered_altitude
        
    # df['interpolated_altitude'] = df['filtered_altitude'].interpolate(method='linear', limit = 3, limit_area='inside')
    df['interpolated_altitude'] = df.set_index('fixed_timestamp')['filtered_altitude']\
                                    .interpolate(method='index', limit = 5, limit_area='inside')\
                                    .values # .reset_index(drop=True)
    df['fixed_altitude'] = df.filtered_altitude.combine_first(df.interpolated_altitude)
    
    return df


def fix_trajectory(data: pd.DataFrame):
    stats = {}

    data = data.copy()
    data = sort_by_position(data, stats, remove_ground_vectors=True)
    data = data.reset_index().rename({'index':'ordenFinal'}, axis=1)
    
    initial = stats.get('initial', -1)
    final = stats.get('final', -1)
    rotation = stats.get('rotation', -1)
    dropped = stats.get('dropped_ground', -1)
    
    data = fix_timestamp(data)
    data = fix_altitude(data)

    print(f'{data.fpId.iloc[0]}: {dropped:3} {int(rotation):5}  {initial:8.3f} -> {final:8.3f} ({((final-initial)/initial)*100:6.2f}%)')

    with open('sort_stats.csv', 'a+', encoding='utf8') as file:
        file.write(f'{int(time.time())},{data.flightDate.iloc[0]},{data.fpId.iloc[0]},{dropped},{rotation},{initial},{final},{((final-initial)/initial)*100:.2f}\n')
    
    return data[['vectorId', 'fpId', 'aerodromeOfDeparture', 'latitude', 'longitude', 
        'fixed_altitude', 'fixed_timestamp', 'ordenInicial', 'ordenFinal',
        # 'altitude', 'timestamp'
        ]]


def main():
    date_start, date_end = '2022-10-01', '2022-10-29'

    date_start = datetime.datetime.strptime(date_start, '%Y-%m-%d')
    date_end   = datetime.datetime.strptime(date_end,   '%Y-%m-%d')
    dates      = [(date_start + datetime.timedelta(days=x))
                for x in range((date_end - date_start).days + 1)][:-1]
    
    with open('bad_trays.txt', 'r', encoding='utf8') as file:
        bad_trays = [x.strip() for x in file.readlines()]

    for date in dates:
        try:
            flights = data_loading.load_raw_data_sort(date)
        except IndexError:
            continue
        indices = data_loading.calculate_indexes(flights)   ## Se puede cambiar por un group by
        if bad_trays:
            indices = indices[~indices.fpId.isin(bad_trays)]
        
        # Test trajectory
        # indices = indices[indices.fpId == 'AT03828689'] # AT02775614 AT05514146

        us_data = []
        for i, (fpId, start_index, end_index) in indices.iterrows():
            us_data.append(flights.iloc[start_index:end_index+1].copy())

        print(f'======== Processing: {date.strftime("%Y-%m-%d")} ({len(us_data)} tray) ========')
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=7) as executor:
            result = list(executor.map(fix_trajectory, us_data))
        
        # fix_trajectory(us_data[0])

        if len(result) > 0:
            output = pd.concat(result).sort_values(['ordenFinal'])
            output.to_parquet(sorted_data_path / f'{date.strftime("%Y%m%d")}.parquet')

        print('Procesado.')

if __name__ == '__main__':
    main()