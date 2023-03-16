from pathlib import Path

base = '..'
base = Path(__file__).parent / '..'

raw_data_path_v   = Path(f'{base}/data/raw/vectors')
raw_data_path_w   = Path(f'{base}/data/raw/weather')
sorted_data_path  = Path(f'{base}/data/sorted')
clean_data_path   = Path(f'{base}/data/clean')
window_data_path  = Path(f'{base}/data/window')
final_data_path   = Path(f'{base}/data/final')
sampled_data_path = Path(f'{base}/data/sampled')

airports_file_path = Path(f'{base}/data/airports.csv')

utils_path  = Path(f'{base}/utils')
models_path = Path(f'{base}/models')