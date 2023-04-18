"""
    This script creates the project's folder structure.
"""

from pathlib import Path

base = '..'
base = Path(__file__).parent / '..'

data_path         = Path(f'{base}/data')
raw_data_path     = Path(f'{base}/data/raw')
raw_data_path_v   = Path(f'{base}/data/raw/vectors')
raw_data_path_w   = Path(f'{base}/data/raw/weather')
sorted_data_path  = Path(f'{base}/data/sorted')
clean_data_path   = Path(f'{base}/data/clean')
window_data_path  = Path(f'{base}/data/window')
final_data_path   = Path(f'{base}/data/final')
sampled_data_path = Path(f'{base}/data/sampled')

utils_path  = Path(f'{base}/utils')
models_path = Path(f'{base}/models')
results_path = Path(f'{base}/results')

paths = [
    data_path,
    raw_data_path,
    raw_data_path_v,
    raw_data_path_w,
    sorted_data_path,
    clean_data_path,
    window_data_path,
    final_data_path,
    sampled_data_path,
    *[sampled_data_path/f's{x}' for x in [15,30,60]],
    utils_path,
    models_path,
    results_path
]
for p in paths:
    if not p.exists():
        p.mkdir(exist_ok=True)