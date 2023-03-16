import math
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (mean_absolute_error,
                             mean_absolute_percentage_error,
                             mean_squared_error)

from paths import *


def get_evaluation_metrics(inv_test_Y: np.array, inv_pred_Y: np.array, feat:str = None, prnt: bool = False, name:str = ''):
    """Calculates and optionally displays metrics given a set of real values and predictions

    Currently, six metrics are calculated:
    - Mean absolute error
    - Root mean squared error
    - Mean absolute percentage error
    - Standard deviation
    - Mean
    - Sample size
    
    Args:
        inv_test_Y: Array of real values
        inv_pred_Y: Array of predicted values
        prnt: Indicates whether the results should be printed on screen
        name: Name of the dataset being evaluated
    """
    mae   = mean_absolute_error(inv_test_Y, inv_pred_Y) # , multioutput='raw_values'
    rmse  = math.sqrt(mean_squared_error(inv_test_Y, inv_pred_Y))
    mape  = mean_absolute_percentage_error(inv_test_Y, inv_pred_Y)
    stdev = statistics.stdev(inv_test_Y-inv_pred_Y)
    mean  = statistics.mean(inv_test_Y-inv_pred_Y)
    sample_size = len(inv_test_Y)

    if prnt:
        print(f'{feat} ===============================')
        print(f'{str.title(name)+" set":18}| MAE:     {mae  :>10.3f}s')
        print(f'{                    "":18}| RMSE:    {rmse :>10.3f}s')
        print(f'{                    "":18}| StDev:   {stdev:>10.3f}s')
        print(f'{                    "":18}| Mean:    {mean :>10.3f}s')
        print(f'{                    "":18}| MAPE:    {mape :>10.3f}')
        print(f'{                    "":18}| Muestra: {sample_size:>10,}')

    metrics = dict(
        mae=mae,
        rmse=rmse,
        mape=mape,
        stdev=stdev,
        mean=mean,
        sample_size=sample_size
    )

    return metrics


def display_errors(test_Y: np.array, pred_Y: np.array, title: str = None):
    """Generates an histogram of the errors"""
    fig,ax    = plt.subplots(1,1, figsize = (14,5))
    error     = pd.DataFrame(test_Y - pred_Y, columns=['error'])

    sns.histplot(error, ax=ax, binwidth=50, log_scale=(False, True))
    ax.set_title(title)
    ax.set(xlabel='Error (real-predicho)', 
           ylabel='Frecuencia')

### Deprecated?
def load_from_dataset(path: Path, date: str, airport: str, randomize:bool = False) -> pd.DataFrame:
    # print(airport, date)
    datasets = [tf.data.Dataset.load(str(ds)) 
                for ds in sorted(path.glob(f'*'))]
    
    if randomize:
        probs = [ds.cardinality().numpy() for ds in datasets]
        probs = [x/sum(probs) for x in probs]
        dataset = tf.data.Dataset.sample_from_datasets(datasets, weights=probs)
    else:
        dataset  = tf.data.Dataset.from_tensor_slices(datasets)
        dataset  = dataset.interleave(lambda x: x, cycle_length=1, 
                                    num_parallel_calls=tf.data.AUTOTUNE)
    
    return dataset