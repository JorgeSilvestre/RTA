import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from typing import NamedTuple

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.layers import LSTM, GRU, Dense, Input, Bidirectional, Conv1D, MaxPool1D, RepeatVector, TimeDistributed
from keras.models import Model, Sequential
from keras.optimizers import Adam, adamw_experimental

import data_loading
import data_preparation
import model_generation
import paths

metrics = ['mae','rmse', 'mape', 'stdev', 'mean', 'sample_size']
report_columns = ['MAE', 'RMSE', 'MAPE', 'StDev', 'Mean', 'Sample']

times = (15, 30, 60, 90, 120, 150, 0)
# times = (25,45,60,100,125,250)
MIN_SAMPLE_SIZE = 5


ExperimentResult = NamedTuple('ExperimentResult', [('dataset',str), ('time',str)]+[(x,str) for x in metrics])

class Experiment:
    """Base class for RNN experiments

    Implements necessary logic, except data formatting and model initialization:
    model loading, data loading (either from parquet or TF Dataset data), and model
    training and evaluation.
    
    Attributes:
        lookback: Length of the sliding window
        sampling: Sampling period used to downsample trajectory data
        batch_size: TF parameter. The amount of examples to be fed before weight update
        months: The months used to train and evaluate the model, in format 'YYYYMM'
        airports: ICAO code of an airport, or * for all available airports
        features: Dictionary with list of strings identifying features of
            each type:
            { numeric:[feat1, ...], categoric:[...], objective:[...] }

    """
    def __init__(self,
                 lookback: int,
                 sampling: int,
                 batch_size: int,
                 months: str,
                 airport: str,
                 features:dict):
        self.lookback = lookback
        self.sampling = sampling
        self.batch_size = batch_size
        self.months = months
        self.airport = airport
        self.features = features
        self.numeric_feat = features.get('numeric',None)
        self.categoric_feat = features.get('categoric',None)
        self.objective_feat = features.get('objective',None)
        self.num_features = len(self.numeric_feat) + len(self.categoric_feat)

        self.model = None
        self.trained_epochs = 0
        self.results = {}

        self.encoders = joblib.load(paths.utils_path / f'encoder_{self.num_features}.joblib')
        self.scaler   = joblib.load(paths.utils_path / f'scaler_{self.num_features}.joblib')

    def init_model(self):
        """Model initialization
        
        To be implemented in child classes."""
        raise NotImplementedError
        
    def _check_trained_epochs(self):
        """For resuming training processes
        
        Checks the latest checkpoint of the model to set the amount of trained epochs"""
        try:
            epochs = pd.read_csv(self.model_path_log).epoch.max() + 1
        except FileNotFoundError:
            # If log file does not exist
            return 0

        return epochs

    def load_model(self, name:str = 'last'):
        """Loads a model checkpoint

        Args:
            name: Name of the model. Can be either 'last', 'best' or a custom file name
        """
        self.model = tf.keras.models.load_model(self.model_path / f'{name}.h5')
        
        self.trained_epochs = self._check_trained_epochs()
        self._init_callbacks()

    def _init_callbacks(self):
        """Initializes callbacks for model training

        Four callbacks are defined:
        - Model checkpoint: saves a checkpoint after every epoch
        - Model checkpoint best: performs an additional save of the best epoch
          based on validation loss
        - Model checkpoint last: performs an additional save of the latest
          movel version (used to ease the model loading)
        - CSVLogger: logs the training results after each epoch
        
        """
        modelCheckpoint = ModelCheckpoint(
            self.model_path_save,
            monitor='val_loss',
            verbose=0,
            mode='auto',
            save_best_only=False
        )

        modelCheckpointBest = ModelCheckpoint(
            self.model_path_best,
            monitor='val_loss',
            verbose=0,
            mode='auto',
            save_best_only=True
        )

        modelCheckpointLast = ModelCheckpoint(
            self.model_path_last,
            monitor='val_loss',
            verbose=0,
            mode='auto',
            save_best_only=False
        )

        csvLogger = CSVLogger(
            self.model_path_log, append=True
        )

        self.callbacks = [modelCheckpoint, modelCheckpointBest, modelCheckpointLast, csvLogger]

    def _format_data(self):
        """Formatting data for use as input to the model 
        
        To be implemented in child classes."""
        raise NotImplementedError

    def _load_data_from_parquet(self, dataset, randomize) -> tf.data.Dataset:
        """Helper function to load data from parquet files

        Loads and merges the parquet files corresponding to months indicated in self.months.
        Optionally it randomizes data (for instance, for training). Randomization is weighted
        according to month's cardinality to ensure an homogeneous distribution.
        
        Args:
            dataset: The dataset to be loaded. Can be either 'train', 'test' or 'val'
            randomize: Boolean to indicate whether the data should be randomized
        """
        data = data_loading.load_final_data(self.months, dataset, self.airport, self.sampling)

        if randomize:
            aps = sorted(data.aerodromeOfDeparture.unique())
            counts = data.aerodromeOfDeparture.value_counts()
            probs = [counts[ap]/len(data) for ap in aps]

            datasets = [data_preparation.get_windows(data[data.aerodromeOfDeparture == ap].copy(),
                                    self.lookback, self.encoders, self.scaler, self.features).shuffle(1000)
                        for ap in aps]

            dataset  = tf.data.Dataset.sample_from_datasets(datasets, weights=probs)
        else:
            dataset = data_preparation.get_windows(data.copy(),
                                    self.lookback, self.encoders, self.scaler, self.features)
        return dataset

    def _load_data_from_dataset(self, dataset, randomize=False) -> tf.data.Dataset:
        """Helper function to load data from a TF dataset
        
        Loads and merges the TF Datasets corresponding to months indicated in self.months.
        Optionally it randomizes data (for instance, for training). Randomization is weighted
        according to month's cardinality to ensure an homogeneous distribution.

        Args:
            dataset: The dataset to be loaded. Can be either 'train', 'test' or 'val'
            randomize: Boolean to indicate whether the data should be randomized
        """
        path = paths.window_data_path / f'data{self.lookback}_s{self.sampling}/{dataset}'
        datasets = [tf.data.Dataset.load(str(ds))
                    for ds in sorted(path.glob(f'{self.months}-{self.airport}'))]

        if randomize:
            freqs = [ds.cardinality().numpy() for ds in datasets]
            probs = [x/sum(freqs) for x in freqs]
            dataset = tf.data.Dataset.sample_from_datasets(datasets, weights=probs)
        else:
            dataset  = tf.data.Dataset.from_tensor_slices(datasets)
            dataset  = dataset.interleave(lambda x: x, cycle_length=1,
                                          num_parallel_calls=tf.data.AUTOTUNE)

        return dataset

    def _load_data(self, dataset, from_parquet: bool=False, randomize:bool=False) -> tf.data.Dataset:
        """Helper function to load data from parquet

        Provides a common interface for the different origins of the data. TF Datasets have a faster
        loading and fit in memory independently of the dataset size, but take up a large amount of
        disk space. Parquet data is loaded and formatted on the fly, but they require to fit in memory.

        Args:
            dataset: The dataset to be loaded. Can be either 'train', 'test' or 'val'
            from_parquet: Boolean to indicate whether the data is loaded from TF Datasets or parquet files
            randomize: Boolean to indicate whether the data should be randomized
        """
        if from_parquet:
            data =  self._load_data_from_parquet(dataset, randomize)
        else:
            data = self._load_data_from_dataset(dataset, randomize)
        return self._format_data(data)

    def train(self, epochs: int, from_parquet: bool = False, add_callbacks: list = None): 
        """Trains the model up to a given number of epochs

        Loads train and validation datasets and trains the model up to a given number of
        epochs. If the model have already been trained some epochs, the amount of trained 
        epochs is taken into account

        Args:
            epochs: Target number of epochs to train the model
            from_parquet: Boolean to indicate whether the data is loaded from parquet files or TF Datasets
            add_callbacks: List of callbacks to be added to the model (optional)       
        """
        train_dataset = self._load_data('train', from_parquet, randomize=True)
        val_dataset = self._load_data('val', from_parquet, randomize=False)

        log_epochs = self._check_trained_epochs()
        if self.trained_epochs != log_epochs:
            print(f'The number of trained epochs has been updated to {log_epochs}')
            self.trained_epochs = log_epochs
    
        h = self.model.fit(
                x = train_dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE),
                validation_data = val_dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE),
                epochs = epochs,
                verbose = 1,
                callbacks = self.callbacks + (add_callbacks if add_callbacks else []),
                initial_epoch = self.trained_epochs)
                
        epochs = pd.read_csv(self.model_path_log).epoch.max() + 1
        self.trained_epochs = epochs

        return h

    def get_y(self, dataset: tf.data.Dataset) -> np.array:
        """Helper function to retrieve the labels of the examples in a TF dataset

        Args:
            dataset: A TF Dataset of labeled windows
        """
        return np.array([i.numpy()[0] for i in dataset.map(lambda x,y: y,
                        num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)])

    def _evaluate_model(self, dataset, print_err = False, name = None, title = None) -> dict:
        """Helper function to compare predicted and real values for a dataset

        Calculates the defined metrics using real and predicted values, and optionally displays
        the results on screen. Retrieves a dictionary with 'metric name:value' pairs.

        Args:
            dataset: A TF Dataset of labeled windows
            print_err: Boolean to indicate if results information should be displayed on screen
            name: Name of the dataset that is being evaluated
            title: Title of the figure for information purposes
        """
        real_Y  = (self.get_y(dataset)/self.scaler.scale_[-1]).reshape((-1,))
        if len(real_Y) < MIN_SAMPLE_SIZE:
            return {}

        pred_Y = (self.model.predict(dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE), verbose = print_err)/self.scaler.scale_[-1])[:,-1].reshape((-1,))[real_Y>0]
        real_Y = real_Y[real_Y>0]
        metrics_values = model_generation.get_evaluation_metrics(real_Y, pred_Y, print_err,name)

        if title:
            model_generation.display_errors(real_Y, pred_Y, title)

        return metrics_values

    def evaluate(self, from_parquet:bool = False, print_err = True):
        """Calculates global metrics for validation and test datasets
        
        Args:
            from_parquet: Boolean to indicate whether the data is loaded from parquet files or TF Datasets
            print_err: Boolean to indicate if results information should be displayed on screen
        """
        val_dataset = self._load_data('val', from_parquet, randomize=False)
        test_dataset = self._load_data('test', from_parquet, randomize=False)

        self.results['val all'] =  ExperimentResult('val', 'all',  **self._evaluate_model(val_dataset, print_err=print_err, name='val',
                                                    title='Distribución del error en el conjunto de validación' if print_err else None))
        self.results['test all'] = ExperimentResult('test', 'all', **self._evaluate_model(test_dataset, print_err=print_err, name='test',
                                                    title='Distribución del error en el conjunto de test' if print_err else None))

    def evaluate_at_times(self):
        """Calculates at-time metrics for validation and test datasets
        
        Data is loaded and formatted on-the-fly from parquet files
        """
        for dataset in ('val','test'):
            # dataframe = self._load_data_from_parquet(dataset, randomize=False)
            dataframe = data_loading.load_final_data(self.months, dataset, self.airport, self.sampling)

            for idx, time in enumerate(times):
                print(f'{dataset}: {idx+1}/{len(times)} Evaluando a {time} minutos     ', end='\r')
                ds = data_preparation.get_windows_at_time(dataframe, time, self.lookback, self.encoders,
                                                          self.scaler, self.features)
                ds = self._format_data(ds)

                if ds.cardinality().numpy() > MIN_SAMPLE_SIZE:
                    self.results[f'{dataset} {time}'] =  ExperimentResult(dataset, time,  **self._evaluate_model(ds))
            print(f'{dataset}: Finalizado' + ' '*50)

    def evaluate_airports(self):
        """Calculates global and at-time metrics for each airport.
        
        Data is loaded and formatted on-the-fly from parquet files
        """
        # test_data = self._load_data_from_parquet('test', randomize=False)
        test_data = data_loading.load_final_data(self.months, 'test', self.airport, self.sampling)
        test_airports = sorted(test_data.aerodromeOfDeparture.unique())

        for idx, ap in enumerate(test_airports):
            airport_data = test_data[test_data.aerodromeOfDeparture == ap].copy()

            print(f'({idx+1}/{len(test_airports)}) Evaluando {ap}' + ' '*30, end='\r')
            ap_ds = data_preparation.get_windows(airport_data.copy(), self.lookback,
                                                 self.encoders, self.scaler, self.features)

            # Revisar por qué todos los datasets tienen cardinality=-2
            # if ap_ds.cardinality().numpy() > MIN_SAMPLE_SIZE:
            ap_ds = self._format_data(ap_ds)
            try:
                # REVISAR: ap_ds.cardinality().numpy() es -2 siempre (TF no puede calcular la cardinalidad)
                # if ap_ds.cardinality().numpy() > MIN_SAMPLE_SIZE:
                self.results[f'{ap} all'] =  ExperimentResult(ap, 'all',  **self._evaluate_model(ap_ds))
            except TypeError:
                pass

            for idx2, time in enumerate(times):
                print(f'({idx+1}/{len(test_airports)}) Evaluando {ap} a {time} minutos' + ' '*30, end='\r')

                ap_ds = data_preparation.get_windows_at_time(airport_data.copy(), time, self.lookback, self.encoders,
                                                             self.scaler, self.features)

                if ap_ds.cardinality().numpy() > MIN_SAMPLE_SIZE:
                    ap_ds = self._format_data(ap_ds)
                    self.results[f'{ap} {time}'] =  ExperimentResult(ap, time,  **self._evaluate_model(ap_ds))
        print(f'({idx+1}/{len(test_airports)})  Done.                        ')

    def export_results_to_csv(self):
        """Pending."""
        raise NotImplementedError


class ExperimentVanilla(Experiment):
    """Experiments that use vanilla LSTM networks

    Implements data formatting and model initialization. Inherits from Experiment class.
    
    Attributes:
        lookback: Length of the sliding window
        sampling: Sampling period used to downsample trajectory data
        model_config: Sets differents configurations of the model, such as batch size,
            activation function, or number of LSTM units
        batch_size: TF parameter. The amount of examples to be fed before weight update
        months: The months used to train and evaluate the model, in format 'YYYYMM'
        airports: ICAO code of an airport, or * for all available airports
        features: Dictionary with list of strings identifying features of
            each type:
            { numeric:[feat1, ...], categoric:[...], objective:[...] }
        model_type: Descriptive name of the model type. By default, 'LSTM'

    """
    def __init__(self,
                 lookback:int,
                 sampling:int,
                 model_config:dict,
                 months:str,
                 airport:str,
                 features:dict,
                 model_type:str = None):
        super().__init__(
            lookback,
            sampling,
            model_config.get('batch_size'),
            months,
            airport,
            features)

        self.model_type = model_type if model_type else 'LSTM'
        self.n_units = model_config.get('n_units')
        self.act_function = model_config.get('act_function', 'tanh')

        self.model_path = paths.models_path / f'{self.model_type}_s{self.sampling}_lb{self.lookback}_u{self.n_units}'
        self.model_path_save = self.model_path / ('ep{epoch:03d}_loss{loss:.4f}_val{val_loss:.4f}.h5')
        self.model_path_best = self.model_path / 'best.h5'
        self.model_path_last = self.model_path / 'last.h5'
        self.model_path_log  = self.model_path / 'log.csv'

    def init_model(self, add_metrics = None):
        self.model = Sequential([
            LSTM(self.n_units,
                 activation=self.act_function,
                 input_shape=(self.lookback, self.num_features)),
            Dense(len(self.objective_feat))
        ])
        self.model.compile(
            loss='mean_absolute_error',
            optimizer=adamw_experimental.AdamW(),
            metrics = ['mean_squared_error'] + (add_metrics if add_metrics else []))

        self._init_callbacks()

    def _format_data(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """Formatting data for use as input to the model 
        
        Uses window data to construct valed examples for the recurrent model.
        """
        return dataset.map(lambda x: (x[:-1,:-3], x[-1,-3]))


class ExperimentGRU(Experiment):
    def __init__(self,
                 lookback:int,
                 sampling:int,
                 model_config:dict,
                 months:str,
                 airport:str,
                 features:dict):
        super().__init__(
            lookback,
            sampling,
            model_config.get('batch_size'),
            months,
            airport,
            features)

        self.model_type = 'GRU'
        self.n_units = model_config.get('n_units')
        self.act_function = model_config.get('act_function', 'tanh')

        self.model_path = paths.models_path / f'{self.model_type}_s{self.sampling}_lb{self.lookback}_u{self.n_units}'
        self.model_path_save = self.model_path / ('ep{epoch:03d}_loss{loss:.4f}_val{val_loss:.4f}.h5')
        self.model_path_best = self.model_path / 'best.h5'
        self.model_path_last = self.model_path / 'last.h5'
        self.model_path_log  = self.model_path / 'log.csv'

    def init_model(self, add_metrics = None):
        self.model = Sequential([
            GRU(self.n_units,
                 activation=self.act_function,
                 input_shape=(self.lookback, self.num_features)),
            Dense(1)
        ])
        self.model.compile(
            loss='mean_absolute_error',
            optimizer=Adam(),
            metrics = ['mean_squared_error'] + (add_metrics if add_metrics else []))

        self._init_callbacks()

    def _format_data(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        return dataset.map(lambda x: (x[:,:-1], x[-1:,-1]))


class ExperimentED(Experiment):
    def __init__(self,
                 lookback:int,
                 sampling:int,
                 model_config:dict,
                 months:str,
                 airport:str,
                 features:dict,
                 model_type:str = None):
        super().__init__(
            lookback,
            sampling,
            model_config.get('batch_size'),
            months,
            airport,
            features)

        self.model_type = model_type if model_type else 'ED'
        self.n_inputs = self.lookback
        self.n_units = model_config.get('n_units')
        self.n_outputs = model_config.get('n_outputs', 1)
        self.act_function = model_config.get('act_function', 'tanh')

        self.ts_feat = features.get('ts',None)
        self.nts_feat = features.get('nts',None)

        self.idx_ts   = [(self.numeric_feat+self.categoric_feat).index(x) for x in self.ts_feat]
        self.idx_nts  = [(self.numeric_feat+self.categoric_feat).index(x) for x in self.nts_feat]

        self.model_path = paths.models_path / f'{self.model_type}_s{self.sampling}_lb{self.lookback}_u{self.n_units}'
        self.model_path_save = self.model_path / ('ep{epoch:03d}_loss{loss:.4f}_val{val_loss:.4f}.h5')
        self.model_path_best = self.model_path / 'best.h5'
        self.model_path_last = self.model_path / 'last.h5'
        self.model_path_log  = self.model_path / 'log.csv'

    def init_model(self):
        # Encoder
        encoder_inputs = Input(shape = (self.n_inputs, len(self.ts_feat)))
        encoder_lstm   = LSTM(self.n_units, activation=self.act_function, return_state = True)
        encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)

        # Decoder
        # decoder_inputs = Input(shape = (len(nts_features), n_outputs))
        decoder_inputs = Input(shape = (self.n_outputs, len(self.nts_feat)))
        decoder_lstm   = LSTM(self.n_units, activation=self.act_function) # return_sequences = True

        decoder_outputs = decoder_lstm(decoder_inputs, initial_state = [state_h, state_c])
        # decoder_outputs = Dense(n_units, activation ='relu')(decoder_outputs)
        # decoder_outputs = Flatten ()( decoder_outputs )
        decoder_outputs = Dense(self.n_outputs)(decoder_outputs)

        self.model = Model(inputs = [encoder_inputs, decoder_inputs], outputs = decoder_outputs)

        self.model.compile(loss='mean_absolute_error',
                    optimizer = Adam(),
                    metrics = ['mean_squared_error'])

        self._init_callbacks()

    def _format_data(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        return dataset.map(lambda x: (
            ( tf.gather(x, self.idx_ts, axis=1),
              tf.reshape(tf.gather(tf.gather(x, self.idx_nts, axis=1), self.lookback-1, axis=0),
                         shape= (1,len(self.idx_nts)))),
            x[-1:,-1]))

        # ED Full
        # return dataset.map(lambda x: (
        #     ( x[:,:-1],
        #       tf.reshape(tf.gather(tf.gather(x, self.idx_nts, axis=1), self.lookback-1, axis=0),
        #                  shape= (1,len(self.idx_nts)))),
        #     x[-1:,-1]))


class ExperimentEDFC(Experiment):
    def __init__(self,
                 lookback:int,
                 sampling:int,
                 model_config:dict,
                 months:str,
                 airport:str,
                 features:dict):
        super().__init__(
            lookback,
            sampling,
            model_config.get('batch_size'),
            months,
            airport,
            features)

        self.model_type = 'EDFC'
        self.n_units = model_config.get('n_units')
        self.act_function = model_config.get('act_function', 'tanh')

        self.model_path = paths.models_path / f'{self.model_type}_s{self.sampling}_lb{self.lookback}_u{self.n_units}'
        self.model_path_save = self.model_path / ('ep{epoch:03d}_loss{loss:.4f}_val{val_loss:.4f}.h5')
        self.model_path_best = self.model_path / 'best.h5'
        self.model_path_last = self.model_path / 'last.h5'
        self.model_path_log  = self.model_path / 'log.csv'

    def init_model(self, add_metrics = None):
        input_layer = Input(shape = (self.lookback, self.num_features))
        lstm_layer = Bidirectional(LSTM(self.n_units,
                 activation=self.act_function))(input_layer)
        fc_layers = Dense(20, activation='relu')(lstm_layer)
        fc_layers = Dense(10, activation='relu')(fc_layers)
        fc_layers = Dense(1)(fc_layers)

        self.model = Model(inputs = input_layer, outputs = fc_layers)

        self.model.compile(
            loss='mean_absolute_error',
            optimizer=Adam(),
            metrics = ['mean_squared_error'] + (add_metrics if add_metrics else []))

        self._init_callbacks()

    def _format_data(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        return dataset.map(lambda x: (x[:,:-1], x[-1:,-1]))


class ExperimentConvLSTM(Experiment):
    def __init__(self,
                 lookback:int,
                 sampling:int,
                 model_config:dict,
                 months:str,
                 airport:str,
                 features:dict):
        super().__init__(
            lookback,
            sampling,
            model_config.get('batch_size'),
            months,
            airport,
            features)

        self.model_type = 'ConvLSTM'
        self.n_units = model_config.get('n_units')
        self.act_function = model_config.get('act_function', 'tanh')

        self.model_path = paths.models_path / f'{self.model_type}_s{self.sampling}_lb{self.lookback}_u{self.n_units}'
        self.model_path_save = self.model_path / ('ep{epoch:03d}_loss{loss:.4f}_val{val_loss:.4f}.h5')
        self.model_path_best = self.model_path / 'best.h5'
        self.model_path_last = self.model_path / 'last.h5'
        self.model_path_log  = self.model_path / 'log.csv'

    def init_model(self, add_metrics = None):
        self.model = Sequential([
            Conv1D(filters=self.num_features,
                   kernel_size=3,
                   activation='relu',
                   input_shape=(self.lookback, self.num_features)),
            Conv1D(filters=self.num_features,
                   kernel_size=3,
                   activation='relu',
                   input_shape=(self.lookback, self.num_features)),
            MaxPool1D(2),
            LSTM(self.n_units,
                 activation=self.act_function),
            Dense(1)
        ])
        self.model.compile(
            loss='mean_absolute_error',
            optimizer=Adam(),
            metrics = ['mean_squared_error'] + (add_metrics if add_metrics else []))

        self._init_callbacks()

    def _format_data(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        return dataset.map(lambda x: (x[:,:-1], x[-1:,-1]))


class ExperimentAE(Experiment):
    def __init__(self,
                 lookback:int,
                 sampling:int,
                 model_config:dict,
                 months:str,
                 airport:str,
                 features:dict):
        super().__init__(
            lookback,
            sampling,
            model_config.get('batch_size'),
            months,
            airport,
            features)

        self.model_type = 'LSTMAE'
        self.n_units = model_config.get('n_units')
        self.act_function = model_config.get('act_function', 'tanh')

        self.model_path = paths.models_path / f'{self.model_type}_s{self.sampling}_lb{self.lookback}_u{self.n_units}'
        self.model_path_save = self.model_path / ('ep{epoch:03d}_loss{loss:.4f}_val{val_loss:.4f}.h5')
        self.model_path_best = self.model_path / 'best.h5'
        self.model_path_last = self.model_path / 'last.h5'
        self.model_path_log  = self.model_path / 'log.csv'

    def init_model(self, add_metrics = None):
        self.model = Sequential([
            LSTM(self.n_units,
                 activation=self.act_function,
                 input_shape=(self.lookback, self.num_features)),
            RepeatVector(self.lookback),
            LSTM(self.n_units,
                 activation=self.act_function,
                 return_sequences=True),
            TimeDistributed(Dense(1))
        ])
        self.model.compile(
            loss='mean_absolute_error',
            optimizer=adamw_experimental.AdamW(),
            metrics = ['mean_squared_error'] + (add_metrics if add_metrics else []))

        self._init_callbacks()

    def _format_data(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        return dataset.map(lambda x: (x[:,:-1], x[:,-1]))

    def get_y(self, dataset: tf.data.Dataset) -> np.array:
        return np.array([i.numpy()[-1] for i in dataset.map(lambda x,y: y,
                        num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)])


class ExperimentMH(Experiment): # WIP: una entrada con situación a corto plazo y otra con resumen de todo el vuelo
    def __init__(self,
                 lookback:int,
                 sampling:int,
                 model_config:dict,
                 months:str,
                 airport:str,
                 features:dict):
        super().__init__(
            lookback,
            sampling,
            model_config.get('batch_size'),
            months,
            airport,
            features)

        self.model_type = 'MH'
        self.n_units = model_config.get('n_units')
        self.act_function = model_config.get('act_function', 'tanh')

        self.model_path = paths.models_path / f'{self.model_type}_s{self.sampling}_lb{self.lookback}_u{self.n_units}'
        self.model_path_save = self.model_path / ('ep{epoch:03d}_loss{loss:.4f}_val{val_loss:.4f}.h5')
        self.model_path_best = self.model_path / 'best.h5'
        self.model_path_last = self.model_path / 'last.h5'
        self.model_path_log  = self.model_path / 'log.csv'

    def init_model(self, add_metrics = None):
        input_layer1 = Input(shape = (self.lookback, self.num_features))
        input_layer2 = Input(shape = (self.lookback, self.num_features))
        lstm_layer1 = LSTM(self.n_units,
                 activation=self.act_function)(input_layer1)
        lstm_layer2 = LSTM(self.n_units,
                 activation=self.act_function)(input_layer2)
        conc = tf.keras.layers.Concatenate(axis=-1)([lstm_layer1,lstm_layer2])
        fc_layers = Dense(1)(conc)

        self.model = Model(inputs = [input_layer1,input_layer2], outputs = fc_layers)

        self.model.compile(
            loss='mean_absolute_error',
            optimizer=Adam(),
            metrics = ['mean_squared_error'] + (add_metrics if add_metrics else []))

        self._init_callbacks()

    def _format_data(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        return dataset.map(lambda x: (x[:,:-1], x[-1:,-1]))


def main():
    n_units      = 20
    act_function = 'relu'
    batch_size   = 128

    lookback     = 20
    sampling     = 60

    epochs       = 20

    model_config = dict(
        n_units=n_units,
        act_function=act_function,
        batch_size=batch_size,
    )
    months       = '20220[123]'
    airport      = '*'
    glob_text    = f'{months}-{airport}' # 202001-LEBL

    numeric_feat   = ['latitude', 'longitude', 'altitude',
                    'departureDelay', 'vspeed', 'speed',
                    'day_of_week', 'track', 'wind_dir_degrees',
                    'wind_speed_kt', 'visibility_statute_mi',
                    'max_temp', 'min_temp', 'clouds', 'hav_distance']
    categoric_feat = ['time_of_day', 'operator', 'aerodromeOfDeparture', 'sky_status']
    objective      = ['RTA']
    ts_features  = ['latitude', 'longitude', 'altitude', 'vspeed', 'speed', 'track', 'hav_distance']
    nts_features = ['departureDelay', 'day_of_week', 'wind_dir_degrees','wind_speed_kt',
                    'visibility_statute_mi', 'max_temp', 'min_temp', 'time_of_day', 'operator',
                    'aerodromeOfDeparture', 'sky_status', 'clouds']

    feat_dict = {
        'numeric':numeric_feat,
        'categoric':categoric_feat,
        'objective':objective,
        'ts':ts_features,
        'nts':nts_features
    }

    experimento = ExperimentVanilla(
        lookback=lookback,
        sampling=sampling,
        model_config=model_config,
        months=months, 
        airport=airport,
        features=feat_dict
    )
    # experimento.init_model()

    experimento.load_model()
    # experimento.train(epochs=epochs, from_parquet=False)

    experimento._load_data(dataset='train',from_parquet=True,randomize=True)
    # experimento.evaluate(from_parquet=True, print_err=True)
    # experimento.evaluate_at_times()
    # experimento.evaluate_airports()

    for k, v in experimento.results.items():
        print(f'{k:10}\t{v}')

if __name__ == '__main__':
    main()