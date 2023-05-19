from abc import ABC, abstractmethod

from .forecast_models import LSTMForecaster
from .utils import DateTimeConverter
from .dataset import TimeSeriesDataset
from .scaler import Scaler

import sklearn
from pathlib import Path
import pandas as pd
import numpy as np
from pmdarima import auto_arima
from tsai.basics import *

class ForecastStrategy(ABC):
    @abstractmethod
    def load_data(self, dataframe):
        pass
    @abstractmethod
    def train(self, input_field, output_field, forecastWindow, **kwargs):
        pass
    @abstractmethod
    def forecast(self, input_field, oput_field, forecastWindow, **kwargs):
        pass

class DeepLearningStrategy(ForecastStrategy):
    def __init__(self,
                 model_path,
                 arch = 'PatchTST',
                 arch_config_path = 'models/PatchTST.json',
                 ):
        self.arch = arch    
        self.arch_config_path = arch_config_path
        
    def load_data(self, dataframe) -> None:
        self._data = dataframe

    def train(self, 
              input_field, 
              output_field,
              forecastWindow,
              lookbackWindow,
              nEpochs: int = 50,
              preprocessPipePath: Path = None, 
              scalePipePath: Path = None,
              datetime_col = None,
              freq = None) -> None:
        
        fields_to_be_process = list(set(input_field + output_field))

        if preprocessPipePath is not None:
            preproc_pipe = load_object(preprocessPipePath)
        else:
            columns = self._data[fields_to_be_process]
            method = 'ffill'
            preproc_pipe = sklearn.pipeline.Pipeline([
            ('datetime_converter', DateTimeConverter(datetime_col=datetime_col)), # convert datetime column to datetime
            ('shrinker', TSShrinkDataFrame()), # shrink dataframe memory usage
            ('drop_duplicates', TSDropDuplicates(datetime_col=datetime_col)), # drop duplicate rows (if any)
            ('add_mts', TSAddMissingTimestamps(datetime_col=datetime_col, freq=freq)), # ass missing timestamps (if any)
            ('fill_missing', TSFillMissing(columns=columns, method=method)), # fill missing data (1st ffill. 2nd value=0)
            ], 
            verbose=True)

        df = preproc_pipe.fit_transform(self._data)

        splits = get_forecasting_splits(df, fcst_history=lookbackWindow, fcst_horizon=forecastWindow, 
                                        val_size = 0.2 , test_size = 0)
        for split in splits:
            if (lookbackWindow+forecastWindow) > len(split):
                raise Exception('The sum of lookback and forecast windows are longer than train/validation/test set')
        
        train_split = splits[0]

        if scalePipePath is not None:
            scale_pipe = load_object(scalePipePath)
        else:
            scale_pipe = sklearn.pipeline.Pipeline([
                ('scaler', TSStandardScaler(fields_to_be_process))
            ], verbose = True)
        
        df = scale_pipe.fit_transform(df, scaler_idx = train_split)
        X, y = prepare_forecasting_data(df, fcst_history=lookbackWindow, fcst_horizon=forecastWindow, 
                                        x_vars=input_field, y_vars=output_field)
        learner = TSForecaster(X, y, splits, batch_size = 16, pipelines = [preproc_pipe, scale_pipe],
                               arch = self.arch, metrics = [mse, mae], cbs = ShowGraph())
        learner.dls.valid.drop_last = True
        lr_max = learner.lr_find(show_plot=False).valley
        learner.fit_one_cycle(nEpochs, lr_max=lr_max)
        self._model = learner


    def forecast(self, 
                input_field, 
                output_field, 
                forecastWindow,
                lookbackWindow,
                preprocessPipePath: Path = None, 
                scalePipePath: Path = None,
                datetime_col = None,
                freq = None) -> None:
        
        fields_to_be_process = list(set(input_field + output_field))

        df = self._data.copy()
        for pipe in self._model.pipelines:
            df = pipe.tranform(df)
        
        splits = get_forecasting_splits(df, fcst_history=lookbackWindow, fcst_horizon=forecastWindow, 
                                        val_size = 0 , test_size = 0)
        
        

class SARIMAStrategy(ForecastStrategy):
    def __init__(self):
        pass
    
    def load_data(self, dataframe):
        self._data = dataframe

    def train(self, input_field, output_field, forecastWindow, m=12):
        self._model = auto_arima(self._data[output_field], seasonal=True, m=m)
        self._model.fit(self._data[output_field])
        return self._model.summary()
    
    def forecast(self, input_field, output_field, forecastWindow):
        preds = self._model.predict(n_periods=forecastWindow)
        nan_df = pd.DataFrame(np.nan, index=preds.index, columns=self._data.columns)
        self._data = pd.concat([self._data, nan_df])
        self._data['pred'] = preds
        return self._data
                                        