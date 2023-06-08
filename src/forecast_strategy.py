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
from tsai.inference import load_learner
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
                 model_path = None,
                 arch = 'PatchTST',
                 arch_config_path = None,
                 ):
        if model_path is not None:
            self._model = load_learner(model_path)
        else :
            self._model = None   
        self.arch = arch   
        if arch_config_path is not None:
            #self.arch_config = load_object(arch_config_path)
            with open(arch_config_path) as f:
                self.arch_config = json.load(f)
        else:
            self.arch_config = None
        
    def load_data(self, dataframe) -> None:
        self._data = dataframe

    def save_checkpoint(self, state, path):
        pass

    def load_checkpoint(self, path):
        pass

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
            columns = self._data.columns[1:]
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
                                        valid_size = 0.2 , test_size = 0)
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
        
        df = scale_pipe.fit_transform(df, scaler__idxs = train_split)
        X, y = prepare_forecasting_data(df, fcst_history=lookbackWindow, fcst_horizon=forecastWindow, 
                                        x_vars=input_field, y_vars=output_field)
        learner = TSForecaster(X, y, splits, batch_size = 16, pipelines = [preproc_pipe, scale_pipe],
                               arch = self.arch, arch_config=self.arch_config, metrics = [mse, mae], cbs = ShowGraph())
        learner.dls.valid.drop_last = True
        lr_max = learner.lr_find(show_plot=False).valley
        learner.fit_one_cycle(nEpochs, lr_max=lr_max)
        self._model = learner

    def forecast(self, 
                input_field, 
                output_field, 
                forecastWindow,
                lookbackWindow,
                datetime_col = None,
                freq = None) -> pd.DataFrame:
         
        df = self._data.copy()
        df = self._model.transform(df)

        old_X, _ = prepare_forecasting_data(df, fcst_history=lookbackWindow, fcst_horizon=forecastWindow,
                                            x_vars=input_field, y_vars=output_field)
        old_preds, *_ = self._model.get_X_preds(old_X)
        old_preds = torch.cat((old_preds[:-1, :,0], old_preds[-1,:,:].T), dim=0)

        lastlookbackWindow, _ = prepare_forecasting_data(df[-lookbackWindow:], fcst_history=lookbackWindow, fcst_horizon=0,
                                                          x_vars=input_field, y_vars=output_field)
        
        new_pred, *_ = self._model.get_X_preds(lastlookbackWindow)
        new_pred = torch.swapaxes(new_pred.squeeze(0), 0, 1)
        preds = torch.concat((old_preds, new_pred))

        dates = pd.date_range(start=df.loc[lookbackWindow, datetime_col],
                            periods = len(df.loc[lookbackWindow:]) + forecastWindow,
                            freq=freq)
        preds_df = pd.DataFrame(dates, columns=[datetime_col])
        preds_df.loc[:, output_field] = preds

        self._model.pipelines[1].inverse_transform(df)
        self._model.pipelines[1].inverse_transform(preds_df)
        
        new_columns = [col + '_pred' if col in output_field else col for col in preds_df.columns]
        preds_df = preds_df.rename(columns=dict(zip(df.columns, new_columns)))

        merged_df = pd.merge(df, preds_df, on=datetime_col, how='outer')

        return merged_df   
      
class SARIMAStrategy(ForecastStrategy):
    def __init__(self):
        pass
    
    def load_data(self, dataframe):
        self._data = dataframe

    def train(self, input_field, output_field, forecastWindow, m=12):
        self._model = auto_arima(self._data[output_field], seasonal=True, m=m)
        self._model.fit(self._data[output_field])
        return self._model.summary()
    
    def forecast(self, datetime_col, freq, input_field, output_field, forecastWindow):
        
        df = self._data.copy()

        dates = pd.date_range(start=df.loc[0, datetime_col],
                            periods = len(df.loc[0:]) + forecastWindow,
                            freq=freq)
        
        df['Month'] = dates[0:144]
        preds = self._model.predict(n_periods=forecastWindow)
        preds_df = pd.DataFrame(dates, columns=[datetime_col])
        preds_df.loc[-forecastWindow:, output_field] = preds

        new_columns = [col + '_pred' if col in output_field else col for col in preds_df.columns]
        preds_df = preds_df.rename(columns=dict(zip(df.columns, new_columns)))
    
        merged_df = pd.concat([df, preds_df])
        return merged_df
                                        