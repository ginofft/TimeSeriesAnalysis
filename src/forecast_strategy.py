from abc import ABC, abstractmethod

from .forecast_models import LSTMForecaster
from .utils import load_checkpoint, save_checkpoint, EarlyStopper
from .dataset import TimeSeriesDataset
from .scaler import Scaler

import pandas as pd
import numpy as np
from pmdarima import auto_arima
import torch
from torch.utils.data import DataLoader
import os

class ForecastStrategy(ABC):
    @abstractmethod
    def load_data(self, dataframe):
        pass
    @abstractmethod
    def train(self, input_field, output_field, h, **kwargs):
        pass
    @abstractmethod
    def forecast(self, input_field, oput_field, h):
        pass

class DeepLearningStrategy(ForecastStrategy):
    def __init__(self,
                 arch = 'PatchTST',
                 arch_config_path = 'models/PatchTST.json'):
        self.arch = arch    
        self.arch_config_path = arch_config_path
        
    def load_data(self, dataframe) -> None:
        pass

    def train(self) -> None:
        pass

    def forecast(self, input_field, output_field, h):
        pass

class SARIMAStrategy(ForecastStrategy):
    def __init__(self):
        pass
    def load_data(self, dataframe):
        self._data = dataframe

    def train(self, input_field, output_field, h, m=12):
        self._model = auto_arima(self._data[output_field], seasonal=True, m=m)
        self._model.fit(self._data[output_field])
        return self._model.summary()
    
    def forecast(self, input_field, output_field, h):
        preds = self._model.predict(n_periods=h)
        nan_df = pd.DataFrame(np.nan, index=preds.index, columns=self._data.columns)
        self._data = pd.concat([self._data, nan_df])
        self._data['pred'] = preds
        return self._data
                                        