from abc import ABC, abstractmethod
from .forecast_models import LSTMForecaster
import pandas as pd
import os

class ForecastStrategy(ABC):
    @property
    @abstractmethod
    def forecast_ready(self):
        pass
    @abstractmethod
    def load_data(self, inputFile):
        pass
    @abstractmethod
    def train(self, output_field, input_field, h):
        pass
    @abstractmethod
    def forecast(self, output_field, input_field, h):
        pass

class LSTMStrategy(ForecastStrategy):
    def __init__(self,
                 modelPath = None,
                 num_layers = 2,
                 hidden_size = 64,
                 lookback_length = 12,
                 ):
        self._modelPath = modelPath
        self.num_layers = num_layers
        self.hidden_size = hidden_size


    def load_data(self, inputFile) -> None:
        self._data = pd.read_csv(inputFile)

    def train(self, output_field, input_field, h):
        self._model = LSTMForecaster(len(input_field), 
                                     len(output_field) * h, 
                                     self.num_layers, 
                                     self.hidden_size)
        
        self._model.train()
    
    def forecast(self, output_field, input_field, h):
        pass
