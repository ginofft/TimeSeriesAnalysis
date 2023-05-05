import pandas as pd
from .forecast_strategy import ForecastStrategy

class Context():
    def __init__(self, strategy, data : pd.DataFrame, outpuf_field, input_field, h):
        self._strategy = strategy
        self._data = data
        self.output_field = outpuf_field
        self.input_field = input_field
        self.h = h
    
    @property
    def strategy(self) -> ForecastStrategy:
        return self._strategy
    @strategy.setter
    def strategy(self, strategy: ForecastStrategy) -> None:
        self._strategy = strategy
    
    @property
    def data(self) -> pd.DataFrame:
        return self._data
    @data.setter
    def data(self, data: pd.DataFrame) -> None:
        self._data = data

    def train(self) -> None:
        self.strategy.load_data(self.data)
        self.strategy.train(self.output_field, self.input_field, self.h)
        self.strategy.forecast_ready = True
        print('------------------------- Training completed!! -------------------------')

    def forecast(self):
        if self.strategy.forecast_ready:
            return self.strategy.forecast(self._data, self._output_field, self._input_field, self._h)
        else:
            raise Exception('Model is not trained yet, please train or load a trained model first')
    
    
