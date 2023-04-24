import torch
import numpy as np
import pandas as pd

class Scaler():
    def __init__(self, 
                data: pd.DataFrame, 
                scaler_type='minmax'):
        self.data = data
        self.scaler_type = scaler_type
        self._calculate_scaler()
    
    def _calculate_scaler(self):
        if self.scaler_type == 'minmax':
            self.min = self.data.min()
            self.max = self.data.max()
        else:
            raise Exception('Not implemented yet')
    
    def transform(self, data):
        if self.scaler_type == 'minmax':
            return (data - self.min) / (self.max - self.min)
        else:
            raise Exception('Not implemented yet')