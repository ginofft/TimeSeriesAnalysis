import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, data: pd.DataFrame, 
                 input_field, 
                 output_field,
                 scaler = None,
                 t=4):
        self.data = data
        self.input_field = input_field
        self.output_field = output_field
        self.t = t
        if scaler is not None:
            self.scaler = scaler
            self.data = self.scaler.transform(self.data)
        
    def __len__(self):
        return len(self.data) - self.t

    def __getitem__(self, idx):
        sequence = torch.tensor(self.data[self.input_field][idx:idx + self.t].values, dtype=torch.float32)
        target = torch.tensor(self.data[self.output_field].iloc[idx+self.t].values, 
                              dtype=torch.float32) # squeeze at 0th dimension, as this is before batch
        return sequence, target
    
    def _inverse_scale(self):
        if self.scaler is None:
            raise Exception('Scaler is not defined')
        else:
            self.data = self.scaler.inverse_transform(self.data)
    
    def _getforecast(self, lastLookBackWindow, h):
        forecastResult = torch.zeros(lastLookBackWindow.shape[0] + h, lastLookBackWindow.shape[1])
        forecastResult[:lastLookBackWindow.shape[0], :] = lastLookBackWindow
        start_index = lastLookBackWindow.shape[0]
        for i in range(h):
            index = start_index + i
            lookbackWindow = forecastResult[index - self.lookback_length:index, :].unsqueeze(0)
            output = self._model(lookbackWindow)
            forecastResult[index, :] = output
        return forecastResult[-h:, :]
        
    def predict(self, model, forecastWindow: int =4, predictPastValues: bool = False):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        col_pred = [col + '_pred' for col in self.output_field]
        self.data[col_pred] = np.nan
        for row_index, (input, _)  in enumerate(self):
            input = input.unsqueeze(0)
            input = input.to(device)
            output = model(input)
            output = output.squeeze(0)
            output = output.cpu().detach().numpy()
            for col_index, col in enumerate(col_pred):
                self.data.loc[row_index+self.t, col] = output[col_index]    
        return self.data

    def plot_forecast_result(self):
        cols = self.output_field.copy()
        for col in self.output_field:
            cols.append(col + '_pred')
            self.scaler.min[col+'_pred'] = self.scaler.min[col]
            self.scaler.max[col+'_pred'] = self.scaler.max[col]
        
        self._inverse_scale()
        for col in cols:
            plt.plot(self.data[col], label=col)
        plt.legend()
        plt.show()
    