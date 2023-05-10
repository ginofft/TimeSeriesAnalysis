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
    
    def predict(self, model, predictPastValues: bool = False):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        lastLookBackWindown = self.data[self.input_field].iloc[-self.t:].values
        lastLookBackWindown = torch.tensor(lastLookBackWindown, dtype=torch.float32) 
        lastLookBackWindown.unsqueeze_(dim=0)
        if len(lastLookBackWindown.shape) == 2:
            lastLookBackWindown.unsqueeze_(dim=2)
        lastLookBackWindown.to(device)

        col_pred = [col + '_pred' for col in self.output_field]
        for col in col_pred:
            self.data[col] = pd.Series()
        
        output = model(lastLookBackWindown)
        output.squeeze_(dim=0)
        output = torch.reshape(output, (self.h, len(self.output_field)))

        for i in range(len(self.data), len(self.data)+self.h):
            for col_index, col in enumerate(col_pred):
                self.data.loc[i, col] = output[i - len(self.data), col_index].item()
        
        if predictPastValues:
            numGroup = (len(self.data) - (self.t+self.h))//(self.h) + 1
            startForecastIndex = (len(self.data) - self.t) % self.h
            for i in range(numGroup):
                startIndex = startForecastIndex + i*self.h
                endIndex = startIndex + self.t
                lookbackWindow = self.data[self.input_field].iloc[startIndex:endIndex].values
                lookbackWindow = torch.tensor(lookbackWindow, dtype=torch.float32)
                lookbackWindow.unsqueeze_(dim=0)
                if len(lookbackWindow.shape) == 2:
                    lookbackWindow.unsqueeze_(dim=2)
                lookbackWindow.to(device)
                output = model(lookbackWindow)
                output.squeeze_(dim=0)
                output = torch.reshape(output, (self.h, len(self.output_field)))
                for j in range(startIndex+self.t, startIndex+self.t+self.h):
                    for col_index, col in enumerate(col_pred):
                        self.data.loc[j, col] = output[j - (startIndex+self.t), col_index].item()
        
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
    