import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd

class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, data: pd.DataFrame, 
                 input_field, 
                 output_field,
                 scaler = None,
                 h=24,
                 t=72):
        self.data = data
        self.input_field = input_field
        self.output_field = output_field
        self.h = h
        self.t = t
        if scaler is not None:
            self.scaler = scaler
            self.data = self.scaler.transform(self.data)
        
    def __len__(self):
        return len(self.data) - (self.h + self.t) + 1

    def __getitem__(self, idx):
        sequence = torch.tensor(self.data[self.input_field][idx:idx + self.t].values, dtype=torch.float32)
        target = torch.tensor(self.data[self.output_field][idx + self.t : idx + self.t + self.h].values, 
                              dtype=torch.float32).squeeze_(dim=0) # squeeze at 0th dimension, as this is before batch
        return sequence, target
    
    def _inverse_scale(self):
        if self.scaler is None:
            raise Exception('Scaler is not defined')
        else:
            self.data = self.scaler.inverse_transform(self.data)
    
    def predict(self, model):
        col_pred = [col + '_pred' for col in self.output_field]
        for col in col_pred:
            self.data[col] = pd.Series()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        dataloader = DataLoader(self, batch_size=1, shuffle=False)
        with torch.no_grad():
            for index, (input, target) in enumerate(dataloader):
                input = input.to(device)
                target = target.to(device)
                embedding = model(input)
                embedding = torch.reshape(embedding, target.shape)
                start_index = index + self.t
                end_index = index + self.t + self.h
                for row_index in range(start_index, end_index):
                    for col_index, col in enumerate(col_pred):
                        self.data.loc[row_index, col] = embedding[0][row_index - start_index, col_index].item()
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
    