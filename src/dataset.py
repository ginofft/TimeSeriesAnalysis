import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd

class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, data: pd.DataFrame, 
                 input_field, 
                 output_field,
                 scaler = None,
                 seq_len: int = 10):
        self.data = data
        self.seq_len = seq_len
        self.input_field = input_field
        self.output_field = output_field
        if scaler is not None:
            self.scaler = scaler
            self.data = self.scaler.transform(self.data)
        
    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        sequence = torch.tensor(self.data[self.input_field][idx:idx + self.seq_len].values, dtype=torch.float32)
        target = torch.tensor(self.data[self.output_field][idx + self.seq_len : idx + self.seq_len + 1].values, 
                              dtype=torch.float32).squeeze_(dim=0) # squeeze at 0th dimension, as this is before batch
        return sequence, target
    
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
                row_index = index + self.seq_len
                for i, col in enumerate(col_pred):
                    self.data.loc[row_index, col] = embedding[0][i].item()
        return self.data

    def plot_forecast_result(self):
        #df = self.data[self.seq_len:]
        df = self.data[self.seq_len:self.seq_len+self.seq_len*10]
        col_pred = [col + '_pred' for col in self.output_field]
        cols = col_pred + self.output_field

        for col in cols:
            plt.plot(df[col], label=col)
        plt.legend()
        plt.show()
            
    


        
    