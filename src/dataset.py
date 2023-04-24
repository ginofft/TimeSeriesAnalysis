import torch
from torch.utils.data import DataLoader
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
    
    def predict(self, model, batchSize=8):
        for col in self.output_field:
            self.data[col+'_pred'] = pd.Series()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        with torch.no_grad():
            for input, targets in enumerate(self):
                input = input.to(device)
                targets = targets.to(device)
                embeddings = model(input)
                raise Exception('Not implemeted yet')


        
    