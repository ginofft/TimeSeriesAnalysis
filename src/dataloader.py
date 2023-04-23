import torch
import pandas as pd

class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, data: pd.DataFrame, 
                 input_field, 
                 output_field, 
                 seq_len: int = 10):
        self.data = data
        self.seq_len = seq_len
        self.input_field = input_field
        self.output_field = output_field
        
    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        sequence = torch.tensor(self.data[self.input_field][idx:idx + self.seq_len].values, dtype=torch.float32)
        target = torch.tensor(self.data[self.output_field][idx + self.seq_len : idx + self.seq_len + 1].values, dtype=torch.float32)
        return sequence, target
    