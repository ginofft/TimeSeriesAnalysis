import torch

class LSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout = 0.2):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.drop_out = dropout
        self.lstm = torch.nn.LSTM(input_size, hidden_size,
                                    num_layers, batch_first=True,
                                    dropout=self.drop_out)
        self.fc = torch.nn.Linear(hidden_size, output_size)