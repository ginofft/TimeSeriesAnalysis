import torch

class LSTMForecaster(torch.nn.Module):
    def __init__(self, 
                input_size,
                output_size,
                hidden_size,
                num_layers,
                dropout = 0.2):
        super(LSTMForecaster, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(input_size = input_size, 
                                  hiden_size = hidden_size,
                                  num_layers = self.num_layers, 
                                  batch_first=True)
        self.dropout = torch.nn.Dropout(dropout)
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output, _ = self.lstm(x)
        output = self.dropout(output[:,-1, :])
        output = self.fc(output)
        return output
