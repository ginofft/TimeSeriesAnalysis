import torch

class LSTMForecaster(torch.nn.Module):
    def __init__(self, 
                input_size,
                hidden_size,
                num_layers, 
                output_size):
        super(LSTMForecaster, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(input_size, hidden_size,
                                num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output, _ = self.lstm(x)
        output = self.fc(output[:,-1, :])
        return output
