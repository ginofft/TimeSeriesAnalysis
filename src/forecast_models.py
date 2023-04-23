import torch

class LSTM(torch.nn.Module):
    def __init__(self, 
                input_size,
                hidden_size,
                num_layers, 
                output_size, 
                dropout = 0.2,
                device = torch.device('')):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.drop_out = dropout
        self.lstm = torch.nn.LSTM(input_size, hidden_size,
                                num_layers, batch_first=True,
                                dropout=self.drop_out)
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        output, _  = self.lstm(x, (h0, c0))
        output = self.fc(output[:,-1, :])
        return output
