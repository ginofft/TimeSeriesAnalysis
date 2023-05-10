from src.context import Context
from src.forecast_strategy import *
import argparse
import pandas as pd

parser = argparse.ArgumentParser(description='Forecasting and Anomaly Detection Toolbox')

parser.add_argument('--csvPath', type = str, default='', help='Path to dataset csv file')
parser.add_argument('--inputField', nargs= '+', help='The field(s) used for training')
parser.add_argument('--outputField', nargs= '+', help='The target field(s)')
parser.add_argument('--h', type=int, default = 24, help = 'forecasting windows')

parser.add_argument('--model', type=str, default='sarima', choices=['sarima', 'lstm'], help='forecasting model')

#samira argument
parser.add_argument('--p', type=int, default = 1, help = 'order of AR')
parser.add_argument('--d', type=int, default = 1, help = 'order of I')
parser.add_argument('--q', type=int, default = 1, help = 'order of MA')
parser.add_argument('--sp', type=int, default = 1, help = 'order of seasonal AR')
parser.add_argument('--sd', type=int, default = 1, help = 'order of seasonal I')
parser.add_argument('--sq', type=int, default = 1, help = 'order of seasonal MA')
parser.add_argument('--s', type=int, default = 24, help = 'seasonal period')

#lstm argument
parser.add_argument('--lr', type = float, default=1e-5, help='learning rate')
parser.add_argument('--batchSize', type=int, default = 16, help='lstm batch size')
parser.add_argument('--nEpochs', type = int, default=50000, help='lstm no. training epochs')
parser.add_argument('--saveEvery', type = int, default = 200, help='lstm no. epoch before a save is created')

parser.add_argument('--t', type = int, default=72, help='lookback window')
parser.add_argument('--hiddenSize', type = int, default=256, help='hidden size')
parser.add_argument('--numLayers', type = int, default=16, help='No. of LSTM layers')

#save, load model
parser.add_argument('--savePath', type = str, default = '', help = 'Path to save checkpoint to')
parser.add_argument('--loadPath', type = str, default = '', help = 'Path to load checkpoint from')

if __name__ == '__main__':
    opt = parser.parse_args()
    if opt.csvPath == '':
        raise ValueError('Please specify the path to the dataset')
    if opt.inputField == None:
        raise ValueError('Please specify the input field(s)')
    if opt.outputField == None:
        raise ValueError('Please specify the output field(s)')
    
    df = pd.read_csv(opt.csvPath)

    if opt.model == 'sarima':
        strategy = SARIMAStrategy()
    if opt.model == 'lstm':
        strategy = LSTMStrategy()
    
    context = Context(strategy=strategy, data = df, 
                      input_field = opt.inputField, output_field = opt.outputField, 
                      h = opt.h)
    
    if opt.model == 'sarima':
        context.train(m = opt.s)
        print(context.forecast(), flush=True)
    if opt.model == 'lstm':
        raise Exception('Not Implemented Yet!')