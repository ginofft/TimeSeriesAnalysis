import argparse
import torch
import pandas as pd
from pathlib import Path

from src.scaler import Scaler
from src.dataset import TimeSeriesDataset
from src.train import train, inference
from src.forecast_models import LSTMForecaster
from src.utils import save_checkpoint, load_checkpoint, EarlyStopper

parser = argparse.ArgumentParser(description='Forecasting and Anomaly Detection Toolbox')

#model, optimizer and criterion parameters
parser.add_argument('--lr', type = float, default=1e-4, help='learning rate')
parser.add_argument('--t', type = int, default=72, help='lookback window')
parser.add_argument('--h', type=int, default = 24, help = 'forecasting windows')
parser.add_argument('--hiddenSize', type = int, default=256, help='hidden size')
parser.add_argument('--numLayers', type = int, default=16, help='No. of LSTM layers')

#training parameters
parser.add_argument('--mode', type=str, default='train', 
                    help='training mode or inference mode',
                    choices=['train', 'inference'],
                    required=True)
parser.add_argument('--nEpochs', type = int, default=50, help='No. epochs')
parser.add_argument('--saveEvery', type = int, default = 10, 
                    help='no. epoch before a save is created')
parser.add_argument('--earlyStopping', type = bool, default = True,
                    help = 'Whether or not to use early stopping')

#Data parameters
parser.add_argument('--batchSize', type=int, default = 16, help='batch size')
parser.add_argument('--datasetPath', type = str, default='',
                    help='Path to dataset csv file')
parser.add_argument('--inputField', nargs= '+', help='The field(s) used for training')
parser.add_argument('--outputField', nargs= '+', help='The target field(s)')

#check point parameters
parser.add_argument('--savePath', type = str, default = '',
                    help = 'Path to save checkpoint to')
parser.add_argument('--loadPath', type = str, default = '',
                    help = 'Path to load checkpoint from')

if __name__ == '__main__':
    opt = parser.parse_args()
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        print('No CUDA GPU detected. Using CPU instead.')
    
    if opt.datasetPath == '':
        raise ValueError('Please specify the path to the dataset')
    
    train_df = pd.read_csv(opt.datasetPath+'/train.csv')
    val_df = pd.read_csv(opt.datasetPath+'/val.csv')
    test_df = pd.read_csv(opt.datasetPath+'/test.csv')       

    field_to_be_scaled = list(set(opt.inputField + opt.outputField))
    scaler = Scaler(train_df[field_to_be_scaled], 'minmax')

    train_dataset = TimeSeriesDataset(train_df, opt.inputField, opt.outputField, h=opt.h, t=opt.t, scaler=scaler)
    val_dataset = TimeSeriesDataset(val_df, opt.inputField, opt.outputField, h=opt.h, t=opt.t, scaler=scaler)
    test_dataset = TimeSeriesDataset(test_df, opt.inputField, opt.outputField, h=opt.h, t=opt.t, scaler=scaler)

    model = LSTMForecaster(input_size=len(opt.inputField), 
                           output_size=len(opt.outputField)*opt.h, 
                           hidden_size=opt.hiddenSize, 
                           num_layers=opt.numLayers)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr = opt.lr)
    criterion = torch.nn.L1Loss()

    if opt.mode == 'train':
        print('------------------------------- Training -------------------------------')
        startEpoch = 0
        val_loss = float('inf')
        train_loss = float('inf')
        
        if opt.loadPath:
            startEpoch, train_loss, val_loss = load_checkpoint(
                                                Path(opt.loadPath),
                                                model,
                                                optimizer)
        
        if opt.earlyStopping:
            early_stopper = EarlyStopper(patience=10, delta=0)
        else:
            early_stopper = None

        for epoch in range(startEpoch+1, opt.nEpochs+1):
            #Calculate Loss
            epoch_train_loss = train(train_dataset, model, 
                                    criterion, optimizer, 
                                    device, opt.batchSize, epoch)
            epoch_val_loss = inference(val_dataset, model, 
                                    criterion, device, 
                                    opt.batchSize)
            print('Epoch {} completed: \nTrain loss: {:.6f} \nValidation loss: {:.6f}'.format(
                epoch, epoch_train_loss, epoch_val_loss), flush = True)
            
            #Setup save
            epoch_state = {
                'epoch' : epoch,
                'train_loss' : epoch_train_loss,
                'val_loss' : epoch_val_loss,
                'input_field' : opt.inputField,
                'output_field' : opt.outputField,
                'model_state_dict' : model.state_dict(),
                'optimizer_state_dict' : optimizer.state_dict(),
            }
            if (epoch_val_loss < val_loss):
                val_loss = epoch_val_loss
                save_checkpoint(epoch_state, Path(opt.savePath), 'best.pth.tar')
            if (epoch % opt.saveEvery) == 0:
                save_checkpoint(epoch_state, Path(opt.savePath), 'epoch{}.pth.tar'.format(epoch))

            #Early stopping 
            if early_stopper is not None:    
                early_stopper(epoch_val_loss)
                if early_stopper.early_stop:
                    print('\n---------------------- Early stopping triggered!! ----------------------\n', flush = True)
                    break
    else:
        if opt.loadPath:
            startEpoch, train_loss, val_loss = load_checkpoint(Path(opt.loadPath),
                                                            model,
                                                            optimizer)
        else:
            raise Exception('Please point to a model using ---loadPath')

        print('---------------------------Running Inferenece---------------------------')
               
        test_loss = inference(test_dataset, model, criterion, device, opt.batchSize)

        print('Test loss: {:.6f}'.format(test_loss), flush=True)
