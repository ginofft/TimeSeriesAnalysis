from abc import ABC, abstractmethod

from .forecast_models import LSTMForecaster
from .utils import load_checkpoint, save_checkpoint, EarlyStopper
from .dataset import TimeSeriesDataset
from .scaler import Scaler


import pandas as pd
import torch
from torch.utils.data import DataLoader
import os

class ForecastStrategy(ABC):
    @abstractmethod
    def load_data(self, inputFile):
        pass
    @abstractmethod
    def train(self, output_field, input_field, h, **kwargs):
        pass
    @abstractmethod
    def forecast(self, output_field, input_field, h):
        pass

class LSTMStrategy(ForecastStrategy):
    def __init__(self,
                 modelPath = None,
                 num_layers = 2,
                 hidden_size = 64,
                 lookback_length = 12,
                 verbose = False
                 ):
        self._modelPath = modelPath
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lookback_length = lookback_length
        self.verbose = verbose

    def load_data(self, inputFile) -> None:
        self._data = pd.read_csv(inputFile)

    def train(self, input_field, output_field, h,
              nEpochs = 10000,
              lr = 1e-5,
              batchSize = 48,
              saveEvery = 200,
              savePath = 'models/',
              ) -> None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
            print('No CUDA GPU detected. Using CPU instead.')

        train_dataset, val_dataset, test_dataset = self._csvToDataset(0.7, 0.15, input_field, output_field, h)
        
        self._model = LSTMForecaster(
                                input_size = len(input_field), 
                                output_size = len(output_field) * h, 
                                num_layers = self.num_layers, 
                                hidden_size = self.hidden_size)
        self._model.to(device)
        optimizer = torch.optim.AdamW(lr = lr, params = self._model.parameters())
        criterion = torch.nn.L1Loss()
        
        startEpoch = 0
        val_loss = float('inf')
        train_loss = float('inf')

        if self._modelPath is not None:
            epoch, train_loss, val_loss = load_checkpoint(self._modelPath, self._model, optimizer)
        early_stopper = EarlyStopper(patience=10, delta = 0)

        for epoch in range(startEpoch+1, nEpochs+1):
            epoch_train_loss = self._train_epoch(train_dataset, batchSize, criterion, optimizer, device)
            epoch_val_loss = self._val_epoch(val_dataset, batchSize, criterion, device)

            if self.verbose:
                print('Epoch {} completed: \nTrain loss: {:.6f} \nValidation loss: {:.6f}'.format(
                    epoch, epoch_train_loss, epoch_val_loss), flush = True)
            #Setup save
            epoch_state = {
                'epoch' : epoch,
                'train_loss' : epoch_train_loss,
                'val_loss' : epoch_val_loss,
                'input_field' : input_field,
                'output_field' : output_field,
                'model_state_dict' : self._model.state_dict(),
                'optimizer_state_dict' : optimizer.state_dict(),
            }
            if (epoch_val_loss < val_loss):
                val_loss = epoch_val_loss
                save_checkpoint(epoch_state, savePath, 'best.pth.tar')
            if epoch % saveEvery == 0:
                save_checkpoint(epoch_state, savePath, 'epoch{}.pth.tar'.format(epoch))
            
            early_stopper(epoch_val_loss)
            if early_stopper.early_stop:
                print(f'\n---------------------- Early stopping triggered!! ----------------------\nLowest val loss is: {early_stopper.best_score}\n', flush = True)                
                break
        
    def forecast(self, input_field, output_field, h):
        self._model = LSTMForecaster(
                                input_size = len(input_field), 
                                output_size = len(output_field) * h, 
                                num_layers = self.num_layers, 
                                hidden_size = self.hidden_size)
        
        load_checkpoint(self._modelPath, self._model)

        field_to_be_scaled = list(set(input_field + output_field))
        scaler = Scaler(self._data[field_to_be_scaled], 'minmax')
        dataset = TimeSeriesDataset(self._data, input_field, output_field, scaler, h, self.lookback_length)
        dataset.predict(self._model, predictPastValues=True)
        return dataset.data

    def _csvToDataset(self, train_ratio, val_ratio, input_field, output_field, h):
        train_end_index = int(train_ratio * len(self._data))
        val_end_index = int((train_ratio + val_ratio) * len(self._data))
        
        train_data = self._data[:train_end_index].reset_index()
        val_data = self._data[train_end_index:val_end_index].reset_index()
        test_data = self._data[val_end_index:].reset_index()

        field_to_be_scaled = list(set(input_field + output_field))
        scaler = Scaler(train_data[field_to_be_scaled], 'minmax')
        train_dataset = TimeSeriesDataset(train_data, input_field, output_field, 
                                          h=h, t=self.lookback_length, 
                                          scaler=scaler)
        val_dataset = TimeSeriesDataset(val_data, input_field, output_field,
                                        h=h, t=self.lookback_length,
                                        scaler=scaler)
        test_dataset = TimeSeriesDataset(test_data, input_field, output_field,
                                        h=h, t=self.lookback_length,
                                        scaler=scaler)
        return train_dataset, val_dataset, test_dataset
    
    def _train_epoch(self, dataset, batchSize, criterion, optimizer, device):
        dataloader = DataLoader(dataset,
                                batch_size=batchSize,
                                shuffle=True,
                                num_workers=2,
                                pin_memory=True)
        n_batches = len(dataloader)

        epoch_loss = 0
        self._model.train()
        for batch_id, (input, target) in enumerate(dataloader, 1):
            input = input.to(device)
            target = target.flatten(start_dim=1).to(device)

            embeddings = self._model(input)
            loss = criterion(embeddings, target).to(device)
            loss.backward()
            optimizer.step()  

            batch_loss = loss.item()
            epoch_loss += batch_loss

            del input, target, embeddings
            del loss
            # if batch_id % 200==0 or n_batches <= 10:
            #     print('Epoch[{}]({}/{}) Loss: {:.6f}'.format(epoch,
            #                                                 batch_id, 
            #                                                 n_batches,
            #                                                 batch_loss))
            del batch_loss
        avg_loss = epoch_loss / n_batches
        del dataloader
        
        if device == torch.device('cuda'):
            torch.cuda.empty_cache()
        return avg_loss
    
    def _val_epoch(self, dataset, batchSize, criterion, device):
        dataloader = DataLoader(dataset, 
                                batch_size = batchSize, 
                                num_workers = 2, 
                                shuffle = False,
                                pin_memory = True)
        epoch_loss = 0
        n_batches = len(dataloader)
        self._model.eval()
        with torch.no_grad():
            for input, target in dataloader:
                input = input.to(device)
                target = target.flatten(start_dim=1).to(device)

                embeddings = self._model(input)
                loss = criterion(embeddings, target).to(device)

                batch_loss = loss.item()
                epoch_loss += batch_loss

                del input, target, embeddings
                del loss
                del batch_loss
        avg_loss = epoch_loss / n_batches

        if device == torch.device('cuda'):
            torch.cuda.empty_cache()
        return avg_loss
    
class ARIMAStrategy(ForecastStrategy):
    
    def __init__(self)
    def load_data(self, inputFile):
        pass
    
