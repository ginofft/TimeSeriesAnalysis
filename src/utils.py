from sklearn.base import TransformerMixin, BaseEstimator
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import torch

def plot_csv_by_time(csv_path, start_time, end_time):
    # Load CSV file into a pandas dataframe
    df = pd.read_csv(csv_path, parse_dates=['date'], index_col=['date'])
    
    # Select rows within the specified time range
    df = df.loc[start_time:end_time]
    
    # Generate a line plot for each column in the dataframe
    for col in df.columns:
        plt.plot(df.index, df[col], label=col)
    
    # Set plot title and axis labels
    plt.title(f"Data from {start_time} to {end_time}")
    plt.xlabel("Time")
    plt.ylabel("Value")
    
    # Display legend and show plot
    plt.legend()
    plt.show()

def save_checkpoint(state, path: Path, filename='latest.pth.tar'):
    outpath = path / filename
    torch.save(state, outpath)

def load_checkpoint(path, model, optimizer = None):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    state = torch.load(path, map_location=device)
    epoch = state['epoch']
    train_loss = state['train_loss']
    val_loss = state['val_loss']

    model.load_state_dict(state['model_state_dict'])
    if optimizer != None:
        optimizer.load_state_dict(state['optimizer_state_dict'])

    print("=> loaded checkpoint '{}' (epoch {})".format(True, epoch))
    print("Checkpoint's train loss is: {:.4f}".format(train_loss))
    print("Checkpoint's validation loss is: {:.4f}".format(val_loss))

    return epoch, train_loss, val_loss

class DateTimeConverter(TransformerMixin, BaseEstimator):
    def __init__(self, 
                 datetime_col):
        self.datetime_col = datetime_col
    def fit(self, X, y=None, **fit_params):
        X[self.datetime_col] = pd.to_datetime(X[self.datetime_col])
        return self
    def transform(self, X, y=None, **fit_params):
        X[self.datetime_col] = pd.to_datetime(X[self.datetime_col])
        return X

class EarlyStopper:
    def __init__(self, patience=10, delta = 0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    

    def __call__(self, val_loss):
        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss > self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.counter = 0    
