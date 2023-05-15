import torch
from torch.utils.data import DataLoader
from .dataset import TimeSeriesDataset

def train(
        train_set:TimeSeriesDataset,
        model,
        criterion,
        optimizer,
        device=torch.device("cuda"),
        batch_size=8,):
    
    dataloader = DataLoader(train_set, 
                            batch_size = batch_size, 
                            num_workers = 2, 
                            shuffle = True,
                            pin_memory = True)
    n_batches = len(dataloader)

    epoch_loss = 0
    model.train()
    for batch_id, (input, target) in enumerate(dataloader, 1):
        input = input.to(device)
        target = target.to(device)

        embeddings = model(input)
        loss = criterion(embeddings, target).to(device)
        optimizer.zero_grad() 
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

def inference(
        testSet : TimeSeriesDataset,
        model,
        criterion, 
        device = torch.device('cuda'),
        batch_size=8):
    dataloader = DataLoader(testSet, 
                            batch_size = batch_size, 
                            num_workers = 2, 
                            shuffle = False,
                            pin_memory = True)
    epoch_loss = 0
    n_batches = len(dataloader)
    model.eval()
    with torch.no_grad():
        for input, target in dataloader:
            input = input.to(device)
            target = target.to(device)

            embeddings = model(input)
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
