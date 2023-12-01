'''
train.py
'''
import os, random, string
import numpy as np

import torch
from torch.utils.data import DataLoader, random_split
import wandb

from dataset.moviedataset import MovieLens100KDataset,MovieLens1MDataset
from model.autorec import AutoRec
from model.metrics import rmse
from model.loss import RMSELoss

SEED = 20231201
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = True

def main():

    device = torch.device('mps')
    modelname = 'I-AutoRec'
    monitor = True
    save = True

    # wandb config
    EPOCHS = 500 
    LEARNING_RATE = 1e-2
    BATCH_SIZE = 128
    LATENT_DIM = 200#512
    
    if monitor:
        config = {"epochs": EPOCHS, "learning_rate": LEARNING_RATE,
                  "batch_size": BATCH_SIZE, "latent_dim": LATENT_DIM}
        run = wandb.init(project='autorec-movielens100k', config=config)
    
    # dataset
    train_dataset = MovieLens1MDataset('data/ml-1m', modelname, True)
    test_dataset = MovieLens1MDataset('data/ml-1m', modelname, False)

    # data loader
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # model training
    if modelname == 'I-AutoRec':
        input_dim = train_dataset.get_num_users()
    elif modelname == 'U-AutoRec':
        input_dim = train_dataset.get_num_items()
    else:
        raise NameError(f'modelname must be "I-AutoRec" or "U-AutoRec", {modelname} not allowed.')

    model = AutoRec(input_dim, LATENT_DIM).to(device)
    model.init_params()
    
    # hyperparameters
    epochs = EPOCHS 
    learning_rate = LEARNING_RATE

    loss_f = RMSELoss().to(device)
    optim = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=(1e-3)/2)

    for e in range(epochs):

        tot_train_loss, tot_train_rmse = 0,0
        tot_test_loss, tot_test_rmse = 0,0
        num_batches = len(train_dataloader)

        for batch, (x, mask) in enumerate(train_dataloader):
            
            x = x.float().to(device)
            predict = model(x)

            optim.zero_grad()
            obs_index = np.where(mask == 1)
            #obs_index = np.where(x.to('cpu') != 3)
            loss = loss_f(predict[obs_index], x[obs_index])
            loss.backward()
            optim.step()

            train_rmse = rmse(predict[obs_index], x[obs_index])

            tot_train_loss += loss.item()
            tot_train_rmse += train_rmse.item()

        tot_train_loss /= num_batches
        tot_train_rmse /= num_batches
        
        num_batches = len(test_dataloader)

        with torch.no_grad():
            for batch, (x, mask) in enumerate(test_dataloader):

                x = x.float().to(device)
                predict = model(x)

                #obs_index = np.where(x.to('cpu') != 3)
                obs_index = np.where(mask == 1)
                loss = loss_f(predict[obs_index], x[obs_index])

                test_rmse = rmse(predict[obs_index], x[obs_index])

                tot_test_loss += loss.item()
                tot_test_rmse += test_rmse.item()
        
#            tot_train_loss = len(train_dataset)
#            tot_train_rmse /= len(train_dataset)
#            
#            tot_test_loss = len(test_dataset)
#            tot_test_rmse /= len(test_dataset)
            tot_test_loss /= num_batches
            tot_test_rmse /= num_batches

        print(f"[epoch {e}] train loss: {round(tot_train_loss,3)}, test loss: {round(tot_test_loss,3)}, train rmse: {round(tot_train_rmse, 3)}, test rmse: {round(tot_test_rmse,3)}")

        if monitor:
            wandb.log({'train_loss': tot_train_loss, 'test_loss': tot_test_loss, 
                       'train_rmse': tot_train_rmse, 'test_rmse': tot_test_rmse})

    if save:
        model_save_dir = './result'
        savename = f"ml100k-{modelname}-{''.join(random.sample(string.ascii_lowercase, 5))}.pt"
        torch.save(model.state_dict(), os.path.join(model_save_dir, savename))

if __name__ == '__main__':
    main()
