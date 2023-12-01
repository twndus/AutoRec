'''
test.py
'''
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset.moviedataset import MovieLens100KDataset
from model.autorec import AutoRec
from model.metrics import rmse

def main():
    
    device = 'cpu'
    save_dir = 'result'
    latest_model = max(os.listdir(save_dir), 
        key=lambda x:os.path.getmtime(os.path.join(save_dir, x)))

    modelpath = os.path.join(save_dir, latest_model)
    modelname = '-'.join(latest_model.split('-')[1:3])

    # dataset
    test_dataset = MovieLens100KDataset('data/ml-100k', modelname, False)
    # dataloader
    test_dataloader = DataLoader(test_dataset, batch_size=256)

    # model training
    if modelname == 'I-AutoRec':
        input_dim = test_dataset.get_num_users()
    elif modelname == 'U-AutoRec':
        input_dim = test_dataset.get_num_items()
    else:
        raise NameError(f'modelname must be "I-AutoRec" or "U-AutoRec", {modelname} not allowed.')

    print(len(test_dataset))
    latent_dim = 50
    model = AutoRec(input_dim, latent_dim).to(device)
    model.load_state_dict(torch.load(modelpath))
    model.eval()

    predicts = []
    with torch.no_grad():
        tot_test_rmse = 0

        for x in test_dataloader:

            x = x.float().to(device)
            predict = model(x)
            
            print(predict)
            print(np.where(predict[0]>=.5, 1, 0))
            obs_index = np.where(x.to('cpu') != 0)

            test_rmse = rmse(predict, x)

            tot_test_rmse += test_rmse
            predicts.append(predict)
        
        tot_test_rmse /= len(test_dataset)

    print(f"test rmse: {tot_test_rmse}")

    movie_df = pd.read_csv('data/ml-100k/u.item', header=None, encoding='latin-1', sep='|')
    item_col = ['movie id', 'movie title', 'release date', 'video release date', 'IMDb URL', 
                'unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime', 
                'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 
                'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    movie_df.columns = item_col

    for i in range(len(predict)):
        topk_index = np.argsort(predict[i])[-5:] 
        print(topk_index)
        #print([movie_df[movie_df['movie id']==i.item()]['movie title'].values[0] for i in topk_index])

pass

if __name__ == '__main__':
    main()
