'''
moviedataset.py
'''
import os, requests, zipfile, io

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset

class MovieLens100KDataset(Dataset):

    def __init__(self, datapath, modelname, train=True):

        super(MovieLens100KDataset, self).__init__()
        self.datapath = datapath
        self.modelname = modelname

        self.download()
        self.ratings = pd.read_csv(
            os.path.join(self.datapath, 'u.data'), sep='\t',
            names=['user_id', 'item_id', 'rating', 'timestamp'])
        self.ratings.rating = 1

        self.num_users = len(self.ratings.user_id.unique())
        self.num_items = len(self.ratings.item_id.unique())
        self.num_data = self.ratings.shape[0]
        
        # user-item matrix
        if self.modelname == 'I-AutoRec':
            self.uimatrix = self.ratings.pivot(
                index='item_id', columns=['user_id'], values=['rating'])
        elif self.modelname == 'U-AutoRec':
            self.uimatrix = self.ratings.pivot(
                index='user_id', columns=['item_id'], values=['rating'])

        # zero-imputation
        self.uimask = self.uimatrix.notna().astype(int)
        self.uimatrix = self.uimatrix.fillna(3)

        train_idx, test_idx = train_test_split(range(self.uimatrix.shape[0]), test_size=.2, random_state=42, shuffle=True)

        if train:
            self.uimatrix = self.uimatrix[train_idx]
            self.uimask = self.uimask[train_idx]
        else:
            self.uimask = self.uimask[test_idx]
            self.uimatrix = self.uimatrix[test_idx]

        self.uimatrix = self.uimatrix.reset_index(drop=True)
        self.uimask = self.uimask.reset_index(drop=True)


    def download(self):
        if not os.path.exists(self.datapath):
            r = requests.get('https://files.grouplens.org/datasets/movielens/ml-100k.zip')
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall('/'.join(self.datapath.split('/')[:-1]))

    
    def __len__(self):
        return self.uimatrix.shape[0]

    def __getitem__(self, id_):
        return self.uimatrix.loc[id_, :].values, self.uimask.loc[id_,:].values

    def get_num_users(self):
        return self.num_users

    def get_num_items(self):
        return self.num_items

    def get_num_data(self):
        return self.num_data


class MovieLens1MDataset(Dataset):

    def __init__(self, datapath, modelname, train=True):

        super(MovieLens1MDataset, self).__init__()
        self.datapath = datapath
        self.modelname = modelname

        self.download()
        self.ratings = pd.read_csv(
            os.path.join(self.datapath, 'ratings.dat'), sep='::',
            names=['user_id', 'item_id', 'rating', 'timestamp'])
        self.ratings.rating = 1

        self.num_users = len(self.ratings.user_id.unique())
        self.num_items = len(self.ratings.item_id.unique())
        self.num_data = self.ratings.shape[0]
        
        # user-item matrix
        if self.modelname == 'I-AutoRec':
            self.uimatrix = self.ratings.pivot(
                index='item_id', columns=['user_id'], values=['rating'])
        elif self.modelname == 'U-AutoRec':
            self.uimatrix = self.ratings.pivot(
                index='user_id', columns=['item_id'], values=['rating'])

        # zero-imputation
        self.uimask = self.uimatrix.notna().astype(int)
        self.uimatrix = self.uimatrix.fillna(3)

        train_idx, test_idx = train_test_split(range(self.uimatrix.shape[0]), test_size=.2, random_state=42, shuffle=True)

        if train:
            self.uimatrix = self.uimatrix.iloc[train_idx,:]
            self.uimask = self.uimask.iloc[train_idx,:]
        else:
            self.uimask = self.uimask.iloc[test_idx,:]
            self.uimatrix = self.uimatrix.iloc[test_idx,:]

        self.uimatrix = self.uimatrix.reset_index(drop=True)
        self.uimask = self.uimask.reset_index(drop=True)


    def download(self):
        if not os.path.exists(self.datapath):
            r = requests.get('https://files.grouplens.org/datasets/movielens/ml-1m.zip')#'https://files.grouplens.org/datasets/movielens/ml-100k.zip')
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall('/'.join(self.datapath.split('/')[:-1]))

    
    def __len__(self):
        return self.uimatrix.shape[0]

    def __getitem__(self, id_):
        return self.uimatrix.loc[id_, :].values, self.uimask.loc[id_,:].values
        #return self.uimatrix.loc[id_, :].values

    def get_num_users(self):
        return self.num_users

    def get_num_items(self):
        return self.num_items

    def get_num_data(self):
        return self.num_data


if __name__ == '__main__':
    ml100k = MovieLens100KDataset('data/ml-100k', 'I-AutoRec')
    print('data length: ', len(ml100k))
    print('1th user data: ', ml100k[1])
