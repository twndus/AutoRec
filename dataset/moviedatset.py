'''
moviedataset.py
'''
import torch
from torch.utils.data import Dataset

class MovieLens100KDataset(Dataset):

    def __init__(self, path):
        super(MovieLens100KDataset, self).__init__()
        pass
    
    def __len__(self):
        pass

    def __get_item__(self, user_id):
        pass


if __name__ == '__main__':
    ml100k = MovieLens100KDataset('./')
