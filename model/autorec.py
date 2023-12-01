'''
autorec.py
'''
import torch
from torch import nn

class AutoRec(nn.Module):

    def __init__(self, input_dim, latent_dim, output_func='sigmoid'):
        
        super(AutoRec, self).__init__()
        pass

    
    def forward(self, x):
        return x
