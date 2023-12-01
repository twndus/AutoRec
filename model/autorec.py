'''
autorec.py
'''
import torch
from torch import nn

class AutoRec(nn.Module):

    def __init__(self, input_dim, latent_dim):
        
        super(AutoRec, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # parameters
        self.encoder = nn.Embedding(self.input_dim, self_latent_dim)
        self.decoder = nn.Embedding(self.latent_dim, self.input_dim)
        
        # function
        self.sigmoid = nn.Sigmoid()

    
    def forward(self, x):
        x = self.encoder(x)
        x = self.sigmoid(x)
        x = self.decoder(x)
        return self.sigmoid(x)
