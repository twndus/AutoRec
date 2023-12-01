'''
autorec.py
'''
import torch
from torch import nn
from torchsummary import summary

class AutoRec(nn.Module):

    def __init__(self, input_dim, latent_dim):
        
        super(AutoRec, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # parameters
        self.encoder = nn.Linear(self.input_dim, self.latent_dim)
        self.decoder = nn.Linear(self.latent_dim, self.input_dim)

        # function
        self.sigmoid = nn.Sigmoid()
        self.identity = nn.Identity()

    def init_params(self):

        for layer in [self.encoder, self.decoder]:
            nn.init.kaiming_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.sigmoid(x)
        x = self.decoder(x)
        return self.identity(x)

if __name__ == '__main__':
    autorec = AutoRec(input_dim=50, latent_dim=10)
    summary(autorec, (1, 50))
