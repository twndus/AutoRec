import torch

def rmse(predict, target):
    return torch.sqrt(torch.mean((predict - target)**2))
