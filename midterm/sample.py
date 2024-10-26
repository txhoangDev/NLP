import numpy as np
import torch.nn as nn
import torch
from torch import optim

class FFNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size_s, output_size_g):
        super().__init__()
        self.V = nn.Linear(input_size, hidden_size)
        self.nonlin = nn.ReLU() 
        
        self.W1 = nn.Linear(hidden_size, output_size_s)
        self.W2 = nn.Linear(hidden_size, output_size_g) 
        self.log_sm_s = nn.LogSoftmax(dim=1)
        self.log_sm_g = nn.LogSoftmax(dim=1)
    
    def forward(self, f_x):
        z = self.nonlin(self.V(f_x))
        logits_s = self.W1(z)
        log_probs_s = self.log_sm_s(logits_s)
        logits_g = self.W2(z)
        log_probs_g = self.log_sm_g(logits_g)
        return log_probs_s, log_probs_g
    
    