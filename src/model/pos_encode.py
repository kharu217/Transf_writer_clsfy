import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module) :
    def __init__(self, d_model, max_seq_length):
        super().__init__()

        self.pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', self.pe.unsqueeze())

    def forward(self, x) :
        return x + self.pe[:, :x.size(1)]
