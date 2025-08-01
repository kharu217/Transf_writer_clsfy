import torch
import torch.nn as nn

class EncoderLayer(nn.Module) :
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
