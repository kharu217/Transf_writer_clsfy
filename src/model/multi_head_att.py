import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module) :
    def __init__(self, d_model : int, num_heads : int):
        super().__init__()
        assert d_model % num_heads == 0, "n_head * n != d_model"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
