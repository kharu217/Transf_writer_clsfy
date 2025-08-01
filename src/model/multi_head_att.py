import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module) :
    def __init__(self, d_model : int, num_heads : int):
        super().__init__()
        assert d_model % num_heads == 0, "n_head * n != d_model"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product(self, Q, K, V, mask=None) :
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None :
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
    
    def split_heads(self, x) :
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        # (batch_size, seq_lenght, d_model) -> (batch_size, seq_length, num_heads, d_k)

    def combine_heads(self, x) :
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None) :
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_k(K))

        attn_output = self.scaled_dot_product(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output
    