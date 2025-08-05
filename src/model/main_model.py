import torch
import torch.nn as nn
import torchsummary
from encoder import EncoderLayer


class attn_classifier(nn.Module) :
    def __init__(self, d_model, num_heads, d_ff, dropout, num_layer):
        super().__init__()
        self.encoder = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layer)])
        
    def forward(self, x) :
        for layer in self.encoder :
            x = layer(x)
        return x

if __name__ == "__main__" :
    test = attn_classifier(512, 2, 10, 0.2, 5)
    torchsummary.summary(test, (100, 512), 1)
