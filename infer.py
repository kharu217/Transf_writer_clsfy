import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.encoder import EncoderLayer

test = EncoderLayer(100, 4, 10, 0.2)
A = torch.rand(10, 10, 100)

print(test(A).shape)
