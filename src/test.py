import torch
import torch.nn as nn

A = torch.randint(low=0, high=99, size=(1, 100))
print(A.shape)
A = nn.functional.pad(A, (0, 100), 'constant', 0)
print(A.shape)
