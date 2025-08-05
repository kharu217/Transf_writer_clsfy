import torch
import torch.nn as nn
import torch.nn.functional as F

tf = nn.Linear(512, 10)
A = torch.rand(2, 512)

B = tf(A)
print(B)
