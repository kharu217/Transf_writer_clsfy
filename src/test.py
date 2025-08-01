import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module) :
    def __init__(self, d_model : int, d_ffn : int):
        super().__init__()
        hidden_dim = int(2 * d_ffn / 3)

        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x : torch.Tensor) :
        #SwiGLU(x) = (xW1 + b) @ silu(xW2)
        gate = F.silu(self.w1(x))
        data = self.w2(x)

        return self.w3(gate * data)

test = torch.rand(5, 4, 15, 15)
swiglu = SwiGLU(15, 1000)
print(swiglu(test).shape)
