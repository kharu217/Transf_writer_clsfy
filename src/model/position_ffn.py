import torch
import torch.nn as nn
from swiglu import SwiGLU

class PositionWiseFeedForward(nn.Module) :
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.swiglu = SwiGLU(d_model)
    
    def forward(self, x) :
        return self.fc2(self.swiglu(self.fc1(x)))
