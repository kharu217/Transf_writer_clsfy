import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
import numpy
from encoder import EncoderLayer


class attn_classifier(nn.Module) :
    def __init__(self, d_model, num_heads, d_ff, dropout, num_layer, embed_d, vocab_size):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.num_layer = num_layer
        self.embed_d = embed_d
        self.vocab_size = vocab_size
    
        self.embedding = nn.Embedding(vocab_size, embed_d)

        self.encoder = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layer)])
        
        self.fc1 = nn.Linear(embed_d * d_model, int((embed_d * d_model)/2))
        self.fc2 = nn.Linear(int((embed_d * d_model)/2), 3)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        
    def forward(self, x) :
        batch, seq_len = x.size()
        assert seq_len < self.d_model, "seq_len must be short than max_seq_len"

        x = self.embedding(x)
        print(x.shape)
        x = F.pad(x, (0, 0, 0, self.d_model - seq_len, 0, 0), 'constant', 0).transpose(1, 2)
        #x.size() == (batch_size, embedding_d, d_model)

        for layer in self.encoder :
            attn_x = layer(x)
        output = torch.flatten(attn_x, 1)
        #attn_x.size() = (batch_size, embedding_d * d_model)

        output = self.relu(self.fc1(output))
        output = self.softmax(self.fc2(output))

        return output

if __name__ == "__main__" :
    test = attn_classifier(d_model=500,
                           num_heads=5, 
                           d_ff=100,
                           dropout=0.2,
                           num_layer=10,
                           embed_d=40,
                           vocab_size=10)

    A = torch.randint(0, 9, (1, 100))
    B = test(A)
