import pandas as pd

import torch
from torch.functional import F
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import build_vocab_from_iterator

from  src.utils.tokenize import tokeinzer, yield_tokens

class text2tensor_dataset(Dataset) :
    def __init__(self, url : str):
        super().__init__()
        self.csv =  pd.read_csv(url).to_dict()

        self.vocab = build_vocab_from_iterator(yield_tokens(self.csv), specials=["<pad>", "<unk>"])
        self.vocab.set_default_index(self.vocab["<unk>"])

    def __len__(self) :
        return len(self.csv["label"])
    
    def __getitem__(self, index):
        text2tensor = self.vocab(self.csv["text"][index].split(" "))
        label = F.one_hot(torch.tensor(self.csv["label"][index]), 3)
        return torch.tensor(text2tensor), label
