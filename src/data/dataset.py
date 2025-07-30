import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class dataset(Dataset) :
    def __init__(self, url : str):
        super().__init__()
        self.csv = pd.read_csv(url)
        self.text = self.csv['text'].values.tolist()
        self.label = self.csv['label'].values.tolist()

    def __len__(self) :
        return len(self.label)
    
    def __getitem__(self, index):
        return self.text[index], self.label[index]
