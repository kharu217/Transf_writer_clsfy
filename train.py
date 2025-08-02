import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from src.data.dataset import text2tensor_dataset
from src.model.encoder import EncoderLayer


def train(epch, lr, batch_size) :
    main_dataset = text2tensor_dataset("data\data.csv")

    train_len = int(0.9 * len(main_dataset))
    eval_len = len(main_dataset) - train_len

    train_dataset, eval_dataset = random_split(main_dataset, [train_len, eval_len])
    
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, shuffle=True)


    optimizer = optim.AdamW()

if __name__ == "__main__" :
    train(10, 19, 19)
