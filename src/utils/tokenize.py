import pandas as pd
from torchtext.data.utils import get_tokenizer

dataset = pd.read_csv("data\data.csv").to_dict()
tokeinzer = get_tokenizer("basic_english")

def yield_tokens(data_iter) :
    for i in range(len(data_iter["label"])) :
        yield tokeinzer(data_iter['text'][i])


