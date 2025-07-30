import pandas as pd
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

dataset = pd.read_csv("data\data.csv").to_dict()
tokeinzer = get_tokenizer("basic_english")

def yield_tokens(data_iter) :
    for i in range(len(data_iter["label"])) :
        yield tokeinzer(data_iter['text'][i])
vocab = build_vocab_from_iterator(yield_tokens(dataset), specials=["<pad>", "<unk>"])
vocab.set_default_index(vocab["<unk>"])
