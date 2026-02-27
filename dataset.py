import torch
from torch.utils.data import Dataset

from tokenizer import Tokenizer


class Book_Dataset(torch.utils.data.Dataset):
    def __init__(self, raw_text, train_len):
        self.tokenizer = Tokenizer(raw_text)
        self.tokens = torch.tensor(self.tokenizer.tokenize(raw_text), dtype=torch.long)
        self.n_context = train_len

    def __len__(self):
        return len(self.tokens) - self.n_context

    def __getitem__(self, index):
        x = self.tokens[index:index + self.n_context]
        y = self.tokens[index + 1:index + self.n_context + 1]
        return x, y
