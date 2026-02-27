import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tokenizer import Tokenizer
from model import transformer


# training loop

def train_loop(samples, batchsize, model):

    # wrap an iterable to enable easy access to samples
    data_loader = DataLoader(samples, batch_size=batchsize, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss()

    # if don't need to do a split then use:
    print("<<<< Training Started >>>>")

    model.train()
    loss_train = 0
    for x, y in data_loader:
        optimizer.zero_grad()
        output = model(x)
        if y.ndim > 1:
            y = torch.argmax(y, dim=1)
        y = y.long()

        batch, training_len, vocab = output.shape

        loss = criterion(output.view(batch * training_len, vocab), y.view(batch * training_len))

        loss.backward()
        optimizer.step()

        loss_train += loss.item()

        print(f"Epoch {i +1}: Loss = {loss_train:.4f}")

    print("<<<< Training Complete >>>>")


def save_model(model, tokens, config, tokenizer, path="my_first_transformer.pt"):
    torch.save({
        "model_state": model.state_dict(),
        "config": config,                   
        "vocab": tokenizer.vocab,          
    }, path)
    print(f"Model saved to {path}")


def load_model(path="my_first_transformer.pt"):
    load_in = torch.load(path)
    config = load_in["config"]
    model_loaded = transformer(config)

    tokenizer = Tokenizer(None)
    tokenizer.vocab = load_in["vocab"]
    print(f"Model loaded from {path}")
    return model_loaded, tokenizer, config