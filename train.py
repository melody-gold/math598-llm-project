import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

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
    loss_train = 0.0
    num_batches = len(data_loader)
    loss_list = []
    for batch_num, (x, y) in enumerate(data_loader, start=1):
        optimizer.zero_grad()
        output = model(x)
        y = y.long()

        batch, training_len, vocab = output.shape
        loss = criterion(output.reshape(batch * training_len, vocab), y.reshape(batch * training_len))

        loss.backward()
        optimizer.step()
        loss_train += loss.item()
        loss_list.append(loss.item())
        
        if batch_num == 1 or batch_num % 100 == 0 or batch_num == num_batches:
            print(f"Batch {batch_num}/{num_batches}: Loss = {loss.item():.4f}, Running Avg = {loss_train/batch_num:.4f}")

    print("<<<< Training Complete >>>>")
    print(f"Final avg loss: {loss_train/len(data_loader):.4f}")

    plt.plot(loss_list, label="Training Loss")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.show()


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