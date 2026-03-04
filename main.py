import time
import torch
import requests

from config import Config
from model import transformer
from dataset import Book_Dataset
from train import train_loop, save_model


def main():
    print("=" * 60)
    print("Downloading dataset from Project Gutenberg...")
    url = "https://www.gutenberg.org/files/67098/67098-0.txt"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    text = response.text

    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"
    start_idx = text.find(start_marker)
    end_idx = text.find(end_marker)
    content_area = text[start_idx:end_idx].split("\n", 1)[1]
    chapter_idx = content_area.upper().find("CHAPTER I")
    raw_text = content_area[chapter_idx:100_000]
    print(f"Dataset loaded: {len(raw_text):,} characters")

    CONTEXT_LEN = 64
    dataset = Book_Dataset(raw_text, train_len=CONTEXT_LEN)
    tokenizer = dataset.tokenizer  # reuse this everywhere — vocab is fixed

    config = Config(
        d_model=256,
        d_vocab=tokenizer.vocab_size,
        d_hidden=512,
        n_context_max=CONTEXT_LEN,
        n_context=CONTEXT_LEN,
        n_layers=4,
    )

    model = transformer(config)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameter count: {param_count:,}")

    print("\n" + "=" * 60)
    print("UNTRAINED MODEL GENERATION")
    print("=" * 60)
    seed = "the captain said"
    untrained_output = model.generate(seed, tokenizer, max_length=100)
    print(f"Seed : '{seed}'")
    print(f"Output:\n{untrained_output}")

    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)
    t0 = time.time()
    loss_list = train_loop(dataset, batchsize=32, model=model)
    elapsed = time.time() - t0
    print(f"Training time: {elapsed:.1f}s")

    print("\n" + "=" * 60)
    print("TRAINED MODEL GENERATION")
    print("=" * 60)
    trained_output = model.generate(seed, tokenizer, max_length=100)
    print(f"Seed : '{seed}'")
    print(f"Output:\n{trained_output}")

    with open("results.txt", "w") as f:
        f.write(f"Model parameter count: {param_count:,}\n")
        f.write(f"Training time: {elapsed:.1f}s\n")
        f.write(f"Final avg loss: {sum(loss_list) / len(loss_list):.4f}\n\n")
        f.write("UNTRAINED MODEL GENERATION\n")
        f.write(f"Seed: '{seed}'\n")
        f.write(f"Output:\n{untrained_output}\n\n")
        f.write("TRAINED MODEL GENERATION\n")
        f.write(f"Seed: '{seed}'\n")
        f.write(f"Output:\n{trained_output}\n")
    print("\nResults saved to results.txt")
    print("Loss curve saved to loss_curve.png")

    save_model(model, None, config, tokenizer, "trained_model.pt")
    print("Model saved to trained_model.pt")


if __name__ == "__main__":
    main()
