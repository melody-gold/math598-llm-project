"""
Tests for the transformer model components.
Run locally with: pytest tests/test_transformer.py -v
"""

import pytest
import torch
import torch.nn as nn

from config import Config
from model import MLP, AttentionHead, TransformerBlock, transformer as Transformer


# Fixtures

@pytest.fixture
def small_config():
    return Config(d_model=16, d_vocab=50, d_hidden=32, n_layers=2,
                  n_context=8, n_context_max=8)


@pytest.fixture
def model(small_config):
    return Transformer(small_config)

# Config tests


def test_config_fields():
    cfg = Config(d_model=64, d_vocab=100, d_hidden=128, n_layers=4,
                 n_context=16, n_context_max=16)
    assert cfg.d_model == 64
    assert cfg.d_vocab == 100
    assert cfg.d_hidden == 128
    assert cfg.n_layers == 4


# MLP tests


def test_mlp_output_shape(small_config):
    mlp = MLP(small_config)
    x = torch.randn(5, small_config.d_model)
    out = mlp(x)
    assert out.shape == x.shape, "MLP must preserve (seq_len, d_model) shape"


# AttentionHead tests


def test_transformer_block_output_shape(small_config):
    block = TransformerBlock(small_config)
    x = torch.randn(6, small_config.d_model)
    out = block(x)
    assert out.shape == x.shape

# Integration Tests


def test_model_creation_does_not_crash(small_config):
    """Just building the model should not raise."""
    model = Transformer(small_config)
    assert model is not None


def test_forward_pass_output_shape(small_config, model):
    """Forward pass should return (seq_len, d_vocab) logits."""
    seq_len = small_config.n_context
    x = torch.randint(0, small_config.d_vocab, (seq_len,))
    out = model(x)
    assert out.shape == (seq_len, small_config.d_vocab), (
        f"Expected ({seq_len}, {small_config.d_vocab}), got {out.shape}"
    )


def test_forward_pass_no_nan(small_config, model):
    """Logits should not contain NaN or Inf after a forward pass."""
    x = torch.randint(0, small_config.d_vocab, (small_config.n_context,))
    out = model(x)
    assert not torch.isnan(out).any(), "NaN in model output"
    assert not torch.isinf(out).any(), "Inf in model output"


def test_two_training_steps_do_not_crash(small_config, model):
    """Two manual gradient-update steps should run without error."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    seq_len = small_config.n_context
    losses = []

    model.train()
    for _ in range(2):
        x = torch.randint(0, small_config.d_vocab, (seq_len,))
        y = torch.randint(0, small_config.d_vocab, (seq_len,))

        optimizer.zero_grad()
        logits = model(x)                        # (seq_len, d_vocab)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        assert not torch.isnan(loss), "Loss became NaN during training"

    assert len(losses) == 2


def test_generate_returns_longer_sequence(small_config, model):
    """generator() should return a string longer than the seed text."""
    seed = "hello"
    num_new = 5
    # generator needs raw_text to build the tokenizer vocab; we fake a small corpus
    fake_corpus = "abcdefghijklmnopqrstuvwxyz .!?"
    result = model.generator(num_tokens=num_new, input_text=seed, raw_text=fake_corpus)
    assert isinstance(result, str), "generator() should return a string"
    assert len(result) >= len(seed), "generated output should be at least as long as the seed"

# TODO: add these tests once the corresponding code is fixed / implemented
# - test_train_loop_two_epochs: call train_loop(..., epochs=2) from train.py
# - test_save_and_load_model: call save_model() then load_model() and verify that the reloaded model produces identical outputs to the original.

'''
create a config
create model
call train
call generate
check that it didn't creash
how many iterations (2)
have a few models in the notebook to show what the outputs have been
'''

