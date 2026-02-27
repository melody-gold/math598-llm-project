"""
Tests for the transformer model components.
Run locally with: pytest tests/test_transformer.py -v
"""

import pytest
import torch
import torch.nn as nn
from dataclasses import dataclass

# Minimal re-definitions so tests are self-contained.


@dataclass
class Config:
    d_model: int
    d_vocab: int
    d_hidden: int
    n_layers: int


class MLP(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_hidden)
        self.fc2 = nn.Linear(config.d_hidden, config.d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class AttentionHead(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.W_q = nn.Linear(config.d_model, config.d_model, bias=False)
        self.W_k = nn.Linear(config.d_model, config.d_model, bias=False)
        self.W_v = nn.Linear(config.d_model, config.d_model, bias=False)
        self.scale = config.d_model ** -0.5

    def forward(self, x):
        Q, K, V = self.W_q(x), self.W_k(x), self.W_v(x)
        scores = (Q @ K.transpose(-2, -1)) * self.scale
        attn = torch.softmax(scores, dim=-1)
        return attn @ V


class TransformerBlock(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.mlp = MLP(config)
        self.attention = AttentionHead(config)

    def forward(self, x):
        return x + self.mlp(x) + self.attention(x)


class Transformer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.d_vocab, config.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])

    def forward(self, x):
        x = self.token_embedding(x)
        for block in self.blocks:
            x = block(x)
        return x @ self.token_embedding.weight.T

# Fixtures


@pytest.fixture
def small_config():
    return Config(d_model=16, d_vocab=50, d_hidden=32, n_layers=2)


@pytest.fixture
def model(small_config):
    return Transformer(small_config)


# Config tests

def test_config_fields():
    cfg = Config(d_model=64, d_vocab=100, d_hidden=128, n_layers=4)
    assert cfg.d_model == 64
    assert cfg.d_vocab == 100
    assert cfg.d_hidden == 128
    assert cfg.n_layers == 4


# MLP tests

def test_mlp_output_shape(small_config):
    mlp = MLP(small_config)
    x = torch.randn(5, small_config.d_model)       # (seq_len, d_model)
    out = mlp(x)
    assert out.shape == x.shape, "MLP must preserve (seq_len, d_model) shape"


# AttentionHead tests

def test_attention_output_shape(small_config):
    attn = AttentionHead(small_config)
    seq_len = 8
    x = torch.randn(seq_len, small_config.d_model)
    out = attn(x)
    assert out.shape == x.shape, "Attention must preserve (seq_len, d_model) shape"


# TransformerBlock tests

def test_transformer_block_output_shape(small_config):
    block = TransformerBlock(small_config)
    x = torch.randn(6, small_config.d_model)
    out = block(x)
    assert out.shape == x.shape


'''
create a config
create model
call train
call generate
check that it didn't creash
how many iterations (2)
have a few models in the notebook to show what the outputs have been
'''

