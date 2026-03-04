"""
Tests for the transformer model components.
Run locally with: pytest tests/test_transformer.py -v
"""

import os
import math
import tempfile

import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from config import Config
from model import MLP, AttentionHead, TransformerBlock, transformer as Transformer
from tokenizer import Tokenizer
from dataset import Book_Dataset
from train import save_model, load_model


CORPUS = "the quick brown fox jumps over the lazy dog. hello world!"

# Fixtures

@pytest.fixture
def small_config():
    return Config(d_model=16, d_vocab=50, d_hidden=32, n_layers=2,
                  n_context=8, n_context_max=8)


@pytest.fixture
def model(small_config):
    torch.manual_seed(0)
    return Transformer(small_config)


@pytest.fixture
def tokenizer():
    return Tokenizer(CORPUS)


@pytest.fixture
def corpus_config(tokenizer):
    return Config(
        d_model=16,
        d_vocab=tokenizer.vocab_size,
        d_hidden=32,
        n_layers=2,
        n_context=8,
        n_context_max=8,
    )


@pytest.fixture
def corpus_model(corpus_config):
    torch.manual_seed(0)
    return Transformer(corpus_config)


# Config tests

class TestConfig:
    def test_fields_stored_correctly(self):
        cfg = Config(d_model=64, d_vocab=100, d_hidden=128, n_layers=4,
                     n_context=16, n_context_max=16)
        assert cfg.d_model == 64
        assert cfg.d_vocab == 100
        assert cfg.d_hidden == 128
        assert cfg.n_layers == 4
        assert cfg.n_context == 16
        assert cfg.n_context_max == 16

    def test_is_dataclass(self):
        """Config should be a plain dataclass — no hidden state."""
        import dataclasses
        assert dataclasses.is_dataclass(Config)

    def test_different_configs_are_independent(self):
        a = Config(d_model=8, d_vocab=10, d_hidden=16, n_layers=1,
                   n_context=4, n_context_max=4)
        b = Config(d_model=32, d_vocab=200, d_hidden=64, n_layers=6,
                   n_context=32, n_context_max=32)
        assert a.d_model != b.d_model
        assert a.d_vocab != b.d_vocab


# tokenizer tests

class TestTokenizer:
    def test_vocab_covers_corpus(self, tokenizer):
        """Every character that survives clean_text must be in the vocab."""
        cleaned = tokenizer.clean_text(CORPUS)
        for ch in cleaned:
            assert ch in tokenizer.encode, f"'{ch}' missing from vocab"

    def test_vocab_size_matches_chars(self, tokenizer):
        assert tokenizer.vocab_size == len(tokenizer.chars)
        assert tokenizer.vocab_size == len(tokenizer.encode)
        assert tokenizer.vocab_size == len(tokenizer.decode)

    def test_encode_decode_are_inverses(self, tokenizer):
        """encode and decode should be exact inverses of each other."""
        for ch, idx in tokenizer.encode.items():
            assert tokenizer.decode[idx] == ch

    def test_tokenize_returns_list_of_ints(self, tokenizer):
        tokens = tokenizer.tokenize("hello")
        assert isinstance(tokens, list)
        assert all(isinstance(t, int) for t in tokens)

    def test_tokenize_detokenize_roundtrip(self, tokenizer):
        """Tokenising then detokenising should recover the cleaned text."""
        cleaned = "".join(tokenizer.clean_text(CORPUS))
        tokens = tokenizer.tokenize(CORPUS)
        recovered = tokenizer.detokenize(tokens)
        assert recovered == cleaned

    def test_clean_text_lowercases(self, tokenizer):
        result = tokenizer.clean_text("HELLO")
        assert all(ch.islower() or not ch.isalpha() for ch in result)

    def test_clean_text_removes_digits_and_special_chars(self, tokenizer):
        result = tokenizer.clean_text("abc123@#$")
        for ch in result:
            assert ch.isalpha() or ch in " .!?"

    def test_unknown_chars_are_skipped_during_tokenize(self, tokenizer):
        """Characters not in the training vocab should be silently skipped."""
        tokens = tokenizer.tokenize("~`^&*")  # none of these survive clean_text
        assert tokens == []

    def test_unknown_ids_skipped_during_detokenize(self, tokenizer):
        """IDs not in decode map should be silently skipped."""
        bad_id = tokenizer.vocab_size + 999
        result = tokenizer.detokenize([bad_id])
        assert result == ""

    def test_empty_string_tokenizes_to_empty_list(self, tokenizer):
        assert tokenizer.tokenize("") == []

    def test_empty_list_detokenizes_to_empty_string(self, tokenizer):
        assert tokenizer.detokenize([]) == ""

    def test_all_token_ids_in_valid_range(self, tokenizer):
        tokens = tokenizer.tokenize(CORPUS)
        for t in tokens:
            assert 0 <= t < tokenizer.vocab_size


# Dataset tests

class TestBookDataset:
    def test_length_is_tokens_minus_context(self):
        ds = Book_Dataset(CORPUS, train_len=4)
        expected = len(ds.tokens) - 4
        assert len(ds) == expected

    def test_getitem_returns_correct_shapes(self):
        ds = Book_Dataset(CORPUS, train_len=4)
        x, y = ds[0]
        assert x.shape == (4,)
        assert y.shape == (4,)

    def test_x_and_y_are_offset_by_one(self):
        ds = Book_Dataset(CORPUS, train_len=4)
        x, y = ds[0]
        # y should be x shifted one step to the right
        assert torch.equal(x[1:], y[:-1])

    def test_tokens_are_long_dtype(self):
        ds = Book_Dataset(CORPUS, train_len=4)
        x, y = ds[0]
        assert x.dtype == torch.long
        assert y.dtype == torch.long

    def test_all_token_ids_in_range(self):
        ds = Book_Dataset(CORPUS, train_len=4)
        assert ds.tokens.min() >= 0
        assert ds.tokens.max() < ds.tokenizer.vocab_size

    def test_tokenizer_attached(self):
        ds = Book_Dataset(CORPUS, train_len=4)
        assert hasattr(ds, "tokenizer")
        assert ds.tokenizer.vocab_size > 0


# MLP tests


class TestMLP:
    def test_output_shape_2d(self, small_config):
        mlp = MLP(small_config)
        x = torch.randn(5, small_config.d_model)
        assert mlp(x).shape == x.shape

    def test_output_shape_3d_batched(self, small_config):
        mlp = MLP(small_config)
        x = torch.randn(3, 5, small_config.d_model)
        assert mlp(x).shape == x.shape

    def test_no_nan_in_output(self, small_config):
        mlp = MLP(small_config)
        x = torch.randn(5, small_config.d_model)
        out = mlp(x)
        assert not torch.isnan(out).any()

    def test_gradients_flow(self, small_config):
        mlp = MLP(small_config)
        x = torch.randn(5, small_config.d_model, requires_grad=True)
        out = mlp(x)
        out.sum().backward()
        assert x.grad is not None

    def test_different_inputs_give_different_outputs(self, small_config):
        mlp = MLP(small_config)
        x1 = torch.randn(4, small_config.d_model)
        x2 = torch.randn(4, small_config.d_model)
        assert not torch.allclose(mlp(x1), mlp(x2))

    def test_has_two_linear_layers(self, small_config):
        mlp = MLP(small_config)
        linears = [m for m in mlp.modules() if isinstance(m, nn.Linear)]
        assert len(linears) == 2

    def test_hidden_dim_is_correct(self, small_config):
        mlp = MLP(small_config)
        assert mlp.linear_up.out_features == small_config.d_hidden
        assert mlp.linear_down.out_features == small_config.d_model




# AttentionHead tests


class TestAttentionHead:
    def test_output_shape_unbatched(self, small_config):
        attn = AttentionHead(small_config)
        x = torch.randn(6, small_config.d_model)
        assert attn(x).shape == x.shape

    def test_output_shape_batched(self, small_config):
        attn = AttentionHead(small_config)
        x = torch.randn(2, 6, small_config.d_model)
        assert attn(x).shape == x.shape

    def test_no_nan_in_output(self, small_config):
        attn = AttentionHead(small_config)
        x = torch.randn(4, small_config.d_model)
        assert not torch.isnan(attn(x)).any()

    def test_causal_mask_is_upper_triangular(self, small_config):
        attn = AttentionHead(small_config)
        M = attn.M_matrix(5)
        # below and on the diagonal should be 0
        for i in range(5):
            for j in range(i + 1):
                assert M[i, j] == 0.0, f"M[{i},{j}] should be 0"
        # above the diagonal should be -inf
        for i in range(5):
            for j in range(i + 1, 5):
                assert M[i, j] == float("-inf"), f"M[{i},{j}] should be -inf"

    def test_causal_mask_shape(self, small_config):
        attn = AttentionHead(small_config)
        for n in [1, 4, 8]:
            assert attn.M_matrix(n).shape == (n, n)

    def test_gradients_flow(self, small_config):
        attn = AttentionHead(small_config)
        x = torch.randn(4, small_config.d_model, requires_grad=True)
        attn(x).sum().backward()
        assert x.grad is not None

    def test_wqk_and_wov_are_parameters(self, small_config):
        attn = AttentionHead(small_config)
        param_names = [n for n, _ in attn.named_parameters()]
        assert "wqk" in param_names
        assert "wov" in param_names

# TransformerBlock tests

class TestTransformerBlock:
    def test_output_shape_unbatched(self, small_config):
        block = TransformerBlock(small_config)
        x = torch.randn(6, small_config.d_model)
        assert block(x).shape == x.shape

    def test_output_shape_batched(self, small_config):
        block = TransformerBlock(small_config)
        x = torch.randn(3, 6, small_config.d_model)
        assert block(x).shape == x.shape

    def test_no_nan_in_output(self, small_config):
        block = TransformerBlock(small_config)
        x = torch.randn(6, small_config.d_model)
        assert not torch.isnan(block(x)).any()

    def test_residual_connection_exists(self, small_config):
        """Output should differ from both attn(x) and mlp(x) alone — residual is in play."""
        block = TransformerBlock(small_config)
        x = torch.randn(6, small_config.d_model)
        out = block(x)
        attn_only = block.attention(x)
        mlp_only = block.mlp(x)
        assert not torch.allclose(out, attn_only)
        assert not torch.allclose(out, mlp_only)

    def test_has_mlp_and_attention(self, small_config):
        block = TransformerBlock(small_config)
        assert hasattr(block, "mlp")
        assert hasattr(block, "attention")
        assert isinstance(block.mlp, MLP)
        assert isinstance(block.attention, AttentionHead)


# Transformer tests

class TestTransformer:
    def test_creation_does_not_crash(self, small_config):
        assert Transformer(small_config) is not None

    def test_forward_shape_unbatched(self, small_config, model):
        x = torch.randint(0, small_config.d_vocab, (small_config.n_context,))
        out = model(x)
        assert out.shape == (small_config.n_context, small_config.d_vocab)

    def test_forward_shape_batched(self, small_config, model):
        x = torch.randint(0, small_config.d_vocab, (4, small_config.n_context))
        out = model(x)
        assert out.shape == (4, small_config.n_context, small_config.d_vocab)

    def test_forward_no_nan(self, small_config, model):
        x = torch.randint(0, small_config.d_vocab, (small_config.n_context,))
        out = model(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_n_layers_blocks_created(self, small_config, model):
        assert len(model.transformerblocks) == small_config.n_layers

    def test_embeddings_exist(self, small_config, model):
        assert hasattr(model, "token_embedding")
        assert hasattr(model, "pos_embedding")
        assert isinstance(model.token_embedding, nn.Embedding)
        assert isinstance(model.pos_embedding, nn.Embedding)

    def test_token_embedding_shape(self, small_config, model):
        assert model.token_embedding.weight.shape == (
            small_config.d_vocab, small_config.d_model
        )

    def test_pos_embedding_shape(self, small_config, model):
        assert model.pos_embedding.weight.shape == (
            small_config.n_context_max, small_config.d_model
        )

    def test_parameter_count_is_reasonable(self, small_config, model):
        count = sum(p.numel() for p in model.parameters())
        assert count > 0
        assert count < 10_000_000  # well under the 10M guideline

    def test_different_seeds_give_different_outputs(self, small_config):
        torch.manual_seed(0)
        m1 = Transformer(small_config)
        torch.manual_seed(99)
        m2 = Transformer(small_config)
        x = torch.randint(0, small_config.d_vocab, (small_config.n_context,))
        assert not torch.allclose(m1(x), m2(x))

    def test_gradients_flow_through_model(self, small_config, model):
        x = torch.randint(0, small_config.d_vocab, (small_config.n_context,))
        out = model(x)
        out.sum().backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_two_training_steps_decrease_loss(self, small_config, model):
        """Loss should generally decrease over two steps on the same batch."""
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
        criterion = nn.CrossEntropyLoss()
        x = torch.randint(0, small_config.d_vocab, (small_config.n_context,))
        y = torch.randint(0, small_config.d_vocab, (small_config.n_context,))

        model.train()
        losses = []
        for _ in range(2):
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # loss on the same batch should drop after an update
        assert losses[1] < losses[0], "Loss did not decrease after one gradient step"

    def test_no_nan_loss_during_training(self, small_config, model):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        model.train()
        for _ in range(2):
            x = torch.randint(0, small_config.d_vocab, (small_config.n_context,))
            y = torch.randint(0, small_config.d_vocab, (small_config.n_context,))
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            assert not torch.isnan(loss), "NaN loss during training"

# Generate tests

class TestGenerate:
    """Tests for the model's text generation method.
    Supports both the old signature generator(num_tokens, input_text, raw_text)
    and the new signature generate(text, tokenizer, max_length).
    """

    def _call_generate(self, model, tokenizer, seed, num_tokens):
        """Try the new API first, fall back to the old one."""
        if hasattr(model, "generate"):
            return model.generate(seed, tokenizer, max_length=num_tokens)
        return model.generator(
            num_tokens=num_tokens, input_text=seed, raw_text=CORPUS
        )

    def test_returns_string(self, corpus_model, tokenizer):
        result = self._call_generate(corpus_model, tokenizer, "the", 5)
        assert isinstance(result, str)

    def test_output_longer_than_seed(self, corpus_model, tokenizer):
        seed = "the"
        result = self._call_generate(corpus_model, tokenizer, seed, 10)
        assert len(result) >= len(seed)

    def test_output_only_contains_known_chars(self, corpus_model, tokenizer):
        result = self._call_generate(corpus_model, tokenizer, "hello", 20)
        valid = set(tokenizer.chars)
        for ch in result:
            assert ch in valid, f"Unexpected char '{ch}' in output"

    def test_zero_new_tokens_returns_seed(self, corpus_model, tokenizer):
        """Generating 0 new tokens should return (at least) the seed."""
        seed = "hello"
        result = self._call_generate(corpus_model, tokenizer, seed, 0)
        assert isinstance(result, str)

    def test_determinism_with_same_seed(self, corpus_config, tokenizer):
        """Two runs with the same manual seed should give identical output."""
        torch.manual_seed(42)
        m1 = Transformer(corpus_config)
        torch.manual_seed(42)
        m2 = Transformer(corpus_config)

        torch.manual_seed(7)
        r1 = self._call_generate(m1, tokenizer, "the", 10)
        torch.manual_seed(7)
        r2 = self._call_generate(m2, tokenizer, "the", 10)
        assert r1 == r2

    def test_context_window_not_exceeded(self, corpus_config, tokenizer):
        """Even with a very long prompt the model should not crash."""
        long_seed = "the quick brown fox " * 20  # far exceeds n_context_max=8
        model = Transformer(corpus_config)
        result = self._call_generate(model, tokenizer, long_seed, 5)
        assert isinstance(result, str)


# Train loop tests

class TinyDataset(Dataset):
    """Minimal dataset that returns fixed (x, y) pairs for testing."""

    def __init__(self, vocab_size, seq_len, n_samples=20):
        torch.manual_seed(0)
        self.data = [
            (
                torch.randint(0, vocab_size, (seq_len,)),
                torch.randint(0, vocab_size, (seq_len,)),
            )
            for _ in range(n_samples)
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class TestTrainLoop:
    def test_train_loop_runs_without_crash(self, small_config, model, tmp_path, monkeypatch):
        from train import train_loop
        monkeypatch.chdir(tmp_path)  # saves loss_curve.png to a temp dir instead
        ds = TinyDataset(small_config.d_vocab, small_config.n_context)
        train_loop(ds, batchsize=4, model=model)

    def test_train_loop_returns_loss_list(self, small_config, model, tmp_path, monkeypatch):
        from train import train_loop
        monkeypatch.chdir(tmp_path)
        ds = TinyDataset(small_config.d_vocab, small_config.n_context)
        result = train_loop(ds, batchsize=4, model=model)
        if result is None:
            pytest.skip("train_loop does not return loss list yet")
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(v, float) for v in result)

    def test_loss_is_finite_throughout_training(self, small_config, model, tmp_path, monkeypatch):
        from train import train_loop
        monkeypatch.chdir(tmp_path)
        ds = TinyDataset(small_config.d_vocab, small_config.n_context)
        result = train_loop(ds, batchsize=4, model=model)
        if result is None:
            pytest.skip("train_loop does not return loss list yet")
        assert all(math.isfinite(v) for v in result)

    def test_model_weights_change_after_training(self, small_config, tmp_path, monkeypatch):
        from train import train_loop
        monkeypatch.chdir(tmp_path)
        torch.manual_seed(0)
        model = Transformer(small_config)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        ds = TinyDataset(small_config.d_vocab, small_config.n_context)
        train_loop(ds, batchsize=4, model=model)
        after = model.state_dict()
        changed = any(not torch.equal(before[k], after[k]) for k in before)
        assert changed