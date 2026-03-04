import torch
import torch.nn as nn
from jaxtyping import Float, Int

from config import Config
from tokenizer import Tokenizer


class MLP(nn.Module):
    def __init__(self, config: Config):  # matrices to initialize
        super().__init__()
        self.linear_up: nn.Linear = nn.Linear(config.d_model, config.d_hidden)
        self.linear_down: nn.Linear = nn.Linear(config.d_hidden, config.d_model)

    def forward(self, x: Float[torch.Tensor, "* d_model"]) -> Float[torch.Tensor, "* d_model"]:
        x = self.linear_up(x)
        x = torch.relu(x)
        x = self.linear_down(x)
        return x


class AttentionHead(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        # weights (use nn.parameter) to create a matrix to track gradients
        self.wqk = nn.Parameter(torch.randn(config.d_model, config.d_model) * 0.02)
        self.wov = nn.Parameter(torch.randn(config.d_model, config.d_model) * 0.02)

        # Create M Matrix
    def M_matrix(self, n, device=None):
        # matrix with 0 at and below the diagonal and -inf above the diagonal
        M = torch.ones((n, n), device=device)
        M = torch.triu(M, diagonal=1)
        M = M.masked_fill(M == 1, float('-inf'))
        # print(M)
        return M

    def forward(self, x: Float[torch.Tensor, "* seq d_model"]) -> Float[torch.Tensor, "* seq d_model"]:
        # x can be (seq, d_model) or (batch, seq, d_model)
        is_batched = x.dim() == 3

        if not is_batched:
            x = x.unsqueeze(0)  # -> (1, seq, d_model)

        batch, n_seq, d_model = x.shape
        M = self.M_matrix(n_seq, x.device)  # (seq, seq)

        # attention pattern: (batch, seq, seq)
        # x @ wqk -> (batch, seq, d_model), then @ x.transpose -> (batch, seq, seq)
        attn = (x @ self.wqk) @ x.transpose(-2, -1) + M
        attn = torch.softmax(attn, dim=-1)

        # (batch, seq, seq) @ (batch, seq, d_model) -> (batch, seq, d_model)
        out = attn @ x @ self.wov

        if not is_batched:
            out = out.squeeze(0)  # -> (seq, d_model)

        return out


class TransformerBlock(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        # self.ln = nn.LayerNorm(config)
        self.mlp = MLP(config)
        self.attention = AttentionHead(config)

    def forward(self, x):
        x = x + self.attention(x)
        x = x + self.mlp(x)
        return x


class transformer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.d_vocab, config.d_model)
        self.pos_embedding = nn.Embedding(config.n_context_max, config.d_model)
        # self.transformerblocks = nn.modules list of transformer blocks
        self.transformerblocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.pos_embedding.weight, std=0.02)

    def forward(self, x: Int[torch.Tensor, "..."]) -> Float[torch.Tensor, "... d_vocab"]:
        # x: (seq,) or (batch, seq)
        is_batched = x.dim() == 2
        seq_len = x.shape[-1]

        x = self.token_embedding(x)  # (..., seq, d_model)
        positions = torch.arange(seq_len, device=x.device)
        x = x + self.pos_embedding(positions)  # broadcasts over batch

        for block in self.transformerblocks:
            x = block(x)

        x = x @ self.token_embedding.weight.T  # (..., seq, d_vocab)
        return x

    def generate(self, text: str, tokenizer, max_length: int) -> str: 
        self.eval()
        with torch.no_grad():
            tokenized = tokenizer.tokenize(text)
            if not tokenized:
                tokenized = [0]  # fallback if seed has no known chars
            input_tensor = torch.tensor(tokenized, dtype=torch.long)
            for _ in range(max_length):
                ctx = input_tensor[-self.config.n_context_max:]
                logits = self(ctx)
                probs = torch.softmax(logits[-1], dim=-1)
                new_token = torch.multinomial(probs, num_samples=1)
                input_tensor = torch.cat([input_tensor, new_token], dim=-1)
        return tokenizer.detokenize(input_tensor.tolist())

# use nn.ModuleList for TB seqeunce & MHA (to create a list of TBS)
# print(f"{x.shape = }") for debugging

# pick a unique dataset to train data on

# if traning models: aim for < 10 million parameters for now
#   sum(x.numel() for x in mymodel.parameters())
