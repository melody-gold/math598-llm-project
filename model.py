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
        self.wqk = nn.Parameter(torch.randn(config.d_model, config.d_model))
        self.wov = nn.Parameter(torch.randn(config.d_model, config.d_model))

        # Create M Matrix
    def M_matrix(self, n):
        # matrix with 0 at and below the diagonal and -inf above the diagonal
        M = torch.ones((n, n))
        M = torch.triu(M, diagonal=1)
        M = M.masked_fill(M == 1, float('-inf'))
        print(M)
        return M

    def forward(self, x: Float[torch.Tensor, "* d_model"]) -> Float[torch.Tensor, "* d_model"]:
        # use weights to compute A⨉
        # X as input: n_seq by d_model
        n_seq = x.shape[0]
        M = self.M_matrix(n_seq)
        attention_pattern = x @ self.wqk @ x.T + M
        attention_of_X = torch.softmax(attention_pattern, dim=-1) @ x @ self.wov

        return attention_of_X


class TransformerBlock(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        # self.ln = nn.LayerNorm(config)
        self.mlp = MLP(config)
        self.attention = AttentionHead(config)

    def forward(self, x: Float[torch.Tensor, "* d_model"]) -> Float[torch.Tensor, "* d_model"]:
        # output = x + mlp(x) + attentionhead(x)
        output_x = x + self.mlp(x) + self.attention(x)
        # x = self.ln(x_1)
        # x = self.ln(x_2)

        return output_x


class transformer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.d_vocab, config.d_model)
        self.pos_embedding = nn.Embedding(config.n_context_max, config.d_model)
        # self.transformerblocks = nn.modules list of transformer blocks
        self.transformerblocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])

    def forward(self, x: Int[torch.Tensor, "n_context"]) -> Float[torch.Tensor, "n_context d_vocab"]:
        x = self.token_embedding(x)  # converts int d-vector to d-model vector
        x = x + self.pos_embedding(torch.arange(x.shape[0])) # x = E + P
        # pos_embedding(x) uses nn.Embedding of torch.arrange(n_context)
        for i in range(self.config.n_layers):
            x = self.transformerblocks[i](x)
        x = x @ self.token_embedding.weight.T  # unembedding 
        # n_contex long - sequence if ints of length n  - float tneosry by n_model  and output is float tencsosr by d-vocab \n",
        # d_model to d_vocab transpose or do a lineear map  - unembed nn.linear
        # dmodel to dvocab

        return x

    def generator(self, num_tokens=10, input_text="", raw_text=""): 
        # some text, number of new token, and return esseuquence of text - tokenzise text, sequence of numbers, numbers in model and get probaility, sample probablities, detonize 
        tokenizer = Tokenizer(raw_text)
        tokenized_text = tokenizer.tokenize(input_text)
        input_tensor = torch.tensor(tokenized_text, dtype=torch.long)
        for i in range(num_tokens):
            input_tensor = input_tensor[-self.config.n_context_max:]
            out = self.forward(input_tensor)
            print("Finished running through forward!")
            probailities = torch.softmax(out[:, -1], dim=-1)
            new_token = torch.multinomial(probailities, num_samples=1)
            new_input_tensor = torch.cat([input_tensor, new_token], dim=-1)
            input_tensor = new_input_tensor
        detokenized_text = tokenizer.detokenize(input_tensor.tolist())

        return detokenized_text

# use nn.ModuleList for TB seqeunce & MHA (to create a list of TBS)
# print(f"{x.shape = }") for debugging

# pick a unique dataset to train data on

# if traning models: aim for < 10 million parameters for now
#   sum(x.numel() for x in mymodel.parameters())
