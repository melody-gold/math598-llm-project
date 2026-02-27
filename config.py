from dataclasses import dataclass 


@dataclass 
class Config:
    d_model : int 
    d_vocab : int
    d_hidden : int # for MLP
    n_context_max : int # important for training loop (max "slice" size)
    n_context: int 
    n_layers : int
    
    # d_head : int # for Attn (if separate wq and wk)
    #no n_context
    #name var : type