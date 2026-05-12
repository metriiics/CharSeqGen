import os
import torch
from torch.distributions.categorical import Categorical
from pathlib import Path

from utils.CharTokenize import CharTokenizer

path_token = Path.cwd() / "models/tokenConfig"
tok = CharTokenizer().load(path_token)

def generate(model, query: str, 
           len_gen: int = 500,
           scale_factor: int = 1.0) -> str:
    encoded_input = torch.tensor(tok.encode(query))
    encoded_input = torch.reshape(encoded_input, (1, -1))

    gen_str = query

    model.eval()
    hidden, cell = model.init_hidden(1)
    hidden = hidden.to('cpu') 
    cell = cell.to('cpu')

    for char in range(len(query) - 1):
        _, hidden, cell = model(encoded_input[:, char].view(1))

    last_char = encoded_input[:, -1]
    for i in range(len_gen):
        logits, hidden, cell = model(last_char.view(1), hidden, cell)
        logits = torch.squeeze(logits, 0)
        scaled_logits = logits * scale_factor
        m = Categorical(logits=scaled_logits)
        last_char = m.sample()
        gen_str += str(tok.decode(last_char))

    return gen_str