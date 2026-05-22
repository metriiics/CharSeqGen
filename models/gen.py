import os
import sys
import torch
from torch.distributions.categorical import Categorical
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.CharTokenize import CharTokenizer

path_token = Path.cwd() / "models/tokenConfig"
tok = CharTokenizer().load(path_token)

def generate(model, query: str, 
           len_gen: int = 500,
           scale_factor: int = 10) -> str:
    encoded_input = torch.tensor(tok.encode(query))
    encoded_input = encoded_input.view(1, -1)

    gen_str = query
    hidden = None

    model.eval()

    for char in range(len(query) - 1):
        _, hidden = model(encoded_input[:, char].view(1, 1), hidden)

    last_char = encoded_input[:, -1].view(1, 1)
    for i in range(len_gen):
        logits, hidden = model(last_char, hidden)
        logits = logits.squeeze(0).squeeze(0)
        scaled_logits = logits * scale_factor
        m = Categorical(logits=scaled_logits)
        last_char = m.sample().view(1, 1)
        gen_str += str(tok.decode(last_char))

    return gen_str