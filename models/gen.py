import os
import torch
from utils.CharTokenize import CharTokenizer

save_dir = "models"
model = torch.load(os.path.join(save_dir, "model.pth"))

tok = CharTokenizer()


def sample(model, query: str, len_gen: int = 500) -> str:
    encoded_input = torch.tensor()