import os
import torch
from models.gen import generate

from models.pipeline import SeqModel

import sys
sys.modules['__main__'].SeqModel = SeqModel

model = torch.load(os.path.join("models/weight/", "model.pth"), weights_only=False)
model.to('cpu')

question = "Привет, как твои дела?"

print(generate(model, question))