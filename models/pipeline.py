import os
import sys
from pathlib import Path
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.CharTokenize import CharTokenizer
from utils.figure import create_figure_loss

import torch 
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as op

DEVICE = torch.device('cuda')

class SequenceDataset(Dataset):
    def __init__(self, text_chunks):
        self.text_chunks = text_chunks

    def __len__(self):
        return len(self.text_chunks)

    def __getitem__(self, index):
        text_chunk = self.text_chunks[index]
        return text_chunk[:-1].long(), text_chunk[1:].long()
    
class SeqModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, rnn_hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn = nn.LSTM(embed_dim, rnn_hidden_size, batch_first=True)
        self.fc = nn.Linear(rnn_hidden_size, vocab_size)

    def forward(self, x, hidden, cell):
        out = self.embedding(x).unsqueeze(1)
        out, (hidden, cell) = self.rnn(out, (hidden, cell))
        out = self.fc(out).reshape(out.size(0), -1)
        return out, hidden, cell

    def init_hidden(self, batch_size):
        hidden = torch.zeros(1, batch_size, self.rnn_hidden_size)
        cell = torch.zeros(1, batch_size, self.rnn_hidden_size)
        return hidden.to(DEVICE), cell.to(DEVICE)

def train(datLoader, device, 
        epochs, model, 
        loss_fn, optim,
        batch_size, seq_length):
    history_losses = []

    scaler = torch.amp.GradScaler("cuda")

    for epoch in range(epochs):
        hidden, cell = model.init_hidden(batch_size)
        seq_batch, target_batch = next(iter(datLoader))
        seq_batch = seq_batch.to(device)
        target_batch = target_batch.to(device)

        optim.zero_grad()
        loss = 0

        with torch.autocast(device_type='cuda', dtype=torch.float16):
            for char in range(seq_length):
                pred, hidden, cell = model(seq_batch[:, char], hidden, cell)
                loss += loss_fn(pred, target_batch[:, char])

        scaler.scale(loss).backward()
        scaler.step(optimizer=optim)

        scaler.update()

        loss = loss.item() / seq_length
        history_losses.append(loss)
        if epoch % 500 == 0:
            print(f"Epoch {epoch} loss: {round(loss, 3)}")
    create_figure_loss(history=history_losses)
    torch.save(model, os.path.join("models/weight", "model.pth"))

if __name__ == "__main__":
    torch.manual_seed(1)

    path = Path.cwd() / 'pre-book/BookInText/scaled_text.txt'
    path_token = Path.cwd() / "models/tokenConfig"

    with open(path, "r", encoding="utf-8") as file:
        text = file.read()

    tokenizer = CharTokenizer()
    tokenizer.fit(text)

    tokenizer.save(path_token)

    seq_length = 128
    chunk_size = seq_length + 1

    tokenized_text = tokenizer.encode(text)
    text_chunks = [tokenized_text[i: i + chunk_size]
                for i in range(len(tokenized_text) - chunk_size + 1)
    ]

    seq_dataset = SequenceDataset(torch.tensor(text_chunks))

    batch_size = 16

    seq_dl = DataLoader(seq_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    vocab_size = tokenizer.get_vocab_size
    embed_dim = 256
    rnn_hidden_size = 512

    model = SeqModel(vocab_size, embed_dim, rnn_hidden_size)
    model = model.to(DEVICE)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = op.Adam(model.parameters(), lr=0.005)

    epochs = 8000
    train(seq_dl, DEVICE, 
        epochs, model, 
        loss_fn, optimizer, 
        batch_size, seq_length)