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
    def __init__(self, vocab_size, embed_dim, rnn_hidden_size, layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn = nn.LSTM(embed_dim, rnn_hidden_size, num_layers=layers, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(rnn_hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        out = self.embedding(x)
        out, hidden = self.rnn(out, hidden)
        out = self.dropout(out)
        out = self.fc(out)
        return out, hidden

def train(datLoader, device, 
        epochs, model, 
        loss_fn, optim):
    history_losses = []

    model.train()

    scaler = torch.amp.GradScaler("cuda")

    for epoch in range(epochs):
        epoch_loss = 0
        for seq_batch, target_batch in datLoader:
            seq_batch = seq_batch.to(device)
            target_batch = target_batch.to(device)

            optim.zero_grad()

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                pred, _ = model(seq_batch)
                loss = loss_fn(pred.reshape(-1, pred.size(-1)), 
                    target_batch.reshape(-1))

            scaler.scale(loss).backward()
            scaler.step(optimizer=optim)

            scaler.update()

            epoch_loss += loss.item()
            
        epoch_loss /= len(datLoader)
        history_losses.append(epoch_loss)
        print(f"Epoch {epoch} loss: {round(epoch_loss, 3)}")
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

    seq_length = 64
    chunk_size = seq_length + 1

    tokenized_text = tokenizer.encode(text)
    text_chunks = [tokenized_text[i: i + chunk_size]
                for i in range(len(tokenized_text) - chunk_size + 1)
    ]
    text_chunks = text_chunks[2312890:3531240]

    seq_dataset = SequenceDataset(torch.tensor(text_chunks))

    batch_size = 128

    seq_dl = DataLoader(seq_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    vocab_size = tokenizer.get_vocab_size
    embed_dim = 256
    rnn_hidden_size = 512
    layers = 1
    drop = 0.5

    model = SeqModel(vocab_size, embed_dim, rnn_hidden_size, layers)
    model = model.to(DEVICE)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = op.Adam(model.parameters(), lr=0.001)

    epochs = 7
    train(seq_dl, DEVICE, 
        epochs, model, 
        loss_fn, optimizer)