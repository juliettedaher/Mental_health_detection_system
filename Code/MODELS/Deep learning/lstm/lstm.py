import torch
import torch.nn as nn


def _init_embedding(embedding):
    nn.init.xavier_uniform_(embedding.weight.data)
    if embedding.padding_idx is not None:
        embedding.weight.data[embedding.padding_idx].zero_()


def _get_last_real_timestep(out, lengths):
    """
    Instead of out[:, -1, :] which reads the last PADDING token,
    this reads the last REAL token for each sequence in the batch.

    out     : (B, seq_len, hidden)
    lengths : (B,) — number of real tokens per sequence (before padding)
    """
    batch_size = out.size(0)
    # clamp so index never goes out of bounds
    idx = (lengths - 1).clamp(min=0)
    # idx shape: (B,) → expand to (B, 1, hidden) → squeeze → (B, hidden)
    idx = idx.unsqueeze(1).unsqueeze(2).expand(batch_size, 1, out.size(2))
    return out.gather(1, idx).squeeze(1)


class LSTMClassifier(nn.Module):

    def __init__(self, vocab_size, embed_dim=128, hidden_size=256,
                 num_layers=2, num_classes=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        _init_embedding(self.embedding)
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (B, seq_len) — token ids, 0 = PAD
        lengths = (x != 0).sum(dim=1).clamp(min=1)   # real token count per row
        emb = self.embedding(x)                        # (B, seq_len, embed_dim)
        out, _ = self.lstm(emb)                        # (B, seq_len, hidden)
        out = _get_last_real_timestep(out, lengths)    # (B, hidden)  ← THE FIX
        out = self.dropout(out)
        return self.fc(out)


class BiLSTMClassifier(nn.Module):

    def __init__(self, vocab_size, embed_dim=128, hidden_size=256,
                 num_layers=2, num_classes=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        _init_embedding(self.embedding)
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True,
                            bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        lengths = (x != 0).sum(dim=1).clamp(min=1)
        emb = self.embedding(x)
        out, _ = self.lstm(emb)
        out = _get_last_real_timestep(out, lengths)
        out = self.dropout(out)
        return self.fc(out)

def get_model(cfg):
    m    = cfg["model"]
    arch = m["architecture"].lower()
    common = dict(
        vocab_size=m["vocab_size"],
        embed_dim=m["embed_dim"],
        hidden_size=m["hidden_size"],
        num_layers=m["num_layers"],
        num_classes=m["num_classes"],
        dropout=m["dropout"],
    )
    if arch == "lstm":
        return LSTMClassifier(**common)
    elif arch == "bilstm":
        return BiLSTMClassifier(**common)
    else:
        raise ValueError("Architecture must be: lstm | bilstm ")