import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):

    def __init__(self, input_size, hidden_size=256, num_layers=2,
                 num_classes=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)        # (B, input_size) → (B, 1, input_size)
        out, _ = self.lstm(x)
        out    = out[:, -1, :]    # (B, hidden_size)
        out    = self.dropout(out)
        return self.fc(out)


def get_model(cfg):
    m = cfg["model"]
    return LSTMClassifier(
        input_size  = m["input_size"],
        hidden_size = m["hidden_size"],
        num_layers  = m["num_layers"],
        num_classes = m["num_classes"],
        dropout     = m["dropout"],
    )
