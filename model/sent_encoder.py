import torch.nn as nn
import torch.nn.functional as F

from model.attention import Attention


class SentEncoder(nn.Module):
    def __init__(self, hidden_dim=256, dropout=0.1):
        super(SentEncoder, self).__init__()

        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, int(hidden_dim / 2), batch_first=True, bidirectional=True)
        self.attention = Attention(hidden_dim, dropout)

    def forward(self, inputs):
        x = self.W(inputs)
        x, h = self.gru(x)
        output = self.attention(x)
        return output
