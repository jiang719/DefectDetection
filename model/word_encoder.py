import torch.nn as nn

from model.attention import Attention


class WordEncoder(nn.Module):
    def __init__(self, dictionary, embed_dim=256, hidden_dim=256, dropout=0.1):
        super(WordEncoder, self).__init__()

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.embedding = nn.Embedding(len(dictionary), embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, int(hidden_dim / 2), batch_first=True, bidirectional=True)
        self.attention = Attention(hidden_dim, dropout)

    def forward(self, inputs):
        x = self.embedding(inputs)
        x, h = self.gru(x)
        output = self.attention(x)
        return output
