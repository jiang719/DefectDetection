import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Attention(nn.Module):
    def __init__(self, hidden_dim, dropout):
        super(Attention, self).__init__()

        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.b = nn.Parameter(torch.FloatTensor(hidden_dim)).to(device)
        nn.init.uniform_(self.b.data)
        self.c = nn.Parameter(torch.FloatTensor(hidden_dim)).to(device)
        nn.init.uniform_(self.c.data)
        self.dropout = dropout

    def forward(self, x):
        # [B, L, H]
        x = self.W(x) + self.b
        x = F.tanh(x)

        c = self.c.repeat(x.size(0), 1, 1)  # [B, 1, H]
        a = torch.bmm(c, x.transpose(1, 2))     # [B, 1, H] x [B, H, L] = [B, 1, L]
        a = F.softmax(a, dim=-1)

        output = torch.bmm(a, x)    # [B, 1, L] x [B, L, H] = [B, 1, H]
        output = output.squeeze(1)  # [B, H]
        output = F.dropout(output, self.dropout, training=self.training)
        return output
