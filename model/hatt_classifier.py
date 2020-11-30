import torch.nn as nn
import torch.nn.functional as F

from model.sent_encoder import SentEncoder
from model.word_encoder import WordEncoder


class HATTClassifier(nn.Module):
    def __init__(self, dictionary, embed_dim=300, hidden_dim=256, dropout=0.1):
        super(HATTClassifier, self).__init__()

        self.dictionary = dictionary
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.word_encoder = WordEncoder(dictionary, embed_dim, hidden_dim, dropout)
        self.tag_embedding = nn.Embedding(3, hidden_dim, padding_idx=0)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.sent_encoder = SentEncoder(hidden_dim, dropout)
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, inputs, tags):
        # [B, S, W], [B, S]
        bsz, sent_num = inputs.size(0), inputs.size(1)
        x = inputs.view(bsz * sent_num, -1)    # [B*S, W]
        x = self.word_encoder(x)    # [B*S, H]
        x = x.view(bsz, sent_num, -1)   # [B, S, H]
        x += self.tag_embedding(tags)
        x = self.layer_norm(x)
        x = self.sent_encoder(x)        # [B, H]

        x = self.fc1(x)
        x = F.tanh(x)
        x = F.dropout(x, self.dropout, training=self.training)

        x = self.fc2(x)
        output = F.log_softmax(x, dim=-1)
        return output
