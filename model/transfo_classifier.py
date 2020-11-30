import torch.nn as nn
import torch.nn.functional as F
from transformers import TransfoXLModel, TransfoXLConfig, AdaptiveEmbedding


class TransfoClassifier(nn.Module):
    def __init__(self, dictionary, embed_dim=256, hidden_dim=256, head_num=4, layer_num=3, inner_dim=256,
                 max_length=512, dropout=0.1):
        super(TransfoClassifier, self).__init__()

        config = TransfoXLConfig(
            vocab_size=len(dictionary),
            div_val=1,
            d_embed=embed_dim,
            d_model=hidden_dim,
            d_head=int(hidden_dim / head_num),
            n_head=head_num,
            n_layer=layer_num,
            d_inner=inner_dim,
            mem_len=max_length
        )
        self.dropout = dropout

        self.word_embedding = self.word_emb = AdaptiveEmbedding(
            config.vocab_size, config.d_embed, config.d_model, config.cutoffs, div_val=config.div_val
        )
        self.tag_embedding = nn.Embedding(3, hidden_dim, padding_idx=0)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        self.transfo = TransfoXLModel(config)
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, inputs, tags):
        # [B, L]
        x = self.word_embedding(inputs) + self.tag_embedding(tags)
        x = self.layer_norm(x)
        x = self.transfo(input_ids=None, inputs_embeds=x, return_dict=True)
        x = x.last_hidden_state     # [B, L, H]
        x = x[:, -1, :]             # [B, H]

        x = self.fc1(x)
        x = F.tanh(x)
        x = F.dropout(x, self.dropout, training=self.training)

        x = self.fc2(x)
        output = F.log_softmax(x, dim=-1)
        return output

