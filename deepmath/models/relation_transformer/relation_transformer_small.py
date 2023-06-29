import einops
import torch
from torch import nn
from deepmath.models.transformer.transformer_encoder_model import TransformerEmbedding


class AttentionRelationSmall(nn.Module):
    def __init__(self, ntoken,
                 embed_dim,
                 edge_dim=2,
                 num_heads=8,
                 dropout=0.,
                 num_layers=4,
                 bias=False,
                 global_pool=True,
                 edge_embed_dim=128,
                 pad=True,
                 inner_dim=None,
                 **kwargs):

        super().__init__()

        self.global_pool = global_pool
        self.embed_dim = embed_dim
        self.bias = bias
        self.pad = pad

        self.num_heads = num_heads

        if inner_dim is not None:
            self.inner_dim = inner_dim
        else:
            self.inner_dim = embed_dim // 2

        head_dim = self.inner_dim // num_heads

        assert head_dim * num_heads == self.inner_dim, "embed_dim must be divisible by num_heads"

        self.transformer_embedding = TransformerEmbedding(ntoken=None, d_model=self.inner_dim, nhead=num_heads,
                                                          d_hid=embed_dim, nlayers=num_layers, dropout=dropout,
                                                          enc=False, in_embed=False, global_pool=False)

        self.edge_embed = nn.Sequential(nn.Embedding(edge_dim + 1, edge_embed_dim * 2, padding_idx=0),
                                        nn.Dropout(dropout),
                                        nn.Linear(edge_embed_dim * 2, edge_embed_dim),
                                        nn.ReLU(),
                                        nn.Dropout(dropout),
                                        nn.Linear(edge_embed_dim, edge_embed_dim),
                                        nn.ReLU())

        if isinstance(ntoken, int):
            self.embedding = nn.Sequential(nn.Embedding(ntoken + 1, self.inner_dim * 2, padding_idx=0),
                                           nn.Dropout(dropout),
                                           nn.Linear(self.inner_dim * 2, self.inner_dim),
                                           nn.ReLU(),
                                           nn.Dropout(dropout),
                                           nn.Linear(self.inner_dim, self.inner_dim),
                                           nn.ReLU())
        elif isinstance(ntoken, nn.Module):
            self.embedding = ntoken
        else:
            raise ValueError("Not implemented!")

        self.scale = head_dim ** -0.5

        self.r_proj = nn.Sequential(nn.Dropout(dropout),
                                    nn.Linear(self.inner_dim * 2 + edge_embed_dim, self.inner_dim, bias=bias),
                                    nn.ReLU(),
                                    nn.Linear(self.inner_dim, self.inner_dim, bias=bias),
                                    nn.ReLU()
                                    )

        self.in_proj = nn.Sequential(nn.Linear(self.inner_dim, self.inner_dim), nn.GELU())
        self.out_proj = nn.Sequential(nn.Linear(self.inner_dim, self.inner_dim), nn.GELU())

        self.cls_token = nn.Parameter(torch.randn(1, self.inner_dim))

        # 1x1 conv equivalent to linear projection in output channel
        self.expand_proj = nn.Sequential(nn.Dropout(dropout),
                                         nn.Linear(self.inner_dim, self.inner_dim * 4),
                                         nn.ReLU(),
                                         nn.Linear(self.inner_dim * 4, self.inner_dim * 8),
                                         nn.ReLU())

    def forward(self,
                data,
                return_attn=False):

        xi = data.xi
        xj = data.xj
        edge_attr = data.edge_attr_

        xi = self.embedding(xi)
        xj = self.embedding(xj)

        xi = self.in_proj(xi)
        xj = self.out_proj(xj)

        if edge_attr is not None:
            edge_attr = self.edge_embed(edge_attr)
            R = torch.cat([xi, edge_attr, xj], dim=-1)
        else:
            R = torch.cat([xi, xj], dim=-1)

        R = self.r_proj(R)

        cls_tokens = einops.repeat(self.cls_token, '() d -> 1 b d', b=R.shape[1])

        R = torch.cat([R, cls_tokens], dim=0)

        enc = self.transformer_embedding(R, mask=data.mask)

        if self.global_pool:
            enc = self.expand_proj(enc)
            # max pool, shape (N , K, D)
            return torch.max(enc, dim=1)[0]
