import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

VOCAB, D, HEADS = 32, 16, 2


class Tiny(nn.Module):
    """Embedding, causal self-attention, one MLP matrix, a tied head: three weight matmuls per token in a chain."""

    def __init__(self):
        super().__init__()
        self.emb = nn.Parameter(torch.randn(VOCAB, D) * 0.1)
        self.w1 = nn.Linear(D, D, bias=False)
        self.w2 = nn.Linear(D, D, bias=False)

    def forward(self, idx):
        x = F.embedding(idx, self.emb)
        b, t, d = x.shape
        q = x.view(b, t, HEADS, d // HEADS).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, q, q, is_causal=True).transpose(1, 2).reshape(b, t, d)
        x = x + self.w1(a)
        x = x + self.w2(F.silu(x))
        return F.linear(x, self.emb)


#: per token, the forward's weight MACs: w1 (D*D), w2 (D*D), the head (VOCAB*D)
PER_TOKEN = 2 * D * D + VOCAB * D


@pytest.fixture
def tiny():
    torch.manual_seed(0)
    m = Tiny()
    theta = {k: v.detach().clone() for k, v in m.state_dict().items()}
    return m, theta


def loss_of(m, idx):
    return F.cross_entropy(m(idx).reshape(-1, VOCAB), idx.reshape(-1))
