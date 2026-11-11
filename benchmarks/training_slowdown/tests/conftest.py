import os

import pytest
import torch
import torch.nn.functional as F
from torch import nn

#: oneDNN's AMX kernels return a wrong bf16 product now and then when the process is preempted on a virtual CPU, on one
#: thread or several: two calls of one Newton-Schulz on one input differ, by a NaN as often as not. Below AMX the
#: product is the same every time. oneDNN reads the cap at its first product, so it is set before any test's.
os.environ["ONEDNN_MAX_CPU_ISA"] = "AVX512_CORE_BF16"

VOCAB, D, HEADS = 32, 16, 2
#: the 45M start's shape, as a start's record states it
SHAPE = {"d_model": 512, "n_layer": 6, "vocab": 50304, "ctx": 1024, "parameters": 45_029_888, "tokens": 154_206_208,
         "forward_macs_per_token": 51_342_080}


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


@pytest.fixture(autouse=True)
def one_thread():
    """Every test of the suite runs torch on one CPU thread: at the tests' shapes more threads only wait on each
    other. It does not make a bf16 product the same every time: the cap above does."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


@pytest.fixture
def tiny():
    torch.manual_seed(0)
    m = Tiny()
    theta = {k: v.detach().clone() for k, v in m.state_dict().items()}
    return m, theta


def loss_of(m, idx):
    return F.cross_entropy(m(idx).reshape(-1, VOCAB), idx.reshape(-1))
