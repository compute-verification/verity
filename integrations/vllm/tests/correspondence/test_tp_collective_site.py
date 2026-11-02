"""The naming of TP collective rows (`AllReduce2Rule` / `AllGather2Rule`, `_collective_hint`).

A collective whose rank-local operand is produced UNDER its own module (vLLM `RowParallelLinear`: the Gemm and its all-reduce in one
forward) keeps the module vocabulary `<module>/<node>[i]/all_reduce` -- the Gemm under the module is the rank's partial.  A
collective issued MID-module, whose rank-local operand is another module's output (OLMoE `OlmoeAttention._apply_qk_norm`: q and k
all-gathered inside `self_attn`, each a slice of the `qkv_proj` Gemm's output), is named at its call SITE `<module>.<fx node>/<node>`,
which the v2 query (`required.collective_sites`) turns into the module's k-th collective identity with its own `shard_rank<r>` member.
Without the site the manifest folded both gathers into one `self_attn/out` of twice the width with no partial to bind (on OLMoE at TP2:
6,944 unbound identities per rank).  CPU, meta tensors; no vLLM.

Run: `PYTHONPATH=.:../.. python -m pytest -q -p no:cacheprovider tests/correspondence/test_tp_collective_site.py`
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from verity_vllm.query.manifest.format import split_name
from verity_vllm.program.frontend.export_ops import ensure_ops
from verity_vllm.program.frontend.derive import derive_report
from verity_vllm.program.frontend.rules.vllm_bindings import vllm_ruleset

PROF = dict(mode="profile", profile="b1-eager-v3")
BF = torch.bfloat16
H, Q = 16, 4                                                                  # hidden, per-rank q (and k) columns


def _meta(*shape):
    return torch.empty(*shape, dtype=BF, device="meta")


class Proj(nn.Module):
    """A bias-free linear (the qkv_proj of a rank: N = 2Q columns = this rank's q and k shards)."""

    def __init__(self, K: int, N: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(N, K, dtype=BF, device="meta"), requires_grad=False)

    def forward(self, x):
        return F.linear(x, self.weight)


class MidModuleAttention(nn.Module):
    """OLMoE's shape: the q and k slices of the qkv Gemm's output are all-gathered INSIDE self_attn (two calls), then returned."""

    def __init__(self) -> None:
        super().__init__()
        self.qkv_proj = Proj(H, 2 * Q)

    def forward(self, x, peer_q, peer_k):
        qkv = self.qkv_proj(x)
        q, k = qkv[..., :Q], qkv[..., Q:]
        gq = torch.ops.verity_tp.all_gather2(q, peer_q)                     # rank order: (own, peer) on rank 0
        gk = torch.ops.verity_tp.all_gather2(k, peer_k)
        return gq, gk


class RowParallel(nn.Module):
    """vLLM `RowParallelLinear`: the Gemm on the K shard and its all-reduce in ONE forward -> the collective is the module's."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(H, H, dtype=BF, device="meta"), requires_grad=False)

    def forward(self, x, peer):
        return torch.ops.verity_tp.all_reduce2(F.linear(x, self.weight), peer)


class Layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = MidModuleAttention()
        self.o_proj = RowParallel()

    def forward(self, x, peer_q, peer_k, peer_o):
        gq, gk = self.self_attn(x, peer_q, peer_k)
        return gq, gk, self.o_proj(x, peer_o)


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([Layer()])

    def forward(self, x, peer_q, peer_k, peer_o):
        return self.layers[0](x, peer_q, peer_k, peer_o)


def _rows(rep) -> list[tuple[str, str]]:
    """(family, name) of every emitted row, from the Program's node hints and node ids."""
    hints = rep.program.annotations["node_hints"]
    nodes = rep.flat_definition.body.nodes
    return [(n.fn.id.split("{")[0], hints[str(i)]) for i, n in enumerate(nodes) if str(i) in hints]


def test_mid_module_collective_is_named_at_its_site_and_the_module_collective_stays_bare():
    ensure_ops()
    T = 3
    rep = derive_report(Model(), (_meta(T, H), _meta(T, Q), _meta(T, Q), _meta(T, H)), **PROF, rules=vllm_ruleset(), name="TpSite")
    assert rep.ok, rep.unsupported
    rows = _rows(rep)
    gathers = sorted({split_name(n)[0] for f, n in rows if f == "AllGather2_v1"})
    reduces = sorted({split_name(n)[0] for f, n in rows if f == "AllReduce2_v1"})
    gemms = sorted({split_name(n)[0] for f, n in rows if f == "Gemm_v1"})
    # the two mid-module gathers: two SITES under self_attn, each named by its own fx node -- never both under the bare module
    assert len(gathers) == 2 and all(m.rsplit(".", 1)[0] == "layers.0.self_attn" for m in gathers), gathers
    assert all(m.rsplit(".", 1)[1].startswith("all_gather2") for m in gathers)
    assert "layers.0.self_attn" not in gathers
    assert sum(1 for f, n in rows if f == "AllGather2_v1") == 2 * T                # one row per token per call
    # the row-parallel all-reduce stays the module's own (its Gemm is under the same module) -- the dense TP2 vocabulary is unchanged
    assert reduces == ["layers.0.o_proj"]
    assert gemms == ["layers.0.o_proj", "layers.0.self_attn.qkv_proj"]
    # provenance: the collective applications still carry the MODULE path (tp_links.json binds peer inputs by it, not by the row name)
    apps = [a for a in rep.provenance["applications"] if a["rule"].startswith("AllGather2")]
    assert len(apps) == 2 and all("layers.0.self_attn (" in a["module_path"] for a in apps), apps
