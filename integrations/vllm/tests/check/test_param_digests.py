"""The chunked, threaded parameter hash records what one whole host copy per tensor recorded (`reference` below is that
implementation, as `weights_of_record.live_param_digests` had it): digests, dtypes, shapes, byte counts, aliases and kept words."""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from verity_vllm.check import weights_of_record as WR
from verity_vllm.check.param_digests import param_digests


def reference(model, *, include_buffers=True, raw_for=WR.ENGINE_ROTARY_RE):
    out, raws, seen = {}, {}, {}
    items = list(model.named_parameters())
    if include_buffers:
        items += list(model.named_buffers())
    for name, p in items:
        if not isinstance(p, torch.Tensor) or p.numel() == 0:
            continue
        if id(p) in seen:
            out[name] = dict(out[seen[id(p)]], alias_of=seen[id(p)])
            if seen[id(p)] in raws:
                raws[name] = raws[seen[id(p)]]
            continue
        seen[id(p)] = name
        flat = p.detach().contiguous().reshape(-1)
        raw = (flat.view(torch.uint8) if flat.dtype != torch.uint8 else flat).cpu().numpy()
        out[name] = {"digest": hashlib.sha256(memoryview(raw).cast("B")).hexdigest(), "dtype": str(p.dtype).replace("torch.", ""),
                     "shape": list(p.shape), "bytes": int(raw.nbytes)}
        if raw_for is not None and raw_for.search(name) and p.dtype == torch.bfloat16:
            raws[name] = np.ascontiguousarray(raw).view(np.uint16).reshape(list(p.shape))
    return {"params": out, "raw": raws, "n": len(out)}


class Layer(torch.nn.Module):
    def __init__(self, g):
        super().__init__()
        self.proj = torch.nn.Parameter(torch.randn(37, 11, generator=g).to(torch.bfloat16))
        self.strided = torch.nn.Parameter(torch.randn(9, 13, generator=g).t())                 # not contiguous
        self.experts = torch.nn.Parameter(torch.randn(3, 8, 5, generator=g).to(torch.float8_e4m3fn), requires_grad=False)
        self.register_buffer("mask", torch.rand(17, generator=g) > 0.5)
        self.register_buffer("ids", torch.randint(-2**40, 2**40, (6,), generator=g))
        self.register_buffer("scale", torch.tensor(0.731))                                     # 0-d
        self.register_buffer("words", torch.randint(0, 255, (29,), generator=g, dtype=torch.uint8))
        self.register_buffer("none", torch.zeros(0))


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(5)
        self.embed = torch.nn.Embedding(19, 7).to(torch.bfloat16)
        self.layers = torch.nn.ModuleList([Layer(g), Layer(g)])
        self.attn, rot = torch.nn.Module(), torch.nn.Module()
        rot.register_buffer("cos_sin_cache", torch.randn(12, 8, generator=g).to(torch.bfloat16))
        self.attn.rotary_emb = rot
        self.lm_head = torch.nn.Linear(7, 19, bias=False).to(torch.bfloat16)
        self.lm_head.weight = self.embed.weight                                                   # tied (named_parameters names it once)
        self.register_buffer("embed_words", self.embed.weight)                                   # a parameter under a buffer name


def same(a, b):
    assert list(a["params"]) == list(b["params"]) and a["params"] == b["params"] and a["n"] == b["n"]
    assert list(a["raw"]) == list(b["raw"]) and all(np.array_equal(a["raw"][k], b["raw"][k]) and a["raw"][k].dtype == b["raw"][k].dtype
                                                    for k in a["raw"])


@pytest.mark.parametrize("chunk,workers", [(1 << 28, 8), (7, 1), (7, 3), (1, 2), (64, 16)])
def test_chunked_threaded_digests_are_the_whole_copy_digests(chunk, workers):
    m = Model()
    same(param_digests(m, raw_for=WR.ENGINE_ROTARY_RE, chunk=chunk, workers=workers), reference(m))
    same(param_digests(m, include_buffers=False, chunk=chunk, workers=workers), reference(m, include_buffers=False, raw_for=None))


def test_the_records_entry_point_is_the_chunked_hash():
    m = Model()
    got = WR.live_param_digests(m)
    same(got, reference(m))
    assert got["params"]["embed_words"]["alias_of"] == "embed.weight"
    assert list(got["raw"]) == ["attn.rotary_emb.cos_sin_cache"] and "layers.0.none" not in got["params"]
