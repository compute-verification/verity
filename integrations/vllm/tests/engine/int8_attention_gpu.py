"""The attention half of `int8_exact_gpu.py` (`--int8-exact attention|all`): taps on served attention layers and their checks.

`AttnTaps` wraps the forward of chosen `Int8AttentionImpl`s and keeps, per call and request, the query rows, the request's window
(k, v gathered from the paged cache after the call, as the step read them) and the served output rows, on the CPU.  `compare`
judges one kept request:

  rerun        `int8_attention.window_attention` again on the GPU                                    (equal, bit for bit)
  gpu_softmax  `window_attention` on the CPU with only its softmax run on the GPU                    (equal: every other step
               of the GPU's served value is the CPU's)
  oracle       `benchmarks/ppl`'s own `edited_attention` on the CPU (`ppl_oracle.int8_attention`)     (counted: torch.softmax
               rounds apart on the two devices, so these may differ; `cpu_oracle` is ours on the CPU against it, equal)
  p_ints       how many of P's integers the two devices' softmax moves, and by how much at most
  cache        a prefill's window from the cache against the K and V the layer was handed (equal)
"""
from __future__ import annotations

import hashlib
from typing import Any

import torch

from tests.engine import ppl_oracle as O
from verity_vllm.engine import int8_attention as A


def _words(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int16)


def _key(*ts: torch.Tensor) -> str:
    h = hashlib.sha256()
    for t in ts:
        h.update(_words(t).numpy().tobytes())
    return h.hexdigest()


class AttnTaps:
    """Keep each tapped layer's first `keep` calls per phase, request by request."""

    def __init__(self, model: torch.nn.Module, names: list[str], keep: int):
        self.phase: str | None = None
        self.keep = keep
        self.got: dict[tuple[str, str], list[dict[str, Any]]] = {}
        self.calls: dict[tuple[str, str], int] = {}
        mods = dict(model.named_modules())
        self.impls = {n: mods[n].impl for n in names}
        self.saved = {n: impl.forward for n, impl in self.impls.items()}
        for n, impl in self.impls.items():
            impl.forward = self._wrap(n, self.saved[n])

    def _wrap(self, name: str, forward):
        def wrapped(layer, query, key, value, kv_cache, attn_metadata, output, *args, **kwargs):
            out = forward(layer, query, key, value, kv_cache, attn_metadata, output, *args, **kwargs)
            if self.phase is None or attn_metadata is None or self.calls.get((self.phase, name), 0) >= self.keep:
                return out
            self.calls[(self.phase, name)] = self.calls.get((self.phase, name), 0) + 1
            impl = self.impls[name]
            kc, vc = kv_cache.transpose(1, 2).split(impl.head_size, dim=-1)
            qsl, lens = attn_metadata.query_start_loc.tolist(), attn_metadata.seq_lens.tolist()
            for r, s in enumerate(lens):
                q0, q1 = qsl[r], qsl[r + 1]
                if q1 == q0:
                    continue
                k, v = A.window_of(kc, vc, attn_metadata.block_table[r], s)
                rec = {"call": self.calls[(self.phase, name)] - 1, "request": r, "s": s, "n": q1 - q0, "scale": impl.scale,
                       "q": query[q0:q1].cpu(), "k": k.cpu(), "v": v.cpu(), "out": output[q0:q1].cpu()}
                if q1 - q0 == s:
                    rec["handed"] = (key[q0:q1].cpu(), value[q0:q1].cpu())
                self.got.setdefault((self.phase, name), []).append(rec)
            return out
        return wrapped

    def close(self) -> None:
        for n, impl in self.impls.items():
            impl.forward = self.saved[n]


def _gpu_softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.softmax(x.cuda(), dim).cpu()


def p_ints(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float) -> tuple[int, int, int]:
    """(P integers, how many the GPU's softmax moves from the CPU's, the largest move) over the window's KV groups."""
    n, h, d = q.shape
    s, g_heads, _ = k.shape
    rep = h // g_heads
    causal = torch.arange(s)[None, :] <= torch.arange(s - n, s)[:, None]
    total = moved = most = 0
    for g in range(g_heads):
        qi, sq = A.quant(q[:, g * rep:(g + 1) * rep].transpose(0, 1).float(), -1)
        ki, sk = A.quant(k[:, g].float(), -1)
        logits = ((A.int_mm(qi.reshape(rep * n, d), ki).view(rep, n, s).float() * (sq * sk.T[None])) * scale).masked_fill(~causal, float("-inf"))
        cpu, gpu = A.quant_p(torch.softmax(logits, -1))[0], A.quant_p(_gpu_softmax(logits, -1))[0]
        diff = (cpu - gpu).abs()
        total, moved, most = total + diff.numel(), moved + int((diff > 0).sum()), max(most, int(diff.max()))
    return total, moved, most


def compare(rec: dict[str, Any]) -> dict[str, Any]:
    q, k, v, out, s, n, scale = rec["q"], rec["k"], rec["v"], rec["out"], rec["s"], rec["n"], rec["scale"]
    q_full = torch.cat([torch.zeros(s - n, *q.shape[1:], dtype=q.dtype), q])          # rows before the call's are not read
    oracle = O.int8_attention(q_full, k, v)[s - n:]
    ours_cpu = A.window_attention(q, k, v, scale)
    rerun = A.window_attention(q.cuda(), k.cuda(), v.cuda(), scale).cpu()
    gpu_sm = A.window_attention(q, k, v, scale, _gpu_softmax)
    total, moved, most = p_ints(q, k, v, scale)
    ow = _words(out)
    got = {"s": s, "n": n, "words": int(ow.numel()), "rerun_mismatch": int((_words(rerun) != ow).sum()),
           "gpu_softmax_mismatch": int((_words(gpu_sm) != ow).sum()), "oracle_mismatch": int((_words(oracle) != ow).sum()),
           "cpu_oracle_mismatch": int((_words(ours_cpu) != _words(oracle)).sum()), "p_ints": total, "p_ints_moved": moved,
           "p_ints_max_move": most, "scale_is_head_dim": scale == q.shape[-1] ** -0.5}
    if "handed" in rec:
        got["cache_mismatch"] = int(not (torch.equal(rec["handed"][0], k) and torch.equal(rec["handed"][1], v)))
    return got


def batch_vs_solo(got: dict[tuple[str, str], list[dict[str, Any]]], names: list[str]) -> dict[str, Any]:
    """Per layer and solo prompt: the kept solo requests whose (q, k, v) bytes are a batched request's, and how many of those
    differ in their output."""
    out: dict[str, Any] = {}
    for name in names:
        batched = {_key(r["q"], r["k"], r["v"]): _words(r["out"]) for r in got.get(("batched", name), [])}
        per = []
        for phase in sorted(p for p, m in got if m == name and p.startswith("solo")):
            recs = got[(phase, name)]
            hits = [batched.get(_key(r["q"], r["k"], r["v"])) for r in recs]
            per.append({"phase": phase, "requests": len(recs), "matched": sum(h is not None for h in hits),
                        "mismatch": sum(h is not None and not torch.equal(h, _words(r["out"])) for h, r in zip(hits, recs))})
        out[name] = per
    return out
