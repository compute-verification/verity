"""The attention half of `int8_exact_gpu.py` (`--int8-exact attention|all`): taps on served attention layers and their checks.

`AttnTaps` wraps the forward of chosen `Int8AttentionImpl`s and keeps, per call and request, the query rows, the request's window
(k, v gathered from the paged cache after the call, as the step read them), the layer's K scales and the served output rows, on
the CPU.  `compare` judges one kept request, every check bit for bit (the LUT softmax has no device-dependent step):

  rerun     `int8_attention.window_attention` (`window_rows` over `kv_quant`, as served) again on the GPU
  cpu       `window_attention` on the CPU
  kv        `kv_quant`'s K', V', s_v and f32(V'(0)) on the GPU against the CPU's
  oracle    `fattn_oracle.attention` (fattn_ref.py's softmax, apv_ref.py's served f) on the CPU
  fattn_ref the window's last query row of its first head: `fattn_ref.check` on the GPU's own S, P, m, e(0), s_q and z_q (head
            dimension 128 only)
  cache     a prefill's window from the cache against the K and V the layer was handed
"""
from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import Any

import torch

from tests.engine import fattn_oracle as O
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
            ks = impl.k_scales(layer.layer_name, query.device).cpu()
            qsl, lens = attn_metadata.query_start_loc.tolist(), attn_metadata.seq_lens.tolist()
            for r, s in enumerate(lens):
                q0, q1 = qsl[r], qsl[r + 1]
                if q1 == q0:
                    continue
                k, v = A.window_of(kc, vc, attn_metadata.block_table[r], s)
                rec = {"call": self.calls[(self.phase, name)] - 1, "request": r, "s": s, "n": q1 - q0, "scale": impl.scale,
                       "k_scales": ks, "q": query[q0:q1].cpu(), "k": k.cpu(), "v": v.cpu(), "out": output[q0:q1].cpu()}
                if q1 - q0 == s:
                    rec["handed"] = (key[q0:q1].cpu(), value[q0:q1].cpu())
                self.got.setdefault((self.phase, name), []).append(rec)
            return out
        return wrapped

    def close(self) -> None:
        for n, impl in self.impls.items():
            impl.forward = self.saved[n]


def fattn_ref_row(q: torch.Tensor, k: torch.Tensor, ks: torch.Tensor, scale: float) -> bool | None:
    """fattn_ref.check on the GPU's softmax of the window's last row of query head 0 (KV head 0); None off head dimension 128."""
    n, _, d = q.shape
    s = k.shape[0]
    if d != 128:
        return None
    qi, sq = A.quant(q[-1:, 0].float().cuda(), -1)
    ki = A.quant_static(k[:, 0].float().cuda(), ks[0])
    S = A.int_mm(qi, ki)
    a = A.fixed_point_a(sq, ks[0], scale)
    causal = torch.ones(1, s, dtype=torch.bool, device=S.device)
    e, m = A.lut_e(S, a, causal)
    p, r, z = A.lut_softmax(S, a, causal)
    served = {"P": p[0].cpu().numpy(), "m": int(m[0, 0]), "e0": int(e[0, 0]), "sp": r[0].cpu().numpy(), "sz": z[0].cpu().numpy()}
    with tempfile.TemporaryDirectory() as tmp:
        got = O.check(O.gate_dir(Path(tmp) / "gate", ki.cpu().numpy(), qi[0].cpu().numpy(), s - 1, int(a[0, 0]), served,
                                 S[0].cpu().numpy()))
    return bool(got["ok"])


def compare(rec: dict[str, Any]) -> dict[str, Any]:
    q, k, v, out, s, n, scale, ks = rec["q"], rec["k"], rec["v"], rec["out"], rec["s"], rec["n"], rec["scale"], rec["k_scales"]
    rerun = A.window_attention(q.cuda(), k.cuda(), v.cuda(), ks.cuda(), scale).cpu()
    cpu = A.window_attention(q, k, v, ks, scale)
    oracle = O.attention(q, k, v, ks, scale)
    kv_gpu, kv_cpu = A.kv_quant(k.cuda(), v.cuda(), ks.cuda()), A.kv_quant(k, v, ks)
    ow = _words(out)
    got = {"s": s, "n": n, "words": int(ow.numel()), "rerun_mismatch": int((_words(rerun) != ow).sum()),
           "cpu_mismatch": int((_words(cpu) != ow).sum()), "oracle_mismatch": int((_words(oracle) != ow).sum()),
           "kv_mismatch": sum(int((a.cpu().view(torch.uint8) != b.view(torch.uint8)).sum()) for a, b in zip(kv_gpu, kv_cpu)),
           "fattn_ref": fattn_ref_row(q, k, ks, scale), "scale_is_head_dim": scale == q.shape[-1] ** -0.5,
           "k_clamped": int((k.float().abs() / ks[None, :, None] > 127.5).sum())}
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
