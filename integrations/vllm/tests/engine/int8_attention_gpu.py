"""The attention half of `int8_exact_gpu.py` (`--int8-exact attention|all`): taps on served attention layers and their checks.

`AttnTaps` wraps the forward of chosen `Int8AttentionImpl`s and keeps, per call and request, the query rows, the request's window
(k, v gathered from the paged cache after the call, as the step read them; under a credited read, `engine.int8_window`, the
rows of the step's buffers pg_window decoded it into), the K and V the layer was handed, the layer's K scales and the served
output rows, on the CPU.  `compare` judges one kept request, every check bit for bit (the LUT softmax has no device-dependent
step):

  rerun     `int8_attention.window_attention` (`window_rows` over `kv_quant`, the reference the served batch is held to) again on the GPU
  cpu       `window_attention` on the CPU
  kv        `kv_quant`'s K', V', s_v and f32(V'(0)) on the GPU against the CPU's
  oracle    `fattn_oracle.attention` (fattn_ref.py's softmax, apv_ref.py's served f) on the CPU
  fattn_ref the window's last query row of its first head: `fattn_ref.check` on the GPU's own S, P, m, e(0), s_q and z_q (head
            dimension 128 only)
  cache     a prefill's window from the cache against the K and V the layer was handed
  defs      the Definitions the int8 fold binds the window to, evaluated on it: `Int8KvQuant_v1{S,D}` of the first and last KV
            heads against the served `kv_quant`, `Int8LutAttentionRow_v1{D,S,SCALE}` of the first and last query rows, every
            head, against the served output

`written` judges a decode's window across the kept calls: its rows before the step's own are the K and V the layer was handed
for this request in the earlier kept calls, and its last rows the step's own (under an encoded credited read, rows written
earlier come back from the cache's encoding).
"""
from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import Any

import torch

from tests.engine import fattn_oracle as O
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import evaluate
from verity_vllm.engine import int8_attention as A
from verity.ml.definitions.registry import int8_exact as X

#: fattn_ref.py reads 128-wide rows (fattn.cu's head dimension); at any other the oracle and the Definitions are the checks
FATTN_REF_HEAD_DIM = 128


def _words(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int16)


def _u(t: torch.Tensor, dtype: torch.dtype, mask: int) -> list[int]:
    return [int(w) & mask for w in t.contiguous().view(dtype).reshape(-1).tolist()]


def _bf16(t: torch.Tensor) -> list[int]:
    return _u(t, torch.int16, 0xFFFF)


def _f32(t: torch.Tensor) -> list[int]:
    return _u(t.float(), torch.int32, 0xFFFFFFFF)


def _s8(t: torch.Tensor) -> list[int]:
    return _u(t, torch.int8, 0xFF)


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
            read = getattr(impl, "credited", None)
            st = read.current if read is not None else None
            if read is not None and (st is None or st.meta is not attn_metadata):
                raise RuntimeError(f"{name}: a credited read with no window of this step")
            for r, s in enumerate(lens[:len(qsl) - 1]):
                q0, q1 = qsl[r], qsl[r + 1]
                if q1 == q0:
                    continue
                if st is None:
                    k, v = A.window_of(kc, vc, attn_metadata.block_table[r], s)
                else:
                    k, v = st.k[st.offsets[r]:st.offsets[r + 1]], st.v[st.offsets[r]:st.offsets[r + 1]]
                rec = {"call": self.calls[(self.phase, name)] - 1, "request": r, "s": s, "n": q1 - q0, "scale": impl.scale,
                       "k_scales": ks, "q": query[q0:q1].cpu(), "k": k.cpu(), "v": v.cpu(), "out": output[q0:q1].cpu(),
                       "window": "cache" if st is None else "credited", "handed": (key[q0:q1].cpu(), value[q0:q1].cpu())}
                self.got.setdefault((self.phase, name), []).append(rec)
            return out
        return wrapped

    def close(self) -> None:
        for n, impl in self.impls.items():
            impl.forward = self.saved[n]


def fattn_ref_row(q: torch.Tensor, k: torch.Tensor, ks: torch.Tensor, scale: float) -> bool | None:
    """fattn_ref.check on the GPU's softmax of the window's last row of query head 0 (KV head 0); None off FATTN_REF_HEAD_DIM."""
    n, _, d = q.shape
    s = k.shape[0]
    if d != FATTN_REF_HEAD_DIM:
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
    got = {"s": s, "n": n, "head_dim": int(q.shape[-1]), "words": int(ow.numel()), "rerun_mismatch": int((_words(rerun) != ow).sum()),
           "cpu_mismatch": int((_words(cpu) != ow).sum()), "oracle_mismatch": int((_words(oracle) != ow).sum()),
           "kv_mismatch": sum(int((a.cpu().view(torch.uint8) != b.view(torch.uint8)).sum()) for a, b in zip(kv_gpu, kv_cpu)),
           "fattn_ref": fattn_ref_row(q, k, ks, scale), "scale_is_head_dim": scale == q.shape[-1] ** -0.5,
           "k_clamped": int((k.float().abs() / ks[None, :, None] > 127.5).sum()), "window": rec["window"],
           **defs(q, k, v, ks, scale, out, [t.cpu() for t in kv_gpu])}
    if n == s:
        got["cache_mismatch"] = int(not (torch.equal(rec["handed"][0], k) and torch.equal(rec["handed"][1], v)))
    return got


def defs(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, ks: torch.Tensor, scale: float, out: torch.Tensor,
         served_kv: list[torch.Tensor]) -> dict[str, int]:
    """`Int8KvQuant_v1` and `Int8LutAttentionRow_v1`, bound as the fold binds them, evaluated on the window (see the module)."""
    n, h, d = q.shape
    s, g = k.shape[0], k.shape[1]
    kq, vq, sv, v0 = served_kv
    kvd = bind(X.Int8KvQuant, S=s, D=d)
    evaluated = mismatch = 0
    for gi in sorted({0, g - 1}):
        got = evaluate(kvd, _bf16(k[:, gi]), _bf16(v[:, gi]), _f32(ks[gi:gi + 1])[0])
        want = _s8(kq[gi]) + _s8(vq[gi]) + _f32(sv[gi]) + _f32(v0[gi])
        evaluated, mismatch = evaluated + 1, mismatch + int(list(got) != want)
    for i in sorted({0, n - 1}):
        p = s - n + i + 1
        row = bind(X.Int8LutAttentionRow, D=d, S=p, SCALE=scale)
        for hd in range(h):
            gi = hd // (h // g)
            got = evaluate(row, _bf16(q[i, hd]), _s8(kq[gi, :p]), _f32(ks[gi:gi + 1])[0], _s8(vq[gi, :p]), _f32(sv[gi]),
                           _f32(v0[gi]))
            evaluated, mismatch = evaluated + 1, mismatch + int(list(got) != _bf16(out[i, hd]))
    return {"defs_evaluated": evaluated, "defs_mismatch": mismatch}


def written(got: dict[tuple[str, str], list[dict[str, Any]]]) -> dict[str, int]:
    """Each kept decode's window against the K and V its request was handed in its phase's earlier kept calls (see the module)."""
    compared = mismatch = 0
    for recs in got.values():
        hist: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for r in sorted(recs, key=lambda x: x["call"]):
            hk, hv = r["handed"]
            prior = hist.get(r["request"])
            if r["n"] < r["s"] and prior is not None and prior[0].shape[0] == r["s"] - r["n"]:
                compared += 1
                mismatch += int(not (torch.equal(torch.cat([prior[0], hk]), r["k"]) and torch.equal(torch.cat([prior[1], hv]), r["v"])))
            hist[r["request"]] = ((torch.cat([prior[0], hk]), torch.cat([prior[1], hv])) if prior is not None and r["n"] < r["s"]
                                  else (hk, hv))
    return {"written_compared": compared, "written_mismatch": mismatch}


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
