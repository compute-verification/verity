"""End-to-end check of the derived-boundary evaluators against a committed B0 record (lane VU-ACQUIRE).

For every norm / attention / silu row of one committed request, reconstruct the derived interior values
(`verity_vllm.registry.derived_rows`, or its torch twin with ``--backend torch``) from the committed *operand*
leaves, and show that the derived boundary plus the committed operands reproduces the committed *downstream*
leaves bit for bit — the property a plan needs to place a VU boundary on an interior gate:

  * `rmsnorm_triton`  (layer-0 input norm; one row per position): derived `inv`; downstream
    `out = bf16((inv*x)*w)` against `layers.0.input_layernorm.rmsnorm/port=0`.
  * `rmsnorm_fused_v2` (all fused norms): derived `s` (and `var`); downstream `out = bf16((res'*s)*w)`
    against the committed normed leaf (residual leaf checked too).
  * `attention` (position x layer x head rows): derived per-64-key-block `(m, l[4], o[D])` states;
    downstream `out = bf16(o * Fa2InvSum((l0+l2)+(l1+l3)))` against the committed
    `self_attn.attn.attn/port=0` rows.  B0 (SmolLM2-135M) runs FA2 `flash_fwd_splitkv_kernel` at head
    dim 64, whose block trait is `model-untested` in the registry (b1.py B0 note): this sweep is the
    empirical test.
  * `silu_mul` (position x layer rows): derived bf16 `mid = silu(gate)`; downstream `out = bf16(mid * up)`
    against `mlp.act_fn.act_mul/port=0`.

Default record: the in-repo B0 commit bundle (`rmsnorm_fused_sweep.BUNDLE`, SmolLM2-135M on A100, vLLM
d9105ea80, VLLM_BATCH_INVARIANT=1; one request, 32 prompt + 7 decode positions committed).  Any commit-v2
store works via ``--commit`` / ``--request`` (e.g. the INT B0 record's, 4 requests x 48 positions):

    PYTHONPATH=. python -m verity_vllm_adapter.derived_b0_check \
        [--commit DIR] [--request RID] [--backend numpy|torch|torch-cuda] [--out report.json]
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from verity_vllm.registry import rmsnorm_fused_sweep as SW  # noqa: E402
from verity_vllm.registry.b1 import B0_CONFIG  # noqa: E402
from verity_vllm.registry import derived_rows as D  # noqa: E402


class Evaluators:
    """The four row evaluators over numpy word arrays, on the chosen backend."""

    def __init__(self, backend: str):
        self.backend = backend
        if backend == "numpy":
            from verity_vllm.registry import derived_rows as mod
            self._m, self._dev = mod, None
        else:
            import torch
            from verity_vllm.registry import derived_rows_torch as mod
            self._m = mod
            self._dev = torch.device("cuda" if backend == "torch-cuda" else "cpu")
            if backend == "torch-cuda" and not torch.cuda.is_available():
                raise SystemExit("--backend torch-cuda: no CUDA device")

    def _in(self, a):
        if self._dev is None:
            return a
        import torch
        return torch.from_numpy(np.ascontiguousarray(a).astype(np.int64)).to(self._dev)

    def _out(self, r):
        if self._dev is None:
            return r
        return {k: (v.cpu().numpy() if hasattr(v, "cpu") else v) for k, v in r.items()}

    def rmsnorm_triton_rows(self, x, w, eps):
        return self._out(self._m.rmsnorm_triton_rows(self._in(x), self._in(w), eps))

    def rmsnorm_fused_v2_rows(self, x, res, w, eps):
        return self._out(self._m.rmsnorm_fused_v2_rows(self._in(x), self._in(res), self._in(w), eps))

    def silu_mul_rows(self, gu):
        return self._out(self._m.silu_mul_rows(self._in(gu)))

    def attention_head_states(self, q, k, v):
        return self._out(self._m.attention_head_states(self._in(q), self._in(k), self._in(v)))


def phase_map(leaves) -> dict[int, str]:
    ph = {}
    for key in leaves:
        t, rest = key.split("/", 1)
        ph[int(t)] = rest.split("ph=", 1)[1].split("/", 1)[0]
    return ph


def _leaf(leaves, ph, t, l, op, port):
    return leaves[f"{t}/ph={ph[t]}/l={l}/op={op}/port={port}"]


def check_triton_norm(ev, leaves, norms, C):
    """Layer-0 input norm (`_rms_norm_kernel`, Triton): one row per position."""
    ph = phase_map(leaves)
    ts = sorted(ph)
    x = np.stack([_leaf(leaves, ph, t, "embed", "embed_tokens.embedding", 0) for t in ts])
    got_leaf = np.stack([_leaf(leaves, ph, t, 0, "layers.0.input_layernorm.rmsnorm", 0) for t in ts])
    w = norms["model.layers.0.input_layernorm.weight"]
    r = ev.rmsnorm_triton_rows(x, w, C["EPS"])
    full_ok = r["out"].astype(np.uint16) == got_leaf
    # downstream from the derived boundary alone: out = bf16((inv * x) * w) with the *derived* inv
    inv = np.asarray(r["inv"], dtype=np.uint32)
    y = D.mul_rn(D.mul_rn(inv[:, None], D.bf16_to_f32_words(x)), D.bf16_to_f32_words(np.broadcast_to(w, x.shape)))
    from_inv = D.f2fp_bf16(y) == got_leaf
    return {"rows_checked": len(ts), "rows_exact": int(np.all(full_ok, axis=1).sum()),
            "words_compared": int(full_ok.size), "words_exact": int(full_ok.sum()),
            "rows_exact_from_derived_inv": int(np.all(from_inv, axis=1).sum()),
            "derived_words_per_row": 1, "derived_bytes_per_row": 4}


def fused_norm_rows(leaves, norms, C):
    """All fused-norm call sites of the request (as `rmsnorm_fused_sweep.fused_norm_rows`, phase-agnostic)."""
    NL = C["NL"]
    ph = phase_map(leaves)

    def leaf(t, l, op, port):
        return _leaf(leaves, ph, t, l, op, port)

    rows = []
    for t in sorted(ph):
        for L in range(NL):
            x = leaf(t, L, f"layers.{L}.self_attn.o_proj.gemm", 0)
            res = leaf(t, "embed", "embed_tokens.embedding", 0) if L == 0 else leaf(t, L, f"layers.{L}.input_layernorm.rmsnorm", 1)
            rows.append((f"t={t}/layers.{L}/post_attention_layernorm", x, res,
                         norms[f"model.layers.{L}.post_attention_layernorm.weight"],
                         leaf(t, L, f"layers.{L}.post_attention_layernorm.rmsnorm", 0),
                         leaf(t, L, f"layers.{L}.post_attention_layernorm.rmsnorm", 1)))
            if L >= 1:
                x = leaf(t, L - 1, f"layers.{L - 1}.mlp.down_proj.gemm", 0)
                res = leaf(t, L - 1, f"layers.{L - 1}.post_attention_layernorm.rmsnorm", 1)
                rows.append((f"t={t}/layers.{L}/input_layernorm", x, res,
                             norms[f"model.layers.{L}.input_layernorm.weight"],
                             leaf(t, L, f"layers.{L}.input_layernorm.rmsnorm", 0),
                             leaf(t, L, f"layers.{L}.input_layernorm.rmsnorm", 1)))
        x = leaf(t, NL - 1, f"layers.{NL - 1}.mlp.down_proj.gemm", 0)
        res = leaf(t, NL - 1, f"layers.{NL - 1}.post_attention_layernorm.rmsnorm", 1)
        rows.append((f"t={t}/norm", x, res, norms["model.norm.weight"],
                     leaf(t, "norm", "norm.rmsnorm", 0), leaf(t, "norm", "norm.rmsnorm", 1)))
    return rows


def check_fused_norm(ev, leaves, norms, C):
    rows = fused_norm_rows(leaves, norms, C)
    X = np.stack([r[1] for r in rows]); RS = np.stack([r[2] for r in rows]); W = np.stack([r[3] for r in rows])
    O0 = np.stack([r[4] for r in rows]); O1 = np.stack([r[5] for r in rows])
    r = ev.rmsnorm_fused_v2_rows(X, RS, W, C["EPS"])
    ok_out = r["normed"].astype(np.uint16) == O0
    ok_res = r["resid"].astype(np.uint16) == O1
    # downstream from the derived boundary alone: out = bf16((res' * s) * w), res' from the committed leaf
    s = np.asarray(r["s"], dtype=np.uint32)
    y = D.mul_rn(D.mul_rn(D.bf16_to_f32_words(O1), s[:, None]), D.bf16_to_f32_words(W))
    from_s = D.f2fp_bf16(y) == O0
    return {"rows_checked": len(rows), "rows_exact": int((np.all(ok_out, 1) & np.all(ok_res, 1)).sum()),
            "words_compared": int(ok_out.size + ok_res.size), "words_exact": int(ok_out.sum() + ok_res.sum()),
            "rows_exact_from_derived_s": int(np.all(from_s, axis=1).sum()),
            "derived_words_per_row": 2, "derived_bytes_per_row": 8}


def check_silu(ev, leaves, C):
    NL, I = C["NL"], C["I"]
    ph = phase_map(leaves)
    ts = sorted(ph)
    gu = np.stack([_leaf(leaves, ph, t, L, f"layers.{L}.mlp.gate_up_proj.gemm", 0)
                   for t in ts for L in range(NL)])
    out_leaf = np.stack([_leaf(leaves, ph, t, L, f"layers.{L}.mlp.act_fn.act_mul", 0)
                         for t in ts for L in range(NL)])
    r = ev.silu_mul_rows(gu)
    ok = r["out"].astype(np.uint16) == out_leaf
    # downstream from the derived boundary alone: bf16(f32(mid) * f32(up))
    mid = np.asarray(r["mid"], dtype=np.uint16)
    from_mid = D.bf16_rn_torch(D.mul_rn(D.bf16_to_f32_words(mid), D.bf16_to_f32_words(gu[:, I:]))) == out_leaf
    return {"rows_checked": gu.shape[0], "rows_exact": int(np.all(ok, axis=1).sum()),
            "words_compared": int(ok.size), "words_exact": int(ok.sum()),
            "rows_exact_from_derived_mid": int(np.all(from_mid, axis=1).sum()),
            "derived_words_per_row": I, "derived_bytes_per_row": 2 * I}


def attention_inputs(leaves, C):
    """KV caches and per-position q / out rows from the committed leaves."""
    NL, NH, KVH, Dh = C["NL"], C["NH"], C["KVH"], C["D"]
    ph = phase_map(leaves)
    ts = sorted(ph)
    Tmax = max(ts) + 1
    kcache = np.zeros((NL, Tmax, KVH, Dh), dtype=np.uint16)
    vcache = np.zeros((NL, Tmax, KVH, Dh), dtype=np.uint16)
    for t in ts:
        for L in range(NL):
            kcache[L, t] = _leaf(leaves, ph, t, L, f"layers.{L}.self_attn.rotary_emb.rope", 1).reshape(KVH, Dh)
            qkv = _leaf(leaves, ph, t, L, f"layers.{L}.self_attn.qkv_proj.gemm", 0)
            vcache[L, t] = qkv[NH * Dh + KVH * Dh:].reshape(KVH, Dh)
    return ts, ph, kcache, vcache


def check_attention(ev, leaves, C):
    NL, NH, KVH, Dh = C["NL"], C["NH"], C["KVH"], C["D"]
    G = NH // KVH
    ts, ph, kcache, vcache = attention_inputs(leaves, C)
    rows_checked = rows_exact = words_exact = words_compared = 0
    from_states_exact = 0
    blocks_total = 0
    nonfinite = 0
    mism = []
    for t in ts:
        T = t + 1
        q = np.stack([_leaf(leaves, ph, t, L, f"layers.{L}.self_attn.rotary_emb.rope", 0).reshape(NH, Dh)[h]
                      for L in range(NL) for h in range(NH)])
        out_leaf = np.stack([_leaf(leaves, ph, t, L, f"layers.{L}.self_attn.attn.attn", 0).reshape(NH, Dh)[h]
                             for L in range(NL) for h in range(NH)])
        k = np.stack([kcache[L, :T, h // G] for L in range(NL) for h in range(NH)])
        v = np.stack([vcache[L, :T, h // G] for L in range(NL) for h in range(NH)])
        r = ev.attention_head_states(q, k, v)
        out = r["out"].astype(np.uint16)
        ok = out == out_leaf
        rows_checked += q.shape[0]
        rows_exact += int(np.all(ok, axis=1).sum())
        words_exact += int(ok.sum()); words_compared += int(ok.size)
        nonfinite += int(np.asarray(r["nonfinite"]).sum())
        blocks_total += q.shape[0] * len(r["blocks"])
        # downstream from the derived boundary alone: the final block state (m, l, o) -> epilogue
        l = np.asarray(r["l"], dtype=np.uint32)[:, -1]
        o = np.asarray(r["o"], dtype=np.uint32)[:, -1]
        total = D.add_ftz(D.add_ftz(l[:, 0], l[:, 2]), D.add_ftz(l[:, 1], l[:, 3]))
        y = D.f2fp_bf16(D.mul_ftz(o, D.fa2_inv_sum(total)[:, None]))
        from_states_exact += int(np.all(y == out_leaf, axis=1).sum())
        if not np.all(ok):
            for i in np.nonzero(~np.all(ok, axis=1))[0][:3]:
                mism.append({"t": t, "L": int(i) // NH, "h": int(i) % NH, "words": int((~ok[i]).sum())})
    return {"rows_checked": rows_checked, "rows_exact": rows_exact,
            "words_compared": words_compared, "words_exact": words_exact,
            "rows_exact_from_derived_states": from_states_exact,
            "blocks_total": blocks_total, "nonfinite_rows": nonfinite,
            "derived_words_per_block": 1 + 4 + Dh, "derived_bytes_per_block": 4 * (1 + 4 + Dh),
            "first_mismatches": mism[:10]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--commit", type=Path, default=None, help="extracted commit store (default: extract the bundle)")
    ap.add_argument("--request", default=None, help="request id (default: the bundle's)")
    ap.add_argument("--backend", choices=["numpy", "torch", "torch-cuda"], default="numpy")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--classes", default="rmsnorm_triton,rmsnorm_fused_v2,attention,silu_mul")
    a = ap.parse_args(argv)
    C = B0_CONFIG
    ev = Evaluators(a.backend)
    with tempfile.TemporaryDirectory() as td:
        commit = a.commit or SW.extract_bundle(SW.BUNDLE, Path(td))
        leaves = SW.load_store_leaves(commit, a.request or SW.REQUEST)
        norms = SW.load_norm_weights(commit)
    report = {"record": str(a.commit or SW.BUNDLE), "request": a.request or SW.REQUEST,
              "backend": a.backend, "classes": {}}
    fns = {"rmsnorm_triton": lambda: check_triton_norm(ev, leaves, norms, C),
           "rmsnorm_fused_v2": lambda: check_fused_norm(ev, leaves, norms, C),
           "attention": lambda: check_attention(ev, leaves, C),
           "silu_mul": lambda: check_silu(ev, leaves, C)}
    for name in a.classes.split(","):
        t0 = time.perf_counter()
        report["classes"][name] = fns[name]()
        report["classes"][name]["wall_s"] = round(time.perf_counter() - t0, 3)
        r = report["classes"][name]
        print(f"{name}: rows {r['rows_exact']}/{r['rows_checked']} exact, "
              f"words {r['words_exact']}/{r['words_compared']}, {r['wall_s']}s")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
