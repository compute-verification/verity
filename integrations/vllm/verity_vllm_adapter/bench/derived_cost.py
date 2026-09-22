"""Acquisition cost of derived boundaries over one RU (lane VU-ACQUIRE): seconds and bytes to
produce-and-hash each derived boundary class from the committed operand leaves, per backend.

For each class the pipeline is exactly what a derived-emitting replay would add on top of a plain replay:
run the bit-exact evaluator (`derived_rows` / `derived_rows_torch`) over every instance of the RU, bring
the derived words to the host, and hash one extra leaf per (instance, name) in the commit framing
(`verity_vllm.hashing.leaf_hash`).  Reported per class: produce seconds, hash seconds, derived bytes and
leaf count, normalised per layer per token.  Two baselines for the comparison "vs plain replay of the same
RU (no derived emission)":

  * ``committed_leaf_hash`` — hashing the RU's committed leaves from the store (the hash side of a plain
    replay commit, same machine);
  * the plain replay wall time itself is a GPU/vLLM measurement made by `replay.py --mode reference`
    (pod); its `replay.json` is quoted next to this table in docs/vllm-poc/derived-boundaries.md.

    PYTHONPATH=. python -m verity_vllm_adapter.bench.derived_cost \
        --commit DIR --request RID [--backends numpy,torch-cuda] [--repeat 3] [--out cost.json]
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from verity_vllm import hashing  # noqa: E402
from verity_vllm.registry import rmsnorm_fused_sweep as SW  # noqa: E402
from verity_vllm.registry.b1 import B0_CONFIG  # noqa: E402
from verity_vllm_adapter.derived_b0_check import (Evaluators, attention_inputs, fused_norm_rows,  # noqa: E402
                                                  phase_map, _leaf)


def _hash_leaves(items) -> tuple[int, int]:
    """items: iterable of (canonical_id, dtype, shape, bytes). Returns (count, total bytes)."""
    n = b = 0
    for cid, dt, shape, val in items:
        hashing.leaf_hash(cid, dt, shape, val)
        n += 1
        b += len(val)
    return n, b


def _f32_bytes(a) -> bytes:
    return np.ascontiguousarray(np.asarray(a, dtype=np.uint32)).view(np.float32).tobytes()


def _bf16_bytes(a) -> bytes:
    return np.ascontiguousarray(np.asarray(a, dtype=np.uint16)).tobytes()


class ClassBench:
    """One derived class: assemble operands once; produce() and hash() timed separately."""

    def __init__(self, name, produce, leaf_items):
        self.name, self.produce, self.leaf_items = name, produce, leaf_items


def build_benches(ev: Evaluators, leaves, norms, C):
    NL, NH, KVH, Dh, I = C["NL"], C["NH"], C["KVH"], C["D"], C["I"]
    G = NH // KVH
    ph = phase_map(leaves)
    ts = sorted(ph)
    benches = []

    # rmsnorm_triton (layer-0 input norm): derived inv, one f32 per position
    x0 = np.stack([_leaf(leaves, ph, t, "embed", "embed_tokens.embedding", 0) for t in ts])
    w0 = norms["model.layers.0.input_layernorm.weight"]

    def triton_produce():
        return ev.rmsnorm_triton_rows(x0, w0, C["EPS"])

    def triton_items(r):
        inv = np.asarray(r["inv"], dtype=np.uint32)
        for i, t in enumerate(ts):
            yield (f"t={t}/ph={ph[t]}/l=0/op=layers.0.input_layernorm.rmsnorm/derived=inv",
                   "float32", (1,), _f32_bytes(inv[i:i + 1]))

    benches.append(ClassBench("rmsnorm_triton.inv", triton_produce, triton_items))

    # rmsnorm_fused_v2: derived (var, s) per fused-norm instance
    rows = fused_norm_rows(leaves, norms, C)
    FX = np.stack([r[1] for r in rows]); FR = np.stack([r[2] for r in rows]); FW = np.stack([r[3] for r in rows])
    flabels = [r[0] for r in rows]

    def fused_produce():
        return ev.rmsnorm_fused_v2_rows(FX, FR, FW, C["EPS"])

    def fused_items(r):
        var = np.asarray(r["var"], dtype=np.uint32)
        s = np.asarray(r["s"], dtype=np.uint32)
        for i, lab in enumerate(flabels):
            yield (f"{lab}/derived=var", "float32", (1,), _f32_bytes(var[i:i + 1]))
            yield (f"{lab}/derived=s", "float32", (1,), _f32_bytes(s[i:i + 1]))

    benches.append(ClassBench("rmsnorm_fused_v2.var_s", fused_produce, fused_items))

    # silu_mul: derived bf16 mid per (t, L) instance
    gu = np.stack([_leaf(leaves, ph, t, L, f"layers.{L}.mlp.gate_up_proj.gemm", 0)
                   for t in ts for L in range(NL)])

    def silu_produce():
        return ev.silu_mul_rows(gu)

    def silu_items(r):
        mid = np.asarray(r["mid"], dtype=np.uint16)
        i = 0
        for t in ts:
            for L in range(NL):
                yield (f"t={t}/ph={ph[t]}/l={L}/op=layers.{L}.mlp.act_fn.act_mul/derived=silu_mid",
                       "bfloat16", (I,), _bf16_bytes(mid[i]))
                i += 1

    benches.append(ClassBench("silu_mul.mid", silu_produce, silu_items))

    # attention: derived per-block (m, l, o) per (t, L) instance, all heads
    _, _, kcache, vcache = attention_inputs(leaves, C)
    qs = {t: np.stack([_leaf(leaves, ph, t, L, f"layers.{L}.self_attn.rotary_emb.rope", 0).reshape(NH, Dh)[h]
                       for L in range(NL) for h in range(NH)]) for t in ts}

    def attn_produce():
        out = {}
        for t in ts:
            T = t + 1
            k = np.stack([kcache[L, :T, h // G] for L in range(NL) for h in range(NH)])
            v = np.stack([vcache[L, :T, h // G] for L in range(NL) for h in range(NH)])
            out[t] = ev.attention_head_states(qs[t], k, v)
        return out

    def attn_items(out):
        for t in ts:
            r = out[t]
            m = np.asarray(r["m"], dtype=np.uint32); l = np.asarray(r["l"], dtype=np.uint32)
            o = np.asarray(r["o"], dtype=np.uint32)
            nb = m.shape[1]
            for L in range(NL):
                sl = slice(L * NH, (L + 1) * NH)
                base = f"t={t}/ph={ph[t]}/l={L}/op=layers.{L}.self_attn.attn.attn"
                yield (f"{base}/derived=fa2.m", "float32", (NH, nb), _f32_bytes(m[sl]))
                yield (f"{base}/derived=fa2.l", "float32", (NH, nb, 4), _f32_bytes(l[sl]))
                yield (f"{base}/derived=fa2.o", "float32", (NH, nb, Dh), _f32_bytes(o[sl]))

    benches.append(ClassBench("attention.m_l_o", attn_produce, attn_items))
    return benches, ts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--commit", type=Path, required=True)
    ap.add_argument("--request", default=None)
    ap.add_argument("--backends", default="numpy")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    C = B0_CONFIG
    leaves = SW.load_store_leaves(a.commit, a.request or SW.REQUEST)
    norms = SW.load_norm_weights(a.commit)
    ntok = len({int(k.split("/")[0]) for k in leaves})
    NL = C["NL"]
    gpu = None
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
    except Exception:
        pass
    report = {"commit": str(a.commit), "request": a.request or SW.REQUEST, "tokens": ntok, "layers": NL,
              "host": {"platform": platform.platform(), "machine": platform.machine(),
                       "python": sys.version.split()[0], "gpu": gpu},
              "repeat": a.repeat, "backends": {}}

    # baseline: hash every committed leaf of the RU (the plain-replay hash side, same framing)
    t0 = time.perf_counter()
    n = b = 0
    for key, arr in leaves.items():
        dt = "bfloat16" if arr.dtype == np.uint16 else "float32"
        hashing.leaf_hash(key, dt, (arr.size,), arr.tobytes())
        n += 1
        b += arr.nbytes
    report["committed_leaf_hash"] = {"leaves": n, "bytes": b, "wall_s": round(time.perf_counter() - t0, 4)}

    for backend in a.backends.split(","):
        ev = Evaluators(backend)
        benches, _ = build_benches(ev, leaves, norms, C)
        rep = {}
        for bench in benches:
            r = bench.produce()  # warmup (tables, jit caches, cuda context)
            produce, hsh = [], []
            for _ in range(a.repeat):
                t0 = time.perf_counter()
                r = bench.produce()
                produce.append(time.perf_counter() - t0)
                t0 = time.perf_counter()
                nl, nb = _hash_leaves(bench.leaf_items(r))
                hsh.append(time.perf_counter() - t0)
            per = NL * ntok
            rep[bench.name] = {
                "produce_s": round(min(produce), 4), "hash_s": round(min(hsh), 4),
                "leaves": nl, "bytes": nb,
                "produce_s_per_layer_token": round(min(produce) / per, 8),
                "hash_s_per_layer_token": round(min(hsh) / per, 8),
                "bytes_per_layer_token": round(nb / per, 1),
            }
            print(f"[{backend}] {bench.name}: produce {rep[bench.name]['produce_s']}s "
                  f"hash {rep[bench.name]['hash_s']}s bytes {nb} leaves {nl}")
        report["backends"][backend] = rep
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
