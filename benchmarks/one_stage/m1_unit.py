"""M1: one step family's unit through Flock, timed: the Definition a Program row names (its ``spec``) as M0's `UnitStager`
statement on the serving rows' ports (an input row per parameter and an output row per returned field, each a
``hm96-sha512/row/v2`` row of its u16 words), flat (`ir_lower`; ``--reads slots``: its MUFU reads in lookup slots,
`read_slots`) or structural (``--structural``: `struct_lower`'s template, the flat unit never built), and with ``--bin`` and ``--lean`` one round of ``--n`` units staged on finite bf16 values (every
parameter of a bf16 family; a stand-in for the served values, which only the window reads), proved by M0 under ``--zk``
(`a0.session`: serve and prove, the verifier's subset draw of all of them) and verified by the Lean verifier of record
(`a0.lean_verify`).  A coordinate template (`serving_rows.COORDS`: the expert-indexed pair GEMMs and FlashInfer's FP8 linear at
N = 1) with ``--call-n N``
stages instead the ``--units`` of a synthetic Call at N read through the serving rows' split (`split_call`), each held to the
Call's output word by the numpy twin before and after staging.  A K-chunk of a w13 coordinate
(`MoeExpertFp8CoordinateChunk`) with ``--chunk-k K`` stages instead the ``--units`` chunks of a synthetic coordinate at K, each
chunk's ``acc`` the last one's output by the word Definition, the last chunk's word held to the twin of the whole coordinate.
A column-block template (`serving_rows.COLS`: MoeSum, the MNNVL all-reduce and torch's multimem all-reduce at a width w) with
``--call-w W`` stages instead the ``--units`` blocks of a synthetic row W wide read through the split (`cols_call`), each held to
its columns of the row's output by the numpy twin of the whole row before and after staging.  A part template
(`serving_rows.PARTS`: the fused MNNVL all-reduce + RMSNorm's all-reduce and residual add, in column blocks, and its norm;
DeepGEMM's FP8 linear's quantisation, whole, and its GEMM, in coordinates) with ``--row SPEC`` stages instead the ``--units``
of a synthetic row of SPEC read through the split (`parts_call`), each held to the row's outputs it returns by the numpy twin
of the whole row before and after staging.  A slice template (`serving_rows.SLICES`: the router GEMM, the LM head, cuBLAS's
split-K at a bf16 or an f32 output and torch.bmm at their coordinates, DSA's logits per key, the embedding shard per column,
the all-gather per shard) with ``--row SPEC`` likewise stages the ``--units`` of a synthetic row of SPEC read through the split
(`slice_call`).
A V-chunk of greedy token selection (`b1.TokenSelectChunk_v1`) or its combine (`TokenSelectCombine_v1`) with ``--vchunks V``
stages the ``--units`` chunks, or the combine, of a synthetic logits row of V (`vchunk_call`), the combine's token held to
`TokenSelect_v1{V}` on the whole row and each staged output to its word Definition's.

    PYTHONPATH=<tree>:<tree>/verity/ml/flock/python:<tree>/integrations/vllm \\
      python benchmarks/one_stage/m1_unit.py --spec 'RMSNormFusedCudaVector8_v1{N=6144,EPS=...,BLOCK=256}' --out DIR \\
      [--structural iv/hybrid/4 | --reads slots] [--bin flock-circuit --lean flock-verify] [--n 1]

Prints ``M1U {json}`` per phase: the build's seconds and peak RSS, the unit's ANDs and rows, k_log, vus_per_block and
chunk_cap (`a6.chunk_cap`'s rule), then the stage, the session and Lean's verdict.
"""
from __future__ import annotations

import argparse
import ctypes
import gc
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.setrecursionlimit(100000)

M_MAX = 35


def say(phase: str, **kw) -> None:
    print("M1U " + json.dumps({"phase": phase, "rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2), **kw},
                              default=str), flush=True)


def release() -> float:
    """Free what the stager held before the prover starts beside this process (at served width its lowering is most of the
    build's peak, which the two would otherwise hold at once), and return this process's resident GB after."""
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass
    return round(int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**30, 2)


def bf16_rows(rng: np.random.Generator, n: int, words: int) -> np.ndarray:
    """``n`` rows of ``words`` finite bf16 values, N(0, 1) rounded to bf16 by truncation."""
    f = rng.standard_normal((n, words)).astype(np.float32)
    return (f.view(np.uint32) >> 16).astype(np.uint16)


def split_call(tp, n_call: int, rng: np.random.Generator) -> dict:
    """A synthetic Call of a coordinate template's family (`serving_rows.COORDS`) at N = ``n_call``, its operands committed
    members and registered weights (finite e4m3 codes, positive f32 scales, a routed id in [0, E), a routed weight in (0, 1);
    FlashInfer's linear: a finite bf16 row and powers of two as GLM-5.3's requantised block scales), read through the serving
    rows' split as a window reads it (`serving_rows.members` and `read`).  Returns the units' rows by port, their output words
    (the Call's, as committed), the Call's output words by the numpy twin (`fp8_moe_rows.moe_expert_fp8_block_row`,
    `deepgemm_twins.flashinfer_linear_row`) and the IR reference of the coordinate on every unit's rows."""
    from verity.ml.circuits.codec import descriptor_id
    from verity.ml.circuits.evaluation import evaluate_batch
    from verity_vllm.pipeline import serving_rows as SRW
    from verity.ml.definitions.numerics import deepgemm_twins as DT
    from verity.ml.definitions.numerics import fp8_moe_rows as FR
    from verity_vllm.query.program_view import spec_base, spec_statics

    S = {**SRW.COORDS[spec_base(tp.word)], **spec_statics(tp.word)}
    E, K, G, N = int(S["E"]), int(S["K"]), int(S["G"]), n_call
    NB, KG = -(-N // G), K // G
    routed = spec_base(tp.word) == "MoeExpertFp8GemmW_v1"
    codes = lambda n: (lambda c: np.where((c & 0x7F) == 0x7F, c ^ 1, c).astype(np.uint8))(rng.integers(0, 256, n, dtype=np.uint8))  # noqa: E731
    scales = lambda n, lo, hi: ((2.0 ** rng.integers(lo, hi, n)) * rng.uniform(1, 2, n)).astype(np.float32)  # noqa: E731
    if spec_base(tp.word) == "FlashInferFp8Linear_v1":
        acts, e = {"x": bf16_rows(rng, 1, K)[0]}, 0
        wts = {"w": codes(N * K), "ws": DT.upcast_e8m0(rng.integers(112, 124, NB * KG))}
        y, nonfin = DT.flashinfer_linear_row(acts["x"], wts["w"].reshape(N, K), wts["ws"].reshape(NB, KG))
        statics, op = {"K": K, "N": N}, "model.layers.3.self_attn.o_proj"
    else:
        acts = {"xq": codes(K), "sx": scales(KG, -6, 0), "e": np.array([rng.integers(0, E)], dtype=np.int32),
                **({"g": rng.uniform(0, 1, 1).astype(np.float32)} if routed else {})}
        wts = {"w": codes(E * N * K), "ws": scales(E * NB * KG, -12, -6)}
        e = int(acts["e"][0])
        y, nonfin = FR.moe_expert_fp8_block_row(acts["xq"], acts["sx"].view(np.uint32), wts["w"].reshape(E, N, K)[e],
                                                wts["ws"].reshape(E, NB, KG)[e].view(np.uint32), G=G, KB=int(S["KB"]),
                                                swap=S["SWAP"] in (True, "true"), g=int(acts["g"].view(np.uint32)[0]) if routed else None,
                                                dot=FR.TWIN_DOT_OF[S["DOT"]])
        statics, op = {"E": E, "K": K, "N": N, "G": G, "KB": int(S["KB"])}, "model.layers.3.mlp.experts"
    at = np.cumsum([0, wts["w"].size])
    names = list(acts)

    class Index:
        pnames = ["weights"]

        def member_at(self, a, lo, hi):
            nm = names[a]
            return {"members": [(nm, acts[nm].size, str(acts[nm].dtype))], "step": 16, "op_path": f"m1.{nm}", "row": 0}, nm, lo, hi, None

        def weight_slice(self, lo, hi):
            f = int(lo >= at[1])
            return ("w", "ws")[f], lo - at[f], hi - at[f], None

    class Store:
        def words(self, rid, step, op, member, lo, hi):
            return (y if member == "out" else acts[member])[lo:hi], None

    spec = descriptor_id(SRW._bind(tp.word, N=N))
    r = {"i": 0, "spec": spec, "args": [[["n", k, 0, acts[nm].size]] for k, nm in enumerate(names)]
         + [[["p", 0, int(at[0]), int(at[1])]], [["p", 0, int(at[1]), int(at[1]) + wts["ws"].size]]]}
    d = {"statics": statics, "op_path": op, "members": [("out", N, "bf16")], "row": 0, "step": 16}
    row = ("r0", Index(), r, d)
    (m,) = SRW.members([tp.id], [[row]], [N])
    data, entries = SRW.read(m, [row], Store(), lambda f, lo, hi: wts[f][lo:hi])
    leaves = [np.ascontiguousarray(data["rows"][p], dtype="<u2").view(np.uint8 if b == 8 else np.dtype(f"<u{b // 8}")).reshape(N, -1)
              .astype(np.uint64) for (p, _w), b in zip(m.rows, tp.packed)]
    word = SRW._bind(tp.word)
    ref = np.asarray(evaluate_batch(word, *leaves)).reshape(N).astype(np.uint16)
    return {"call": spec, "member": m, "rows": data["rows"], "outs": data["outs"][m.outs[0][0]].reshape(N), "twin": y,
            "nonfinite": int(nonfin.sum()), "reference": ref, "entries": entries, "expert": e}


def cols_call(tp, w_call: int, rng: np.random.Generator) -> dict:
    """A synthetic row of a column-block template's family (`serving_rows.COLS`) ``w_call`` wide on finite bf16 values, its operand
    a committed member, read through the serving rows' split as a window reads it (`serving_rows.members` and `read`).  Returns
    the units' rows by port, their outputs (the row's, as committed), the row's output by the numpy twin of the whole row
    (`rows.moe_sum`, `tp_allreduce_check.mnnvl_rows`, `tp_allreduce_check.multimem_rows` through the row's own owners' trees), and
    the IR reference of the template's word on every unit's rows."""
    from verity.ml.circuits.codec import descriptor_id
    from verity.ml.circuits.evaluation import evaluate_batch
    from verity_vllm.pipeline import serving_rows as SRW
    from verity.ml.definitions.numerics import rows as KR
    from verity.ml.definitions.registry import tp_allreduce as TA
    from verity.ml.definitions.registry import tp_allreduce_check as TAC
    from verity_vllm.query.program_view import spec_base, spec_statics

    axis, S = SRW.COLS[spec_base(tp.word)], spec_statics(tp.word)
    lead = int(S["TOPK"] if axis == "H" else TA.ranks(S["WORLD"]))
    x = bf16_rows(rng, lead, w_call)
    y = (KR.moe_sum({"TOPK": lead, "H": w_call}, [x])["sum"] if axis == "H"
         else TAC.multimem_rows(x, S["TREES"], S["FTZ"] in (True, "true")) if spec_base(tp.word) == "AllReduceMultimem_v1"
         else TAC.mnnvl_rows(x, int(S["SHOT"])))
    statics = {**S, axis: w_call}
    op = "model.layers.3.mlp.experts" if axis == "H" else "model.layers.3.mlp"

    class Index:
        pnames: list[str] = []

        def member_at(self, a, lo, hi):
            return {"members": [("x", x.size, "bf16")], "step": 16, "op_path": "m1.x", "row": 0}, "x", lo, hi, None

    class Store:
        def words(self, rid, step, op, member, lo, hi):
            return (y if member == "out" else x.reshape(-1))[lo:hi], None

    spec = descriptor_id(SRW._bind(tp.word, **{axis: w_call}))
    r = {"i": 0, "spec": spec, "args": [[["n", 0, 0, x.size]]]}
    d = {"statics": statics, "op_path": op, "members": [("out", w_call, "bf16")], "row": 0, "step": 16}
    row = ("r0", Index(), r, d)
    nh = w_call // int(S[axis])
    (m,) = SRW.members([tp.id], [[row]], [nh])
    data, entries = SRW.read(m, [row], Store(), None)
    ref = np.asarray(evaluate_batch(SRW._bind(tp.word), data["rows"]["in0"].astype(np.uint64))).reshape(nh, -1).astype(np.uint16)
    return {"call": spec, "member": m, "rows": data["rows"], "outs": data["outs"][m.outs[0][0]], "twin": np.asarray(y).reshape(nh, -1),
            "reference": ref, "entries": entries}


def parts_call(tp, row: str, rng: np.random.Generator) -> dict:
    """A synthetic row of ``row`` (a descriptor id of a `serving_rows.PARTS` family: the fused MNNVL all-reduce + RMSNorm, or
    DeepGEMM's FP8 linear) on finite values, its activations committed members and its weights registered weights (DeepGEMM's:
    finite e4m3 codes and powers of two as GLM-5.3's requantised block scales), read through the split for the part template
    ``tp`` (a ``COLS`` part in column blocks, a ``COORDS`` part in coordinates) as a window reads it.  Returns the units' rows by
    port, their outputs (the row's, as committed; every output port's words in order), those outputs by the numpy twin of the
    whole row (`tp_allreduce_check.mnnvl_norm_rows`, `deepgemm_twins.deepgemm_linear_row`), and the IR reference of the
    template's word on every unit's rows."""
    from verity.ml.circuits.codec import descriptor_id
    from verity_vllm.pipeline import serving_rows as SRW
    from verity.ml.definitions.numerics import deepgemm_twins as DT
    from verity.ml.definitions.registry import tp_allreduce_check as TAC
    from verity.ml.definitions.registry.fp8 import FP8_BLOCK
    from verity_vllm.query.program_view import spec_base, spec_statics

    S, p = spec_statics(row), SRW.part(tp.word, row)
    if p is None or descriptor_id(SRW._bind(row)) != row:
        raise ValueError(f"{tp.id} is no part of {row} (serving_rows.PARTS), or {row} is no descriptor id")
    if spec_base(row) == "DeepGemmFp8Linear_v1":
        K, N = int(S["K"]), int(S["N"])
        NB, KG = -(-N // FP8_BLOCK), K // FP8_BLOCK
        codes = rng.integers(0, 256, N * K, dtype=np.uint8)
        acts = {"x": bf16_rows(rng, 1, K)[0]}
        wts = {"w": np.where((codes & 0x7F) == 0x7F, codes ^ 1, codes).astype(np.uint8),
               "ws": DT.upcast_e8m0(rng.integers(112, 124, NB * KG))}
        q, s, y, _nf = DT.deepgemm_linear_row(acts["x"], wts["w"].reshape(N, K), wts["ws"].reshape(NB, KG))
        members, outs = [("x_q", K, "e4m3"), ("x_s", KG, "f32"), ("out", N, "bf16")], [q.astype(np.uint8), s.astype(np.uint32),
                                                                                      y.astype(np.uint16)]
        op, width = "model.layers.3.mlp.shared_experts.gate_up_proj", N
    else:
        world, width = int(S["WORLD"]), int(S["N"])
        parts, res, w = bf16_rows(rng, world, width), bf16_rows(rng, 1, width)[0], bf16_rows(rng, 1, width)[0]
        normed, r = TAC.mnnvl_norm_rows(parts[:, None], res[None], w, int(S["SHOT"]), int(S["CLUSTER"]), float(S["EPS"]),
                                        S["FMA"] in (True, "true"))
        acts, wts = {"parts": parts.reshape(-1), "res": res}, {"w": w}
        members, outs = [("normed", width, "bf16"), ("residual", width, "bf16")], [normed[0], r[0]]
        op = "model.layers.3.post_attention_layernorm"
    vals = {**acts, **{mm[0]: o for mm, o in zip(members, outs, strict=True)}}
    names, fields, at = list(acts), list(wts), np.cumsum([0, *(a.size for a in wts.values())])

    class Index:
        pnames = ["weights"]

        def member_at(self, a, lo, hi):
            nm = names[a]
            return {"members": [(nm, acts[nm].size, str(acts[nm].dtype))], "step": 16, "op_path": f"m1.{nm}", "row": 0}, nm, lo, hi, None

        def weight_slice(self, lo, hi):
            f = int(np.searchsorted(at, lo, side="right")) - 1
            return fields[f], lo - int(at[f]), hi - int(at[f]), None

    class Store:
        def words(self, rid, step, op, member, lo, hi):
            return vals[member][lo:hi], None

    r_ = {"i": 0, "spec": row, "args": [[["n", k, 0, acts[nm].size]] for k, nm in enumerate(names)]
          + [[["p", 0, int(at[f]), int(at[f + 1])]] for f in range(len(fields))]}
    d = {"statics": S, "op_path": op, "row": 0, "step": 16, "members": members}
    one = ("r0", Index(), r_, d)
    axis = SRW.COLS.get(spec_base(tp.word))
    n = int(S["N"]) if tp.layout == "coord" else width // int(spec_statics(tp.word)[axis]) if axis else 1
    (m,) = SRW.members([tp.id], [[one]], [n])
    data, entries = SRW.read(m, [one], Store(), lambda f, lo, hi: wts[f][lo:hi])
    packed = tp.packed or (16,) * (len(m.rows) + len(m.outs))
    twin = np.concatenate([_packed(outs[j], b).reshape(n, -1) for j, b in zip(p[2], packed[len(m.rows):], strict=True)], axis=1)
    return {"call": row, "member": m, "rows": data["rows"], "outs": np.concatenate([data["outs"][q_] for q_, _w in m.outs], axis=1),
            "twin": twin, "reference": _reference(SRW._bind(tp.word), data["rows"], m.rows, packed), "entries": entries}


def slice_call(tp, row: str, rng: np.random.Generator) -> dict:
    """A synthetic row of ``row`` (a descriptor id of a `serving_rows.SLICES` family) on finite operands (bf16 N(0, 1), e4m3 codes
    but NaN, f32 in [0.5, 2), the embedding shard's token inside its shard), each a committed member, read through the split for
    the slice template ``tp`` as a window reads it.  Returns the units' rows by port, their outputs (the row's, as committed),
    those outputs by the row's twin (`glm53_rows.ROWS`' kernel where the family has one, cuBLAS's split-K's
    `cublas_gemm_rows.ROWS`, the embedding shard's table row by index, else the row's word Definition by `evaluate_batch`), and
    the IR reference of the template's word on every unit's rows, each as u16 words."""
    from verity.ml.circuits.codec import descriptor_id
    from verity.ml.circuits.evaluation import evaluate_batch
    from verity_vllm.pipeline import serving_rows as SRW
    from verity_vllm.pipeline import slice_rows as SL
    from verity.ml.definitions.numerics import cublas_gemm_rows as CGR
    from verity.ml.definitions.numerics import glm53_rows as GR
    from verity_vllm.query.program_view import spec_base, spec_statics

    fn, S = SRW._bind(row), spec_statics(row)
    if spec_base(row) not in SRW.SLICES or fn is None or descriptor_id(fn) != row:
        raise ValueError(f"{row} is no row of a serving_rows.SLICES family, or no descriptor id")
    params, ret = fn.signature()
    ops = []
    for nm, ty in params:
        n, w = len(ty.leaf_widths()), ty.leaf_widths()[0]
        ops.append(np.array([int(S["START"]) + int(S["VS"]) // 2], np.int32) if nm == "tok" else bf16_rows(rng, 1, n)[0] if w == 16
                   else (lambda c: np.where((c & 0x7F) == 0x7F, c ^ 1, c).astype(np.uint8))(rng.integers(0, 256, n, dtype=np.uint8))
                   if w == 8 else rng.uniform(0.5, 2, n).astype(np.float32))
    bits = lambda a: a.view(f"<u{a.itemsize}").astype(np.uint64).reshape(1, -1)  # noqa: E731
    if spec_base(row) in GR.ROWS:
        y = np.asarray(GR.ROWS[spec_base(row)][0](S, ops)["0"]).reshape(-1)
    elif spec_base(row) in CGR.ROWS:
        y = np.asarray(CGR.ROWS[spec_base(row)](S, ops)["0"]).reshape(-1)
    elif spec_base(row) == "EmbeddingShard_v1":
        t, VS = int(ops[0][0]) - int(S["START"]), int(S["VS"])
        y = ops[1].reshape(VS, -1)[t] if 0 <= t < VS else np.zeros(int(S["H"]), np.uint16)
    else:
        y = np.asarray(evaluate_batch(fn, *(bits(o) for o in ops))).reshape(-1)
    y = y.astype(np.uint16 if ret.leaf_widths()[0] == 16 else np.uint32)

    class Index:
        pnames: list[str] = []

        def member_at(self, a, lo, hi):
            return {"members": [(params[a][0], ops[a].size, str(ops[a].dtype))], "step": 16, "op_path": f"m1.{params[a][0]}",
                    "row": 0}, params[a][0], lo, hi, None

    class Store:
        def words(self, rid, step, op, member, lo, hi):
            return (y if member == "out" else ops[[nm for nm, _t in params].index(member)])[lo:hi], None

    r = {"i": 0, "spec": row, "args": [[["n", k, 0, o.size]] for k, o in enumerate(ops)]}
    d = {"statics": S, "op_path": "model.layers.3.m1", "members": [("out", y.size, str(y.dtype))], "row": 0, "step": 16}
    one = ("r0", Index(), r, d)
    n = SL.extent(tp, S)
    (m,) = SRW.members([tp.id], [[one]], [n])
    data, entries = SRW.read(m, [one], Store(), None)
    packed = tp.packed or (16,) * (len(params) + 1)
    unit = [np.ascontiguousarray(data["rows"][p], dtype="<u2").view(f"<u{b // 8}").astype(np.uint64) for (p, _w), b in zip(m.rows, packed)]
    ref = np.asarray(evaluate_batch(SRW._bind(tp.word), *unit)).reshape(n, -1).astype(f"<u{packed[-1] // 8}")
    return {"call": row, "member": m, "rows": data["rows"], "outs": data["outs"][m.outs[0][0]], "entries": entries,
            "twin": np.ascontiguousarray(y.reshape(n, -1)).view("<u2"), "reference": np.ascontiguousarray(ref).view("<u2")}


def _packed(a: np.ndarray, bits: int) -> np.ndarray:
    """A port's elements as its row holds them: their little-endian bytes in u16 words (`serving_rows`' packed ports)."""
    return np.ascontiguousarray(np.asarray(a).reshape(-1).astype(np.dtype(f"<u{bits // 8}"))).view("<u2")


def _reference(fn, rows: dict, ports, packed) -> np.ndarray:
    """Word Definition ``fn`` on every unit's rows (``ports``' u16 words as their elements of ``packed`` bits): each unit's every
    output port's elements as their words, in order."""
    from verity.ml.circuits.evaluation import evaluate_batch

    leaves = [np.ascontiguousarray(rows[p], dtype="<u2").view(np.uint8 if b == 8 else np.dtype(f"<u{b // 8}"))
              .reshape(len(rows[p]), -1).astype(np.uint64) for (p, _w), b in zip(ports, packed)]
    out = np.asarray(evaluate_batch(fn, *leaves)).reshape(len(leaves[0]), -1)
    ret = fn.signature()[1]
    cuts = np.cumsum([0, *(len(t.leaf_widths()) for t in (getattr(ret, "fields", None) or (ret,)))])
    return np.concatenate([np.stack([_packed(u[a:b], bits) for u in out]) for a, b, bits in zip(cuts, cuts[1:], packed[len(ports):])],
                          axis=1)


def chunk_call(tp, k_call: int, rng: np.random.Generator) -> dict:
    """A synthetic w13 coordinate at K = ``k_call`` (finite e4m3 codes, positive f32 scales, a routed id in [0, E)) cut into
    `MoeExpertFp8CoordinateChunk_v1` chunks of the unit's K, chunk c reading the coordinate's columns of its weights and chunk c - 1's
    output as its ``acc``: every chunk's rows and output word by the word Definition in order, and the last one's word against the
    numpy twin of the whole coordinate (`fp8_moe_rows.moe_expert_fp8_block_row`) and `MoeExpertFp8Gemm_v1{N=1}` at K = ``k_call``."""
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.evaluation import evaluate_batch
    from verity_vllm.pipeline import serving_rows as SRW
    from verity.ml.definitions.numerics import fp8_moe_rows as FR
    from verity.ml.definitions.registry import fp8_moe_experts as X
    from verity_vllm.query.program_view import spec_statics

    S = spec_statics(tp.word)
    E, KC, G, KB, swap = int(S["E"]), int(S["K"]), int(S["G"]), int(S["KB"]), S["SWAP"] in (True, "true")
    if k_call % KC or KC % G:
        raise ValueError(f"a coordinate of K = {k_call} is no whole number of chunks of {KC} whole scale blocks of {G}")
    C = k_call // KC
    codes = lambda n: (lambda c: np.where((c & 0x7F) == 0x7F, c ^ 1, c).astype(np.uint8))(rng.integers(0, 256, n, dtype=np.uint8))  # noqa: E731
    scales = lambda n, lo, hi: ((2.0 ** rng.integers(lo, hi, n)) * rng.uniform(1, 2, n)).astype(np.float32).view(np.uint32)  # noqa: E731
    xq, sx, e = codes(k_call), scales(k_call // G, -6, 0), int(rng.integers(0, E))
    slab, ws = codes(E * k_call).reshape(E, k_call), scales(E * k_call // G, -12, -6).reshape(E, k_call // G)
    y, nonfin = FR.moe_expert_fp8_block_row(xq, sx, slab[e][None], ws[e][None], G=G, KB=KB, swap=swap, dot=FR.TWIN_DOT_OF[S["DOT"]])
    chunks, acc = [], None
    for c in range(C):
        a, b = c * KC, (c + 1) * KC
        word = SRW._bind(tp.word, FIRST=c == 0, LAST=c == C - 1)
        vals = [xq[a:b], sx[a // G:b // G], np.array([e], np.uint32)] + ([] if acc is None else [np.array([acc], np.uint32)]) \
            + [slab[:, a:b], ws[:, a // G:b // G]]
        out = int(np.asarray(evaluate_batch(word, *(v.reshape(1, -1).astype(np.uint64) for v in vals))).reshape(-1)[0])
        chunks.append({"vals": vals, "out": out, "last": c == C - 1})
        acc = out
    whole = bind(X.MoeExpertFp8Gemm, E=E, K=k_call, N=1, G=G, KB=KB, SWAP=swap, DOT=SRW._bind(tp.word).bindings["DOT"])
    ref = int(np.asarray(evaluate_batch(whole, *(v.reshape(1, -1).astype(np.uint64) for v in (xq, sx, np.array([e], np.uint32), slab, ws))))
              .reshape(-1)[0])
    return {"chunks": chunks, "twin": int(y[0]), "nonfinite": int(nonfin.sum()), "reference": ref, "expert": e, "C": C}


def vchunk_call(tp, v_call: int, rng: np.random.Generator) -> dict:
    """A synthetic logits row of V = ``v_call`` (finite bf16 words, a NaN after logits[0], the row's largest word on both sides of
    the middle chunk boundary) cut into V-chunks of a `TokenSelectChunk_v1` or `TokenSelectCombine_v1` template's C (`b1.chunks`):
    each chunk's partial by `TokenSelectChunk_v1`, the combine's token from logits[0] and the partials, and the row's by
    `TokenSelect_v1{V}`, which is the first of the two largest words."""
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.evaluation import evaluate_batch
    from verity.ml.definitions.registry import b1
    from verity_vllm.query.program_view import spec_statics

    def ev(fn, *xs):
        return np.asarray(evaluate_batch(fn, *(np.asarray(x).reshape(1, -1).astype(np.uint64) for x in xs))).reshape(-1)

    C = int(spec_statics(tp.word)["C"])
    cuts = b1.chunks(v_call, C)
    x = bf16_rows(rng, 1, v_call)[0]
    x[1 + int(rng.integers(v_call - 1))] = 0x7FC0
    at = (len(cuts) // 2) * C
    x[[at - 1, at]] = 0x7F00
    parts = [ev(bind(b1.TokenSelectChunk, C=hi - lo), x[lo:hi]) for lo, hi in cuts]
    best, idx = np.array([p[0] for p in parts], np.uint16), np.array([p[1] for p in parts], np.uint32)
    token = int(ev(bind(b1.TokenSelectCombine, V=v_call, C=C), x[:1], best, idx)[0])
    return {"x": x, "cuts": cuts, "best": best, "idx": idx, "token": token, "reference": int(ev(bind(b1.TokenSelect, V=v_call), x)[0]),
            "largest": at - 1}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--structural", default=None, metavar="LAYOUT", help="the structural stager at this layout mode (iv/hybrid/4)")
    ap.add_argument("--reads", choices=("inline", "slots"), default="inline",
                    help="a flat unit's MUFU table reads: inline rows of the unit, or Flock lookup slots (read_slots)")
    ap.add_argument("--stage-ands", type=lambda v: v if v in ("auto", "none") else int(v), default="auto", metavar="N|auto|none",
                    help="with --reads slots: at most N ANDs a net, the caps whose nets take the fewest cells (auto), or no cap "
                         "(read_slots.split's stage_ands)")
    ap.add_argument("--bin", default=None, help="M0's flock-circuit: prove one round")
    ap.add_argument("--lean", default=None, help="the Lean verifier of record (flock-verify)")
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--port", type=int, default=7611)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--call-n", type=int, default=None, metavar="N",
                    help="a coordinate template (serving_rows.COORDS): stage units of a synthetic Call at this N, read through the split")
    ap.add_argument("--chunk-k", type=int, default=None, metavar="K",
                    help="a MoeExpertFp8CoordinateChunk unit: stage chunks of a synthetic coordinate at this K, cut at the unit's K")
    ap.add_argument("--call-w", type=int, default=None, metavar="W",
                    help="a column-block template (serving_rows.COLS): stage blocks of a synthetic row this wide, read through the split")
    ap.add_argument("--row", default=None, metavar="SPEC",
                    help="a part template (serving_rows.PARTS, a COORDS part in coordinates) or a slice template "
                         "(serving_rows.SLICES): stage its units of a synthetic row of this descriptor id, read through the split")
    ap.add_argument("--vchunks", type=int, default=None, metavar="V",
                    help="a TokenSelectChunk or TokenSelectCombine unit: stage it on a synthetic logits row of V cut at the unit's C")
    ap.add_argument("--units", default=None,
                    help="with --call-n: the coordinates staged; with --chunk-k or --vchunks: the chunks; with --call-w or --row: the "
                         "column blocks or parts (comma-separated); else the first --n")
    a = ap.parse_args()

    import a0
    import a11_kinds as K
    from verity_flock import class_statement as CS
    from verity_flock import ir_lower as IL
    from verity_vllm.pipeline import serving_rows as SRW

    from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as V2

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fn = SRW.template(a.spec).fn
    params, ret = fn.signature()
    rows = [(nm, V2, K.leaf_bits(t), [nm]) for nm, t in params]
    n_out = len(ret.fields) if hasattr(ret, "fields") else 1
    say("bound", spec=a.spec, word_gates=fn.gates, rows=[[r, b] for r, _s, b, _p in rows], outputs=n_out)

    d = out / "stage"
    t = time.time()
    with K.no_cyclic_gc():
        st = CS.UnitStager(fn, f"m1/{K.family(fn.id)}", rows, K.out_rows(n_out), circuit=d / "circuit.txt", reads=a.reads,
                           stage_ands=None if a.stage_ands == "none" else a.stage_ands,
                           **({"layout": a.structural, "structural": True} if a.structural else {}))
    m, c = st.meta, getattr(st.low, "counts", None) or st.low.layout.counts
    cap = int(m["vus_per_block"]) << (M_MAX - int(m["k_log"])) if M_MAX - int(m["k_log"]) >= 3 else 0
    say("built", seconds=round(time.time() - t, 1), structural=a.structural, reads=a.reads, ands=c["ands"], unit_rows=c.get("rows"),
        nnz=c.get("nnz_a", 0) + c.get("nnz_b", 0), lookups=m.get("per_vu", {}), k_log=m["k_log"], unit_log=m["unit_log"],
        vus_per_block=m["vus_per_block"], chunk_cap=cap, pin=st.pin, circuit_bytes=(d / "circuit.txt").stat().st_size,
        domain=st.low.domain, stage_ands=getattr(st.low, "caps", None))
    if not a.bin:
        return 0

    rng = np.random.default_rng(a.seed)
    if a.call_n:
        t = time.time()
        sc = split_call(SRW.template(a.spec), a.call_n, rng)
        units = [int(u) for u in a.units.split(",")] if a.units else list(range(a.n))
        a.n = len(units)
        say("split", seconds=round(time.time() - t, 1), call=sc["call"], expert=sc["expert"], units=sc["member"].n, entries=sc["entries"],
            ports=[list(p) for p in sc["member"].rows + sc["member"].outs], staged=units, twin_nonfinite=sc["nonfinite"],
            outs_are_twin=bool((sc["outs"] == sc["twin"]).all()), reference_is_twin=bool((sc["reference"] == sc["twin"]).all()))
        if not ((sc["outs"] == sc["twin"]).all() and (sc["reference"] == sc["twin"]).all()):
            return 1
        vals = {nm: sc["rows"][f"in{g}"][units] for g, (nm, _s, _b, _p) in enumerate(rows)}
    elif a.chunk_k:
        t = time.time()
        tp = SRW.template(a.spec)
        sc = chunk_call(tp, a.chunk_k, rng)
        units = [int(u) for u in a.units.split(",")] if a.units else [sc["C"] // 2]
        a.n = len(units)
        place = {c: (c == 0, c == sc["C"] - 1) for c in units}
        S = tp.fn.bindings
        last = sc["chunks"][-1]["out"]
        say("chunks", seconds=round(time.time() - t, 1), chunks=sc["C"], expert=sc["expert"], staged=units, twin_nonfinite=sc["nonfinite"],
            last_is_twin=last == sc["twin"], reference_is_twin=sc["reference"] == sc["twin"])
        if not (last == sc["twin"] and sc["reference"] == sc["twin"]) or any(p != (S["FIRST"], S["LAST"]) for p in place.values()):
            return 1
        vals = {nm: np.stack([_packed(sc["chunks"][c]["vals"][g], b) for c in units])
                for g, ((nm, _s, _b, _p), b) in enumerate(zip(rows, tp.packed))}
        want = np.stack([_packed(np.array([sc["chunks"][c]["out"]]), 16 if S["LAST"] else 32) for c in units])
    elif a.vchunks:
        t = time.time()
        tp = SRW.template(a.spec)
        sc = vchunk_call(tp, a.vchunks, rng)
        combine = "TokenSelectCombine" in tp.word
        units = [0] if combine else [int(u) for u in a.units.split(",")] if a.units else [len(sc["cuts"]) // 2]
        a.n = len(units)
        say("vchunks", seconds=round(time.time() - t, 1), chunks=len(sc["cuts"]), staged=units, token=sc["token"],
            token_is_token_select=sc["token"] == sc["reference"], largest=sc["largest"])
        if sc["token"] != sc["reference"] or sc["token"] != sc["largest"]:
            return 1
        if combine:
            vals = {"x0": _packed(sc["x"][:1], 16)[None], "best": _packed(sc["best"], 16)[None], "idx": _packed(sc["idx"], 32)[None]}
            want = _packed(np.array([sc["token"]]), 32)[None]
        else:
            C = int(tp.fn.bindings["C"])
            if any(sc["cuts"][c][1] - sc["cuts"][c][0] != C for c in units):
                return 1
            vals = {"x": np.stack([sc["x"][sc["cuts"][c][0]:sc["cuts"][c][1]] for c in units])}
            want = np.stack([np.concatenate([_packed(sc["best"][c:c + 1], 16), _packed(sc["idx"][c:c + 1], 32)]) for c in units])
    elif a.call_w or a.row:
        t = time.time()
        tp = SRW.template(a.spec)
        sc = (cols_call(tp, a.call_w, rng) if a.call_w else slice_call(tp, a.row, rng) if tp.layout == "slice"
              else parts_call(tp, a.row, rng))
        units = [int(u) for u in a.units.split(",")] if a.units else list(range(a.n))
        a.n = len(units)
        say("split", seconds=round(time.time() - t, 1), call=sc["call"], units=sc["member"].n, entries=sc["entries"],
            ports=[list(p) for p in sc["member"].rows + sc["member"].outs], staged=units,
            outs_are_twin=bool((sc["outs"] == sc["twin"]).all()), reference_is_twin=bool((sc["reference"] == sc["twin"]).all()))
        if not ((sc["outs"] == sc["twin"]).all() and (sc["reference"] == sc["twin"]).all()):
            return 1
        vals = {nm: sc["rows"][f"in{g}"][units] for g, (nm, _s, _b, _p) in enumerate(rows)}
        want = sc["twin"][units]
    else:
        vals = {r: bf16_rows(rng, a.n, b // 16) for r, _s, b, _p in rows}
    t = time.time()
    with K.no_cyclic_gc():
        got = st.stage(vals, d, os.urandom(32))
    for f in ("inst", "pub"):
        (d / f"{f}.bin").rename(d / f"{f}-{a.n}.bin")
    IL.write_tables(d / "tables")
    calls = bool((got["outs"]["o0"].reshape(-1) == sc["twin"][units]).all()) if a.call_n else None
    if a.chunk_k or a.call_w or a.row or a.vchunks:
        calls = bool(np.array_equal(np.concatenate([np.asarray(got["outs"][f"o{j}"]).reshape(a.n, -1) for j in range(n_out)], axis=1),
                                    want))
    say("staged", seconds=round(time.time() - t, 1), n=a.n, ands=got["ands"],
        **({"outs_are_calls": calls} if a.call_n or a.chunk_k or a.call_w or a.row or a.vchunks else {}))
    if calls is False:
        return 1
    pin = st.pin
    del st, got
    say("released", rss_now_gb=release())

    t = time.time()
    record, live = a0.session(a.bin, {"dir": d, "pin": pin}, a.n, f"subset:{a.n}", a.port, out / "session", zk=True)
    say("proved", seconds=round(time.time() - t, 1), accepted=(live or {}).get("accepted"), why=(live or {}).get("why"),
        **{k: (live or {}).get(k) for k in ("k_log", "nbl", "dense_m", "prove_total_s", "witness_s", "proof_bytes", "units")},
        verdict=(record or {}).get("verdict"))
    v = a0.lean_verify(a.lean, {"dir": d}, a.n, out / "session", zk=True)
    say("verified", **v)
    return 0 if (live or {}).get("accepted") and v["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
