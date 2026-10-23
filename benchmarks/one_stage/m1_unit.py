"""M1: one step family's unit through C-Flock, timed: the Definition a Program row names (its ``spec``) as M0's `UnitStager`
statement on the serving rows' ports (an input row per parameter and an output row per returned field, each a
``hm96-sha512/row/v2`` row of its u16 words), flat (`ir_lower`) or structural (``--structural``: `struct_lower`'s template,
the flat unit never built), and with ``--bin`` and ``--lean`` one round of ``--n`` units staged on finite bf16 values (every
parameter of a bf16 family; a stand-in for the served values, which only the window reads), proved by M0 under ``--zk``
(`a0.session`: serve and prove, the verifier's subset draw of all of them) and verified by the Lean verifier of record
(`a0.lean_verify`).  A coordinate template (`serving_rows.COORDS`, the expert-indexed pair GEMMs at N = 1) with ``--call-n N``
stages instead the ``--units`` of a synthetic Call at N read through the serving rows' split (`split_call`), each held to the
Call's output word by the numpy twin before and after staging.  A K-chunk of a w13 coordinate
(`MoeExpertFp8CoordinateChunk`) with ``--chunk-k K`` stages instead the ``--units`` chunks of a synthetic coordinate at K, each
chunk's ``acc`` the last one's output by the word Definition, the last chunk's word held to the twin of the whole coordinate.

    PYTHONPATH=<tree>:<tree>/verity/ml/flock/python:<tree>/integrations/vllm \\
      python benchmarks/one_stage/m1_unit.py --spec 'RMSNormFusedCudaVector8_v1{N=6144,EPS=...,BLOCK=256}' --out DIR \\
      [--structural iv/hybrid/4] [--bin flock-circuit --lean flock-verify] [--n 1]

Prints ``M1U {json}`` per phase: the build's seconds and peak RSS, the unit's ANDs and rows, k_log, vus_per_block and
chunk_cap (`a6.chunk_cap`'s rule), then the stage, the session and Lean's verdict.
"""
from __future__ import annotations

import argparse
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


def bf16_rows(rng: np.random.Generator, n: int, words: int) -> np.ndarray:
    """``n`` rows of ``words`` finite bf16 values, N(0, 1) rounded to bf16 by truncation."""
    f = rng.standard_normal((n, words)).astype(np.float32)
    return (f.view(np.uint32) >> 16).astype(np.uint16)


def split_call(tp, n_call: int, rng: np.random.Generator) -> dict:
    """A synthetic Call of a coordinate template's family (`serving_rows.COORDS`) at N = ``n_call``, its operands committed
    members and registered weights (finite e4m3 codes, positive f32 scales, a routed id in [0, E), a routed weight in (0, 1)),
    read through the serving rows' split as a window reads it (`serving_rows.members` and `read`).  Returns the units' rows by
    port, their output words (the Call's, as committed), the Call's output words by the numpy twin
    (`fp8_moe_rows.moe_expert_fp8_block_row`) and the IR reference of the coordinate on every unit's rows."""
    from verity.ml.circuits.codec import descriptor_id
    from verity.ml.circuits.evaluation import evaluate_batch
    from verity_vllm.pipeline import serving_rows as SRW
    from verity_vllm.program.kernels import fp8_moe_rows as FR
    from verity_vllm.query.program_view import spec_base, spec_statics

    S = spec_statics(tp.word)
    E, K, G, KB, N = int(S["E"]), int(S["K"]), int(S["G"]), int(S["KB"]), n_call
    NB, KG = -(-N // G), K // G
    routed = spec_base(tp.word) == "MoeExpertFp8GemmW_v1"
    codes = lambda n: (lambda c: np.where((c & 0x7F) == 0x7F, c ^ 1, c).astype(np.uint8))(rng.integers(0, 256, n, dtype=np.uint8))  # noqa: E731
    scales = lambda n, lo, hi: ((2.0 ** rng.integers(lo, hi, n)) * rng.uniform(1, 2, n)).astype(np.float32)  # noqa: E731
    acts = {"xq": codes(K), "sx": scales(KG, -6, 0), "e": np.array([rng.integers(0, E)], dtype=np.int32),
            **({"g": rng.uniform(0, 1, 1).astype(np.float32)} if routed else {})}
    wts = {"w": codes(E * N * K), "ws": scales(E * NB * KG, -12, -6)}
    e = int(acts["e"][0])
    y, nonfin = FR.moe_expert_fp8_block_row(acts["xq"], acts["sx"].view(np.uint32), wts["w"].reshape(E, N, K)[e],
                                            wts["ws"].reshape(E, NB, KG)[e].view(np.uint32), G=G, KB=KB, swap=S["SWAP"] in (True, "true"),
                                            g=int(acts["g"].view(np.uint32)[0]) if routed else None, dot=FR.TWIN_DOT_OF[S["DOT"]])
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
    d = {"statics": {"E": E, "K": K, "N": N, "G": G, "KB": KB}, "op_path": "model.layers.3.mlp.experts", "members": [("out", N, "bf16")],
         "row": 0, "step": 16}
    row = ("r0", Index(), r, d)
    (m,) = SRW.members([tp.id], [[row]], [N])
    data, entries = SRW.read(m, [row], Store(), lambda f, lo, hi: wts[f][lo:hi])
    leaves = [np.ascontiguousarray(data["rows"][p], dtype="<u2").view(np.uint8 if b == 8 else np.dtype(f"<u{b // 8}")).reshape(N, -1)
              .astype(np.uint64) for (p, _w), b in zip(m.rows, tp.packed)]
    word = SRW._bind(tp.word)
    ref = np.asarray(evaluate_batch(word, *leaves)).reshape(N).astype(np.uint16)
    return {"call": spec, "member": m, "rows": data["rows"], "outs": data["outs"][m.outs[0][0]].reshape(N), "twin": y,
            "nonfinite": int(nonfin.sum()), "reference": ref, "entries": entries, "expert": e}


def _packed(a: np.ndarray, bits: int) -> np.ndarray:
    """A port's elements as its row holds them: their little-endian bytes in u16 words (`serving_rows`' packed ports)."""
    return np.ascontiguousarray(np.asarray(a).reshape(-1).astype(np.dtype(f"<u{bits // 8}"))).view("<u2")


def chunk_call(tp, k_call: int, rng: np.random.Generator) -> dict:
    """A synthetic w13 coordinate at K = ``k_call`` (finite e4m3 codes, positive f32 scales, a routed id in [0, E)) cut into
    `MoeExpertFp8CoordinateChunk_v1` chunks of the unit's K, chunk c reading the coordinate's columns of its weights and chunk c - 1's
    output as its ``acc``: every chunk's rows and output word by the word Definition in order, and the last one's word against the
    numpy twin of the whole coordinate (`fp8_moe_rows.moe_expert_fp8_block_row`) and `MoeExpertFp8Gemm_v1{N=1}` at K = ``k_call``."""
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.evaluation import evaluate_batch
    from verity_vllm.pipeline import serving_rows as SRW
    from verity_vllm.program.kernels import fp8_moe_rows as FR
    from verity_vllm.program.registry import fp8_moe_experts as X
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--structural", default=None, metavar="LAYOUT", help="the structural stager at this layout mode (iv/hybrid/4)")
    ap.add_argument("--bin", default=None, help="M0's flock-circuit: prove one round")
    ap.add_argument("--lean", default=None, help="the Lean verifier of record (flock-verify)")
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--port", type=int, default=7611)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--call-n", type=int, default=None, metavar="N",
                    help="a coordinate template (serving_rows.COORDS): stage units of a synthetic Call at this N, read through the split")
    ap.add_argument("--chunk-k", type=int, default=None, metavar="K",
                    help="a MoeExpertFp8CoordinateChunk unit: stage chunks of a synthetic coordinate at this K, cut at the unit's K")
    ap.add_argument("--units", default=None,
                    help="with --call-n: the coordinates staged; with --chunk-k: the chunks (comma-separated); else the first --n")
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
        st = CS.UnitStager(fn, f"m1/{K.family(fn.id)}", rows, K.out_rows(n_out), circuit=d / "circuit.txt",
                           **({"layout": a.structural, "structural": True} if a.structural else {}))
    m, c = st.meta, st.low.layout.counts
    cap = int(m["vus_per_block"]) << (M_MAX - int(m["k_log"])) if M_MAX - int(m["k_log"]) >= 3 else 0
    say("built", seconds=round(time.time() - t, 1), structural=a.structural, ands=c["ands"], unit_rows=c.get("rows"),
        k_log=m["k_log"], unit_log=m["unit_log"], vus_per_block=m["vus_per_block"], chunk_cap=cap, pin=st.pin,
        circuit_bytes=(d / "circuit.txt").stat().st_size, domain=st.low.domain)
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
    else:
        vals = {r: bf16_rows(rng, a.n, b // 16) for r, _s, b, _p in rows}
    t = time.time()
    with K.no_cyclic_gc():
        got = st.stage(vals, d, os.urandom(32))
    for f in ("inst", "pub"):
        (d / f"{f}.bin").rename(d / f"{f}-{a.n}.bin")
    IL.write_tables(d / "tables")
    calls = bool((got["outs"]["o0"].reshape(-1) == sc["twin"][units]).all()) if a.call_n else None
    if a.chunk_k:
        calls = bool(np.array_equal(np.asarray(got["outs"]["o0"]).reshape(a.n, -1), want))
    say("staged", seconds=round(time.time() - t, 1), n=a.n, ands=got["ands"],
        **({"outs_are_calls": calls} if a.call_n or a.chunk_k else {}))
    if calls is False:
        return 1

    t = time.time()
    record, live = a0.session(a.bin, {"dir": d, "pin": st.pin}, a.n, f"subset:{a.n}", a.port, out / "session", zk=True)
    say("proved", seconds=round(time.time() - t, 1), accepted=(live or {}).get("accepted"), why=(live or {}).get("why"),
        **{k: (live or {}).get(k) for k in ("k_log", "nbl", "dense_m", "prove_total_s", "witness_s", "proof_bytes", "units")},
        verdict=(record or {}).get("verdict"))
    v = a0.lean_verify(a.lean, {"dir": d}, a.n, out / "session", zk=True)
    say("verified", **v)
    return 0 if (live or {}).get("accepted") and v["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
