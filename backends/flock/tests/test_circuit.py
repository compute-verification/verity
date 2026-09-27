"""verity/flock-circuit's Python side (lane flock-circuit): the composite circuit, its instance files, and the tail pieces in
the circuit. The Rust statement's own tests are ``flock-circuit selftest`` on a pod (every negative) and the crate's unit tests."""
from __future__ import annotations

import json

import numpy as np
import pytest

from verity_flock import ir_lower as IL
from verity_flock import circuit as FC
from verity_flock import tail as TL
from verity_flock.gf2 import Circuit

pytest.importorskip("verity_numerical")
from verity_numerical.bench import templates as TM  # noqa: E402
from verity_numerical.bench.generate import generate  # noqa: E402
from verity_numerical.bench.input_sets import InputSet  # noqa: E402


def _set(sid: str, n: int, tmp_path):
    summary = generate(TM.parse_subcircuit_id(sid), n, 20260926, tmp_path)
    return InputSet.open(summary["path"])


def test_rope_composes_private_by_default(tmp_path):
    s = _set("rope-head/d64/neox-bf16", 4, tmp_path)
    low = FC.lowering_for_set(s)
    text = FC.compose(low, s.subcircuit)
    meta = json.loads(text.split("\n", 2)[1][5:])
    assert meta["statement"] == FC.STATEMENT and meta["leaf_scheme"] == FC.LEAF_SCHEME
    assert not any(k in meta for k in ("tail", "cut", "public_ports", "native")), "no verifier-evaluated part"
    assert {r["net"] for r in meta["ranges"]} == {FC.SHA_NET, FC.HM_NET, "unit", "mask"}
    spans = sorted(((r["pos0"] << r["slot_log"]), ((r["pos0"] + (r["count"] if r.get("packed") else FC._pow2(r["count"]))) << r["slot_log"]),
                    bool(r.get("packed")), r["slot_log"]) for r in meta["ranges"])
    assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:])), "ranges are disjoint"
    assert all(start % ((1 << sl) if packed else end - start) == 0 for start, end, packed, sl in spans), "ranges are aligned"
    assert [r["net"] for r in meta["ranges"] if r.get("packed")] == [FC.SHA_NET], "only the compressions are packed"
    assert meta["program_digests"] and meta["partition"]["rule"] == FC.PARTITION_RULE
    FC.stage(low, s, 0, 4, tmp_path / "st")
    with pytest.raises(ValueError, match="salt key"):
        FC.stage(low, s, 0, 4, tmp_path / "v", public_only=True)
    prover, verifier = (tmp_path / "st" / "inst-4.bin").read_bytes(), (tmp_path / "st" / "pub-4.bin").read_bytes()
    hp, hv = (json.loads(b.split(b"\n", 1)[0]) for b in (prover, verifier))
    assert hp["rows"] and not hv["rows"], "the verifier's file holds no row"
    assert prover.split(b"\n", 1)[1].endswith(verifier.split(b"\n", 1)[1]), "the public part is the same bytes"
    assert hp["circuit_sha512"] == FC.digest(text) and len(FC.digest(text)) == 128
    G = meta["vus_per_block"]
    assert hp["units"]["indices"] == [0, 1, 2, 3] and hp["units"]["blocks"] == [[0, 1, 2, 3][b:b + G] for b in range(0, 4, G)]
    assert len(meta["partition"]["digest"]) == 128, "a 64-byte SHA-512 partition digest"
    # a row's public b || c is hm96-sha512 of its private row and salt
    from verity.commitments import hm96
    from verity.commitments.rowleaf import ROLE_X, sha512_row_digest

    ports = meta["ports"]
    rb, np_ = sum(2 * w for _, w, _ in ports), len(ports)
    body = prover.split(b"\n", 1)[1]
    salt0, bc0 = body[4 * rb:4 * rb + FC.SALT_BYTES], body[4 * rb + 4 * np_ * FC.SALT_BYTES:][:FC.BC_BYTES]
    words = np.frombuffer(body[:2 * ports[0][1]], dtype="<u2").tolist()
    assert bc0 == hm96.SHA512.commit_string(hm96.SHA512.default_key, sha512_row_digest(words, 16, ROLE_X), salt0)


def test_leaf_scheme_reserves_what_hm96_needs():
    import hashlib

    from verity.commitments import hm96

    t = FC.LEAF_SCHEME
    assert (t["id"], t["salt_bytes"]) == (hm96.SHA512.name, hm96.SHA512.salt_bytes)
    assert t["key_sha512"] == hashlib.sha512(hm96.SHA512.default_key).hexdigest(), "the key is the scheme's pinned constant"


def test_returned_outputs_need_a_cut_tail():
    class U:
        cut, tail_rets = [], [0]

    class Low:
        units, name = U(), "attention"

    with pytest.raises(NotImplementedError, match="no verifier-evaluated form"):
        FC.compose(Low(), None)


@pytest.mark.parametrize("prim", sorted(p for p in TL.MUFU if p not in ("MufuEx2Ftz_v1", "Fa2InvSum_v1")))
def test_mufu_pieces_match_the_ir(prim):
    from verity_vllm.program.kernels import rms_relation
    from verity_vllm.program.registry import prims as P

    t = rms_relation.tables("cuda")
    tabs = {"rsq": np.asarray(t.rsq, dtype=np.uint32), "sqrt": np.asarray(t.sqrt, dtype=np.uint32), "rcp": np.asarray(t.rcp, dtype=np.uint32)}
    rng = np.random.default_rng(3)
    xs = np.array([0, 0x80000000, 1, 0x80000001, 0x007FFFFF, 0x00800000, 0x7F7FFFFF, 0x7F800000, 0xFF800000, 0x7FC00000, 0x3F800000,
                   0x7E800000, 0x7E800001, 0xFE800001, 0x40000000] + list(rng.integers(0, 2**32, 400, dtype=np.uint64)), dtype=np.uint64)
    n = len(xs)
    pre, post = TL.MUFU[prim]
    c1 = Circuit()
    table, idx, ctx = pre(c1, [c1.input("x", 32)])
    c1.output("idx", idx)
    for i, w in enumerate(ctx):
        c1.output(f"c{i}", w)
    o1 = {k: IL.unslice_words(b, n) for k, b in c1.evaluate({"x": IL.slice_words(xs, 32)}, n)[0].items()}
    v = tabs[table][o1["idx"].astype(np.int64)].astype(np.uint64) & np.uint64((1 << TL.VALUE_BITS) - 1)
    c2 = Circuit()
    cw = [c2.input(f"c{i}", 32) for i in range(len(ctx))]
    c2.output("y", post(c2, cw, c2.input("v", TL.VALUE_BITS)))
    ins = {**{f"c{i}": IL.slice_words(o1[f"c{i}"], 32) for i in range(len(ctx))}, "v": IL.slice_words(v, TL.VALUE_BITS)}
    y = IL.unslice_words(c2.evaluate(ins, n)[0]["y"], n)
    nan = lambda u: (u >> 23) & 0xFF == 0xFF and u & 0x7FFFFF  # noqa: E731
    ref = [getattr(P, prim[:-3]).evaluate(int(a)) for a in xs]
    bad = [(hex(int(a)), hex(int(b)), hex(c)) for a, b, c in zip(xs, y, ref) if int(b) != c and not (nan(int(b)) and nan(c))]
    assert not bad, bad[:5]


def test_lookup_rows_match_the_rust_layout():
    # 128 input rows, decoders of 13 + 10 bits, 31 * 2^10 products, output word, constant (lookup.rs)
    assert FC.lookup_rows(23, 13, 31) == -(-(128 + FC._decode_rows(13) + FC._decode_rows(10) + 31 * 1024) // 128) * 128 + 129
    assert FC._decode_rows(2) == 4 and FC._decode_rows(3) == 8 + 4


def _f32_samples(n, seed):
    rng = np.random.default_rng(seed)
    special = [0, 0x80000000, 1, 0x80000001, 0x007FFFFF, 0x00800000, 0x7F7FFFFF, 0x7F800000, 0xFF800000, 0x7FC00000, 0x3F800000,
               0xBF800000, 0x4E800000, 0xCE800000, 0x42FE0000, 0xC3000000, 0x19DB6BEB, 0x9D60AFCB]
    return special + [int(x) for x in rng.integers(0, 2**32, n, dtype=np.uint64)] + \
        [int(np.float32(x).view(np.uint32)) for x in rng.normal(0, 30, n)]


def _run(C, inputs, t):
    got, _ = C.evaluate({k: [sum(((v >> i) & 1) << j for j, v in enumerate(vs)) for i in range(bits)] for k, (vs, bits) in inputs.items()}, t)
    return {k: [sum(((b >> j) & 1) << i for i, b in enumerate(v)) for j in range(t)] for k, v in got.items()}


@pytest.mark.parametrize("name,arity", [("F32AddFtz_v1", 2), ("F32MulFtz_v1", 2), ("F32FmaFtz_v1", 3), ("F32FmaSubFtz_v1", 3),
                                        ("F32Max_v1", 2), ("F2fpBf16_v1", 1), ("F32SubFtz_v1", 2), ("GuardNegInfZero_v1", 1)])
def test_attention_tail_ops_match_the_ir(name, arity):
    """Exact on encodings, but for a NaN result's payload (canonical here, as for F32Add; every NaN meets a bf16 cast or a
    NaN-blind consumer before it leaves the tail)."""
    from verity_vllm.program.registry import prims as P

    args = [_f32_samples(150, 11 + i) for i in range(arity)]
    t = len(args[0])
    C = Circuit()
    C.output("y", list(TL.OPS[name](C, *[C.input(f"x{i}", 32) for i in range(arity)])))
    got = _run(C, {f"x{i}": (args[i], 32) for i in range(arity)}, t)["y"]
    nan = lambda u: (u & 0x7F800000) == 0x7F800000 and u & 0x7FFFFF
    for k in range(t):
        want = getattr(P, name[:-3]).evaluate(*[a[k] for a in args]) & 0xFFFFFFFF
        assert got[k] == want or (nan(got[k]) and nan(want)), (name, [hex(a[k]) for a in args], hex(got[k]), hex(want))


def test_mufu_ex2_matches_the_ir():
    from verity_vllm.program.kernels import fa2_relation
    from verity_vllm.program.registry import prims as P

    ex2 = np.asarray(fa2_relation.tables().ex2, dtype=np.uint32)
    xs = _f32_samples(300, 5)
    t = len(xs)
    pre, post = TL.MUFU["MufuEx2Ftz_v1"]
    c1 = Circuit()
    table, idx, ctx = pre(c1, [c1.input("x", 32)])
    assert table == "ex2"
    c1.output("idx", idx)
    for i, w in enumerate(ctx):
        c1.output(f"c{i}", w)
    o1 = _run(c1, {"x": (xs, 32)}, t)
    c2 = Circuit()
    c2.output("y", post(c2, [c2.input(f"c{i}", 32) for i in range(len(ctx))], c2.input("v", TL.VALUE_BITS)))
    ins = {f"c{i}": (o1[f"c{i}"], 32) for i in range(len(ctx))}
    ins["v"] = ([int(ex2[j]) & 0x7FFFFFFF for j in o1["idx"]], TL.VALUE_BITS)
    got = _run(c2, ins, t)["y"]
    assert [hex(x) for x, g in zip(xs, got) if g != P.MufuEx2Ftz.evaluate(x)] == []


def test_fa2_inv_sum_matches_the_ir():
    from verity_vllm.program.kernels import fa2_relation
    from verity_vllm.program.registry import prims as P

    rcp = np.asarray(fa2_relation.tables().rcp, dtype=np.uint32)
    xs = _f32_samples(300, 9)
    t = len(xs)
    pre, post = TL.MUFU["Fa2InvSum_v1"]
    c1 = Circuit()
    table, idx, ctx = pre(c1, [c1.input("x", 32)])
    assert table == "rcp"
    c1.output("idx", idx)
    for i, w in enumerate(ctx):
        c1.output(f"c{i}", w)
    o1 = _run(c1, {"x": (xs, 32)}, t)
    c2 = Circuit()
    c2.output("y", post(c2, [c2.input(f"c{i}", 32) for i in range(len(ctx))], c2.input("v", TL.VALUE_BITS)))
    ins = {f"c{i}": (o1[f"c{i}"], 32) for i in range(len(ctx))}
    ins["v"] = ([int(rcp[j]) & 0x7FFFFFFF for j in o1["idx"]], TL.VALUE_BITS)
    got = _run(c2, ins, t)["y"]
    assert [hex(x) for x, g in zip(xs, got) if g != P.Fa2InvSum.evaluate(x)] == []
