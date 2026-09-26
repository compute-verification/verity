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
    kinds = {r["type"] for r in meta["ranges"]}
    assert kinds == {"comp", "unit", "mask"}
    spans = sorted(((r["pos0"] << r["slot_log"]), ((r["pos0"] + FC._pow2(r["count"])) << r["slot_log"])) for r in meta["ranges"])
    assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:])), "ranges are disjoint"
    assert all(start % (end - start) == 0 for start, end in spans), "ranges are aligned to their spans"
    FC.stage(low, s, 0, 4, tmp_path / "st")
    FC.stage(low, s, 0, 4, tmp_path / "st", public_only=True)
    prover, verifier = (tmp_path / "st" / "inst-4.bin").read_bytes(), (tmp_path / "st" / "pub-4.bin").read_bytes()
    hp, hv = (json.loads(b.split(b"\n", 1)[0]) for b in (prover, verifier))
    assert hp["rows"] and not hv["rows"], "the verifier's file holds no row"
    assert prover.split(b"\n", 1)[1].endswith(verifier.split(b"\n", 1)[1]), "the public part is the same bytes"
    assert hp["circuit_sha256"] == FC.digest(text)


def test_leaf_scheme_reserves_what_hm96_needs():
    hm96 = pytest.importorskip("verity.commitments.hm96")
    t = FC.LEAF_SCHEME["target"]
    assert (t["scheme"], t["salt_bytes_per_leaf"], t["key_bytes_per_proof"]) == (hm96.NAME, hm96.SALT_BYTES, hm96.KEY_BYTES)


def test_attention_has_no_verifier_evaluated_form():
    class U:
        cut, tail_rets = [0], [0]

    class Low:
        units, name = U(), "attention"

    with pytest.raises(NotImplementedError, match="no verifier-evaluated form"):
        FC.compose(Low(), None)


@pytest.mark.parametrize("prim", sorted(TL.MUFU))
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
