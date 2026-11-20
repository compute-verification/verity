"""The replay row kernel of the dense Definitions (`program.kernels.ir_rows`): on a Gemma-2-style dense Program every row is an evaluable
unit of the replay population, and each dense row's words are the IR reference's (edge words included)."""
from __future__ import annotations


import numpy as np
import pytest
from verity.circuits.defs import bind
from verity.circuits.evaluation import evaluate

from tests.acquire.call_boundary_programs import DENSE, LP, dense_program
from verity_vllm.definitions.numerics import ir_rows
from verity_vllm.definitions.numerics.rows import ROWS
from verity_vllm.definitions.registry import dense
from verity_vllm.query.manifest import format as RM
from verity_vllm.query.manifest.format import spec_family
from verity_vllm.query.program_view import spec_statics, specialization

PROMPT = [0, 1, 2, 3, 4, 5, 9]


def _fields(rng, V=16, H=8):
    def bf16(n, s=1.0):
        return (np.asarray(rng.standard_normal(n) * s, dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16)
    table = bf16(V * H).reshape(V, H)
    table[:4] = [[0x7FC0, 0xFFC1, 0x7F80, 0xFF80, 0x0000, 0x8000, 0x0001, 0x7F7F]] * 4
    return {n: (table.reshape(-1) if c == V * H else bf16(c, 0.5)) for n, c, _b, _t in DENSE}


def test_each_dense_row_is_the_ir_reference():
    b, _at = dense_program()
    doc = b.doc(LP)
    fields = _fields(np.random.default_rng(5))
    weights = [int(w) for n, *_ in DENSE for w in fields[n]]
    vals: dict[int, list[int]] = {}
    seen: set[str] = set()
    for r in doc["rows"]:
        args = [[v for kind, a, lo, hi in g for v in (PROMPT[lo:hi] if kind == "p" and a == 0 else weights[lo:hi] if kind == "p" else vals[a][lo:hi])]
                for g in r["args"]]
        fam, st = spec_family(r["spec"]), spec_statics(r["spec"])
        vals[r["i"]] = list(evaluate(specialization(fam, st), *args))
        if fam in ir_rows.FAMILIES:
            # the replay reads statics as the manifest spells them (an f64 literal stays `{"f64":"<hex>"}` text)
            got = ROWS[fam].fn(RM.spec_statics(r["spec"]), [np.asarray(a, dtype=np.uint64) for a in args])
            assert got["0"] is got["out"] and got["out"].tolist() == vals[r["i"]], r["name"]
            seen.add(fam)
    assert seen == set(ir_rows.FAMILIES) - {"GeluTanhMul_v1"}


def test_gelu_tanh_mul_at_gemma_width_and_its_declines():
    rng = np.random.default_rng(6)
    width = 9216
    gu = (np.asarray(rng.standard_normal(2 * width) * 3, dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16)
    gu[:6] = [0x7FC0, 0xFF80, 0x7F80, 0x8000, 0x0001, 0x7F7F]
    got = ROWS["GeluTanhMul_v1"].fn({"I": width}, [gu])
    assert list(got) == ["0", "out"] and got["out"].dtype == np.uint16
    assert got["out"].tolist() == list(evaluate(bind(dense.GeluTanhMul, I=width), gu.tolist()))
    with pytest.raises(ir_rows.Declined, match="arg words"):
        ROWS["GeluTanhMul_v1"].fn({"I": width}, [gu[:-1]])
    with pytest.raises(ir_rows.Declined, match="past"):
        ROWS["GeluTanhMul_v1"].fn({"I": ir_rows.MAX_WORDS}, [np.zeros(2 * ir_rows.MAX_WORDS, dtype=np.uint16)])


