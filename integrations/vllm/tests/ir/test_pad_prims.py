"""Lane PAD-IMPL: the flag/select primitives of the padded serving circuit (spec §9 of
`docs/vllm-poc/serve3-spec.md`) — `verity_vllm/registry/pad_prims.py` (`BitAnd_v1`/`BitNot_v1` new; the
selects and `I32Eq_v1` reused from `prims.py`) — and their agreement with the `veritor.pad-serve@1` gate
set (`src/veritor/core/pad_serve.py`, the Python twin of the SP1 statement table
`zk/sp1/common/src/gateset.rs` and the Lean table `lean/VeritorSpec/PadServe.lean`).  The Rust test
`pad_serve_manifest_matches_the_python_family` pins the same digest this file pins."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from verity_vllm.registry import pad_prims as PP  # noqa: E402
from verity_vllm.registry import prims as P  # noqa: E402

#: `veritor.core.pad_serve.make_pad_serve_gate_set().digest` — the identity the Rust statement table pins
#: (`zk/sp1/common/src/gateset.rs::PAD_SERVE_DIGEST_HEX`).
PAD_SERVE_DIGEST_HEX = "3aa430e2d7e6cb4dca307acc5defcff84cdb7ce9b79423cd01af12aa3a10c536"


def test_bit_gates_truth_tables_and_absorption():
    assert [PP.BitAnd.evaluate(a, b) for a in (0, 1) for b in (0, 1)] == [0, 0, 0, 1]
    assert [PP.BitNot.evaluate(a) for a in (0, 1)] == [1, 0]
    # the finished chain is absorbing (spec §4): live' = and(live, not(stop)) is 0 whenever live is 0
    for stop in (0, 1):
        assert PP.BitAnd.evaluate(0, PP.BitNot.evaluate(stop)) == 0
    # and forwards the negated stop bit whenever live is 1
    for stop in (0, 1):
        assert PP.BitAnd.evaluate(1, PP.BitNot.evaluate(stop)) == (stop ^ 1)


def test_new_primitives_are_registered_and_reused_ones_untouched():
    assert PP.BitAnd.id == "BitAnd_v1" and PP.BitNot.id == "BitNot_v1"
    assert PP.BitAnd.conformance == "protocol-defined" and PP.BitNot.conformance == "protocol-defined"
    # the spec's "EqI32" / typed selects are the existing registered primitives, reused not re-registered
    assert [p.id for p in PP.REUSED] == ["I32Eq_v1", "SelectBf16_v1", "SelectF32_v1", "SelectI32_v1"]
    assert all(p.conformance == "exact-model-tested" for p in PP.REUSED)


def test_gate_set_digest_and_agreement_with_registry_evaluators():
    """The `veritor.pad-serve@1` gate set: pinned digest (= the Rust table's), and each gate bit-identical to
    the verity.ir registry evaluator it names, on edge vectors."""
    core = pytest.importorskip("veritor.core.pad_serve")
    s = core.make_pad_serve_gate_set()
    assert s.id == "veritor.pad-serve@1"
    digest = s.digest.hex() if hasattr(s.digest, "hex") else str(s.digest)
    assert digest == PAD_SERVE_DIGEST_HEX
    assert s.input_gates == ("in",) and s.weight_gates == ("weight",)

    rng = np.random.default_rng(29)
    bits16 = [0x0000, 0x3F80, 0x7FFF, 0x7FC0, 0x8000, 0xFFFF] + [int(x) for x in rng.integers(0, 1 << 16, 20)]
    bits32 = [0, 1, 0x3F800000, 0x7F800000, 0x7FC00000, 0x80000000, 0xFFFFFFFF] + [int(x) for x in rng.integers(0, 1 << 32, 20)]
    for c in (0, 1):
        for a, b in zip(bits16, reversed(bits16)):
            assert s["select_bf16"].evaluate((c, a, b)) == P.SelectBf16.evaluate(c, a, b)
        for a, b in zip(bits32, reversed(bits32)):
            assert s["select_f32"].evaluate((c, a, b)) == P.SelectF32.evaluate(c, a, b)
            assert s["select_i32"].evaluate((c, a, b)) == P.SelectI32.evaluate(c, a, b)
    for a in bits32:
        for b in (a, (a + 1) & 0xFFFFFFFF, 0):
            assert s["eq_i32"].evaluate((a, b)) == P.I32Eq.evaluate(a, b)
    for a in (0, 1):
        assert s["bit_not"].evaluate((a,)) == PP.BitNot.evaluate(a)
        for b in (0, 1):
            assert s["bit_and"].evaluate((a, b)) == PP.BitAnd.evaluate(a, b)


def test_gate_set_rejects_out_of_width_words():
    core = pytest.importorskip("veritor.core.pad_serve")
    s = core.make_pad_serve_gate_set()
    for gate, args in [("select_bf16", (2, 0, 0)), ("select_bf16", (1, 1 << 16, 0)),
                       ("eq_i32", (1 << 32, 0)), ("bit_not", (2,)), ("select_bf16", (1, 0)), ("bit_and", (1,))]:
        with pytest.raises(Exception):
            s[gate].evaluate(args)
