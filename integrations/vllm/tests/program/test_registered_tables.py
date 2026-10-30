"""The tables a program brings as registered inputs (`registry.tables`): the read takes the table as an operand, and the Definitions
that read it compute what their twins, whose primitive evaluator holds the table, compute. A `registered_values` derivation of
Gemma-2's GeGLU registers no table: its Boolean version computes the activation (`registry.boolean_tanh`)."""

from __future__ import annotations

import numpy as np
import pytest

from verity.ml.circuits.evaluation import RegisteredTable, evaluate
from verity.ml.circuits.codec import encode_program, program_digest
from verity.ml.circuits.constants import REGISTRATION, inputs
from verity.ml.circuits.defs import bind
from verity.ml.circuits.program import Program
from verity.ml.circuits.types import Array, Value
from verity.ml.definitions.registry import dense as D
from verity.ml.definitions.registry import tables as T


def test_the_read_takes_the_entry_at_the_index_low_bits_its_words_low_first():
    rng = np.random.default_rng(11)
    r16 = T.table_read(16, 1)
    assert r16.id == "TableRead16x1_v1" and r16.params[0][1] == Value(16) and r16.ret == Value(16)
    assert r16.params[1][1] == T.table_type("gelu_tanh_bf16") == Array(1024, Array(64, Value(16)))
    words = rng.integers(0, 1 << 16, 1 << 16).tolist()
    for i in [0, 1, 0xFFFF, *rng.integers(0, 1 << 16, 8).tolist()]:
        assert evaluate(r16, i, words) == (words[i],)
    r27 = T.table_read(27, 2)
    assert r27.id == "TableRead27x2_v1" and r27.params[0][1] == Value(32) and r27.ret == Value(32)
    assert r27.params[1][1] == T.table_type("tanh_rn") == Array(1 << 22, Array(64, Value(16)))
    word = lambda j: (j * 0x9E37) & 0xFFFF   # noqa: E731
    for idx in [0, (1 << 27) - 1, 0xF8000005, 0x0123_4567]:
        e = idx & ((1 << 27) - 1)
        assert r27.evaluate_lazy(lambda j: idx if j == 0 else word(j - 1)) == word(2 * e) | word(2 * e + 1) << 16
    with pytest.raises(ValueError, match="do not fill rows"):
        T.table_read(4, 1)


def test_a_read_has_its_boolean_form_up_to_max_words():
    """`table_read` makes `TableRead{IB}x{EW}_v2` with the read (`verity.ml.circuits.boolean.table`, by which Flock lowers it) up
    to `BT.MAX_WORDS` table words, and `gelu_tanh_bf16`'s and `tanh_rn`'s reads, past it, have none."""
    from verity.ml.circuits.boolean import table as BT
    from verity.ml.circuits.defs import REGISTRY

    r = T.table_read(6, 1)
    assert REGISTRY.defs["TableRead6x1_v2"] is BT.table_read(r, 6) and REGISTRY.defs["TableRead6x1_v2"].word is r
    for name in ("gelu_tanh_bf16", "tanh_rn"):
        p = T.table_read(*T.TABLES[name])
        assert f"{p.name}_v2" not in REGISTRY.defs


def test_a_program_reading_a_table_decodes_where_it_was_not_built(tmp_path):
    """`TableRead16x1_v1` comes from its lazy family in a process that never built a Definition reading it."""
    import json
    import subprocess
    import sys

    prog = Program(T.GeluTanhMulBf16V3, registered=("gelu_tanh_bf16",))
    d = tmp_path / "p.json"
    d.write_text(json.dumps(encode_program(prog)))
    code = ("import json, sys; from verity.ml.circuits.codec import decode_program, program_digest; from verity.ml.circuits.defs import REGISTRY; "
            "import verity.ml.definitions.registry.tables; assert 'TableRead16x1_v1' not in REGISTRY; "
            "print(program_digest(decode_program(json.load(open(sys.argv[1])), REGISTRY)))")
    out = subprocess.run([sys.executable, "-c", code, str(d)], capture_output=True, text=True, check=True).stdout.split()
    assert out == [program_digest(prog)]
    assert len(inputs(prog.circuit)[REGISTRATION]) == 1 << 16


def test_tanh_rn_entries_are_tanh_f32_rn_at_the_region_words_low_27_bits():
    rng = np.random.default_rng(12)
    lo, hi = T.TANH_RN_LO, T.TANH_RN_HI
    region = [lo, lo + 1, 0x3F800000, 0x40000000, 0x3FFFFFFF, hi - 1, *rng.integers(lo, hi, 200).tolist()]
    for w in region:
        e = T.tanh_rn_entry(w & 0x7FFFFFF)
        assert e == D.tanh_f32_bits(w) and e < 1 << 31
    assert hi - lo < 1 << 27   # every region word has its own index
    assert T.tanh_rn_entry((lo - 1) & 0x7FFFFFF) == 0 == T.tanh_rn_entry(hi & 0x7FFFFFF)


def test_tanh_f32_rn_v2_is_v1_reading_the_registered_table_lazily():
    """`TanhF32Rn_v2` reads `tanh_rn`'s 2^28 words lazily (`tanh_rn_table`), two per instance: on both signs of each rule's and
    each region's edges, every exponent, and random words of the region and of all f32, it is `TanhF32Rn_v1`."""
    rng = np.random.default_rng(15)
    lo, hi = T.TANH_RN_LO, T.TANH_RN_HI
    edges = [0, 1, 0x007FFFFF, 0x00800000, lo - 1, lo, lo + 1, 0x37FFFFFF, 0x38000000, 0x3F800000, 0x3FFFFFFF, 0x40000000, hi - 1,
             hi, 0x41200000, 0x7F7FFFFF, 0x7F800000, 0x7F800001, 0x7FC00000, 0x7FFFFFFF]
    edges += [0x38000000 | e for r in T.TANH_RN_REGIONS for e in (r[0], r[1] - 1)] + [0x40000000 | e for r in T.TANH_RN_REGIONS for e in r]
    exponents = [e << 23 | m for e in range(256) for m in (0, 1, 0x400000, 0x7FFFFF)]
    words = edges + exponents + rng.integers(lo, hi, 2000).tolist() + rng.integers(0, 1 << 31, 1000).tolist()
    words = [(w & 0x7FFFFFFF) | s for w in words for s in (0, 1 << 31)]
    table, reads = T.tanh_rn_table(), []
    assert (table.name, table.leaves, table.declared_by) == ("tanh_rn", T.TANH_RN_TABLE.leaves, D.TanhF32Rn)
    counted = RegisteredTable("tanh_rn", lambda j: reads.append(j) or table.word(j), table.leaves, D.TanhF32Rn)
    fn = bind(T.TanhF32RnV2)
    assert fn.id == "TanhF32Rn_v2"
    for w in words:
        assert evaluate(fn, w, counted) == (D.TanhF32Rn.evaluate(w),), hex(w)
    assert len(reads) == 2 * len(words)


def test_the_strict_lift_of_a_table_read_holds_its_widths_as_runs():
    """The strict lift's operand widths (`operand_widths.leaf_widths`) are one run per parameter for `TableRead27x2_v1`, not a list
    of 2^28, and a list for a narrow primitive; the lift of `TableRead16x1_v1` (past the layout's wide fan-in too) still maps an
    active, a ⊥ and an invalid table word as the flat widths do."""
    from verity.ml.circuits.ops import scalar as SC
    from verity.ml.definitions.registry import lifted as L
    from verity.ml.definitions.registry import operand_widths as OW

    assert OW.leaf_widths(T.table_read(27, 2).params).runs == ((1, 32), (1 << 28, 16))
    assert OW.leaf_widths(SC.SelectF32.params) == [1, 32, 32]
    lp = L.lifted_prim_v1(T.table_read(16, 1))
    words = list(range(1 << 16))
    assert lp.evaluate(5, *words) == 5
    assert lp.evaluate(5, *words[:9], L.enc_bot(16), *words[10:]) == L.enc_bot(16)
    assert lp.evaluate(5, *words[:-1], L.inv(16)) == L.inv(16)
    assert lp.evaluate(L.enc_bot(16), *words) == L.enc_bot(16)


def test_gelu_tanh_mul_bf16_v3_is_v1_reading_the_registered_table():
    """`GeluTanhMulBf16_v3` on special words, `gelu_tanh_bf16` its operand, against `GeluTanhMulBf16_v1` (the Lean evaluator's
    difftest runs it on every gate word)."""
    table = T.table_words("gelu_tanh_bf16")
    for g, u in [(0x3F80, 0x8000), (0xFF80, 0), (0x7FC0, 0x3F80), (0x0001, 0x7F80), (0x8000, 0xFF80)]:
        assert evaluate(T.GeluTanhMulBf16V3, g, u, table) == (D.GeluTanhMulBf16.evaluate(g, u),)


def test_a_registered_values_derivation_registers_no_gelu_table():
    """`_C.gelu_tanh_and_mul` under `registered_values`: `GeluTanhMul_v1`, as the constants derivation binds it, with no table
    parameter (its Boolean version, `boolean_tanh.GeluTanhMul_v4`, computes the activation), and the Program computes what the
    constants derivation computes."""
    import torch

    from verity_vllm.program.frontend.derive import derive_report
    from verity_vllm.program.frontend.rules.vllm_bindings import vllm_ruleset
    from verity.ml.definitions.numerics.serve3_reference import materialize

    class GeGLU(torch.nn.Module):
        def __init__(self, I: int) -> None:
            super().__init__()
            self.I, self.weight = I, torch.nn.Parameter(torch.empty(2 * I, dtype=torch.bfloat16))

        def forward(self, x):
            y = torch.empty_like(x)
            torch.ops.vllm_iface.rms_norm(y, x, self.weight, 1e-6)
            out = torch.empty(x.shape[0], self.I, dtype=x.dtype)
            torch.ops.vllm_iface.gelu_tanh_and_mul(out, y)
            return out

    with torch.device("meta"):
        m = GeGLU(8)
    x = (torch.empty(2, 16, dtype=torch.bfloat16, device="meta"),)
    kw = dict(mode="profile", profile="b1-eager-v3", name="GeGLU", rules=vllm_ruleset())
    const, reg = derive_report(m, x, **kw), derive_report(m, x, registered_values=True, **kw)
    assert const.ok and reg.ok, reg.unsupported
    prog, twin = reg.program, Program(const.flat_definition)
    assert [n for n, _ in prog.fn.body.params] == ["x", "values", "weights"]
    assert prog.bindings == ("run", "registration", "run")
    assert not reg.provenance["supplied"].get("tables")
    calls = {n.fn.id for n in reg.flat_definition.body.nodes}
    assert calls == {"GeluTanhMul_v1{I=8}", "RMSNormTriton_v3{N=16}"}
    assert {n.fn.id for n in const.flat_definition.body.nodes} >= {"GeluTanhMul_v1{I=8}"}
    assert len(inputs(prog.circuit)[REGISTRATION]) == 1
    rng = np.random.default_rng(14)
    xs = [int(v) >> 16 for v in (rng.standard_normal(32).astype(np.float32) * 2).view(np.uint32)]
    w = [int(v) >> 16 for v in (rng.standard_normal(16).astype(np.float32)).view(np.uint32)]
    eps = int(np.float32(1e-6).view(np.uint32))
    a = materialize(twin, [xs, w])
    b = materialize(prog, [xs, [eps], w])
    assert [b[g] for g in prog.circuit.root_outputs()] == [a[g] for g in twin.circuit.root_outputs()]

