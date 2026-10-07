"""`TargetProfile.ir = "boolean"` (`verity_vllm.program.boolean`): a derived Program on bits.

* the table: each word Definition's Boolean version is the registered Boolean Definition whose `word` names it, a function-valued
  static bound to its own Boolean version (`DotBf16_v2{K, DOT}` -> `DotBf16_v3{K, DOT=HopperBF16WgmmaDot16_v2}`), and a static
  the word has no counterpart for to the Boolean version of the word Definition its `word_statics` names;
* `lift` re-emits a derived function's Calls on bits, every word run mapped to the bits it occupies (affine, strided, explicit,
  concatenated and `PartLog`-cut operands, a record of mixed widths), and computes the word function's outputs (bit-sliced
  evaluation of the lift against the plain-integer reference of the word function);
* a Definition with no Boolean version is refused by name, at `lift` and at derive;
* `purity` counts the not-Boolean Definitions a Program reaches: zero for the lift, every one for the word Program;
* the default target derives on words: `ir` is absent from its serialisation, so no profile digest moves.
"""
from __future__ import annotations

import numpy as np
import pytest

from verity.ml.circuits.evaluation import evaluate, evaluate_words
from verity.ml.circuits import boolean as BL
from verity.ml.circuits.codec import _spec_id
from verity.ml.circuits.defs import Builder, CompositeDefinition, PrimitiveDefinition, bind
from verity.ml.circuits.program import Program
from verity.ml.circuits.refs import Affine, Coll, Concat, Explicit, PartLog, Strided, coll_concat
from verity.ml.circuits.types import Array, Record, Tuple as TupleT, Value
from verity.ml.circuits.ops import fp32 as W
from verity.ml.circuits.ops import gemm as G
from verity.ml.circuits.ops import prims as MP
from verity_vllm.program import boolean as BB
from verity_vllm.program.frontend.target_profile import DEFAULT_TARGET, TargetProfile
from verity_vllm.program.replay import replayer

F32, BF16 = Value(32), Value(16)
PARAMS = (("x", Array(4, F32)), ("y", Array(2, Array(4, F32))), ("weights", Record(("s", F32), ("t", Array(3, BF16)), ("u", Array(2, F32)))))


def _derived(fns=None):
    """A derived function as `frontend.derive` writes one: an unregistered `Derived[...]` composite replaying a flat body of
    registered word Definitions, whose operands are every kind of reference sequence."""
    fns = {"add": W.F32AddFtz, "mul": W.F32MulFtz, "cast": MP.F2fpBf16, "max": W.F32Max, "guard": W.GuardNegInfZero, **(fns or {})}
    B = Builder(PARAMS, TupleT())
    x, y, w = (Coll(t, Affine("p", i, 0, t.leaves)) for i, (_, t) in enumerate(PARAMS))
    n0 = B.batch(fns["add"], x, y[1], axes=(0, 0))
    y_t = Coll(Array(8, F32), Strided("p", 1, 0, 2, 1, 4, 8))                  # y transposed, [4][2]
    x_twice = Coll(Array(8, F32), Concat([Affine("p", 0, 0, 4), Explicit(tuple(("p", 0, j) for j in (3, 2, 1, 0)))]))
    n1 = B.batch(fns["mul"], y_t, x_twice, axes=(0, 0))
    n2 = B.batch(fns["cast"], n1, axes=(0,))
    n3 = B.call(fns["max"], w.field("s"), n0[2])
    n4 = B.batch(fns["guard"], w.field("u"), axes=(0,))
    log = PartLog([Affine("n", 0, 3, 1), Affine("n", 0, 0, 1), Affine("n", 3, 0, 1)])
    cut = log.cut(3)
    assert isinstance(cut, Concat) and cut.prefix is not None
    n5 = B.batch(fns["add"], Coll(Array(3, F32), cut), Coll(Array(3, F32), Affine("p", 0, 1, 3)), axes=(0, 0))
    ret_t = TupleT(Array(8, BF16), F32, Array(2, F32), Array(3, BF16), Array(3, F32))
    B.ret_type = ret_t
    body = B.finish(coll_concat(ret_t, [n2, n3, n4, w.field("t"), n5]))
    defn = CompositeDefinition("Derived[Tiny]", 1, ("SRC",), lambda S: (PARAMS, ret_t), replayer(body), register=False)
    sd = bind(defn, SRC={"class": "tests.Tiny"})
    sd.set_body(body)
    return sd


def _inputs(n: int, seed: int = 7):
    rng = np.random.default_rng(seed)
    special = np.array([0, 0x80000000, 0x3F800000, 0xBF800000, 0x7F800000, 0xFF800000, 0x7FC00000, 0x00000001, 0x007FFFFF,
                        0x7F7FFFFF], dtype=np.uint64)

    def words(k):
        a = rng.integers(0, 1 << 32, size=(n, k), dtype=np.uint64)
        pick = rng.random((n, k)) < 0.3
        return np.where(pick, special[rng.integers(0, len(special), size=(n, k))], a)

    rec = np.concatenate([words(1), rng.integers(0, 1 << 16, size=(n, 3), dtype=np.uint64), words(2)], axis=1)
    return words(4), words(8), rec


def test_the_table_reads_the_boolean_definitions_word_links():
    from verity.ml.circuits.ops.boolean import attention as BA, gemm as BG
    assert BB.boolean_version(W.F32AddFtz) is bind(BA.F32AddFtz)
    assert BB.boolean_version(MP.F2fpBf16) is bind(BG.F2fpBf16)
    word = bind(G.DotBf16, K=64, DOT=MP.HopperBF16WgmmaDot16)
    assert _spec_id(BB.boolean_version(word)) == 'DotBf16_v3{K=64,DOT={"fn":"HopperBF16WgmmaDot16_v2"}}'


def test_a_static_the_word_lacks_is_bound_to_the_boolean_version_of_its_word_statics_entry(monkeypatch):
    from verity.ml.circuits.boolean.trace import word_of

    sig = lambda t: lambda S: ((("x", Array(S.N, t)),), Array(S.N, t))  # noqa: E731
    word = CompositeDefinition("Twice", 1, ("N",), sig(F32), lambda B, S, x: B.batch(W.F32AddFtz, x, x, axes=(0, 0)), register=False)
    bits = CompositeDefinition("Twice", 2, ("N", "ADD"), sig(BL.bit_type(F32)), lambda B, S, x: B.batch(S.ADD, x, x, axes=(0, 0)),
                               register=False)

    def twice_word(b):
        if word_of(b["ADD"]).id != W.F32AddFtz.id:
            raise ValueError(f"ADD computes {word_of(b['ADD']).id}")
        return bind(word, N=b["N"])
    bits.word, bits.word_statics = twice_word, {"ADD": W.F32AddFtz}
    exact, by_name, errors = BB._load()
    monkeypatch.setattr(BB, "_TABLE", [(exact, {**by_name, "Twice": [bits]}, errors)])
    monkeypatch.setattr(BB, "_VERSIONS", {})
    got = BB.boolean_version(bind(word, N=3))
    assert _spec_id(got) == 'Twice_v2{N=3,ADD={"fn":"F32AddFtz_v3"}}'
    x = _inputs(32)[0][:, :3]
    want = np.array([evaluate(bind(word, N=3), list(map(int, r))) for r in x], dtype=np.uint64)
    assert (evaluate_words(got, x).astype(np.uint64) == want).all()

    lonely = PrimitiveDefinition("LonelyWord", 1, (("a", F32), ("b", F32)), F32, lambda a, b: a, register=False)
    monkeypatch.setattr(bits, "word_statics", {"ADD": lonely})
    monkeypatch.setattr(BB, "_VERSIONS", {})
    with pytest.raises(BB.NoBooleanVersion, match=r"Twice_v2's ADD: no Boolean version of LonelyWord_v1") as e:
        BB.boolean_version(bind(word, N=4))
    assert list(e.value.missing) == ["Twice_v1{N=4}"]


def test_the_served_norms_bind_their_mufu_statics_to_mufus_boolean_versions():
    from verity.ml.circuits.ops.boolean import mufu as BM
    from verity_vllm.program.registry import b1, boolean_norms as BN
    cuda = BB.boolean_version(bind(b1.RMSNormFusedCuda, N=16, EPS=1e-5))
    triton = BB.boolean_version(bind(b1.RMSNormTriton, N=16, EPS=1e-5))
    assert _spec_id(cuda) == _spec_id(bind(BN.RMSNormFusedCuda, N=16, EPS=1e-5, RSQRT=bind(BM.RsqrtApprox)))
    assert _spec_id(triton) == _spec_id(bind(BN.RMSNormTriton, N=16, EPS=1e-5, SQRT=bind(BM.MufuSqrtFtz),
                                             SCALEA=bind(BM.DivFullScaleA), RCP=bind(BM.DivFullRcp)))
    assert BL.is_boolean(cuda) and BL.is_boolean(triton)


def test_lift_computes_the_word_function_on_bits():
    word = _derived()
    bits = BB.lift(word)
    assert _spec_id(bits).endswith(',IR="boolean"}') and bits.name == word.name
    assert [n.fn for n in bits.body.nodes] == [BB.boolean_version(n.fn) for n in word.body.nodes]
    assert bits.signature() == (tuple((n, BL.bit_type(t)) for n, t in PARAMS), BL.bit_type(word.signature()[1]))
    x, y, rec = _inputs(64)
    got = evaluate_words(bits, x, y, rec)
    want = np.array([evaluate(word, *(list(map(int, a[i])) for a in (x, y, rec))) for i in range(len(x))], dtype=np.uint64)
    assert got.shape == want.shape == (64, 17)
    assert (got.astype(np.uint64) == want).all()


def test_the_lifted_program_is_bits_throughout_and_the_word_program_is_not():
    word = _derived()
    bits = Program(BB.lift(word))
    assert bits.input_gates == sum(BL.bit_type(t).leaves for _, t in PARAMS)
    rep = BB.purity(bits)
    assert rep["non_boolean"] == {"definitions": 0, "specializations": 0, "list": []}
    rep = BB.purity(Program(word))
    ids = [r["id"] for r in rep["non_boolean"]["list"]]
    assert ids == sorted(["F2fpBf16_v1", "F32AddFtz_v1", "F32Max_v1", "F32MulFtz_v1", "GuardNegInfZero_v1",
                          "Input16_v1", "Input32_v1", "Root[Derived[Tiny]_v1]"])
    assert rep["non_boolean"]["definitions"] == len(ids)
    dry = BB.dry_run(Program(word))
    assert dry["gaps"] == [] and dry["non_boolean"]["definitions"] == 0
    assert {r["word"]: r["boolean"] for r in dry["derived_calls"]}["F32AddFtz_v1"] == ["F32AddFtz_v3"]


def test_a_word_derivation_report_becomes_its_program_on_bits():
    from verity.ml.circuits.codec import encode_program, program_digest
    from verity_vllm.program.frontend.derive import DeriveReport

    P = Program(_derived(), annotations={"note": "carried"})
    desc = encode_program(P)
    prov = {"result": {"root": _spec_id(P.fn), "digest": program_digest(desc), "gates": P.gates}}
    word = DeriveReport(P, None, prov, prov["result"]["digest"], _spec_id(P.fn), P.gates, descriptor=desc)
    rep = BB.derived(word)
    bits = Program(BB.lift(P.fn))
    assert rep.ok and rep.digest == program_digest(bits) == program_digest(rep.descriptor) != word.digest
    assert rep.root == _spec_id(bits.fn) and rep.gates == bits.gates and rep.program.annotations == {"note": "carried"}
    assert rep.provenance["ir"] == {"ir": "boolean", "word_root": word.root, "word_digest": word.digest,
                                    "sigma_check": {"nodes": 6, "definitions": 5, "ok": True}}
    assert rep.provenance["result"]["digest"] == rep.digest and word.provenance["result"]["digest"] == word.digest
    assert BB.purity(rep.program)["non_boolean"]["definitions"] == 0


def test_sigma_check_reads_the_boolean_function_against_sigma_of_the_word_function():
    def one_add(row):
        B = Builder(PARAMS[:2], Array(4, F32))
        x, y = (Coll(t, Affine("p", i, 0, t.leaves)) for i, (_, t) in enumerate(PARAMS[:2]))
        body = B.finish(B.batch(W.F32AddFtz, x, y[row], axes=(0, 0)))
        defn = CompositeDefinition("Derived[OneAdd]", 1, (), lambda S: (PARAMS[:2], Array(4, F32)), replayer(body), register=False)
        sd = bind(defn)
        sd.set_body(body)
        return sd

    word = _derived()
    assert BB.sigma_check(word, BB.lift(word)) == {"nodes": 6, "definitions": 5, "mismatches": [], "mismatch_n": 0, "ok": True}
    callee = BB.sigma_check(word, BB.lift(_derived({"mul": W.F32AddFtz})))
    assert not callee["ok"] and callee["mismatches"] == [f"node 1: Calls {_spec_id(BB.boolean_version(W.F32AddFtz))}, "
                                                         f"σ(F32MulFtz_v1) = {_spec_id(BB.boolean_version(W.F32MulFtz))}"]
    assert BB.sigma_check(one_add(1), BB.lift(one_add(1)))["ok"]
    assert BB.sigma_check(one_add(1), BB.lift(one_add(0)))["mismatches"] == ["node 0 operand 1: not the word operand's bits"]


def test_a_definition_with_no_boolean_version_is_refused_by_name():
    lonely = PrimitiveDefinition("LonelyWord", 1, (("a", F32),), F32, lambda a: a, register=False)
    word = _derived({"guard": lonely})
    with pytest.raises(BB.NoBooleanVersion, match="LonelyWord_v1") as e:
        BB.lift(word)
    assert list(e.value.missing) == ["LonelyWord_v1"]
    dry = BB.dry_run(Program(word))
    assert dry["gaps"] == ["LonelyWord_v1"]
    assert {r["id"] for r in dry["non_boolean"]["list"]} == {"LonelyWord_v1", "Root[Derived[Tiny]_v1]"}


def test_derive_refuses_a_call_with_no_boolean_version_by_name():
    torch = pytest.importorskip("torch")
    from verity_vllm.program.frontend.derive import derive_report

    class M(torch.nn.Module):
        def forward(self, x, y):
            return x * y + x

    ex = [torch.empty(4, dtype=torch.float32, device="meta")] * 2
    word = derive_report(M(), ex, mode="reference")
    assert word.ok and "ir" not in word.provenance
    rep = BB.derived(word)
    assert not rep.ok and rep.unsupported.reason == "numerics_unregistered"
    assert sorted(rep.unsupported.detail["missing"]) == ['AddRef_v1{N=4,DT="f32"}', 'MulRef_v1{N=4,DT="f32"}']
    assert rep.provenance["ir"] == {"ir": "boolean", "word_root": word.root, "word_digest": word.digest}
    assert BB.derived(rep) is rep


def test_bit_offsets_are_the_prefix_sums_of_the_leaf_widths():
    for t in (Record(("s", F32), ("t", Array(3, BF16)), ("u", Array(2, F32))), Array(3, TupleT(BF16, Array(2, F32), Value(1))),
              TupleT(Array(0, F32), F32, Array(2, TupleT()), BF16), Array(5, BF16)):
        widths = t.leaf_widths()
        off = BB.bit_offsets(t)
        assert [off(j) for j in range(len(widths) + 1)] == list(np.cumsum([0] + widths))


def test_the_default_target_derives_on_words():
    assert DEFAULT_TARGET.ir is None and "ir" not in DEFAULT_TARGET.to_json()
    on_bits = DEFAULT_TARGET.with_overrides(ir="boolean")
    assert on_bits.to_json()["ir"] == "boolean" and on_bits.digest() != DEFAULT_TARGET.digest()
    assert TargetProfile.from_json(on_bits.to_json()) == on_bits
    for bad in ("word", "bits", ""):
        with pytest.raises(ValueError, match="ir must be None"):
            TargetProfile(ir=bad)
