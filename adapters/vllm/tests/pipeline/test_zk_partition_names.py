"""The templates a GLM-5.3 partition for ``verify --zk`` sessions names for the five unit families whose word Definitions read
a MUFU table (`chunk_rows.zk_refusal`, a partition file's ``zk``): FlashInfer's FP8 linear coordinate and its quantisation,
the sigmoid router's row and its slots, and the DSA indexer's query row.

``verify --zk`` proves knowledge soundness for a unit that returns its own outputs (`one_stage.proofs.ks_form_why`).  A unit
whose table reads are laid out in lookup slots returns them from a tail stage, so a partition for ``--zk`` names the Boolean
version, whose MUFU is gates:

* at GLM-5.3's served statics, each family's unit template is its ``_v2``: the Boolean version of the word Definition the
  rows hold, read through its word link with the word template's layout and rows, its statement's reads inline;
* on a small binding, a row's units under the ``_v2`` name are the members the word name gives, and the ``_v2`` Definition
  computes the word Definition's words;
* a partition that says ``zk`` and names a template whose statement is the lookup-slots form is refused by name, with the
  tables and the Boolean version to name; the same file without ``zk`` is not, and neither is the Boolean version;
* staged as its statement says, each ``_v2`` is one flat net that `ks_form_why` takes, and each word unit in lookup slots
  is the form it refuses, which `zk_refusal` says of the same template without staging it."""
from __future__ import annotations

__models__ = ("zai-org/GLM-5.3",)

from types import SimpleNamespace

import numpy as np
import pytest
from verity_flock.class_statement import UnitStager

from verity.protocols.spot.verifier.one_stage import proofs as PF
from verity.circuits import evaluation as E
from verity.circuits.codec import descriptor_id
from verity.circuits.evaluation import bits as BS
from verity.definitions import boolean as BB
from verity.definitions.numerics import deepgemm_twins as DT
from verity_vllm.pipeline import chunk_rows as CR
from verity_vllm.pipeline import serving_rows as SRW
from verity_vllm.query.program_view import spec_statics

from tests.pipeline import test_serving_rows_splits as SPL

ROUTER = 'E=256,TOPK=8,SCALE={"f64":"0x1.4000000000000p+1"},KERNEL="block"'
#: GLM-5.3's unit templates of the five families as its rows hold them (`serving_templates.unit_templates`), at its statics:
#: hidden 6144, q_lora 2048, the shared expert's 1536 and 256, 256 experts and 8 slots, an indexer of 32 heads of 128
SERVED = {
    "FlashInferFp8Linear": [f"FlashInferFp8Linear_v1{{K={K},N=1}}" for K in (6144, 2048, 1536, 256)],
    "FlashInferFp8Quant1x128": [f"FlashInferFp8Quant1x128_v1{{K={K}}}" for K in (6144, 2048, 1536, 256)],
    "MoeRouterSigmoidBias": [f"MoeRouterSigmoidBias_v1{{{ROUTER}}}"],
    "MoeRouterSigmoidBiasSlots": [f"MoeRouterSigmoidBiasSlots_v1{{{ROUTER}}}"],
    "DsaIndexerQ": ["DsaIndexerQ_v1{H=32,D=128,ROPE=64}"],
}
ROUTER_S = 'E=32,TOPK=2,SCALE={"f64":"0x1.4000000000000p+1"},KERNEL="block"'
_ROUTED = [("topk_weights", 2, "f32"), ("topk_ids", 2, "i32")]
#: each family at the least statics Flock composes (some row of a statement takes a compression before its last block, so
#: one is at least 112 bytes): its unit template, the row that holds it, the row's committed members, and the tables the
#: word unit reads
SMALL = {
    "FlashInferFp8Linear": ("FlashInferFp8Linear_v1{K=128,N=1}", "FlashInferFp8Linear_v1{K=128,N=4}", [("out", 4, "bf16")], ("rcp",)),
    "FlashInferFp8Quant1x128": ("FlashInferFp8Quant1x128_v1{K=128}", "FlashInferFp8Linear_v1{K=128,N=4}", [("out", 4, "bf16")],
                                ("rcp",)),
    "MoeRouterSigmoidBias": (f"MoeRouterSigmoidBias_v1{{{ROUTER_S}}}", f"MoeRouterSigmoidBias_v1{{{ROUTER_S}}}", _ROUTED,
                             ("ex2", "rcp")),
    "MoeRouterSigmoidBiasSlots": (f"MoeRouterSigmoidBiasSlots_v1{{{ROUTER_S}}}", f"MoeRouterSigmoidBias_v1{{{ROUTER_S}}}", _ROUTED,
                                  ("ex2", "rcp")),
    "DsaIndexerQ": ("DsaIndexerQ_v1{H=1,D=64,ROPE=2}", "DsaIndexerQ_v1{H=1,D=64,ROPE=2}",
                    [("index_q_fp8", 64, "e4m3"), ("index_weights_out", 1, "f32")], ("ex2",)),
}
FAMILIES = list(SERVED)


def gates(word: str) -> str:
    """The id of the Boolean version of word template ``word``."""
    return descriptor_id(BB.boolean_version(SRW.template(word).fn))


def _bf16(rng, shape) -> np.ndarray:
    return ((rng.standard_normal(shape) * 3).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)


def _f32(rng, shape) -> np.ndarray:
    return (rng.standard_normal(shape) * 2).astype(np.float32).view(np.uint32)


def _e4m3(rng, shape) -> np.ndarray:
    w = rng.integers(0, 256, shape, dtype=np.uint8)
    w[(w & 0x7F) == 0x7F] ^= 1
    return w


def inputs(family: str, n: int) -> list[np.ndarray]:
    """``n`` rows of the family's small unit: activation-like values, then as many rows of uniform words (non-finite ones
    among them) for the operands serving computes; a weight row keeps finite E4M3 codes and power-of-two scales."""
    rng = np.random.default_rng(sorted(SMALL).index(family))
    S = spec_statics(SMALL[family][0])

    def both(like, width, shape):
        return np.concatenate([like(rng, (n, *shape)), rng.integers(0, 1 << width, (n, *shape), dtype=np.uint64).astype(f"<u{width // 8}")])

    if family.startswith("FlashInfer"):
        x = both(_bf16, 16, (S["K"],))
        if family == "FlashInferFp8Quant1x128":
            return [x]
        return [x, _e4m3(rng, (2 * n, 1, S["K"])), DT.upcast_e8m0(rng.integers(112, 124, 2 * n)).reshape(2 * n, 1, 1)]
    if family.startswith("MoeRouter"):
        return [both(_f32, 32, (S["E"],)), both(lambda r, s: (r.standard_normal(s) * 1e-2).astype(np.float32).view(np.uint32), 32, (S["E"],))]
    return [both(_bf16, 16, (S["H"], S["D"])), both(_bf16, 16, (S["ROPE"],)), both(_bf16, 16, (S["H"],))]


def _flat(out, n: int) -> np.ndarray:
    return np.concatenate([np.asarray(a, np.uint64).reshape(n, -1) for a in (out if isinstance(out, (tuple, list)) else [out])], axis=1)


@pytest.mark.parametrize("family", FAMILIES)
def test_a_served_unit_s_template_is_its_boolean_version(family):
    for word in SERVED[family]:
        wt, bt = SRW.template(word), SRW.template(gates(word))
        assert bt.id == word.replace("_v1{", "_v2{", 1) and bt.id != wt.id
        assert (bt.word, bt.layout, bt.packed) == (wt.id, wt.layout, wt.packed)
        assert CR.unit_rows(bt) == CR.unit_rows(wt)
        assert CR.statement(bt) == {"reads": "inline"} and CR.zk_refusal(bt) is None


@pytest.mark.parametrize("family", FAMILIES)
def test_a_row_s_units_under_the_boolean_name_are_the_word_name_s_members(family):
    word, spec, committed, _tables = SMALL[family]
    args = [[["x", 70 + k, 0, 1]] for k in range(len(SRW.template(spec).fn.signature()[0]))]
    row = ("r0", None, {"i": 7, "spec": spec, "args": args},
           {"statics": spec_statics(spec), "op_path": "model.layers.0.op", "row": 0, "step": 16, "members": committed})
    got = {t: SRW.members([t], [[row]], None, {t: 8}) for t in (word, gates(word))}
    assert [m.template for m in got[gates(word)]] == [gates(word)]
    assert [[(m.rows, m.outs, m.n, m.base, m.schemas, m.tables) for m in ms] for ms in got.values()] == [
        [(m.rows, m.outs, m.n, m.base, m.schemas, m.tables) for m in got[word]]] * 2
    assert got[word][0].n == (4 if family == "FlashInferFp8Linear" else 1) and got[word][0].schemas


@pytest.mark.slow
@pytest.mark.parametrize("family", FAMILIES)
def test_the_boolean_version_computes_the_word_definition_s_words(family):
    """Each ``_v2`` on its small binding, evaluated on bits, against its ``_v1`` (5 to 40 s each)."""
    word, n = SMALL[family][0], 6
    args = inputs(family, n)
    got = _flat(BS.evaluate_words(SRW.template(gates(word)).fn, *args), 2 * n)
    want = _flat(E.evaluate_batch(SRW.template(word).fn, *args), 2 * n)
    np.testing.assert_array_equal(got, want)
    assert len({r.tobytes() for r in want}) > n


def test_only_a_statement_in_lookup_slots_that_reads_a_slot_s_table_is_refused():
    """`zk_refusal` on the split chain's templates of `test_serving_rows_splits`: the whole first key block's word template reads
    EX2 in slots and is refused with its Boolean version, whose reads are inline; the partial first block, the column end and
    the combine read their tables inline and are not."""
    f16, f8, e0, c2 = (SRW.template(t) for t in (SPL.F16, SPL.F8, SPL.E0, SPL.C2))
    assert [CR.statement(tp)["reads"] for tp in (f16, f8, e0, c2)] == ["slots", "inline", "inline", "inline"]
    assert [CR.slot_tables(tp.fn) for tp in (f16, f8, e0, c2)] == [("ex2",), ("ex2",), ("rcp",), ("ex2", "rcp")]
    assert CR.zk_refusal(f8) is None and CR.zk_refusal(e0) is None and CR.zk_refusal(c2) is None
    gates16 = SRW.template(SPL.block(16, boolean=True))
    assert gates16.word == SPL.F16 and CR.statement(gates16) == {"reads": "inline"} and CR.zk_refusal(gates16) is None
    why = CR.zk_refusal(f16)
    assert why.startswith(f"{SPL.F16}: its statement reads ex2 in lookup slots") and "`verify --zk` refuses" in why
    assert why.endswith(f"name its Boolean version {SPL.block(16, boolean=True)}")


def test_a_partition_for_zk_sessions_refuses_a_slots_form_template_by_name():
    gates16 = SPL.block(16, boolean=True)
    partition = {"templates": [SPL.F8, SPL.F16, SPL.C2], "law": "stratified:1", "chunk_cap": SPL.CAPS, "zk": True}
    with pytest.raises(ValueError) as e:
        SRW.population({}, partition, "00" * 64)
    assert str(e.value) == f"a partition for --zk sessions names {CR.zk_refusal(SRW.template(SPL.F16))}"
    assert SPL.F16 in str(e.value) and gates16 in str(e.value)
    for other in ({**partition, "zk": False}, {k: v for k, v in partition.items() if k != "zk"},
                  {**partition, "templates": [SPL.F8, gates16, SPL.C2], "chunk_cap": {**SPL.CAPS, gates16: 3}}):
        with pytest.raises(LookupError, match="no rows of SparseMlaHeadBlock_v1{DR=16,DV=32,NVIS=8,"):
            SRW.population({}, other, "00" * 64)
    window = SimpleNamespace(members=[], law="stratified:1", query={"name": SRW.TEMPLATE_QUERIES[0]}, row_tables=(), work_table=None,
                             partition={}, partition_digest="00" * 64, population=0)
    assert SRW.partition_doc(window, partition)["zk"] is True and "zk" not in SRW.partition_doc(window, {**partition, "zk": False})


@pytest.mark.parametrize("family", FAMILIES)
def test_a_word_unit_laid_out_in_lookup_slots_is_refused_with_its_boolean_version(family, monkeypatch):
    """The five families' reads are inline under either name; were a word unit's laid out in slots (``compose_circuit.py
    --reads slots``), the partition would name the form ``--zk`` refuses, and `zk_refusal` says so with the ``_v2`` name."""
    word, _spec, _committed, tables = SMALL[family]
    wt = SRW.template(word)
    assert CR.statement(wt) == {"reads": "inline"} and CR.zk_refusal(wt) is None and CR.slot_tables(wt.fn) == tables
    for served in SERVED[family]:
        assert CR.slot_tables(SRW.template(served).fn) == tables
    monkeypatch.setattr(CR, "reads", lambda tp: "slots")
    why = CR.zk_refusal(wt)
    assert why.startswith(f"{word}: its statement reads {', '.join(tables)} in lookup slots")
    assert why.endswith(f"name its Boolean version {gates(word)}")
    with pytest.raises(ValueError, match="a partition for --zk sessions names " + word.split("{")[0]):
        SRW.population({}, {"templates": [word], "law": "subset:1", "chunk_cap": {word: 8}, "zk": True}, "00" * 64)


def _staged(tp, d, reads: str) -> str | None:
    """`ks_form_why` on template ``tp``'s unit staged as the served composer stages it, its table reads laid out ``reads``."""
    in_rows, out_rows = CR.unit_rows(tp)
    d.mkdir()
    UnitStager(tp.fn, "unit/serving-rows", in_rows, out_rows, circuit=d / "circuit.txt", reads=reads)
    return PF.ks_form_why(d / "circuit.txt", d / "pub.bin")


@pytest.mark.slow
@pytest.mark.parametrize("family", FAMILIES)
def test_staged_the_boolean_version_is_flat_and_the_word_unit_in_slots_is_the_form_zk_refuses(family, tmp_path, monkeypatch):
    """Each family's small unit composed by `UnitStager`, the Boolean version as its statement says and the word unit in
    slots (a minute or two and a few GB each; the largest is the router's): the staged circuit's own head decides, where
    every faster test here reads the Definitions.  It catches a ``_v2`` body that reads a table after all, and a
    `zk_refusal` that no longer agrees with `ks_form_why`."""
    word, _spec, _committed, _tables = SMALL[family]
    wt, bt = SRW.template(word), SRW.template(gates(word))
    assert CR.slot_tables(bt.fn) == () and CR.statement(bt) == {"reads": "inline"}
    assert _staged(bt, tmp_path / "gates", "inline") is None
    why = _staged(wt, tmp_path / "word-slots", "slots")
    assert why is not None and PF.SCOPE_WHY in why
    monkeypatch.setattr(CR, "reads", lambda tp: "slots")
    assert CR.zk_refusal(wt) is not None
