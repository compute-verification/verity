"""circuit-check over every registered Definition and subcircuit template, a served row's request Program, and the checks' own
negatives (each check fires by name on a circuit built to break it).

A failure is a correctness or invariant violation.  The ones `main` already has are listed in `circuit_check.known` with their
cause; a test whose only failures are known is an expected failure, and `test_known_failures_are_current` fails once a known one
is fixed, so the entry is dropped.  Redundant and dead gates are warnings: counted in the report, never a test failure here.
The per-target tests are marked `circuit_suite` and collected only under `-m circuit_suite` (see conftest.py): `check` runs the
same checks as `circuit-check --all`.
"""
from __future__ import annotations

import dataclasses
import gzip
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest

from circuit_check import checks as CK
from circuit_check import known as KN
from circuit_check import targets as TG
from circuit_check import vectors as VX

REPO = Path(__file__).resolve().parents[4]
OPTS = CK.Options()
MEMO: dict = {}
PINS = CK.load_pins()
SUITE = TG.suite()
TEMPLATES = {label: (sub, sts) for label, sub, sts in TG.templates()}

_known = KN.known


def _clean(entry: dict) -> None:
    new = [f["name"] for f in entry["failures"] if _known(f) is None]
    assert not new, "\n".join(new)
    known = sorted({_known(f) for f in entry["failures"]} - {None})
    if known:
        pytest.xfail(" | ".join(known))


# --- coverage -----------------------------------------------------------------------------------------------------------------

def test_every_registered_definition_is_checked():
    """Every registered Definition family is checked at a binding the catalog reaches, and every catalog root builds: a new
    Definition without one fails here by name (add its binding to its module's sidecar, `circuit_check.sidecars`)."""
    assert not TG.ROOT_ERRORS, TG.ROOT_ERRORS
    assert len(TG.REGISTERED) > 200
    checked = {TG.family(fn) for fn in SUITE.values()}
    missing = sorted(set(TG.REGISTERED) - checked)
    assert not missing, f"registered Definitions no circuit-check binding reaches: {missing}"


def test_the_registries_import_every_module_with_a_sidecar_and_key_every_target_on_them():
    """A Definition module is registered by having a sidecar, not by a line in `targets.py`, so the set of them is in every
    target's key (`cache.registries`): a new one reruns every target, as a new import line did."""
    import hashlib
    import sys

    from circuit_check import cache as CC
    from circuit_check import sidecars as SC

    assert len(SC.discover()) > 50 and not sorted(set(SC.discover()) - set(sys.modules))
    want = hashlib.sha256("\n".join(sorted(SC.discover())).encode()).hexdigest()
    assert CC.key({})["registries"] == want


def test_every_operation_is_checked_and_reaches_no_table_outside_the_library():
    """The guard does not visit an operation (`verity.ml.circuits.operations`), so a table an operation reached would be hidden: every
    listed operation has a specialization here, and none reaches a primitive that holds a table the library does not list."""
    from verity.ml.circuits import constants as K
    from verity.ml.circuits import operations as O

    ops = [fn for fn in SUITE.values() if O.is_operation(fn)]
    assert {fn.base_id for fn in ops} == O.OPERATIONS
    for fn in ops:
        held = sorted(p.id for p in TG.reachable([fn]).values() if K.held_tables(p, O.is_library_table))
        assert not held, (fn.id, held)


def test_every_template_is_checked():
    from verity_numerical.bench import templates as TM

    assert {sub.template.id for sub, _ in TEMPLATES.values()} == set(TM.TEMPLATES)


def _descriptor_digest(fn) -> str:
    """A target's canonical descriptor digest: `program_digest` of the one-call Program at its statics."""
    from verity.ml.circuits import codec
    from verity.ml.circuits.defs import CompositeDefinition, bind
    from verity.ml.circuits.program import Program

    params, ret = fn.signature()
    shell = CompositeDefinition(f"Show[{fn.id}]", 1, (), lambda S: (tuple(params), ret), lambda B, S, *a: B.call(fn, *a),
                                register=False)
    return codec.program_digest(codec.encode_program(Program(bind(shell))))


SPLIT = Path(__file__).parent / "fixtures" / "split"
#: split fixtures another test reads, in that split script's own format
SPLIT_READERS = {"prims_serve": "test_split_prims_serve.py"}


def test_every_split_fixture_another_test_reads_is_read():
    for stem, reader in SPLIT_READERS.items():
        assert (SPLIT / f"{stem}.json").exists(), stem
        assert f"{stem}.json" in (Path(__file__).parent / reader).read_text(), (stem, reader)


@pytest.mark.parametrize("pinned", sorted(p for p in SPLIT.glob("*.json") if p.stem not in SPLIT_READERS), ids=lambda p: p.stem)
def test_a_split_family_keeps_its_targets_and_digests(pinned):
    """`fixtures/split/<family>.json` was written before that family's registry modules were split into Definitions, twins and
    vocabulary (`integrations/vllm/tools/migrate/split_<family>.py pin`). Every target it lists is still checked, at the same
    canonical descriptor digest, and every Definition family it lists is still registered. A new binding may add targets."""
    pin = json.loads(pinned.read_text())
    missing = sorted(set(pin["definitions"]) - set(SUITE))
    assert not missing, f"{pin['family']}: targets no longer checked: {missing}"
    moved = {fid: (want, got) for fid, want in pin["definitions"].items() if (got := _descriptor_digest(SUITE[fid])) != want}
    assert not moved, f"{pin['family']}: descriptor digests changed: {moved}"
    assert not sorted(set(pin["registered"]) - set(TG.REGISTERED)), pin["family"]


def test_the_partition_applies_to_the_calls_of_served_programs():
    calls = TG.call_families()
    assert {"Gemm_v1", "RMSNormTriton_v1", "Attention_v3", "TokenSelect_v1", "GumbelTopPTokenSelect_v1", "RoPEHead_v1"} <= calls
    assert not {"AttnBlock_v3", "DotBf16_v1", "Serve_v1", "TopPMask_v1"} & calls


def test_an_id_only_the_catalog_registers_resolves_through_the_suite(monkeypatch):
    """`circuit-check <id>` finds a suite target that the registries hold only once the catalog's roots are traced (the gathers
    `boolean_gather.gather_bf16(n)` builds) or whose id writes a dict static as its digest (`gf2k.GfResiduals`'s schema), and
    an id neither knows still raises."""
    from verity_vllm.query import word as W

    real = W.specialization

    def unregistered(fam, statics):
        if fam.startswith("GatherBf16x"):
            raise KeyError(f"primitive {fam} is not in the registry")
        return real(fam, statics)

    monkeypatch.setattr(W, "specialization", unregistered)
    monkeypatch.setattr(TG, "suite", lambda: SUITE)
    for fid in ("GatherBf16x64_v2", "GatherBf16x1100_v2", "GatherBf16x1024u10_v1"):
        assert TG.definition(fid) is SUITE[fid]
    res = [fid for fid in SUITE if fid.startswith("GfResiduals_v1{S={")]
    assert res and all(TG.definition(fid) is SUITE[fid] for fid in res)
    with pytest.raises(KeyError):
        TG.definition("NoSuchDefinition_v9")


def test_a_target_with_a_record_static_parses_as_its_id_prints_it(monkeypatch):
    """Every suite target whose statics hold a record (printed as a digest, ``C={d0dc81b7fa41}``) parses back from its id to
    itself, the record also at statics the suite does not bind, and a digest of no record the suite binds raises."""
    import re

    monkeypatch.setattr(TG, "suite", lambda: SUITE)
    records = [fid for fid in SUITE if re.search(r"=\{[0-9a-f]{12}\}", fid)]
    assert "StepBody_v1{C={e2f2f43eb046},T=4,GEMM={4130e3ca726d},ATTENTION={35db6738c571},NORM={4d524c30089f}}" in records
    for fid in records:
        assert TG.definition(fid).id == fid
    step = next(fid for fid in records if fid.startswith("StepBody_v1{"))
    other = TG.definition(step.replace(",T=4,", ",T=6,"))
    assert other.id == step.replace(",T=4,", ",T=6,") and other.bindings["C"] is SUITE[step].bindings["C"]
    with pytest.raises(KeyError, match="no Definition the suite checks"):
        TG.definition("StepBody_v1{C={000000000000},T=4}")


# --- the suite ----------------------------------------------------------------------------------------------------------------

@pytest.mark.circuit_suite
@pytest.mark.parametrize("fid", sorted(SUITE))
def test_definition(fid, results):
    e = CK.check_definition(SUITE[fid], OPTS, memo=MEMO, pins=PINS)
    results.append({**e, "kind": "definition", "target": fid})
    _clean(e)


@pytest.mark.circuit_suite
@pytest.mark.parametrize("label", sorted(TEMPLATES))
def test_template(label, results):
    sub, sts = TEMPLATES[label]
    r = CK.check_subcircuit(sub, OPTS, instance_statics=sts, memo=MEMO, pins=PINS)
    results.append({**r, "kind": "template", "target": label})
    assert all(low["lowered"] or low.get("reason") for low in r["lowering"])
    _clean(r)


def _request_program(tmp_path: Path, **kw) -> Path:
    """The vLLM tests' synthetic request Program (per token Embedding -> RMSNorm -> Gemm -> RoPE -> SiluMul -> Gemm -> RMSNorm, per
    step the LM head and TokenSelect; ``attn``: the attention and its o_proj, whose output is a leaf) as ``instances.json.gz``."""
    path = REPO / "integrations" / "vllm" / "tests" / "check" / "test_sampled_replay.py"
    spec = importlib.util.spec_from_file_location("circuit_check_request_program", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    inst, _w = mod._program(**kw)
    out = tmp_path / "instances.json.gz"
    out.write_bytes(gzip.compress(json.dumps(inst).encode()))
    return out


@pytest.mark.slow
def test_row(tmp_path, results):
    r = CK.check_row(_request_program(tmp_path, rope=True), OPTS, memo=MEMO, pins=PINS)
    results.append({**r, "kind": "row", "target": "synthetic request Program"})
    assert r["calls"] == 42 and len(r["definitions"]) >= 8
    _clean(r)


@pytest.mark.slow
def test_a_call_output_no_commitment_holds_is_named(tmp_path):
    """The attention fixture's o_proj Gemm returns a value no later Call reads and the query does not require: a unit would end
    at an uncommitted value, across Calls."""
    r = CK.check_row(_request_program(tmp_path, attn=True), OPTS, memo=MEMO, pins=PINS)
    assert "partition/output-not-committed Gemm_v1{K=48,N=16} (4 Call(s) under model.model.layers.0.self_attn.o_proj)" in \
        [f["name"] for f in r["failures"]]


@pytest.mark.circuit_suite
def test_known_failures_are_current():
    """Each known entry still fails somewhere: a fixed one is dropped from `circuit_check.known`."""
    fails = [f for fid in sorted(SUITE) for f in CK.check_definition(SUITE[fid], OPTS, memo=MEMO, pins=PINS)["failures"]]
    assert not KN.stale(fails), KN.stale(fails)


# --- the vectors ----------------------------------------------------------------------------------------------------------------

def test_the_vectors_start_with_every_edge():
    from verity.ml.circuits.types import Array, Value

    a, x = VX.vectors([Value(32), Array(3, Value(16))], 40, 1)
    assert a.dtype == np.uint32 and x.shape == (40, 3) and x.dtype == np.uint16
    e32, e16 = VX.edges(32), VX.edges(16)
    for w in (0x7FC00000, 0xFFC00000, 0x7F800001, 0x7F800000, 0xFF800000, 0, 0x80000000, 1, 0x007FFFFF, 0x7F7FFFFF, 0xFF7FFFFF):
        assert w in e32
    for w in (0x7FC0, 0x7F81, 0x7F80, 0xFF80, 0x8000, 0x0001, 0x007F, 0x7F7F, 0x7C00, 0x7E00, 0x7BFF):   # bf16 and fp16
        assert w in e16
    for w in (0x7F, 0xFF, 0x7E, 0x01, 0x7C, 0x7B):                                                  # e4m3 and e5m2
        assert w in VX.edges(8)
    assert a[:len(e32)].tolist() == list(e32) and (x[:len(e16)] == np.array(e16)[:, None]).all()


# --- registered tables ----------------------------------------------------------------------------------------------------------

#: a table of 2^12 two-word entries (8,192 leaves), past `TABLE_OPTS.input_budget`, held by `_HOLDS` as a registered input
TABLE_IB = 12
TABLE_OPTS = dataclasses.replace(OPTS, input_budget=1024)
TABLE_REGIONS = ((0, 1000), (1000, 1 << TABLE_IB))


def _table_entry(i: int) -> int:
    return (i * 0x9E3779B1 + 0x7F4A7C15) & 0xFFFFFFFF


def _table_word(j: int) -> int:
    return (_table_entry(j >> 1) >> (16 * (j & 1))) & 0xFFFF


def _holds():
    from verity.ml.circuits.defs import PrimitiveDefinition
    from verity.ml.circuits.types import Value

    return PrimitiveDefinition("CircuitCheckHolds", 1, (("x", Value(32)),), Value(32), _table_entry, register=False,
                               tables=(("cc_table", "registered"),))


def _table_read(lazy: bool = True):
    """(index, table) -> the table's two-word entry at the index's low bits, as `TableRead` reads it."""
    from verity.ml.circuits.defs import PrimitiveDefinition
    from verity.ml.circuits.types import Array, Value

    mask = (1 << TABLE_IB) - 1
    p = PrimitiveDefinition("CircuitCheckTableRead", 1, (("index", Value(32)), ("table", Array((2 << TABLE_IB) // 64, Array(64, Value(16))))),
                            Value(32), lambda idx, *w: w[2 * (idx & mask)] | w[2 * (idx & mask) + 1] << 16, register=False)
    if lazy:
        p.evaluate_lazy = lambda get: get(1 + 2 * (get(0) & mask)) | get(2 + 2 * (get(0) & mask)) << 16  # type: ignore[attr-defined]
    return p


def _table_probe(holds):
    from verity.ml.circuits.evaluation import RegisteredTable

    return VX.TableProbe(1, RegisteredTable("cc_table", _table_word, 2 << TABLE_IB, holds), 2, TABLE_REGIONS,
                         lambda e: [[e | (0x5A << 24)]])


def test_the_probed_entries_are_the_ends_and_each_regions_edges():
    assert VX.probe_entries(1 << 27, [(0, 0x01102CB4), (0x01102CB4, 0x01B89BA3), (0x01B89BA3, 1 << 27)]) == \
        [0, 0x01102CB3, 0x01102CB4, 0x01B89BA2, 0x01B89BA3, (1 << 27) - 1]
    assert VX.probe_entries(8, []) == [0, 7]
    with pytest.raises(ValueError, match=r"region \[4, 9\) is not within the table's 8 entries"):
        VX.probe_entries(8, [(4, 9)])


def test_a_registered_table_is_probed_and_never_flattened(monkeypatch):
    """A Definition whose table parameter is over the input budget is checked on its other parameters' vectors plus one
    instance per probed entry, each reading the table lazily; the record names the probed entries and counts the words read."""
    from verity.ml.circuits.evaluation import evaluate

    read, holds = _table_read(), _holds()
    probe = _table_probe(holds)
    monkeypatch.setitem(TG.TABLE_PROBES, read.id, lambda: probe)
    monkeypatch.setattr(VX.RegisteredTable, "__iter__", lambda self: pytest.fail("the table was flattened"), raising=False)
    params, _ = CK.signature_of(read)
    instances, got, _rec = VX.table_instances(params, probe, CK._n_vectors(read, TABLE_OPTS, 1), TABLE_OPTS.seed)
    for args in instances:
        evaluate(read, *args)
    assert got == {2 * (args[0] & ((1 << TABLE_IB) - 1)) + k for args in instances for k in (0, 1)}
    e = CK.check_definition(read, TABLE_OPTS, memo={}, pins=PINS)
    assert not e["failures"], e["failures"]
    corr = e["correspondence"]
    assert corr["table"] == {"name": "cc_table", "param": 1, "entries": 1 << TABLE_IB, "leaves": 2 << TABLE_IB,
                             "regions": [list(r) for r in TABLE_REGIONS], "probed_entries": [0, 999, 1000, (1 << TABLE_IB) - 1],
                             "probe_instances": 4, "words_read": len(got)}
    assert corr["vectors"] == len(instances) == TABLE_OPTS.vectors + 4 and len(got) < 2 << TABLE_IB
    assert corr["realizations"] == [{"via": "c-flock:ir_lower", "lowered": False, "reason": "a lowering takes every input leaf, and "
                                     "this Definition's correspondence reads a registered table lazily"}]


def test_a_probe_that_reads_past_the_table_is_a_partial_reference(monkeypatch):
    """An entry index one past the last (an off-by-one in ``evaluate_lazy``) raises on the last entry's probe, by name."""
    read = _table_read()
    mask = (1 << TABLE_IB) - 1
    read.evaluate_lazy = lambda get: get(3 + 2 * (get(0) & mask)) | get(2 + 2 * (get(0) & mask)) << 16  # type: ignore[attr-defined]
    probe = _table_probe(_holds())
    monkeypatch.setitem(TG.TABLE_PROBES, read.id, lambda: VX.TableProbe(probe.param, probe.table, 2, (), lambda e: [[e]]))
    e = CK.check_definition(read, dataclasses.replace(TABLE_OPTS, vectors=4, min_vectors=4), memo={}, pins=PINS)
    (f,) = e["failures"]
    assert f["name"].startswith("correspondence/reference-not-total CircuitCheckTableRead_v1 raised on ")
    assert any("IndexError" in x["error"] for x in f["examples"])


def test_a_parameter_over_the_input_budget_without_a_probe_fails_by_name():
    e = CK.check_definition(_table_read(), TABLE_OPTS, memo={}, pins=PINS)
    assert [f["name"] for f in e["failures"]] == [
        "correspondence/unprobed-table CircuitCheckTableRead_v1 parameter 1 has 8192 leaves, over the input budget 1024, and no "
        "registered table probes it (`circuit_check.targets.TABLE_PROBES`)"]


def test_a_definition_over_the_leaf_budget_is_not_partitioned(monkeypatch):
    """The word query and `Q_call` resolve every parameter leaf: over ``partition_max_leaves`` the partition is a warning, not run."""
    read = _table_read()
    monkeypatch.setitem(TG.TABLE_PROBES, read.id, lambda: _table_probe(_holds()))
    e = CK.check_definition(read, dataclasses.replace(TABLE_OPTS, partition_max_leaves=4096), memo={}, pins=PINS, as_call=True)
    assert not e["failures"]
    assert [w["name"] for w in e["warnings"]] == [
        "partition/unchecked CircuitCheckTableRead_v1: 1 its parameters' 8193 leaves are over the partition's leaf budget 4096: the "
        "word query and Q_call resolve every leaf"]
    assert e["partition"]["checked"] is False and "q_call" not in e["partition"]
    assert CK.check_definition(read, TABLE_OPTS, memo={}, pins=PINS, as_call=True)["partition"]["q_call"]["checked"]


# --- negatives: each check fires, by name ---------------------------------------------------------------------------------------

def _composite(name, sig, body):
    from verity.ml.circuits.defs import CompositeDefinition, bind

    return bind(CompositeDefinition(name, 1, (), lambda S: sig, body, register=False))


def _f32x2():
    from verity.ml.circuits.types import Array, Value

    return (("x", Array(2, Value(32))),)


def test_a_value_computed_twice_in_one_unit_is_a_redundant_gate():
    from verity.ml.circuits.types import Value
    from verity_vllm.program.registry import prims as P

    fn = _composite("CircuitCheckRedundant", (_f32x2(), Value(32)),
                    lambda B, S, x: B.call(P.F32Mul, B.call(P.F32Add, x[0], x[1]), B.call(P.F32Add, x[0], x[1])))
    e = CK.check_definition(fn, OPTS, as_call=True)
    (w,) = [w for w in e["warnings"] if w["check"] == "redundant-gates" and w["code"] == "ir"]
    assert w["count"] == 1 and w["where"] == {"F32Add_v1 @ CircuitCheckRedundant_v1": 1}
    assert not [f for f in e["failures"] if f["check"] == "partition"]


def test_a_value_computed_in_two_units_is_a_recompute_failure():
    """Two identical calls, each returned by its own unit (the body reads only its parameters, which `word.unit_rule` partitions
    node by node; the whole Definition's cut compares them)."""
    from verity.ml.circuits.refs import array_of
    from verity.ml.circuits.types import Array, Value
    from verity_vllm.program.registry import prims as P

    fn = _composite("CircuitCheckRecompute", (_f32x2(), Array(2, Value(32))),
                    lambda B, S, x: array_of([B.call(P.F32Add, x[0], x[1]), B.call(P.F32Add, x[0], x[1])]))
    e = CK.check_definition(fn, OPTS, as_call=True)
    assert "partition/gate-recomputed CircuitCheckRecompute_v1 (1 gate(s))" in [f["name"] for f in e["failures"]]
    assert e["partition"]["q_word"] == 1                                # a word Call keeps v1: the recompute fails
    assert e["partition"]["whole"]["q_word_v2"] == {"codes": [], "recomputed_across": 1}   # v2 reports it, refuses nothing
    inner = CK.check_definition(fn, OPTS, as_call=False)                 # not a Call: its copies are redundant gates, not a failure
    assert not [f for f in inner["failures"] if f["check"] == "partition"] and inner["partition"]["isolated"]
    assert [w["count"] for w in inner["warnings"] if w["check"] == "redundant-gates" and w["code"] == "ir"] == [1]


#: a Boolean Call family (`RoPEHead_v1`'s Boolean version) whose cut computes 1,120 values in two units
BOOL_RECOMPUTE = "RoPEHead_v2{D=8}"


def _partition(fn, q_word, opts=OPTS):
    F = CK.Findings()
    rec, _rule = CK.partition(fn, opts, F, {}, applies=True, q_word=q_word)
    rec.pop("_graph", None)
    return rec, [f["name"] for f in F.failures if f["check"] == "partition"]


def test_a_boolean_call_that_recomputes_across_units_passes_with_the_count():
    """A Boolean Call is judged by `Q_word` v2: the values two units compute are recorded, in the entry and the summary, not
    failed; under v1 the same cut fails by name."""
    from verity.ml.circuits import boolean as BL

    fn = TG.definition(BOOL_RECOMPUTE)
    assert BL.is_boolean(fn) and TG.family(fn) in TG.call_families()
    rec, failed = _partition(fn, 2)
    assert not failed and rec["q_word"] == 2 and rec["recomputed_across"] == 1120
    assert CK.summarize([{"definition": fn.id, "partition": rec, "failures": [], "warnings": []}])["recomputed_across"] == {fn.id: 1120}
    _rec, failed = _partition(fn, 1)
    assert failed == [f"partition/gate-recomputed {fn.id} (1120 gate(s))"]
    rec, failed = _partition(fn, 2, dataclasses.replace(OPTS, partition_max_gates=10))   # over the budget: unchecked, no count
    assert not failed and not rec["checked"] and rec["recomputed_across"] is None


def test_a_boolean_call_with_another_partition_code_still_fails(monkeypatch):
    """Under v2 only `gate-recomputed` is reported instead of refused: a Boolean Call whose cut has any other code fails by name."""
    from verity.ml.circuits.partition import CutVerdict
    from verity_vllm.query import word as W

    real = W.units

    def wide(G, *a, **k):
        R = real(G, *a, **k)
        c = R["cut"]
        detail = {**c.detail, "unit-too-wide": {"n": 1, "gates": [[0, 17, 2]], "limits": [16, 32, 0]}}
        return {**R, "cut": CutVerdict(False, tuple(c.codes) + ("unit-too-wide",), detail)}

    monkeypatch.setattr(W, "units", wide)
    fn = TG.definition(BOOL_RECOMPUTE)
    rec, failed = _partition(fn, 2)
    assert failed == [f"partition/unit-too-wide {fn.id} (1 gate(s))"], failed
    assert rec["recomputed_across"] == 1120 and rec["whole"]["q_word_v2"]["codes"] == ["unit-too-wide"]


def test_q_call_is_recorded_per_call_and_refuses_a_gate_wider_than_32_bits():
    """`Q_call` v1 beside `Q_word`: a Boolean Call's units, committed bits and the values two units compute (reported); a Call with
    a 64-bit gate is refused, a warning and the summary's `q_call_refused`, not a failure."""
    from verity.ml.circuits.defs import PrimitiveDefinition
    from verity.ml.circuits.types import Array, Value

    fn = TG.definition(BOOL_RECOMPUTE)
    rec, failed = _partition(fn, 2)
    assert not failed and rec["q_call"] == QCALL_ROPE
    add64 = PrimitiveDefinition("CircuitCheckAdd64", 1, (("a", Value(64)), ("b", Value(64))), Value(64), lambda a, b: (a + b) % (1 << 64),
                                register=False)
    wide = _composite("CircuitCheckWide64", ((("a", Array(2, Value(64))),), Value(64)), lambda B, S, a: B.call(add64, a[0], a[1]))
    F = CK.Findings()
    rec, _rule = CK.partition(wide, OPTS, F, {}, applies=True)
    assert rec["q_call"]["inapplicable"] == {"definition": wide.id, "gate": 0, "primitive": add64.id, "width": 64, "X": 32, "why": "gate-too-wide"}
    assert [w["name"] for w in F.warnings if w["code"] == "q-call-inapplicable"] == [f"partition/q-call-inapplicable {wide.id}: 1 gate-too-wide"]
    assert not [f for f in F.failures if f["code"].startswith("q-call")]
    assert CK.summarize([{"definition": wide.id, "partition": rec, "failures": [], "warnings": []}])["q_call_refused"] == {wide.id: "gate-too-wide"}


#: `Q_call{32}` on `BOOL_RECOMPUTE`: its eight 16-bit outputs, two to a unit, and the same values computed in two units as v2 reports
QCALL_ROPE = {"X": 32, "checked": True, "units": 8, "committed": 128, "committed_bits": 128, "recomputed_across": 1120, "redundant_gates": 0,
              "codes": []}


def test_the_boolean_versions_of_the_served_calls_are_calls():
    """Sigma's image of the word Call families (`boolean_version`, the switch's `word=` table) is held as Calls."""
    calls = TG.call_families()
    assert {"RMSNormFusedCuda_v3", "RMSNormTriton_v2", "Gemm_v3", "Attention_v8", "RoPE_v2", "SiluMul_v3", "TokenSelect_v2",
            "Embedding_v2"} <= calls
    assert {"RMSNormFusedCuda_v2", "RMSNormTriton_v1", "Gemm_v1"} <= calls
    assert not {"DotBf16_v3", "AttnBlock_v6"} & calls


def test_every_call_the_suite_checks_is_partitioned():
    """No Call passes with its partition unchecked: each is within `Options.partition_max_gates` (the Boolean attention Calls
    at T = 17 are about 1.9M gates, 2.1M with Gemma-2's softcap) and `Options.partition_max_leaves`."""
    from verity.ml.circuits.evaluation.reference import signature_of

    calls = TG.call_families()
    over = {k: fn.gates for k, fn in SUITE.items() if TG.family(fn) in calls and getattr(fn, "gates", 0) > OPTS.partition_max_gates}
    assert not over
    wide = {k: n for k, fn in SUITE.items() if TG.family(fn) in calls
            and (n := sum(pt.leaves for pt in signature_of(fn)[0])) > OPTS.partition_max_leaves}
    assert not wide


def test_a_boolean_family_over_the_budget_keeps_its_boolean_binding():
    """The suite keeps a family's smallest specialization over `MAX_GATES`; a Boolean norm's smallest binds the word MUFU, so its
    smallest Boolean binding (the one the Program on bits calls) is checked too."""
    from verity.ml.circuits import boolean as BL
    for word, bits in [("RMSNormFusedCuda_v3{N=16,EPS=1e-05,RSQRT=RsqrtApprox_v1}", "RMSNormFusedCuda_v3{N=16,EPS=1e-05,RSQRT=RsqrtApprox_v2}"),
                       ("RMSNormTriton_v2{N=16,EPS=1e-05,SQRT=MufuSqrtFtz_v1,SCALEA=DivFullScaleA_v1,RCP=DivFullRcp_v1}",
                        "RMSNormTriton_v2{N=16,EPS=1e-05,SQRT=MufuSqrtFtz_v2,SCALEA=DivFullScaleA_v3,RCP=DivFullRcp_v2}")]:
        assert {word, bits} <= set(SUITE) and SUITE[bits].gates > TG.MAX_GATES
        assert BL.is_boolean(SUITE[bits]) and not BL.is_boolean(SUITE[word])


def test_a_gate_no_output_reads_is_a_dead_gate():
    from verity.ml.circuits.types import Value
    from verity_vllm.program.registry import prims as P

    def body(B, S, x):
        B.call(P.F32Mul, x[0], x[1])
        return B.call(P.F32Add, x[0], x[1])

    e = CK.check_definition(_composite("CircuitCheckDead", (_f32x2(), Value(32)), body), OPTS)
    (w,) = [w for w in e["warnings"] if w["check"] == "dead-gates"]
    assert w["count"] == 1 and w["where"] == {"F32Mul_v1 @ CircuitCheckDead_v1": 1}


def test_a_unit_wider_than_the_word_fails_the_width_rule():
    from verity.ml.circuits.defs import PrimitiveDefinition
    from verity.ml.circuits.types import Value

    wide = PrimitiveDefinition("CircuitCheckWide", 1, (("a", Value(32)), ("b", Value(32))), Value(64), lambda a, b: (a << 32) | b,
                               register=False)
    fn = _composite("CircuitCheckWidth", ((("a", Value(32)), ("b", Value(32))), Value(64)), lambda B, S, a, b: B.call(wide, a, b))
    names = [f["name"] for f in CK.check_definition(fn, OPTS, as_call=True)["failures"]]
    assert any(n.startswith("partition/width CircuitCheckWidth_v1 output:CircuitCheckWide_v1") for n in names), names


def test_a_partial_reference_is_named():
    from verity.ml.circuits.defs import PrimitiveDefinition
    from verity.ml.circuits.types import Value

    def finite_only(a):
        if (a >> 23) & 0xFF == 0xFF:
            raise ValueError("non-finite")
        return a

    prim = PrimitiveDefinition("CircuitCheckPartial", 1, (("a", Value(32)),), Value(32), finite_only, register=False)
    (f,) = CK.check_definition(prim, OPTS)["failures"]
    assert f["name"].startswith("correspondence/reference-not-total CircuitCheckPartial_v1 raised on")


def test_a_kernel_wrong_only_on_edge_words_is_caught():
    """A kernel that returns 0 for a NaN operand: uniform random words rarely hold a NaN, the edge vectors always do."""
    from verity.ml.circuits.evaluation import register_kernel
    from verity.ml.circuits.evaluation.batch import _KERNELS
    from verity.ml.circuits.defs import PrimitiveDefinition
    from verity.ml.circuits.types import Value

    prim = PrimitiveDefinition("CircuitCheckKernel", 1, (("a", Value(32)),), Value(32), lambda a: a ^ 1, register=False)
    nan = lambda a: ((a >> 23) & 0xFF == 0xFF) & ((a & 0x7FFFFF) != 0)  # noqa: E731
    register_kernel(prim, lambda a: (np.where(nan(a), 0, a ^ 1).astype(np.uint32), None))
    try:
        e = CK.check_definition(prim, OPTS)
    finally:
        del _KERNELS[prim.id]
    (f,) = e["failures"]
    assert f["name"].startswith("correspondence/kernel-mismatch CircuitCheckKernel_v1 [numpy]")


def test_a_lowering_that_differs_from_the_ir_is_caught(monkeypatch):
    from verity_flock import ir_lower as IL
    from verity_vllm.program.registry import prims as P

    add = IL.PIECES["F32Add_v1"]
    monkeypatch.setitem(IL.PIECES, "F32Mul_v1", lambda C, a, b: add(C, a, b))
    e = CK.check_definition(P.F32Mul, OPTS, pins=PINS)
    assert any(f["name"].startswith("correspondence/lowering-mismatch F32Mul_v1 [c-flock]") for f in e["failures"])


def test_a_lowering_count_off_its_pin_is_named():
    from verity_vllm.program.registry import prims as P

    pins = {"pieces": {"F32Mul_v1": {"and": 1}}, "units": {}}
    e = CK.check_definition(P.F32Mul, OPTS, pins=pins)
    assert any(f["name"].startswith("lowering/pin-mismatch F32Mul_v1 [F32Mul_v1]: and 1 -> ") for f in e["failures"])


def test_redundant_ands_are_counted():
    from verity_flock import gf2

    C = gf2.Circuit()
    a, b = C.input("a", 2)
    x = C.AND(a, b)
    y = C.AND(b, a)                       # the same product again
    z = C.AND(x ^ b, y ^ b)               # equal operands once y is x
    C.output("o", [x, y, z])
    assert CK.redundant_ands(C)[0] == 2


@pytest.mark.slow
def test_the_cli_writes_a_report_and_fails_by_name(tmp_path, capsys):
    from circuit_check import cli

    # Every target here is a Call of a served Program (`targets.call_families`), so `--as-call` is the default it would resolve;
    # passing it skips building the whole catalog in each run's worker (~10 min each).
    call = ["--as-call", "-q"]
    out = tmp_path / "r.json"
    assert cli.main(["RoPEHead_v1{D=8}", BOOL_RECOMPUTE, "--out", str(out), *call]) == 0
    r = json.loads(out.read_text())
    assert r["schema"] == CK.SCHEMA and [t["target"] for t in r["targets"]] == ["RoPEHead_v1{D=8}", BOOL_RECOMPUTE]
    assert r["summary"]["ok"] and all(t["kind"] == "definition" for t in r["targets"])
    assert cli.main(["RoPEHead_v1{D=8}", "--lower-max-ands", "1", "--out", str(out), *call]) == 0
    [real] = json.loads(out.read_text())["targets"][0]["correspondence"]["realizations"]
    assert not real["lowered"] and real["reason"].endswith("over the whole-Definition budget (1)")
    capsys.readouterr()
    fp8 = "ScaledMmFp8Block_v1{K=128,N=128,G=128}"
    assert cli.main([fp8, *call]) == 0                                           # a known failure: named, not failing
    assert f"known partition/gate-recomputed {fp8}" in capsys.readouterr().out
    assert cli.main([fp8, *call, "--strict"]) == 1
    capsys.readouterr()
    assert cli.main([BOOL_RECOMPUTE, *call]) == 0
    assert "1 Boolean Call(s) recomputing 1120 gate(s) across units (Q_word v2)" in capsys.readouterr().out


def test_a_result_after_a_read_the_tracer_cannot_see_is_not_stored(tmp_path):
    """A check that starts a program (its reads are untraced) is not stored, nor is any later one in that process; a library
    built in the tree and loaded by the dynamic loader is an input by its bytes."""
    import subprocess
    import sys

    script = f"""
import json, ctypes.util
from pathlib import Path
from circuit_check import cache as CC, checks as CK, cli
CC.trace(cli.REPO)
ok = lambda: {{"failures": [], "warnings": []}}
def spawns():
    import subprocess
    subprocess.run(["true"])
    return ok()
cli._tasks = lambda *a: [("definition", "A", ok), ("definition", "B", spawns), ("definition", "C", ok)]
rep = cli.run([], CK.Options(), log=None, cache=Path({str(tmp_path)!r}))
print(json.dumps(rep["cache"]))
"""
    r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, r.stderr
    stats = json.loads(r.stdout.strip().splitlines()[-1])
    assert stats["stored"] == 1 and [u["target"] for u in stats["untraced"]] == ["B", "C"]
    assert "subprocess.Popen" in stats["untraced"][0]["because"][0]


def test_a_check_that_raised_or_read_another_trees_module_is_not_stored(tmp_path):
    """A check that raised is no verdict (T11: an ImportError from a tree a concurrent run was rewriting, replayed on the next
    tree); a module imported from outside the tree is a read no key holds, so that result and later ones are not stored."""
    import subprocess
    import sys

    (tmp_path / "elsewhere").mkdir()
    (tmp_path / "elsewhere" / "other_tree_mod.py").write_text("X = 1\n")
    script = f"""
import json, sys
from pathlib import Path
from circuit_check import cache as CC, checks as CK, cli
CC.trace(cli.REPO)
ok = lambda: {{"failures": [], "warnings": []}}
def broken():
    from verity_numerical.bench import no_such_module  # noqa: F401
def outside():
    sys.path.insert(0, {str(tmp_path / 'elsewhere')!r})
    import other_tree_mod  # noqa: F401
    return ok()
cli._tasks = lambda *a: [("definition", "A", ok), ("definition", "B", broken), ("definition", "C", outside), ("definition", "D", ok)]
rep = cli.run([], CK.Options(), log=None, cache=Path({str(tmp_path / 'cache')!r}))
print(json.dumps(rep["cache"]))
"""
    r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, r.stderr
    stats = json.loads(r.stdout.strip().splitlines()[-1])
    assert stats["stored"] == 1 and stats["errors"] == ["B"] and [u["target"] for u in stats["untraced"]] == ["C", "D"]
    assert "imported other_tree_mod from" in stats["untraced"][0]["because"][0] and "outside the tree" in stats["untraced"][0]["because"][0]


def test_a_result_is_not_stored_when_the_tree_changed_while_it_ran(tmp_path):
    """The run remembers each read's digest from its first look; the process that ran the check digests its reads again right
    after it. When they differ (a concurrent run rewrote the tree), nothing is stored: the report may be of a tree no digest names."""
    import subprocess

    from circuit_check import cache as CC

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "m.py").write_text("")
    got = {"files": ["a.txt"], "dirs": ["pkg"], "probes": [], "libs": []}
    CC._VISIBLE.clear()
    CC._MEMO.clear()
    k, rep = {"version": "v", "request": {}}, {"failures": [], "warnings": []}
    CC._digests(tmp_path, got)                                          # the run's first look, before the check
    (tmp_path / "pkg" / "m.py").unlink()                                # a concurrent run removes a file while it runs
    CC._VISIBLE.clear()
    assert CC.store(tmp_path / "cache", k, tmp_path, got, rep, CC.seen(tmp_path, got)) is None
    assert not list((tmp_path / "cache").rglob("*.json"))
    (tmp_path / "pkg" / "m.py").write_text("")
    CC._VISIBLE.clear()
    assert CC.store(tmp_path / "cache", k, tmp_path, got, rep, CC.seen(tmp_path, got))    # the same tree: stored
    CC._VISIBLE.clear()
    CC._MEMO.clear()


@pytest.mark.slow
@pytest.mark.skipif(not os.environ.get("VERITY_CIRCUIT_CHECK_SLOW"),
                    reason="8 min on a node, nearly all of it each worker's cold start: VERITY_CIRCUIT_CHECK_SLOW=1")
def test_parallel_jobs_give_the_serial_report(tmp_path):
    """It checks four real targets serially and in worker processes, so it catches a `--jobs` worker whose report differs from
    the serial one. Opt-in: run it after a change to `cli.run`, `cli.parallel` or the task list's order."""
    from circuit_check import cli

    specs = ["F32Mul_v1", "RMSNormTriton_v1{N=64,EPS=1e-05}", "TokenSelect_v1{V=64}", "rope-head"]
    strip = lambda rep: [{k: v for k, v in t.items() if k != "seconds"} for t in rep["targets"]]  # noqa: E731
    assert strip(cli.run(specs, OPTS, log=None, jobs=3)) == strip(cli.run(specs, OPTS, log=None, jobs=1))


def test_an_isolated_target_stays_traced():
    """`parallel` forks each item from its zygote, and the child keeps the tracer's hook: the fork is the checker's own, so it
    must not make every isolated target's verdict uncacheable."""
    import operator

    from circuit_check import cache as CC
    from circuit_check import cli

    (got,) = [v for _, v, _, _ in cli.parallel(operator.call, [CC.untraced], 1, CC.trace, (cli.REPO,))]
    assert got == []


def test_the_cache_reads_the_tree_as_git_has_it(tmp_path):
    """A build or bytecode inside the tree (ignored by git) changes no recorded read; a new file git sees does."""
    import subprocess

    from circuit_check import cache as CC

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text("build/\n")
    (tmp_path / "a.txt").write_text("a")
    got = {"files": ["a.txt", "build/lib.so"], "dirs": ["."], "probes": ["build/lib.so", "b.txt"]}
    CC._VISIBLE.clear()
    CC._MEMO.clear()
    before = CC._digests(tmp_path, got)
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "lib.so").write_bytes(b"built")
    CC._VISIBLE.clear()
    CC._MEMO.clear()
    assert CC._digests(tmp_path, got) == before
    (tmp_path / "b.txt").write_text("b")
    CC._VISIBLE.clear()
    CC._MEMO.clear()
    after = CC._digests(tmp_path, got)
    assert after["dirs"] != before["dirs"] and after["probes"]["b.txt"] == "file"
    CC._VISIBLE.clear()
    CC._MEMO.clear()


def test_the_cache_key_is_the_same_on_hosts_that_differ_only_in_their_kernel(monkeypatch):
    """Pods of one image run on hosts with different kernels; a report one pod stored must count on another."""
    import platform

    from circuit_check import cache as CC

    here = CC.key({"target": "F32Mul_v1"})
    monkeypatch.setattr(platform, "release", lambda: "0.0.0-another-host")
    monkeypatch.setattr(platform, "version", lambda: "#1 SMP another build")
    monkeypatch.setattr(platform, "platform", lambda *a, **k: "Linux-0.0.0-another-host-x86_64")
    assert CC.key({"target": "F32Mul_v1"}) == here
    monkeypatch.setattr(platform, "machine", lambda: "aarch64")
    assert CC.key({"target": "F32Mul_v1"}) != here                  # a different CPU is a different machine


def test_the_cache_key_is_the_same_in_every_check_slot(monkeypatch):
    """Each check slot keeps its tests' results apart (`VERITY_TEST_CACHE`, `tools/verity/check/slot.py`); a report one slot
    stored must count in another, while a variable that can change what a check computes still keys it."""
    from circuit_check import cache as CC

    monkeypatch.setenv("VERITY_TEST_CACHE", "/home/research/.cache/verity/tests-slot-a")
    here = CC.key({"target": "F32Mul_v1"})
    monkeypatch.setenv("VERITY_TEST_CACHE", "/home/research/.cache/verity/tests-slot-b")
    assert CC.key({"target": "F32Mul_v1"}) == here
    monkeypatch.setenv("VERITY_REGRESSION", "1" if os.environ.get("VERITY_REGRESSION") != "1" else "0")
    assert CC.key({"target": "F32Mul_v1"}) != here
