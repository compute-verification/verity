"""circuit-check's coverage of every registered Definition and subcircuit template, a served workload's request Program, and the
checks' own negatives (each check fires by name on a circuit built to break it).  `circuit-check --all`, which `check` runs, is
where every target is checked.

A failure is a correctness or invariant violation, and fails its test: whether one may stay red is drive's policy
(`research.policy`), never the checker's.  Redundant and dead gates are warnings: counted in the report, never a test failure
here.

The tests that read the suite (the `suite` fixture) or the Calls it gives (`targets.call_families`) are marked slow: the process
that runs one builds circuit-check's catalog first (conftest.py), from the bindings' records when they count and binding by
binding without them, and only the whole suite shows what they hold (every registered family checked, every Call partitioned).
The two walks of every Call over the partition budget (`_unchecked_walk`) are opt-in (`_opt_in`), and the nightly cold check runs
them by name (check.py's `OPT_IN`): `circuit-check --all` fails a Call whose partition nothing checks by its name, and the walks
hold `targets.CALLEE_CALLS` and `OVER_PARTITION_BUDGET` current.
A test that selects targets by family reads the family off the catalog (`Catalog.reached`), so it makes only those.
"""
from __future__ import annotations

import dataclasses
import gzip
import importlib.util
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pytest

from circuit_check import checks as CK
from circuit_check import targets as TG
from circuit_check import vectors as VX

REPO = Path(__file__).resolve().parents[4]
OPTS = CK.Options()
MEMO: dict = {}
PINS = CK.load_pins()
TEMPLATES = {label: (sub, sts) for label, sub, sts in TG.templates()}


def _clean(entry: dict) -> None:
    assert not entry["failures"], "\n".join(f["name"] for f in entry["failures"])


def _opt_in(takes: str):
    """The mark of a test that takes ``takes`` on a node: it stays out of `check` and runs where VERITY_CIRCUIT_CHECK_SLOW asks."""
    return pytest.mark.skipif(not os.environ.get("VERITY_CIRCUIT_CHECK_SLOW"),
                              reason=f"{takes} on a node: VERITY_CIRCUIT_CHECK_SLOW=1")


# --- coverage -----------------------------------------------------------------------------------------------------------------

@pytest.mark.slow
def test_every_registered_definition_is_checked(catalog):
    """Every registered Definition family is checked at a binding the catalog reaches, and every catalog root builds: a new
    Definition without one fails here by name (add its binding to its module's sidecar, `circuit_check.sidecars`)."""
    assert not TG.ROOT_ERRORS, TG.ROOT_ERRORS
    assert len(TG.REGISTERED) > 200
    checked = {catalog.reached[fid][0] for fid in catalog.suite}
    missing = sorted(set(TG.REGISTERED) - checked)
    assert not missing, f"registered Definitions no circuit-check binding reaches: {missing}"


def test_the_registries_import_every_module_with_a_sidecar_and_key_the_task_list_on_them():
    """A Definition module is registered by having a sidecar, not by a line in `targets.py`, so the set of them is in the task
    list's key (`cache.registries`), and a target's holds none of them: it holds whether a module whose import registered what
    its check read has a sidecar (``registrars``)."""
    import hashlib
    import sys

    from circuit_check import cache as CC
    from circuit_check import sidecars as SC

    assert len(SC.discover()) > 50 and not sorted(set(SC.discover()) - set(sys.modules))
    want = hashlib.sha256("\n".join(sorted(SC.discover())).encode()).hexdigest()
    assert CC.key({"target": ["task-list", "--all"]})["registries"] == want
    assert "registries" not in CC.key({"target": ["definition", "F32Mul_v1"]})


@pytest.mark.slow
def test_every_operation_is_checked_and_reaches_no_table_outside_the_library(catalog):
    """The guard does not visit an operation (`verity.definitions.circuits.operations`), so a table an operation reached would be hidden: every
    listed operation has a specialization here, and none reaches a primitive that holds a table the library does not list."""
    from verity.definitions.circuits import constants as K
    from verity.definitions.circuits import operations as O

    suite = catalog.suite
    ops = [fn for fid in suite if catalog.reached[fid][0] in O.OPERATIONS and O.is_operation(fn := suite[fid])]
    assert {fn.base_id for fn in ops} == O.OPERATIONS
    for fn in ops:
        held = sorted(p.id for p in TG.reachable([fn]).values() if K.held_tables(p, O.is_library_table))
        assert not held, (fn.id, held)


def test_every_template_is_checked():
    from verity_flock import subcircuit_templates as TM

    assert {sub.template.id for sub, _ in TEMPLATES.values()} == set(TM.TEMPLATES)


def _descriptor_digest(fn) -> str:
    """A target's canonical descriptor digest: `program_digest` of the one-call Program at its statics."""
    from verity.circuits import codec
    from verity.circuits.defs import CompositeDefinition, bind
    from verity.circuits.program import Program

    params, ret = fn.signature()
    shell = CompositeDefinition(f"Show[{fn.id}]", 1, (), lambda S: (tuple(params), ret), lambda B, S, *a: B.call(fn, *a),
                                register=False)
    return codec.program_digest(codec.encode_program(Program(bind(shell))))


SPLIT = Path(__file__).parent / "fixtures" / "split"
#: split fixtures another test reads, in that test's own format
SPLIT_READERS = {"prims_serve": "test_split_prims_serve.py"}


def test_every_split_fixture_another_test_reads_is_read():
    for stem, reader in SPLIT_READERS.items():
        assert (SPLIT / f"{stem}.json").exists(), stem
        assert f"{stem}.json" in (Path(__file__).parent / reader).read_text(), (stem, reader)


@pytest.mark.slow
@pytest.mark.parametrize("pinned", sorted(p for p in SPLIT.glob("*.json") if p.stem not in SPLIT_READERS), ids=lambda p: p.stem)
def test_a_split_family_keeps_its_targets_and_digests(pinned, suite):
    """`fixtures/split/<family>.json` was written before that family's registry modules were split into Definitions, twins and
    vocabulary. Every target it lists is still checked, at the same canonical descriptor digest, and every Definition family it
    lists is still registered. A new binding may add targets."""
    pin = json.loads(pinned.read_text())
    missing = sorted(set(pin["definitions"]) - set(suite))
    assert not missing, f"{pin['family']}: targets no longer checked: {missing}"
    moved = {fid: (want, got) for fid, want in pin["definitions"].items() if (got := _descriptor_digest(suite[fid])) != want}
    assert not moved, f"{pin['family']}: descriptor digests changed: {moved}"
    assert not sorted(set(pin["registered"]) - set(TG.REGISTERED)), pin["family"]


@pytest.mark.slow
@pytest.mark.usefixtures("suite")
def test_the_partition_applies_to_the_calls_of_served_programs():
    calls = TG.call_families()
    assert {"Gemm_v1", "RMSNormTriton_v1", "Attention_v3", "TokenSelect_v1", "GumbelTopPTokenSelect_v1", "RoPEHead_v1"} <= calls
    assert not {"AttnBlock_v3", "DotBf16_v1", "Serve_v1", "TopPMask_v1"} & calls


def _without_the_catalog(monkeypatch) -> None:
    """`is_call` from scratch, failing a test that builds the catalog or asks for its Calls."""
    monkeypatch.setattr(TG, "_CALLS", set())
    monkeypatch.setattr(TG, "_SHORT", {})
    monkeypatch.setattr(TG, "suite", lambda: pytest.fail("the catalog was built"))
    monkeypatch.setattr(TG, "call_families", lambda found=None: pytest.fail("call_families was asked"))


@pytest.mark.slow
def test_is_call_holds_a_family_exactly_when_call_families_does(monkeypatch, catalog):
    """`is_call` decides without the catalog by a superset of the Boolean Calls (`boolean_call_candidates`, the Boolean table's
    links) and a subset of them (`certified_boolean_calls`, the catalog's roots), and refuses the rest (`Undecided`) rather than
    build the catalog: on every family the suite checks, every registered one and every Call, each one it decides it decides as
    `call_families` does, and it decides most."""
    calls = TG.call_families()
    words = TG.word_call_families()
    assert TG.certified_boolean_calls(words) <= calls
    _without_the_catalog(monkeypatch)
    fams = set(TG.REGISTERED) | {catalog.reached[fid][0] for fid in catalog.suite} | calls
    decided: dict[str, bool] = {}
    undecided: dict[str, str] = {}
    for f in fams:
        try:
            decided[f] = TG.is_call(f)
        except TG.Undecided as e:
            undecided[f] = str(e)
    assert {f for f, call in decided.items() if call} == set(decided) & calls
    assert len(undecided) < len(fams) // 10, f"most families need no catalog: {sorted(undecided.values())[:20]}"


@pytest.mark.slow
def test_the_boolean_call_candidates_drop_only_families_boolean_version_refuses(catalog):
    """`boolean_call_candidates`, narrowed to the statics a binding of the word Call family could give, drops only Boolean
    families `boolean_version` refuses: on the real catalog, the Boolean version `boolean_version` accepts for each suite
    Definition of a word Call family (`boolean_call_families`) is a candidate, unless its family is a word Call family itself,
    which is decided as one first.  `units.call_families` looks for ``--all``'s Boolean Calls only among the candidates, and
    `is_call` says a family outside them is no Call, so a family dropped here would leave both short of `call_families`."""
    words = TG.word_call_families()
    suite = catalog.suite
    accepted = TG.boolean_call_families(words, {fid: suite[fid] for fid in suite if catalog.reached[fid][0] in words}) - words
    assert accepted, "the served word Calls have Boolean versions"
    missing = sorted(accepted - TG.boolean_call_candidates(words))
    assert not missing, f"Boolean versions boolean_version accepts that the candidates drop: {missing}"


def test_a_boolean_version_whose_statics_no_word_call_binds_is_not_a_candidate(monkeypatch):
    """A Boolean Definition linked to its word by name is the Boolean version only at a binding that gives each of its statics
    (`verity.definitions.boolean._find`), so one with a static no word Call family of its name has is not a candidate, and
    `is_call` says it is no Call without the catalog: a new port shape (``{K,R}`` beside the word Call's ``{K}``) is not
    refused."""
    from types import SimpleNamespace

    from verity.definitions import boolean as BV

    TG.load_registries()
    exact, by_name, errors = BV._load()
    wider = SimpleNamespace(name="Int8RowQuant", version=99, statics=("K", "R"))
    monkeypatch.setattr(BV, "_load", lambda: (exact, {**by_name, "Int8RowQuant": [*by_name.get("Int8RowQuant", ()), wider]}, errors))
    got = TG.boolean_call_candidates({"Int8RowQuant_v1"})
    assert "Int8RowQuant_v2" in got and "Int8RowQuant_v99" not in got


def test_a_family_is_call_cannot_decide_alone_is_refused_by_name_with_the_flags_that_decide_it(monkeypatch):
    """`is_call` never builds the catalog for one family: a Boolean candidate no catalog root certifies is `Undecided`, and a
    targeted run fails on that target by name, naming ``--as-call`` and ``--no-as-call``, in place of building it."""
    from circuit_check import cli

    _without_the_catalog(monkeypatch)
    TG.word_call_families()
    TG._SHORT["candidates"] = {"RoPEHead_v2"}
    monkeypatch.setattr(TG, "certified_boolean_calls", lambda words, only=None: set())
    with pytest.raises(TG.Undecided, match="give --as-call or --no-as-call"):
        TG.is_call("RoPEHead_v2")
    (target,) = cli.run(["RoPEHead_v2{D=8}"], OPTS, log=None, isolate=False)["targets"]
    (f,) = target["failures"]
    assert f["code"] == "error" and "Undecided" in f["name"] and "--no-as-call" in f["name"], f["name"]


@pytest.mark.slow
@pytest.mark.usefixtures("suite")
def test_a_boolean_version_made_on_demand_is_a_call_by_its_id(monkeypatch):
    """A word Call family made on demand (a `Const<w>[0x..]` or a gather) has its Boolean version, made on demand too, among the
    Calls by its id: no module registers one ahead, so the catalog can't name it, and `is_call` decides it without the catalog."""
    words = {"Const32[0x00000000]_v1", "GatherI32x16_v1", "GatherBf16x64_v1", "F32Add_v1"}
    made = {"Const32[0x00000000]_v2", "GatherI32x16_v2", "GatherBf16x64_v2"}
    assert TG.on_demand_boolean_families(words) == made
    assert made <= TG.boolean_call_candidates(words) and made <= TG.certified_boolean_calls(words)
    assert {"Const32[0x00000000]_v2", "GatherI32x16_v2"} <= TG.call_families()
    _without_the_catalog(monkeypatch)
    assert TG.is_call("Const32[0x00000000]_v2") and TG.is_call("GatherI32x16_v2")


def test_the_boolean_versions_made_on_demand_are_made_in_sorted_order(monkeypatch):
    """Making a Boolean version on demand registers it, so `on_demand_boolean_families` makes them in sorted order: the
    registry's order, which a slice's shared part notes (`units.slices`' ``families``), is not a set's."""
    from verity.definitions import boolean

    asked: list[str] = []
    monkeypatch.setattr(boolean, "made_on_demand", asked.append)
    words = ["GatherI32x16_v1", "Const32[0x00000000]_v1", "GatherI32x129280_v1", "F32Add_v1"]
    for order in (words, words[::-1]):
        asked.clear()
        assert TG.on_demand_boolean_families(set(order)) == set()
        assert asked == sorted(words), asked


#: one Definition target for each way `is_call` decides: a vocabulary kind's family, a template's, a Boolean version one of their
#: bindings certifies, a word composite and a Boolean one outside the Boolean table's bound, and a primitive
DECIDED = {"TokenSelect_v1{V=16}": "words", "RoPEHead_v1{D=8}": "words", "RoPEHead_v2{D=8}": "certified",
           "DotBf16_v1{K=16}": "not a candidate", "DotBf16_v3{K=16,DOT=AmpereBF16TcDot16_v3}": "not a candidate",
           "F32Add_v1": "not a candidate"}


@pytest.mark.slow
@pytest.mark.usefixtures("suite")
def test_a_targeted_run_without_the_catalog_reports_what_one_with_it_does(monkeypatch):
    """A targeted run does not build the whole catalog (`call_families`) before checking a target.  The same
    targets checked with each `as_call` read off `call_families` and with it decided by `is_call` without the catalog get the
    same report."""
    from circuit_check import cli

    TG.call_families()
    full = cli.run(list(DECIDED), OPTS, log=None, isolate=False)
    _without_the_catalog(monkeypatch)
    fast = cli.run(list(DECIDED), OPTS, log=None, isolate=False)
    short = TG._SHORT
    assert {t: "words" if TG.family(TG.definition(t)) in short["words"] else
            "certified" if TG.family(TG.definition(t)) in short["certified"] else
            "not a candidate" if TG.family(TG.definition(t)) not in short["candidates"] else "?" for t in DECIDED} == DECIDED

    def strip(rep):
        return {**{k: v for k, v in rep.items() if k != "prologue_s"},
                "targets": [{k: v for k, v in t.items() if k not in ("seconds", "seconds_by_check")} for t in rep["targets"]]}

    assert strip(fast) == strip(full)
    assert [t["partition"]["applies"] for t in full["targets"]] == [v != "not a candidate" for v in DECIDED.values()]


def test_the_cli_resolves_each_targets_call_without_the_catalog(monkeypatch):
    """The process the targets' processes fork from resolves each Definition target's `is_call` (`cli._calls`), without the
    catalog for any target `is_call` decides alone; one that doesn't resolve is left to its check."""
    from circuit_check import cli

    _without_the_catalog(monkeypatch)
    attention = "Attention_v8{T=40,NH=2,KVH=1,D=16,BN=16,DOT=HopperBF16WgmmaDot16_v2,INV=Fa2InvSum_v2,MASKED_FROM=2}"
    cli._calls(["TokenSelect_v1{V=16}", "RoPEHead_v2{D=8}", attention, "F32Add_v1", "rope-head", "RoPEHead_v1{Q=8}"])
    assert {"TokenSelect_v1", "RoPEHead_v1"} <= TG._SHORT["words"]
    assert {"RoPEHead_v2", "Attention_v8"} <= TG._SHORT["certified"], "a template's binding, and a root function's"


@pytest.mark.slow
def test_an_id_only_the_catalog_registers_resolves_through_the_suite(monkeypatch, suite):
    """`circuit-check <id>` finds a suite target that the registries hold only once the catalog's roots are traced (the gathers
    `boolean_gather.gather_bf16(n)` builds) or whose id writes a dict static as its digest (`gf2k.GfResiduals`'s schema), and
    an id neither knows still raises."""
    from verity.definitions import word as W

    real = W.specialization

    def unregistered(fam, statics):
        if fam.startswith("GatherBf16x"):
            raise KeyError(f"primitive {fam} is not in the registry")
        return real(fam, statics)

    monkeypatch.setattr(W, "specialization", unregistered)
    for fid in ("GatherBf16x64_v2", "GatherBf16x1100_v2", "GatherBf16x1024u10_v1"):
        assert TG.definition(fid) is suite[fid]
    res = [fid for fid in suite if fid.startswith("GfResiduals_v1{S={")]
    assert res and all(TG.definition(fid) is suite[fid] for fid in res)
    with pytest.raises(KeyError):
        TG.definition("NoSuchDefinition_v9")


@pytest.mark.slow
def test_a_target_with_a_record_static_parses_as_its_id_prints_it(suite):
    """Every suite target whose statics hold a record (printed as a digest, ``C={d0dc81b7fa41}``) parses back from its id to
    itself, the record also at statics the suite does not bind, and a digest of no record the suite binds raises."""
    import re

    records = [fid for fid in suite if re.search(r"=\{[0-9a-f]{12}\}", fid)]
    assert "StepBody_v1{C={e2f2f43eb046},T=4,GEMM={4130e3ca726d},ATTENTION={35db6738c571},NORM={4d524c30089f}}" in records
    for fid in records:
        assert TG.definition(fid).id == fid
    step = next(fid for fid in records if fid.startswith("StepBody_v1{"))
    other = TG.definition(step.replace(",T=4,", ",T=6,"))
    assert other.id == step.replace(",T=4,", ",T=6,") and other.bindings["C"] is suite[step].bindings["C"]
    with pytest.raises(KeyError, match="no Definition the suite checks"):
        TG.definition("StepBody_v1{C={000000000000},T=4}")


# --- a served workload ----------------------------------------------------------------------------------------------------------

def _request_program(tmp_path: Path, **kw) -> Path:
    """The vLLM tests' synthetic request Program (per token Embedding -> RMSNorm -> Gemm -> RoPE -> SiluMul -> Gemm -> RMSNorm, per
    step the LM head and TokenSelect; ``attn``: the attention and its o_proj, whose output is a leaf) as ``instances.json.gz``."""
    path = REPO / "adapters" / "vllm" / "tests" / "check" / "test_sampled_replay.py"
    spec = importlib.util.spec_from_file_location("circuit_check_request_program", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    inst, _w = mod._program(**kw)
    out = tmp_path / "instances.json.gz"
    out.write_bytes(gzip.compress(json.dumps(inst).encode()))
    return out


@pytest.mark.slow
def test_row(tmp_path):
    r = CK.check_row(_request_program(tmp_path, rope=True), OPTS, memo=MEMO, pins=PINS)
    assert r["calls"] == 42 and len(r["definitions"]) >= 8
    _clean(r)


@pytest.mark.slow
def test_a_call_output_no_commitment_holds_is_named(tmp_path):
    """The attention fixture's o_proj Gemm returns a value no later Call reads and the query does not require: a proof block
    would end at an uncommitted value, across Calls."""
    r = CK.check_row(_request_program(tmp_path, attn=True), OPTS, memo=MEMO, pins=PINS)
    assert "partition/output-not-committed Gemm_v1{K=48,N=16} (4 Call(s) under model.model.layers.0.self_attn.o_proj)" in \
        [f["name"] for f in r["failures"]]


# --- the vectors ----------------------------------------------------------------------------------------------------------------

def test_the_vectors_start_with_every_edge():
    from verity.circuits.types import Array, Value

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


def test_a_caller_keeps_room_for_random_vectors_after_fewer_edges():
    from verity.circuits.types import Value

    e16 = VX.edges(16)
    (a,) = VX.vectors([Value(16)], 6, 1, edge=4)
    (b,) = VX.vectors([Value(16)], 6, 1)
    assert a[:4].tolist() == list(e16[:4]) and b.tolist() == list(e16[:6]) and a[4:].tolist() != list(e16[4:6])


def _f32_sum():
    from verity.circuits.types import Value
    from verity.definitions.registry import prims as P

    return _composite("CircuitCheckFloorSum", (_f32x2(), Value(32)), lambda B, S, x: B.call(P.F32Add, x[0], x[1]))


def test_a_realization_compares_at_least_two_random_vectors_within_the_budget():
    """Past ``reference_budget`` the vector rule leaves ``min_vectors`` vectors, every one all-edge (an f32 word has 17 edges):
    the floor adds two random ones while the reference's work on them is within ``random_budget``, so the lowering compares
    the Definition on a random vector.  Past the budget it compares on edge vectors only (``vectors/edge-only``, a warning);
    a Definition no realization takes keeps its vectors and is compared on none (``vectors/none``)."""
    fn, floor = _f32_sum(), dataclasses.replace(OPTS, reference_budget=1)

    def run(opts):
        e = CK.check_definition(fn, opts, memo={}, pins=PINS)
        _clean(e)
        c = e["correspondence"]
        low = next(r for r in c["realizations"] if r["via"] == "c-flock:ir_lower")
        return e, (c["vectors"], c["edge_vectors"], low.get("compared"), low.get("random"))

    e, got = run(floor)
    assert got == (6, 4, 6, 2) and CK.vector_class(e) == "random" and CK.classified(e) is e
    e, got = run(dataclasses.replace(floor, random_budget=1))
    assert got == (4, 4, 4, 0) and CK.vector_class(e) == "edge-only"
    assert [w["name"] for w in CK.classified(e)["warnings"] if w["check"] == "vectors"] == [
        "vectors/edge-only CircuitCheckFloorSum_v1: 1 its realizations compared it with the reference on all-edge vectors only, "
        "on no random one"]
    e, got = run(dataclasses.replace(floor, lower_max_gates=0))
    assert got == (4, 4, None, None) and CK.vector_class(e) == "none"
    assert [w["code"] for w in CK.classified(e)["warnings"] if w["check"] == "vectors"] == ["none"]


def test_a_kernel_takes_its_floor_from_its_own_sampler():
    """No lowering takes a Definition Flock has no piece for, so past ``reference_budget`` its vectors stay all-edge; each
    kernel is then compared on two vectors its own sampler samples in its domain (a random vector over many leaves almost always
    holds a word a finite-only kernel declines).  One right on every edge word and wrong elsewhere fails by name; one whose
    sampler samples only outside its domain compared on edge vectors only; past ``random_budget`` none is sampled."""
    from verity.circuits.defs import PrimitiveDefinition
    from verity.circuits.evaluation import register_kernel
    from verity.circuits.evaluation.batch import _KERNELS
    from verity.circuits.types import Value

    prim = PrimitiveDefinition("CircuitCheckFloorKernel", 1, (("a", Value(16)),), Value(16), lambda a: a ^ 1, register=False)
    edges = np.array(VX.edges(16), dtype=np.uint16)

    def finite(rng, n, params, **statics):
        return [rng.integers(0, 0x7F00, n).astype(np.uint16)]

    register_kernel(prim, lambda a: (a ^ 1, (a & 0x7F80) == 0x7F80), kernel="finite", partial=True, sample=finite)
    register_kernel(prim, lambda a: (np.where(np.isin(a, edges), a ^ 1, a), None), kernel="edges-only", sample=finite)
    register_kernel(prim, lambda a: (a ^ 1, ~np.isin(a, edges)), kernel="outside", partial=True, sample=finite)
    floor = dataclasses.replace(OPTS, reference_budget=1)
    try:
        e = CK.check_definition(prim, floor, pins=PINS)
        over = CK.check_definition(prim, dataclasses.replace(floor, random_budget=1), pins=PINS)
    finally:
        del _KERNELS[prim.id]
    assert (e["correspondence"]["vectors"], e["correspondence"]["edge_vectors"]) == (4, 4)
    via = {r["via"]: (r["compared"], r["random"], r.get("sampled")) for r in e["correspondence"]["realizations"]
           if r["via"].startswith("kernel:")}
    assert via == {"kernel:edges-only": (4, 2, {"vectors": 2, "compared": 2, "mismatches": 2}),
                   "kernel:finite": (4, 2, {"vectors": 2, "compared": 2, "mismatches": 0}),
                   "kernel:outside": (4, 0, {"vectors": 2, "compared": 0, "mismatches": 0})}
    assert [f["name"] for f in e["failures"]] == [
        "correspondence/kernel-mismatch CircuitCheckFloorKernel_v1 [edges-only]: 2 of the 2 sampled vectors"]
    assert CK.vector_class(e) == "random"
    assert not over["failures"] and CK.vector_class(over) == "edge-only"
    assert all("sampled" not in r for r in over["correspondence"]["realizations"])


def test_the_summary_counts_and_names_the_targets_by_vector_class():
    """Each target is ``random``, ``edge-only`` or ``none`` by its realizations (a row by its worst Definition, a template by its
    Definitions', target's and proof blocks'), each named; those read lazily are named too; a by-calls root and a constant (no
    input, so no vector to sample) are none of them."""
    def entry(name, *recs, **corr):
        corr = {"vectors": 4, "edge_vectors": 4, **corr}
        return {"definition": name, "correspondence": {"realizations": list(recs), **corr}, "failures": [], "warnings": []}

    out = [CK.classified(r) for r in (
        entry("R", {"via": "kernel:numpy", "compared": 6, "random": 2}),
        entry("E", {"via": "kernel:numpy", "compared": 4, "random": 0}),
        entry("N", {"via": "kernel:numpy", "declines": "a kernel takes arrays"}, edge_vectors=0, wide=[{"param": 1}]),
        {"row": "build", "definitions": [entry("R2", {"compared": 3, "random": 1}), entry("N2", {"lowered": False})],
         "failures": [], "warnings": []},
        {"subcircuit": "rope-head", "definitions": [entry("D", {"compared": 4, "random": 0})],
         "lowering": [{"lowered": True, "compared": 3, "random": 1}], "failures": [], "warnings": []},
        {"definition": "B", "mode": "by-calls", "failures": [], "warnings": []},
        entry("C", {"via": "c-flock:ir_lower", "lowered": False, "reason": "a constant: circuit structure, no gates"},
              vectors=64, edge_vectors=0))]
    assert CK.vector_class(out[-1]) is None and out[-1]["warnings"] == []
    s = CK.summarize(out)
    assert s["vectors"] == {"random": {"count": 2, "targets": ["R", "rope-head"]}, "edge-only": {"count": 1, "targets": ["E"]},
                            "none": {"count": 2, "targets": ["N", "build"]}, "lazy": {"count": 1, "targets": ["N"]}}
    assert s["warnings"]["vectors/edge-only"]["by_definition"] == {"E": 1}
    assert s["warnings"]["vectors/none"]["by_definition"] == {"N": 1, "build": 1}


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
    from verity.circuits.defs import PrimitiveDefinition
    from verity.circuits.types import Value

    return PrimitiveDefinition("CircuitCheckHolds", 1, (("x", Value(32)),), Value(32), _table_entry, register=False,
                               tables=(("cc_table", "registered"),))


def _table_read(lazy: bool = True):
    """(index, table) -> the table's two-word entry at the index's low bits, as `TableRead` reads it."""
    from verity.circuits.defs import PrimitiveDefinition
    from verity.circuits.types import Array, Value

    mask = (1 << TABLE_IB) - 1
    p = PrimitiveDefinition("CircuitCheckTableRead", 1, (("index", Value(32)), ("table", Array((2 << TABLE_IB) // 64, Array(64, Value(16))))),
                            Value(32), lambda idx, *w: w[2 * (idx & mask)] | w[2 * (idx & mask) + 1] << 16, register=False)
    if lazy:
        p.evaluate_lazy = lambda get: get(1 + 2 * (get(0) & mask)) | get(2 + 2 * (get(0) & mask)) << 16  # type: ignore[attr-defined]
    return p


def _table_probe(holds):
    from verity.circuits.evaluation import RegisteredTable

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
    vector per probed entry, each reading the table lazily; the record names the probed entries and counts the words read."""
    from verity.circuits.evaluation import evaluate

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
    assert e["conformance"] == "not-yet-established"
    assert [f["name"] for f in e["failures"]] == [
        "correspondence/unprobed-table CircuitCheckTableRead_v1 parameter 1 has 8192 leaves, over the input budget, and is neither a "
        "registered table a probe reads (`circuit_check.targets.TABLE_PROBES`) nor an operand read lazily "
        "(`circuit_check.targets.LAZY_OPERANDS`)"]


def _wide_gemm():
    """`Gemm_v2` at K=32, N=64, with an input budget its weight's 2,048 leaves are over and 32 vectors of its x fit in."""
    from verity.circuits.defs import bind
    from verity.definitions.ops import gemm as G
    from verity.definitions.ops import prims as P

    return bind(G.Gemm, K=32, N=64, DOT=P.AmpereBF16TcDot16), dataclasses.replace(OPTS, input_budget=1024, vectors=40)


def test_an_operand_over_the_input_budget_is_read_lazily_on_the_vectors_words(monkeypatch):
    """A weight over the input budget (`targets.LAZY_OPERANDS`) takes the edge and random words every parameter takes, one
    vector at a time, and is read lazily, never flattened; the reference on each vector equals it on the same words flattened."""
    from verity.circuits.evaluation import RegisteredTable, evaluate

    t, opts = _wide_gemm()
    (x, w), _ = CK.signature_of(t)
    monkeypatch.setattr(RegisteredTable, "__iter__", lambda self: pytest.fail("the operand was flattened"), raising=False)
    F = CK.Findings()
    rec, arrays, ref = CK.correspondence(t, opts, F)
    assert not F.failures and arrays is None
    n, n_edge = rec["vectors"], rec["edge_vectors"]
    assert n == CK._n_vectors(t, opts, [1]) == 32 and n_edge == len(VX.edges(16)) < n
    assert rec["wide"] == [{"param": 1, "name": "w", "leaves": 2048, "width": 16, "reads": rec["wide"][0]["reads"]}]
    assert rec["wide"][0]["reads"] >= 2048 * n
    assert rec["realizations"] == [{"via": f"kernel:{k}", "declines": "a kernel takes arrays, not 'w' read lazily"}
                                   for k in sorted(CK.kernels(t))]
    xs = VX.vectors([x], n, opts.seed)
    for i in range(n):
        words = [VX.wide_words(w, i, n_edge, opts.seed, 1)(k) for k in range(w.leaves)]
        assert ref[i] == tuple(evaluate(t, VX.instance_args([x], xs, i)[0], words))
    e = CK.check_definition(t, opts, memo={}, pins=PINS)
    assert [f["name"] for f in e["failures"] if f["check"] == "correspondence"] == []
    assert {"via": "c-flock:ir_lower", "lowered": False, "reason": "a lowering takes every input leaf, and this Definition's "
            "correspondence reads an operand over the input budget lazily"} in e["correspondence"]["realizations"]


def test_an_operand_read_lazily_past_the_random_budget_takes_one_edge_vector_and_the_random_floor():
    """Where the vectors `Options.min_vectors` adds past what `Options.reference_budget` pays for would take the reference's work
    past `Options.random_budget` (`checks._lazy_vectors`), a Definition read lazily takes ``random_floor + 1`` vectors: one
    all-edge, the others random in every operand, each the reference on the same words flattened.  Within it, it keeps
    `_n_vectors`' (the test above)."""
    from verity.circuits.evaluation import evaluate

    t, opts = _wide_gemm()
    (x, w), _ = CK.signature_of(t)
    opts = dataclasses.replace(opts, reference_budget=t.gates - 1, random_budget=opts.min_vectors * t.gates - 1)
    assert CK._lazy_vectors(t, dataclasses.replace(opts, random_budget=opts.min_vectors * t.gates), [1]) == (opts.min_vectors, False)
    F = CK.Findings()
    rec, arrays, ref = CK.correspondence(t, opts, F)
    assert not F.failures and arrays is None
    n = rec["vectors"]
    assert (n, rec["edge_vectors"]) == (opts.random_floor + 1, 1) and n < opts.min_vectors
    assert rec["wide"][0]["reads"] >= w.leaves * n
    xs = VX.vectors([x], n, opts.seed, 1)
    for i in range(n):
        words = [VX.wide_words(w, i, 1, opts.seed, 1)(k) for k in range(w.leaves)]
        assert ref[i] == tuple(evaluate(t, VX.instance_args([x], xs, i)[0], words))
        assert (len(set(words)) == 1) == (i == 0)


def _wide_experts(E: int):
    """`MoeExpertFp8Gemm_v1` over E experts at the sidecar's small statics, with an input budget its expert stack's E N K leaves
    are over and 8 vectors of its other parameters fit in."""
    from verity.circuits.defs import bind
    from verity.definitions.ops.prims import BlackwellE4m3QmmaDot32
    from verity.definitions.registry import fp8_moe_experts as X

    return (bind(X.MoeExpertFp8Gemm, E=E, K=128, N=80, G=64, KB=32, SWAP=False, DOT=BlackwellE4m3QmmaDot32),
            dataclasses.replace(OPTS, input_budget=8192, vectors=8))


def test_an_expert_gemm_resolves_only_the_expert_a_vector_selects(monkeypatch):
    """An expert GEMM's stack read lazily: each vector that selects an expert reads its N K weights and no other's, the walk's
    parameter resolutions (`verity.circuits.layout._arg_ref`) are the same over 4 experts as over 16, so a vector costs
    what its expert's rows cost and not every expert's, and the reference equals it on the same words flattened.  On random
    words and operands (the vectors here are edge vectors, whose experts' words are all equal), for each expert and for none, the
    lazy evaluation's whole trace is the eager walk's."""
    import random

    from verity.circuits import layout
    from verity.circuits.evaluation import evaluate, evaluate_region
    from verity.circuits.evaluation.reference import standalone_call

    resolutions = {}
    for E in (4, 16):
        t, opts = _wide_experts(E)
        params, _ = CK.signature_of(t)
        calls = [0]
        resolve_arg = layout._arg_ref

        def counted(*a, _f=resolve_arg, _c=calls):
            _c[0] += 1
            return _f(*a)

        monkeypatch.setattr(layout, "_arg_ref", counted)
        F = CK.Findings()
        rec, _arrays, ref = CK.correspondence(t, opts, F)
        monkeypatch.setattr(layout, "_arg_ref", resolve_arg)
        assert not F.failures and rec["probes"] == [{"e": E - 1}]
        n = rec["vectors"] - 1
        other = [params[j] for j in (0, 1, 2, 4)]
        es = [VX.instance_args(other, VX.vectors(other, n, opts.seed), i)[2] for i in range(n)]
        selecting = sum(0 <= v < E for v in es) + 1
        assert 2 <= selecting < rec["vectors"] and rec["wide"][0]["reads"] == selecting * 80 * 128
        resolutions[E] = calls[0]
        if E == 4:
            xs = VX.vectors(other, n, opts.seed)
            for i in range(n):
                xq, sx, e, ws = VX.instance_args(other, xs, i)
                words = [VX.wide_words(params[3], i, rec["edge_vectors"], opts.seed, 3)(k) for k in range(params[3].leaves)]
                assert ref[i] == tuple(evaluate(t, xq, sx, e, words, ws))
            rng = random.Random(opts.seed)
            xq, sx, words, ws = ([rng.getrandbits(w) for w in params[j].leaf_widths()] for j in (0, 1, 3, 4))
            prog, call = standalone_call(t)
            for e in range(E + 1):
                bound = dict(zip(call.args, [*xq, *sx, e, *words, *ws]))
                lazy = evaluate_region(prog.circuit, call, dict(bound))
                assert lazy == evaluate_region(prog.circuit, list(call), dict(bound)), e
    assert resolutions[4] == resolutions[16], resolutions


def _wide_shard():
    """`EmbeddingShard_v1` at VS=64, H=32, START=1000, with an input budget its table's 2,048 leaves are over: no edge or
    random token lies in the shard."""
    from verity.circuits.defs import bind
    from verity.definitions.registry import b1_tp2

    return bind(b1_tp2.EmbeddingShard, VS=64, H=32, START=1000), dataclasses.replace(OPTS, input_budget=1024, vectors=8)


def test_an_operand_probe_sets_the_index_that_reads_the_operands_first_and_last_rows():
    """A shard's edge and random tokens lie outside it, so only its probes (`targets.OPERAND_PROBES`: the shard's first and last
    token) read the table, one row each, on random words; the reference there equals it on the same words flattened."""
    from verity.circuits.evaluation import evaluate

    t, opts = _wide_shard()
    (_tok, table), _ = CK.signature_of(t)
    F = CK.Findings()
    rec, _arrays, ref = CK.correspondence(t, opts, F)
    assert not F.failures and not F.warnings
    n = CK._n_vectors(t, opts, [1])
    assert rec["probes"] == [{"tok": 1000}, {"tok": 1063}] and rec["vectors"] == n + 2
    assert rec["wide"][0]["reads"] == 2 * 32
    for q, tok in enumerate((1000, 1063)):
        words = [VX.wide_words(table, n + q, rec["edge_vectors"], opts.seed, 1)(k) for k in range(table.leaves)]
        assert ref[n + q] == tuple(evaluate(t, tok, words)) != (0,) * 32


def test_an_operand_no_vector_reads_is_unchecked_and_a_probe_of_a_non_scalar_is_refused(monkeypatch):
    t, opts = _wide_shard()
    monkeypatch.setitem(TG.OPERAND_PROBES, "EmbeddingShard_v1", lambda S: [])
    F = CK.Findings()
    rec, _arrays, _ref = CK.correspondence(t, opts, F)
    assert not F.failures and rec["wide"][0]["reads"] == 0 and "probes" not in rec
    assert [w["name"] for w in F.warnings] == [
        "correspondence/operand-unread EmbeddingShard_v1{VS=64,H=32,START=1000}: 1 no vector read 'table', so its words are "
        "checked on none (`circuit_check.targets.OPERAND_PROBES` sets an index that reads it)"]
    monkeypatch.setitem(TG.OPERAND_PROBES, "EmbeddingShard_v1", lambda S: [{"table": 0}])
    with pytest.raises(ValueError, match=r"isn't one of its Value parameters \['tok'\]"):
        CK.correspondence(t, opts, CK.Findings())


def test_an_operands_lazy_words_are_its_edges_then_seeded_random_words():
    """Vector i of a lazily read operand is the i-th edge of its width in every leaf, then random words of that width, the same
    on every sample and different per vector and per parameter."""
    from verity.circuits.types import Array, Value

    pt = Array(64, Array(80, Value(8)))
    ew = VX.edges(8)
    for i in range(len(ew)):
        assert {VX.wide_words(pt, i, len(ew), 7, 2)(k) for k in range(pt.leaves)} == {ew[i]}
    draw = lambda i, param=2: [VX.wide_words(pt, i, len(ew), 7, param)(k) for k in range(pt.leaves)]  # noqa: E731
    a = draw(len(ew))
    assert a == draw(len(ew)) and all(0 <= v < 256 for v in a) and len(set(a)) > 64
    assert a != draw(len(ew) + 1) and a != draw(len(ew), param=3)
    with pytest.raises(ValueError, match="one width of at most 64 bits"):
        VX.wide_words(Array(4, Value(128)), 0, 1, 7, 0)


def test_a_definition_over_the_leaf_budget_is_not_partitioned(monkeypatch):
    """The word query and `Q_call` resolve every parameter leaf: over ``partition_max_leaves`` the partition is not run, a warning
    where the Definition is no Call; a primitive Call has no callees to partition it, so it fails by its name."""
    read = _table_read()
    monkeypatch.setitem(TG.TABLE_PROBES, read.id, lambda: _table_probe(_holds()))
    opts = dataclasses.replace(TABLE_OPTS, partition_max_leaves=4096)
    why = "its parameters' 8193 leaves are over the partition's leaf budget 4096: the word query and Q_call resolve every leaf"
    e = CK.check_definition(read, opts, memo={}, pins=PINS, as_call=False)
    assert not e["failures"] and [w["name"] for w in e["warnings"]] == [f"partition/unchecked CircuitCheckTableRead_v1: 1 {why}"]
    assert e["partition"]["checked"] is False and "q_call" not in e["partition"]
    e = CK.check_definition(read, opts, memo={}, pins=PINS, as_call=True)
    assert [f["name"] for f in e["failures"]] == [
        f"partition/unpartitioned CircuitCheckTableRead_v1 {why}; CircuitCheckTableRead_v1 is a primitive, with no callees to "
        "partition it"]
    assert not e["warnings"] and e["partition"]["checked"] is False and "q_call" not in e["partition"]
    assert CK.check_definition(read, TABLE_OPTS, memo={}, pins=PINS, as_call=True)["partition"]["q_call"]["checked"]


# --- negatives: each check fires, by name ---------------------------------------------------------------------------------------

def _composite(name, sig, body):
    from verity.circuits.defs import CompositeDefinition, bind

    return bind(CompositeDefinition(name, 1, (), lambda S: sig, body, register=False))


def _f32x2():
    from verity.circuits.types import Array, Value

    return (("x", Array(2, Value(32))),)


def test_a_value_computed_twice_in_one_unit_is_a_redundant_gate():
    from verity.circuits.types import Value
    from verity.definitions.registry import prims as P

    fn = _composite("CircuitCheckRedundant", (_f32x2(), Value(32)),
                    lambda B, S, x: B.call(P.F32Mul, B.call(P.F32Add, x[0], x[1]), B.call(P.F32Add, x[0], x[1])))
    e = CK.check_definition(fn, OPTS, as_call=True)
    (w,) = [w for w in e["warnings"] if w["check"] == "redundant-gates" and w["code"] == "ir"]
    assert w["count"] == 1 and w["where"] == {"F32Add_v1 @ CircuitCheckRedundant_v1": 1}
    assert not [f for f in e["failures"] if f["check"] == "partition"]


def test_a_value_computed_in_two_units_is_a_recompute_failure():
    """Two identical calls, each returned by its own proof block (the body reads only its parameters, which `word.unit_rule` partitions
    node by node; the whole Definition's cut compares them)."""
    from verity.circuits.refs import array_of
    from verity.circuits.types import Array, Value
    from verity.definitions.registry import prims as P

    fn = _composite("CircuitCheckRecompute", (_f32x2(), Array(2, Value(32))),
                    lambda B, S, x: array_of([B.call(P.F32Add, x[0], x[1]), B.call(P.F32Add, x[0], x[1])]))
    e = CK.check_definition(fn, OPTS, as_call=True)
    assert "partition/gate-recomputed CircuitCheckRecompute_v1 (1 gate(s))" in [f["name"] for f in e["failures"]]
    assert e["partition"]["q_word"] == 1                                # a word Call keeps v1: the recompute fails
    assert e["partition"]["whole"]["q_word_v2"] == {"codes": [], "recomputed_across": 1}   # v2 reports it, refuses nothing
    inner = CK.check_definition(fn, OPTS, as_call=False)                 # not a Call: its copies are redundant gates, not a failure
    assert not [f for f in inner["failures"] if f["check"] == "partition"] and inner["partition"]["isolated"]
    assert [w["count"] for w in inner["warnings"] if w["check"] == "redundant-gates" and w["code"] == "ir"] == [1]


def test_a_call_over_the_partition_budget_fails_by_its_name(monkeypatch):
    """Over ``partition_max_gates`` the partition is not run.  A Definition that is no Call is partitioned inside the Call that
    contains it, so its record says it is unchecked and nothing fails; a Call whose callees can't partition it either (one is no
    Call) fails by its name, with that callee, so its entry never reads as a pass."""
    from verity.circuits.types import Value
    from verity.definitions.registry import prims as P

    fn = _composite("CircuitCheckOverBudget", (_f32x2(), Value(32)),
                    lambda B, S, x: B.call(P.F32Mul, B.call(P.F32Add, x[0], x[1]), x[1]))
    monkeypatch.setattr(TG, "is_call", lambda fam: fam == TG.family(fn))
    opts = dataclasses.replace(OPTS, partition_max_gates=1)
    e = CK.check_definition(fn, opts, as_call=False)
    assert e["partition"]["checked"] is False and e["partition"]["applies"] is False and not e["failures"]
    e = CK.check_definition(fn, opts, as_call=True)
    assert [f["name"] for f in e["failures"] if f["check"] == "partition"] == [
        f"partition/unpartitioned {fn.id} 2 gates: over the partition budget 1; {P.F32Add.id} is no Call"]
    assert e["partition"]["checked"] is False and e["partition"]["callee_gap"] == f"{P.F32Add.id} is no Call"
    assert not [w for w in e["warnings"] if w["check"] == "partition"] and not CK.summarize([e])["ok"]


def _two_callees():
    """A Call over a budget of 2 gates whose body (not separable: its second node reads the first's output) calls a 2-gate
    callee that computes one value twice, each copy returned by its own proof block, and a 1-gate one."""
    from verity.circuits.refs import array_of
    from verity.circuits.types import Array, Value
    from verity.definitions.registry import prims as P

    twice = _composite("CircuitCheckTwice", (_f32x2(), Array(2, Value(32))),
                       lambda B, S, x: array_of([B.call(P.F32Add, x[0], x[1]), B.call(P.F32Add, x[0], x[1])]))
    mul = _composite("CircuitCheckMul", (_f32x2(), Value(32)), lambda B, S, x: B.call(P.F32Mul, x[0], x[1]))
    top = _composite("CircuitCheckByCallees", (_f32x2(), Value(32)), lambda B, S, x: B.call(mul, B.call(twice, x)))
    return top, twice, mul, dataclasses.replace(OPTS, partition_max_gates=2)


def test_a_call_over_the_budget_whose_callees_are_calls_is_partitioned_by_them(monkeypatch):
    """Over ``partition_max_gates`` and not cut member by member, a Call whose every callee is a Call (`targets.CALLEE_CALLS`) has
    each callee's partition checked as a Call's: a violation fails by the callee's name, and no ``partition/unchecked`` warning
    is left; the report's summary names the callees."""
    top, twice, mul, opts = _two_callees()
    monkeypatch.setattr(TG, "is_call", lambda fam: fam in {TG.family(f) for f in (top, twice, mul)})
    assert CK.uncut(top, opts) == "3 gates: over the partition budget 2" and CK.uncut(twice, opts) is None
    e = CK.check_definition(top, opts, memo={}, pins=PINS, as_call=True)
    assert [f["name"] for f in e["failures"] if f["check"] == "partition"] == [f"partition/gate-recomputed {twice.id} (1 gate(s))"]
    assert not [w for w in e["warnings"] if w["check"] == "partition"]
    p = e["partition"]
    assert p["checked"] and p["reason"] == "3 gates: over the partition budget 2" and sorted(p["by_callees"]) == sorted([twice.id, mul.id])
    assert p["by_callees"][mul.id]["checked"] and e["redundant_gates"] == {"checked": False}
    assert p["max_out_bits"] == max(r["max_out_bits"] for r in p["by_callees"].values()) == 32
    assert CK.summarize([e])["by_callees"] == {top.id: sorted([twice.id, mul.id])}


def test_a_call_cut_member_by_member_down_to_a_member_over_the_budget_is_partitioned_by_its_callees(monkeypatch):
    """A Call whose body `cut.separable` cuts member by member, down to a member over the budget that isn't (``too-large``), is
    `checks.uncut` too, as `DsaButterflyRowsPad_v1` is around `DsaButterflyRows_v1`: its callees partition it, the member by its
    own callees, and where one of those is no Call it fails by its name with that gap."""
    from verity.circuits.types import Value

    top, twice, mul, opts = _two_callees()
    wrap = _composite("CircuitCheckWrap", (_f32x2(), Value(32)), lambda B, S, x: B.call(top, x))
    assert CK.uncut(wrap, opts) == f"its member {top.id}'s 3 gates are over the partition budget 2, and not cut member by member"
    monkeypatch.setattr(TG, "is_call", lambda fam: fam in {TG.family(f) for f in (wrap, top, twice, mul)})
    e = CK.check_definition(wrap, opts, memo={}, pins=PINS, as_call=True)
    assert [f["name"] for f in e["failures"] if f["check"] == "partition"] == [f"partition/gate-recomputed {twice.id} (1 gate(s))"]
    p = e["partition"]
    assert p["checked"] and sorted(p["by_callees"]) == [top.id]
    assert sorted(p["by_callees"][top.id]["by_callees"]) == sorted([twice.id, mul.id])
    monkeypatch.setattr(TG, "is_call", lambda fam: fam in {TG.family(f) for f in (wrap, top, mul)})
    e = CK.check_definition(wrap, opts, memo={}, pins=PINS, as_call=True)
    assert [f["name"] for f in e["failures"] if f["check"] == "partition"] == [
        f"partition/unpartitioned {wrap.id} its member {top.id}'s 3 gates are over the partition budget 2, and not cut member by "
        f"member; {top.id}: {twice.id} is no Call"]


def test_a_call_over_the_budget_with_a_callee_that_is_no_call_fails_by_its_name(monkeypatch):
    """A callee that is no Call is partitioned inside the Call that contains it, which here is over the budget: nothing checks
    the Call's partition, so it fails by its name (``partition/unpartitioned``), with the callee, rather than pass as a warning."""
    top, twice, mul, opts = _two_callees()
    monkeypatch.setattr(TG, "is_call", lambda fam: fam in {TG.family(f) for f in (top, mul)})
    e = CK.check_definition(top, opts, memo={}, pins=PINS, as_call=True)
    assert e["partition"]["checked"] is False and "by_callees" not in e["partition"]
    assert e["partition"]["callee_gap"] == f"{twice.id} is no Call" == CK.callee_gap(top, opts)
    assert [f["name"] for f in e["failures"] if f["check"] == "partition"] == [
        f"partition/unpartitioned {top.id} 3 gates: over the partition budget 2; {twice.id} is no Call"]
    assert not [w for w in e["warnings"] if w["check"] == "partition"]


def test_a_call_over_the_budget_with_a_primitive_callee_over_it_too_fails_by_its_name(monkeypatch):
    """A primitive callee over ``partition_max_leaves`` has no callees to partition it, so nothing checks its Call's partition
    though it is a Call: `checks.callee_gap` names it, in the record, in the Call's failure and for the suite's walk."""
    from verity.circuits.types import Value
    from verity.definitions.registry import prims as P

    top = _composite("CircuitCheckOverPrimitive", (_f32x2(), Value(32)), lambda B, S, x: B.call(P.F32Mul, x[0], x[1]))
    monkeypatch.setattr(TG, "is_call", lambda fam: fam in {TG.family(f) for f in (top, P.F32Mul)})
    opts = dataclasses.replace(OPTS, partition_max_leaves=1)
    assert CK.uncut(P.F32Mul, opts) is not None and not CK.callees_are_calls(top, opts)
    e = CK.check_definition(top, opts, memo={}, pins=PINS, as_call=True)
    assert e["partition"]["checked"] is False and "by_callees" not in e["partition"]
    assert e["partition"]["callee_gap"].startswith(f"{P.F32Mul.id} is a primitive and its parameters' 2 leaves are over")
    assert CK.callee_gap(P.F32Mul, opts) == f"{P.F32Mul.id} is a primitive, with no callees to partition it"
    (f,) = [f for f in e["failures"] if f["check"] == "partition"]
    assert f["code"] == "unpartitioned" and f["definition"] == top.id and f["name"].endswith(e["partition"]["callee_gap"])
    assert not [w for w in e["warnings"] if w["check"] == "partition"]


def test_a_call_over_the_leaf_budget_partitioned_by_its_callees_still_compares_its_members(monkeypatch):
    """Over ``partition_max_leaves`` a Call whose body nodes read only its parameters is partitioned by its callees too, and a
    value two of its members compute fails under `Q_word` v1 by the Call's name (`cross_call.members_check`), as `word.unit_rule`
    fails it for one cut member by member."""
    from verity.circuits.refs import array_of
    from verity.circuits.types import Array, Value

    _top, _twice, mul, _opts = _two_callees()
    pair = _composite("CircuitCheckPairByCallees", ((("x", Array(3, Value(32))),), Array(2, Value(32))),
                      lambda B, S, x: array_of([B.call(mul, array_of([x[0], x[1]])), B.call(mul, array_of([x[0], x[1]]))]))
    monkeypatch.setattr(TG, "is_call", lambda fam: fam in {TG.family(f) for f in (pair, mul)})
    opts = dataclasses.replace(OPTS, partition_max_leaves=2)
    e = CK.check_definition(pair, opts, memo={}, pins=PINS, as_call=True)
    assert e["partition"]["reason"].startswith("its parameters' 3 leaves are over the partition's leaf budget 2")
    assert list(e["partition"]["by_callees"]) == [mul.id] and not [w for w in e["warnings"] if w["check"] == "partition"]
    assert [f["name"] for f in e["failures"] if f["check"] == "partition"] == [
        f"partition/gate-recomputed {pair.id} (1 gate(s) between node members)"]


def test_a_call_over_the_budget_fails_all_by_the_name_of_a_callee_that_breaks_q_word(monkeypatch):
    """``--all`` takes a Call's callees as Calls from `targets.CALLEE_CALLS` (through `targets.call_families`, as every task
    does), checks each callee's partition inside the Call's check and fails by the callee's name, and its summary reads the
    Call partitioned by its callees.  Without the entries nothing checks the Call's partition, so ``--all`` fails by the
    Call's name (``partition/unpartitioned``) rather than pass a value computed in two proof blocks as a warning."""
    from circuit_check import block_held, cli

    top, twice, mul, opts = _two_callees()
    monkeypatch.setattr(TG, "load_registries", lambda: None)
    monkeypatch.setattr(TG, "suite", lambda: {top.id: top})
    # the holds name the registry's Definitions, which this one-target catalog doesn't check (block/absent-hold)
    monkeypatch.setattr(block_held, "HELD", {})
    monkeypatch.setattr(TG, "templates", lambda: [])
    monkeypatch.setattr(TG, "word_call_families", lambda: {TG.family(top)})
    monkeypatch.setattr(TG, "boolean_call_families", lambda words, found=None: set())
    monkeypatch.setattr(TG, "_CALLS", set())

    def everything(callee_calls):
        monkeypatch.setattr(TG, "CALLEE_CALLS", callee_calls)
        TG._CALLS.clear()
        return cli.run([], opts, everything=True, log=None, isolate=False)

    rep = everything({})
    (unchecked,) = rep["targets"]
    assert not rep["summary"]["ok"] and unchecked["partition"]["checked"] is False
    assert [f["name"] for f in unchecked["failures"]] == [
        f"partition/unpartitioned {top.id} 3 gates: over the partition budget 2; {twice.id} is no Call"]
    assert not [w for w in unchecked["warnings"] if w["check"] == "partition"]
    rep = everything({TG.family(twice): (TG.family(top),), TG.family(mul): (TG.family(top),)})
    (target,) = rep["targets"]
    assert [f["name"] for f in target["failures"] if f["check"] == "partition"] == [
        f"partition/gate-recomputed {twice.id} (1 gate(s))"]
    assert not [w for w in target["warnings"] if w["check"] == "partition"] and target["partition"]["checked"]
    assert not rep["summary"]["ok"] and rep["summary"]["by_callees"] == {top.id: sorted([twice.id, mul.id])}


#: a Boolean Call family (`RoPEHead_v1`'s Boolean version) whose cut computes 1,120 values in two proof blocks
BOOL_RECOMPUTE = "RoPEHead_v2{D=8}"


def _partition(fn, q_word, opts=OPTS):
    F = CK.Findings()
    rec, _rule = CK.partition(fn, opts, F, {}, applies=True, q_word=q_word)
    rec.pop("_graph", None)
    return rec, [f["name"] for f in F.failures if f["check"] == "partition"]


@pytest.mark.slow
@pytest.mark.usefixtures("suite")
def test_a_boolean_call_that_recomputes_across_units_passes_with_the_count():
    """A Boolean Call is judged by `Q_word` v2: the values two proof blocks compute are recorded, in the entry and the summary, not
    failed; under v1 the same cut fails by name."""
    from verity.circuits import boolean as BL

    fn = TG.definition(BOOL_RECOMPUTE)
    assert BL.is_boolean(fn) and TG.family(fn) in TG.call_families()
    rec, failed = _partition(fn, 2)
    assert not failed and rec["q_word"] == 2 and rec["recomputed_across"] == 1120
    assert CK.summarize([{"definition": fn.id, "partition": rec, "failures": [], "warnings": []}])["recomputed_across"] == {fn.id: 1120}
    _rec, failed = _partition(fn, 1)
    assert failed == [f"partition/gate-recomputed {fn.id} (1120 gate(s))"]
    rec, failed = _partition(fn, 2, dataclasses.replace(OPTS, partition_max_gates=10))   # over the budget: unchecked, no count
    assert failed and {f.split()[0] for f in failed} == {"partition/unpartitioned"} and not rec["checked"]
    assert rec["recomputed_across"] is None


def test_a_boolean_call_with_another_partition_code_still_fails(monkeypatch):
    """Under v2 only `gate-recomputed` is reported instead of refused: a Boolean Call whose cut has any other code fails by name."""
    from verity.circuits.partition import CutVerdict
    from verity.definitions import word as W

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


def test_the_partition_cuts_a_call_once(monkeypatch):
    """`partition` checks the cut `word.unit_rule` made of the whole Definition (its ``graphs``) rather than cutting it again: one
    `word.Graph` of it, and the record and failures of cutting it twice; a Definition cut member by member (`word.separable`) is
    cut whole once, by `partition` alone."""
    from verity.circuits.types import Value
    from verity.definitions import word as W
    from verity.definitions.registry import prims as P

    real_rule, cuts = W.unit_rule, []

    class Counted(W.Graph):
        def __init__(self, fn, *a, **k):
            cuts.append(fn.id)
            super().__init__(fn, *a, **k)

    monkeypatch.setattr(W, "Graph", Counted)
    chained = _composite("CircuitCheckCutOnce", (_f32x2(), Value(32)),
                         lambda B, S, x: B.call(P.F32Mul, B.call(P.F32Add, x[0], x[1]), B.call(P.F32Add, x[0], x[1])))
    rope = TG.definition(BOOL_RECOMPUTE)
    assert W.separable(chained) is None and W.separable(rope) is not None
    for fn, q_word, twice in ((chained, 1, 2), (rope, 2, 1), (rope, 1, 1)):
        cuts.clear()
        once = _partition(fn, q_word)
        assert cuts.count(fn.id) == 1, fn.id
        with monkeypatch.context() as m:
            m.setattr(W, "unit_rule", lambda *a, graphs=None, **k: real_rule(*a, **k))
            cuts.clear()
            assert _partition(fn, q_word) == once and cuts.count(fn.id) == twice, fn.id
    assert once[1] == [f"partition/gate-recomputed {rope.id} (1120 gate(s))"]


def test_q_call_is_recorded_per_call_and_refuses_a_gate_wider_than_32_bits():
    """`Q_call` v1 beside `Q_word`: a Boolean Call's proof blocks, committed bits and the values two proof blocks compute
    (reported); a Call with
    a 64-bit gate is refused, a warning and the summary's `q_call_refused`, not a failure."""
    from verity.circuits.defs import PrimitiveDefinition
    from verity.circuits.types import Array, Value

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


#: `Q_call{32}` on `BOOL_RECOMPUTE`: its eight 16-bit outputs, two to a proof block, and the same values computed in two proof blocks as v2 reports
QCALL_ROPE = {"X": 32, "checked": True, "units": 8, "committed": 128, "committed_bits": 128, "recomputed_across": 1120, "redundant_gates": 0,
              "codes": []}


@pytest.mark.slow
@pytest.mark.usefixtures("suite")
def test_the_boolean_versions_of_the_served_calls_are_calls():
    """Sigma's image of the word Call families (`boolean_version`, the switch's `word=` table) is held as Calls."""
    calls = TG.call_families()
    assert {"RMSNormFusedCuda_v3", "RMSNormTriton_v2", "Gemm_v3", "Attention_v8", "RoPE_v2", "SiluMul_v3", "TokenSelect_v2",
            "Embedding_v2"} <= calls
    assert {"RMSNormFusedCuda_v2", "RMSNormTriton_v1", "Gemm_v1"} <= calls
    assert not {"DotBf16_v3", "AttnBlock_v6"} & calls


#: Calls over `Options.partition_max_gates` or `Options.partition_max_leaves` whose partition stays unchecked, each with why it
#: is allowed: the suite's, and any range spot or callee of one `_unchecked_walk` reaches; none is.  Each passes no Call status
#: to its callees (`targets.CALLEE_CALLS`): their boundaries are not the proof blocks its pending cut makes.  A hold keeps these
#: walks from failing on it; circuit-check still fails it by name (``partition/unpartitioned``), and whether that may stay red is
#: drive's policy (policy.toml's allowances).  Over the gate
#: budget alone, GLM-5.3 TP8's served two-shot MNNVL all-reduces (past 22 rows of 6,144) and multimem ones (past 50) are cut
#: member by member; the σ Definitions at their smallest bindings (`boolean_sigma.circuit-check.py`, whose split moves
#: GLM-5.3's and DeepSeek-V4's word digests), multimem at 86 rows and more, over the leaf budget too, and GLM-5.3's served
#: expert GEMMs, over the leaf budget (an E = 256 stack of FP8 weights: 805M leaves for w13, 403M for w2), are partitioned by
#: their callees
OVER_PARTITION_BUDGET: dict[str, str] = {}


def _over_budget(catalog, calls: set[str]) -> dict[str, str]:
    """{the suite's Calls over `Options.partition_max_gates` or `Options.partition_max_leaves`: which}.  Gate counts are the
    catalog's (`Catalog.reached`): a target made from its record has no body built, and building every Call's would take about
    half an hour."""
    from verity.circuits.evaluation.reference import signature_of

    suite, out = catalog.suite, {}
    for k in suite:
        if catalog.reached[k][0] in calls:
            if (n := catalog.reached[k][1]) > OPTS.partition_max_gates:
                out[k] = f"{n} gates"
            elif (n := sum(pt.leaves for pt in signature_of(suite[k])[0])) > OPTS.partition_max_leaves:
                out[k] = f"{n} leaves"
    return out


def _gap(fn) -> str | None:
    """Why a Call of ``fn`` over the budget passes with its partition unchecked, or None: for one `checks.uncut` leaves to its
    callees (a member that the cut member by member leaves whole included), `checks.callee_gap`."""
    return CK.callee_gap(fn, OPTS) if CK.uncut(fn, OPTS) is not None else None


_WALK: dict = {}


def _unchecked_walk(catalog) -> tuple[dict[str, set[str]], set[str], dict[str, str]]:
    """What `targets.CALLEE_CALLS` should be, the `OVER_PARTITION_BUDGET` ids reached, and each `checks.callee_gap` reached: from
    the suite's Calls over the budget (`_over_budget`) and the range spots that are Calls (`circuit_check.ranges`), the families
    of each `checks.uncut` Call's callees that are no Call otherwise, each with the families of the Calls that pass it, through
    each callee `uncut` itself, primitives included; one `OVER_PARTITION_BUDGET` holds passes nothing."""
    if not _WALK:
        from circuit_check import ranges as RG

        suite, calls = catalog.suite, TG.call_families()
        own = calls - set(TG.CALLEE_CALLS)
        stack = [suite[k] for k in _over_budget(catalog, calls)]
        stack += [fn for spot in RG.spots() if TG.family(fn := TG.definition(spot)) in calls]
        want: dict[str, set[str]] = {}
        held: set[str] = set()
        gaps: dict[str, str] = {}
        seen: set[str] = set()
        while stack:
            fn = stack.pop()
            if fn.id in seen or CK.uncut(fn, OPTS) is None:
                continue
            seen.add(fn.id)
            if fn.id in OVER_PARTITION_BUDGET:
                held.add(fn.id)
                continue
            if (gap := CK.callee_gap(fn, OPTS)) is not None:
                gaps[fn.id] = gap
            for n in fn.body.nodes if hasattr(fn, "body") else ():
                if (f := TG.family(n.fn)) not in own:
                    want.setdefault(f, set()).add(TG.family(fn))
                stack.append(n.fn)
        _WALK.update(want=want, held=held, gaps=gaps)
    return _WALK["want"], _WALK["held"], _WALK["gaps"]


@pytest.mark.slow
@_opt_in("~40 min")
def test_every_call_the_suite_checks_is_partitioned(catalog):
    """No Call passes with its partition unchecked: each is within `Options.partition_max_gates` (the Boolean attention Calls
    at T = 17 are about 1.9M gates, 2.1M with Gemma-2's softcap; DeepSeek-V4's Boolean FlashMLA prefill and mHC Calls at their
    smallest whole blocks 4.1M to 4.6M) and `Options.partition_max_leaves`, is cut member by member with each member within
    them, or is partitioned by its callees (`_gap` finds none), apart from `OVER_PARTITION_BUDGET`, which holds only Calls
    that are still none of these."""
    suite, calls = catalog.suite, TG.call_families()
    over = _over_budget(catalog, calls)
    gaps = {k: f"{why}; {gap}" for k, why in over.items() if k not in OVER_PARTITION_BUDGET and (gap := _gap(suite[k])) is not None}
    assert gaps == {}, ("a Call over the partition budget that its callees can't partition: list a callee that is no Call in "
                        "targets.CALLEE_CALLS, cut an uncut primitive callee or member, or hold the Call here with why")
    assert sorted(set(OVER_PARTITION_BUDGET) - set(over) - _unchecked_walk(catalog)[1]) == [], \
        "no longer an over-budget Call the suite checks, or a range spot or callee of one: remove it from OVER_PARTITION_BUDGET"
    assert sorted(k for k in OVER_PARTITION_BUDGET if k in suite and _gap(suite[k]) is None) == [], \
        "cut member by member or partitioned by its callees now: remove it from OVER_PARTITION_BUDGET"


@pytest.mark.slow
@_opt_in("~40 min")
def test_a_call_whose_partition_is_unchecked_passes_call_status_to_its_callees(catalog):
    """`targets.CALLEE_CALLS` is the families of the Definitions an `checks.uncut` Call calls (and, through each callee `uncut`
    itself, its callees), apart from those that are Calls otherwise, each with the families of the Calls that pass it
    (`_unchecked_walk`).  The uncut Calls are the suite's and the range spots' (`circuit_check.ranges`), apart from
    `OVER_PARTITION_BUDGET`'s: they pass nothing, and stay unchecked until their cut lands.  Every other one the walk reaches,
    primitives included, is partitioned by its callees: an uncut primitive callee has none, so it fails here by name, as
    circuit-check fails its Call (``partition/unpartitioned``).  A listed Call that no longer passes a family Call status only warns:
    it keeps a family a Call, so its partition is checked, never left unchecked."""
    want, _held, gaps = _unchecked_walk(catalog)
    have = {f: set(by) for f, by in TG.CALLEE_CALLS.items()}
    assert {f: sorted(by - have.get(f, set())) for f, by in want.items() if by - have.get(f, set())} == {}, (
        f"set targets.CALLEE_CALLS to {dict(sorted((f, tuple(sorted(by))) for f, by in want.items()))}, or hold in "
        "OVER_PARTITION_BUDGET, with why, a Call whose callees can't be Calls yet")
    if stale := {f: sorted(by - want.get(f, set())) for f, by in have.items() if by - want.get(f, set())}:
        warnings.warn(f"targets.CALLEE_CALLS lists Calls that no longer pass these families Call status: {stale}; drop them",
                      stacklevel=1)
    assert {f: by for f, by in TG.CALLEE_CALLS.items() if list(by) != sorted(by)} == {}, "list each family's Calls in order"
    assert gaps == {}, ("Calls whose partition nothing checks (`checks.callee_gap`: a callee that is no Call, or a primitive "
                        "over the budget): cut each, or hold it in OVER_PARTITION_BUDGET with why")


@pytest.mark.slow
def test_a_boolean_family_over_the_budget_keeps_its_boolean_binding(suite):
    """The suite keeps a family's smallest specialization over `MAX_GATES`; a Boolean norm's smallest binds the word MUFU, so its
    smallest Boolean binding (the one the Program on bits calls) is checked too."""
    from verity.circuits import boolean as BL
    for word, bits in [("RMSNormFusedCuda_v3{N=16,EPS=1e-05,RSQRT=RsqrtApprox_v1}", "RMSNormFusedCuda_v3{N=16,EPS=1e-05,RSQRT=RsqrtApprox_v2}"),
                       ("RMSNormTriton_v2{N=16,EPS=1e-05,SQRT=MufuSqrtFtz_v1,SCALEA=DivFullScaleA_v1,RCP=DivFullRcp_v1}",
                        "RMSNormTriton_v2{N=16,EPS=1e-05,SQRT=MufuSqrtFtz_v2,SCALEA=DivFullScaleA_v3,RCP=DivFullRcp_v2}")]:
        assert {word, bits} <= set(suite) and suite[bits].gates > TG.MAX_GATES
        assert BL.is_boolean(suite[bits]) and not BL.is_boolean(suite[word])


def test_a_gate_no_output_reads_is_a_dead_gate():
    from verity.circuits.types import Value
    from verity.definitions.registry import prims as P

    def body(B, S, x):
        B.call(P.F32Mul, x[0], x[1])
        return B.call(P.F32Add, x[0], x[1])

    e = CK.check_definition(_composite("CircuitCheckDead", (_f32x2(), Value(32)), body), OPTS)
    (w,) = [w for w in e["warnings"] if w["check"] == "dead-gates"]
    assert w["count"] == 1 and w["where"] == {"F32Mul_v1 @ CircuitCheckDead_v1": 1}


def test_a_unit_wider_than_the_word_fails_the_width_rule():
    from verity.circuits.defs import PrimitiveDefinition
    from verity.circuits.types import Value

    wide = PrimitiveDefinition("CircuitCheckWide", 1, (("a", Value(32)), ("b", Value(32))), Value(64), lambda a, b: (a << 32) | b,
                               register=False)
    fn = _composite("CircuitCheckWidth", ((("a", Value(32)), ("b", Value(32))), Value(64)), lambda B, S, a, b: B.call(wide, a, b))
    names = [f["name"] for f in CK.check_definition(fn, OPTS, as_call=True)["failures"]]
    assert any(n.startswith("partition/width CircuitCheckWidth_v1 output:CircuitCheckWide_v1") for n in names), names


def test_a_partial_reference_is_named():
    from verity.circuits.defs import PrimitiveDefinition
    from verity.circuits.types import Value

    def finite_only(a):
        if (a >> 23) & 0xFF == 0xFF:
            raise ValueError("non-finite")
        return a

    prim = PrimitiveDefinition("CircuitCheckPartial", 1, (("a", Value(32)),), Value(32), finite_only, register=False)
    (f,) = CK.check_definition(prim, OPTS)["failures"]
    assert f["name"].startswith("correspondence/reference-not-total CircuitCheckPartial_v1 raised on")


def test_a_declared_interval_the_reference_leaves_is_named():
    """A primitive's declared return interval is a fact the Build takes (`verity.circuits.ports`): checked on operands within
    its declared bounds, so a product of two s8 operands holds in [-16256, 16384] and one off by a unit is named."""
    from verity.circuits.defs import PrimitiveDefinition
    from verity.circuits.ports import In, Out, Ports
    from verity.circuits.types import Value

    def mul(a, b):
        return ((a - (1 << 16) if a >> 15 else a) * (b - (1 << 16) if b >> 15 else b)) & 0xFFFFFFFF

    s8 = {"a": In(bound=(-128, 127)), "b": In(bound=(-128, 127))}
    good = PrimitiveDefinition("CircuitCheckS8Mul", 1, (("a", Value(16)), ("b", Value(16))), Value(32), mul, register=False,
                               ports=Ports(params=s8, ret=Out(interval=(-16256, 16384))))
    e = CK.check_definition(good, OPTS, as_call=False)
    assert e["ports"] == {"vectors": e["correspondence"]["vectors"], "intervals": [[0, 1, [-16256, 16384]]], "outside": 0}
    assert not e["failures"]
    bad = PrimitiveDefinition("CircuitCheckS8MulTight", 1, (("a", Value(16)), ("b", Value(16))), Value(32), mul, register=False,
                              ports=Ports(params=s8, ret=Out(interval=(-16256, 16383))))
    (f,) = CK.check_definition(bad, OPTS, as_call=False)["failures"]
    assert f["name"].startswith("ports/interval CircuitCheckS8MulTight_v1") and f["examples"][0]["value"] == 16384
    assert "ports" not in CK.check_definition(PrimitiveDefinition("CircuitCheckS8MulUntyped", 1, (("a", Value(16)), ("b", Value(16))),
                                                                  Value(32), mul, register=False), OPTS, as_call=False)


def test_a_transfer_the_reference_leaves_is_named():
    """A primitive's transfer (its return's interval from its operands') is a fact the Build takes: checked on sampled operand
    intervals, so an exact sum of two s8 operands holds in [a0 + b0, a1 + b1] and a transfer one short of it is named."""
    from verity.circuits.defs import PrimitiveDefinition
    from verity.circuits.ports import Ports, as_reading
    from verity.circuits.types import Value

    def add(a, b):
        return (((a ^ 0x80) - 0x80) + ((b ^ 0x80) - 0x80)) & 0xFFFF

    def transfer(short):
        def f(a, b):
            a, b = as_reading(a, 8), as_reading(b, 8)
            return None if a is None or b is None else (a[0] + b[0], a[1] + b[1] - short)
        return f

    def prim(name, short):
        return PrimitiveDefinition(name, 1, (("a", Value(8)), ("b", Value(8))), Value(16), add, register=False,
                                   ports=Ports(transfer=transfer(short)))

    e = CK.check_definition(prim("CircuitCheckS8Add", 0), OPTS, as_call=False)
    t = e["ports"]["transfer"]
    assert t["draws"] == CK.TRANSFER_DRAWS and t["intervals"] > 0 and t["outside"] == 0 and not e["failures"]
    fails = CK.check_definition(prim("CircuitCheckS8AddShort", 1), OPTS, as_call=False)["failures"]
    assert fails and all(f["name"].startswith("ports/transfer CircuitCheckS8AddShort_v1") for f in fails)


def test_a_kernel_wrong_only_on_edge_words_is_caught():
    """A kernel that returns 0 for a NaN operand: uniform random words rarely hold a NaN, the edge vectors always do."""
    from verity.circuits.evaluation import register_kernel
    from verity.circuits.evaluation.batch import _KERNELS
    from verity.circuits.defs import PrimitiveDefinition
    from verity.circuits.types import Value

    prim = PrimitiveDefinition("CircuitCheckKernel", 1, (("a", Value(32)),), Value(32), lambda a: a ^ 1, register=False)
    nan = lambda a: ((a >> 23) & 0xFF == 0xFF) & ((a & 0x7FFFFF) != 0)  # noqa: E731
    register_kernel(prim, lambda a: (np.where(nan(a), 0, a ^ 1).astype(np.uint32), None))
    try:
        e = CK.check_definition(prim, OPTS)
    finally:
        del _KERNELS[prim.id]
    (f,) = e["failures"]
    assert f["name"].startswith("correspondence/kernel-mismatch CircuitCheckKernel_v1 [numpy]")


def test_a_kernel_that_compared_nothing_or_declined_inside_the_domain_is_not_reported_checked():
    """A kernel that declines every vector, and one not registered partial that declines vectors the reference accepts,
    each fail `correspondence` under their `SelfCheck.fault`; the partial one that computes some vectors passes."""
    from verity.circuits.evaluation import register_kernel
    from verity.circuits.evaluation.batch import _KERNELS
    from verity.circuits.defs import PrimitiveDefinition
    from verity.circuits.types import Value

    prim = PrimitiveDefinition("CircuitCheckDeclines", 1, (("a", Value(16)),), Value(16), lambda a: a ^ 1, register=False)
    odd = lambda a: (a ^ 1, a % 2 == 1)  # noqa: E731
    register_kernel(prim, lambda a: (a ^ 1, np.ones(a.shape, dtype=bool)), kernel="every")
    register_kernel(prim, odd, kernel="odd")
    register_kernel(prim, odd, kernel="odd-partial", partial=True)
    try:
        e = CK.check_definition(prim, OPTS)
    finally:
        del _KERNELS[prim.id]
    assert sorted(f["name"].split(":")[0] for f in e["failures"]) == [
        "correspondence/kernel-compared-nothing CircuitCheckDeclines_v1 [every]",
        "correspondence/kernel-declined-in-domain CircuitCheckDeclines_v1 [odd]"]
    via = {r["via"]: r for r in e["correspondence"]["realizations"] if r["via"].startswith("kernel:")}
    assert (via["kernel:every"]["compared"], via["kernel:every"]["fault"]) == (0, "compared-nothing")
    assert via["kernel:odd-partial"]["compared"] > 0 and "fault" not in via["kernel:odd-partial"]


def test_a_word_definition_that_differs_from_its_boolean_definition_is_caught():
    """The Boolean Definition is the definition: the word Definition it names must equal its gates through the word view."""
    from verity.circuits import boolean as BL
    from verity.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
    from verity.circuits.refs import Coll, concat
    from verity.circuits.types import Array, Value

    nib = Array(4, BL.BIT)

    def xor_bits(B, S, x, y):
        return B.batch(BL.XOR, Coll(Array(4, Array(2, BL.BIT)), concat([c.refs for i in range(4) for c in (x[i], y[i])])), axes=(0,))

    for name, word in (("CircuitCheckXorBits", lambda x, y: x ^ y), ("CircuitCheckOrBits", lambda x, y: x | y)):
        d = CompositeDefinition(name, 1, (), lambda S: ((("x", nib), ("y", nib)), nib), xor_bits, register=False)
        d.word = PrimitiveDefinition(f"{name}Word", 1, (("x", Value(4)), ("y", Value(4))), Value(4), word, register=False)
        F = CK.Findings()
        rec = CK.word_view(bind(d), OPTS, F)
        assert rec["via"] == f"word:{name}Word_v1" and rec["compared"] > 0
        if name == "CircuitCheckXorBits":
            assert rec["mismatches"] == 0 and not F.failures
        else:
            (f,) = F.failures
            assert rec["mismatches"] > 0 and f["name"].startswith(f"correspondence/word-mismatch {name}_v1 [{name}Word_v1]: ")


def test_a_boolean_constant_is_checked_on_its_one_instance():
    """A Boolean constant (`Const<w>[0x..]_v2`, `verity.definitions.ops.boolean.scalar.const_word`) takes no operands: its lowering
    and its word view each compare its one vector, every bit, so a constant one bit off its word constant is named."""
    from verity.circuits import boolean as BL
    from verity.circuits.boolean.forms import const_bits
    from verity.circuits.boolean.trace import definition
    from verity.definitions.ops import prims as MP
    from verity.definitions.ops.boolean import scalar as BS
    from verity.circuits.types import Array

    e = CK.check_definition(BS.const_word(32, 0x3727C5AC), OPTS, pins=PINS)
    via = {r["via"]: r for r in e["correspondence"]["realizations"]}
    assert not e["failures"] and via["c-flock:ir_lower"]["ands"] == 0
    assert [(r["compared"], r["mismatches"]) for r in via.values()] == [(1, 0), (1, 0)]
    off = definition("CircuitCheckConst", 2, lambda C: const_bits(0x3727C5AD, 32), (), Array(32, BL.BIT),
                     word=MP.const(32, 0x3727C5AC), register=False)
    names = [f["name"] for f in CK.check_definition(off, OPTS, pins=PINS)["failures"]]
    assert names == ["correspondence/word-mismatch CircuitCheckConst_v2 [Const32[0x3727c5ac]_v1]: 1 of 1 vectors"], names


def test_a_lowering_that_differs_from_the_ir_is_caught(monkeypatch):
    from verity_flock import ir_lower as IL
    from verity.definitions.registry import prims as P

    add = IL.PIECES["F32Add_v1"]
    monkeypatch.setitem(IL.PIECES, "F32Mul_v1", lambda C, a, b: add(C, a, b))
    e = CK.check_definition(P.F32Mul, OPTS, pins=PINS)
    assert any(f["name"].startswith("correspondence/lowering-mismatch F32Mul_v1 [c-flock]") for f in e["failures"])


def test_a_lowering_count_off_its_pin_is_named():
    from verity.definitions.registry import prims as P

    pins = {"pieces": {"F32Mul_v1": {"and": 1}}, "units": {}}
    e = CK.check_definition(P.F32Mul, OPTS, pins=pins)
    assert any(f["name"].startswith("lowering/pin-mismatch F32Mul_v1 [F32Mul_v1]: and 1 -> ") for f in e["failures"])


def test_a_proof_blocks_export_counts_its_circuit_types():
    """`checks._export_counts`, which every lowered template's check reads, exports one proof block's circuit through
    `boolean_export` and counts it: four ANDs of 4-bit inputs are one circuit type of 4 ANDs, none shared."""
    from verity_flock import gf2

    def build():
        C = gf2.Circuit(cse=True)
        a, b = C.input("a", 4), C.input("b", 4)
        C.output("out0", [C.AND(x, y) for x, y in zip(a, b)])
        return C

    want = {"and": 4, "xor": 0, "not": 0, "subcircuit_types": 1, "flat_ands": 4, "shared_across_types": 0}
    assert CK._export_counts(build, "test unit") == want == CK._export_counts(build, "test unit", build())


def test_a_proof_blocks_export_is_evaluated_against_the_circuit_it_is_handed():
    """`checks._export_counts` given the circuit ``build`` made (`template_lowering` hands it the lowering's) evaluates the
    export's types against that circuit: one that computes something else is refused."""
    from verity_flock import gf2

    def build(xor: bool = False):
        C = gf2.Circuit(cse=True)
        a, b = C.input("a", 4), C.input("b", 4)
        C.output("out0", [x ^ y if xor else C.AND(x, y) for x, y in zip(a, b)])
        return C

    with pytest.raises(AssertionError, match="disagree with its circuit"):
        CK._export_counts(build, "test unit", build(xor=True))


def _lowering_pin(e: dict) -> dict:
    (r,) = [r for r in e["correspondence"]["realizations"] if r["via"] == "c-flock:ir_lower"]
    return r["pin"]


def test_a_lowering_type_is_its_flat_circuit_type():
    """A pin's ``type`` (`checks.lowering_type`) is `circuit_types.from_gf2`'s digest of the lowering: its named inputs and
    outputs as ports, its assertions as one more port, its tables by their SHA-512; an assertion and a table read each enter it."""
    from verity_flock import circuit_types as CT
    from verity_flock import gf2
    from verity_flock import tail_pieces as TP
    from verity_flock import type_trace as TT

    def build(asserted: bool):
        C = gf2.Circuit()
        a, b = C.input("a", 2), C.input("b", TP.TABLE_BITS["ex2"])
        x = C.AND(a[0], a[1])
        v = C.lookup("ex2", [x ^ b[0], *b[1:]], TP.VALUE_BITS)
        C.output("o", [x, v[0] ^ a[0]])
        if asserted:
            C.assert_zero("z", v[1] ^ b[2])
        return C

    C = build(True)
    ins, outs = [*C.inputs["a"], *C.inputs["b"]], [*C.outputs["o"], C.asserts[0][1]]
    t = CT.from_gf2(C, ins, outs, [(1, 2), (1, len(C.inputs["b"]))], [(1, 2), (1, 1)], "program",
                    {"ex2": (TT.table_sha512("ex2"), TP.TABLE_BITS["ex2"], TP.VALUE_BITS)})
    assert [it[0] for it in t.items] == [CT.AND, CT.READ] and CK.lowering_type(C) == t.digest()
    assert CK.lowering_type(build(False)) != t.digest()


def test_a_lowering_with_its_pinned_and_count_but_another_circuit_is_refused_on_its_type(monkeypatch):
    """The same ANDs computing the same function, but built another way (every AND's operands swapped), match the pin's AND count
    and every correspondence check: only the pin's ``type`` refuses it."""
    from verity.circuits.boolean import forms
    from verity.definitions.registry import prims as P

    pin = _lowering_pin(CK.check_definition(P.F32Mul, OPTS, pins={"pieces": {}, "units": {}}))
    pins = {"pieces": {"F32Mul_v1": pin["counts"]}, "units": {}}
    assert not CK.check_definition(P.F32Mul, OPTS, pins=pins)["failures"]
    AND = forms.Circuit.AND
    monkeypatch.setattr(forms.Circuit, "AND", lambda self, a, b: AND(self, b, a))
    e = CK.check_definition(P.F32Mul, OPTS, pins=pins)
    assert _lowering_pin(e)["counts"]["and"] == pin["counts"]["and"]
    assert [f["name"].split(" (")[0] for f in e["failures"]] == [
        f"lowering/pin-mismatch F32Mul_v1 [F32Mul_v1]: type {pin['counts']['type'][:16]}... -> {_lowering_pin(e)['counts']['type'][:16]}..."]


def test_a_pin_without_a_type_is_named_as_untyped_and_update_pins_types_it(monkeypatch):
    """A pin with the lowering's AND count and no ``type`` (a branch written before ``type``) passes with an ``untyped``
    warning, and `--update-pins` writes the lowering's ``type`` into it."""
    from circuit_check import cli
    from circuit_check import sidecars as SC
    from verity.definitions.registry import prims as P

    pin = _lowering_pin(CK.check_definition(P.F32Mul, OPTS, pins={"pieces": {}, "units": {}}))
    e = CK.check_definition(P.F32Mul, OPTS, pins={"pieces": {"F32Mul_v1": {"and": pin["counts"]["and"]}}, "units": {}})
    assert not e["failures"] and _lowering_pin(e)["status"] == "untyped"
    assert [w["name"].split(" (")[0] for w in e["warnings"] if w["code"] == "untyped"] == [
        "lowering/untyped F32Mul_v1: 1 [F32Mul_v1]: no type"]
    written = {}
    monkeypatch.setattr(SC, "write_pins", written.update)
    cli.update_pins({"targets": [e]})
    assert written == {("pieces", "F32Mul_v1"): pin["counts"]}


def test_a_gated_definitions_circuit_digest_is_held_to_its_pin():
    """A Definition a Program holds as one gate naming its circuit digest (``gated``: N2's served counts here) has that
    digest, core's `program_digest` of its Program, checked against its ``digests`` pin, which a verifier reads in its
    place: its own digest passes, another fails by name, and none warns."""
    from verity.circuits.codec import program_digest
    from verity.circuits.program import Program
    from verity.protocols.certifier.verifier import served as SV

    d = SV.unit(8, 3, 1, 2, 2, 0)
    digest = program_digest(Program(d))
    assert SV.NetworkServedCounts.gated and PINS["digests"][d.id] == {"sha512": digest}
    for pinned, status in ((digest, "ok"), ("0" * 128, "mismatch"), (None, "unpinned")):
        F = CK.Findings()
        rec = CK.circuit_digest_pin(d, {"digests": {} if pinned is None else {d.id: {"sha512": pinned}}}, F)
        assert rec["status"] == status and rec["counts"] == {"sha512": digest}
        assert [f["name"].split(" [")[0] for f in F.failures] == ([f"lowering/pin-mismatch {d.id}"] if status == "mismatch" else [])
        assert [w["code"] for w in F.warnings] == (["unpinned"] if status == "unpinned" else [])


def test_any_definition_holding_a_digests_pin_is_held_to_it():
    """A Definition that isn't gated but holds a ``digests`` pin (a binding at served statics over `targets.MAX_GATES`, which
    the pin keeps in ``--all``, `targets.pinned`) has its circuit digest checked against it, so no held pin goes uncompared:
    its own digest passes and another fails by name. Without a pin it gets no digest check."""
    from verity.circuits.codec import program_digest
    from verity.circuits.defs import bind
    from verity.circuits.program import Program
    from verity.definitions.registry import tp_allreduce as TA

    d = bind(TA.AllReduceMultimemVector, WORLD=8, TREE=TA.GLM53_H200_TREE, FTZ=False)
    digest = program_digest(Program(d))
    assert not getattr(getattr(d, "defn", d), "gated", False)
    assert "digest" not in CK.check_definition(d, OPTS, memo={}, pins={**PINS, "digests": {}}, as_call=False)
    for pinned, status in ((digest, "ok"), ("0" * 128, "mismatch")):
        e = CK.check_definition(d, OPTS, memo={}, pins={**PINS, "digests": {d.id: {"sha512": pinned}}}, as_call=False)
        assert e["digest"]["status"] == status and e["digest"]["counts"] == {"sha512": digest}
        assert [f["name"].split(" [")[0] for f in e["failures"]] == ([f"lowering/pin-mismatch {d.id}"] if status == "mismatch" else [])


def test_redundant_ands_are_counted():
    from verity_flock import gf2

    C = gf2.Circuit()
    a, b = C.input("a", 2)
    x = C.AND(a, b)
    y = C.AND(b, a)                       # the same product again
    z = C.AND(x ^ b, y ^ b)               # equal operands once y is x
    C.output("o", [x, y, z])
    assert CK.redundant_ands(C)[0] == 2


def test_a_commit_is_a_copy_not_a_redundant_and():
    """A commit row (``A = form, B = 1``) is never a redundant AND, whatever it copies; an AND that reads one reads its form."""
    from verity_flock import gf2

    C = gf2.Circuit()
    a, b, c = C.input("a", 3)
    (x,) = C.commit([a ^ b])              # a carried form committed: a row, no product
    (y,) = C.commit([a ^ b])              # the same form again, and one bit's NOT: copies too
    (n,) = C.commit([c.inv()])
    C.output("o", [x, y, n, C.AND(x, c)])
    assert CK.redundant_ands(C)[0] == 0 and sum(CK.is_commit(C, v) for v in C.live()) == 3
    C.output("p", [C.AND(y, c), C.AND(n, c)])   # AND(x, c) again once y is x, and c AND NOT c
    assert CK.redundant_ands(C)[0] == 2


def test_a_genuine_and_with_one_is_a_redundant_and():
    """An AND whose operand is the constant 1 once a duplicate is replaced is a product computed for nothing, not a commit."""
    from verity_flock import gf2

    C = gf2.Circuit()
    a, b, c = C.input("a", 3)
    p, q = C.AND(a, b), C.AND(a, b)       # one product twice
    g = C.AND(c, p ^ q ^ gf2.ONE)         # c AND 1 once q is p
    C.output("o", [g])
    (gv,), (qv,) = g.s, q.s
    assert not CK.is_commit(C, gv)
    assert CK.redundant_ands(C, owner=str) == (2, {str(qv): 1, str(gv): 1})


def test_a_template_s_unit_is_checked_as_its_lowering_built_it_and_traced_only_to_name_a_redundant_and(monkeypatch):
    """`template_lowering` reads the unit circuit its lowering built (a traced build makes the same circuit): its type, export
    and redundant ANDs; it builds the unit again, traced, only to name where a redundant AND is, as the traced build names it."""
    from collections import Counter

    from verity_flock import lowerings
    from verity_flock import subcircuit_templates as TM

    sub = TM.TEMPLATES["rope-head"].bind(D=8)
    low = lowerings.registry("C-Flock").module("rope-head").lowering(sub)
    C, root = CK._traced(low.units.circuit)
    assert CK.lowering_type(low.circuit) == CK.lowering_type(C) and low.circuit.kind == C.kind
    traced: list[int] = []
    real = CK._traced
    monkeypatch.setattr(CK, "_traced", lambda build: traced.append(1) or real(build))
    e = CK.check_subcircuit(sub, OPTS, pins=PINS)
    assert not e["failures"] and e["lowering"][0]["redundant_ands"] == {"count": 0, "where": {}} and not traced
    monkeypatch.setattr(CK, "redundant_ands", lambda C, owner=None: (1, Counter([owner(len(C.kind) - 1)])))
    (got,) = CK.check_subcircuit(sub, OPTS, pins=PINS)["lowering"]
    assert got["redundant_ands"] == {"count": 1, "where": {CK._owner_names(root, len(C.kind))(len(C.kind) - 1): 1}}
    assert len(traced) == 1


def _whole_boolean(name, build):
    """`checks.whole_lowering` of the Boolean Definition of three bits that ``build`` computes: its record and findings."""
    from verity.circuits.boolean.trace import definition
    from verity.circuits.defs import bind
    from verity.circuits.evaluation.reference import resolve, signature_of
    from verity.definitions.ops.boolean import scalar as SB

    t = resolve(bind(definition(name, 1, build, (("a", SB.BIT), ("b", SB.BIT), ("c", SB.BIT)), SB.BIT, register=False)))
    F = CK.Findings()
    _corr, arrays, ref = CK.correspondence(t, OPTS, F)
    return CK.whole_lowering(t, signature_of(t)[0], arrays, ref, OPTS, F, PINS), F


def test_a_builders_commit_is_a_row_of_its_lowering_and_no_redundant_gate():
    """A builder that commits a carried form (`forms.Circuit.commit`, as `ChargeCount_v1` commits its carries) lowers to a
    commit row, counted under ``commits``, and no redundant-gates warning.  One that commits a form twice and ANDs c with
    their sum plus one writes a genuine AND(c, 1), no commit: that warns, and is an AND its Definition doesn't have."""
    from verity_flock.gf2 import ONE

    def once(C, a, b, c):
        (x,) = C.commit([a[0] ^ b[0]])
        return [C.AND(x, c[0])]

    def twice(C, a, b, c):
        (x,), (y,) = C.commit([a[0] ^ b[0]]), C.commit([a[0] ^ b[0]])
        return [C.AND(c[0], x ^ y ^ ONE) ^ C.AND(a[0], b[0])]

    def redundant(F):
        return [w["count"] for w in F.warnings if (w["check"], w["code"]) == ("redundant-gates", "boolean")]

    rec, F = _whole_boolean("CircuitCheckCommitOnce", once)
    assert (rec["mismatches"], rec["ands"], rec["commits"], rec["definition_ands"], rec["redundant_ands"]) == (0, 2, 1, 1, 0)
    assert redundant(F) == [] and F.failures == []
    rec, F = _whole_boolean("CircuitCheckCommitTwice", twice)
    assert (rec["mismatches"], rec["ands"], rec["commits"], rec["definition_ands"], rec["redundant_ands"]) == (0, 4, 2, 1, 1)
    assert redundant(F) == [1] and [f["code"] for f in F.failures] == ["ands-added"]


@pytest.mark.slow
def test_the_cli_writes_a_report_and_fails_by_name(tmp_path, capsys):
    from circuit_check import cli

    # Every target here is a Call of a served Program, which `targets.is_call` decides without the catalog.  Uncached: a cache
    # would trace this test process for good (`cache.trace`).
    call = ["-q", "--no-cache"]
    out = tmp_path / "r.json"
    assert cli.main(["RoPEHead_v1{D=8}", BOOL_RECOMPUTE, "--out", str(out), *call]) == 0
    r = json.loads(out.read_text())
    assert r["schema"] == CK.SCHEMA and [t["target"] for t in r["targets"]] == ["RoPEHead_v1{D=8}", BOOL_RECOMPUTE]
    assert r["summary"]["ok"] and all(t["kind"] == "definition" for t in r["targets"])
    assert cli.main(["RoPEHead_v1{D=8}", "--lower-max-ands", "1", "--out", str(out), *call]) == 0
    [real] = json.loads(out.read_text())["targets"][0]["correspondence"]["realizations"]
    assert not real["lowered"] and real["reason"].endswith("over the whole-Definition budget (1)")
    capsys.readouterr()
    fp8 = "ScaledMmFp8Block_v1{K=128,N=128,G=128}"           # no Call: its shared-scale form is served in its place
    assert cli.main([fp8, *call]) == 0
    with pytest.MonkeyPatch.context() as mp:                  # as a Call, its coordinates' copies of each scale product fail by name
        mp.setattr(TG, "_CALLS", set())
        mp.setitem(TG._SHORT, "words", TG.word_call_families() | {"ScaledMmFp8Block_v1"})
        assert cli.main([fp8, *call]) == 1
    assert f"FAIL partition/gate-recomputed {fp8}" in capsys.readouterr().out
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
    """A check that raised is no verdict (such as an ImportError from a tree a concurrent run was rewriting, replayed on the next
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


@pytest.mark.parametrize("racing", ["record", "entry"])
def test_two_writers_of_one_entry_at_once_both_store_it(tmp_path, monkeypatch, racing):
    """Two checks on one node that store the same entry, and so the same read records, don't share a temporary file, where the
    second's rename would find it gone (FileNotFoundError) and fail the whole check: the other writer stores it whole between
    the first's write of a read record, or of the entry, and its rename."""
    import subprocess

    from circuit_check import cache as CC

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "a.txt").write_text("a")
    got = {"files": ["a.txt"], "dirs": [], "probes": [], "libs": []}
    k, rep = {"version": "v", "request": {}}, {"failures": [], "warnings": []}
    CC._VISIBLE.clear()
    CC._MEMO.clear()
    replace, inner = os.replace, []

    def other(src, dst):
        if not inner and (CC.READS in Path(dst).parts) == (racing == "record"):
            inner.append(None)
            inner[0] = CC.store(tmp_path / "cache", k, tmp_path, got, rep)
        replace(src, dst)

    monkeypatch.setattr(CC.os, "replace", other)
    eid = CC.store(tmp_path / "cache", k, tmp_path, got, rep)
    assert eid and inner == [eid]
    assert [p.name for p in (tmp_path / "cache").glob("*/*/*.json")] == [f"{eid}.json"]
    assert [p for p in (tmp_path / "cache").rglob("*") if p.is_file() and p.suffix != ".json"] == []
    CC._RECORDS.clear()
    assert CC.read(next((tmp_path / "cache").glob("*/*/*.json")))["reads"] == CC._digests(tmp_path, got)
    CC._VISIBLE.clear()
    CC._MEMO.clear()


def test_an_entry_in_the_old_form_is_a_miss(tmp_path):
    """Every key holds the cache's schema (`cache.key`), so an entry stored in the old form, its reads whole in it, is under
    another key and never looked up; one under a key of the new form anyway is a miss, not a failure, and the run stores the
    entry again."""
    import subprocess

    from circuit_check import cache as CC

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "a.txt").write_text("a")
    got = {"files": ["a.txt"], "dirs": [], "probes": [], "libs": []}
    k, rep = {"version": "v", "request": {}}, {"failures": [], "warnings": []}
    CC._VISIBLE.clear()
    CC._MEMO.clear()
    assert CC.key({"target": ["definition", "X_v1"]})["schema"] == CC.SCHEMA
    old = CC._dir(tmp_path / "cache", k) / "old.json"
    old.parent.mkdir(parents=True)
    old.write_text(json.dumps({"key": k, "reads": CC._digests(tmp_path, got), "report": rep}))
    assert CC.lookup(tmp_path / "cache", k, tmp_path) is None and CC.recorded(tmp_path / "cache", k, tmp_path) is None
    assert f"{old}: KeyError: 'own'; counted as no entry" in CC.unread()
    eid = CC.store(tmp_path / "cache", k, tmp_path, got, rep)
    assert CC.lookup(tmp_path / "cache", k, tmp_path) == (rep, eid)
    CC._VISIBLE.clear()
    CC._MEMO.clear()


@pytest.mark.slow
@_opt_in("~2.5 h")
def test_parallel_jobs_give_the_serial_report(tmp_path):
    """It checks four real targets serially and in worker processes, so it catches a `--jobs` worker whose report differs from
    the serial one. Opt-in: run it after a change to `cli.run`, `cli.parallel` or the task list's order. Only an isolated
    result has ``peak_gb`` and ``rss_start_gb`` (`cli.run`), so they are stripped with ``seconds`` and ``seconds_by_check``."""
    from circuit_check import cli

    specs = ["F32Mul_v1", "RMSNormTriton_v1{N=64,EPS=1e-05}", "TokenSelect_v1{V=64}", "rope-head"]
    measured = {"seconds", "seconds_by_check", "peak_gb", "rss_start_gb"}
    strip = lambda rep: [{k: v for k, v in t.items() if k not in measured} for t in rep["targets"]]  # noqa: E731
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


def test_the_cache_key_is_the_same_under_two_builds_of_one_python_release(monkeypatch):
    """Two machines' builds of one Python release differ in `sys.version` by their build's date; a report one stored must
    count on the other, while another release is another key."""
    import re
    import sys

    from circuit_check import cache as CC

    here = CC.key({"target": "F32Mul_v1"})
    built = re.sub(r"\(([^,]*), [^)]*\)", r"(\1, Jan  1 2000, 00:00:00)", sys.version, count=1)
    assert built != sys.version
    monkeypatch.setattr(sys, "version", built)
    assert CC.key({"target": "F32Mul_v1"}) == here
    v = sys.version_info
    monkeypatch.setattr(sys, "version", built.replace(f"{v.major}.{v.minor}.{v.micro}", f"{v.major}.{v.minor}.{v.micro + 1}", 1))
    assert CC.key({"target": "F32Mul_v1"}) != here


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


def test_the_cache_key_is_the_same_on_every_node(monkeypatch, tmp_path):
    """A report one node stored counts on every other node of the image (`check.py --plan`, `research merge --verdicts`): the key
    names no host, kernel, working directory, home, PATH or check slot (each slot sets its own `VERITY_TEST_CACHE`,
    `tools/verity/check/slot.py`), while the CPU, a variable that can change what a check computes and the request still key it."""
    import platform

    from circuit_check import cache as CC

    def node(name: str) -> dict:
        home = tmp_path / name
        return {"HOME": str(home), "PATH": f"{home}/.local/bin:/usr/bin", "HOSTNAME": name, "TMPDIR": f"{home}/tmp",
                "VIRTUAL_ENV": f"{home}/trees/{name}/.venv", "SSH_CONNECTION": f"10.0.0.{len(name)} 22", "CUDA_VISIBLE_DEVICES": "",
                "VERITY_TEST_CACHE": f"{home}/.cache/verity/tests-slot-{name}", "CHECK_SLOT": name, "VERITY_REGRESSION": "0"}

    def on(name: str) -> None:
        for k, v in node(name).items():
            monkeypatch.setenv(k, v)
        (tmp_path / name).mkdir(exist_ok=True)
        monkeypatch.chdir(tmp_path / name)
        monkeypatch.setattr(platform, "node", lambda: name)
        monkeypatch.setattr(platform, "release", lambda: f"6.{len(name)}.0-{name}")
        monkeypatch.setattr(platform, "version", lambda: f"#1 SMP {name}")
        monkeypatch.setattr(platform, "platform", lambda *a, **k: f"Linux-6.{len(name)}.0-{name}-x86_64")

    on("vy-nebius-1")
    here = CC.key({"target": "F32Mul_v1"})
    on("vy-nebius-cpu-11")
    assert CC.key({"target": "F32Mul_v1"}) == here
    assert CC.key({"target": "F32Add_v1"}) != here
    monkeypatch.setenv("VERITY_REGRESSION", "1")
    assert CC.key({"target": "F32Mul_v1"}) != here
    monkeypatch.undo()
    here = CC.key({"target": "F32Mul_v1"})
    monkeypatch.setattr(platform, "machine", lambda: "aarch64")
    assert CC.key({"target": "F32Mul_v1"}) != here                  # a different CPU is a different machine
