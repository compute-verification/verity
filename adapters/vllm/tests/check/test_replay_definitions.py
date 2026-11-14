"""The replay refuses a Definition the Build of record did not bind (`verity_vllm.check.replay.definitions`).

The sampled replay evaluates a row with its family's row kernel, which answers for the Definition in the replay's own tree, while
the Commit bound the Program the Build compiled.  A small Llama's request Program in a Build dir: both loaders
(`tp.rank_replay.load`, `c2_replay.programs_of_record`) take it as built, and refuse it by name once the Build's descriptor holds
another body for a specialization the replay evaluates (the Program a Build at another body of that Definition compiles, digest
of record and all), or is not the Program of record.  Two Definitions under one id show that the check reads bodies, not ids.
Both also refuse a row kernel that fails its self-check against its Definition.

A Boolean Build's descriptor holds its Boolean Definitions, and each of its rows names one (``bits``) beside the word Definition
the row is graded as (``spec``).  A small Program lifted to bits and written as the Build writes it: both loaders take it, and
refuse it by name once a Boolean body moved, a row's ``bits`` names another word Definition than its ``spec``, or a row names no
``bits``.  A Boolean row's kernel is also held to its ``bits`` family's gates, on the kernel self-check's own specialisations and
inputs: both loaders refuse gates that give another word, and the check refuses a ``bits`` family none of whose Definitions
names those specialisations as its word and a function the Build binds at another version than the gates were checked with.
What a pass still leaves unshown is on the replay's record, and the verdict's rule carries it."""
from __future__ import annotations

import copy
import gzip
import json
from types import SimpleNamespace

import pytest
from verity.circuits import Array, Program, Value
from verity.circuits.codec import descriptor_id, encode_program, program_digest
from verity.circuits.defs import (
    REGISTRY,
    Builder,
    CompositeDefinition,
    PrimitiveDefinition,
    bind,
)
from verity.circuits.refs import Affine, Coll
from verity.circuits.types import RecordType
from verity.definitions import boolean as BB
from verity.definitions.numerics import derived_rows as DR
from verity.definitions.registry.catalog import load
from verity.definitions.replay import replayer

from tests.program.predict.test_predict import TINY, _phi
from verity_vllm.check import c2_rules
from verity_vllm.check.replay import definitions as DF
from verity_vllm.pipeline import c2_replay as C2
from verity_vllm.pipeline.build import _instance_rows
from verity_vllm.pipeline.build_instances import write_instances
from verity_vllm.pipeline.tp import rank_replay as RR
from verity_vllm.program.frontend.boolean import derived
from verity_vllm.program.frontend.derive import DeriveReport
from verity_vllm.program.predict.build import build_program
from verity_vllm.query.manifest.format import spec_family

NORM = "RMSNormTriton_v5{"
SILU, SILU_BITS, PICK, PICK_BITS = "SiluMul_v1{I=4}", "SiluMul_v3{I=4}", "TokenSelect_v1{V=8}", "TokenSelect_v2{V=8}"
REQ = {"request_id": "q0", "index": 0, "prompt_token_ids": [3, 1, 4, 1], "max_tokens": 2, "arrive_step": 0}
V16, V32 = Value(16), Value(32)


@pytest.fixture(scope="module")
def program():
    return build_program(_phi(TINY), "request", LP=4, T=1).program


@pytest.fixture(scope="module")
def boolean():
    """The report of a derived function on bits, as the Build of a Boolean target takes it (`program.frontend.boolean.derived`):
    an unregistered ``Derived[...]`` composite of five Calls of word Definitions the replay grades by row kernels, lifted."""
    V, H = 8, 4
    defs = load()[0].defs
    weights = RecordType(("embed", Array(V, Array(H, V16))), ("up", Array(2 * H, Array(H, V16))), ("head", Array(V, Array(H, V16))))
    params = (("prompt", Array(1, V32)), ("weights", weights))
    B = Builder(params, V32)
    prompt, w = (Coll(t, Affine("p", i, 0, t.leaves)) for i, (_, t) in enumerate(params))
    x = B.call(bind(defs["Embedding_v1"], V=V, H=H), prompt[0], w.field("embed"))
    gu = B.call(bind(defs["Gemm_v1"], K=H, N=2 * H), x, w.field("up"))
    logits = B.call(bind(defs["Gemm_v1"], K=H, N=V), B.call(bind(defs["SiluMul_v1"], I=H), gu), w.field("head"))
    body = B.finish(B.call(bind(defs["TokenSelect_v1"], V=V), logits))
    fn = bind(CompositeDefinition("Derived[Tiny]", 1, ("SRC",), lambda S: (params, V32), replayer(body), register=False),
              SRC={"class": "tests.Tiny"})
    fn.set_body(body)
    word = Program(fn)
    desc = encode_program(word)
    prov = {"result": {"root": descriptor_id(fn), "digest": program_digest(desc), "gates": word.gates}}
    rep = derived(DeriveReport(word, None, prov, prov["result"]["digest"], descriptor_id(fn), word.gates, descriptor=desc))
    assert rep.ok and rep.program.word is word, rep.unsupported
    return rep


def _write(d, desc):
    d.mkdir(parents=True)
    with gzip.open(d / "descriptor.json.gz", "wt") as f:
        json.dump(desc, f)
    return program_digest(desc)


def _build(d, program, desc=None) -> str:
    """Build dir ``d`` of ``program`` as the replay loads it, with descriptor ``desc`` (the program's by default); its digest."""
    dg = _write(d, encode_program(program) if desc is None else desc)
    inst = {"schema": "cb-a/instance-sequence/v1", "digest": dg, "params": [[n, str(t)] for n, t in program.fn.body.params],
            "rows": list(_instance_rows(program))}
    with gzip.open(d / "instances.json.gz", "wt") as f:
        json.dump(inst, f)
    (d / "result.json").write_text(json.dumps({"ok": True, "program": {"digest": dg}}))
    return dg


def _boolean_build(d, rep, desc=None, row=None) -> str:
    """Build dir ``d`` of the Boolean report ``rep`` as the Build writes one (`build._write_program`): its descriptor (``desc`` in
    its place), its instance sequence through the word link, each row as ``row`` returns it, and its word Program; its digest."""
    dg = _write(d, rep.descriptor if desc is None else desc)
    write_instances(str(d), rep.program, dg, None, {})
    if row is not None:
        with gzip.open(d / "instances.json.gz", "rt") as f:
            inst = json.load(f)
        inst["rows"] = [row(r) for r in inst["rows"]]
        with gzip.open(d / "instances.json.gz", "wt") as f:
            json.dump(inst, f)
    word = BB.write_word(d, rep.program, rep.provenance["ir"]["word_digest"])
    (d / "result.json").write_text(json.dumps({"ok": True, "program": {"digest": dg, "word_digest": word}}))
    return dg


def _unchecked(monkeypatch) -> None:
    """Forget what this process already checked of the row kernels, so the next check runs."""
    monkeypatch.setattr(DF, "_KERNEL_CHECKS", {})
    monkeypatch.setattr(DF, "_GATE_CHECKS", {})


def _moved(desc: dict, prefix: str = NORM) -> tuple[dict, str]:
    """``desc`` with one node of its first ``prefix`` specialization reading its two operands the other way round, and that
    specialization's id."""
    desc = copy.deepcopy(desc)
    for fid, e in desc["definitions"].items():
        if fid.startswith(prefix):
            for nd in e["body"]["nodes"]:
                a = nd["args"]
                if len(a) == 2 and a[0]["type"] == a[1]["type"] and a[0]["refs"] != a[1]["refs"]:
                    a.reverse()
                    return desc, fid
    raise AssertionError(f"no {prefix} node with two operands of one type")


def _rank(d, dg):
    return RR.load({"requests": [REQ], "programs": {"0": str(d)}, "of_record": {"0": [dg]}}, 0, 1)


def _c2(d, dg):
    site = SimpleNamespace(args=SimpleNamespace(program_digest=dg, programs_root=None, program_dir=str(d)), manifest=None, reqs=[REQ],
                           sampling={})
    return C2.programs_of_record(site, C2.request_labels([REQ]))


def test_the_rank_replay_refuses_a_body_its_build_did_not_bind(tmp_path, program):
    dg = _build(tmp_path / "built", program)
    assert _rank(tmp_path / "built", dg).digs == {"q0": dg}
    desc, fid = _moved(encode_program(program))
    moved = _build(tmp_path / "moved", program, desc)
    with pytest.raises(RuntimeError, match=r"^rank 0: REFUSED \(not the Definitions of record\)") as e:
        _rank(tmp_path / "moved", moved)
    assert f"{fid}: {fid} is not the Definition the Build bound" in str(e.value)


def test_the_c2_replay_refuses_it_too(tmp_path, program):
    dg = _build(tmp_path / "built", program)
    assert _c2(tmp_path / "built", dg)[1] == {"r0": dg}
    desc, fid = _moved(encode_program(program))
    moved = _build(tmp_path / "moved", program, desc)
    with pytest.raises(RuntimeError, match=r"^replay: REFUSED \(not the Definitions of record\)") as e:
        _c2(tmp_path / "moved", moved)
    assert fid in str(e.value)


def test_both_loaders_refuse_a_row_kernel_that_is_not_its_definition(tmp_path, program, monkeypatch):
    """The Build is of record, but the row kernel the replay grades its norm rows by is one bit off its Definition: refused by
    name before any row is graded, by the kernel's self-check."""
    dg = _build(tmp_path / "built", program)
    _unchecked(monkeypatch)
    assert DF.kernel_refusal(DF.evaluated_specs(_instance_rows(program))) is None
    right = DR.rmsnorm_triton_rows
    monkeypatch.setattr(DR, "rmsnorm_triton_rows", lambda x, w, eps: {"out": right(x, w, eps)["out"] ^ 1})
    _unchecked(monkeypatch)
    with pytest.raises(RuntimeError, match=r"^rank 0: REFUSED \(a row kernel is not its Definition\): RMSNormTriton_v5: "):
        _rank(tmp_path / "built", dg)
    _unchecked(monkeypatch)
    with pytest.raises(RuntimeError, match=r"^replay: REFUSED \(a row kernel is not its Definition\): RMSNormTriton_v5: .* mismatches"):
        _c2(tmp_path / "built", dg)


def test_a_build_whose_descriptor_is_not_the_program_of_record_is_refused(tmp_path, program):
    d = tmp_path / "built"
    dg = _build(d, program)
    specs = DF.evaluated_specs(_instance_rows(program))
    assert any(spec.startswith(NORM) and bound == spec for spec, bound in specs) and DF.refusal(str(d), dg, specs) is None
    assert f"is Program {dg[:16]}, not the Program of record {'ab' * 8}" in DF.refusal(str(d), "ab" * 64, specs)
    (d / "descriptor.json.gz").unlink()
    assert "has no descriptor.json.gz" in DF.refusal(str(d), dg, specs)
    assert DF.refusal(str(d), dg, []) is None      # a Program the replay evaluates no row of


def test_the_check_reads_bodies_not_ids(tmp_path, monkeypatch):
    add = PrimitiveDefinition("SkewTestAdd16", 1, (("a", V16), ("b", V16)), V16, lambda a, b: (a + b) & 0xFFFF, register=False)
    sub = PrimitiveDefinition("SkewTestSub16", 1, (("a", V16), ("b", V16)), V16, lambda a, b: (a - b) & 0xFFFF, register=False)
    for p in (add, sub):
        monkeypatch.setitem(REGISTRY.defs, p.id, p)

    def fold(op):
        def make(B, S, x):
            acc = x[0]
            for k in range(1, S.N):
                acc = B.call(op, acc, x[k])
            return acc
        return CompositeDefinition("SkewTestFold", 1, ("N",), lambda S: ((("x", Array(S.N, V16)),), V16), make, register=False)

    built, moved = fold(add), fold(sub)
    spec = bind(built, N=3)
    caller = CompositeDefinition("SkewTestCaller", 1, (), lambda S: spec.signature(), lambda B, S, x: B.call(spec, x), register=False)
    fid, d = descriptor_id(spec), tmp_path / "built"
    dg = _write(d, encode_program(Program(caller)))
    monkeypatch.setitem(REGISTRY.defs, built.id, built)
    assert DF.refusal(str(d), dg, [(fid, fid)]) is None
    monkeypatch.setitem(REGISTRY.defs, built.id, moved)
    why = DF.refusal(str(d), dg, [(fid, fid)])
    assert f"{fid}: {fid} is not the Definition the Build bound" in why
    assert f"{fid}: this tree's reaches {sub.id}, which the Build's descriptor does not hold" in why


def test_both_loaders_take_a_boolean_build_and_refuse_a_boolean_body_it_did_not_bind(tmp_path, boolean):
    dg = _boolean_build(tmp_path / "built", boolean)
    assert _rank(tmp_path / "built", dg).digs == {"q0": dg} and _c2(tmp_path / "built", dg)[1] == {"r0": dg}
    held, specs = set(boolean.descriptor["definitions"]), DF.evaluated_specs(_instance_rows(boolean.program))
    assert (SILU, SILU_BITS) in specs and all(bound in held and spec not in held for spec, bound in specs)
    desc, fid = _moved(boolean.descriptor, SILU_BITS)
    moved = _boolean_build(tmp_path / "moved", boolean, desc)
    for loader, what in ((_rank, "rank 0"), (_c2, "replay")):
        with pytest.raises(RuntimeError, match=rf"^{what}: REFUSED \(not the Definitions of record\)") as e:
            loader(tmp_path / "moved", moved)
        assert f"{fid}: {fid} is not the Definition the Build bound" in str(e.value)


def test_a_boolean_row_is_held_to_the_word_definition_it_is_graded_as(tmp_path, boolean):
    """Each of these Builds is of record and every Boolean Definition its rows name is the one it bound, but the replay would
    grade a row by the kernel of a word Definition its ``bits`` does not compute, or of one no Definition of the Build answers
    for: a row whose ``bits`` names another word Definition than its ``spec``, and rows that name no ``bits``."""
    other = _boolean_build(tmp_path / "other", boolean, row=lambda r: {**r, "bits": PICK_BITS} if r["spec"] == SILU else r)
    bare = _boolean_build(tmp_path / "bare", boolean, row=lambda r: {k: v for k, v in r.items() if k != "bits"})
    for loader, what in ((_rank, "rank 0"), (_c2, "replay")):
        with pytest.raises(RuntimeError, match=rf"^{what}: REFUSED \(not the Definitions of record\)") as e:
            loader(tmp_path / "other", other)
        assert f"{PICK_BITS}: this tree's names the word Definition {PICK}, and its row is graded as {SILU}" in str(e.value)
        with pytest.raises(RuntimeError, match=rf"^{what}: REFUSED \(not the Definitions of record\)") as e:
            loader(tmp_path / "bare", bare)
        assert f"{SILU}: this tree does not rebuild it" in str(e.value) and f"the Build's descriptor holds no {SILU}" in str(e.value)


def test_both_loaders_refuse_a_boolean_row_whose_kernel_is_not_its_gates(tmp_path, boolean, monkeypatch):
    """The Build is of record and each row kernel is its word Definition, but the gates of the Boolean Definition the Build
    bound its SiluMul rows under give another word than the kernel: refused by name before any row is graded.  Every family
    whose rows carry ``bits`` is compared with that ``bits`` family's gates."""
    from verity.circuits import evaluation as EV

    dg = _boolean_build(tmp_path / "built", boolean)
    specs = DF.evaluated_specs(_instance_rows(boolean.program))
    _unchecked(monkeypatch)
    assert DF.kernel_refusal(specs) is None
    assert dict(DF._GATE_CHECKS) == {(spec_family(spec), spec_family(bound)): ([], set()) for spec, bound in specs}
    right = EV.evaluate_words
    monkeypatch.setattr(EV, "evaluate_words", lambda b, *a: right(b, *a) ^ (descriptor_id(b) == "SiluMul_v3{I=8}"))
    for loader, what in ((_rank, "rank 0"), (_c2, "replay")):
        _unchecked(monkeypatch)
        with pytest.raises(RuntimeError, match=rf"^{what}: REFUSED \(a row kernel is not its Definition\): SiluMul_v1: SiluMul_v1{{I=8}}: "
                                                 rf"the row kernel's rows are not SiluMul_v3{{I=8}}'s gates on (\d) of the \1 inputs"):
            loader(tmp_path / "built", dg)


def test_the_gates_are_compared_on_the_self_check_s_own_inputs(monkeypatch):
    """The kernel is asked for the same rows twice, at the same specialisation: by its self-check against the word Definition,
    then against the gates."""
    from verity.definitions.numerics import derived_rows as DR

    asked, right = [], DR.silu_mul_rows

    def rows(gu):
        asked.append(gu.tobytes())
        return right(gu)

    monkeypatch.setattr(DR, "silu_mul_rows", rows)
    _unchecked(monkeypatch)
    assert DF.kernel_refusal([(SILU, SILU_BITS)]) is None
    assert len(asked) == 2 * DF.SAMPLES and asked[:DF.SAMPLES] == asked[DF.SAMPLES:] and len(set(asked)) == DF.SAMPLES


def test_a_boolean_row_s_gates_are_its_bits_family_s_with_the_functions_its_build_binds(monkeypatch):
    """A row whose ``bits`` is of a family that does not name the kernel's check specialisations as its word is refused: no
    gates of that family were compared.  The gates compared are those of the version the row names, not of the newest that
    names the word (`MoeSum_v1` has two).  A row whose Build binds a function at another version than the gates were checked
    with is refused; a function the Build binds nowhere (the check's other dot) is this tree's."""
    from verity.circuits import evaluation as EV

    _unchecked(monkeypatch)
    assert DF.kernel_refusal([(SILU, "SiluMul_v4{I=4}")]) == \
        "SiluMul_v1: no Boolean version of SiluMul_v1: SiluMul_v1{I=8}: no registered SiluMul_v4 names it as its word"
    right, evaluated = EV.evaluate_words, []

    def recorded(b, *inputs):
        evaluated.append(descriptor_id(b))
        return right(b, *inputs)

    monkeypatch.setattr(EV, "evaluate_words", recorded)
    assert DF.kernel_refusal([("MoeSum_v1{TOPK=4,H=16}", "MoeSum_v2{TOPK=4,H=16}")]) is None
    assert DF.kernel_refusal([("MoeSum_v1{TOPK=4,H=16}", "MoeSum_v3{TOPK=4,H=16}")]) is None
    assert evaluated == ["MoeSum_v2{TOPK=2,H=8}", "MoeSum_v3{TOPK=2,H=8}"]
    gemm = 'Gemm_v2{K=4,N=4,DOT={"fn":"HopperBF16WgmmaDot16_v1"}}'

    def bits(dot: str) -> str:
        return 'Gemm_v3{K=4,N=4,DOT={"fn":"' + dot + '"}}'

    assert DF.kernel_refusal([(gemm, bits("AnotherDot16_v1"))]) is None
    whys, checked_with = DF._GATE_CHECKS["Gemm_v2", "Gemm_v3"]
    (hopper,) = (f for f in checked_with if f.startswith("HopperBF16WgmmaDot16_v"))
    assert whys == [] and DF.kernel_refusal([(gemm, bits(hopper))]) is None
    refused = f"Gemm_v2: its gates are checked with {hopper}, and the Build binds {hopper}0"
    assert DF.kernel_refusal([(gemm, bits(f"{hopper}0"))]) == refused
    assert DF.kernel_refusal([(gemm, bits('Outer_v1{F={\\"fn\\":\\"' + hopper + '0\\"}}'))]) == refused


@pytest.mark.parametrize("error", [ValueError, AttributeError])
def test_a_specialisation_whose_gates_this_tree_does_not_build_is_refused_by_name(monkeypatch, error):
    """A Boolean Definition whose body is not built at one of the kernel's check specialisations (statics its lowering does
    not take, whichever way its builder fails on them) is a refusal that names it, not an exception out of the replay."""
    from verity.circuits import evaluation as EV

    def unbuilt(b, *inputs):
        raise error("unsupported statics")

    monkeypatch.setattr(EV, "evaluate_words", unbuilt)
    _unchecked(monkeypatch)
    assert DF.kernel_refusal([(SILU, SILU_BITS)]) == ("SiluMul_v1: SiluMul_v1{I=8}: this tree does not build SiluMul_v3{I=8}'s gates, so "
                                                      f"no row is held to them ({error.__name__}: unsupported statics)")


def test_a_replay_of_boolean_rows_says_what_its_pass_leaves_unshown(program, boolean):
    """The replay's record carries the line only for rows of a Boolean Definition, and the verdict's sampled-replay rule passes
    with it in its notes."""
    assert f"{DF.SAMPLES} inputs" in DF.BOOLEAN_PREMISE and "nothing holds the kernel to the gates at a served binding" in DF.BOOLEAN_PREMISE
    assert DF.premises(_instance_rows(program)) == [] and DF.premises(_instance_rows(boolean.program)) == [DF.BOOLEAN_PREMISE]

    def rule(sr):
        (r,) = (c for c in c2_rules.c2_rules({"sampled_replay": sr, "boundary_linkage": {"result": True}}, {})
                if c.name == "local_replay/sampled_replay")
        return r

    sr = {"result": True, "grade": "complete"}
    told = rule(sr | {"premises": DF.premises(_instance_rows(boolean.program))})
    assert told.outcome == rule(sr).outcome == "PASS"
    assert DF.BOOLEAN_PREMISE in told.notes and DF.BOOLEAN_PREMISE not in rule(sr).notes
