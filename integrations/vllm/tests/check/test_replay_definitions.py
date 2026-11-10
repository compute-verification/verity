"""The replay refuses a Definition the Build of record did not bind (`verity_vllm.check.replay.definitions`).

The sampled replay evaluates a row with its family's row kernel, which answers for the Definition in the replay's own tree, while
the Commit bound the Program the Build compiled.  A small Llama's request Program in a Build dir: both loaders
(`tp.rank_replay.load`, `c2_replay.programs_of_record`) take it as built, and refuse it by name once the Build's descriptor holds
another body for a specialization the replay evaluates (the Program a Build at another body of that Definition compiles, digest
of record and all), or is not the Program of record.  Two Definitions under one id show that the check reads bodies, not ids.
Both also refuse a row kernel that fails its self-check against its Definition."""
from __future__ import annotations

import copy
import gzip
import json
from types import SimpleNamespace

import pytest

from tests.program.predict.test_predict import TINY, _phi
from verity.ml.circuits import Array, Program, Value
from verity.ml.circuits.codec import descriptor_id, encode_program, program_digest
from verity.ml.circuits.defs import REGISTRY, CompositeDefinition, PrimitiveDefinition, bind
from verity_vllm.check.replay import definitions as DF
from verity_vllm.pipeline import c2_replay as C2
from verity_vllm.pipeline.build import _instance_rows
from verity_vllm.pipeline.tp import rank_replay as RR
from verity_vllm.program.predict.build import build_program

NORM = "RMSNormTriton_v5{"
REQ = {"request_id": "q0", "index": 0, "prompt_token_ids": [3, 1, 4, 1], "max_tokens": 2, "arrive_step": 0}
V16 = Value(16)


@pytest.fixture(scope="module")
def program():
    return build_program(_phi(TINY), "request", LP=4, T=1).program


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
    from verity.ml.definitions.numerics import derived_rows as DR

    dg = _build(tmp_path / "built", program)
    monkeypatch.setattr(DF, "_KERNEL_CHECKS", {})
    assert DF.kernel_refusal(DF.evaluated_specs(_instance_rows(program))) is None
    right = DR.rmsnorm_triton_rows
    monkeypatch.setattr(DR, "rmsnorm_triton_rows", lambda x, w, eps: {"out": right(x, w, eps)["out"] ^ 1})
    monkeypatch.setattr(DF, "_KERNEL_CHECKS", {})
    with pytest.raises(RuntimeError, match=r"^rank 0: REFUSED \(a row kernel is not its Definition\): RMSNormTriton_v5: "):
        _rank(tmp_path / "built", dg)
    monkeypatch.setattr(DF, "_KERNEL_CHECKS", {})
    with pytest.raises(RuntimeError, match=r"^replay: REFUSED \(a row kernel is not its Definition\): RMSNormTriton_v5: .* mismatches"):
        _c2(tmp_path / "built", dg)


def test_a_build_whose_descriptor_is_not_the_program_of_record_is_refused(tmp_path, program):
    d = tmp_path / "built"
    dg = _build(d, program)
    specs = DF.evaluated_specs(_instance_rows(program))
    assert any(s.startswith(NORM) for s in specs) and DF.refusal(str(d), dg, specs) is None
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
    assert DF.refusal(str(d), dg, [fid]) is None
    monkeypatch.setitem(REGISTRY.defs, built.id, moved)
    why = DF.refusal(str(d), dg, [fid])
    assert f"{fid}: {fid} is not the Definition the Build bound" in why
    assert f"{fid}: this tree's reaches {sub.id}, which the Build's descriptor does not hold" in why
