"""The required-value manifest FORMAT (`manifest_format`): name / spec parsing, the identity digest, the coverage check's negatives
the manifest makes detectable BY CONSTRUCTION (omit a whole producer family; omit one step; drop one member and bind another twice; a member
committed short) and the binding loader's two engine-step shapes -- over manifests of record built by the v2 query
(`test_sampled_replay.v2_manifest`) on a synthetic wired Program (no torch, no pod)."""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

from tests.check.test_sampled_replay import _program, v2_manifest
from verity_vllm.program.instances_file import write_instances_file
from verity_vllm.query.manifest import format as MF
from verity_vllm.query.manifest.format import (
    canonical_identity,
    manifest_digest,
    spec_statics,
    split_name,
)

EPS = 'EPS={"f64":"0x1.4f8b588e368f1p-17"}'


def _layouts_from(man: dict) -> dict[int, list[dict]]:
    """A collector's per-step layouts that cover the manifest exactly (the positive case), in the committed-name vocabulary."""
    lay: dict[int, list[dict]] = {}
    for r in man["identities"]:
        if r["family"] == "weights":
            continue
        lay.setdefault(int(r["step"]), []).append({"name": f"{r['op_path']}/{r['output_member']}", "numel": r["element_range"][1], "inv": r["invocation"]})
    for step in lay:                     # committed order: a module's k-th call is the k-th occurrence in the step
        lay[step].sort(key=lambda m: (m["name"], m["inv"]))
    return lay


def test_parsing():
    assert spec_statics("Gemm_v1{K=576,N=960}") == {"K": 576, "N": 960}
    assert spec_statics(f"RMSNormFusedCuda_v2{{N=576,{EPS}}}")["N"] == 576
    assert split_name("model.model.layers.3.self_attn.rotary_emb/rotary_embedding_3/key[7]") == ("model.model.layers.3.self_attn.rotary_emb", "rotary_embedding_3", "key", 7)
    assert split_name("argmax_2[0]") == ("", "argmax_2", "", 0)


def test_digest_determinism():
    inst, _woff = _program()
    man = v2_manifest(inst)
    assert man["complete"] is True and man["steps"] == [0, 1, 2] and man["engine_steps"] == [0, 1, 2]
    assert v2_manifest(inst)["manifest_digest"] == man["manifest_digest"]                     # rebuild -> same
    ids = copy.deepcopy(man["identities"])
    for r in ids:                                                                             # annotations do not enter
        r.pop("geometry", None)
        r.pop("rows", None)
    assert manifest_digest(ids) == man["manifest_digest"]
    rows = b"".join(json.dumps([r["family"]] + canonical_identity(r), separators=(",", ":"), sort_keys=True).encode() + b"\n"
                    for r in sorted(ids, key=MF.row_key))
    assert man["manifest_digest"] == hashlib.sha512(rows).hexdigest() and len(man["manifest_digest"]) == 128
    assert v2_manifest(inst, digest="q" * 64)["manifest_digest"] != man["manifest_digest"]   # the Program digest does
    assert len(canonical_identity(man["identities"][0])) == 8








def _seq_planes(rows: list[int], *, h: int = 4, nb: int = 2, ms: bool = False) -> list[tuple[str, int]]:
    return [(f"s{i}/g{h}x{sl}x{nb}/{c}", h * sl * w) for i, sl in enumerate(rows) for c, w in MF.hidden_stream_widths(nb, 128, 64, ms).items()]


@pytest.mark.parametrize("active, members, why", [
    ([3, 1], _seq_planes([3, 1]), None),
    ([3, 1], _seq_planes([3, 1], ms=True), "s0 carries"),                                   # MS without the guarded-max policy
    ([3], _seq_planes([3], nb=1), None),                                                     # one of two executing: its key blocks may be fewer
    ([3], _seq_planes([3], nb=3), "NB 3"),                                                   # ... never more than the launch declares
    ([3, 1], _seq_planes([3, 1], nb=1), "NB 1"),                                             # every declared sequence executing: the launch's NB
    ([3, 1], _seq_planes([3, 1]) + [("g8x3x2", MF.mat_total_words(8, 3, 2, 128, 64))], "beside the per-sequence planes"),
    ([3, 1], _seq_planes([3, 1]) + _seq_planes([3])[:1], "s0/g4x3x2/S committed twice"),
    ([3, 1], _seq_planes([3, 1]) + [("s0/g4x3x2/bot", 1)], None),                           # a ⊥ padding plane is not the launch's stream
    ([3, 1], [("g8x3x2", MF.mat_total_words(8, 3, 2, 128, 64))], None),
    ([3, 1], [("g8x12x2", MF.mat_total_words(8, 12, 2, 128, 64))], None),                # FA3's rows with its KV group's heads, an old manifest's M without
    ([3, 1], [("g8x2x2", MF.mat_total_words(8, 2, 2, 128, 64))], "2 executing sequence(s) of this launch stage g8x>=3x2"),
    ([3], [("g4x3x2", MF.mat_total_words(4, 3, 2, 128, 64))], None),
    ([3], [("g8x3x2", MF.mat_total_words(8, 3, 2, 128, 64))], "1 executing sequence(s) of this launch stage g4x>=3x<=2")])
def test_batch_launch_shortfall_names_what_the_committed_stream_lacks(active, members, why):
    """`format.batch_launch_shortfall` over a two-sequence FA2 launch (4 heads a sequence, NB 2, the legacy tile): None when the committed
    tensors are its whole stream at the executing sequences' rows, else the shortfall named."""
    geometry = {"HB": 8, "M": 3, "NB": 2, "BN": 128, "D": 64, "n_seq": 2}
    got = MF.batch_launch_shortfall(geometry, active, members)
    assert (got is None) if why is None else (why in (got or "")), got


def _members_by_step_op(ids: list[dict]) -> dict[tuple[int, str], set[str]]:
    out: dict[tuple[int, str], set[str]] = {}
    for r in ids:
        if r["family"] == "instance_outputs" and int(r["invocation"]) == 0:
            out.setdefault((int(r["step"]), r["op_path"]), set()).add(r["output_member"])
    return out




def test_program_dirs_for_passes_over_a_result_it_cannot_read(tmp_path):
    """The lookup by a derive's result.json skips a candidate whose result.json is not JSON or cannot be opened, and takes the
    next one that declares the request's shape; with no such candidate left it refuses by the request's name."""
    fx = {"requests": [{"request_id": "r0", "prompt_len": 168, "max_tokens": 10}]}
    (tmp_path / "build_request_a").mkdir()
    (tmp_path / "build_request_a" / "result.json").write_text('{"ok": true, "LP": 168')
    (tmp_path / "build_request_b").mkdir()
    (tmp_path / "build_request_b" / "result.json").mkdir()
    (tmp_path / "build_request_c").mkdir()
    (tmp_path / "build_request_c" / "result.json").write_text(json.dumps({"ok": True, "LP": 168, "T": 9}))
    assert MF.program_dirs_for(fx, str(tmp_path)) == {"r0": str(tmp_path / "build_request_c")}
    (tmp_path / "build_request_c" / "result.json").write_bytes(b"\xff\xfe{}")
    with pytest.raises(SystemExit, match="r0 .*168,9"):
        MF.program_dirs_for(fx, str(tmp_path))


def test_program_dirs_for_passes_over_a_result_nested_past_the_parser(tmp_path):
    """A result.json nested deeper than Python's parser reads, whole or in one part, is no JSON the lookup can read (`json`
    raises `RecursionError` for it, which is no `ValueError`): the candidate is passed over like one that is not JSON."""
    fx = {"requests": [{"request_id": "r0", "prompt_len": 168, "max_tokens": 10}]}
    nested = "[" * 100_000 + "]" * 100_000
    with pytest.raises(RecursionError):
        json.loads(nested)
    (tmp_path / "build_request_a").mkdir()
    (tmp_path / "build_request_a" / "result.json").write_text(nested)
    (tmp_path / "build_request_b").mkdir()
    (tmp_path / "build_request_b" / "result.json").write_text('{"ok": true, "LP": 168, "T": 9, "C": ' + nested + "}")
    with pytest.raises(SystemExit, match="r0 .*168,9"):
        MF.program_dirs_for(fx, str(tmp_path))
    (tmp_path / "build_request_c").mkdir()
    (tmp_path / "build_request_c" / "result.json").write_text(json.dumps({"ok": True, "LP": 168, "T": 9}))
    assert MF.program_dirs_for(fx, str(tmp_path)) == {"r0": str(tmp_path / "build_request_c")}


def test_load_program_reads_the_dirs_result_and_its_artifact_when_there_is_one(tmp_path):
    """`load_program` returns the Program dir's result.json and artifact.json as the files hold them, and None for an artifact
    the dir does not have."""
    head: dict[str, object] = {"schema": "cb-a/instance-sequence/v2", "digest": "d" * 64, "correspondence_digest": None, "params": []}
    write_instances_file(str(tmp_path / "instances.json.gz"), head, iter([]))
    (tmp_path / "result.json").write_text(json.dumps({"ok": True, "LP": 3}))
    got = MF.load_program(str(tmp_path))
    assert (got["result"], got["artifact"], got["dir"]) == ({"ok": True, "LP": 3}, None, str(tmp_path))
    assert got["instances"] == {**head, "rows": []}
    (tmp_path / "artifact.json").write_text(json.dumps({"program": {"digest": "d" * 64}}))
    assert MF.load_program(str(tmp_path))["artifact"] == {"program": {"digest": "d" * 64}}


def test_programs_for_loads_the_program_of_each_declared_request(tmp_path):
    """One loaded Program a declared request, under its label: the dir `program_dirs_for` finds for its shape, a warm request's
    its own."""
    fx = {"requests": [{"request_id": "r0", "prompt_len": 168, "max_tokens": 10}, {"request_id": "r1", "prompt_len": 168, "max_tokens": 10, "cached": 128}]}
    head: dict[str, object] = {"schema": "cb-a/instance-sequence/v2", "digest": "d" * 64, "correspondence_digest": None, "params": []}
    for name, C in (("build_request_LP168_T9", 0), ("build_request_LP168_T9_C128", 128)):
        (tmp_path / name).mkdir()
        write_instances_file(str(tmp_path / name / "instances.json.gz"), head, iter([]))
        (tmp_path / name / "result.json").write_text(json.dumps({"ok": True, "LP": 168, "T": 9, "C": C}))
    dirs = MF.program_dirs_for(fx, str(tmp_path))
    assert dirs == {"r0": str(tmp_path / "build_request_LP168_T9"), "r1": str(tmp_path / "build_request_LP168_T9_C128")}
    got = MF.programs_for(fx, str(tmp_path))
    assert got == {rid: MF.load_program(d) for rid, d in dirs.items()}
    assert {rid: (p["dir"], p["result"]["C"], p["artifact"]) for rid, p in got.items()} == {"r0": (dirs["r0"], 0, None), "r1": (dirs["r1"], 128, None)}
    assert MF.programs_for({"requests": []}, str(tmp_path)) == {}
    with pytest.raises(SystemExit, match="no request Program for r2 .*169,9"):
        MF.programs_for({"requests": [*fx["requests"], {"request_id": "r2", "prompt_len": 169, "max_tokens": 10}]}, str(tmp_path))


def test_a_manifests_rows_are_read_under_the_name_its_layout_gives_them():
    a, b = {"op_path": "a"}, {"op_path": "b"}
    assert MF.manifest_rows({"identities": [a], "entries": [b]}) == [a], "the identity key first"
    assert MF.manifest_rows({"identities": [], "entries": [b]}) == [b] and MF.manifest_rows({"rows": [a, b]}) == [a, b]
    assert MF.manifest_rows({"identities": None}) == [] and MF.manifest_rows({}) == []
    assert MF.identity_rows({"identities": [a]}) == [a] and MF.identity_rows({"entries": [b]}) == [] and MF.identity_rows(None) == []
    assert MF.identity_rows({"identities": 0}) == [] and MF.identity_rows({"identities": ""}) == [], "an empty value of any type holds none"


def test_a_workloads_sweep_and_requests_are_empty_where_it_holds_none():
    assert MF.sweep_and_requests(None) == ({}, []) and MF.sweep_and_requests({"sweep": None}) == ({}, [])
    assert MF.sweep_and_requests({"sweep": 0, "requests": ""}) == ({}, []), "an empty value of any type holds none"
    w = {"sweep": {"batch": 2, "arrive_steps": [0, 3]}, "requests": [{"request_id": "r0"}, {"request_id": "r1"}]}
    assert MF.sweep_and_requests(w) == (w["sweep"], w["requests"])


def test_importing_the_format_loads_only_the_standard_library_and_records():
    """What importing `format` loads is part of its contract: a suite that lists its inputs by file reads this module, and its
    read guard fails on a file the list lacks.  So `load_program` and `program_dirs_for` import `program` inside themselves, and
    the module's own imports are the standard library and `verity_vllm.records`."""
    loaded: set[str] = set()
    for node in ast.parse(Path(MF.__file__).read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Import):
            loaded |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            loaded.add("." * node.level + (node.module or ""))
    assert {m for m in loaded if m.split(".")[0] not in sys.stdlib_module_names} == {"verity_vllm.records"}
