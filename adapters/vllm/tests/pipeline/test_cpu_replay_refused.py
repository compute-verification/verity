"""What `pipeline.cpu_replay` refuses by name of a source that is not the record's: a keep or a bundle of another schema, a keep
whose `slim.json` names another pair, run root or binding map than its files, a bundle's tensor of another size than registered,
a required manifest that is not the Commit's; and what its readers that decide nothing take: a record already decided
(`of_record` with ``awaiting=False``), a bundle that names no engine of record.  Last, the order `decide` leaves a run's
components in.  A block left out of a record is `test_cpu_replay_absent.py`; the whole command is `test_cpu_replay_command.py`."""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import verity_vllm.pipeline.cpu_replay as CPU
import verity_vllm.pipeline.replay_bundle as RB
from tests.pipeline import test_cpu_replay as OF_RECORD
from verity_vllm.check import cublas_route_pin, verdict

Record = dict[str, Any]
ABSENT = object()
ROPE = "model.rotary_emb.cos_sin_cache"
COS_SIN = OF_RECORD.COS_SIN.tobytes()
SHA = "cd" * 32
AWAITING = "pair 1: no instrumented run awaiting its deferred replay in "


def with_(doc: Record, field: str, value: object) -> Record:
    """`doc` with `field` left out (`ABSENT`) or holding `value`."""
    rest = {k: v for k, v in doc.items() if k != field}
    return rest if value is ABSENT else {**rest, field: value}


def rewritten(path: Path, change: Callable[[Record], Record]) -> None:
    """The one JSON document at `path` (a line of runs.jsonl, a keep's slim.json, a bundle's manifest) replaced by `change` of it."""
    path.write_text(json.dumps(change(json.loads(path.read_text()))) + "\n")


# ---------------------------------------------------------------------------------------------- a keep or a bundle that is not one

@pytest.mark.parametrize("schema", [ABSENT, None, "verity-vllm/another-record/v1"])
def test_a_keep_and_a_bundle_of_another_schema_are_refused_by_name(tmp_path: Path, schema: object) -> None:
    out, _awaiting = OF_RECORD._decided(tmp_path)
    named = None if schema is ABSENT else schema
    for source, doc, what, read in ((CPU.slim_path(out, 1), CPU.SLIM_DOC, "keep", CPU.SLIM_SCHEMA),
                                    (RB.path(out, 1), RB.MANIFEST, "manifest", RB.SCHEMA)):
        assert CPU.open_source(source)["path"] == str(source)
        rewritten(source / doc, lambda d: with_(d, "schema", schema))
        with pytest.raises(ValueError, match=re.escape(f"{source}: {what} schema {named!r} is not {read}")):
            CPU.open_source(source)


@pytest.mark.parametrize(("field", "value"), [("pair", 2), ("run_root", "cd" * 64), ("binding_map_sha256", "00" * 32)])
def test_a_keep_whose_slim_json_names_another_pair_run_root_or_binding_map_than_its_files_is_refused(tmp_path: Path, field: str,
                                                                                                    value: object) -> None:
    """A keep's files are checked against `slim.json`'s list, and then `slim.json`'s own pair, run root and binding map against the
    context and the binding map among those files: a document that lists the files as they are and names another record is refused."""
    out, _awaiting = OF_RECORD._decided(tmp_path)
    keep = CPU.slim_path(out, 1)
    assert json.loads((keep / CPU.SLIM_DOC).read_text())[field] != value
    rewritten(keep / CPU.SLIM_DOC, lambda doc: doc | {field: value})
    with pytest.raises(ValueError, match=re.escape(f"{keep}: its context and binding map are not slim.json's (pair, run root, binding map sha256)")):
        CPU.open_keep(keep)


@pytest.mark.parametrize("held", [COS_SIN[:-1], COS_SIN + b"\x00", b""])
def test_a_bundle_s_tensor_of_another_size_than_registered_is_read_and_does_not_open(tmp_path: Path, held: bytes) -> None:
    """The size is held before the root is folded: a bundle's file shorter or longer than the tree registers says both sizes, and
    one of the registered size with other bytes says its root."""
    (tmp_path / "weights" / "extra").mkdir(parents=True)
    tree = OF_RECORD.weights_tree({ROPE: ("float32", OF_RECORD.COS_SIN)}) | {"extra": {ROPE: 0}}
    (tmp_path / "weights" / "extra" / "0.bin").write_bytes(held)
    assert CPU._open_one(tmp_path, tree, 0, None) == ("bundle", None, f"{len(held)} B, registered {len(COS_SIN)} B", True)
    (tmp_path / "weights" / "extra" / "0.bin").write_bytes(bytes(len(COS_SIN)))
    assert CPU._open_one(tmp_path, tree, 0, None) == ("bundle", None, "tensor root != the committed tensor root", True)


# ---------------------------------------------------------------------------------------------- a reader that decides nothing

def test_a_reader_that_decides_nothing_takes_a_decided_record_as_it_stands(tmp_path: Path) -> None:
    """`of_record` with ``awaiting=False`` reads a bundle's run for a reader that decides nothing: the record may be decided, and
    its row is given as it stands, from the bundle or from its keep (which a deciding replay puts back to the awaiting state)."""
    out, awaiting = OF_RECORD._decided(tmp_path)
    bundle, keep = RB.open_bundle(RB.path(out, 1)), CPU.open_keep(CPU.slim_path(out, 1))
    decided = json.loads((out / "runs.jsonl").read_text())
    for source in (bundle, keep):
        rows, row, _verdict, _store, run = CPU.of_record(source, out, awaiting=False)
        assert rows == [decided] and row == decided and row["commit"]["commit_pass"] is True
        assert run.run_root.hex() == decided["commit"]["run_root"]
    with pytest.raises(CPU.Refused, match=AWAITING):
        CPU.of_record(bundle, out)
    assert CPU.of_record(keep, out)[1] == awaiting


@pytest.mark.parametrize("change", [lambda row: row | {"arm": "control"}, lambda row: row | {"pair": 2}], ids=["control", "another-pair"])
def test_a_reader_that_decides_nothing_is_refused_a_record_with_no_instrumented_run_of_the_pair(tmp_path: Path,
                                                                                               change: Callable[[Record], Record]) -> None:
    out = OF_RECORD._deferred(tmp_path)
    b = RB.open_bundle(RB.path(out, 1))
    assert CPU.of_record(b, out, awaiting=False)[1]["pair"] == 1
    rewritten(out / "runs.jsonl", change)
    with pytest.raises(CPU.Refused, match=re.escape(f"pair 1: no instrumented run in {out / 'runs.jsonl'}")):
        CPU.of_record(b, out, awaiting=False)
    with pytest.raises(CPU.Refused, match=AWAITING):
        CPU.of_record(b, out)


@pytest.mark.parametrize("engine", [ABSENT, None])
def test_a_bundle_that_names_no_engine_of_record_leaves_the_replay_s_own_facts_where_they_are(tmp_path: Path, engine: object) -> None:
    cpu = {"VLLM_BATCH_INVARIANT": None, "compute_capability": "unavailable: no CUDA device"}
    row: Record = {"commit": {"sampled_replay": {"engine_of_record": dict(cpu)}}}
    art = tmp_path / "sampled_replay_p1.json"
    CPU.engine_of_record(row, with_({}, "engine_of_record", engine), art)
    assert row == {"commit": {"sampled_replay": {"engine_of_record": cpu}}} and not art.exists()


# ---------------------------------------------------------------------------------------------- the Commit's required manifest

def test_the_required_manifest_is_the_one_the_command_names_else_the_commit_s_own_else_the_row_dir_s_and_is_held_to_its_sha256(
        tmp_path: Path) -> None:
    """The replay reads the Commit's required manifest where `--required-manifest` names it, else at the path the Commit recorded
    when a file is there, else as `<row-dir>/manifest.json`; the file it reads is refused unless its sha256 is the Commit's, and a
    file that is not is never passed over for the next place."""
    manifest: Record = {"entries": []}
    text = json.dumps(manifest)
    sha = hashlib.sha256(text.encode()).hexdigest()
    row_dir, named, recorded = tmp_path / "row", tmp_path / "named.json", tmp_path / "commit" / "manifest.json"
    for p in (row_dir / "manifest.json", named, recorded):
        p.parent.mkdir(exist_ok=True)
        p.write_text(text)
    args = SimpleNamespace(programs_root=None, program_dir=None)

    def of(command: Path | None, commit: Path | None) -> tuple[str | None, Any]:
        o = CPU.Options(bundle="B", row_dir=str(row_dir), checkpoint_dir="CK", required_manifest=None if command is None else str(command))
        ctx = {"required_manifest_sha256": sha, "args": {"required_manifest": None if commit is None else str(commit)}}
        return CPU.manifest_of(o, ctx, args)

    assert of(named, recorded) == (str(named), manifest)
    assert of(None, recorded) == (str(recorded), manifest)
    assert of(None, tmp_path / "gone" / "manifest.json") == of(None, None) == (str(row_dir / "manifest.json"), manifest)
    for command, commit, read in ((named, recorded, named), (None, recorded, recorded), (None, None, row_dir / "manifest.json")):
        read.write_text(json.dumps({"entries": [1]}))
        with pytest.raises(CPU.Refused, match=re.escape(f"{read} is not the Commit's required manifest (sha256 {sha[:16]})")):
            of(command, commit)
        read.write_text(text)
    assert of(named, recorded) == (str(named), manifest)


@pytest.mark.parametrize("want", [ABSENT, None, ""])
def test_a_commit_that_recorded_no_required_manifest_gives_the_replay_none_and_reads_no_file(tmp_path: Path, want: object) -> None:
    """A Commit that recorded no required manifest's sha256 (left out, null or empty) has none to replay under: `manifest_of` gives
    no path and no manifest, and reads no file, not one the command names nor one the Commit recorded."""
    o = CPU.Options(bundle="B", row_dir=str(tmp_path / "row"), checkpoint_dir="CK", required_manifest=str(tmp_path / "named.json"))
    ctx = with_({"args": {"required_manifest": str(tmp_path / "recorded.json")}}, "required_manifest_sha256", want)
    assert CPU.manifest_of(o, ctx, SimpleNamespace(programs_root=None, program_dir=None)) == (None, None)


# ---------------------------------------------------------------------------------------------- the decided run's components

def test_a_decided_run_s_components_end_on_the_weights_pin_and_the_replay_and_its_why_names_each_failure_in_that_order(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """`decide` puts the run's weights pin and its `replay` component last, in that order, as the in-process Commit writes them;
    the correspondence's why names each failing component in the order they then stand, and none when none fails.  The route pin
    is stamped before, and the verdict read after, over the run's own tokens and the bundle's declaration."""
    seen: list[tuple[Any, ...]] = []

    def commit_verdict(c: Record, tokens_equal: bool | None, **kw: Any) -> tuple[bool, str]:
        vc = c["value_correspondence"]
        seen.append(("verdict", list(vc["components"]), vc["why"], c["weights_attest"]["root_openings"], tokens_equal, kw))
        return False, "C2 FAIL"

    monkeypatch.setattr(cublas_route_pin, "stamp", lambda c: seen.append(("stamp", list(c["value_correspondence"]["components"]))))
    monkeypatch.setattr(verdict, "commit_verdict", commit_verdict)
    components: Record = {"weights_pin": {"result": False, "why": "pin differs"}, "replay": {"result": True},
                          "sampled_replay": {"result": True}, "boundary_linkage": {"result": False, "why": "2 unlinked"},
                          "negatives": {"suite": "s"}}
    as_given, as_decided = list(components), ["sampled_replay", "boundary_linkage", "negatives", "weights_pin", "replay"]
    row: Record = {"tokens_equal_ref": True, "commit": {"value_correspondence": {"method": "m", "result": False, "components": components}}}
    ctx = {"min_openings": 64, "declaration": {"classes": ["c"]}, "required_classes": ["c"]}
    why = "boundary_linkage: 2 unlinked; weights_pin: pin differs"

    CPU.decide(row, ctx, {"equal": True}, {"manifest_sha256": SHA}, set(), {}, {})
    c = row["commit"]
    assert list(c["value_correspondence"]["components"]) == as_decided and c["value_correspondence"]["why"] == why
    assert (c["commit_pass"], c["commit_fail_reason"]) == (False, "C2 FAIL")
    assert seen == [("stamp", as_given), ("verdict", as_decided, why, "0/0", True, ctx)]

    for x in components.values():
        x.pop("why", None)
        x["result"] = True
    CPU.decide(row, ctx, {"equal": True}, {"manifest_sha256": SHA}, set(), {}, {})
    assert list(c["value_correspondence"]["components"]) == as_decided and c["value_correspondence"]["why"] is None
