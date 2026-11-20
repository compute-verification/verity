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


@pytest.mark.parametrize("held", [COS_SIN[:-1], COS_SIN + b"\x00", b""])
def test_a_bundle_s_tensor_of_another_size_than_registered_is_read_and_does_not_open(tmp_path: Path, held: bytes) -> None:
    """The size is held before the root is folded: a bundle's file shorter or longer than the tree registers says both sizes, and
    one of the registered size with other bytes says its root."""
    (tmp_path / "weights" / "extra").mkdir(parents=True)
    tree = OF_RECORD.weights_tree({ROPE: ("float32", OF_RECORD.COS_SIN)}) | {"extra": {ROPE: 0}}
    OF_RECORD.write_salts(tmp_path, tree)
    (tmp_path / "weights" / "extra" / "0.bin").write_bytes(held)
    assert CPU._open_one(tmp_path, tree, 0, None) == ("bundle", None, f"{len(held)} B, registered {len(COS_SIN)} B", True)
    (tmp_path / "weights" / "extra" / "0.bin").write_bytes(bytes(len(COS_SIN)))
    assert CPU._open_one(tmp_path, tree, 0, None) == ("bundle", None, "tensor root != the committed tensor root", True)


# ---------------------------------------------------------------------------------------------- a reader that decides nothing


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

