"""What `pipeline.cpu_replay` reads of a record, a verdict, a weights tree and a bundle's context that leave a block out: a block
may be absent, null or empty, and each reader gives the one answer for all three.  The whole replay over a run with no weight
registration and a tree with no tensor of the bundle's is `test_cpu_replay_command.py`."""
from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import verity_vllm.pipeline.cpu_replay as CPU
import verity_vllm.pipeline.replay_bundle as RB
from tests.pipeline import test_cpu_replay as OF_RECORD

Record = dict[str, Any]
Lines = Callable[[], list[tuple[str, str]]]
ABSENT = object()
ROPE = "model.rotary_emb.cos_sin_cache"
DECIDED = "sampled-exact-replay(complete) + boundary-linkage + offline-suite"
SHA = "cd" * 32


def with_(doc: Record, field: str, value: object) -> Record:
    """`doc` with `field` left out (`ABSENT`) or holding `value`."""
    rest = {k: v for k, v in doc.items() if k != field}
    return rest if value is ABSENT else {**rest, field: value}


def rewritten(path: Path, change: Callable[[Record], Record]) -> None:
    """The one JSON document at `path` (a line of runs.jsonl, or verdict.json) replaced by `change` of it."""
    path.write_text(json.dumps(change(json.loads(path.read_text()))) + "\n")


# ---------------------------------------------------------------------------------------------- a decided record, put back

def test_a_decided_record_goes_back_to_its_deferred_state_less_what_the_replay_wrote() -> None:
    components = {"weights_pin": {"result": True}, "sampled_replay": {"result": True}, "boundary_linkage": {"result": True}, "negatives": {"suite": "s"}}
    c: Record = {"commit_pass": True, "commit_fail_reason": "PASS", "weights_attest": {"root_openings": "2/2"},
                 "opened_values": {"method": "opened", "replay_linkage": {"n": 2}},
                 "value_correspondence": {"method": DECIDED, "result": True, "components": components}}
    CPU.undecided(c, "/row/commit/replay_bundle_p1")
    sr, lk = RB.deferred_records(Path("/row/commit/replay_bundle_p1"))
    assert c == {"commit_pass": None, "commit_fail_reason": RB.PENDING, "sampled_replay": sr, "boundary_linkage": lk,
                 "value_correspondence": {"method": "offline-suite", "result": None, "components": {"weights_pin": {"result": True}}}}


@pytest.mark.parametrize("components", [ABSENT, None, {}])
def test_a_decided_record_whose_correspondence_holds_no_component_has_none_to_take_off(components: object) -> None:
    c: Record = {"commit_pass": True, "value_correspondence": with_({"method": DECIDED, "result": True}, "components", components)}
    CPU.undecided(c, "/row/commit/replay_bundle_p1")
    assert c["value_correspondence"] == with_({"method": "offline-suite", "result": None}, "components", components)
    assert (c["commit_pass"], c["commit_fail_reason"], c["sampled_replay"]["grade"]) == (None, RB.PENDING, "deferred")


# ---------------------------------------------------------------------------------------------- the weights tree and its attest

@pytest.mark.parametrize("extra", [ABSENT, None, {}, {"another.tensor": 3}])
def test_a_tensor_the_tree_does_not_list_as_the_bundle_s_is_read_from_the_checkpoint(tmp_path: Path, extra: object) -> None:
    tree = with_(OF_RECORD.weights_tree({ROPE: ("float32", OF_RECORD.COS_SIN)}), "extra", extra)
    OF_RECORD.write_salts(tmp_path, tree)
    assert CPU._open_one(tmp_path, tree, 0, None) == ("checkpoint", None, "no checkpoint given", False)


def test_a_tensor_the_tree_lists_as_the_bundle_s_is_read_from_the_bundle(tmp_path: Path) -> None:
    (tmp_path / "weights" / "extra").mkdir(parents=True)
    (tmp_path / "weights" / "extra" / "0.bin").write_bytes(OF_RECORD.COS_SIN.tobytes())
    tree = OF_RECORD.weights_tree({ROPE: ("float32", OF_RECORD.COS_SIN)}) | {"extra": {ROPE: 0}}
    OF_RECORD.write_salts(tmp_path, tree)
    source, raw, why, read = CPU._open_one(tmp_path, tree, 0, None)
    assert (source, why, read) == ("bundle", None, True) and raw is not None and raw.tobytes() == OF_RECORD.COS_SIN.tobytes()


# ---------------------------------------------------------------------------------------------- the bundle is the record's

@pytest.mark.parametrize("deferred", [ABSENT, None, {}, {"bundles": None}, {"bundles": []}, {"pending_pairs": [1]}])
def test_a_verdict_that_names_no_deferred_bundle_refuses_every_bundle(tmp_path: Path, deferred: object) -> None:
    out = OF_RECORD._deferred(tmp_path)
    b = RB.open_bundle(RB.path(out, 1))
    rewritten(out / "verdict.json", lambda v: with_(v, "replay_deferred", deferred))
    with pytest.raises(CPU.Refused, match=f"the Commit's verdict names no bundle of pair 1 with manifest sha256 {b['manifest_sha256'][:16]}"):
        CPU.of_record(b, out)


@pytest.mark.parametrize("commit", [ABSENT, None, {}])
def test_a_run_that_holds_no_commit_record_awaits_no_replay_of_its_bundle(tmp_path: Path, commit: object) -> None:
    out = OF_RECORD._deferred(tmp_path)
    rewritten(out / "runs.jsonl", lambda row: with_(row, "commit", commit))
    with pytest.raises(CPU.Refused, match="pair 1: no instrumented run awaiting its deferred replay in "):
        CPU.of_record(RB.open_bundle(RB.path(out, 1)), out)


# ---------------------------------------------------------------------------------------------- the slim keep


@pytest.mark.parametrize("field", ["key", "tensor_keys"])
def test_a_tree_registered_under_a_weights_key_is_refused_by_name(field: str) -> None:
    """Only a registration with a salt file is read: a tree recorded under a weights key (its ``key``, or a keep's ``tensor_keys``)
    is refused before anything opens."""
    tree = OF_RECORD.weights_tree({ROPE: ("float32", OF_RECORD.COS_SIN)})
    assert CPU.tree_refusal(tree | {field: "00" * 64}, tree["root"])[1] == (
        "the bundle's weights tree was registered under a weights key: only a registration with a salt file is read")
    assert CPU.tree_refusal(tree, tree["root"])[1] is None
