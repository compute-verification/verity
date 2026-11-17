"""Which replay records a row's verdict cites (`verity_vllm.check.verdict.from_record`).

The verdict cites by sha256 the replay records under the row's `commit/` that are its runs' own: for each pair `commit/runs.jsonl`
holds a run of that is not the control arm, the pair's `sampled_replay_p<pair>.json` and its ranks'
`sampled_replay_p<pair>_rank<r>.json`.  A Commit removes nothing under its `--out`, so a later Commit of fewer pairs, or of the
control arm alone, leaves an earlier one's records beside its own rows.  Those are not cited, under `verdict.inputs` or under a
run's replay check, and a verdict is the same with them on the row as without them.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tests.check.test_verdict import _base, _row_dir
from verity_vllm.check import verdict as V
from verity_vllm.pipeline import cli

REPLAY_CHECK = "local_replay/sampled_replay"


def row_of(tmp_path: Path, pairs: list[int], *, arms: tuple[str, ...] = ("control", "instrumented")) -> Path:
    """A row whose Commit passed: the two runs of `_row_dir`, the control and the instrumented one, restated for each of ``pairs``,
    with the runs of ``arms`` kept."""
    d, _ = _row_dir(tmp_path, _base(), commit_pass=True, rc=0)
    runs = d / "commit" / "runs.jsonl"
    rows = [json.loads(line) for line in runs.read_text().splitlines()]
    runs.write_text("".join(json.dumps(dict(r, pair=p)) + "\n" for p in pairs for r in rows if r["arm"] in arms))
    return d


def replay_records(d: Path, names: list[str]) -> dict[str, str]:
    """A replay record under each of ``names`` in the row's `commit/`, no two of the same bytes: its path from the row, and its
    sha256."""
    written = {}
    for seed, name in enumerate(names):
        doc = json.dumps({"sampled_replay": dict(_base()["sampled_replay"], seed=seed)}).encode()
        (d / "commit" / name).write_bytes(doc)
        written[f"commit/{name}"] = hashlib.sha256(doc).hexdigest()
    return written


def cited(inputs: dict) -> dict[str, str]:
    return {k: v for k, v in inputs.items() if k.startswith("commit/sampled_replay")}


@pytest.mark.parametrize("pairs, arms, on_the_row, own", [
    ([0], ("control", "instrumented"), ["sampled_replay_p0.json"], ["sampled_replay_p0.json"]),
    ([0, 1], ("control", "instrumented"), ["sampled_replay_p0.json", "sampled_replay_p1.json"], ["sampled_replay_p0.json", "sampled_replay_p1.json"]),
    ([0], ("control", "instrumented"), ["sampled_replay_p0.json", "sampled_replay_p1.json"], ["sampled_replay_p0.json"]),
    ([1], ("control", "instrumented"), ["sampled_replay_p1.json", "sampled_replay_p10.json", "sampled_replay_p11.json", "sampled_replay_p0.json"],
     ["sampled_replay_p1.json"]),
    ([0], ("control", "instrumented"), ["sampled_replay_p0_rank0.json", "sampled_replay_p0_rank1.json", "sampled_replay_p1_rank0.json"],
     ["sampled_replay_p0_rank0.json", "sampled_replay_p0_rank1.json"]),
    ([1], ("control", "instrumented"), ["sampled_replay_p1_rank0.json", "sampled_replay_p10_rank0.json", "sampled_replay_p1.json"],
     ["sampled_replay_p1.json", "sampled_replay_p1_rank0.json"]),
    ([2, 10], ("control", "instrumented"), ["sampled_replay_p2.json", "sampled_replay_p10.json", "sampled_replay_p10_rank0.json"],
     ["sampled_replay_p2.json", "sampled_replay_p10.json", "sampled_replay_p10_rank0.json"]),
    ([0], ("instrumented",), ["sampled_replay_p0.json", "sampled_replay_p1.json"], ["sampled_replay_p0.json"]),
    ([0], ("control",), ["sampled_replay_p0.json"], []),
    ([0, 1], ("control", "instrumented"), [], []),
], ids=["its-pairs-record", "two-pairs-and-their-records", "fewer-pairs-than-an-earlier-commit", "a-pair-whose-number-begins-another's",
        "its-pairs-rank-records", "a-pairs-record-and-its-rank-record", "in-the-order-of-their-names", "the-instrumented-arm-alone",
        "the-control-arm-alone", "no-replay-record"])
def test_a_verdict_cites_the_replay_records_of_its_own_runs_and_no_other(tmp_path, pairs, arms, on_the_row, own):
    d = row_of(tmp_path, pairs, arms=arms)
    written = replay_records(d, on_the_row)
    v = V.from_record(d, rc=0)
    want = {f"commit/{name}": written[f"commit/{name}"] for name in sorted(own)}
    assert cited(v.inputs) == want and list(cited(v.inputs)) == list(want)
    assert v.inputs["commit/runs.jsonl"] and v.runs["instrumented"] == ("instrumented" in arms) * len(pairs)
    assert [cited(c.inputs) for c in v.checks if c.name == REPLAY_CHECK] == [want] * ("instrumented" in arms)


@pytest.mark.parametrize("arms", [("control", "instrumented"), ("control",)], ids=["both-arms", "the-control-arm-alone"])
def test_a_verdict_is_the_same_with_another_commits_replay_records_on_the_row_as_without_them(tmp_path, arms):
    d = row_of(tmp_path, [0], arms=arms)
    replay_records(d, ["sampled_replay_p0.json"])
    alone = V.from_record(d, rc=0).to_dict()
    replay_records(d, ["sampled_replay_p0.json", "sampled_replay_p1.json", "sampled_replay_p1_rank0.json", "sampled_replay_p00.json"])
    assert V.from_record(d, rc=0).to_dict() == alone
    assert alone["outcome"] == (V.PASS if "instrumented" in arms else V.NOT_RUN)


def test_a_row_with_no_runs_cites_no_replay_record(tmp_path):
    d = row_of(tmp_path, [0])
    replay_records(d, ["sampled_replay_p0.json"])
    (d / "commit" / "runs.jsonl").unlink()
    v = V.from_record(d, rc=0)
    assert cited(v.inputs) == {} and "commit/runs.jsonl" not in v.inputs and v.inputs["commit/verdict.json"]
    assert v.outcome == V.NOT_RUN and any("commit/runs.jsonl not on the row" in note for note in v.notes)


def test_the_verdict_the_row_keeps_cites_the_same_records(tmp_path, capsys):
    """`verity-vllm verdict from-record`, as the row's Commit stage runs it: the `verdict.json` it writes cites the pair's record and
    not the earlier Commit's."""
    d = row_of(tmp_path, [0])
    written = replay_records(d, ["sampled_replay_p0.json", "sampled_replay_p1.json"])
    assert cli.main(["verdict", "from-record", str(d), "--rc", "0"]) == 0 and capsys.readouterr().out.startswith(f"PASS {d / 'verdict.json'} [")
    kept = json.loads((d / "verdict.json").read_text())
    assert cited(kept["inputs"]) == {"commit/sampled_replay_p0.json": written["commit/sampled_replay_p0.json"]}
    assert kept["outcome"] == V.PASS and V.CommitChecks.load(d / "verdict.json").outcome == V.PASS
