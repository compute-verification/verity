"""The restore's reading of a served restore arm (edges, states, the arrivals it replays, the served outputs) and its
comparisons, on a synthetic arm."""
from __future__ import annotations

import gzip
import json
import pickle
from pathlib import Path

import boundary as B
import pytest
import restore as R


class Request:
    def __init__(self, request_id: str) -> None:
        self.request_id = request_id


def rid(tag: str, i: int) -> str:
    return f"cmpl-{tag}-{i:05d}-0-0123abcd"


def arm(tmp_path: Path, gz: bool = False) -> tuple[Path, Path]:
    """A restore arm whose edge 0 holds ``a`` in flight at step 2 and whose edge 1 is at step 4; ``b`` arrived before
    step 1 (held at edge 0) and ``c`` before step 3."""
    d, kv = tmp_path / "restore", tmp_path / "kv"
    (d / "server" / "edges").mkdir(parents=True)
    (d / "medium" / "rep-1").mkdir(parents=True)
    kv.mkdir()
    plan = tmp_path / "plan.jsonl"
    plan.write_text("".join(json.dumps({"i": i, "prompt_ids": [i, i + 1]}) + "\n" for i in range(3)))
    (d / "meta.json").write_text(json.dumps({"plan": str(plan)}))
    (d / "medium" / "rep-1" / "client.jsonl").write_text("".join(
        json.dumps({"id": f"medium1-{i:05d}", "i": i}) + "\n" for i in range(3)))
    a, b, c = rid("medium1", 0), rid("medium1", 1), rid("medium1", 2)
    s = {"max_tokens": 4, "seed": 7}
    recs = [{"kind": "header", "config": {"block_size": 16}},
            {"kind": "arrive", "request": a, "before_step": 0, "sampling": s},
            {"kind": "step", "step": 0, "rows": [[a, 0, 2]]},
            {"kind": "arrive", "request": b, "before_step": 1, "sampling": s},
            {"kind": "step", "step": 1, "rows": [[a, 2, 1], [b, 0, 2]]},
            {"kind": "step", "step": 2, "rows": [[a, 3, 1], [b, 2, 1]]},
            {"kind": "arrive", "request": c, "before_step": 3, "sampling": s},
            {"kind": "step", "step": 3, "rows": [[c, 0, 2]]},
            {"kind": "step", "step": 4, "rows": [[c, 2, 1]]}]
    trace = d / "server" / "step_trace.9.jsonl"
    trace.write_text("".join(json.dumps(r) + "\n" for r in recs))
    edges = [{"edge": 0, "next_step": 2}, {"edge": 0, "state_write_s": 0.1}, {"edge": 1, "next_step": 4}]
    (d / "server" / "edges.9.jsonl").write_text("".join(json.dumps(r) + "\n" for r in edges))
    for k, held in ((0, [a, b]), (1, [c])):
        st = {"requests": B.pickle_requests([Request(x) for x in held], {"block_hasher": None}), "next_step": 2 + 2 * k}
        (kv / f"edge-{k}.state.pkl").write_bytes(pickle.dumps(st))
        (d / "server" / "edges" / f"edge-{k}.dump.json").write_text(json.dumps({"sched.x": k}))
    outs = d / "server" / "edges" / "outputs.9.jsonl"
    outs.write_text("".join(json.dumps({"step": t, "out": {a: [[t], None]}}) + "\n" for t in range(1, 5)))
    if gz:
        for f in (trace, outs):
            with gzip.open(f.with_suffix(".jsonl.gz"), "wt") as z:
                z.write(f.read_text())
            f.unlink()
    return d, kv


@pytest.mark.parametrize("gz", [False, True])
def test_served_cuts_the_replay_block_and_its_arrivals(tmp_path: Path, gz: bool):
    d, kv = arm(tmp_path, gz)
    w = R.served(d, kv, 0, 1)
    assert (w["lo"], w["hi"], sorted(w["steps"]), sorted(w["outputs"])) == (2, 4, [2, 3], [2, 3])
    assert [(r["engine_id"], r["before_step"], r["prompt_ids"], r["max_tokens"]) for r in w["arrive"]] == [
        (rid("medium1", 2), 3, [2, 3], 4)]
    assert w["dumps"][1] == {"sched.x": 1} and w["header"]["kind"] == "header"


def test_served_refuses_an_edge_not_taken_and_a_missing_plan(tmp_path: Path):
    d, kv = arm(tmp_path)
    with pytest.raises(SystemExit, match="not 0 and 2"):
        R.served(d, kv, 0, 2)
    (d / "meta.json").write_text(json.dumps({"plan": str(tmp_path / "elsewhere.jsonl")}))
    with pytest.raises(SystemExit, match="no plan at"):
        R.served(d, kv, 0, 1)


def test_step_diff_names_the_first_differing_step_and_fields():
    want = {2: {"rows": [["a", 3, 1]], "new": [], "t": 1.0}, 3: {"rows": [["c", 0, 2]], "new": ["c"], "t": 2.0}}
    got = {2: {"rows": [["a", 3, 1]], "new": [], "t": 9.0}, 3: {"rows": [["c", 0, 1]], "new": ["c"], "t": 9.5}}
    assert R.step_diff(want, {k: dict(v) for k, v in want.items()})["steps_differing"] == 0
    d = R.step_diff(want, got)
    assert (d["steps_differing"], d["rows_served"], d["first"]["step"], d["first"]["fields"]) == (1, 2, 3, ["rows"])
    assert R.step_diff(want, {2: got[2]})["first"]["step"] == 3


def test_fingerprint_diff_counts_blocks_and_layers():
    np = pytest.importorskip("numpy")
    a = np.arange(12, dtype=np.int64).reshape(4, 3)
    b = a.copy()
    assert R.fingerprint_diff(a, b)["blocks_differing"] == 0
    b[2, 1] += 1
    d = R.fingerprint_diff(a, b)
    assert (d["blocks_differing"], d["first"], d["layers_differing_first_block"]) == (1, [2], [1])
    assert R.fingerprint_diff(a, b[:3]) == {"compared": False} and R.fingerprint_diff(None, b) == {"compared": False}
