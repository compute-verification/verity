import json
import sys
from types import SimpleNamespace

import pytest

from te_core import VERSION, frontier, launch
from te_core.ledger import Ledger
from te_core.run import file_sha256
from te_core.score import K_BITS, R

PIN = "a" * 40


class Clock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


def sub(led, **kw):
    keys = [f"{k}/r{r}" for k in K_BITS for r in R]
    body = {"macs": 100, "val_bpb": 1.0, "tier": 0, "start": "small", "start_bpb": 1.1, "mechanism": "dense",
            "reference": False, "program_sha256": "", "invalid": [], "run_te_core": VERSION,
            "measurement_commit": PIN} | kw
    for name, default in (("rho", 1.0), ("beta", 1.6), ("phi", 0.5)):
        body[name] = dict.fromkeys(keys, kw.get(name, default))
    return led.append("submission", body)


def test_records_are_append_only_and_per_agent(tmp_path):
    a, b = Ledger(tmp_path, "agent-a"), Ledger(tmp_path, "agent-b")
    r1, r2, r3 = sub(a), sub(a), sub(b)
    assert [r1["id"], r2["id"], r3["id"]] == ["agent-a-000000", "agent-a-000001", "agent-b-000000"]
    with pytest.raises(PermissionError):
        (tmp_path / "records" / "agent-a" / "agent-a-000000.json").write_text("{}")
    with pytest.raises(PermissionError):
        a.promote(r1["id"], 1, "self-promotion")
    with pytest.raises(ValueError):
        Ledger(tmp_path, "../x")
    (tmp_path / "records" / "agent-b" / "forged.json").write_text(json.dumps({"agent": "scorer", "kind": "promotion"}))
    assert {r["id"] for r in Ledger(tmp_path).records()} == {r1["id"], r2["id"], r3["id"]}


def test_index_joins_promotions_and_final_scores(tmp_path):
    a, s = Ledger(tmp_path, "agent-a"), Ledger(tmp_path, "scorer")
    r = sub(a)
    s.promote(r["id"], 1, "frontier")
    s.final(r["id"], 0.97)
    (row,) = Ledger(tmp_path).index()
    assert row["promoted"] == 1 and row["test_bpb"] == 0.97
    assert (tmp_path / "index.jsonl").read_text().count("\n") == 1


def test_launch_gates(tmp_path):
    clock = Clock()
    prog = tmp_path / "train.py"
    prog.write_text("def train(te):\n    pass\n")
    nodes = tmp_path / "nodes.txt"
    nodes.write_text("# the coordinator's list\nvy-dc-h200-1\nvy-nebius-2\n")
    a, s = Ledger(tmp_path, "agent-a", clock=clock), Ledger(tmp_path, "scorer", clock=clock)
    (rec,) = launch.launch(a, prog, 0, node_file=nodes, direction="d", mechanism="m", dry_run=True, clock=clock)
    assert rec["node"] == "vy-nebius-2"
    assert rec["argv"][:6] == ["research", "run", "--on", "vy-nebius-2", "--gpus", "1"]
    assert "--budget-minutes 9" in rec["argv"][-1] and "gpu-lease" not in rec["argv"][-1]
    pooled = tmp_path / "pooled.txt"
    pooled.write_text("vy-nebius-2 pool=gpu\n")
    (rec,) = launch.launch(a, prog, 0, node_file=pooled, direction="d", mechanism="m", dry_run=True, clock=clock,
                           script=True, train_args=["--lr", "1e-3"])
    assert rec["argv"][3:8] == ["vy-nebius-2", "--pool", "gpu", "--as", "training"] and "--gpus" not in rec["argv"]
    assert "gpu-lease 1 --wait" in rec["argv"][-1] and "--script" in rec["argv"][-1]
    assert "--start s100 --out $S/sub --budget-fraction 0.294 --lr 1e-3" in rec["argv"][-1]
    entry = sub(a, program_sha256=file_sha256(prog))["id"]
    with pytest.raises(launch.Refused, match="not promoted"):
        launch.launch(a, prog, 1, node_file=nodes, direction="d", mechanism="m", entry=entry, dry_run=True, clock=clock)
    s.promote(entry, 1, "frontier")
    for _ in range(launch.TIER1_PER_HOUR):
        launch.launch(a, prog, 1, node_file=nodes, direction="d", mechanism="m", entry=entry, dry_run=True, clock=clock)
    with pytest.raises(launch.Refused, match="last hour"):
        launch.launch(a, prog, 1, node_file=nodes, direction="d", mechanism="m", entry=entry, dry_run=True, clock=clock)
    clock.t += 3601
    launch.launch(a, prog, 1, node_file=nodes, direction="d", mechanism="m", entry=entry, dry_run=True, clock=clock)
    with pytest.raises(launch.Refused, match="tier 2"):
        launch.launch(a, prog, 2, node_file=nodes, direction="d", mechanism="m", entry=entry, dry_run=True, clock=clock)
    other = tmp_path / "other.py"
    other.write_text("def train(te):\n    return 1\n")
    with pytest.raises(launch.Refused, match="not .*'s program"):
        launch.launch(a, other, 1, node_file=nodes, direction="d", mechanism="m", entry=entry, dry_run=True, clock=clock)
    s.promote(entry, 2, "elite")
    recs = launch.launch(a, prog, 2, node_file=nodes, direction="d", mechanism="m", entry=entry, dry_run=True,
                         clock=clock)
    assert len(recs) == 6 and all("--keep-checkpoints" in r["argv"][-1] for r in recs)
    only_h200 = tmp_path / "h200.txt"
    only_h200.write_text("vy-dc-h200-2\n")
    with pytest.raises(launch.Refused, match="no usable node"):
        launch.launch(a, prog, 0, node_file=only_h200, direction="d", mechanism="m", dry_run=True, clock=clock)


def test_frontier_elites_targets_and_promotion_candidates(tmp_path):
    a = Ledger(tmp_path, "agent-a")
    ref = sub(a, macs=100, val_bpb=1.08, reference=True, mechanism="reference")
    cheap = sub(a, macs=50, val_bpb=1.085, rho=1.0, mechanism="lowrank", beta=0.01)
    good = sub(a, macs=80, val_bpb=1.05, rho=4.0, mechanism="dense")
    dominated = sub(a, macs=500, val_bpb=1.10, rho=4.0, mechanism="dense")
    bad = sub(a, macs=1, val_bpb=0.5, invalid=["no step"])
    other = sub(a, macs=1, val_bpb=0.5, measurement_commit="b" * 40)
    rows = Ledger(tmp_path).index()
    rep = frontier.report(rows, PIN)
    key = "10kB/r10"
    assert [f["id"] for f in rep[key]["frontier"]] == [cheap["id"], ref["id"], good["id"]]
    assert dominated["id"] not in {e["id"] for e in rep[key]["elites"]}
    assert rep[key]["coverage"]["filled"] == 3
    t = next(x for x in rep[key]["targets"] if x["drop"] == 0.01)
    assert t["l_star"] == pytest.approx(1.1 * 0.99)
    assert t["honest"] == 100 and t["w_star"] == 50
    assert t["effective_slowdown"] == pytest.approx(0.5)
    assert set(frontier.candidates(rows, 1, PIN)) == {ref["id"], cheap["id"], good["id"]}
    assert "| 10kB/r10 |" in frontier.markdown(rep)
    assert frontier.status(rows, PIN) == {bad["id"]: "invalid", other["id"]: "other-version"}


def test_a_record_repeating_an_earlier_e_is_left_out_of_the_frontier(tmp_path):
    clock = Clock()
    a, b = Ledger(tmp_path, "agent-a", clock=clock), Ledger(tmp_path, "agent-b", clock=clock)
    first = sub(a, macs=500, val_bpb=1.0, output={"e_sha256": "e1"})
    clock.t += 1
    copy = sub(b, macs=1, val_bpb=1.0, output={"e_sha256": "e1"})
    clock.t += 1
    own = sub(b, macs=2, val_bpb=1.0, output={"e_sha256": "e2"})
    rows = Ledger(tmp_path).index()
    assert frontier.status(rows, PIN) == {copy["id"]: "duplicate-e"}
    assert {r["id"] for r in frontier.valid(rows, PIN)} == {first["id"], own["id"]}


def test_an_agents_relaunch_at_a_new_pin_counts_and_a_copy_of_an_old_pins_e_does_not(tmp_path):
    clock = Clock()
    a, b = Ledger(tmp_path, "agent-a", clock=clock), Ledger(tmp_path, "agent-b", clock=clock)
    old = sub(a, val_bpb=1.0, output={"e_sha256": "e1"}, measurement_commit="b" * 40)
    clock.t += 1
    copy = sub(b, val_bpb=1.0, output={"e_sha256": "e1"})
    clock.t += 1
    relaunch = sub(a, val_bpb=1.0, output={"e_sha256": "e1"})
    clock.t += 1
    again = sub(a, val_bpb=1.0, output={"e_sha256": "e1"})
    rows = Ledger(tmp_path).index()
    assert frontier.status(rows, PIN) == {old["id"]: "other-version", copy["id"]: "duplicate-e",
                                          again["id"]: "duplicate-e"}
    assert [r["id"] for r in frontier.valid(rows, PIN)] == [relaunch["id"]]


def test_the_commits_of_one_version_are_one_pin(tmp_path, monkeypatch):
    clock = Clock()
    a, b = Ledger(tmp_path, "agent-a", clock=clock), Ledger(tmp_path, "agent-b", clock=clock)
    earlier = sub(a, val_bpb=1.0, output={"e_sha256": "e1"}, measurement_commit="b" * 40)
    clock.t += 1
    later = sub(b, val_bpb=1.0, output={"e_sha256": "e2"})
    clock.t += 1
    copy = sub(b, val_bpb=1.0, output={"e_sha256": "e1"})
    clock.t += 1
    other = sub(a, val_bpb=1.0, measurement_commit="c" * 40)
    rows = Ledger(tmp_path).index()
    assert frontier.status(rows, (PIN, "b" * 40)) == {copy["id"]: "duplicate-e", other["id"]: "other-version"}
    assert [r["id"] for r in frontier.valid(rows, (PIN, "b" * 40))] == [earlier["id"], later["id"]]
    monkeypatch.setitem(sys.modules, "pin", SimpleNamespace(pinned=lambda: {"commit": PIN, "same_version": ["b" * 40]}))
    monkeypatch.setattr(sys, "path", list(sys.path))
    assert frontier.pinned() == (PIN, "b" * 40)


def test_a_run_with_covered_steps_counts_once_rho_is_priced_by_thresholds():
    covered = [{"covers": [0, 1], "beta": {"0": 1e-6, "1": 32.0}}]
    rows = [{"id": name, "val_bpb": 1.0, "invalid": [], "measurement_commit": PIN, "te_core": ingest, "families": fams}
            for name, ingest, fams in [("plain", "2.3", []), ("old-priced", "2.3", covered),
                                       ("repriced", "3.0", covered), ("repriced-later", "3.10", covered)]]
    assert frontier.status(rows, PIN) == {"old-priced": "re-ingest"}
    assert [r["id"] for r in frontier.valid(rows, PIN)] == ["plain", "repriced", "repriced-later"]
