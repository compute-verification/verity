import json
import random
import sys
from types import SimpleNamespace

import pytest

from te_core import VERSION, frontier, launch
from te_core.ledger import Ledger
from te_core.run import file_sha256

PIN = "a" * 40
#: a start's shape (te_core.policy), the small start's
SHAPE = {"d_model": 512, "n_layer": 6, "vocab": 50304, "ctx": 1024, "parameters": 45_029_888, "tokens": 154_206_208,
         "forward_macs_per_token": 51_342_080}
CELL = "reference"


class Clock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


def sub(led, **kw):
    """A submission whose formalism block makes its |C| `macs` gates with no dependent gates (ρ = 1 everywhere); `rho`
    is its ρ in `CELL` for `scored`."""
    body = {"macs": 100, "val_bpb": 1.0, "tier": 0, "start": "small", "start_bpb": 1.1, "mechanism": "dense",
            "reference": False, "program_sha256": "", "invalid": [], "run_te_core": VERSION,
            "measurement_commit": PIN} | kw
    body.setdefault("formalism", {"parts": [{"gates": body["macs"], "tokens": 10, "dep_gates": 0.0}], "u1_bits": 1e6})
    return led.append("submission", body)


def scored(rows):
    """The valid rows as `frontier.target` reads them: |C| in gates (`macs` here), ρ in `CELL`."""
    return [r | {"gates": r["macs"], "rho": {CELL: r.get("rho", 1.0)}} for r in frontier.valid(rows, PIN)]


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


def test_frontier_elites_and_promotion_candidates(tmp_path):
    a = Ledger(tmp_path, "agent-a")
    ref = sub(a, macs=100, val_bpb=1.08, reference=True, mechanism="reference", direction="ref-optim")
    cheap = sub(a, macs=50, val_bpb=1.085, mechanism="lowrank")
    good = sub(a, macs=80, val_bpb=1.05, mechanism="dense")
    dominated = sub(a, macs=500, val_bpb=1.10, mechanism="dense")
    bad = sub(a, macs=1, val_bpb=0.5, invalid=["no step"])
    other = sub(a, macs=1, val_bpb=0.5, measurement_commit="b" * 40)
    rows = Ledger(tmp_path).index()
    on, skipped = frontier.formalism_rows(frontier.valid(rows, PIN), {"small": SHAPE})
    assert not skipped and all(r["rho_by"][name][CELL] == 1.0 for r in on for name in r["rho_by"])
    on = frontier.reading(on, "tail-charged")
    assert [r["id"] for r in frontier.pareto(on, CELL)] == [cheap["id"], good["id"]]
    assert dominated["id"] not in {r["id"] for r in frontier.elites(on, CELL).values()}
    assert set(frontier.candidates(rows, 1, PIN, {"small": SHAPE})) == {ref["id"], cheap["id"], good["id"]}
    assert frontier.candidates(rows, 1, PIN, {}) == []  # no start shape: nothing the formalism scores
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


def test_a_later_ingest_of_a_run_supersedes_its_earlier_rows_and_is_not_a_duplicate_of_its_own_e():
    def row(name, ts, run_id="r1", e="e1", agent="a"):
        return {"id": name, "written_ts": ts, "val_bpb": 1.0, "invalid": [], "measurement_commit": PIN, "te_core": "4.0",
                "families": [], "run_id": run_id, "checkpoint_step": 3, "agent": agent, "output": {"e_sha256": e}}

    rows = [row("first", 1), row("other-run", 2, run_id="r2", e="e2"), row("again", 3, agent="scorer")]
    assert frontier.status(rows, PIN) == {"first": "re-ingested"}
    assert [r["id"] for r in frontier.valid(rows, PIN)] == ["other-run", "again"]
    rows.append(row("copy", 4, run_id="r3", agent="b"))  # another agent's run of the same E
    assert frontier.status(rows, PIN)["copy"] == "duplicate-e"


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


def test_a_row_is_measured_once_its_run_is_ingested_with_its_traced_circuit():
    """Every row before 4.0 holds stubs for what the formalism measures, as does a 4.0 row with an updated step the
    capture didn't trace or with no trace in its block; all stay valid (scored on their labelled stubs)."""
    rows = [{"id": name, "val_bpb": 1.0, "invalid": [], "measurement_commit": PIN, "te_core": ingest,
             "annotation": {"untraced": untraced}, "formalism": {"trace": {}} if trace else {}}
            for name, ingest, untraced, trace in [("old", "3.13", 0, True), ("older", "2.3", 0, False),
                                                  ("traced", "4.0", 0, True), ("later", "4.10", 0, True),
                                                  ("untraced", "4.0", 3, True), ("stubs", "4.0", 0, False)]]
    assert frontier.status(rows, PIN) == {}
    assert [r["id"] for r in rows if frontier.measured(r)] == ["traced", "later"]


def at_one_percent(rows):
    t = frontier.target(scored(rows), 1.1 * 0.99, [CELL])
    return t | t["cells"][CELL]


def test_honest_and_w_star_read_every_checkpoint_of_every_record(tmp_path):
    """The coordinator's 19:50Z ruling: honest(l*) is the least |C| over every checkpoint of every valid record that
    reaches l*, mid-schedule ones and covert records included, and W* the least |C| ρ over the same checkpoints."""
    a = Ledger(tmp_path, "agent-a")
    sub(a, run_id="ref", checkpoint_step=9, macs=100, val_bpb=1.08, direction="ref-optim", reference=True)
    ref_mid = sub(a, run_id="ref", checkpoint_step=2, macs=20, val_bpb=1.085, direction="ref-optim", reference=True)
    sub(a, run_id="cov", checkpoint_step=4, macs=40, val_bpb=1.085, rho=5.0, direction="wildcard")
    cov_mid = sub(a, run_id="cov", checkpoint_step=1, macs=10, val_bpb=1.088, rho=5.0, direction="wildcard")
    t = at_one_percent(Ledger(tmp_path).index())
    assert t["honest"] == 10 and t["honest_of"]["id"] == cov_mid["id"] and "bound" not in t
    assert t["honest_reference"] == 20 and t["honest_reference_of"]["id"] == ref_mid["id"]
    assert t["w_star"] == 20 and t["of"]["id"] == ref_mid["id"]  # 20 x 1 < 10 x 5
    assert t["effective_slowdown"] == pytest.approx(2.0)


def test_w_star_over_honest_is_never_below_one(tmp_path):
    rng = random.Random(0)
    a = Ledger(tmp_path, "agent-a")
    for i in range(200):
        sub(a, run_id=f"r{i % 23}", checkpoint_step=i // 23, macs=rng.randint(1, 1000), val_bpb=rng.uniform(1.0, 1.1),
            rho=rng.choice([1.0, 1.0, rng.uniform(1.0, 50.0)]), direction=rng.choice(["ref-optim", "sparse", "wildcard"]))
    on = scored(Ledger(tmp_path).index())
    slowdowns = [frontier.target(on, 1.1 * (1 - f), [CELL])["cells"][CELL]["effective_slowdown"]
                 for f in frontier.TARGETS]
    assert all(s is not None for s in slowdowns) and min(slowdowns) >= 1.0


def test_a_target_only_a_mid_schedule_checkpoint_reaches_is_scored_like_any_other(tmp_path):
    a = Ledger(tmp_path, "agent-a")
    mid = sub(a, run_id="long", checkpoint_step=3, macs=30, val_bpb=1.085, direction="sparse")
    sub(a, run_id="long", checkpoint_step=9, macs=90, val_bpb=1.095, direction="sparse")
    t = at_one_percent(Ledger(tmp_path).index())
    assert t["honest"] == 30 and t["honest_of"]["id"] == mid["id"] and t["of"]["id"] == mid["id"]
    assert t["honest_reference"] is None and t["effective_slowdown"] == pytest.approx(1.0)


def test_a_read_out_leak_is_left_out_and_a_tied_lookup_record_is_not(tmp_path):
    """4.0 reads a tied embedding's update at its lookup's update-dependent rows, so the 13:32Z tied-lookup rule is
    retired; a read-out leak, given by `--facts` for an older row, still is."""
    a = Ledger(tmp_path, "agent-a")
    top = sub(a, run_id="top", changed=["embed", "blocks.1.attn.weight"], layers=2)
    leak = sub(a, run_id="leak", readout_leaks=[{"step": 0, "tokens": 8, "consumed": 2, "exited": 2}])
    old = sub(a, run_id="old")
    rows = Ledger(tmp_path).index()
    assert frontier.status(rows, PIN) == {leak["id"]: "read-out-leak"}
    rows = frontier.with_facts(rows, {"old": {"readout_leaks": [{"step": 1}]}})
    assert frontier.status(rows, PIN) == {leak["id"]: "read-out-leak", old["id"]: "read-out-leak"}
    assert [r["id"] for r in frontier.valid(rows, PIN)] == [top["id"]]
