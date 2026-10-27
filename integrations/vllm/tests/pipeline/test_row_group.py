"""A config run's Commit stage as one grouped Commit of several rows of one engine (`pipeline.row_group`).

* the members file names every row of the group, the lead among them, each in a row dir of its own;
* a member is the lead's row with its own id, model and sweep dir: the lead's configuration (but the group) and environment, its own
  timeline, no research run of its own;
* the lead refuses by name a group it can't run: not a config run, a replay that isn't deferred, the hot worker, serving rows,
  protocol options, a tensor-parallel row, a members file that doesn't name it;
* a member whose precheck or Commit preparation stops, that is tensor-parallel or whose Commit environment differs from the lead's in
  more than its row dir and timeline is refused by name, as is one whose command runs another interpreter;
* every row gets its own verdict from the group's result, the members' first, the lead's last; each records the group's one lease
  and its part in the group; a member refused by the group, or that the group never reached, has a Commit FAIL that says why;
* the lead's replay stage replays each member whose Commit in the group ended PENDING, then the lead;
* the lead's commit.log, which the GPU task's stall watchdog watches, moves while any row of the group writes its log.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.pipeline.test_row import ROW, _row
from verity_vllm.config import UsageError
from verity_vllm.pipeline import cli
from verity_vllm.pipeline import gpu_lease as GL
from verity_vllm.pipeline import row_config as RC
from verity_vllm.pipeline import row_group as RG
from verity_vllm.pipeline.row_driver import Stop
from verity_vllm.pipeline.row_stages import SingleRow

TP_ROW = "llama32-1b__bf16__l40s__tp2__b1__i256__o32__mixed__greedy__bi-eager"


def _members(path: Path, *rows: tuple[str, str]) -> str:
    path.write_text(json.dumps({"schema": RG.MEMBERS_SCHEMA, "rows": [{"row": r, "role": "R", "repo": "o/r", "rev": "v", "sweep_dir": s}
                                                                       for r, s in rows]}))
    return str(path)


class Fake:
    """A row as `row_group` reads it: its records in memory, its stages.txt and row.log on disk."""

    def __init__(self, tmp: Path, row: str, *, group: str = "", trap: bool = True):
        self.row, self.d, self.trap, self.lines = row, str(tmp / row), trap, []
        self.env = SimpleNamespace(sweep_dir=str(tmp), environ={})
        self.ck_ = SimpleNamespace(group=group, replay_deferred="1", hot_engine="0", serving_rows=None)
        self.config_run, self.protocols = True, None
        os.makedirs(self.d, exist_ok=True)

    def verdict(self, stage, status, detail, *, quiet=False):
        self.lines.append(f"{stage} {status} {detail}")

    def tl(self, *a):
        self.lines.append(f"tl {a[0]} {a[1]}")

    def log(self, msg):
        self.lines.append(f"log {msg}")

    def commit_stop(self, rc, detail, stage="commit"):
        self.verdict(stage, "FAIL", detail)
        raise Stop(rc)

    def install_signal_forwarding(self):
        self.lines.append("signals")

    def prepare(self):
        self.lines.append("prepare")


# ---------------------------------------------------------------------------------------------------------------- members

def test_the_members_file_names_every_row_each_in_a_row_dir_of_its_own(tmp_path):
    p = tmp_path / "m.json"
    ms = RG.load(_members(p, ("a", str(tmp_path)), ("b", str(tmp_path))))
    assert [(m.row, m.role, m.repo, m.rev, m.sweep_dir, m.gpu_util) for m in ms] == [("a", "R", "o/r", "v", str(tmp_path), None),
                                                                                       ("b", "R", "o/r", "v", str(tmp_path), None)]
    for doc, why in [({"schema": "x", "rows": [{}]}, "not a"), ({"schema": RG.MEMBERS_SCHEMA, "rows": []}, "not a"),
                     ({"schema": RG.MEMBERS_SCHEMA, "rows": [{"row": "a"}]}, "a row needs")]:
        p.write_text(json.dumps(doc))
        with pytest.raises(ValueError, match=why):
            RG.load(str(p))
    os.symlink(tmp_path, tmp_path / "same")
    with pytest.raises(ValueError, match="share a row dir"):
        RG.load(_members(p, ("a", str(tmp_path)), ("a", str(tmp_path / "same"))))


def test_a_member_is_the_leads_row_with_its_own_id_model_sweep_dir_and_timeline(tmp_path):
    lead = _row(tmp_path / "lead", "--config-run", "1", "--replay-deferred", "1", "--commit-group", "m.json",
                environ={"VERITY_TIMELINE": str(tmp_path / "lead.jsonl"), "RESEARCH_RUN_DIR": "/runs/r1", "RESEARCH_RUN_ID": "r1"})
    assert lead.ck_.group == "m.json" and lead.timeline == str(tmp_path / "lead.jsonl")
    m = RG.member(lead, RG.Member(row="other-row", role="R2", repo="o/r2", rev="v2", sweep_dir=str(tmp_path / "b"), gpu_util="0.6"))
    assert isinstance(m, SingleRow) and (m.row, m.role, m.repo, m.rev, m.gpu_util) == ("other-row", "R2", "o/r2", "v2", "0.6")
    assert m.d == f"{tmp_path}/b/other-row" and m.timeline == f"{m.d}/timeline.jsonl" and m.ck_.group == ""
    assert m.env.environ["SWEEP_DIR"] == str(tmp_path / "b") and "VERITY_TIMELINE" not in m.env.environ
    assert m.env.research_run_dir is None and m.env.research_run_id is None and m.protocols is None
    assert m.ck_.config_run == lead.ck_.config_run and m.cb is lead.cb and m.stages == lead.stages
    assert RG.is_lead(lead, RG.Member(row=ROW, role="R", repo="o/r", rev="v", sweep_dir=str(tmp_path / "lead")))
    assert not RG.is_lead(lead, RG.Member(row=ROW, role="R", repo="o/r", rev="v", sweep_dir=str(tmp_path / "b")))
    assert RG.member(lead, RG.Member(row="x", role="R", repo="o/r", rev="v", sweep_dir=str(tmp_path))).gpu_util == lead.gpu_util


def test_the_lead_refuses_by_name_a_group_it_cannot_run(tmp_path):
    lead = Fake(tmp_path, "a")
    assert RG.refusal(lead) is None
    for change, why in [(lambda r: setattr(r, "config_run", False), "--config-run 1"),
                        (lambda r: setattr(r.ck_, "replay_deferred", "0"), "--replay-deferred"),
                        (lambda r: setattr(r.ck_, "hot_engine", "1"), "--hot-engine 1"),
                        (lambda r: setattr(r.ck_, "serving_rows", "p.json"), "--serving-rows"),
                        (lambda r: setattr(r, "protocols", ("pouw:ncp", "sampled-proofs:rows-sha512")), "protocol options")]:
        r = Fake(tmp_path, "a")
        change(r)
        assert why in RG.refusal(r)


def test_a_tensor_parallel_row_refuses_a_grouped_commit(tmp_path):
    from verity_vllm.pipeline import row
    o = cli.parse("row", ["run", TP_ROW, "R", "o/r", "v", "--sweep-dir", str(tmp_path), "--py", sys.executable, "--commit-group", "m.json"],
                  {"PY": sys.executable})
    with pytest.raises(UsageError, match="--commit-group: .* is a tensor-parallel row"):
        row.run_row(o, "commit")


def test_a_lead_the_members_file_does_not_name_stops_its_commit_by_name(tmp_path):
    lead = Fake(tmp_path, "a", group=_members(tmp_path / "m.json", ("b", str(tmp_path))))
    with pytest.raises(Stop) as e:
        RG.commit(lead, 14400)
    assert e.value.rc == 70 and "does not name this row (a in" in lead.lines[-1] and lead.lines[0] == "tl stage.commit start"
    lead = Fake(tmp_path, "a", group=str(tmp_path / "absent.json"))
    with pytest.raises(Stop):
        RG.commit(lead, 14400)
    assert "absent.json" in lead.lines[-1] and "REFUSED" in lead.lines[-1]


# ---------------------------------------------------------------------------------------------------------------- a member's preparation

class Member(Fake):
    def __init__(self, tmp, row, *, stop=None, env=None, cmd=None):
        super().__init__(tmp, row)
        self.wl, self.stop, self.child_env, self.cmd = "wl.json", stop, {"A": "1", "SWEEP_DIR": "s", "VERITY_TIMELINE": row, **(env or {})}, cmd
        (tmp / "wl.json").write_text(json.dumps({"model": "m"}))
        self.tmp = tmp

    def path(self, rel):
        return str(self.tmp / rel)

    def prepare(self):
        if self.stop == "precheck":
            raise Stop(3)

    def commit_prepare(self):
        if self.stop == "commit":
            raise Stop(12)
        return 600, self.cmd or ["py", "-m", "verity_vllm.pipeline.cli", "commit", "--label", self.row], ""


def test_a_member_whose_preparation_stops_or_whose_environment_differs_is_refused_by_name(tmp_path):
    lead = Member(tmp_path, "a", env={"SWEEP_DIR": "lead", "VERITY_TIMELINE": "lead"})
    assert RG.prepared(lead, Member(tmp_path, "b")) == (600, ["py", "-m", "verity_vllm.pipeline.cli", "commit", "--label", "b"])
    assert RG.prepared(lead, Member(tmp_path, "c", stop="precheck")) == f"its precheck stopped (exit 3; {tmp_path}/c/stages.txt)"
    assert RG.prepared(lead, Member(tmp_path, "d", stop="commit")) == f"its Commit preparation stopped (exit 12; {tmp_path}/d/stages.txt)"
    assert RG.prepared(lead, Member(tmp_path, "e", env={"A": "2", "B": "x"})) == "its Commit's environment differs from the lead's in ['A', 'B']"
    assert "tensor-parallel" in RG.prepared(lead, Member(tmp_path, TP_ROW))


def test_the_group_runs_the_rows_it_could_prepare_and_refuses_the_others_by_name(tmp_path, monkeypatch):
    lead = Member(tmp_path, "a")
    lead.ck_.group = _members(tmp_path / "m.json", *((r, str(tmp_path)) for r in "abcd"))
    rows = {"b": Member(tmp_path, "b"), "c": Member(tmp_path, "c", stop="precheck"),
            "d": Member(tmp_path, "d", cmd=["/other/py", "-m", "verity_vllm.pipeline.cli", "commit"])}
    monkeypatch.setattr(RG, "member", lambda lead, m: rows[m.row])
    ran = {}

    def run_group(lead, ready, cap):
        ran.update(rows=[r.row for r, _, _ in ready], cap=cap)
        return 0, 30, str(tmp_path / "none.json")

    def verdicts(lead, ready, refused, rc, wall, result):
        ran.update(refused={r.row: why for r, why in refused}, rc=rc, wall=wall)
        return True

    monkeypatch.setattr(RG, "run_group", run_group)
    monkeypatch.setattr(RG, "verdicts", verdicts)
    assert RG.commit(lead, 14400) is True
    assert ran["rows"] == ["a", "b"] and ran["cap"] == 14400 and (ran["rc"], ran["wall"]) == (0, 30)
    assert ran["refused"] == {"c": f"its precheck stopped (exit 3; {tmp_path}/c/stages.txt)",
                              "d": "its Commit runs another interpreter or command (['/other/py', '-m', 'verity_vllm.pipeline.cli', 'commit'])"}
    assert lead.lines[-2:] == [f"log grouped Commit: c refused: {ran['refused']['c']}", f"log grouped Commit: d refused: {ran['refused']['d']}"]


def test_the_group_is_one_process_under_the_leads_lease_over_each_rows_own_command_line(tmp_path):
    class Lead(Member):
        def cli(self, *a):
            return ["py", "-m", "verity_vllm.pipeline.cli", *a]

        def gpu_exclusive(self, why):
            self.lines.append(f"exclusive {why}")

        def run_gpu(self, stage, to, lf, cmd):
            self.ran = (stage, to, lf, cmd)
            return 0

    lead, b = Lead(tmp_path, "a"), Member(tmp_path, "b")
    lead.ck_.group = "m.json"
    Path(lead.d, "commit_group").mkdir()
    Path(lead.d, "commit_group", "result.json").write_text("{}")                # a stale result is not read as this group's
    rc, _, result = RG.run_group(lead, [(lead, 600, lead.commit_prepare()[1]), (b, 900, b.commit_prepare()[1])], 1000)
    spec_path = f"{lead.d}/commit_group/spec.json"
    assert rc == 0 and result == f"{lead.d}/commit_group/result.json" and not os.path.exists(result)
    assert lead.ran == ("commit.commit_group", 1000, f"{lead.d}/commit_group.log", ["py", "-m", "verity_vllm.pipeline.cli", "commit-group", spec_path])
    spec = json.loads(Path(spec_path).read_text())
    assert (spec["schema"], spec["lead"], spec["members"], spec["result"]) == ("verity-vllm/commit-group/v1", "a", "m.json", result)
    assert spec["rows"][1] == {"row": "b", "dir": b.d, "argv": ["--label", "b"], "env": {"SWEEP_DIR": "s", "VERITY_TIMELINE": "b"},
                               "log": f"{b.d}/commit.log"}
    assert Path(b.d, "commit.log").read_text() == (f"$ py -m verity_vllm.pipeline.cli commit --label b\n# row 2 of 2 of the grouped Commit py -m "
                                                   f"verity_vllm.pipeline.cli commit-group {spec_path} (lead a)\n")
    assert "tl commit.commit_delta start" in b.lines and "tl commit.commit_delta start" not in lead.lines and "exclusive commit-group" in lead.lines


# ---------------------------------------------------------------------------------------------------------------- verdicts

def test_every_row_has_its_own_verdict_from_the_groups_result_the_leads_last(tmp_path, monkeypatch):
    lead, b, c, d, e = (Fake(tmp_path, r) for r in "abcde")
    result = tmp_path / "result.json"
    result.write_text(json.dumps({"rows": {"a": {"rc": 0, "refused": None, "wall_s": 40.0}, "b": {"rc": 0, "refused": None, "wall_s": 20.0},
                                           "c": {"rc": 70, "refused": "HOT-KEY-MISMATCH: live engine differs"},
                                           "x": {"rc": 1}}}))
    (tmp_path / "a" / GL.RECORD).write_text(json.dumps({"schema": "verity-vllm/gpu-lease/v1", "held_min": 2.5}))
    order, records, discarded = [], [], []

    def commit_verdict(row, rck, wall):
        order.append((row.row, rck, wall))
        return True

    monkeypatch.setattr(RC, "commit_verdict", commit_verdict)
    monkeypatch.setattr(RC, "config_record", lambda row: records.append(row.row))
    monkeypatch.setattr(RC, "discard_bundles", lambda row: discarded.append(row.row))
    ready = [(lead, 600, []), (b, 600, []), (c, 600, []), (d, 600, [])]
    assert RG.verdicts(lead, ready, [(e, "its precheck stopped (exit 3)")], 0, 90, str(result)) is True
    assert order == [("b", 0, 20), ("a", 0, 40)] and records == ["c", "d", "e"] and discarded == ["c", "d", "e"]
    assert "commit FAIL rc=70 REFUSED by the grouped Commit: HOT-KEY-MISMATCH: live engine differs" in c.lines
    assert "commit FAIL rc=70 REFUSED by the grouped Commit: the grouped Commit ended (exit 0) before this row's Commit ran" in d.lines
    assert "commit FAIL rc=70 REFUSED by the grouped Commit: its precheck stopped (exit 3)" in e.lines
    group = {"lead": "a", "rows": ["a", "b", "c", "d"], "record": f"{lead.d}/{GL.RECORD}", "wall_s": 90, "rc": 0}
    for r in (lead, b, c, d, e):
        assert json.loads(Path(r.d, GL.RECORD).read_text()) == {"schema": "verity-vllm/gpu-lease/v1", "held_min": 2.5, "group": group}
    rec = {r.row: json.loads(Path(r.d, "commit_group.json").read_text()) for r in (lead, b, c, d, e)}
    assert {k: v["pending"] for k, v in rec.items()} == {"a": True, "b": True, "c": False, "d": False, "e": False}
    assert rec["a"]["this"] == {"rc": 0, "refused": None, "wall_s": 40.0} and rec["a"]["refused"] is None
    assert rec["e"]["refused"] == "its precheck stopped (exit 3)" and rec["c"]["schema"] == RG.ROW_SCHEMA


def test_a_failed_member_is_recorded_and_the_lead_still_decides_its_own(tmp_path, monkeypatch):
    lead, b = Fake(tmp_path, "a"), Fake(tmp_path, "b")
    result = tmp_path / "result.json"
    result.write_text(json.dumps({"rows": {"a": {"rc": 1, "refused": None}, "b": {"rc": 1, "refused": None}}}))
    records: list[str] = []

    def commit_verdict(row, rck, wall):
        raise Stop(12)

    monkeypatch.setattr(RC, "commit_verdict", commit_verdict)
    monkeypatch.setattr(RC, "config_record", lambda row: records.append(row.row))
    with pytest.raises(Stop):
        RG.verdicts(lead, [(lead, 600, []), (b, 600, [])], [], 0, 90, str(result))
    assert records == ["b"]
    assert json.loads(Path(lead.d, "commit_group.json").read_text())["pending"] is False


def test_a_lead_the_group_refused_stops_with_its_reason(tmp_path, monkeypatch):
    lead = Fake(tmp_path, "a")
    result = tmp_path / "result.json"
    result.write_text(json.dumps({"rows": {"a": {"rc": 70, "refused": "HOT-UNSAFE: --arms"}}}))
    monkeypatch.setattr(RC, "commit_verdict", lambda *a: pytest.fail("a refused lead has no Commit to decide"))
    with pytest.raises(Stop) as e:
        RG.verdicts(lead, [(lead, 600, [])], [], 0, 90, str(result))
    assert e.value.rc == 70 and lead.lines[-1] == "commit FAIL rc=70 REFUSED by the grouped Commit: HOT-UNSAFE: --arms"


# ---------------------------------------------------------------------------------------------------------------- replay

def test_the_leads_replay_stage_replays_each_pending_member_then_the_lead(tmp_path, monkeypatch):
    lead = Fake(tmp_path, "a", group=_members(tmp_path / "m.json", *((r, str(tmp_path)) for r in "abcd")))
    rows = {r: Fake(tmp_path, r) for r in "bcd"}
    for r, doc in {"b": {"lead": "a", "pending": True}, "c": {"lead": "a", "pending": False}, "d": {"lead": "other", "pending": True}}.items():
        Path(rows[r].d, "commit_group.json").write_text(json.dumps(doc))
    monkeypatch.setattr(RG, "member", lambda lead, m: rows[m.row])
    replays, records = [], []

    def replay(row):
        replays.append(row.row)
        if row.row == "b":
            raise Stop(12)

    monkeypatch.setattr(RC, "replay", replay)
    monkeypatch.setattr(RC, "config_record", lambda row: records.append(row.row))
    RG.replay(lead)
    assert replays == ["b", "a"] and records == ["b"] and rows["b"].lines[0] == "prepare"
    assert any("c not replayed" in ln for ln in lead.lines) and any("d not replayed" in ln for ln in lead.lines)
    assert any(ln.startswith("log grouped replay: b stopped (exit 12") for ln in lead.lines) and "signals" in lead.lines


def test_a_singlerow_with_a_group_runs_its_commit_and_replay_through_the_group(tmp_path, monkeypatch):
    r = _row(tmp_path, "--config-run", "1", "--replay-deferred", "1", "--commit-group", "m.json", "--stages", "replay")
    calls: list[str] = []
    monkeypatch.setattr(RG, "commit", lambda row, cap: calls.append(f"commit {cap}") or True)
    monkeypatch.setattr(SingleRow, "commit_prepare", lambda self: pytest.fail("the group prepares the lead's Commit"))
    r.commit()
    assert calls == ["commit 14400"] and r.replay_pending is True
    monkeypatch.setattr(SingleRow, "prepare", lambda self: None)
    monkeypatch.setattr(SingleRow, "timeline_report", lambda self: None)
    monkeypatch.setattr(RG, "replay", lambda row: calls.append("group replay"))
    monkeypatch.setattr(RC, "replay", lambda row: pytest.fail("a grouped row replays through the group"))
    monkeypatch.setattr(RC, "config_record", lambda row: calls.append("record"))
    assert r.run() == 0 and calls[1:] == ["group replay", "record"]


# ---------------------------------------------------------------------------------------------------------------- the watchdog's log

def test_the_leads_commit_log_moves_to_the_newest_log_of_the_group(tmp_path):
    target, a, b = tmp_path / "commit.log", tmp_path / "a.log", tmp_path / "b.log"
    for p, t in ((target, 100), (a, 50), (b, 300)):
        p.write_text("x")
        os.utime(p, (t, t))
    assert RG.mirror(str(target), [str(a), str(b), str(tmp_path / "absent.log")]) == 300 and os.stat(target).st_mtime == 300
    assert RG.mirror(str(target), [str(a), str(b)]) is None
    assert RG.mirror(str(target), [str(tmp_path / "absent.log")]) is None
    assert RG.mirror(str(tmp_path / "no-target.log"), [str(b)]) is None
