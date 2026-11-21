"""The challenger as the party `verity run` starts first (`served_table.STARTS`): its hosts on the run's own machine,
each under a key made for the run, write SPOT's and Pearl+'s bindings, which SPOT's piece hands the runner; it is
stopped when the run ends. Its hosts and Pearl+'s gate are stand-ins written by each test."""
from __future__ import annotations

import ast
import inspect
import json
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from verity.channel import Refused
from verity_cli import run as R

ROOT = Path(R.ROOT)
CONFIG = ROOT / "applications" / "onsite_full.toml"
REG = R.registry()
T = sys.modules["served_table"]
LINE = ["c" * 40, "b" * 40, "a" * 40]
KEY = "0" * 128
ADDRESS = "127.0.0.1:40001"


class Host:
    made: list[Host] = []

    def __init__(self, root: Path, main: list[str], local) -> None:
        self.root, self.main, self.local, self.stopped = root, main, local, False
        Host.made.append(self)

    def start(self) -> dict[str, str]:
        return {"address": ADDRESS, "pin": self.main[0]}

    def stop(self) -> None:
        self.stopped = True


@pytest.fixture
def stand_ins(tmp_path, monkeypatch):
    """The run's state with its run.json, the git line, Pearl+'s gate (``init`` answers ``pin``, or exits ``rc``) and
    SPOT's host, each a stand-in; the calls they got."""
    state, calls, gate = tmp_path / "state", [], {"rc": 0, "pin": LINE[0], "stop": 0}
    state.mkdir()
    (state / "run.json").write_text(json.dumps({"run": "r-test", "program": LINE[0], "periods": 1}))
    Host.made.clear()

    def run(argv, **kw):
        calls.append(list(argv))
        if argv[:3] == ["git", "rev-list", "--first-parent"]:
            return subprocess.CompletedProcess(argv, 0, "\n".join(LINE) + "\n", "")
        if argv[2] == "init":
            Path(argv[4]).mkdir(parents=True)
            doc = {"gate": f"{sys.executable} {T.PEARLPLUS_GATE} gate --root {argv[4]}", "peer": "local",
                   "lane": "vllm", "pin": gate["pin"], "key_sha512": KEY}
            return subprocess.CompletedProcess(argv, gate["rc"], f"[coins] built\n{json.dumps(doc)}\n", "no lake")
        if argv[2] == "stop":
            return subprocess.CompletedProcess(argv, gate["stop"], "", "no ledger to copy")
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(T.subprocess, "run", run)
    monkeypatch.setattr(T.CP, "Host", Host)
    return state, calls, gate


def test_every_party_the_table_starts_first_is_registered_into_the_one_starts_table():
    """A second `STARTS = ...` merges without a conflict and drops every party registered before it."""
    assert set(T.STARTS) == {T.CHALLENGER, T.NETWORK_ATTESTER}
    module = ast.parse(Path(T.__file__).read_text())
    assigned = [n.lineno for n in module.body if isinstance(n, (ast.Assign, ast.AnnAssign))
                for t in (n.targets if isinstance(n, ast.Assign) else [n.target]) if isinstance(t, ast.Name)
                and t.id == "STARTS"]
    assert len(assigned) == 1, f"served_table.py assigns STARTS at lines {assigned}: register with STARTS[PARTY] = ..."


def test_the_challenger_writes_spots_and_pearl_plus_bindings():
    assert {c: list(k) for c, k in T.STARTS[T.CHALLENGER].writes.items()} == {
        "sampled-proofs": ["loopback", "main-file"], "pearl-plus": ["gate", "peer", "lane", "key-sha512", "main-file"]}
    assert (ROOT / T.PEARLPLUS_GATE).is_file()


def test_its_serve_takes_the_machines_this_host_stands_for_or_none():
    serve = inspect.signature(T.STARTS[T.CHALLENGER].serve)
    serve.bind({}, Path("state"))
    serve.bind({}, Path("state"), ["prover-node"])


def test_its_hosts_run_under_keys_made_for_the_run_pinned_on_the_runs_commit_and_its_bindings_are_whole(stand_ins):
    state, calls, _ = stand_ins
    cfg = R.load(CONFIG, REG)
    T.STARTS[T.CHALLENGER].serve(cfg, state)
    mine, public = state / "parties" / T.CHALLENGER, state / "public" / T.CHALLENGER
    got = R.wait_bindings(state, T.CHALLENGER, T.STARTS[T.CHALLENGER], lambda: True)
    line = str(public / "main.txt")
    assert got == {"sampled-proofs": {"loopback": ADDRESS, "main-file": line},
                   "pearl-plus": {"gate": f"{sys.executable} {T.PEARLPLUS_GATE} gate --root {mine / 'pearl-plus'}",
                                  "peer": "local", "lane": "vllm", "key-sha512": KEY, "main-file": line}}
    assert (public / "main.txt").read_text().split() == LINE
    assert [(h.root, h.main, h.local) for h in Host.made] == [(mine / "sampled-proofs", LINE, T._deployed_local)]
    assert calls[0] == ["git", "rev-list", "--first-parent", LINE[0]]
    assert calls[1] == [sys.executable, str(ROOT / T.PEARLPLUS_GATE), "init", "--root", str(mine / "pearl-plus")]
    assert not (public / f".{T.BINDINGS}").exists()

    T.STARTS[T.CHALLENGER].stop(cfg, state)
    assert Host.made[0].stopped
    assert calls[-1] == [sys.executable, str(ROOT / T.PEARLPLUS_GATE), "stop", "--root", str(mine / "pearl-plus"),
                         "--out", str(public / "pearl-plus")]


@pytest.mark.parametrize("gate", [{"rc": 2}, {"pin": "d" * 40}], ids=["init-fails", "pinned-elsewhere"])
def test_a_gate_not_made_or_pinned_off_the_runs_commit_is_refused_and_writes_no_bindings(stand_ins, gate):
    state, _, given = stand_ins
    given.update(gate)
    with pytest.raises(Refused) as e:
        T.STARTS[T.CHALLENGER].serve(R.load(CONFIG, REG), state)
    assert e.value.name == "challenger"
    assert not (state / "public" / T.CHALLENGER / T.BINDINGS).exists() and not Host.made


def test_spots_loopback_host_makes_its_key_at_the_deployed_parameters_v2_reads_coins_under(monkeypatch, tmp_path):
    made = []
    monkeypatch.setattr(T.LC, "Local", lambda root, main, **kw: made.append((root, main, kw)))
    T._deployed_local(tmp_path, LINE)
    assert made == [(tmp_path, LINE, {"params": (1, 20), "workers": len(T.os.sched_getaffinity(0))})]
    assert (T.lms.DEPLOYED.W, T.lms.DEPLOYED.H) == (1, 20)


def test_a_failed_stop_of_pearl_plus_gate_fails_the_challenger_and_the_run(stand_ins, capsys):
    state, _, gate = stand_ins
    gate["stop"] = 1
    cfg = R.load(CONFIG, REG)
    runner = R.InProcess(cfg, REG, state, True)
    assert R.execute(cfg, REG, state, [T.CHALLENGER], runner, [], ["gate-host"], [T.CHALLENGER]) == 2
    assert capsys.readouterr().out.splitlines()[-1] == 'PARTY {"exit": 1, "party": "challenger"}'
    error = (state / "logs" / f"{T.CHALLENGER}.error").read_text()
    assert error.startswith("Refused") and "stop exited 1" in error and "no ledger to copy" in error
    assert Host.made[0].stopped


def test_spots_piece_hands_the_runner_the_loopback_host_its_development_marker_whatever_the_hold_and_a_gate_first(
        monkeypatch):
    """The hold's development run is served_verdicts.sh's to decide (placement.sh's ``development_under_hold``): the
    hooks hand the same variables held or not."""
    cfg, piece = R.load(CONFIG, REG), T.PIECES["sampled-proofs"][T.WORKERS]
    loop = {"loopback": ADDRESS, "main-file": "/s/public/challenger/main.txt"}
    for hold in (None, "1"):
        if hold is None:
            monkeypatch.delenv("VERITY_HOLD_PLACEMENT", raising=False)
        else:
            monkeypatch.setenv("VERITY_HOLD_PLACEMENT", hold)
        assert piece.variables(cfg, {"challenger": {"sampled-proofs": loop}}) == {
            "PP_CHALLENGER": f"loopback:{ADDRESS}", "MAIN_FILE": "/s/public/challenger/main.txt",
            "PP_DEVELOPMENT_UNDER_HOLD": "1"}
        assert piece.variables(cfg, {}) == {"PP_DEVELOPMENT_UNDER_HOLD": "1"}
        gated = piece.variables(cfg, {"challenger": {"sampled-proofs": {**loop, "gate": "user@gate-1"}}})
        assert gated["PP_CHALLENGER"] == "gate:user@gate-1"
        assert "DEVELOPMENT" not in gated and "PP_DEVELOPMENT_UNDER_HOLD" not in gated


def test_verity_run_starts_it_first_binds_what_it_wrote_and_stops_it_at_the_end(stand_ins, capsys):
    state, _, _ = stand_ins
    cfg = R.load(CONFIG, REG)
    runner = R.InProcess(cfg, REG, state, True)
    assert R.execute(cfg, REG, state, [T.CHALLENGER], runner, [], ["gate-host"], [T.CHALLENGER]) == 0
    out = capsys.readouterr().out.splitlines()
    started = [json.loads(x.split(" ", 1)[1]) for x in out if x.startswith("STARTED ")]
    assert started == [{"party": T.CHALLENGER, "writes": {"pearl-plus": ["gate", "key-sha512", "lane", "main-file", "peer"],
                                                          "sampled-proofs": ["loopback", "main-file"]}}]
    assert out[-1] == 'PARTY {"exit": 0, "party": "challenger"}'
    bound = json.loads((state / R.BINDINGS).read_text())[T.CHALLENGER]
    assert bound["sampled-proofs"]["loopback"] == ADDRESS and bound["pearl-plus"]["key-sha512"] == KEY
    assert Host.made[0].stopped
    transcript = [json.loads(x)["file"] for x in (state / "transcripts" / f"{T.CHALLENGER}.jsonl").read_text().splitlines()]
    assert {"public/challenger/bindings.json", "public/challenger/main.txt"} <= set(transcript)


def test_a_dry_run_counts_what_the_challenger_writes_as_bound(capsys):
    assert R.run_main([str(CONFIG), "--dry-run"], REG) == 0
    out = capsys.readouterr().out.splitlines()
    plan = {d["party"]: d for d in (json.loads(x.split(" ", 1)[1]) for x in out if x.startswith("PLAN "))}
    assert plan[T.CHALLENGER]["writes"] == {"sampled-proofs": ["loopback", "main-file"],
                                            "pearl-plus": ["gate", "peer", "lane", "key-sha512", "main-file"]}
    dry = json.loads(next(x for x in out if x.startswith("DRY ")).split(" ", 1)[1])
    assert not [k for k in dry["unbound"] if k.startswith(f"{T.CHALLENGER}.")]


def test_the_auditors_own_gate_runs_the_challenger_where_it_says_and_half_of_it_is_refused(tmp_path, capsys):
    auditors = tmp_path / "auditors.toml"
    auditors.write_text(f'[challenger.pearl-plus]\ngate = "gate-command"\npeer = "peer-1"\nkey-sha512 = "{KEY}"\n'
                        '[challenger.sampled-proofs]\ngate = "gate-command"\n')
    assert R.run_main([str(CONFIG), "--dry-run", "--bindings", str(auditors)], REG) == 0
    assert R.starting(REG, [T.CHALLENGER], tomllib.loads(auditors.read_text())) == []
    capsys.readouterr()
    with pytest.raises(Refused, match="challenger.pearl-plus.gate of what challenger writes, not all of it"):
        R.starting(REG, [T.CHALLENGER], {T.CHALLENGER: {"pearl-plus": {"gate": "gate-command"}}})


def test_it_says_before_each_long_build_what_is_being_built_and_that_it_takes_minutes(stand_ins, monkeypatch, capsys):
    """One line before the Lean verifier's build and Pearl+'s gate start, one before the loopback host's key is made at the
    deployed height: each printed before the work it says, not after."""
    state, calls, _ = stand_ins
    seen = []
    run, Made = T.subprocess.run, T.CP.Host

    def run_say(argv, **kw):
        if len(argv) > 2 and argv[2] == "init":
            seen.append(("gate", capsys.readouterr().err))
        return run(argv, **kw)

    class HostSay(Made):
        def __init__(self, *a, **kw):
            seen.append(("host", capsys.readouterr().err))
            super().__init__(*a, **kw)

    monkeypatch.setattr(T.subprocess, "run", run_say)
    monkeypatch.setattr(T.CP, "Host", HostSay)
    T.STARTS[T.CHALLENGER].serve(R.load(CONFIG, REG), state)
    gate, host = dict(seen)["gate"], dict(seen)["host"]
    assert "building the Lean verifier and Pearl+'s gate" in gate and "takes minutes" in gate and "loopback" not in gate
    assert "making the loopback host's key at the deployed height" in host and "takes minutes" in host
    assert [w for w, _ in seen] == ["gate", "host"]
