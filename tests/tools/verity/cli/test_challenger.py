"""The challenger as the party `verity run` starts first (`served_table.STARTS`): its hosts on the run's own machine,
each under a key made for the run, write SPOT's and Pearl+'s bindings, which SPOT's piece hands the runner; it is
stopped when the run ends. Its hosts and Pearl+'s gate are stand-ins written by each test."""
from __future__ import annotations

import json
import subprocess
import sys
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
    state, calls, gate = tmp_path / "state", [], {"rc": 0, "pin": LINE[0]}
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
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(T.subprocess, "run", run)
    monkeypatch.setattr(T.CP, "Host", Host)
    return state, calls, gate


def test_the_challenger_is_the_party_the_table_starts_first_and_writes_spots_and_pearl_plus_bindings():
    assert set(T.STARTS) == {T.CHALLENGER}
    assert {c: list(k) for c, k in T.STARTS[T.CHALLENGER].writes.items()} == {
        "sampled-proofs": ["loopback", "main-file"], "pearl-plus": ["gate", "peer", "lane", "key-sha512", "main-file"]}
    assert (ROOT / T.PEARLPLUS_GATE).is_file()


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
    assert [(h.root, h.main, h.local) for h in Host.made] == [(mine / "sampled-proofs", LINE, T.LC.Local)]
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


def test_spots_piece_hands_the_runner_the_loopback_host_in_a_development_run_and_a_gate_before_it():
    cfg, piece = R.load(CONFIG, REG), T.PARTIES["sampled-proofs"][T.WORKERS]
    loop = {"loopback": ADDRESS, "main-file": "/s/public/challenger/main.txt"}
    assert piece.variables(cfg, {"challenger": {"sampled-proofs": loop}}) == {
        "PP_CHALLENGER": f"loopback:{ADDRESS}", "MAIN_FILE": "/s/public/challenger/main.txt", "DEVELOPMENT": "1"}
    gated = piece.variables(cfg, {"challenger": {"sampled-proofs": {**loop, "gate": "user@gate-1"}}})
    assert gated["PP_CHALLENGER"] == "gate:user@gate-1" and "DEVELOPMENT" not in gated
    assert piece.variables(cfg, {}) == {"DEVELOPMENT": "1"}


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


def test_a_dry_run_counts_what_the_challenger_writes_as_bound_and_leaves_pouws_gate_unbound(capsys):
    assert R.run_main([str(CONFIG), "--dry-run"], REG) == 0
    out = capsys.readouterr().out.splitlines()
    plan = {d["party"]: d for d in (json.loads(x.split(" ", 1)[1]) for x in out if x.startswith("PLAN "))}
    assert plan[T.CHALLENGER]["writes"] == {"sampled-proofs": ["loopback", "main-file"],
                                            "pearl-plus": ["gate", "peer", "lane", "key-sha512", "main-file"]}
    dry = json.loads(next(x for x in out if x.startswith("DRY ")).split(" ", 1)[1])
    assert {"challenger.pouw.gate", "challenger.pouw.peer", "challenger.pouw.key-sha512"} <= set(dry["unbound"])
