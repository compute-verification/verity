"""serve-openai's PoUW H1 arm (`pipeline.serve_h1`, ``--protocols pouw:eps1-sm120-h1fx1``): each Commit window's H1 window
taps the server's own rows-sha512 Commit (its step-input port and the bench row's int8 source) and ties at ROOTS1, on the
epoch's in-process coins, which credit nothing, or with ``--challenger-gate`` on a session of its own on the auditor host's
challenger, which the auditor's ledger opens and ends for it; `CommitWindows` opens the arm once the committer is
installed and closes it before the committer comes off, its outcome the window's line; and the flags are refused before
the model loads where the server can't serve them."""
from __future__ import annotations

import functools
import json
import shlex
from concurrent.futures import Future
from datetime import datetime, UTC
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from verity.core.protocols.sampled_proofs.challenger import CoinError
from verity_vllm.engine import handle
from verity_vllm.pipeline import cli
from verity_vllm.pipeline import commit_coins as CC
from verity_vllm.pipeline import serve_h1 as SH
from verity_vllm.pipeline import serve_openai as so
from verity_vllm.protocols import pouw
from verity_vllm.protocols.pouw import h1 as PH
from verity_vllm.protocols.pouw.h1 import challenger as PHC

from tests.pipeline import predicted
from tests.pipeline.test_commit_coins import GATE, KEY, PEER, _Auditor
from tests.pipeline.test_serve_openai import CERTIFIER_BODY, Serving, records, serve_concurrently
from tests.protocols import challenger_host as CH
from tests.protocols.test_pouw_h1 import Runner, _so
from tests.protocols.test_pouw_h1_commit import K, ROWS, STEPS, Committer, Linear, Model, _ids, _manifest_sites, _served_commit
from tests.protocols.test_pouw_h1_served import _unit

#: each window's two requests: step 0 admits both (prompt tokens, cap), step 1 decodes a row of each
PROMPTS = ({"a": [5, 6, 7], "b": [8, 9]}, {"c": [11, 12], "d": [13, 14, 15, 16]})


class _Server:
    """The server's Commit as the arm sees it: the committer (`test_pouw_h1_commit.Committer`) with its step-input port
    and the bench row's int8 source at engine steps 0..S-1 of each window (the committer's steps restart at its reset),
    the stand-in model, and the engine's runner, which `handle.model_runner` hands H1."""

    def __init__(self, monkeypatch, quant=None):
        monkeypatch.setattr(pouw, "linear_types", lambda: (Linear,))
        self.runner, self.model, self.com = Runner(), Model(), Committer()
        monkeypatch.setattr(handle, "model_runner", lambda llm: self.runner)
        self.port, self.src = _served_commit(self.com, self.model, _manifest_sites(range(STEPS)) if quant is None else quant)
        self.src.install()

    def serve(self, prompts, seed):
        """One window's two engine steps through the model, the Commit acquiring each as serve-openai's does."""
        g, run = torch.Generator().manual_seed(seed), pouw.h1_run(SH.ENTRY)
        for s, sched in enumerate((_so(*((q, p, 4) for q, p in prompts.items())), _so(running=dict.fromkeys(prompts, 1)))):
            self.runner.execute_model(sched)
            self.com._cur = SimpleNamespace(step=s)
            self.src.on_step_begin()
            got = _ids(run, s)
            self.com._plan_shape = (len(got), len(prompts))
            self.port._pre(None, (torch.tensor(got), torch.arange(len(got))), {})
            for layer in self.model.model.layers:
                layer((torch.randn(len(run.owners[s]), K, generator=g) * 3).bfloat16())
            self.com._cur = None
        return run


def test_each_commit_window_is_an_h1_window_tied_to_the_servers_commit_on_coins_that_credit_nothing(tmp_path, monkeypatch):
    """Two Commit windows, each one H1 window over the committer's steps 0 and 1: each window's A tables and input ids are
    the rows the server's Commit acquired (ROOTS1 tied, none of the Commit's own trees), under the period's one Program and
    weights' root, on the epoch's in-process coins (a fresh epoch a window), so no row is gate-passed; at each close H1's
    hooks and its tap come off, leaving the committer as it was. A window that served no step is neither tied nor not."""
    srv = _Server(monkeypatch)
    arm = SH.H1Windows(srv.model, srv.com, ROWS, STEPS, engine=object(), out=tmp_path, unit=_unit())
    described = arm.describe()
    assert {k: described[k] for k in ("entry", "rows", "steps", "coins", "linears")} == \
        {"entry": SH.ENTRY, "rows": ROWS, "steps": STEPS, "coins": "in-process", "linears": 2}
    salts = []
    for w, prompts in enumerate(PROMPTS):
        arm.open(w)
        assert "_acquire" in vars(srv.com) and "execute_model" in vars(srv.runner)
        run = srv.serve(prompts, w)
        salts.append(run.salt)
        out = arm.close(w)
        assert out["tied"] and out["roots1"] == {"tied": True, "reason": None, "values": 4, "own_trees": 0,
                                                 "steps": {"0": 0, "1": 1}}, out
        assert (out["window"], out["steps_served"], out["coin_source"], out["refused"]) == (w, STEPS, "in-process", [])
        assert out["weights_root"] == described["weights_root"] and run.served.program_digest == described["program"]
        assert out["gate"]["coins"] == "in-process" and out["gate"]["gate_passed"]["rows"] == 0
        assert (tmp_path / "h1" / f"w{w}" / PH.CALLS_FILE).is_file()
        assert "_acquire" not in vars(srv.com) and "execute_model" not in vars(srv.runner)
        assert arm.close(w) is None
    assert salts[0] != salts[1]
    arm.open(2)
    idle = arm.close(2)
    assert (idle["tied"], idle["steps_served"], idle["gate"]["gate_passed"]["rows"]) == (None, 0, 0), "it served no step"
    assert arm.summary() == {"windows": 3, "tied": 2, "untied": [], "gate_passed_rows": 0, "releasing": 0, "unused": []}
    srv.src.uninstall()


class _GateAuditor(_Auditor):
    """`test_commit_coins._Auditor` answering ``status`` as infra's gate does, with the session's open and end entries;
    `settle` (the arm's wait between polls) ends each session whose client has closed it, as serve.py's exit does."""

    def run(self, argv: list[str]) -> tuple[int, str]:
        words = shlex.split(argv[-1])
        if words[0] != "status":
            return super().run(argv)
        self.argv.append(argv)
        entries = {e["kind"]: e for e in map(json.loads, self.gate.lines) if e["session"] == words[1]}
        return 0, json.dumps({"session": words[1], "commitment": self.sessions[words[1]].get("commitment"),
                              "open": entries.get("open"), "end": entries.get("end")})

    def settle(self, _seconds: float) -> None:
        for sid, s in self.sessions.items():
            if "thread" in s and not s.get("ended"):
                self.end(sid)
                s["ended"] = True


def test_each_window_draws_its_coins_from_a_gate_session_of_its_own(tmp_path, monkeypatch):
    """``--challenger-gate`` and ``--challenger-peer``: each window reserves a session of its own (H1's scheme and
    families for S rounds, as this peer), round 0 opens it on the window's open root once its port listens, and the
    window's close waits for its end entry, closed and complete. The window's salt, round and draw coins are that
    session's, its record carries the session without its secret, and the verifier's check of the record against the
    auditor's ledger (`pouw_h1_challenger.session_reason`) passes."""
    srv, auditor = _Server(monkeypatch), _GateAuditor(tmp_path, monkeypatch)
    ports: list[int] = []
    sessions = SH.GateSessions(CC.Gate(GATE, auditor.run), PEER, STEPS, wait=auditor.settle,
                               listens=lambda port: ports.append(port) or True)
    arm = SH.H1Windows(srv.model, srv.com, ROWS, STEPS, engine=object(), out=tmp_path, unit=_unit(), sessions=sessions)
    assert (arm.describe()["coins"], arm.describe()["peer"]) == ("challenger", PEER)
    pins, main = [CH.write_pins(tmp_path / "pins.jsonl")], list(CH.MAIN)
    sids = []
    for w, prompts in enumerate(PROMPTS):
        arm.open(w)
        run = srv.serve(prompts, w)
        out = arm.close(w)
        sid = out["session"]["session"]
        sids.append(sid)
        assert out["tied"] and out["coin_source"] == PH.CHALLENGER and out["gate"]["gate_passed"]["rows"] > 0, out
        assert {k: out["session"][k] for k in ("window", "peer", "port", "listen_polls", "unopened", "closed", "end")} == {
            "window": w, "peer": PEER, "port": 7811, "listen_polls": 0, "unopened": None, "closed": {"closed": True},
            "end": {"opened": True, "rc": 0, "closed": True, "complete": True}}
        record = json.loads((tmp_path / "h1" / f"w{w}" / PH.CALLS_FILE).read_text())
        given = record["commitment"]["session"]
        assert given["session"] == sid and "secret" not in given
        assert record["commitment"]["coins"] == record["commitment"]["round_coins"]
        ledger = PHC.Ledger.load(auditor.heads(), KEY, pins=pins, main=main)
        assert PHC.session_reason(record["commitment"], ledger, lambda s: tmp_path / s if s == sid else None) is None
        assert run.details["challenger"] == {"closed": True}
    assert len(set(sids)) == 2 and ports == [7811, 7811]
    reserves = [shlex.split(a[-1]) for a in auditor.argv if shlex.split(a[-1])[0] == "reserve"]
    assert reserves == [["reserve", "--peer", PEER, "--scheme", PHC.SCHEME, "--families", PHC.families(STEPS)]] * 2
    srv.src.uninstall()


def test_a_window_whose_reserve_the_gate_refuses_installs_nothing(tmp_path, monkeypatch):
    """The gate refuses the window's reserve: its open raises the refusal before any tap or hook goes on, so the Commit
    serves on and the window's line carries the error."""
    srv = _Server(monkeypatch)
    sessions = SH.GateSessions(CC.Gate(GATE, lambda argv: (3, json.dumps({"refused": "--peer 'x': one of []"}))), PEER,
                               STEPS, wait=lambda s: pytest.fail("no poll"), listens=lambda p: pytest.fail("no port"))
    arm = SH.H1Windows(srv.model, srv.com, ROWS, STEPS, engine=object(), out=tmp_path, unit=_unit(), sessions=sessions)
    with pytest.raises(CoinError, match="the challenger gate refused reserve"):
        arm.open(0)
    assert "_acquire" not in vars(srv.com) and "execute_model" not in vars(srv.runner) and arm.close(0) is None
    srv.src.uninstall()


def test_a_commit_that_skips_a_linear_at_a_step_opens_no_h1_window(tmp_path, monkeypatch):
    """The bench row's manifest names no int8 site of one linear at the window's last step: the arm refuses the window
    before it installs anything, so the Commit serves on untouched, and the next window may open."""
    sites = _manifest_sites(range(STEPS))
    srv = _Server(monkeypatch, {k: v for k, v in sites.items() if k != (STEPS - 1, "model.layers.1")})
    arm = SH.H1Windows(srv.model, srv.com, ROWS, STEPS, engine=object(), out=tmp_path, unit=_unit())
    with pytest.raises(ValueError, match=r"acquires 1 of H1's \(engine step, linear\) sites nowhere"):
        arm.open(0)
    assert "_acquire" not in vars(srv.com) and "execute_model" not in vars(srv.runner) and arm.close(0) is None
    srv.src.uninstall()


class Later:
    """`serve_h1.worker` as the tests drive it: what it is handed waits, in turn, until `run` (``shutdown`` too), and
    ``inside`` says whether one of its jobs is running now."""

    def __init__(self) -> None:
        self.jobs: list[tuple[Future, Any]] = []
        self.inside = False

    def submit(self, fn: Any, *args: Any) -> Future:
        self.jobs.append((Future(), functools.partial(fn, *args)))
        return self.jobs[-1][0]

    def run(self) -> None:
        while self.jobs:
            fut, job = self.jobs.pop(0)
            self.inside = True
            try:
                got = job()
            except (ValueError, RuntimeError, OSError) as e:
                self.inside = False
                fut.set_exception(e)
            else:
                self.inside = False
                fut.set_result(got)

    def shutdown(self, wait: bool = True) -> None:
        self.run()


class _InOrderAuditor(_GateAuditor):
    """`_GateAuditor` whose wait between polls ends only the oldest session still unended, the one whose window is
    releasing (windows release in order), so a later window's session stays open meanwhile."""

    def settle(self, _seconds: float) -> None:
        sid = next((sid for sid, s in self.sessions.items() if "thread" in s and not s.get("ended")), None)
        if sid is not None:
            self.end(sid)
            self.sessions[sid]["ended"] = True


def test_reserves_and_releases_run_off_the_loop_while_the_next_window_serves(tmp_path, monkeypatch):
    """With ``ahead`` and ``closer``: window 0's session is reserved at its open (after any period the run opened) and
    window k+1's while window k serves, and window k's release (its manifest, draw coin, session close and the wait for
    its end entry) runs after window k+1 has opened and served on a session of its own. The loop's gate calls are window
    0's reserve and each window's open; each window keeps its own session, coins and transcript, and the outcomes are
    recorded in window order."""
    srv, auditor = _Server(monkeypatch), _InOrderAuditor(tmp_path, monkeypatch)
    ahead, closer = Later(), Later()
    calls: list[tuple[str, bool]] = []

    def gate(argv: list[str]) -> tuple[int, str]:
        calls.append((shlex.split(argv[-1])[0], ahead.inside or closer.inside))
        return auditor.run(argv)
    sessions = SH.GateSessions(CC.Gate(GATE, gate), PEER, STEPS, wait=auditor.settle, listens=lambda port: True,
                               ahead=ahead, now=lambda: datetime(2026, 10, 20, 11, 8, tzinfo=UTC))
    arm = SH.H1Windows(srv.model, srv.com, ROWS, STEPS, engine=object(), out=tmp_path, unit=_unit(), sessions=sessions,
                       closer=closer)
    assert (arm.describe()["reserve"], arm.describe()["release"]) == ("ahead", "off the loop")
    assert not ahead.jobs and not calls, "nothing is reserved before window 0 opens"
    pending, runs = [], []
    for w, prompts in enumerate(PROMPTS):
        ahead.run()
        arm.open(w)
        runs.append(srv.serve(prompts, w))
        pending.append(arm.close(w))
        assert not pending[-1].done() and not (tmp_path / "h1" / f"w{w}" / PH.CALLS_FILE).exists()
        assert arm.summary()["releasing"] == w + 1 and arm.outcomes == []
    assert "execute_model" not in vars(srv.runner) and "_acquire" not in vars(srv.com), "both windows' hooks are off"
    closer.run()
    outs = [f.result() for f in pending]
    assert [o["window"] for o in outs] == [o["window"] for o in arm.outcomes] == [0, 1]
    for w, (out, run) in enumerate(zip(outs, runs)):
        assert "error" not in out and out["tied"] and out["coin_source"] == PH.CHALLENGER, out
        assert out["session"]["end"] == {"opened": True, "rc": 0, "closed": True, "complete": True}
        assert out["session"]["window"] == w and run.details["challenger"] == {"closed": True}
        record = json.loads((tmp_path / "h1" / f"w{w}" / PH.CALLS_FILE).read_text())
        assert record["commitment"]["session"]["session"] == out["session"]["session"]
        assert set(out["seconds"]) == {"reserve", "install", "close_on_loop", "queued", "release", "ended"}
    assert outs[0]["session"]["session"] != outs[1]["session"]["session"]
    arm.drain()
    assert [verb for verb, off in calls if not off] == ["reserve"] + ["open", "status"] * 2, \
        "window 0's reserve, then round 0's open and its commitment by status"
    assert [verb for verb, _off in calls].count("reserve") == 3, "window 2's, reserved ahead, is never opened"
    assert arm.summary()["unused"] == [sessions.held[-1]["session"]] and sessions.held[-1]["window"] == 2
    srv.src.uninstall()


def test_a_reserve_ahead_that_expires_before_its_window_is_replaced_and_none_is_made_ahead_in_a_period(tmp_path,
                                                                                                    monkeypatch):
    """A session reserved ahead that expires within `MARGIN` of its window's open is left unused and the window reserves
    another; a reserve the gate answers inside a period makes none ahead, since one no window opened would leave the
    period incomplete."""
    answers = iter([{"expires_utc": "2026-10-27T20:04:00Z", "period": None},
                    {"expires_utc": "2026-10-27T22:00:00Z", "period": "p1"}])
    seen: list[str] = []

    def gate(argv: list[str]) -> tuple[int, str]:
        seen.append(shlex.split(argv[-1])[0])
        return 0, json.dumps({"session": f"s20261027T2000{len(seen):02d}Z-{len(seen):04x}", "seq": len(seen),
                              "peer": PEER, "by": "vllm", **next(answers)})
    ahead = Later()
    sessions = SH.GateSessions(CC.Gate(GATE, gate), PEER, STEPS, ahead=ahead,
                               now=lambda: datetime(2026, 10, 27, 20, 0, tzinfo=UTC))
    sessions.ahead_of(0)
    ahead.run()
    got = sessions.take(0)
    assert got["period"] == "p1" and sessions.held[0]["unused"].startswith("its reserve expires at 2026-10-27T20:04:00Z")
    assert seen == ["reserve", "reserve"] and not ahead.jobs, "none ahead inside a period"


def test_a_release_that_fails_off_the_loop_is_its_windows_outcome_in_order(tmp_path, monkeypatch):
    """Window 0's release raises off the loop after its session closed: its outcome carries the error, recorded before
    window 1's, which opened, served and released as its own."""
    from verity_vllm.protocols.pouw.h1 import window as PHW

    srv, closer = _Server(monkeypatch), Later()
    finish = PHW.finish

    def failing(run: Any, client: Any) -> None:
        finish(run, client)
        if str(run.transcript_dir).endswith("w0"):
            raise RuntimeError("the draw's record was not written")
    monkeypatch.setattr(PHW, "finish", failing)
    arm = SH.H1Windows(srv.model, srv.com, ROWS, STEPS, engine=object(), out=tmp_path, unit=_unit(), closer=closer)
    pending = []
    for w, prompts in enumerate(PROMPTS):
        arm.open(w)
        srv.serve(prompts, w)
        pending.append(arm.close(w))
    closer.run()
    outs = [f.result() for f in pending]
    assert outs[0]["error"] == "close: RuntimeError: the draw's record was not written" and outs[0]["tied"]
    assert "error" not in outs[1] and outs[1]["tied"]
    assert [o["window"] for o in arm.outcomes] == [0, 1] and arm.summary()["releasing"] == 0
    srv.src.uninstall()


class ArmSpy:
    """`serve_h1.H1Windows` as `CommitWindows` drives it: at each open and close, whether the committer's patches of the
    runner are on; ``fail`` the windows whose open raises."""

    def __init__(self, runner: Any, fail: tuple[int, ...] = ()) -> None:
        self.runner, self.fail, self.seen, self.open_ = runner, fail, [], None

    def open(self, window: int) -> None:
        self.seen.append(("open", window, "execute_model" in vars(self.runner)))
        if window in self.fail:
            raise ValueError("no int8 source")
        self.open_ = window

    def close(self, window: int) -> dict[str, Any] | None:
        self.seen.append(("close", window, "execute_model" in vars(self.runner)))
        got, self.open_ = ({"window": window, "tied": True} if self.open_ == window else None), None
        return got

    def summary(self) -> dict[str, Any]:
        return {"windows": sum(1 for e in self.seen if e[0] == "close")}


@pytest.mark.parametrize("fail", [(), (1,)])
def test_commit_windows_open_the_arm_on_the_installed_committer_and_close_it_first(tmp_path, fail):
    """Every window opens the arm once the committer's patches are on and closes it while they still are, before the
    committer comes off; the arm's outcome is its window's line's ``pouw_h1``, a failed open that window's error while the
    Commit serves on, and the summary counts both."""
    spies: list[ArmSpy] = []

    def arm(runner: Any) -> ArmSpy:
        spies.append(ArmSpy(runner, fail))
        return spies[-1]

    bodies = [dict(CERTIFIER_BODY, prompt=[128000, 10 + i], max_tokens=3, min_tokens=3) for i in range(3)]
    with Serving(tmp_path, window_steps=2, arm=arm) as s:
        serve_concurrently(s, bodies)
    (spy,) = spies
    wins = records(tmp_path, "windows.jsonl")
    assert len(wins) > 1 and all(w["passed"] for w in wins)
    assert spy.seen == [(e, w["window"], True) for w in wins for e in ("open", "close")]
    assert [w["pouw_h1"] for w in wins] == [{"error": "open: ValueError: no int8 source"} if w["window"] in fail
                                            else {"window": w["window"], "tied": True} for w in wins]
    got = s.windows.summary()
    assert got["passed"] and got["pouw_h1"] == {"windows": len(wins), "failed": list(fail), "closing": []}


class LaterArm(ArmSpy):
    """`ArmSpy` whose close hands back a `Future` for its outcome, which the test resolves once serving has ended."""

    def __init__(self, runner: Any) -> None:
        super().__init__(runner)
        self.futures: dict[int, Future] = {}

    def close(self, window: int) -> Future:
        super().close(window)
        self.futures[window] = Future()
        return self.futures[window]


def test_commit_windows_write_a_windows_line_once_its_arms_release_ends(tmp_path):
    """An arm whose release runs off the loop: the loop serves every window without waiting for any release, each
    window's line is held until its arm's outcome comes, the lines are written in window order however the outcomes
    come, and a release that raised is that window's failure, its error the line's ``pouw_h1``."""
    spies: list[LaterArm] = []

    def arm(runner: Any) -> LaterArm:
        spies.append(LaterArm(runner))
        return spies[-1]

    bodies = [dict(CERTIFIER_BODY, prompt=[128000, 10 + i], max_tokens=3, min_tokens=3) for i in range(3)]
    with Serving(tmp_path, window_steps=2, arm=arm) as s:
        serve_concurrently(s, bodies)
    (spy,) = spies
    closed = sorted(spy.futures)
    assert len(closed) > 1 and not (tmp_path / "windows.jsonl").exists()
    held = s.windows.summary()
    assert held["pouw_h1"]["closing"] == closed and held["windows"] == 0 and not held["passed"]
    failed = closed[1]
    for w in closed:
        if w == failed:
            spy.futures[w].set_exception(RuntimeError("its session's end entry never came"))
        else:
            spy.futures[w].set_result({"window": w, "tied": True})
    wins = records(tmp_path, "windows.jsonl")
    assert [w["window"] for w in wins] == closed and all(w["passed"] for w in wins)
    assert [w["pouw_h1"] for w in wins] == [{"error": "close: RuntimeError: its session's end entry never came"}
                                            if w["window"] == failed else {"window": w["window"], "tied": True} for w in wins]
    got = s.windows.summary()
    assert got["passed"] and got["pouw_h1"] == {"windows": len(wins), "failed": [failed], "closing": []}


def test_serve_openai_refuses_protocols_it_cant_serve_before_the_model_loads(tmp_path):
    """Among them a token budget the rows-sha512 Commit's step-input port can't take (whole 512-id rows), with H1 or
    without, so H1's own 32-row rule is met by every T the Commit takes."""
    man = predicted.row(tmp_path)

    def refusal(*more: str) -> str | None:
        opts = cli.parse("serve-openai", ["--model", "m", "--revision", "r", "--required-manifest", str(man), *more],
                         environ={})
        try:
            so.preflight(opts, which=lambda _: "/v/bin/ninja")
        except cli.UsageError as e:
            return str(e)
        return None

    h1 = ("--protocols", SH.ENTRY)
    assert refusal(*h1) is None
    assert refusal(*h1, "--protocols-config", json.dumps({"pouw": {"admission": "per-round"}})) is None
    (cfg := tmp_path / "pouw.json").write_text(json.dumps({"pouw": {"retain_steps": 2}}))
    assert refusal(*h1, "--protocols-config", str(cfg)) is None
    assert "PoUW H1) only" in refusal("--protocols", "pouw:ncp-v2")
    assert "--commit, --scheme rows-sha512" in refusal(*h1, "--no-commit")
    for t in ("2050", "2016"):
        assert refusal(*h1, "--max-num-batched-tokens", t) == refusal("--max-num-batched-tokens", t), t
        assert f"--max-num-batched-tokens a multiple of 512, not {t}" in refusal("--max-num-batched-tokens", t)
    assert refusal(*h1, "--max-num-batched-tokens", "1536") is None
    assert "multiple of 32" in SH.refusal(cli.parse("serve-openai", ["--model", "m", "--revision", "r", "--required-manifest",
                                                                     str(man), *h1, "--max-num-batched-tokens", "2050"], environ={}))
    assert refusal("--protocols-config", "{}") == "--protocols-config without --protocols"
    assert "the server's own" in refusal(*h1, "--protocols-config", json.dumps({"pouw": {"rows": 64}}))
    assert "one gate session a window" in refusal(*h1, "--protocols-config", json.dumps({"pouw": {"challenger": "h:1"}}))
    gate = ("--challenger-gate", GATE, "--challenger-peer", PEER)
    assert refusal(*h1, *gate) is None
    assert "go together" in refusal(*h1, *gate[:2]) and "go together" in refusal(*h1, *gate[2:])
    assert "needs --protocols" in refusal(*gate)
    assert "(pouw)" in refusal(*h1, "--protocols-config", json.dumps({"pous": {}}))
    assert "'admission' is one of" in refusal(*h1, "--protocols-config", json.dumps({"pouw": {"admission": "never"}}))
    assert "--protocols-config {" in refusal(*h1, "--protocols-config", "{not json")


def test_the_window_program_holds_a_tile_unit_for_each_32_rows_of_slot_and_32_columns_of_each_linear_at_each_step():
    """Llama-3.2-1B's int8 linears (16 layers of qkv, o, gate_up and down) at T 2048 and S 1: 64 row tiles of 11,776
    column tiles. A T between powers of two takes the next one's slot, and S multiplies through."""
    llama = [(3072, 2048), (2048, 2048), (16384, 2048), (2048, 8192)] * 16
    assert SH.units(llama, 2048, 1) == 64 * 11_776 == 753_664
    assert SH.units(llama, 1536, 1) == SH.units(llama, 2048, 1)
    assert SH.units(llama, 512, 16) == 16 * 16 * 11_776
