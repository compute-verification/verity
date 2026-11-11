"""serve-openai's tie of each PoUW H1 window to the audited window circuit (`serve_tie`, ``--h1-window-commit``): at the
window's close, on its release thread and before its sample coin, the window's registration W (its tables' entries, chained
after the log's last record) is written, the command (V3c's window-commit, a stand-in here) writes R reusing W's tables, and
R is the window's Commit record, bound into calls_root before draw_root and the coin. The release waits for the window's
seal line; a window whose seal failed or never came, whose release runs on the loop, or whose command fails or writes no
R that reuses W binds no record and opens no coin."""
from __future__ import annotations

import json
import shlex
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest

from verity.protocols.spot.verifier.one_stage import registration as REG
from verity.circuits.codec import canonical_json
from verity_vllm.pipeline import serve_h1 as SH
from verity_vllm.pipeline import serve_tie as TIE
from verity_vllm.protocols.pouw import h1 as PH
from verity_vllm.protocols.pouw.h1 import commit as CT

from tests.pipeline.test_serve_h1 import PROMPTS, Later, _Server
from tests.protocols.test_pouw_h1_commit import ROWS, STEPS, _reusing
from tests.protocols.test_pouw_h1_served import _unit

#: what the tie hears of a sealed window (`serve_commit.CommitWindows`' seal line)
SEALED = {"run_id": "t/w", "run_root": "ab" * 64, "row_dir": "windows/w"}


class WindowCommit:
    """V3c's window-commit as the tie runs it (`serve_tie.run_command`'s place): each call's arguments and which of the
    run's calls_root, draw and commit were bound when it ran, then R as `test_pouw_h1_commit._reusing` makes it of W
    (``edit`` the record; none with ``write`` False) and the exit code ``rc``."""

    def __init__(self, runs: dict[int, Any], *, edit: Callable[[dict], dict] = lambda r: r, rc: int = 0, write: bool = True,
                 then: Callable[[], Any] = lambda: None) -> None:
        self.runs, self.edit, self.rc, self.write, self.then, self.calls = runs, edit, rc, write, then, []

    def __call__(self, argv: list[str], log: Any) -> int:
        self.then()
        k, wf, rf = int(argv[5]), Path(argv[6]), Path(argv[7])
        self.calls.append({"argv": argv, "bound": sorted({"calls_root", "draw", "commit"} & set(self.runs[k].commits))})
        log.write("window-commit: committed\n")
        if self.write:
            rf.write_text(json.dumps(self.edit(_reusing(json.loads(wf.read_text()), lambda r: r))))
        return self.rc


def _tied(tmp_path, monkeypatch, closer: Any, **commit: Any) -> tuple[_Server, SH.H1Windows, WindowCommit, dict, list]:
    srv, runs, flushed = _Server(monkeypatch), {}, []
    wc = WindowCommit(runs, **commit)
    tie = TIE.WindowTie("v3c-window-commit", tmp_path / "serve", flush=lambda: flushed.append("flush"), runner=wc)
    arm = SH.H1Windows(srv.model, srv.com, ROWS, STEPS, engine=object(), out=tmp_path / "serve", unit=_unit(),
                       prover=tmp_path / "prover", closer=closer, tie=tie)
    return srv, arm, wc, runs, flushed


def _log(serve: Path) -> list[dict[str, Any]]:
    p = serve / "h1" / TIE.LOG
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


def test_a_tied_window_names_r_reusing_its_tables_from_w_as_its_commit_record_before_its_sample_coin(tmp_path, monkeypatch):
    """Two windows released off the loop once sealed. Each writes W, a v2 record of its tables' entries chained after the
    log's last record (window 0's after none, window 1's after window 0's R), then runs the command, with the serve dir,
    the window, W's and R's files and the window's prover directory, once the advice trace is flushed and before calls_root,
    draw_root or the commit are bound. R is the window's Commit record: its canonical JSON is the commitment's ``commit``,
    the draw coin opens after it, the outcome records both digests, and the log names W then R, so a tie over the same
    serve dir chains after the last R."""
    serve = tmp_path / "serve"
    srv, arm, wc, runs, flushed = _tied(tmp_path, monkeypatch, closer := Later())
    assert arm.describe()["commit_record"] == {"named": "at the close", "log": "h1/log.jsonl",
                                               "window": "h1/w<k>/registration.json", "record": "h1/w<k>/record.json"}
    pending = []
    for w, prompts in enumerate(PROMPTS):
        arm.open(w)
        runs[w] = srv.serve(prompts, w)
        pending.append(arm.close(w))
    for w in range(len(PROMPTS)):
        arm.sealed(w, {"window": w, **SEALED})
    closer.run()
    outs, previous, want = [f.result() for f in pending], None, []
    for w, out in enumerate(outs):
        d = serve / "h1" / f"w{w}"
        reg, rec = (json.loads((d / f).read_text()) for f in (TIE.W_FILE, TIE.R_FILE))
        assert "error" not in out and out["tied"] and out["kept"]["calls"] > 0, out
        assert reg == CT.registration(runs[w], previous) and rec["previous"] == REG.digest(reg)
        assert out["commit_record"] == {"named": True, "registration": REG.digest(reg), "record": REG.digest(rec)}
        record = json.loads((d / PH.CALLS_FILE).read_text())
        c = record["commitment"]
        assert c["commit"] == canonical_json(rec).hex() and c["draw"]["coin"] is not None and "refused" not in c["draw"]
        assert CT.digest(c["commit"]).hex() == REG.digest(rec) and c["calls_root"] == runs[w].calls_root().hex()
        assert wc.calls[w] == {"argv": ["bash", "-c", "v3c-window-commit", "_", str(serve), str(w), str(d / TIE.W_FILE),
                                        str(d / TIE.R_FILE), str(tmp_path / "prover" / f"w{w}")], "bound": []}
        assert (d / TIE.COMMAND_LOG).read_text() == "window-commit: committed\n"
        want += [{"window": w, "kind": "registration", "file": f"w{w}/{TIE.W_FILE}", "digest": REG.digest(reg)},
                 {"window": w, "kind": "record", "file": f"w{w}/{TIE.R_FILE}", "digest": REG.digest(rec)}]
        previous = REG.digest(rec)
    assert flushed == ["flush"] * len(PROMPTS) and _log(serve) == want
    assert TIE.WindowTie("again", serve).last == previous
    arm.drain()
    srv.src.uninstall()


def test_a_tied_release_off_the_loop_waits_for_its_windows_seal(tmp_path, monkeypatch):
    """On the server's own release thread (`serve_h1.worker`), the window's release waits for its seal line, which
    `CommitWindows` hands the arm after the close returns: the trace's flush and the command come after it."""
    events: list[str] = []
    srv, arm, _wc, runs, _flushed = _tied(tmp_path, monkeypatch, SH.worker("h1-release"),
                                          then=lambda: events.append("window-commit"))
    arm.tie.flush = lambda: events.append("flush")
    arm.open(0)
    runs[0] = srv.serve(PROMPTS[0], 0)
    pending = arm.close(0)
    assert isinstance(pending, Future)
    events.append("sealed")
    arm.sealed(0, {"window": 0, **SEALED})
    out = pending.result()
    assert events == ["sealed", "flush", "window-commit"] and out["commit_record"]["named"], out
    arm.drain()
    srv.src.uninstall()


def _seal(arm: SH.H1Windows) -> None:
    arm.sealed(0, {"window": 0, **SEALED})


@pytest.mark.parametrize("act,commit,why,logged", [
    (lambda arm: arm.sealed(0, {"window": 0, "error": "not sealed: below the disk floor"}), {},
     "window 0's seal failed, so window-commit has no row dir: not sealed: below the disk floor", []),
    (lambda arm: arm.tie.stop(), {}, "window 0 has no seal: the server stopped before it was sealed", []),
    (_seal, {"rc": 2}, "--h1-window-commit exited 2: its output is in the serve dir's h1/w0/window-commit.log",
     ["registration"]),
    (_seal, {"write": False}, "--h1-window-commit wrote no record R to h1/w0/record.json", ["registration"]),
    (_seal, {"edit": lambda r: {**r, "previous": None}}, "previous is not the window registration's digest",
     ["registration"]),
], ids=["seal-failed", "server-stopped", "command-failed", "no-r", "r-not-after-w"])
def test_a_tied_window_with_no_r_reusing_w_binds_no_record_and_opens_no_coin(tmp_path, monkeypatch, act, commit, why,
                                                                              logged):
    """No seal (it failed, or the server stopped first), a command that fails or writes no R, or an R that isn't chained
    right after W: the window binds no Commit record and opens no draw coin, its outcome's error names why, and only a W
    written goes into the log, so the next window chains after it. Nothing is retried."""
    serve = tmp_path / "serve"
    srv, arm, wc, runs, _flushed = _tied(tmp_path, monkeypatch, closer := Later(), **commit)
    arm.open(0)
    runs[0] = srv.serve(PROMPTS[0], 0)
    pending = arm.close(0)
    act(arm)
    closer.run()
    out = pending.result()
    assert "the window names no Commit record" in out["error"] and why in out["error"], out["error"]
    assert out["commit_record"]["named"] is False and why in out["commit_record"]["reason"]
    c = json.loads((serve / "h1" / "w0" / PH.CALLS_FILE).read_text())["commitment"]
    assert c["commit"] is None and c["draw"]["coin"] is None
    assert [e["kind"] for e in _log(serve)] == logged and len(wc.calls) == len(logged)
    if logged:
        assert TIE.WindowTie("again", serve).last == REG.digest(json.loads((serve / "h1" / "w0" / TIE.W_FILE).read_text()))
    arm.drain()
    srv.src.uninstall()


def test_a_tied_window_whose_hooks_stay_on_is_released_on_the_loop_and_names_no_record(tmp_path, monkeypatch):
    """A window whose hooks refuse to come off is released on the engine loop, which seals it only afterwards: its tie
    refuses at once rather than wait for that seal, so the loop never blocks, and the window opens no coin."""
    srv, arm, wc, runs, _flushed = _tied(tmp_path, monkeypatch, Later())
    arm.open(0)
    runs[0] = srv.serve(PROMPTS[0], 0)

    def stuck() -> None:
        raise RuntimeError("a hook is still on the engine")
    monkeypatch.setattr(arm.comp, "unhook", stuck)
    out = arm.close(0)
    assert not isinstance(out, Future) and out["error"] == "close: RuntimeError: a hook is still on the engine", out
    assert out["commit_record"]["reason"].startswith("ValueError: window 0 can't wait for its seal: its hooks refused to "
                                                     "come off (RuntimeError: a hook is still on the engine)")
    assert wc.calls == [] and _log(tmp_path / "serve") == []
    arm.drain()
    srv.src.uninstall()


def test_a_tie_needs_its_release_off_the_loop(tmp_path, monkeypatch):
    srv = _Server(monkeypatch)
    with pytest.raises(ValueError, match="its release runs off the loop, on closer"):
        SH.H1Windows(srv.model, srv.com, ROWS, STEPS, engine=object(), out=tmp_path, unit=_unit(),
                     tie=TIE.WindowTie("true", tmp_path))
    srv.src.uninstall()


def _run() -> SimpleNamespace:
    """A window's run as `pouw_h1_commit.registration` reads it."""
    served = SimpleNamespace(program_digest="ab" * 64, root="cd" * 64, refused=[], entries=lambda: [])
    return SimpleNamespace(entry=SH.ENTRY, served=served)


def test_the_command_runs_under_bash_on_the_windows_files_and_a_failure_points_to_its_output(tmp_path):
    """``bash -c CMD _ SERVE_DIR K W_FILE R_FILE PROVER_DIR`` (PROVER_DIR empty without one), its stdout and stderr in
    ``h1/w<k>/window-commit.log``; a command that exits nonzero is refused by its exit code alone, since its output may
    name the window's values."""
    serve, args, r = tmp_path / "serve", tmp_path / "args", {"format": "stand-in"}
    (tmp_path / "r.json").write_text(json.dumps(r))
    tie = TIE.WindowTie(f'printf "%s\\n" "$@" > {shlex.quote(str(args))}; echo "rows of window $2" >&2; '
                        f'cp {shlex.quote(str(tmp_path / "r.json"))} "$4"', serve)
    tie.sealed(3, {"window": 3, **SEALED})
    w, got = tie.record(3, _run(), tmp_path / "prover" / "w3", None)
    d = serve / "h1" / "w3"
    assert got == r and w == CT.registration(_run(), None) == json.loads((d / TIE.W_FILE).read_text())
    assert args.read_text().splitlines() == [str(serve), "3", str(d / TIE.W_FILE), str(d / TIE.R_FILE),
                                             str(tmp_path / "prover" / "w3")]
    assert (d / TIE.COMMAND_LOG).read_text() == "rows of window 3\n"
    failing = TIE.WindowTie('printf "%s|" "$@" >&2; echo "row 7 holds 0x1234" >&2; exit 5', serve)
    assert failing.last == REG.digest(w), "the log's last record is window 3's W: its R was never bound"
    failing.sealed(4, {"window": 4, **SEALED})
    with pytest.raises(ValueError) as e:
        failing.record(4, _run(), None, None)
    d4 = serve / "h1" / "w4"
    assert str(e.value) == "--h1-window-commit exited 5: its output is in the serve dir's h1/w4/window-commit.log"
    assert (d4 / TIE.COMMAND_LOG).read_text() == f"{serve}|4|{d4 / TIE.W_FILE}|{d4 / TIE.R_FILE}||row 7 holds 0x1234\n"
    assert json.loads((d4 / TIE.W_FILE).read_text())["previous"] == REG.digest(w)
