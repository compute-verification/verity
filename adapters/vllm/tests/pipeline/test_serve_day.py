"""serve-openai's day (``--day``, `serve_day`): before the model loads, the server declares the day's one circuit in the
deployment's registration log under core's own check, so a second declaration of the day is refused by core's name before
anything serves. Without ``--day`` nothing is declared. What ``--h1-window-commit`` does with the day (the window log
registers the declaration first, each W names it, an R of another program is refused at its close, and without ``--day``
every byte is as before) is in `test_serve_tie`, with the windows it serves."""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from verity.hardware import site as ST
from verity.hardware import registration as G
from verity_vllm.pipeline import cli
from verity_vllm.pipeline import serve_day as SD
from verity_vllm.pipeline import serve_h1 as SH
from verity_vllm.pipeline import serve_openai as so
from verity_vllm.pipeline import serve_tie as TIE

#: the day's circuit's Program, and another
PROGRAM = "5a" * 64
OTHER = "0" * 128


def _params(tmp: Path) -> Path:
    p = tmp / "params.json"
    p.write_text(json.dumps(G.params("onsite", ST.Auditor())))
    return p


def _opts(tmp: Path, *day: str) -> object:
    return cli.parse("serve-openai", ["--model", "m", "--revision", "r", "--out", str(tmp / "serve"), "--no-commit", *day],
                     environ={})


def _day(tmp: Path, day: int, program: str, *more: str) -> list[str]:
    return ["--day", str(day), "--day-log", str(tmp / "log.json"), "--day-program", program, "--day-params",
            str(_params(tmp)), *more]


def test_without_day_nothing_is_declared(tmp_path):
    """The default: no day option, nothing declared or written."""
    opts = _opts(tmp_path)
    assert opts.day == SD.DayOptions() and SD.refusal(opts) is None
    assert SD.declare(opts, tmp_path) is None and list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("argv,why", [
    (["--day-log", "l.json"], "--day-log without --day"),
    (["--day-program", OTHER, "--day-blocks", "a"], "--day-program, --day-blocks without --day"),
    (["--day", "-1", "--day-log", "l", "--day-program", OTHER], "a day is its index"),
    (["--day", "0", "--day-program", OTHER], "--day needs --day-log"),
    (["--day", "0", "--day-log", "l", "--day-program", "AB" * 64], "not a SHA-512 digest in lowercase hex"),
], ids=["log-alone", "program-alone", "negative", "no-log", "upper-hex"])
def test_day_options_that_cant_declare_are_refused_before_the_model_loads(tmp_path, argv, why):
    with pytest.raises(cli.UsageError, match=why):
        so.preflight(_opts(tmp_path, *argv), which=lambda _: None)


def test_serve_openai_declares_the_day_in_the_registration_log_before_the_model_loads(tmp_path, monkeypatch):
    """`serve_openai.main` appends the day's declaration to ``--day-log`` and writes ``declaration.json`` before it builds
    the engine: by then the log holds the declaration as core's check accepts it, chained from the composite's genesis."""
    seen = {}

    def build(opts, session=None, row=None):
        seen["log"] = G.log_of(json.loads((tmp_path / "log.json").read_text()))
        seen["file"] = json.loads((tmp_path / "serve" / SD.FILE).read_text())
        raise RuntimeError("the model would load here")
    monkeypatch.setattr(so, "build", build)
    with pytest.raises(RuntimeError, match="the model would load here"):
        so.main(_opts(tmp_path, *_day(tmp_path, 2, PROGRAM, "--day-blocks", "t0, t1")))
    composite, log = seen["log"]
    entry = G.Declaration(2, PROGRAM, ("t0", "t1"), ST.genesis("onsite"))
    assert composite == "onsite" and log == [entry]
    assert G.check(ST.Auditor(), composite, log).view().declared(2) == dataclasses.replace(entry, previous="")
    assert seen["file"] == {"day": 2, "program": PROGRAM, "blocks": ["t0", "t1"],
                            "declaration": G.Declaration(2, PROGRAM, ("t0", "t1")).digest(),
                            "log": str(tmp_path / "log.json"), "head": entry.digest(), "declared": "now", "file": SD.FILE,
                            "entry": entry.to_json()}


def test_a_window_names_the_declaration_by_its_digest_with_no_previous_not_the_logs_chained_one(tmp_path):
    """W names the day's declaration by `registration.Declaration`'s digest rebuilt from its day, program and blocks with
    no ``previous`` (`named`), the digest R2 rebuilds from the challenger device's entry too; the log's chained entry,
    whose digest is the log's head, is not what a window names."""
    day = SD.declare(_opts(tmp_path, *_day(tmp_path, 3, PROGRAM, "--day-blocks", "t0")), tmp_path)
    _composite, (entry,) = G.log_of(json.loads((tmp_path / "log.json").read_text()))
    rebuilt = G.Declaration(3, PROGRAM, ("t0",)).digest()
    assert day.naming() == {"day": 3, "declaration": rebuilt} == {"day": 3, "declaration": SD.named(entry)}
    assert entry.previous and entry.digest() == day.head != rebuilt


def test_a_second_declaration_of_the_day_is_refused_by_name_and_leaves_the_log_as_it_was(tmp_path):
    """Day 0 declared, then declared again to the same circuit (a server restarted within its day: the declaration is
    read again, the log unchanged), then to another: core's check refuses it (``duplicate-declaration``), before anything
    is written. Another day takes its own declaration, chained after the first."""
    log = tmp_path / "log.json"
    first = SD.declare(_opts(tmp_path, *_day(tmp_path, 0, PROGRAM)), tmp_path)
    assert first.declared == "now" and first.naming() == {"day": 0, "declaration": first.digest}
    body = log.read_bytes()
    again = SD.declare(_opts(tmp_path, *_day(tmp_path, 0, PROGRAM)), tmp_path)
    assert (again.declared, again.digest, again.head) == ("earlier", first.digest, first.head) and log.read_bytes() == body
    (tmp_path / "other").mkdir()
    with pytest.raises(cli.UsageError, match="refuses its declaration, entry 1: duplicate-declaration: day 0 is declared "
                                             "already, to another circuit"):
        SD.declare(_opts(tmp_path, *_day(tmp_path, 0, OTHER)), tmp_path / "other")
    assert log.read_bytes() == body and list((tmp_path / "other").iterdir()) == []
    second = SD.declare(_opts(tmp_path, *_day(tmp_path, 1, OTHER)), tmp_path)
    composite, entries = G.log_of(json.loads(log.read_text()))
    assert second.declared == "now" and entries[1].previous == first.head and second.digest == SD.named(entries[1])
    view = G.check(ST.Auditor(), composite, entries).view()
    assert (view.declared(0).program, view.declared(1).program) == (PROGRAM, OTHER)
    wrong = tmp_path / "remote.json"
    wrong.write_text(json.dumps(G.params("remote", ST.Auditor())))
    with pytest.raises(cli.UsageError, match="the log is onsite's, not --day-params' remote's"):
        SD.declare(_opts(tmp_path, *_day(tmp_path, 2, OTHER)[:-1], str(wrong)), tmp_path)
    log.unlink()
    with pytest.raises(cli.UsageError, match="a new log takes its composite from --day-params"):
        SD.declare(_opts(tmp_path, "--day", "0", "--day-log", str(log), "--day-program", PROGRAM), tmp_path)


def test_the_day_reaches_the_tie_through_serve_h1s_arm(tmp_path, monkeypatch):
    """`serve_h1.arm_of` hands the day to the tie it makes for ``--h1-window-commit``, and none without one."""
    made = []
    monkeypatch.setattr(TIE, "WindowTie", lambda *a, **kw: made.append(kw.get("day")) or SimpleNamespace())
    monkeypatch.setattr(SH, "sessions_of", lambda opts: None)
    monkeypatch.setattr(SH, "H1Windows", lambda *a, **kw: SimpleNamespace(ready=lambda: None, tie=kw["tie"]))
    monkeypatch.setattr(SH, "worker", lambda name: None)
    monkeypatch.setattr(SH, "config_of", lambda c: None)
    gate = SimpleNamespace(h1_window_commit="cmd", h1_model=None, h1_prover=None, h1_receipts=False, h1_weights_root=None)
    opts = SimpleNamespace(protocols="pouw:x", gate=gate, max_num_batched_tokens=1, window_steps=1, protocols_config=None)
    llm = SimpleNamespace(model=None, vllm=None)
    SH.arm_of(opts, llm, None, tmp_path, day="the day")
    SH.arm_of(opts, llm, None, tmp_path)
    assert made == ["the day", None]
