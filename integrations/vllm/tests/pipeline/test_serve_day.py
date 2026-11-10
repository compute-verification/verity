"""serve-openai's day (``--day``, `serve_day`): before the model loads, the server declares the day's one circuit in the
deployment's registration log under core's own check, so a second declaration of the day is refused by core's name before
anything serves; under ``--h1-window-commit`` the window log registers the declaration first, each W names it, and an R of
another program is refused at its close. Without ``--day`` every byte is as before."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from verity.core import site as ST
from verity.core.protocols import registration as G
from verity.core.protocols.spot.one_stage import registration as REG
from verity_vllm.pipeline import cli
from verity_vllm.pipeline import serve_day as SD
from verity_vllm.pipeline import serve_h1 as SH
from verity_vllm.pipeline import serve_openai as so
from verity_vllm.pipeline import serve_tie as TIE
from verity_vllm.protocols.pouw import h1 as PH
from verity_vllm.protocols.pouw.h1 import commit as CT

from tests.pipeline.test_serve_h1 import PROMPTS, Later, _Server
from tests.pipeline.test_serve_tie import SEALED, WindowCommit, _log, _run
from tests.protocols.test_pouw_h1_commit import AUDITED, ROWS, STEPS
from tests.protocols.test_pouw_h1_served import _unit

#: `_run`'s W chained after none and after ``"ef" * 64``, as `pouw_h1_commit.registration` wrote it before ``--day``
DEFAULT_W = {None: "5294483d21133ab067f5437bc5de486f0d4e07571259f9aa64edcc137b02d3c7eed735d83d7f78eb51482fdc675f7cbc491002c8da47"
                   "4a08709ad4a4b140229d",
             "ef" * 64: "0fed5f69a3da222fff14260aa531125c6bf3ab9058e3c0d0fa795b76f733a3ed76088534f0da88503355eb9413858f74fffcae4d"
                        "ec56811b70cbc2dd7cb58412"}
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


def test_without_day_nothing_is_declared_and_w_the_window_log_and_the_tie_are_byte_for_byte_as_before(tmp_path):
    """The default: no day option, nothing declared or written, W's bytes (pinned by digest, as the code before ``--day``
    wrote them) and its window's keys unchanged, the tie's log not made until a window and its description as before."""
    opts = _opts(tmp_path)
    assert opts.day == SD.DayOptions() and SD.refusal(opts) is None
    assert SD.declare(opts, tmp_path) is None and list(tmp_path.iterdir()) == []
    for previous, digest in DEFAULT_W.items():
        w = CT.registration(_run(), previous)
        assert REG.digest(w) == digest and CT.registration(_run(), previous, None) == w
        assert sorted(w["window"]) == ["entry", "kind", "open_root", "refused"] and CT.day_of(w) is None
    tie = TIE.WindowTie("true", tmp_path / "serve")
    assert tie.describe() == {"log": "h1/log.jsonl", "window": "h1/w<k>/registration.json", "record": "h1/w<k>/record.json"}
    assert tie.day is None and tie.last is None and not (tmp_path / "serve").exists()


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
        so.main(_opts(tmp_path, *_day(tmp_path, 2, AUDITED, "--day-blocks", "t0, t1")))
    composite, log = seen["log"]
    entry = G.Declaration(2, AUDITED, ("t0", "t1"), ST.genesis("onsite"))
    assert composite == "onsite" and log == [entry]
    assert G.check(ST.Auditor(), composite, log).view().declared(2) == entry.declared()
    assert seen["file"] == {"day": 2, "program": AUDITED, "blocks": ["t0", "t1"],
                            "declaration": G.Declaration(2, AUDITED, ("t0", "t1")).digest(),
                            "log": str(tmp_path / "log.json"), "head": entry.digest(), "declared": "now", "file": SD.FILE,
                            "entry": entry.to_json()}


def test_a_window_names_the_declaration_by_its_digest_with_no_previous_not_the_logs_chained_one(tmp_path):
    """W names the day's declaration by `registration.Declaration`'s digest rebuilt from its day, program and blocks with
    no ``previous`` (`named`), the digest R2 rebuilds from the challenger device's entry too; the log's chained entry,
    whose digest is the log's head, is not what a window names."""
    day = SD.declare(_opts(tmp_path, *_day(tmp_path, 3, AUDITED, "--day-blocks", "t0")), tmp_path)
    _composite, (entry,) = G.log_of(json.loads((tmp_path / "log.json").read_text()))
    rebuilt = G.Declaration(3, AUDITED, ("t0",)).digest()
    assert day.naming() == {"day": 3, "declaration": rebuilt} == {"day": 3, "declaration": SD.named(entry)}
    assert entry.previous and entry.digest() == day.head != rebuilt


def test_a_second_declaration_of_the_day_is_refused_by_name_and_leaves_the_log_as_it_was(tmp_path):
    """Day 0 declared, then declared again to the same circuit (a server restarted within its day: the declaration is
    read again, the log unchanged), then to another: core's check refuses it (``duplicate-declaration``), before anything
    is written. Another day takes its own declaration, chained after the first."""
    log = tmp_path / "log.json"
    first = SD.declare(_opts(tmp_path, *_day(tmp_path, 0, AUDITED)), tmp_path)
    assert first.declared == "now" and first.naming() == {"day": 0, "declaration": first.digest}
    body = log.read_bytes()
    again = SD.declare(_opts(tmp_path, *_day(tmp_path, 0, AUDITED)), tmp_path)
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
    assert (view.declared(0)["program"], view.declared(1)["program"]) == (AUDITED, OTHER)
    wrong = tmp_path / "remote.json"
    wrong.write_text(json.dumps(G.params("remote", ST.Auditor())))
    with pytest.raises(cli.UsageError, match="the log is onsite's, not --day-params' remote's"):
        SD.declare(_opts(tmp_path, *_day(tmp_path, 2, OTHER)[:-1], str(wrong)), tmp_path)
    log.unlink()
    with pytest.raises(cli.UsageError, match="a new log takes its composite from --day-params"):
        SD.declare(_opts(tmp_path, "--day", "0", "--day-log", str(log), "--day-program", AUDITED), tmp_path)


def _tied_day(tmp_path, monkeypatch, program: str):
    day = SD.declare(_opts(tmp_path, *_day(tmp_path, 0, program)), tmp_path)
    srv, runs = _Server(monkeypatch), {}
    wc = WindowCommit(runs)
    tie = TIE.WindowTie("v3c-window-commit", tmp_path / "serve", runner=wc, day=day)
    arm = SH.H1Windows(srv.model, srv.com, ROWS, STEPS, engine=object(), out=tmp_path / "serve", unit=_unit(),
                       prover=tmp_path / "prover", closer=(closer := Later()), tie=tie)
    return day, srv, arm, runs, closer


def test_a_tied_window_names_the_days_declaration_which_the_window_log_registers_first(tmp_path, monkeypatch):
    """With ``--day`` the window log's first line is the declaration, written when the tie is made, before any window.
    Each W names the day and the declaration's digest in its window and chains after the log's last W or R, never the
    declaration (a deployment log's entry, not a one-stage record); R, the day's program, is the window's Commit record. A
    tie made again on the serve dir doesn't register the declaration twice."""
    serve = tmp_path / "serve"
    day, srv, arm, runs, closer = _tied_day(tmp_path, monkeypatch, AUDITED)
    assert _log(serve) == [{"window": None, "kind": "declaration", "file": f"../{SD.FILE}", "digest": day.digest}]
    assert arm.tie.describe()["day"] == {"day": 0, "declaration": day.digest}
    pending = []
    for w, prompts in enumerate(PROMPTS):
        arm.open(w)
        runs[w] = srv.serve(prompts, w)
        pending.append(arm.close(w))
        arm.sealed(w, {"window": w, **SEALED})
    closer.run()
    previous = None
    for w, out in enumerate(f.result() for f in pending):
        d = serve / "h1" / f"w{w}"
        reg, rec = (json.loads((d / f).read_text()) for f in (TIE.W_FILE, TIE.R_FILE))
        assert "error" not in out and out["commit_record"]["named"] is True, out
        assert reg == CT.registration(runs[w], previous, day.naming()) and reg["previous"] == previous
        assert CT.day_of(reg) == {"day": 0, "declaration": day.digest} and rec["program"] == day.program == AUDITED
        assert CT.reused(runs[w], reg, rec) is None
        previous = REG.digest(rec)
    assert [x["kind"] for x in _log(serve)] == ["declaration"] + ["registration", "record"] * len(PROMPTS)
    again = TIE.WindowTie("again", serve, day=day)
    assert again.last == previous and [x["kind"] for x in _log(serve)].count("declaration") == 1
    arm.drain()
    srv.src.uninstall()


def test_an_r_whose_program_isnt_the_days_is_refused_at_its_close_and_opens_no_coin(tmp_path, monkeypatch):
    """The day declared another program than R's: the window binds no Commit record and opens no sampling coin, its error
    names R2's refusal, and only the declaration and W are in the log."""
    serve = tmp_path / "serve"
    _day_, srv, arm, runs, closer = _tied_day(tmp_path, monkeypatch, OTHER)
    arm.open(0)
    runs[0] = srv.serve(PROMPTS[0], 0)
    pending = arm.close(0)
    arm.sealed(0, {"window": 0, **SEALED})
    closer.run()
    out = pending.result()
    assert "the window names no Commit record" in out["error"] and "R2-declared" in out["error"], out["error"]
    c = json.loads((serve / "h1" / "w0" / PH.CALLS_FILE).read_text())["commitment"]
    assert c["commit"] is None and c["draw"]["coin"] is None
    assert [x["kind"] for x in _log(serve)] == ["declaration", "registration"]
    arm.drain()
    srv.src.uninstall()


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
