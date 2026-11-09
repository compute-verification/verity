"""`served_pearlplus.sh` in toy trees whose steps each note their argv.

Under PEARLPLUS_TIE: Pearl+'s piece over the window's own epoch, which serve-openai's Pearl+ arm (`serve_pearl`) retained
in the serve dir's ``pearl/w<k>/``: the epoch taken as the piece's passes, the give-up on the card, ``retain
--window-record W``, PEARLPLUS_TIE_CMD's window commit before the coin, and ``verify --tie W``.

The give-up before the verifier's coin, with stand-ins for served_passes.py, debit.py, served_verdicts.py,
pearl_c_sm90.sh's setup and gpu-lease: it asks ``served_passes.py volunteer-needed`` before anything is built or served,
gives the rows over a debit cap up (on the card by default) exactly when the answer is yes, refuses to serve with the
give-up turned off then, leaves it off when the answer is no unless PEARLPLUS_VOLUNTEER asks for it, stops on any other
answer, and hands PEARLPLUS_OVER_CAP to both the question and the verdict."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

HERE = Path(__file__).resolve().parent

#: the toy venv's python: each served_passes.py step, debit.py and served_verdicts.py noted and answered as the real one
#: writes its files; anything else (the script's inline checks) is the real interpreter
STEP = r'''
import json, sys
from pathlib import Path
argv = sys.argv[1:]
with open(sys.argv[0] + ".log", "a") as f:
    f.write(json.dumps(argv) + "\n")
def opt(name):
    return argv[argv.index(name) + 1] if name in argv else None
script, cmd = Path(argv[0]).name, argv[1] if len(argv) > 1 else None
if script == "served_passes.py" and cmd == "volunteer-needed":
    print("yes reject")
elif script == "debit.py":
    Path(opt("--out")).write_text("{}")
elif script == "served_passes.py" and cmd == "volunteer":
    Path(opt("--out")).write_text(json.dumps({"s": 1.0, "jobs": 1, "passes": []}))
    print("volunteer PASS: 0 rows given up")
elif script == "served_passes.py" and cmd == "retain":
    if w := opt("--window-record"):
        Path(w).write_text(json.dumps({"window": {"scheme": "pearl-c-h100-v1-h3"}}))
        print("retain PASS: 3 of 3 passes with calls retained")
        print(f"retain W: {w}, rounds 0..2 (3), 384 values, digest abcd…")
    else:
        print("retain PASS: 3 of 3 passes with calls retained")
elif script == "served_passes.py" and cmd == "verify":
    Path(opt("--passes"), "coin.json").write_text("{}")
    Path(opt("--out")).write_text(json.dumps({"answer": {"verdict": "accept"}}))
    print("ACCEPT verity/pouw/pearl-plus/v1")
elif script == "served_passes.py" and cmd == "lean":
    print("lean: no executable")
elif script == "served_verdicts.py" and cmd == "coin":
    print("coin: kept")
elif script == "served_verdicts.py" and argv[1:3] == ["pearlplus-period", "rule"]:
    print(f"RULE no other reserve on lane {opt('--lane')} between period pp-r1's close and the draw: HELD (toy)")
elif script == "served_verdicts.py" and cmd == "pearlplus":
    print("POUW-DEV {}")
else:
    sys.exit(f"unexpected step {argv}")
'''


def _piece(tmp_path: Path, served: dict | None, period: dict | None = None,
           **extra: str) -> tuple[subprocess.CompletedProcess, Path, Path]:
    """The piece on window 7 of a toy serve dir whose ``pearl/w7/served.json`` is ``served`` (none if None) and whose
    window serve left ``pearlplus/period.json`` `period` (none if None), under PEARLPLUS_TIE with the window serve's cubin
    and a toy gpu-lease: the run, the serve dir and the toy tree."""
    tree, bin_, venv, sd = tmp_path / "tree", tmp_path / "bin", tmp_path / "venv" / "bin", tmp_path / "serve"
    for p in (tree / "benchmarks" / "one_stage", tree / "benchmarks" / "harness", tree / ".venv" / "bin", bin_, venv,
              sd / "pearl" / "w7", tmp_path / "piece", tmp_path / "cubin"):
        p.mkdir(parents=True)
    (tree / "benchmarks" / "one_stage" / "served_pearlplus.sh").symlink_to(HERE.parent / "served_pearlplus.sh")
    (tree / "benchmarks" / "harness" / "env.sh").write_text("")
    (tmp_path / "step.py").write_text(STEP)
    py = (f'#!/bin/sh\ncase "$1" in -c|-) exec {sys.executable} "$@";; esac\n'
          f'exec {sys.executable} {tmp_path}/step.py "$@"\n')
    for path, body in ((venv / "python", py), (tree / ".venv" / "bin" / "python", py),
                       (bin_ / "gpu-lease", '#!/bin/sh\nwhile [ "$1" != -- ]; do shift; done\nshift\nexec "$@"\n')):
        path.write_text(body)
        path.chmod(0o755)
    (tmp_path / "cubin" / "pearl_c_sm90.cubin").write_bytes(b"cubin")
    if served is not None:
        (sd / "pearl" / "w7" / "served.json").write_text(json.dumps(served))
    if period is not None:
        (sd / "pearlplus").mkdir()
        (sd / "pearlplus" / "period.json").write_text(json.dumps(period))
    env = {"PATH": f"{bin_}:{os.defpath}", "HOME": str(tmp_path), "VENV": str(venv.parent), "PEARLPLUS_TIE": "1",
           "PEARLC_KERNEL": "pearl_c_sm90", "PEARLC_CUBIN": str(tmp_path / "cubin" / "pearl_c_sm90.cubin"),
           "PEARLPLUS_JOBS": "1", "PEARLPLUS_TIE_CMD": f'printf "%s\\n" "$@" > {tmp_path}/tie.args; echo "PEARL-TIE: tied"',
           **extra}
    run = subprocess.run(["bash", str(tree / "benchmarks" / "one_stage" / "served_pearlplus.sh"), str(sd), "7",
                          str(tmp_path / "piece")], cwd=tmp_path, env=env, check=False, capture_output=True, text=True)
    return run, sd, tree


def _steps(tmp_path: Path) -> list[list[str]]:
    log = tmp_path / "step.py.log"
    return [json.loads(x) for x in log.read_text().splitlines()] if log.exists() else []


def _epoch(ok: bool = True) -> dict:
    return {"schema": "verity-pouw/pearl-c-served-passes/v1", "entry": "pouw:pearl-c-h100-v1-h3", "window": 7,
            "commit_steps": 3, "passes_with_calls": 3, "coins": {"source": "in-process"}, "ok": ok,
            "error": None if ok else "its passes with calls are 2 and its Commit's steps 3, so round s isn't step s",
            "passes": [{"pass": i, "timed": True, "units": 64, "manifest": f"m{i}"} for i in range(3)]}


def test_under_pearlplus_tie_the_piece_ties_the_windows_own_epoch_to_its_w(tmp_path):
    """The piece serves nothing: its passes are the window's epoch (a symlink to ``pearl/w7``), the card gives up its rows
    over a cap (VOLUNTEER's default), `retain` writes W, PEARLPLUS_TIE_CMD commits the window tied to W before the coin
    (the serve dir, the window, W and the passes), `verify` holds the passes to W, and the piece reads the epoch's
    served.json."""
    run, sd, tree = _piece(tmp_path, _epoch())
    q = sd / "private" / "pearlplus"
    assert run.returncode == 0, (run.stdout, run.stderr)
    assert (q / "passes").is_symlink() and (q / "passes").resolve() == (sd / "pearl" / "w7").resolve()
    steps = [(Path(a[0]).name, a[1] if len(a) > 1 else None) for a in _steps(tmp_path)]
    assert steps == [("served_passes.py", "volunteer-needed"), ("debit.py", "--cubin"), ("served_passes.py", "volunteer"),
                     ("served_passes.py", "retain"), ("served_passes.py", "verify"), ("served_verdicts.py", "coin"),
                     ("served_passes.py", "lean"), ("served_verdicts.py", "pearlplus")]
    argv = {Path(a[0]).name + " " + a[1]: a for a in _steps(tmp_path)}
    w = str(q / "window-record.json")
    retain = argv["served_passes.py retain"]
    assert retain[retain.index("--passes") + 1] == str(q / "passes") and retain[retain.index("--window-record") + 1] == w
    assert retain[retain.index("--volunteer") + 1] == str(q / "volunteer.json")
    assert (tmp_path / "tie.args").read_text().split() == [str(sd), "7", w, str(q / "passes")]
    verify = argv["served_passes.py verify"]
    assert verify[verify.index("--tie") + 1] == w and verify[verify.index("--passes") + 1] == str(q / "passes")
    pearlplus = argv["served_verdicts.py pearlplus"]
    assert pearlplus[pearlplus.index("--served") + 1] == str(q / "passes" / "served.json")
    assert pearlplus[pearlplus.index("--tie") + 1] == w
    lines = run.stdout.splitlines()
    assert any(x.startswith("POUW window: the window's own Pearl+ epoch ") and x.endswith("coins in-process -> OK")
               for x in lines), lines
    assert "POUW W: rounds 0..2 (3), 384 values, digest abcd…" in lines
    assert any(x.startswith("POUW tie: served_verdicts.sh pearl-tie exit 0 after ") and x.endswith(": PEARL-TIE: tied")
               for x in lines), lines
    assert (sd / "pearl" / "w7" / "coin.json").exists()


def test_under_pearlplus_tie_a_window_whose_epoch_failed_or_is_missing_gets_no_retain_tie_or_verdict(tmp_path):
    """No retries: an epoch whose served.json isn't ok, or none at all, is the window's failure, and the piece gives up,
    retains, ties, verifies or asks the Lean nothing, still writing its piece over what it has, as the window's own
    epoch (``--tie W``)."""
    for name, served in (("failed", _epoch(ok=False)), ("missing", None)):
        d = tmp_path / name
        run, sd, _tree = _piece(d, served)
        assert [Path(a[0]).name + " " + a[1] for a in _steps(d)] == ["served_passes.py volunteer-needed",
                                                                     "served_verdicts.py pearlplus"], run.stdout
        pearlplus = _steps(d)[-1]
        assert pearlplus[pearlplus.index("--tie") + 1] == str(sd / "private" / "pearlplus" / "window-record.json")
        assert not (d / "tie.args").exists()
        said = [x for x in run.stdout.splitlines() if x.startswith("POUW window: ")]
        assert len(said) == 1 and (said[0].endswith("so round s isn't step s") if served else
                                   said[0].endswith("so there are no passes to tie")), said


@pytest.mark.parametrize("case", ["held", "unclosed", "refused", "unread"])
def test_under_pearlplus_tie_and_the_gate_the_piece_draws_only_after_a_period_whose_reserves_held(tmp_path, case):
    """Under the gate the window serve closed its Pearl-C period (served_verdicts.sh's serve()), its period.json saying
    whether the lane's reserves since the open were the window's salt sessions alone (``reserves``) and holding the gate's
    close (``close``): the piece draws only after one that closed and held, and once the coin is drawn its RULE line reads
    the ledger (``pearlplus-period rule`` on the serve dir's pearlplus/, the lane and the coin record). Otherwise, the
    gate's close refused too, it says why, draws nothing and still writes its piece."""
    reserves = {"held": {"ok": True}, "unclosed": {"ok": True}, "refused": {"ok": False}, "unread": None}[case]
    period = {"name": "pp-r1", **({} if reserves is None else {"reserves": reserves}),
              **({} if case == "unclosed" else {"close": {"kind": "period-close"}})}
    run, sd, _tree = _piece(tmp_path, _epoch(), period=period, PEARLC_CHALLENGER_GATE="ssh gate",
                            PEARLC_CHALLENGER_LANE="vllm")
    assert run.returncode == 0, (run.stdout, run.stderr)
    names = [" ".join([Path(a[0]).name, *a[1:3 if a[1:2] == ["pearlplus-period"] else 2]]) for a in _steps(tmp_path)]
    given = ["served_passes.py volunteer-needed", "debit.py --cubin", "served_passes.py volunteer", "served_passes.py retain"]
    lines = run.stdout.splitlines()
    if case == "held":
        assert names == [*given, "served_passes.py verify", "served_verdicts.py coin",
                         "served_verdicts.py pearlplus-period rule", "served_passes.py lean",
                         "served_verdicts.py pearlplus"], lines
        rule = _steps(tmp_path)[names.index("served_verdicts.py pearlplus-period rule")]
        q = sd / "private" / "pearlplus"
        assert rule[3:] == [str(sd / "pearlplus"), "--gate", "ssh gate", "--lane", "vllm", "--coin",
                            str(q / "passes" / "coin.json")]
        assert "POUW RULE no other reserve on lane vllm between period pp-r1's close and the draw: HELD (toy)" in lines
    else:
        assert names == [*given, "served_verdicts.py pearlplus"], lines
        assert ("POUW draw: refused before it: the window serve's Pearl-C period didn't close with the lane's reserves "
                "since its open the window's salt sessions alone (served_verdicts.sh's PEARLPLUS-PERIOD line), so no draw "
                "follows it") in lines
        assert not (sd / "pearl" / "w7" / "coin.json").exists()


RECORD = '''import json, os, sys
from pathlib import Path
with open(os.environ["CALLS"], "a") as f:
    f.write(json.dumps({"tool": Path(sys.argv[0]).name, "argv": sys.argv[1:]}) + "\\n")
rest = sys.argv[1:]
at = lambda flag: Path(rest[rest.index(flag) + 1])
'''
PASSES = '''if rest[0] == "volunteer-needed":
    print(os.environ["NEEDED"])
    sys.exit(0 if os.environ["NEEDED"] else 2)
if rest[0] == "serve":
    at("--passes").mkdir(parents=True, exist_ok=True)
if rest[0] in ("serve", "volunteer", "verify"):
    at("--out").write_text("{}")
print(rest[0], "ok")
'''
SETUP = '''echo '{"tool": "pearl_c_sm90.sh", "argv": []}' >> "$CALLS"
mkdir -p "$RESEARCH_RUN_DIR/k/debit" && touch "$RESEARCH_RUN_DIR/k/pearl_c_sm90.cubin" "$RESEARCH_RUN_DIR/k/debit/debit_sm90.cubin"
echo '{"ok": true}' > "$RESEARCH_RUN_DIR/k/harness_gate.json"
PEARLC_KERNEL=pearl_c_sm90 PEARLC_CUBIN=$RESEARCH_RUN_DIR/k/pearl_c_sm90.cubin
'''
SERVED = ["volunteer-needed", "pearl_c_sm90.sh", "serve"]
CHECKED = ["verify", "lean"]


def _untied(tmp_path: Path, needed: str, volunteer: str | None = None) -> Any:
    """The untied piece on window 3 in a toy tree whose served_passes.py, served_verdicts.py, debit.py and setup each note
    their argv in ``calls``, ``volunteer-needed`` answering `needed`: a function running it under more environment, which
    returns its exit code, its output and the calls in order."""
    tree, d, p, calls = tmp_path / "tree", tmp_path / "serve", tmp_path / "piece", tmp_path / "calls"
    one = tree / "benchmarks" / "one_stage"
    one.mkdir(parents=True)
    shutil.copy(HERE.parent / "served_pearlplus.sh", one)
    (tree / ".venv" / "bin").mkdir(parents=True)
    (tree / ".venv" / "bin" / "python").symlink_to(sys.executable)
    (one / "served_verdicts.py").write_text(RECORD + 'print("POUW-DEV {}")\n')
    pouw = tree / "benchmarks" / "pouw"
    (pouw / "pearl_c_vllm").mkdir(parents=True)
    (pouw / "pearl_c_vllm" / "served_passes.py").write_text(RECORD + PASSES)
    (pouw / "pearl_c_sm90").mkdir()
    (pouw / "pearl_c_sm90" / "debit.py").write_text(RECORD + 'at("--out").write_text("{}")\n')
    (tree / "benchmarks" / "harness" / "setup").mkdir(parents=True)
    (tree / "benchmarks" / "harness" / "env.sh").write_text('VENV=$VERITY_BENCH_VENV\n')
    (tree / "benchmarks" / "harness" / "setup" / "pearl_c_sm90.sh").write_text(SETUP)
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    (bin_ / "gpu-lease").write_text('#!/bin/sh\nwhile [ "$1" != -- ]; do shift; done\nshift\nexec "$@"\n')
    (bin_ / "gpu-lease").chmod(0o755)
    d.mkdir()
    env = {"PATH": f"{bin_}:{os.defpath}", "HOME": str(tmp_path), "CALLS": str(calls), "NEEDED": needed,
           "VERITY_BENCH_VENV": str(tree / ".venv"), "PEARLPLUS_PASSES_RAM": "", "PEARLPLUS_UNRETAINED": "",
           **({} if volunteer is None else {"PEARLPLUS_VOLUNTEER": volunteer})}

    def run(**more: str) -> tuple[int, str, list[dict]]:
        calls.unlink(missing_ok=True)
        r = subprocess.run(["bash", "benchmarks/one_stage/served_pearlplus.sh", str(d), "3", str(p)], cwd=tree,
                           env={**env, **more}, capture_output=True, text=True, check=False)
        return r.returncode, r.stdout + r.stderr, [json.loads(x) for x in calls.read_text().splitlines()]
    return run


@pytest.mark.parametrize("needed, volunteer, steps", [
    ("yes reject", None, [*SERVED, "debit.py", "volunteer --device", "retain --volunteer", *CHECKED, "pearlplus --volunteer"]),
    ("yes reject", "cpu", [*SERVED, "volunteer", "retain --volunteer", *CHECKED, "pearlplus --volunteer"]),
    ("yes reject", "", ["volunteer-needed"]),
    ("no uncredited", None, [*SERVED, *CHECKED, "pearlplus"]),
    ("no uncredited", "", [*SERVED, *CHECKED, "pearlplus"]),
    ("no uncredited", "1", [*SERVED, "debit.py", "volunteer --device", "retain --volunteer", *CHECKED, "pearlplus --volunteer"]),
    ("", None, ["volunteer-needed"]),
])
def test_the_give_up_runs_exactly_when_volunteer_needed_says_yes_and_cant_be_skipped_then(tmp_path, needed, volunteer, steps):
    d, run = tmp_path / "serve", _untied(tmp_path, needed, volunteer)
    rc, said, got = run()
    flags = {"volunteer": "--device", "retain": "--volunteer", "pearlplus": "--volunteer"}
    names = []
    for c in got:
        name = c["argv"][0] if c["tool"] in ("served_passes.py", "served_verdicts.py") else c["tool"]
        names.append(f"{name} {flags[name]}" if flags.get(name) in c["argv"] else name)
    assert names == steps, said
    if steps == ["volunteer-needed"]:
        assert rc == 2 and not (d / "private" / "pearlplus" / "setup.log").exists(), said
        assert ("POUW volunteer: refused, so nothing is served" if needed else "volunteer-needed answered ''") in said
        return
    assert rc == 0, said
    if "verify" in steps and not any(s.startswith("volunteer ") or s == "volunteer" for s in steps):
        assert f"POUW volunteer: not run: Pearl+'s verdict credits a drawn row over its debit cap nothing (over-cap rule " \
               f"{needed.split()[1]})" in said
    if needed == "yes reject" and volunteer is None:
        rc, said, got = run(PEARLPLUS_OVER_CAP="reject")
        asked = {c["argv"][0]: c["argv"] for c in got if c["tool"] == "served_passes.py"}
        assert rc == 0 and asked["volunteer-needed"] == ["volunteer-needed", "--over-cap", "reject"], said
        assert asked["verify"][asked["verify"].index("--over-cap") + 1] == "reject"


def test_under_the_gate_the_untied_piece_holds_its_pearl_c_period_open_from_its_serve_to_right_before_its_draw(tmp_path):
    """Under the gate the untied piece opens the lane's Pearl-C period right before its retained serve and closes it only
    right before verify draws, after the give-up and retain: while it is open its cap of 1, which the serve's salt session
    fills, refuses every other reserve of the lane, so none comes between its close and the draw, which a compose takes
    only as the lane's first reserve after the close."""
    rc, said, got = _untied(tmp_path, "yes reject")(PEARLC_CHALLENGER_GATE="ssh gate")
    names = [" ".join(c["argv"][:2]) if c["argv"][:1] == ["pearlplus-period"] else
             c["argv"][0] if c["tool"] in ("served_passes.py", "served_verdicts.py") else c["tool"] for c in got]
    assert rc == 0 and names == ["volunteer-needed", "pearl_c_sm90.sh", "pearlplus-period open", "serve", "debit.py",
                                 "volunteer", "retain", "pearlplus-period close", *CHECKED, "pearlplus"], said
