"""`served_pearlplus.sh` under PEARLPLUS_TIE: Pearl+'s piece over the window's own epoch, which serve-openai's Pearl+ arm
(`serve_pearl`) retained in the serve dir's ``pearl/w<k>/``, in a toy tree whose steps each note their argv: the epoch
taken as the piece's passes, the give-up on the card, ``retain --window-record W``, PEARLPLUS_TIE_CMD's window commit
before the coin, and ``verify --tie W``."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

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
if script == "debit.py":
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
elif script == "served_verdicts.py" and cmd == "coin":
    print("coin: kept")
elif script == "served_verdicts.py" and cmd == "pearlplus":
    print("POUW-DEV {}")
else:
    sys.exit(f"unexpected step {argv}")
'''


def _piece(tmp_path: Path, served: dict | None, **extra: str) -> tuple[subprocess.CompletedProcess, Path, Path]:
    """The piece on window 7 of a toy serve dir whose ``pearl/w7/served.json`` is ``served`` (none if None), under
    PEARLPLUS_TIE with the window serve's cubin and a toy gpu-lease: the run, the serve dir and the toy tree."""
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
    assert steps == [("debit.py", "--cubin"), ("served_passes.py", "volunteer"), ("served_passes.py", "retain"),
                     ("served_passes.py", "verify"), ("served_verdicts.py", "coin"), ("served_verdicts.py", "pearlplus")]
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
    lines = run.stdout.splitlines()
    assert any(x.startswith("POUW window: the window's own Pearl+ epoch ") and x.endswith("coins in-process -> OK")
               for x in lines), lines
    assert "POUW W: rounds 0..2 (3), 384 values, digest abcd…" in lines
    assert any(x.startswith("POUW tie: served_verdicts.sh pearl-tie exit 0 after ") and x.endswith(": PEARL-TIE: tied")
               for x in lines), lines
    assert (sd / "pearl" / "w7" / "coin.json").exists()


def test_under_pearlplus_tie_a_window_whose_epoch_failed_or_is_missing_gets_no_retain_tie_or_verdict(tmp_path):
    """No retries: an epoch whose served.json isn't ok, or none at all, is the window's failure, and the piece gives up,
    retains, ties and verifies nothing, still writing its piece over what it has."""
    for name, served in (("failed", _epoch(ok=False)), ("missing", None)):
        d = tmp_path / name
        run, _sd, _tree = _piece(d, served)
        assert [Path(a[0]).name + " " + a[1] for a in _steps(d)] == ["served_verdicts.py pearlplus"], run.stdout
        assert not (d / "tie.args").exists()
        said = [x for x in run.stdout.splitlines() if x.startswith("POUW window: ")]
        assert len(said) == 1 and (said[0].endswith("so round s isn't step s") if served else
                                   said[0].endswith("so there are no passes to tie")), said
