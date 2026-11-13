"""served_pouw.sh, a remote run's POUW_CMD, with stand-ins for live.sh, drawn.sh, drawn.py, served_verdicts.py,
pouw_piece.py and lake: the live rounds from the window's --h1-prover dump and the drawn units are proved first, their
scratch and logs under the serve dir's private pouw/ (drawn.sh's kept, its check left to credit), then the piece and
the period's case get their statements, live evidence and one memo in the verifier's own scratch (0700, outside the
serve dir, removed on exit), each line saying what it took by key; its lines name no window and quote neither tool; and
H1_POUW_REUSE=1 reruns the piece and the case on the kept proofs alone."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RECORD = '''import json, os, sys
from pathlib import Path
with open(os.environ["CALLS"], "a") as f:
    f.write(json.dumps({"tool": Path(sys.argv[0]).name, "argv": sys.argv[1:], **{k: os.environ.get(k) for k in (
        "RESEARCH_RUN_DIR", "H1_LIVE", "H1_KEEP", "H1_DRAWN_ONLY", "H1_DRAWN_CHECK", "H1_DRAWN_WORK")}}) + "\\n")
'''
PIECE = "LEAN-ANSWERS 0 of 2 runs by key, 2 ran"
PERIOD = "LEAN-ANSWERS 2 of 2 runs by key (2 from the challenger check), 0 ran"


def _tool(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(RECORD + body)


def test_the_pouw_command_proves_then_writes_the_piece_and_the_period_case(tmp_path):
    tree, d, p, calls = tmp_path / "tree", tmp_path / "serve", tmp_path / "out" / "pieces" / "pouw", tmp_path / "calls"
    one = tree / "benchmarks" / "one_stage"
    one.mkdir(parents=True)
    shutil.copy(HERE.parent / "served_pouw.sh", one)
    (tree / ".venv" / "bin").mkdir(parents=True)
    (tree / ".venv" / "bin" / "python").symlink_to(sys.executable)
    (tree / "verity" / "protocols" / "pouw" / "verifier" / "lean" / "check").mkdir(parents=True)
    h1 = tree / "benchmarks" / "pouw" / "h1_prove"
    for name, body in (("live.sh", 'mkdir -p "$RESEARCH_RUN_DIR/out/live" && echo "{}" > "$RESEARCH_RUN_DIR/out/live/live-evidence.json"'),
                       ("drawn.sh", 'mkdir -p "$3/drawn" && echo "[]" > "$3/drawn/statements.json"')):
        _tool(h1 / f"{name}.py", "")
        (h1 / name).write_text(f'"{sys.executable}" "$(dirname "$0")/{name}.py" "$@"\n{body}\necho "window 3 secret"\n')
    _tool(one / "served_verdicts.py", 'a = Path(sys.argv[sys.argv.index("--lean-answers") + 1])\n'
                                      'Path(os.environ["CALLS"] + ".mode").write_text(oct(a.parent.stat().st_mode & 0o777))\n'
                                      'c = Path(sys.argv[4]) / "check"\nc.mkdir(parents=True, exist_ok=True)\n'
                                      f'(c / "check.log").write_text("window 3 secret\\n{PIECE}\\n")\n'
                                      'print("POUW-DEV {}")\nprint("window 3 secret")\n')
    _tool(one / "pouw_piece.py", f'print("PIECE window 3 secret")\nprint("{PERIOD}")\nsys.exit(0)\n')
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    (bin_ / "lake").write_text('#!/bin/sh\nmkdir -p .lake/build/bin && touch ".lake/build/bin/$2" && chmod +x ".lake/build/bin/$2"\n')
    (bin_ / "lake").chmod(0o755)
    (d / "private" / "h1" / "w3" / "live").mkdir(parents=True)
    vtmp = tmp_path / "vtmp"
    vtmp.mkdir()
    env = {"PATH": f"{bin_}:{os.defpath}", "HOME": str(tmp_path), "CALLS": str(calls), "H1_KEY_SHA512": "ab" * 64,
           "MAIN_FILE": str(tmp_path / "main.txt"), "VERIFIER_TMP": str(vtmp)}

    def run(**more: str) -> list[str]:
        calls.unlink(missing_ok=True)
        r = subprocess.run(["bash", "benchmarks/one_stage/served_pouw.sh", str(d), "3", str(p)], cwd=tree,
                           env={**env, **more}, capture_output=True, text=True, check=False)
        assert r.returncode == 0, r.stdout + r.stderr
        return r.stdout.splitlines()

    said = run()
    q = d / "private" / "pouw"
    st, le = q / "drawn" / "statements.json", q / "live" / "out" / "live" / "live-evidence.json"
    ev = ["--statements", str(st), "--live-evidence", str(le)]
    exe = tree / "verity/protocols/pouw/verifier/lean/check/.lake/build/bin/pouw-period"
    got = [json.loads(x) for x in calls.read_text().splitlines()]
    memo = Path(got[2]["argv"][got[2]["argv"].index("--lean-answers") + 1])
    assert [(c["tool"], c["argv"]) for c in got] == [
        ("live.sh.py", []), ("drawn.sh.py", [str(d), "3", str(q)]),
        ("served_verdicts.py", ["pouw", str(d), "3", str(p), *ev, "--lean-answers", str(memo)]),
        ("pouw_piece.py", ["period", "--out", str(q / "period"), "--exe", str(exe), "--serve-run", str(d), "--window", "3",
                           "--pouw", str(p), "--key-sha512", "ab" * 64, "--main", env["MAIN_FILE"], *ev,
                           "--lean-answers", str(memo)])]
    assert memo.parent.parent == vtmp and not memo.parent.exists()
    assert Path(str(calls) + ".mode").read_text() == "0o700"
    assert (got[0]["RESEARCH_RUN_DIR"], got[0]["H1_LIVE"], got[0]["H1_KEEP"]) == (
        str(q / "live"), str(d / "private" / "h1" / "w3" / "live"), str(q / "live" / "live-proofs"))
    assert (got[1]["H1_DRAWN_ONLY"], got[1]["H1_DRAWN_CHECK"], got[1]["H1_DRAWN_WORK"]) == ("1", "0", str(q / "drawn-work"))
    assert said[0].startswith("POUW live rounds: live.sh exit 0 after ") and said[0].endswith(", live evidence written")
    assert said[1].startswith("POUW drawn units: drawn.sh exit 0 after ") and said[1].endswith(", statements written")
    assert said[2] == "POUW-DEV {}" and said[3].startswith("POUW piece: served_verdicts.py pouw exit 0")
    assert said[3].endswith("; its Lean answers: 0 of 2 runs by key, 2 ran")
    assert said[4].startswith("POUW period: accept (pouw_piece.py period exit 0") and len(said) == 5
    assert said[4].endswith("; it took the piece's Lean answers by key: 2 of 2 runs by key (2 from the challenger check), "
                            "0 ran")
    assert not any("secret" in x or " 3" in x for x in said)
    assert oct(q.stat().st_mode & 0o777) == "0o700"

    said = run(H1_POUW_REUSE="1")
    assert said[0] == "POUW proofs: the kept statements and live evidence (H1_POUW_REUSE=1)"
    got = [json.loads(x) for x in calls.read_text().splitlines()]
    assert [c["tool"] for c in got] == ["served_verdicts.py", "pouw_piece.py"]
    assert not any(" 3" in x for x in said)
