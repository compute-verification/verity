"""served_pearlplus.sh's give-up before the verifier's coin, with stand-ins for served_passes.py, debit.py,
served_verdicts.py, pearl_c_sm90.sh's setup and gpu-lease: it asks ``served_passes.py volunteer-needed`` before anything is
built or served, gives the rows over a debit cap up (on the card by default) exactly when the answer is yes, refuses to
serve with the give-up turned off then, leaves it off when the answer is no unless PEARLPLUS_VOLUNTEER asks for it, stops
on any other answer, and hands PEARLPLUS_OVER_CAP to both the question and the verdict."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
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
