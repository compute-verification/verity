"""prove_time.sh on a templated statement (`flock-circuit/tmpl`, a census read such as `EmbeddingRead_v1`), on its own
functions lifted out of the script and run against stubs of stage.py, flock-tmpl-unit and flock-circuit. Its stage step
places the unit stage.py writes (unit.tmpl at unit.layout's mode) as unit.placed, also for a statement staged before this
step placed units, and leaves no unit.placed when flock-tmpl-unit fails. Its prove step passes unit.placed as --tmpl-unit to
both the server and the prover, which load the circuit with it alike (a server without it refuses the statement), passes
nothing to a flat statement's, and runs no session for a templated statement with no unit.placed."""
from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import pytest

from conftest import HERE

PT = (HERE / "prove_time.sh").read_text()
LAYOUT = "iv/hybrid/4"

#: stage.py UNIT N DIR: a statement in DIR, and a template and its layout mode beside it for a unit named `Tmpl`
STAGE = r'''import json, sys
from pathlib import Path
u, n, d = sys.argv[2], sys.argv[3], Path(sys.argv[4])
with open(d.parent.parent / "stage-calls", "a") as f:
    f.write(u + "\n")
(d / "circuit.txt").write_text("circuit " + u + "\n")
(d / "stage.json").write_text(json.dumps({"ands": 1, "k_log": 22, "vus_per_block": 1, "seconds": 0.0, "n": int(n)}))
if "Tmpl" in u:
    (d / "unit.tmpl").write_text("template of " + u + "\n")
    (d / "unit.layout").write_text("''' + LAYOUT + r'''")
'''
#: flock-tmpl-unit TMPL --layout MODE --out F (TU_FAIL: its refusal)
TMPL_UNIT = r'''#!/usr/bin/env bash
[ "$2" = --layout ] && [ "$4" = --out ] || exit 2
[ -n "${TU_FAIL:-}" ] && { echo 'TMPL {"ok":false,"error":"stub"}'; exit 1; }
{ cat "$1"; echo "ORDER $3"; } > "$5"
echo 'TMPL {"ok":true}'
'''
#: flock-circuit serve|prove ARGS: each call's arguments a line in ARGS_LOG
FLOCK_CIRCUIT = r'''#!/usr/bin/env bash
printf '%s\n' "$*" >> "$ARGS_LOG"
case $1 in serve) echo SERVING ;; prove) echo '{"accepted":true}' ;; esac
'''


def fn(name: str) -> str:
    f = re.search(rf"^{name}\(\) \{{(?:\n.*?\n\}}\n|[^\n]*\}}\n)", PT, re.M | re.S)
    assert f, f"prove_time.sh's {name}() is not where this test reads it"
    return f[0]


def exe(path: Path, text: str) -> Path:
    path.write_text(text)
    path.chmod(0o755)
    return path


@pytest.fixture
def sh(tmp_path):
    """Runs bash lines after prove_time.sh's shell options and functions, with the stubs on its variables; `taskset` runs its
    command as it is, and `wait_serving` waits for the stub server's exit rather than polling its log."""
    py = tmp_path / "py"
    py.mkdir()
    (py / "stage.py").write_text(STAGE)
    env = {**os.environ, "ARGS_LOG": str(tmp_path / "args.log"),
           "PT_TMPL_UNIT": str(exe(tmp_path / "flock-tmpl-unit", TMPL_UNIT))}
    head = "\n".join([
        re.search(r"^set -\w+ pipefail$", PT, re.M)[0],
        f'REPO={tmp_path}/repo FW={tmp_path}/fw O={tmp_path}/out KEY=k TIMEV= DEV=cpu PY="python3 {py}/stage.py"',
        f"B={exe(tmp_path / 'flock-circuit', FLOCK_CIRCUIT)} VCPU=0 PCPU=0 NC=2 VC=1 WARM=1 RUNS=4 NSESS=5",
        "mkdir -p $O $FW/data",
        *(fn(f) for f in ("name", "tmpl_unit", "place_unit", "stage", "free_port", "stamp", "session_at")),
        'taskset() { shift 2; "$@"; }',
        "wait_serving() { wait $2; grep -q SERVING $1; }",
    ])

    def run(lines: str, **extra: str) -> subprocess.CompletedProcess:
        return subprocess.run(["bash", "-c", head + "\n" + lines], capture_output=True, text=True, env={**env, **extra})

    run.root = tmp_path
    return run


def statement(root: Path, unit: str) -> Path:
    return root / "fw/data" / f"census-{unit.lower()}-n2-k"


def calls(root: Path) -> list[str]:
    log = root / "args.log"
    return log.read_text().splitlines() if log.exists() else []


def test_stage_places_a_templated_statements_unit(sh):
    r = sh("stage census:Tmpl 2 && stage census:Flat 2")
    assert r.returncode == 0, r.stdout + r.stderr
    t, f = statement(sh.root, "Tmpl"), statement(sh.root, "Flat")
    assert (t / "unit.placed").read_text() == f"template of census:Tmpl\nORDER {LAYOUT}\n"
    assert not (f / "unit.placed").exists() and not (f / "unit.tmpl").exists()
    assert json.loads((sh.root / "out/stage/census-tmpl-n2/tmpl-unit.log").read_text().split(" ", 1)[1])["ok"]


def test_stage_places_the_unit_of_a_statement_staged_without_one(sh):
    assert sh("stage census:Tmpl 2").returncode == 0
    t = statement(sh.root, "Tmpl")
    (t / "unit.placed").unlink()
    r = sh("stage census:Tmpl 2")
    assert r.returncode == 0, r.stdout + r.stderr
    assert (t / "unit.placed").exists()
    assert (sh.root / "fw/stage-calls").read_text().splitlines() == ["census:Tmpl"], "the statement was staged again"


def test_stage_fails_without_a_placed_unit_when_flock_tmpl_unit_fails(sh):
    r = sh("stage census:Tmpl 2", TU_FAIL="1")
    assert r.returncode != 0 and "flock-tmpl-unit FAILED" in r.stdout
    t = statement(sh.root, "Tmpl")
    assert not (t / "unit.placed").exists() and not (t / "unit.placed.part").exists()


def test_session_passes_the_placed_unit_to_server_and_prover(sh):
    r = sh("stage census:Tmpl 2 && session_at census-tmpl-n2 $FW/data/census-tmpl-n2-k")
    assert r.returncode == 0, r.stdout + r.stderr
    placed = statement(sh.root, "Tmpl") / "unit.placed"
    serve, prove = calls(sh.root)
    assert serve.startswith("serve ") and prove.startswith("prove ")
    for c in (serve, prove):
        assert f"--circuit {placed.parent}/circuit.txt --tmpl-unit {placed} " in c, c
    assert (sh.root / "out/cpu/census-tmpl-n2/exit").read_text().strip() == "0"


def test_session_passes_no_unit_for_a_flat_statement(sh):
    r = sh("stage census:Flat 2 && session_at census-flat-n2 $FW/data/census-flat-n2-k")
    assert r.returncode == 0, r.stdout + r.stderr
    serve, prove = calls(sh.root)
    assert "--tmpl-unit" not in serve + prove


def test_session_refuses_a_templated_statement_without_its_unit(sh):
    assert sh("stage census:Tmpl 2").returncode == 0
    (statement(sh.root, "Tmpl") / "unit.placed").unlink()
    r = sh("session_at census-tmpl-n2 $FW/data/census-tmpl-n2-k")
    assert r.returncode != 0 and "without unit.placed" in r.stdout
    assert calls(sh.root) == [], "a session ran without its unit"
    o = sh.root / "out/cpu/census-tmpl-n2"
    assert (o / "exit").read_text().strip() == "1" and (o / "done").exists()
