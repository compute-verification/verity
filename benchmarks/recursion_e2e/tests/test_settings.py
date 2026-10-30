"""The recursion bench's settings (``settings.py``): every variable job.sh reads is declared, an unknown ``REPEAT_*`` or
``OPROVE_*`` variable is refused by run.py and by job.sh's first command, the record's digest, and the ratchet on direct
environment uses."""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
import run as RUN
import settings as RS
from verity_bench.settings import direct_uses

HERE = Path(__file__).resolve().parents[1]
#: The files of this member that still use the environment directly, and how often.  It only shrinks.
BASELINE: dict[str, int] = {}
#: declared but not read by name in job.sh: REPEAT_<STEP> through ``${!v:-$REPEAT}``, 85's OPROVE_MEM_GB under the job's
#: prefix, and run.py's own
NOT_IN_JOB = {*(f"REPEAT_{s.upper()}" for s in RS.REPEATED), "OPROVE_MEM_GB", "RESEARCH_RUNS_DIR"}


def test_every_variable_job_sh_reads_is_declared_and_every_declared_one_is_read() -> None:
    read = set(re.findall(r"\$\{([A-Z][A-Z0-9_]*):?[-+=?]", (HERE / "job.sh").read_text()))
    declared = set(RS.REGISTRY.settings)
    assert not read - declared, f"declare it in benchmarks/recursion_e2e/settings.py: {sorted(read - declared)}"
    assert declared - read == NOT_IN_JOB, sorted(declared - read)
    assert set(RS.REPEATED) <= set(RUN.STEPS)


def test_run_refuses_an_unknown_job_variable_before_it_launches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(RUN.subprocess, "run", lambda *a, **k: pytest.fail("launched"))
    for bad, said in (("REPEAT_LAEN=1", "unknown setting REPEAT_LAEN: nothing in the tree reads it"),
                      ("OPROVE_RUNZ=2", "unknown setting OPROVE_RUNZ"), ("REPEAT_LEAN=one", "REPEAT_LEAN='one'")):
        with pytest.raises(SystemExit, match=re.escape(said)):
            RUN.main(["run", "--dry-run", "--env", bad])
    assert RUN.main(["run", "--dry-run", "--env", "REPEAT_LEAN=1", "--env", "LEVELS=L3"]) == 0


def _record(tmp_path: Path, **env: str) -> subprocess.CompletedProcess:
    base = {k: v for k, v in RS.REGISTRY.environ().items() if not k.startswith(RS.REGISTRY.prefixes) and k != "RESEARCH_RUN_ID"}
    return subprocess.run([sys.executable, str(HERE / "settings.py"), "record", str(tmp_path / "settings.json")],
                          env={**base, **env}, capture_output=True, text=True, check=False)


def test_the_jobs_first_command_refuses_an_unknown_variable_and_records_the_rest(tmp_path: Path) -> None:
    p = _record(tmp_path, OPROVE_WARMUP="1")
    assert p.returncode == 2 and "unknown setting OPROVE_WARMUP" in p.stderr and not (tmp_path / "settings.json").exists()
    p = _record(tmp_path, K="4k")
    assert p.returncode == 2 and "K='4k'" in p.stderr
    p = _record(tmp_path, K="64", N="16", MODES="firewall", TAG="t", LEVELS="L0")
    assert p.returncode == 0, p.stderr
    doc = json.loads((tmp_path / "settings.json").read_text())
    assert doc["values"] == {"K": "64", "MODES": "firewall", "N": "16"} and p.stdout.strip() == doc["digest"]
    assert doc["digest"] == RS.record({"K": "64", "N": "16", "MODES": "firewall", "TAG": "t"})["digest"] == (
        "451b3c49825b835f8cd1fc20bf415bcea9d7eda13b5c169a4995e68e423aba28"
        "9293a16edbd1edcead0ecf7085f45a747b8665d86fb3c1c7cfb67942f5b9a719")


def test_direct_environment_uses_only_shrink() -> None:
    found = {}
    for p in sorted(HERE.rglob("*.py")):
        if "__pycache__" not in p.parts and (uses := direct_uses(p.read_text())):
            found[str(p.relative_to(HERE))] = len(uses)
    grown = {f: n for f, n in found.items() if n > BASELINE.get(f, 0)}
    assert not grown, f"read the environment through benchmarks/recursion_e2e/settings.py (declare it there): {grown}"
    shrunk = {f: found.get(f, 0) for f, n in BASELINE.items() if found.get(f, 0) < n}
    assert not shrunk, f"fewer direct uses than BASELINE allows: lower it in tests/test_settings.py to {shrunk}"
