"""Each run that starts the records audit builds, with Flock's verifier, every executable the prover's Python looks up in that
package's build without building it (`read_tmpl.BUILT`), so a missing one stops the run's build, not its audit hours later."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FLOCK = "verity/protocols/proof/flock/verifier"
PROVER = ROOT / "verity/protocols/proof/flock/prover/python/verity_flock"
AUDITING = ["benchmarks/one_stage/served_verdicts.sh", "benchmarks/prove_path/run.sh"]


def looked_up() -> set[str]:
    return {m for f in sorted(PROVER.glob("*.py")) for m in re.findall(rf"{re.escape(FLOCK)}/\.lake/build/bin/([\w-]+)", f.read_text())}


def built(script: str) -> set[str]:
    lines = (ROOT / script).read_text().splitlines()
    return {t for line in lines if f"cd {FLOCK} " in line for m in re.findall(r"lake build ([\w -]+)\)", line) for t in m.split()}


def test_the_prover_looks_up_the_read_template_in_flocks_build():
    assert "flock-read-tmpl" in looked_up()


def test_each_auditing_run_builds_every_executable_the_prover_looks_up():
    want = looked_up()
    missing = {s: sorted(want - built(s)) for s in AUDITING if want - built(s)}
    assert not missing, f"add these to the script's `lake build` in {FLOCK}: {missing}"


def test_every_script_that_runs_the_records_audit_is_held():
    scripts = [p for d in ("benchmarks/one_stage", "benchmarks/prove_path") for p in sorted((ROOT / d).rglob("*.sh"))]
    runs = {str(p.relative_to(ROOT)) for p in scripts
            if re.search(r"records_audit\.py (audit|warm)", p.read_text(errors="replace")) and f"cd {FLOCK} " in p.read_text(errors="replace")}
    assert runs <= set(AUDITING), f"hold these to the prover's lookups in AUDITING too: {sorted(runs - set(AUDITING))}"
