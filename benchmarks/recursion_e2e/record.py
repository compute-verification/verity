"""The recursion bench's numbers (``metrics.toml``): ``run.py run`` with its job wrapped so that the run records them.

    uv run python benchmarks/recursion_e2e/record.py run [run.py run's options]   one recorded run, as `run.py run` launches it
    python3 benchmarks/recursion_e2e/record.py job KEY=VALUE...                    the run's command: job.sh, then its numbers

``job`` runs job.sh with its arguments (stage.sh, which ends in job.sh, with ``WINDOW=``), forwarding a SIGTERM to it (job.sh then writes the result so far), and then, in a
research run, writes `METRICS` from job.sh's result (``$OUT/recursion_e2e.json``) into the run's ``result.json`` through
the harness kit (`record`), exiting with job.sh's status, so that `run.py` and job.sh's own result step stay free of
the kit.

Stdlib only with the kit and ``verity_config``: the node's ``python3`` runs it under the research run, whose
``PYTHONPATH`` holds the shipped tree's workspace roots, and ``uv run`` runs it at the launcher (``metrics.toml``).
"""
from __future__ import annotations

import hashlib
import json
import re
import signal
import subprocess
import sys
from pathlib import Path

import rec_profile as RP
import rec_table as RT
import run as R
from verity_bench import kit, settings

HERE = Path(__file__).resolve().parent

#: the metrics `benchmarks/recursion_e2e/metrics.toml` holds this benchmark to, by id: each the figure of the result's
#: headline it names, in seconds (`record`)
METRICS = {"benchmarks/recursion-run-seconds": "pipeline_sum_s",
           "benchmarks/recursion-lean-verifier-seconds": "Lean step_s"}
#: the run's command before job.sh's KEY=VALUE arguments, as `run.py` runs it from the tree's root
JOB = ("python3", "benchmarks/recursion_e2e/record.py", "job")
#: job.sh's result under its output directory (`run.py result --out`)
RESULT = "recursion_e2e.json"
#: a served window's public files; every other file of it is the prover's (rows, salts, its registration's secrets)
PUBLIC = re.compile(r"window\.json|registration\.json|m\d+-public\.bin")
#: what a staging of a window names its private values: a member's rows and salts, the instance file, the prover's record
PRIVATE_NAMES = re.compile(r"m\d+-private\.bin|inst-\d+\.bin|private\.json")


def private_in(out: Path, window: Path) -> list[Path]:
    """The files under `out` that hold a private file of `window` or of its staging: by name (`PRIVATE_NAMES`, or a
    non-public file's name of the window's) or by bytes (a copy of one, under any name)."""
    private = [f for f in window.rglob("*") if f.is_file() and not PUBLIC.fullmatch(f.name)]
    names = {f.name for f in private}
    digests: dict[int, set[str]] = {}
    for f in private:
        digests.setdefault(f.stat().st_size, set()).add(hashlib.sha256(f.read_bytes()).hexdigest())
    return sorted(f for f in out.rglob("*") if f.is_file() and not f.is_symlink() and (
        PRIVATE_NAMES.fullmatch(f.name) or f.name in names
        or hashlib.sha256(f.read_bytes()).hexdigest() in digests.get(f.stat().st_size, ())))


def member(doc: dict | None, serves: str | None) -> str | None:
    """The measurements' member for a run over a subset of a served window: the served workload's and N
    (``llama-3.2-1b-h200-n16``); None for seeded inputs, whose line is the metric's own."""
    st = (doc or {}).get("staged") or {}
    return f"{serves}-n{st['n']}" if serves and st.get("subset") else None


def record(doc: dict | None, kept: str | None, rc: int, member: str | None = None, removed: list[str] | None = None) -> None:
    """Write `METRICS` into this research run's result: each the headline's figure it names, measured when the run was
    whole (every step one recursive run needs ran) and partial when it wasn't, with no value and why where the run has
    none (or job.sh, exiting `rc`, wrote no result `doc`); the run's seconds have none while a step of `RP.E2E_STEPS` is
    missing, since the sum would be only the steps that ran; the run's parameters in the fingerprint; and the validation,
    passed only for an accepted recursive run (the headline's ``ok``), its detail the table's verdict and its evidence
    the result, at `kept` in the run's directory (None when job.sh wrote it elsewhere). `removed`: private files of a window
    found in the run's out/ and deleted there, which fail the validation whatever the verdict."""
    w = kit.writer(settings.RESEARCH_RUN_DIR)
    h, p, st = (doc or {}).get("headline") or {}, (doc or {}).get("params") or {}, (doc or {}).get("staged")
    shape = f"{st['subcircuit']} x {st['n']}" if st else f"K = {p.get('K')}, N = {p.get('N')}"
    w.fingerprint(benchmark="recursion_e2e", shape=shape, modes=p.get("modes"), params=p)
    evidence = [kept] if doc is not None and kept else []
    for f in evidence:
        w.measurement_file(f)
    ran = (doc or {}).get("steps") or {}
    missing = [x for x in RP.E2E_STEPS if not ran.get(x)]
    figures = {"pipeline_sum_s": None if missing else h.get("pipeline_sum_s"),
               "Lean step_s": (h.get("pipeline") or {}).get("Lean step_s")}
    for name, figure in METRICS.items():
        value = figures[figure]
        why = (f"job.sh wrote no result (exit {rc})" if doc is None else
               f"the run is missing {', '.join(missing)}: its {figure} would sum only the steps that ran"
               if figure == "pipeline_sum_s" and missing else
               f"the run has no {figure}: its steps were {', '.join(sorted(ran)) or 'none'}")
        w.measurement(name, value, "s", why=None if value is not None else why, member=member,
                      mark="measured" if h.get("complete") else "partial", extra={"shape": shape, "modes": p.get("modes")})
    if removed:
        w.validation("failed", "record.private_in", f"the window's private files were in out/ and are deleted: {', '.join(removed)}",
                     [])
    elif doc is None:
        w.validation("failed", "rec_profile.headline", f"job.sh wrote no result (exit {rc})", [])
    else:
        w.validation("passed" if h.get("ok") else "failed", "rec_profile.headline", RT.verdict(h, p.get("modes", "?")),
                     evidence)


def job(env: list[str]) -> int:
    """job.sh with `env` (stage.sh with ``WINDOW=``), then, in a research run, every private file of the window deleted
    from out/ and its numbers recorded (`record`); job.sh's exit status, or 1 where a private file was found."""
    given = dict(e.split("=", 1) for e in env if "=" in e)
    window = given.get("WINDOW")
    p = subprocess.Popen(["bash", str(HERE / ("stage.sh" if window else "job.sh")), *env])
    signal.signal(signal.SIGTERM, lambda *_: p.send_signal(signal.SIGTERM))
    rc = p.wait()
    if not settings.is_set("RESEARCH_RUN_DIR"):
        return rc
    run_dir = Path(settings.RESEARCH_RUN_DIR).resolve()
    out = Path(settings.OUT).resolve() if settings.is_set("OUT") else run_dir / "out"
    result = out / RESULT
    doc = json.loads(result.read_text()) if result.is_file() else None
    kept = result.relative_to(run_dir).as_posix() if result.is_relative_to(run_dir) else None
    removed = []
    if window and out.is_dir():
        for f in private_in(out, Path(window)):
            removed.append(f.relative_to(run_dir).as_posix() if f.is_relative_to(run_dir) else str(f))
            f.unlink()
    record(doc, kept, rc, member(doc, given.get("SERVES")), removed)
    return 1 if removed else rc


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["job"]:
        return job(argv[1:])
    if argv[:1] == ["run"]:
        return R.main(argv, job=JOB)
    raise SystemExit(__doc__)


if __name__ == "__main__":
    sys.exit(main())
