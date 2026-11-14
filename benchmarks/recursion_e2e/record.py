"""The recursion bench's numbers (``metrics.toml``): ``run.py run`` with its job wrapped so that the run records them.

    python3 benchmarks/recursion_e2e/record.py run [run.py run's options]   one recorded run, as `run.py run` launches it
    python3 benchmarks/recursion_e2e/record.py job KEY=VALUE...              the run's command: job.sh, then its numbers

``job`` runs job.sh with its arguments, forwarding a SIGTERM to it (job.sh then writes the result so far), and then, in a
research run, writes `METRICS` from job.sh's result (``$OUT/recursion_e2e.json``) into the run's ``result.json`` through
the harness kit (`record`), exiting with job.sh's status, so that `run.py` and job.sh's own result step stay free of
the kit.

Stdlib only with the kit and ``verity_config``, which it puts on the path from the tree: the node's ``python3`` runs it.
"""
from __future__ import annotations

import json
import signal
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path += [str(HERE.parent / "harness"), str(HERE.parents[1] / "tools" / "config")]

import rec_table as RT  # noqa: E402
import run as R  # noqa: E402
from verity_bench import kit, settings  # noqa: E402

#: the metrics `benchmarks/recursion_e2e/metrics.toml` holds this benchmark to, by id: each the figure of the result's
#: headline it names, in seconds (`record`)
METRICS = {"benchmarks/recursion-run-seconds": "pipeline_sum_s",
           "benchmarks/recursion-lean-verifier-seconds": "Lean step_s"}
#: the run's command before job.sh's KEY=VALUE arguments, as `run.py` runs it from the tree's root
JOB = ("python3", "benchmarks/recursion_e2e/record.py", "job")
#: job.sh's result under its output directory (`run.py result --out`)
RESULT = "recursion_e2e.json"


def record(doc: dict | None, kept: str | None, rc: int) -> None:
    """Write `METRICS` into this research run's result: each the headline's figure it names, measured when the run was
    whole (every step one recursive run needs ran) and partial when it wasn't, with no value and why where the run has
    none (or job.sh, exiting `rc`, wrote no result `doc`); the run's parameters in the fingerprint; and the validation,
    passed only for an accepted recursive run (the headline's ``ok``), its detail the table's verdict and its evidence
    the result, at `kept` in the run's directory (None when job.sh wrote it elsewhere)."""
    w = kit.writer(settings.RESEARCH_RUN_DIR)
    h, p, st = (doc or {}).get("headline") or {}, (doc or {}).get("params") or {}, (doc or {}).get("staged")
    shape = f"{st['subcircuit']} x {st['n']}" if st else f"K = {p.get('K')}, N = {p.get('N')}"
    w.fingerprint(benchmark="recursion_e2e", shape=shape, modes=p.get("modes"), params=p)
    evidence = [kept] if doc is not None and kept else []
    for f in evidence:
        w.measurement_file(f)
    figures = {"pipeline_sum_s": h.get("pipeline_sum_s"), "Lean step_s": (h.get("pipeline") or {}).get("Lean step_s")}
    for name, figure in METRICS.items():
        value = figures[figure]
        why = (f"job.sh wrote no result (exit {rc})" if doc is None else
               f"the run has no {figure}: its steps were {', '.join(sorted(doc.get('steps') or {})) or 'none'}")
        w.measurement(name, value, "s", why=None if value is not None else why,
                      mark="measured" if h.get("complete") else "partial", extra={"shape": shape, "modes": p.get("modes")})
    if doc is None:
        w.validation("failed", "rec_profile.headline", f"job.sh wrote no result (exit {rc})", [])
    else:
        w.validation("passed" if h.get("ok") else "failed", "rec_profile.headline", RT.verdict(h, p.get("modes", "?")),
                     evidence)


def job(env: list[str]) -> int:
    """job.sh with `env`, then, in a research run, its numbers recorded (`record`); job.sh's exit status."""
    p = subprocess.Popen(["bash", str(HERE / "job.sh"), *env])
    signal.signal(signal.SIGTERM, lambda *_: p.send_signal(signal.SIGTERM))
    rc = p.wait()
    if not settings.is_set("RESEARCH_RUN_DIR"):
        return rc
    run_dir = Path(settings.RESEARCH_RUN_DIR).resolve()
    out = Path(settings.OUT).resolve() if settings.is_set("OUT") else run_dir / "out"
    result = out / RESULT
    doc = json.loads(result.read_text()) if result.is_file() else None
    kept = result.relative_to(run_dir).as_posix() if result.is_relative_to(run_dir) else None
    record(doc, kept, rc)
    return rc


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["job"]:
        return job(argv[1:])
    if argv[:1] == ["run"]:
        return R.main(argv, job=JOB)
    raise SystemExit(__doc__)


if __name__ == "__main__":
    sys.exit(main())
