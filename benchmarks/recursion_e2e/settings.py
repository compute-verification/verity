"""The recursion bench's environment variables, each declared once in a `verity_config.settings.Registry`, checked by
verity-bench's `check` (``benchmarks/harness/verity_bench/settings.py``): what ``job.sh`` reads (its own, and those of
``verity/ml/flock/pod/85-rec-reprice.sh``'s it sets a default for), and what ``run.py`` and ``forge_after.py`` read.

``job.sh`` takes its variables as ``KEY=VALUE`` arguments (``run.py``'s ``job_env``) and passes every one on to 85.  It runs
``python3 benchmarks/recursion_e2e/settings.py record FILE`` first, which refuses, by name, a variable under the job's own
prefixes (``REPEAT_``, a step's repeats; ``OPROVE_``, oprove's) that nothing reads, a declared value that doesn't parse, and a
debug setting on a run of record (one with ``RESEARCH_RUN_ID``); then writes the value settings in force to FILE and prints
their digest, which ``params.txt`` (so the result's ``params``) carries as ``settings``.  ``run.py`` refuses the same before it
launches.  85's other variables (``LEVELS``, ``PCPUS``, ...) pass through undeclared: 85 owns them.

Stdlib only, as ``run.py`` is, and so are ``verity_bench.settings`` and ``verity_config``, which it imports from the tree:
``job.sh`` runs it with the pod's python3.
"""
from __future__ import annotations

import functools
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "benchmarks" / "harness"), str(ROOT / "tools" / "config")]

from verity_bench.settings import check
from verity_config.settings import VALUE, WHERE, Registry, Setting, csv

#: the steps job.sh repeats, each of which REPEAT_<STEP> may set (job.sh's ``reps``)
REPEATED = ("icheck", "vstage", "vcap", "oreplay", "lean")

S = Setting
REGISTRY = Registry(__file__, ("REPEAT_", "OPROVE_"),
    # ---- the job's own (job.sh)
    S("STEPS", csv, ("build", "inner", "icheck", "vstage", "vcap", "oprove", "oreplay", "lean", "negative"), VALUE,
      "the steps the job runs, in rec_profile.ORDER's order"),
    S("REPEAT", int, 3, VALUE, "each CPU step's invocations"),
    *(S(f"REPEAT_{s.upper()}", int, None, VALUE, f"the {s} step's invocations; unset, REPEAT") for s in REPEATED),
    S("MODES", csv, ("firewall",), VALUE, "oprove's prover sides; the last one's releases are the ones oreplay and lean read"),
    S("OPROVE_WARM", int, 0, VALUE, "warm releases per V* statement, untimed"),
    S("OPROVE_RUNS", int, 1, VALUE, "timed releases per V* statement"),
    S("OPROVE_GPU", str, None, VALUE, "V*'s prover: 0 85's CPU prover, 1 the GPU one under a lease; unset, 0 under the firewall, "
      "else GPU"),
    S("NEG", str, "challenge,query", VALUE, "the inner coins of rep 0 the negative step flips (empty: none)"),
    S("FORGE", csv, (), VALUE, "85's forgeries, which vstage stages and oprove proves"),
    S("FORGE_AFTER", csv, ("coef", "dirs", "message"), VALUE, "the forgeries forge-after makes on the honest releases"),
    S("LEAN_JOBS", int, 4, VALUE, "V* releases Lean verifies at once (85's JOBS for the lean step), which its times depend on"),
    S("KEEP_DATA", str, None, WHERE, "1 keeps 85's data/TAG for a later job: retention, not what is measured"),
    # ---- 85's, which the job sets a default for (verity/ml/flock/pod/85-rec-reprice.sh)
    S("K", int, 4096, VALUE, "the inner statement's GemmCoordinate_v2 K"),
    S("N", int, 2, VALUE, "the inner statement's instances"),
    S("WARM", int, 1, VALUE, "the inner step's warm runs"),
    S("RUNS", int, 3, VALUE, "the inner step's timed runs"),
    S("GPU", str, None, VALUE, "0: 85 proves on the CPU; unset, 1"),
    S("OPROVE_JOBS", int, 1, VALUE, "V* statements 85 proves at once (85's own default is 4), which their times depend on"),
    S("OPROVE_MEM_GB", int, None, VALUE, "the estimated peak GB the statements 85 proves at once may sum to; unset, 85's 48"),
    S("GPU_MAX_MIN", int, 75, WHERE, "each prove's GPU lease in minutes: a bound on how long it may hold the GPU"),
    S("GPU_PREEMPTIBLE", str, "", WHERE, "85's leases are preemptible unless empty (the job's default): how a lease is held"),
    S("TAG", str, None, WHERE, "85's data/TAG directory; unset, rb-<run id>"),
    S("FLOCK_WORK", str, "/workspace/jobs/rec-bench", WHERE, "85's work tree: its caches and data"),
    S("CUDA_TK", str, None, WHERE, "the CUDA toolkit 85 builds the prover with, a path; the build records the binary's sha256"),
    S("WARM_FROM", str, None, WHERE, "a warm cargo tree 85 builds from, read only; the build records the binary's sha256"),
    # ---- research run's
    S("OUT", str, None, WHERE, "the directory a job, or forge_after.py, writes its outputs in"),
    S("RESEARCH_RUN_DIR", str, None, WHERE, "the run directory; the job's OUT is its out/"),
    S("RESEARCH_RUN_ID", str, None, WHERE, "the research run the job is, which makes it a run of record"),
    S("RESEARCH_SOURCE_SHA", str, None, VALUE, "the commit research run shipped, which params.txt names"),
    S("RESEARCH_RUNS_DIR", str, None, WHERE, "where `research fetch` puts a run's records; unset, ~/.research/runs (run.py's load)"),
    shared=False)
del S

get, record = REGISTRY.get, REGISTRY.record
validate = functools.partial(check, REGISTRY)


def __getattr__(name: str) -> Any:
    if name in REGISTRY.settings:
        return get(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def main(argv: list[str]) -> int:
    """``record FILE``: validate this process's environment, write its record to FILE and print the record's digest."""
    if len(argv) != 2 or argv[0] != "record":
        print("usage: settings.py record FILE", file=sys.stderr)
        return 2
    try:
        validate(of_record=REGISTRY.is_set("RESEARCH_RUN_ID"))
    except ValueError as e:
        print(f"recursion_e2e: {e}", file=sys.stderr)
        return 2
    doc = record()
    Path(argv[1]).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(doc["digest"])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
