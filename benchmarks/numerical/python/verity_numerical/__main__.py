"""``python -m verity_numerical JOB [ARGS]``, which ``verity bench numerical JOB [ARGS]`` runs: one of the package's tools,
started with ARGS as ``python -m <its module>`` starts it.  With no JOB it prints the jobs, one line each.

A job is a module of the package with its own command line (`JOBS`: its name, its module under ``verity_numerical`` and what it
does).  The module's docstring is its manual: what it reads, what it writes and what it needs.
"""
from __future__ import annotations

import importlib
import sys
from collections.abc import Sequence

__all__ = ["JOBS", "main"]

#: job -> (its module under ``verity_numerical``, what it does)
JOBS: dict[str, tuple[str, str]] = {
    "generate": ("bench.generate", "write synthetic input sets, one or a named suite, and register them in the evidence store"),
    "fixture_sets": ("bench.fixture_sets", "rewrite, add to or check the synthetic input sets Flock's tests read"),
    "instances": ("bench.instances", "build, verify or summarize the hash-pinned input set bench-instances/v1"),
    "subcircuit_catalog": ("bench.subcircuit_catalog", "build the catalog of every census subcircuit's Boolean export"),
    "unit_costs": ("bench.unit_costs", "count what one unit of each served workload's unit types costs as a Boolean circuit"),
    "unit_costs_padding": ("bench.unit_costs_padding", "price proving an audit's sampled proof blocks in fixed-shape sessions"),
    "unit_costs_prover": ("bench.unit_costs_prover", "fit the inner prover's time and memory per session from recorded runs"),
    "unit_costs_menu": ("bench.unit_costs_menu", "pin each model's menu of session shapes"),
    "latency": ("bench.latency", "price a result's rounds as network round trips, per batch"),
    "cell": ("bench.cell", "plan, launch, check and register one measured cell on its prover and verifier pods"),
    "placement": ("bench.placement", "probe a machine's identity and check where a run's parties are placed"),
    "sweep": ("checker.sweep", "run the reference checker over bench-instances/v1"),
    "search": ("checker.search", "search the checker's gadgets for an accepted wrong output"),
    "search_v2": ("checker.v2.search", "the same search over the v2 gadgets"),
}


def usage() -> str:
    """The jobs, one line each."""
    width = max(map(len, JOBS))
    return "verity bench numerical JOB [ARGS]: JOB is one of\n" + "".join(
        f"  {name:{width}}  {what}\n" for name, (_, what) in JOBS.items())


def main(argv: Sequence[str] | None = None) -> int:
    """Run the job ``argv`` names with the rest of ``argv``, and return its exit code; 2, with the jobs, for a name that is none."""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        sys.stdout.write(usage())
        return 0 if args else 2
    if args[0] not in JOBS:
        sys.stderr.write(f"verity bench numerical: no job {args[0]!r}\n{usage()}")
        return 2
    module = importlib.import_module(f"{__package__}.{JOBS[args[0]][0]}")
    return int(module.main(args[1:]) or 0)


if __name__ == "__main__":
    sys.exit(main())
