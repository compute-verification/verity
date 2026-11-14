#!/usr/bin/env python3
"""The onsite demo's minutes from its first command to its verdict, read off its two runs' records: the metric
``vllm/onsite-demo-minutes`` (`demos/onsite_served/metrics.toml`). Standard library only.

    research run --cwd source -- python3 demos/onsite_served/minutes.py PROVER_RUN AUDITOR_RUN [--runs-dir DIR] [--main REF]

A run of record runs the demo's two commands (README.md, "Running it") each under ``research run``: PROVER_RUN is the
prover node's, AUDITOR_RUN the auditor host's, whose log (``stdout.log``) holds the period's ``VERDICT`` line. The
minutes run from PROVER_RUN's ``started_utc`` to AUDITOR_RUN's ``ended_utc``. Finding the runs, the served commit (the
prover node's run's), what is refused and what is written are `demos/vllm/minutes.py`'s, the remote demo's reader,
whose code this one runs with its own verdict (`verdict`). It also refuses an AUDITOR_RUN whose log holds no
``VERDICT`` line, a line whose ``run`` is ``local`` (a `verity run` outside any research run, whose record holds no
time), and one whose ``components`` give a verdict on fewer than the goal's seven checks (`CHECKS`).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

METRIC = "vllm/onsite-demo-minutes"
#: the goal's seven checks, by the names a ``VERDICT`` line's ``components`` give them (served_table's ONSITE_COMPONENTS)
CHECKS = ("sampled-proofs", "pouw", "inference-only", "pous", "bandwidth", "certifier", "wipe")
VLLM = Path(__file__).resolve().parents[1] / "vllm"


def _reader():
    """The remote demo's reader, loaded by its path under a name of its own, with its directory first on the path for
    the `settings` it imports."""
    sys.path.insert(0, str(VLLM))
    spec = importlib.util.spec_from_file_location("vllm_demo_minutes", VLLM / "minutes.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _reader()


def verdict(d: Path) -> str:
    """The period's verdict in run ``d``'s ``VERDICT`` lines (`verity run`'s), accept only where every line accepts;
    `Refused` where there is none, or one is outside a research run or short of the goal's checks."""
    log = d / "stdout.log"
    lines = [json.loads(x.removeprefix("VERDICT ")) for x in (log.read_text().splitlines() if log.is_file() else ())
             if x.startswith("VERDICT {")]
    if not lines:
        raise R.Refused(f"{d.name}: no verdict line in its stdout.log")
    for x in lines:
        if x.get("run") == "local":
            raise R.Refused(f"{d.name}: its verdict line's run is local, a `verity run` outside any research run")
        given = {k for k, v in (x.get("components") or {}).items() if v != "absent"}
        if gone := [c for c in CHECKS if c not in given]:
            raise R.Refused(f"{d.name}: its verdict covers {len(CHECKS) - len(gone)} of the goal's {len(CHECKS)} "
                            f"checks, not {', '.join(gone)}")
    return next((x["verdict"] for x in lines if x.get("verdict") != "accept"), "accept")


def main(argv: list[str] | None = None) -> int:
    return R.main(argv, read=verdict, metric=METRIC, checker="demos/onsite_served/minutes.py", nargs=2, doc=__doc__)


if __name__ == "__main__":
    sys.exit(main())
