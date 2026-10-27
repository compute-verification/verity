"""Every environment variable `verity_all_protocols` reads, declared once (``tests/lint/test_repo_env_reads.py``'s settings
registry; the template is ``integrations/vllm/verity_vllm/settings.py``). Each is read at the call, never cached.

- ``RESEARCH_RUN_ID``: the research run the parties run in (`research run` sets it); ``local`` outside one.
- ``VERITY_ALL_PROTOCOLS_TOOLS``: ``build.sh``'s ``tools.json``, which opts the tests into the parties' end-to-end run.
- ``RAYON_NUM_THREADS``, ``RUST_MIN_STACK``: the prover's threads and stack, set for the records audit when unset.
- ``PYTHONPATH``: the path a child process of the parties is given, this package's directory first.
- ``CERTIFIER_TAMPER``: the network certifier's relay tamper (`legs/certifier.py` reads an unset one as ``extra``, its
  benchmark's self-test); step 0's network certifier (`deployment`) relays honest traffic unless the caller set one.
"""
from __future__ import annotations

import os
from pathlib import Path


def run_id() -> str:
    return os.environ.get("RESEARCH_RUN_ID") or "local"


def tools() -> str | None:
    return os.environ.get("VERITY_ALL_PROTOCOLS_TOOLS") or None


def prover_defaults() -> None:
    """The prover's threads (every CPU) and stack (64 MiB), where the environment doesn't say."""
    os.environ.setdefault("RAYON_NUM_THREADS", str(os.cpu_count() or 1))
    os.environ.setdefault("RUST_MIN_STACK", str(64 << 20))


def certifier_defaults() -> None:
    """Honest relay (no ``CERTIFIER_TAMPER``), where the environment doesn't say."""
    os.environ.setdefault("CERTIFIER_TAMPER", "")


def child_env(first: Path) -> dict[str, str]:
    """This process's environment for a child, with ``first`` ahead of its ``PYTHONPATH``."""
    return {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(first), os.environ.get("PYTHONPATH")]))}
