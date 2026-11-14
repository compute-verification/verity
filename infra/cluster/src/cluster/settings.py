"""The environment variables `cluster` reads, declared once (`tests/lint/test_repo_env_reads.py`: reads belong here)."""
from __future__ import annotations

import os


def host_var(node: str) -> str:
    """The variable that holds `node`'s address: the node's name in capitals, ``-`` as ``_``, then ``_HOST``
    (vy-nebius-1's is ``VY_NEBIUS_1_HOST``)."""
    return node.upper().replace("-", "_") + "_HOST"


def host(node: str) -> str | None:
    """`node`'s address as `machines.d/<node>.toml` registers it, which the environment holds (`host_var`) and no file
    here does; None while it is unset or empty."""
    return os.environ.get(host_var(node)) or None
