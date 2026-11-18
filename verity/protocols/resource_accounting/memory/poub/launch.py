"""The served runner's launcher of the bandwidth audit's piece (`Launch.entry`): called with the scheme's declaration,
it gives the piece's hooks, ``hooks(cfg, params, bound, pinned)``, the runner's variables for the piece, and the scripts
they run. A launcher starts and collects; it decides no verdict."""

from __future__ import annotations

import shlex
from typing import Any

BANDWIDTH_SCRIPT = "benchmarks/pous/served_piece.py"
DEVELOPMENT = {"DEVELOPMENT": "1"}
#: served_piece.py's flag for each binding of PoUB's bandwidth verifier
_BANDWIDTH = {"host": "--bw-host", "command": "--bw-party"}


def _bandwidth(cfg, params, bound, pinned):
    """PoUB's prover inside PoUS's piece (served_piece.py's --bw-*): the held store proved to the bandwidth verifier a
    pass at a time, its answer collected by BANDWIDTH_CMD. The close's key is the configuration's pin for the bandwidth verifier alone; with no pin the session has no
    case. Unbound, PoUS's run keys start the bandwidth party too, under PoUS's key made for the run."""
    words = [*(["--bw-trusted-sha512", *pinned] if pinned else []),
             *(w for k, flag in _BANDWIDTH.items() if k in bound for w in (flag, bound[k]))]
    return {**({"POUS_ARGS": shlex.join(words)} if words else DEVELOPMENT),
            "BANDWIDTH_CMD": f'"$PY" {BANDWIDTH_SCRIPT} collect-bandwidth --serve-dir "$1" --window "$2" --out "$3"'}


def bandwidth(declaration: Any) -> tuple[Any, tuple[str, ...]]:
    """The bandwidth audit's prover piece (`Launch.entry` of ``declaration``, the bandwidth audit's scheme)."""
    return _bandwidth, (BANDWIDTH_SCRIPT,)


__all__ = ["BANDWIDTH_SCRIPT", "bandwidth"]
