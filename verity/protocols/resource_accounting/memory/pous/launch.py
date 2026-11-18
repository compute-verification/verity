"""The served runner's launchers of PoUS's and the wipe's pieces (`Launch.entry`): each called with its scheme's
declaration, it gives the piece's hooks, ``hooks(cfg, params, bound, pinned)``, the runner's variables for the piece, and
the scripts they run. A launcher starts and collects; it decides no verdict."""

from __future__ import annotations

import shlex
from collections.abc import Callable, Mapping
from typing import Any

#: the scripts' one flag for a development run's attester: on the workload's machine, under a key made for the run
RUN_KEYS = "--run-keys"
DEVELOPMENT = {"DEVELOPMENT": "1"}

WIPE_SCRIPT = "benchmarks/pous/streamed_wipe/served_wipe.sh"
POUS_SCRIPT, ONSITE_PIECES = "benchmarks/pous/served_piece.sh", "benchmarks/pous/onsite_pieces.sh"
ATTESTER_SCRIPT = "benchmarks/pous/run_attester.sh"
#: served_wipe.sh's variable for each binding of the wipe's memory attester
_WIPE = {"address": "WIPE_ADDRESS", "host": "WIPE_CHALLENGER", "command": "WIPE_COMMAND"}
#: served_piece.py's flag for each binding of PoUS's memory attester
_POUS = {"address": "--challenger-address", "host": "--challenger-host", "command": "--challenger-command"}

Hooks = Callable[[Mapping[str, Any], Mapping[str, Any], Mapping[str, str], list[str]], dict[str, str]]


def _wipe(cfg, params, bound, pinned):
    """The streamed wipe of the card before the workload's engine loads (WIPE_CMD, served_wipe.sh). The key is the configuration's pin alone, on the command line, so neither the environment nor a binding gives
    another; with no pin served_wipe.sh refuses. Unbound, the key is the run's (`RUN_KEYS`), and no pin is passed."""
    if "address" not in bound:
        return {"WIPE_CMD": f'bash {WIPE_SCRIPT} "$1" "$2" "$3" {RUN_KEYS}', **DEVELOPMENT}
    return {"WIPE_CMD": f'WIPE_KEY_SHA512={"".join(pinned)} bash {WIPE_SCRIPT} "$1" "$2" "$3"',
            **{v: bound[k] for k, v in _WIPE.items() if k in bound}}


def _pous(cfg, params, bound, pinned):
    """PoUS's prover after the workload (POUS_CMD, served_piece.sh): the memory attester's timed reads answered from
    the served weights held as P2's C* on the card, with the memory attester on the workload's machine in a development
    run. The memory attester's key is the configuration's pin alone, on the command line; with no pin the piece
    trusts no session. Unbound, the key is the run's (`RUN_KEYS`), and no pin is passed."""
    words = [*(["--trusted-sha512", *pinned] if pinned else []), *([] if "address" in bound else [RUN_KEYS]),
             "--rounds", str(params["challenges"]),
             *(w for k, flag in _POUS.items() if k in bound for w in (flag, bound[k]))]
    return {"POUS_CMD": f'bash {POUS_SCRIPT} "$@"', "POUS_ARGS": shlex.join(words), "ONSITE_PIECES": ONSITE_PIECES,
            **({} if "address" in bound else DEVELOPMENT)}


def wipe(declaration: Any) -> tuple[Hooks, tuple[str, ...]]:
    """The wipe's prover piece (`Launch.entry` of ``declaration``, the wipe's scheme)."""
    return _wipe, (WIPE_SCRIPT,)


def pous(declaration: Any) -> tuple[Hooks, tuple[str, ...]]:
    """PoUS's prover piece (`Launch.entry` of ``declaration``, PoUS's scheme)."""
    return _pous, (POUS_SCRIPT, ONSITE_PIECES, ATTESTER_SCRIPT)


__all__ = ["ATTESTER_SCRIPT", "DEVELOPMENT", "ONSITE_PIECES", "POUS_SCRIPT", "RUN_KEYS", "WIPE_SCRIPT", "pous", "wipe"]
