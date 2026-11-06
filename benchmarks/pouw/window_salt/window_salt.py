"""A live H1 window's salt from the auditor's challenger: the salt-only session `core_check.window_salt` reads.

`issue(gate, rec, k, main, dest)` reserves a session for the salt coin alone (`window.SALT_FAMILIES`, scheme
`window.LIVE_SCHEME`) on `gate`, opens it on the epoch's root R = `salt.anchored_epoch_root(W, X, k)` (W and X the
window's weights' and activations' roots, `core_check.window_anchors` of ``rec``, the ``roots`` and ``window`` its
registration will carry; k its salt epoch's number among W's epochs), opens ``pouw/salt`` coin 0 after R, closes the
session and reads back what the gate ended. It returns the salt the window's H1 tiles take (`h1tile.live_salt` of that
coin at round 0 after R), the ``public.salt`` its service answer carries (``{coins, epoch, opening, salt, session}``) and
the verifier's ``live`` params (``{key, ledger, main, pins, session}``), and refuses (`CoinError`) unless
`core_check.window_salt` accepts them.

The window's order is: serve, commit its activations (their entries ``rec["window"]["activations"]``), `issue`, then
its H1 tiles under the salt, then its registration with those same entries. The gate opens a root once, ever, and the
coin follows it, so activations committed after the salt name another X, whose R no session was opened on.

`gate` is `a5_coins.Local` (a development session on this machine, under a key made there: tests and demos) or
`a5_coins.GateClient` (vy-challenger-1 through infra's gate); `local(root, main)` makes the first. `main` is the
verifier's own copy of main's first-parent history (``git rev-list --first-parent origin/main``, newest first).
`anchors(weights, activations)` is the ``rec`` of a window whose weights are registered values (the torch loop's pinned
``w`` and ``sw``).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verity.core.protocols.pouw import core_check as CC
from verity.core.protocols.pouw import salt as PS
from verity.core.protocols.pouw import window as Wd
from verity.ml.protocols.pouw.circuit import h1tile as HT
from verity.core.protocols.sampled_proofs import challenger as Ch
from verity.core.protocols.sampled_proofs import remote

TREE = Path(__file__).resolve().parents[3]


def a5_coins():
    """`benchmarks/one_stage/a5_coins.py`, whose `Local` and `GateClient` open a session on a gate."""
    if "a5_coins" not in sys.modules:
        spec = importlib.util.spec_from_file_location("a5_coins", TREE / "benchmarks" / "one_stage" / "a5_coins.py")
        module = sys.modules["a5_coins"] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return sys.modules["a5_coins"]


def local(root: Path, main: list[str]):
    """A development gate on this machine, its auditor host's root in scratch `root`, pinned on `main`."""
    return a5_coins().Local(root, main, by="compute-accounting")


def anchors(weights: dict[str, Any], activations: dict[str, Any]) -> dict[str, Any]:
    """The ``roots`` and ``window`` `core_check.window_anchors` reads of a window registering no ``W`` root: its
    registered weights' entries and its committed activations' entries, each ``{program, values}``."""
    return {"roots": [], "window": {"activations": activations, "registered": weights}}


@dataclass(frozen=True)
class Issued:
    """A live window's salt (`issue`): the bytes its H1 tiles take, its answer's ``public.salt`` and the verifier's
    ``live`` params."""

    salt: bytes
    answer: dict[str, Any]
    live: dict[str, Any]


def issue(gate: Any, rec: dict[str, Any], k: int, main: list[str], dest: Path, verify: Ch.Verify = CC.lms_verify
          ) -> Issued:
    """The salt of the live window whose registration will carry `rec`'s ``roots`` and ``window``, in salt epoch `k`,
    from one salt-only session of `gate`, its files copied to `dest` (module docstring)."""
    got = CC.window_anchors(rec)
    if isinstance(got, str):
        raise Ch.CoinError(f"the window can't be salted yet: {got}")
    r = PS.anchored_epoch_root(*got, k)
    address, secret, commitment = gate.open(r.hex(), [list(f) for f in Wd.SALT_FAMILIES], scheme=Wd.LIVE_SCHEME)
    with remote.RemoteChallenger.connect(address, bytes.fromhex(secret)) as client:
        c = client.commitment
        if c.root.hex() != commitment:
            raise Ch.CoinError(f"the challenger's hello answered commitment {c.root.hex()[:16]}..., not the gate's open's "
                               f"{commitment[:16]}...")
        opening = client.open(Wd.SALT, 0, r)
    directory, ledger, pins, key = gate.finish(dest)
    lines = remote.ledger_lines(ledger)
    opened = next(e for e in map(json.loads, lines) if e["kind"] == "open" and e["session"] == gate.session)
    answer = {"coins": c.to_json(), "epoch": k, "opening": opening.to_json(),
              "salt": HT.live_salt(c, opening.coin, 0, r).hex(),
              "session": {"session": opened["session"], "seq": opened["seq"], "root": r.hex(), "commitment": c.root.hex()}}
    live = {"key": key.to_json(), "ledger": lines, "main": list(main),
            "pins": [x for x in Path(pins).read_text().split("\n") if x], "session": remote.session_files(directory)}
    why = CC.window_salt({"check": {Wd.COINS: "live"}, "live": live}, rec, answer, verify)
    if why is not None:
        raise Ch.CoinError(f"the salt session {gate.session} doesn't salt the window: {why}")
    return Issued(bytes.fromhex(answer["salt"]), answer, live)


__all__ = ["Issued", "a5_coins", "anchors", "issue", "local"]
