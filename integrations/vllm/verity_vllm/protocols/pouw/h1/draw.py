"""H1's draw coin from a draw session of its own (`window.DRAW_FAMILIES`): the session the auditor's ledger opens on the
window's `pouw_h1_requests.draw_root`, withheld (`challenger.withheld_on`), so its challenger released its coin root only
after that root, the prover's commitment. `pouw_h1_challenger` reads a window's draw through `draw_session_reason`.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from verity.core.protocols.pouw import window as Wd
from verity.core.protocols.sampled_proofs import challenger as Ch
from verity.core.protocols.sampled_proofs import remote
from verity.core.protocols.sampled_proofs.challenger import (ChallengerKey, CoinError, LiveCommitment, LiveOpening,
                                                             live_reason)

from verity_vllm.protocols.pouw.h1 import requests as RQ

_ROOT = re.compile(r"^[0-9a-f]{128}$")
#: why a window's draw coin has no session to be read from
NO_DRAW_SESSION = ("the verifier names no draw session: a window's draw coin is its own draw session's, opened withheld "
                   "on its draw root (CMP-177)")


def _draw(commits: Mapping[str, Any]) -> Mapping[str, Any]:
    draw = commits.get("draw")
    return draw if isinstance(draw, Mapping) else {}


def draw_session_reason(commits: Mapping[str, Any], key: ChallengerKey, lines: list[str], pins: Any, main: Any,
                        files: Any, verify: Any) -> str | None:
    """Why the window's draw coin (its H1 commitment `commits`'s ``draw``) isn't its draw session's: the session the
    auditor's ledger `lines` opens on the window's draw root under `key`, run from a commit the pins log `pins` records
    on `main` (`challenger.trusted_session_of` over its files `files`, as infra's gate publishes them; None names none),
    opened withheld (`challenger.withheld_on`), so its challenger released its coin root only after that root, of a draw
    session's layout (`window.draw_epoch_reason`), and the coin its draw coin 0 opened after that root
    (`draw_reason`); None when it is."""
    root = _draw(commits).get("root")
    if not (isinstance(root, str) and _ROOT.match(root)):
        return "the window's draw root is not 128 lowercase hex"
    if files is None:
        return NO_DRAW_SESSION
    try:
        trusted, received = Ch.trusted_session_of(lines, {"root": root}, key, pins, main, files, verify)
    except CoinError as e:
        return f"the window's draw session is not trusted: {e}"
    if not Ch.withheld_on(lines, root):
        return f"the window's draw session: {Ch.WITHHELD}"
    why = Wd.draw_epoch_reason(trusted)
    if why is not None:
        return f"the window's draw session's signed coins: {why}"
    return draw_reason(commits, trusted, received.get(Wd.DRAW, ()))


def draw_reason(commits: Mapping[str, Any], trusted: LiveCommitment, received: Sequence[bytes]) -> str | None:
    """Why the window's draw coin isn't its one draw coin 0, opened after its `pouw_h1_requests.draw_root` (over the
    manifest's ``calls_root``, the last round's receipts and every round's root and coin), given its draw session's
    signed commitment `trusted` and the draw roots its log shows it received; None when it is."""
    if not received:
        return "the draw session closed with no draw coin opened, so the window has no draw of the challenger's"
    draw = _draw(commits)
    try:
        root = RQ.draw_root(bytes.fromhex(commits["calls_root"]), bytes.fromhex(draw["receipts"]),
                            list(commits.get("rounds") or ()))
    except (KeyError, TypeError, ValueError) as e:
        return f"the window's draw root is malformed ({e})"
    if draw.get("root") != root.hex():
        return "the window's draw root is not over its manifest, its last round's receipts and every round's root and coin"
    if tuple(received) != (root,):
        return ("the challenger's log does not show the window's draw root as the one root its draw coin followed: a draw "
                "opened before the window's rounds, or after another root")
    try:
        opening = LiveOpening.from_json(draw.get("coin"))
    except CoinError as e:
        return f"the window's draw coin is malformed ({e})"
    why = live_reason(trusted, opening, Wd.DRAW, 0, root)
    return None if why is None else f"the window's draw coin: {why}"


def draw_files(commits: Mapping[str, Any], ledger: Any, fetch: Any, verify: Any) -> Any:
    """The files of the window's draw session (`draw_session_reason`), the one `ledger` (`pouw_h1_challenger.Ledger`)
    opens on the window's draw root (its H1 commitment `commits`'s ``draw``'s ``root``), from its ``out/`` as `fetch`
    gives it; None when the ledger opens none there or `fetch` has no tree for it, which `draw_session_reason` refuses."""
    root = _draw(commits).get("root")
    try:
        tree = fetch(ledger.pinned({"root": root}, verify).session) if isinstance(root, str) else None
        return None if tree is None else remote.session_files(Path(tree) / "out")
    except CoinError:
        return None


__all__ = ["NO_DRAW_SESSION", "draw_files", "draw_reason", "draw_session_reason"]
