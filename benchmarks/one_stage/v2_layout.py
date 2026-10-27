"""A v2 window's withheld challenger layout, sized from the statements its sessions prove (`OneStageZk.roundsWhy`).

v2 reads every coin of a window from the one withheld challenger session the auditor's ledger opens on it, one release
a use (`proofs.session_uses`, `coins.round_coins`): the draw's family (`a5_coins.families`), then for each session `i`
the window proves, family ``slot/<i>/session`` (its link points) and ``slot/<i>/<stream>`` (each round of a rep stream).
A use of n F128s reads ceil(n/4) coins of 64 bytes, and every count is public: the statement's schedule
(`Flock.Firewall.Session.planOf`), which `flock-firewall` prints from the circuit and public files, `hello` its link
points (``points * m``) and `session --schedule` its rounds. So a family's count is the sum of its uses' coins.

The layout only bounds the uses. A family too small refuses an honest session's use; coins past the uses stay
unreleased, and v2 refuses any release no use reads (`coins.exactly_reason`). The layout is fixed before the draw, and
a session's table grows with the units it proves (m = k_log + max(3, ceil(log2 ceil(n/g)))), so a reserve made before the
draw sizes each slot by `slot_families` over every statement the slot may prove, the population's (draw None) among
them."""
from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from verity.core.protocols.sampled_proofs import challenger as Ch

TREE = Path(__file__).resolve().parents[2]
#: the Lean firewall the sessions' provers run behind (`lake build flock-firewall` in the verifier)
FIREWALL = TREE / "verity" / "core" / "service" / "flock" / ".lake" / "build" / "bin" / "flock-firewall"
#: the stream of a session's link points (`Coins.SESSION`), which names no rep stream
SESSION = "session"

#: a statement's schedule: its link points' F128s, and each opening after them as (stream, F128s) in the prover's order
Schedule = tuple[int, list[tuple[str, int]]]


def coins(n: int) -> int:
    """The 64-byte coins a use of `n` F128s reads."""
    return -(-n // 4)


def schedule(flags: Sequence[str], draw: Mapping[str, Any] | None = None, firewall: Path = FIREWALL) -> Schedule:
    """The schedule of the statement `flags` name (`flock-firewall`'s statement flags: ``--statement``, ``--circuit``,
    ``--public``, ``--partition``, ``--program``, ...) at the verifier's draw `draw` (None: the population's): its
    `Hello`'s link points and `session --schedule`'s openings. Raises ValueError with the firewall's refusal."""
    text = json.dumps(draw).encode()
    p = subprocess.run([str(firewall), "hello", *flags], input=b"draw %d\n" % len(text) + text, capture_output=True)
    head, _, rest = p.stdout.partition(b"\n")
    if not head.startswith(b"ok "):
        raise ValueError(f"flock-firewall hello: {(head or p.stderr).decode(errors='replace').strip()}")
    link = json.loads(rest[:int(head[3:])])["link"]
    p = subprocess.run([str(firewall), "session", "--schedule", *flags], input=text, capture_output=True)
    lines = p.stdout.decode().splitlines()
    if not lines or not lines[0].startswith("ok ") or len(lines) != 1 + int(lines[0][3:]):
        raise ValueError(f"flock-firewall session --schedule: {(lines[0] if lines else p.stderr.decode()).strip()}")
    return link["points"] * link["m"], [(s, int(n)) for s, n in (x.rsplit(" ", 1) for x in lines[1:])]


def slot_families(slot: int, *schedules: Schedule) -> list[list[Any]]:
    """Session `slot`'s families, each at its largest count over `schedules`: ``slot/<slot>/session``, the coins of its
    link points, then ``slot/<slot>/<stream>`` for each stream in schedule order, the sum of its rounds' coins. A family
    of no coins is left out. Raises ValueError for no schedule, or a stream named as the link points' family."""
    if not schedules:
        raise ValueError("a slot's families come from at least one schedule")
    out = {f"slot/{slot}/{SESSION}": 0}
    for links, plan in schedules:
        one = {f"slot/{slot}/{SESSION}": coins(links)}
        for stream, n in plan:
            if stream == SESSION:
                raise ValueError(f"a stream is named {SESSION}, the link points' family")
            one[f"slot/{slot}/{stream}"] = one.get(f"slot/{slot}/{stream}", 0) + coins(n)
        for label, n in one.items():
            out[label] = max(out.get(label, 0), n)
    return [[label, n] for label, n in out.items() if n]


def window_layout(scheme: str, draws: Sequence[Sequence[Any]], *slots: Sequence[Sequence[Any]]) -> list[list[Any]]:
    """The window's layout: the draws' families (`a5_coins.families`), then each slot's (`slot_families`). Raises
    ValueError when it is no live coin layout of `scheme` (`challenger.layout_problem`: labels distinct, at most
    `challenger.MAX_COINS` coins)."""
    layout = [list(f) for f in draws] + [list(f) for s in slots for f in s]
    why = Ch.layout_problem(scheme, 0, tuple(tuple(f) for f in layout))
    if why is not None:
        raise ValueError(f"the window's layout {layout}: {why}")
    return layout
