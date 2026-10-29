"""serve-openai's PoUW H1 arm's draw (`serve_h1`): how many tile units a window Program holds (`units`), which proof
blocks a closed window's draw coin draws (`drawn_units`), and what each opens of the server's Commit, as the draw-gated
release keeps them (`drawn_reads`)."""
from __future__ import annotations

import bisect
from collections.abc import Mapping, Sequence
from typing import Any


def units(shapes: Any, rows: int, steps: int) -> int:
    """The tile units of the window Program `served.declared` builds over ``shapes`` (each ``(n, K)``) at T ``rows`` and S
    ``steps``: the members of its batches, one a call (S × the linears), so the descriptor, the declare and the digest
    grow with the calls, not the units."""
    from verity.core.protocols.pouw.circuit import h1window as HW

    from verity_vllm.protocols.pouw import h1 as PH

    r, c = PH.TILE
    return int(steps) * (HW.slot(int(rows), PH.TILE) // r) * sum(int(n) // c for n, _k in shapes)


def drawn_units(commits: Mapping[str, Any], layout: Mapping[str, Any], k: int, stream: str) -> list[int]:
    """The proof blocks a closed window's draw coin draws at K ``k`` (`credit.drawn` over the window Program's
    ``layout``, `credit.layout_of`'s, on stream ``stream``, registration the draw root and receipt its receipts, as
    `pouw_h1_challenger.drawn_of` draws them), from its commitment ``commits``. ValueError when it opened no draw coin or
    core refuses the draw."""
    from verity.core.protocols.pouw import credit as CR
    from verity.core.protocols.sampled_proofs.challenger import LiveOpening

    draw = commits.get("draw")                              # None: the window committed no draw root
    if draw is None or draw.get("coin") is None:
        raise ValueError("the window opened no draw coin" + (f" ({draw['refused']})" if draw and draw.get("refused") else ""))
    got, why = CR.drawn(LiveOpening.from_json(draw["coin"]).coin.hex(), stream, draw.get("root"),
                        {"receipts": draw.get("receipts")}, dict(layout), k)
    if why is not None:
        raise ValueError(f"{why[0]}: {why[1]}")
    return list(got)


def drawn_reads(tap: Any, layout: Mapping[str, Any], drawn: Sequence[int],
                s_a_bytes: int) -> list[tuple[int, str, int, int]]:
    """What the proof blocks ``drawn`` of the window ``layout`` open of the server's Commit, as the draw-gated release takes
    them (`serve_release.Read`: engine step, member, byte lo, hi): unit u, tile t of call c, opens A rows R·I .. R·I + R - 1
    of its call, I = t // (n/C) its row block (`h1window.table_reads`), which are the Commit's x_q (K bytes a row) and s_a
    (``s_a_bytes`` a row) rows of the call's site at the engine step its step maps to, as H1's tap ``tap`` registered them
    (``where``), under the member names the Commit acquired them by (``acquired``), cut to the rows it acquired; a call of
    an empty step opens none. KeyError on a drawn call the Commit registered no rows of."""
    (r, cw), m = layout["tile"], int(layout["m"])
    ks = {name: (int(k), int(n)) for name, k, n in layout["linears"]}
    firsts, at = [0], 0
    for lin, _blk in layout["calls"]:
        at += (m // r) * (ks[lin][1] // cw)
        firsts.append(at)
    calls: dict[int, tuple[int | None, str, int]] = {}
    for w in tap.where.values():
        if "calls" in w:                                   # an A table's rows; a round's ids are no call's
            for c, j, s, n in zip(w["calls"], w["steps"], w["sites"], w["rows"]):
                calls[int(c)] = (j, s, int(n))
    out: set[tuple[int, str, int, int]] = set()
    for u in drawn:
        c = bisect.bisect_right(firsts, u) - 1
        if not 0 <= c < len(layout["calls"]):
            raise KeyError(f"unit {u} is no unit of the window's {firsts[-1]}")
        k, n = ks[layout["calls"][c][0]]
        i = (u - firsts[c]) // (n // cw)
        if c not in calls:
            raise KeyError(f"unit {u}'s call {c}: the Commit registered no rows of it, so what it opens can't be kept")
        j, site, rows = calls[c]
        lo, hi = r * i, min(r * (i + 1), rows)
        if j is None or hi <= lo:
            continue
        x_q, s_a = (tap.acquired[(int(j), site, port)] for port in ("0", "1"))
        out |= {(int(j), x_q, lo * k, hi * k), (int(j), s_a, lo * s_a_bytes, hi * s_a_bytes)}
    return sorted(out)
