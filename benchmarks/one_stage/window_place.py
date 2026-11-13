"""A sample's place in its window's session, and the family of that session its challenges are read from.

A window's challenger session reserves one family of challenges a sample, and a family's challenges follow one
registration (`coins.FAMILY_ROOT`). Which family a sample reads is no file's to say: it is the sample's place, a number
its record's window fixes (`of`), and the family is ``draw/<place>`` with the place in decimal (`family`). The first
stage reserves by it (`two_stage._first_all`), and whoever reads a sample again names the family by it and refuses a
hand-off that names another (`two_stage.resume`, `stop_audit.resume`, `stop_audit.link`). This is the one Python
definition of a place: every caller uses it, and the Lean verifier's is held to it row by row.

    place = ((prompt · 2^32 + outputs) · 2 + stopped) · 2 + blocks

read from four items of the record's window and no other: ``prompt`` and ``outputs``, the class's two sizes, each a
natural number below 2^32 (`SIZES`); ``stopped``, 1 where it is true (a stop class) and 0 where it is false or absent;
and ``population``, 1 where it is "stop" (a stop record's, `boundary_stop.stop_record`: the sample is of the class's
stop blocks) and 0 where absent (the sample is of the class's spans). So two classes of one window have two places,
and a stop class's spans and its stop blocks have two. A window with any other word in one of the four has no place.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

#: a class's two sizes are each below this, so that its place (`of`) names it and no other class
SIZES = 1 << 32


def of(window: Mapping[str, Any]) -> int:
    """The place of the sample a record registers, from the record's window (the module's formula). ValueError for a
    size that is no natural number below 2^32, for a ``stopped`` the window names that is neither true nor false, for
    a ``population`` it names that is not "stop", and for the stop blocks of a class that is no stop class."""
    prompt, outputs = window.get("prompt"), window.get("outputs")
    if not all(type(v) is int and 0 <= v < SIZES for v in (prompt, outputs)):
        raise ValueError(f"class (prompt {prompt!r}, outputs {outputs!r}) has no place in a window's session: a place "
                         f"names two sizes, natural numbers below {SIZES}")
    stopped, blocks = window.get("stopped", False), "population" in window
    if type(stopped) is not bool or (blocks and window["population"] != "stop"):
        raise ValueError(f"class (prompt {prompt}, outputs {outputs}) has no place in a window's session: where its "
                         f"window names them, ``stopped`` is true or false and ``population`` is \"stop\"")
    if blocks and not stopped:
        raise ValueError(f"class (prompt {prompt}, outputs {outputs}) is no stop class: it has no stop blocks, and no "
                         f"place for a sample of them")
    return ((prompt * SIZES + outputs) * 2 + int(stopped)) * 2 + int(blocks)


def of_layout(layout: Any, blocks: bool = False) -> int:
    """A class's place (`of`), of the sample of its spans or, with ``blocks``, of its stop blocks: the place the window
    of its boundary record names (`boundary_rows.Layout.provenance`), or of its stop record."""
    return of({"prompt": layout.prompt, "outputs": layout.outputs, "stopped": layout.stopped,
               **({"population": "stop"} if blocks else {})})


def family(window: Mapping[str, Any]) -> str:
    """The family of its window's session that the sample a record registers reads its challenges from: the one of its
    place (`of`), ``draw/<place>`` with the place in decimal (`a5_coins.family`)."""
    import a5_coins as AC

    return AC.family(of(window))


def family_of(layout: Any, blocks: bool = False) -> str:
    """A class's family (`family`), of the sample of its spans or, with ``blocks``, of its stop blocks. The first
    stage reserves it and samples from it, and whoever reads the sample again names it from the class's layout, never
    from a file the prover wrote."""
    import a5_coins as AC

    return AC.family(of_layout(layout, blocks))


def another_family(named: Any, mine: str) -> str:
    """Why a hand-off that names the family ``named`` for a sample is refused, the sample's place's being ``mine``."""
    return (f"the hand-off names the family {named!r} for the sample's challenges, and the sample's place in its "
            f"window's session is the family {mine!r}: a class is sampled once, on its own family, and a hand-off does "
            f"not choose among the families the session reserved")
