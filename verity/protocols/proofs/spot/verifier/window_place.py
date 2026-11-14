"""A sample's place in its window's session, and the family of that session its challenges are read from.

A window's challenger session reserves one family of challenges a sample, and a family's challenges follow one
registration (`coins.FAMILY_ROOT`). Which family a sample reads is no prover's file's to say: it is the sample's place,
a number its registration record's window fixes (`of`), the family is ``draw/<place>`` with the place in decimal
(`family`), and the session is the one opened on the window's run (`run`). Whoever reserves a session's families
reserves by place, and whoever reads a sample again names its family by place and refuses another. This is the one
Python definition of a place: every caller uses it, and a stream of any sampling design holds its challenge source
to it where it samples a registration, a two-stage stream at its first stage (`service.Stream.select`).

    place = ((prompt · 2^32 + outputs) · 2 + stopped) · 2 + blocks

read from four items of the record's window and no other, as a serving system's adapter writes them: ``prompt`` and
``outputs``, the two sizes of the class the record registers, each a natural number below 2^32 (`SIZES`); ``stopped``,
1 where it is true (the class is a stop class) and 0 where it is false or absent; and ``population``, 1 where it is
"stop" (the record registers the class's stop blocks) and 0 where absent (it registers the class's spans). So two
classes of one window have two places, and a stop class's spans and its stop blocks have two. A window with any other
word in one of the four has no place, and neither has one that is no object.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

#: a class's two sizes are each below this, so that its place (`of`) names it and no other class
SIZES = 1 << 32
#: the name a window's samples' families begin with
FAMILY = "draw"


def of(window: Any) -> int:
    """The place of the sample a record registers, from the record's window (the module's formula). ValueError for a
    window that is no object, a size that is no natural number below 2^32, a ``stopped`` the window names that is
    neither true nor false, a ``population`` it names that is not "stop", and the stop blocks of a class that is no
    stop class."""
    if not isinstance(window, Mapping):
        raise ValueError(f"a record's window is an object, not {type(window).__name__}: it has no place in a window's "
                         f"session")
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


def family(window: Any) -> str:
    """The family of its window's session that the sample a record registers reads its challenges from: the one of its
    place (`of`), ``draw/<place>`` with the place in decimal. ValueError as `of`."""
    return f"{FAMILY}/{of(window)}"


def run(window: Any) -> str | None:
    """The root the window's session is opened on: its ``run``, where that is a string; None otherwise."""
    r = window.get("run") if isinstance(window, Mapping) else None
    return r if isinstance(r, str) else None
