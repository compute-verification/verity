"""A class's own Definition on its boundary commitment (`boundary_rows`): its signature (`class_signature`) and the names
round 2 gives its ends (`ends`).

**Its ends.** The class's own Definition has its boundary values as its ends: its parameters are a span's `b.request`
and `b.prompt` rows and it returns the span's `b.egress` row, so round 2's own derivation links a template's port that
reads a parameter to the value of its name, and the template output it returns to the name round 2 gives a return.

**Laid out under it** (`boundary_rows.Layout.own`), a class has it, whole, as its replay block and commits those three
values and no other: `b.request`, `b.prompt` and, under the return's name, `b.egress`'s ids, each in the rows round 2's
link reads of it (`Layout.cuts`: a prompt read an id at a time is P rows a span), audited with the templates its body
calls. A parameter of any other name is the interior's, as a stand-in's all are. It takes no stop rule, whose check is
the Definition's own to make, and no history tables.
"""
from __future__ import annotations

from collections.abc import Iterable

from verity.circuits.types import Array
from verity_vllm.pipeline.boundary_rows import EGRESS, PARAMS, class_size, linked, return_port, row_type


def class_signature(prompt: int, outputs: int) -> tuple[tuple[tuple[str, Array], ...], Array]:
    """The signature of a class's Definition at (P, O) = (``prompt``, ``outputs``): its parameters `PARAMS`, a span's
    `b.request` and `b.prompt` rows, and its return, the span's `b.egress` row (`row_type`). Round 2 then links a
    template's port that reads one of them to the class's value of that name, and the template output it returns to
    the name `ends` gives `b.egress`."""
    if not (class_size(prompt) and class_size(outputs)):
        raise ValueError(f"a class's Definition is at a class's (P, O), each a class's size (`class_size`), not "
                         f"({prompt!r}, {outputs!r})")
    return tuple((v, row_type(v, prompt)) for v in PARAMS), row_type(EGRESS, outputs)


def ends(block: str, ports: Iterable[str] = ()) -> dict[str, str]:
    """Each boundary value a class's Definition ``block`` (`class_signature`) reads or returns, and the name round 2
    gives the value its link reads in a record whose other ports are ``ports`` (`linked`): `b.request` and `b.prompt`
    their own, and `b.egress` the return's (`return_port`)."""
    ports = tuple(ports)
    return {**{v: linked(block, v, ports) for v in PARAMS}, EGRESS: return_port(block, ports)}


__all__ = ["class_signature", "ends"]
