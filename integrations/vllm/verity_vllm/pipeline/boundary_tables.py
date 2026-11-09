"""A two-stage window's history tables (`boundary_rows`): each finished request's latent rows at each layer, one
`commit.history_tables` table a (span, layer), committed at serve beside its ``b.ingress`` and ``b.egress``.

**Names.** A table is named from public items alone (`name`): its window's run, its span's class (P, O) and slot r in the
class, and its layer, never the engine's id for the request, which only the prover's index holds. So the challenger
derives every table's name and domain from the class's ``commit.json``, as it does each boundary value's.
"""
from __future__ import annotations

from verity_vllm.commit import history_tables as H

__all__ = ["KEY", "key", "name"]

#: a span's key in its tables' names (`history_tables.name`): the window's run, the class and the slot
KEY = "{run}/p{prompt}-o{outputs}/s{slot}"


def key(run: str, prompt: int, outputs: int, slot: int) -> str:
    """Slot ``slot``'s key in class (``prompt``, ``outputs``) of the window whose run is ``run``."""
    for k, v in (("prompt", prompt), ("outputs", outputs), ("slot", slot)):
        if type(v) is not int or v < 0:
            raise ValueError(f"a table's {k} is a natural number, not {v!r}")
    if not isinstance(run, str) or not run:
        raise ValueError("a table's name binds its window's run: give it")
    return KEY.format(run=run, prompt=prompt, outputs=outputs, slot=slot)


def name(run: str, prompt: int, outputs: int, slot: int, layer: int) -> str:
    """The table of slot ``slot`` of class (``prompt``, ``outputs``) in the window of run ``run`` at layer ``layer``."""
    return H.name(key(run, prompt, outputs, slot), layer)
