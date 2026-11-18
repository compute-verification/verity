"""The masked versions of the decode units that read the key cache, for a class's Program (`class_window.class_request`).

A decode step past the request's stop is switched off by one bit, its step's ``live``: a unit that reads the cache's
``kv`` writes zeros there, whatever ``kv`` holds, so nothing a dead step computes reads the cache, and every unit after it
on the step reads only zeros, the same-step query and the chain's state. Two units of a key block read ``kv``:

- `SparseMlaHeadBlock_v1`, a whole key block (the chain `node_window_sparse` lowers): masked as
  `SparseMlaHeadBlockLive_v1`, its running max, lanes and O selected by ``live``;
- `SparseMlaHeadKeysWire_v1`, a column block's writer in the chain cut at P (`sparse_mla.block_units`), the one unit of
  the cut that reads the key rows: masked as `SparseMlaHeadKeysWireLive_v1`, its rows selected by ``live``.

`SparseMlaHeadScoresCols_v1` and `SparseMlaHeadValues_v1` read the writer's output row by wire, never the cache, so they
keep their Definitions: past the stop they read the zeros it writes."""
from __future__ import annotations

from typing import TYPE_CHECKING

from verity.circuits.defs import bind

from verity_vllm.definitions.registry import sparse_mla as SM

if TYPE_CHECKING:
    from verity.circuits.defs import SpecializedDefinition

#: the masked units' added parameter, the last
LIVE = SM.LIVE
#: each family that reads the key cache and its masked version, a registered Definition at the same statics
MASKED = {"SparseMlaHeadBlock_v1": SM.SparseMlaHeadBlockLive, "SparseMlaHeadKeysWire_v1": SM.SparseMlaHeadKeysWireLive}


def masked(fn: SpecializedDefinition) -> SpecializedDefinition:
    """The masked version of ``fn``, a unit of a `MASKED` family at any of its statics: its masked family bound at ``fn``'s
    statics, with ``fn``'s parameters and then `LIVE`, one row of 16 bits (`sparse_mla.LIVE_WORD`: the step before's
    select's ``o1`` as a row holds it, the step's bit at bit 0), and its outputs ``fn``'s where that bit is 1 and zeros
    where it is 0, word for word. Refused: any other family, since every other decode unit reads no cache row and keeps its
    Definition."""
    family = f"{fn.defn.name}_v{fn.defn.version}"
    if family not in MASKED:
        raise ValueError(f"masked takes a unit that reads the key cache ({', '.join(MASKED)}), not {fn.id}: every other "
                         f"decode unit keeps its Definition")
    return bind(MASKED[family], **fn.bindings)


__all__ = ["LIVE", "MASKED", "masked"]
