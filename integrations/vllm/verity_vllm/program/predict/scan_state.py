"""The network certifier's scan state as a window class's Programs read it (VLLM-126, form (a)): the one place the window's
lowering knows its layout.

The Programs at a boundary b read the scan's state row S_{b − 1} (`ingress_scan`'s "The token-id row"), registered as
`SCAN_STATE`'s row b − 1: NS u16 words, its fields bit-packed least significant bit first in `layout`'s order, slot j's
``j.<field>``. A slot's Program reads its ``occ`` (a 0/1 word: 1 when the slot carries a request whose Program starts
at b), its prompt's length ``fields``, its ``max_tokens`` and its ``ids`` (prompt position p at
`io_shell.position`'s index), through `io_shell.ScanSlots_v1`, which is wires (`bridge`). The circuit reads the window's
rows from the scan's row t0 − 1 on, t0 the window's first bucket: its row a is the scan's row t0 + a − 1, the one the
Programs anchored at the window's bucket a read.

`layout` is network-accounting's PR2 form as its docstring states it (VLLM-126): `ingress_scan.slot_layout` with ``occ``
between ``anchor`` and ``n``, a 0/1 word. The scan this tree's `ingress_scan` runs has no ``occ`` yet; once PR2 lands,
`layout` is its own (`ingress_scan.layout`).
"""
from __future__ import annotations

from typing import Any

from verity.core.protocols.certifier import ingress_scan as IS
from verity.ml.circuits.defs import bind
from verity.ml.definitions.registry import io_shell as IO

#: the registered value whose rows are the scan's states, one a bucket
SCAN_STATE = "verity/certifier/scan-state/v1"
#: the network certifier run of record's cells (w63, `benchmarks/network_traces/scan_zk.W63`): B-byte cells, r a connection a bucket, C
#: cells a request, ids of IDB bits, a request's ingress within WITHIN buckets
B, R, C, IDB, WITHIN = 512, 4, 47, 17, 12


def scan_class(lanes: int, outputs: int, every: int) -> IS.Class:
    """The scan's class at a window class's lanes, outputs (KO) and timestep, on the run of record's cells; its delay
    the least it takes, WITHIN + EVERY − 1 (27 at EVERY 16, w63's)."""
    return IS.Class(B=B, r=R, lanes=lanes, C=C, IDB=IDB, EVERY=every, WITHIN=WITHIN, D=WITHIN + every - 1, KO=outputs)


def _slot_layout(cls: IS.Class) -> tuple[tuple[str, int], ...]:
    fields = IS.slot_layout(cls)
    at = [k for k, _ in fields].index("n")
    return fields[:at] + (("occ", 32),) + fields[at:]


def layout(cls: IS.Class) -> dict[str, tuple[int, int]]:
    """Each field's (bit offset, width) in the state row: the head's, slot j's as ``"j.name"``, then ``pad`` to whole words."""
    out, at = {}, 0
    for name, w in IS.HEAD:
        out[name], at = (at, w), at + w
    for j in range(cls.NR):
        for name, w in _slot_layout(cls):
            out[f"{j}.{name}"], at = (at, w), at + w
    words = -(-at // 16)
    out["pad"] = (at, 16 * words - at)
    return out


def words(cls: IS.Class) -> int:
    """The state row's u16 words."""
    pad = layout(cls)["pad"]
    return (pad[0] + pad[1]) // 16


def bridge(cls: IS.Class, prompt: int) -> Any:
    """`io_shell.ScanSlots_v1` at ``cls``'s state row for prompts of at most ``prompt`` ids: each lane's occupancy word,
    prompt length, ``max_tokens`` and ids where `layout` puts them."""
    lay = layout(cls)
    at = {f: tuple(lay[f"{j}.{f}"][0] for j in range(cls.NR)) for f in ("occ", "fields", "max_tokens", "ids")}
    return bind(IO.ScanSlots, NS=words(cls), NP=prompt, IDB=cls.IDB, K0=cls.K0, K=cls.K, OCC=at["occ"], LP=at["fields"],
                MAX=at["max_tokens"], IDS=at["ids"])


__all__ = ["SCAN_STATE", "bridge", "layout", "scan_class", "words"]
