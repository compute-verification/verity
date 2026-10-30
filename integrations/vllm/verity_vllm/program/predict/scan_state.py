"""The network certifier's scan state as a window class's Programs read it: the one place the window's lowering knows it.

The Programs at a boundary b read the scan's state row S_{b − 1} (`ingress_scan`'s "The token-id row"), the empty
state's at b = 0: NS u16 words, its fields bit-packed least significant bit first in `ingress_scan.layout`'s order, slot
j's ``j.<field>``. A slot's Program reads its ``occ`` (a 0/1 word: 1 when the slot carries a request whose Program starts
at b), its prompt's length ``fields``, its ``max_tokens`` and its ``ids`` (prompt position p at `io_shell.position`'s
index), through `boolean_io_shell.ScanSlots_v2` (`bridge`), which also gives each lane its first position's token and the
slot it carries on. The window's registered `SCAN_STATE` is the rows
`ingress_scan.window_state` gives for its link-window: for the window of ``buckets`` buckets from scan bucket t0, the
scan's rows t0 − 1 … t0 + buckets − 2, its row a S_{t0 + a − 1}, the one the Programs anchored at the window's bucket a
read, and window 0's row 0 the empty state's, a constant rather than a scan unit's. The scan's prompt cap ``KI`` changes
no field: a slot's ``ids`` are its C cells' whatever the cap.
"""
from __future__ import annotations

from typing import Any

from verity.ml.protocols.certifier import ingress_scan as IS
from verity.ml.circuits.defs import bind
from verity.ml.definitions.registry import boolean_io_shell as BIO

#: the registered value whose rows are the scan's states, one a bucket of the window (`ingress_scan.window_state`'s)
SCAN_STATE = IS.SCAN_STATE
#: the network certifier's cells at the class of record (w63, `benchmarks/network_traces/scan_zk.W63`): B-byte cells, r a
#: connection a bucket, C cells a request, ids of IDB bits, a request's ingress within WITHIN buckets
B, R, C, IDB, WITHIN = 512, 4, 47, 17, 12


def scan_class(lanes: int, outputs: int, every: int) -> IS.Class:
    """The scan's class at a window class's lanes, outputs (KO) and timestep, on the class of record's cells; its delay
    the least it takes, WITHIN + EVERY − 1 (27 at EVERY 16, w63's)."""
    return IS.Class(B=B, r=R, lanes=lanes, C=C, IDB=IDB, EVERY=every, WITHIN=WITHIN, D=WITHIN + every - 1, KO=outputs)


def words(cls: IS.Class) -> int:
    """The state row's u16 words."""
    return IS.state_bits(cls) // 16


def bridge(cls: IS.Class, prompt: int) -> Any:
    """`boolean_io_shell.ScanSlots_v2` at ``cls``'s state row for prompts of at most ``prompt`` ids: each lane's occupancy
    word, prompt length, ``max_tokens`` and ids where `ingress_scan.layout` puts them."""
    lay = IS.layout(cls)
    at = {f: tuple(lay[f"{j}.{f}"][0] for j in range(cls.NR)) for f in ("occ", "fields", "max_tokens", "ids")}
    return bind(BIO.ScanSlotsV2, NS=words(cls), NP=prompt, IDB=cls.IDB, K0=cls.K0, K=cls.K, OCC=at["occ"], LP=at["fields"],
                MAX=at["max_tokens"], IDS=at["ids"])


__all__ = ["SCAN_STATE", "bridge", "scan_class", "words"]
