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
no field: a slot's ``ids`` are its C cells' whatever the cap. `scan_state` gives a window those rows off the scan of the
warden's ingress rows, with the request each slot's Program starts.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from verity.core.protocols.certifier import ingress_scan as IS
from verity.ml.circuits.defs import bind
from verity.ml.definitions.registry import boolean_io_shell as BIO

#: the registered value whose rows are the scan's states, one a bucket of the window (`ingress_scan.window_state`'s)
SCAN_STATE = IS.SCAN_STATE
#: the network certifier's cells at the class of record (w63, `benchmarks/network_traces/scan_zk.W63`): B-byte cells, r a
#: connection a bucket, C cells a request, ids of IDB bits, a request's ingress within WITHIN buckets
B, R, C, IDB, WITHIN = 512, 4, 47, 17, 12


def scan_class(lanes: int, outputs: int, every: int, prompt: int = 1024) -> IS.Class:
    """The scan's class at a window class's lanes, outputs (KO), timestep and prompt ids (KI), on the class of record's
    cells; its delay the least it takes, WITHIN + EVERY − 1 (27 at EVERY 16, w63's)."""
    return IS.Class(B=B, r=R, lanes=lanes, C=C, IDB=IDB, EVERY=every, WITHIN=WITHIN, D=WITHIN + every - 1, KO=outputs,
                    KI=prompt)


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


def _scan(scan: IS.Class, ingress: Sequence[bytes], outputs: Mapping[int, Sequence[int]]) -> list[IS.State]:
    got = IS.scan(scan, list(ingress), {int(q): tuple(v) for q, v in outputs.items()})
    if got is None:
        raise ValueError(f"the scan at {scan} refuses the warden's ingress rows to bucket {len(ingress) - 1}: a request it "
                         f"can't hold (more than {scan.KI} prompt ids or {scan.KO} outputs) or rows that aren't the "
                         "warden's")
    return got[0]


def scan_anchors(cls: Any, ingress: Sequence[bytes], outputs: Mapping[int, Sequence[int]]) -> dict[int, int]:
    """Each request's anchor in the scan of ``ingress`` by the window class ``cls`` (a `window_circuit.WindowClass`, which
    imports this module), as `scan_state` scans it: the bucket whose Programs start it (its slot's
    ``occ`` bit 0 set in the state before it), by its ordinal q. ValueError where the scan refuses the rows."""
    out: dict[int, int] = {}
    for t, st in enumerate(_scan(cls.scan, ingress, outputs)):
        for s in st.slots:
            if s.occ & 1:
                out.setdefault(s.q, t + 1)
    return out


def scan_state(cls: Any, ingress: Sequence[bytes], outputs: Mapping[int, Sequence[int]],
               t0: int = 0) -> tuple[list[list[int]], list[int | None]]:
    """The network certifier's scan as a window of ``cls`` (a `window_circuit.WindowClass`) from scan bucket ``t0`` reads
    it: its state rows
    (`window_rows.state_value`'s ``rows``, `ingress_scan.window_state`'s) and the request each class Program starts,
    slot k's the ordinal q of the request whose Program starts at its anchor in its lane (``occ``'s bit 0 set in the row
    there, `ingress_scan.programs`' read), None for a slot no request starts. The scan is of the warden's registered
    ingress rows ``ingress`` (a row a bucket, `warden_rows.window_rows`) from the empty state at bucket 0 to the window's
    last row, each request's Program giving its output ids ``outputs[q]``. ValueError where the scan refuses the rows."""
    scan, n = cls.scan, t0 + cls.buckets - 1
    if len(ingress) < n:
        raise ValueError(f"the window from scan bucket {t0} reads the scan's states to bucket {n - 1}: give the ingress "
                         f"rows to it, {n}, not {len(ingress)}")
    states = _scan(scan, ingress[:n], outputs)
    rows = IS.window_state(scan, [IS.encode(scan, s) for s in states], t0, cls.buckets)
    at = {a: states[t0 + a - 1] if t0 + a else IS.empty(scan) for a in {cls.anchor(k) for k in range(cls.slots)}}
    starts = []
    for k in range(cls.slots):
        s = at[cls.anchor(k)].slots[cls.lane(k)]
        starts.append(s.q if s.occ & 1 else None)
    return rows, starts


__all__ = ["SCAN_STATE", "bridge", "scan_anchors", "scan_class", "scan_state", "words"]
