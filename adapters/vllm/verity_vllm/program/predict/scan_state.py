"""The network attester's scan state as a window class's Programs read it: the one place the window's lowering knows it.

The Programs at a boundary b read the scan's state row S_{b − 1} (`ingress_scan`'s "The token-id row"), the empty
state's at b = 0: NS u16 words, its fields bit-packed least significant bit first in `ingress_scan.layout`'s order, slot
j's ``j.<field>``. A slot's Program reads its ``occ`` (a 0/1 word: 1 when the slot carries a request whose Program starts
at b), its prompt's length ``fields``, its ``max_tokens`` and its ``ids`` (prompt position p at `io_shell.position`'s
index), through `boolean_io_shell.ScanSlots_v3` (`bridge`), which also gives each lane its first position's token and the
slot it carries on. The window's registered `SCAN_STATE` is the rows
`ingress_scan.window_state` gives for its window: for the window of ``buckets`` buckets from scan bucket t0, the
scan's rows t0 − 1 … t0 + buckets − 2, its row a S_{t0 + a − 1}, the one the Programs anchored at the window's bucket a
read, and window 0's row 0 the empty state's, a constant rather than a scan unit's. The scan's prompt bound ``KI`` changes
no field: a slot's ``ids`` are its C cells' whatever the bound. The network attester's prover computes those rows
(`network.prover.scan_state_rows`).
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from typing import Any

from verity.circuits.defs import bind
from verity.protocols.resource_accounting import scheme as schemes

from verity_vllm import messages
from verity_vllm.definitions.registry import boolean_io_shell as BIO

#: the network attester's cells at the class of record (w63,
#: `verity.protocols.resource_accounting.network.prover.scan_zk.W63`): B-byte cells, r a connection a bucket, C cells a
#: request, ids of IDB bits, a request's ingress within WITHIN buckets
B, R, C, IDB, WITHIN = 512, 4, 47, 17, 12
#: the port whose layout the state row is, looked up on the declared scheme that has it (`messages.declaring`)
PORT = "scan-state"


@dataclass(frozen=True)
class Scan:
    """The scan's class as a window's lowering reads it: ``shape``, its statics by name (an attribute each), and
    ``layout``, the state row's `scheme.Layout` at them, ``K0`` and ``K`` its counts."""

    shape: Mapping[str, int]
    layout: schemes.Layout

    def __getattr__(self, name: str) -> int:
        if name in ("shape", "layout") or name not in self.__dict__.get("shape", {}):
            raise AttributeError(name)
        return self.shape[name]

    @property
    def K0(self) -> int:
        return self.layout.counts["K0"]

    @property
    def K(self) -> int:
        return self.layout.counts["K"]


@cache
def _port() -> schemes.Port:
    return next(p for p in messages.declaring(PORT).ports if p.name == PORT)


def __getattr__(name: str) -> str:
    if name == "SCAN_STATE":
        return _port().value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


@cache
def _layout(shape: tuple[tuple[str, int], ...]) -> schemes.Layout:
    return schemes.layout(messages.declaring(PORT), PORT, dict(shape))


def scan_class(lanes: int, outputs: int, every: int, prompt: int = 1024) -> Scan:
    """The scan's class at a window class's lanes, outputs (KO), timestep and prompt ids (KI), on the class of record's
    cells; its delay the least it takes, WITHIN + EVERY − 1 (27 at EVERY 16, w63's)."""
    shape = {"B": B, "r": R, "lanes": lanes, "C": C, "IDB": IDB, "EVERY": every, "WITHIN": WITHIN, "D": WITHIN + every - 1,
             "KO": outputs, "KI": prompt}
    return Scan(shape, _layout(tuple(shape.items())))


def words(scan: Scan) -> int:
    """The state row's u16 words."""
    return scan.layout.bits // 16


def bridge(scan: Scan, prompt: int) -> Any:
    """`boolean_io_shell.ScanSlots_v3` at ``scan``'s state row for prompts of at most ``prompt`` ids: each lane's occupancy
    word, prompt length, ``max_tokens`` and ids where its layout puts them."""
    lay = scan.layout.fields
    lanes = sum(1 for k in lay if k.endswith(".occ"))
    at = {f: tuple(lay[f"{j}.{f}"][0] for j in range(lanes)) for f in ("occ", "fields", "max_tokens", "ids")}
    return bind(BIO.ScanSlotsV3, NS=words(scan), NP=prompt, IDB=scan.IDB, K0=scan.K0, K=scan.K, OCC=at["occ"],
                LP=at["fields"], MAX=at["max_tokens"], IDS=at["ids"])


__all__ = ["PORT", "SCAN_STATE", "Scan", "bridge", "scan_class", "words"]
