"""A call's public table's rows as H1's served window registers them (`pouw_h1_served.Served._pubs`): `PublicRows`, built
again on read, on the host or on a torch device, rather than held."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from verity.core.protocols.pouw import core_check as CC


@dataclass(frozen=True)
class PublicRows(Sequence):
    """A call's public table's rows (`Served._pubs`), built again on read rather than held: row t `core_check.public_words`
    of the window's Program ``salt``, position u0 + t and the call's two ``roots``, and with ``blocks`` ``(M, K, (R, C),
    q)`` a J10-R call's u0 and unit t's blocks (`core_check.j10r_blocks` of unit (t // q, t % q), as
    `h1window.call_blocks` lays them). Items as `pouw_h1_registered.Rows` gives them; `words` builds rows of a NumPy
    array there, `tensor` every row on a torch device."""

    salt: bytes
    roots: tuple[bytes, bytes]
    u0: int
    n: int
    blocks: tuple[int, int, tuple[int, int], int] | None = None

    def __post_init__(self) -> None:
        if any(len(x) % 4 for x in (self.salt, *self.roots)) or len(self._base()) * 32 != (
                CC.PUBLIC_ROW_BITS if self.blocks is None else CC.PUBLIC_ROW_J10R_BITS):
            raise ValueError(f"a public row is {CC.PUBLIC_ROW_BITS} bits, a J10-R unit's {CC.PUBLIC_ROW_J10R_BITS}, "
                             "its position on a u32 word")

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, j: Any) -> Any:
        if isinstance(j, slice):
            return self.words(np.arange(self.n)[j])
        if not -self.n <= j < self.n:
            raise IndexError(j)
        return tuple(int(v) for v in self.words(np.array([j % self.n]))[0])

    def at(self, positions: Sequence[int] | np.ndarray) -> np.ndarray:
        """Rows ``positions`` (each in [0, n)) as their words ``(·, W)`` u16."""
        at = np.asarray(positions, dtype=np.int64)
        if at.size and (int(at.min()) < 0 or int(at.max()) >= self.n):
            raise IndexError(f"a position outside the table's {self.n} rows")
        return self.words(at)

    def words(self, t: np.ndarray) -> np.ndarray:
        """Rows ``t`` (unit indices, int64) as ``(len(t), W)`` u16 words."""
        out = self._fill(np.repeat(self._base()[None], len(t), axis=0), np.asarray(t, dtype=np.int64))
        return out.astype("<u4").view("<u2").astype(np.uint16)

    def tensor(self, device: Any) -> Any:
        """Every row as an ``(n, W)`` u16 tensor on torch ``device``, written on its current stream; the only copy to it is
        the one row of what's common to all of them, from pinned memory, so the host waits on nothing there."""
        import torch

        base = torch.from_numpy(self._base())
        base = (base.pin_memory() if torch.device(device).type == "cuda" else base).to(device, non_blocking=True)
        out = self._fill(base.expand(self.n, -1).clone(), torch.arange(self.n, dtype=torch.int64, device=device))
        return out.view(torch.uint8).view(self.n, -1, 8)[:, :, :4].contiguous().view(torch.uint16).view(self.n, -1)

    def _base(self) -> np.ndarray:
        """The row's u32 words (`core_check.public_row`'s 4-byte groups, each big-endian) with the position, and a J10-R
        call's u0 and blocks, zero; as int64."""
        tail = bytes(24) if self.blocks is not None else b""
        return np.frombuffer(self.salt + bytes(8) + self.roots[0] + self.roots[1] + tail, ">u4").astype(np.int64)

    def _fill(self, out: Any, t: Any) -> Any:
        """``out`` (rows ``t``'s `_base`, a NumPy array or a tensor) with each row's own: its position, and a J10-R call's
        u0 and blocks, each u64 its high word then its low."""
        cols = [(len(self.salt) // 4, self.u0 + t)]
        if self.blocks is not None:
            m, k, tile, q = self.blocks
            b0, b1 = CC.j10r_blocks(m, k, tile, t // q, t % q)
            at = (len(self.salt) + 8 + len(self.roots[0]) + len(self.roots[1])) // 4
            cols += [(at, self.u0), (at + 2, b0), (at + 4, b1)]
        for i, v in cols:
            out[:, i], out[:, i + 1] = v >> 32, v & 0xFFFFFFFF
        return out


__all__ = ["PublicRows"]
