"""`tensors.H1Device` over torch and the CUDA kernels (`H1`): what the PyTorch adapter hands PoUW's serving code
(`tensors.use_h1`). Each method is its module's: `registered_device` (the registered rows' hashes), `tables_device` (the
call tables), `records_device` (the records' check where they are) and `credit_gate_device` (the credit gate's trace)."""
from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any

import numpy as np

from verity_torch.pouw import credit_gate_device as CGD
from verity_torch.pouw.h1 import records_device as ZRD
from verity_torch.pouw.h1 import registered_device as RD
from verity_torch.pouw.h1 import tables_device as TD


class TorchH1Device:
    """`tensors.H1Device` on torch tensors and this process's CUDA device."""

    def available(self) -> bool:
        return RD.available()

    def device_hash(self, words: Any, key: bytes, nonce: int, row_salts: bytes | None, domain_id: bytes,
                    public: tuple[bytes, int, bool] | None = None, also: tuple[bytes, int] | None = None) -> Any:
        return RD.device_hash(words, key, nonce, row_salts, domain_id, public, also)

    def device_public_salts(self, head: bytes, first: int, n: int, ok: bool = False) -> bytes:
        return RD.device_public_salts(head, first, n, ok)

    def hashing(self) -> AbstractContextManager[Any]:
        return RD.hashing()

    def commit_tables(self, run: Any, call: Any, a: Any, s_a: Any, records: np.ndarray | None, zy: Any = None) -> None:
        TD.commit_tables(run, call, a, s_a, records, zy)

    def settle(self, run: Any, wait: bool) -> None:
        TD.settle(run, wait)

    def flush(self, run: Any) -> None:
        TD.flush(run)

    def take_z(self, run: Any, call: Any, words: Any) -> tuple[bytes, int] | None:
        return TD.take_z(run, call, words)

    def row_words(self, t: Any, m: int, dtype: str) -> np.ndarray:
        return TD.row_words(t, m, dtype)

    def holds_on(self, zy: Any, rows: int, n: int, y: Any, s_a: Any, s_w: Any) -> bool:
        return ZRD.holds_on(zy, rows, n, y, s_a, s_w)

    def trace(self, linear: str) -> AbstractContextManager[Any]:
        return CGD.Trace(linear)

    def outside(self) -> AbstractContextManager[None]:
        return CGD.outside()


H1 = TorchH1Device()

__all__ = ["H1", "TorchH1Device"]
