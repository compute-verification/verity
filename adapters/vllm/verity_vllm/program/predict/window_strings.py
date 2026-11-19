"""The strings a served window's close hands `Window.close` (``strings``): each of its `leaf_rows` ports' salts, unit u's the
stream of the window's key at index (template, port), leaf u (`salt_stream.fill`, one call a port), and its rows' ``b || c``
under them: a row the serve acquired hashed where it is committed (`window_plan.stage`; on a card native_rows.cu's
`verity_rows_h_launch`, one parameter build and one launch a port), every other row on the host from the window's recorded
rows. `HostTwin` computes the card's strings on the processor."""
from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable, Collection, Mapping, Sequence
from typing import Any

import numpy as np
from verity.commitments import hm96
from verity.commitments.cuda.salt_stream.engine import LEAF_BYTES, fill
from verity.commitments.rowleaf import (
    SCHEMA_HM96_SHA512_ROW_V2,
    sha512_row_v2_digest,
    sha512_row_v2_prefix,
)

from .window_plan import stage

__all__ = ["Device", "HostTwin", "acquired", "leaf_strings", "strings_of"]

INST = hm96.SHA512
KEY = INST.default_key
BC_BYTES = 128


def _stride(lb: int) -> int:
    return -(-lb // 16) * 16


class HostTwin:
    """The card's strings on the processor: numpy rows and salts, the salts from the host salt library
    (`salt_stream.host_lib`), each row's ``b || c`` its v2 row digest's hm96 string."""

    def __init__(self, lib: Any) -> None:
        self.lib = lib

    def salts(self, key: Any, index: tuple[int, int], n: int) -> np.ndarray:
        out = np.empty((n, LEAF_BYTES), dtype=np.uint8)
        fill(self.lib, out, key, index, 0)
        return out

    def rows(self, rows: Sequence[Any], stride: int) -> np.ndarray:
        out = np.zeros((len(rows), stride), dtype=np.uint8)
        for i, r in enumerate(rows):
            r = np.asarray(r, dtype=np.uint8).reshape(-1)
            out[i, :len(r)] = r
        return out

    def strings(self, rows: np.ndarray, lb: int, ys: np.ndarray) -> np.ndarray:
        return host_strings(rows, lb, ys)

    def host(self, x: Any) -> np.ndarray:
        return np.asarray(x)


class Device:
    """The same on a CUDA device: the salts from the device salt library ``lib`` (`salt_stream.device_lib`) into device
    memory, the strings native_rows.cu's with the port's parameters built once (`RowStep`), on the device's current stream."""

    def __init__(self, lib: Any, device: Any) -> None:
        import torch

        from verity_vllm.commit import step_rows
        from verity_vllm.commit.committer.native_ext import ext

        self.lib, self.device, self.ext = lib, torch.device(device), ext()
        self.hasher = self.ext.StepHasher()
        self.tn = torch.frombuffer(bytearray(step_rows.t_nibbles(KEY)), dtype=torch.uint8).to(self.device)

    def _stream(self) -> int:
        import torch

        return int(torch.cuda.current_stream(self.device).cuda_stream)

    def salts(self, key: Any, index: tuple[int, int], n: int) -> Any:
        import torch

        out = torch.empty((n, LEAF_BYTES), dtype=torch.uint8, device=self.device)
        fill(self.lib, out, key, index, 0, stream=self._stream())
        return out

    def rows(self, rows: Sequence[Any], stride: int) -> Any:
        import torch

        out = torch.zeros((len(rows), stride), dtype=torch.uint8, device=self.device)
        for i, r in enumerate(rows):
            r = r.reshape(-1)
            out[i, :r.numel()] = r
        return out

    def strings(self, rows: Any, lb: int, ys: Any) -> Any:
        # the domain id and the salt key are unread: no frame leaf is made, and every row's salt is given
        rs = self.ext.RowStep(bytes(64), bytes(64), sha512_row_v2_prefix(8 * lb), INST.salt_prefix, INST.leaf_prefix(KEY),
                              SCHEMA_HM96_SHA512_ROW_V2, lb)
        n, stride = int(rows.shape[0]), int(rows.shape[1])
        return self.hasher.rows_h(rs, rows, n * stride, 0, self.tn, self._stream(), stride=stride, salts=ys.contiguous(),
                                  want_bc=True)[1]

    def host(self, x: Any) -> np.ndarray:
        return x.cpu().numpy()


def host_strings(rows: np.ndarray, lb: int, ys: np.ndarray) -> np.ndarray:
    """``(m, 128)`` uint8: each row's first ``lb`` bytes' ``b || c`` under its salt, as the kernel computes it."""
    out = b"".join(INST.commit_string(KEY, sha512_row_v2_digest(bytes(r[:lb]), 8 * lb), bytes(y)) for r, y in zip(rows, ys))
    return np.frombuffer(out, dtype=np.uint8).reshape(len(rows), BC_BYTES)


def acquired(plan: Any, block: Callable[[int, str], tuple[Any, int]], ports: Collection) -> dict[tuple[str, str], dict[int, Any]]:
    """By each of ``ports`` (template id, output) a served node feeds (`Plan.feeds`), its acquired units' rows where they are
    committed (`stage`), by unit."""
    out: dict[tuple[str, str], dict[int, Any]] = {}
    for f in plan.feeds:
        if (f.tid, f.out) in ports:
            rows = stage(f, block)
            out.setdefault((f.tid, f.out), {}).update({f.unit + i: rows[i] for i in range(f.rows)})
    return out


def leaf_strings(leaf_rows: Mapping[tuple[str, str], Any], key: Any, hasher: Any,
                 staged: Mapping[tuple[str, str], Mapping[int, Any]]) -> tuple[dict[tuple[str, str], tuple[bytes, bytes]], dict]:
    """Each `leaf_rows` port's salts and strings (`Window.close`'s ``strings``) under ``key``: a unit in ``staged`` hashed
    by ``hasher`` from its acquired row, every other unit on the host from its recorded row (`LeafRows.rows`); a port of a
    schema other than ``hm96-sha512/row/v2``, which the kernel's row hash is, refused by name. -> the strings and their cost:
    units hashed where acquired and on the host, and the host's seconds."""
    out: dict[tuple[str, str], tuple[bytes, bytes]] = {}
    cost = {"acquired": 0, "host": 0, "host_s": 0.0}
    for k, lr in leaf_rows.items():
        if lr.schema != SCHEMA_HM96_SHA512_ROW_V2:
            raise ValueError(f"strings: {k} is {lr.schema}, and the row hash computes {SCHEMA_HM96_SHA512_ROW_V2}'s strings alone")
        units, rows = list(lr.units), lr.rows
        lb = 2 * int(rows.shape[1])
        ys = hasher.salts(key, (int(lr.template), int(lr.port)), units[-1] + 1)[units]
        hy, bc = hasher.host(ys), np.empty((len(units), BC_BYTES), dtype=np.uint8)
        fed = staged.get(k, {})
        at = [i for i, u in enumerate(units) if u in fed]
        if at:
            bc[at] = hasher.host(hasher.strings(hasher.rows([fed[units[i]] for i in at], _stride(lb)), lb, ys[at]))
        if rest := [i for i, u in enumerate(units) if u not in fed]:
            t0 = time.perf_counter()
            r = np.zeros((len(rest), lb), dtype=np.uint8)
            r[:] = np.ascontiguousarray(rows[rest]).astype("<u2").view(np.uint8)
            bc[rest] = host_strings(r, lb, hy[rest])
            cost["host_s"] += time.perf_counter() - t0
        cost["acquired"], cost["host"] = cost["acquired"] + len(at), cost["host"] + len(rest)
        out[k] = (hy.tobytes(), bc.tobytes())
    cost["host_s"] = round(cost["host_s"], 3)
    return out, cost


def strings_of(win: Any, key: Any, hasher: Any, plan: Any = None,
               block: Callable[[int, str], tuple[Any, int]] | None = None) -> dict[tuple[str, str], tuple[bytes, bytes]]:
    """`leaf_strings` of ``win``'s `leaf_rows`, the units ``plan`` feeds staged from ``block``; prints one
    ``WINDOW-STRINGS`` line, its cost."""
    lrs = win.leaf_rows()
    staged = acquired(plan, block, lrs) if plan is not None and block is not None else {}
    strings, cost = leaf_strings(lrs, key, hasher, staged)
    sys.stderr.write("WINDOW-STRINGS " + json.dumps({"ports": len(strings), **cost}) + "\n")
    sys.stderr.flush()
    return strings
