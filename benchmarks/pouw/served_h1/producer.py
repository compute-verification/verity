"""h1u.cu's units kernel for the served H1 receipt producer (`verity_vllm.protocols.pouw.h1.receipts.Receipts`, the served
path's `Served.receipts`), and `check`, which holds the receipts it kept to `h1tile.reference`.

`Units` wraps `kh_seeds`, `kh_unit` and `kh_units_multi` of an -hbc build (`benchmarks/hash/dither/sbench.sh`'s
h1u_kc.so). `Receipts(served, units=Units(so), device="cuda")` computes each call's receipts with it, word for word the
torch chain's (`pouw_h1_receipts`, the reference the units kernels are gated against): a call at a time, or a step's
calls of one K in a few `kh_units_multi` launches (`Receipts.prepare`), each launch's codes copied to the device and its
receipts back on streams of their own, beside the launches before and after. The kernel needs 512 | K (`kh_unit`); its
``live`` mask is a v4 window's dead tiles (`Receipts`' ``skip_dead``).

`check` seeds its reference from each unit's public row, row t of the call's ``pub@<c>`` as the verifier builds it
(`credit.public_table`, `Receipts.public`): its salt, position and roots (`public_fields`).

**J10-R.** A window Program over J10-R's unit (`h1window.is_j10r`: `h1tile.PouwH1Tile_v5`, ungated) takes J10-R's
receipts: the call's one seed seed_c (`seeds.call_seed` at the call's first unit u0, its ``call``) and one stream at call
scale, from which each unit gathers its rows and columns (`h1tile.j10r_noise`), its fold keys σ_R's at u0 + u. The units
kernel computes them with `kh_seeds_j10r` and `kh_units_multi_j10r` (h1u.cu's), each call's row blocks hashed once into a
workspace of the build's size (dither's `j10r_ws.py`). Unit u0 + t's public row then holds its call u0 and its ``blocks``
too (`h1window.call_blocks`, `j10r_fields`), and `check` seeds `h1tile.j10r_reference` from them.
"""
from __future__ import annotations

import ctypes as C
import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import torch

from verity.protocols.pouw.verifier import core_check as CC
from verity.protocols.pouw.prover.circuit import h1tile as HT
from verity_vllm.protocols.pouw.h1 import receipts as RC

HERE = Path(__file__).resolve().parent
_ws_spec = importlib.util.spec_from_file_location("dither_j10r_ws", HERE.parents[1] / "hash" / "dither" / "j10r_ws.py")
WS = importlib.util.module_from_spec(_ws_spec)
_ws_spec.loader.exec_module(WS)


#: a public row's salt bytes: the row's bytes before its position and two roots
_SALT = CC.PUBLIC_ROW_BITS // 8 - 8 - 128


def _row_bytes(words: Any) -> bytes:
    if len(words) * 16 not in (CC.PUBLIC_ROW_BITS, CC.PUBLIC_ROW_J10R_BITS):
        raise ValueError(f"a public row is {CC.PUBLIC_ROW_BITS // 16} 16-bit words (a J10-R unit's "
                         f"{CC.PUBLIC_ROW_J10R_BITS // 16}), not {len(words)}")
    x = b"".join(int(w).to_bytes(2, "little") for w in words)
    return b"".join(x[i:i + 4][::-1] for i in range(0, len(x), 4))


def public_fields(words: Any) -> tuple[bytes, int, tuple[bytes, bytes]]:
    """A unit's public row (`core_check.public_words`, a ``row/v2`` row of 16-bit leaves, or a J10-R unit's) -> the salt,
    position and (root(A), root(B)) it carries (`core_check.public_row`), its seeds' inputs. ValueError on a row of
    another width."""
    x, s = _row_bytes(words), _SALT
    return x[:s], int.from_bytes(x[s:s + 8], "big"), (x[s + 8:s + 72], x[s + 72:s + 136])


def j10r_fields(words: Any) -> tuple[int, tuple[int, int]] | None:
    """A J10-R unit's public row -> the ``call`` and ``blocks`` it carries past `public_fields`' (`core_check.public_row`'s
    ``j10r``); None for a v1 row."""
    x = _row_bytes(words)
    if len(x) * 8 == CC.PUBLIC_ROW_BITS:
        return None
    call, b0, b1 = (int.from_bytes(x[_SALT + 136 + 8 * i:_SALT + 144 + 8 * i], "big") for i in range(3))
    return call, (b0, b1)


def _stream(t: torch.Tensor) -> int:
    return torch.cuda.current_stream(t.device).cuda_stream


#: h1u.cu's `UCall`, `kh_units_multi`'s record of one call: its A, B, seeds and receipts' device addresses, N, K, its first
#: unit's index in the launch and n_digits(K / 32)
UCALL = np.dtype([("A", "<u8"), ("B", "<u8"), ("seeds", "<u8"), ("out", "<u8"),
                  ("N", "<u4"), ("K", "<u4"), ("u0", "<u4"), ("md", "<u4")])
#: h1u.cu's `UCallR`, `kh_units_multi_j10r`'s record of one call: `UCALL`'s with its workspace ``rws`` (the call's row
#: blocks of its stream, and an H1U_JW build's column blocks and W' after them: `j10r_ws.workspace_bytes(M, N, K)`
#: bytes, the build's own size)
UCALLR = np.dtype([("A", "<u8"), ("B", "<u8"), ("seeds", "<u8"), ("out", "<u8"), ("rws", "<u8"),
                   ("N", "<u4"), ("K", "<u4"), ("u0", "<u4"), ("md", "<u4")])


class Units:
    """h1u.cu's `kh_seeds`, `kh_unit` and `kh_units_multi` from an -hbc build: a call's units' seeds and receipts, or the
    receipts of many calls in one launch (`multi`), on torch's current stream, every unit live (``live`` NULL: v1's
    ungated unit) unless given a row block's live bits; and J10-R's (``j10r``: `kh_seeds_j10r`, `kh_units_multi_j10r`)."""

    def __init__(self, lib: str | Path | C.CDLL):
        h = lib if isinstance(lib, C.CDLL) else C.CDLL(str(lib))
        vp, u32 = C.c_void_p, C.c_uint
        h.kh_seeds.argtypes = [vp, C.c_ulonglong, u32, vp, vp]
        h.kh_seeds.restype = C.c_int
        h.kh_unit.argtypes = [vp, vp, u32, u32, u32, vp, vp, vp, vp]
        h.kh_unit.restype = C.c_int
        h.kh_units_multi.argtypes = [vp, u32, u32, u32, vp, vp]
        h.kh_units_multi.restype = C.c_int
        for f in (h.kh_words, h.kh_digits):
            f.argtypes, f.restype = [u32], u32
        h.kh_ucall_bytes.argtypes, h.kh_ucall_bytes.restype = [], u32
        if h.kh_ucall_bytes() != UCALL.itemsize:
            raise ValueError(f"h1u.cu's UCall is {h.kh_ucall_bytes()} bytes, not the {UCALL.itemsize} `UCALL` lays out")
        self.j10r = hasattr(h, "kh_units_multi_j10r")
        if self.j10r:
            h.kh_seeds_j10r.argtypes = [vp, C.c_ulonglong, C.c_ulonglong, u32, vp, vp]
            h.kh_seeds_j10r.restype = C.c_int
            h.kh_units_multi_j10r.argtypes = [vp, u32, u32, u32, vp, C.c_int, vp]
            h.kh_units_multi_j10r.restype = C.c_int
            h.kh_ucallr_bytes.argtypes, h.kh_ucallr_bytes.restype = [], u32
            if h.kh_ucallr_bytes() != UCALLR.itemsize:
                raise ValueError(f"h1u.cu's UCallR is {h.kh_ucallr_bytes()} bytes, not the {UCALLR.itemsize} `UCALLR` "
                                 "lays out")
            WS.flags(h)
        self.h = h

    def words(self, k: int) -> int:
        return int(self.h.kh_words(k))

    def seeds(self, record: torch.Tensor, base: int, units: int, out: torch.Tensor | None = None,
              call: int | None = None) -> torch.Tensor:
        """(units, 16) int64 on ``record``'s device (into ``out``, if given): unit u's seed_j then σ_R, big-endian words, at
        position base + u, from ``record`` (`pouw_h1_receipts.seed_record`, (30,) int64). With ``call`` (J10-R,
        `kh_seeds_j10r`, from `seed_record`'s ``j10r`` record): unit u's seed_c, its call's at the call's first position
        ``call``, then σ_R."""
        if tuple(record.shape) != (30,) or record.dtype != torch.int64 or not record.is_cuda:
            raise ValueError(f"kh_seeds: a (30,) int64 record on the device, not {record.dtype} {tuple(record.shape)}")
        if out is None:
            out = torch.empty(units, 16, dtype=torch.int64, device=record.device)
        elif tuple(out.shape) != (units, 16) or out.dtype != torch.int64 or out.device != record.device \
                or not out.is_contiguous():
            raise ValueError(f"kh_seeds: out is {out.dtype} {tuple(out.shape)} on {out.device}, not ({units}, 16) int64")
        if call is None:
            rc = self.h.kh_seeds(record.contiguous().data_ptr(), int(base), int(units), out.data_ptr(), _stream(record))
        else:
            self._need_j10r()
            rc = self.h.kh_seeds_j10r(record.contiguous().data_ptr(), int(call), int(base), int(units), out.data_ptr(),
                                      _stream(record))
        if rc:
            raise RuntimeError(f"kh_seeds{'' if call is None else '_j10r'} returned {rc} (U = {units})")
        return out

    def _need_j10r(self) -> None:
        if not self.j10r:
            raise ValueError("this h1u.cu build has no J10-R units (kh_seeds_j10r, kh_units_multi_j10r)")

    def multi(self, calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
              live: torch.Tensor | None = None, j10r: bool = False) -> torch.Tensor:
        """Every unit of ``calls``, each ``(A [M, K], B [N, K], seeds)`` as `receipts` takes them, every A M rows of one K,
        in one launch (`kh_units_multi`) -> their receipts (sum of units, kh_words(K)) int32, call i's from the sum of the
        units before it, each word for word its `receipts`. ``live`` as `receipts`', every call's. ``j10r``: J10-R's units
        (`kh_units_multi_j10r`, each call's seeds `seeds`' with its ``call``), each call's row blocks hashed once into a
        workspace of its own, of the build's size for its N (`j10r_ws.workspace_bytes`: an H1U_JW build's column blocks
        and W' too); `j10r_ws.WorkspaceError` where the build and that size disagree."""
        if not calls:
            raise ValueError("kh_units_multi: no calls")
        if j10r:
            self._need_j10r()
        (m, k), dev = calls[0][0].shape, calls[0][0].device
        n_of, rec = [], np.zeros(len(calls) + 1, UCALLR if j10r else UCALL)
        for i, (a, b, seeds) in enumerate(calls):
            n = b.shape[0]
            units = (m // RC.R) * (n // RC.C_)
            if tuple(a.shape) != (m, k) or (a.dtype, b.dtype) != (torch.int8, torch.int8) or tuple(b.shape) != (n, k) \
                    or m % RC.R or n % RC.C_ or a.device != dev or b.device != dev or not dev.type == "cuda" \
                    or tuple(seeds.shape) != (units, 16) or seeds.dtype != torch.int64 or seeds.device != dev \
                    or not (a.is_contiguous() and b.is_contiguous() and seeds.is_contiguous()):
                raise ValueError(f"kh_units_multi: call {i}'s A {a.dtype} {tuple(a.shape)}, B {b.dtype} {tuple(b.shape)}, "
                                 f"seeds {tuple(seeds.shape)}: contiguous s8 rows on one device, every A {m} × {k}, 32 | "
                                 "M, N, and a unit's seeds each")
            rec[i]["A"], rec[i]["B"], rec[i]["seeds"] = a.data_ptr(), b.data_ptr(), seeds.data_ptr()
            rec[i]["N"], rec[i]["K"], rec[i]["md"], rec[i]["u0"] = n, k, self.h.kh_digits(k), sum(n_of)
            n_of.append(units)
        if k % 512 or rec[0]["md"] not in (2, 3):
            raise ValueError(f"kh_units_multi: no units kernel at K = {k} (512 | K, n_digits(K / 32) 2 or 3)")
        if live is not None and (live.dtype != torch.uint8 or tuple(live.shape) != (m // RC.R,) or live.device != dev
                                 or not live.is_contiguous()):
            raise ValueError(f"kh_units_multi: live is {live.dtype} {tuple(live.shape)} on {live.device}, not a contiguous "
                             f"uint8 bit a row block, {m // RC.R}, on {dev}")
        total = rec[-1]["u0"] = sum(n_of)
        out = torch.empty(total, self.words(k), dtype=torch.int32, device=dev)
        for i in range(len(calls)):
            rec[i]["out"] = out[int(rec[i]["u0"])].data_ptr() if n_of[i] else out.data_ptr()
        if j10r:
            per = [WS.workspace_bytes(self.h, m, int(rec[i]["N"]), k) for i in range(len(calls))]
            rws = torch.empty(sum(per), dtype=torch.uint8, device=dev)
            for i in range(len(calls)):
                rec[i]["rws"] = rws.data_ptr() + sum(per[:i])
        table = torch.from_numpy(rec.view(np.uint8)).pin_memory().to(dev, non_blocking=True)
        lv = None if live is None else live.data_ptr()
        if j10r:
            rc = self.h.kh_units_multi_j10r(table.data_ptr(), len(calls), total, m, lv, 0, _stream(out))
        else:
            rc = self.h.kh_units_multi(table.data_ptr(), len(calls), total, m, lv, _stream(out))
        if rc:
            raise RuntimeError(f"kh_units_multi{'_j10r' if j10r else ''} returned {rc} ({len(calls)} calls, {total} units, "
                               f"M = {m}, K = {k})")
        return out

    def receipts(self, a: torch.Tensor, b: torch.Tensor, seeds: torch.Tensor,
                 live: torch.Tensor | None = None, j10r: bool = False) -> torch.Tensor:
        """A [M, K] and B [N, K] s8 on the device and the units' seeds (`seeds`) -> receipts (units, kh_words(K)) int32,
        unit u row block u // (N / 32) and column block u % (N / 32). ``live``: (M / 32,) uint8 on the device, a row
        block's units zero words, no noise derived, where its bit is 0 (v4's dead tile). ValueError on operands no call's
        or a K the kernel has no instance of (512 | K and n_digits(K / 32) 2 or 3). ``j10r``: J10-R's units, `multi`'s of
        the one call."""
        if j10r:
            return self.multi([(a, b, seeds)], live, j10r=True)
        (m, k), (n, kb) = a.shape, b.shape
        units = (m // RC.R) * (n // RC.C_)
        if (a.dtype, b.dtype) != (torch.int8, torch.int8) or kb != k or m % RC.R or n % RC.C_ or not (a.is_cuda and b.is_cuda) \
                or tuple(seeds.shape) != (units, 16) or not (a.is_contiguous() and b.is_contiguous()):
            raise ValueError(f"kh_unit: A {a.dtype} {tuple(a.shape)}, B {b.dtype} {tuple(b.shape)}, seeds {tuple(seeds.shape)}: "
                             "contiguous s8 rows on the device, 32 | M, N, and a unit's seeds each")
        if live is not None and (live.dtype != torch.uint8 or tuple(live.shape) != (m // RC.R,) or live.device != a.device
                                 or not live.is_contiguous()):
            raise ValueError(f"kh_unit: live is {live.dtype} {tuple(live.shape)} on {live.device}, not a contiguous uint8 "
                             f"bit a row block, {m // RC.R}, on {a.device}")
        out = torch.empty(units, self.words(k), dtype=torch.int32, device=a.device)
        rc = self.h.kh_unit(a.data_ptr(), b.data_ptr(), m, n, k, seeds.contiguous().data_ptr(),
                            None if live is None else live.data_ptr(), out.data_ptr(), _stream(a))
        if rc == 2:
            raise ValueError(f"kh_unit: no units kernel at K = {k} (512 | K, n_digits(K / 32) 2 or 3)")
        if rc:
            raise RuntimeError(f"kh_unit returned {rc} (M = {m}, N = {n}, K = {k})")
        return out


def check(kept: dict[str, Any], units: list[int]) -> dict[str, Any]:
    """A kept call's (`Receipts.kept`) receipts of ``units`` against `h1tile.reference` on the same operands, seeded from
    each unit's public row as the verifier builds it (``pub``, `public_fields`), word for word, v4's under the served rows'
    admit bits where the producer skipped dead tiles (``skip``), J10-R's (`h1tile.j10r_reference`, from the row's call and
    blocks, `j10r_fields`) where it computed J10-R's (``j10r``): ``{unit: {"live": its row block holds a served row,
    "pub": the row is the producer's salt, position and roots (and under J10-R the call's first position and the tile's
    `h1tile.j10r_blocks`), "ok": bool}}``."""
    ncb, out = kept["b"].shape[0] // RC.C_, {}
    m, k = kept["a"].shape
    for u in units:
        i, j = divmod(u, ncb)
        a = (kept["a"][RC.R * i:RC.R * (i + 1)].to(torch.int64) & 0xFF).tolist()
        b = (kept["b"][RC.C_ * j:RC.C_ * (j + 1)].to(torch.int64) & 0xFF).tolist()
        salt, position, roots = public_fields(kept["pub"][u])
        pub = (salt, position, roots) == (kept["salt"], kept["base"] + u, kept["roots"])
        extra = j10r_fields(kept["pub"][u])
        if kept.get("j10r"):
            if extra is None:
                pub, want = False, None
            else:
                pub = pub and extra == (kept["base"], HT.j10r_blocks(m, k, (RC.R, RC.C_), i, j))
                _z, want = HT.j10r_reference(a, b, salt, position, roots, extra[0], extra[1], pairs=[(0, 0)])
        else:
            pub = pub and extra is None
            gated = [int(RC.R * i + r < kept["m"]) for r in range(RC.R)] if kept.get("skip") else None
            _z, want = HT.reference(a, b, salt, position, roots, pairs=[(0, 0)], gated=gated)
        out[u] = {"live": RC.R * i < kept["m"], "pub": pub, "ok": kept["receipts"][u].tolist() == want}
    return out


__all__ = ["Units", "check", "j10r_fields", "public_fields"]
