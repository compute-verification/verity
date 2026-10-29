"""Kernels' producer of vllm's `Served.receipts` (BRIDGE2) on the H1 served path
(`integrations/vllm/verity_vllm/protocols/pouw/h1/served.py`): each call's receipts, one a unit of the call in the
window Program's tile order, computed by h1u.cu's units kernel (`Units`: `kh_seeds` and `kh_unit` of an -hbc build,
`benchmarks/hash/dither/sbench.sh`'s h1u_kc.so) or, with no kernel, by receipts.py's torch chain (the reference the units
kernels are gated against).

Call c of the window (`pouw_h1.Call`) at contraction K, as the Program's units read it (`h1window.program`):

* A is the call's M slot rows (`Served.m`, the Program's slot: its served codes, then zero rows) and B its linear's N
  weight rows; unit u of the call is row block u // (N / 32) and column block u % (N / 32), at position u0 + u, u0 the
  call's first unit of the window Program (`h1window.call_units`), the position its statement's public row carries;
* its seeds (`seeds.seed`, `seeds.receipt_seed`) take the window Program's salt (`H1Run.salt`) and the call's roots
  (`Receipts.roots`): root(A) its step's A table ``a/<K>@<s>``'s and root(B) ``b/<K>``'s, the Registered row/v2 roots
  `Served` committed (rows under `audit.hm96_salt`, domains `registered.domain`), not served_h1's AROW roots
  (`commit.py`). `Served` asks for a step's receipts of K at its last call of K, once both are registered. Salt,
  position and roots are unit u0 + t's public row, row t of the call's ``pub@<c>`` that the verifier builds itself
  (`credit.public_table`, `Receipts.public`), and `check` seeds its reference from those rows (`public_fields`);
* the unit is v1's (`h1tile.PouwH1Tile`, `h1window.tile_unit`'s): no admit gating, so a row block past the served rows
  has the receipt its zero codes and its noise give it, not v4's zero receipt.

With ``skip_dead`` (``H1_SKIP_DEAD=1``; off by default) the units of a row block wholly past the call's served rows are
skipped: such a tile's admit bits are all 0, and its row is `h1tile.PouwH1Tile_v4`'s dead-tile receipt, zero words with no
noise derived (`kh_unit`'s ``live`` mask, receipts.py's `gated_receipts`). A row block holding a served row is computed in
full, and the receipts keep one row a unit in tile order. It stays off while the window Program's unit is v1, whose dead
tile has its noise's receipt, not zero.

`Served` hands a step's calls of one K to `Receipts.prepare` before it asks each of them. With the units kernel, the
producer computes all of their units in a few `kh_units_multi` launches (``launches``), with one copy of a launch's codes
to the device and one of its receipts back, each on a stream of its own and beside the launches before and after, so a
call's ask returns its rows with no launch of its own once its launch's copy is done. A call asked without `prepare` is
computed on its own, to the same words.

It returns (units, 2 (KS + m)) uint32 (KS = K / 32, m = `segment.n_digits(KS)`: 260 words at K = 4096, 902 at 14336), the
input of `pouw_h1_served.receipt_words`, which `Served` refuses at any other shape. It takes no coin: the noise is the
salt's and the unit's position's. The kernel needs 512 | K (`kh_unit`); the torch chain any 32 | K, K >= 64.

**J10-R.** A window Program over J10-R's unit (`h1window.is_j10r`: `h1tile.PouwH1Tile_v5`, ungated) takes J10-R's
receipts: the call's one seed seed_c (`seeds.call_seed` at the call's first unit u0, its ``call``) and one stream at call
scale, from which each unit gathers its rows and columns (`h1tile.j10r_noise`), its fold keys σ_R's at u0 + u. The units
kernel computes them with `kh_seeds_j10r` and `kh_units_multi_j10r` (h1u.cu's), the torch chain with receipts.py's
``j10r`` (each call's stream from `noise.cu`'s `h1_noise_calls` when ``cubin`` is given, else hashlib). Unit u0 + t's
public row then holds its call u0 and its ``blocks`` too (`h1window.call_blocks`), and `check` seeds
`h1tile.j10r_reference` from them. ``skip_dead`` is refused there: v5 has no dead tile.
"""
from __future__ import annotations

import ctypes as C
import importlib.util
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import receipts as RC  # noqa: E402

_ws_spec = importlib.util.spec_from_file_location("dither_j10r_ws", HERE.parents[1] / "hash" / "dither" / "j10r_ws.py")
WS = importlib.util.module_from_spec(_ws_spec)
_ws_spec.loader.exec_module(WS)

from verity.core.protocols.pouw import core_check as CC  # noqa: E402
from verity.core.protocols.pouw import credit as CR  # noqa: E402
from verity.core.protocols.pouw.circuit import h1tile as HT  # noqa: E402
from verity.core.protocols.pouw.circuit import h1window as HW  # noqa: E402
from verity.core.protocols.pouw.circuit import seeds as SD  # noqa: E402
from verity_config.settings import of  # noqa: E402

settings = of(__file__)


def _be(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=">u8").astype(np.uint64)


def seed_record(salt: bytes, root_a: bytes, root_b: bytes, j10r: bool = False) -> np.ndarray:
    """`kh_seeds`' data, 30 int64: SEED_TAG and RECEIPT_TAG (3 big-endian u64 words each), then the salt, root(A) and
    root(B) (8 each); with `j10r`, `kh_seeds_j10r`'s, CALL_SEED_TAG in SEED_TAG's place. ValueError unless each of the
    three is 64 bytes."""
    for name, v in (("salt", salt), ("root(A)", root_a), ("root(B)", root_b)):
        if len(v) != 64:
            raise ValueError(f"a unit's seeds take a 64-byte {name}, not {len(v)} bytes")
    tag = SD.CALL_SEED_TAG if j10r else SD.SEED_TAG
    return np.concatenate([_be(tag), _be(SD.RECEIPT_TAG), _be(salt), _be(root_a), _be(root_b)]).view(np.int64)


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
        position base + u, from ``record`` (`seed_record`, (30,) int64). With ``call`` (J10-R, `kh_seeds_j10r`, from
        `seed_record`'s ``j10r`` record): unit u's seed_c, its call's at the call's first position ``call``, then σ_R."""
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


class Receipts:
    """`Served.receipts` for the window ``served`` (`pouw_h1_served.Served`) of the run ``run`` (`pouw_h1.H1Run`, its
    salt; `bind` when the run is made after the producer): ``(call, codes, s_a, weight codes, s_w)`` -> the call's
    receipts (units, 2 (KS + m)) uint32 (the module docstring). With ``units`` (`Units`) on ``device`` (CUDA), else
    receipts.py's torch chain on ``device`` (the CPU by default).

    ``weight(call)``: the call's B [N, K] s8 on ``device`` (the serving engine's own int8 weight, so the GPU holds no second
    copy); by default the weight codes `Served` hands over, copied to ``device`` once a linear. ``roots(call)``: the call's
    (root(A), root(B)), by default `Receipts.roots`. ``keep(call)``: keep that call's operands, roots and receipts in
    ``kept`` (by call index) for a check against `h1tile.reference` (`check`). ``skip_dead``: skip the units of row blocks
    wholly past the served rows (the module docstring), by default ``H1_SKIP_DEAD=1`` in the environment.
    ``launches``: `prepare` splits a batch into that many launches of whole calls, about equal in units, each one's codes
    staged and copied to the device while the one before computes, and its receipts copied back while the one after does.
    ``cubin``: the torch chain's noise from `noise.cu` (`receipts.Cubin`) on ``device``, else hashlib's. ``j10r``: J10-R's
    receipts (the module docstring), by default the window Program's form (`h1window.is_j10r`)."""

    def __init__(self, served: Any, run: Any = None, units: Units | None = None, device: Any = None,
                 weight: Callable[[Any], torch.Tensor] | None = None,
                 roots: Callable[[Any], tuple[bytes, bytes]] | None = None, keep: Callable[[Any], bool] | None = None,
                 skip_dead: bool | None = None, launches: int = 4, cubin: Any = None, j10r: bool | None = None):
        if launches < 1:
            raise ValueError(f"a batch takes at least one launch, not {launches}")
        self.served, self.run, self.units, self.launches, self.cubin = served, run, units, launches, cubin
        self._j10r = j10r
        self.skip_dead = settings.H1_SKIP_DEAD if skip_dead is None else bool(skip_dead)
        d = torch.device(device if device is not None else ("cuda" if units is not None or cubin is not None else "cpu"))
        self.device = torch.device("cuda", torch.cuda.current_device()) if d.type == "cuda" and d.index is None else d
        self._weight, self._roots, self._keep = weight, roots, keep
        self._b: dict[int, tuple[Any, torch.Tensor]] = {}
        self._starts: tuple[Any, list[tuple[int, int]]] | None = None
        self._records: dict[tuple[bytes, bytes, bytes, bool], torch.Tensor] = {}
        self._ready: dict[int, tuple[Any, np.ndarray, Any]] = {}
        self._copy: Any = None
        self._up: Any = None
        self.kept: dict[int, dict[str, Any]] = {}

    def bind(self, run: Any) -> "Receipts":
        self.run = run
        return self

    def roots(self, call: Any) -> tuple[bytes, bytes]:
        """Call ``call``'s (root(A), root(B)): its step's A table ``a/<K>@<s>``'s and the weights' ``b/<K>``'s registered
        roots (`Served.registered`), the Registered row/v2 roots its tables were committed under. ValueError while
        either isn't registered."""
        if self._roots is not None:
            return self._roots(call)
        s = self.served
        names = (s.a_table(call.index), s.program.fn.bindings["PREFIX"] + f"b/{call.k}")
        if missing := [v for v in names if v not in s.registered]:
            raise ValueError(f"call {call.index}'s seeds take the roots of {names[0]} and {names[1]}, and {missing} isn't "
                             "registered yet: a step's receipts of K are asked for at its last call of K")
        return s.registered[names[0]].root, s.registered[names[1]].root

    @property
    def j10r(self) -> bool:
        """Whether the producer computes J10-R's receipts: ``j10r`` if given, else the window Program's form."""
        return HW.is_j10r(self.served.program) if self._j10r is None else bool(self._j10r)

    def public(self, call: Any) -> list[list[int]]:
        """Call ``call``'s public rows as the verifier builds them (`credit.public_table`, the statement's ``pub`` port):
        row t unit u0 + t's, the window Program's salt, u0 + t and the call's roots (`roots`), and under J10-R u0 and the
        unit's blocks (`h1window.call_blocks`)."""
        s, (u0, n), (ra, rb) = self.served, self.start(call), self.roots(call)
        blocks = HW.call_blocks(s.program, call.index) if self.j10r else None
        rows, _salts, _reg = CR.public_table(s.program_digest, HW.table(s.program, HW.PUBS, call.index), self.run.salt.hex(),
                                             s.root, u0, n, (ra.hex(), rb.hex()), blocks)
        return rows

    def start(self, call: Any) -> tuple[int, int]:
        """Call ``call``'s first unit of the window Program and its units (`h1window.call_units`): its units' positions are
        u0 .. u0 + n - 1."""
        prog = self.served.program
        if self._starts is None or self._starts[0] is not prog:
            self._starts = (prog, HW.call_units(prog))
        return self._starts[1][call.index]

    def weight_of(self, call: Any, w: Any) -> torch.Tensor:
        """Call ``call``'s B [N, K] s8 on the device (``weight``, else ``w`` copied there once a linear)."""
        if self._weight is not None:
            b = self._weight(call)
        else:
            held = self._b.get(call.layer)
            if held is None or held[0] is not w:
                if w is None:
                    raise ValueError(f"call {call.index}: no weight of linear {call.layer} was handed over")
                held = self._b[call.layer] = (w, torch.from_numpy(np.ascontiguousarray(w, np.int8)).to(self.device))
            b = held[1]
        if b.dtype != torch.int8 or tuple(b.shape) != (call.n, call.k) or b.device != self.device or not b.is_contiguous():
            raise ValueError(f"call {call.index}: its weight is {b.dtype} {tuple(b.shape)} on {b.device}, not its {call.n} × "
                             f"{call.k} contiguous int8 codes on {self.device}")
        return b

    def operands(self, call: Any, codes: Any, w: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Call ``call``'s A [M, K] (its served ``codes`` [m, K], then zero rows to the Program's slot M) and B [N, K], s8
        on the device."""
        m, k = int(self.served.m), int(call.k)
        x = np.asarray(codes)
        if x.dtype != np.int8 or x.ndim != 2 or x.shape[1] != k or len(x) > m:
            raise ValueError(f"call {call.index}: its codes are {x.dtype} {x.shape}, not at most {m} rows of {k} int8 codes")
        a = torch.zeros(m, k, dtype=torch.int8, device=self.device)
        if len(x):
            a[:len(x)] = torch.from_numpy(np.ascontiguousarray(x)).to(self.device)
        return a, self.weight_of(call, w)

    def _record(self, salt: bytes, ra: bytes, rb: bytes) -> torch.Tensor:
        key = (salt, ra, rb, self.j10r)
        if key not in self._records:
            self._records = {key: torch.from_numpy(seed_record(salt, ra, rb, self.j10r)).to(self.device)}
        return self._records[key]

    def _live(self, m: int) -> torch.Tensor | None:
        if not self.skip_dead:
            return None
        if self.j10r:
            raise ValueError("skip_dead under J10-R: its unit (PouwH1Tile_v5) is ungated, a dead tile has its noise's "
                             "receipt; unset H1_SKIP_DEAD")
        return (torch.arange(int(self.served.m) // RC.R, device=self.device) * RC.R < m).to(torch.uint8)

    def prepare(self, batch: list[tuple[Any, Any, Any, Any, Any]]) -> None:
        """`Served`'s hook before it asks a step's calls of one K, ``batch`` their ``(call, codes, s_a, weight codes, s_w)``
        in the order it asks them: with the units kernel, every unit of them in `launches` launches (`Units.multi`) and
        their receipts copied to the host on a stream of their own, so each call's ask returns its rows (the same words)
        with no launch of its own, once its launch's copy is done. A launch's codes are staged in pinned memory and go to
        the device in one copy on a stream of their own. Nothing with the torch chain, or unless every call of the batch
        serves the same rows at one K: each is then asked on its own."""
        self._ready = {}
        if self.units is None or self.run is None or len(batch) < 2:
            return
        m, k, slot = len(batch[0][1]), int(batch[0][0].k), int(self.served.m)
        if any(len(codes) != m or int(call.k) != k for call, codes, *_rest in batch):
            return
        xs = [np.asarray(codes) for _call, codes, *_rest in batch]
        if any(x.dtype != np.int8 or x.shape != (m, k) for x in xs) or m > slot:
            raise ValueError(f"calls {[c.index for c, *_r in batch]}: their codes are not {m} rows of {k} int8 codes each, "
                             f"at most {slot}")
        bs = [self.weight_of(call, w) for call, _codes, _s_a, w, _s_w in batch]
        spans = [self.start(call) for call, *_rest in batch]
        for (call, *_rest), b, (_u0, n) in zip(batch, bs, spans):
            if (slot // RC.R) * (b.shape[0] // RC.C_) != n:
                raise ValueError(f"call {call.index}: {n} units in the window Program, not its {slot} × {b.shape[0]} tiles")
        offs = [int(x) for x in np.cumsum([0] + [n for _u0, n in spans])]
        n = min(self.launches, len(batch))
        cuts = sorted({0, len(batch)} | {int(np.searchsorted(offs, offs[-1] * j / n)) for j in range(1, n)})
        recs = [self._record(self.run.salt, *self.roots(call)) for call, *_rest in batch]
        main = torch.cuda.current_stream(self.device)
        if self._copy is None:
            self._copy, self._up = torch.cuda.Stream(device=self.device), torch.cuda.Stream(device=self.device)
        a = (torch.empty if m == slot else torch.zeros)(len(batch), slot, k, dtype=torch.int8, device=self.device)
        self._up.wait_stream(main)
        a.record_stream(self._up)
        host = torch.empty(len(batch), m, k, dtype=torch.int8, pin_memory=True) if m else None
        seeds = torch.empty(offs[-1], 16, dtype=torch.int64, device=self.device)
        got = torch.empty(offs[-1], self.units.words(k), dtype=torch.int32, pin_memory=True)
        words, live, ready, j10r = got.numpy().view(np.uint32), self._live(m), {}, self.j10r
        for lo, hi in zip(cuts[:-1], cuts[1:]):
            if host is not None:
                hn = host.numpy()
                for i in range(lo, hi):
                    hn[i] = xs[i]
                with torch.cuda.stream(self._up):
                    a[lo:hi, :m].copy_(host[lo:hi], non_blocking=True)
                main.wait_stream(self._up)
            i = lo
            while i < hi:
                j = i + 1
                while not j10r and j < hi and recs[j] is recs[i] and spans[j][0] == spans[j - 1][0] + spans[j - 1][1]:
                    j += 1
                self.units.seeds(recs[i], spans[i][0], offs[j] - offs[i], out=seeds[offs[i]:offs[j]],
                                 call=spans[i][0] if j10r else None)
                i = j
            out = self.units.multi([(a[i], bs[i], seeds[offs[i]:offs[i + 1]]) for i in range(lo, hi)], live, j10r)
            self._copy.wait_stream(main)
            with torch.cuda.stream(self._copy):
                got[offs[lo]:offs[hi]].copy_(out, non_blocking=True)
            out.record_stream(self._copy)
            copied = torch.cuda.Event()
            copied.record(self._copy)
            for i in range(lo, hi):
                ready[batch[i][0].index] = (batch[i][1], words[offs[i]:offs[i + 1]], copied)
        self._ready = ready

    def __call__(self, call: Any, codes: Any, s_a: Any, w: Any, s_w: Any) -> np.ndarray:
        if self.run is None:
            raise ValueError("the receipt producer is bound to no run: its units' seeds take the run's salt (`bind`)")
        salt, j10r = self.run.salt, self.j10r
        u0, n = self.start(call)
        ra, rb = self.roots(call)
        m = len(codes)
        live = self._live(m)
        ready = self._ready.pop(call.index, None)
        if ready is not None and ready[0] is codes:
            ready[2].synchronize()
            out, a, b = ready[1], None, None
        else:
            a, b = self.operands(call, codes, w)
            if (a.shape[0] // RC.R) * (b.shape[0] // RC.C_) != n:
                raise ValueError(f"call {call.index}: {n} units in the window Program, not its {a.shape[0]} × "
                                 f"{b.shape[0]} tiles")
            if self.units is not None:
                seeds = self.units.seeds(self._record(salt, ra, rb), u0, n, call=u0 if j10r else None)
                out = self.units.receipts(a, b, seeds, live, j10r).cpu().numpy().view(np.uint32)
            elif self.skip_dead:
                out = RC.gated_receipts(RC.Call(a, b), np.arange(a.shape[0]) < m, salt, u0, ra, rb, self.cubin)[0]
            else:
                out = RC.call_receipts(RC.Call(a, b), salt, u0, ra, rb, self.cubin, j10r=j10r)
        if self._keep is not None and self._keep(call):
            if a is None:
                a, b = self.operands(call, codes, w)
            self.kept[call.index] = {"a": a.cpu(), "b": b.cpu(), "salt": salt, "base": u0, "roots": (ra, rb),
                                     "receipts": out.copy(), "m": m, "pub": self.public(call), "skip": self.skip_dead,
                                     "j10r": j10r}
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


__all__ = ["Receipts", "Units", "check", "j10r_fields", "public_fields", "seed_record"]
