"""Kernels' producer of vllm's `Served.receipts` (BRIDGE2) on the H1 served path
(`integrations/vllm/verity_vllm/protocol_options/pouw_h1_served.py`): each call's receipts, one a unit of the call in the
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

It returns (units, 2 (KS + m)) uint32 (KS = K / 32, m = `segment.n_digits(KS)`: 260 words at K = 4096, 902 at 14336), the
input of `pouw_h1_served.receipt_words`, which `Served` refuses at any other shape. It takes no coin: the noise is the
salt's and the unit's position's. The kernel needs 512 | K (`kh_unit`); the torch chain any 32 | K, K >= 64.
"""
from __future__ import annotations

import ctypes as C
import os
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import receipts as RC  # noqa: E402

from verity.core.protocols.pouw import core_check as CC  # noqa: E402
from verity.core.protocols.pouw import credit as CR  # noqa: E402
from verity.core.protocols.pouw.circuit import h1tile as HT  # noqa: E402
from verity.core.protocols.pouw.circuit import h1window as HW  # noqa: E402
from verity.core.protocols.pouw.circuit import seeds as SD  # noqa: E402


def _be(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=">u8").astype(np.uint64)


def seed_record(salt: bytes, root_a: bytes, root_b: bytes) -> np.ndarray:
    """`kh_seeds`' data, 30 int64: SEED_TAG and RECEIPT_TAG (3 big-endian u64 words each), then the salt, root(A) and
    root(B) (8 each). ValueError unless each of the three is 64 bytes."""
    for name, v in (("salt", salt), ("root(A)", root_a), ("root(B)", root_b)):
        if len(v) != 64:
            raise ValueError(f"a unit's seeds take a 64-byte {name}, not {len(v)} bytes")
    return np.concatenate([_be(SD.SEED_TAG), _be(SD.RECEIPT_TAG), _be(salt), _be(root_a), _be(root_b)]).view(np.int64)


def public_fields(words: Any) -> tuple[bytes, int, tuple[bytes, bytes]]:
    """A unit's public row (`core_check.public_words`, a ``row/v2`` row of 16-bit leaves) -> the salt, position and
    (root(A), root(B)) it carries (`core_check.public_row`), its seeds' inputs. ValueError on a row of another width."""
    if len(words) * 16 != CC.PUBLIC_ROW_BITS:
        raise ValueError(f"a public row is {CC.PUBLIC_ROW_BITS // 16} 16-bit words, not {len(words)}")
    x = b"".join(int(w).to_bytes(2, "little") for w in words)
    x = b"".join(x[i:i + 4][::-1] for i in range(0, len(x), 4))
    s = len(x) - 8 - 128
    return x[:s], int.from_bytes(x[s:s + 8], "big"), (x[s + 8:s + 72], x[s + 72:])


def _stream(t: torch.Tensor) -> int:
    return torch.cuda.current_stream(t.device).cuda_stream


class Units:
    """h1u.cu's `kh_seeds` and `kh_unit` from an -hbc build: a call's units' seeds and receipts, on torch's current stream,
    every unit live (``live`` NULL: v1's ungated unit) unless `receipts` is given a row block's live bits."""

    def __init__(self, lib: str | Path | C.CDLL):
        h = lib if isinstance(lib, C.CDLL) else C.CDLL(str(lib))
        vp, u32 = C.c_void_p, C.c_uint
        h.kh_seeds.argtypes = [vp, C.c_ulonglong, u32, vp, vp]
        h.kh_seeds.restype = C.c_int
        h.kh_unit.argtypes = [vp, vp, u32, u32, u32, vp, vp, vp, vp]
        h.kh_unit.restype = C.c_int
        h.kh_words.argtypes = [u32]
        h.kh_words.restype = u32
        self.h = h

    def words(self, k: int) -> int:
        return int(self.h.kh_words(k))

    def seeds(self, record: torch.Tensor, base: int, units: int) -> torch.Tensor:
        """(units, 16) int64 on ``record``'s device: unit u's seed_j then σ_R, big-endian words, at position base + u, from
        ``record`` (`seed_record`, (30,) int64)."""
        if tuple(record.shape) != (30,) or record.dtype != torch.int64 or not record.is_cuda:
            raise ValueError(f"kh_seeds: a (30,) int64 record on the device, not {record.dtype} {tuple(record.shape)}")
        out = torch.empty(units, 16, dtype=torch.int64, device=record.device)
        rc = self.h.kh_seeds(record.contiguous().data_ptr(), int(base), int(units), out.data_ptr(), _stream(record))
        if rc:
            raise RuntimeError(f"kh_seeds returned {rc} (U = {units})")
        return out

    def receipts(self, a: torch.Tensor, b: torch.Tensor, seeds: torch.Tensor,
                 live: torch.Tensor | None = None) -> torch.Tensor:
        """A [M, K] and B [N, K] s8 on the device and the units' seeds (`seeds`) -> receipts (units, kh_words(K)) int32,
        unit u row block u // (N / 32) and column block u % (N / 32). ``live``: (M / 32,) uint8 on the device, a row
        block's units zero words, no noise derived, where its bit is 0 (v4's dead tile). ValueError on operands no call's
        or a K the kernel has no instance of (512 | K and n_digits(K / 32) 2 or 3)."""
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
    wholly past the served rows (the module docstring), by default ``H1_SKIP_DEAD=1`` in the environment."""

    def __init__(self, served: Any, run: Any = None, units: Units | None = None, device: Any = None,
                 weight: Callable[[Any], torch.Tensor] | None = None,
                 roots: Callable[[Any], tuple[bytes, bytes]] | None = None, keep: Callable[[Any], bool] | None = None,
                 skip_dead: bool | None = None):
        self.served, self.run, self.units = served, run, units
        self.skip_dead = os.environ.get("H1_SKIP_DEAD") == "1" if skip_dead is None else bool(skip_dead)
        d = torch.device(device if device is not None else ("cuda" if units is not None else "cpu"))
        self.device = torch.device("cuda", torch.cuda.current_device()) if d.type == "cuda" and d.index is None else d
        self._weight, self._roots, self._keep = weight, roots, keep
        self._b: dict[int, tuple[Any, torch.Tensor]] = {}
        self._starts: tuple[Any, list[tuple[int, int]]] | None = None
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

    def public(self, call: Any) -> list[list[int]]:
        """Call ``call``'s public rows as the verifier builds them (`credit.public_table`, the statement's ``pub`` port):
        row t unit u0 + t's, the window Program's salt, u0 + t and the call's roots (`roots`)."""
        s, (u0, n), (ra, rb) = self.served, self.start(call), self.roots(call)
        rows, _salts, _reg = CR.public_table(s.program_digest, HW.table(s.program, HW.PUBS, call.index), self.run.salt.hex(),
                                             s.root, u0, n, (ra.hex(), rb.hex()))
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

    def __call__(self, call: Any, codes: Any, s_a: Any, w: Any, s_w: Any) -> np.ndarray:
        if self.run is None:
            raise ValueError("the receipt producer is bound to no run: its units' seeds take the run's salt (`bind`)")
        salt = self.run.salt
        u0, n = self.start(call)
        ra, rb = self.roots(call)
        a, b = self.operands(call, codes, w)
        if (a.shape[0] // RC.R) * (b.shape[0] // RC.C_) != n:
            raise ValueError(f"call {call.index}: {n} units in the window Program, not its {a.shape[0]} × {b.shape[0]} tiles")
        m = len(codes)
        if self.units is not None:
            seeds = self.units.seeds(torch.from_numpy(seed_record(salt, ra, rb)).to(self.device), u0, n)
            live = None
            if self.skip_dead:
                live = (torch.arange(a.shape[0] // RC.R, device=self.device) * RC.R < m).to(torch.uint8)
            out = self.units.receipts(a, b, seeds, live).cpu().numpy().view(np.uint32)
        elif self.skip_dead:
            out = RC.gated_receipts(RC.Call(a, b), np.arange(a.shape[0]) < m, salt, u0, ra, rb)[0]
        else:
            out = RC.call_receipts(RC.Call(a, b), salt, u0, ra, rb)
        if self._keep is not None and self._keep(call):
            self.kept[call.index] = {"a": a.cpu(), "b": b.cpu(), "salt": salt, "base": u0, "roots": (ra, rb),
                                     "receipts": out.copy(), "m": m, "pub": self.public(call), "skip": self.skip_dead}
        return out


def check(kept: dict[str, Any], units: list[int]) -> dict[str, Any]:
    """A kept call's (`Receipts.kept`) receipts of ``units`` against `h1tile.reference` on the same operands, seeded from
    each unit's public row as the verifier builds it (``pub``, `public_fields`), word for word, v4's under the served rows'
    admit bits where the producer skipped dead tiles (``skip``): ``{unit: {"live": its row block holds a served row, "pub":
    the row is the producer's salt, position and roots, "ok": bool}}``."""
    ncb, out = kept["b"].shape[0] // RC.C_, {}
    for u in units:
        i, j = divmod(u, ncb)
        a = (kept["a"][RC.R * i:RC.R * (i + 1)].to(torch.int64) & 0xFF).tolist()
        b = (kept["b"][RC.C_ * j:RC.C_ * (j + 1)].to(torch.int64) & 0xFF).tolist()
        salt, position, roots = public_fields(kept["pub"][u])
        gated = [int(RC.R * i + r < kept["m"]) for r in range(RC.R)] if kept.get("skip") else None
        _z, want = HT.reference(a, b, salt, position, roots, pairs=[(0, 0)], gated=gated)
        out[u] = {"live": RC.R * i < kept["m"],
                  "pub": (salt, position, roots) == (kept["salt"], kept["base"] + u, kept["roots"]),
                  "ok": kept["receipts"][u].tolist() == want}
    return out


__all__ = ["Receipts", "Units", "check", "public_fields", "seed_record"]
