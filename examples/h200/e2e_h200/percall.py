"""PoUW's per-call-noise units over every served linear call of the step, coverage f = 1: a
`units.Subcircuit` whose ``call`` the driver runs on each linear's served call (`driver.Driver._linear`), with that call's
A codes, weight and s32 sums.

Each call (rows ``lo .. lo + m`` of one micro-batch through one of a layer's six linears or the head: 4 x 313 calls a
step at batch 4,096 and micro-batch 1,024) is J10-R's call: its units are the 32 x 32 tiles of A [m, K] times B [N, K],
``PouwH1Tile_v6`` (`h1tile.j10r_reference`), unit u = ri (N / 32) + cj at position ``base + u``, the step's units numbered
in serving order from ``(step + 1) << 40``. Per call:

* **receipts**, every unit's: one of kernels' sm_90a builds (``kraqwxsc``, the fastest exact one, or ``kraq``;
  `benchmarks/torch_gpu/j10r_sm90/units.py`'s `H1uCall`, the served launch on every SM) on CUDA, or
  `h1tile.j10r_reference` on the host (``python``, small shapes);
* **z held to the served accumulators**: a unit's z is the served call's s32 sums, not recomputed. They're committed as
  served, and ``samples`` units a call (a FIXTURE coin) go through the exact fallback (`RefCall`, j10r_ref.so, or
  `j10r_reference` on the host), whose z must equal the served tile and whose receipt the CUDA unit's;
* **commitments**, under the step's salt key (sha512_rows' engine, the commit call's): each unit's receipt one
  hm96-sha512 row leaf (salt nonce its position), each row of z one leaf per 256 columns (nonce its row), a tree of each
  per call, and the step's PoUW root over the calls' (receipts, z) roots, all frame-v3-sha512.

Credit is not computed here, and not reported as zero: the two blockers stay named in `CREDIT` (h1zy's J10-R lane
Definitions, Credit.window reading the tile stratum), and the stand-ins in `STANDS_IN`.

``--subcircuit e2e_h200.percall:kraqwxsc`` (or ``:kraq``) reads ``PERCALL_H1U`` (the h1u build, default
``h1u_<backend>.so``), ``PERCALL_REF`` (j10r_ref.so) and ``PERCALL_SAMPLES`` (2); ``e2e_h200.percall:python`` is the
host twin.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from verity.core.protocols.pouw.circuit import h1tile as HT

from .commit import _slots
from .shape import Shape

T = 32
#: z's leaf: one row's columns, at most this many s32 sums
Z_COLUMNS = 256
#: the step's first unit position is (step + 1) << POS_SHIFT
POS_SHIFT = 40
#: kernels' sm_90a h1u builds (`benchmarks/torch_gpu/j10r_sm90/job.sh`'s arms), each a CUDA backend
BUILDS = ("kraqwxsc", "kraq")
TEMPLATE = "PouwH1Tile_v6{R=32, C=32, K} (J10-R, whole K a unit)"
J10R_SM90 = Path(__file__).resolve().parents[3] / "benchmarks" / "torch_gpu" / "j10r_sm90"

CREDIT = {
    "status": "blocked",
    "eps": None,
    "blockers": [
        {"id": "h1zy-j10r-lanes", "what": "J10-R's lane Definitions in h1zy are pending: a drawn unit has no "
                                          "registered lane statement for C-Flock to prove at the served K (16,384 and "
                                          "65,536)"},
        {"id": "CMP-127 TileDrawIsCoin", "what": "PouwCheck.Credit.window does not yet read the tile stratum of the single "
                                                 "stratified draw, so no unit here is credited"},
    ],
}
STANDS_IN = {
    "salt": "SHA-512 of the step's salt key: window_salt.issue's live salt replaces it",
    "roots": "root(A) and root(B) are SHA-512 tags of the call and the tensor: the step's table root and the "
             "weights' registration root replace them",
    "check_draw": "a FIXTURE coin picks the checked units; the tile stratum's draw from live coins replaces it",
    "commit_layout": "this module's per-call trees under the engine's hm96 leaves; the registered Program's layout "
                     "replaces it",
}


def _h(*parts: bytes) -> bytes:
    return hashlib.sha512(b"\0".join(parts)).digest()


def _u32(x: int) -> bytes:
    return int(x).to_bytes(4, "big")


@dataclass
class Call:
    op: str
    lo: int
    m: int
    n: int
    k: int
    base: int
    units: int
    words: int
    roots: tuple[Any, Any] = (None, None)
    picks: list[int] = field(default_factory=list)
    receipts_equal: Any = None
    z_equal: Any = None
    marks: tuple = ()
    seconds: dict[str, float] = field(default_factory=dict)


class PerCall:
    """The subcircuit (`units.Subcircuit`): ``backend`` one of `BUILDS` (CUDA, `H1uCall` on ``h1u``, that build, checked by
    `RefCall` on ``ref``) or "python" (the host, `h1tile.j10r_reference`)."""

    name = "pouw-percall"

    def __init__(self, shape: Shape, weights: Any = None, run_id: str = "e2e-h200-fixture", backend: str = "python",
                 h1u: str | None = None, ref: str | None = None, samples: int = 2) -> None:
        if backend not in (*BUILDS, "python"):
            raise ValueError(f"the backend is one of {', '.join(BUILDS)} or python, not {backend!r}")
        self.shape, self.run_id, self.backend, self.samples = shape, run_id, backend, samples
        _, G = _slots()
        self.E = G.E
        self.cuda = backend in BUILDS
        self.eng = self.E.Engine("cuda" if self.cuda else "host")
        self.libs: dict[str, Any] = {}
        if self.cuda:
            if str(J10R_SM90) not in sys.path:
                sys.path.insert(0, str(J10R_SM90))
            import units as UN
            self.UN = UN
            self.h1u = UN.H1u(h1u, sm=UN.H1u.has_sm(h1u))
            self.ref = UN.Ref(ref)
            self.libs = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (h1u, ref)}
        self._leaf_tables: dict[tuple, Any] = {}
        self.step: int | None = None
        self.calls: list[Call] = []
        self.steps: list[dict] = []
        self.mismatches = 0

    # -- keys and positions

    def _begin(self, st: Any) -> None:
        if self.calls:
            raise RuntimeError(f"step {self.step}'s calls were not served")
        self.step = st.step
        key = st.salt_key or _h(b"e2e-h200/percall/salt-key-fixture", self.run_id.encode(), _u32(st.step))[:32]
        self.salt = _h(b"e2e-h200/percall/salt-stand-in", key)
        self.receipt_key = _h(b"e2e-h200/percall/receipt-salts", key)[:32]
        self.z_key = key
        self.next = (st.step + 1) << POS_SHIFT
        self.t0 = time.perf_counter()

    def _did(self, kind: str, op: str = "") -> bytes:
        return _h(b"e2e-h200/percall/v1", kind.encode(), op.encode())

    def _leaf_table(self, kind: str, op: str, g: int) -> Any:
        key = (kind, op, g)
        if key not in self._leaf_tables:
            self._leaf_tables[key] = self.eng.leaf_table([self._did(kind, op)], [0] * g, list(range(g)))
        return self._leaf_tables[key]

    def _arr(self, t: torch.Tensor) -> Any:
        return t if self.cuda else np.ascontiguousarray(t.numpy())

    # -- the served call

    def call(self, driver: Any, st: Any, op: str, lo: int, codes: torch.Tensor, w: torch.Tensor, acc: torch.Tensor) -> None:
        if self.step != st.step:
            self._begin(st)
        m, k = codes.shape
        n = w.shape[0]
        if m % T or n % T or k % T:
            raise ValueError(f"{op}: a call of {m} x {n} x {k} is no whole 32 x 32 tiles")
        units = (m // T) * (n // T)
        words = 2 * (k // T + HT.Layout(T, T, k).m)
        c = Call(op, lo, m, n, k, self.next, units, words)
        self.next += units
        ra = _h(b"e2e-h200/percall/root-a-tag", _u32(st.step), op.encode(), _u32(lo))
        rb = _h(b"e2e-h200/percall/root-b-tag", op.encode())
        h = _h(b"e2e-h200/percall/check-fixture", self.run_id.encode(), _u32(st.step), op.encode(), _u32(lo))
        rng = np.random.Generator(np.random.PCG64(int.from_bytes(h[:16], "big")))
        c.picks = sorted({int(u) for u in rng.integers(0, units, size=min(self.samples, units))})
        if self.cuda:
            self._cuda(c, codes, w, acc, ra, rb)
        else:
            self._python(c, codes, w, acc, ra, rb)
        self.calls.append(c)

    def _commit(self, c: Call, receipts: Any, acc: Any) -> None:
        eng = self.eng
        x = eng.row_digests(eng.rows([[(0, 0, 4 * c.words)]], [receipts]))
        rl = eng.leaves(self._leaf_table("receipts", c.op, 1), x, self.E.SaltKey(self.receipt_key, c.base))
        cols = min(c.n, Z_COLUMNS)
        g = c.n // cols
        zx = eng.row_digests(eng.rows([[(0, 4 * cols * j, 4 * cols)] for j in range(g)], [acc]))
        zkey = _h(b"e2e-h200/percall/z-salts", self.z_key, c.op.encode())[:32]
        zl = eng.leaves(self._leaf_table("z", c.op, g), zx, self.E.SaltKey(zkey, c.lo))
        zl = zl.permute(1, 0, 2).reshape(1, c.m * g, 8) if self.cuda else np.ascontiguousarray(zl.transpose(1, 0, 2)).reshape(1, -1, 8)
        c.roots = (eng.root(rl, [self._did("receipts", c.op)]), eng.root(zl, [self._did("z", c.op)]))

    def _tile(self, acc: torch.Tensor, n: int, u: int) -> torch.Tensor:
        ri, cj = divmod(u, n // T)
        return acc[T * ri:T * ri + T, T * cj:T * cj + T]

    def _cuda(self, c: Call, codes: torch.Tensor, w: torch.Tensor, acc: torch.Tensor, ra: bytes, rb: bytes) -> None:
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
        ev[0].record()
        kc = self.UN.H1uCall(self.h1u, codes, w, self.salt, (ra, rb), c.base)
        if kc.out.shape[1] != c.words:
            raise RuntimeError(f"{c.op}: the h1u build writes {kc.out.shape[1]} receipt words, J10-R {c.words}")
        kc.run()
        ev[1].record()
        self._commit(c, kc.out, acc)
        ev[2].record()
        rc = self.UN.RefCall(self.ref, codes, w, self.salt, (ra, rb), c.base, units=c.picks, z=True)
        rc.run()
        picks = torch.tensor(c.picks, device=acc.device)
        c.receipts_equal = (rc.out == kc.out.index_select(0, picks)).all(dim=1)
        c.z_equal = (rc.z == torch.stack([self._tile(acc, c.n, u) for u in c.picks])).flatten(1).all(dim=1)
        ev[3].record()
        c.marks = tuple(ev)

    def _python(self, c: Call, codes: torch.Tensor, w: torch.Tensor, acc: torch.Tensor, ra: bytes, rb: bytes) -> None:
        t0 = time.perf_counter()
        a8, b8 = codes.numpy().astype(np.int64) & 0xFF, w.numpy().astype(np.int64) & 0xFF
        served = acc.numpy().view(np.uint32)
        receipts = np.empty((c.units, c.words), np.uint32)
        z_equal = []
        for u in range(c.units):
            ri, cj = divmod(u, c.n // T)
            blocks = HT.j10r_blocks(c.m, c.k, (T, T), ri, cj)
            z, rec = HT.j10r_reference(a8[T * ri:T * ri + T].tolist(), b8[T * cj:T * cj + T].tolist(), self.salt,
                                       c.base + u, (ra, rb), c.base, blocks, stream=True)
            if len(rec) != c.words:
                raise RuntimeError(f"{c.op}: j10r_reference gives {len(rec)} receipt words, not {c.words}")
            receipts[u] = rec
            z_equal.append(np.array_equal(np.asarray(z, np.uint64).astype(np.uint32).reshape(T, T),
                                          served[T * ri:T * ri + T, T * cj:T * cj + T]))
        t1 = time.perf_counter()
        self._commit(c, receipts.view(np.int32), self._arr(acc))
        t2 = time.perf_counter()
        c.receipts_equal = np.ones(len(c.picks), bool)
        c.z_equal = np.array(z_equal, bool)
        c.seconds = {"units": t1 - t0, "commit": t2 - t1, "check": 0.0}

    # -- the step

    def serve(self, driver: Any, result: Any) -> None:
        """The step's PoUW root over its calls' (receipts, z) roots, its seconds and its checks."""
        if not self.calls or self.step != result.step:
            raise RuntimeError(f"step {result.step} served no linear call through this subcircuit")
        if self.cuda:
            torch.cuda.synchronize()
            for c in self.calls:
                e = c.marks
                c.seconds = {"units": e[0].elapsed_time(e[1]) / 1e3, "commit": e[1].elapsed_time(e[2]) / 1e3,
                             "check": e[2].elapsed_time(e[3]) / 1e3}
            stack = torch.stack
        else:
            stack = np.stack
        did = self._did("step")
        values = stack([r.reshape(8) for c in self.calls for r in c.roots]).reshape(1, -1, 8)
        root = self.E.digests(self.eng.root(self.eng.frame_leaves(values, [did]), [did]))[0]
        r_bad = sum(int((~torch.as_tensor(c.receipts_equal)).sum()) for c in self.calls)
        z_bad = sum(int((~torch.as_tensor(c.z_equal)).sum()) for c in self.calls)
        self.mismatches += r_bad + z_bad
        sec = {k: sum(c.seconds[k] for c in self.calls) for k in ("units", "commit", "check")}
        doc = {"step": result.step, "root": root.hex(), "calls": len(self.calls), "units": sum(c.units for c in self.calls),
               "first_position": self.calls[0].base, "receipt_bytes": sum(4 * c.units * c.words for c in self.calls),
               "z_bytes": sum(4 * c.m * c.n for c in self.calls),
               "z_checked_units": sum(len(c.z_equal) for c in self.calls),
               "receipt_mismatches": r_bad, "z_mismatches": z_bad, "seconds": sec,
               "serve_seconds": dict(result.seconds),
               "by_linear": self._by_linear()}
        self.steps.append(doc)
        print(json.dumps({"pouw": {k: doc[k] for k in ("step", "calls", "units", "receipt_mismatches", "z_mismatches",
                                                       "seconds")} | {"root": doc["root"][:32]}}), flush=True)
        self.calls = []

    def _by_linear(self) -> dict:
        out: dict[str, dict] = {}
        for c in self.calls:
            kind = c.op.split(".")[-1]
            d = out.setdefault(kind, {"calls": 0, "units": 0, "units_s": 0.0, "commit_s": 0.0, "n": c.n, "k": c.k})
            d["calls"] += 1
            d["units"] += c.units
            d["units_s"] += c.seconds["units"]
            d["commit_s"] += c.seconds["commit"]
        return out

    def families(self, replayer: Any, step: int) -> list:
        return []

    def manifest(self) -> dict:
        return {"template": TEMPLATE, "issue": "CMP-126", "coverage": "f = 1: every served linear call, every unit",
                "backend": f"{self.backend} (h1u sm_90a, the served launch) checked by j10r_ref" if self.cuda
                else "h1tile.j10r_reference on the host", "libs_sha256": self.libs, "samples_a_call": self.samples,
                "z": "the served call's s32 sums (z held to the served accumulators)", "credit": CREDIT,
                "stands_in": STANDS_IN, "steps": self.steps}


def _build(backend: str, shape: Shape, weights: Any, run_id: str) -> PerCall:
    return PerCall(shape, weights, run_id, backend, os.environ.get("PERCALL_H1U", f"h1u_{backend}.so"),
                   os.environ.get("PERCALL_REF", "j10r_ref.so"), int(os.environ.get("PERCALL_SAMPLES", "2")))


def kraqwxsc(shape: Shape, weights: Any = None, run_id: str = "e2e-h200-fixture") -> PerCall:
    return _build("kraqwxsc", shape, weights, run_id)


def kraq(shape: Shape, weights: Any = None, run_id: str = "e2e-h200-fixture") -> PerCall:
    return _build("kraq", shape, weights, run_id)


def python(shape: Shape, weights: Any = None, run_id: str = "e2e-h200-fixture") -> PerCall:
    return PerCall(shape, weights, run_id, "python", samples=int(os.environ.get("PERCALL_SAMPLES", "2")))
