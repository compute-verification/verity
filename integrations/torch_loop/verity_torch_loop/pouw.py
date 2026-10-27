"""The torch loop's linears as PoUW's pinned H1 tile, so PoUW's window check (`verity/pouw/v1`, `core_check.answer`) can
credit them: no new pin; PoUW's check takes the window's operands (`operands`) beside its items.

* **The unit** is the tile `core_check.TILES` pins, `PouwH1Tile_v1{EXACT=GemmAccumulatorS8_v2{K=64}, R=4, C=8, PAIRS=full}`
  (`tile`): on R = 4 A rows and C = 8 B rows of 64 s8 codes, its salt, position and roots, it returns z = A·Bᵀ exactly
  (s32 in u32 words) and the tile's receipt (`h1tile.reference`, the Definition's function, which the Definition evaluates
  to bit for bit on these words).
* **The linear** at width H = 128 is two tiles a cell, its K split in halves: z = z_lo + z_hi in int32 is the loop's
  exact accumulator, and ``bf16((f32(z) * sx) * sw)`` is its output (`loop.linear`). Every step's linear is batched over the
  requests (rows are independent: each request is one token row), the rows padded to a multiple of R with zero rows, so a
  step's linear is ``ceil(requests / R) * (H / C) * (H / 64)`` tiles; `H1Loop` asserts that the tiles' z gives the loop's
  own output, coordinate for coordinate. The scales and the cast are not in the unit (one-stage's window of the loop
  covers them).
* **The window** (`H1Loop.window`) is an `interface.Window` of ``N = pad(N_real + 1)`` tiles (`pouw.window`'s grid: N_real
  real tiles, credited ``floor_point(N_real)``, the rest filler tiles of zero rows), unit u at position u, under the law
  subset:K′ that ε and δ fix (`window.k_prime`).
* **The operands** (`H1Loop.operands`, CMP-91) are committed before the salt, in the tile's own row shape: ``a``, one row
  a (step, layer, row block, K half) of R tokens' 64 codes, and ``b``, one a (layer, column block, K half) of C channels'
  64 codes, each with one zero row for the fillers: per-window values of PoUW's `operands` (domains binding the run
  root), named in the registration's window (``activations``). The tiles' ``a`` and ``b`` ports declare reads of them
  (`interface.Read`), so a member commits each row under the value's salt and Lean's registered reads open it against
  the value's root; the tiles' roots are those two roots. Their root X (`H1Loop.X`, `core_check.values_root` of
  ``activations``) is what the window's salt is released against, so the salt comes after them (`H1Loop.window`'s
  ``salt``), the same for every tile.
* **The live salt** (``live``, CMP-90): the registration names the pinned weights' entries (``registered``,
  `weights_group`) and no ``W`` root, so W is their root; `window_salt.issue` opens one salt-only session on
  R = `salt.anchored_epoch_root(W, X, k)` (`H1Loop.anchors`) and the tiles take its salt. `SALT` beside the window holds
  the answer's ``public.salt`` and the verifier's ``live`` params, and the check item's coins are live.
* **The registration** is one-stage's record whose window is PoUW's form, ``{context, kind: "served", public}``
  (`core_check.registered`): ``public`` holds the registered items (`window.REGISTERED`: work, check, sampling, length,
  root_W), ``context`` is the run root and ``activations`` the operands. What one-stage's window would name (the
  circuit record's digest, the run root, the index's digest, the order) goes to ``provenance.json`` beside it
  (`Registrar`), which the audit holds to the circuit the verifier registered (`pouw_audit`). The check item's coins are
  live for a live window and seeded otherwise, and ε, δ and the entries bound are all-protocols' PoUW leg's (`EPS`,
  `DELTA`, `ENTRIES`).
  `params` gives the verifier's parameters it is checked under.
* **Tamper** flips bit 0 of every real tile's z before it is recorded: a drawn real tile's proof refuses it.

Gaps, stated: a live window's salt is the session's, but the tiles' salt port is a value the prover records, so nothing
in the proof yet ties the proved tiles to it (the statement takes no coin input from the verifier); the A rows are not proved to be the served ``x`` rows (one token's
row each, an A row R tokens' half rows: a gather, CIR-100), and the B rows not to be the registered weights' (one
channel's row each); the roots port holds the operands' roots but no read declares it, so the tile's receipt binds them
only as the prover recorded them; the unit's proof is of the window's staged statement, not the pinned flat circuit
`core_check.opening` reads; root_W is null.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from fractions import Fraction
from pathlib import Path
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from verity.core.protocols.pouw import core_check as CC, operands as OP, window as Wd
from verity.ml.protocols.pouw.circuit import h1tile as Ht, seeds as SD
from verity.core.protocols.sampled_proofs.one_stage import draw as D, registration as R
from verity.ml.circuits.codec import canonical_json, descriptor_id
from verity.ml.circuits.defs import bind
from verity.ml.circuits.ops.boolean import gemm as BG
from verity.ml.windows import interface as IF

from . import loop as L

TILE_K, TILE_R, TILE_C = 64, 4, 8
PAIRS = tuple((i, j) for i in range(TILE_R) for j in range(TILE_C))
#: the tile's work: its MACs
TILE_WORK = TILE_K * TILE_R * TILE_C
#: the verifier's ε and δ and its entries bound (all-protocols' PoUW leg's ``H1_EPS``, ``H1_DELTA``, ``H1_ENTRIES``)
EPS, DELTA, ENTRIES = Fraction(1, 2), Fraction(1, 1000000), 1024
#: the verifier's declared range of tile counts (the leg's ``H1_RANGE``)
RANGE = (1, 1 << 16)
#: tiles a statement
CAP = 8
#: the operands' salts, the prover's, beside the window
OPERANDS_PRIVATE = "operands-private.json"
#: a live window's salt session, beside the window (the leg's ``SALT``): ``{answer, live}``
SALT = "salt.json"
SALT_BYTES = Ht.SALT_BYTES


def tile() -> Any:
    """The pinned H1 tile (`core_check.TILES`)."""
    return bind(Ht.PouwH1Tile, EXACT=bind(BG.GemmAccumulatorS8, K=TILE_K, **BG.GEMM_SM120_S8), R=TILE_R, C=TILE_C,
                PAIRS=PAIRS)


def resolve(tid: str) -> Any:
    """The verifier's catalog for the loop's PoUW window: the pinned tile alone."""
    t = tile()
    if tid != descriptor_id(t):
        raise KeyError(f"{tid}: not the pinned H1 tile")
    return t


def check_item(entries: int = ENTRIES, live: bool = False) -> dict[str, Any]:
    """The check item, its coins live (`window.COINS`) when the window's salt is a live session's."""
    return {**({Wd.COINS: "live"} if live else {}), "entries": entries, "scheme": CC.SCHEME, "template": tile().id,
            "tile_work": TILE_WORK}


class Registrar:
    """One-stage's registrar (`registration`), the record's window replaced by PoUW's ``{context, kind, public}``:
    ``public`` the registered ``items``, ``context`` the run root, ``activations`` the tiles' operands
    (`operands.block`), ``registered`` the pinned weights' entries when given (the W `core_check.window_anchors` reads),
    and the window one-stage would name kept as ``provenance``."""
    KINDS, FORMAT, digest = R.KINDS, R.FORMAT, staticmethod(R.digest)

    def __init__(self, items: Mapping[str, Any], operands: Mapping[str, Any],
                 registered: Mapping[str, Any] | None = None) -> None:
        self.items, self.operands, self.provenance = dict(items), dict(operands), None
        self.registered = None if registered is None else dict(registered)

    def record(self, program, query, law, scheme, roots, window, **kw) -> dict[str, Any]:
        self.provenance = dict(window)
        return R.record(program, query, law, scheme, roots, {"context": window["run"], "kind": window["kind"],
                                                             "public": self.items, OP.KEY: self.operands,
                                                             **({} if self.registered is None
                                                                else {"registered": self.registered})}, **kw)


def _u32(xs) -> np.ndarray:
    return np.asarray(xs, dtype=np.int64).astype(np.uint32)


def _words(rows: np.ndarray) -> list[int]:
    """An operand row (s8 codes) as the port's u16 words."""
    return IF.words((rows.astype(np.int64) & 0xFF).astype(np.uint8)[None], 8, 1, rows.size // 2)[0].tolist()


class H1Loop:
    """``loop``'s linears as H1 tiles (module docstring) at ε ``eps`` and δ ``delta``: the loop served and the tiles'
    operands committed (`operands`) on construction, before any salt; the tiles recorded under one (`window`).
    ``weights``: the pinned weights' entries (``{program, values}``, `weights_group`), which the registration names as
    ``registered``; ``live``: the salt is a live session's (`anchors`, `window_salt`), which needs them."""

    def __init__(self, loop: L.TorchLoop, eps: Fraction = EPS, delta: Fraction = DELTA, bits: int = Wd.GRID_BITS,
                 salts: Mapping[str, list[bytes]] | None = None, weights: Mapping[str, Any] | None = None,
                 live: bool = False) -> None:
        h = loop.model.h
        if h % TILE_K or h % TILE_C:
            raise ValueError(f"width {h}: the tiles need a multiple of {TILE_K} and {TILE_C}")
        if live and weights is None:
            raise ValueError("a live window's salt binds its registered weights' root W: give the pinned weights")
        self.loop, self.eps, self.delta, self.bits = loop, Fraction(eps), Fraction(delta), bits
        self.weights, self.live = (None if weights is None else dict(weights)), live
        self.blocks = -(-len(loop.requests) // TILE_R)
        self.calls = loop.steps * len(loop.model.layers)
        self.cols, self.ks = h // TILE_C, h // TILE_K
        self.real = self.calls * self.blocks * self.cols * self.ks
        self.work = Wd.bucket(self.real, bits)
        self.k = Wd.k_prime(self.eps, self.delta, self.work.credited * TILE_WORK, self.work.n, TILE_WORK)
        self.served = self._serve()
        a = [r[rb * TILE_R:(rb + 1) * TILE_R, kh * TILE_K:(kh + 1) * TILE_K]
             for _s, _l, r, _sx, _y in self.served for rb in range(self.blocks) for kh in range(self.ks)]
        b = [lin.wq.numpy()[cb * TILE_C:(cb + 1) * TILE_C, kh * TILE_K:(kh + 1) * TILE_K]
             for lin in loop.model.layers for cb in range(self.cols) for kh in range(self.ks)]
        self.a_rows = [*a, np.zeros((TILE_R, TILE_K), np.int64)]
        self.b_rows = [*b, np.zeros((TILE_C, TILE_K), np.int64)]
        self.values = OP.commit(self.program, loop.run_root,
                                {"a": [_words(x) for x in self.a_rows], "b": [_words(x) for x in self.b_rows]}, salts)
        self.operands = OP.block(self.program, self.values)
        self.roots = (self.values["a"].root, self.values["b"].root)

    @property
    def program(self) -> str:
        return f"torch-loop/pouw-h1-h{self.loop.model.h}"

    @property
    def X(self) -> bytes:
        """The operands' root (`operands.root`), fixed before the window's salt."""
        return OP.root(self.operands)

    def anchors(self) -> dict[str, Any]:
        """The ``roots`` and ``window`` `core_check.window_anchors` reads of this window's registration: no ``W`` root, its
        registered weights (W) and its operands (X), which `window_salt.issue` opens the salt session over."""
        return {"roots": [], "window": {OP.KEY: self.operands, "registered": self.weights}}

    @property
    def law(self) -> dict[str, Any]:
        return D.parse_law(f"subset:{self.k}")

    def circuit(self) -> IF.Circuit:
        """The window's circuit: its tiles, each one's ``a`` and ``b`` read from the committed operands, unit (step, layer,
        rb, cb, kh) reading A row (step, layer, rb, kh) and B row (layer, cb, kh), and the filler tiles the zero rows."""
        tid, words = descriptor_id(tile()), dict(IF.layout(tile()).rows)
        wa, wb, nl, fill = words["a"], words["b"], len(self.loop.model.layers), self.work.n - self.real
        radix = (self.loop.steps, nl, self.blocks, self.cols, self.ks)
        reads = (IF.Read(tid, "a", "a", 0, radix, 0, (nl * self.blocks * self.ks * wa, self.blocks * self.ks * wa,
                                                       self.ks * wa, 0, wa)),
                 IF.Read(tid, "b", "b", 0, radix, 0, (0, self.cols * self.ks * wb, 0, self.ks * wb, wb)),
                 IF.Read(tid, "a", "a", self.real, (fill,), (len(self.a_rows) - 1) * wa, (0,)),
                 IF.Read(tid, "b", "b", self.real, (fill,), (len(self.b_rows) - 1) * wb, (0,)))
        reg = IF.Registered({"a": len(self.a_rows) * wa, "b": len(self.b_rows) * wb}, reads=reads)
        return IF.Circuit(self.program, ((tile(), self.work.n),), registered=reg)

    def items(self) -> dict[str, Any]:
        """The registered items (`window.REGISTERED`), as `core_check.expected_items` computes them."""
        return {"check": check_item(live=self.live), "root_W": None, "work": self.work.to_json(TILE_WORK),
                "sampling": Wd.sampling(self.eps, self.delta, self.work, TILE_WORK, self.bits),
                "length": Wd.length_item([self.loop.steps] * len(self.loop.requests))}

    def _serve(self) -> list[tuple[int, int, np.ndarray, torch.Tensor, torch.Tensor]]:
        """Each (step, layer)'s linear as the loop serves it: its token rows padded to whole blocks of R, its scales and
        the loop's own output."""
        m, h, n = self.loop.model, self.loop.model.h, len(self.loop.requests)
        x, out = m.embed[self.loop.requests], []
        for step in range(self.loop.steps):
            for layer, lin in enumerate(m.layers):
                xq, sx = L.quantize(x)
                y = L.linear(xq, sx, lin.wq, lin.sw)
                rows = np.zeros((self.blocks * TILE_R, h), np.int64)
                rows[:n] = xq.numpy()
                out.append((step, layer, rows, sx, y))
                x = torch.relu(y.to(torch.float32)) if layer == 0 else y.to(torch.float32)
        return out

    def _tile(self, win: IF.Window, u: int, a: np.ndarray, b: np.ndarray, salt: bytes, tamper: bool, label: Any
              ) -> np.ndarray:
        """Tile u on rows ``a`` (R × 64 s8) and ``b`` (C × 64 s8), recorded; -> its z, (R, C) int64."""
        a8, b8 = (a.astype(np.int64) & 0xFF), (b.astype(np.int64) & 0xFF)
        z, rc = Ht.reference(a8.tolist(), b8.tolist(), salt, u, self.roots)
        zw = _u32(z) ^ np.uint32(1) if tamper else _u32(z)
        win.record(tile(), [a8.astype(np.uint8)[None], b8.astype(np.uint8)[None], np.asarray([SD.halves(salt)], np.uint32),
                            np.asarray([SD.halves(u.to_bytes(8, "big"))], np.uint32),
                            np.asarray([[SD.halves(self.roots[0]), SD.halves(self.roots[1])]], np.uint32)],
                   [zw[None], _u32(rc)[None]], label)
        return np.asarray(z, np.int64).astype(np.uint32).view(np.int32).astype(np.int64).reshape(TILE_R, TILE_C)

    def run(self, win: IF.Window, salt: bytes) -> None:
        """Every linear's tiles recorded into ``win`` under ``salt`` in unit order, their z held to the loop's own output;
        then the filler tiles."""
        if len(salt) != SALT_BYTES:
            raise ValueError(f"the tile's salt is {SALT_BYTES} bytes, not {len(salt)}")
        m, n, u = self.loop.model, len(self.loop.requests), 0
        nb = self.cols * self.ks
        for c, (step, layer, _rows, sx, y) in enumerate(self.served):
            lin, acc = m.layers[layer], np.zeros((self.blocks * TILE_R, m.h), np.int64)
            for rb in range(self.blocks):
                for cb in range(self.cols):
                    for kh in range(self.ks):
                        a = self.a_rows[(c * self.blocks + rb) * self.ks + kh]
                        b = self.b_rows[layer * nb + cb * self.ks + kh]
                        acc[rb * TILE_R:(rb + 1) * TILE_R, cb * TILE_C:(cb + 1) * TILE_C] += self._tile(
                            win, u, a, b, salt, self.loop.tamper,
                            {"step": step, "layer": layer, "op": m.op_path(layer), "rows": rb, "cols": cb, "k": kh})
                        u += 1
            got = ((torch.from_numpy(acc[:n]).to(torch.float32) * sx[:, None]) * lin.sw[None, :]).to(torch.bfloat16)
            if not torch.equal(got.view(torch.int16), y.view(torch.int16)):
                raise AssertionError(f"step {step} layer {layer}: the tiles' z is not the loop's linear")
        while u < self.work.n:
            self._tile(win, u, self.a_rows[-1], self.b_rows[-1], salt, False, {"filler": u})
            u += 1

    def window(self, salt: bytes, cap: int = CAP) -> IF.Window:
        """The window under ``salt`` (64 bytes, released against `X` after the operands were committed)."""
        win = IF.Window(self.circuit(), self.loop.run_root, self.law, cap, registered=self.values)
        self.run(win, salt)
        return win

    def write(self, win: IF.Window, out: str | Path, provenance: Mapping[str, Any], salt_key: bytes | None = None,
              issued: Any = None) -> dict[str, Any]:
        """The window's files (`Window.write`) under PoUW's registration (`Registrar`), ``provenance.json``,
        ``operands-private.json`` (the operands' salts, the prover's) and, for a live window, `SALT` (``issued``,
        `window_salt.issue`'s: its answer's ``public.salt`` and the verifier's ``live`` params)."""
        reg = Registrar(self.items(), self.operands, self.weights)
        win.write(win.close(salt_key=salt_key), out, provenance, registrar=reg)
        if issued is not None:
            (Path(out) / SALT).write_text(json.dumps({"answer": issued.answer, "live": issued.live}, sort_keys=True))
        (Path(out) / "provenance.json").write_bytes(canonical_json(reg.provenance))
        fd = os.open(Path(out) / OPERANDS_PRIVATE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump({"program": self.program, "salts": {v: [x.hex() for x in r.salts] for v, r in self.values.items()}}, f)
        return json.loads((Path(out) / "registration.json").read_text())

    def params(self, live: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """The verifier's parameters of the window (`core_check.params`): the pinned tile, the run root as its context, ε,
        δ, the grid, the requests' max_tokens and the declared work range `RANGE`; a live window's salt session's inputs
        ``live`` (`window_salt.Issued.live`)."""
        return {"check": check_item(live=self.live), "context": self.loop.run_root,
                "delta": str(self.delta), "eps": str(self.eps), "grid_bits": self.bits,
                "length": {"max_tokens": [self.loop.steps] * len(self.loop.requests), "stop": Wd.STOP_FIXED},
                "query": CC.TILES[tile().id].query, "range": list(RANGE), **({CC.LIVE_PARAM: dict(live)} if live else {})}


def fresh_salt() -> bytes:
    return os.urandom(SALT_BYTES)


def weights_group(model: L.Model, values: Mapping[str, Any]) -> dict[str, Any]:
    """The pinned weights' entries as the registration's window names them (``registered``, `loop.pin`'s program and
    values): the group whose root `core_check.values_root` is the window's W."""
    return {"program": model.name, "values": {v: r.entry() for v, r in values.items()}}


def window_salt() -> Any:
    """`benchmarks/pouw/window_salt/window_salt.py`, which opens a live window's salt session (`issue`)."""
    if "window_salt" not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            "window_salt", Path(__file__).resolve().parents[3] / "benchmarks" / "pouw" / "window_salt" / "window_salt.py")
        module = sys.modules["window_salt"] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return sys.modules["window_salt"]
