"""`LazyProgramLocator`: the verifier's gate -> located-unit derivation over a lazy `Serve_v1` program (LZ, 2026-09-10).

Same answers as :class:`e2e.verifier_v3.locator.ProgramLocator` (it *is* one: the class inherits `gemms_of`,
`locate_gemm_coordinate` and `locate_input` unchanged and replaces only the construction and `root_of`), but the
construction touches no root call and `root_of` is arithmetic over :class:`verity_vllm.ir.lazy.ServeLayout`, so the
memory of a `locate_selected` run is bounded by the selected units, not by the program's runs:

* `ProgramLocator.__init__` builds `StructuralCircuitMap(program)` -- `list(circuit.root_calls())`, one `FunctionCall`
  per root call, then names them by counting and checks the dataflow -- O(roots) = O(NL * P) objects on top of the
  fully materialised `Program` (A3: 1.7 s and O(runs) memory at 32x15, not buildable at 6500x1599).
* Here `root_of(g)` is `layout.node_at(g)` -> structural role -> name, and one `FunctionCall` over the lazy root for
  the touched node only (an LRU of 64 root calls); the weight ranges and the prompt range are the same
  `_coll_range` computations over the same `input_coll`s (O(NL) ranges).

The structural naming is the one `StructuralCircuitMap` reconstructs by counting (`t=<t>/embed`, `t=<t>/layers.<L>/
{pre,attn,post}`, `t=<t>/final`) and the one `registry.b1.Serve` annotates; the dataflow checks the map performs at
construction are available on demand for a sample of positions (:meth:`check_dataflow`) instead of for every root.

Equality with `ProgramLocator` (digest bytes, `root_of`, `locate_gemm_coordinate`, `locate_input`, `Circuit.gate`) is
asserted by `tests/contract/lz_lazy_program_test.py` at TINY (both TINY configurations) and 32x15, and on a CPU pod at
the larger shapes where the full build still exists (`data/contract/lz-lazy-program/`).

Usage (the verifier's side; see `data/contract/lz-lazy-program/patches/verifier_v3.diff` for the wiring):

    from verity_vllm.ir.lazy import LazyProgram, lazy_program_digest
    program = LazyProgram(bind(b1.Serve, C=C, LP=LP, STEPS=STEPS))
    digest = lazy_program_digest(program)             # == program_digest(encode_program(Program(sp)))
    loc = LazyProgramLocator(program, C)
    units = locate_selected(loc, rows, klass=..., ru_request=..., n_v=...)
"""

from __future__ import annotations

import re
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE.parent.parent), str(_HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from e2e.verifier_v3.locator import ProgramLocator

__all__ = ["LazyProgramLocator", "build_lazy_program"]


class _Capped(OrderedDict):
    """A dict keeping at most `cap` most-recently-used entries: the inherited `gemms_of` / `locate_gemm_coordinate` caches (a Gemm's member
    table is one entry per member, 49 152 for lm_head) would otherwise grow with every distinct root a query touches (MEASURED: 1.2 GB for
    10^4 random gates at 32x15 with the unbounded dicts)."""

    def __init__(self, cap: int) -> None:
        super().__init__()
        self.cap = cap

    def __setitem__(self, key, value) -> None:
        super().__setitem__(key, value)
        if len(self) > self.cap:
            self.popitem(last=False)

    def __getitem__(self, key):
        v = super().__getitem__(key)
        self.move_to_end(key)
        return v


class LazyProgramLocator(ProgramLocator):
    """`ProgramLocator` over a :class:`verity_vllm.ir.lazy.LazyProgram`; see the module docstring."""

    def __init__(self, program, C: dict[str, Any], *, call_cache: int = 64, gemm_cache: int = 256, member_cache: int = 128) -> None:
        from verity_vllm.correspondence.core import _LAYER_FIELDS, _WEIGHT_FIELDS, _coll_range
        from verity_vllm.ir.lazy import LazyProgram

        if not isinstance(program, LazyProgram):
            raise TypeError(f"LazyProgramLocator needs a verity_vllm.ir.lazy.LazyProgram, got {type(program).__name__} (use ProgramLocator)")
        self.program = program
        self.layout = program.layout
        self.cmap = _LazyMap(self)  # `cmap.named[name]` by arithmetic, no StructuralCircuitMap
        # the two caches ProgramLocator keeps (same keys, same contents), bounded: memory follows the working set of roots, not |Q|
        self._gemms: dict[str, list] = _Capped(gemm_cache)
        self._members: dict[tuple[str, int], tuple[dict[tuple[int, int], int], int]] = _Capped(member_cache)
        self._calls: OrderedDict[int, Any] = OrderedDict()
        self._call_cache = call_cache
        circuit = program.circuit
        # weights: field collections -> [start, end) gate ranges with the weights leaf canonical id (== ProgramLocator)
        W = program.input_coll(1)
        ranges: list[tuple[int, int, str, str]] = []
        s, e = _coll_range(circuit, W["embed"])
        ranges.append((s, e, _WEIGHT_FIELDS["embed"][0], "weights.embed"))
        for L in range(C["NL"]):
            w = W["layers"][L]
            for fname, (cid_t, _srcs) in _LAYER_FIELDS.items():
                if fname == "qkv_b" and not C.get("BIAS", True):
                    continue
                s, e = _coll_range(circuit, w[fname])
                ranges.append((s, e, cid_t.format(L=L), f"weights.layers[{L}].{fname}"))
        s, e = _coll_range(circuit, W["norm"])
        ranges.append((s, e, _WEIGHT_FIELDS["norm"][0], "weights.norm"))
        s, e = _coll_range(circuit, W["cos_sin"])
        ranges.append((s, e, _WEIGHT_FIELDS["cos_sin"][0], "weights.cos_sin"))
        ranges.sort()
        self.w_ranges = ranges
        self.w_starts = [r[0] for r in ranges]
        P = program.input_coll(0)
        self.prompt_range = _coll_range(circuit, P)
        self.input_gates = program.input_gates

    # -- root calls, arithmetically -------------------------------------------------------------------------
    def root_call(self, k: int):
        """The `FunctionCall` of root node k (a named root: not one of the two Input batches)."""
        from verity.ir.layout import FunctionCall

        c = self._calls.get(k)
        if c is None:
            circuit = self.program.circuit
            c = FunctionCall(circuit, circuit.root_scope, k, 0)
            self._calls[k] = c
            if len(self._calls) > self._call_cache:
                self._calls.popitem(last=False)
        else:
            self._calls.move_to_end(k)
        return c

    def root_of(self, g: int):
        """`(structural name, FunctionCall)` of the root call containing gate g; `(None, None)` for an Input gate or a gate
        outside the circuit -- exactly `ProgramLocator.root_of` (whose bisection over the named roots misses the same gates)."""
        lay = self.layout
        if not (0 <= g < lay.gates):
            return None, None
        k = lay.node_at(g)
        if lay.role(k)[0] == "input":
            return None, None
        return lay.name(k), self.root_call(k)

    # -- optional structural checks (what StructuralCircuitMap asserts for every root, here for chosen positions) -------
    def check_dataflow(self, positions: list[int]) -> list[str]:
        """`StructuralCircuitMap._check_dataflow` for the given positions t: Attention's static T == t+1; LayerPost's `attn`
        argument is the Attention call's returns; a decode Embedding reads the previous Final's token; a prefill Embedding
        reads prompt Input gate t.  Returns the list of violations (empty == the checked positions agree)."""
        from verity.ir.layout import param_leaf, resolve

        lay = self.layout
        circuit = self.program.circuit
        errs: list[str] = []

        def first_args(c, n: int) -> tuple[int, ...]:
            sc = c.scope()
            return tuple(param_leaf(sc, 0, j) for j in range(n))

        for t in positions:
            for L in range(lay.NL):
                attn = self.root_call(lay.index(("attn", t, L)))
                T = attn.fn.bindings.get("T")
                if T != t + 1:
                    errs.append(f"t={t}/layers.{L}/attn: Attention static T={T} but position t={t}")
                post = self.root_call(lay.index(("post", t, L)))
                rets = tuple(attn.returns)
                if first_args(post, len(rets)) != rets:
                    errs.append(f"t={t}/layers.{L}/post: arg[0:{len(rets)}] is not the returns of {attn!r}")
            emb = self.root_call(lay.index(("embed", t)))
            if t >= lay.LP:
                fin = self.root_call(lay.index(("final", t - 1)))
                if first_args(emb, 1) != tuple(fin.returns[:1]):
                    errs.append(f"t={t}/embed: token operand {first_args(emb, 1)} is not Final(t={t - 1}) return {fin.returns[:1]}")
            else:
                g = resolve(circuit.root_scope, self.program.input_coll(0).refs[t])
                if first_args(emb, 1)[0] != g:
                    errs.append(f"t={t}/embed: token operand {first_args(emb, 1)[0]} is not prompt Input gate {g}")
        return errs


_NAME_RE = re.compile(r"^t=(\d+)/(?:(embed|final)|layers\.(\d+)/(pre|attn|post))$")


class _LazyNamed:
    """`StructuralCircuitMap.named` by arithmetic: `named["t=8/layers.3/attn"]` is the root call of that structural name."""

    def __init__(self, loc: LazyProgramLocator) -> None:
        self.loc = loc

    def _index(self, name: str) -> int | None:
        m = _NAME_RE.match(name)
        if not m:
            return None
        lay = self.loc.layout
        t = int(m.group(1))
        if m.group(2) == "embed":
            role: tuple = ("embed", t)
        elif m.group(2) == "final":
            role = ("final", t)
        else:
            role = (m.group(4), t, int(m.group(3)))
        if not (0 <= t < lay.P) or (role[0] == "final" and t < lay.LP - 1) or (len(role) == 3 and not (0 <= role[2] < lay.NL)):
            return None
        return lay.index(role)

    def __contains__(self, name: str) -> bool:
        return self._index(name) is not None

    def __getitem__(self, name: str):
        k = self._index(name)
        if k is None:
            raise KeyError(name)
        return self.loc.root_call(k)

    def get(self, name: str, default=None):
        k = self._index(name)
        return default if k is None else self.loc.root_call(k)

    def __len__(self) -> int:
        return self.loc.layout.n_nodes - 2

    def __iter__(self):
        lay = self.loc.layout
        return (lay.name(k) for k in range(2, lay.n_nodes))

    def items(self):
        for name in self:
            yield name, self[name]


class _LazyMap:
    """The two attributes of `StructuralCircuitMap` the locator's callers use (`named`, `structural_names` on demand)."""

    def __init__(self, loc: LazyProgramLocator) -> None:
        self.named = _LazyNamed(loc)
        self.program = loc.program
        self.circuit = loc.program.circuit
        lay = loc.layout
        self.LP, self.STEPS, self.NL = lay.LP, lay.STEPS, lay.NL


def build_lazy_program(C: dict[str, Any], LP: int, STEPS: int):
    """`LazyProgram(bind(Serve, C=C, LP=LP, STEPS=STEPS))` -- the verifier's in-process program without the root body."""
    from verity.ir import bind
    from verity_vllm.ir.lazy import LazyProgram
    from verity_vllm.registry import b1

    return LazyProgram(bind(b1.Serve, C=C, LP=LP, STEPS=STEPS))
