"""E1 — stream every computed gate value of ONE declared TINY request (RU) with the `verity.ir` reference interpreter.

What it does (brief `out/briefs/EOD-2026-09-11.md`, interface 2; scoping `data/contract/a3-sequence/interior-commit-scope.md` §2 (a)):

1. Builds the pinned program (`protocol_instance.json#program.identity`, lazy `verity_vllm.ir.lazy.LazyProgram`, digest checked against the
   pin by `e2e.verifier_v3.locator.build_pinned_program`).
2. Binds the RU's committed inputs: prompt token ids from the RU boundary (`boundary.json`, re-hashed into the boundary tree and compared
   with the committed root in `commitments.json`), weights from the checkpoint (every weights leaf's sha256 compared with the committed
   `weights/index.json`), the `cos_sin_cache` table from the GPU-built fixture (sha256 compared with its committed leaf; the CPU rebuild
   does not reproduce it -- `verity_vllm.correspondence.core.COS_SIN_REBUILT_FROM`).
3. Evaluates every root call of the program in canonical (gate) order with `verity.ir.evaluate.evaluate_region` -- the same evaluator,
   the same primitive semantics; only the gate enumeration is a hierarchy walk (`CallWalk`) instead of a per-gate bisection, checked
   against `Circuit.gate` on a sample -- and streams each root call's gates as value-record JSONL rows `{"g","k","w","v"}` sorted by gate
   index to `interior/<req>/<step>/<call id>.jsonl`.  Memory: the transcript keeps only the returns of finished root calls.
4. After each root call, hands the interpreter's words of every committed operator-boundary leaf completed by that call to
   `compare_boundaries.BoundaryIndex` (hash exactly as the store, compare with the committed leaf).
5. Writes `inventory.json` (required gates = every VU output gate of the RU that is NOT at an operator boundary; boundary derived
   from the committed index.json of attempt 6 through `CircuitMap.locate` until E3's `RecordPartition.locate` oracle lands -- stated in
   the file), `required_gates.bin` (uint32 LE, sorted), `boundary_compare.json`, `rate.json`, `progress.jsonl`.

Laptop: smoke tests only (`--prefill-layers N` / `--max-root-calls N`), under `zk/campaign/lane_run.py` with a declared budget.  The
full RU (107,636,385 computed gates -- MEASURED from the layout; A3's 26.9 M divided the per-request count by the 4 requests) runs on a
CPU pod.

    PYTHONPATH=. python -m e2e.interior.producer --record ~/veritor-archive/A9-TINY-elem-b8177939/rec/V5-TINY-bex0-inc-elem \
        --request 35285052-1526-5943-b9a6-3a2c8bbecc4a --out out/eod/e1/smoke --prefill-layers 2
"""

from __future__ import annotations

import argparse
import array
import bisect
import glob
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
for _p in (REPO, REPO / "vllm-poc", REPO / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from verity_vllm.correspondence.core import _LAYER_FIELDS, _WEIGHT_FIELDS, _coll_range, load_safetensors  # noqa: E402
from verity.ir.defs import PrimitiveDefinition  # noqa: E402
from verity.ir.evaluate import MissingValue, evaluate_region  # noqa: E402
from verity.ir.layout import WIDE_FAN_IN, FunctionCall, Gate, LazyOperands, Scope, _prim_member_args, resolve  # noqa: E402
from verity.ir.refs import Affine  # noqa: E402

from . import vecwalk as _vw  # noqa: E402
from .compare_boundaries import BoundaryIndex  # noqa: E402

DEFAULT_RECORD = Path.home() / "veritor-archive" / "A9-TINY-elem-b8177939" / "rec" / "V5-TINY-bex0-inc-elem"
DEFAULT_REQUEST = "35285052-1526-5943-b9a6-3a2c8bbecc4a"  # RU 1 of attempt 6 (J = [1])
DEFAULT_COS_SIN = REPO / "fixtures" / "B0-divergence-20260907T1604Z" / "layers.0.self_attn.rotary_emb" / "cos_sin_cache.npy"
HF_GLOB = str(Path.home() / ".cache" / "huggingface" / "hub" / "models--HuggingFaceTB--SmolLM2-135M" / "snapshots" / "*" / "model.safetensors")
SCHEMA_VALUES = "veritor/e1/value-record/v1"


def rss_gb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024 ** 3) if sys.platform == "darwin" else r / (1024 ** 2)


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ---------------------------------------------------------------------------------------------------------------------------------
# transcript: computed values in the dict, prescribed inputs from two flat arrays
# ---------------------------------------------------------------------------------------------------------------------------------


class StreamTranscript(dict):
    """`ExecutionTranscript` whose Input gates are read from flat arrays: prompt words at [0, LP), weights words at [ws, we).
    `array.array` indexing yields Python ints (the evaluator's operand type; never numpy scalars)."""

    def __init__(self, lp: int, prompt: array.array, ws: int, weights: array.array) -> None:
        super().__init__()
        self.lp, self.prompt, self.ws, self.we, self.weights = lp, prompt, ws, ws + len(weights), weights
        # [E2 SCALE follow-up] call blocks: the words of one whole root call `[lo, hi)` as a uint64 column while the call is being
        # written / compared (no 200 k dict inserts + deletes per call); `drop_block` keeps the returns as dict entries
        self.blk_lo: list[int] = []
        self.blk: dict[int, tuple[int, np.ndarray]] = {}

    def add_block(self, lo: int, vals: np.ndarray) -> None:
        vals = np.ascontiguousarray(vals, dtype=np.uint64)
        self.blk[lo] = (lo + len(vals), vals)
        bisect.insort(self.blk_lo, lo)

    def drop_block(self, lo: int, keep) -> None:
        """Remove the block at `lo`; the gates `keep` (a call's returns) become ordinary entries."""
        hi, vals = self.blk.pop(lo)
        self.blk_lo.remove(lo)
        for g in keep:
            if lo <= g < hi:  # a call's returns may pass a parameter leaf through (a gate before the call): not this block's
                dict.__setitem__(self, g, int(vals[g - lo]))

    def _block_word(self, g: int):
        i = bisect.bisect_right(self.blk_lo, g) - 1
        if i >= 0:
            lo = self.blk_lo[i]
            hi, vals = self.blk[lo]
            if g < hi:
                return int(vals[g - lo])
        return None

    def __missing__(self, g: int) -> int:
        if self.ws <= g < self.we:
            return self.weights[g - self.ws]
        if 0 <= g < self.lp:
            return self.prompt[g]
        if self.blk_lo:
            v = self._block_word(g)
            if v is not None:
                return v
        raise KeyError(g)

    def __contains__(self, g: object) -> bool:  # type: ignore[override]
        return dict.__contains__(self, g) or (isinstance(g, int) and ((self.ws <= g < self.we) or (0 <= g < self.lp) or
                                                                       (bool(self.blk_lo) and self._block_word(g) is not None)))

    def gather(self, gates: np.ndarray) -> np.ndarray:
        """[E2 SCALE] the words of many gates as one uint64 column: weights / prompt straight from their arrays, computed values
        through the dict (same values as `self[g]` one by one)."""
        gates = np.asarray(gates, dtype=np.int64)
        out = np.empty(len(gates), dtype=np.uint64)
        w = (gates >= self.ws) & (gates < self.we)
        if w.any():
            out[w] = np.frombuffer(self.weights, dtype=np.uint16)[gates[w] - self.ws]
        p = (gates >= 0) & (gates < self.lp)
        if p.any():
            out[p] = np.frombuffer(self.prompt, dtype=np.uintc)[gates[p]]  # array('I') == C unsigned int
        r = ~(w | p)
        if r.any():
            for lo in self.blk_lo:
                hi, vals = self.blk[lo]
                b = r & (gates >= lo) & (gates < hi)
                if b.any():
                    out[b] = vals[gates[b] - lo]
                    r &= ~b
            if r.any():
                out[r] = np.array([self[g] for g in gates[r].tolist()], dtype=np.uint64)
        return out


# ---------------------------------------------------------------------------------------------------------------------------------
# gate enumeration of one call by hierarchy walk (== Circuit.gate(g) for every g of the call, checked by `check_walk`)
# ---------------------------------------------------------------------------------------------------------------------------------


class FastArgs:
    """Operand resolution with two semantics-preserving shortcuts over `layout._prim_member_args` (checked against `Circuit.gate` by
    `check_walk`): (i) a parameter reference `("p", pi, leaf)` whose chain of argument reference sequences up to a root `Input` batch is
    all `Affine` (unit stride) resolves arithmetically, `gate = A + leaf` -- every weight operand; (ii) every other body-scoped
    reference is memoised per live scope object (the activation operands a batch's members share).  Scopes are looked up by `id` and
    the stored scope is identity-checked, so a recycled id can never return another scope's result.  Wide fan-in nodes keep
    `LazyOperands` unchanged."""

    def __init__(self) -> None:
        self.aff: dict[int, tuple[Scope, dict]] = {}
        self.memo: dict[int, tuple[Scope, dict]] = {}
        self.stats = {"affine": 0, "memo_hit": 0, "memo_miss": 0}

    def _entry(self, table: dict, scope: Scope) -> dict:
        ent = table.get(id(scope))
        if ent is None or ent[0] is not scope:
            ent = (scope, {})
            table[id(scope)] = ent
        return ent[1]

    def affine_param(self, scope: Scope, pi: int):
        """`(A, B)` with `param_leaf(scope, pi, leaf) == A + B*leaf` for every leaf of parameter `pi`, or None when the chain is not
        provably affine (then the generic `resolve` is used)."""
        d = self._entry(self.aff, scope)
        if pi in d:
            return d[pi]
        res = None
        parent = scope.parent
        if parent is not None:
            node = parent.spec.body.nodes[scope.node_idx]
            refs = off = None
            if node.form == "call":
                refs, off = node.args[pi].refs, 0
            elif node.form == "batch":
                ax = node.axes[pi]
                a = node.args[pi]
                if ax is None:
                    refs, off = a.refs, 0
                elif ax == 0:
                    refs, off = a.refs, scope.member * a.type.elem.leaves
            if isinstance(refs, Affine):
                if refs.space == "p":
                    inner = self.affine_param(parent, refs.idx)
                    if inner is not None:
                        res = (inner[0] + inner[1] * (refs.base + off), inner[1])
                else:
                    pbody = parent.spec.body
                    nd = pbody.nodes[refs.idx]
                    if nd.form == "batch" and isinstance(nd.fn, PrimitiveDefinition):  # leaf j of a primitive batch is its gate node_off + j
                        res = (parent.offset + pbody.offsets[refs.idx] + refs.base + off, 1)
        d[pi] = res
        return res

    def res(self, scope: Scope, ref) -> int:
        """== `layout.resolve(scope, ref)`."""
        space, idx, leaf = ref
        if space == "n":
            body = scope.spec.body
            node = body.nodes[idx]
            fn = node.fn
            if isinstance(fn, PrimitiveDefinition):  # a primitive node's leaf is its own gate
                node_off = scope.offset + body.offsets[idx]
                return node_off if node.form == "primitive" else node_off + leaf // fn.out_leaves
            d = self._entry(self.memo, scope)  # composite target: chase its returns once per (scope, ref)
            v = d.get(ref)
            if v is None:
                v = d[ref] = resolve(scope, ref)
                self.stats["memo_miss"] += 1
            else:
                self.stats["memo_hit"] += 1
            return v
        ab = self.affine_param(scope, idx)
        if ab is not None:
            self.stats["affine"] += 1
            return ab[0] + ab[1] * leaf
        parent = scope.parent  # non-affine parameter: the same mapping as `layout.param_leaf`, continued through `res` in the parent scope
        node = parent.spec.body.nodes[scope.node_idx]
        if node.form == "call":
            return self.res(parent, node.args[idx].refs[leaf])
        if node.form == "batch":
            ax = node.axes[idx]
            a = node.args[idx]
            if ax is None:
                return self.res(parent, a.refs[leaf])
            if ax == 0:
                return self.res(parent, a.refs[scope.member * a.type.elem.leaves + leaf])
            return self.res(parent, a.take(ax, scope.member).refs[leaf])
        return resolve(scope, ref)  # scan parameters: generic

    def member_args(self, scope: Scope, node, m: int):
        """== `layout._prim_member_args(scope, node, m)`."""
        if sum(p.leaves for _, p in node.fn.params) > WIDE_FAN_IN:
            return LazyOperands(scope, node, m)
        r = self.res
        if node.form == "primitive":
            return tuple(r(scope, x) for a in node.args for x in a.refs)
        out: list[int] = []
        for a, ax in zip(node.args, node.axes):
            if ax is None:
                out.extend(r(scope, x) for x in a.refs)
            elif ax == 0:
                L = a.type.elem.leaves
                refs = a.refs
                out.extend(r(scope, refs[m * L + j]) for j in range(L))
            else:
                out.extend(r(scope, x) for x in a.take(ax, m).refs)
        return tuple(out)


class CallWalk:
    """Iterable of `Gate`s of a composite `FunctionCall`, in gate order; records `runs` = [(start, count, kind, width)] per primitive
    node as it goes (kind = `<primitive id>|<node form>`, the population's naming for primitive VUs).  Operands through `FastArgs`
    (default) or `layout._prim_member_args`."""

    def __init__(self, call: FunctionCall, fast: bool = True, vectorize: bool = False, vec_min_members: int = 4, flat: bool = True,
                 flat_check_all: bool = False) -> None:
        self.call = call
        # [E2 SCALE follow-up] flat (level-scheduled) template for calls the member template gets nothing out of (attention)
        self.flat, self.flat_check_all = flat, flat_check_all
        self.runs: list = []  # (start, count, kind, width) tuples and, with `vectorize`, `vecwalk.RunBlock`s -- in gate order
        self.fast = FastArgs() if fast else None
        # [E2 SCALE 2026-09-12] member-vectorised evaluation of composite batch nodes (`vecwalk`): the same gates and values, ~all
        # members of a GEMM at once in numpy instead of one Python gate at a time
        self.vectorize = vectorize
        self.vec_min_members = vec_min_members
        self._vec = _vw.evaluators() if vectorize else None

    def _scope(self, sc: Scope) -> Iterator[Gate]:
        body = sc.spec.body
        offs = body.offsets
        runs = self.runs
        args = self.fast.member_args if self.fast is not None else _prim_member_args
        for k, node in enumerate(body.nodes):
            node_off = sc.offset + offs[k]
            fn = node.fn
            if isinstance(fn, PrimitiveDefinition):
                form = node.form
                if form == "primitive":
                    runs.append((node_off, 1, f"{fn.id}|primitive", fn.ret.w))
                    yield Gate(node_off, fn, args(sc, node, 0), sc)
                else:
                    n = node.n
                    runs.append((node_off, n, f"{fn.id}|{form}", fn.ret.w))
                    for m in range(n):
                        yield Gate(node_off + m, fn, args(sc, node, m), sc)
            elif node.form == "call":
                yield from self._scope(Scope(fn, node_off, sc, k, 0))
            else:
                g = fn.gates
                for m in range(node.n):
                    yield from self._scope(Scope(fn, node_off + m * g, sc, k, m))

    def __iter__(self) -> Iterator[Gate]:
        self.runs = []
        return self._scope(self.call.scope())

    def _scope_selective(self, sc: Scope, base: int, member_los: set[int], keep) -> Iterator[Gate]:
        """[E2 SCALE follow-up] `_scope` that, at the composite batch nodes a `RootTemplate` evaluates as member templates (relative
        node offsets `member_los`), enumerates only the members `keep(n)` (the root template's check compares those; every other
        gate is enumerated).  Runs are not recorded."""
        body = sc.spec.body
        offs = body.offsets
        args = self.fast.member_args if self.fast is not None else _prim_member_args
        for k, node in enumerate(body.nodes):
            node_off = sc.offset + offs[k]
            fn = node.fn
            if isinstance(fn, PrimitiveDefinition):
                if node.form == "primitive":
                    yield Gate(node_off, fn, args(sc, node, 0), sc)
                else:
                    for m in range(node.n):
                        yield Gate(node_off + m, fn, args(sc, node, m), sc)
            elif node.form == "call":
                yield from self._scope_selective(Scope(fn, node_off, sc, k, 0), base, member_los, keep)
            else:
                g = fn.gates
                members = keep(node.n) if (node.form == "batch" and (node_off - base) in member_los) else range(node.n)
                for m in members:
                    yield from self._scope_selective(Scope(fn, node_off + m * g, sc, k, m), base, member_los, keep)

    # -- split evaluation: the members of large composite batches (the GEMM dot batches, ~95 % of the gates) across forked workers --
    def _members(self, sc: Scope, k: int, node, node_off: int, m0: int, m1: int) -> Iterator[Gate]:
        fn = node.fn
        g = fn.gates
        for m in range(m0, m1):
            yield from self._scope(Scope(fn, node_off + m * g, sc, k, m))

    def _split_scope(self, sc: Scope, circuit, tr, nproc: int, min_gates: int, stats: dict) -> None:
        """Evaluate the gates of scope `sc` in gate order into `tr`: primitive nodes and `call` nodes here (recursing), a composite
        `batch` node of >= `min_gates` gates member-sliced over `nproc` processes (this one + nproc-1 forked children that inherit the
        transcript copy-on-write, evaluate their slice and send back (runs, values); slices are contiguous in gate order, so `runs`
        stays in gate order).  Every value a child produces is computed by the same `evaluate_region` over the same gates as the
        sequential walk; only the process differs."""
        body = sc.spec.body
        offs = body.offsets
        args = self.fast.member_args if self.fast is not None else _prim_member_args
        for k, node in enumerate(body.nodes):
            node_off = sc.offset + offs[k]
            fn = node.fn
            if isinstance(fn, PrimitiveDefinition):
                form = node.form
                if form == "primitive":
                    self.runs.append((node_off, 1, f"{fn.id}|primitive", fn.ret.w))
                    evaluate_region(circuit, (Gate(node_off, fn, args(sc, node, 0), sc),), tr)
                else:
                    n = node.n
                    self.runs.append((node_off, n, f"{fn.id}|{form}", fn.ret.w))
                    evaluate_region(circuit, (Gate(node_off + m, fn, args(sc, node, m), sc) for m in range(n)), tr)
            elif node.form == "call":
                self._split_scope(Scope(fn, node_off, sc, k, 0), circuit, tr, nproc, min_gates, stats)
            else:  # batch (independent members) or scan (each member consumes the previous member's carry: never sliced)
                n = node.n
                # [E1 night 2026-09-12, for G4] a batch with FEWER members than processes (attention: one member per head, 9 / 15 / 14 heads;
                # at T >= 32 each member is large) was evaluated in the parent (E8 measured 35-57 us/gate on the attn root calls at 32x15):
                # slice it over min(nproc, n) processes -- one member per worker at least.  Same gates, same order, same values.
                if node.form == "batch" and self.vectorize and n >= self.vec_min_members and n * fn.gates >= min_gates \
                        and self._vectorized(sc, k, node, node_off, tr, stats):
                    continue
                if node.form == "batch" and nproc > 1 and n >= 2 and n * fn.gates >= min_gates:
                    self._parallel_members(sc, k, node, node_off, circuit, tr, min(nproc, n), stats, min_gates)
                else:
                    self._eval_members(sc, k, node, node_off, 0, n, circuit, tr, min_gates, stats)

    def _eval_members(self, sc: Scope, k: int, node, node_off: int, m0: int, m1: int, circuit, tr, min_gates: int, stats: dict) -> None:
        """Members `[m0, m1)` of a composite batch/scan node in one process.  With `vectorize`, each member is walked with
        `_split_scope` (nproc 1) so that composite batch nodes INSIDE it (the per-key score / PV batches of an attention head) are
        vectorised; otherwise the plain gate generator (same gates, same order)."""
        if not self.vectorize:
            evaluate_region(circuit, self._members(sc, k, node, node_off, m0, m1), tr)
            return
        fn = node.fn
        g = fn.gates
        for m in range(m0, m1):
            self._split_scope(Scope(fn, node_off + m * g, sc, k, m), circuit, tr, 1, min_gates, stats)

    def _vectorized(self, sc: Scope, k: int, node, node_off: int, tr, stats: dict) -> bool:
        """Evaluate every member of composite batch `node` at once (`vecwalk.Template`): template with the member index symbolic,
        checked against this walk's own gate enumeration of members 0, 1 and n-1, then evaluated column-wise.  False (nothing
        evaluated, nothing appended) when the node's references are not linear in the member index."""
        n = node.n
        try:
            tpl = _vw.Template(sc, k, node, node_off)
            for m in sorted({0, 1, n - 1}):
                tpl.check(m, _vw.concrete_gates_of_member(self, sc, k, node, node_off, m))
        except _vw.NotVectorizable as e:
            stats["not_vectorizable"].append(f"{node.fn.id} x{n}: {e}")
            return False
        tpl.evaluate(tr, self._vec, stats)
        self.runs.append(tpl.run_block())
        stats["vec_nodes"] += 1
        return True

    def _parallel_members(self, sc: Scope, k: int, node, node_off: int, circuit, tr, nproc: int, stats: dict, min_gates: int = 20000) -> None:
        import pickle
        fn = node.fn
        n, g = node.n, fn.gates
        bounds = [n * i // nproc for i in range(nproc + 1)]
        children = []
        for i in range(1, nproc):
            rfd, wfd = os.pipe()
            pid = os.fork()
            if pid == 0:  # child: slice i, then exit without running any parent cleanup
                os.close(rfd)
                code = 0
                try:
                    runs: list = []
                    saved, self.runs = self.runs, runs
                    self._eval_members(sc, k, node, node_off, bounds[i], bounds[i + 1], circuit, tr, min_gates, stats)
                    lo, hi = node_off + bounds[i] * g, node_off + bounds[i + 1] * g
                    payload = pickle.dumps((runs, [tr[x] for x in range(lo, hi)]), protocol=pickle.HIGHEST_PROTOCOL)
                except BaseException as e:  # noqa: BLE001
                    payload = pickle.dumps(("ERROR", repr(e)), protocol=pickle.HIGHEST_PROTOCOL)
                    code = 1
                with os.fdopen(wfd, "wb") as f:
                    f.write(payload)
                os._exit(code)
            os.close(wfd)
            children.append((pid, rfd, i))
        stats["forks"] += nproc - 1
        self._eval_members(sc, k, node, node_off, bounds[0], bounds[1], circuit, tr, min_gates, stats)  # parent: slice 0
        for pid, rfd, i in children:
            with os.fdopen(rfd, "rb") as f:
                data = f.read()
            _, status = os.waitpid(pid, 0)
            obj = pickle.loads(data) if data else ("ERROR", f"child {pid} sent nothing (status {status})")
            if obj[0] == "ERROR":
                raise RuntimeError(f"parallel member slice {i} of {fn.id} x{n}: {obj[1]}")
            runs, values = obj
            self.runs.extend(runs)
            lo = node_off + bounds[i] * g
            for j, v in enumerate(values):
                tr[lo + j] = v
            stats["values_received"] += len(values)

    def evaluate(self, circuit, tr, nproc: int = 1, min_gates: int = 20000) -> dict:
        """Evaluate the whole call into `tr` (== `evaluate_region(circuit, self, tr)`), member-sliced over `nproc` processes."""
        self.runs = []
        stats = {"forks": 0, "values_received": 0, "vec_nodes": 0, "vec_gates": 0, "scalar_gates": 0, "not_vectorizable": [], "flat": 0}
        if self.vectorize and self.flat and self._flat_evaluate(tr, stats):
            return stats
        if nproc <= 1 and not self.vectorize:
            evaluate_region(circuit, self, tr)
        else:
            self._split_scope(self.call.scope(), circuit, tr, nproc, min_gates, stats)
        return stats

    def _flat_evaluate(self, tr, stats: dict) -> bool:
        """[E2 SCALE follow-up] The whole call through `vecwalk.RootTemplate`: one template per FUNCTION id (member templates for
        the GEMM batches, level-scheduled flat segments for everything else -- the attention heads' per-key nodes, the norm chains),
        built and checked once, re-used by every call of the function.  The first call of every function id is checked gate by gate
        against this walk's own concrete enumeration (`flat_check_all`: every call).  False: not eligible (wide fan-in, > 4 M gates)
        or not buildable -- nothing evaluated, the per-node path applies."""
        ft = _vw.flat_template(self.call.fn, self.vec_min_members, stats)
        if ft is None:
            return False
        base = self.call.lo
        pl = _vw.param_leaves(self.call)
        if self.flat_check_all or ft.fn.id not in _vw.FLAT_CHECKED:
            # every flat gate and members {0, 1, n-1} of each member template vs this walk's concrete enumeration (raises on any difference)
            keep = _vw.check_members
            ft.check(base, pl.tolist(), self._scope_selective(self.call.scope(), base, ft.member_los(), keep), keep)
            _vw.FLAT_CHECKED.add(ft.fn.id)
        ft.evaluate(base, pl, tr, self._vec, stats)
        self.runs.extend(ft.run_blocks(base))
        stats["flat"] = 1
        return True


def check_walk(circuit, call: FunctionCall, sample: int, rng: np.random.Generator, fast: bool = True) -> dict:
    """`CallWalk` vs `Circuit.gate` on `sample` gates of the call (every gate when `sample` >= the call's size): same index sequence,
    primitive and operands."""
    walk = CallWalk(call, fast=fast)
    full = sample >= call.hi - call.lo
    idx = None if full else set(int(x) for x in rng.integers(call.lo, call.hi, size=sample))
    n = 0
    expect = call.lo
    for gv in walk:
        if gv.index != expect:
            raise AssertionError(f"walk order: got gate {gv.index}, expected {expect}")
        expect += 1
        if full or gv.index in idx:
            ref = circuit.gate(gv.index)
            a_ops, b_ops = ref.operands, gv.operands
            if len(a_ops) <= 64:
                same = tuple(a_ops) == tuple(b_ops)
            else:  # wide fan-in (LazyOperands): length + 32 probed positions, not 49,153 resolves per sample
                probe = [0, len(a_ops) - 1] + [int(x) for x in rng.integers(0, len(a_ops), size=30)]
                same = len(a_ops) == len(b_ops) and all(a_ops[j] == b_ops[j] for j in probe)
            if ref.prim is not gv.prim or not same:
                raise AssertionError(f"walk differs from Circuit.gate at {gv.index}: {gv!r} vs {ref!r}")
            n += 1
    if expect != call.hi:
        raise AssertionError(f"walk covered [{call.lo},{expect}) not [{call.lo},{call.hi})")
    return {"call": call.id, "gates": call.hi - call.lo, "gates_checked": n, "full": full, "runs": len(walk.runs),
            "fast_args": walk.fast.stats if walk.fast is not None else None}


# ---------------------------------------------------------------------------------------------------------------------------------
# committed inputs
# ---------------------------------------------------------------------------------------------------------------------------------


def authenticate_boundary(rec_dir: Path, request: str) -> dict:
    """Rebuild the request's boundary tree from `boundary.json` with the store's own code and compare with the committed root."""
    from verity_vllm.capture import Boundary, boundary_tree_id
    from verity_vllm.commit import MerkleTree
    sel = rec_dir / "adapter" / "selective"
    doc = json.loads((sel / "ru" / request / "boundary.json").read_text())
    commitments = json.loads((sel / "commitments.json").read_text())
    profile_id = commitments["profile_id"]
    b = Boundary.from_json(doc)
    problems = b.validate()
    tree = MerkleTree.from_values(b.leaves(profile_id), boundary_tree_id(request))
    committed = commitments["trees"][f"boundary/{request}"]
    root = tree.root.hex()
    return {"prompt_token_ids": list(b.prompt_token_ids), "output_token_ids": list(b.output_token_ids), "steps": [s.to_json() for s in b.steps],
            "validate_problems": problems, "root_rebuilt": root, "root_committed": committed["root"], "leaf_count": tree.leaf_count,
            "root_matches_committed": root == committed["root"] and tree.leaf_count == committed["leaf_count"], "profile_id": profile_id}


def bind_inputs(program, circuit, rec_dir: Path | None, request: str, checkpoint: Path, cos_sin_npy: Path | None, C: dict, prompt: list[int]) -> tuple[StreamTranscript, dict]:
    """Prompt words at the prompt Input gates; every weight Input gate from the checkpoint (bf16 raw words), the cos_sin table from the
    fixture; each field's sha256 compared with the committed weights leaf.  Returns the transcript and the authentication record."""
    lay = program.layout
    lp, W = lay.LP, lay.W
    pstart, pend = _coll_range(circuit, program.input_coll(0))
    assert (pstart, pend) == (0, lp), (pstart, pend, lp)
    if len(prompt) != lp:
        raise ValueError(f"prompt has {len(prompt)} tokens, the program's LP is {lp}")
    pw = array.array("I", [int(x) for x in prompt])
    widx = ({e["canonical_id"]: e for e in json.loads((rec_dir / "adapter" / "selective" / "weights" / "index.json").read_text())["entries"]}
            if rec_dir is not None else {})  # declared-inputs mode: no committed leaves; every field is `sha256_committed: null`
    tensors = load_safetensors([checkpoint])
    Wc = program.input_coll(1)
    ws, we = _coll_range(circuit, Wc)
    assert we - ws == W, (ws, we, W)
    words = np.zeros(W, dtype=np.uint16)
    filled = np.zeros(W, dtype=bool)
    auth: list[dict] = []

    def one(coll, cid: str, arr: np.ndarray, source: str) -> None:
        s, e = _coll_range(circuit, coll)
        flat = np.ascontiguousarray(arr).reshape(-1)
        if flat.dtype != np.uint16:
            raise TypeError(f"{cid}: expected raw bf16 uint16 words, got {flat.dtype}")
        if e - s != flat.size:
            raise ValueError(f"{cid}: {e - s} Input gates vs {flat.size} words")
        sha = hashlib.sha256(flat.tobytes()).hexdigest()
        ent = widx.get(cid)
        words[s - ws:e - ws] = flat
        filled[s - ws:e - ws] = True
        auth.append({"leaf_id": cid, "input_gates": [s, e], "words": int(flat.size), "source": source, "sha256_rebuilt": sha,
                     "sha256_committed": ent["sha256"] if ent else None, "leaf_committed": ent["leaf"] if ent else None,
                     "sha256_matches_committed": bool(ent and sha == ent["sha256"])})

    one(Wc["embed"], _WEIGHT_FIELDS["embed"][0], tensors["model.embed_tokens.weight"], "checkpoint")
    for L in range(C["NL"]):
        w = Wc["layers"][L]
        for fname, (cid_t, srcs) in _LAYER_FIELDS.items():
            if fname == "qkv_b" and not C.get("BIAS", True):
                continue
            parts = [tensors[s.format(L=L)] for s in srcs]
            arr = parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0)
            one(w[fname], cid_t.format(L=L), arr, "checkpoint")
    one(Wc["norm"], _WEIGHT_FIELDS["norm"][0], tensors["model.norm.weight"], "checkpoint")
    cs_id = _WEIGHT_FIELDS["cos_sin"][0]
    cs_ent = widx.get(cs_id)
    vbin = (rec_dir / "adapter" / "selective" / "weights" / "values.bin") if rec_dir is not None else None
    if cos_sin_npy is None and cs_ent is not None and vbin is not None and vbin.exists() and cs_ent.get("offset") is not None:
        # [E1 night 2026-09-12] no fixture named: the served record's own committed table words (weights/values.bin at the index's offset;
        # the sha256 check below authenticates them against the committed leaf like every other weight) -- the GPU-built table of THIS model
        cs = np.frombuffer(vbin.read_bytes()[int(cs_ent["offset"]):int(cs_ent["offset"]) + int(cs_ent["nbytes"])], dtype=np.uint16).reshape(cs_ent["shape"])
        cs_src = f"the record's committed weights values ({vbin}, offset {cs_ent['offset']}; vllm._compute_cos_sin_cache on the serving GPU)"
    else:
        if cos_sin_npy is None:
            cos_sin_npy = DEFAULT_COS_SIN
        cs = np.load(cos_sin_npy)
        cs_src = f"GPU-built fixture {cos_sin_npy} (vllm._compute_cos_sin_cache on CUDA; the CPU rebuild differs)"
    if cs.dtype != np.uint16:
        cs = cs.view(np.uint16)
    one(Wc["cos_sin"], cs_id, cs, cs_src)
    if not filled.all():
        raise ValueError(f"{int((~filled).sum())} weight Input gates unbound")
    wa = array.array("H")
    wa.frombytes(words.tobytes())
    del words, filled
    rec = {"prompt": {"input_gates": [pstart, pend], "token_ids": prompt,
                      "authenticated_by": f"boundary/{request} tree (prompt_token/t leaves)" if rec_dir is not None else "DECLARED (--prompt-ids; no served record yet)"},
           "weights": auth, "weights_all_match": all(a["sha256_matches_committed"] for a in auth),
           "weights_mismatching": [a["leaf_id"] for a in auth if not a["sha256_matches_committed"]]}
    return StreamTranscript(lp, pw, ws, wa), rec


# ---------------------------------------------------------------------------------------------------------------------------------
# inventory: required gates = VU output gates of the RU not at an operator boundary
# ---------------------------------------------------------------------------------------------------------------------------------


def call_at(circuit, lo: int, hi: int):
    """The `FunctionCall` whose interval is exactly [lo, hi), descending the canonical layout by `node_at`, or None (E3's
    `verifier_v3.locator.call_at` rule, used verbatim so both inventories spell one output-gate set)."""
    if not (0 <= lo < hi <= circuit.size):
        return None
    scope = circuit.root_scope
    while True:
        body = scope.spec.body
        k = body.node_at(lo - scope.offset)
        node = body.nodes[k]
        node_off = scope.offset + body.offsets[k]
        m = 0 if node.form in ("primitive", "call") else (lo - node_off) // node.fn.gates
        c_lo = node_off + m * node.fn.gates
        if (c_lo, c_lo + node.fn.gates) == (lo, hi):
            return FunctionCall(circuit, scope, k, m)
        if isinstance(node.fn, PrimitiveDefinition) or c_lo + node.fn.gates < hi or lo < c_lo:
            return None
        scope = Scope(node.fn, c_lo, scope, k, m)


def vu_output_offsets(circuit, lo: int, usz: int) -> tuple[int, ...]:
    """Output gates of the VU [lo, lo+usz) relative to lo: its one gate (unit size 1) or the distinct declared returns of its call inside
    the interval, ascending (PROTOCOL step 3 "each output gate of V"; == E3's `locator.output_gates_of`).  A scan step such as
    `ArgmaxStep_v1` returns 3 leaves (running max / index / count) -> 3 output gates per VU."""
    if usz == 1:
        return (0,)
    c = call_at(circuit, lo, lo + usz)
    if c is None:
        # [E1 night 2026-09-12] E6's night populations regroup the sampler as `TokenSelect_v1|whole-scan` (one VU = the whole ArgmaxStep scan,
        # 245,755 gates at V=49,152): no call spans it.  E3's oracle defines the unit's outputs (the ENCLOSING call's returns inside the
        # interval = the scan's final token word; the per-step carries are interior) -- use it, do not reimplement.
        from e2e.verifier_v3 import locator as vloc

        class _P:  # locator.output_gates_of reads `.circuit` only
            pass
        p = _P(); p.circuit = circuit
        try:
            return tuple(int(g) - lo for g in vloc.output_gates_of(p, lo, lo + usz))
        except vloc.PositionRefusal as ex:
            raise ValueError(f"VU [{lo},{lo + usz}) is no call of the pinned program ({ex})") from ex
    return tuple(sorted({int(g) - lo for g in c.returns if lo <= g < lo + usz}))


def build_inventory(rec_dir: Path, request: str, program, runs: list[tuple[int, int]], out_dir: Path, files: list[dict]) -> dict:
    """From the pinned population rows (`population.v5.json`, per-RU basis): the output gates of VU j of a row are
    lo + j*stride + `vu_output_offsets` (one specialization per row -> one layout, derived once per row from the program);
    boundary iff the gate lies in a committed run (the located committed leaves of the record's step trees).  E3's
    `RecordPartition.locate` / `GateResolver.required_gates` derive the same set (cross-checked: `out/eod/e1/e3_crosscheck.py`)."""
    pop = json.loads((rec_dir / "population.v5.json").read_text())
    kinds = {k["id"]: k for k in pop["kinds"]["entries"]}
    lo_arr = np.array([a for a, _ in runs], dtype=np.int64)
    hi_arr = np.array([b for _, b in runs], dtype=np.int64)

    def committed(gs: np.ndarray) -> np.ndarray:
        i = np.searchsorted(lo_arr, gs, side="right") - 1
        ok = i >= 0
        out = np.zeros(gs.shape, dtype=bool)
        out[ok] = gs[ok] < hi_arr[i[ok]]
        return out

    lay = program.layout
    circuit = program.circuit
    per_kind: dict[str, dict] = {}
    req_rows: list[list[int]] = []
    required: list[np.ndarray] = []
    tot = {"vus": 0, "boundary": 0, "interior": 0, "interior_bytes": 0}
    rel_cache: dict[tuple[int, int, int], tuple[tuple[int, ...], tuple[int, ...]]] = {}
    for first_vu, count, ru, kid, lo, stride, usz in pop["rows"]:
        k = kinds[kid]
        name = f"{k['fn']}|{k['form']}"
        key = (int(kid), int(stride), int(usz))
        cached = rel_cache.get(key)
        if cached is None:
            rel = vu_output_offsets(circuit, int(lo), int(usz))
            widths = tuple(int(circuit.gate(int(lo) + r).prim.ret.w) for r in rel)
            cached = rel_cache[key] = (rel, widths)
        rel, widths = cached
        members = lo + np.arange(count, dtype=np.int64) * stride
        outs = (members[:, None] + np.asarray(rel, dtype=np.int64)[None, :]).reshape(-1)  # ascending: rel ascending, stride >= usz
        c = committed(outs)
        d = per_kind.setdefault(name, {"vus": 0, "boundary": 0, "interior": 0, "out_bits": k["out_bits"], "unit_size": usz,
                                       "out_gates_per_vu": len(rel), "out_widths": list(widths)})
        nb, ni = int(c.sum()), int((~c).sum())
        d["vus"] += count
        d["boundary"] += nb
        d["interior"] += ni
        tot["vus"] += count
        tot["boundary"] += nb
        tot["interior"] += ni
        wb = np.tile(np.asarray([(w + 7) // 8 for w in widths], dtype=np.int64), count)  # bytes per output gate at its declared width
        tot["interior_bytes"] += int(wb[~c].sum())
        if ni:
            required.append(outs[~c])
            req_rows.append([int(first_vu), int(count), int(kid), int(lo), int(stride), int(usz), ni])
    req = np.sort(np.concatenate(required)) if required else np.zeros(0, dtype=np.int64)
    if req.size and (np.diff(req) == 0).any():
        raise AssertionError("duplicate required gate")
    (out_dir / "required_gates.bin").write_bytes(req.astype("<u4").tobytes())
    # per root call: how many required gates fall in each root call's interval
    offs = np.array([lay.offset(k) for k in range(lay.n_nodes + 1)], dtype=np.int64)
    per_root = np.diff(np.searchsorted(req, offs)) if req.size else np.zeros(lay.n_nodes, dtype=np.int64)
    for f in files:
        f["required_gates"] = int(per_root[f["k"]])
    inv = {"schema": "veritor/e1/inventory/v1", "label": "MEASURED (counts from the pinned population rows and the committed index of attempt 6)",
           "request_id": request, "ru": 1, "program_digest": pop["program_digest"], "partition_digest": pop["partition_digest"],
           "population_content_sha256": pop.get("content_sha256"), "population_vu_count": pop["vu_count"],
           "boundary_derivation": "committed operator-boundary runs = gates of every captured canonical id of the RU's step trees "
                                  "(adapter/selective/ru/<req>/<step>/index.json) located in the pinned program by verity_vllm.correspondence.core.CircuitMap.locate; "
                                  "cross-checked gate-for-gate against E3's verifier_v3.locator.GateResolver.required_gates (eod/e3 182f6bf4) -- "
                                  "out/eod/e1/e3_crosscheck.py",
           "required_definition": "every output gate of every VU of the RU (its one gate, or the declared returns of its call: 3 per ArgmaxStep_v1 scan step) "
                                  "that is not in a committed run",
           "totals": {**tot, "required_gates": int(req.size), "committed_runs": len(runs)},
           "by_kind": dict(sorted(per_kind.items(), key=lambda kv: -kv[1]["interior"])),
           "required_rows": {"fields": ["first_vu", "count", "kind_id", "lo", "stride", "unit_size", "interior_count"], "rows": req_rows,
                             "kinds": {str(i): f"{k['fn']}|{k['form']}" for i, k in kinds.items()}},
           "required_gates_bin": {"path": "required_gates.bin", "dtype": "<u4", "count": int(req.size), "sorted": True},
           "value_record": {"schema": SCHEMA_VALUES, "row": '{"g": <gate index>, "k": "<primitive id>|<node form>", "w": <width bits>, "v": "<hex, little-endian bytes of the word>"}',
                            "file": "interior/<req>/<step>/<call id>.jsonl, one per root call, rows sorted by g; call id = t=<t>_ph=<phase>_l=<layer|embed|final>_call=<role>",
                            "rows_are": "EVERY computed gate of the root call (boundary and interior alike); required (interior VU output) gates are listed in required_gates.bin -- "
                                        "E2 must never emit a gate that has a boundary position as interior (interface 3)"},
           "files": files}
    (out_dir / "inventory.json").write_text(json.dumps(inv, indent=1))
    return inv


# ---------------------------------------------------------------------------------------------------------------------------------
# the producer
# ---------------------------------------------------------------------------------------------------------------------------------


def call_file_id(lay, k: int) -> tuple[str, int, str]:
    """(file id, position t, phase) of root node k."""
    r = lay.role(k)
    kind = r[0]
    t = r[1]
    ph = "prefill" if t < lay.LP else "decode"
    if kind in ("embed", "final"):
        l = kind
    else:
        l = str(r[2])
    return f"t={t}_ph={ph}_l={l}_call={kind}", t, ph


def step_of(t: int, lp: int) -> int:
    return 0 if t < lp else t - lp + 1


class NoBoundary:
    """Stand-in for :class:`BoundaryIndex` in declared-inputs mode (no served record yet): nothing to compare; the offline
    `python -m e2e.interior.compare_boundaries --record <rec> --request <id> --values <out>` does the comparison once the record exists."""

    def __init__(self, request: str) -> None:
        self.request = request
        self.by_root: dict[int, list] = {}
        self.runs: list = []
        self.n_entries = self.n_words = 0
        self.not_in_program: list = []
        self.locate_errors: list = []
        self.build_s = 0.0
        self.rows_compared = self.rows_agree = 0
        self.mismatches: list = []

    def compare_root(self, k: int, transcript, *, on_mismatch=None) -> list[dict]:
        return []

    def summary(self, *, label: str = "NOT COMPARED") -> dict[str, Any]:
        return {"schema": "veritor/e1/boundary-compare/v1", "label": label, "request_id": self.request, "committed_leaves": 0, "committed_words": 0,
                "located_leaves": 0, "rows_compared": 0, "rows_agree": 0, "rows_mismatch": 0, "rows_not_evaluated": 0, "rows_pending": 0, "per_step": {},
                "verdict": "NOT COMPARED: declared-inputs mode (no served record); run e2e.interior.compare_boundaries --values <this dir> against the record",
                "first_mismatch": None, "mismatches": [], "not_evaluated": []}


class Producer:
    def __init__(self, a: argparse.Namespace) -> None:
        self.a = a
        self.out = Path(a.out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.t_start = time.perf_counter()
        self.progress = open(self.out / "progress.jsonl", "a")
        self.log("start", argv=sys.argv, record=str(a.record), request=a.request, pid=os.getpid())

    def log(self, event: str, **kw: Any) -> None:
        rec = {"utc": utc(), "event": event, "elapsed_s": round(time.perf_counter() - self.t_start, 3), "max_rss_gb": round(rss_gb(), 3), **kw}
        self.progress.write(json.dumps(rec, default=str) + "\n")
        self.progress.flush()
        print(json.dumps(rec, default=str), flush=True)

    def run(self) -> int:
        a = self.a
        from e2e.verifier_v3 import locator as vloc
        declared = a.declared_case is not None  # [E1 night 2026-09-12] no served record yet: program from the registry case, prompt from --prompt-ids
        # [E2 CURVE 2026-09-13] declared program + a SERVED record without protocol_instance.json (E10's numerical transition records): the
        # program is the registry's (`--declared-case --lp --steps`), the prompt is the record's committed boundary, and the committed RU leaves
        # are compared live exactly as in record mode (`BoundaryIndex`); no population -> no inventory (`--only-boundary` writes the boundary rows)
        served = (not declared) or a.prompt_ids is None
        if declared and served:
            self.log("declared_program_served_record", note="program from --declared-case (no protocol_instance.json in the record); prompt + boundary leaves from the record; live compare")
        t0 = time.perf_counter()
        if declared:
            from verity.ir import bind
            from verity_vllm.ir.lazy import LazyProgram, lazy_program_digest
            from verity_vllm.registry import b1
            if not hasattr(b1, f"{a.declared_case}_CONFIG"):
                self.log("program_problems", problems=[f"registry has no {a.declared_case}_CONFIG"])
                return 2
            C = dict(getattr(b1, f"{a.declared_case}_CONFIG"))
            geom = json.loads(a.attention) if getattr(a, "attention", None) else None  # [E7 D57] declared geometry -> Serve_v4; none -> Serve_v1
            fam = "Serve_v4" if geom is not None else b1.DEFAULT_SERVE_FAMILY
            program = LazyProgram(b1.serve_program(C, a.lp, a.steps, family=fam, attention=geom))
            inst = {"digest": lazy_program_digest(program), "identity": {"family": fam, "case": a.declared_case, "LP": a.lp, "STEPS": a.steps, "function": program.fn.id,
                                                                          **({"attention": geom} if geom is not None else {})}}
            (self.out / "declared_program.json").write_text(json.dumps({**inst, "gates": program.gates, "input_gates": program.input_gates, "C": C,
                                                                         "label": "DECLARED (no protocol_instance.json; the pin must reproduce this digest)"}, indent=1))
        else:
            inst = json.loads((a.record / "protocol_instance.json").read_text())["program"]
            if getattr(a, "attention", None):
                # [E7 D57] DIAGNOSTIC: re-interpret a record pinned under one geometry with another (D54 acceptance on E1's 128×64 dry record,
                # pinned Serve_v1): the identity is rewritten to Serve_v4 + the declared geometry, the pinned digest/gates no longer apply
                geom = json.loads(a.attention)
                inst = {"identity": {**inst["identity"], "family": "Serve_v4", "attention": geom}, "digest": None, "gates": None}
                inst["identity"].pop("function", None)
                self.log("program_geometry_override", identity=inst["identity"], label="DIAGNOSTIC (the record's pinned program is NOT this one)")
            program, C, problems = vloc.build_pinned_program(inst["identity"], expected_digest=inst.get("digest"), expected_gates=inst.get("gates"), lazy=True)
            if problems:
                self.log("program_problems", problems=problems)
                return 2
        circuit = program.circuit
        lay = program.layout
        self.log("program", function=program.fn.id, digest=inst.get("digest"), gates=program.gates, input_gates=program.input_gates,
                 computed_gates=program.gates - program.input_gates, root_nodes=lay.n_nodes, build_s=round(time.perf_counter() - t0, 3), declared=declared)

        # committed inputs (or, declared mode, the declared prompt)
        if not served:
            ids = [int(x) for x in a.prompt_ids.split(",")]
            bnd = {"prompt_token_ids": ids, "profile_id": a.profile_id, "root_matches_committed": None, "leaf_count": None, "validate_problems": [],
                   "label": "DECLARED: prompt token ids from --prompt-ids; no boundary tree (no served record yet)"}
            self.log("declared_inputs", prompt=ids, case=a.declared_case, lp=a.lp, steps=a.steps)
        else:
            bnd = authenticate_boundary(a.record, a.request)
            self.log("boundary_tree", root_matches_committed=bnd["root_matches_committed"], leaf_count=bnd["leaf_count"], prompt=bnd["prompt_token_ids"],
                     validate_problems=bnd["validate_problems"])
            if not bnd["root_matches_committed"]:
                self.log("abort", reason="rebuilt boundary tree root != committed root")
                return 3
        ckpt = Path(a.checkpoint) if a.checkpoint else Path(sorted(glob.glob(HF_GLOB))[-1])
        t0 = time.perf_counter()
        tr, auth = bind_inputs(program, circuit, a.record if served else None, a.request, ckpt, Path(a.cos_sin_npy) if a.cos_sin_npy else None, C, bnd["prompt_token_ids"])
        self.log("inputs_bound", weights_leaves=len(auth["weights"]), weights_all_match=auth["weights_all_match"], weights_mismatching=auth["weights_mismatching"],
                 checkpoint=str(ckpt), seconds=round(time.perf_counter() - t0, 3))
        (self.out / "inputs_authenticated.json").write_text(json.dumps({"boundary": bnd, "inputs": auth}, indent=1))
        if not auth["weights_all_match"] and not a.allow_weight_mismatch and served:
            self.log("abort", reason="a weights field does not match its committed leaf sha256", mismatching=auth["weights_mismatching"])
            return 3
        if not served:
            self.log("inputs_unauthenticated", note="declared-inputs mode: the weights' sha256s are recorded, not compared (no committed leaves); "
                     "compare_boundaries --values <out> against the served record authenticates the boundary rows afterwards")

        # committed boundary leaves -> gates
        bi = BoundaryIndex(program, a.record, a.request) if served else NoBoundary(a.request)
        self.log("boundary_index", committed_leaves=bi.n_entries, located=sum(len(v) for v in bi.by_root.values()), words=bi.n_words,
                 not_in_program=len(bi.not_in_program), locate_errors=len(bi.locate_errors), seconds=round(bi.build_s, 3))
        if a.only_boundary:
            # [E2 CURVE] rows = exactly the committed boundary gates (the leaves' gate runs): small, and enough for the OFFLINE re-check
            # (`python -m e2e.interior.compare_boundaries --values <out>`) to repeat the comparison from the written words
            if not served:
                self.log("abort", reason="--only-boundary needs a served record (the committed RU index names the boundary gates)")
                return 2
            bg = np.concatenate([np.arange(lo, hi, dtype=np.int64) for lo, hi in bi.runs]) if bi.runs else np.zeros(0, dtype=np.int64)
            bg = np.unique(bg)
            (self.out / "boundary_gates.bin").write_bytes(bg.astype("<u4").tobytes())
            a.only_required_from = bg
            self.log("only_boundary", gates=int(bg.size), runs=len(bi.runs))

        # inventory first (population rows + committed runs; independent of the evaluation), so it exists even if the run is killed
        files: list[dict] = []
        if a.reinventory:  # rebuild inventory.json / required_gates.bin of a finished or running output dir from its files_so_far.json
            files = json.loads((self.out / "files_so_far.json").read_text())
            inv = build_inventory(a.record, a.request, program, bi.runs, self.out, files)
            self.log("reinventory", files=len(files), **{k: v for k, v in inv["totals"].items()})
            return 0
        only_required = a.only_required_from
        # [E7 D57] a geometry override re-interprets the record under a program the population was NOT pinned to: its rows (gate offsets
        # per kind) do not index this program -> no inventory, as in declared-inputs mode; the boundary compare (committed leaves located
        # by canonical id in THIS program) is the point of such a run.  Write nothing (--only-required-from <empty>) or everything.
        no_inventory = declared or (not declared and getattr(a, "attention", None) is not None)
        if no_inventory:
            if a.only_required:
                self.log("abort", reason="--only-required needs the inventory (population + committed index): not available in declared-inputs mode / under a geometry override; write every gate and --reinventory later")
                return 2
            self.log("inventory", skipped="declared-inputs mode or geometry override (the pinned population does not index this program); rebuild with --reinventory --record <rec> when a record exists")
        else:
            inv = build_inventory(a.record, a.request, program, bi.runs, self.out, files)
            self.log("inventory", **{k: v for k, v in inv["totals"].items()})
            if a.only_required:
                only_required = np.frombuffer((self.out / "required_gates.bin").read_bytes(), dtype="<u4").astype(np.int64)

        # root calls in gate order; which to run
        roots = sorted(circuit.root_calls(), key=lambda c: c.lo)
        plan: list[tuple[int, FunctionCall]] = []
        for c in roots:
            k = lay.node_at(c.lo)
            r = lay.role(k)
            if a.prefill_layers is not None:
                if r[0] in ("final",) or r[1] >= lay.LP or (r[0] in ("pre", "attn", "post") and r[2] >= a.prefill_layers):
                    continue
            plan.append((k, c))
            if a.max_root_calls and len(plan) >= a.max_root_calls:
                break
        planned_gates = sum(c.hi - c.lo for _, c in plan)
        self.log("plan", root_calls=len(plan), gates=planned_gates, of_root_calls=len(roots), of_computed_gates=program.gates - program.input_gates,
                 parallel=a.parallel, parallel_min_gates=a.parallel_min_gates)

        # walk self-check: sampled on the first planned calls, then the named calls (`--walk-check-named`)
        rng = np.random.default_rng(a.seed)
        fast = not a.no_fast_args
        checks = [check_walk(circuit, c, a.walk_check_sample, rng, fast) for _, c in plan[: a.walk_check_calls]]
        if a.walk_check_named:
            by_name = {lay.name(lay.node_at(c.lo)): c for c in roots}
            for item in a.walk_check_named.split(","):
                nm, _, cnt = item.strip().partition(":")
                c = by_name[nm]
                t1 = time.perf_counter()
                r = check_walk(circuit, c, int(cnt) if cnt else c.hi - c.lo, rng, fast)
                r["name"], r["seconds"] = nm, round(time.perf_counter() - t1, 1)
                checks.append(r)
                self.log("walk_check_named", **r)
        self.log("walk_check", fast_args=fast, calls=checks)

        req_dir = self.out / "interior" / a.request
        done_gates = 0
        t_eval0 = time.perf_counter()
        rate_rows: list[dict] = []
        first_mismatch: dict | None = None

        def on_mismatch(rec: dict) -> None:
            nonlocal first_mismatch
            if first_mismatch is None:
                first_mismatch = rec
            self.log("BOUNDARY_MISMATCH", canonical_id=rec["canonical_id"], gates=[rec["gate_lo"], rec["gate_hi"]],
                     leaf_interpreter=rec["leaf_interpreter"][:16], leaf_committed=rec["leaf_committed"][:16], first_differing_word=rec.get("first_differing_word"))

        stop = False

        def finish(seq: int, k: int, call: FunctionCall, runs: list, pstats: dict, t_eval: float) -> None:
            """Everything after a call's values are in `tr`: rows, boundary compare, retention, logs, checkpoints."""
            nonlocal done_gates, stop
            fid, t, ph = call_file_id(lay, k)
            step = step_of(t, lay.LP)
            d = req_dir / f"{step:06d}"
            d.mkdir(parents=True, exist_ok=True)
            path = d / (fid + (".jsonl.gz" if a.gzip else ".jsonl"))
            # stream rows (every computed gate of the call), then compare boundaries, then retain only the returns
            t2 = time.perf_counter()
            n_rows = self._write_rows(path, runs, tr, only_required, call.lo, call.hi)
            t_write = time.perf_counter() - t2
            t3 = time.perf_counter()
            cmp = bi.compare_root(k, tr, on_mismatch=on_mismatch)
            t_cmp = time.perf_counter() - t3
            rets = set(call.returns)
            lo, hi = call.lo, call.hi
            if lo in tr.blk:
                tr.drop_block(lo, rets)
            else:
                for g in range(lo, hi):
                    if g not in rets:
                        del tr[g]
            n = hi - lo
            done_gates += n
            files.append({"seq": seq, "k": k, "path": str(path.relative_to(self.out)), "call_id": fid, "canonical_id": f"p={bnd['profile_id']}/r={a.request}/t={t}/ph={ph}/l={fid.split('_l=')[1].split('_')[0]}/call={call.id}",
                          "name": lay.name(k), "function": call.id, "lo": lo, "hi": hi, "gates": n, "rows": n_rows, "step": step, "t": t,
                          "boundary_leaves_compared": len(cmp), "boundary_leaves_agree": sum(1 for r in cmp if r["agrees"]),
                          "eval_s": round(t_eval, 3), "us_per_gate": round(1e6 * t_eval / n, 1)})
            elapsed = time.perf_counter() - t_eval0
            rate_rows.append({"seq": seq, "name": lay.name(k), "gates": n, "eval_s": round(t_eval, 3), "write_s": round(t_write, 3), "compare_s": round(t_cmp, 3),
                              "us_per_gate": round(1e6 * t_eval / n, 1), "cum_gates": done_gates, "cum_s": round(elapsed, 1), "retained": len(tr), "max_rss_gb": round(rss_gb(), 3),
                              "forks": pstats["forks"], "vec_nodes": pstats["vec_nodes"], "vec_gates": pstats["vec_gates"],
                              "scalar_gates": pstats["scalar_gates"], "flat": pstats.get("flat", 0), "not_vectorizable": pstats["not_vectorizable"][:8]})
            self.log("root_call", **rate_rows[-1], boundary_compared=bi.rows_compared, boundary_agree=bi.rows_agree, boundary_mismatch=len(bi.mismatches))
            if len(rate_rows) % a.checkpoint_every == 0 or len(rate_rows) == len(plan):
                self._write_summaries(bi, rate_rows, done_gates, planned_gates, program, files, t_eval0)
            if first_mismatch is not None and a.stop_on_mismatch:
                self.log("stop", reason="first boundary mismatch (--stop-on-mismatch)")
                stop = True

        if a.pool > 1:
            rc = self._run_pool(plan, circuit, tr, fast, finish, lambda: stop)
            if rc:
                return rc
        else:
            for seq, (k, call) in enumerate(plan):
                walk = CallWalk(call, fast=fast, vectorize=not a.no_vectorize, flat=not a.no_flat, flat_check_all=a.flat_check_all)
                t1 = time.perf_counter()
                try:
                    pstats = walk.evaluate(circuit, tr, nproc=a.parallel, min_gates=a.parallel_min_gates)
                except MissingValue as e:
                    self.log("abort", reason=f"MissingValue in {lay.name(k)}: {e}")
                    return 4
                except RuntimeError as e:
                    self.log("abort", reason=f"parallel evaluation failed in {lay.name(k)}: {e}")
                    return 4
                finish(seq, k, call, walk.runs, pstats, time.perf_counter() - t1)
                if stop:
                    break

        files.sort(key=lambda f: f["seq"])  # the pool finishes calls out of plan order; the inventory lists them in plan order
        self._write_summaries(bi, rate_rows, done_gates, planned_gates, program, files, t_eval0)
        if not no_inventory:
            inv = build_inventory(a.record, a.request, program, bi.runs, self.out, files)  # again, now with the files written
        s = bi.summary()
        self.log("done", verdict=s["verdict"], rows_compared=s["rows_compared"], rows_agree=s["rows_agree"], rows_mismatch=s["rows_mismatch"],
                 rows_pending=s["rows_pending"], gates=done_gates, wall_s=round(time.perf_counter() - self.t_start, 1), max_rss_gb=round(rss_gb(), 3))
        return 0 if not s["rows_mismatch"] else 1

    def _run_pool(self, plan, circuit, tr, fast: bool, finish, stopped) -> int:
        """[E2 SCALE follow-up] Root calls over a POOL of `--pool N` forked workers (forked once, after the inputs are bound; not one
        fork per batch node as `--parallel` does): a root call is dispatched as soon as the words its parameter leaves need are in
        the parent's transcript (its dependencies are complete), the first ready call in plan order within a look-ahead window;
        the worker evaluates it exactly as the sequential path does (same `CallWalk`, same templates, same values) on its own copy
        of the transcript, and sends back the call's words and runs; the parent writes rows / compares / retains in completion
        order (one file per call, so the rows are the same bytes).  The prefill's calls of one layer are independent across
        positions (pre, attn given the layer's pre calls, post): that is where the pool scales; decode is one chain."""
        import pickle
        import select
        a = self.a
        n_workers = a.pool
        by_seq = {seq: (k, call) for seq, (k, call) in enumerate(plan)}

        def act_leaves(call) -> np.ndarray:
            pl = _vw.param_leaves(call)
            keep = ~(((pl >= tr.ws) & (pl < tr.we)) | ((pl >= 0) & (pl < tr.lp)))
            return np.unique(pl[keep])

        # dependencies as root calls of the plan (a set of seqs, ready when all are finished), not as gates: the per-gate
        # `g in tr` scan over the look-ahead window cost ≈ 100 ms per dispatch on the decode chain (20 k K/V-cache leaves per call)
        plan_lo = np.array([call.lo for _, call in plan], dtype=np.int64)
        plan_hi = np.array([call.hi for _, call in plan], dtype=np.int64)
        order = np.argsort(plan_lo, kind="stable")
        lo_sorted = plan_lo[order]
        done_seqs: set[int] = set()

        def deps_of(act: np.ndarray) -> tuple[set[int], np.ndarray]:
            """(seqs of the plan's calls that produce `act`, the leaves no call of the plan produces -- checked in `tr` directly)."""
            if len(act) == 0:
                return set(), act
            i = np.searchsorted(lo_sorted, act, side="right") - 1
            seqs = order[np.maximum(i, 0)]
            inside = (i >= 0) & (act < plan_hi[seqs])
            return set(seqs[inside].tolist()), act[~inside]

        def ready(seq: int) -> bool:
            ent = acts.get(seq)
            if ent is None:
                act = act_leaves(by_seq[seq][1])
                ent = acts[seq] = (act, *deps_of(act))
            act, deps, extra = ent
            return deps <= done_seqs and all(g in tr for g in extra.tolist())

        workers = []
        for _ in range(n_workers):
            p_r, p_w = os.pipe()  # parent -> child
            c_r, c_w = os.pipe()  # child -> parent
            pid = os.fork()
            if pid == 0:
                os.close(p_w)
                os.close(c_r)
                code = 0
                try:
                    rf, wf = os.fdopen(p_r, "rb"), os.fdopen(c_w, "wb")
                    while True:
                        hdr = rf.read(8)
                        if len(hdr) < 8:
                            break
                        job = pickle.loads(rf.read(int.from_bytes(hdr, "little")))
                        if job is None:
                            break
                        seq, act_g, act_v = job
                        k, call = by_seq[seq]
                        tr.update(zip(act_g.tolist(), act_v.tolist(), strict=True))
                        walk = CallWalk(call, fast=fast, vectorize=not a.no_vectorize, flat=not a.no_flat, flat_check_all=a.flat_check_all)
                        t1 = time.perf_counter()
                        try:
                            pstats = walk.evaluate(circuit, tr, nproc=1, min_gates=a.parallel_min_gates)
                            if call.lo in tr.blk:
                                vals = tr.blk[call.lo][1]
                            else:
                                vals = np.fromiter((tr[g] for g in range(call.lo, call.hi)), dtype=np.uint64, count=call.hi - call.lo)
                            msg = (seq, vals, walk.runs, pstats, time.perf_counter() - t1)
                        except BaseException as e:  # noqa: BLE001
                            msg = ("ERROR", seq, repr(e))
                        if call.lo in tr.blk:
                            tr.drop_block(call.lo, ())
                        for g in range(call.lo, call.hi):
                            tr.pop(g, None)
                        for g in act_g.tolist():
                            tr.pop(g, None)
                        payload = pickle.dumps(msg, protocol=pickle.HIGHEST_PROTOCOL)
                        wf.write(len(payload).to_bytes(8, "little"))
                        wf.write(payload)
                        wf.flush()
                except BaseException:  # noqa: BLE001
                    code = 1
                os._exit(code)
            os.close(p_r)
            os.close(c_w)
            workers.append({"pid": pid, "w": os.fdopen(p_w, "wb"), "r": os.fdopen(c_r, "rb"), "fd": c_r, "busy": None})
        self.log("pool", workers=n_workers, pids=[w["pid"] for w in workers])

        def send(w, seq: int, act: np.ndarray) -> None:
            vals = tr.gather(act) if len(act) else np.zeros(0, dtype=np.uint64)
            payload = pickle.dumps((seq, act, vals), protocol=pickle.HIGHEST_PROTOCOL)
            w["w"].write(len(payload).to_bytes(8, "little"))
            w["w"].write(payload)
            w["w"].flush()
            w["busy"] = seq

        def recv(w):
            hdr = w["r"].read(8)
            if len(hdr) < 8:
                raise RuntimeError(f"pool worker {w['pid']} died")
            return pickle.loads(w["r"].read(int.from_bytes(hdr, "little")))

        pending = list(range(len(plan)))
        acts: dict[int, tuple[np.ndarray, set[int], np.ndarray]] = {}
        lookahead = a.pool_lookahead
        rc = 0
        try:
            while pending or any(w["busy"] is not None for w in workers):
                idle = [w for w in workers if w["busy"] is None]
                if idle:
                    ready_seqs = []
                    for seq in pending[:lookahead]:
                        if ready(seq):
                            ready_seqs.append(seq)
                            if len(ready_seqs) == len(idle):
                                break
                    if len(ready_seqs) == 1 and len(idle) == n_workers:
                        # a dependency chain (decode: embed -> pre -> attn -> post -> ... one call ready at a time, every worker
                        # idle): evaluate in the parent -- the sequential path, no round trip (the pool's decode was 1.5x slower)
                        seq = ready_seqs[0]
                        pending.remove(seq)
                        acts.pop(seq, None)
                        k, call = by_seq[seq]
                        walk = CallWalk(call, fast=fast, vectorize=not a.no_vectorize, flat=not a.no_flat, flat_check_all=a.flat_check_all)
                        t1 = time.perf_counter()
                        try:
                            pstats = walk.evaluate(circuit, tr, nproc=1, min_gates=a.parallel_min_gates)
                        except (MissingValue, RuntimeError) as e:
                            self.log("abort", reason=f"evaluation failed in {lay_name(plan, seq)}: {e!r}")
                            return 4
                        finish(seq, k, call, walk.runs, pstats, time.perf_counter() - t1)
                        done_seqs.add(seq)
                        if stopped():
                            pending.clear()
                        continue
                    for w, seq in zip(idle, ready_seqs, strict=False):
                        pending.remove(seq)
                        send(w, seq, acts.pop(seq)[0])
                busy = [w for w in workers if w["busy"] is not None]
                if not busy:
                    if pending:
                        self.log("abort", reason=f"pool deadlock: no ready call among the next {lookahead} of {len(pending)} pending")
                        return 4
                    break
                readable, _, _ = select.select([w["fd"] for w in busy], [], [])
                for w in busy:
                    if w["fd"] not in readable:
                        continue
                    msg = recv(w)
                    w["busy"] = None
                    if msg[0] == "ERROR":
                        self.log("abort", reason=f"pool worker failed on call {msg[1]} ({lay_name(plan, msg[1])}): {msg[2]}")
                        return 4
                    seq, vals, runs, pstats, t_eval = msg
                    k, call = by_seq[seq]
                    if len(vals) != call.hi - call.lo:
                        raise RuntimeError(f"pool worker returned {len(vals)} words for call {seq} of {call.hi - call.lo} gates")
                    tr.add_block(call.lo, vals)
                    finish(seq, k, call, runs, pstats, t_eval)
                    done_seqs.add(seq)
                    if stopped():
                        pending.clear()
        finally:
            for w in workers:
                try:
                    w["w"].write((0).to_bytes(8, "little"))
                    w["w"].flush()
                    w["w"].close()
                except OSError:
                    pass
                try:
                    w["r"].close()  # a worker still writing a result (parent left on an error) gets EPIPE instead of blocking forever
                except OSError:
                    pass
            for w in workers:
                os.waitpid(w["pid"], 0)
        return rc

    def _write_rows(self, path: Path, runs: list, tr: dict, only_required: np.ndarray | None, lo: int, hi: int) -> int:
        """Rows `{"g","k","w","v"}` of the gates of `[lo, hi)` (all, or the required ones) in gate order -- one line per gate, the
        same bytes as the run-by-run writer.  [E2 SCALE 2026-09-12] the run table (starts / counts / kinds / widths, `RunBlock`s
        expanded with numpy) is built once per call and every gate's run is found by ONE vectorised searchsorted: the old writer
        did two searchsorted per run, i.e. per gate for the one-gate `AmpereBF16TcDot16` runs (754 s of E8's 32x16 job)."""
        import gzip
        opener = gzip.open if path.suffix == ".gz" else open
        starts_l: list[np.ndarray] = []
        counts_l: list[np.ndarray] = []
        widths_l: list[np.ndarray] = []
        kinds: list[str] = []
        pend: list[tuple[int, int, str, int]] = []

        def flush() -> None:
            if pend:
                starts_l.append(np.fromiter((r[0] for r in pend), dtype=np.int64, count=len(pend)))
                counts_l.append(np.fromiter((r[1] for r in pend), dtype=np.int64, count=len(pend)))
                widths_l.append(np.fromiter((r[3] for r in pend), dtype=np.int64, count=len(pend)))
                kinds.extend(r[2] for r in pend)
                pend.clear()

        for r in runs:
            if isinstance(r, _vw.RunBlock):
                flush()
                s, c, k, w = r.arrays()
                starts_l.append(s)
                counts_l.append(c)
                widths_l.append(w)
                kinds.extend(k)
            else:
                pend.append(r)
        flush()
        starts = np.concatenate(starts_l) if starts_l else np.zeros(0, dtype=np.int64)
        counts = np.concatenate(counts_l) if counts_l else np.zeros(0, dtype=np.int64)
        widths = np.concatenate(widths_l) if widths_l else np.zeros(0, dtype=np.int64)
        if only_required is not None:
            gates = only_required[int(np.searchsorted(only_required, lo)):int(np.searchsorted(only_required, hi))]
        else:
            gates = np.arange(lo, hi, dtype=np.int64)
        if len(gates) == 0:
            with opener(path, "wt"):
                pass
            return 0
        idx = np.searchsorted(starts, gates, side="right") - 1
        if idx.min() < 0 or not bool(np.all(gates < starts[idx] + counts[idx])):
            raise RuntimeError(f"gate outside every run of the call [{lo}, {hi})")
        wid = widths[idx].tolist()
        gather = getattr(tr, "gather", None)
        words = gather(gates).tolist() if gather is not None else [tr[g] for g in gates.tolist()]
        lines = [f'{{"g":{g},"k":"{kinds[i]}","w":{w},"v":"{v.to_bytes((w + 7) // 8, "little").hex()}"}}\n'
                 for g, i, w, v in zip(gates.tolist(), idx.tolist(), wid, words, strict=True)]
        with opener(path, "wt") as fh:
            fh.write("".join(lines))
        return len(lines)

    def _write_summaries(self, bi: BoundaryIndex, rate_rows: list[dict], done_gates: int, planned_gates: int, program, files: list[dict], t_eval0: float) -> None:
        s = bi.summary()
        (self.out / "boundary_compare.json").write_text(json.dumps(s, indent=1))
        elapsed = time.perf_counter() - t_eval0
        computed = program.gates - program.input_gates
        us = 1e6 * elapsed / done_gates if done_gates else None
        rate = {"schema": "veritor/e1/rate/v1", "label": "MEASURED (this process, this machine)", "utc": utc(), "host": os.uname().nodename, "python": sys.version.split()[0],
                "gates_evaluated": done_gates, "planned_gates": planned_gates, "root_calls": len(rate_rows), "eval_wall_s": round(elapsed, 1),
                "gates_per_s": round(done_gates / elapsed, 1) if elapsed else None, "us_per_gate": round(us, 2) if us else None, "max_rss_gb": round(rss_gb(), 3),
                "retained_transcript_entries": rate_rows[-1]["retained"] if rate_rows else 0,
                "projection": {"label": "MODELED (measured rate x the layout's computed gate count)", "computed_gates_per_ru": computed,
                               "full_ru_hours_at_measured_rate": round(computed * us / 3.6e9, 2) if us else None,
                               "note": "A3's 26.9 M gates per RU divided the per-request count by the 4 requests; the pinned program is one request = one RU"},
                "per_root_call": rate_rows[-50:], "files_so_far": len(files)}
        (self.out / "rate.json").write_text(json.dumps(rate, indent=1))
        (self.out / "files_so_far.json").write_text(json.dumps(files))


def lay_name(plan, seq: int) -> str:
    try:
        return plan[seq][1].id
    except Exception:  # noqa: BLE001
        return "?"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--record", type=Path, default=DEFAULT_RECORD, help="attempt-6 record dir (…/rec/V5-TINY-bex0-inc-elem)")
    ap.add_argument("--request", default=DEFAULT_REQUEST)
    ap.add_argument("--checkpoint", type=Path, default=None, help="model.safetensors (default: the HF cache snapshot)")
    ap.add_argument("--cos-sin-npy", type=Path, default=None, help="GPU-built cos_sin_cache words (uint16 [MAX_POS, D]); default: the record's committed words (weights/values.bin) when stored, else the B0 fixture; sha256-checked against the committed leaf either way")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--prefill-layers", type=int, default=None, help="smoke: only prefill root calls of layers < N (embed + pre/attn/post)")
    ap.add_argument("--max-root-calls", type=int, default=0, help="smoke: stop after N root calls")
    ap.add_argument("--gzip", action="store_true", help="write .jsonl.gz instead of .jsonl")
    ap.add_argument("--only-required", action="store_true", help="write only the required (interior VU output) gates of this run's own inventory; default: every computed gate")
    ap.add_argument("--only-required-from", type=Path, default=None, help="write only the gates listed in this required_gates.bin (uint32 LE sorted)")
    ap.add_argument("--only-boundary", action="store_true", help="[E2 CURVE] write only the committed operator-boundary gates of the record's RU (no population needed; enough for the offline compare_boundaries re-check)")
    ap.add_argument("--stop-on-mismatch", action="store_true")
    ap.add_argument("--allow-weight-mismatch", action="store_true")
    ap.add_argument("--checkpoint-every", type=int, default=5, help="rewrite boundary_compare.json / rate.json every N root calls")
    ap.add_argument("--walk-check-calls", type=int, default=3)
    ap.add_argument("--walk-check-sample", type=int, default=64)
    ap.add_argument("--walk-check-named", default=None,
                    help='root calls checked against Circuit.gate before the run, "name[:sample],..." (no sample = every gate), e.g. "t=0/layers.0/post,t=3/final:20000"')
    ap.add_argument("--no-fast-args", action="store_true", help="resolve operands with layout._prim_member_args only (no affine/memo shortcuts)")
    ap.add_argument("--declared-case", default=None, help="[declared-inputs mode] registry case (b1.<case>_CONFIG) of a model with no served record yet; with --lp/--steps/--prompt-ids; no boundary comparison, no inventory")
    ap.add_argument("--attention", default=None, help='[E7 D57] declared attention kernel geometry JSON ({"kernel":"fa2-splitkv-align","kBlockM":64,"kBlockN":128,"num_splits":1}): with --declared-case -> Serve_v4; '
                                                      'with --record -> DIAGNOSTIC override of the pinned identity (D54 acceptance on a Serve_v1-pinned record). Omitted: the identity as pinned / Serve_v1')
    ap.add_argument("--lp", type=int, default=4)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--prompt-ids", default=None, help="[declared-inputs mode] comma-separated prompt token ids (LP of them)")
    ap.add_argument("--profile-id", default="000000000000", help="[declared-inputs mode] the p= field of the canonical ids in inventory (unknown before serving)")
    ap.add_argument("--reinventory", action="store_true", help="only rebuild inventory.json + required_gates.bin of --out from its files_so_far.json, then exit")
    ap.add_argument("--pool", type=int, default=0, help="[E2 SCALE follow-up] evaluate root calls over a pool of N forked workers (forked once; a call is dispatched when its inputs are complete); 0/1 = sequential")
    ap.add_argument("--pool-lookahead", type=int, default=96, help="[E2 SCALE follow-up] how many pending root calls (plan order) the pool scheduler scans for a ready one")
    ap.add_argument("--no-flat", action="store_true", help="[E2 SCALE follow-up] no flat (level-scheduled) template for the attention calls; member-vectorised batches only")
    ap.add_argument("--flat-check-all", action="store_true", help="[E2 SCALE follow-up] check EVERY flat-templated call against the concrete gate walk (default: the first call of each function id)")
    ap.add_argument("--no-vectorize", action="store_true", help="[E2 SCALE] evaluate every gate one at a time (the pre-2026-09-12 path); default: composite batch nodes are evaluated member-vectorised (e2e.interior.vecwalk)")
    ap.add_argument("--parallel", type=int, default=1, help="processes per root call: the members of large composite batches (GEMM dots) are sliced over forked workers")
    ap.add_argument("--parallel-min-gates", type=int, default=20000, help="composite batches with fewer gates are evaluated in the parent")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args(argv)
    if a.only_required_from is not None:
        a.only_required_from = np.frombuffer(Path(a.only_required_from).read_bytes(), dtype="<u4").astype(np.int64)
    return Producer(a).run()


if __name__ == "__main__":
    raise SystemExit(main())
