"""One-stage audit, run A11: GLM-5.3 TP8 B1 (row ``glm53__fp8__h200__tp8__b1__i256__o32__mixed__greedy__eager``), rank 0, through
the one-stage audit A6 ran on stand-in coordinates: each Definition kind (family) whose unit M0 lowers (`a11_kinds.py`'s table)
is a template, its units staged from the step's own values, sampled by the stratified law in one audit, proved by M0 under
``--zk`` and held by the Lean verifier of record.

    PYTHONPATH=<tree>:<tree>/verity/ml/flock/python:<tree>/integrations/vllm \\
      python benchmarks/one_stage/a11.py --values DIR --kinds kinds.json --roles derive-report.json --bin flock-circuit \\
      --lean flock-verify --out DIR [--per-kind 8] [--steps 1,0] [--layers 0,3] [--k K]

``--save-values DIR`` stops after the values, writing each kind's rows and its unit Definitions (a few MB); ``--load-values
DIR`` stages from them without the fold, which is most of a run's time on a shared CPU node.

* **Values.** ``--values`` is `a11_slice.py`'s tar extracted: rank 0's observation log, the capture's snapshots of a few
  layers' events (activations, copied right after each call) and ``ckpt/``, the checkpoint of record's tensors of those
  layers. The log is folded into the request Program's root Calls (`observe.fold`, the integration's derived profile). A
  Call's operand is the snapshot of its view taken before the Call ran, else (an operand produced outside the slice's
  events) the producing Call's output (its allocation's last snapshot before another Call writes it, as a reader without
  a snapshot of its own read it); a weight field is composed for rank 0 of 8 from the checkpoint by
  `check.weights_of_record.compose` and the Build's load-time ue8m0 rewrite (`check.fp8_ue8m0`, roles from ``--roles``), the
  composition whose per-field digests the Commit's weights pin held equal to the served parameters; the rotary table is the
  capture's ``cos_sin_cache.npy``. A Call reading another rank's values (``peers``) has none here and is not staged.
* **Units.** The Call is evaluated gate by gate (`circuits.evaluation.reference`) on those operands and held to the capture's output
  snapshot (the twin; for a kernel whose write lands after its own event's snapshot, the first later one before another
  writer that holds it, `call_twin`); its first ``--per-kind`` unit calls (the kind's unit, `a11_kinds.chain`) give each
  unit's arguments and returns. Each is packed into the unit's rows (`a11_kinds.rows_of`, an output row per output port,
  `a11_kinds.out_rows`) and staged by C-Flock's `UnitStager` bound to the partition, whose IR reference check and the
  transcript's returns must agree. A kind the table lowers through its Boolean version (``boolean``: MoeSum_v1 as
  MoeSum_v3's 64-column members, TokenSelect_v1 as TokenSelect_v2's argmax leaves) evaluates that version bit-sliced on the
  same operands (`boolean_units`), holds its returns to the snapshot as words, and stages its unit's bits.
* **Audit.** As `a6.py`: kinds sharing a unit Definition share its template, one member each; one registration before the
  draw; the verifier's Lean stratified draw over its own program and partition; each member serves its share
  (``serve --zk --draw-file``) and Lean ``verify --zk --partition --program`` holds each session; the audit record and
  a6's negatives. A tampered copy of one kind's member (bit 0 of every instance's first output word flipped before it is
  committed) is registered and proved on its own (``subset:2``), and the Lean verifier must refuse it.

The window is a stand-in: the Commit (``retain=host``) kept no values, so these are the Match capture's, an independent
instrumented run of the same row whose equality to the Commit's committed values rests on the row's oracle comparison and
token equality, provenance and not proof.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import socket
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "served_units"))
sys.setrecursionlimit(100000)

import a0  # noqa: E402
import a6  # noqa: E402
import a11_kinds as K  # noqa: E402
import r2_population as R2  # noqa: E402
from verity.ml.circuits.codec import canonical_json, encode_program  # noqa: E402
from verity.ml.circuits.defs import PrimitiveDefinition  # noqa: E402
from verity.ml.circuits.evaluation.reference import evaluate_region  # noqa: E402
from verity.ml.circuits.program import Program  # noqa: E402
from verity.ml.circuits.refs import runs  # noqa: E402
from verity_flock import circuit as FC  # noqa: E402
from verity_flock import class_statement as CS  # noqa: E402
from verity.core.protocols.sampled_proofs.one_stage import draw as D, partition as P, registration as R, verifier as V  # noqa: E402

ROW = "glm53__fp8__h200__tp8__b1__i256__o32__mixed__greedy__eager"
WINDOW_OF = ("GLM-5.3 TP8 B1 rank 0's step values: the Match capture's activation snapshots of row " + ROW
             + " and its weights composed for rank 0 from the checkpoint of record (zai-org/GLM-5.3 aca966e4)")
LINK = ("provenance, not proof: the Commit (retain=host) kept no values; the Match capture is an independent instrumented run of "
        "the same row, equal to the Commit by the row's oracle comparison (TP partials) and token equality, and its weights by "
        "the Commit's weights pin (per-field sha256 of the served parameters against this composition)")
LAYER = re.compile(r"layers\.(\d+)/")
WRAPPER = "inner.model."


def field_type(t, name: str):
    """The type of weights field ``name`` (`_field_intervals`' ``a[i].b`` names) in the weights Record ``t``."""
    for part in name.split("."):
        m = re.fullmatch(r"(\w+)(?:\[(\d+)\])?", part)
        t = t.fields[t.index(m[1])][1]
        if m[2] is not None:
            t = t.elem
    return t


def shape_bits(t) -> tuple[list[int], int]:
    shape = []
    while hasattr(t, "elem"):
        shape.append(t.n)
        t = t.elem
    return shape, t.w


class Weights:
    """Leaves of the Program's ``weights`` parameter: each field composed for rank ``rank`` of ``world`` from the checkpoint of
    record, as `weights_of_record` composes the field it pins."""

    def __init__(self, wtype, profile, ckpt: Path, roles_file: Path | None, cos_sin: Path, world: int = 8, rank: int = 0) -> None:
        from verity_vllm.check import fp8_ue8m0, weights_of_record as WR
        from verity_vllm.check.checkpoint_index import CheckpointIndex
        from verity_vllm.program.frontend.rules.family import _field_intervals
        self.WR, self.wtype, self.world, self.rank, self.cos_sin = WR, wtype, world, rank, cos_sin
        self.ck = CheckpointIndex(str(ckpt), verify_shards=False)
        self.intervals = _field_intervals(wtype, "")
        roles = {}
        if roles_file is not None:
            roles = fp8_ue8m0.build_roles(str(roles_file.parent)) if roles_file.name.startswith("derive-report") else {}
        # every served name the Program can read: the Build's roles, the checkpoint's own names and their fused names
        names = {f[len(WRAPPER):] if f.startswith(WRAPPER) else f for f in roles} | set(self.ck.tensors)
        for n in list(self.ck.tensors):
            for fused, srcs in WR.FUSED.items():
                for s in srcs:
                    if f".{s}." in n:
                        names.add(n.replace(f".{s}.", f".{fused}."))
            m = re.match(r"^(model\.layers\.\d+)\.self_attn\.kv_b_proj\.weight$", n)
            if m:
                names |= {f"{m[1]}.self_attn.W_UK_T", f"{m[1]}.self_attn.W_UV"}
        self.served: dict[str, str] = {}
        for n in sorted(names):
            path = profile.weight_field(n)
            if path:
                key = ".".join(f"{p}[{path[i + 1]}]" if i + 1 < len(path) and isinstance(path[i + 1], int) else str(p)
                               for i, p in enumerate(path) if not isinstance(p, int))
                self.served.setdefault(key, n)
        self.dims = {}
        for f, s in self.served.items():
            self.dims[WRAPPER + s] = shape_bits(field_type(wtype, f))
        for n, (_fn, _dt, shape, _lo, _hi) in self.ck.tensors.items():
            self.dims.setdefault(WRAPPER + n, ([shape[0] // world] + list(shape[1:]) if shape else [], 0))
        self.rw = fp8_ue8m0.Rewrites(list(self.dims), roles, self._raw, self.ck.config)
        self.notes: dict[str, dict] = {}
        self._cache: dict[str, np.ndarray] = {}

    def _raw(self, field: str):
        shape, bits = self.dims[field]
        live = field[len(WRAPPER):]
        return self.WR.compose(live, shape, bits or {"F8_E4M3": 8, "F32": 32, "BF16": 16}[self.ck.dtype(live)], self.ck,
                               world=self.world, rank=self.rank)

    def field(self, name: str) -> np.ndarray:
        if name in self._cache:
            return self._cache[name]
        shape, bits = shape_bits(field_type(self.wtype, name))
        if name == "cos_sin":
            arr, note = np.load(self.cos_sin), {"rule": "the capture's rank0/cos_sin_cache.npy (the served table's words)"}
        else:
            live = self.served.get(name)
            if live is None:
                raise LookupError(f"weights field {name}: no served parameter maps to it")
            full = WRAPPER + live
            arr, note = (self.rw.compose(full, shape) if self.rw.covers(full)
                         else self.WR.compose(live, shape, bits, self.ck, world=self.world, rank=self.rank))
        w = np.ascontiguousarray(arr).reshape(-1)
        if w.size != int(np.prod(shape)):
            raise LookupError(f"weights field {name}: {w.size} words for shape {shape}")
        self._cache[name] = w
        self.notes[name] = {"served": self.served.get(name), "sha256": hashlib.sha256(w.tobytes()).hexdigest(),
                            **{k: note.get(k) for k in ("rule", "shard", "cast")}}
        return w

    def words(self, lo: int, hi: int) -> np.ndarray:
        out = []
        for flo, fhi, name in self.intervals:
            if fhi <= lo or flo >= hi:
                continue
            try:
                w = self.field(name)
            except KeyError as e:
                raise LookupError(f"weights field {name}: {e}") from None
            out.append(w[max(lo, flo) - flo:min(hi, fhi) - flo].astype(np.uint64))
        w = np.concatenate(out) if out else np.zeros(0, np.uint64)
        if w.size != hi - lo:
            raise LookupError(f"weights leaves [{lo}, {hi}) not covered")
        return w


def fold_values(vdir: Path, profile_name: str, roles: Path | None, last_step: int | None = None, weights_cls=None):
    """The rank's log folded into the request Program's root Calls, its snapshots, and the weights (`Weights`, or
    ``weights_cls``); with ``last_step``, only the log up to that step's end (the fold is causal, so its Calls are those of
    the whole log's first steps)."""
    from verity_vllm.check import fold_compare
    from verity_vllm.check.fold_compare import SnapshotStore, Values, _attach_specs, load_profile
    from verity_vllm.observe.events import RunHeader, StepStart
    from verity_vllm.observe.fold.fold import fold
    from verity_vllm.observe.fold.patterns.pouw import with_protocols
    from verity_vllm.observe.log import read_log
    t0 = time.time()
    for dt in ("float8_e4m3fn", "float8_e5m2", "float8_e8m0fnu"):
        fold_compare._NP.setdefault(dt, np.uint8)
    profile = load_profile(profile_name)
    events = []
    for e in read_log(next(vdir.glob("log.rank*.jsonl.gz"))):
        if last_step is not None and isinstance(e, StepStart) and e.step > last_step:
            break
        events.append(e)
    header = next((e for e in events if isinstance(e, RunHeader)), None)
    res = fold(events, with_protocols(profile, header), strict=None, with_program=False)
    _attach_specs(res)
    store = SnapshotStore(vdir / "rank0")
    weights = (weights_cls or Weights)(res.params[-1][1], profile, vdir / "ckpt", roles, vdir / "rank0" / "cos_sin_cache.npy")

    class A11Values(Values):
        def of_run(self, space, idx, lo, hi):
            if space == "p":
                if idx == self.W:
                    return weights.words(lo, hi)
                name = self.r.params[idx][0]
                if name == "prompt":
                    toks = self.r.requests[0]["prompt_token_ids"]
                    return np.array([int(t) & 0xFFFFFFFF for t in toks[lo:hi]], dtype=np.uint64)
                raise LookupError(f"parameter {name!r}: another rank's values, which this slice does not hold")
            if space == "n":
                return producer_words(self.by_node.get(idx), store, lo, hi, idx, write_events(self.r))
            return super().of_run(space, idx, lo, hi)

    return res, store, weights, A11Values(res, store, None), round(time.time() - t0, 1)


_CONSTANTS: dict[str, np.ndarray] = {}


def constant_words(spec) -> np.ndarray:
    """The returns of a Definition without parameters (a ``Const32`` Call, which the capture has no snapshot of)."""
    if spec.id not in _CONSTANTS and isinstance(spec, PrimitiveDefinition):
        _CONSTANTS[spec.id] = np.array([spec.evaluate()], dtype=np.uint64)
    if spec.id not in _CONSTANTS:
        C = Program(spec).circuit
        t: dict[int, int] = {}
        evaluate_region(C, C.region(0, C.size), t)
        _CONSTANTS[spec.id] = np.array([t[g] for g in C.root_outputs()], dtype=np.uint64)
    return _CONSTANTS[spec.id]


def producer_words(m: dict | None, store, lo: int, hi: int, idx: int, writes: dict[int, list[int]] | None = None) -> np.ndarray:
    """Return leaves [lo, hi) of the Call ``m`` from its output snapshots, its views aligned to its trailing return components
    (as `call_twin`); a Call without parameters is evaluated. With ``writes`` (`write_events`), each view is read from its
    allocation's last snapshot before the next Call writes it (what a reader without its own snapshot read: the DSA top-k's
    logits), else from the one at the Call's event."""
    from verity_vllm.check.fold_compare import _component_ranges
    if m is None:
        raise LookupError(f"node {idx} has no instance record")
    comps, views = _component_ranges(m), m["output_views"]
    spec = m["_spec"]
    if not spec.signature()[0]:
        return constant_words(spec)[lo:hi]
    first = len(comps) - len(views)
    out = []
    for k, (clo, chi) in enumerate(comps):
        if chi <= lo or clo >= hi:
            continue
        view = views[k - first] if k >= first else None
        if view is None:
            raise LookupError(f"node {idx} ({m['name']}) component {k}: no output view")
        later = later_snapshots(store, view, m["event"], writes) if writes is not None else []
        s = later[-1] if later else store.find(view, at=m["event"])
        if s is None:
            raise LookupError(f"node {idx} ({m['name']}): no snapshot of event {m['event']} covers {view!r}")
        out.append(store.extract(s, view)[max(lo, clo) - clo:min(hi, chi) - clo])
    w = np.concatenate(out) if out else np.zeros(0, np.uint64)
    if w.size != hi - lo:
        raise LookupError(f"node {idx}: leaves [{lo}, {hi}) outside its outputs")
    return w


def operands(res, store, vals, m: dict, first: str = "snapshot") -> list[np.ndarray]:
    """Each operand's words from one of two sources, ``first`` and the other as fallback: the operand view's snapshot the
    capture took at the Call (the latest before it, for an in-place Call), or provenance (weights, prompt, the producing Call's
    output snapshot). Neither is right everywhere: a producer's snapshot at its own event can predate its write (the router's
    logits, which the gate GEMM writes later), so only the reader's snapshot holds them; a view's latest snapshot can predate
    its producer's write."""
    from verity_vllm.check.fold_compare import _in_place
    node = res.body.nodes[m["node"]]
    out = []
    for k, a in enumerate(node.args):
        view = m["operand_views"][k] if k < len(m["operand_views"]) else None

        def snap():
            if all(s == "p" for s, *_ in runs(a.refs)) or view is None or view.numel != a.refs.count:
                raise LookupError(f"operand {k}: no snapshot view")
            s = store.find(view, before=m["event"] if _in_place(m) else m["event"] + 1)
            if s is None:
                raise LookupError(f"operand {k}: no snapshot of {view!r} before event {m['event']}")
            return store.extract(s, view)

        a_, b_ = (snap, lambda: vals.of_refs(a.refs)) if first == "snapshot" else (lambda: vals.of_refs(a.refs), snap)
        try:
            w = a_()
        except LookupError:
            w = b_()
        out.append(np.asarray(w).astype(np.uint64))
    return out


def write_events(res) -> dict[int, list[int]]:
    """Each allocation's writing Calls' events (an output view's), ascending."""
    if not hasattr(res, "_a11_writes"):
        w: dict[int, set[int]] = {}
        for m in res.instances:
            for v in m["output_views"]:
                if v is not None:
                    w.setdefault(v.alloc, set()).add(m["event"])
        res._a11_writes = {a: sorted(e) for a, e in w.items()}
    return res._a11_writes


def later_snapshots(store, view, event: int, writes: dict[int, list[int]]) -> list:
    """The snapshots covering ``view`` after ``event`` and before the next Call that writes its allocation, in event order."""
    nxt = next((e for e in writes.get(view.alloc, []) if e > event), None)
    return [s for s in store.by_alloc.get(view.alloc, []) if s.event > event and (nxt is None or s.event < nxt) and store.covers(s, view)]


def call_twin(store, m: dict, got: list, writes: dict[int, list[int]] | None = None) -> dict:
    """The evaluated Call's returns against the capture's output snapshots, on the returns evaluated. The capture has a view
    per kernel output; a Definition that also returns intermediates ahead of them (DeepGEMM's quantized input and scales)
    has fewer views than return components, and the views are its trailing components. Some kernels' writes land after their
    own event's snapshot (the MoE router's gate GEMM; the DSA indexer's logits, which hold them only at the top-k's event):
    with ``writes`` (`write_events`), a component that disagrees there is held to the first later snapshot of its allocation,
    before any other Call writes it, that holds it (``late``, its event)."""
    from verity_vllm.check.fold_compare import _component_ranges
    comps, views = _component_ranges(m), m["output_views"]
    if len(views) > len(comps):
        return {"compared": 0, "agree": False, "why": f"{len(views)} output views for {len(comps)} return components"}
    compared, diff, held, late = 0, [], [], {}
    for k, ((clo, chi), view) in enumerate(zip(comps[len(comps) - len(views):], views), start=len(comps) - len(views)):
        if view is None:
            continue
        if view.numel != chi - clo:
            return {"compared": 0, "agree": False, "why": f"component {k}: {chi - clo} returns vs a {view.numel}-element view"}
        s = store.find(view, at=m["event"])
        if s is None:
            continue
        held.append(k)
        idx = [i for i in range(clo, chi) if got[i] is not None]
        def bad(snap, view=view, clo=clo, idx=idx):
            want = store.extract(snap, view)
            return [i for i in idx if int(got[i]) != int(want[i - clo])]
        d = bad(s)
        if d and writes is not None:
            s2 = next((x for x in later_snapshots(store, view, m["event"], writes) if not bad(x)), None)
            if s2 is not None:
                d, late[k] = [], s2.event
        compared += len(idx)
        diff += d
    if not held:
        return {"compared": 0, "agree": None, "why": "no output snapshot"}
    return {"compared": compared, "of": len(got), "components": held, "of_components": len(comps),
            "agree": (not diff) if compared else None, "first_diff": diff[:4], **({"late": late} if late else {})}


def find_units(circuit, unit, n: int) -> list:
    out: list = []

    def walk(calls) -> None:
        for c in calls:
            if len(out) >= n:
                return
            if c.is_primitive:
                continue
            if c.fn.id == unit.id:
                out.append(c)
            elif c.fn.gates > unit.gates:
                walk(c.children())

    walk(circuit.root_calls())
    return out


def evaluate_units(spec, unit, args: list[np.ndarray], n: int, full_gates: int) -> dict:
    """The Call evaluated gate by gate on ``args``; the first ``n`` unit calls' argument and return leaves."""
    t0 = time.time()
    prog = Program(spec)
    C = prog.circuit
    t: dict[int, int] = {}
    for pi, w in enumerate(args):
        gates = [(lo, hi) for _s, _i, lo, hi in runs(prog.input_coll(pi).refs)]
        if sum(hi - lo for lo, hi in gates) != w.size:
            raise ValueError(f"parameter {pi}: {w.size} words for {sum(hi - lo for lo, hi in gates)} input gates")
        vals, at = w.tolist(), 0
        for lo, hi in gates:
            t.update(zip(range(lo, hi), vals[at:at + hi - lo]))
            at += hi - lo
    n_in = prog.input_gates
    if unit.id == spec.id:
        units = None
        hi = C.size
    else:
        units = find_units(C, unit, n)
        if not units:
            raise ValueError(f"no {unit.id} call in {spec.id}")
        hi = C.size if C.size - n_in <= full_gates else max(c.hi for c in units)
    evaluate_region(C, C.region(n_in, hi), t)
    outs = C.root_outputs()
    got = [t.get(g) for g in outs]
    if units is None:
        leaves_in = [[t[g] for g in range(n_in)]]
        leaves_out = [got]
    else:
        leaves_in = [[t[g] for g in c.args] for c in units]
        leaves_out = [[t[g] for g in c.returns] for c in units]
    return {"in": leaves_in, "out": leaves_out, "call_out": got, "gates": C.size - n_in, "evaluated": hi - n_in,
            "seconds": round(time.time() - t0, 1)}


def _reaches(c, unit_id: str) -> bool:
    return any(s[0] in ("call", "scan") and (s[1].target == unit_id or _reaches(s[1], unit_id)) for s in c.steps)


def _run_units(c, args: np.ndarray, unit_id: str, sink: list, n: int) -> np.ndarray:
    """`evaluation.bits._run`, keeping in ``sink`` the argument and return bits of instance 0's first ``n`` calls of
    ``unit_id``."""
    from verity.ml.circuits.evaluation import bits as BS
    words = args.shape[1]
    W = np.empty((c.size, words), dtype=np.uint64)
    W[0] = 0
    W[1] = ~np.uint64(0)
    W[2:2 + c.n_params] = args
    for step in c.steps:
        if step[0] == "gates":
            for kind, out, a, b in step[1]:
                W[out] = W[a] & W[b] if kind == 0 else W[a] ^ W[b] if kind == 1 else ~W[a]
        elif step[0] == "call":
            _, child, members, base, w = step
            if child.target != unit_id and _reaches(child, unit_id):
                for i, mem in enumerate(members):
                    W[base + i * w:base + (i + 1) * w] = _run_units(child, W[mem], unit_id, sink, n)
                continue
            per = max(1, BS._ROW_BUDGET // (8 * child.size * words))
            for lo in range(0, len(members), per):
                m = members[lo:lo + per]
                k = len(m)
                rows = W[m].transpose(1, 0, 2).reshape(m.shape[1], k * words)
                W[base + lo * w:base + (lo + k) * w] = BS._run(child, rows).reshape(w, k, words).transpose(1, 0, 2).reshape(k * w, words)
            if child.target == unit_id:
                for i in range(min(len(members), n - len(sink))):
                    sink.append(([int(x) for x in W[members[i], 0] & np.uint64(1)],
                                 [int(x) for x in W[base + i * w:base + (i + 1) * w, 0] & np.uint64(1)]))
        else:
            if _reaches(step[1], unit_id) or step[1].target == unit_id:
                raise NotImplementedError(f"{c.target}: a call of {unit_id} inside a scan")
            _, child, init, xs, shared, nn, cl, base = step
            carry, ys = W[init], []
            for m in range(nn):
                parts = [carry] + [W[r[m * e:(m + 1) * e]] for r, e in xs] + [W[r] for r in shared]
                r = BS._run(child, np.concatenate(parts) if parts else np.empty((0, words), np.uint64))
                carry = r[:cl]
                ys.append(r[cl:])
            W[base:base + cl] = carry
            if ys:
                y = np.concatenate(ys)
                W[base + cl:base + cl + len(y)] = y
    return W[c.ret_idx]


def boolean_units(spec, unit, args: list[np.ndarray], n: int, full_gates: int) -> dict:
    """`evaluate_units` for a Boolean Call: ``spec`` evaluated bit-sliced (`evaluation.bits`) on the word operands (element j
    of a parameter's word view is its word leaf j), its first ``n`` calls of ``unit`` with their argument and return bits, and
    the Call's returns as words (``call_out``, element j of the return's word view at word leaf j, as `call_twin` reads a
    word Call's)."""
    from verity.ml.circuits import boolean as BL
    from verity.ml.circuits.evaluation import bits as BS
    t0 = time.time()
    c = BS.compile(spec)
    cols = []
    for pi, (w, t) in enumerate(zip(args, c.params)):
        widths = BL.elements(t)
        if w.size != len(widths):
            raise ValueError(f"parameter {pi}: {w.size} words for {len(widths)} elements")
        cols.append(BS._to_bits(np.asarray(w, dtype=np.uint64).reshape(1, -1), widths))
    sink: list = []
    ret = _run_units(c, BS._pack(np.concatenate(cols, axis=1)), unit.id, sink, n)
    if not sink:
        raise ValueError(f"no {unit.id} call in {spec.id}")
    got = BS._to_words(BS._unpack(ret, 1), BL.elements(c.ret))[0]
    return {"in": [s[0] for s in sink], "out": [s[1] for s in sink], "call_out": [int(x) for x in got], "gates": spec.gates,
            "evaluated": spec.gates, "seconds": round(time.time() - t0, 1)}


def pack(vals, widths, pad16: bool = False) -> np.ndarray:
    """Leaves little-endian bit-concatenated (`class_statement.unit_lowering`'s rows) as u16 words."""
    bits = []
    i = 0
    while i < len(widths):
        j = i
        while j < len(widths) and widths[j] == widths[i]:
            j += 1
        w, v = widths[i], vals[i:j]
        if w in (8, 16, 32, 64):
            dt = {8: "<u1", 16: "<u2", 32: "<u4", 64: "<u8"}[w]
            bits.append(np.unpackbits(np.asarray([int(x) & ((1 << w) - 1) for x in v], dtype=np.uint64).astype(dt).view(np.uint8),
                                      bitorder="little"))
        elif w < 64:
            a = np.asarray([int(x) for x in v], dtype=np.uint64)
            bits.append(((a[:, None] >> np.arange(w, dtype=np.uint64)) & 1).astype(np.uint8).reshape(-1))
        else:
            bits.append(np.array([(int(x) >> b) & 1 for x in v for b in range(w)], dtype=np.uint8))
        i = j
    b = np.concatenate(bits) if bits else np.zeros(0, np.uint8)
    if b.size % 16:
        if not pad16:
            raise ValueError(f"{b.size} bits are not whole u16 words")
        b = np.concatenate([b, np.zeros(16 - b.size % 16, np.uint8)])
    return np.packbits(b, bitorder="little").view("<u2").copy()


def unit_rows(unit, rows_spec, leaves_in: list[list[int]], leaves_out: list[list[int]]):
    """Each instance's input rows ``{row: (n, words)}`` and output ports ``{o{j}: (n, words)}`` from its leaves."""
    params, ret = unit.signature()
    spans, at = {}, 0
    for name, t in params:
        spans[name] = (at, at + t.leaves, t.leaf_widths())
        at += t.leaves
    rows = {}
    for r, _s, _b, ps in rows_spec:
        per = []
        for lv in leaves_in:
            vals, ws = [], []
            for p in ps:
                lo, hi, w = spans[p]
                vals += lv[lo:hi]
                ws += w
            per.append(pack(vals, ws))
        rows[r] = np.stack(per)
    fields = ret.fields if hasattr(ret, "fields") else (ret,)
    outs, at = {}, 0
    for j, f in enumerate(fields):
        outs[f"o{j}"] = np.stack([pack(lv[at:at + f.leaves], f.leaf_widths(), pad16=True) for lv in leaves_out])
        at += f.leaves
    return rows, outs


def pick(res, kind: str, unit_fam: str, steps: list[int], layers: set[int], unit_gates: int, probed: str | None = None,
         boolean: bool = False):
    """The kind's Calls in the chosen steps and layers (layer-less Calls too), each with the Definition evaluated (the Call's,
    or with ``boolean`` its Boolean version) and its unit: the deepest Definition of that one's chain in the probed unit's
    family. Calls of the probed specialization come first (the unit the table sized; prefill's attention, say, has one per
    prefix length), then by step preference."""
    from verity.ml.definitions.boolean import boolean_version
    cands = []
    for m in res.instances:
        if K.family(m["spec"]) != kind or m["step"] not in steps:
            continue
        lm = LAYER.search(m["name"])
        if lm and int(lm[1]) not in layers:
            continue
        cands.append(m)
    cands.sort(key=lambda m: (m["spec"] != probed, steps.index(m["step"]), m["event"]))
    for m in cands:
        spec = res.body.nodes[m["node"]].fn
        top = boolean_version(spec) if boolean else spec
        unit = next((f for f, _n in reversed(K.chain(top, unit_gates, boolean)) if K.family(f.id) == unit_fam), None)
        if unit is not None:
            yield m, spec, top, unit


def kind_values(res, store, vals, rec: dict, a) -> dict:
    """One kind: a Call whose operands resolve, evaluated and held to its output snapshot, and its first ``--per-kind`` units'
    rows; a kind whose unit is the Call itself takes one unit from each of up to ``--per-kind`` Calls of that Definition. A
    kind the table lowers through its Boolean version (``boolean``) evaluates that version on the Call's operands
    (`boolean_units`) and stages its unit from it."""
    steps, layers = [int(s) for s in a.steps.split(",")], {int(x) for x in a.layers.split(",")}
    why, tried = [], 0
    acc: dict | None = None
    boolean = bool(rec.get("boolean"))
    evaluate = boolean_units if boolean else evaluate_units
    for m, spec, top, unit in pick(res, rec["kind"], K.family(rec["unit"]), steps, layers, a.unit_gates, rec.get("call"), boolean):
        if acc is not None and len(acc["in"]) >= a.per_kind:
            break
        if acc is not None and spec.id != acc["spec"]:
            continue
        tried += 1
        if acc is None and tried > a.tries:
            break
        try:
            args = operands(res, store, vals, m)
        except LookupError as e:
            why.append(f"{m['name']}: {e}")
            continue
        rows_spec = K.rows_of(unit)
        if rows_spec is None:
            return {"staged": False, "why": f"{unit.id}: the parameters' bits are not whole u16 words"}
        n_want = a.per_kind - (len(acc["in"]) if acc else 0)
        ev = evaluate(top, unit, args, n_want, a.full_gates)
        twin = {**call_twin(store, m, ev.pop("call_out"), write_events(res)), "operands": "snapshot"}
        if twin["agree"] is False:
            try:
                alt = operands(res, store, vals, m, first="provenance")
            except LookupError:
                alt = args
            if any(not np.array_equal(x, y) for x, y in zip(alt, args)):
                ev2 = evaluate(top, unit, alt, n_want, a.full_gates)
                twin2 = {**call_twin(store, m, ev2.pop("call_out"), write_events(res)), "operands": "provenance"}
                if twin2["agree"] is not False:
                    ev, twin = ev2, twin2
        if twin["agree"] is False:
            why.append(f"{m['name']}: the Call evaluated on the capture's operands disagrees with its output snapshot "
                       f"({json.dumps({x: twin.get(x) for x in ('compared', 'first_diff', 'why')})})")
            continue
        if acc is None:
            acc = {"unit": unit, "spec": spec.id, "rows_spec": rows_spec, "in": [], "out": [], "calls": [], "twins": [], "evals": []}
        acc["in"] += ev.pop("in")
        acc["out"] += ev.pop("out")
        acc["calls"].append({"name": m["name"], "step": m["step"], "event": m["event"], "node": m["node"]})
        acc["twins"].append(twin)
        acc["evals"].append(ev)
        if unit.id != top.id:
            break
    if acc is None:
        return {"staged": False, "why": "; ".join(why[-3:]) or "no Call of this kind in the slice's steps and layers"}
    rows, outs = unit_rows(acc["unit"], acc["rows_spec"], acc["in"], acc["out"])
    agree = [t["agree"] for t in acc["twins"]]
    twin = {"calls": len(agree), "agree": False if False in agree else (True if all(agree) else None),
            "compared": sum(t["compared"] for t in acc["twins"]), "each": acc["twins"][:4]}
    c0 = acc["calls"][0]
    return {"staged": True, "unit": acc["unit"], "spec": acc["spec"], "name": c0["name"], "step": c0["step"], "event": c0["event"],
            "calls": acc["calls"], "rows_spec": acc["rows_spec"], "rows": rows, "outs": outs, "n": len(acc["in"]), "twin": twin,
            "eval": acc["evals"][0], "skipped_calls": why[-3:], **({"boolean": rec["boolean"]} if boolean else {})}


def stage_tampered(st, rows: dict, out_dir: Path, salt_key: bytes, indices, set_name: str) -> dict:
    """`UnitStager.stage` with bit 0 of every instance's first output word flipped before the set is committed."""
    from types import SimpleNamespace
    iset = SimpleNamespace(**{**vars(st.iset), "name": set_name})
    n = len(rows[st.in_rows[0][0]])
    flat = np.concatenate([np.asarray(rows[r], dtype=np.uint16) for r in st.row_ports], axis=1)
    got, _cut, ok = st.low.evaluate(flat.astype(np.uint64))
    if int(ok) != (1 << n) - 1:
        raise AssertionError(f"{st.name}: a unit is unsatisfied on its rows")
    got = got.astype(np.uint16)
    at = np.cumsum([0] + st.out_words)
    outs = {f"o{j}": got[:, at[j]:at[j + 1]].copy() for j in range(len(st.out_words))}
    st.reference_check(rows, outs)
    outs["o0"][:, 0] ^= 1
    salts = FC.stage_salts(iset, 0, n, st.ports, salt_key)
    out_dir.mkdir(parents=True, exist_ok=True)
    FC.write(st.text, iset, 0, n, out_dir / "inst.bin", out_dir / "pub.bin", salt_key,
             rows={r: np.asarray(rows[r], dtype=np.uint16) for r, *_ in st.in_rows}, outs=outs, salts=salts, public_words={},
             digests=(st.pin, st.class_sha512), indices=[int(u) for u in indices])
    return {"meta": json.loads(st._meta), "tampered": "o0", "word": 0, "bit": 0}


def session_port(base: int, j: int) -> int:
    """Session ``j``'s port: ``base + j``, else (``base`` 0) a port free now, so that audits sharing a node never cross one
    audit's prover with another's verifier."""
    if base:
        return base + j
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def save_values(d: Path, staged: dict, rows_of_kind: dict, weights_notes, a) -> None:
    """Each staged kind's rows and outputs (``<kind>.npz``), its unit Definitions as one encoded Program, and the staging notes."""
    d.mkdir(parents=True, exist_ok=True)
    units = {kv["unit"].id: kv["unit"] for kv in rows_of_kind.values()}
    (d / "units.json").write_bytes(canonical_json(encode_program(P.mixed_population_program([(u, 1) for u in units.values()]))))
    for f, kv in rows_of_kind.items():
        np.savez(d / f"{f}.npz", **{f"i.{r}": np.asarray(v, dtype=np.uint16) for r, v in kv["rows"].items()},
                 **{f"o.{o}": np.asarray(v, dtype=np.uint16) for o, v in kv["outs"].items()})
    (d / "values.json").write_text(json.dumps({"staged": staged, "weights": weights_notes, "values": a.values, "steps": a.steps,
                                               "layers": a.layers, "per_kind": a.per_kind}, indent=1, sort_keys=True, default=str))


#: Definitions built on first use, outside the registry's catalog: name -> (module, builder) of its one static. Building one
#: also registers its Boolean form for M0's lowering (`ir_lower._boolean_build`), which finds no other in a fresh process.
# TODO: once its `tail_pieces.FAMILIES` lowers `GatherE4m3x{V}_v1` and `MoeSelectBf16x{E}_v1` in a fresh process, lower
# through those pieces and keep this only for the decoded units' builders.
LAZY = {r"GatherE4m3x(\d+)_v2": ("verity.ml.definitions.registry.boolean_fp8_moe", "gather_e4m3"),
        r"GatherF32x(\d+)_v2": ("verity.ml.definitions.registry.boolean_fp8_moe", "gather_f32")}


def live(fn, registry):
    """A decoded Definition as the registry's own: a specialization bound again on its composite (the decoded one has no
    builder, which `Program` and the stager's binding need), function-valued statics likewise; one of `LAZY` built again."""
    import importlib
    from verity.ml.circuits.defs import FunctionDefinition, SpecializedDefinition
    if not isinstance(fn, SpecializedDefinition):
        return fn
    name = f"{fn.defn.name}_v{fn.defn.version}"
    for pat, (mod, builder) in LAZY.items():
        if name not in registry and (m := re.fullmatch(pat, name)):
            return getattr(importlib.import_module(mod), builder)(int(m[1])).bind()

    def static(v):
        if isinstance(v, FunctionDefinition):
            return live(v, registry)
        if isinstance(v, tuple):
            return tuple(static(x) for x in v)
        if isinstance(v, dict):
            return {k: static(x) for k, x in v.items()}
        return v
    return registry[name].bind(**{k: static(v) for k, v in fn.bindings.items()})


def load_values(d: Path, lowers: list[str]) -> tuple[dict, dict, dict]:
    """`save_values`' directory read back: the staged kinds among ``lowers``, each unit decoded from ``units.json`` and bound
    again from the registry, refused unless it encodes as the decoded one."""
    from verity.ml.circuits.codec import decode_program
    from verity.ml.definitions.registry.catalog import load as load_registry
    registry = load_registry()[0]
    doc = json.loads((d / "values.json").read_text())
    defs = {x.id: x for x in decode_program(json.loads((d / "units.json").read_text()), registry).defs.values()}
    enc = lambda u: canonical_json(encode_program(P.mixed_population_program([(u, 1)])))  # noqa: E731
    staged = {f: s for f, s in doc["staged"].items() if f in lowers}
    rows_of_kind = {}
    for f, s in staged.items():
        if not s.get("staged"):
            continue
        unit = live(defs[s["unit"]], registry)
        if unit.id != s["unit"] or enc(unit) != enc(defs[s["unit"]]):
            raise SystemExit(f"{f}: the registry's {unit.id} is not the saved {s['unit']}")
        z = np.load(d / f"{f}.npz")
        rows_of_kind[f] = {**s, "unit": unit, "rows_spec": K.rows_of(unit),
                           "rows": {k[2:]: z[k] for k in z.files if k.startswith("i.")},
                           "outs": {k[2:]: z[k] for k in z.files if k.startswith("o.")}}
    return staged, rows_of_kind, {"from": {x: doc.get(x) for x in ("values", "steps", "layers", "per_kind")}, "weights": doc["weights"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--values", default=None, help="a11_slice.py's tar, extracted")
    ap.add_argument("--save-values", default=None, help="write each kind's staged rows and its unit's Definition here and stop")
    ap.add_argument("--load-values", default=None, help="a --save-values directory: stage from it instead of folding --values")
    ap.add_argument("--kinds", required=True, help="a11_kinds.py's table")
    ap.add_argument("--roles", default=None, help="the Build's derive-report.json (weight_roles: the ue8m0 rewrite)")
    ap.add_argument("--profile", default="derived_GLM53_tp8")
    ap.add_argument("--bin", required=True)
    ap.add_argument("--lean", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-kind", type=int, default=8, help="units staged per kind (at most the statement cap)")
    ap.add_argument("--steps", default="1,0", help="steps to take a kind's Call from, in preference order")
    ap.add_argument("--layers", default="0,3")
    ap.add_argument("--only", default=None, help="comma-separated kinds")
    ap.add_argument("--try", dest="try_kinds", default=None,
                    help="comma-separated kinds staged though the table records no lowering for them (the stager's build is the test)")
    ap.add_argument("--k", type=int, default=None, help="the stratified rule's k (default: one unit a template)")
    ap.add_argument("--tries", type=int, default=4, help="Calls tried per kind")
    ap.add_argument("--unit-gates", type=int, default=4096)
    ap.add_argument("--full-gates", type=int, default=4_000_000, help="a Call up to this many gates is evaluated whole")
    ap.add_argument("--tamper", default=None, help="the kind whose member is tampered (default: the first staged; none: no tampered member)")
    ap.add_argument("--delta", type=float, default=2.0 ** -20)
    ap.add_argument("--port", type=int, default=0, help="the sessions' base port (default: a free port each)")
    ap.add_argument("--values-only", action="store_true", help="stop after staging values (no registration or proofs)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summary: dict = {"row": ROW, "rank": 0, "steps": a.steps, "layers": a.layers, "per_kind": a.per_kind}
    # M0 lowers a word primitive by the Boolean Definition registered for it (`ir_lower._boolean_build`), some of which only
    # the integration's registry modules register (U32And_v1's); a11_kinds' decode loads them all the same way
    from verity.ml.definitions.registry.catalog import load as load_registry
    summary["registry_errors"] = load_registry()[1]

    table = json.loads(Path(a.kinds).read_text())
    kinds = {f: r for f, r in table["kinds"].items() if r.get("gates", 0) > 0 and not f.startswith("Const")}
    tries = set(a.try_kinds.split(",")) if a.try_kinds else set()
    lowers = [f for f, r in kinds.items() if (r.get("outcome") == "lowers" or f in tries) and (not a.only or f in a.only.split(","))]
    summary["tried"] = sorted(tries & set(lowers))
    if a.load_values:
        staged, rows_of_kind, notes = load_values(Path(a.load_values), lowers)
        summary["values_from"] = {"dir": a.load_values, **notes["from"]}
        weights_notes = notes["weights"]
        print(f"A11 values loaded: {sorted(rows_of_kind)}", flush=True)
    else:
        if not a.values:
            raise SystemExit("--values (a11_slice.py's tar, extracted) or --load-values")
        res, store, weights, vals, fold_s = fold_values(Path(a.values), a.profile, Path(a.roles) if a.roles else None)
        print(f"A11 fold: {len(res.instances)} Calls, {len(store.entries)} snapshots, {len(weights.served)} weight fields mapped, "
              f"{fold_s} s", flush=True)
        staged, rows_of_kind = {}, {}
        for f in lowers:
            t = time.time()
            try:
                kv = kind_values(res, store, vals, {"kind": f, **kinds[f]}, a)
            except Exception as e:  # noqa: BLE001
                kv = {"staged": False, "why": f"{type(e).__name__}: {str(e)[:400]}", "trace": traceback.format_exc()[-1500:]}
            kv["seconds"] = round(time.time() - t, 1)
            staged[f] = {x: (y.id if x == "unit" else y) for x, y in kv.items() if x not in ("rows", "outs", "rows_spec")}
            if kv["staged"]:
                rows_of_kind[f] = kv
            print(f"A11 values {f}: " + json.dumps(staged[f], default=str)[:700], flush=True)
        weights_notes = weights.notes
        if a.save_values:
            save_values(Path(a.save_values), staged, rows_of_kind, weights_notes, a)
            print(f"A11 values saved to {a.save_values}: {sorted(rows_of_kind)}", flush=True)
            return 0 if rows_of_kind else 1
    summary["values"] = staged
    summary["weights"] = weights_notes
    if a.values_only or not rows_of_kind:
        (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
        return 0 if rows_of_kind else 1

    # templates: one per unit Definition; kinds sharing one are consecutive members of its range
    templates: dict[str, dict] = {}
    for f, kv in rows_of_kind.items():
        templates.setdefault(kv["unit"].id, {"defn": kv["unit"], "kinds": []})["kinds"].append(f)
    defs = [t["defn"] for t in templates.values()]
    ns = [sum(rows_of_kind[f]["n"] for f in t["kinds"]) for t in templates.values()]
    program = P.mixed_population_program(list(zip(defs, ns)))
    query = P.template_instances_query(defs)
    part = P.partition(program, query)
    pd = P.digest(part)
    n_all = P.units(program, query)
    ranges = P.ranges(program, query)
    (out / "partition.json").write_text(json.dumps(part, sort_keys=True))
    (out / "program.json").write_bytes(canonical_json(encode_program(program)))
    k = a.k or len(defs)
    law = D.parse_law(f"stratified:{k}")
    bound = {"partition": pd, "program": part["program"]}

    members, tags = [], {}
    for (uid, tpl), (tid, base, cnt) in zip(templates.items(), ranges):
        assert tid == P.template_id(tpl["defn"])
        t0 = time.time()
        # one stage directory (and circuit) per template, though two specializations share a family
        tag = K.family(uid)
        tags[tag] = tags.get(tag, 0) + 1
        tag = tag if tags[tag] == 1 else f"{tag}-{tags[tag]}"
        sdir = out / f"stage-{tag}"
        kv0 = rows_of_kind[tpl["kinds"][0]]
        with K.no_cyclic_gc():
            st = CS.UnitStager(tpl["defn"], f"unit/a11-{tag}", kv0["rows_spec"], K.out_rows(len(kv0["outs"])),
                               circuit=sdir / "circuit.txt", bind=bound)
        tpl["stager"], t_build = st, round(time.time() - t0, 1)
        cap = a6.chunk_cap(st.meta)
        salt_key = hashlib.sha256(f"verity/one-stage/a11/salts\0{uid}".encode()).digest()
        lo = 0
        for f in tpl["kinds"]:
            kv, t1 = rows_of_kind[f], time.time()
            n = kv["n"]
            if n > cap:
                raise SystemExit(f"{f}: {n} units over the statement cap {cap}; lower --per-kind")
            d, set_name = sdir / f, f"{st.iset.name}-{f}"
            got = st.stage(kv["rows"], d, salt_key, indices=range(base + lo, base + lo + n), set_name=set_name)
            (d / "inst.bin").rename(d / f"inst-{n}.bin")
            (d / "pub.bin").rename(d / f"pub-{n}.bin")
            (d / "tables").mkdir(exist_ok=True)
            if not (d / "circuit.txt").exists():
                (d / "circuit.txt").symlink_to("../circuit.txt")
            agree = all(np.array_equal(np.asarray(got["outs"][o], dtype=np.uint16)[:, :kv["outs"][o].shape[1]], kv["outs"][o])
                        for o in kv["outs"])
            ms = {"tag": f, "dir": d, "meta": st.meta, "pin": st.pin, "class": st.class_sha512, "set": set_name, "lo": lo, "n": n,
                  "cap": cap, "twin": {"agrees": agree, "instances": n, "call": kv["twin"]}, "ands": got["ands"],
                  "unit_nnz": got["unit_nnz"], "seconds": {"build": t_build if lo == 0 else 0.0, "stage": round(time.time() - t1, 1)},
                  "partition_file": out / "partition.json", "program_file": out / "program.json"}
            members.append({"name": f, "kind": f, "n": n, "base": base + lo, "defn": tpl["defn"], "st": ms,
                            "header": a0.public_header(d, n), "salt_key": salt_key, "stager": st, "rows": kv["rows"]})
            print(f"A11 staged {f}: {n} units of {uid} from {base + lo} (cap {cap}), {got['ands']} ANDs, k_log {st.meta.get('k_log')}, "
                  f"unit twin {agree}, Call twin {kv['twin']}, {ms['seconds']}", flush=True)
            lo += n
    summary["members"] = [{"name": m["name"], "template": P.template_id(m["defn"]), "n": m["n"], "base": m["base"], "set": m["st"]["set"],
                           "cap": m["st"]["cap"], "ands": m["st"]["ands"], "unit_nnz": m["st"]["unit_nnz"],
                           "k_log": m["st"]["meta"].get("k_log"), "pin": m["st"]["pin"], "class": m["st"]["class"],
                           "twin": m["st"]["twin"], "stage_s": m["st"]["seconds"]} for m in members]

    # one registration, before the draw
    want, roots, publics = {}, [], {}
    for m in members:
        fv = m["header"]["frame_v3"]
        for port, w in a6.frame_ports(m["st"]["set"], m["st"]["meta"], m["n"]).items():
            want[f"{m['name']}/{port}"] = w
            roots.append({"port": f"{m['name']}/{port}", "root": fv["roots"][port], "leaves": w["leaves"], "domain": fv["domain_ids"][port]})
        publics[m["name"]] = a0.sha512_file(m["st"]["dir"] / f"pub-{m['n']}.bin")
    leaf_layer = hashlib.sha512(canonical_json(publics)).hexdigest()
    window = {"kind": "stand-in", "of": WINDOW_OF, "link": LINK,
              "members": {m["name"]: {"template": P.template_id(m["defn"]), "units": [m["base"], m["base"] + m["n"]],
                                      "call": staged[m["kind"]]["name"], "step": staged[m["kind"]]["step"]} for m in members}}
    rec = R.record(program, query, law, a0.SCHEME, roots, window, leaf_layer=leaf_layer)
    (out / "registration.json").write_text(json.dumps(rec, indent=1, sort_keys=True))
    checks = {"registration": V.check(rec, program, query, law, a0.SCHEME, want, leaf_layer=leaf_layer).to_json()}
    for m in members:
        checks[f"statement/{m['name']}"] = a6.statement_checks(m, rec, want)
        ok = m["st"]["twin"]["agrees"] and m["st"]["twin"]["call"].get("agree") is not False
        checks[f"twin/{m['name']}"] = {"ok": ok, "codes": [] if ok else ["twin"], "detail": m["st"]["twin"]}
    receipt = R.receipt(rec, "one-stage-e2e verifier (a11.py)")
    (out / "receipt.json").write_text(json.dumps(receipt, indent=1, sort_keys=True))
    summary.update(population=n_all, partition=pd, registration=R.digest(rec), law=rec["law"],
                   checks_before_draw={c: v["ok"] for c, v in checks.items()})
    if not all(v["ok"] for v in checks.values()):
        summary["checks"] = checks
        (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
        print("A11 " + json.dumps({"refused_before_draw": summary["checks_before_draw"]}, default=str))
        return 1

    # the verifier's stratified draw, by its own Lean over its own program and partition object; each member serves its share
    p = a0.run([a.lean, "draw", "--program", str(out / "program.json"), "--partition", str(out / "partition.json"),
                "--stratified", str(k)], out / "draw.log")
    lean_draw = json.loads(next(x for x in p.stdout.splitlines() if x.startswith("DRAW ")).split(" ", 1)[1])
    (out / "draw.json").write_text(json.dumps(lean_draw, sort_keys=True))
    served, proved, instances, verdicts, statement, sessions, counts, session_roots = [], [], 0, {}, {}, [], {}, {}
    for j, m in enumerate(members):
        local = sorted(u - m["base"] for u in lean_draw["units"] if m["base"] <= u < m["base"] + m["n"])
        counts[m["name"]] = len(local)
        if not local:
            continue
        sdir = out / f"session-{m['name']}"
        sdir.mkdir(parents=True, exist_ok=True)
        (sdir / "draw.json").write_text(json.dumps({"law": "subset", "population": m["n"], "k": len(local), "units": local}, sort_keys=True))
        t = time.time()
        record, live = a0.session(a.bin, m["st"], m["n"], f"subset:{len(local)}", session_port(a.port, j), sdir, draw_file=sdir / "draw.json", zk=True)
        session_s = round(time.time() - t, 1)
        draw_t = (record or {}).get("unit_draw") or {}
        v = V.check_draw(draw_t, {"population": m["n"]}, {"law": "subset", "k": len(local)})
        served += [m["base"] + u for u in draw_t.get("units", [])]
        session_roots.update({f"{m['name']}/{port}": r for port, r in ((live or {}).get("roots") or {}).items()})
        proved += a0.proved_units(m["header"], live or {})
        instances += (live or {}).get("instances", 0)
        lean_u = a0.lean_draw_test(a.lean, [{"record": draw_t or None, "statement": (live or {}).get("unit_draw"),
                                             "instances": (live or {}).get("instances", 0)}], sdir, "u1-u3")
        verdicts[f"m0-verifier/{m['name']}"] = {"accepted": ((record or {}).get("verdict") or {}).get("accepted") is True
                                                and (live or {}).get("roots") == m["header"]["frame_v3"]["roots"] and v.ok,
                                                "why": ((record or {}).get("verdict") or {}).get("why"), "draw_check": v.to_json()}
        verdicts[f"lean-draw/{m['name']}"] = {"accepted": bool(lean_u) and all(c["ok"] for c in lean_u), "detail": lean_u}
        verdicts[f"lean-verify/{m['name']}"] = {**a0.lean_verify(a.lean, m["st"], m["n"], sdir, zk=True),
                                                "tool": "flock-verify verify --zk --archive --partition --program"}
        statement[m["name"]] = {"session_s": session_s, **{x: (live or {}).get(x) for x in ("circuit_sha512", "statement_digest", "sigma")},
                                "timing": {x: (live or {}).get(x) for x in ("e2e_s", "witness_s", "prove_total_s", "k_log", "units", "proof_bytes")},
                                "verify_s": ((record or {}).get("verdict") or {}).get("verify_s"),
                                "lean_verify_s": verdicts[f"lean-verify/{m['name']}"].get("verify_s"),
                                "lean_partition": verdicts[f"lean-verify/{m['name']}"].get("partition")}
        sessions += [a0.sha512_file(f) for f in sorted((sdir / "sessions").glob("*/session.json"))]
        lv, mv = verdicts[f"lean-verify/{m['name']}"], verdicts[f"m0-verifier/{m['name']}"]
        print(f"A11 session {m['name']}: {len(local)} units, M0 {mv['accepted']}, Lean {lv['accepted']} {lv.get('why') or ''}, "
              f"{session_s} s", flush=True)
    result = V.audit_record(rec, receipt, lean_draw, rec["law"], sorted(proved), instances, statement, verdicts, a.delta,
                            sessions=sessions, served=sorted(served), session_roots=session_roots)
    result["draw_counts"] = counts
    result["checks_before_draw"] = checks
    (out / "audit.json").write_text(json.dumps(result, indent=1, sort_keys=True, default=str))

    # negatives: a6's, each re-deciding one input altered
    neg: dict[str, dict] = {}
    fewer = R2.program(list(zip(defs, ns)), mixed=True)
    v = V.check(rec, fewer, query, law, a0.SCHEME, want, leaf_layer=leaf_layer)
    neg["registration-partition-mismatch (R2)"] = {"refused": not v.ok, "codes": list(v.codes)}
    if len(defs) > 1:
        v = V.check(rec, program, P.template_instances_query(defs[:-1]), law, a0.SCHEME, want, leaf_layer=leaf_layer)
        neg["registration-query-drops-a-template (R2)"] = {"refused": not v.ok, "codes": list(v.codes)}
    for name, other in (("registration-law-another-k (R1)", D.parse_law(f"stratified:{k + 1}")),
                        ("registration-law-subset-not-stratified (R1)", D.parse_law(f"subset:{k}"))):
        v = V.check(rec, program, query, other, a0.SCHEME, want, leaf_layer=leaf_layer)
        neg[name] = {"refused": not v.ok, "codes": list(v.codes)}
    v = V.check(rec, program, query, law, a0.SCHEME, want,
                leaf_layer=hashlib.sha512(canonical_json({**publics, members[0]["name"]: "00" * 64})).hexdigest())
    neg["public-file-not-the-registered-leaf-layer (R4)"] = {"refused": not v.ok, "codes": list(v.codes)}
    v = V.check_draw({**lean_draw, "units": lean_draw["units"][:-1] + [n_all]}, rec, rec["law"])
    neg["draw-outside-the-partition"] = {"refused": not v.ok, "codes": list(v.codes)}
    if proved:
        v = V.check_proved(lean_draw, proved[1:], len(proved) - 1)
        neg["proved-set-short-of-the-draw"] = {"refused": not v.ok, "codes": list(v.codes)}
        moved = sorted(proved)
        moved[0] = next(u for u in range(n_all) if u not in lean_draw["units"])
        v = V.check_proved(lean_draw, sorted(moved), instances)
        neg["proved-unit-not-drawn"] = {"refused": not v.ok, "codes": list(v.codes)}
    m0 = members[0]
    other_pd = P.digest(P.partition(fewer, query))
    c = a6.statement_checks({**m0, "st": {**m0["st"], "meta": {**m0["st"]["meta"], "partition": {"rule": P.FORMAT, "digest": other_pd}}}},
                            rec, want)
    neg["statement-binds-another-partition"] = {"refused": not c["ok"], "codes": c["codes"]}
    neg.update(a0.audit_negatives(result, rec["law"], a.delta, session_roots, served=sorted(served)))

    # a tampered copy of one kind's member: wrong outputs committed, registered and proved on its own; Lean must refuse it
    tm = None if a.tamper == "none" else (next((m for m in members if m["kind"] == a.tamper), None)
                                          or next((m for m in members if m["n"] >= 2), None))
    if tm is not None:
        tn = min(tm["n"], 4)
        tdef = tm["defn"]
        tprog = P.mixed_population_program([(tdef, tn)])
        tq = P.template_instances_query([tdef])
        tpart = P.partition(tprog, tq)
        tdir = out / "tampered"
        tdir.mkdir(exist_ok=True)
        (tdir / "partition.json").write_text(json.dumps(tpart, sort_keys=True))
        (tdir / "program.json").write_bytes(canonical_json(encode_program(tprog)))
        with K.no_cyclic_gc():
            tst = CS.UnitStager(tdef, f"unit/a11-tampered-{tm['kind']}", rows_of_kind[tm["kind"]]["rows_spec"],
                                K.out_rows(len(rows_of_kind[tm["kind"]]["outs"])), circuit=tdir / "circuit.txt",
                                bind={"partition": P.digest(tpart), "program": tpart["program"]})
        d, set_name = tdir / "set", f"{tst.iset.name}-tampered"
        stage_tampered(tst, {r: v[:tn] for r, v in tm["rows"].items()}, d, tm["salt_key"], range(tn), set_name)
        (d / "inst.bin").rename(d / f"inst-{tn}.bin")
        (d / "pub.bin").rename(d / f"pub-{tn}.bin")
        (d / "tables").mkdir(exist_ok=True)
        (d / "circuit.txt").symlink_to("../circuit.txt")
        tst_d = {"dir": d, "meta": tst.meta, "pin": tst.pin, "partition_file": tdir / "partition.json", "program_file": tdir / "program.json"}
        theader = a0.public_header(d, tn)
        tlaw = D.parse_law("subset:2")
        twant = a6.frame_ports(set_name, tst.meta, tn)
        trec = R.record(tprog, tq, tlaw, a0.SCHEME, a0.registered_roots(theader, twant),
                        {"kind": "stand-in", "of": f"tampered copy of {tm['kind']}'s first {tn} units: bit 0 of every instance's "
                                                   "first output word flipped before commitment"},
                        leaf_layer=a0.sha512_file(d / f"pub-{tn}.bin"))
        tcheck = V.check(trec, tprog, tq, tlaw, a0.SCHEME, twant, leaf_layer=a0.sha512_file(d / f"pub-{tn}.bin"))
        trecord, tlive = a0.session(a.bin, tst_d, tn, "subset:2", session_port(a.port, len(members) + 1), tdir / "session", zk=True)
        tlean = a0.lean_verify(a.lean, tst_d, tn, tdir / "session", zk=True)
        tm0 = ((trecord or {}).get("verdict") or {})
        neg[f"wrong-output-word ({tm['kind']})"] = {
            "refused": a0.lean_refused(tlean) and tm0.get("accepted") is not True, "registration_ok": tcheck.ok,
            "m0_verifier": tm0.get("why"), "m0_accepted": tm0.get("accepted"), "lean": tlean.get("why"),
            "lean_refused": a0.lean_refused(tlean), "prover": (tlive or {}).get("why"), "draw": (trecord or {}).get("unit_draw")}
    (out / "negatives.json").write_text(json.dumps(neg, indent=1, sort_keys=True, default=str))

    summary["audit"] = {x: result.get(x) for x in ("accepted", "complete", "wrong_units", "profile")}
    summary["audit"]["verdicts"] = {x: y.get("accepted") for x, y in verdicts.items()}
    summary["audit"]["why"] = {x: y.get("why") for x, y in verdicts.items() if y.get("accepted") is not True}
    summary.update(draw=lean_draw["units"], draw_counts=counts, served_is_lean_draw=sorted(served) == sorted(lean_draw["units"]),
                   timing=statement, negatives={x: y["refused"] for x, y in neg.items()})
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
    ok = bool(result.get("accepted")) and all(summary["negatives"].values())
    print("A11 " + json.dumps({x: summary[x] for x in ("population", "draw_counts", "audit", "negatives")}, sort_keys=True, default=str))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
