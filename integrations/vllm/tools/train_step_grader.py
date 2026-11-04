"""The grader: the products kind's slowdown ρ of a run's training step, read off the step's standard Program (the Boolean
circuit a lane hands over, Daniel 3 Nov) by training's slowdown code's bound.

A lane hands over one directory (`write_handover`), and the grader refuses anything else, each refusal with its fix:

    handover.json      {"format": "grader-handover/v1",
                        "run": {"id", "checkpoints": [{"checkpoint", "steps"}]},   the run record: each checkpoint's steps
                        "registered": [...],                                       theta's input names, as registered
                        "frontend": {"commit"},                                    the commit step_program ran at
                        "step": {"program", "digest", "annotations", "batch"},
                        "structure": {"program", "digest", "annotations", "batch"}}
    step.json.gz       the run's step: `step_program(model, loss_fn, optimizer, batch)`'s Program at the run's batch shape
                       (`codec.encode_program`), named by its digest (`codec.program_digest`, which leaves the annotations
                       out, so `annotations` is the sha512 of their canonical JSON)
    structure.json.gz  the same step at another batch shape, which says which dims are the tokens'

The grader reads every input of the bound off the step Program. The products are its matrix-product Definition uses (an
mm or bmm occurrence, lowered to `LinearRef_v1` cells), each priced at its census class's Boolean gates a cell; the
operands' widths, what a block reads of them beyond theta, the chains and their classes come from the Program's dataflow:
its occurrences and the views they read (`_Flow`). On that dataflow it applies the rules of training's slowdown code's
capture (its `capture.products`, which read the same rules off an op graph), one for one: the rates and chains, the
whole-making system's reach and P-match's check, the gathered rows and the matrix products written with elementwise
ops. The bound's math is the slowdown code's own: its formalism (`products_of_trace`, `fresh_products`) and io_bound.
|C| is the Program's census at the Boolean prices of the Definitions it calls (`circuit_size`).

Out (`grade`, "grader-graded/v1"), per checkpoint: |C|, and ρ at the console's default block and the served block (and
any `--policy c,k`): the largest valid lower bound the grader computes, floored at 1, its binding kind and its premises.
The headline assumes that blocks hold whole multiply cells (P-cell; Daniel, 3 Nov) and that every value a block reads
costs its full width (P-width); the figure without P-cell is beside it as a sensitivity. A pure function of its inputs:
no network, nothing published.

What reads today: the hand-over and its refusals, and the circuit's products (`Circuit`: each multiply-add cell's MACs,
outputs, operand dependence on θ′, word widths and phase, by occurrence) with |C|. The rules' passes and `grade` don't.
"""
from __future__ import annotations

import gzip
import hashlib
import importlib
import json
import math
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np

FORMAT = "grader-handover/v1"
GRADED = "grader-graded/v1"
VERSION = "1"
#: the capture whose rules `_Flow` reads on a Program (the slowdown code's `capture.products.CAPTURE`)
CAPTURE = "4.12.9"
THE_CODE = "training's slowdown code"
RETIRED = "derive the step with step_program; the op-graph capture is retired"
#: the matrix products the grader reads, and the cell Definition each must be lowered to
PRODUCTS = ("mm", "bmm")
CELL = "LinearRef_v1"
#: the premises the headline and its sensitivity rest on (the slowdown code's formalism.PREMISES names them)
HEADLINE = ("P-cell", "P-width", "P-chain", "P-match")
SENSITIVITY = ("P-width", "P-chain", "P-match")


class Refused(ValueError):
    """What the grader won't grade, with how to fix it."""


# -- the hand-over -------------------------------------------------------------------------------------------------------

def _annotations_digest(desc: dict) -> str:
    from verity.ml.circuits.codec import canonical_json

    return hashlib.sha512(canonical_json(desc.get("annotations", {}))).hexdigest()


def write_handover(out: Path, step, structure, *, run: str, checkpoints: list[tuple[str, int]], frontend: str,
                   step_batch: tuple[int, ...], structure_batch: tuple[int, ...]) -> Path:
    """Write a lane's hand-over to directory `out`: the run's step Program (`step_program` at the run's batch shape
    `step_batch`), the same step at `structure_batch`, the run record (its id and each checkpoint's steps), theta as
    registered (the step Program's theta) and the frontend commit both were derived at. Returns handover.json's path."""
    from verity.ml.circuits.codec import encode_program, program_digest

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    doc = {"format": FORMAT, "run": {"id": run, "checkpoints": [{"checkpoint": str(c), "steps": int(s)} for c, s in checkpoints]},
           "registered": list(step.theta), "frontend": {"commit": frontend}}
    for key, prog, batch in (("step", step, step_batch), ("structure", structure, structure_batch)):
        desc = encode_program(prog)
        name = f"{key}.json.gz"
        (out / name).write_bytes(gzip.compress(json.dumps(desc, separators=(",", ":")).encode(), mtime=0))
        doc[key] = {"program": name, "digest": program_digest(desc), "annotations": _annotations_digest(desc),
                    "batch": [int(x) for x in batch]}
    path = out / "handover.json"
    path.write_text(json.dumps(doc, indent=1) + "\n")
    return path


@dataclass
class Handover:
    """A hand-over the grader accepted: its record and both Programs, decoded."""
    path: Path
    doc: dict
    step: object
    structure: object


def _need(doc: dict, key: str, kind, where: str):
    v = doc.get(key)
    if not isinstance(v, kind) or (isinstance(v, (str, list, dict)) and not v):
        raise Refused(f"the hand-over's {where} has no {key!r}: write it with train_step_grader.write_handover "
                      f"from the run's own step Program ({FORMAT})")
    return v


def _batch_ok(step: list, structure: list) -> None:
    fix = ("derive the structure Program at a batch of B' rows of L' tokens with B' != B, L' != L, B'L' != BL and "
           "B'(L'-1) != B(L-1), both with B >= 2 and L >= 3")
    for b in (step, structure):
        if len(b) != 2 or not all(isinstance(x, int) and not isinstance(x, bool) for x in b) or b[0] < 2 or b[1] < 3:
            raise Refused(f"batch shape {b} isn't two sizes [B, L] with B >= 2 and L >= 3: {fix}")
    (B, L), (B2, L2) = step, structure
    if B == B2 or L == L2 or B * L == B2 * L2 or B * (L - 1) == B2 * (L2 - 1):
        raise Refused(f"the step's batch {step} and the structure's {structure} share a token-indexed size, so the "
                      f"grader can't tell which dims are the tokens': {fix}")


def load_handover(path: Path) -> Handover:
    """The hand-over at `path` (its directory or its handover.json), checked: its format and fields, each Program file's
    digest and annotations against the record, both Programs a frontend step (`step_program`'s annotations), theta as
    registered, and two batch shapes that tell the token dims apart. Raises `Refused` with the fix."""
    from verity.ml.circuits.codec import decode_program, program_digest
    from verity.ml.circuits.defs import REGISTRY

    path = Path(path)
    path = path / "handover.json" if path.is_dir() else path
    try:
        doc = json.loads(path.read_text())
    except (OSError, ValueError) as e:
        raise Refused(f"cannot read a hand-over at {path} ({type(e).__name__}: {e}): hand over the directory "
                      f"train_step_grader.write_handover wrote") from e
    if not isinstance(doc, dict) or doc.get("format") != FORMAT:
        got = doc.get("format") if isinstance(doc, dict) else type(doc).__name__
        raise Refused(f"{path} isn't a grader hand-over (format {got!r}, not {FORMAT!r}); {RETIRED}, and write the "
                      f"hand-over with train_step_grader.write_handover")
    run = _need(doc, "run", dict, "record")
    _need(run, "id", str, "run")
    cps = _need(run, "checkpoints", list, "run")
    for cp in cps:
        if not isinstance(cp, dict) or not isinstance(cp.get("steps"), int) or cp["steps"] < 1 or "checkpoint" not in cp:
            raise Refused(f"checkpoint {cp!r} isn't {{'checkpoint', 'steps'}} with at least one step: list each "
                          f"checkpoint the run reached with the steps it took to it")
    registered = _need(doc, "registered", list, "record")
    _need(_need(doc, "frontend", dict, "record"), "commit", str, "frontend")
    decoded = {}
    for key in ("step", "structure"):
        rec = _need(doc, key, dict, "record")
        for f in ("program", "digest", "annotations"):
            _need(rec, f, str, key)
        _need(rec, "batch", list, key)
        file = path.parent / rec["program"]
        try:
            raw = file.read_bytes()
            desc = json.loads(gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw)
        except (OSError, ValueError, EOFError) as e:
            raise Refused(f"cannot read the {key} Program {file} ({type(e).__name__}: {e}): hand over the file "
                          f"write_handover wrote, unmodified") from e
        if program_digest(desc) != rec["digest"]:
            raise Refused(f"the {key} Program's digest is {program_digest(desc)[:16]}..., the hand-over says "
                          f"{rec['digest'][:16]}...: hand over the Program the run record names, unmodified")
        if _annotations_digest(desc) != rec["annotations"]:
            raise Refused(f"the {key} Program's annotations aren't the ones the hand-over names: hand over the "
                          f"Program step_program wrote, with its annotations, unmodified")
        ann = desc.get("annotations", {})
        if "train_step" not in ann or "correspondence" not in ann:
            raise Refused(f"the {key} Program isn't a training step the frontend derived (no train_step or "
                          f"correspondence annotation): {RETIRED}")
        try:
            decoded[key] = decode_program(desc, REGISTRY)
        except Exception as e:
            raise Refused(f"the {key} Program doesn't decode here ({type(e).__name__}: {str(e)[:300]}): grade it at a "
                          f"commit that has every Definition it calls, registered before the grader reads it") from e
    names = [decoded["step"].root.body.nodes[i].name for i in sorted(decoded["step"].root.theta)]
    if sorted(registered) != sorted(names):
        extra, missing = sorted(set(registered) - set(names)), sorted(set(names) - set(registered))
        raise Refused(f"theta as registered isn't the step Program's theta (registered only: {extra[:5]}, the "
                      f"Program's only: {missing[:5]}): register the step Program's theta, or derive the step from "
                      f"the model whose weights were registered")
    _batch_ok(doc["step"]["batch"], doc["structure"]["batch"])
    return Handover(path, doc, decoded["step"], decoded["structure"])


# -- training's slowdown code --------------------------------------------------------------------------------------------

def load_slowdown_code(directory: Path, commit: str) -> SimpleNamespace:
    """Training's slowdown code from `directory` (the one that holds its package), at `commit` (checked where the
    directory is a git checkout): the modules the grader calls, its version and its commit."""
    d = Path(directory).resolve()
    if not (d / "te_core" / "io_bound.py").is_file():
        raise Refused(f"{d} doesn't hold {THE_CODE}: pass the directory its package sits in (benchmarks/covert_witness "
                      f"on its branch)")
    head = subprocess.run(["git", "-C", str(d), "rev-parse", "HEAD"], capture_output=True, text=True)
    if head.returncode == 0 and not head.stdout.strip().startswith(commit):
        raise Refused(f"{THE_CODE} at {d} is at {head.stdout.strip()[:12]}, not {commit}: check out {commit}, or "
                      f"name the commit it is at")
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
    pkg = importlib.import_module("te_core")
    mods = {m: importlib.import_module(f"te_core.{p}") for m, p in (
        ("io_bound", "io_bound"), ("formalism", "formalism"), ("products", "capture.products"),
        ("structure", "capture.structure"), ("panels", "panels"), ("policy", "policy"))}
    if mods["products"].CAPTURE != CAPTURE:
        raise Refused(f"the grader reads capture {CAPTURE}'s rules on the Program, and {THE_CODE} at {commit} "
                      f"captures with {mods['products'].CAPTURE}'s: update the grader's rules to its capture first")
    return SimpleNamespace(**mods, version=pkg.VERSION, commit=commit, directory=str(d))


# -- one Program: units, values and views --------------------------------------------------------------------------------

@dataclass
class View:
    """What occurrence `occ` reads through its fx input `j`: the fx node `fx` (a view of one or more values, or a value
    itself), at `shape` and `width` bits an entry. `parts` holds each value it reads with the distinct entries it reads
    of it; `consts` the distinct constants it reads. `kind`: `walk` (every entry of one value once: a reshape, a
    permutation), `take` (some entries of one value: a slice), `repeat` (entries repeated: a broadcast), `pad` (one
    value and constants), `concat` (several values side by side), `const` (constants alone), `none` (read for its
    shape only)."""
    occ: int
    j: int
    arg: int | None
    fx: str
    source: str
    shape: tuple
    width: int
    numel: int
    parts: dict = field(default_factory=dict)
    consts: int = 0
    kind: str = "none"


def _dtype_width(dtype: str | None) -> int:
    import torch

    t = getattr(torch, dtype or "", None)
    if not isinstance(t, torch.dtype):
        return 64
    return 1 if t == torch.bool else t.itemsize * 8


def _leaf_parts(refs):
    from verity.ml.circuits.refs import Concat

    if isinstance(refs, Concat):
        for p in refs.parts:
            yield from _leaf_parts(p)
    else:
        yield refs


def _strided(r, add) -> None:
    from verity.ml.circuits.intervals import IntervalLimit, strided_intervals

    try:
        strided_intervals(r, add)
    except IntervalLimit:
        j = np.arange(r.count, dtype=np.int64)
        leaves = np.unique(r.base + (j // r.inner) * r.outer_stride + (j % r.inner) * r.inner_stride)
        cut = np.flatnonzero(np.diff(leaves) != 1) + 1
        for run in np.split(leaves, cut):
            add(int(run[0]), int(run[-1]) + 1)


def _refs_by_node(refs, out: dict, seen: set) -> None:
    """Accumulate the root leaves `refs` names into `out` ({node: [(lo, hi)]}, unnormalized)."""
    from verity.ml.circuits.refs import Affine, Explicit, Strided

    for p in _leaf_parts(refs):
        if isinstance(p, Affine):
            if p.space != "n":
                raise Refused(f"a root reference outside the root's nodes ({p!r}): {RETIRED}")
            if p.count:
                out.setdefault(p.idx, []).append((p.base, p.base + p.count))
        elif isinstance(p, Strided):
            if p.space != "n":
                raise Refused(f"a root reference outside the root's nodes ({p!r}): {RETIRED}")
            key = (p.idx, p.base, p.inner, p.outer_stride, p.inner_stride, p.count)
            if key not in seen:
                seen.add(key)
                lst = out.setdefault(p.idx, [])
                _strided(p, lambda lo, hi, lst=lst: lst.append((lo, hi)))
        elif isinstance(p, Explicit):
            for s, i, leaf in p.items:
                if s != "n":
                    raise Refused(f"a root reference outside the root's nodes ({s}, {i}): {RETIRED}")
                out.setdefault(i, []).append((leaf, leaf + 1))
        else:
            for s, i, leaf in p:
                out.setdefault(i, []).append((leaf, leaf + 1))


def _ref_nodes(refs, logs: dict | None = None) -> set:
    """The root nodes `refs` reads. With `logs`, a `Concat` cut from a `PartLog` (`prefix`: a causal attention's rows, each
    call's the call before's plus one) reads the log's parts once over all its cuts (`_log_nodes`)."""
    from verity.ml.circuits.refs import Affine, Concat, Explicit, Strided

    if isinstance(refs, Concat):
        got, start = set(), 0
        if logs is not None and refs.prefix is not None:
            log, start = refs.prefix
            got = _log_nodes(log, start, logs)
        for i in range(start, len(refs.parts)):
            got |= _ref_nodes(refs.parts[i], logs)
        return got
    if isinstance(refs, (Affine, Strided)):
        return {refs.idx} if refs.count else set()
    if isinstance(refs, Explicit):
        return {i for _, i, _ in refs.items}
    return {i for _, i, _ in refs}


def _log_nodes(log, j: int, logs: dict) -> set:
    """The root nodes `log.parts[:j]` reads, `logs` holding each log's parts read so far and the first part reading each node."""
    rec = logs.setdefault(id(log), [0, {}])
    while rec[0] < j:
        for x in _ref_nodes(log.parts[rec[0]], logs):
            rec[1].setdefault(x, rec[0])
        rec[0] += 1
    return {x for x, i in rec[1].items() if i < j}


def _distinct(ivs: list) -> int:
    from verity.ml.circuits.intervals import norm, total

    return total(norm(ivs))


class _Program:
    """One step Program as units, values and views. A unit is an input node (index i) or an occurrence of the step's
    graph (index n_in + o, its calls the root nodes `calls` assigns it); a value is one output (unit, i); a view is what
    an occurrence reads through one fx input, its entries' producers found from the calls' references (`View`, read by
    `read_views`). `reads` holds the root nodes each root node reads, and `ret` the root's return (the loss first)."""

    def __init__(self, decoded, role: str) -> None:
        import torch
        from verity.ml.circuits.program import is_input

        self.role = role
        root = decoded.root
        nodes = self.nodes = root.body.nodes
        ann = decoded.annotations
        corr = ann["correspondence"]
        self.layout = ann["train_step"]
        n_in = self.n_in = sum(1 for nd in nodes if is_input(nd.fn))
        occs, calls = self.occs, self.calls = corr["occurrences"], corr["calls"]
        if len(calls) != len(nodes) - n_in:
            raise Refused(f"the {role} Program's correspondence has {len(calls)} calls for its {len(nodes) - n_in} "
                          f"root calls: {RETIRED}")
        self.names = [nd.name for nd in nodes[:n_in]]
        self.theta = sorted(root.theta)
        self.ops, self.targets = [], []
        for o in occs:
            if o["written"]:
                raise Refused(f"occurrence {o['node']} ({o['op']}) writes an input in place: {RETIRED} "
                              f"(it functionalizes the step)")
            ns, name, overload = (o["op"].split(".") + ["", ""])[:3]
            t = getattr(getattr(getattr(torch.ops, ns, None), name, None), overload, None)
            if not isinstance(t, torch._ops.OpOverload):
                raise Refused(f"occurrence {o['node']}'s op {o['op']} isn't a torch op here: grade with the torch "
                              f"the frontend derived the step with")
            self.targets.append(t)
            self.ops.append(t.overloadpacket.__name__)
        # each root node's unit and value (None: a call of no occurrence; port None: a call returning other than one
        # port); constants (no operand, or constants alone), which a statement hardwires
        unit: list = list(range(n_in))
        value: list = [(i, 0) for i in range(n_in)]
        for c in calls:
            u = None if c["occ"] is None else n_in + c["occ"]
            unit.append(u)
            value.append(None if u is None else (u, c["returns"][0]["output"] if len(c["returns"]) == 1 else None))
        self.ret = root.body.ret
        logs: dict = {}
        self.reads = [frozenset()] * n_in + [frozenset().union(*(_ref_nodes(a.refs, logs) for a in nd.args))
                                             for nd in nodes[n_in:]]
        const = [False] * len(nodes)
        for i in range(n_in, len(nodes)):
            const[i] = all(const[x] for x in self.reads[i])
        self.unit, self.value, self.const = unit, value, const
        self.out_shapes = [[tuple(x["shape"]) if x else None for x in o["outputs"]] for o in occs]
        self.out_widths = [[_dtype_width(x["dtype"]) if x else 0 for x in o["outputs"]] for o in occs]
        self.leaves = [nodes[i].out.leaves for i in range(n_in)]
        self.in_widths = [nodes[i].fn.ret.w for i in range(n_in)]
        self.views = [[View(o, j, x["arg"], x["node"], x["source"], tuple(x["view"]["shape"]),
                            _dtype_width(x["view"]["dtype"]), math.prod(x["view"]["shape"]))
                       for j, x in enumerate(occ["inputs"])] for o, occ in enumerate(occs)]
        self.unplaced = 0
        self._viewed = False

    def read_views(self) -> None:
        """Read each view's producers and constants (`_read_views`), once."""
        if not self._viewed:
            self._read_views()
            self._viewed = True

    def size(self, v: tuple) -> int:
        """Value v's entries."""
        u, i = v
        if u < self.n_in:
            return self.leaves[u]
        s = self.out_shapes[u - self.n_in][i] if i is not None else None
        return math.prod(s) if s is not None else 0

    def _read_views(self) -> None:
        """Each view's producers and constants, from the references of the calls that read it. A call's argument names
        the fx input it reads (`input`), or not (null): a transposed weight a cell reads in its own layout, a constant
        written into the call. An unnamed argument goes to the input the occurrence's other calls read at the same
        argument of the same Definition, else to its one input no call names (a constant to its one constant input);
        an unnamed argument of constants with no such input is a constant the call hardwires."""
        n_in, nodes = self.n_in, self.nodes
        by_occ: dict = {}
        for k, c in enumerate(self.calls):
            if c["occ"] is not None:
                by_occ.setdefault(c["occ"], []).append(k)
        for o, ks in by_occ.items():
            own = n_in + o
            reads: dict = {}
            pending: list = []
            at: dict = {}
            for k in ks:
                fid = nodes[n_in + k].fn.id
                for a, (rec, arg) in enumerate(zip(self.calls[k]["args"], nodes[n_in + k].args)):
                    nodes_read = _ref_nodes(arg.refs)
                    if all(self.unit[x] == own and not self.const[x] for x in nodes_read):
                        continue
                    if rec["input"] is not None:
                        reads.setdefault(rec["input"], []).append(arg.refs)
                        at.setdefault((fid, a), set()).add(rec["input"])
                    else:
                        pending.append((fid, a, arg.refs, all(self.const[x] for x in nodes_read)))
            named = set(reads)
            views = self.views[o]
            for fid, a, refs, consts in pending:
                same = at.get((fid, a), set())
                if len(same) == 1:
                    j = next(iter(same))
                else:
                    free = [v.j for v in views if v.j not in named and (v.source in ("other", "constant")) == consts]
                    if len(free) != 1:
                        if consts:
                            continue
                        raise Refused(f"occurrence {self.occs[o]['node']} ({self.occs[o]['op']}) reads a value through "
                                      f"an argument the correspondence doesn't name, and {len(free)} of its inputs could "
                                      f"be it: {RETIRED}")
                    j = free[0]
                reads.setdefault(j, []).append(refs)
            for j, refs_list in reads.items():
                self._view_parts(views[j], refs_list)

    def _view_parts(self, view: View, refs_list: list) -> None:
        own = self.n_in + view.occ
        by_node: dict = {}
        seen: set = set()
        for refs in refs_list:
            _refs_by_node(refs, by_node, seen)
        parts: dict = {}
        consts = 0
        for x, ivs in by_node.items():
            if self.const[x]:
                consts += _distinct(ivs)
                continue
            if self.unit[x] == own:
                continue
            v = self.value[x]
            if v is None:
                raise Refused(f"occurrence {self.occs[view.occ]['node']} reads root call {x - self.n_in}, which the "
                              f"correspondence ties to no occurrence: {RETIRED}")
            if v[1] is None:
                outs = [i for i, s in enumerate(self.out_shapes[v[0] - self.n_in]) if s is not None]
                self.unplaced += 1
                v = (v[0], outs[0] if outs else 0)
            parts[v] = parts.get(v, 0) + _distinct(ivs)
        view.parts, view.consts = parts, consts
        V, D = view.numel, sum(parts.values())
        if not parts:
            view.kind = "const" if consts else "none"
        elif len(parts) == 1:
            (v, d), = parts.items()
            if D >= V:
                view.kind = "walk" if d >= self.size(v) else "take"
            else:
                view.kind = "pad" if consts else "repeat"
        else:
            view.kind = "concat" if D >= V or consts else "repeat"


# -- the step's dataflow, both Programs ----------------------------------------------------------------------------------

def _same_step(a: _Program, b: _Program) -> None:
    fix = ("derive both with step_program from the same model, loss and optimizer, at two batch shapes, at the same "
           "frontend commit")
    if a.names != b.names:
        raise Refused(f"the structure Program's inputs aren't the step's: {fix}")
    for i in a.theta:
        if a.leaves[i] != b.leaves[i]:
            raise Refused(f"theta input {a.names[i]} has {a.leaves[i]} entries in the step and {b.leaves[i]} in the "
                          f"structure Program: {fix}")
    if len(a.occs) != len(b.occs):
        raise Refused(f"the step has {len(a.occs)} occurrences and the structure Program {len(b.occs)}: {fix}")
    for x, y in zip(a.occs, b.occs):
        if (x["node"], x["op"], [i["node"] for i in x["inputs"]]) != (y["node"], y["op"], [i["node"] for i in y["inputs"]]):
            raise Refused(f"occurrence {x['node']} ({x['op']}) of the step is {y['node']} ({y['op']}) in the structure "
                          f"Program: {fix}")


class _Flow:
    """The step's dataflow as the slowdown code's capture reads a captured step (`capture.products`' `_Values`): which
    values depend on θ′ (`dep`: the start's weights and the optimizer's state and scalars, and everything computed from
    them), which are token-indexed (`tok`: their shape differs at the structure Program's batch shape), each
    occurrence's phase, and its matrix products. Views are values of their own, as the capture's view ops are."""

    def __init__(self, step: _Program, structure: _Program, code) -> None:
        _same_step(step, structure)
        step.read_views()
        structure.read_views()
        self.p, self.code = step, code
        P = len(step.layout["params"])
        n_in = step.n_in
        seeds = set(step.layout["inputs"][:3 * P + len(step.layout["scalars"])])
        self.batch = step.names.index("batch") if "batch" in step.names else None
        self.units = list(range(n_in + len(step.occs)))
        dep = [step.names[i] in seeds for i in range(n_in)]
        for o in range(len(step.occs)):
            dep.append(any(self.view_dep(v, dep) for v in step.views[o]))
        self.dep = dep
        # token-indexed values, views and entries read
        self.tok: dict = {(i, 0): step.leaves[i] != structure.leaves[i] for i in range(n_in)}
        for o in range(len(step.occs)):
            for i, (s, t) in enumerate(zip(step.out_shapes[o], structure.out_shapes[o])):
                self.tok[(n_in + o, i)] = s != t
        self.vtok: dict = {}
        self.dtok: dict = {}
        for o in range(len(step.occs)):
            for v, w in zip(step.views[o], structure.views[o]):
                self.vtok[(o, v.j)] = v.shape != w.shape
                for p, d in v.parts.items():
                    self.dtok[(o, v.j, p)] = w.parts.get(p) != d
        circuit = Circuit(step)
        self.phase = {n_in + o: circuit.occ_phase(o) for o in range(len(step.occs))}
        # matrix products: dims, token dims, operand views
        self.gemm: dict = {}
        for o, name in enumerate(step.ops):
            if name in PRODUCTS:
                self.gemm[n_in + o] = self._gemm(o, name, structure)
            elif any(s in name for s in code.structure._GEMM_LIKE) and name not in code.structure._NOT_GEMM:
                raise Refused(f"occurrence {step.occs[o]['node']} ({step.occs[o]['op']}) computes matrix products the "
                              f"grader doesn't read (it reads {', '.join(PRODUCTS)}): derive the step with a rule that "
                              f"lowers it to mm or bmm")

    def view_dep(self, v: View, dep=None) -> bool:
        dep = self.dep if dep is None else dep
        return any(dep[p[0]] for p in v.parts)

    def _gemm(self, o: int, name: str, structure: _Program) -> dict:
        p = self.p
        A, B = self.at(o, 0), self.at(o, 1)
        dims, dims2 = _dims(p, o), _dims(structure, o)
        Bt, M, K, N = (dims[d] for d in ("batch", "m", "k", "n"))
        cells = [k for k, c in enumerate(p.calls) if c["occ"] == o]
        fns = {p.nodes[p.n_in + k].fn for k in cells}
        bad = [f.id for f in fns if getattr(f, "base_id", f.id) != CELL or f.bindings.get("K") != K
               or f.bindings.get("N") != N]
        if bad or sum(p.nodes[p.n_in + k].n for k in cells) != Bt * M:
            raise Refused(f"occurrence {p.occs[o]['node']} ({name}, {Bt}x{M}x{K}x{N}) isn't lowered to {CELL} cells "
                          f"of K={K}, N={N} over its {Bt * M} rows ({bad[:2] or 'other row counts'}): derive the step "
                          f"with step_program's training rules")
        return {"dims": dims, "token_dims": sorted(d for d in dims if dims[d] != dims2[d]), "A": A, "B": B,
                "cells": cells}

    def at(self, o: int, arg: int) -> View | None:
        """Occurrence o's view at fx position `arg` (its op's Python argument `arg`, where every earlier one is a
        tensor, as `self` and the operands of the ops this is asked for are)."""
        return next((v for v in self.p.views[o] if v.arg == arg), None)

    def outs(self, u: int) -> list:
        p = self.p
        if u < p.n_in:
            return [(u, 0)]
        return [(u, i) for i, s in enumerate(p.out_shapes[u - p.n_in]) if s is not None]

    def numel(self, u: int) -> int:
        return sum(self.p.size(v) for v in self.outs(u))

    def width(self, u: int) -> int:
        p = self.p
        if u < p.n_in:
            return p.in_widths[u]
        return next((w for w in p.out_widths[u - p.n_in] if w), 64)

    def gather_by_token(self, o: int) -> bool:
        """The capture's gather by token data: a gather (`_GATHERS`) whose Python argument 1 is a tensor that doesn't
        depend on θ′ (an embedding's ids)."""
        p, code = self.p, self.code
        if p.ops[o] not in code.structure._GATHERS:
            return False
        args = p.targets[o]._schema.arguments
        import torch

        if len(args) < 2 or not isinstance(args[1].type, torch._C.TensorType):
            return False
        v = self.at(o, 1)
        return v is not None and not self.view_dep(v)

    def elems(self, v: View) -> int:
        """The capture's `elems`: a view's entries, at most its storage's (one value's, for a view of it)."""
        if v.kind in ("walk", "take", "repeat") and len(v.parts) == 1:
            return min(v.numel, self.p.size(next(iter(v.parts))))
        return v.numel

    def root(self, x, scale: bool):
        """io_bound's `_root` on the Program: back through views of one value and the occurrences of the capture's walk
        ops (and, with `scale`, scalings by a constant), to a unit or a view of several values."""
        p, code = self.p, self.code
        for _ in range(100_000):
            if isinstance(x, View):
                if x.kind in ("walk", "take", "repeat") and len(x.parts) == 1 and not x.consts:
                    x = next(iter(x.parts))[0]
                    continue
                return ("view", x.occ, x.j)
            u = x
            if u < p.n_in:
                return ("unit", u)
            o = u - p.n_in
            name, views = p.ops[o], p.views[o]
            first = self.at(o, 0)
            if first is not None and (name in code.structure._WALK or scale and name in code.structure._SCALE
                                      and len(views) == 1):
                x = first
                continue
            return ("unit", u)
        return ("unit", x)

    def passed_width(self, v: View) -> int:
        """io_bound's `_passed_width` on the Program: back through views that read every entry of one value once and
        occurrences with one same-size argument that depends on θ′ (not a concatenation)."""
        p, code = self.p, self.code
        b = v.width
        x = v
        for _ in range(100_000):
            if isinstance(x, View):
                if x.kind != "walk" or not self.view_dep(x):
                    break
                u = next(iter(x.parts))[0]
                b = min(b, self.width(u))
                x = u
                continue
            if x < p.n_in:
                break
            o = x - p.n_in
            same = [w for w in p.views[o] if w.numel == self.numel(x) and self.view_dep(w)]
            if len(same) != 1 or p.ops[o] in code.io_bound._CONCAT:
                break
            x = same[0]
            b = min(b, x.width)
        return b


def circuit_size(program) -> dict:
    """|C|: the step Program's census at the Boolean prices of the primitives it calls (`train_step_census.priced`): its
    gates and ANDs by root callee class, and the word gates of the primitives with no Boolean version, which it leaves
    out."""
    from train_step_census import boolean_prices, priced
    from verity.ml.circuits.census import census
    from verity_vllm.program.frontend.train_step import _primitives

    return priced(census(program), boolean_prices(_primitives(program)))


def _dims(p: _Program, o: int) -> dict:
    """An mm or bmm occurrence's (batch, m, k, n), from its operands' views."""
    (A,), (B,) = ([v for v in p.views[o] if v.arg == a] for a in (0, 1))
    if p.ops[o] == "mm":
        (M, K), (_, N) = A.shape, B.shape
        return {"batch": 1, "m": M, "k": K, "n": N}
    (Bt, M, K), (_, _, N) = A.shape, B.shape
    return {"batch": Bt, "m": M, "k": K, "n": N}


# -- the step's multiply-add cells, on its circuit -----------------------------------------------------------------------

#: the multiply-add cells a step's circuit computes products with, each instance's MACs and outputs from its bindings: a
#: row of a matrix product (`LinearRef_v1{K, N}`: N dots of K, what every mm and bmm is lowered to) and one dot
#: (`DotRef_v1{K}`, what other ops' sums of products are lowered to)
MAC_CELLS = {CELL: lambda b: (b["K"] * b["N"], b["N"]), "DotRef_v1": lambda b: (b["K"], 1)}


def _instances(nd) -> int:
    return nd.n if nd.form in ("batch", "scan") else 1


def _first_node(refs) -> int:
    from verity.ml.circuits.refs import Affine, Explicit, Strided

    p = next(iter(_leaf_parts(refs)))
    return p.idx if isinstance(p, (Affine, Strided)) else p.items[0][1] if isinstance(p, Explicit) else p[0][1]


class Circuit:
    """A step Program's circuit as the products bound reads it, root node by root node: which nodes depend on θ′ (`dep`:
    the start's weights, the optimizer's state and scalars, and every node computed from them), which the loss is
    computed from (`forward`), which read the optimizer's state (`opt`), and each multiply-add cell (`MAC_CELLS`): its
    occurrence, instances, MACs and outputs, whether each of its two operands depends on θ′, and the operands' and the
    outputs' word widths, from the cell Definition's signature."""

    def __init__(self, p: _Program) -> None:
        self.p = p
        P, S = len(p.layout["params"]), len(p.layout["scalars"])
        seeds = set(p.layout["inputs"][:3 * P + S])
        state = set(p.layout["inputs"][P:3 * P + S])
        n, n_in = len(p.nodes), p.n_in
        dep = [p.names[i] in seeds for i in range(n_in)] + [False] * (n - n_in)
        opt = [p.names[i] in state for i in range(n_in)] + [False] * (n - n_in)
        for i in range(n_in, n):
            dep[i] = any(dep[x] for x in p.reads[i])
            opt[i] = any(opt[x] for x in p.reads[i])
        self.loss = _first_node(p.ret)
        fwd = [False] * n
        fwd[self.loss] = True
        for i in range(self.loss, -1, -1):
            if fwd[i]:
                for x in p.reads[i]:
                    fwd[x] = True
        self.dep, self.opt, self.forward = dep, opt, fwd
        self.cells: list[dict] = []
        logs: dict = {}
        for i in range(n_in, n):
            nd = p.nodes[i]
            per = MAC_CELLS.get(getattr(nd.fn, "base_id", None))
            if per is None:
                continue
            macs, outs = per(nd.fn.bindings)
            params, ret = nd.fn.signature()
            k = _instances(nd)
            self.cells.append({
                "node": i, "occ": p.calls[i - n_in]["occ"], "cell": nd.fn.id, "base": nd.fn.base_id,
                "phase": self.phase(i), "instances": k, "macs": k * macs, "outputs": k * outs,
                "out_width": max(ret.width_set()), "widths": [max(t.width_set()) for _, t in params[:2]],
                "dep": [any(dep[x] for x in _ref_nodes(a.refs, logs)) for a in nd.args[:2]]})

    def phase(self, i: int) -> str:
        """Root node i's phase: the forward (the loss is computed from it), the optimizer (it reads the optimizer's
        state or scalars), or the backward."""
        return "forward" if self.forward[i] else "optimizer" if self.opt[i] else "backward"

    def occ_phase(self, o: int) -> str:
        """Occurrence o's phase: its calls' (the forward if any of them is the forward's)."""
        got = {self.phase(self.p.n_in + k) for k, c in enumerate(self.p.calls) if c["occ"] == o}
        return next((x for x in ("forward", "optimizer", "backward") if x in got), "backward")

    def products(self, structure: _Program | None = None) -> list[dict]:
        """Each occurrence with multiply-add cells (and the cells of no occurrence, as one row): its op, phase, cells and
        their instances, MACs, outputs and output wires, those of cells with both operands depending on θ′ (`*_both`),
        each operand's dependence (any cell's), the operands' and outputs' word widths, and for an mm or bmm its dims
        from its operands' views (`dims`, their MACs `dims_macs`) and, given the structure Program, its token dims
        (`token_dims`: those that differ there)."""
        p = self.p
        by: dict = {}
        for c in self.cells:
            by.setdefault(c["occ"], []).append(c)
        rows = []
        for o, cs in sorted(by.items(), key=lambda kv: -1 if kv[0] is None else kv[0]):
            both = [c for c in cs if all(c["dep"])]
            row = {"occ": o, "node": None if o is None else p.occs[o]["node"], "op": None if o is None else p.ops[o],
                   "phase": cs[0]["phase"] if o is None else self.occ_phase(o),
                   "cells": sorted({c["cell"] for c in cs}), "instances": sum(c["instances"] for c in cs),
                   "macs": sum(c["macs"] for c in cs), "outputs": sum(c["outputs"] for c in cs),
                   "out_wires": sum(c["outputs"] * c["out_width"] for c in cs),
                   "macs_both": sum(c["macs"] for c in both), "outputs_both": sum(c["outputs"] for c in both),
                   "out_wires_both": sum(c["outputs"] * c["out_width"] for c in both),
                   "dep": [any(c["dep"][a] for c in cs) for a in (0, 1)],
                   "widths": sorted({tuple(c["widths"]) for c in cs}), "out_widths": sorted({c["out_width"] for c in cs})}
            if row["op"] in PRODUCTS:
                dims = _dims(p, o)
                row["dims"] = dims
                row["dims_macs"] = math.prod(dims.values())
                if structure is not None:
                    d2 = _dims(structure, o)
                    row["token_dims"] = sorted(d for d in dims if dims[d] != d2[d])
            rows.append(row)
        return rows
