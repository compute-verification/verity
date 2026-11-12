"""The grader: a run's training step read off its standard Program, the Boolean circuit a lane hands over, beside training's
slowdown code.

    python grader.py HANDOVER --slowdown-code DIR --commit SHA [--trace STEP.products.json] --out OUT.json

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
    structure.json.gz  the same step at another batch shape, which says which dims are the tokens': only its occurrences'
                       views are read, so it may be derived without the step's `cc`, whose routes needn't cover its shapes

Training's slowdown code is read where it sits, at the commit named, and never copied (`load_slowdown_code`): the grader
calls its names of the ops that compute products and its Definitions' output widths, and refuses a capture other than the
one it reads beside (`CAPTURE`).

What it reads (`read`, "grader-products/v1"):

- the step's matrix products (`Cells.products`): one row per mm or bmm occurrence, lowered to cells that each compute
  one row of it (`ROWS`: `LinearRef_v1`, torch CPU's arithmetic, or, derived at a `cc`, the route its cuBLAS takes, each
  k16 step the tensor core's dot), with its phase (forward, backward or optimizer), MACs, outputs and output wires, those
  of the cells whose two operands depend on θ′ (the start's weights and the optimizer's state and scalars: `*_both`), the
  operands' and outputs' word widths as the circuit wires them, its dims, the tokens' among them, and its cells' |C|; the
  dots other ops are lowered to (`DotRef_v1`) are rows of their own;
- N, the MACs of the matrix products whose two operands depend on θ′, and O, their output wires (`totals`);
- |C| (`circuit_size`): the step Program's census at the Boolean prices of the primitives it calls;
- given a trace (`against`): the slowdown code's product rows of the same step (its capture's `.products.json`) beside the
  circuit's products, by phase, op and dims: their count, N, O, operand widths, token dims and |C| (the trace's at its
  gates a MAC), and both sides' |C| of the step, of each op's products and of the rest.

A pure function of its inputs: no network, nothing published. It doesn't compute the bound ρ.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import importlib
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

FORMAT = "grader-handover/v1"
READ = "grader-products/v1"
#: the capture of training's slowdown code the grader reads beside (its `capture.products.CAPTURE`)
CAPTURE = "4.25.0"
THE_CODE = "training's slowdown code"
#: its package, which sits in `benchmarks/training_slowdown`
PACKAGE = "training_slowdown"
DERIVE = "derive the step with step_program, the training-step frontend"
#: the matrix products the grader reads, and the cells each must be lowered to, one cell a row of the product (N dots of K,
#: its bindings `K` and `N`): torch CPU's arithmetic (`CELL`), or a cuBLAS route's Definition on words
#: (`cublas_bf16.route_gemm`'s word: unsplit, split-K with a bf16 workspace, serial split-K, or split-K over grid.z with a
#: bf16 workspace, an f32 one or none)
PRODUCTS = ("mm", "bmm")
CELL = "LinearRef_v1"
ROWS = (CELL, "Gemm_v2", "CublasSplitKGemmBf16Partials_v1", "CublasSerialSplitKGemm_v1",
        "CublasSplitKGemmBf16PartialsAligned_v1", "CublasSplitKGemmAligned_v1", "CublasSerialSplitKGemmAligned_v1")
DIMS = ("batch", "m", "k", "n")
PHASES = ("forward", "backward", "optimizer")


class Refused(ValueError):
    """What the grader won't read, with how to fix it."""


# -- the hand-over -------------------------------------------------------------------------------------------------------

def _annotations_digest(desc: dict) -> str:
    from verity.circuits.codec import canonical_json

    return hashlib.sha512(canonical_json(desc.get("annotations", {}))).hexdigest()


def write_handover(out: Path, step, structure, *, run: str, checkpoints: list[tuple[str, int]], frontend: str,
                   step_batch: tuple[int, ...], structure_batch: tuple[int, ...]) -> Path:
    """Write a lane's hand-over to directory `out`: the run's step Program (`step_program` at the run's batch shape
    `step_batch`), the same step at `structure_batch`, the run record (its id and each checkpoint's steps), theta as
    registered (the step Program's theta) and the frontend commit both were derived at. Returns handover.json's path."""
    from verity.circuits.codec import encode_program, program_digest

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
        raise Refused(f"the hand-over's {where} has no {key!r}: write it with grader.write_handover from the run's own step "
                      f"Program ({FORMAT})")
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


def _input_names(decoded, role: str) -> list[str]:
    """The names of a decoded Program's input nodes, which its encoding doesn't hold: the program function's inputs as its
    `correspondence` annotation names them, each one input node, the root's first."""
    from verity.circuits.program import is_input

    nodes = decoded.root.body.nodes
    names = decoded.annotations["correspondence"].get("inputs")
    n_in = sum(1 for nd in nodes if is_input(nd.fn))
    if not (isinstance(names, list) and len(names) == n_in and all(isinstance(n, str) for n in names)
            and all(is_input(nd.fn) for nd in nodes[:n_in])):
        raise Refused(f"the {role} Program's correspondence doesn't name its {n_in} input nodes, one input each, the "
                      f"root's first: {DERIVE}")
    return names


def load_handover(path: Path) -> Handover:
    """The hand-over at `path` (its directory or its handover.json), checked: its format and fields, each Program file's
    digest and annotations against the record, both Programs a frontend step (`step_program`'s annotations), theta as
    registered, and two batch shapes that tell the token dims apart. Raises `Refused` with the fix."""
    from verity.circuits.codec import decode_program, program_digest
    from verity.circuits.defs import REGISTRY
    # registers the Definitions the frontend's rules lower to, which a decoded step calls by id
    import verity_vllm.program.frontend.train_step  # noqa: F401

    path = Path(path)
    path = path / "handover.json" if path.is_dir() else path
    try:
        doc = json.loads(path.read_text())
    except (OSError, ValueError) as e:
        raise Refused(f"cannot read a hand-over at {path} ({type(e).__name__}: {e}): hand over the directory "
                      f"grader.write_handover wrote") from e
    if not isinstance(doc, dict) or doc.get("format") != FORMAT:
        got = doc.get("format") if isinstance(doc, dict) else type(doc).__name__
        raise Refused(f"{path} isn't a grader hand-over (format {got!r}, not {FORMAT!r}): {DERIVE}, and write the "
                      f"hand-over with grader.write_handover")
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
                          f"correspondence annotation): {DERIVE}")
        try:
            decoded[key] = decode_program(desc, REGISTRY)
        except Exception as e:
            raise Refused(f"the {key} Program doesn't decode here ({type(e).__name__}: {str(e)[:300]}): read it at a "
                          f"commit that has every Definition it calls, registered before the grader reads it") from e
    inputs = _input_names(decoded["step"], "step")
    _input_names(decoded["structure"], "structure")
    theta = sorted(decoded["step"].root.theta)
    names = [inputs[i] for i in theta if isinstance(i, int) and 0 <= i < len(inputs)]
    reg = [n for n in registered if isinstance(n, str)]
    if len(reg) != len(registered) or len(names) != len(theta) or sorted(reg) != sorted(names):
        extra, missing = sorted(set(reg) - set(names)), sorted(set(names) - set(reg))
        raise Refused(f"theta as registered isn't the step Program's theta (registered only: {extra[:5]}, the "
                      f"Program's only: {missing[:5]}): register the step Program's theta, or derive the step from "
                      f"the model whose weights were registered")
    _batch_ok(doc["step"]["batch"], doc["structure"]["batch"])
    return Handover(path, doc, decoded["step"], decoded["structure"])


# -- training's slowdown code --------------------------------------------------------------------------------------------

def load_slowdown_code(directory: Path, commit: str) -> SimpleNamespace:
    """Training's slowdown code from `directory`, the one its package (`PACKAGE`) sits in (`benchmarks/training_slowdown`), at
    `commit`: checked against the checkout's HEAD where `directory` is in a git checkout (`checked`), else taken as named.
    Returns the modules the grader calls (`opnames`, `work`, `capture.products` as `products`), its version and the
    commit. Refuses a directory without it, another commit, the package already loaded from another directory in this
    process, and another capture."""
    d = Path(directory).resolve()
    if not (d / PACKAGE / "__init__.py").is_file():
        raise Refused(f"{d} doesn't hold {THE_CODE}: pass the directory its package sits in (benchmarks/training_slowdown)")
    if not re.fullmatch(r"[0-9a-f]{7,40}", commit):
        raise Refused(f"{commit!r} isn't a commit: name the commit {THE_CODE} is at, in 7 to 40 hex digits")
    head = subprocess.run(["git", "-C", str(d), "rev-parse", "HEAD"], capture_output=True, text=True)
    checked = head.returncode == 0
    if checked and not head.stdout.strip().startswith(commit):
        raise Refused(f"{THE_CODE} at {d} is at {head.stdout.strip()[:12]}, not {commit}: check out {commit}, or "
                      f"name the commit it is at")
    loaded = sys.modules.get(PACKAGE)
    if loaded is not None and Path(loaded.__file__).resolve().parent.parent != d:
        raise Refused(f"{THE_CODE} is already loaded from {Path(loaded.__file__).resolve().parent.parent} in this process: "
                      f"read it from one directory a process")
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
    pkg = importlib.import_module(PACKAGE)
    mods = {m: importlib.import_module(f"{PACKAGE}.{p}") for m, p in (
        ("opnames", "opnames"), ("work", "work"), ("products", "capture.products"))}
    if mods["products"].CAPTURE != CAPTURE:
        raise Refused(f"the grader reads beside capture {CAPTURE}, and {THE_CODE} at {commit} captures with "
                      f"{mods['products'].CAPTURE}: update the grader to its capture first")
    return SimpleNamespace(**mods, version=pkg.VERSION, commit=commit, checked=checked, directory=str(d))


# -- one Program: its root nodes, what each reads, and its occurrences' views --------------------------------------------

@dataclass(frozen=True)
class View:
    """What an occurrence reads through one fx input: the Python argument it is (`arg`, None where it is none), at `shape`
    and `width` bits an entry."""
    arg: int | None
    shape: tuple
    width: int


def _dtype_width(dtype: str | None) -> int:
    import torch

    t = getattr(torch, dtype or "", None)
    if not isinstance(t, torch.dtype):
        return 64
    return 1 if t == torch.bool else t.itemsize * 8


def _leaf_parts(refs):
    from verity.circuits.refs import Concat

    if isinstance(refs, Concat):
        for p in refs.parts:
            yield from _leaf_parts(p)
    else:
        yield refs


def _ref_nodes(refs, logs: dict | None = None) -> set:
    """The root nodes `refs` reads. With `logs`, a `Concat` cut from a `PartLog` (`prefix`: a causal attention's rows, each
    call's the call before's plus one) reads the log's parts once over all its cuts (`_log_nodes`)."""
    from verity.circuits.refs import Affine, Concat, Explicit, Strided

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


class _Program:
    """One step Program as the grader reads it: its root nodes (the inputs first, then the calls, each tied to an occurrence
    of the step's graph, or to none, by the `correspondence` annotation), the root nodes each reads (`reads`), the root's
    return (`ret`, the loss first), and each occurrence's op and the views it reads its inputs through (`views`)."""

    def __init__(self, decoded, role: str) -> None:
        import torch

        self.role = role
        root = decoded.root
        nodes = self.nodes = root.body.nodes
        ann = decoded.annotations
        corr = ann["correspondence"]
        self.layout = ann["train_step"]
        self.names = _input_names(decoded, role)
        n_in = self.n_in = len(self.names)
        occs, calls = self.occs, self.calls = corr["occurrences"], corr["calls"]
        if len(calls) != len(nodes) - n_in:
            raise Refused(f"the {role} Program's correspondence has {len(calls)} calls for its {len(nodes) - n_in} "
                          f"root calls: {DERIVE}")
        self.theta = sorted(root.theta)
        self.leaves = [nodes[i].out.leaves for i in range(n_in)]
        self.ops, self.aten = [], []
        for o in occs:
            if o["written"]:
                raise Refused(f"occurrence {o['node']} ({o['op']}) writes an input in place: {DERIVE} (it functionalizes "
                              f"the step)")
            ns, name, overload = (o["op"].split(".") + ["", ""])[:3]
            t = getattr(getattr(getattr(torch.ops, ns, None), name, None), overload, None)
            if not isinstance(t, torch._ops.OpOverload):
                raise Refused(f"occurrence {o['node']}'s op {o['op']} isn't a torch op here: read the step with the torch "
                              f"the frontend derived it with")
            self.ops.append(t.overloadpacket.__name__)
            self.aten.append(t.namespace == "aten")
        logs: dict = {}
        self.reads = [frozenset()] * n_in + [frozenset().union(*(_ref_nodes(a.refs, logs) for a in nd.args))
                                             for nd in nodes[n_in:]]
        self.ret = root.body.ret
        self.views = [[View(x["arg"], tuple(x["view"]["shape"]), _dtype_width(x["view"]["dtype"])) for x in occ["inputs"]]
                      for occ in occs]


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
#: row of a matrix product (`ROWS`, `{K, N}`: N dots of K, what every mm and bmm is lowered to) and one dot
#: (`DotRef_v1{K}`, what other ops' sums of products are lowered to)
MAC_CELLS = {**dict.fromkeys(ROWS, lambda b: (b["K"] * b["N"], b["N"])), "DotRef_v1": lambda b: (b["K"], 1)}


def _instances(nd) -> int:
    return nd.n if nd.form in ("batch", "scan") else 1


def _first_node(refs) -> int:
    from verity.circuits.refs import Affine, Explicit, Strided

    p = next(iter(_leaf_parts(refs)))
    return p.idx if isinstance(p, (Affine, Strided)) else p.items[0][1] if isinstance(p, Explicit) else p[0][1]


def _check_products(p: _Program, code) -> None:
    """Refuse a step whose matrix products the grader doesn't read whole: an aten op that computes products (the slowdown
    code's `opnames.gemm_like`) other than an mm or bmm, and an mm or bmm whose multiply-add cells aren't rows (`ROWS`) of
    its K and N, one for each of its rows."""
    cells: dict = collections.defaultdict(list)
    for k, c in enumerate(p.calls):
        nd = p.nodes[p.n_in + k]
        if getattr(nd.fn, "base_id", None) in MAC_CELLS:
            cells[c["occ"]].append(nd)
    for o, name in enumerate(p.ops):
        where = f"occurrence {p.occs[o]['node']} ({p.occs[o]['op']})"
        if name not in PRODUCTS:
            if p.aten[o] and code.opnames.gemm_like(name):
                raise Refused(f"{where} computes matrix products the grader doesn't read (it reads "
                              f"{', '.join(PRODUCTS)}): derive the step with a rule that lowers it to mm or bmm")
            continue
        d = _dims(p, o)
        bad = sorted({nd.fn.id for nd in cells[o] if nd.fn.base_id not in ROWS or nd.fn.bindings.get("K") != d["k"]
                      or nd.fn.bindings.get("N") != d["n"]})
        rows = sum(_instances(nd) for nd in cells[o])
        if bad or rows != d["batch"] * d["m"]:
            raise Refused(f"{where}, {d['batch']}x{d['m']}x{d['k']}x{d['n']}, isn't row cells ({', '.join(ROWS)}) of "
                          f"K={d['k']} and N={d['n']}, one for each of its {d['batch'] * d['m']} rows "
                          f"({', '.join(bad[:2]) or f'{rows} rows'}): derive the step with step_program's training rules")


class Cells:
    """A step Program's multiply-add cells as the products bound reads its circuit, root node by root node: which nodes
    depend on θ′ (`dep`: the start's weights, the optimizer's state and scalars, and every node computed from them), which
    the loss is computed from (`forward`), which read the optimizer's state (`opt`), and each multiply-add cell
    (`MAC_CELLS`): its occurrence, census class, instances, MACs and outputs, whether each of its two operands depends on θ′,
    and the operands' and the outputs' word widths, from the cell Definition's signature."""

    def __init__(self, p: _Program) -> None:
        from verity.circuits.codec import descriptor_id

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
                "class": descriptor_id(nd.fn),
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
        return next((x for x in PHASES if x in got), "backward")

    def products(self, structure: _Program | None = None, C: dict | None = None) -> list[dict]:
        """Each occurrence with multiply-add cells (and the cells of no occurrence, as one row): its op, phase, cells and
        their instances, MACs, outputs and output wires, those of cells with both operands depending on θ′ (`*_both`),
        each operand's dependence (any cell's), the operands' and outputs' word widths, for an mm or bmm its dims
        from its operands' views (`dims`, their MACs `dims_macs`) and, given the structure Program, its token dims
        (`token_dims`: those that differ there), and, given the step's |C| (`circuit_size`), its cells' gates (`gates`,
        each instance at its class's) and the primitives they leave out, with no Boolean version (`unpriced`)."""
        p = self.p
        classes = None if C is None else {r["class"]: r for r in C["classes"]}
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
            if classes is not None:
                cl = [classes[c["class"]] for c in cs]
                row["gates"] = sum(c["instances"] * (k["gates"] // k["instances"]) for c, k in zip(cs, cl))
                row["unpriced"] = sorted({u for k in cl for u in k["unpriced"]})
            rows.append(row)
        return rows


# -- |C| -----------------------------------------------------------------------------------------------------------------

def boolean_prices(prims: dict) -> dict[str, dict]:
    """Each primitive's Boolean price, by id: its Boolean version (`definitions.boolean.boolean_version`) and that circuit's
    ANDs, XORs, NOTs and gates (`evaluation.bits.compile`'s counts); a primitive with no Boolean version has none, and why.
    Inputs (`program.is_input`) are left out: they are bound, not computed, and `priced` counts no input class."""
    from verity.circuits.codec import descriptor_id
    from verity.circuits.evaluation.bits import compile as compile_bits
    from verity.circuits.program import is_input
    from verity.definitions.boolean import boolean_version

    out = {}
    for pid, fn in sorted(prims.items()):
        if is_input(fn):
            continue
        try:
            bv = boolean_version(fn)
        except Exception as e:  # noqa: BLE001  (named in the output: a primitive without a Boolean version is unpriced)
            out[pid] = {"boolean": None, "why": f"{type(e).__name__}: {str(e)[:200]}"}
            continue
        c = compile_bits(bv).counts
        a, x, n = c.get("and", 0), c.get("xor", 0), c.get("not", 0)
        out[pid] = {"boolean": descriptor_id(bv), "and": a, "xor": x, "not": n, "gates": a + x + n}
    return out


def priced(cen: dict, prices: dict) -> dict:
    """Each root callee class's word gates, Boolean gates and ANDs (its primitives at their prices, times its instances under
    the root) and the primitives it calls without a Boolean version (`unpriced`), and those primitives' word gates."""
    rows, unpriced = [], collections.Counter()
    for cid in cen["callees"]:
        cl = cen["classes"][cid]
        if cl["kind"] == "input":
            continue
        inst = sum(n for c, n in zip(cen["nodes"]["callee"], cen["nodes"]["n"]) if cen["callees"][c] == cid)
        g = a = 0
        for pid, k in cl["primitives"].items():
            p = prices.get(pid, {})
            if p.get("boolean") is None:
                unpriced[pid] += inst * k
                continue
            g += inst * k * p["gates"]
            a += inst * k * p["and"]
        rows.append({"class": cid, "instances": inst, "word_gates": inst * cl["gates"], "gates": g, "ands": a,
                     "unpriced": sorted(pid for pid in cl["primitives"] if prices.get(pid, {}).get("boolean") is None)})
    rows.sort(key=lambda r: -r["gates"])
    return {"classes": rows, "gates": sum(r["gates"] for r in rows), "ands": sum(r["ands"] for r in rows),
            "word_gates": sum(r["word_gates"] for r in rows), "unpriced_word_gates": dict(unpriced)}


def circuit_size(program) -> dict:
    """|C|: the step Program's census (`verity.definitions.circuits.census`) at the Boolean prices of the primitives it calls
    (`boolean_prices`): its gates and ANDs by root callee class, and the word gates of the primitives with no Boolean
    version, which it leaves out."""
    from verity.definitions.circuits.census import census
    from verity_vllm.program.frontend.train_step import primitives

    return priced(census(program), boolean_prices(primitives(program)))


# -- the front door ------------------------------------------------------------------------------------------------------

def totals(rows: list[dict]) -> dict:
    """The matrix products' count, MACs, N (the MACs of those whose two operands depend on θ′), O (their output wires) and
    gates, with N and gates by phase, gates by op and the operands' and outputs' widths; the other ops' dots apart (`dots`)."""
    prods = [r for r in rows if r["op"] in PRODUCTS]
    dots = [r for r in rows if r["op"] not in PRODUCTS]
    return {"products": len(prods), "macs": sum(r["macs"] for r in prods), "N": sum(r["macs_both"] for r in prods),
            "O": sum(r["out_wires_both"] for r in prods), "outputs": sum(r["outputs_both"] for r in prods),
            "gates": sum(r["gates"] for r in prods),
            "N_by_phase": {ph: sum(r["macs_both"] for r in prods if r["phase"] == ph) for ph in PHASES},
            "gates_by_phase": {ph: sum(r["gates"] for r in prods if r["phase"] == ph) for ph in PHASES},
            "gates_by_op": {op: sum(r["gates"] for r in prods if r["op"] == op) for op in PRODUCTS},
            "unpriced": sorted({u for r in prods for u in r["unpriced"]}),
            "widths": sorted({tuple(w) for r in prods for w in r["widths"]}),
            "out_widths": sorted({w for r in prods for w in r["out_widths"]}),
            "dots": {"rows": len(dots), "macs": sum(r["macs"] for r in dots), "N": sum(r["macs_both"] for r in dots),
                     "O": sum(r["out_wires_both"] for r in dots), "gates": sum(r["gates"] for r in dots)}}


def read(h: Handover, code) -> dict:
    """What the grader reads of hand-over `h` (the module's "What it reads", "grader-products/v1"), with the hand-over and
    the slowdown code (`load_slowdown_code`) it read beside. Refuses a structure Program of another step and a step whose
    matrix products it doesn't read whole."""
    step, structure = _Program(h.step, "step"), _Program(h.structure, "structure")
    _same_step(step, structure)
    _check_products(step, code)
    C = circuit_size(h.step)
    rows = Cells(step).products(structure, C)
    return {"format": READ,
            "handover": {"path": str(h.path), "run": h.doc["run"]["id"], "frontend": h.doc["frontend"]["commit"],
                         **{k: {"digest": h.doc[k]["digest"], "batch": h.doc[k]["batch"]} for k in ("step", "structure")}},
            "slowdown_code": {"version": code.version, "commit": code.commit, "checked": code.checked, "capture": CAPTURE},
            "products": rows, "totals": totals(rows), "C": C}


def load_trace(path: Path) -> dict:
    """The slowdown code's product rows of a step (its capture's `.products.json`): its rows, |C| (`gates`) and capture."""
    try:
        trace = json.loads(Path(path).read_text())
    except (OSError, ValueError) as e:
        raise Refused(f"cannot read a trace at {path} ({type(e).__name__}: {e}): pass the .products.json {THE_CODE}'s "
                      f"capture wrote for the step") from e
    if not isinstance(trace, dict) or not isinstance(trace.get("rows"), list) or "gates" not in trace or "capture" not in trace:
        raise Refused(f"{path} isn't {THE_CODE}'s product rows of a step (rows, gates and capture): pass the .products.json "
                      f"its capture wrote for the step")
    return trace


def _key(phase: str, op: str, dims: dict) -> tuple:
    return (phase, op, *(dims[d] for d in DIMS))


def against(result: dict, trace: dict, code) -> dict:
    """The circuit's matrix products (`read`'s) beside the slowdown code's product rows of the same step (`trace`,
    `load_trace`), by (phase, op, batch, m, k, n): on each side the products' count, N, O, operand widths, token dims and
    gates. The trace's widths are those its capture passes; its O has each output at its Definition's output width b_y, as
    the slowdown code's `he_of_rows` reads it from `work.HE_BITS` (the narrower operand's width for a Definition it
    doesn't list); its gates are each product's MACs at its gates a MAC (`gates_per_mac`), as its |C| prices them. The
    circuit's are as it wires and prices them. `agree` says which of count, N, O, widths and token dims match at every key;
    `C` holds both sides' |C| and their ratio: the step's, each op's products', all products' and the rest's (the step's
    less its products')."""
    if trace["capture"] != CAPTURE:
        raise Refused(f"the trace is capture {trace['capture']}'s, and the grader reads beside {CAPTURE}'s: capture the "
                      f"step with {THE_CODE} at a commit whose capture is {CAPTURE}")
    by: dict = {}

    def side(key: tuple, s: str) -> dict:
        e = by.setdefault(key, {})
        return e.setdefault(s, {"count": 0, "N": 0, "O": 0, "gates": 0, "widths": set(), "token_dims": set()})

    for r in result["products"]:
        if r["op"] not in PRODUCTS:
            continue
        e = side(_key(r["phase"], r["op"], r["dims"]), "circuit")
        e["count"] += 1
        e["N"] += r["macs_both"]
        e["O"] += r["out_wires_both"]
        e["gates"] += r["gates"]
        e["widths"] |= {tuple(w) for w in r["widths"]}
        e["token_dims"].add(tuple(r.get("token_dims") or ()))
    for r in trace["rows"]:
        e = side(_key(r["phase"], r["op"], r["dims"]), "trace")
        e["count"] += r["count"]
        if all(r["affected"]):
            b_y = code.work.HE_BITS.get(r["cell_gates"], (min(r["widths"]),) * 2)[1]
            e["N"] += r["count"] * r["macs"]
            e["O"] += r["count"] * r["outputs"] * b_y
        e["gates"] += r["count"] * r["macs"] * r["gates_per_mac"]
        e["widths"].add(tuple(r["widths"]))
        e["token_dims"].add(tuple(sorted(r["token_dims"])))
    empty = {"count": 0, "N": 0, "O": 0, "gates": 0, "widths": [], "token_dims": []}
    rows = []
    for key in sorted(by, key=lambda k: (PHASES.index(k[0]) if k[0] in PHASES else len(PHASES), k[1], -math.prod(k[2:]), k)):
        row = {"phase": key[0], "op": key[1], "dims": dict(zip(DIMS, key[2:]))}
        for s in ("circuit", "trace"):
            e = by[key].get(s)
            row[s] = empty if e is None else {**e, "widths": sorted(e["widths"]), "token_dims": sorted(e["token_dims"])}
        rows.append(row)
    agree = {x: all(r["circuit"][x] == r["trace"][x] for r in rows) for x in ("count", "N", "O", "widths", "token_dims")}

    def ratio(circuit, trace_) -> dict:
        return {"circuit": circuit, "trace": trace_, "ratio": circuit / trace_ if trace_ else None}

    ops = {s: {op: sum(r[s]["gates"] for r in rows if r["op"] == op) for op in PRODUCTS} for s in ("circuit", "trace")}
    prods = {s: sum(ops[s].values()) for s in ops}
    step = {"circuit": result["C"]["gates"], "trace": trace["gates"]}
    C = {"step": ratio(step["circuit"], step["trace"]), **{op: ratio(ops["circuit"][op], ops["trace"][op]) for op in PRODUCTS},
         "products": ratio(prods["circuit"], prods["trace"]),
         "rest": ratio(step["circuit"] - prods["circuit"], None if step["trace"] is None else step["trace"] - prods["trace"])}
    return {"rows": rows, "totals": {s: {x: sum(r[s][x] for r in rows) for x in ("count", "N", "O", "gates")}
                                     for s in ("circuit", "trace")},
            "agree": agree, "C": C}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="grader.py", description="Read a run's training step off the Program a lane hands over, "
                                                               f"beside {THE_CODE} (grader.py's docstring).")
    ap.add_argument("handover", type=Path, help="the hand-over's directory (grader.write_handover's) or its handover.json")
    ap.add_argument("--slowdown-code", type=Path, required=True,
                    help=f"the directory {THE_CODE}'s package sits in (benchmarks/training_slowdown)")
    ap.add_argument("--commit", required=True, help=f"the commit {THE_CODE} is at")
    ap.add_argument("--trace", type=Path, help=f"{THE_CODE}'s product rows of the same step (its capture's .products.json)")
    ap.add_argument("--out", type=Path, required=True, help="where to write what it read (JSON)")
    a = ap.parse_args(argv)
    try:
        code = load_slowdown_code(a.slowdown_code, a.commit)
        out = read(load_handover(a.handover), code)
        if a.trace is not None:
            out["against"] = against(out, load_trace(a.trace), code)
    except Refused as e:
        print(f"grader: refused: {e}", file=sys.stderr)
        return 2
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    t = out["totals"]
    line = f"grader: {t['products']} products, N {t['N']:,}, O {t['O']:,}, |C| {out['C']['gates']:,} ({t['gates']:,} products')"
    if "against" in out:
        C = out["against"]["C"]
        line += (f"; against {THE_CODE}: agree {out['against']['agree']}, |C| ratio "
                 + ", ".join(f"{k} {C[k]['ratio']}" for k in ("step", "products", *PRODUCTS, "rest")))
    print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
