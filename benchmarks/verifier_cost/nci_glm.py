"""``nci_glm.py measure``: the NCI check (`nci-check --policy`, `NciCheck.Desc`) on GLM-5.3's registered Programs, the B1
Build's workload Program of each TP rank, and on Programs derived from them for the divisions the registered ones can't
express (README.md).

  measure --rank PATH ... --pins PINS --out DIR [--case NAME ...] [--gamma G] [--alpha NUM/DEN] [--jobs N]
          [--scratch DIR] [--exe PATH] [--check-mem-gb G]
  measure --case registered-node --node PATH --node-digest HEX [--node-trusted PATH --node-provenance PATH
          [--node-registration PATH]] [--node-points POINT ...] --out DIR [...]

`--rank` names each rank's workload descriptor (`build/rank<r>/build_workload/descriptor.json.gz` of the Build), in rank
order, and `--pins` the Build's meta (`rank_workload_digests`), which each must match. `--check-mem-gb` caps each
`nci-check`'s address space, so one that outgrows its share fails rather than the node. The cases:

  registered  each rank's Program as registered, whose root is its inputs and one Call, the request: θ its weights
              (`annotations["theta"]`), then θ empty, one IU (the Call)
  rank        rank 0's Program with its weights' input nodes bound at registration (`RegisteredInput<w>_v1`) and
              nothing else changed: θ the weights, one IU (the request, its peers' partials read as inputs)
  slice       Definitions of rank 0 as Programs (`slice`), their weight parameters registered, one IU per Call and one
              in all: `--policy`, the reference (`nci_reference.check_desc`) and `--policy --unrolled` on the instance
              expanded gate by gate must agree
  node        the eight ranks' requests joined into one Definition called once by a root of input nodes, each collective
              reading the other ranks' partials where it read its `peers_*` inputs (the TP exchange inside the node),
              the weights registered and θ: one IU, the node's session, at (Γ, α) and at its neighbours (α/2, 2α, Γ/2, 2Γ)
  registered-node
              the node Program `verity-vllm predict node` registered (`--node`), whose SHA-512 must be
              `--node-digest`. One IU, every root node that isn't an input (the whole node's session), θ its
              `RegisteredInput` root nodes (the descriptor's `annotations["theta"]` too, where it has them). `--policy`
              at A (`GAMMA`, `ALPHA`), its neighbours and C (`ALPHA_C` at `GAMMA` and `GAMMA_C`). Given its
              registration, the node is held to it before any check (it is the program of the record `--node-trusted`
              cites as the run's own, `run`, and `--node-provenance` has one entry per θ node, each citing the log's
              first record), and the check of record runs too, at A (`nci-check INSTANCE TRUSTED`,
              `NciCheck.Baseline`: the policy check, then θ's provenance against the registration log).
              `--node-points` picks some of `NODE_POINTS`

A finer division of a whole session (per engine step, per layer, per Call) is not checked here. Its root would be the
request's body, whose reads `NciCheck.Desc` holds as segments of gates, each Definition's return coalesced into runs
(`Desc.coalesce`); `nci_cost.py glm` counts and times them (`summarize_steps`).

DIR/results.json has a row per check: the Program (SHA-512 digest, bytes, root nodes, gates), the division, the verdict
summarised and the time and peak RSS; DIR/ius/<case>.json each verdict's IUs.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import multiprocessing
import re
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from run import EXE, _reference, timed

SCHEMA = "verity/nci-glm/v1"
DESCRIPTOR = "verity-ir/descriptor/v1"
#: the demo's policy (README.md): Γ gates per IU, α novel bits per gate
GAMMA = 2 ** 44
ALPHA = (1, 2 ** 25)
#: the stricter point C: α = 2^-28, at Γ and at Γ = 2^52
ALPHA_C = (1, 2 ** 28)
GAMMA_C = 2 ** 52
#: the registered node's checks, in the order they start: tag -> (Γ, α, the check of record or `--policy`)
NODE_POINTS = {
    "full": (GAMMA, ALPHA, True),
    "A": (GAMMA, ALPHA, False),
    "C": (GAMMA, ALPHA_C, False),
    "C-2^52": (GAMMA_C, ALPHA_C, False),
    "alpha/2": (GAMMA, (ALPHA[0], 2 * ALPHA[1]), False),
    "2alpha": (GAMMA, (2 * ALPHA[0], ALPHA[1]), False),
    "gamma/2": (GAMMA // 2, ALPHA, False),
    "2gamma": (2 * GAMMA, ALPHA, False),
}
INPUT = re.compile(r"^(Registered)?Input[1-9][0-9]*_v1$")
#: rank 0's Definitions checked gate by gate: the router, the fused all-reduce and norm, and an FP8 linear layer
SLICES = ("MoeRouterSigmoidBias_v1{", "AllReduceRmsNormMnnvl_v1{", "DeepGemmFp8Linear_v1{K=6144,N=2624}")
#: a slice's divisions: per root node (its body node `k`, or None for an input node), its IU label
SLICE_DIVISIONS = {"session": lambda k: 0, "call": lambda k: k}


def canonical(v) -> str:
    return json.dumps(v, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def load(path: Path) -> tuple[dict, dict]:
    with (gzip.open(path) if path.suffix == ".gz" else open(path, "rb")) as f:
        desc = json.load(f)
    return desc, desc.pop("annotations", {})


def leaves(t) -> int:
    if t[0] == "Value":
        return 1
    if t[0] == "Array":
        return t[1] * leaves(t[2])
    return sum(leaves(f) for f in (t[1] if t[0] == "Tuple" else [f for _, f in t[1]]))


def widths(t) -> set[int]:
    if t[0] == "Value":
        return {t[1]}
    if t[0] == "Array":
        return widths(t[2])
    return set().union(*(widths(f) for f in (t[1] if t[0] == "Tuple" else [f for _, f in t[1]])))


def members(nd: dict) -> int:
    return nd["n"] if nd["form"] in ("batch", "scan") else 1


def remap(r: dict, p, n) -> dict:
    """`r` with each parameter and node reference moved: `p(j)` and `n(k)` give the (space, index) read instead, and an
    alias names the node `n` moves its node to."""
    (form, v), = r.items()
    if form in ("affine", "strided"):
        return {form: [*(p if v[0] == "p" else n)(v[1]), *v[2:]]}
    if form == "explicit":
        return {form: [[*(p if s == "p" else n)(i), j] for s, i, j in v]}
    if form == "concat":
        return {form: [remap(x, p, n) for x in v]}
    if form == "alias":
        return {form: [n(v[0])[1], v[1]]}
    raise ValueError(f"reference form {form!r}")


def atoms(r: dict, nodes: list) -> list[dict]:
    """`r` as its affine, strided and explicit parts, in order, aliases resolved against `nodes` (PROTOCOL.md §4.5)."""
    (form, v), = r.items()
    if form == "concat":
        return [a for x in v for a in atoms(x, nodes)]
    if form == "alias":
        return atoms(nodes[v[0]]["args"][v[1]]["refs"], nodes)
    return [r]


def count(a: dict) -> int:
    (form, v), = a.items()
    return v[3] if form == "affine" else v[6] if form == "strided" else len(v)


def reads(r: dict, nodes: list) -> set[tuple[str, int]]:
    out = set()
    for a in atoms(r, nodes):
        (form, v), = a.items()
        out |= {(s, i) for s, i, _ in v} if form == "explicit" else {(v[0], v[1])}
    return out


def record_fns(v) -> list[str]:
    """The functions a binding record names (`{"fn": id}`, at any depth)."""
    if isinstance(v, dict):
        return [v["fn"]] if set(v) == {"fn"} else [f for x in v.values() for f in record_fns(x)]
    if isinstance(v, list):
        return [f for x in v for f in record_fns(x)]
    return []


class Source:
    """A descriptor read, whose nodes and definitions a `Writer` copies with their table indices translated."""

    def __init__(self, desc: dict) -> None:
        self.desc = desc
        self.defs, self.callees, self.types = desc["definitions"], desc["callees"], desc["types"]


class Writer:
    """A descriptor written from parts of others: fresh interning tables, and every definition the written nodes reach
    copied once; a definition two sources both give must be the same."""

    def __init__(self) -> None:
        self.callees: list[str] = []
        self.types: list = []
        self._c: dict[str, int] = {}
        self._t: dict[str, int] = {}
        self._maps: dict[int, tuple[dict[int, int], dict[int, int]]] = {}
        self.defs: dict[str, dict] = {}
        self._todo: list[tuple[str, Source]] = []

    def callee(self, key: str) -> int:
        if key not in self._c:
            self._c[key] = len(self.callees)
            self.callees.append(key)
        return self._c[key]

    def type(self, t) -> int:
        s = canonical(t)
        if s not in self._t:
            self._t[s] = len(self.types)
            self.types.append(t)
        return self._t[s]

    def fn(self, src: Source, i: int) -> int:
        cmap = self._maps.setdefault(id(src), ({}, {}))[0]
        j = cmap.get(i)
        if j is None:
            j = cmap[i] = self.callee(src.callees[i])
            self._todo.append((src.callees[i], src))
        return j

    def ty(self, src: Source, i: int) -> int:
        tmap = self._maps.setdefault(id(src), ({}, {}))[1]
        j = tmap.get(i)
        if j is None:
            j = tmap[i] = self.type(src.types[i])
        return j

    def node(self, src: Source, nd: dict, move, first=None) -> dict:
        """`nd` of `src`, its references moved by `move`; `first`, when given, is its first argument's references."""
        args = [{"refs": first if first is not None and a == 0 else move(x["refs"]), "type": self.ty(src, x["type"])}
                for a, x in enumerate(nd["args"])]
        return nd | {"fn": self.fn(src, nd["fn"]), "out": self.ty(src, nd["out"]), "args": args}

    def input_node(self, t, registered: bool) -> dict:
        """One parameter of type `t` as an input node (`program._lower_input`), bound at registration or per run."""
        (w,) = widths(t)
        n = leaves(t)
        key = f"{'RegisteredInput' if registered else 'Input'}{w}_v1"
        self.defs.setdefault(key, {"form": "primitive", "id": key, "params": [], "ret": ["Value", w]})
        return {"args": [], "axes": [], "fn": self.callee(key), "form": "batch", "n": n,
                "out": self.type(["Array", n, ["Value", w]])}

    def program(self, root: str, program_fn: str, nodes: list, ret: dict, ret_type) -> dict:
        same = lambda r: r
        seen: set[tuple[str, int]] = set()
        while self._todo:
            key, src = self._todo.pop()
            if (key, id(src)) in seen:
                continue
            seen.add((key, id(src)))
            d = src.defs[key]
            if d["form"] == "composite":
                d = d | {"body": d["body"] | {"nodes": [self.node(src, nd, same) for nd in d["body"]["nodes"]]}}
                self._todo += [(f, src) for f in record_fns(d["bindings"])]
            have = self.defs.setdefault(key, d)
            if have is not d and canonical(have) != canonical(d):
                raise ValueError(f"definition {key!r} differs between the sources")
        defs = self.defs | {root: {"bindings": {}, "form": "composite", "id": root,
                                   "body": {"nodes": nodes, "params": [], "ret": ret, "ret_type": ret_type}}}
        return {"schema": DESCRIPTOR, "root": root, "program_fn": program_fn, "definitions": defs,
                "callees": self.callees, "types": self.types}


def gates_of(desc: dict):
    """Gates per definition key: a primitive is one gate, a composite its nodes' members times their callees'."""
    defs, callees = desc["definitions"], desc["callees"]
    memo: dict[str, int] = {}

    def g(key: str) -> int:
        if key not in memo:
            d = defs[key]
            memo[key] = 1 if d["form"] == "primitive" else sum(members(nd) * g(callees[nd["fn"]]) for nd in d["body"]["nodes"])
        return memo[key]
    return lambda nd: members(nd) * g(callees[nd["fn"]])


def request(desc: dict) -> tuple[str, dict]:
    """A workload Program's one request: its root is input nodes and one Call, of the request's Definition."""
    nodes = desc["definitions"][desc["root"]]["body"]["nodes"]
    calls = [k for k, nd in enumerate(nodes) if not INPUT.match(desc["callees"][nd["fn"]])]
    if calls != [len(nodes) - 1] or nodes[-1]["form"] != "call":
        raise ValueError(f"root: input nodes then one Call, not non-input nodes {calls[:4]}")
    key = desc["callees"][nodes[-1]["fn"]]
    return key, desc["definitions"][key]["body"]


def structure(body: dict) -> tuple[list[int], list[int]]:
    """A request body's engine steps, each its last node (the node its sampled token is), and its collectives: the nodes
    that read another rank's partial, a `peers_*` parameter."""
    nodes = body["nodes"]
    peer = {j for j, (n, _) in enumerate(body["params"]) if n.startswith("peers_")}
    ends = []
    for a in atoms(body["ret"], nodes):
        (form, v), = a.items()
        if form != "affine" or v[0] != "n" or v[3] != 1:
            raise ValueError(f"ret: one token per step, not {a}")
        ends.append(v[1])
    if ends != sorted(ends):
        raise ValueError("ret: steps out of order")
    ends[-1] = len(nodes) - 1
    colls = [k for k, nd in enumerate(nodes) if any(s == "p" and i in peer for x in nd["args"] for s, i in reads(x["refs"], nodes))]
    return ends, colls


def call_owners(desc: dict) -> list:
    """A Program whose root is input nodes and Calls: "input" for each input node, one IU per Call."""
    nodes = desc["definitions"][desc["root"]]["body"]["nodes"]
    ids: dict[int, int] = {}
    return ["input" if INPUT.match(desc["callees"][nd["fn"]]) else ids.setdefault(k, len(ids)) for k, nd in enumerate(nodes)]


def register(desc: dict, inputs: list[int]) -> dict:
    """`desc` with its root's input nodes `inputs` bound at registration (`RegisteredInput<w>_v1` for `Input<w>_v1`) and
    nothing else changed."""
    callees, defs = list(desc["callees"]), dict(desc["definitions"])
    at = {k: i for i, k in enumerate(callees)}
    root = defs[desc["root"]]
    nodes = list(root["body"]["nodes"])
    for k in inputs:
        key = callees[nodes[k]["fn"]]
        if not INPUT.match(key) or key.startswith("Registered"):
            raise ValueError(f"root node {k} calls {key}, not an input bound per run")
        reg = "Registered" + key
        if reg not in at:
            at[reg] = len(callees)
            callees.append(reg)
            defs[reg] = defs[key] | {"id": reg}
        nodes[k] = nodes[k] | {"fn": at[reg]}
    defs[desc["root"]] = root | {"body": root["body"] | {"nodes": nodes}}
    return desc | {"callees": callees, "definitions": defs}


def reroot(src: Source, key: str, registered: list[bool]) -> tuple[dict, list]:
    """A Program whose root inlines `key`'s body, as `Program` lowers a function: one input node per parameter, in order,
    bound at registration where `registered`, then the body's nodes. Per root node, None or its body node."""
    W = Writer()
    body = src.defs[key]["body"]
    nodes = [W.input_node(t, reg) for (_, t), reg in zip(body["params"], registered)]
    P = len(nodes)
    move = lambda r: remap(r, lambda j: ("n", j), lambda k: ("n", k + P))
    nodes += [W.node(src, nd, move) for nd in body["nodes"]]
    name = src.defs[key]["id"]
    desc = W.program(f"Root[{name}]_v1", key, nodes, move(body["ret"]), body["ret_type"])
    return desc, [None] * P + list(range(len(body["nodes"])))


def join(srcs: list[Source], bodies: list[dict], colls: list[list[int]]) -> dict:
    """The ranks' requests joined into one Definition, `NodeTp<R>_v1`, called once by a root of input nodes: its
    parameters each rank's but its peers' partials (weights bound at registration), its nodes each rank's, collective by
    collective; each collective's first argument, rank r's partial in slot r and `peers_*` inputs in the others, reads
    every rank's own partial instead. `colls` are each rank's collectives (`structure`)."""
    R = len(srcs)
    W = Writer()
    params: list[tuple[str, list, bool]] = []
    nodes: list = []
    pidx: list[list[int | None]] = []
    for r, b in enumerate(bodies):
        m: list[int | None] = []
        for name, t in b["params"]:
            if name.startswith("peers_"):
                m.append(None)
                continue
            params.append((f"r{r}_{name}", t, name.startswith("weights")))
            m.append(len(params) - 1)
        pidx.append(m)
    nmap = [[-1] * len(b["nodes"]) for b in bodies]

    def mover(r: int):
        def p(j: int) -> tuple[str, int]:
            if pidx[r][j] is None:
                raise ValueError(f"rank {r}: {bodies[r]['params'][j][0]} read outside a collective's peer slot")
            return "p", pidx[r][j]

        def n(k: int) -> tuple[str, int]:
            if nmap[r][k] < 0:
                raise ValueError(f"rank {r}: node {k} read before it is placed")
            return "n", nmap[r][k]
        return lambda ref: remap(ref, p, n)
    move = [mover(r) for r in range(R)]

    def place(r: int, k: int, nd: dict) -> None:
        nodes.append(nd)
        nmap[r][k] = len(nodes) - 1

    def slots(r: int, c: int) -> list[list[dict]]:
        b = bodies[r]
        parts = atoms(b["nodes"][c]["args"][0]["refs"], b["nodes"])
        total = sum(map(count, parts))
        if total % R:
            raise ValueError(f"rank {r} node {c}: {total} leaves in {R} slots")
        M, at, out = total // R, 0, [[] for _ in range(R)]
        for a in parts:
            if at // M != (at + count(a) - 1) // M:
                raise ValueError(f"rank {r} node {c}: a part spans two slots")
            out[at // M].append(a)
            at += count(a)
        for s in range(R):
            peer = [a for a in out[s] if next(iter(a.values()))[0] == "p"
                    and bodies[r]["params"][next(iter(a.values()))[1]][0].startswith("peers_")]
            if (s == r) == bool(peer) or (s != r and len(out[s]) != 1):
                raise ValueError(f"rank {r} node {c}: slot {s} is not {'its own partial' if s == r else 'one peer input'}")
        return out

    K = len(colls[0])
    if any(len(c) != K for c in colls) or any(len({srcs[r].callees[bodies[r]["nodes"][colls[r][i]]["fn"]] for r in range(R)}) != 1
                                              for i in range(K)):
        raise ValueError("the ranks' collectives differ")
    prev = [-1] * R
    for i in range(K):
        for r in range(R):
            for k in range(prev[r] + 1, colls[r][i]):
                place(r, k, W.node(srcs[r], bodies[r]["nodes"][k], move[r]))
        own = [p for r in range(R) for p in map(move[r], slots(r, colls[r][i])[r])]
        first = own[0] if len(own) == 1 else {"concat": own}
        for r in range(R):
            c = colls[r][i]
            place(r, c, W.node(srcs[r], bodies[r]["nodes"][c], move[r], first))
            prev[r] = c
    for r in range(R):
        for k in range(prev[r] + 1, len(bodies[r]["nodes"])):
            place(r, k, W.node(srcs[r], bodies[r]["nodes"][k], move[r]))
    t = bodies[0]["ret_type"]
    if t[0] != "Array" or any(canonical(b["ret_type"]) != canonical(t) for b in bodies):
        raise ValueError("the ranks' requests return different types")
    fn, ret_type = f"NodeTp{R}_v1", ["Array", R * t[1], t[2]]
    W.defs[fn] = {"bindings": {}, "form": "composite", "id": fn,
                  "body": {"nodes": nodes, "params": [[name, t] for name, t, _ in params],
                           "ret": {"concat": [move[r](b["ret"]) for r, b in enumerate(bodies)]}, "ret_type": ret_type}}
    root = [W.input_node(t, reg) for _, t, reg in params]
    call = {"args": [{"refs": {"affine": ["n", j, 0, leaves(t)]}, "type": W.type(t)} for j, (_, t, _) in enumerate(params)],
            "fn": W.callee(fn), "form": "call", "out": W.type(ret_type)}
    return W.program(f"Root[{fn}]_v1", fn, root + [call], {"affine": ["n", len(root), 0, R * t[1]]}, ret_type)


class Runs:
    """Instances written to scratch and checked by `nci-check` on a pool; rows and IUs collected into `out`."""

    def __init__(self, args) -> None:
        self.args, self.out = args, Path(args.out)
        (self.out / "ius").mkdir(parents=True, exist_ok=True)
        self.pool = ThreadPoolExecutor(max_workers=args.jobs)
        self.futures: list = []
        self.rows: list[dict] = []

    def submit(self, path: Path, row: dict, unrolled: bool = False, trusted: Path | None = None) -> None:
        """`nci-check --policy` on the instance `path`, or with `trusted` the check of record against it."""
        argv = [str(self.args.exe), *([str(path), str(trusted)] if trusted else
                                      ["--policy", *(["--unrolled"] if unrolled else []), str(path)])]
        if self.args.check_mem_gb:
            argv = ["bash", "-c", 'ulimit -v "$0" && exec "$@"', str(self.args.check_mem_gb << 20), *argv]
        self.futures.append(self.pool.submit(self._run, argv, path, row))

    def _run(self, argv: list[str], path: Path, row: dict) -> dict:
        r = timed(argv)
        path.unlink()
        v = r.pop("verdict", None)
        row = row | r | summary(v)
        if v and "ius" in v:
            (self.out / "ius" / f"{row['case'].replace('/', '__')}.json").write_text(json.dumps(v["ius"], separators=(",", ":")))
        self.rows.append(row)
        print(json.dumps({k: row[k] for k in ("case", "rc", "seconds", "max_rss_kb", "complies", "failure", "error",
                                              "ius", "max_size", "max_nci", "stderr") if k in row}), flush=True)
        return row

    def finish(self, meta: dict) -> list[dict]:
        for f in self.futures:
            f.result()
        self.pool.shutdown()
        rows = sorted(self.rows, key=lambda r: r["case"])
        (self.out / "results.json").write_text(json.dumps({"schema": SCHEMA, **meta, "rows": rows}, indent=1) + "\n")
        return rows


def summary(v: dict | None) -> dict:
    if not v:
        return {}
    if "error" in v:
        return {"error": v["error"]}
    ius = v["ius"]
    worst = max(range(len(ius)), key=lambda u: Fraction(ius[u][1], ius[u][0]), default=None)
    return {"gates": v["gates"], "ius": len(ius), "complies": v["complies"], "failure": v["failure"],
            "failing_iu": ius[v["failure"][0]] if v["failure"] else None,
            "computed": sum(s for s, _ in ius), "novel": sum(x for _, x in ius),
            "max_size": max((s for s, _ in ius), default=0),
            "max_nci": float(Fraction(ius[worst][1], ius[worst][0])) if worst is not None else None,
            "worst_iu": ius[worst] if worst is not None else None}


def write_instance(path: Path, ptext: str, theta: list, owner: list, gamma: int, alpha: tuple[int, int],
                   provenance: list | None = None) -> Path:
    """`NciCheck.Desc`'s instance, `ptext` the program as a JSON string; with `provenance`, `NciCheck.Baseline`'s."""
    with open(path, "w") as f:
        f.write(f'{{"alpha":[{alpha[0]},{alpha[1]}],"gamma":{gamma},"owner":')
        f.write(json.dumps(owner, separators=(",", ":")))
        f.write(',"program":')
        f.write(ptext)
        if provenance is not None:
            f.write(',"provenance":')
            f.write(json.dumps(provenance, separators=(",", ":")))
        f.write(',"theta":')
        f.write(json.dumps(theta, separators=(",", ":")))
        f.write("}")
    return path


def described(desc: dict) -> tuple[str, dict]:
    """The program's canonical text and its sizes: SHA-512, bytes, root nodes, definitions, gates (computed and input),
    and the bits of its input nodes by binding."""
    text = canonical(desc)
    nodes = desc["definitions"][desc["root"]]["body"]["nodes"]
    gate = gates_of(desc)
    bits = {"run": 0, "registration": 0}
    gates = {"computed": 0, "input": 0}
    for nd in nodes:
        key = desc["callees"][nd["fn"]]
        if INPUT.match(key):
            bits["registration" if key.startswith("Registered") else "run"] += nd["n"] * int(key.rsplit("Input", 1)[1][:-3])
            gates["input"] += nd["n"]
        else:
            gates["computed"] += gate(nd)
    return text, {"sha512": hashlib.sha512(text.encode()).hexdigest(), "bytes": len(text), "root_nodes": len(nodes),
                  "definitions": len(desc["definitions"]), "gates": gates, "input_bits": bits}


def points(gamma: int, alpha: tuple[int, int], neighbours: bool) -> list[tuple[str, int, tuple[int, int]]]:
    (a, b) = alpha
    out = [("", gamma, alpha)]
    if neighbours:
        out += [("@alpha/2", gamma, (a, 2 * b)), ("@2alpha", gamma, (2 * a, b)), ("@gamma/2", gamma // 2, alpha),
                ("@2gamma", 2 * gamma, alpha)]
    return out


def check_program(runs: Runs, scratch: Path, name: str, desc: dict, theta: list, info: dict, neighbours: bool) -> None:
    """A Program whose root is input nodes and one Call, the Call one IU, at the policy and, when `neighbours`, at its
    neighbours."""
    text, size = described(desc)
    ptext = json.dumps(text)
    del text
    owner = call_owners(desc)
    for tag, g, a in points(runs.args.gamma, runs.args.alpha, neighbours):
        case = f"{name}/session{tag}"
        path = write_instance(scratch / (case.replace("/", "_") + ".json"), ptext, theta, owner, g, a)
        runs.submit(path, {"case": case, "program": name, "division": "session", "gamma": g, "alpha": list(a),
                           "theta_nodes": len(theta), **size, **info})


def registered(runs: Runs, scratch: Path, r: int, desc: dict, ann: dict, pin: str, with_theta: bool) -> dict:
    """Rank r's Program as registered: its digest against the pin, θ its weights (refused unless they are registered
    inputs), then θ empty with the request one IU."""
    text, size = described(desc)
    if size["sha512"] != pin:
        raise ValueError(f"rank {r}: digest {size['sha512'][:16]} is not the pinned {pin[:16]}")
    ptext = json.dumps(text)
    del text
    nodes = desc["definitions"][desc["root"]]["body"]["nodes"]
    owner = ["input" if INPUT.match(desc["callees"][nd["fn"]]) else 0 for nd in nodes]
    key, body = request(desc)
    names = [n for n, _ in body["params"]]
    by = {}
    for j, a in enumerate(nodes[-1]["args"]):
        (_, k, _, cnt), = a["refs"].values()
        grp = re.sub(r"[\[_].*", "", names[j])
        key_w = desc["callees"][nodes[k]["fn"]]
        by[grp] = by.get(grp, 0) + cnt * int(key_w.rsplit("Input", 1)[1][:-3])
    info = {"rank": r, "request": key[:120], "input_bits_by_group": by}
    for tag, theta in (("theta-weights", ann.get("theta", [])), ("theta-empty", [])):
        if tag == "theta-weights" and not with_theta:
            continue
        case = f"registered/r{r}/{tag}"
        path = write_instance(scratch / (case.replace("/", "_") + ".json"), ptext, theta, owner, runs.args.gamma,
                              runs.args.alpha)
        runs.submit(path, {"case": case, "program": f"registered/r{r}", "division": "session", "gamma": runs.args.gamma,
                           "alpha": list(runs.args.alpha), "theta_nodes": len(theta), **size, **info})
    return size


def slices(runs: Runs, scratch: Path, src: Source, body: dict, ref) -> None:
    """Each of `SLICES` as a Program of its own, its weight parameters registered: `--policy`, the reference and
    `--policy --unrolled` on the expanded instance."""
    names = [n for n, _ in body["params"]]
    for prefix in SLICES:
        site = next((nd for nd in body["nodes"] if src.callees[nd["fn"]].startswith(prefix)), None)
        if site is None:
            raise ValueError(f"no Call of {prefix} in rank 0's request")
        key = src.callees[site["fn"]]
        reg = [bool(rs) and all(s == "p" and names[i].startswith("weights") for s, i in rs)
               for rs in (reads(a["refs"], body["nodes"]) for a in site["args"])]
        desc, origin = reroot(src, key, reg)
        text, size = described(desc)
        nodes = desc["definitions"][desc["root"]]["body"]["nodes"]
        gate = gates_of(desc)
        theta = [k for k, nd in enumerate(nodes) if desc["callees"][nd["fn"]].startswith("RegisteredInput")]
        for div, label in SLICE_DIVISIONS.items():
            ids: dict[int, int] = {}
            owner = ["input" if k is None or gate(nd) == 0 else ids.setdefault(label(k), len(ids)) for nd, k in zip(nodes, origin)]
            inst = {"program": text, "theta": theta, "owner": owner,
                    "gamma": runs.args.gamma, "alpha": list(runs.args.alpha)}
            case = f"slice/{prefix.split('{')[0]}/{div}"
            row = {"case": case, "program": f"slice/{key[:120]}", "division": div, "gamma": runs.args.gamma,
                   "alpha": list(runs.args.alpha), "theta_nodes": len(theta), **size}
            t = time.perf_counter()
            row_ref = {"reference": summary(ref.check_desc(inst)), "reference_seconds": round(time.perf_counter() - t, 3)}
            path = scratch / (case.replace("/", "_") + ".json")
            path.write_text(json.dumps(inst))
            runs.submit(path, row | {"check": "desc"} | row_ref)
            upath = scratch / (case.replace("/", "_") + "-unrolled.json")
            upath.write_text(json.dumps(ref.expand(inst)))
            runs.submit(upath, row | {"case": case + "/unrolled", "check": "unrolled"}, unrolled=True)


def node_instances(args, scratch: Path) -> tuple[list[tuple], dict]:
    """The registered node's instances, one per `--node-points` point, once its descriptor is its digest and, given its
    registration, held to it (`registered-node`); run in a process of its own, as the decoded descriptor is tens of GB."""
    from verity.core.protocols.sampled_proofs.one_stage import registration as REG

    desc, ann = load(Path(args.node))
    text, size = described(desc)
    names = [desc["callees"][nd["fn"]] for nd in desc["definitions"][desc["root"]]["body"]["nodes"]]
    del desc
    theta = [k for k, n in enumerate(names) if n.startswith("RegisteredInput")]
    owner = ["input" if INPUT.match(n) else 0 for n in names]
    wrong = []
    if size["sha512"] != args.node_digest:
        wrong.append(f"its SHA-512 is {size['sha512'][:16]}, not --node-digest {args.node_digest[:16]}")
    if not theta:
        wrong.append("its root has no RegisteredInput node")
    if "theta" in ann and list(ann["theta"]) != theta:
        wrong.append(f"its annotations' θ ({len(ann['theta'])} nodes) isn't its RegisteredInput root nodes ({len(theta)})")
    provenance = own = trusted = None
    if args.node_trusted:
        trusted = json.loads(Path(args.node_trusted).read_text())
        provenance = json.loads(Path(args.node_provenance).read_text())
        digests = [REG.digest(rec) for rec in trusted["log"]]
        own = trusted["log"][digests.index(trusted["run"])] if trusted["run"] in digests else None
        if own is None:
            wrong.append(f"the trusted input's run {trusted['run'][:16]} is the digest of no record of its log")
        elif own["program"] != size["sha512"]:
            wrong.append(f"the run's record registers program {own['program'][:16]}, not this one")
        if len(provenance) != len(theta):
            wrong.append(f"{len(provenance)} provenance entries for {len(theta)} θ nodes")
        elif any(p.get("registration") != digests[0] for p in provenance):
            wrong.append("a provenance entry cites a record other than the log's first")
        if args.node_registration:
            reg = json.loads(Path(args.node_registration).read_text())
            if (reg["program_digest"], reg["record_digest"]) != (size["sha512"], trusted["run"]):
                wrong.append("the registration's summary names another program or record")
    if wrong:
        raise ValueError("the node is not the one named, or not its registration's: " + "; ".join(wrong))
    info = {"root_calls": owner.count(0), "theta_nodes": len(theta), **size}
    if own is not None:
        info |= {"record": trusted["run"], "window": own["window"]}
    ptext = json.dumps(text)
    del text
    made = []
    for tag in args.node_points:
        g, a, full = NODE_POINTS[tag]
        case = "registered-node/full" if full else "registered-node/session" + ("" if tag == "A" else f"@{tag}")
        path = write_instance(scratch / f"registered-node_{tag.replace('/', '_')}.json", ptext, theta, owner, g, a,
                              provenance if full else None)
        made.append((case, path, full, g, a))
    return made, info


def registered_node(runs: Runs, scratch: Path) -> dict:
    with ProcessPoolExecutor(1, mp_context=multiprocessing.get_context("spawn")) as ex:
        made, info = ex.submit(node_instances, runs.args, scratch).result()
    for case, path, full, g, a in made:
        runs.submit(path, {"case": case, "program": "registered-node", "division": "session", "gamma": g,
                           "alpha": list(a), "check": "baseline" if full else "desc", **info},
                    trusted=Path(runs.args.node_trusted) if full else None)
    return info


def measure(args) -> int:
    if "registered-node" in args.case:
        if not (args.node and args.node_digest):
            raise SystemExit("registered-node: give --node and --node-digest")
        if bool(args.node_trusted) != bool(args.node_provenance):
            raise SystemExit("registered-node: give both --node-trusted and --node-provenance (the registration), or neither")
        record = {t for t, (_, _, full) in NODE_POINTS.items() if full}
        args.node_points = args.node_points or [t for t in NODE_POINTS if args.node_trusted or t not in record]
        if not args.node_trusted and record & set(args.node_points):
            raise SystemExit("registered-node: the check of record (--node-points full) needs the registration: give "
                             "--node-trusted and --node-provenance")
    runs = Runs(args)
    t0 = time.perf_counter()
    meta: dict = {}
    with tempfile.TemporaryDirectory(dir=args.scratch) as tmp:
        scratch = Path(tmp)
        if "registered-node" in args.case:
            meta["node"] = registered_node(runs, scratch)
        if set(args.case) - {"registered-node"}:
            meta |= ranks(runs, scratch, args)
        rows = runs.finish(meta | {"gamma": args.gamma, "alpha": list(args.alpha), "seconds": round(time.perf_counter() - t0, 1)})
    bad = [r["case"] for r in rows if r.get("rc") != 0 or ("error" in r and not r["case"].endswith("theta-weights"))]
    keys = ("gates", "ius", "complies", "failure", "computed", "novel", "max_size", "worst_iu")
    by = {r["case"]: r for r in rows}
    for case in {c.removesuffix("/unrolled") for c in by if c.startswith("slice/")}:
        d, u = by.get(case, {}), by.get(case + "/unrolled", {})
        if any(d.get(k) != u.get(k) or d.get("reference", {}).get(k) != d.get(k) for k in keys):
            bad.append(case + " (the three checks disagree)")
    if bad:
        print("failures or disagreements:", ", ".join(bad), file=sys.stderr)
        return 1
    return 0


def ranks(runs: Runs, scratch: Path, args) -> dict:
    """The cases on the Build's rank Programs (`registered`, `rank`, `slice`, `node`); their meta."""
    if not (args.rank and args.pins):
        raise SystemExit(f"{', '.join(c for c in args.case if c != 'registered-node')}: give --rank and --pins")
    pins = json.loads(Path(args.pins).read_text())
    digests = pins["rank_workload_digests"]
    if len(args.rank) != len(digests):
        raise SystemExit(f"--rank names {len(args.rank)} descriptors; the Build pins {len(digests)} ranks")
    descs: list[dict] = []
    for r, path in enumerate(args.rank):
        desc, ann = load(Path(path))
        if r == 0:
            ann0 = ann
        if "registered" in args.case:
            registered(runs, scratch, r, desc, ann, digests[r], with_theta=r == 0)
        descs.append(desc)
        if "node" not in args.case and r > 0:
            descs[r] = None  # type: ignore[call-overload]
    src = Source(descs[0])
    key, body = request(descs[0])
    ends, colls = structure(body)
    meta = {"pins": pins, "rank0": {"request": key[:200], "body_nodes": len(body["nodes"]), "steps": len(ends),
                                    "collectives": len(colls), "step_ends": ends}}
    if "rank" in args.case:
        reg = register(descs[0], ann0["theta"])
        check_program(runs, scratch, "rank/r0", reg, ann0["theta"], {"rank": 0}, neighbours=False)
        del reg
    if "slice" in args.case:
        slices(runs, scratch, src, body, _reference())
    if "node" in args.case:
        srcs = [Source(d) for d in descs]
        bodies = [request(d)[1] for d in descs]
        t = time.perf_counter()
        desc = join(srcs, bodies, [structure(b)[1] for b in bodies])
        meta["join_seconds"] = round(time.perf_counter() - t, 1)
        del srcs, bodies, descs
        nodes = desc["definitions"][desc["root"]]["body"]["nodes"]
        theta = [k for k, nd in enumerate(nodes) if desc["callees"][nd["fn"]].startswith("RegisteredInput")]
        check_program(runs, scratch, "node", desc, theta, {"ranks": len(args.rank)}, neighbours=True)
        del desc, nodes
    return meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="nci_glm.py", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure")
    m.add_argument("--rank", nargs="+")
    m.add_argument("--pins")
    m.add_argument("--out", required=True)
    m.add_argument("--case", action="append", choices=("registered", "rank", "slice", "node", "registered-node"))
    m.add_argument("--gamma", type=int, default=GAMMA)
    m.add_argument("--alpha", type=lambda s: tuple(map(int, s.split("/"))), default=ALPHA)
    m.add_argument("--jobs", type=int, default=4)
    m.add_argument("--scratch", default=None)
    m.add_argument("--exe", type=Path, default=EXE)
    m.add_argument("--check-mem-gb", type=int, default=None)
    m.add_argument("--node")
    m.add_argument("--node-digest")
    m.add_argument("--node-trusted")
    m.add_argument("--node-provenance")
    m.add_argument("--node-registration")
    m.add_argument("--node-points", nargs="+", choices=tuple(NODE_POINTS), default=None)
    args = ap.parse_args(argv)
    args.case = args.case or ["registered", "rank", "slice", "node"]
    return measure(args)


if __name__ == "__main__":
    sys.exit(main())
