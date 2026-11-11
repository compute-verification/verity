"""The NVSwitch's multimem reduction trees and the node records that state them.

A multimem tree is a binary tree over the ranks whose nodes are f32 adds, in canonical form (`multimem_tree`): the TREE of
`tp_allreduce.AllReduceMultimemVector_v1`, one per owner as its TREES.  The trees are the node's, not the H200's: two 8x H200 nodes
have shown different ones.  So each node has a record (`NODE_FILE`, `check_node_record`), written by the 8-GPU capture's
``--register`` on it: the device, versions, each owner's tree and FTZ, NCCL's tuning and runs past symmetric memory, the routes its
launches showed and the run records they rest on.  `tp_allreduce.CAPTURED` keys those routes on the record's device and node tree
(`node_key`), which a registration states (`stated_node`).
"""
from __future__ import annotations

import json
from pathlib import Path

__all__ = ["NODE_FILE", "NODE_NCCL_FITS", "NODE_RECORDS", "NODE_ROUTES", "NODE_SCHEMA", "check_node_record", "load_node_records",
           "multimem_tree", "multimem_trees", "node_key", "nvls_tree", "stated_node", "tree_leaves"]

NODE_SCHEMA = "tp-allreduce-node/1"
#: every node record (`check_node_record`), one per node the 8-GPU capture's ``--register`` (or a whole capture) ran on
NODE_FILE = Path(__file__).with_suffix(".json")
#: the models an NCCL probe fits: bf16 at every hop, or one in-switch f32 sum rounded once, IEEE or FTZ
NODE_NCCL_FITS = ("bf16", "f32", "f32_ftz")
#: the routes a node record keys: torch's symmetric memory, and NCCL's past it when its probe fits one in-switch f32 sum (NVLS)
NODE_ROUTES = ("symm_mem_comm", "pynccl_comm")
_NODE_KEYS = ("schema", "device", "world", "versions", "multimem", "nccl", "routes", "source")


def tree_leaves(t) -> tuple[int, ...]:
    return (t,) if isinstance(t, int) else tuple(r for c in t for r in tree_leaves(c))


def multimem_tree(t, world: int):
    """`t` as a reduction tree over ranks 0..WORLD-1 in canonical form: a rank, or a pair of trees (one f32 add), every rank once,
    each pair's children ordered by their least rank (an IEEE add commutes)."""
    def canon(x):
        if isinstance(x, bool) or not isinstance(x, (int, tuple, list)):
            raise ValueError(f"TREE {t!r}: a node is a rank or a pair of nodes, not {x!r}")
        if isinstance(x, int):
            return x
        if len(x) != 2:
            raise ValueError(f"TREE {t!r}: the node {x!r} adds {len(x)} subtrees; each add takes two")
        a, b = canon(x[0]), canon(x[1])
        return (a, b) if min(tree_leaves(a)) < min(tree_leaves(b)) else (b, a)

    c = canon(t)
    if sorted(tree_leaves(c)) != list(range(world)):
        raise ValueError(f"TREE {t!r}: its leaves must be the ranks 0..{world - 1}, each once")
    return c


def multimem_trees(trees, world: int) -> tuple:
    """One canonical tree per owner rank."""
    trees = tuple(trees)
    if len(trees) != world:
        raise ValueError(f"TREES: one tree per owner rank, {world}, not {len(trees)}")
    return tuple(multimem_tree(t, world) for t in trees)


def stated_node(trees, ftz, world: int) -> tuple | None:
    """A registration's stated (TREES, FTZ) in canonical form, or None when it states neither or not one tree per owner of `world`
    ranks; a stated tree that is not a reduction tree over the ranks is refused (`multimem_tree`'s ValueError)."""
    if trees is None or ftz is None:
        return None
    trees = tuple(trees)
    return (multimem_trees(trees, world), bool(ftz)) if len(trees) == world else None


def _fields(where: str, d, keys) -> dict:
    if not isinstance(d, dict) or sorted(d) != sorted(keys):
        got = sorted(d) if isinstance(d, dict) else type(d).__name__
        raise ValueError(f"node record: {where} has {got}, not the keys {sorted(keys)}")
    return d


def _int(where: str, v, lo: int = 1) -> int:
    if isinstance(v, bool) or not isinstance(v, int) or v < lo:
        raise ValueError(f"node record: {where} is {v!r}, not an integer >= {lo}")
    return v


def _text(where: str, v, *, none: bool = False):
    if not (isinstance(v, str) and v) and not (none and v is None):
        raise ValueError(f"node record: {where} is {v!r}, not a string{' or null' if none else ''}")
    return v


def _list(where: str, v, *, empty: bool = False) -> list:
    if not isinstance(v, list) or (not v and not empty):
        raise ValueError(f"node record: {where} is {v!r}, not a {'' if empty else 'nonempty '}list")
    return v


def _trees(where: str, trees, world: int) -> tuple:
    try:
        return tuple(multimem_tree(t, world) for t in _list(where, trees))
    except ValueError as e:
        raise ValueError(f"node record: {where}: {e}") from None


def nvls_tree(nccl: dict | None, world: int):
    """The one tree a node record's NCCL all-reduces (its ``nccl``) take at every probed size over `world` ranks, when its fit is
    one in-switch f32 sum (NVLS, `AllReduceMultimem_v1` at that tree on every owner), else ValueError."""
    if nccl is None or nccl["fit"] not in ("f32", "f32_ftz"):
        raise ValueError("node record: a pynccl_comm route needs nccl.fit f32 or f32_ftz (one in-switch f32 sum)")
    trees = {multimem_tree(s["trees"][s["runs"][0][2]], world) for s in nccl["sizes"] if len(s["runs"]) == 1}
    if len(trees) != 1 or any(len(s["runs"]) != 1 for s in nccl["sizes"]):
        raise ValueError("node record: a pynccl_comm route needs NCCL's every probed size one run of one tree, the same tree")
    return trees.pop()


def check_node_record(rec, *, cited: bool = True) -> dict:
    """`rec` when it is a node record (`NODE_SCHEMA`), else ValueError naming the field. ``device`` {name, compute_capability
    "major.minor", sms}, ``world``, ``versions`` {driver, nccl, torch, vllm, flashinfer: each a string, or null where the run did
    not log it}; ``multimem`` {trees: one per owner rank as nested [a, b] lists, ftz, probes: the [N, M] all-reduces the probes
    read them off}; ``nccl``: null, or NCCL's all-reduces past symmetric memory {fit (`NODE_NCCL_FITS`), sizes: per probed [N, M]
    its bytes, NCCL's tuning (algo, proto, channels) and the runs [lo, hi, t] of the decoded ``trees`` over every element};
    ``routes``: per (setting, N, route) the lo..hi tokens its GO launches showed and the run records they rest on, a pynccl_comm
    one needing NCCL's fit one in-switch f32 sum in one run of one tree; ``source``: each run record with the commit it ran. A
    `cited` record (the tree's) has every route resting on run records ``source`` lists; the probe writes one uncited, before its
    run has a record.  Which settings a route may name is `tp_allreduce`'s (`SETTINGS`), checked where it keys the routes."""
    _fields("the record", rec, _NODE_KEYS)
    if rec["schema"] != NODE_SCHEMA:
        raise ValueError(f"node record: schema {rec['schema']!r}, not {NODE_SCHEMA!r}")
    dev = _fields("device", rec["device"], ("name", "compute_capability", "sms"))
    _text("device.name", dev["name"])
    cc = _text("device.compute_capability", dev["compute_capability"])
    if len(cc.split(".")) != 2 or not all(p.isdigit() for p in cc.split(".")):
        raise ValueError(f"node record: device.compute_capability {cc!r}, not major.minor")
    _int("device.sms", dev["sms"])
    world = _int("world", rec["world"], 2)
    for k, v in _fields("versions", rec["versions"], ("driver", "nccl", "torch", "vllm", "flashinfer")).items():
        _text(f"versions.{k}", v, none=True)
    mm = _fields("multimem", rec["multimem"], ("trees", "ftz", "probes"))
    if len(_trees("multimem.trees", mm["trees"], world)) != world:
        raise ValueError(f"node record: multimem.trees holds {len(mm['trees'])} trees, one per owner rank of {world}")
    if not isinstance(mm["ftz"], bool):
        raise ValueError(f"node record: multimem.ftz is {mm['ftz']!r}, not a boolean")
    for i, p in enumerate(_list("multimem.probes", mm["probes"])):
        if not isinstance(p, list) or len(p) != 2:
            raise ValueError(f"node record: multimem.probes[{i}] is {p!r}, not [N, M]")
        _int(f"multimem.probes[{i}].N", p[0])
        _int(f"multimem.probes[{i}].M", p[1])
    if rec["nccl"] is not None:
        n = _fields("nccl", rec["nccl"], ("fit", "sizes"))
        if n["fit"] not in NODE_NCCL_FITS:
            raise ValueError(f"node record: nccl.fit {n['fit']!r}, not one of {NODE_NCCL_FITS}")
        for i, s in enumerate(_list("nccl.sizes", n["sizes"])):
            w = f"nccl.sizes[{i}]"
            _fields(w, s, ("N", "M", "bytes", "algo", "proto", "channels", "runs", "trees"))
            size = _int(f"{w}.N", s["N"]) * _int(f"{w}.M", s["M"])
            if _int(f"{w}.bytes", s["bytes"]) != 2 * size:
                raise ValueError(f"node record: {w}.bytes {s['bytes']}, not the {2 * size} of {s['M']} x {s['N']} bf16")
            for k in ("algo", "proto", "channels"):
                _text(f"{w}.{k}", s[k])
            trees = _trees(f"{w}.trees", s["trees"], world)
            at = 0
            for j, r in enumerate(_list(f"{w}.runs", s["runs"])):
                if (not isinstance(r, list) or len(r) != 3 or r[0] != at or _int(f"{w}.runs[{j}].hi", r[1]) <= at
                        or not 0 <= _int(f"{w}.runs[{j}].t", r[2], 0) < len(trees)):
                    raise ValueError(f"node record: {w}.runs[{j}] is {r!r}, not [lo, hi, t] from {at} with t a tree of {w}.trees")
                at = r[1]
            if at != size:
                raise ValueError(f"node record: {w}.runs end at {at}, not the call's {size} elements")
    source = _list("source", rec["source"])
    for i, s in enumerate(source):
        _fields(f"source[{i}]", s, ("run_record", "commit"))
        _text(f"source[{i}].run_record", s["run_record"], none=not cited)
        _text(f"source[{i}].commit", s["commit"], none=True)
    records = {s["run_record"] for s in source}
    seen = set()
    for i, r in enumerate(_list("routes", rec["routes"])):
        w = f"routes[{i}]"
        _fields(w, r, ("setting", "N", "route", "lo", "hi", "run_records"))
        _text(f"{w}.setting", r["setting"])
        if r["route"] not in NODE_ROUTES:
            raise ValueError(f"node record: {w} is route {r['route']!r}, not one of {NODE_ROUTES}")
        if _int(f"{w}.hi", r["hi"]) < _int(f"{w}.lo", r["lo"]):
            raise ValueError(f"node record: {w} lo {r['lo']} > hi {r['hi']}")
        key = (r["setting"], _int(f"{w}.N", r["N"]), r["route"])
        if key in seen:
            raise ValueError(f"node record: {w} repeats {key}")
        seen.add(key)
        arts = _list(f"{w}.run_records", r["run_records"], empty=not cited)
        if cited and not set(arts) <= records:
            raise ValueError(f"node record: {w}.run_records {sorted(set(arts) - records)} are not in source")
        if r["route"] == "pynccl_comm":
            nvls_tree(rec["nccl"], world)
    return rec


def node_key(rec: dict) -> tuple:
    """((compute capability, SMs), world, TREES, FTZ) of a node record: what a registration's stated node must equal."""
    d = rec["device"]
    return (d["compute_capability"], d["sms"]), rec["world"], multimem_trees(rec["multimem"]["trees"], rec["world"]), \
        rec["multimem"]["ftz"]


def load_node_records(path: Path = NODE_FILE) -> tuple[dict, ...]:
    """Every node record of `path` (a JSON list), each `check_node_record` cited, no two of one node tree (`node_key`)."""
    recs = tuple(check_node_record(r) for r in json.loads(Path(path).read_text()))
    keys = [node_key(r) for r in recs]
    if len(set(keys)) != len(keys):
        raise ValueError(f"{path}: two node records hold one node tree")
    return recs


NODE_RECORDS = load_node_records()
