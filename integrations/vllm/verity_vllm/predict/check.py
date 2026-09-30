"""CHECK: built descriptor == accepted descriptor by digest, else an exact structural diff.

    report = check_descriptors(built_desc, accepted_desc, prefix_nodes=None)

The diff is over the *denoted* structure (descriptor v1 of `verity.ir.codec`): root / program_fn ids, the set of reachable
definitions (each compared by its self-contained v0 encoding via `descriptor_equivalence`-style normalisation: callee ids
and types resolved through the interning tables), and the root body node by node — form, callee id, argument types,
argument reference runs (aliases and concats resolved to `(space, idx, lo, hi)` runs), output type, batch axes / n — plus
the return runs.  Nothing is weakened: any difference is reported with its first occurrences; `prefix_nodes` compares only
the first n root nodes (development: a prefill-only build against the full accepted Program) and says so in the report.
"""

from __future__ import annotations

import json
from typing import Any

from verity.ir.codec import SCHEMA_V0, SCHEMA_V1, program_digest


def _tables(desc: dict):
    if desc.get("schema") == SCHEMA_V0:
        return (lambda f: f), (lambda t: json.dumps(t, sort_keys=True, separators=(",", ":")))
    callees, types = desc["callees"], desc["types"]
    return callees.__getitem__, (lambda i: json.dumps(types[i], sort_keys=True, separators=(",", ":")))


def _runs_of(j: Any, nodes: list[dict], schema: str) -> list[tuple]:
    """The `(space, idx, lo, hi)` runs a descriptor reference denotes (aliases resolved, concats flattened, adjacent merged)."""
    out: list[list] = []

    def push(space, idx, lo, hi):
        if out and out[-1][0] == space and out[-1][1] == idx and out[-1][3] == lo:
            out[-1][3] = hi
        else:
            out.append([space, idx, lo, hi])

    def walk(x):
        if isinstance(x, dict):
            if "alias" in x:
                k, a = x["alias"]
                walk(nodes[k]["args"][a]["refs"])
            elif "concat" in x:
                for p in x["concat"]:
                    walk(p)
            elif "affine" in x:
                space, idx, base, count = x["affine"]
                if count:
                    push(space, idx, base, base + count)
            elif "strided" in x:
                # kept as one token (not expanded: a definition body's strided view can denote 28M leaves — Embedding's
                # column view of the table); both sides encode the same registry code, so token equality is what we compare
                out.append(["strided", *x["strided"]])
            elif "explicit" in x or "items" in x:
                for space, idx, leaf in (x.get("explicit") or x.get("items") or []):     # an empty explicit = an Array<0> (empty output)
                    push(space, idx, leaf, leaf + 1)
            else:
                raise ValueError(f"unknown refs form {list(x)[:4]}")
        elif isinstance(x, list):
            for p in x:
                walk(p)
        else:
            raise ValueError(f"unknown refs json {type(x).__name__}")

    walk(j)
    return [tuple(r) for r in out]


def _node_key(nd: dict, nodes: list[dict], callee, typ, schema: str) -> tuple:
    args = tuple((typ(a["type"]), tuple(_runs_of(a["refs"], nodes, schema))) for a in nd["args"])
    return (nd["form"], callee(nd["fn"]), args, typ(nd["out"]), tuple(nd.get("axes", ())), nd.get("n", 1), nd.get("n_xs", 0))


def _root_body(desc: dict) -> dict:
    return desc["definitions"][desc["root"]]["body"]


def _def_key(desc: dict, fid: str) -> Any:
    """A self-contained comparable form of one definition (interning resolved)."""
    d = desc["definitions"][fid]
    if d["form"] == "primitive":
        return ("primitive", d["id"], json.dumps(d["params"], sort_keys=True), json.dumps(d["ret"], sort_keys=True))
    callee, typ = _tables(desc)
    b = d["body"]
    nodes = b["nodes"]
    return ("composite", d["id"], json.dumps(d["bindings"], sort_keys=True), json.dumps(b["params"], sort_keys=True),
            json.dumps(b["ret_type"], sort_keys=True), tuple(_node_key(nd, nodes, callee, typ, desc["schema"]) for nd in nodes),
            tuple(_runs_of(b["ret"], nodes, desc["schema"])))


def check_descriptors(built: dict, accepted: dict, *, prefix_nodes: int | None = None, max_examples: int = 12) -> dict:
    rep: dict[str, Any] = {"digest_built": program_digest(built), "digest_accepted": program_digest(accepted)}
    rep["digest_equal"] = rep["digest_built"] == rep["digest_accepted"]
    rep["prefix_nodes"] = prefix_nodes
    if rep["digest_equal"] and prefix_nodes is None:
        rep["verdict"] = "IDENTICAL (digest)"
        return rep
    diffs: list[dict] = []
    if built.get("schema") != accepted.get("schema"):
        diffs.append({"what": "schema", "built": built.get("schema"), "accepted": accepted.get("schema")})
    for key in ("root", "program_fn"):
        if built.get(key) != accepted.get(key):
            diffs.append({"what": key, "built": built.get(key), "accepted": accepted.get(key)})
    bdefs, adefs = set(built["definitions"]), set(accepted["definitions"])
    broot, aroot = built["root"], accepted["root"]
    only_b = sorted(bdefs - adefs - {broot})
    only_a = sorted(adefs - bdefs - {aroot})
    rep["definitions"] = {"built": len(bdefs), "accepted": len(adefs), "only_built": only_b[:max_examples], "only_accepted": only_a[:max_examples],
                          "n_only_built": len(only_b), "n_only_accepted": len(only_a)}
    if prefix_nodes is None and (only_b or only_a):
        diffs.append({"what": "definitions", "only_built": only_b[:max_examples], "only_accepted": only_a[:max_examples]})
    differing_defs = []
    for fid in sorted((bdefs & adefs) - {broot, aroot}):
        if _def_key(built, fid) != _def_key(accepted, fid):
            differing_defs.append(fid)
    if differing_defs:
        diffs.append({"what": "definition bodies differ", "ids": differing_defs[:max_examples], "n": len(differing_defs)})
    # root body, node by node
    bb, ab = _root_body(built), _root_body(accepted)
    bc, bt = _tables(built)
    ac, at = _tables(accepted)
    bn, an = bb["nodes"], ab["nodes"]
    n = min(len(bn), len(an)) if prefix_nodes is None else min(prefix_nodes, len(bn), len(an))
    rep["root_nodes"] = {"built": len(bn), "accepted": len(an), "compared": n}
    if prefix_nodes is None and len(bn) != len(an):
        diffs.append({"what": "root node count", "built": len(bn), "accepted": len(an)})
    node_diffs = []
    for k in range(n):
        kb = _node_key(bn[k], bn, bc, bt, built["schema"])
        ka = _node_key(an[k], an, ac, at, accepted["schema"])
        if kb != ka:
            node_diffs.append({"node": k, "built": _describe(kb), "accepted": _describe(ka)})
            if len(node_diffs) >= max_examples:
                break
    rep["first_node_diff"] = node_diffs[0]["node"] if node_diffs else None
    if node_diffs:
        diffs.append({"what": "root body nodes differ", "examples": node_diffs})
    if json.dumps(bb["params"], sort_keys=True) != json.dumps(ab["params"], sort_keys=True):
        diffs.append({"what": "root params", "built": bb["params"], "accepted": ab["params"]})
    if prefix_nodes is None:
        if json.dumps(bb["ret_type"], sort_keys=True) != json.dumps(ab["ret_type"], sort_keys=True):
            diffs.append({"what": "ret_type", "built": bb["ret_type"], "accepted": ab["ret_type"]})
        rb, ra = _runs_of(bb["ret"], bn, built["schema"]), _runs_of(ab["ret"], an, accepted["schema"])
        if rb != ra:
            diffs.append({"what": "ret refs", "built": rb[:max_examples], "accepted": ra[:max_examples]})
    rep["diffs"] = diffs
    if prefix_nodes is not None:
        rep["verdict"] = ("PREFIX-EQUAL" if not node_diffs and not any(d["what"] in ("root params", "root", "program_fn", "definition bodies differ") for d in diffs)
                          else "PREFIX-DIFFERS") + f" (first {n} root nodes; digest not expected to match a partial build)"
    else:
        rep["verdict"] = "IDENTICAL (digest)" if rep["digest_equal"] else ("STRUCTURALLY-EQUAL-BUT-DIGEST-DIFFERS (encoding)" if not diffs else "DIFFERS")
    return rep


def _describe(key: tuple) -> dict:
    form, fn, args, out, axes, n, n_xs = key
    return {"form": form, "fn": fn, "out": out, "axes": list(axes), "n": n,
            "args": [{"type": t, "runs": [list(r) for r in runs[:6]], "n_runs": len(runs)} for t, runs in args]}
