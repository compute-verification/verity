"""Capture maps (`verity-vllm/capture-map/v0`): what the committer captures from a served model, where, and what checks it.

A capture map is data (`manifests/capture-maps/*.json`), the capture half of a semantic profile for one checkpoint served one way
(a target device, a tensor-parallel size, an environment).  It lists every module the pinned vLLM builds for the checkpoint, each
with a disposition (`taps`, `container`, `fused`, `weights-only`, `inactive`) and the taps that cover it, and every tap: its op and
port in the capture-v2 identity, its layers, the tensor's dtype and shape on one tensor-parallel rank, where it is captured, the
weights it reads and the Program that checks it, by status (`STATUSES`).  Shapes and layer sets are expressions over the map's
symbols, which name `config.json` keys and the tensor-parallel size, so nothing in the map restates a number the config holds.

`check` holds a map against a module inventory (`verity_vllm.engine.meta_inventory`: a meta-device build of the same config, or
`inventory(model)` of a live serving rank): every (path pattern, class) the inventory lists has exactly one module entry and
every entry is listed, each entry's layers are the inventory's, each tap names an inventory module and reads tensors of the
shape the inventory gives that rank, every kernel expectation (`expect`) holds, and every Definition a Program cites is
registered.  `identities` expands the taps of one step into canonical value ids (`verity_vllm.commit.identity`), and `counts`
totals them by Program status.

    python -m verity_vllm.correspondence.capture_map check --map M.json --config config.json --inventory inv.json [--rank R]
    python -m verity_vllm.correspondence.capture_map ids --map M.json --config config.json --t 0 --phase prefill --rank 0
"""
from __future__ import annotations

import argparse
import ast
import json
import operator
import sys
from pathlib import Path
from typing import Any

from verity_vllm.commit.identity import NAMED_LAYERS, format_id, value_id

SCHEMA = "verity-vllm/capture-map/v0"
#: a tap's Program today: registered and conformance-tested; being built (on a named branch); a registered family whose kernel
#: variant is not yet pinned; a value only moved (checked by word equality against the taps it copies); none (bound by the Commit,
#: reported as an unverified unit)
STATUSES = ("exists", "tonight", "port", "copy", "committed-only")
DISPOSITIONS = ("taps", "container", "fused", "weights-only", "inactive")
TP_KINDS = ("replicated", "column", "partial", "reduced", "gathered", "vocab-shard")
LAYER_VAR = "i"

_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod}
_CMP = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt, ast.LtE: operator.le, ast.Gt: operator.gt,
        ast.GtE: operator.ge}
_CALLS = {"max": max, "min": min}


class MapError(ValueError):
    """A capture map that cannot be read: an unknown symbol, an expression outside the grammar, a malformed field."""


def load(path: Path | str) -> dict[str, Any]:
    cmap = json.loads(Path(path).read_text())
    if cmap.get("schema") != SCHEMA:
        raise MapError(f"{path}: schema {cmap.get('schema')!r}, expected {SCHEMA}")
    return cmap


def evaluate(expr: str, env: dict[str, int]) -> int | bool:
    """An integer or boolean expression over `env`: + - * // %, comparisons (chained), and/or/not, max/min, integer literals."""

    def ev(n: ast.AST):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, bool)):
            return n.value
        if isinstance(n, ast.Name):
            if n.id not in env:
                raise MapError(f"{expr!r}: unknown symbol {n.id!r}")
            return env[n.id]
        if isinstance(n, ast.BinOp) and type(n.op) in _BIN:
            return _BIN[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            return -ev(n.operand)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.Not):
            return not ev(n.operand)
        if isinstance(n, ast.BoolOp):
            vals = [ev(v) for v in n.values]
            return all(vals) if isinstance(n.op, ast.And) else any(vals)
        if isinstance(n, ast.Compare) and all(type(o) in _CMP for o in n.ops):
            left = ev(n.left)
            for o, c in zip(n.ops, n.comparators):
                right = ev(c)
                if not _CMP[type(o)](left, right):
                    return False
                left = right
            return True
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _CALLS and not n.keywords:
            return _CALLS[n.func.id](*(ev(a) for a in n.args))
        raise MapError(f"{expr!r}: {ast.dump(n)[:60]} is outside the expression grammar")

    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise MapError(f"{expr!r}: {e.msg}") from None
    return ev(tree)


def config_value(config: dict[str, Any], path: str) -> Any:
    """`a.b.0` reads config["a"]["b"][0]."""
    v: Any = config
    for part in path.split("."):
        if isinstance(v, list) and part.isdigit():
            v = v[int(part)]
        elif isinstance(v, dict) and part in v:
            v = v[part]
        else:
            raise MapError(f"config has no {path!r} (at {part!r})")
    return v


def symbols(cmap: dict[str, Any], config: dict[str, Any], tp: int) -> dict[str, int]:
    """The map's symbols for one config at tensor-parallel size `tp`: a config key, or `=expr` over the symbols before it and `tp`."""
    env: dict[str, int] = {"tp": tp}
    for name, src in cmap["symbols"].items():
        v = evaluate(src[1:], env) if src.startswith("=") else config_value(config, src)
        if not isinstance(v, int) or isinstance(v, bool):
            raise MapError(f"symbol {name} = {src!r} is {v!r}, not an integer")
        env[name] = v
    return env


def layer_set(cmap: dict[str, Any], name: str, env: dict[str, int]) -> list[int]:
    expr = cmap["layer_sets"].get(name)
    if expr is None:
        raise MapError(f"no layer set {name!r}")
    return [i for i in range(env["L"]) if evaluate(expr, {**env, LAYER_VAR: i})]


def dim(expr: int | str, env: dict[str, int], free: dict[str, Any]) -> int | str:
    """A shape entry: an integer, a symbol or `=expr`; one that reads a free symbol (a per-step size such as the token count)
    is kept as written."""
    if isinstance(expr, int):
        return expr
    src = expr[1:] if expr.startswith("=") else expr
    names = {n.id for n in ast.walk(ast.parse(src, mode="eval")) if isinstance(n, ast.Name)}
    if names & free.keys():
        return expr
    return int(evaluate(src, env))


def tap_layers(cmap: dict[str, Any], tap: dict[str, Any], env: dict[str, int]) -> list[int] | None:
    """The layers a per-layer tap is captured at; None for a model-level tap (`layer`: one of the named layers)."""
    return None if "layer" in tap else layer_set(cmap, tap["layers"], env)


def _at(pattern: str, layer: int | None) -> str:
    return pattern.replace("{" + LAYER_VAR + "}", str(layer)) if layer is not None else pattern


def _walk(rec: dict[str, Any], path: str) -> Any:
    v: Any = rec
    for part in path.split("."):
        if not isinstance(v, dict) or part not in v:
            raise KeyError(path)
        v = v[part]
    return v


def _group_tensors(group: dict[str, Any], layer: int | None) -> dict[str, Any]:
    for s in group["shapes"]:
        if layer is None or layer in s["layers"] or s["layers"] == [-1]:
            return s["tensors"]
    return {}


def check(cmap: dict[str, Any], inventory: dict[str, Any], config: dict[str, Any], tp: int,
          registry: Any = None) -> list[str]:
    """Every way `cmap` disagrees with one rank's `inventory` (`meta_inventory.inventory`'s record) for `config` at `tp`;
    empty when the map covers exactly the modules the inventory lists.  `registry` (`catalog.load()[0]`), when given, must
    hold every Definition the map's Programs cite."""
    problems: list[str] = []
    env = symbols(cmap, config, tp)
    free = cmap.get("free_symbols", {})
    groups = {(g["pattern"], g["type"]): g for g in inventory["groups"]}
    by_pattern: dict[str, list[dict[str, Any]]] = {}
    for g in inventory["groups"]:
        by_pattern.setdefault(g["pattern"], []).append(g)
    taps = {t["id"]: t for t in cmap["taps"]}
    if len(taps) != len(cmap["taps"]):
        problems.append("tap ids are not unique")
    programs = cmap["programs"]

    entries: dict[tuple[str, str], dict[str, Any]] = {}
    for m in cmap["modules"]:
        key = (m["pattern"], m["type"])
        if key in entries:
            problems.append(f"module {key} has two entries")
        entries[key] = m
    for key in groups.keys() - entries.keys():
        problems.append(f"module {key[0] or '<root>'} ({key[1]}) is in the inventory but not in the map")
    for key in entries.keys() - groups.keys():
        problems.append(f"module {key[0] or '<root>'} ({key[1]}) is in the map but not in the inventory")

    listed: set[str] = set()
    for key, m in entries.items():
        where = f"module {key[0] or '<root>'} ({key[1]})"
        if m["disposition"] not in DISPOSITIONS:
            problems.append(f"{where}: disposition {m['disposition']!r} is not one of {DISPOSITIONS}")
        mtaps = m.get("taps", [])
        if m["disposition"] in ("taps", "fused", "weights-only") and not mtaps:
            problems.append(f"{where}: disposition {m['disposition']} lists no taps")
        if m["disposition"] in ("container", "inactive") and (mtaps or not m.get("reason")):
            problems.append(f"{where}: a {m['disposition']} entry has no taps and says why")
        for tid in mtaps:
            if tid not in taps:
                problems.append(f"{where}: unknown tap {tid!r}")
        listed.update(mtaps)
        g = groups.get(key)
        if g is None or not g["layers"]:
            continue
        expect = layer_set(cmap, m.get("when", "all"), env)
        if m.get("shared"):
            expect = expect[:1]
        if sorted(g["layers"]) != expect:
            problems.append(f"{where}: the inventory has it at {len(g['layers'])} layers, the map at {len(expect)} "
                            f"(first difference {sorted(set(g['layers']) ^ set(expect))[:3]})")
    for tid in taps.keys() - listed:
        if taps[tid]["module"] is not None:
            problems.append(f"tap {tid} covers no module entry")

    for tid, t in taps.items():
        where = f"tap {tid}"
        if ("layer" in t) == ("layers" in t):
            problems.append(f"{where}: needs exactly one of layer (a named layer) or layers (a layer set)")
            continue
        if "layer" in t and t["layer"] not in NAMED_LAYERS:
            problems.append(f"{where}: layer {t['layer']!r} is not one of {NAMED_LAYERS}")
        if "/" in t["op"] or ("layers" in t and "{" + LAYER_VAR + "}" not in t["op"]):
            problems.append(f"{where}: op {t['op']!r} must name its layer and hold no '/'")
        if t["tp"] not in TP_KINDS:
            problems.append(f"{where}: tp {t['tp']!r} is not one of {TP_KINDS}")
        p = programs.get(t["program"])
        if p is None:
            problems.append(f"{where}: unknown program {t['program']!r}")
        route = t.get("route")
        if route is not None and route not in cmap.get("routes", {}):
            problems.append(f"{where}: unknown route {route!r}")
        try:
            layers = tap_layers(cmap, t, env)
            shape = [dim(d, env, free) for d in t["shape"]]
        except MapError as e:
            problems.append(f"{where}: {e}")
            continue
        if t["module"] is not None:
            gs = by_pattern.get(t["module"])
            if not gs:
                problems.append(f"{where}: module {t['module']!r} is not in the inventory")
            elif layers is not None and not set(layers) <= {i for g in gs for i in g["layers"]}:
                problems.append(f"{where}: captured at layers the inventory's {t['module']} does not have")
        for w in t.get("weights", []):
            mod, _, name = w["param"].rpartition(".")
            gs = by_pattern.get(mod, [])
            for layer in ([None] if w.get("shared") or not layers else layers):
                tensors = {}
                for g in gs:
                    tensors.update(_group_tensors(g, layer))
                ten = tensors.get(name)
                if ten is None:
                    if not w.get("after_load"):
                        problems.append(f"{where}: weight {_at(w['param'], layer)} is not in the inventory")
                    break
                if "rows" in w and ten["shape"][-2:-1] != [dim(w["rows"], env, free)]:
                    problems.append(f"{where}: weight {_at(w['param'], layer)} has shape {ten['shape']}, the map's rows "
                                    f"{w['rows']} = {dim(w['rows'], env, free)}")
                    break

    for pid, p in programs.items():
        if p["status"] not in STATUSES:
            problems.append(f"program {pid}: status {p['status']!r} is not one of {STATUSES}")
        if p["status"] == "exists" and not p["definitions"]:
            problems.append(f"program {pid}: status exists names no Definition")
        if p["status"] == "tonight" and not p.get("branch"):
            problems.append(f"program {pid}: status tonight names no branch")
        if registry is not None:
            for d in p["definitions"]:
                if d not in registry:
                    problems.append(f"program {pid}: Definition {d} is not registered")
    used = {t["program"] for t in cmap["taps"]}
    for pid in programs.keys() - used:
        alt = programs[pid].get("alternative_to")
        if alt is None:
            problems.append(f"program {pid} is cited by no tap")
        elif alt not in used:
            problems.append(f"program {pid} is the alternative to {alt!r}, which no tap cites")

    for e in cmap.get("expect", []):
        g = groups.get((e["pattern"], e["type"]))
        want = e["value"]
        if isinstance(want, str) and want.startswith("="):
            want = evaluate(want[1:], env)
        if g is None:
            problems.append(f"expect {e['pattern']} ({e['type']}): not in the inventory")
            continue
        try:
            got = _walk(g, e["path"])
        except KeyError:
            problems.append(f"expect {e['pattern']} {e['path']}: not recorded")
            continue
        if got != want:
            problems.append(f"expect {e['pattern']} {e['path']} = {want!r}, the inventory has {got!r}")
    return problems


def check_switches(cmap: dict[str, Any], doc: dict[str, Any]) -> list[str]:
    """The process-wide switches the map's target pins (`target.switches`) against an inventory document's `selections`
    (`meta_inventory`: values read from vLLM after the model is built)."""
    got = doc.get("selections", {})
    return [f"switch {k} = {v!r}, the inventory has {got.get(k)!r}" for k, v in cmap["target"].get("switches", {}).items()
            if got.get(k) != v]


def identities(cmap: dict[str, Any], env: dict[str, int], *, profile: str, request: str, t: int, phase: str,
               routes: tuple[str, ...] | None = None, rank: int | None = None) -> list[str]:
    """The canonical value ids of every tap one step captures on one rank (`s=<rank>` when `rank` is given), for the step's
    `routes` (all of them when None)."""
    out = []
    for tap in cmap["taps"]:
        if tap.get("route") is not None and routes is not None and tap["route"] not in routes:
            continue
        if tap.get("phases") and phase not in tap["phases"]:
            continue
        layers = tap_layers(cmap, tap, env)
        for layer in (layers if layers is not None else [None]):
            name = tap["layer"] if layer is None else layer
            out.append(format_id(value_id(profile, request, t, phase, name, _at(tap["op"], layer), tap["port"], shard=rank)))
    return out


def counts(cmap: dict[str, Any], env: dict[str, int], routes: tuple[str, ...] | None = None) -> dict[str, Any]:
    """Module entries, module types, tap templates and taps per step and rank (for `routes`; all when None), each by status."""
    status = {pid: p["status"] for pid, p in cmap["programs"].items()}
    per_status = dict.fromkeys(STATUSES, 0)
    templates = dict.fromkeys(STATUSES, 0)
    h200 = 0
    for tap in cmap["taps"]:
        if tap.get("route") is not None and routes is not None and tap["route"] not in routes:
            continue
        layers = tap_layers(cmap, tap, env)
        n = 1 if layers is None else len(layers)
        per_status[status[tap["program"]]] += n
        templates[status[tap["program"]]] += 1
        h200 += n if tap.get("h200_only") else 0
    return {"module_entries": len(cmap["modules"]), "module_types": len({m["type"] for m in cmap["modules"]}),
            "tap_templates": sum(templates.values()), "tap_templates_by_status": templates,
            "taps_per_step": sum(per_status.values()), "taps_per_step_by_status": per_status, "h200_only_taps_per_step": h200}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="capture_map", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("check", "counts", "ids"):
        s = sub.add_parser(name)
        s.add_argument("--map", type=Path, required=True)
        s.add_argument("--config", type=Path, required=True, help="the checkpoint's config.json")
        s.add_argument("--tp", type=int, default=None, help="tensor-parallel size (default: the map's target)")
        if name == "check":
            s.add_argument("--inventory", type=Path, required=True, help="a meta_inventory JSON (meta build or live rank)")
            s.add_argument("--rank", default=None, help="which rank of the inventory (default: every rank it holds)")
        if name == "ids":
            s.add_argument("--profile", default="000000000000")
            s.add_argument("--request", default="r0")
            s.add_argument("--t", type=int, default=0)
            s.add_argument("--phase", default="prefill")
            s.add_argument("--route", action="append", default=None)
            s.add_argument("--rank", type=int, default=None)
    a = ap.parse_args(argv)
    cmap = load(a.map)
    config = json.loads(a.config.read_text())
    tp = a.tp or cmap["target"]["tensor_parallel_size"]
    env = symbols(cmap, config, tp)
    if a.cmd == "ids":
        for cid in identities(cmap, env, profile=a.profile, request=a.request, t=a.t, phase=a.phase,
                              routes=tuple(a.route) if a.route else None, rank=a.rank):
            print(cid)
        return 0
    if a.cmd == "counts":
        print(json.dumps({"all routes": counts(cmap, env), **{r: counts(cmap, env, (r,)) for r in cmap.get("routes", {})}}, indent=1))
        return 0
    from verity_vllm.program.registry import catalog

    registry, errors = catalog.load()
    inv = json.loads(a.inventory.read_text())
    ranks = inv["ranks"] if "ranks" in inv else {"live": inv}
    chosen = [a.rank] if a.rank is not None else [r for r in ranks if "groups" in ranks[r]]
    bad = 0
    switch_docs = {f"rank {r}": rec for r, rec in ranks.items() if "selections" in rec} or {"inventory": inv}
    for where, d in switch_docs.items():
        for p in check_switches(cmap, d):
            print(f"FAIL {where}: {p}")
            bad += 1
    sigs = {r: ranks[r].get("signature") for r in ranks}
    if len(set(sigs.values())) > 1:
        print(f"FAIL ranks disagree on the inventory signature: {sigs}")
        bad += 1
    for r in chosen:
        problems = errors + check(cmap, ranks[r], config, tp, registry.defs)
        for p in problems:
            print(f"FAIL rank {r}: {p}")
        bad += len(problems)
        print(f"rank {r}: {len(problems)} problems, signature {sigs.get(r)}")
    print(json.dumps(counts(cmap, env), indent=1))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
