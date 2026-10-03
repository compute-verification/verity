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
registered.  `check_target` holds the device and the process-wide switches the inventory was built under against the map's
target.  `identities` expands the taps of one step into canonical value ids (`verity_vllm.commit.identity`), and `counts`
totals them by Program status.  A tap marked `target_only` runs only on the map's target device (its kernels have no build
for the devices the inventory and the tests run on).

    verity-vllm capture-map check --map M.json --config config.json --inventory inv.json [--rank R]
    verity-vllm capture-map counts --map M.json --config config.json
    verity-vllm capture-map ids --map M.json --config config.json --t 0 --phase prefill --rank 0
"""
from __future__ import annotations

import ast
import json
import operator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from verity_vllm.commit.identity import NAMED_LAYERS, format_id, value_id
from verity_vllm.config import option

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
        c = t.get("commit")
        if c is not None and c.get("binder") not in cmap.get("binders", {}):
            problems.append(f"{where}: commit binder {c.get('binder')!r} is not one of the map's binders")
        if c is not None and "layers" in t and "{" + LAYER_VAR + "}" not in c.get("name", ""):
            problems.append(f"{where}: commit name {c.get('name')!r} must name its layer")
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


def check_target(cmap: dict[str, Any], doc: dict[str, Any]) -> list[str]:
    """The map's target against one inventory record (`meta_inventory`: values read back from vLLM once the model is built):
    the compute capability and SM count vLLM answered (`applied`) and the process-wide switches (`target.switches` against
    `selections`).  A record that lacks one of them disagrees."""
    target, applied, got = cmap["target"], doc.get("applied", {}), doc.get("selections", {})
    out = [f"{k} = {target[k]!r}, the inventory has {applied.get(k)!r}" for k in ("compute_capability", "num_sms")
           if applied.get(k) != target[k]]
    return out + [f"switch {k} = {v!r}, the inventory has {got.get(k)!r}" for k, v in target.get("switches", {}).items()
                  if got.get(k) != v]


def _step_taps(cmap: dict[str, Any], env: dict[str, int], phase: str, routes: tuple[str, ...] | None,
               binders: tuple[str, ...] | None = None):
    """(tap, layer) for every tap one step captures on `routes` (all when None); with `binders`, only the taps one of them binds."""
    for tap in cmap["taps"]:
        if tap.get("route") is not None and routes is not None and tap["route"] not in routes:
            continue
        if tap.get("phases") and phase not in tap["phases"]:
            continue
        if binders is not None and (tap.get("commit") or {}).get("binder") not in binders:
            continue
        layers = tap_layers(cmap, tap, env)
        for layer in (layers if layers is not None else [None]):
            yield tap, layer


def _value_id(tap: dict[str, Any], layer: int | None, *, profile: str, request: str, t: int, phase: str, rank: int | None) -> str:
    name = tap["layer"] if layer is None else layer
    return format_id(value_id(profile, request, t, phase, name, _at(tap["op"], layer), tap["port"], shard=rank))


def identities(cmap: dict[str, Any], env: dict[str, int], *, profile: str, request: str, t: int, phase: str,
               routes: tuple[str, ...] | None = None, rank: int | None = None, binders: tuple[str, ...] | None = None) -> list[str]:
    """The canonical value ids of every tap one step captures on one rank (`s=<rank>` when `rank` is given), for the step's
    `routes` (all of them when None); with `binders`, only the taps those committer binders bind (`commit.binder`)."""
    return [_value_id(tap, layer, profile=profile, request=request, t=t, phase=phase, rank=rank)
            for tap, layer in _step_taps(cmap, env, phase, routes, binders)]


def committer_names(cmap: dict[str, Any], env: dict[str, int], *, phase: str = "prefill", routes: tuple[str, ...] | None = None,
                    rank: int = 0, binders: tuple[str, ...] | None = None) -> dict[str, tuple[dict[str, Any], int | None]]:
    """The committer's tensor name (`commit.name`: `{i}` the layer, `{r}` the rank) of every bound tap one step captures on one
    rank -> (tap, layer).  Two taps of one step under one name is a `MapError`."""
    out: dict[str, tuple[dict[str, Any], int | None]] = {}
    for tap, layer in _step_taps(cmap, env, phase, routes, binders):
        c = tap.get("commit")
        if c is None:
            continue
        name = _at(c["name"], layer).replace("{r}", str(rank))
        if name in out:
            raise MapError(f"taps {out[name][0]['id']} and {tap['id']} are both committed as {name}")
        out[name] = (tap, layer)
    return out


def bind_names(cmap: dict[str, Any], env: dict[str, int], names, *, profile: str, request: str, t: int, phase: str,
               routes: tuple[str, ...] | None = None, rank: int | None = None, binders: tuple[str, ...] | None = None
               ) -> dict[str, list[str]]:
    """One step's committed tensor names on one rank against the map: `ids`, the value id of every name the map lists; `unknown`,
    the names it does not list; `missing`, the names of bound taps (of `binders`, when given) the step did not commit."""
    want = committer_names(cmap, env, phase=phase, routes=routes, rank=rank or 0, binders=binders)
    got = set(names)
    return {"ids": sorted(_value_id(*want[n], profile=profile, request=request, t=t, phase=phase, rank=rank) for n in got & want.keys()),
            "unknown": sorted(got - want.keys()), "missing": sorted(want.keys() - got)}


def counts(cmap: dict[str, Any], env: dict[str, int], routes: tuple[str, ...] | None = None) -> dict[str, Any]:
    """Module entries, module types, tap templates and taps per step and rank (for `routes`; all when None), each by status."""
    status = {pid: p["status"] for pid, p in cmap["programs"].items()}
    per_status = dict.fromkeys(STATUSES, 0)
    templates = dict.fromkeys(STATUSES, 0)
    target_only = 0
    by_binder: dict[str, int] = {}
    for tap in cmap["taps"]:
        if tap.get("route") is not None and routes is not None and tap["route"] not in routes:
            continue
        layers = tap_layers(cmap, tap, env)
        n = 1 if layers is None else len(layers)
        per_status[status[tap["program"]]] += n
        templates[status[tap["program"]]] += 1
        target_only += n if tap.get("target_only") else 0
        if tap.get("commit"):
            b = tap["commit"]["binder"]
            by_binder[b] = by_binder.get(b, 0) + n
    return {"module_entries": len(cmap["modules"]), "module_types": len({m["type"] for m in cmap["modules"]}),
            "tap_templates": sum(templates.values()), "tap_templates_by_status": templates,
            "taps_per_step": sum(per_status.values()), "taps_per_step_by_status": per_status,
            "target_only_taps_per_step": target_only, "bound_taps_per_step": sum(by_binder.values()),
            "bound_taps_per_step_by_binder": dict(sorted(by_binder.items()))}


def check_document(cmap: dict[str, Any], doc: dict[str, Any], config: dict[str, Any], tp: int, registry: Any = None,
                   rank: str | None = None) -> dict[str, list[str]]:
    """Every problem of an inventory document (`meta_inventory`'s, or `live_document` of a serving engine's ranks), by where:
    each rank's `check_target`, its `check` (`rank`: that rank alone; else every rank that carries its modules), and the
    ranks' disagreement on the inventory signature (under `ranks`)."""
    ranks = doc["ranks"]
    out: dict[str, list[str]] = {}
    sigs = {r: rec.get("signature") for r, rec in ranks.items()}
    if len(set(sigs.values())) > 1:
        out["ranks"] = [f"the ranks disagree on the inventory signature: {sigs}"]
    for r in ([rank] if rank is not None else [r for r in ranks if "groups" in ranks[r]]):
        out[f"rank {r}"] = check_target(cmap, ranks[r]) + check(cmap, ranks[r], config, tp, registry)
    return out


@dataclass(frozen=True, kw_only=True)
class CheckOptions:
    "a capture map against a module inventory (a meta-device build, or a serving engine's ranks)"

    map: str = option("--map")
    config: str = option("--config", help="the checkpoint's config.json")
    inventory: str = option("--inventory", help="a meta-inventory document (`verity-vllm meta-inventory`, or a live one)")
    rank: str | None = option("--rank", default=None, help="one rank of the inventory (default: every rank it holds)")
    tp: int | None = option("--tp", type=int, default=None, help="tensor-parallel size (default: the map's target)")
    cmd: str = field(default="check", init=False)


@dataclass(frozen=True, kw_only=True)
class CountsOptions:
    "module entries, module types, tap templates and taps per step and rank, by Program status and by route"

    map: str = option("--map")
    config: str = option("--config", help="the checkpoint's config.json")
    tp: int | None = option("--tp", type=int, default=None, help="tensor-parallel size (default: the map's target)")
    cmd: str = field(default="counts", init=False)


@dataclass(frozen=True, kw_only=True)
class IdsOptions:
    "the canonical value ids of one step's taps on one rank"

    map: str = option("--map")
    config: str = option("--config", help="the checkpoint's config.json")
    tp: int | None = option("--tp", type=int, default=None, help="tensor-parallel size (default: the map's target)")
    profile: str = option("--profile", default="000000000000", help="the capture profile id (12 hex)")
    request: str = option("--request", default="r0")
    t: int = option("--t", type=int, default=0, help="the step's first position")
    phase: str = option("--phase", default="prefill", choices=("prefill", "decode"))
    route: list[str] | None = option("--route", action="append", default=None, help="the step's routes (default: all)")
    rank: int | None = option("--rank", type=int, default=None, help="the tensor-parallel rank (`s=`; default: none)")
    cmd: str = field(default="ids", init=False)


COMMANDS = {"check": CheckOptions, "counts": CountsOptions, "ids": IdsOptions}


def main(a: CheckOptions | CountsOptions | IdsOptions) -> int:
    cmap = load(a.map)
    config = json.loads(Path(a.config).read_text())
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
    doc = json.loads(Path(a.inventory).read_text())
    found = check_document(cmap, doc, config, tp, registry.defs, a.rank)
    if errors:
        found["registry"] = list(errors)
    for where, problems in found.items():
        for p in problems:
            print(f"FAIL {where}: {p}")
    sigs = {r: rec.get("signature") for r, rec in doc["ranks"].items()}
    for where, problems in found.items():
        if where.startswith("rank "):
            print(f"{where}: {len(problems)} problems, signature {sigs.get(where[5:])}")
    print(json.dumps(counts(cmap, env), indent=1))
    return 1 if any(found.values()) else 0
