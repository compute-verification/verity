"""Reading a capture map (`verity-vllm/capture-map/v0`) and expanding it into one step's taps on one rank.

A map's shapes and layer sets are expressions over its symbols (`symbols`: `config.json` keys and the tensor-parallel size).
`identities` gives the canonical value id (`verity_vllm.commit.identity`) of every tap a step captures on its routes, and
`committer_names` the tensor name each bound tap is committed under (`commit.name`), which is what the committer's sources
check a step against (`acquire.sources.route_source`).  A tap whose leading dimension is a row count (`row_counts`: the tokens
whose position + 1 is a multiple of a ratio) writes no row on a step where the count is 0, and the step commits nothing under its
name, and on every other step commits exactly the count's rows under it.  From the step's committed positions `empty_names`
recomputes those names and `rows_differ` the names whose committed rows are not the count, for the committer's identity check and
for `bind_names`, which holds one step's committed names to the map.  `counts` totals the taps by Program status and by binder.
Holding a map against a module inventory is `verity_vllm.check.capture_map`.
"""
from __future__ import annotations

import ast
import json
import operator
from pathlib import Path
from typing import Any

from verity_vllm.commit.identity import format_id, value_id

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


def at_layer(pattern: str, layer: int | None) -> str:
    return pattern.replace("{" + LAYER_VAR + "}", str(layer)) if layer is not None else pattern


def route_sites(cmap: dict[str, Any]) -> list[dict[str, Any]]:
    """The calls that decide a step's routes (`acquire.sources.route_source`): `route_site` itself, or each of its `sites`."""
    site = cmap.get("route_site")
    if not site:
        return []
    return list(site["sites"]) if "sites" in site else [site]



def _step_taps(cmap: dict[str, Any], env: dict[str, int], phase: str, routes: tuple[str, ...] | None,
               binders: tuple[str, ...] | None = None, on_target: bool = True):
    """(tap, layer) for every tap one step captures on `routes` (all when None); with `binders`, only the taps one of them binds;
    off the map's target device (`on_target` False), no `target_only` tap."""
    for tap in cmap["taps"]:
        if tap.get("route") is not None and routes is not None and tap["route"] not in routes:
            continue
        if tap.get("target_only") and not on_target:
            continue
        if tap.get("phases") and phase not in tap["phases"]:
            continue
        if binders is not None and (tap.get("commit") or {}).get("binder") not in binders:
            continue
        layers = tap_layers(cmap, tap, env)
        for layer in (layers if layers is not None else [None]):
            yield tap, layer


def tap_value_id(tap: dict[str, Any], layer: int | None, *, profile: str, request: str, t: int, phase: str, rank: int | None) -> str:
    name = tap["layer"] if layer is None else layer
    return format_id(value_id(profile, request, t, phase, name, at_layer(tap["op"], layer), tap["port"], shard=rank))


def identities(cmap: dict[str, Any], env: dict[str, int], *, profile: str, request: str, t: int, phase: str,
               routes: tuple[str, ...] | None = None, rank: int | None = None, binders: tuple[str, ...] | None = None,
               on_target: bool = True) -> list[str]:
    """The canonical value ids of every tap one step captures on one rank (`s=<rank>` when `rank` is given), for the step's
    `routes` (all of them when None); with `binders`, only the taps those committer binders bind (`commit.binder`); off the
    target (`on_target` False), none of the `target_only` taps."""
    return [tap_value_id(tap, layer, profile=profile, request=request, t=t, phase=phase, rank=rank)
            for tap, layer in _step_taps(cmap, env, phase, routes, binders, on_target)]


def committer_names(cmap: dict[str, Any], env: dict[str, int], *, phase: str = "prefill", routes: tuple[str, ...] | None = None,
                    rank: int = 0, binders: tuple[str, ...] | None = None, on_target: bool = True
                    ) -> dict[str, tuple[dict[str, Any], int | None]]:
    """The committer's tensor name (`commit.name`: `{i}` the layer, `{r}` the rank) of every bound tap one step captures on one
    rank -> (tap, layer), the `target_only` ones only on the target.  Two taps of one step under one name is a `MapError`."""
    out: dict[str, tuple[dict[str, Any], int | None]] = {}
    for tap, layer in _step_taps(cmap, env, phase, routes, binders, on_target):
        c = tap.get("commit")
        if c is None:
            continue
        name = at_layer(c["name"], layer).replace("{r}", str(rank))
        if name in out:
            raise MapError(f"taps {out[name][0]['id']} and {tap['id']} are both committed as {name}")
        out[name] = (tap, layer)
    return out


def row_count_of(cmap: dict[str, Any], tap: dict[str, Any]) -> str | None:
    """The row count (`row_counts.ratios`) a tap's leading dimension is, or None."""
    lead = tap["shape"][0] if tap.get("shape") else None
    return lead if isinstance(lead, str) and lead in ((cmap.get("row_counts") or {}).get("ratios") or {}) else None


def row_counts(cmap: dict[str, Any], env: dict[str, int], positions) -> dict[str, int]:
    """Each row count of the map (`row_counts.ratios`: the count's symbol -> its ratio's) on a step whose tokens are at `positions`
    (the values the step commits under the `row_counts.positions` tap): the tokens whose position + 1 is a multiple of the ratio,
    the rows the compress kernels' mask selects, recomputed from the step's own positions.  Empty for a map without row counts."""
    ratios = (cmap.get("row_counts") or {}).get("ratios") or {}
    out: dict[str, int] = {}
    for sym, ratio in ratios.items():
        r = int(evaluate(ratio, env))
        out[sym] = sum(1 for p in positions if (int(p) + 1) % r == 0)
    return out


def empty_names(cmap: dict[str, Any], env: dict[str, int], positions, names: dict[str, tuple[dict[str, Any], int | None]]) -> set[str]:
    """Of a step's `names` (`committer_names`), the ones it writes no row for: those whose tap's leading dimension is a row count
    (`row_count_of`) that is 0 at its `positions` (`row_counts`).  The committer commits no empty tensor, so the step commits none
    of them and every other name: the committer's identity check (`acquire.sources.route_source`) and a reader of a commit
    (`bind_names`) hold a step to that."""
    zero = {s for s, n in row_counts(cmap, env, positions).items() if n == 0}
    return {n for n, (tap, _layer) in names.items() if row_count_of(cmap, tap) in zero} if zero else set()


def rows_differ(cmap: dict[str, Any], env: dict[str, int], positions, names: dict[str, tuple[dict[str, Any], int | None]],
                rows: dict[str, int]) -> dict[str, tuple[int | None, int, int]]:
    """Of a step's committed `names` (`committer_names`), the ones whose tap's leading dimension is a row count (`row_count_of`)
    and whose committed leading dimension (`rows`, name -> the dimension; -1 for a name it does not give) is not that count at the
    step's `positions` (`row_counts`): name -> (its layer, the committed dimension, the count).  The count is the rows the pinned
    kernels write, whatever the phase, the ratio or the compressor state an earlier chunk left (`row_counts`), so a step that
    commits one row more or fewer under such a name than its positions give committed another computation's rows."""
    want = row_counts(cmap, env, positions)
    out: dict[str, tuple[int | None, int, int]] = {}
    for n, (tap, layer) in names.items():
        sym = row_count_of(cmap, tap)
        if sym is not None and int(rows.get(n, -1)) != want[sym]:
            out[n] = (layer, int(rows.get(n, -1)), want[sym])
    return out


def bind_names(cmap: dict[str, Any], env: dict[str, int], names, *, profile: str, request: str, t: int, phase: str,
               routes: tuple[str, ...] | None = None, rank: int | None = None, binders: tuple[str, ...] | None = None,
               on_target: bool = True, positions=None, rows: dict[str, int] | None = None) -> dict[str, Any]:
    """One step's committed tensor names on one rank against the map: `ids`, the value id of every name the map lists; `unknown`,
    the names it does not list; `missing`, the names of bound taps (of `binders`, when given) the step did not commit and writes
    rows for; `empty`, the names it committed and writes no row for; `rows_differ`, the names it committed and writes rows for
    whose committed leading dimension (`rows`, name -> the dimension) is not the count its positions give (`rows_differ`: name ->
    (layer, committed, count)).  Which names a step writes no row for, and how many rows the others hold, is `empty_names` and
    `row_counts` at its `positions` (the values it committed under the `row_counts.positions` tap), which a step with a row-counted
    name must give, and a step that commits one must give `rows` too."""
    want = committer_names(cmap, env, phase=phase, routes=routes, rank=rank or 0, binders=binders, on_target=on_target)
    if positions is None and any(row_count_of(cmap, tap) for tap, _layer in want.values()):
        raise MapError(f"capture map {cmap.get('id')}: a step whose names have row counts needs its positions")
    empty = empty_names(cmap, env, positions, want) if positions is not None else set()
    got = set(names)
    counted = {n: want[n] for n in sorted(got & want.keys() - empty) if row_count_of(cmap, want[n][0])}
    if counted and rows is None:
        raise MapError(f"capture map {cmap.get('id')}: a step that commits row-counted names needs their leading dimensions")
    return {"ids": sorted(tap_value_id(*want[n], profile=profile, request=request, t=t, phase=phase, rank=rank) for n in got & want.keys()),
            "unknown": sorted(got - want.keys()), "missing": sorted(want.keys() - got - empty), "empty": sorted(got & empty),
            "rows_differ": rows_differ(cmap, env, positions, counted, rows) if counted else {}}


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
