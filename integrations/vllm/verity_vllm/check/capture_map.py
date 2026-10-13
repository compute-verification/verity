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
target.  Reading a map and expanding one step's taps (`identities`, `committer_names`, `counts`) is `verity_vllm.commit.capture_map`,
which the committer's sources read too.  A tap marked `target_only` runs only on the map's target device (its kernels have no
build for the devices the inventory and the tests run on).

    verity-vllm capture-map check --map M.json --config config.json --inventory inv.json [--rank R]
    verity-vllm capture-map counts --map M.json --config config.json
    verity-vllm capture-map ids --map M.json --config config.json --t 0 --phase prefill --rank 0 [--route R] [--binder B --names]
        [--off-target]
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from verity_vllm.commit.capture_map import (DISPOSITIONS, LAYER_VAR, STATUSES, TP_KINDS, MapError, at_layer, committer_names,
                                            counts, dim, evaluate, identities, layer_set, load, route_sites, symbols, tap_layers,
                                            tap_value_id)
from verity_vllm.commit.identity import NAMED_LAYERS
from verity_vllm.config import option


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
                        problems.append(f"{where}: weight {at_layer(w['param'], layer)} is not in the inventory")
                    break
                if "rows" in w and ten["shape"][-2:-1] != [dim(w["rows"], env, free)]:
                    problems.append(f"{where}: weight {at_layer(w['param'], layer)} has shape {ten['shape']}, the map's rows "
                                    f"{w['rows']} = {dim(w['rows'], env, free)}")
                    break

    for site in route_sites(cmap):
        problems += [f"route_site {k}: unknown route {r!r}" for k in ("true", "false", "decode_tokens") for r in site.get(k, ())
                     if r not in cmap.get("routes", {})]
        if not by_pattern.get(site["module"]):
            problems.append(f"route_site: module {site['module']!r} is not in the inventory")

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
    binder: list[str] | None = option("--binder", action="append", default=None,
                                      help="only the taps this committer binder binds (`commit.binder`; default: every tap)")
    names: bool = option("--names", action="store_true", default=False,
                         help="each bound tap's id and the tensor name the committer binds it as")
    off_target: bool = option("--off-target", action="store_true", default=False,
                              help="a rank off the map's target device: leave out the target_only taps")
    cmd: str = field(default="ids", init=False)


COMMANDS = {"check": CheckOptions, "counts": CountsOptions, "ids": IdsOptions}


def main(a: CheckOptions | CountsOptions | IdsOptions) -> int:
    cmap = load(a.map)
    config = json.loads(Path(a.config).read_text())
    tp = a.tp or cmap["target"]["tensor_parallel_size"]
    env = symbols(cmap, config, tp)
    if a.cmd == "ids":
        step = {"phase": a.phase, "routes": tuple(a.route) if a.route else None, "binders": tuple(a.binder) if a.binder else None,
                "on_target": not a.off_target}
        if a.names:
            for name, (tap, layer) in committer_names(cmap, env, rank=a.rank or 0, **step).items():
                print(tap_value_id(tap, layer, profile=a.profile, request=a.request, t=a.t, phase=a.phase, rank=a.rank), name)
            return 0
        for cid in identities(cmap, env, profile=a.profile, request=a.request, t=a.t, rank=a.rank, **step):
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
