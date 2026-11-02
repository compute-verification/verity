"""Derive once, relabel per rank: rank r's derive report from the derivation of a template rank t of the same TP world.

Two ranks' exported graphs differ in two kinds of argument only (design note `build-derive-once-relabel` §1):

* each collective (`verity_tp::all_reduce2` / `all_gather2`, and `all_reduce` / `all_gather` over the world's `parts`) lists its
  operands in rank order, the rank's own partial at its own index and the peer inputs `peers_<k*(W-1)+j>` around it in rank order:
  from rank t to rank r the own operand moves from index t to index r and the peers keep their order;
* each `verity_tp::embedding_shard(ids, shard, START, END)` carries the rank's vocab range, which the caller supplies from vLLM's own
  range function (`export_ops.rank_embedding_shards`) as a map from the template's (START, END) to rank r's.

`rank_report` rewrites those arguments in the template's fx graph, re-emits only those nodes with the rule that lowered them (the
first applicable rule, as `derive._translate_node` picks it), each Call written into the body slot the template's emission used,
of the same form, output type, gate count and name; the arguments read as the template read them.  Their application records are
rebuilt by the rule and take the template's places.  Then the derivation's tail (`derive._finish_report`) runs over the re-emitted
body with `rank=(W, r)`, so every byte of rank r's report is produced by the code that produces a real derive's.  The template's
state is restored afterwards.  Anything outside those two facts raises `RelabelRefused`, and the caller derives rank r for real;
a relabel is trusted only as far as the Build's guard has compared it with a real derive (`pipeline.tp.relabel_build`).
"""
from __future__ import annotations

from typing import Any

import torch

from verity.ml.circuits.defs import Body, Builder
from verity.ml.circuits.refs import Affine, Coll, runs
from verity.ml.circuits.types import Value
from verity_vllm.program.frontend.derive import _finish_report, _flat_definition, fx_arg_nodes
from verity_vllm.program.frontend.report import DeriveReport, DerivationTail

COLLECTIVES = ("all_reduce2", "all_gather2", "all_reduce", "all_gather")
EMBEDDING = "embedding_shard"


class RelabelRefused(RuntimeError):
    """The template's derivation has something the relabel does not reproduce: derive the rank for real."""


def _op(node) -> str | None:
    """The `verity_tp` op name of fx `node`, None for any other node."""
    t = node.target if node.op == "call_function" else None
    if isinstance(t, torch._ops.OpOverload) and t.namespace == "verity_tp":
        return t.__name__.split(".")[0]
    return None


def tp_sites(ctx) -> list:
    """The template graph's `verity_tp` nodes, in graph order; an op the relabel does not know is refused."""
    sites = []
    for node in ctx.ep.graph.nodes:
        op = _op(node)
        if op is None:
            continue
        if op not in COLLECTIVES and op != EMBEDDING:
            raise RelabelRefused(f"{node.name}: verity_tp::{op} is not an op the relabel rewrites")
        sites.append(node)
    return sites


def _operands(node) -> tuple[list, bool]:
    """A collective's operands in rank order, and whether they are one `parts` list (world > 2) or two positional arguments."""
    if isinstance(node.args[0], (list, tuple)):
        return list(node.args[0]), True
    return list(node.args[:2]), False


def _rank_args(ctx, node, t: int, r: int, world: int, shards: dict[tuple[int, int], tuple[int, int]]) -> tuple:
    """`node`'s arguments at rank r, from the template's at rank t."""
    op = _op(node)
    if op == EMBEDDING:
        ids, table, start, end = node.args[:4]
        if (start, end) not in shards:
            raise RelabelRefused(f"{node.name}: no rank-{r} vocab range for the template's ({start}, {end})")
        return (ids, table, *shards[(start, end)], *node.args[4:])
    ops, listed = _operands(node)
    if len(ops) != world:
        raise RelabelRefused(f"{node.name}: {len(ops)} operands in a world of {world}")
    own = [i for i, a in enumerate(ops) if not isinstance(a, torch.fx.Node) or ctx.is_input_storage(a) is None]
    if own != [t]:
        raise RelabelRefused(f"{node.name}: the rank's own operands are at {own}, not at the template's rank {t}")
    x = ops.pop(t)
    ops.insert(r, x)
    return (ops, *node.args[1:]) if listed else (*ops, *node.args[2:])


def shard_map(template: list[dict], rank: list[dict]) -> dict[tuple[int, int], tuple[int, int]]:
    """Template (START, END) -> rank (START, END), over the two ranks' `embedding_shards` records (same modules, same order)."""
    if [d["module"] for d in template] != [d["module"] for d in rank]:
        raise RelabelRefused(f"the embedding shards' modules differ: {[d['module'] for d in template]} vs {[d['module'] for d in rank]}")
    out: dict[tuple[int, int], tuple[int, int]] = {}
    for a, b in zip(template, rank):
        k, v = (int(a["START"]), int(a["END"])), (int(b["START"]), int(b["END"]))
        if out.setdefault(k, v) != v:
            raise RelabelRefused(f"the template's vocab range {k} maps to {out[k]} and {v}")
    return out


def rank_report(session: DerivationTail, r: int, *, shards: dict[tuple[int, int], tuple[int, int]]) -> DeriveReport:
    """Rank `r`'s derive report from the template's derivation `session` (see module docstring).  Raises `RelabelRefused`."""
    if session.rank is None:
        raise RelabelRefused("the template is not a TP rank derivation")
    world, t = int(session.rank[0]), int(session.rank[1])
    if world < 2 or not 0 <= r < world:
        raise RelabelRefused(f"rank {r} of a world of {world}")
    if session.family is not None:
        raise RelabelRefused(f"the template's Program is family {session.family.get('rule')!r}'s: no correspondence to relabel")
    ctx, body = session.ctx, session.flat.body
    sites = tp_sites(ctx)
    site_set = set(sites)
    slots: dict[Any, list[int]] = {n: [] for n in sites}
    for k, fx in ctx.node_fx.items():
        if fx in site_set:
            slots[fx].append(k)
    app_at: dict[str, list[int]] = {n.name: [] for n in sites}
    for i, a in enumerate(ctx.applications):
        if a["node"] in app_at:
            app_at[a["node"]].append(i)
    saved = {"args": [(n, n.args) for n in sites], "applications": ctx.applications, "occurrence_bindings": ctx.occurrence_bindings,
             "unresolved": ctx.unresolved_module_paths, "current": ctx.current}
    nodes = list(body.nodes)
    applications, bindings = list(ctx.applications), dict(ctx.occurrence_bindings)
    ctx.applications, ctx.occurrence_bindings = applications, bindings
    try:
        for node in sites:
            if len(app_at[node.name]) != 1:
                raise RelabelRefused(f"{node.name}: {len(app_at[node.name])} application records, not one")
            _reemit(ctx, node, _rank_args(ctx, node, t, r, world, shards), sorted(slots[node]), app_at[node.name][0], nodes, applications)
        ctx.unresolved_module_paths = dict(session.unresolved)
        flat = _flat_definition(session.name, Body(body.params, body.ret_type, nodes, body.ret, body.offsets),
                                session.flat.bindings["SRC"], session.mode)
        rep = _finish_report(flat, ctx, dict(session.head), mode=session.mode, ruleset=session.ruleset, rank=(world, r),
                             check_registry=session.check_registry, dead_free=session.dead_free)
    finally:
        for n, a in saved["args"]:
            n.args = a
        ctx.applications, ctx.occurrence_bindings = saved["applications"], saved["occurrence_bindings"]
        ctx.unresolved_module_paths, ctx.current, ctx.emit_into = saved["unresolved"], saved["current"], None
    if rep.family is not None:
        raise RelabelRefused(f"rank {r}'s Program matched family {rep.family.get('rule')!r}, the template's did not")
    return rep


class _Slots:
    """`Ctx.emit_into` while fx `node` is re-emitted: each call is written into the next body slot the template's emission of
    `node` used, which must hold a node of the same form, output type, gate count and name."""

    def __init__(self, ctx, node, slots: list[int], nodes: list) -> None:
        self.ctx, self.node, self.slots, self.nodes, self.used = ctx, node, slots, nodes, 0

    def __call__(self, fn, args, fx, rule, name) -> Coll:
        if fx is not self.node or self.used == len(self.slots):
            raise RelabelRefused(f"{self.node.name}: the re-emission makes a call the template's emission did not ({getattr(fn, 'id', fn)})")
        k = self.slots[self.used]
        self.used += 1
        scratch = Builder((), Value(1))
        try:
            scratch.call(fn, *args, name=name)
        except TypeError as e:
            raise RelabelRefused(f"{self.node.name}: {e}") from None
        new, old = scratch.nodes[0], self.nodes[k]
        if (new.form, new.out, new.name, new.gates) != (old.form, old.out, old.name, old.gates) or (name is not None and self.ctx.node_hints.get(k) != name):
            raise RelabelRefused(f"{self.node.name}: node {k} re-emits as {new.fn.id} {new.out} {new.name!r}, the template's is "
                                 f"{old.fn.id} {old.out} {old.name!r}")
        self.nodes[k] = new
        return Coll(new.out, Affine("n", k, 0, new.out.leaves))


def _reemit(ctx, node, args: tuple, slots: list[int], app_i: int, nodes: list, applications: list) -> None:
    """Re-emit fx `node` with arguments `args` through its rule, into `nodes[slots]`, its application record at `applications[app_i]`."""
    frozen = ctx.occurrence_bindings.get(node)
    if frozen is None:
        raise RelabelRefused(f"{node.name}: no bindings were frozen at its lowering")
    as_read = {a: frozen["in"].get(j) for j, a in fx_arg_nodes(node)}
    template_out = frozen["out"]
    node.args = args
    keys = list(as_read) + [node]
    env = {a: ctx.env[a] for a in keys if a in ctx.env}
    stale = {a: ctx.stale.pop(a) for a in keys if a in ctx.stale}
    writer = _Slots(ctx, node, slots, nodes)
    try:
        ctx.env.update(as_read)
        ctx.current = node
        ctx.emit_into = writer
        rule = next((ru for ru in ctx.ruleset.for_target(node.target, ctx.mode) if ru.applies(node, ctx) is True), None)
        if rule is None:
            raise RelabelRefused(f"{node.name}: no rule applies to the rank's arguments")
        n_apps = len(applications)
        k0, before = ctx.begin_emit(node)
        rule.emit(node, ctx)
        ctx.note_emitted(node, k0, before)
        if writer.used != len(slots):
            raise RelabelRefused(f"{node.name}: the re-emission made {writer.used} calls, the template's {len(slots)}")
        added = applications[n_apps:]
        del applications[n_apps:]
        if len(added) != 1 or added[0]["rule"] != applications[app_i]["rule"] or added[0]["node"] != node.name:
            raise RelabelRefused(f"{node.name}: the re-emission recorded {[a['rule'] for a in added]}, the template {applications[app_i]['rule']}")
        applications[app_i] = added[0]
        out = ctx.env.get(node)
        if not isinstance(out, Coll) or not isinstance(template_out, Coll) or out.type != template_out.type or runs(out.refs) != runs(template_out.refs):
            raise RelabelRefused(f"{node.name}: the re-emitted value is not the template's")
    finally:
        ctx.emit_into = None
        for a in keys:
            if a in env:
                ctx.env[a] = env[a]
            else:
                ctx.env.pop(a, None)
        ctx.stale.update(stale)


__all__ = ["RelabelRefused", "rank_report", "shard_map", "tp_sites"]
