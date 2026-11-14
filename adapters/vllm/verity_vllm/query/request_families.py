"""The families of a request manifest that are not a module's required Values, and what the query header says of the families a
manifest holds.  `required.request_manifest` writes each through its row adder
(`add(family, step, invocation, op_path, member, lo, hi, **extra)`):

* the values the served run decides (`run_value_rows`): one request-owned identity per `<op_path>/<member>` field of the Program's
  `fmt.RUN_PARAM` input, at the step whose forward reads it;
* the MoE block planes (`moe_plane_rows`): kernel geometry, per (MoE body, step);
* the header's `required_families`, policy lines and TP block for the families present (`header_families`).
"""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from verity_vllm.correspondence.reader_for_query import Correspondence
from verity_vllm.query.manifest import format as fmt
from verity_vllm.query.program_view import CallView, ProgramView, param_reads
from verity_vllm.query.units import ValueRef

__all__ = ["run_value_rows", "moe_plane_rows", "header_families"]


def run_value_rows(P: ProgramView, corr: Correspondence, step_of: Callable[[int], int], add: Callable[..., None],
                   unmodelled: Counter) -> None:
    """The values the served run decides (`fmt.RUN_PARAM`): one request-owned identity per `<op_path>/<member>` field, at its
    readers' step.  A field no step's forward reads, or several steps' do, is unmodelled."""
    impl = corr.implementation_path
    run = {ValueRef(p.id, port.name): port for p in P.params if p.name == fmt.RUN_PARAM for port in p.port_table if "/" in port.name}
    if run:
        read_at: dict[ValueRef, set[int]] = defaultdict(set)
        reader: dict[ValueRef, CallView] = {}
        for c in P.calls:
            for r in param_reads(c):
                if r in run:
                    read_at[r].add(step_of(c.id))
                    reader.setdefault(r, c)
        for v, port in run.items():
            steps = sorted(read_at.get(v, ()))
            if len(steps) != 1:
                unmodelled[f"run value {port.name}: read at steps {steps} (one step's forward reads it)"] += 1
                continue
            op, member = port.name.rsplit("/", 1)
            add(fmt.RUN_FAMILY, steps[0], 0, impl(op), member, 0, port.elems, dtype="u8", spec=reader[v].family, leaves=[port.lo, port.hi],
                attention_module=impl(corr.module_of(reader[v].id)), statics=dict(reader[v].statics))


def moe_plane_rows(P: ProgramView, events: Sequence[int], impl: Callable[[str], str], add: Callable[..., None]) -> set[int]:
    """The MoE block planes (kernel geometry, v1's), per (MoE body, step); returns the layers that have them."""
    moe_layers: set[int] = set()
    lo = 0
    for step, ev in enumerate(events):                     # the step's Calls are the body slice up to and including its sampling event
        seg = P.calls[lo: ev + 1]
        lo = ev + 1
        for mod, g in fmt.moe_geometry({"spec": c.definition, "name": c.name} for c in seg).items():
            layer = fmt.layer_of(mod)
            if g["M"] <= 0 or g["TOPK"] <= 0:
                continue
            moe_layers.add(layer)
            for plane, nw, dtype in fmt.moe_identities(g):
                add("moe_block_stream", step, 0, f"moe/L{layer}", plane, 0, nw, dtype=dtype, rows=g["M"], topk=g["TOPK"], moe_module=impl(mod))
    return moe_layers


def header_families(q: dict[str, Any], pop: Any, world: int, moe_layers: set[int], fam_counts: Mapping[str, int]) -> None:
    """Adds to the query header `q` what the families of the manifest of `pop` (a `required.Population`) have it say: a rank Program's
    partials and its TP block, the MoE layers, the sampler's inputs and the run's values."""
    if world > 1:
        q["required_families"].append("tp_rank_partials")
        q["policy"]["protocol_required"].append(f"TP-04 (world {world}): every collective output and its rank-local partial, with every "
                                                "port of the partial's producer")
        q["tp"] = {"world": world, "collectives": len(pop.collectives), "peer_inputs": len(pop.peer_inputs)}
        if pop.witness:
            q["tp"]["witness_inputs"] = {fmt.SCHEDULE_PARAM: {"readers": len(pop.witness_readers),
                                                              "binding": "each reader's own committed rank (port 0)"}}
    if moe_layers:
        q["required_families"].append("moe_block_stream")
        q["moe_layers"] = sorted(moe_layers)
    if any(f[:8] == "sampler_" for f in fam_counts):
        q["required_families"] += sorted(f for f in fam_counts if f[:8] == "sampler_")
    if fam_counts.get(fmt.RUN_FAMILY):
        q["required_families"].append(fmt.RUN_FAMILY)
        q["policy"]["protocol_required"].append(f"every `{fmt.RUN_PARAM}` field `<op_path>/<member>`: the request's value of the step "
                                                f"whose forward reads it (leaves = its leaf range in `{fmt.RUN_PARAM}`)")
