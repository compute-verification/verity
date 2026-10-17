"""Where a rank Program's carried collectives are spelt: the collectives whose rank-local operand another module body produced
(`required.Population.carries`).  Two identity encodings:

* `<norm site>` (`norm_sites`): vLLM's `fused_allreduce_rms_norm(hidden_states, residual, norm)` all-reduces a row-parallel output
  inside the norm after it, so the collective is spelt under that norm, at invocation 0: the fused route keeps its ports (`0`, `1`),
  the unfused path's one all-reduced tensor is `NORM_SITE_MEMBER`, and the partial is `part_rank<r>`
  (`acquire.sources.allreduce_norm` commits each under those labels).  The collective is that norm's: the fused route reads
  `<norm>.weight`, and the all-reduce of the unfused path is read by the Calls of `<norm>`, which read `<norm>.weight`;
* `<collective site>` (`collective_sites`): every other carried collective lives at its enclosing module, invocation k.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Callable, Mapping

from verity_vllm.correspondence.reader_for_query import Correspondence
from verity_vllm.query.program_view import ProgramView, param_reads, reads_any_of
from verity_vllm.query.units import ValueRef

__all__ = ["NORM_SITE_MEMBER", "norm_sites", "collective_sites"]

#: the member of a norm site's one output (the unfused path's all-reduce); a fused route keeps its ports `0` / `1`
NORM_SITE_MEMBER = "all_reduce"


def norm_sites(P: ProgramView, corr: Correspondence, record_fields: Mapping[ValueRef, str], carries: set[int], collectives: Mapping[int, str],
               trailing: Callable[[str], int], step_of: Callable[[int], int]) -> tuple[dict[int, str], list[int]]:
    """(carried collective -> its norm module, the carried collectives refused): a fused all-reduce + RMSNorm reading one
    `<norm>.weight` among its `trailing` operands, or a one-output all-reduce that Calls of `<norm>` reading `<norm>.weight` read.  A
    collective with several or no such norm, or a norm two collectives of one step are found for, is refused (never guessed)."""
    norms: dict[int, set[str]] = defaultdict(set)
    unfused: set[int] = set()
    for cid in carries:
        c = P.call(cid)
        t = trailing(c.family)
        if t:
            norms[cid] = {f[: -len(".weight")] for r in c.reads[len(c.reads) - t:] if (f := record_fields.get(r, "")).endswith(".weight")}
        elif collectives[cid] == "all_reduce" and len(c.port_table) == 1:
            unfused.add(cid)
    if unfused:
        for c in P.calls:
            m = corr.module_of(c.id)
            if m and any(record_fields.get(v) == f"{m}.weight" for v in param_reads(c)) and reads_any_of(c, unfused):
                for r in c.reads:
                    if r.call in unfused:
                        norms[r.call].add(m)
    once = Counter((next(iter(ms)), step_of(cid)) for cid, ms in norms.items() if len(ms) == 1)
    out: dict[int, str] = {}
    refused: list[int] = []
    for cid, ms in sorted(norms.items()):
        if len(ms) == 1 and once[(next(iter(ms)), step_of(cid))] == 1:
            out[cid] = next(iter(ms))
        else:
            refused.append(cid)
    return out, refused


def collective_sites(pop: Any) -> dict[str, tuple[str, int]]:
    """site path -> (enclosing module, invocation) for the collective SITES of a rank Program's `required.Population` `pop` (the
    `<collective site>` encoding): the collectives whose rank-local operand another body produced (`pop.carries`) and that are no norm's,
    each at its site path `<enclosing module>.<fx node>` (`pop.site`).  k = the site's order among its enclosing module's sites in the same
    step (Program order of their first Call), from 1 when the enclosing module's own body -- its Calls other than sites -- produces required
    Values that step."""
    if not pop.site:
        return {}
    bodies = {u.id: u.calls for u in pop.B.units}
    ports = pop._ports()
    first: dict[str, int] = {}
    for cid, path in pop.site.items():
        first[path] = min(cid, first.get(path, cid))
    out: dict[str, tuple[str, int]] = {}
    next_k: dict[tuple[str, int], int] = {}
    for f, path in sorted((f, path) for path, f in first.items()):
        parent, step = path.rsplit(".", 1)[0], pop.step_of(f)
        k = next_k.get((parent, step))
        if k is None:
            k = 1 if any(c in ports and c not in pop.site and pop.step_of(c) == step for c in bodies.get(parent, ())) else 0
        out[path] = (parent, k)
        next_k[(parent, step)] = k + 1
    return out
