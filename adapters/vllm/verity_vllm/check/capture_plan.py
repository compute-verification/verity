"""The capture plan: what the serve commits of a row, read from the registered compiler's Programs alone, and how each value is
captured. The serve builds it at its start from the row's Build (`of_build`); the acquisition and the seal's binder read it, and
the window's audit reads the same rows of each request's Program (`captured`).

A plan row (`captured`) is one row of a slot Program that the window commits:

    field       type                          what
    op_path     str                           where the value is bound: the collector's path of its module (`<module>`, or
                                              `<module>/<fx node>` for a call boundary; `runner.sampler` for the token select)
    module      str                           the module whose forward produces it
    members     [(name, width, dtype)]        the committed members, each `width` elements of `dtype` per row
    classes     {member: capture class}       how each member is captured (`CLASSES`)
    row         int                           the request-local row of the op path's member at its step
    step        int                           the request-local step
    family      str                           the Definition's name (`spec_family`); `spec` is its full spec
    statics     {str: Any}                    the Definition's statics

A plan class (`of_build`) is one value the serve acquires, over every row and step of every request Program of the Build:

    field       type                          what
    op_path     str                           as a row's, but a call boundary's `<module>/<op>`, its fx node's count over the
                                              request dropped (`NODE`): the count is the request's, the capture the op's
    member      str                           the member's name; `in` for the prompt ids at the embedding's input
    capture     str                           its capture class (`CLASSES`)
    width       int                           elements per row
    dtype       str                           element type
    definitions [str]                         the Definitions whose rows produce it, sorted
"""
from __future__ import annotations

import gzip
import json
import re
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

from verity_vllm.check.replay.checks import check_rows
from verity_vllm.records.capture import (
    CALL_BOUNDARY,
    CLASSES,
    INPUT_IDS,
    MODULE_OUTPUT,
    RUN_VALUE,
    SAMPLED,
    SCALE_PRODUCTS,
)

#: the query's Value families (`query.required.CallMember.family`) the plan captures, by capture class
CLASS_OF_FAMILY = {"instance_outputs": MODULE_OUTPUT, "call_boundaries": CALL_BOUNDARY, "scale_products": SCALE_PRODUCTS,
                   "run_values": RUN_VALUE}
#: the families no served application captures, with why
NOT_SERVED = {"tp_rank_partials": "a tensor-parallel rank's partial, and the multi-GPU model is not served",
              "moe_block_stream": "a fused-MoE block's plane, and no application serves MoE"}

#: a call boundary's op path: its module, then its fx node, its op counted over the request (`naming`: ``row_quant``,
#: ``row_quant_1``, ...)
NODE = re.compile(r"(.+)/(.+?)(?:_\d+)?")

INPUT_IDS_MEMBER = "in"
INPUT_IDS_DTYPE = "i32"


def captured(pi: Any, rid: str = "") -> list[dict[str, Any]]:
    """The plan rows of a slot Program (its `ProgramIndex`), each with its members' capture classes. A check row (`check_rows`)
    is not served and is left out; any other row with no committed member, or a member of a family `NOT_SERVED` or unknown to
    the plan, is refused by name."""
    checks = {i for _what, i, _lo, _hi in check_rows(pi)}
    rows = []
    for r in pi.rows:
        d = pi.describe(r)
        if d is None or not d["members"]:
            if int(r["i"]) in checks:
                continue
            raise ValueError(f"{rid}: row {r['i']} ({r['spec']}) has no committed member")
        rows.append({**d, "spec": r["spec"], "classes": _classes(pi, r, d, rid)})
    return rows


def _classes(pi: Any, r: Mapping[str, Any], d: Mapping[str, Any], rid: str) -> dict[str, str]:
    if d["op_path"] == _sampler_op():
        return {str(m[0]): SAMPLED for m in d["members"]}
    families = pi.value_families(int(r["i"]))
    out = {}
    for m in d["members"]:
        fam = families.get(str(m[0]))
        if fam in NOT_SERVED:
            raise ValueError(f"{rid}: row {r['i']} ({r['spec']}) commits {d['op_path']}/{m[0]}, {NOT_SERVED[fam]}")
        if fam not in CLASS_OF_FAMILY:
            raise ValueError(f"{rid}: row {r['i']} ({r['spec']}) commits {d['op_path']}/{m[0]} of family {fam!r}, which the "
                             f"capture plan has no class for ({sorted(CLASS_OF_FAMILY)})")
        out[str(m[0])] = CLASS_OF_FAMILY[fam]
    return out


def _sampler_op() -> str:
    from verity_vllm.check.replay.index import SAMPLER_OP
    return SAMPLER_OP


def _index(inst: Mapping[str, Any], program_dir: str) -> Any:
    from verity_vllm.check.replay.index import ProgramIndex
    return ProgramIndex(dict(inst), program_dir=program_dir)


def of_build(build: str | Path, *, index: Callable[[Mapping[str, Any], str], Any] = _index) -> list[dict[str, Any]]:
    """The plan classes of the row whose Build of record is the directory ``build``: every value a row of its request Programs
    (`build_request_*`) commits, and the prompt ids at each Program's embedding, sorted by op path and member. A value two
    rows give another width, dtype or capture class is refused by name, and so are two calls of one op in one module at one
    step, which fold to one class name, and a Build with no request Program."""
    out: dict[tuple[str, str], dict[str, Any]] = {}
    spelled: dict[tuple[str, Any, str], str] = {}

    def add(op: str, member: str, capture: str, width: int, dtype: str, definition: str, where: str) -> None:
        c = out.setdefault((op, member), {"op_path": op, "member": member, "capture": capture, "width": int(width), "dtype": str(dtype),
                                          "definitions": []})
        if (c["capture"], c["width"], c["dtype"]) != (capture, int(width), str(dtype)):
            raise ValueError(f"{where}: {op}/{member} is {capture} {width} {dtype} here and {c['capture']} {c['width']} {c['dtype']} "
                             "in another row")
        if definition not in c["definitions"]:
            c["definitions"].append(definition)

    for d, pi in _programs(build, index):
        for r in captured(pi, d.name):
            for name, width, dtype in r["members"]:
                capture = r["classes"][str(name)]
                op = "/".join(NODE.fullmatch(r["op_path"]).groups()) if capture == CALL_BOUNDARY else str(r["op_path"])
                if spelled.setdefault((d.name, r["step"], op), r["op_path"]) != r["op_path"]:
                    raise ValueError(f"{d.name}: {spelled[(d.name, r['step'], op)]} and {r['op_path']} are two calls of one op at step "
                                     f"{r['step']}, and both fold to the class {op}")
                add(op, str(name), capture, width, dtype, str(r["family"]), d.name)
        embed = pi.embed_op_path()
        if embed is None:
            raise ValueError(f"{d.name}: no row reads the prompt ids")
        add(embed, INPUT_IDS_MEMBER, INPUT_IDS, 1, INPUT_IDS_DTYPE, "", d.name)
    for c in out.values():
        c["definitions"] = sorted(x for x in c["definitions"] if x)
    return [out[k] for k in sorted(out)]


def rows_of_build(build: str | Path, *, index: Callable[[Mapping[str, Any], str], Any] = _index) -> list[dict[str, Any]]:
    """What one engine step of the Build ``build`` commits at each op path: its first plan row (`captured`, request Programs in
    name order) with the members of every row at it, in the order they first appear."""
    out: dict[str, dict[str, Any]] = {}
    for d, pi in _programs(build, index):
        for r in captured(pi, d.name):
            at = out.setdefault(str(r["op_path"]), {**r, "members": []})
            at["members"] += [m for m in r["members"] if m[0] not in {x[0] for x in at["members"]}]
    return list(out.values())


def _programs(build: str | Path, index: Callable[[Mapping[str, Any], str], Any]) -> Iterator[tuple[Path, Any]]:
    dirs = sorted(p for p in Path(build).glob("build_request_*") if (p / "instances.json.gz").is_file())
    if not dirs:
        raise ValueError(f"{build}: no request Program (build_request_*/instances.json.gz)")
    for d in dirs:
        yield d, index(json.loads(gzip.open(d / "instances.json.gz").read()), str(d))


__all__ = ["CALL_BOUNDARY", "CLASSES", "CLASS_OF_FAMILY", "INPUT_IDS", "MODULE_OUTPUT", "NODE", "NOT_SERVED", "RUN_VALUE", "SAMPLED",
           "SCALE_PRODUCTS", "captured", "of_build", "rows_of_build"]
