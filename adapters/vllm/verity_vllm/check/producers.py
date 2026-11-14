"""Which Call produces each member the oracle compare reads, from the required manifest and from the Programs.

`producers_of` reads the manifest's rows; `producers_of_programs` and `program_producer_facts` read the Programs of record
by dataflow; `merge_producers` completes the first by the second; `promote_handed_down` marks the rows whose Value the
producer module's body hands down.  `_family_of` and `_query_family_of` name the family of a binding-map entry, and
`query_families_of` and `runtime_rows_of` read the manifest's identities.  Nothing here reads the Match record.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

NO_ORACLE_FAMILY_PREFIXES = ("fa2.m1/", "fa3.m1/", "moe/")          # kernel-interior streams: no dispatcher-level value in the Match record


def _family_of(entry: dict) -> str:
    ident = entry["identity"]
    if entry.get("kind") == "registered-input" or ident.get("op_path") == "weights":
        return "weights"
    name = str(entry.get("name") or ident.get("op_path") or "")
    if name.startswith(NO_ORACLE_FAMILY_PREFIXES) or str(ident.get("op_path", "")).startswith(NO_ORACLE_FAMILY_PREFIXES) or (m := ident.get("output_member")) == "norm_scale" or m == "router_softmax" or m == "vocab_range" or m == "scale_products":
        return "hidden_stream"                                     # ... and a tap's Call-interior value (query.norm_scales / router_softmax / vocab_range / scale_products MEMBER): no Match value
    if ident.get("output_member") == "sampled_token_ids":
        return "sampled_token_ids"
    if ident.get("output_member") == "in":
        return "input_token_ids"
    return "instance_outputs"


def _query_family_of(entry: dict) -> str:
    """The Q(P) family (required-manifest vocabulary) a binding-map entry belongs to -- `fa2.m1/` -> fa2_hidden_m1_stream,
    `fa3.m1/` -> fa3_hidden_m1_stream, `moe/` -> moe_block_stream, registered inputs -> weights; the oracle's own coarse classes
    (`_family_of`: weights / hidden_stream / ids / instance_outputs) stay what the verdict reads, this is the label the POPULATION uses."""
    ident = entry["identity"]
    if entry.get("kind") == "registered-input" or ident.get("op_path") == "weights":
        return "weights"
    name = str(entry.get("name") or ident.get("op_path") or "")
    op = str(ident.get("op_path", ""))
    for pref, qf in (("fa2.m1/", "fa2_hidden_m1_stream"), ("fa3.m1/", "fa3_hidden_m1_stream"), ("moe/", "moe_block_stream")):
        if name.startswith(pref) or op.startswith(pref):
            return qf
    return _family_of(entry)


def query_families_of(manifest: dict[str, Any] | None) -> dict[str, int] | None:
    """Identities per Q(P) family from the required manifest's rows (the manifest's own identity rule -- an M1 stream identity is one per
    (layer, ENGINE step), a moe plane one per (plane, request, step); binding ENTRIES are more numerous than either).  None when no
    manifest was supplied (the statement then says so rather than counting entries as identities)."""
    if not isinstance(manifest, dict) or not isinstance(manifest.get("identities"), list):
        return None
    return dict(Counter(str(r.get("family")) for r in manifest["identities"]))


def manifest_rows(manifest: dict[str, Any] | None) -> list[dict[str, Any]]:
    """The required manifest's rows (`identities`): none where no manifest was supplied, and for a manifest that states none."""
    rows = manifest.get("identities") if manifest else None
    return rows if rows else []


def runtime_rows_of(manifest: dict[str, Any] | None) -> dict[tuple, tuple[int, int]]:
    """(step, request, op_path, member, invocation) -> (r0, rows) of each identity whose Values are the runtime's rows r0.. (`runtime_rows`)."""
    return {(int(r["step"]), str(r["request_id"]), str(r["op_path"]), str(r.get("output_member", "out")), int(r.get("invocation", 0))):
            (int(r["runtime_rows"][0]), int(r["runtime_rows"][1]) - int(r["runtime_rows"][0])) for r in manifest_rows(manifest) if r.get("runtime_rows")}


def producers_of(manifest: dict[str, Any] | None) -> dict[tuple[str, str], dict[str, Any]]:
    """(op_path, output_member) -> {"spec": producer family base, "ordinal": producer_ordinal | None, "operand": producer_operand | None}
    from the required manifest's rows (`spec` is the Program op family that produces the member; `producer_ordinal` the optional index
    among same-family producers per row; `producer_operand` the family whose output the producer consumes -- the order-free selector).
    A PROMOTED row additionally carries `promoted: True` and `consumers` (the reading modules, Program order, unioned over the
    member's rows): the oracle resolves it through the Value's dataflow, not the hint prefix (MatchOracle._promoted_instances)."""
    out: dict[tuple[str, str], dict[str, Any]] = {}
    consumers: dict[tuple[str, str], list[str]] = {}
    for r in manifest_rows(manifest):
        if r.get("family") not in ("instance_outputs", "call_boundaries") or not r.get("spec"):
            continue
        key = (str(r.get("op_path")), str(r.get("output_member")))
        cur = out.get(key)
        # a PROMOTED row (`promoted: true`, `consumers` = the reading modules, Program order) is resolved by the
        # oracle through the Value's dataflow (MatchOracle._promoted_instances); `promoted` is part of the producer's identity here (a
        # promoted and a vocabulary row on one member is a conflict), the consumer lists of the rows are unioned in first-seen order
        ent = {"spec": str(r["spec"]).split("{", 1)[0], "ordinal": r.get("producer_ordinal"), "operand": r.get("producer_operand")}
        if r.get("port") is not None:
            ent["port"] = str(r["port"])
        if r.get("promoted"):
            ent["promoted"] = True                    # a vocabulary row's entry is byte-for-byte what it was (no key added)
        if r.get("family") == "call_boundaries":
            ent["interior"] = True
        if cur is None:
            out[key] = ent
        elif cur != ent:
            out[key] = {"spec": None, "ordinal": None, "operand": None, "conflict": True}          # two producers claim one member: no filter, fail visibly
            consumers.pop(key, None)
            continue
        if ent.get("promoted"):
            cl = consumers.setdefault(key, [])
            named = r.get("consumers")   # the reading modules the promoted row names; a row that names none adds none
            for c in named if named else ():
                if str(c) not in cl:
                    cl.append(str(c))
    for key, cl in consumers.items():
        if out[key].get("promoted"):
            out[key]["consumers"] = cl
    return out


def producers_of_programs(programs: dict[str, dict[str, Any]], aliases: dict[str, Any] | None = None, query: str | None = None) -> dict[tuple[str, str], dict[str, Any]]:
    """(op_path, output_member) -> {"spec", "operand", "consumers"} read from the Programs of record by dataflow, under `query`, the one
    the query path built the manifest with (`required.population`, the Correspondence `query.cli` reads; None: the query of record) --
    such a manifest carries no producer annotations.  Per required `instance_outputs` Value of every producing Call: `spec` = the Call's family; `operand` = the
    family of the Call whose output its FIRST operand reads (the order-free selector among same-family instances of one module: a fused
    norm's two narrowings read AddWidenedBf16 -> `1` and MulVecF32 -> `0`); `consumers` = the modules of the Calls that read the Value
    outside its own module, Program order (the dataflow resolution when the fold observed the producer under another path).  A fact the
    member's rows disagree on is None (the oracle then names the ambiguity; never guessed); a family disagreement is a `conflict`.
    `programs`: request label -> `manifest_format.load_program(dir)`; `aliases`: the manifest's `op_path_aliases` (alias -> [canon, k])."""
    from verity_vllm.correspondence.reader_for_query import Correspondence
    from verity_vllm.query.program_view import from_instances

    out: dict[tuple[str, str], dict[str, Any]] = {}
    seen: set[int] = set()
    for prog in programs.values():
        inst = prog["instances"]
        if id(inst) in seen:
            continue
        seen.add(id(inst))
        res = prog.get("result")   # the Build's result.json; a Program handed over without one names no implementation path and no rank
        P = from_instances(inst)
        corr = Correspondence.of(P, prog.get("dir"), implementation_paths=Correspondence.implementation_paths_of(res))
        tpinfo = t if res and isinstance(t := res.get("tp"), dict) else {}
        world = int(tpinfo.get("world") or 1)
        program_producer_facts(P, corr, tp=({"world": world, "rank": int(tpinfo.get("rank") or 0)} if world > 1 else None), aliases=aliases, into=out, query=query)
        del P, corr
    return out


def program_producer_facts(P: Any, corr: Any, *, tp: dict[str, Any] | None = None, aliases: dict[str, Any] | None = None,
                           into: dict[tuple[str, str], dict[str, Any]] | None = None, query: str | None = None) -> dict[tuple[str, str], dict[str, Any]]:
    """`producers_of_programs` for one ProgramView + Correspondence, accumulated into `into` (the member's facts over every Program)."""
    from verity_vllm.query.required import QUERY_OF_RECORD, population

    canon = {str(k): str(v[0]) for k, v in (aliases or {}).items() if isinstance(v, (list, tuple)) and v}
    out = {} if into is None else into
    pop = population(P, corr, tp=tp, query=query or QUERY_OF_RECORD)
    members = {cid: [cm.member for cm in pop.members_of(cid) if cm.family == "instance_outputs"] for cid in pop.producers()}
    members = {cid: ms for cid, ms in members.items() if ms}
    readers: dict[int, list[int]] = defaultdict(list)
    for c in P.calls:
        for r in c.reads:
            if r.call >= 0 and r.call in members and (not readers[r.call] or readers[r.call][-1] != c.id):
                readers[r.call].append(c.id)
    for cid, ms in members.items():
        c = P.call(cid)
        own = corr.module_of(cid)
        path = corr.implementation_path(pop.program_path(cid))
        path = canon.get(path, path)
        first = next(iter(c.reads), None)
        operand = P.call(first.call).family if first is not None and first.call >= 0 else None
        cons: list[str] = []
        for rc in readers.get(cid, ()):
            m = corr.module_of(rc)
            if m != own:
                m = corr.implementation_path(m)
                if m not in cons:
                    cons.append(m)
        for mem in ms:
            key = (path, mem)
            cur = out.get(key)
            if cur is None:
                out[key] = {"spec": c.family, "operand": operand, "consumers": list(cons)}
                continue
            if cur.get("conflict"):
                continue
            if cur["spec"] != c.family:
                out[key] = {"spec": None, "operand": None, "consumers": [], "conflict": True}
                continue
            if cur["operand"] != operand:
                cur["operand"] = None
            for m in cons:
                if m not in cur["consumers"]:
                    cur["consumers"].append(m)
    return out


def merge_producers(declared: dict[tuple[str, str], dict[str, Any]], derived: dict[tuple[str, str], dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    """The manifest's producer entries (`producers_of`) completed by the Program-derived facts (`producers_of_programs`) where the
    manifest row declares no selector of its own: `operand` (when the manifest gives neither `producer_operand` nor `producer_ordinal`)
    and `consumers` (when it is not a promoted row).  Only for members both name with the SAME family; a manifest entry is never
    overridden, a member the manifest does not name is never added."""
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for key, ent in declared.items():
        d = derived.get(key)
        if not d or d.get("conflict") or ent.get("conflict") or not ent.get("spec") or d.get("spec") != ent.get("spec"):
            out[key] = ent
            continue
        ent = dict(ent)
        if d.get("operand") and ent.get("operand") is None and ent.get("ordinal") is None:
            ent["operand"] = d["operand"]
        if d.get("consumers") and not ent.get("promoted"):
            ent["consumers"] = list(d["consumers"])
        out[key] = ent
    return out


def promote_handed_down(manifest: dict[str, Any] | None, derived: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    """Mark, in place, the `instance_outputs` rows of a `v2-query` manifest whose Value the producer module's body hands DOWN: every module
    the Programs name as reading it (`derived[...]["consumers"]`, `producers_of_programs`) is inside the producer's module subtree
    (`required.under`), so it is no return of that module.  Such a row gets `promoted: true` + those `consumers` (Program order) and the
    acquisition observes it at its first consumer's input (`manifest_format.PROMOTED_OBSERVATION_RULE`) instead of hooking the module's
    forward output: row #57's `model/out` is Gemma2's embed scale (the model body's `hidden *= normalizer`, read by `model.layers.0`)
    while `Gemma2Model` returns the final norm's output -- the two differ from element 0 (form (B): 432
    mismatches, 159,408 equal).  A v2 manifest carries no `promoted` flag (`required`: such a Value simply crosses the body boundary).
    Only rows whose family the Programs agree on (`spec` equal to the row's, no conflict) and that are not already promoted; a member
    read by any module outside the subtree is a return and stays a module output.  Returns {"members": {"op/member": consumers}, "rows": n}."""
    from verity_vllm.query.required import under

    picked: dict[tuple[str, str], list[str]] = {}
    for key, d in derived.items():
        named = d.get("consumers")   # the modules the Programs name as reading the Value; a fact that names none hands nothing down
        cons = [str(c) for c in named] if named else []
        if cons and not d.get("conflict") and all(under(c, key[0]) for c in cons):
            picked[key] = cons
    rows = 0
    members: dict[str, list[str]] = {}
    for r in manifest_rows(manifest):
        if r.get("family") != "instance_outputs" or r.get("promoted"):
            continue
        key = (str(r.get("op_path")), str(r.get("output_member")))
        consumers = picked.get(key)
        if consumers is None or str(derived[key].get("spec") or "") != str(r.get("spec") or "").split("{", 1)[0].strip():
            continue
        r["promoted"] = True
        r["consumers"] = list(consumers)
        members[f"{key[0]}/{key[1]}"] = list(consumers)
        rows += 1
    return {"members": members, "rows": rows}
