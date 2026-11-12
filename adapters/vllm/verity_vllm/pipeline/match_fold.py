"""One fold of an observation log, for the Match driver's fold worker (`verity-vllm match --worker fold`).
`fold_summary_result` folds the log under a profile and gives the `gates.FOLD_SUMMARY_SCHEMA` summary with the live fold
result.  `persist_program` writes that result into a run directory: the Program (`program.json`), its instance sequence
(`instances.jsonl`, its operand references in the form `instances_form` reads) and the access journal
(`accesses.jsonl.gz`).  `run_input_bindings` binds each request's `run` Inputs of a workload root to the committed
`run_values` identities of the required-value manifest."""

from __future__ import annotations

import importlib
import json
import sys
import time
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

from verity_vllm import settings
from verity_vllm.check import gates as G
from verity_vllm.query.manifest.format import RUN_FAMILY, RUN_PARAM, canonical_identity
from verity_vllm.records.schema_ids import (
    REALIZED_INSTANCES_PROGRESSIONS_SCHEMA,
    REALIZED_INSTANCES_SCHEMA,
    REALIZED_PROGRAM_SCHEMA,
)

__all__ = ["INSTANCES_FORM_ENV", "INSTANCES_SCHEMA", "INSTANCES_SCHEMA_V2", "PROGRAM_SCHEMA", "fold_summary_result",
           "instances_form", "persist_program", "run_input_bindings"]


def _max_rss_gb() -> float:
    import resource
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return ru / (1024 ** 3) if sys.platform == "darwin" else ru / (1024 ** 2)


def fold_summary_result(log_path: str | Path, profile_name: str, *, with_program: bool = True, moe_construction: str | None = None, sampler_construction: str | None = None, registered_values: bool = False) -> tuple[dict, Any]:
    """Fold one log: the `gates.FOLD_SUMMARY_SCHEMA` document and the live `FoldResult` (for `persist_program`).  `moe_construction`: the workload's
    DECLARED construction, which a profile without a `with_moe_construction` hook cannot fold (FAILS by name); `sampler_construction` and `registered_values`: the Programs' (`--program-dir`)."""
    from verity_vllm.check.fold_compare import load_profile
    from verity_vllm.observe.events import Root, RunHeader
    from verity_vllm.observe.fold.fold import fold, summarize
    from verity_vllm.observe.fold.patterns.pouw import with_protocols
    from verity_vllm.observe.fold.patterns.sampling import with_sampler_construction
    from verity_vllm.observe.log import read_log
    from verity_vllm.program.frontend.target_profile import TargetProfile

    t0 = time.time()
    profile = load_profile(profile_name)
    prof_mod = importlib.import_module(f"verity_vllm.observe.fold.profiles.{profile_name}")
    if moe_construction not in (None, TargetProfile.DEFAULT_MOE_CONSTRUCTION):
        hook = getattr(prof_mod, "with_moe_construction", None)
        if hook is None:
            raise SystemExit(f"workload declares moe_construction={moe_construction!r} but profile {profile_name} has no with_moe_construction hook: it cannot fold that construction (refused by name; the MOE-01 fold is NOT a substitute)")
        profile = hook(profile, moe_construction)
    profile = replace(with_sampler_construction(profile, sampler_construction), registered_values=registered_values)
    events = list(read_log(log_path))
    header = next((e for e in events if isinstance(e, RunHeader)), None)
    result = fold(events, with_protocols(profile, header), strict=None, with_program=with_program)   # PoUW's calls: `patterns.pouw`
    summ = summarize(result)
    scope_pattern = next((type(pt).__name__ for pt in profile.patterns if type(pt).__name__.startswith("MoEFusedExperts")), None)
    # roots, classified as resolve_log does (the profile's `root_class`, else `weight_field`)
    classify = getattr(prof_mod, "root_class", None)
    classes: Counter = Counter()
    unknown: list[str] = []
    seen: set[str] = set()
    for ev in events:
        if isinstance(ev, Root) and ev.name not in seen:
            seen.add(ev.name)
            if ev.role in ("weight", "buffer"):
                cls = classify(ev.name) if classify else ("weight" if profile.weight_field(ev.name) else "unknown")
                classes[cls] += 1
                if cls == "unknown":
                    unknown.append(ev.name)
            else:
                classes[ev.role] += 1
    unbound = sorted({n.split("Root ")[1].split(" (")[0].strip("'") for n in result.notes if n.startswith("unbound root:")})
    # per step
    by_step: dict[int, Counter] = {}
    for inst in result.instances:
        by_step.setdefault(int(inst["step"]), Counter())[inst["spec"]] += 1
    step_meta = {int(s["step"]): s for s in result.steps}
    steps_sorted = sorted(set(by_step) | set(step_meta))
    first = steps_sorted[0] if steps_sorted else 0
    per_step = []
    for s in steps_sorted:
        m = step_meta.get(s, {})
        per_step.append({"step": s, "phase": "prefill" if s == first else "decode", "rows": m.get("rows"),
                         "n_instances": sum(by_step.get(s, Counter()).values()), "computes": m.get("computes"),
                         "spec_histogram": dict(sorted(by_step.get(s, Counter()).items())), "unsupported": m.get("unsupported"),
                         "errors": m.get("errors"), "seconds": m.get("seconds")})
    tokens = {str(r): [int(t) for _, t in v] for r, v in result.sampled.items()}
    sampled_steps = {str(r): [int(s) for s, _ in v] for r, v in result.sampled.items()}
    requests = []
    hdr_reqs = list(getattr(result, "requests", None) or ((header.workload or {}).get("requests") if header else None) or [])
    for r in hdr_reqs:
        rid = getattr(r, "request_id", None) if not isinstance(r, dict) else r.get("request_id")
        ptl = getattr(r, "prompt_token_ids", None) if not isinstance(r, dict) else r.get("prompt_token_ids")
        mt = getattr(r, "max_tokens", None) if not isinstance(r, dict) else r.get("max_tokens")
        requests.append({"request_id": rid, "prompt_len": len(ptl or []), "max_tokens": mt, "n_tokens": len(tokens.get(str(rid), []))})
    opaque = sum(1 for e in result.errors if str(e.get("class", "")).lower() in ("opaque", "prescribed"))
    return {
        "schema": G.FOLD_SUMMARY_SCHEMA, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "log": str(log_path),
        "profile": profile_name, "profile_id": profile.id, "with_program": with_program, "digest": result.digest, "spec": summ.get("spec"),
        # the DECLARED construction the scope folded under (None = undeclared = indexed-read) and the pattern that did
        "moe_construction": moe_construction, "moe_scope_pattern": scope_pattern, "sampler_construction": sampler_construction, "registered_values": registered_values,
        "n_instances": len(result.instances), "n_steps": len(per_step), "unsupported": result.report.unsupported,
        "unresolved": result.unresolved, "opaque_fallbacks": opaque, "errors": len(result.errors), "error_classes": summ.get("error_classes"),
        "error_examples": summ.get("error_examples"), "unknown_roots": unknown, "root_classes": dict(classes), "unbound_roots": unbound,
        "per_step": per_step, "tokens": tokens, "sampled_steps": sampled_steps, "requests": requests, "params": summ.get("params"),
        "ret_type": summ.get("ret_type"), "memory": result.memory, "timings": result.timings, "seconds": round(time.time() - t0, 3),
        "max_rss_gb": round(_max_rss_gb(), 3), "fold": summ,
    }, result


PROGRAM_SCHEMA = REALIZED_PROGRAM_SCHEMA
INSTANCES_SCHEMA = REALIZED_INSTANCES_SCHEMA
INSTANCES_SCHEMA_V2 = REALIZED_INSTANCES_PROGRESSIONS_SCHEMA   # operand_refs: "p" runs + node-space PROGRESSIONS ["N", idx, lo, hi, dI, dL, dH, count]
INSTANCES_FORM_ENV = "VERITY_INSTANCES_FORM"      # runs (default, v1) | progressions (v2); every reader accepts both (cb_a.compact.runs_from_file_items)


def instances_form() -> str:
    """The STORED form of `instances.jsonl` operand references.  `runs` = the record form
    (every maximal run one item, the node-space K/V gathers listed run by run: 54 M items / 1.63 GB on a B8 1024/128 row); `progressions`
    = the same references with the node space as compact progressions built from the fold's own Refs objects in O(1) per part (no
    `runs()` expansion at write; ~3 % of the text), parameter-space runs unchanged.  Lossless either way.  The LIBRARY default stays `runs`
    (tests and ad-hoc folds unchanged); the rows of record set VERITY_INSTANCES_FORM=progressions in row_pod.sh's Match phase ("compact
    refs stay compact end to end": readers accept both forms)."""
    v = settings.get(INSTANCES_FORM_ENV)
    v = "runs" if v is None else v
    if v not in ("runs", "progressions"):
        raise SystemExit(f"{INSTANCES_FORM_ENV}={v!r}: expected runs | progressions")
    return v


def persist_program(result: Any, run_dir: Path, *, profile_name: str, log_path: str | Path, journal_level: int = 9) -> dict:
    """Freeze the candidate.  `program.json` is the Verity Program through the existing codec (`verity.circuits.codec`,
    digest == `fold_summary.digest`); `instances.jsonl` is the runtime-correspondence record the fold already holds in
    memory -- one line per Compute instance (step, root event id, pattern, spec id, rows, operand/output `ViewDesc`s, and
    the operand `Refs` as runs) -- and `accesses.jsonl.gz` the full memory-access journal (alloc / root / read / write /
    move / token with event ids).  With these three files a checker can locate every
    operand and result of every instance in the trusted record without importing the profile's patterns."""
    import dataclasses
    import gzip

    from verity.circuits import codec
    from verity.circuits.refs import runs

    from verity_vllm.observe.events import ViewDesc

    # the writer converts by SHAPE, not by a generic walk: a ViewDesc is six known fields (the
    # same dict, in field order, that dataclasses.asdict + _jsonable produced), an operand_refs item is a list of a str and ints already
    # (runs() / cb_a.compact.file_items_from_refs_json), so neither goes through _jsonable's per-element recursion (6.5 M calls on a 32k-instance
    # fold, ~40 % of the writer; at B8 1024/128 the refs alone were 54 M runs x 4 elements).  Output bytes are unchanged (test_compact:
    # test_persist_program_writer_is_byte_identical...); only the meta fields, which may carry non-JSON values, keep the generic conversion.
    def view(v):
        if type(v) is ViewDesc:
            return {"alloc": int(v.alloc), "offset": int(v.offset), "shape": [int(s) for s in v.shape], "strides": [int(s) for s in v.strides],
                    "dtype": v.dtype, "device": v.device}
        return _jsonable(dataclasses.asdict(v)) if dataclasses.is_dataclass(v) and not isinstance(v, type) else _jsonable(v)

    def run_items(rs):      # runs(): [(space, idx, lo, hi), ...] -> the stored lists, ints coerced exactly as _jsonable's tolist() did
        return [[r[0], int(r[1]), int(r[2]), int(r[3])] for r in rs]

    written: dict[str, Any] = {}
    if result.program is not None:
        desc = getattr(result, "descriptor", None)
        if desc is None:      # a result without the fold's own encoding (older producer): encode here, and the digest must agree
            desc = codec.encode_program(result.program)
            digest = codec.program_digest(desc)
            if result.digest and digest != result.digest:
                raise SystemExit(f"program.json digest {digest[:16]} != fold digest {result.digest[:16]}")
        else:                 # the descriptor `_finish` digested: one encoding, program.json's digest is the fold's by construction
            digest = result.digest or codec.program_digest(desc)
        n_root_nodes = len(result.program.root.body.nodes)
        n_input_nodes = n_root_nodes - len(result.body.nodes)
        doc = {"schema": PROGRAM_SCHEMA, "profile": profile_name, "log": str(log_path), "digest": digest,
               "n_input_nodes": n_input_nodes, "n_instances": len(result.body.nodes), "program": desc,
               # the fold's root-parameter layout by NAME (prompt_<i> / seed_<i> / peers_<i> / weights, in parameter order = Input
               # node order for homogeneous parameters) and, on a tensor-parallel rank fold, rank/world + the `peers_<i>` field tables
               # (step, k, kind, site, rows, N, leaf offset) -- cb_a.compare locates every peer leaf range from these, never by guessing
               "params": [[n, repr(t)] for n, t in (getattr(result, "params", None) or ())],
               "tp": getattr(result, "tp", None), "peers_layout": _jsonable(getattr(result, "peers_layout", None) or []),
               "witness_layout": _jsonable(getattr(result, "witness_layout", None) or []),   # the witness parameters by name
               "supplied": _jsonable(getattr(result, "supplied", None) or {})}   # a `registered_values` fold's, with `input_nodes`
        p = run_dir / "program.json"
        _ptext = json.dumps(doc, separators=(",", ":"), sort_keys=True) + "\n"
        p.write_text(_ptext)
        written["program"] = {"path": str(p), "bytes": p.stat().st_size, "digest": digest, "n_input_nodes": n_input_nodes}
        del _ptext
    p = run_dir / "instances.jsonl"
    # the header carries the request table (fold order = prompt parameter order `prompt_<i>` of program.json) with each
    # request's prompt length and number of SERVED samples, and every instance line its `request`/`position` attribution
    # (`fold._attribution`; X-09's `meta` channel reads exactly these keys -- without them here the record never carried them and the
    # decomposition always fell through to the data-flow / row-count channels).
    reqs_tab = []
    sampled = getattr(result, "sampled", None)   # request id -> its served samples; a fold of no sampler has none
    for i, r in enumerate(getattr(result, "requests", None) or []):
        rid = r.get("request_id")
        prompt, served = r.get("prompt_token_ids"), sampled.get(rid) if sampled else None
        reqs_tab.append({"index": i, "request_id": rid, "prompt_len": len(prompt) if prompt else 0,
                         "served_samples": len(served) if served else 0, "max_tokens": r.get("max_tokens")})
    _form = instances_form()
    if _form == "progressions":
        from verity_vllm.program.compact import file_items_from_refs as _file_items
    with p.open("w") as f:
        line = json.dumps({"schema": INSTANCES_SCHEMA_V2 if _form == "progressions" else INSTANCES_SCHEMA, "profile": profile_name,
                           "digest": result.digest, "n_instances": len(result.instances),
                           "note": "root-circuit node of instance k = n_input_nodes + k (program.json)", "requests": reqs_tab,
                           # how operand_refs items are stored: "runs" = every maximal run ["space", idx, lo, hi]; "progressions" =
                           # node- and parameter-space progressions ["N"|"P", idx, lo, hi, dI, dL, dH, count] (cb_a.compact.runs_from_file_items)
                           "operand_refs_form": _form}) + "\n"
        f.write(line)
        _dumps = json.dumps
        for m in result.instances:
            node = result.body.nodes[m["node"]] if result.body is not None and m.get("node") is not None else None
            rec = {k: m.get(k) for k in ("node", "step", "event", "resolution", "pattern", "spec", "row", "rows", "name", "hint", "request", "position")}
            if m.get("operand_roots"):
                rec["operand_roots"] = m["operand_roots"]     # per-operand weight names (multi-weight Computes)
            if m.get("collective"):
                rec["collective"] = m["collective"]           # kind/step/k/site/rank/world + the peers field the row reads
            rec = _jsonable(rec)                              # the meta fields only (small; may carry tuples / numpy scalars / objects)
            operand_views, output_views = m.get("operand_views"), m.get("output_views")   # an instance may list no view of a side
            rec["operand_views"] = [view(v) for v in operand_views] if operand_views else []
            rec["output_views"] = [view(v) for v in output_views] if output_views else []
            if node is not None:
                if _form == "progressions":   # from the fold's own Refs objects: no runs() expansion, no to_json() tree
                    rec["operand_refs"] = [_file_items(arg.refs) for arg in node.args]   # (== file_items_from_refs_json(refs.to_json()))
                else:
                    rec["operand_refs"] = [run_items(runs(arg.refs)) for arg in node.args]
            f.write(_dumps(rec, separators=(",", ":")) + "\n")
    # no compact sidecar is written beside the files of record: a producer-written copy in the record's own directory is prover input
    # the checker cannot authenticate; the compact STORED form above is the file of record itself.
    written["instances"] = {"path": str(p), "bytes": p.stat().st_size, "n": len(result.instances), "operand_refs_form": _form}
    acc = getattr(result, "accesses", None) or []
    if acc:
        p = run_dir / "accesses.jsonl.gz"
        # the journal is 22 % of the fold phase at B8 1024/128 (161 s: jsonable + dumps + gzip; 237 MB) and no
        # Match-pipeline stage reads it (the regression fixtures pin it).  The fold worker under row_pod's
        # MATCH_PIPELINE=shared compresses it at `journal_level` 1 (same lines, ~4x less compressor time, a larger file).
        with gzip.open(p, "wt", compresslevel=journal_level) as f:
            for r in acc:
                # the journal entry (fold._journal): view -> the six fields, refs -> its runs() lists, everything else by the generic conversion;
                # key order is the entry's own (unchanged bytes)
                r2 = {}
                for k, v in r.items():
                    if k == "view":
                        r2[k] = view(v) if v is not None else None
                    elif k == "refs":     # under `progressions` a Refs is stored compact too (a transposed weight read: ~1e6 runs)
                        r2[k] = None if v is None else _file_items(v) if _form == "progressions" and not isinstance(v, list) else run_items(v if isinstance(v, list) else runs(v))
                    else:
                        r2[str(k)] = _jsonable(v)
                f.write(_dumps(r2, separators=(",", ":")) + "\n")
        written["accesses"] = {"path": str(p), "bytes": p.stat().st_size, "n": len(acc)}
    return written


# ---------------------------------------------------------------------------------------------------------
# the root's `run` Inputs
# ---------------------------------------------------------------------------------------------------------

def run_input_bindings(address_map: dict, identities: list[dict], *, rank: int = 0) -> dict:
    """Each request's `run` Input batches of the workload root, bound to the committed `run_values` identities of the required
    manifest (`identities`, `RUN_FAMILY`, rank `rank`).  `run` (`RUN_PARAM`) is a Record of one `Array<leaves, Value<1>>` per field, so
    the root holds it as one Input batch per field in field order (`workload.request_component`), `request[<id>].run[<k>]` (`run`
    alone for one field), whose gates `WorkloadProgram.address_map` gives; field k is the request's row whose `leaves` are the k-th
    of its `run`, and its batch has one gate a leaf.  A request whose rows don't tile its `run` batch by batch, or whose rows the map
    has no `run` for, is refused by name and binds nothing.
    -> {"bindings": [{request_id, input, field, gates, identity, engine_step}], "refused": [str]}"""
    batches: dict[str, list[tuple[int, str, int, int]]] = defaultdict(list)
    for r in address_map["requests"]:
        for i in r["inputs"]:
            name = str(i["name"])
            if name == RUN_PARAM or (name.startswith(RUN_PARAM + "[") and name.endswith("]") and name[len(RUN_PARAM) + 1:-1].isdigit()):
                k = int(name[len(RUN_PARAM) + 1:-1]) if name != RUN_PARAM else 0
                batches[f"r{int(r['index'])}"].append((k, str(i["global"]), int(i["gates"][0]), int(i["gates"][1])))
    fields: dict[str, list[dict]] = defaultdict(list)
    for row in identities:
        if row.get("family") == RUN_FAMILY and int(row.get("rank", 0)) == rank:
            fields[str(row["request_id"])].append(row)
    bindings: list[dict] = []
    refused: list[str] = []
    for rid in sorted(set(batches) | set(fields), key=lambda q: (len(q), q)):
        rows = sorted(fields.get(rid, ()), key=lambda x: int(x["leaves"][0]))
        held = sorted(batches.get(rid, ()))
        if not held:
            refused.append(f"{rid}: {len(rows)} run values and no `{RUN_PARAM}` Input in the address map")
            continue
        spans = [(int(x["leaves"][0]), int(x["leaves"][1])) for x in rows]
        sizes = [(k, b - a) for k, _p, a, b in held]
        tiled = len(spans) == len(held) and [k for k, _ in sizes] == list(range(len(held)))
        lo = 0
        for (a, b), (_k, n) in zip(spans, sizes):
            tiled = tiled and a == lo and b - a == n
            lo = b
        if not tiled:
            refused.append(f"{rid}: its run values' leaves {spans} don't tile its `{RUN_PARAM}` Input batches {sizes} (field k, one gate a leaf)")
            continue
        for row, (k, param, a, b) in zip(rows, held):
            bindings.append({"request_id": rid, "input": param, "field": f"{row['op_path']}/{row['output_member']}", "gates": [a, b],
                             "identity": canonical_identity(row), "engine_step": row.get("engine_step")})
    return {"bindings": bindings, "refused": refused}


_JSON_PRIMS = frozenset((str, int, float, bool, type(None)))


def _jsonable(x: Any) -> Any:
    if type(x) in _JSON_PRIMS:      # the common leaf, one set lookup (subclasses -- IntEnum, numpy scalars -- take the general path below)
        return x
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if hasattr(x, "tolist"):
        return x.tolist()
    return repr(x)
