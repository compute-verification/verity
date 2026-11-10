"""(T0) The Program-side step segmentation of every request component (GP-01; X-09's unit): `workload.request_component` cuts a request
body into request-relative steps, one segment per sampling event (the instance that selects the step's token), and the workload
Program's chronology programs segment s of request r for engine step `arrive_step + s`.  A rule that does not name the row's sampling
family leaves the body as ONE segment: every step past s = 0 has no programmed instance set (`chronology.json` member `nodes: null`) and
`component_steps_equal_cap` is false -- v1's rule named `TokenSelect_v1` only, so a seeded-Gumbel row records exactly that.

Lifted from the row's OWN documents: per declared request (`build_workload/workload_program.json` `requests[*]`) its cap, component
digest, `n_nodes`, `steps`, sha256 of `step_nodes`, and the component's sampling events as its Build counted them (`result.json`
`spec_histogram`, the families of `verity_vllm.program.sampling_event`); from `build_workload/chronology.json` the flag and the number of
members with no segment.  Recomputed: `request_component` over each component's `descriptor.json.gz` with the CURRENT code, the
sampling events counted in the decoded body, the chronology facts by `WorkloadProgram.chronology`'s rule (segment s exists iff
s < len(step_nodes)).  In candidate mode the candidate's own build_workload/ documents are lifted the same way.

`actual` also states `consistent` (not in the contract, not diffed): steps == cap == sampling events per request and every segment
ending at a sampling event -- what a correct segmentation of a one-token-per-step request is, whichever side of the diff it is on.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os

from .. import resolver

NAME = "step_segmentation"
TIER = "T0"          # inputs: build_workload/ + the components' descriptors (MBs); the decode is CPU-seconds to minutes per component
STRENGTH = "strong"
REBUILD = True       # every component decoded with the current code (#73: 9 descriptors, 228 MB, 542 s): the post-train job's, like a rebuild
INPUTS = ("build_workload/workload_program.json", "build_workload/chronology.json", "build_request*/result.json", "build_request*/descriptor.json.gz")

_KEYS = ("requests", "component_steps_equal_cap", "unsegmented_members")


def applies(row: resolver.Row) -> str | None:
    if not (row.has("build_workload/workload_program.json") and row.has("build_workload/chronology.json")):
        return "no build_workload/workload_program.json + chronology.json on this row (no workload Program composed)"
    return None


def _sha(obj) -> str:
    return hashlib.sha256(resolver.canonical(obj).encode()).hexdigest()


def component_dirs(wp: dict) -> dict[str, str]:
    """`"<prompt_len>,<T>"` -> the row-relative Build dir of that request shape (`inputs.derived` records absolute pod paths; the row dir
    is their parent, so the basename is the row-relative dir: `build_request`, `build_request_LP27_T100`, ...)."""
    return {k: os.path.basename(str(v).rstrip("/")) for k, v in ((wp.get("inputs") or {}).get("derived") or {}).items()}


def sampling_events(hist: dict) -> int:
    from verity_vllm.program import sampling_event as SE
    return sum(int(n) for spec, n in (hist or {}).items() if SE.is_sampling_event(spec))


def _dir_of(r: dict, dirs: dict[str, str]) -> str | None:
    return dirs.get(f"{int(r['prompt_len'])},{int(r['cap']) - 1}")


def lift(wp: dict, chron: dict, histograms: dict[str, dict]) -> dict:
    """The contract from a workload_program.json + chronology.json pair and the components' Build histograms (dir -> spec_histogram)."""
    dirs = component_dirs(wp)
    reqs = {}
    for r in wp["requests"]:
        c = r["component"]
        d = _dir_of(r, dirs)
        ent = {"cap": int(r["cap"]), "component": d, "program_digest": c["program_digest"], "n_nodes": int(c["n_nodes"]),
               "steps": int(c["steps"]), "step_nodes_sha256": _sha(c["step_nodes"])}
        if d in histograms:
            ent["sampling_events"] = sampling_events(histograms[d])
        reqs[r["request_id"]] = ent
    unseg = sum(1 for e in chron["engine_steps"] for m in e["members"] if m.get("nodes") is None)
    return {"requests": reqs, "component_steps_equal_cap": bool(chron["component_steps_equal_cap"]), "unsegmented_members": unseg}


def _histograms(read, dirs) -> dict[str, dict]:
    out = {}
    for d in sorted(set(dirs)):
        for rel in (f"{d}/result.json", f"{d}/construction_manifest.json"):
            doc = read(rel)
            if doc is not None and isinstance(doc.get("spec_histogram"), dict):
                out[d] = doc["spec_histogram"]
                break
    return out


def expected(row: resolver.Row) -> dict:
    wp = row.json("build_workload/workload_program.json")
    return lift(wp, row.json("build_workload/chronology.json"),
                _histograms(lambda rel: row.json(rel) if row.has(rel) else None, component_dirs(wp).values()))


def recompute(desc: dict) -> dict:
    """`request_component` (current code) over one component descriptor: its segmentation and the sampling events in its body."""
    from verity.ml.circuits.codec import descriptor_id
    from verity_vllm.pipeline import global_program as GP  # the registry set build_workload decodes components with
    from verity_vllm.program import sampling_event as SE
    from verity_vllm.program import workload as W
    comp = W.request_component(desc, GP.REGISTRY, None, canon={})
    specs = [descriptor_id(n.fn) for n in comp.fn.body.nodes]
    return {"program_digest": comp.program_digest, "n_nodes": comp.n_nodes, "steps": len(comp.step_nodes),
            "step_nodes": [list(x) for x in comp.step_nodes], "sampling_events": sum(1 for s in specs if SE.is_sampling_event(s)),
            "segments_end_at_sampling_event": all(SE.is_sampling_event(specs[hi - 1]) for lo, hi in comp.step_nodes if hi > lo)}


def _consistent(reqs: dict, ends: dict[str, bool]) -> dict:
    return {rid: {"steps_eq_cap": r["steps"] == r["cap"], "steps_eq_sampling_events": r["steps"] == r.get("sampling_events"),
                  **({"segments_end_at_sampling_event": ends[r["component"]]} if r["component"] in ends else {})} for rid, r in reqs.items()}


def actual(row: resolver.Row) -> dict:
    if row.candidate_root() is not None:
        wp_p, ch_p = row.candidate("build_workload/workload_program.json"), row.candidate("build_workload/chronology.json")
        if wp_p is None or ch_p is None:
            return {"source": "candidate", **{k: resolver.not_applicable("the candidate has no build_workload/ documents") for k in _KEYS}}
        wp = json.load(open(wp_p))

        def read(rel):
            p = row.candidate(rel)
            return json.load(open(p)) if p is not None else None
        out = lift(wp, json.load(open(ch_p)), _histograms(read, component_dirs(wp).values()))
        out["consistent"] = _consistent(out["requests"], {})
        return dict(out, source="candidate")
    wp = row.json("build_workload/workload_program.json")
    dirs = component_dirs(wp)
    comps: dict[str, dict] = {}
    for d in sorted(set(dirs.values())):
        with gzip.open(row.path(f"{d}/descriptor.json.gz"), "rt") as f:
            comps[d] = recompute(json.load(f))
    reqs = {}
    for r in wp["requests"]:
        d = _dir_of(r, dirs)
        if d not in comps:
            raise resolver.NotResolvable(f"{row.key}: request {r['request_id']} names no derived component dir (inputs.derived: {sorted(dirs)})")
        c = comps[d]
        reqs[r["request_id"]] = {"cap": int(r["cap"]), "component": d, "program_digest": c["program_digest"], "n_nodes": c["n_nodes"],
                                 "steps": c["steps"], "step_nodes_sha256": _sha(c["step_nodes"]), "sampling_events": c["sampling_events"]}
    unseg = sum(max(0, r["cap"] - r["steps"]) for r in reqs.values())
    return {"requests": reqs, "component_steps_equal_cap": all(r["steps"] == r["cap"] for r in reqs.values()), "unsegmented_members": unseg,
            "consistent": _consistent(reqs, {d: c["segments_end_at_sampling_event"] for d, c in comps.items()}), "source": "request_component"}
