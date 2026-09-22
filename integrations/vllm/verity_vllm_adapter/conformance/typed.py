"""Level 2: typed correspondence between a port's `Program` and its capture (the TA1 engine,
`verity_vllm.correspondence.core`).

Two sources of correspondence evidence:

* recorded `verity-ir/correspondence/v0` files (TA1's `results/TA1-typed-b1-20260907/*/correspondence/*.json`,
  or a previous harness run's), checked here against the port: the capture's profile must be a bundle of this
  port, the program's function identity must be the port's static configuration at the recorded shape, every
  non-exact record is a mismatch, every `not_in_program` entry and every unconsumed/derived weights leaf is
  classified by the port's `expectations.json`;
* a live run (`--commit STORE --checkpoint SHARD.. --request R --positions .. --layers ..`) that calls
  `verity_vllm.correspondence.core.run` with the port's `program()` / `circuit_map()` (the `program_factory`,
  `circuit_map` and `static_config` seams) and then checks the result the same way.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

from .evidence import PID_RE, ResultsDir, _json, rel
from .port import Port

SHAPE_RE = re.compile(r"LP=(?P<LP>\d+),STEPS=(?P<STEPS>\d+)")
FN_BASE_RE = re.compile(r"^(?P<name>[A-Za-z0-9]+)_v(?P<ver>\d+)")


def _fn_base(function_id: str) -> tuple[str, int] | None:
    m = FN_BASE_RE.match(function_id or "")
    return (m.group("name"), int(m.group("ver"))) if m else None


def _root_id(port: Port, **shape: int) -> str | None:
    """The program's function identity at `shape` without lowering the body (`static_config.root`)."""
    root = getattr(port.static_config, "root", None)
    if callable(root):
        try:
            return root(**shape).id
        except NotImplementedError:
            return None
    return None


def _full_shape(port: Port, **shape: int) -> dict[str, Any]:
    """Lower the port's program at `shape` once per process (≈7 s / 1.2 GB for B1 at LP=512, STEPS=15)."""
    cache = getattr(port, "_full_shape_cache", None)
    if cache is None:
        cache = port._full_shape_cache = {}  # type: ignore[attr-defined]
    key = tuple(sorted(shape.items()))
    if key not in cache:
        t0 = time.time()
        prog = port.program(**shape)
        cache[key] = {"function": prog.fn.id, "gates": prog.gates, "input_gates": prog.input_gates, "build_seconds": round(time.time() - t0, 1)}
    return cache[key]


def check_correspondence_file(port: Port, path: Path, dirs: list[ResultsDir]) -> dict[str, Any]:
    """Check one recorded correspondence against the port; returns the per-file record used by `correspond`."""
    doc = _json(path)
    problems: list[str] = []
    if doc.get("schema") != "verity-ir/correspondence/v0":
        problems.append(f"schema {doc.get('schema')!r}")
    pids = {m.group(1) for r in doc.get("records", []) for m in [PID_RE.search(r.get("capture_id", ""))] if m}
    pids |= {m.group(1) for r in doc.get("not_in_program", []) for m in [PID_RE.search(r.get("capture_id", ""))] if m}
    matching = {rd.profile_id: rd for rd in dirs if rd.matches_port and rd.profile_id}
    for pid in sorted(pids):
        if pid not in matching:
            problems.append(f"capture profile {pid} is not a bundle of this port among the supplied results")
    prog = doc.get("program") or {}
    fid = prog.get("function", "")
    m = SHAPE_RE.search(fid)
    shape = {k: int(v) for k, v in m.groupdict().items()} if m else {}
    expected = _root_id(port, **shape) if shape else None
    identity = {"recorded": fid, "port": expected, "match": None}
    if expected is not None and expected == fid:
        identity["match"] = "exact"
    elif expected is not None:
        # The static-configuration digest differs (the registry dict's text changed since the run). Structural
        # identity is still decidable: lower the port's program at the recorded shape and compare gate counts.
        full = _full_shape(port, **shape)
        identity.update({"port_gates": full["gates"], "port_input_gates": full["input_gates"], "recorded_gates": prog.get("gates"),
                         "recorded_input_gates": prog.get("input_gates"), "build_seconds": full["build_seconds"]})
        if full["gates"] == prog.get("gates") and full["input_gates"] == prog.get("input_gates"):
            identity["match"] = "structural"
            identity["note"] = (f"static-configuration digest differs (recorded {fid!r}, committed registry {expected!r}); the port's program at "
                                f"LP={shape.get('LP')},STEPS={shape.get('STEPS')} has the recorded gate count and Input gate count")
        else:
            identity["match"] = "none"
            problems.append(f"program function {fid!r} is not this port's static configuration at LP={shape.get('LP')},STEPS={shape.get('STEPS')} "
                            f"({expected!r}: {full['gates']:,} gates vs recorded {prog.get('gates'):,})")
    if doc.get("model") and doc["model"] != port.config.model_repo:
        problems.append(f"model {doc['model']!r} != port checkpoint {port.config.model_repo!r}")
    summary = dict(doc.get("summary") or {})
    records = doc.get("records", [])
    families: dict[str, dict[str, int]] = {}
    mismatches: list[dict[str, Any]] = []
    version_drift: dict[str, tuple[int, int, int]] = {}  # function name -> (recorded version, bound version, records)
    for r in records:
        vid = r.get("value_identity") or {}
        base = _fn_base(vid.get("function", ""))
        binding = port.bindings.get(base[0]) if base else None
        if base and binding:
            # IT-F3 (`independent-tests.md`): the program identity above hashes the static configuration, not the
            # definition versions, so evidence computed under `Name_v1` stayed "met" after the port moved to `Name@2`.
            # Every record names the composite version it was evaluated under; it must be the one the port binds now.
            bound_ver = int(binding.registered.rsplit("@", 1)[1])
            if bound_ver != base[1]:
                name, ver, n = base[0], base[1], version_drift.get(base[0], (0, 0, 0))[2]
                version_drift[name] = (ver, bound_ver, n + 1)
        fam = (binding.family if binding else None) or "(structural)"
        st = (r.get("check") or {}).get("status", "?")
        families.setdefault(fam, {})
        families[fam][st] = families[fam].get(st, 0) + 1
        if st != "exact":
            ch = r.get("check") or {}
            mismatches.append({"canonical_id": r["capture_id"], "status": st, "composite": vid.get("function"), "call": vid.get("call"),
                               "family": fam, "words_compared": ch.get("words_compared"), "mismatches": ch.get("mismatches"),
                               "words": [{"word": w.get("word"), "gate": w.get("gate"), "ir": w.get("ir"), "captured": w.get("captured")}
                                         for w in (ch.get("first_mismatches") or [])],
                               "check": ch, "authentication_location": r.get("authentication_location"),
                               "binding": {"registered": binding.registered, "status": binding.status,
                                           "divergence_recorded": binding.divergence_recorded} if binding else None,
                               "recorded_in_binding": bool(binding and (binding.divergence_recorded or binding.status != "exact-model-tested")),
                               "source": rel(path)})
    if version_drift:
        problems.append("definition-version drift: " + ", ".join(
            f"{n} record(s) evaluated under {name}_v{rec} but the port binds {name}@{bound}" for name, (rec, bound, n) in sorted(version_drift.items()))
            + " (evidence predates the binding; re-run the correspondence under the current binding)")
    return {"path": rel(path), "request": doc.get("request"), "positions": doc.get("positions"), "layers": doc.get("layers"),
            "profiles": sorted(pids), "program": prog, "program_identity_ok": identity["match"] in (None, "exact", "structural"),
            "program_identity": identity, "shape": shape, "summary": summary, "records": len(records), "families": families,
            "definition_version_drift": {name: {"recorded": rec, "bound": bound, "records": n} for name, (rec, bound, n) in sorted(version_drift.items())},
            "mismatches": mismatches, "not_in_program": doc.get("not_in_program", []),
            "weights": (doc.get("input_gate_records") or {}).get("weights", []),
            "prompt": (doc.get("input_gate_records") or {}).get("prompt"), "problems": problems}


def classify_missing(port: Port, files: list[dict[str, Any]]) -> dict[str, Any]:
    """`missing_values.json`: every captured value with no counterpart in the program, every program input not
    taken from the checkpoint, every declared-unmaterialised value, each classified by the port's expectations
    (an entry no expectation covers is `unclassified` and blocks Level 2)."""
    patterns = [(re.compile(p["pattern"]), p["reason"]) for p in port.expectations.not_in_program]
    seen_nip: dict[str, dict[str, Any]] = {}
    unclassified = 0
    for f in files:
        for e in f["not_in_program"]:
            cid = e["capture_id"]
            key = re.sub(r"/r=[^/]+/", "/r=*/", re.sub(r"/t=\d+/", "/t=*/", cid))
            rec = seen_nip.setdefault(key, {"pattern_of_capture_ids": key, "count": 0, "tool_reason": e.get("reason"), "examples": [],
                                            "classification": "unclassified", "port_reason": None})
            rec["count"] += 1
            if len(rec["examples"]) < 2:
                rec["examples"].append(cid)
            for rx, reason in patterns:
                if rx.search(cid):
                    rec["classification"], rec["port_reason"] = "expected-not-in-program", reason
                    break
    for rec in seen_nip.values():
        if rec["classification"] == "unclassified":
            unclassified += rec["count"]
    derived: dict[str, dict[str, Any]] = {}
    unconsumed: dict[str, dict[str, Any]] = {}
    checkpoint_ok, checkpoint_bad = 0, 0
    for f in files:
        for w in f["weights"]:
            lid = w.get("leaf_id", "")
            if "not_in_program" in w:
                rec = unconsumed.setdefault(lid, {"leaf_id": lid, "tool_reason": w["not_in_program"], "classification": "unclassified", "port_reason": None,
                                                  "authentication_location": w.get("authentication_location")})
                for rx, reason in patterns:
                    if rx.search(lid):
                        rec["classification"], rec["port_reason"] = "expected-unconsumed-leaf", reason
                        break
            elif w.get("rebuilt_from"):
                derived[lid] = {"leaf_id": lid, "classification": "derived-input",
                                "provenance": w["rebuilt_from"], "rebuilt_sha256_matches_leaf": w.get("sha256_matches_leaf"),
                                "cpu_rebuild": w.get("cpu_rebuild"), "value_identity": w.get("value_identity"),
                                "authentication_location": w.get("authentication_location")}
            elif w.get("sha256_matches_leaf") is True:
                checkpoint_ok += 1
            else:
                checkpoint_bad += 1
    for rec in unconsumed.values():
        if rec["classification"] == "unclassified":
            unclassified += 1
    for d in derived.values():
        if d["rebuilt_sha256_matches_leaf"] is not True:
            unclassified += 1
            d["classification"] = "derived-input-unreproduced"
    missing_operand = sum(int(f["summary"].get("missing-operand", 0)) for f in files)
    return {"not_in_program": sorted(seen_nip.values(), key=lambda r: r["pattern_of_capture_ids"]),
            "unconsumed_leaves": sorted(unconsumed.values(), key=lambda r: r["leaf_id"]),
            "derived_inputs": sorted(derived.values(), key=lambda r: r["leaf_id"]),
            "checkpoint_leaves": {"sha256_matches": checkpoint_ok, "sha256_differs": checkpoint_bad},
            "missing_operand_records": missing_operand,
            "unmaterialised_declared": list(port.expectations.unmaterialised),
            "unclassified": unclassified}


def expectations_check(port: Port, files: list[dict[str, Any]], description: dict | None) -> dict[str, Any]:
    """Per family: the port's declared status against what the capture recorded and what correspondence found."""
    captured: dict[str, int] = {}
    relations: dict[str, set[str]] = {}
    if description:
        for n in description.get("nodes", []):
            fam = n.get("family")
            if fam:
                captured[fam] = captured.get(fam, 0) + 1
                rid = n.get("relation")
                if rid:
                    relations.setdefault(fam, set()).add(str(rid))
    corr: dict[str, dict[str, int]] = {}
    for f in files:
        for fam, sts in f["families"].items():
            for st, n in sts.items():
                corr.setdefault(fam, {})
                corr[fam][st] = corr[fam].get(st, 0) + n
    out = {}
    problems = []
    for fam, e in port.expectations.families.items():
        c = corr.get(fam, {})
        exact, other = c.get("exact", 0), sum(v for k, v in c.items() if k != "exact")
        consistent = True
        if e.status == "supported-exact" and c and exact == 0:
            consistent = False
        if e.status == "unsupported" and c:
            consistent = False  # a family the port calls unsupported must not be matched by the program
        if e.relations and fam in relations and not set(e.relations) & relations[fam]:
            consistent = False
        out[fam] = {"declared": e.status, "kind": e.kind, "reason": e.reason, "captured_nodes": captured.get(fam, 0),
                    "captured_relations": sorted(relations.get(fam, ())), "correspondence": c, "consistent": consistent}
        if not consistent:
            problems.append(f"family {fam}: declared {e.status}, correspondence {c}, relations {sorted(relations.get(fam, ()))}")
    for fam in sorted(set(captured) - set(port.expectations.families)):
        out[fam] = {"declared": None, "captured_nodes": captured[fam], "consistent": False}
        problems.append(f"family {fam}: captured but not in expectations.json")
    return {"families": out, "problems": problems}


def correspond(port: Port, files: list[Path], dirs: list[ResultsDir], description: dict | None) -> dict[str, Any]:
    """Assemble Level 2 from recorded correspondence files (already checked against the port)."""
    checked = [check_correspondence_file(port, p, dirs) for p in files]
    summary: dict[str, Any] = {}
    qualifications: list[str] = []
    for f in checked:
        for k, v in f["summary"].items():
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                # verity_vllm.correspondence.core ≥ 7da4eda carries `summary.qualifications` (free-text list); kept, not summed
                if k == "qualifications" and isinstance(v, list):
                    qualifications += [str(q) for q in v]
                continue
            summary[k] = summary.get(k, 0) + int(v)
    if qualifications:
        summary["qualifications"] = qualifications
    records = sum(f["records"] for f in checked)
    mismatches = [m for f in checked for m in f["mismatches"]]
    missing = classify_missing(port, checked)
    exp = expectations_check(port, checked, description)
    problems = [f"{f['path']}: {p}" for f in checked for p in f["problems"]]
    problems += exp["problems"]
    if not port.level2_claimed:
        status, reason = "not-claimed", port.level2_reason
    elif not checked:
        status, reason = "not-attempted", "no correspondence evidence supplied (recorded files or a live run)"
    elif records == 0:
        status, reason = "not-attempted", "correspondence files contain no records"
    elif problems:
        status, reason = "not-established", "; ".join(problems)
    elif summary.get("shape_mismatch", 0):
        status, reason = "not-established", f"{summary['shape_mismatch']} shape/type mismatches"
    elif summary.get("missing-operand", 0):
        status, reason = "not-established", f"{summary['missing-operand']} records with missing operands"
    elif missing["unclassified"]:
        status, reason = "not-established", f"{missing['unclassified']} missing/unconsumed/derived values not classified by expectations.json"
    elif any(not m["recorded_in_binding"] for m in mismatches):
        status, reason = "not-established", "a mismatch is not recorded on its composite's binding (status not downgraded)"
    else:
        status, reason = "established", None
    return {"status": status, "reason": reason, "files": [{k: v for k, v in f.items() if k not in ("weights", "not_in_program", "mismatches")} for f in checked],
            "summary": summary, "records": records, "mismatches": mismatches, "missing": missing, "expectations": exp}


def live_run(port: Port, commit: Path, checkpoint: list[Path], request: str, positions: list[int], layers: list[int], out: Path,
             coord_samples: int = 32, full: set[tuple[int, int]] | None = None, seed: int = 0,
             cos_sin_npy: Path | None = None, cos_sin_cpu_npy: Path | None = None) -> Path:
    """Run TA1's engine on a commit store with the port's program and mapping (GPU-free but needs numpy + the
    checkpoint shards); writes a `verity-ir/correspondence/v0` file and returns its path."""
    from verity_vllm.correspondence.core import CommitStore, run  # noqa: PLC0415

    t0 = time.time()
    out.parent.mkdir(parents=True, exist_ok=True)
    run(CommitStore(commit), checkpoint, request, positions, layers, coord_samples, full or set(), out, seed,
        cos_sin_npy=cos_sin_npy, cos_sin_cpu_npy=cos_sin_cpu_npy,
        program_factory=lambda **shape: port.program(**shape), circuit_map=port.circuit_map,
        static_config=port.static_config_record())
    print(f"[conformance] live correspondence written to {out} in {time.time() - t0:.1f}s", flush=True)
    return out


def bytes_by_field(desc: dict) -> dict[str, Any]:
    """Descriptor bytes by top-level field and by definition family. Uses B1Q's `verity_vllm.ir.profile_descriptor`
    when it is importable (branch `ta1-b1-qualifications`), otherwise a plain canonical-JSON size split."""
    try:
        from verity_vllm.ir.profile_descriptor import profile  # type: ignore  # noqa: PLC0415

        return {"profiler": "verity_vllm.ir.profile_descriptor", **profile(desc)}
    except ImportError:
        pass
    from verity.ir import canonical_json  # noqa: PLC0415

    top = {k: len(canonical_json(v)) for k, v in desc.items() if k != "annotations"}
    fams: dict[str, dict[str, int]] = {}
    forms = {"primitive": 0, "composite": 0}
    for fid, d in desc["definitions"].items():
        base = FN_BASE_RE.match(fid)
        fam = base.group(0) if base else fid.split("{")[0]
        rec = fams.setdefault(fam, {"definitions": 0, "bytes": 0, "nodes": 0})
        rec["definitions"] += 1
        rec["bytes"] += len(canonical_json(d))
        rec["nodes"] += len((d.get("body") or {}).get("nodes", []))
        forms[d.get("form", "composite")] = forms.get(d.get("form", "composite"), 0) + 1
    return {"profiler": "conformance.bytes_by_field (fallback)", "top_level": top, "forms": forms,
            "families": dict(sorted(fams.items(), key=lambda kv: -kv[1]["bytes"]))}


def descriptor_stats(port: Port, files: list[dict[str, Any]], full: bool = False) -> dict[str, Any]:
    """`descriptor_stats.json`: the probe-shape program's descriptor (built here) plus the recorded full-shape
    program facts from the correspondence files; `full=True` also lowers the full-shape program."""
    from verity.ir import descriptor_size, encode_program, program_digest  # noqa: PLC0415
    from verity.ir.defs import PrimitiveDefinition, SpecializedDefinition  # noqa: PLC0415

    if not port.level2_claimed:
        return {"status": "not-claimed", "reason": port.level2_reason}
    probe = dict(getattr(port.static_config, "PROBE_SHAPE", {"LP": 2, "STEPS": 1}))
    t0 = time.time()
    prog = port.program(**probe)
    build_s = time.time() - t0
    desc = encode_program(prog)
    comps: dict[str, dict[str, Any]] = {}
    prims: dict[str, int] = {}

    def visit(fn) -> None:
        if isinstance(fn, PrimitiveDefinition):
            prims[fn.id] = prims.get(fn.id, 0) + 1
            return
        assert isinstance(fn, SpecializedDefinition)
        if fn.id in comps:
            return
        comps[fn.id] = {"gates": fn.gates, "nodes": len(fn.body.nodes), "conformance": getattr(fn.defn, "conformance", None)}
        for n in fn.body.nodes:
            visit(n.fn)
        for v in fn.bindings.values():
            if isinstance(v, (PrimitiveDefinition, SpecializedDefinition)):
                visit(v)

    visit(prog.fn)
    probe_stats = {"shape": probe, "function": prog.fn.id, "gates": prog.gates, "input_gates": prog.input_gates,
                   "descriptor_bytes": descriptor_size(desc), "program_digest": program_digest(desc),
                   "definitions": len(desc["definitions"]), "distinct_specializations": len(comps), "distinct_primitives": len(prims),
                   "root_nodes": len(prog.root.body.nodes), "build_seconds": round(build_s, 2),
                   "bytes_by_field": bytes_by_field(desc),
                   "specializations": dict(sorted(comps.items(), key=lambda kv: -kv[1]["gates"])[:64])}
    recorded = [{"source": f["path"], **f["program"]} for f in files if f.get("program")]
    out: dict[str, Any] = {"status": "computed", "probe": probe_stats, "recorded_full_shape": recorded}
    shapes = {tuple(sorted(f["shape"].items())) for f in files if f.get("shape")}
    cached = getattr(port, "_full_shape_cache", {}) or {}
    if full and shapes:
        for s in shapes:
            _full_shape(port, **dict(s))
        cached = port._full_shape_cache  # type: ignore[attr-defined]
    if cached:
        out["full_shape"] = [{"shape": dict(k), **v,
                              "matches_recorded": all(r.get("gates") == v["gates"] and r.get("input_gates") == v["input_gates"]
                                                      for r in recorded if SHAPE_RE.search(r.get("function", "")) and
                                                      {kk: int(vv) for kk, vv in SHAPE_RE.search(r["function"]).groupdict().items()} == dict(k))}
                             for k, v in cached.items()]
    return out
