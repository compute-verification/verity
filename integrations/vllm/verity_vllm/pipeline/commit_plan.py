"""The Commit's plan: what the Commit reads before it arms the committer and needs no engine for, derived on a 0-GPU task before
the GPU is held (`verity-vllm commit-plan`, the row's `plan` stage).

    verity-vllm commit-plan --workload W --required-manifest M --program-digest D (--program-dir P | --programs-root R) --out PLAN

The plan is the form (B) producer facts of a query-path manifest (`oracle_producers_of_record`: the Programs of record read by
dataflow; 588 s of a Qwen3-4B B32 Commit's 949 s).  The Commit (`--plan PLAN`) takes them from the plan only when the plan's key
equals its own (`key`: the Program digest, the sha256 of the required manifest and of the workload, the Programs it reads, the code
identity over the integration and core, the vLLM version); a plan it cannot read or whose key differs, and it derives the facts
itself and records `plan_recomputed` with the reason (`use`).  The facts are the same either way: the plan stores them in the
order they were derived.  Nothing sampled from the run root is planned: the replay sample and the slim bundle's reads come after
the Commit, from the root it fixed.
"""
from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata
import json
import os
import time
from pathlib import Path
from typing import Any

from verity_vllm.config import option, workload_arg

SCHEMA = "verity-vllm/commit-plan/v1"


def oracle_producers_of_record(args: Any, required_manifest: dict[str, Any] | None, reqs: list[dict[str, Any]],
                               sampling: dict[str, Any]) -> tuple[dict[tuple[str, str], dict[str, Any]] | None, dict[str, Any]]:
    """The oracle's producer facts from the Programs of record (`oracle_compare.producers_of_programs`, under the manifest's query) for a
    manifest the query path built (`query.query_id` in `required.QUERIES`), which carries none: (facts | None, record).  The Programs are
    the replay's (same lookup, same of-record digest check); None + the named reason for any other manifest, no Program dir or one not of record."""
    from verity_vllm.check import oracle_compare as OC
    from verity_vllm.query.manifest import format as MF
    from verity_vllm.query.required import QUERIES
    q = required_manifest.get("query") if isinstance(required_manifest, dict) else None
    if not (isinstance(q, dict) and q.get("query_id") in QUERIES):
        return None, {"used": False, "why": "manifest is not the query path's: its rows carry the producer annotations"}
    t0 = time.monotonic()
    labels = {str(r.get("request_id")): f"r{int(r.get('index', i))}" for i, r in enumerate(reqs)}
    if getattr(args, "programs_root", None):
        fx = {"requests": [{"request_id": labels[str(r.get("request_id"))], "prompt_len": len(r["prompt_token_ids"]),
                            "max_tokens": int(r.get("max_tokens") or sampling.get("max_tokens") or 1), "cached": int(r.get("cached") or 0)}
                           for r in reqs]}
        loaded = MF.programs_for(fx, args.programs_root)
    elif getattr(args, "program_dir", None):
        one = MF.load_program(args.program_dir)
        loaded = {labels[str(r.get("request_id"))]: one for r in reqs}
    else:
        return None, {"used": False, "why": "no --programs-root / --program-dir: producer facts not derivable"}
    of_record = {str(args.program_digest)} if getattr(args, "program_digest", None) else set()
    of_record.update(str(x) for x in (required_manifest or {}).get("component_digests") or [])
    of_record.update(str(c["program_digest"]) for c in ((required_manifest or {}).get("components") or {}).values()
                     if isinstance(c, dict) and c.get("program_digest"))
    digs = {k: str((v.get("result") or {}).get("program_digest") or (v.get("instances") or {}).get("digest") or "") for k, v in loaded.items()}
    bad = sorted(k for k, d in digs.items() if d not in of_record)
    if bad:
        return None, {"used": False, "why": f"Program(s) {bad[:4]} not of record (of-record {sorted(x[:16] for x in of_record)})"}
    facts = OC.producers_of_programs(loaded, aliases=(required_manifest or {}).get("op_path_aliases"), query=q["query_id"])
    del loaded
    return facts, {"used": True, "members": len(facts), "with_operand": sum(1 for f in facts.values() if f.get("operand")),
                   "with_consumers": sum(1 for f in facts.values() if f.get("consumers")),
                   "conflicts": sum(1 for f in facts.values() if f.get("conflict")), "wall_s": round(time.monotonic() - t0, 1),
                   "method": "v1_bridge.population over the Programs of record: producing Call family, first-operand producer family, "
                             "reading modules outside the producer's module (oracle_compare.producers_of_programs)"}


def _sha256(path: str | None) -> str | None:
    if not path:
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def _version(dist: str) -> str | None:
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None


def key(args: Any) -> dict[str, Any]:
    """What the producer facts are a function of, as the Commit with `args` would read it: the Program digest of record, the bytes of
    the required manifest and of the workload, which Programs are read (the dir or the root, resolved), the executing code (the code
    identity over the integration and core, `source_identity.code_identity`) and the vLLM version of this interpreter."""
    from verity_vllm.pipeline import source_identity as SI

    def real(p: str | None) -> str | None:
        return os.path.realpath(p) if p else None

    return {"program_digest": getattr(args, "program_digest", None) or None,
            "required_manifest_sha256": _sha256(getattr(args, "required_manifest", None)),
            "workload_sha256": _sha256(args.workload),
            "program_dir": real(getattr(args, "program_dir", None)), "programs_root": real(getattr(args, "programs_root", None)),
            "code": SI.code_identity(SI.TREE)["sha256"], "vllm": _version("vllm")}


def facts_to_json(facts: dict[tuple[str, str], dict[str, Any]] | None) -> list[list[Any]] | None:
    return None if facts is None else [[op, member, ent] for (op, member), ent in facts.items()]


def facts_of_json(rows: list[list[Any]] | None) -> dict[tuple[str, str], dict[str, Any]] | None:
    return None if rows is None else {(str(op), str(member)): dict(ent) for op, member, ent in rows}


def derive(args: Any) -> dict[str, Any]:
    """The plan of the Commit `args` describes: its key and the producer facts (the facts None + the named reason when there are none)."""
    from verity_vllm.engine import vllm_adapter as va

    t0 = time.monotonic()
    k = key(args)
    man = json.loads(Path(args.required_manifest).read_text()) if getattr(args, "required_manifest", None) else None
    wl = va.load_workload(args.workload)
    facts, rec = oracle_producers_of_record(args, man, wl["requests"], wl["sampling"])
    return {"schema": SCHEMA, "key": k, "producer_facts": {"facts": facts_to_json(facts), "record": rec},
            "wall_s": round(time.monotonic() - t0, 1), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def write(path: str | os.PathLike[str], doc: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".partial")
    tmp.write_text(json.dumps(doc, separators=(",", ":")) + "\n")
    os.replace(tmp, p)


def keyed(path: str, args: Any, schema: str) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """The plan document at `path` when it is of `schema` and keyed as the Commit `args` describes (`key`), and its record: (None, record
    with `plan_recomputed` and the reason) when it is unreadable, of another schema, or keyed differently."""
    rec: dict[str, Any] = {"path": str(path), "schema": schema}
    try:
        doc = json.loads(Path(path).read_text())
    except (OSError, ValueError) as e:
        return None, {**rec, "used": False, "plan_recomputed": True, "why": f"plan unreadable: {type(e).__name__}: {e}"[:300]}
    if not isinstance(doc, dict) or doc.get("schema") != schema:
        return None, {**rec, "used": False, "plan_recomputed": True, "why": f"plan schema {doc.get('schema') if isinstance(doc, dict) else None!r} != {schema}"}
    want, have = key(args), doc.get("key") or {}
    diff = sorted(k for k in set(want) | set(have) if want.get(k) != have.get(k))
    rec.update(key=want, planned_utc=doc.get("utc"), plan_wall_s=doc.get("wall_s"))
    if diff:
        return None, {**rec, "used": False, "plan_recomputed": True,
                      "why": "plan key differs from this Commit's: " + ", ".join(f"{k} {str(have.get(k))[:16]} != {str(want.get(k))[:16]}" for k in diff)}
    return doc, rec


def use(path: str, args: Any) -> tuple[tuple[dict[tuple[str, str], dict[str, Any]] | None, dict[str, Any]] | None, dict[str, Any]]:
    """The producer facts of the plan at `path` for the Commit `args` describes, as `oracle_producers_of_record` returns them, and the
    plan's record: (None, record with `plan_recomputed` and the reason) when the plan is unreadable, of another schema, or keyed
    differently from this Commit (`keyed`) -- the Commit then derives the facts itself."""
    t0 = time.monotonic()
    doc, rec = keyed(path, args, SCHEMA)
    if doc is None:
        return None, rec
    pf = doc.get("producer_facts") or {}
    out = (facts_of_json(pf.get("facts")), dict(pf.get("record") or {}))
    return out, {**rec, "used": True, "plan_recomputed": False, "members": len(out[0] or {}), "check_s": round(time.monotonic() - t0, 3)}


def derived(args: Any, required_manifest: dict[str, Any] | None, reqs: list[dict[str, Any]],
            sampling: dict[str, Any]) -> tuple[dict[tuple[str, str], dict[str, Any]] | None, dict[str, Any]]:
    """`oracle_producers_of_record`, with an exception as the named reason there are no facts (the manifest's own entries stand)."""
    try:
        return oracle_producers_of_record(args, required_manifest, reqs, sampling)
    except Exception as e:  # noqa: BLE001  -- no facts: the reason is on the record
        return None, {"used": False, "why": f"producer facts not derived: {e!r}"[:300]}


def producer_facts(args: Any, required_manifest: dict[str, Any] | None, reqs: list[dict[str, Any]], sampling: dict[str, Any], *, tl: Any
                   ) -> tuple[tuple[dict[tuple[str, str], dict[str, Any]] | None, dict[str, Any]], dict[str, Any] | None]:
    """The Commit's producer facts, from its plan (`--plan`) when `use` takes it, else `derived` here; and the plan's record (None
    without a plan).  `tl` times them (`prep.producer_facts`, with their source)."""
    tl.start("prep.producer_facts")
    got, rec = use(args.plan, args) if getattr(args, "plan", None) else (None, None)
    if rec is not None:
        print(f"[delta] plan {args.plan}: " + (f"producer facts from the plan ({rec['members']} members, planned {rec.get('planned_utc')})"
                                               if rec["used"] else f"plan_recomputed: {rec['why']}"), flush=True)
    got = got if got is not None else derived(args, required_manifest, reqs, sampling)
    tl.end("prep.producer_facts", source="plan" if (rec or {}).get("used") else "derived", members=len(got[0] or {}))
    return got, rec


def verdict_entry(args: Any, rec: dict[str, Any] | None) -> dict[str, Any]:
    """The Commit verdict's `plan` when it was given one: the plan's record, or why no producer facts were read."""
    if not getattr(args, "plan", None):
        return {}
    return {"plan": rec or {"path": args.plan, "used": False, "plan_recomputed": False, "why": "no producer facts read (no manifest or no committer)"}}


@dataclasses.dataclass(frozen=True, kw_only=True)
class Options:
    """Derive a Commit's plan (the producer facts of its manifest, keyed by what they are a function of) without an engine."""

    workload: str = option("--workload", type=workload_arg, help="the row's workload (the Commit's --workload)")
    required_manifest: str | None = option("--required-manifest", default=None, help="the Commit's required-value manifest")
    program_digest: str | None = option("--program-digest", default=None, help="the Commit's Program digest of record")
    program_dir: str | None = option("--program-dir", default=None, help="B=1: the Build's request Program dir")
    programs_root: str | None = option("--programs-root", default=None, help="B>=2: the directory holding the per-request Programs")
    out: str = option("--out", help="where the plan is written (the Commit's --plan)")


def main(o: Options) -> int:
    doc = derive(o)
    write(o.out, doc)
    pf = doc["producer_facts"]
    print(f"[plan] {o.out}: producer facts {pf['record']} in {doc['wall_s']}s; key " +
          " ".join(f"{k}={str(v)[:16]}" for k, v in doc["key"].items()), flush=True)
    return 0
