"""verity_vllm/properties/golden.py — PROTECTED (gate G6, regression half).  The frozen golden corpus: every previously accepted
observation log must keep resolving to exactly its recorded pattern attribution and Program digest.

    uv run verity-vllm golden --check [--out golden.json]          # CPU only, no GPU, every candidate commit
    uv run verity-vllm golden --record NAME --log LOG --profile MOD  # integrator only (protected path)

Why: a change that adds a Gemma pattern which accidentally also matches a Qwen operation, or loosens a matcher, changes how an
*old* capture resolves.  Re-folding the corpus catches that without a GPU: `attribution_sha` is the sha256 of the node-ordered
sequence `(step, event, pattern, specialization)` of every instance plus the fold summary histogram, `digest` is the Program's
canonical digest.  Additive specializations leave both unchanged; anything else is a regression that needs an explicit,
versioned migration (a new corpus entry recorded by the integrator with the decision id), never a silent edit.

Corpus manifest: `verity_vllm/properties/golden/corpus.json`.  Logs live under `data/logs/`, registered fixtures
(`fixtures/artifacts.json`) that `check` fetches where they are not.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verity_vllm.config import ROOT as REPO
from verity_vllm.config import UsageError, option
from verity_vllm.properties.record import seal

CORPUS = Path(__file__).resolve().parent / "golden" / "corpus.json"


def attribution(result) -> tuple[str, dict[str, Any]]:
    """sha256 of the instance attribution sequence + histogram (deterministic in the fold's node order)."""
    seq = [(int(m["step"]), int(m["event"]), str(m.get("pattern") or ""), str(m["spec"])) for m in result.instances]
    rep = result.report.to_dict()
    summ = {k: rep.get(k) for k in ("steps", "computes", "transparent", "unsupported", "by_pattern", "by_spec", "unsupported_ops")}
    doc = {"sequence": seq, "report": summ, "n_nodes": len(result.body.nodes) if result.body is not None else 0}
    blob = json.dumps(doc, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(blob).hexdigest(), {"n_instances": len(seq), "n_steps": len({s for s, *_ in seq}),
                                              "patterns": sorted({p for _, _, p, _ in seq}), "specs": len({sp for *_, sp in seq})}


def measure(log: str | Path, profile_name: str) -> dict[str, Any]:
    from verity_vllm.check.fold_compare import fold_log, load_profile

    t0 = time.time()
    profile = load_profile(profile_name)
    result = fold_log(log, profile, strict=False, with_program=True)
    sha, info = attribution(result)
    return {"digest": result.digest, "attribution_sha": sha, "unsupported": int(result.report.unsupported),
            "unresolved": result.unresolved, "seconds": round(time.time() - t0, 2), **info}


def _fetch(log: Path) -> str | None:
    """None once `log` is here, fetched first when absent (a registered fixture: `research data fetch-fixtures` checks every byte
    against its id); else why it is not."""
    if log.exists():
        return None
    p = subprocess.run([sys.executable, "-m", "research", "data", "fetch-fixtures", "--repo", str(REPO.parent.parent), str(log)],
                       capture_output=True, text=True)
    return None if log.exists() else f"log not present in this checkout; fetch-fixtures rc {p.returncode}: {p.stderr.strip()[-300:]}"


def load_corpus(path: Path = CORPUS) -> dict[str, Any]:
    if not path.exists():
        return {"schema": "verity-gen/golden/v1", "entries": []}
    return json.loads(path.read_text())


def check(corpus_path: Path = CORPUS, only: list[str] | None = None) -> dict[str, Any]:
    corpus = load_corpus(corpus_path)
    out: dict[str, Any] = {"schema": "verity-gen/golden-check/v1", "status": "pass", "utc": _utc(), "corpus": str(corpus_path), "entries": []}
    for e in corpus["entries"]:
        if only and e["name"] not in only:
            continue
        log = REPO / e["log"]
        ent = {"name": e["name"], "log": e["log"], "profile": e["profile"], "expected_digest": e["digest"], "expected_attribution_sha": e["attribution_sha"]}
        missing = _fetch(log)
        if missing:
            ent.update(status="skipped", reason=missing)
        else:
            try:
                m = measure(log, e["profile"])
                ent.update(got_digest=m["digest"], got_attribution_sha=m["attribution_sha"], n_instances=m["n_instances"], seconds=m["seconds"],
                           status="pass" if (m["digest"] == e["digest"] and m["attribution_sha"] == e["attribution_sha"]) else "fail")
                if ent["status"] == "fail":
                    ent["reason"] = ("digest changed" if m["digest"] != e["digest"] else "") + \
                                    (" attribution changed" if m["attribution_sha"] != e["attribution_sha"] else "")
                    ent["expected_n_instances"] = e.get("n_instances")
            except Exception as ex:  # noqa: BLE001
                ent.update(status="fail", reason=f"{type(ex).__name__}: {ex}")
        out["entries"].append(ent)
    st = [x["status"] for x in out["entries"]]
    out["n_pass"], out["n_fail"], out["n_skipped"] = st.count("pass"), st.count("fail"), st.count("skipped")
    out["status"] = "fail" if out["n_fail"] else ("pass" if out["n_pass"] else "skipped")
    return seal(out, out["status"] == "pass")


def record(name: str, log: str, profile_name: str, *, decision: str | None = None, corpus_path: Path = CORPUS) -> dict[str, Any]:
    corpus = load_corpus(corpus_path)
    m = measure(REPO / log if not Path(log).is_absolute() else log, profile_name)
    rel = str(Path(log)) if not Path(log).is_absolute() else str(Path(log).resolve().relative_to(REPO))
    entry = {"name": name, "log": rel, "profile": profile_name, "digest": m["digest"], "attribution_sha": m["attribution_sha"],
             "n_instances": m["n_instances"], "n_steps": m["n_steps"], "patterns": m["patterns"], "recorded_utc": _utc(), "decision": decision}
    corpus["entries"] = [e for e in corpus["entries"] if e["name"] != name] + [entry]
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    corpus_path.write_text(json.dumps(corpus, indent=1) + "\n")
    return entry


def _utc() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True, kw_only=True)
class Options:
    check: bool = option("--check", default=False, action="store_true")
    record: str | None = option("--record", default=None, metavar="NAME")
    log: str | None = option("--log", default=None)
    profile: str | None = option("--profile", default=None)
    decision: str | None = option("--decision", default=None, help="decision id authorizing a re-record of an existing entry")
    only: str | None = option("--only", default=None, help="comma list of entry names to check")
    out: str | None = option("--out", default=None)


def main(a: Options) -> int:
    if a.record:
        if not (a.log and a.profile):
            raise UsageError("--record needs --log and --profile")
        e = record(a.record, a.log, a.profile, decision=a.decision)
        print(json.dumps(e, indent=1))
        return 0
    rep = check(only=a.only.split(",") if a.only else None)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(rep, indent=1) + "\n")
    for e in rep["entries"]:
        print(f"[golden] {e['status']:7s} {e['name']}: {e.get('reason', '')} ({e.get('n_instances', '?')} instances, {e.get('seconds', '?')} s)", file=sys.stderr)
    print(f"[golden] {rep['status']}: {rep['n_pass']} pass, {rep['n_fail']} fail, {rep['n_skipped']} skipped", file=sys.stderr)
    return 0 if rep["status"] in ("pass", "skipped") else 1
