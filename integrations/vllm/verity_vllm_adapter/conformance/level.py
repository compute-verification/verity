"""The level rules (contract Part 2) and the `level.json` verdict. Never above the evidence.

Level 1  reproducible          production run twice with identical tokens and chosen logprobs; replay followed
                               the trajectory, reproduced the logprobs, agreed with the unforced sampler; the
                               commit capture is bit-identical to fresh-process references (`compare.json`).
Level 2  typed-correspondence  the port constructs a `Program`; TA1 correspondence on a capture of this port
                               found every located value exact, or every mismatch is listed and recorded on its
                               composite's binding; every value not in the program and every program input not
                               from the checkpoint is classified by `expectations.json`.
Level 3  protocol-bundle       a Table-1 row whose recomputed §10 verdict is PASS, the verifier's report over
                               the same bundle, and the bundle's file digests re-hashed — for a manifest that
                               matches the port field by field.
Level 4  proof-backed          an exported typed obligation derived from a capture of this port, and a proof
                               record whose host-verified statement digest is the obligation's.

The levels are independent claims; `level` (the headline) is the highest level met and `gaps` lists every
lower level not met, so "L1/L3 with L2 not attempted" is expressible and an L3 never implies an L2.
`assert_claim` refuses any claim above the evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from . import SCHEMA_LEVEL

LEVEL_NAMES = {1: "reproducible", 2: "typed-correspondence", 3: "protocol-bundle", 4: "proof-backed"}


class OverClaim(ValueError):
    """A level was claimed that the evidence does not support."""


@dataclass
class LevelResult:
    level: int
    met: bool
    status: str  # met | not-met | not-attempted | not-claimed | absent
    reason: str | None
    evidence: list[dict[str, Any]]

    def as_json(self) -> dict[str, Any]:
        return {"level": self.level, "name": LEVEL_NAMES[self.level], "met": self.met, "status": self.status,
                "reason": self.reason, "evidence": self.evidence}


def _ptr(kind: str, path: str | None, **extra: Any) -> dict[str, Any]:
    return {"kind": kind, "path": path, **{k: v for k, v in extra.items() if v is not None}}


def level1(production: dict[str, Any], replay: dict[str, Any]) -> LevelResult:
    ev: list[dict[str, Any]] = []
    for r in production.get("runs", []):
        ev.append(_ptr("production", r["path"], profile_id=r.get("profile_id"), requests=r.get("requests"),
                       in_process_second_pass_identical=r.get("in_process_second_pass_identical")))
    for p in production.get("pairs", []):
        ev.append(_ptr("production-pair", None, a=p["a"], b=p["b"], identical=p["identical"], requests_shared=p["requests_shared"]))
    for r in replay.get("replays", []):
        ev.append(_ptr("replay-commit", r["commit"], profile_id=r.get("profile_id"), status=r["status"], problems=r["problems"] or None,
                       missing=r.get("missing") or None))
        if r.get("compare"):
            ev.append(_ptr("compare", None, **r["compare"]))
    if production["status"] == "absent" and replay["status"] == "absent":
        return LevelResult(1, False, "absent", "no production run and no replay supplied", ev)
    if production["status"] == "absent":
        return LevelResult(1, False, "not-met", "replay evidence only: no production run to compare (a commit store carries the replay half of Level 1)", ev)
    if production["status"] != "reproducible":
        return LevelResult(1, False, "not-met", f"production not shown reproducible ({production['status']}: {production.get('method')})", ev)
    if replay["status"] != "reproducible":
        why = "; ".join(p for r in replay.get("replays", []) for p in r["problems"] + r.get("missing", [])) or replay["status"]
        return LevelResult(1, False, "not-met", f"replay not shown reproducible: {why}", ev)
    rows = replay.get("rows") or {}
    note = f"; {rows['incomplete']} row(s) incomplete: " + "; ".join(
        f"{r['profile_id']}: {m}" for r in replay["replays"] if r["status"] == "incomplete" for m in r["missing"]) if rows.get("incomplete") else ""
    return LevelResult(1, True, "met", f"{production.get('method')}; replay bit-identical to fresh references in {rows.get('reproducible', len(replay['replays']))} row(s){note}", ev)


def level2(corr: dict[str, Any], port_claims: bool) -> LevelResult:
    ev = [_ptr("correspondence", f["path"], request=f.get("request"), positions=f.get("positions"), layers=f.get("layers"),
               summary=f.get("summary"), program=(f.get("program") or {}).get("function"),
               program_identity=(f.get("program_identity") or {}).get("match"), program_identity_note=(f.get("program_identity") or {}).get("note"))
          for f in corr.get("files", [])]
    for m in corr.get("mismatches", []):
        ev.append(_ptr("mismatch", m.get("source"), canonical_id=m["canonical_id"], composite=m["composite"], status=m["status"],
                       words=m.get("words"), binding=m.get("binding"), recorded_in_binding=m["recorded_in_binding"]))
    for d in (corr.get("missing") or {}).get("derived_inputs", []):
        ev.append(_ptr("derived-input", None, leaf_id=d["leaf_id"], provenance=d["provenance"],
                       rebuilt_sha256_matches_leaf=d["rebuilt_sha256_matches_leaf"], cpu_rebuild=d.get("cpu_rebuild")))
    st = corr.get("status")
    if not port_claims or st == "not-claimed":
        return LevelResult(2, False, "not-claimed", corr.get("reason"), ev)
    if st == "not-attempted":
        return LevelResult(2, False, "not-attempted", corr.get("reason"), ev)
    if st == "established":
        s = corr.get("summary", {})
        return LevelResult(2, True, "met", f"{s.get('exact', 0)} exact, {s.get('mismatch', 0)} mismatch (recorded), "
                                             f"{s.get('sampled', 0)} coordinate-sampled Gemms, {corr['records']} records", ev)
    return LevelResult(2, False, "not-met", corr.get("reason"), ev)


def level3(l3: dict[str, Any]) -> LevelResult:
    ev: list[dict[str, Any]] = []
    for b in l3.get("bundles", []):
        for r in b.get("table1_rows", []):
            ev.append(_ptr("table1-row", r["table1"], experiment_id=r.get("experiment_id"), profile_id=r.get("profile_id"),
                           result=r.get("result"), recomputed_result=r.get("recomputed_result"), verification_evidence=r.get("verification_evidence")))
        if b.get("verify"):
            v = b["verify"]
            ev.append(_ptr("verify-report", v["verify"], profile_id=v.get("profile_id"), challenge_digest=v.get("challenge_digest"),
                           statuses=v.get("statuses"), authentication=v.get("authentication"), negatives=v.get("negatives")))
        if b.get("bundle_integrity"):
            i = b["bundle_integrity"]
            ev.append(_ptr("bundle-digests", i["bundle_json"], files_checked=i["files_checked"], files_ok=i["files_ok"], files_bad=i["files_bad"] or None))
        if b.get("audit"):
            ev.append(_ptr("audit", b["audit"]["path"], result=b["audit"].get("result"), worker=b["audit"].get("worker")))
        if b["problems"]:
            ev.append(_ptr("bundle-problems", b["commit"], problems=b["problems"]))
    if l3.get("status") == "absent":
        return LevelResult(3, False, "absent", "no bundle of this port among the supplied results", ev)
    if l3.get("status") == "verified":
        good = [b for b in l3["bundles"] if b["status"] == "verified"]
        return LevelResult(3, True, "met", f"{len(good)} re-verified bundle(s): " + ", ".join(b["profile_id"] for b in good), ev)
    return LevelResult(3, False, "not-met", "; ".join(p for b in l3.get("bundles", []) for p in b["problems"]), ev)


def level4(l4: dict[str, Any]) -> LevelResult:
    ev: list[dict[str, Any]] = []
    for r in l4.get("records", []):
        ev.append(_ptr("typed-obligation", r["obligation"], name=r.get("name"), canonical_id=r.get("canonical_id"),
                       profile_id=r.get("profile_id"), results=r.get("results"), statement_digest=r.get("statement_digest"),
                       verdict=r.get("verdict"), fixture_files=r.get("fixture_files")))
        for p in r.get("proofs", []):
            ev.append(_ptr("proof-record", p["record"], backend=p.get("backend"), vk_hash=p.get("vk_hash"), ok=p.get("ok"),
                           verdict=p.get("verdict"), statement_match=p.get("statement_match"), matches_obligation=p["matches_obligation"]))
        if r["problems"]:
            ev.append(_ptr("obligation-problems", r["obligation"], problems=r["problems"]))
    if l4.get("status") == "absent":
        return LevelResult(4, False, "not-attempted", "no typed obligation / proof record supplied for this port", ev)
    if l4.get("status") == "proved":
        good = [r for r in l4["records"] if r["status"] == "proved"]
        return LevelResult(4, True, "met", f"{len(good)} proved obligation(s): " + ", ".join(r.get("name") or r["obligation"] for r in good), ev)
    return LevelResult(4, False, "not-met", "; ".join(p for r in l4.get("records", []) for p in r["problems"]), ev)


def verdict(port_name: str, results: list[LevelResult], *, utc: str, run_dir: str | None = None) -> dict[str, Any]:
    by = {r.level: r for r in results}
    met = sorted(l for l, r in by.items() if r.met)
    headline = max(met) if met else 0
    gaps = [l for l in range(1, headline) if l not in met]
    claims = []
    for l in (1, 2, 3, 4):
        r = by[l]
        tag = f"L{l} {LEVEL_NAMES[l]}"
        claims.append(f"{tag}: {'met' if r.met else r.status}" + (f" — {r.reason}" if r.reason else ""))
    return {"schema": SCHEMA_LEVEL, "port": port_name, "utc": utc, "run_dir": run_dir,
            "level": headline, "level_name": LEVEL_NAMES.get(headline, "none"), "met": met, "gaps": gaps,
            "summary": (f"L{headline}" if headline else "none") + (f" (gaps: {', '.join(f'L{g}' for g in gaps)})" if gaps else ""),
            "levels": {str(l): by[l].as_json() for l in (1, 2, 3, 4)}, "claims": claims}


def assert_claim(level_doc: dict[str, Any], claimed: int) -> None:
    """Refuse a claim the evidence does not support (`claimed` must be a met level)."""
    if claimed not in level_doc.get("met", []):
        r = (level_doc.get("levels") or {}).get(str(claimed)) or {}
        raise OverClaim(f"port {level_doc.get('port')}: Level {claimed} claimed but the evidence supports {level_doc.get('summary')}"
                        f" (L{claimed}: {r.get('status')}: {r.get('reason')})")
