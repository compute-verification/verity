"""Is `manifest.json` THE manifest of record of this Program x workload?  `verify` compares the manifest a Commit used with the one the
integration's query path rebuilds and returns the `verity-sweep/manifest-verify/v1` record; `verity-vllm manifest-verify`
(`verity_vllm.pipeline.manifest_verify`) rebuilds it and writes the record beside the Commit (`commit/manifest_verify.json`).

`ok` iff the used manifest's `manifest_digest` == the digest of its own rows == the rebuilt digest, the program digests agree and
the rebuilt manifest is complete.  The manifest of record is a FUNCTION of the Program x declared workload: a weakened
manifest in the evidence dir has the wrong digest here even when its own `manifest_digest` is honest.  A verify FAIL is a Commit FAIL
(`required_value_coverage/manifest_of_record_verified` in verdict.json), never a footnote.  Replaces `required_manifest verify`
(the v1 builders' rebuild) on the row path since the v2 flip; the record's fields are unchanged so the regression harness lifts
`used_digest` / `used_program_digest` / `used_identities` as before.
"""
from __future__ import annotations

from typing import Any

from verity_vllm.query.manifest.format import manifest_digest

SCHEMA = "verity-sweep/manifest-verify/v1"
ENGINE = "v2-query"


def verify(used: dict[str, Any], rebuilt: dict[str, Any], *, manifest_path: str | None = None) -> dict[str, Any]:
    rows_digest = manifest_digest(used.get("identities") or [])
    rec = {"schema": SCHEMA, "manifest": manifest_path, "used_digest": used.get("manifest_digest"), "used_rows_digest": rows_digest,
           "rebuilt_digest": rebuilt["manifest_digest"], "used_program_digest": used.get("program_digest"), "rebuilt_program_digest": rebuilt["program_digest"],
           "used_identities": len(used.get("identities") or []), "rebuilt_identities": rebuilt["populations"]["identities"], "rebuilt_complete": rebuilt.get("complete"),
           "used_query": (used.get("query") or {}).get("engine") if isinstance(used.get("query"), dict) else None,
           "rebuilt_engine": ENGINE}
    rec["ok"] = bool(rec["used_digest"] == rows_digest == rec["rebuilt_digest"] and rec["used_program_digest"] == rec["rebuilt_program_digest"]
                     and rebuilt.get("complete") is True)
    if not rec["ok"]:
        why = []
        if rec["used_digest"] != rows_digest:
            why.append(f"manifest_digest {str(rec['used_digest'])[:16]} != digest of its rows {rows_digest[:16]} (edited after digesting)")
        if rows_digest != rec["rebuilt_digest"]:
            why.append(f"rows digest {rows_digest[:16]} != the Program x workload's manifest {rec['rebuilt_digest'][:16]} ({rec['used_identities']} vs {rec['rebuilt_identities']} identities): not the manifest of record")
        if rec["used_program_digest"] != rec["rebuilt_program_digest"]:
            u, b = str(rec["used_program_digest"]), str(rec["rebuilt_program_digest"])
            if u[:16] == b[:16] or len(u) != len(b):                          # a prefix passed as --workload-digest: say so, full width
                why.append(f"program_digest {u} ({len(u)} hex) != the rebuilt {b} ({len(b)} hex): pass the FULL digest (workload_program.json), not a prefix")
            else:
                why.append(f"program_digest {u[:16]} != the Build's {b[:16]}")
        if rebuilt.get("complete") is not True:
            why.append(f"the Program x workload manifest is incomplete: {rebuilt.get('unmodelled')}")
        rec["why"] = "; ".join(why)
    return rec

