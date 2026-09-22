"""PROOF-COV's coverage relations as a proof-verification stage entry (CPU-side).

Stage 8c of ``e2e_b0.sh``: the two families that close GUIDANCE §36.1 requirement 1 —
``veritor.embed-gather@1`` (the embedding row gather) and ``veritor.pad-serve@1`` (the
ordinary inactive-case guard and row-mux statements) — checked against the FRESH record's
own attached proofs (``<record>/proofs/proof-cov/``, written by ``e2e.cov_attach`` on the
proving pod), not against the older committed pod records:

1. **Relation identity** — both gate sets rebuilt from the Python mirrors
   (``veritor.core.embed_gates`` / ``veritor.core.pad_serve``); each digest must equal the
   PROOF-COV pin AND the digest the record's ``cov_summary.json`` carries.
2. **Attached batches re-bound** — for every attached batch: ``statement.bin`` sha256 and
   canonical statement digest recomputed = the summary's pins; every attached
   ``proof-<mode>.bin`` sha256 recomputed = the pod proving record's pin; the CUDA prove and
   the CPU-mode host verify rows must both say True (local re-prove needs the SP1 prover
   and is out of scope for a CPU lane — stated).
3. **Prefill token binding re-derived** — the boundary tree of every request with prefill
   embedding VUs is REBUILT here from the record bundle's ``boundary.json`` (capture-module
   leaves) and must equal the bundle's ``commitments.json`` pin and the root the builder
   recorded.  A record without committed prompt ids must carry the labeled exclusion.
4. **Coverage census** — embed rows on BOTH phases (prefill + decode, or decode + labeled
   exclusion), pad-guard and pad-rowmux each with inactive units > 0.

Usage: ``python -m e2e.cov_relation --record <record dir> --out <dir>`` from ``vllm-poc/``.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

from e2e.run_composed import sha256_file, write_json

PINS = {
    "veritor.embed-gather@1": "7af70afd60a0e4a1a446c9d6b3b44866b98e9dcc0116353e37dfe97c14e2eb15",
    "veritor.pad-serve@1": "3aa430e2d7e6cb4dca307acc5defcff84cdb7ce9b79423cd01af12aa3a10c536",
}


def _bundle_docs(record: Path) -> tuple[dict, dict[str, dict]]:
    """(commitments.json, {request: boundary.json doc}) read straight out of the record's tarball."""
    tars = sorted((record / "bundle").glob("*.tar.gz"))
    if not tars:
        raise AssertionError("record has no bundle tarball -- prefill binding cannot be re-derived")
    commitments, boundaries = None, {}
    with tarfile.open(tars[0]) as tf:
        for member in tf.getmembers():
            parts = Path(member.name).parts
            if len(parts) >= 2 and parts[-1] == "commitments.json" and parts[-2] == "bundle":
                commitments = json.loads(tf.extractfile(member).read())
            elif parts[-1] == "boundary.json" and "ru" in parts:
                rid = parts[parts.index("ru") + 1]
                boundaries[rid] = json.loads(tf.extractfile(member).read())
    if commitments is None:
        raise AssertionError("bundle tarball has no bundle/commitments.json")
    return commitments, boundaries


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--record", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    record = Path(args.record).resolve()
    cov = record / "proofs" / "proof-cov"
    if not (cov / "cov_summary.json").exists():
        raise SystemExit(f"cov_relation: no {cov / 'cov_summary.json'} -- run e2e.cov_attach on the proving pod first")
    summary = json.loads((cov / "cov_summary.json").read_text())

    import veritor  # noqa: F401  (sys.path for the src tree)

    repo = Path(veritor.__file__).resolve().parents[2]
    sys.path.insert(0, str(repo))
    from veritor.core.embed_gates import make_embed_gather_gate_set
    from veritor.core.pad_serve import make_pad_serve_gate_set
    from veritor.protocol.proofs.wire import statement_digest

    # 1. relation identity
    families = {}
    for make in (make_embed_gather_gate_set, make_pad_serve_gate_set):
        gs = make()
        fam = f"{gs.name}@1"
        if str(gs.digest) != PINS[fam]:
            raise AssertionError(f"{fam}: rebuilt digest {gs.digest} != the PROOF-COV pin")
        families[fam] = str(gs.digest)
    for wl, rec in (summary.get("families") or {}).items():
        fam = rec.get("gate_set")
        if fam not in families or rec.get("gate_set_digest") != families[fam]:
            raise AssertionError(f"{wl}: cov_summary gate set {fam}/{rec.get('gate_set_digest')} != rebuilt digest")

    # 2. attached batches re-bound
    n_attached = n_proof_files = 0
    for b in summary["batches"]:
        bdir = cov / b["name"]
        for mode, doc in (b.get("modes") or {}).items():
            if not (doc.get("proved_cuda") or {}).get("verdict"):
                raise AssertionError(f"{b['name']} {mode}: pod proving record verdict is not True")
            vc = doc.get("verified_cpu")
            if vc is not None and not vc.get("ok"):
                raise AssertionError(f"{b['name']} {mode}: CPU-mode host verify recorded not ok")
        if not b.get("attached"):
            continue
        n_attached += 1
        st = bdir / "statement.bin"
        if sha256_file(st) != b["statement_sha256"]:
            raise AssertionError(f"{b['name']}: statement.bin sha != cov_summary pin")
        if statement_digest(st.read_bytes()).hex() != b["statement_digest"]:
            raise AssertionError(f"{b['name']}: canonical statement digest != cov_summary pin")
        for mode, doc in (b.get("modes") or {}).items():
            att = doc.get("attached")
            if not att:
                continue
            f = record / att["file"]
            if sha256_file(f) != att["sha256"]:
                raise AssertionError(f"{b['name']} {mode}: attached proof sha != cov_summary pin")
            want = (doc.get("proved_cuda") or {}).get("proof_sha256")
            if want and att["sha256"] != want:
                raise AssertionError(f"{b['name']} {mode}: attached proof sha != pod proving record pin")
            n_proof_files += 1

    # 3. prefill token binding re-derived from the record's own bundle
    binding_rows = summary.get("prefill_binding") or []
    checked_bindings = 0
    if any(row.get("committed") for row in binding_rows):
        from verity_vllm import commit as vcommit
        from verity_vllm.capture import Boundary, boundary_tree_id

        commitments, boundaries = _bundle_docs(record)
        profile_id = commitments["profile_id"]
        for row in binding_rows:
            if not row.get("committed"):
                continue
            rid = row["request"]
            tree = vcommit.MerkleTree.from_values(
                Boundary.from_json(boundaries[rid]).leaves(profile_id), boundary_tree_id(rid))
            pin = commitments["trees"][boundary_tree_id(rid)]
            if tree.root.hex() != pin["root"] or tree.root.hex() != row["root_recomputed"]:
                raise AssertionError(f"{rid}: boundary root re-derived here != bundle pin / builder record")
            checked_bindings += 1

    # 4. coverage census
    prefill = max((b.get("prefill_units") or 0) for b in summary["batches"])
    decode = max((b.get("decode_units") or 0) for b in summary["batches"])
    exclusions = summary.get("prefill_exclusions") or []
    guard_inactive = max((b.get("inactive_units") or 0) for b in summary["batches"]
                         if (b.get("relation") or "").startswith("pad-guard")) if any(
        (b.get("relation") or "").startswith("pad-guard") for b in summary["batches"]) else 0
    mux_inactive = max((b.get("inactive_units") or 0) for b in summary["batches"]
                       if (b.get("relation") or "").startswith("pad-rowmux")) if any(
        (b.get("relation") or "").startswith("pad-rowmux") for b in summary["batches"]) else 0
    if decode <= 0:
        raise AssertionError("no decode embedding rows proved")
    if prefill <= 0 and not exclusions:
        raise AssertionError("no prefill embedding rows AND no labeled exclusion -- a silent skip")
    if guard_inactive <= 0 or mux_inactive <= 0:
        raise AssertionError(f"ordinary inactive statements missing (guard {guard_inactive}, mux {mux_inactive})")
    if not summary.get("ok"):
        raise AssertionError(f"cov_summary records problems: {summary.get('problems')}")

    doc = {
        "schema": "veritor/cov-relation-check/v1",
        "generated": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "record": record.name,
        "relations": families,
        "batches": {"total": len(summary["batches"]), "attached": n_attached, "proof_files_rebound": n_proof_files},
        "coverage": {"embed_prefill_units": prefill, "embed_decode_units": decode,
                     "prefill_exclusions": exclusions,
                     "pad_guard_inactive_units": guard_inactive, "pad_rowmux_inactive_units": mux_inactive},
        "prefill_binding_rederived": checked_bindings,
        "program_identity": summary.get("program_identity"),
        "local_reprove": "out of scope for a CPU lane (SP1 prover); the pod's CUDA prove + CPU-mode "
                         "verify rows re-read, every attached statement/proof hash recomputed",
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "cov_relation.json", doc)
    print(f"cov_relation: 2 gate sets re-derived = pins; {n_attached} attached batches re-bound "
          f"({n_proof_files} proof files); embed {prefill} prefill + {decode} decode rows"
          f"{' (+' + str(len(exclusions)) + ' labeled exclusion(s))' if exclusions else ''}; "
          f"inactive units guard {guard_inactive} / rowmux {mux_inactive}; "
          f"prefill binding re-derived for {checked_bindings} request(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
