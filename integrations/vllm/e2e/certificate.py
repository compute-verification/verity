"""The single-instance whole-program certificate (GUIDANCE §36.1 requirement 4).

One JSON document describing ONE record: the whole-serving-program description + its validated
plan, the pre-challenge sampling policy (with the realized selections held SEPARATE), the
staged commitments, the authenticated proofs for every selected obligation, the served-token
correspondence + enforcement status, and the reconciled cost record.  Every hash the
certificate cites is RECOMPUTED here from the record's files — the certificate binds, it does
not transcribe.  The certificate's own sha256 is printed so a transcript can pin it.

Usage: ``python -m e2e.certificate --record <completed run dir> [--out FILE]``.
Exit 0 iff every recomputed hash matches the manifest's pin and the record's verdict is PASS.

v3 (PROTO-A, 2026-09-09; brief A.4): the certificate REQUIRES the record's pre-challenge protocol instance
(``protocol_instance.json``, pinned in the manifest), the digest-pinned compact whole-program validation artifact
(``compact_validation.json``), the population file and -- for a Bernoulli law -- the commitment -> challenge chronology
(``e2e.instance_check``).  Any missing or inconsistent item is a binding problem: NOT BOUND, exit 1.  Nothing is skipped
with a reason.  Historical records (INT-PAD-2 and earlier, which carry none of this) certify only under ``--historical-v2``,
which writes the v2 schema with ``attests_protocol_instance = false`` and ``bound_attached = false``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from e2e.run_composed import sha256_file, write_json


def _load(p: Path):
    return json.loads(p.read_text())


def _bundle_file(d: Path, name: str) -> bytes | None:
    """A ``bundle/<name>`` file read out of the record's bundle tarball (adapter/ lives in the
    pod archive, not the committed record; the tarball carries the run's own copies)."""
    import tarfile

    for tar in sorted((d / "bundle").glob("*.tar.gz")):
        with tarfile.open(tar) as tf:
            for member in tf.getmembers():
                p = Path(member.name)
                if p.name == name and p.parent.name == "bundle":
                    return tf.extractfile(member).read()
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--record", required=True)
    ap.add_argument("--out", default=None, help="default RECORD/certificate.json")
    ap.add_argument("--require-coverage", action="store_true",
                    help="a missing proofs/proof-cov/ attachment is a binding problem "
                         "(the complete-audit certificate, GUIDANCE §36.1 req. 1)")
    ap.add_argument("--historical-v2", action="store_true",
                    help="certify a record that predates the protocol instance (v2 schema; the certificate states that it attests "
                         "no protocol instance and attaches no Bound). Without this flag the instance + compact validation "
                         "artifact + population (+ chronology for a Bernoulli law) are REQUIRED and their absence is NOT BOUND.")
    ap.add_argument("--instance", default=None, help="protocol_instance.json (default RECORD/protocol_instance.json)")
    ap.add_argument("--validation", default=None, help="compact_validation.json (default RECORD/compact_validation.json)")
    ap.add_argument("--population", default=None, help="population.json (default RECORD/population.json)")
    ap.add_argument("--chronology", default=None, help="chronology.json (default RECORD/chronology.json)")
    ap.add_argument("--approved-guests", default=None,
                    help="the verifier's approved-guests policy (default e2e/approved_guests.json). STRICT path: an ELF/vk not in it is NOT BOUND "
                         "(GUEST_NOT_APPROVED) whatever the record claims; --historical-v2 warns instead.")
    args = ap.parse_args(argv)
    d = Path(args.record).resolve()
    manifest, summary = _load(d / "manifest.json"), _load(d / "summary.json")
    inp = manifest["inputs"]
    scope = summary.get("scope") or {}
    problems: list[str] = []

    # 0. v3: the pre-challenge protocol instance and its evidence (hard requirement; e2e.instance_check)
    if args.historical_v2:
        protocol_instance = {"attests_protocol_instance": False, "bound_attached": False,
                             "note": "--historical-v2: this record predates the protocol instance (no pre-challenge instance, no compact "
                                     "whole-program validation artifact, fixed-size selector); it is described after the fact by "
                                     "fixtures/results/PROTO-A/*/protocol_instance.json (status historical-reconstruction) and carries no Bound"}
    else:
        from e2e.instance_check import check_instance_evidence

        ev_problems, ev_summary = check_instance_evidence(d, manifest, instance=Path(args.instance) if args.instance else None,
                                                          validation=Path(args.validation) if args.validation else None,
                                                          population=Path(args.population) if args.population else None,
                                                          chronology=Path(args.chronology) if args.chronology else None)
        problems.extend(f"instance: {p}" for p in ev_problems)
        protocol_instance = {"attests_protocol_instance": not ev_problems, **ev_summary,
                             "bound_attached": bool((ev_summary.get("instance") or {}).get("bound_attached")) and not ev_problems}

    def pin(rel: str, key: str) -> str | None:
        """Recompute the file's sha and hold it against the manifest pin (when one exists)."""
        f = d / rel
        if not f.exists():
            problems.append(f"{rel} missing from the record")
            return None
        got = sha256_file(f)
        want = inp.get(key)
        if want and got != want:
            problems.append(f"{rel} sha {got[:12]} != manifest inputs.{key} {str(want)[:12]}")
        return got

    # 1. the instance
    instance = {
        "record_id": d.name, "case": manifest.get("case"), "utc": [manifest.get("utc_start"), manifest.get("utc_end")],
        "manifest_sha256": sha256_file(d / "manifest.json"),
        "workload_sha256": pin("workload.json", "workload_sha256") if (d / "workload.json").exists() else inp.get("workload_sha256"),
        "semantic_profile": inp.get("semantic_profile"),
        "production_record_sha256": inp.get("production_record_sha256"),
        "trusted_replay_equivalence_class": inp.get("trusted_replay_equivalence_class"),
        "workload_description": scope.get("workload"),
    }

    # 2. the whole serving program + its validated plan
    program = {
        "description_sha256": inp.get("description_sha256"),  # recomputed + compared in the enforcement block below
        "program_digest": inp.get("program_digest"), "layer_program_digest": inp.get("layer_program_digest"),
        "plan_digest": inp.get("plan_digest"),
        "full_program_validation": scope.get("full_program_validation"),
    }

    # 3. the pre-challenge sampling policy, realized selections SEPARATE
    sv = _load(d / "verify" / "selective_verify.json") if (d / "verify" / "selective_verify.json").exists() else {}
    policy_doc = _load(d / "sampling_policy.json") if (d / "sampling_policy.json").exists() else None
    if policy_doc is None:
        problems.append("sampling_policy.json missing (pre-challenge q,s not bound)")
    else:
        pin("sampling_policy.json", "sampling_policy_sha256")
        if policy_doc.get("selector") == "fixed_size" and sv and sv.get("k") != policy_doc.get("k"):
            problems.append(f"realized |J| = {sv.get('k')} != declared k = {policy_doc.get('k')}")
    sampling = {
        "declared": policy_doc,
        "realized": {"r_q": ((sv.get("challenge") or {}).get("r_q") if sv else None), "k": sv.get("k"),
                     "selected_requests": sv.get("selected"), "requests_total": sv.get("requests_total"),
                     "certified_sample": sv.get("certified_sample"), "coverage_sample": sv.get("coverage_sample")},
        "separation": "declared lives in sampling_policy.json bound before READY_PHASE0; realized lives in verify/",
    }

    # 4. commitments + proofs for every selected obligation
    # GUEST (S12 opt-in): which statement format h_L is the digest of (2 = the historical record; 3 carries both digests)
    statement_format = int(inp.get("statement_format", 2))
    commitments = {"h_O": inp.get("h_O"), "h_L": inp.get("h_L"), "statement_format": statement_format,
                   "statement_v2_digest": (inp.get("statement_v2_digest") if statement_format == 3 else inp.get("h_L")),
                   "commitments_sha256": inp.get("commitments_sha256"), "challenge_digest": inp.get("challenge_digest")}
    if statement_format not in (2, 3):
        problems.append(f"manifest inputs.statement_format {statement_format!r} is not 2 or 3")
    if statement_format == 3 and not inp.get("statement_v2_digest"):
        problems.append("format-3 record without statement_v2_digest")
    # [REC] the exported obligation object, statement and witness are held against the manifest: h_L == sha256(statement.bin), h_O == the
    # object's recorded digest, and (records that carry them) the per-file pins + obligation count -- a tampered value or a dropped obligation is refused here
    odir = d / "obligations"
    if (odir / "statement.bin").exists() and inp.get("h_L") and sha256_file(odir / "statement.bin") != inp["h_L"]:
        problems.append(f"obligations/statement.bin sha != manifest h_L {str(inp['h_L'])[:12]} (statement tampered or obligations dropped)")
    if (odir / "composed.json").exists():
        comp = _load(odir / "composed.json")
        if inp.get("h_O") and comp.get("digest") != inp["h_O"]:
            problems.append("obligations/composed.json digest != manifest h_O")
        n_obl = len(((comp.get("object") or {}).get("obligations")) or [])
        if inp.get("obligations_count") is not None and n_obl != inp["obligations_count"]:
            problems.append(f"obligations/composed.json carries {n_obl} obligations != manifest obligations_count {inp['obligations_count']} (an obligation was dropped or added)")
        commitments["obligations_in_object"] = n_obl
    for name, want in (inp.get("obligations_sha256") or {}).items():
        f = odir / name
        if not f.exists():
            problems.append(f"obligations/{name} missing (pinned in the manifest)")
        elif sha256_file(f) != want:
            problems.append(f"obligations/{name} sha != manifest pin (a value or an obligation was changed after the record was made)")
    commitments["obligations_sha256"] = inp.get("obligations_sha256")
    proofs = {"elf_sha256": inp.get("elf_sha256"), "vk_hash": inp.get("vk_hash"), "sp1_version": inp.get("sp1_version"),
              "host_sha256": inp.get("host_sha256"), "guest_features_claimed": inp.get("guest_features"), "files": {}}
    # GUEST (S12 caveat, fail-closed): the (ELF, vk) must be one the VERIFIER approves (e2e/approved_guests.json); the
    # record's claimed guest_features / guest_build never substitute for that.  STRICT: NOT BOUND (GUEST_NOT_APPROVED).
    # --historical-v2: a loud warning (that certificate is already labelled as attesting no instance).
    from e2e.approved_guests import check_guest

    guest, g_problems, g_warnings = check_guest(inp.get("elf_sha256"), inp.get("vk_hash"), inp.get("guest_build"), strict=not args.historical_v2,
                                               policy_path=Path(args.approved_guests) if args.approved_guests else None, what="proofs guest")
    problems.extend(g_problems)
    for w in g_warnings:
        print(f"[certificate] WARNING: {w}", file=sys.stderr)
    proofs["guest"] = {**guest, "warnings": g_warnings}
    proofs["guest_build"] = guest.get("guest_build")
    for f in sorted((d / "proofs").glob("proof-*.bin")):
        proofs["files"][f.name] = {"sha256": sha256_file(f), "bytes": f.stat().st_size}
    if not proofs["files"]:
        problems.append("no proofs/proof-*.bin in the record")
    proofs["families_proved"] = scope.get("proved_families")
    binding = d / "proofs" / "binding.json"
    proofs["binding"] = _load(binding) if binding.exists() else None
    obligations = summary.get("accounting")

    # 4b. kind coverage: the record-attached PROOF-COV family proofs (embed-gather +
    # ordinary inactive pad-serve statements), every cited hash recomputed from the files.
    # Absence is a binding problem under --require-coverage (the complete-audit certificate);
    # a PRESENT summary whose pins do not recompute is a problem unconditionally.
    cov_file = d / "proofs" / "proof-cov" / "cov_summary.json"
    coverage = None
    if cov_file.exists():
        cs = _load(cov_file)
        cov_problems: list[str] = list(cs.get("problems") or [])
        n_attached = n_files = 0
        for b in cs.get("batches") or []:
            for mode, docm in (b.get("modes") or {}).items():
                if not (docm.get("proved_cuda") or {}).get("verdict"):
                    cov_problems.append(f"{b['name']} {mode}: proving record verdict not True")
                vc = docm.get("verified_cpu")
                if vc is not None and not vc.get("ok"):
                    cov_problems.append(f"{b['name']} {mode}: CPU-mode verify not ok")
                att = docm.get("attached")
                if att:
                    f = d / att["file"]
                    if not f.exists() or sha256_file(f) != att["sha256"]:
                        cov_problems.append(f"{b['name']} {mode}: attached proof missing or sha mismatch")
                    else:
                        n_files += 1
            if b.get("attached"):
                n_attached += 1
                st = d / "proofs" / "proof-cov" / b["name"] / "statement.bin"
                if not st.exists() or sha256_file(st) != b.get("statement_sha256"):
                    cov_problems.append(f"{b['name']}: statement.bin missing or sha mismatch")
        prefill = max(((b.get("prefill_units") or 0) for b in cs.get("batches") or []), default=0)
        decode = max(((b.get("decode_units") or 0) for b in cs.get("batches") or []), default=0)
        exclusions = cs.get("prefill_exclusions") or []
        kinds = cs.get("kinds_proved") or []
        if decode <= 0:
            cov_problems.append("no decode embedding rows proved")
        if prefill <= 0 and not exclusions:
            cov_problems.append("no prefill embedding rows and no labeled exclusion")
        if not any("PadGuard" in k for k in kinds) or not any("PadRowMux" in k for k in kinds):
            cov_problems.append(f"ordinary inactive kinds missing from {kinds}")
        cov_host = ((cs.get("program_identity") or {}).get("host_sha256"))
        cov_bin = d / "bin" / "veritor-zk-host-cov"
        host_bound = cov_bin.exists() and sha256_file(cov_bin) == cov_host
        if not host_bound:
            cov_problems.append("family-proof host binary bin/veritor-zk-host-cov absent or unbound")
        # the family-proof guest is a second executable: it must be approved too
        cov_pi = cs.get("program_identity") or {}
        cov_guest, cg_problems, cg_warnings = check_guest(cov_pi.get("elf_sha256"), cov_pi.get("vk_hash"), cov_pi.get("guest_build"), strict=not args.historical_v2,
                                                          policy_path=Path(args.approved_guests) if args.approved_guests else None, what="proof-cov guest")
        cov_problems.extend(cg_problems)
        for w in cg_warnings:
            print(f"[certificate] WARNING: {w}", file=sys.stderr)
        problems.extend(cov_problems)
        coverage = {
            "kinds_proved": kinds,
            "families": cs.get("families"),
            "batches_attached": n_attached, "proof_files_rebound": n_files,
            "embed": {"prefill_units": prefill, "decode_units": decode, "exclusions": exclusions,
                      "weights_sha256": cs.get("weights_sha256"), "table_root": cs.get("table_root")},
            "prefill_binding": cs.get("prefill_binding"),
            "program_identity": cs.get("program_identity"),
            "guest": {**cov_guest, "warnings": cg_warnings},
            "problems": cov_problems,
        }
    elif getattr(args, "require_coverage", False):
        problems.append("kind coverage incomplete: no proofs/proof-cov/cov_summary.json "
                        "(embed-gather + ordinary-inactive proofs not attached to the record)")

    # 5. enforcement + correspondence
    served = _load(d / "served_tokens.json") if (d / "served_tokens.json").exists() else None
    if served is None:
        problems.append("served_tokens.json missing (served-token correspondence not recorded)")
    elif not served.get("ok"):
        problems.append("served-token correspondence recorded NOT ok")
    # [PAD-INT, additive] the served transcript is a committed input (run_composed pins inputs.served_tokens_sha256 and
    # verify_composed re-hashes it); the certificate must too, or an edited transcript (a token appended after the EOS
    # of an early-terminated request) keeps a BOUND certificate — found by padint_negatives.py N3b on PADINT-B0-20260909T082901Z.
    if served is not None and inp.get("served_tokens_sha256"):
        pin("served_tokens.json", "served_tokens_sha256")
    # the declaration as ENFORCED: read from the record's own description, not from the profile file
    declared = None
    desc_doc = None
    for rel in ("adapter/selective/description.json", "adapter/commit-v2/description.json"):
        desc_file = d / rel
        if desc_file.exists():
            if inp.get("description_sha256") and sha256_file(desc_file) != inp["description_sha256"]:
                problems.append(f"{rel} sha != manifest inputs.description_sha256")
            desc_doc = _load(desc_file)
            break
    else:
        # the record's bundle tarball carries the run's own description.json -- accepted
        # ONLY when it hashes to the manifest's inputs.description_sha256 pin
        import hashlib

        raw = _bundle_file(d, "description.json")
        if raw is None:
            problems.append("description.json missing from the record (adapter/ stripped and no pinned bundle copy)")
        elif inp.get("description_sha256") and hashlib.sha256(raw).hexdigest() != inp["description_sha256"]:
            problems.append("bundle description.json sha != manifest inputs.description_sha256")
        else:
            desc_doc = json.loads(raw)
    for node in (desc_doc or {}).get("nodes") or []:
        if node.get("op") == "sampler.select":
            declared = node.get("declared_domain")
    linkage_rows = sv.get("output_linkage") or []  # one row per request: {request_id, checked, problems}
    linkage_ok = bool(linkage_rows) and all(r.get("checked", 0) > 0 and not r.get("problems") for r in linkage_rows)
    active = declared == 16 and linkage_ok
    if not active:
        problems.append(f"16-bit output-domain enforcement NOT active (declared_domain={declared}, "
                        f"output_linkage rows={len(linkage_rows)}, all checked clean={linkage_ok})")
    enforcement = {
        "declared_domain": declared, "active": active,
        "output_linkage": {"requests": len(linkage_rows), "leaves_checked": sum(r.get("checked", 0) for r in linkage_rows),
                           "problems": [p for r in linkage_rows for p in (r.get("problems") or [])]},
        "note": "ACTIVE iff the record's description declares declared_domain:16 on sampler.select and output linkage checked every request clean",
        "served_token_correspondence": (None if served is None else {"ok": served.get("ok"), "requests": len(served.get("requests") or [])}),
    }

    # 6. the reconciled cost record
    sp = d / "adapter" / "selective" / "selective_prover.json"
    if sp.exists():
        split = _load(sp).get("leaf_hash_timer_split")
    else:
        raw = _bundle_file(d, "selective_prover.json")
        split = json.loads(raw).get("leaf_hash_timer_split") if raw is not None else None
    cost = {
        "wall_seconds": summary.get("wall_seconds"), "protocol_wall_seconds": summary.get("protocol_wall_seconds"),
        "timings_seconds": summary.get("timings_seconds"),
        "T_native_sha256": pin("production_timings.json", "production_timings_sha256") if (d / "production_timings.json").exists() else None,
        "leaf_hash_timer_split": split,
        "verifier_cost": scope.get("verifier_cost"),
    }
    binaries = {}
    for name, sha in (inp.get("archived_binaries") or {}).items():
        f = d / "bin" / name
        ok = f.exists() and sha256_file(f) == sha
        binaries[name] = {"sha256": sha, "present_and_bound": ok}
        if not ok:
            problems.append(f"archived binary bin/{name} absent or unbound")

    verdict = summary.get("verdict")
    if verdict != "PASS":
        problems.append(f"record verdict is {verdict}, not PASS")
    # S4 (user ruling 11:54 PT): the legacy path stays available but is UNCERTIFIED in every output -- loud, first key, in the status word and on stdout;
    # `bound` keeps its old meaning (the record's own commitments/proofs/binaries are mutually consistent -- what the negatives test) and is NOT certification
    cert = {
        "schema": "veritor/whole-program-certificate/v2" if args.historical_v2 else "veritor/whole-program-certificate/v3",
        **({"UNCERTIFIED": "LEGACY --historical-v2 PATH: this document is NOT a v3 certificate; attests_protocol_instance=false; no mandatory v3 check was run; no Bound is "
                           "attached; integration evidence only (GUIDANCE §57.10 S2/S4)", "certification_status": "UNCERTIFIED"} if args.historical_v2
           else {"certification_status": "certified-v3" if not problems else "not-bound"}),
        "certified": (not problems) and not args.historical_v2,
        "protocol_instance": protocol_instance,
        "guest_policy": {"file": "e2e/approved_guests.json" if not args.approved_guests else str(args.approved_guests),
                         "enforced": not args.historical_v2,
                         "note": ("STRICT: every proving executable (ELF, vk) is one the verifier approves; a provisional-legacy approval is NOT build-verified (see proofs.guest.approval)"
                                  if not args.historical_v2 else
                                  "--historical-v2: NOT ENFORCED -- unapproved executables are only warned about; this certificate attests no protocol instance and no guest approval")},
        "instance": instance, "program": program, "sampling_policy": sampling,
        "commitments": commitments, "proofs": proofs, "obligations_accounting": obligations,
        "kind_coverage": coverage,
        "enforcement": enforcement, "cost_record": cost, "archived_binaries": binaries,
        "record_verdict": verdict, "binding_problems": problems, "bound": not problems,
    }
    out = Path(args.out) if args.out else d / "certificate.json"
    write_json(out, cert)
    if args.historical_v2:
        print("[certificate] ******** UNCERTIFIED: LEGACY --historical-v2 PATH -- NOT a v3 certificate; no protocol instance attested; no Bound ********")
    print(f"[certificate] {('UNCERTIFIED legacy v2 -- ' + ('record internally consistent (bound=true means ONLY that; certified=false)' if not problems else 'NOT BOUND (record inconsistent; certified=false)')) if args.historical_v2 else ('BOUND (certified v3)' if not problems else 'NOT BOUND')} -> {out} (sha256 {sha256_file(out)})")
    for p in problems:
        print(f"[certificate]   problem: {p}")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
