"""The audit-definition negatives that mutate a COMPLETED record (GUIDANCE §36.1 / §34).

The five required negative tests and where each lives:

* **mismatched policies** — here: the declared pre-challenge sampling policy is mutated to a
  different ``k`` (with the manifest re-pinned, so hash checks alone cannot save the verifier);
  ``verify_composed``'s ``format`` row must refuse on realized-vs-declared, not on a hash.
  A second variant tampers the policy file without re-pinning: the hash binding must refuse.
* **missing required proofs** — here: a selected obligation's ``proof-*.bin`` is deleted; the
  ``proofs`` row must refuse (nothing silently substitutes a replay for the missing proof).
* **invalid output domains** — ``tests/test_output_domain.py`` (consistent out-of-domain claims
  rejected on every audit; run by e2e_b0.sh stage 9).
* **inconsistent authentication** — the wa-m12/wa-m8b artefact-mutation negatives
  (``run_composed --negative``; refused by coverage/placement/authentication checks).
* **empty-sample output-check** — the ``--k 0``-style capture: output typing runs on every
  token output when the random sample is empty (pod capture stage 2).

Usage: ``python -m e2e.negative_audit --record <completed run dir> --out <dir> [--host BIN]``.
Each case hardlink-copies the record, applies one mutation, runs ``e2e.verify_composed`` and
asserts the PRECISE refusing row.  Exit 0 iff every negative is refused by its intended check.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from e2e.run_composed import sha256_file, write_json


def _clone(record: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(record, dst, copy_function=os.link, ignore=shutil.ignore_patterns("reverify"))


def _rewrite(path: Path, doc: dict) -> None:
    path.unlink()  # break the hardlink before writing: never mutate the source record
    path.write_text(json.dumps(doc, indent=1) + "\n")


def _mut_policy_mismatch(d: Path) -> str:
    pol = json.loads((d / "sampling_policy.json").read_text())
    pol["k"] = (pol.get("k") or 0) + 1
    pol["q_declared"] = f"{pol['k']}/{pol.get('requests_total')}"
    _rewrite(d / "sampling_policy.json", pol)
    man = json.loads((d / "manifest.json").read_text())
    man["inputs"]["sampling_policy_sha256"] = sha256_file(d / "sampling_policy.json")  # re-pinned: hashes agree
    _rewrite(d / "manifest.json", man)
    return f"declared k rewritten to {pol['k']} and re-pinned; realized |J| unchanged"


def _mut_policy_tamper(d: Path) -> str:
    pol = json.loads((d / "sampling_policy.json").read_text())
    pol["s_declared"] = str(pol.get("s_declared") or "0") + "1"  # s is a fraction string: "1/4096" -> "1/40961"
    _rewrite(d / "sampling_policy.json", pol)  # manifest pin NOT updated
    return "sampling_policy.json edited without re-pinning the manifest"


def _mut_missing_proof(d: Path) -> str:
    proofs = sorted((d / "proofs").glob("proof-*.bin"))
    if not proofs:
        raise AssertionError("record has no proofs/proof-*.bin -- nothing to delete")
    proofs[0].unlink()
    return f"deleted {proofs[0].relative_to(d)} ({len(proofs)} were present)"


def _mut_missing_family_proof(d: Path) -> str:
    proofs = sorted((d / "proofs" / "proof-cov").glob("*/proof-*.bin"))
    if not proofs:
        raise AssertionError("record has no proofs/proof-cov/*/proof-*.bin -- nothing to delete")
    proofs[0].unlink()
    return f"deleted {proofs[0].relative_to(d)} ({len(proofs)} family proof files were present)"


CASES = {
    "policy-mismatch": (_mut_policy_mismatch, "format"),
    "policy-tamper": (_mut_policy_tamper, "format"),
    "missing-proof": (_mut_missing_proof, "proofs"),
    # INT-PAD-3: a deleted kind-coverage (embed / ordinary-inactive) proof must break the
    # whole-program certificate's binding -- refused by e2e.certificate, not verify_composed
    "missing-family-proof": (_mut_missing_family_proof, "certificate"),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--record", required=True, help="a completed positive run_composed directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--host", default=None, help="SP1 host binary; without it proofs verify from the recorded host log")
    args = ap.parse_args()
    record, out = Path(args.record).resolve(), Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    for name, (mutate, expect_row) in CASES.items():
        work = out / name
        _clone(record, work)
        try:
            what = mutate(work)
        except AssertionError as e:
            if name == "missing-family-proof":
                results[name] = {"mutation": None, "expected_row": expect_row,
                                 "skipped": f"not applicable: {e}", "refused_by_intended_check": False}
                print(f"[negative_audit] {name:20s} SKIPPED: {e}")
                shutil.rmtree(work)
                continue
            raise
        if expect_row == "certificate":
            # the coverage negatives are refused by the whole-program certificate's binding
            cmd = [sys.executable, "-m", "e2e.certificate", "--record", str(work),
                   "--require-coverage", "--out", str(out / f"{name}-certificate.json")]
            if not (work / "protocol_instance.json").exists():
                cmd.append("--historical-v2")  # [PROTO-A v3] a pre-instance record: the coverage negative must be refused on ITS check, not on the absent instance
            p = subprocess.run(cmd, capture_output=True, text=True)
            cert_file = out / f"{name}-certificate.json"
            cert = json.loads(cert_file.read_text()) if cert_file.exists() else {}
            cov_problems = [q for q in (cert.get("binding_problems") or []) if "proof" in q or "coverage" in q]
            refused = p.returncode != 0 and not cert.get("bound") and bool(cov_problems)
            results[name] = {"mutation": what, "expected_row": expect_row,
                             "certificate_bound": cert.get("bound"), "problems": cov_problems[:3],
                             "verify_rc": p.returncode, "refused_by_intended_check": refused}
            print(f"[negative_audit] {name:20s} {'REFUSED' if refused else 'NOT REFUSED'} by the certificate binding: {cov_problems[:1]}")
            shutil.rmtree(work)
            continue
        rv_out = out / f"{name}-reverify"
        cmd = [sys.executable, "-m", "e2e.verify_composed", "--dir", str(work), "--out", str(rv_out),
               "--program", "recorded", "--plan", "recorded", "--skip-bundle"]
        cmd += (["--host", args.host] if args.host else ["--allow-recorded-proofs"])
        p = subprocess.run(cmd, capture_output=True, text=True)
        doc = json.loads((rv_out / "reverify.json").read_text()) if (rv_out / "reverify.json").exists() else {}
        row = (doc.get("rows") or {}).get(expect_row) or {}
        refused = p.returncode != 0 and doc.get("fresh_verdict") == "FAIL" and row.get("status") in ("FAIL", "error")
        results[name] = {"mutation": what, "expected_row": expect_row, "row_status": row.get("status"),
                         "row_detail": row.get("detail"), "fresh_verdict": doc.get("fresh_verdict"),
                         "verify_rc": p.returncode, "refused_by_intended_check": refused}
        print(f"[negative_audit] {name:20s} {'REFUSED' if refused else 'NOT REFUSED'} by row '{expect_row}': {row.get('status')} -- {row.get('detail')}")
        shutil.rmtree(work)  # the clones hardlink the weights; drop them once verified

    ok = all(r["refused_by_intended_check"] for r in results.values())
    write_json(out / "negative_audit.json", {"schema": "veritor/negative-audit/v1", "record": str(record), "ok": ok, "cases": results})
    print(f"[negative_audit] {'ALL REFUSED' if ok else 'FAILURE: some mutation was not refused by its intended check'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
