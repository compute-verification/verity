"""[REC 2026-09-09] The corrected record's three negatives (GUIDANCE §57.10 deliverable 4): each edits a hard-linked COPY of the
positive record and runs the STRICT v3 entry point (``e2e.certificate``, no ``--historical-v2``); every run must end NOT BOUND.

* ``tamper-value``        one committed value of a sampled unit's witness (``obligations/witness.bin``, first byte flipped): the exported
                          record's witness digest no longer matches; the certificate's records/proofs binding refuses.
* ``drop-obligation``     one sampled (security) obligation removed from ``obligations/composed.json``: the statement digest ``h_L`` and the
                          obligation count differ from the manifest / proofs -> refused.
* ``swap-seed``           the verifier's revealed seed in ``verify/challenge_transcript.json`` replaced by another 32 bytes: the seed no longer
                          opens the committed seed commitment inside T_0 and r_q / r_s do not re-derive -> the chronology check refuses.

Usage:  python -m e2e.rec_negatives --record RECORD --out DIR
Writes DIR/rec_negatives.json with the three runs (mutation, certificate problems, refused).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _clone(record: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(record, dst, copy_function=os.link, ignore=shutil.ignore_patterns("reverify", "adapter", "bundle", "handshake", "logs"))


def _rewrite_json(path: Path, doc: dict) -> None:
    path.unlink()  # break the hardlink: the source record is never modified
    path.write_text(json.dumps(doc, indent=1) + "\n")


def _rewrite_bytes(path: Path, data: bytes) -> None:
    path.unlink()
    path.write_bytes(data)


def mut_tamper_value(d: Path) -> str:
    w = d / "obligations" / "witness.bin"
    raw = bytearray(w.read_bytes())
    if not raw:
        raise AssertionError("empty witness")
    i = len(raw) // 2
    raw[i] ^= 0x01
    _rewrite_bytes(w, bytes(raw))
    return f"obligations/witness.bin byte {i} flipped (one committed value of a sampled unit changed by 1 bit)"


def mut_drop_obligation(d: Path) -> str:
    p = d / "obligations" / "composed.json"
    doc = json.loads(p.read_text())
    obj = doc.get("object") or doc
    key = next((k for k in ("obligations", "units", "items") if isinstance(obj.get(k), list) and obj[k]), None)
    if key is None:
        # fall back: the statement hex is the exported object; drop 32 bytes of it (one obligation's leaf) -- still a dropped obligation
        if "statement" in doc and len(doc["statement"]) > 128:
            doc["statement"] = doc["statement"][:-64]
            _rewrite_json(p, doc)
            return "obligations/composed.json statement shortened by one leaf digest (an obligation dropped from the exported object)"
        raise AssertionError(f"no obligation list in composed.json (keys {sorted(doc)[:8]})")
    dropped = obj[key].pop()
    _rewrite_json(p, doc)
    return f"obligations/composed.json: dropped 1 of {len(obj[key]) + 1} obligations ({str(dropped)[:60]}...)"


def mut_swap_seed(d: Path) -> str:
    t = d / "verify" / "challenge_transcript.json"
    doc = json.loads(t.read_text())
    old = doc["reveal"]["seed"]
    new = bytes(b ^ 0xFF for b in bytes.fromhex(old)).hex()
    doc["reveal"]["seed"] = new
    _rewrite_json(t, doc)
    # the chronology copies the seed: swap it there too (an attacker rewriting the transcript would)
    c = d / "chronology.json"
    if c.exists():
        cd = json.loads(c.read_text())
        for e in cd.get("events", []):
            if e.get("name") == "seed_reveal":
                e["seed"] = new
        cd["transcript"]["sha256"] = __import__("hashlib").sha256(t.read_bytes()).hexdigest()
        _rewrite_json(c, cd)
    return f"verify/challenge_transcript.json reveal.seed {old[:12]}... -> {new[:12]}... (chronology.json seed_reveal swapped alike, transcript sha re-pinned)"


CASES = {"tamper-value": mut_tamper_value, "drop-obligation": mut_drop_obligation, "swap-seed": mut_swap_seed}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--record", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    record, out = Path(a.record).resolve(), Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}
    all_refused = True
    for name, mutate in CASES.items():
        work = out / name
        _clone(record, work)
        what = mutate(work)
        cert = out / f"{name}-certificate.json"
        p = subprocess.run([sys.executable, "-m", "e2e.certificate", "--record", str(work), "--out", str(cert)], capture_output=True, text=True)
        doc = json.loads(cert.read_text()) if cert.exists() else {}
        problems = list(doc.get("binding_problems") or [])
        refused = p.returncode != 0 and not doc.get("bound") and not doc.get("certified", False)
        all_refused &= refused
        results[name] = {"mutation": what, "certificate_rc": p.returncode, "bound": doc.get("bound"), "certified": doc.get("certified"),
                         "certification_status": doc.get("certification_status"), "problems": problems[:6], "refused": refused,
                         "stdout_tail": p.stdout.strip().splitlines()[-3:]}
        print(f"[rec_negatives] {name:16s} {'NOT BOUND (refused)' if refused else 'STILL BOUND -- NEGATIVE FAILED'}: {problems[:1]}")
        shutil.rmtree(work)
    (out / "rec_negatives.json").write_text(json.dumps({"schema": "veritor/rec-negatives/v1", "record": str(record), "all_refused": all_refused, "runs": results},
                                                        indent=1) + "\n")
    print(f"[rec_negatives] {'ALL THREE REFUSED' if all_refused else 'SOME NEGATIVE NOT REFUSED'} -> {out / 'rec_negatives.json'}")
    return 0 if all_refused else 1


if __name__ == "__main__":
    sys.exit(main())
