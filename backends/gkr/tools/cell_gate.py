"""E1: the evidence gate of a route (a) cell (red-team-flock third audit).  A Flock session record counts for the cell
only if every check below holds; the gate then writes the ``link.txt`` the prime verifier checks the prime proof
against, so the points, y and root_F the prime proof used are the record's by construction (the Rust verifier refuses
a proof whose root_F, points or y differ: ``verifier/src/link.rs``).

  exchange      ``link_mode == "exchange"`` and a ``link`` block (not the stub)
  gated         ``config.require_link`` is true (E0: the library refuses the exchange without it)
  sigma         Σ in ``hello`` and in ``link`` equals the cell's Σ (SHA-256 of its ``sigma.txt``)
  accepted      the verifier's verdict accepted every rep and the session did not abort
  context       two points of m coordinates and one y per point (m from ``sigma.txt``)
  preserved     every proof the record hashes is kept beside it with that SHA-256, both reps of every table
  non_producer  ``verifier.json``'s operator is not a producer lane, and its pod is not a producer pod
  prime_live    (NON_ZK_PROOF, PROTOCOL 17.6) the session served the prime coins: ``config.prime_live``, ``hello`` names
                live-coins/v1, at least one prime round before Commit (root_F) and one after y, every round well formed
  replayable    (G2) the record keeps root_F, every root_B and the committed public words, matching their hashes
  prime         (with --rust) the prime verifier accepts the prime proof against the record's link.txt and its prime
                rounds (prime_coins.txt, ``--require-live-coins``), with the circuit set and the instance commitment pinned
  flock_replay  (with --flock) ``flock-link replay`` re-verifies the Flock side from the record and the kept proofs
                alone, with the verifier's statement (the leaf digests Σ names); the record's own verdict is not used

    python tools/cell_gate.py --session DIR --sigma sigma.txt --producer LANE [--producer-pod ID]... [--out DIR]
                              [--rust VB --statement DIR --proof proof.bin --vus N --relation R]
                              [--flock BIN --digests leaf_digests.bin --prime-commitment HEX]

Exit 0 iff every check holds; the verdict (``cell-gate/v1``) goes to stdout and ``--out/gate.json``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

LINK_TXT = "link session-2xgf2_128/v2"
PRIME_MODE = "live-coins/v1"
P = 2013265921
DEG = 6


def link_txt(rec: dict, table: str = "chain") -> str:
    ln = rec["link"]
    return (f"{LINK_TXT}\nsigma {ln['sigma']}\nroot_f_sha256 {ln['root_f_sha256']}\nroot_b_sha256 {ln['root_b_sha256'][table]}\n"
            + "".join(f"point {' '.join(p)}\n" for p in ln["points"]) + f"y {' '.join(ln['y'])}\n")


def prime_coins_txt(rec: dict) -> str:
    """The record's prime rounds as the prime verifier reads them (``gpu/transcript.py::coins_text``)."""
    pr = rec["prime"]
    return f"prime-coins {PRIME_MODE}\nbefore_commit {pr['before_commit']}\n" + "".join(
        f"round {r['state']} {r['n']} {' '.join(str(w) for w in r['words'])}\n" for r in pr["rounds"])


def _prime_ok(rec: dict, hello: dict) -> tuple[bool, str]:
    pr = rec.get("prime") or {}
    rounds, before = pr.get("rounds") or [], pr.get("before_commit")
    shaped = all(isinstance(r.get("state"), str) and len(r["state"]) == 64 and int(r.get("n") or 0) > 0
                 and len(r.get("words") or []) == DEG * int(r["n"]) and all(0 <= int(w) < P for w in r["words"]) for r in rounds)
    ok = ((rec.get("config") or {}).get("prime_live") is True and hello.get("prime") == PRIME_MODE and pr.get("mode") == PRIME_MODE
          and isinstance(before, int) and 1 <= before < len(rounds) and shaped)
    return ok, f"prime_live {(rec.get('config') or {}).get('prime_live')!r}, hello {hello.get('prime')!r}, {len(rounds)} rounds, before_commit {before!r}, well formed {shaped}"


def _replayable(ln: dict) -> tuple[bool, str]:
    h = lambda s: hashlib.sha256(bytes.fromhex(s)).hexdigest()   # noqa: E731
    try:
        ok = (h(ln["root_f"]) == ln["root_f_sha256"] and set(ln["root_b"]) == set(ln["root_b_sha256"]) and set(ln["publics"]) == set(ln["publics_sha256"])
              and all(h(v) == ln["root_b_sha256"][t] for t, v in ln["root_b"].items())
              and all(h(v) == ln["publics_sha256"][t] for t, v in ln["publics"].items()) and all(ln["publics"].values()))
        return ok, f"root_f, root_b {sorted(ln['root_b'])}, publics {[len(v) // 2 for v in ln['publics'].values()]} bytes"
    except (KeyError, TypeError, ValueError) as e:
        return False, f"missing or malformed: {e!r}"


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def check(session: Path, sigma_txt: Path, producers: list[str], producer_pods: list[str]) -> tuple[dict, dict]:
    rec = json.loads((session / "session.json").read_text())
    sig_text = sigma_txt.read_text()
    sigma = hashlib.sha256(sig_text.encode()).hexdigest()
    m = int(next(ln.split()[1] for ln in sig_text.splitlines() if ln.startswith("m ")))
    hello = json.loads(rec.get("hello") or "{}")
    ln = rec.get("link") or {}
    verdict = rec.get("verdict") or {}
    c: dict[str, tuple[bool, str]] = {}
    c["exchange"] = (rec.get("link_mode") == "exchange" and ln.get("mode") == "exchange", f"link_mode {rec.get('link_mode')!r}")
    c["gated"] = ((rec.get("config") or {}).get("require_link") is True, f"require_link {(rec.get('config') or {}).get('require_link')!r}")
    c["sigma"] = ((hello.get("link") or {}).get("sigma") == sigma and ln.get("sigma") == sigma,
                  f"hello Σ {(hello.get('link') or {}).get('sigma')}, link Σ {ln.get('sigma')}, cell Σ {sigma}")
    reps = verdict.get("reps") or []
    c["accepted"] = (verdict.get("accepted") is True and rec.get("aborted") is None and len(reps) > 0 and all(r.get("ok") for r in reps),
                     f"accepted {verdict.get('accepted')!r}, aborted {rec.get('aborted')!r}, reps {[r.get('ok') for r in reps]}")
    pts, y = ln.get("points") or [], ln.get("y") or []
    c["context"] = (len(pts) == 2 and all(len(p) == m for p in pts) and len(y) == 2 and (hello.get("link") or {}).get("points") == 2
                    and (hello.get("link") or {}).get("m") == m, f"{len(pts)} points of {[len(p) for p in pts]}, {len(y)} y, m {m}")
    want = {f"{t}/rep{r}" for t in (ln.get("root_b_sha256") or {}) for r in range((rec.get("config") or {}).get("reps", 0))}
    got = {}
    for e in rec.get("proof_sha256") or []:
        f = session / f"{e['stream'].replace('/', '.')}.bin"
        got[e["stream"]] = f.is_file() and sha256_file(f) == e["sha256"]
    c["preserved"] = (bool(want) and set(got) == want and all(got.values()), f"proofs kept {got}, want {sorted(want)}")
    c["prime_live"] = _prime_ok(rec, hello)
    c["replayable"] = _replayable(ln)
    vj = json.loads((session / "verifier.json").read_text()) if (session / "verifier.json").is_file() else {}
    op, pod = vj.get("operator") or "", vj.get("runpod_pod_id") or ""
    c["non_producer"] = (bool(op) and op not in producers and (not pod or pod not in producer_pods),
                         f"operator {op!r} pod {pod!r}; producers {producers} pods {producer_pods}")
    return {k: {"ok": v[0], "detail": v[1]} for k, v in c.items()}, rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="tools/cell_gate.py")
    ap.add_argument("--session", required=True)
    ap.add_argument("--sigma", required=True)
    ap.add_argument("--producer", action="append", default=[])
    ap.add_argument("--producer-pod", action="append", default=[])
    ap.add_argument("--out", default=None)
    ap.add_argument("--rust", default=None)
    ap.add_argument("--statement", default=None)
    ap.add_argument("--proof", default=None)
    ap.add_argument("--vus", type=int, default=None)
    ap.add_argument("--relation", default="bf16-ampere+blake3")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--flock", default=None, help="flock-link: replay the Flock side from the record (G2)")
    ap.add_argument("--digests", default=None)
    ap.add_argument("--prime-commitment", default=None)
    a = ap.parse_args(argv)
    if not a.producer:
        ap.error("--producer: the lane(s) that produced the proofs")
    session = Path(a.session)
    checks, rec = check(session, Path(a.sigma), a.producer, a.producer_pod)
    out = Path(a.out) if a.out else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
    text = link_txt(rec) if checks["exchange"]["ok"] and checks["context"]["ok"] else None
    if a.rust:
        if not (a.statement and a.proof and out and text):
            checks["prime"] = {"ok": False, "detail": "no link context, or --statement / --proof / --out missing"}
        else:
            vd = out / "statement"
            shutil.copytree(a.statement, vd, dirs_exist_ok=True)
            (vd / "link.txt").write_text(text)
            (vd / "prime_coins.txt").unlink(missing_ok=True)
            if checks["prime_live"]["ok"]:
                (vd / "prime_coins.txt").write_text(prime_coins_txt(rec))
            shutil.copyfile(a.sigma, vd / "sigma.txt")
            cmd = [a.rust, "verify", "--dir", str(vd), "--proof", a.proof, "--threads", str(a.threads), "--relation", a.relation,
                   "--require-live-coins", "--json", str(out / "prime_verify.json")] + (["--vus", str(a.vus)] if a.vus else [])
            r = subprocess.run(cmd, capture_output=True, text=True)
            doc = json.loads((out / "prime_verify.json").read_text()) if (out / "prime_verify.json").is_file() else {}
            checks["prime"] = {"ok": r.returncode == 0 and doc.get("accepted") is True and doc.get("circuit_pinned") is True
                               and (doc.get("commitment") or {}).get("pinned") is True,
                               "detail": f"rc {r.returncode}, accepted {doc.get('accepted')}, circuit pinned {doc.get('circuit_pinned')}, "
                                         f"commitment pinned {(doc.get('commitment') or {}).get('pinned')}, "
                                         f"error {doc.get('error')}, {doc.get('verify_seconds')} s", "proof_sha256": sha256_file(Path(a.proof))}
    if a.flock:
        if not (a.digests and a.prime_commitment and a.vus):
            checks["flock_replay"] = {"ok": False, "detail": "--digests, --prime-commitment and --vus are needed"}
        else:
            cmd = [a.flock, "replay", "--session", str(session), "--vus", str(a.vus), "--digests", a.digests,
                   "--prime-commitment", a.prime_commitment] + (["--json", str(out / "flock_replay.json")] if out else [])
            r = subprocess.run(cmd, capture_output=True, text=True)
            line = next((ln.split("\t", 1)[1] for ln in r.stdout.splitlines() if ln.startswith("REPLAY\t")), "{}")
            rv = json.loads(line)
            checks["flock_replay"] = {"ok": r.returncode == 0 and rv.get("accepted") is True,
                                      "detail": f"rc {r.returncode}, accepted {rv.get('accepted')}, why {rv.get('why')!r}, {rv.get('replay_s')} s",
                                      "digests_sha256": sha256_file(Path(a.digests))}
    ok = all(v["ok"] for v in checks.values())
    doc = {"kind": "cell-gate/v1", "admitted": ok, "session": str(session), "sigma": hashlib.sha256(Path(a.sigma).read_bytes()).hexdigest(),
           "record_sha256": sha256_file(session / "session.json"), "checks": checks}
    if out:
        (out / "gate.json").write_text(json.dumps(doc, indent=1))
        if text:
            (out / "link.txt").write_text(text)
        if checks["prime_live"]["ok"]:
            (out / "prime_coins.txt").write_text(prime_coins_txt(rec))
    print(json.dumps(doc, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
