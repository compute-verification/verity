"""The memory challenger's process in the harness's timed audit (`p2-v3-mc-gpu`): `verity.protocols.accounting.space.pous.
memory_challenger` against the responder, over the harness's responder wire (`band_gpu/live.py`).

    python benchmarks/pous/memory_challenger/live.py --ready READY.json --leaves DIGESTS.bin --out AUDIT.json [--audits N]
        [--pings N] [--k K] [--verifier-cpu C] [--limit-ns N] [--calibrate-only] [--clock logical] [--strict] [--signer test]

`READY.json` names the responder's port, the scheme, its manifest and the setup salt.  `DIGESTS.bin` is SHA-512 of every
block of `C`, in block order: what `memory_challenger.verdict` checks each round's `H(block)` against, the verifier-side
table a proof will replace.

- **The limit** is `Δ + RTT_ALLOWANCE_CAP_NS` (spec v1.1's 354 µs cap, `Δ + RTT = 854 µs`), fixed before any load.  The
  calibration (`--calibrate-only`) still measures the loopback round trip and records whether its p99.9 fits under the cap.
- **A session** draws a fresh session id and challenger secret, commits its coin tree, runs `k` rounds (the certificate's
  `k`), signs once and opens its coins.  `--strict` refuses a `k` or limit no certificate covers (`covering_certificate`)
  before any round.
- **The verdict** is `memory_challenger.verdict` under `memory_challenger.SCHEMES`, which holds no scheme until S1 lands, so
  every session is rejected: it fails closed.  With `--signer test` the sessions are signed by the PoUS tests' XMSS-style
  test signer (`pous_test_signer.py`), and `accepted_under_test_signer` is the same verdict under that signer's own mapping:
  every check but the scheme's acceptance.  It is never the verdict.
- **The record** separates what is published (`public`: the commitment, the verdict, the coin openings) from the witness
  (`witness`: the signed session, the reasons, per-round on-time and digest bits), and keeps the responder's reports beside.

Exit 0 when every session passes under the signer the run used, 1 otherwise.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from band_gpu.live import (  # noqa: E402
    ALLOWANCE_KIND,
    HEADER,
    OP_CHALLENGE,
    LiveResponder,
    LogicalClock,
    calibrate,
    calibrate_only,
    cgroup_cpu,
    params_of,
    quantiles,
    scheme_of,
)

ROOT = Path(__file__).resolve().parents[3]
TEST_SIGNER = ROOT / "verity" / "protocols" / "accounting" / "space" / "pous" / "tests" / "pous_test_signer.py"


def test_signer() -> Any:
    """The PoUS tests' XMSS-style test signer module (`SCHEME`, `TestSigner`, `verify`, `SCHEMES`)."""
    spec = importlib.util.spec_from_file_location("pous_test_signer", TEST_SIGNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rtt_cap_ns() -> int:
    from verity.protocols.accounting.space.pous.schemes import p2
    return p2.RTT_ALLOWANCE_CAP_NS


def binding_of(salt: bytes, digests: bytes) -> bytes:
    """What a session audits: SHA-512 of the setup salt and the table of block digests."""
    return hashlib.sha512(b"verity-pous/bench/memory-challenger/binding/v1\0" + salt + hashlib.sha512(digests).digest()).digest()


def judge(sessions: list[dict[str, Any]], expected: Callable[[int], bytes], trusted: set[bytes],
          test_schemes: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """Each session's verdict under `memory_challenger.SCHEMES` and, when given, under the test signer's mapping."""
    from verity.protocols.accounting.space.pous import memory_challenger as mc

    out = []
    for s in sessions:
        v = mc.verdict(s["signed"], s["commitment"], s["openings"], expected, trusted=trusted)
        t = mc.verdict(s["signed"], s["commitment"], s["openings"], expected, trusted=trusted,
                       schemes=test_schemes) if test_schemes is not None else None
        rounds = s["signed"].rounds
        answers = [{"index": r.index, "elapsed_ns": r.elapsed_ns, "device_ns": d, "server_ns": sv,
                    "correct": r.digest == expected(r.index), "on_time": r.elapsed_ns <= s["commitment"].limit_ns}
                   for r, (d, sv) in zip(rounds, s["reports"], strict=True)]
        out.append({"answers": answers,
                    "public": {"commitment": s["commitment"].to_json(), "verdict": v.public(),
                               "openings": [o.to_json() for o in s["openings"]]},
                    "witness": {"signed": s["signed"].to_json(), "reasons": list(v.reasons),
                                "reasons_under_test_signer": None if t is None else list(t.reasons)},
                    "accepted": v.accepted, "accepted_under_test_signer": None if t is None else t.accepted,
                    "sign_ns": s["sign_ns"]})
    return out


def signature_control(doc: dict[str, Any], expected: Callable[[int], bytes]) -> dict[str, Any]:
    """The bad-signature control, on the honest audit's signed sessions: each with one signature bit flipped, and each with
    one round's receipt time moved 1 ns earlier after signing (an on-time bit forged), judged under the test signer's
    mapping; every one must be rejected for its signature, and the unedited sessions must pass."""
    from verity.protocols.accounting.space.pous import memory_challenger as mc

    ts = test_signer()
    out = {"sessions": 0, "unedited_accepted": 0, "flipped_rejected": 0, "edited_rejected": 0}
    for rec in doc.get("records") or []:
        signed = mc.SignedSession.from_json(rec["witness"]["signed"])
        commitment = mc.SessionCommitment.from_json(rec["public"]["commitment"])
        openings = [mc.CoinOpening.from_json(o) for o in rec["public"]["openings"]]
        trusted = {signed.public_key}

        def reasons(s: mc.SignedSession) -> tuple[str, ...]:
            return mc.verdict(s, commitment, openings, expected, trusted=trusted, schemes=ts.SCHEMES).reasons

        sig = bytearray(signed.signature)
        sig[len(sig) // 2] ^= 1
        r = signed.rounds[len(signed.rounds) // 2]
        rounds = list(signed.rounds)
        rounds[r.j] = mc.Round(r.j, r.index, r.send_ns, r.receipt_ns - 1, r.digest)
        bad = "the signature does not verify"
        out["sessions"] += 1
        out["unedited_accepted"] += reasons(signed) == ()
        out["flipped_rejected"] += bad in reasons(mc.SignedSession(commitment, signed.ended, signed.rounds, signed.scheme,
                                                                   signed.public_key, bytes(sig)))
        out["edited_rejected"] += bad in reasons(mc.SignedSession(commitment, signed.ended, tuple(rounds), signed.scheme,
                                                                  signed.public_key, signed.signature))
    out["passed"] = bool(out["sessions"]) and out["flipped_rejected"] == out["edited_rejected"] == out["sessions"]
    return out


def audit(host: str, port: int, scheme: Any, salt: bytes, digests: bytes, *, audits: int, pings: int, k: int | None = None,
          cpu: int | None = None, allowance: str = "p99.9", deadline_ns: int | None = None, limit_ns: int | None = None,
          clock: Callable[[], int] = time.perf_counter_ns, strict: bool = True, signer: str = "test") -> dict[str, Any]:
    """`audits` sessions of the memory challenger against `limit_ns` (else `Δ` plus the cap), each judged after all of them
    ran, off the timed path."""
    from verity.protocols.accounting.space.pous import audit as pous_audit
    from verity.protocols.accounting.space.pous import memory_challenger as mc
    from verity.protocols.accounting.space.pous.protocol import StorageProfile

    if signer != "test":
        raise ValueError(f"no signer {signer!r}: S1's scheme has not landed, and the test signer is 'test'")
    params = scheme.params
    blocks, nb = params.blocks, params.block_bytes
    if len(digests) != 64 * blocks:
        raise ValueError(f"{len(digests) // 64} block digests for a store of {blocks} blocks")
    deadline_ns = deadline_ns or pous_audit.DEADLINE_NS
    cap = rtt_cap_ns()
    fixed = limit_ns is not None
    limit = limit_ns if fixed else deadline_ns + cap
    cert = scheme.certificate
    k = k or (cert.k if cert is not None else pous_audit.CHALLENGES)
    base = {"verifier": "memory-challenger", "protocol": mc.PROTOCOL, "coin_tree": mc.COIN_TREE, "records_tree": mc.RECORDS_TREE,
            "strict": strict, "limit_ns": limit, "limit_fixed_before_load": fixed, "deadline_ns": deadline_ns,
            "allowance": "cap", "allowance_kind": ALLOWANCE_KIND, "rtt_allowance_ns": limit - deadline_ns, "rtt_cap_ns": cap,
            "k": k, "scheme": scheme.name}
    gaps: tuple[str, ...] = ()
    try:
        cert = mc.covering_certificate(scheme, k, limit, experimental=True)
        if cert.segments is not None and params.segments != cert.segments:
            gaps = (f"W has {params.segments} segments; {cert.theorem} is about {cert.segments}",)
    except ValueError as e:
        if strict:
            return {**base, "refused": str(e), "audits": 0, "audits_accepted": 0, "audits_accepted_under_test_signer": 0,
                    "answers": 0, "late": 0, "wrong": 0, "records": []}
        cert, gaps = None, (str(e),)
    ts = test_signer()
    sig = ts.TestSigner(os.urandom(32), height=max(1, (max(1, audits) - 1).bit_length()))
    binding = binding_of(salt, digests)
    resolution = None if clock is time.perf_counter_ns else 1
    if cpu is not None:
        os.sched_setaffinity(0, {cpu})
    gc.collect()
    gc.freeze()
    gc.disable()
    r = LiveResponder(host, port, nb, clock)
    cpu_before = cgroup_cpu()
    sessions = []
    try:
        before = calibrate(r, pings)
        for _ in range(audits):
            coins = mc.CoinTree.draw(os.urandom(32), os.urandom(32), blocks, k)
            c = mc.MemoryChallenger(coins, sig, limit_ns=limit, block_bytes=nb, binding=binding, clock=clock,
                                    resolution_ns=resolution)
            reports = []
            while not c.done:
                i = c.challenge()
                r.roundtrip(OP_CHALLENGE, i)
                c.receive(r.view[HEADER:])
                reports.append((r.device_ns, r.server_ns))
            t0 = time.perf_counter_ns()
            signed = c.close()
            sessions.append({"commitment": c.commitment, "signed": signed, "openings": c.openings(), "reports": reports,
                             "sign_ns": time.perf_counter_ns() - t0})
        after = calibrate(r, pings)
    finally:
        r.close()
        gc.enable()
    expected = lambda i: digests[64 * i:64 * (i + 1)]  # noqa: E731
    records = judge(sessions, expected, {sig.public_key}, ts.SCHEMES)
    answers = [a for rec in records for a in rec["answers"]]
    sha = [0] * 2000
    probe = digests[:nb].ljust(nb, b"\0")
    for n in range(len(sha)):
        t0 = time.perf_counter_ns()
        hashlib.sha512(probe).digest()
        sha[n] = time.perf_counter_ns() - t0
    first = records[0] if records else None
    accepted = first is not None and first["accepted"]
    profile = (StorageProfile(scheme.name, params.manifest(), blocks, params.m, k, limit, cert, gaps,
                              scheme.experimental).to_json() if accepted and cert is not None else None)
    return {**base, "gaps": list(gaps), "profile": profile,
            "signer": {"scheme": sig.scheme, "public_key": sig.public_key.hex(), "test": True,
                       "note": "a test signer: memory_challenger.SCHEMES accepts no record it signs (fail closed)"},
            "calibration_here": {"rtt_s": quantiles(before), "within_cap": quantiles(before).get("p99.9", 0) * 1e9 <= cap},
            "rtt_after_s": quantiles(after), "pings_raw_s": {"before": before, "after": after},
            "audit_of_record": None if first is None else {"accepted": first["accepted"],
                                                           "accepted_under_test_signer": first["accepted_under_test_signer"],
                                                           "rounds": len(first["answers"])},
            "audits": len(records), "audits_accepted": sum(rec["accepted"] for rec in records),
            "audits_accepted_under_test_signer": sum(bool(rec["accepted_under_test_signer"]) for rec in records),
            "answers": len(answers), "late": sum(not a["on_time"] for a in answers),
            "wrong": sum(not a["correct"] for a in answers),
            "elapsed_s": quantiles([a["elapsed_ns"] / 1e9 for a in answers]),
            "margin_s": quantiles([(limit - a["elapsed_ns"]) / 1e9 for a in answers]),
            "device_s": quantiles([a["device_ns"] / 1e9 for a in answers]),
            "server_s": quantiles([a["server_ns"] / 1e9 for a in answers]),
            "sha512_block_s": quantiles([x / 1e9 for x in sha]),
            "sign_s": quantiles([rec["sign_ns"] / 1e9 for rec in records]),
            "cgroup_cpu": {"before": cpu_before, "after": cgroup_cpu()}, "records": records}


def main(argv: list[str] | None = None) -> int:
    from verity.protocols.accounting.space.pous import audit as pous_audit

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ready", required=True)
    ap.add_argument("--leaves", required=True, help="SHA-512 of every block of C, in block order")
    ap.add_argument("--out", required=True)
    ap.add_argument("--audits", type=int, default=20)
    ap.add_argument("--pings", type=int, default=5000)
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--verifier-cpu", type=int, default=None)
    ap.add_argument("--allowance", default="p99.9", choices=["p50", "p99.9", "max"], help="the calibration's quantile, recorded")
    ap.add_argument("--deadline-ns", type=int, default=0)
    ap.add_argument("--limit-ns", type=int, default=0, help="the limit fixed before any load (0: Δ + the RTT cap)")
    ap.add_argument("--calibrate-only", action="store_true")
    ap.add_argument("--clock", default="wall", choices=["wall", "logical"])
    ap.add_argument("--strict", action="store_true", help="refuse a k or limit no certificate covers, before any round")
    ap.add_argument("--signer", default="test", choices=["test"])
    a = ap.parse_args(argv)
    ready = json.loads(Path(a.ready).read_text())
    params = params_of(ready["scheme"], ready["params"])
    clock = LogicalClock() if a.clock == "logical" else time.perf_counter_ns
    deadline = a.deadline_ns or pous_audit.DEADLINE_NS
    if a.calibrate_only:
        doc = calibrate_only("127.0.0.1", ready["port"], params.block_bytes, pings=a.pings, cpu=a.verifier_cpu,
                             allowance=a.allowance, deadline_ns=deadline, clock=clock)
        cap = rtt_cap_ns()
        doc = {**doc, "measured_allowance_ns": doc["rtt_allowance_ns"], "within_cap": doc["rtt_allowance_ns"] <= cap,
               "allowance": "cap", "rtt_allowance_ns": cap, "limit_ns": deadline + cap, "clock": a.clock}
        Path(a.out).write_text(json.dumps(doc, indent=1) + "\n")
        return 0
    scheme = scheme_of(ready["scheme"], params)
    doc = {"clock": a.clock,
           **audit("127.0.0.1", ready["port"], scheme, bytes.fromhex(ready["salt"]), Path(a.leaves).read_bytes(),
                   audits=a.audits, pings=a.pings, k=a.k, cpu=a.verifier_cpu, allowance=a.allowance, deadline_ns=deadline,
                   limit_ns=a.limit_ns or None, clock=clock, strict=a.strict, signer=a.signer)}
    Path(a.out).write_text(json.dumps(doc, indent=1) + "\n")
    return 0 if doc["audits"] and doc["audits_accepted_under_test_signer"] == doc["audits"] else 1


if __name__ == "__main__":
    sys.exit(main())
