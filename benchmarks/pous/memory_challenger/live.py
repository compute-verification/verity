"""The memory challenger's process in the harness's timed audit (`p2-v3-mc-gpu`): `verity.core.protocols.pous.
memory_challenger` against the responder, over the harness's responder wire (`band_gpu/memory_challenger.py`).

    python benchmarks/pous/memory_challenger/live.py --ready READY.json --leaves DIGESTS.bin --out AUDIT.json [--audits N]
        [--pings N] [--k K] [--verifier-cpu C] [--limit-ns N] [--calibrate-only] [--clock logical] [--strict]
        [--signer lms --lms-key DIR --session-log FILE [--keygen-workers N] | --signer test] [--sessions FILE]

`READY.json` names the responder's port, the scheme, its manifest and the setup salt.  `DIGESTS.bin` is the `sha512/row/v2`
digest of every block of `C` (`memory_challenger.block_digest`), in block order: what `memory_challenger.verdict` checks
each round's `H(block)` against, the table a proof will replace (`answer_circuit.PousAnswerOpens_v1` opens `H(block)`
against C's leaf).

- **The limit** is `Δ + RTT_ALLOWANCE_CAP_NS` (P2's 354 µs cap, `Δ + RTT = 854 µs`), fixed before any load.  The
  calibration (`--calibrate-only`) still measures the loopback round trip and records whether its p99.9 fits under the cap.
- **A session** draws a fresh session id (the auditor's designation: here the harness is the auditor) and challenger
  secret and commits its coin tree, every session's before the first round of any; it runs `k` rounds (the certificate's
  `k`) after `WARMUP_PINGS` pings, signs once and opens its coins.  `--strict` refuses a `k` or limit no
  certificate covers (`covering_certificate`) before any round.  Every session's binding is `setup_root`, the root the setup
  publishes for the `C*` of these digests under READY's salt (`VerifierKey.from_row_digests`), recorded as `setup_root`.
- **The session log** (`--session-log FILE`, with `--signer lms`) is the memory challenger's `memory_challenger.SessionLog`:
  every session's id goes into it, fsynced, when the session commits, and a used id is refused.  It is a file of its own,
  refused inside `--lms-key`; it is made with a new key, and for a key that exists a missing log is refused (one made
  before the log existed takes `memory_challenger.SessionLog.create(FILE)` once).  Under `--signer test` it is in a
  temporary directory the audit removes once its sessions are signed.
- **The signer.** `--signer lms` signs with S1's LMS at `lms.DEPLOYED` (`memory_challenger.LMS`), the key in `--lms-key DIR`:
  `seed` (secret, 0600), `public` (`I || T[1]`), `tree.bin` (the key's tree, a cache checked against `public` and
  `tree.sha512`) and `state.json` (`lms.LeafState`, which refuses a used leaf across runs).  A missing key is made there
  first (`--keygen-workers`; about 1.1G SHA-512 hashes).  `--signer test` signs with the PoUS tests' test signer (`pous_test_signer.py`: LMS at a small height under a
  `test/` id), its leaf state in that temporary directory.
- **The verdict** is `memory_challenger.verdict` under `memory_challenger.SCHEMES`, which accepts S1's scheme alone, so a
  test-signed session is rejected: it fails closed.  `judge` computes it twice on every session, with the Lean port's
  (`lean_verdict`, built with `lake` on first use) and with the Python, which stays the reference, and keeps both in the
  witness (`verdicts`).  `RECORD_VERDICT` names the one that is the record's (the Lean's); where the two disagree, on the
  accept bit or the reasons, the session is rejected for it (`DISAGREE`, listing both).
  `verity/core/devices/memory_challenger/lean/tests/test_verdict.py` holds the Lean to the Python on generated records.
  `accepted_under_signer` is the same verdict under the signer's own mapping (`memory_challenger.SCHEMES` for `lms`, the
  test signer's for `test`): for `test`, every check but the scheme's acceptance, and never the verdict.
- **The record** separates what is published (`public`: the commitment, the verdict, the coin openings) from the witness
  (`witness`: the signed session, the record's reasons, both verdicts, C's digests at the blocks the rounds revealed,
  per-round on-time and digest bits), and keeps the responder's reports beside: with the signer's public key, every input
  of the verdict, so a record replays.
- **The signed sessions** go to `--sessions` (`sessions-<--out's name>` beside it) as the session loop ends, however it
  ends, before any verdict runs, and the Lean verdict is built and serving before the first session signs: a verdict that
  fails never takes signed sessions, or the key's leaves they spent, with it.

Exit 0 when every session passes under the signer the run used, 1 otherwise.
"""
from __future__ import annotations

import argparse
import functools
import gc
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from band_gpu.memory_challenger import (  # noqa: E402
    ALLOWANCE_KIND,
    HEADER,
    OP_CHALLENGE,
    OP_PING,
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
TEST_SIGNER = ROOT / "verity" / "core" / "protocols" / "pous" / "tests" / "pous_test_signer.py"
LEAN_VERDICT = ROOT / "verity" / "core" / "devices" / "memory_challenger" / "lean" / "verdict.py"
RECORD_VERDICT = "lean"
"""Which verdict is the record's: `"lean"` (`lean_verdict`) or `"python"` (`memory_challenger.verdict`, the reference).
`judge` computes both on every session and records both either way."""
DISAGREE = "the Lean and Python verdicts disagree"
WARMUP_PINGS = 8
"""Pings before each session's first round, after the previous session's signature: a session's round 0 otherwise follows
the challenger's own Python work (one measured round 0 took 655.8 µs, 26 µs of it at the responder)."""


def test_signer() -> Any:
    """The PoUS tests' test signer module (`TestSigner`, `scheme`, `SCHEMES`)."""
    spec = importlib.util.spec_from_file_location("pous_test_signer", TEST_SIGNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@functools.cache
def lean_verdict() -> Any:
    """The Lean port's verdict module (`verdict.py` beside its Lake package): `verdict`, with `memory_challenger.verdict`'s
    signature, computed by `memory-challenger-verdict` (built with `lake` on first use)."""
    spec = importlib.util.spec_from_file_location("memory_challenger_lean_verdict", LEAN_VERDICT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def lean_ready() -> None:
    """The Lean verdict loaded, its executable built (or found built: `pod.sh PREPARE=1`) and serving, before any session
    signs: where it can't run, the audit stops here and no leaf of the key is spent."""
    lean_verdict().server()


def write_sessions(path: str | os.PathLike, sig: Any, sessions: list[dict[str, Any]]) -> None:
    """The signed sessions as they left the session loop, on disk before any verdict runs: each one's commitment, signed
    session, coin openings, the responder's reports and its signing time, with the signer's scheme and public key."""
    doc = {"signer": {"scheme": sig.scheme, "public_key": sig.public_key.hex()},
           "sessions": [{"commitment": s["commitment"].to_json(), "signed": s["signed"].to_json(),
                         "openings": [o.to_json() for o in s["openings"]], "reports": [list(r) for r in s["reports"]],
                         "sign_ns": s["sign_ns"]} for s in sessions]}
    p = Path(path)
    tmp = p.with_name(p.name + ".tmp")
    with open(tmp, "w") as f:
        f.write(json.dumps(doc, indent=1) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)


def rtt_cap_ns() -> int:
    from verity.core.protocols.pous.schemes import p2
    return p2.RTT_ALLOWANCE_CAP_NS


def lms_signer(directory: str | os.PathLike, workers: int = 1, params: Any = None) -> Any:
    """The memory challenger's S1 signer whose key is in `directory`, made there first when absent; at `params` other than
    `lms.DEPLOYED` (tests), under a `test/` id."""
    from verity.ml.crypto import lms
    from verity.core.protocols.pous import memory_challenger as mc

    p = params or lms.DEPLOYED
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    seed_p, public_p, tree_p, state_p = d / "seed", d / "public", d / "tree.bin", d / "state.json"
    check_p = d / "tree.sha512"
    if not seed_p.exists():
        fd = os.open(seed_p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.write(fd, os.urandom(p.N))
            os.fsync(fd)
        finally:
            os.close(fd)
    seed = seed_p.read_bytes()
    if tree_p.exists() and public_p.exists() and check_p.exists():
        public, raw = public_p.read_bytes(), tree_p.read_bytes()
        tree = (b"",) + tuple(raw[p.N * r:p.N * (r + 1)] for r in range(len(raw) // p.N))
        key = lms.PrivateKey(p, public[:16], seed, tree)
        if (len(tree) != 2 ** (p.H + 1) or lms.public_key(key) != public
                or hashlib.sha512(public + raw).digest() != check_p.read_bytes()):
            raise ValueError(f"{tree_p} is not the tree of {public_p}")
    else:
        key = lms.keygen(p, seed, workers)
        raw, public = b"".join(key.tree[1:]), lms.public_key(key)
        for path, data in ((tree_p, raw), (public_p, public), (check_p, hashlib.sha512(public + raw).digest())):
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_bytes(data)
            os.replace(tmp, path)
    if not state_p.exists():
        lms.LeafState.create(state_p, lms.public_key(key), p.H)
    scheme = mc.LMS if p == lms.DEPLOYED else mc.TEST_PREFIX + f"lms-sha512/n{p.N}-w{p.W}-h{p.H}/v0"
    return mc.LmsSigner(key, state_p, scheme=scheme)


def session_log(path: str | os.PathLike, new_key: bool) -> Any:
    """The memory challenger's used-session log at `path` (`memory_challenger.SessionLog`), made there when the key is new."""
    from verity.core.protocols.pous import memory_challenger as mc

    if new_key and not Path(path).exists():
        mc.SessionLog.create(path)
    return mc.SessionLog(path)


def signer_schemes(scheme: str) -> Mapping[str, Any]:
    """The mapping a session signed under `scheme` is judged by beside `memory_challenger.SCHEMES`: that one for S1's
    scheme, the test signer's for a test one."""
    from verity.core.protocols.pous import memory_challenger as mc

    return mc.SCHEMES if scheme in mc.SCHEMES else test_signer().SCHEMES


def setup_root(params: Any, salt: bytes, digests: bytes) -> bytes:
    """What a session audits: the root the setup publishes for the `C*` whose block digests are `digests`, under `salt`
    (`VerifierKey.root`)."""
    from verity.core.protocols.pous.verifier import VerifierKey

    return VerifierKey.from_row_digests(params, salt, [digests[64 * i:64 * (i + 1)] for i in range(params.blocks)]).root


def verdicts(s: dict[str, Any], expected: Callable[[int], bytes], trusted: set[bytes],
             schemes: Mapping[str, Any] | None = None) -> tuple[Any, dict[str, dict[str, Any]]]:
    """Session `s` judged by the Lean verdict and by the Python's: the record's verdict, which is `RECORD_VERDICT`'s unless
    the two disagree (accept bit or reasons), and then a rejection for it naming both; and both as recorded.  A record the
    Lean refuses to read (`ValueError`) is its rejection."""
    from verity.core.protocols.pous import memory_challenger as mc

    args = (s["signed"], s["commitment"], s["openings"], expected)
    kw = {"trusted": trusted, **({} if schemes is None else {"schemes": schemes})}
    got = {"python": mc.verdict(*args, **kw)}
    try:
        got["lean"] = lean_verdict().verdict(*args, **kw)
    except ValueError as e:
        got["lean"] = mc.Verdict(False, (str(e),))
    both = {name: {"accepted": v.accepted, "reasons": list(v.reasons)} for name, v in sorted(got.items())}
    v = got[RECORD_VERDICT]
    if both["lean"] != both["python"]:
        v = mc.Verdict(False, (f"{DISAGREE}: lean {json.dumps(both['lean'])}, python {json.dumps(both['python'])}",),
                       v.late, v.wrong)
    return v, both


def judge(sessions: list[dict[str, Any]], expected: Callable[[int], bytes], trusted: set[bytes],
          signer_schemes: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Each session's verdict under `memory_challenger.SCHEMES` and under the signer's own mapping (`verdicts`)."""
    out = []
    for s in sessions:
        v, both = verdicts(s, expected, trusted)
        t, both_t = verdicts(s, expected, trusted, signer_schemes)
        rounds = s["signed"].rounds
        answers = [{"index": r.index, "elapsed_ns": r.elapsed_ns, "device_ns": d, "server_ns": sv,
                    "correct": r.digest == expected(r.index), "on_time": r.elapsed_ns <= s["commitment"].limit_ns}
                   for r, (d, sv) in zip(rounds, s["reports"], strict=True)]
        out.append({"answers": answers,
                    "public": {"commitment": s["commitment"].to_json(), "verdict": v.public(),
                               "openings": [o.to_json() for o in s["openings"]]},
                    "witness": {"signed": s["signed"].to_json(), "reasons": list(v.reasons),
                                "record_verdict": RECORD_VERDICT, "verdicts": both, "verdicts_under_signer": both_t,
                                "expected": {str(r.index): expected(r.index).hex() for r in rounds
                                             if 0 <= r.index < s["commitment"].blocks},
                                "reasons_under_signer": list(t.reasons)},
                    "accepted": v.accepted, "accepted_under_signer": t.accepted,
                    "sign_ns": s["sign_ns"]})
    return out


def signature_control(doc: dict[str, Any], expected: Callable[[int], bytes]) -> dict[str, Any]:
    """The bad-signature control, on the honest audit's signed sessions: each with one signature bit flipped, and each with
    one round's receipt time moved 1 ns earlier after signing (an on-time bit forged), judged under the signer's own
    mapping (`signer_schemes`); every one must be rejected for its signature, and the unedited sessions must pass."""
    from verity.core.protocols.pous import memory_challenger as mc

    schemes = signer_schemes(doc["signer"]["scheme"])
    out = {"sessions": 0, "unedited_accepted": 0, "flipped_rejected": 0, "edited_rejected": 0}
    for rec in doc.get("records") or []:
        signed = mc.SignedSession.from_json(rec["witness"]["signed"])
        commitment = mc.SessionCommitment.from_json(rec["public"]["commitment"])
        openings = [mc.CoinOpening.from_json(o) for o in rec["public"]["openings"]]
        trusted = {signed.public_key}

        def reasons(s: mc.SignedSession) -> tuple[str, ...]:
            return mc.verdict(s, commitment, openings, expected, trusted=trusted, schemes=schemes).reasons

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
          clock: Callable[[], int] = time.perf_counter_ns, strict: bool = True, signer: str = "test",
          lms_key: str | os.PathLike | None = None, keygen_workers: int = 1,
          sessions_out: str | os.PathLike | None = None, session_log_path: str | os.PathLike | None = None) -> dict[str, Any]:
    """`audits` sessions of the memory challenger against `limit_ns` (else `Δ` plus the cap), each judged after all of them
    ran, off the timed path.  The Lean verdict is made ready before any session signs (`lean_ready`), and the signed
    sessions are written to `sessions_out` (`write_sessions`) when the session loop ends, however it ends, before any
    verdict runs.  Every session's id is in `session_log_path` (`session_log`) before its first round; without one (the
    test signer), in a temporary directory with the test signer's leaf state, removed once the sessions are signed."""
    from verity.core.protocols.pous import audit as pous_audit
    from verity.core.protocols.pous import memory_challenger as mc
    from verity.core.protocols.pous.protocol import StorageProfile

    if signer not in ("lms", "test") or (signer == "lms") != (lms_key is not None):
        raise ValueError(f"the signer is 'lms' with a key directory or 'test' without one, not {signer!r}, {lms_key!r}")
    if signer == "lms" and session_log_path is None:
        raise ValueError("an LMS-signed audit takes the memory challenger's session log (--session-log FILE)")
    if lms_key is not None and Path(session_log_path).resolve().is_relative_to(Path(lms_key).resolve()):
        raise ValueError(f"the session log {session_log_path} is inside the key's directory {lms_key}: it is a file of its own")
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
    binding = setup_root(params, salt, digests)
    base = {"memory_challenger": "memory_challenger.MemoryChallenger", "protocol": mc.PROTOCOL, "coin_tree": mc.COIN_TREE, "records_tree": mc.RECORDS_TREE,
            "strict": strict, "warmup_pings": WARMUP_PINGS, "limit_ns": limit, "limit_fixed_before_load": fixed, "deadline_ns": deadline_ns,
            "allowance": "cap", "allowance_kind": ALLOWANCE_KIND, "rtt_allowance_ns": limit - deadline_ns, "rtt_cap_ns": cap,
            "k": k, "scheme": scheme.name, "setup_root": binding.hex()}
    gaps: tuple[str, ...] = ()
    try:
        cert = mc.covering_certificate(scheme, k, limit, experimental=True)
        if cert.segments is not None and params.segments != cert.segments:
            gaps = (f"W has {params.segments} segments; {cert.theorem} is about {cert.segments}",)
    except ValueError as e:
        if strict:
            return {**base, "refused": str(e), "audits": 0, "audits_accepted": 0, "audits_accepted_under_signer": 0,
                    "answers": 0, "late": 0, "wrong": 0, "records": []}
        cert, gaps = None, (str(e),)
    lean_ready()
    new_key = signer == "lms" and not (Path(lms_key) / "seed").exists()
    with tempfile.TemporaryDirectory(prefix="pous-live-") as scratch:
        t0 = time.perf_counter_ns()
        sig = (lms_signer(lms_key, keygen_workers) if signer == "lms"
               else test_signer().TestSigner(os.urandom(32), height=max(1, (max(1, audits) - 1).bit_length()),
                                             directory=scratch))
        key_load_s = (time.perf_counter_ns() - t0) / 1e9
        log = session_log(session_log_path or Path(scratch) / "sessions", new_key or session_log_path is None)
        resolution = None if clock is time.perf_counter_ns else 1
        if cpu is not None:
            os.sched_setaffinity(0, {cpu})
        gc.collect()
        gc.freeze()
        gc.disable()
        r = LiveResponder(host, port, nb, clock)
        cpu_before = cgroup_cpu()
        sessions = []
        challengers = [mc.MemoryChallenger(mc.CoinTree.draw(os.urandom(32), os.urandom(32), blocks, k), sig, log=log,
                                           limit_ns=limit, block_bytes=nb, binding=binding, clock=clock,
                                           resolution_ns=resolution)
                       for _ in range(audits)]
        try:
            before = calibrate(r, pings)
            for c in challengers:
                for _ in range(WARMUP_PINGS):
                    r.roundtrip(OP_PING, 0)
                reports = []
                while not c.done:
                    i = c.challenge()
                    r.roundtrip(OP_CHALLENGE, i)
                    c.receive(r.view[HEADER:])
                    reports.append((r.device_ns, r.server_ns))
                t0 = time.perf_counter_ns()
                signed = c.close()
                sessions.append({"commitment": c.commitment, "signed": signed, "openings": c.openings(),
                                 "reports": reports, "sign_ns": time.perf_counter_ns() - t0})
            after = calibrate(r, pings)
        finally:
            r.close()
            gc.enable()
            if sessions_out is not None:
                write_sessions(sessions_out, sig, sessions)
    expected = lambda i: digests[64 * i:64 * (i + 1)]  # noqa: E731
    records = judge(sessions, expected, {sig.public_key}, signer_schemes(sig.scheme))
    answers = [a for rec in records for a in rec["answers"]]
    sha = [0] * 2000
    probe = digests[:nb].ljust(nb, b"\0")
    for n in range(len(sha)):
        t0 = time.perf_counter_ns()
        mc.block_digest(probe)
        sha[n] = time.perf_counter_ns() - t0
    first = records[0] if records else None
    accepted = first is not None and first["accepted"]
    profile = (StorageProfile(scheme.name, params.manifest(), blocks, params.m, k, limit, cert, gaps,
                              scheme.experimental).to_json() if accepted and cert is not None else None)
    return {**base, "gaps": list(gaps), "profile": profile,
            "signer": {"kind": signer, "scheme": sig.scheme, "public_key": sig.public_key.hex(),
                       "test": sig.scheme not in mc.SCHEMES, "key_load_s": key_load_s, "session_log": str(log.path),
                       "note": ("S1's LMS at lms.DEPLOYED, its leaf state on disk" if sig.scheme in mc.SCHEMES else
                                "a test signer: memory_challenger.SCHEMES accepts no record it signs (fail closed)")},
            "calibration_here": {"rtt_s": quantiles(before), "within_cap": quantiles(before).get("p99.9", 0) * 1e9 <= cap},
            "rtt_after_s": quantiles(after), "pings_raw_s": {"before": before, "after": after},
            "audit_of_record": None if first is None else {"accepted": first["accepted"],
                                                           "accepted_under_signer": first["accepted_under_signer"],
                                                           "rounds": len(first["answers"])},
            "audits": len(records), "audits_accepted": sum(rec["accepted"] for rec in records),
            "audits_accepted_under_signer": sum(rec["accepted_under_signer"] for rec in records),
            "answers": len(answers), "late": sum(not a["on_time"] for a in answers),
            "wrong": sum(not a["correct"] for a in answers),
            "elapsed_s": quantiles([a["elapsed_ns"] / 1e9 for a in answers]),
            "margin_s": quantiles([(limit - a["elapsed_ns"]) / 1e9 for a in answers]),
            "device_s": quantiles([a["device_ns"] / 1e9 for a in answers]),
            "server_s": quantiles([a["server_ns"] / 1e9 for a in answers]),
            "block_digest_s": quantiles([x / 1e9 for x in sha]),
            "sign_s": quantiles([rec["sign_ns"] / 1e9 for rec in records]),
            "cgroup_cpu": {"before": cpu_before, "after": cgroup_cpu()}, "records": records}


def main(argv: list[str] | None = None) -> int:
    from verity.core.protocols.pous import audit as pous_audit

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ready", required=True)
    ap.add_argument("--leaves", required=True, help="the sha512/row/v2 digest of every block of C, in block order")
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
    ap.add_argument("--signer", default="test", choices=["lms", "test"])
    ap.add_argument("--lms-key", default=None, help="the S1 key's directory (--signer lms), made there when absent")
    ap.add_argument("--session-log", default=None,
                    help="the memory challenger's used-session log (--signer lms): a file of its own, outside --lms-key")
    ap.add_argument("--keygen-workers", type=int, default=len(os.sched_getaffinity(0)))
    ap.add_argument("--sessions", default=None,
                    help="where the signed sessions go before any verdict runs (default: sessions-<--out's name> beside it)")
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
                   limit_ns=a.limit_ns or None, clock=clock, strict=a.strict, signer=a.signer, lms_key=a.lms_key,
                   keygen_workers=a.keygen_workers, session_log_path=a.session_log,
                   sessions_out=a.sessions or Path(a.out).with_name("sessions-" + Path(a.out).name))}
    Path(a.out).write_text(json.dumps(doc, indent=1) + "\n")
    return 0 if doc["audits"] and doc["audits_accepted_under_signer"] == doc["audits"] else 1


if __name__ == "__main__":
    sys.exit(main())
