"""The red team's resume attack on the band fill, adapted to the streamed-label wipe and run against
it at fixture scale on CPU.

**What the attack uses in the band fill.** A band label is the deployed `Π₂` (Feistel10 over SHAKE256), which the
prover computes itself from the public salt. Its first `m` bits are `pad ⊕` five SHAKE256 squeeze streams, so the
prover keeps the five 200-byte post-absorb states (1,000 bytes of a 64 KiB block, 1.53%) and squeezes the block back in
481 sequential Keccak-f, with no `Π₂` call, inside the 1.449 ms read deadline that `(D, Q) = (5, 2^20)` was converted
to.

**The streamed wipe.** `L_i = SHA-512(LABEL_TAG ‖ K ‖ u64 i)` under the memory challenger's key `K`, which the prover
never holds. The only state short of a label from which labels resume is the challenger's: SHA-512's state after
the 67 bytes of `LABEL_TAG ‖ K` (rounds 0–7 in `wipe_stream.c`), which is `K`'s worth. Each prover below runs whole
sessions of `streamed_wipe.Challenger` (sequential reads, batch size 1), signed by a test signer and judged by
`wipe_verdict.judge`:

- `honest`: keeps every label. Expect accept.
- `naive`: the `--foreign-gb fM` cheat: the room of `dropped(N, k)` labels kept as prior content, one label past the
  `foreign_bits` `verity/wipe/v0` publishes at `k` reads (`wipe_period.certify`), those labels dropped. Expect reject.
- `resume`: the attack transported. Each slot keeps the attack's share of its label (1,000/65,536 of it, rounded up
  to one byte) and 63 bytes of prior content; slots `0 … SEEDS−1` keep whole labels, its "states". At a read it
  resumes the label's computation, which needs `K`: it searches keys against its whole labels for as long as the
  read's deadline allows, then answers with the label under a key that reproduces them, or else what it kept.
  Expect reject.
- `resume-untimed`: the same under a 10-minute deadline with `UNTIMED_TRIES` keys searched at the first read.
  Expect reject: the deadline isn't what stops it.
- `key-leak` (a premise broken): the prover is handed `K`, as a memory challenger on the prover's own machine can
  leak it. It keeps SHA-512's state after `LABEL_TAG ‖ K` and prior content in the whole store, and answers a read
  with one compression. Expect accept: the attack's mechanism, once the resume state is the prover's.
- `untiered` (a premise broken): every label kept in memory the commitment doesn't declare, prior content in the
  whole declared store. Expect accept: the verdict sees answers, not where they were kept.
- `sampling`: the best keyless strategy, `naive`'s `g = N − dropped(N, k)` labels kept at the sessions' `k`, over many
  sessions at a small `k`: its acceptance rate against `(g/N)^k`, the bound's sampling term.

For each it records the verdict (each side's), how the session ended, the prior content (its size, its share of the
declared store, and whether `keep` returns it after the reads) and the read times. `main` writes `resume.json`.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import tempfile
import time
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "verity/core/protocols/pous/tests")]

import pous_test_signer as ts  # noqa: E402
from verity.core.protocols.pous import streamed_wipe as sw  # noqa: E402
from verity.core.protocols.pous import wipe_period as W  # noqa: E402

WIPE_VERDICT = ROOT / "verity/core/devices/memory_challenger/lean/wipe_verdict.py"
LABEL = sw.LABEL_BYTES
BAND_SHARE = Fraction(5 * 200, 65_536)
"""What the band attack keeps of a block: five 200-byte SHAKE256 states of a 64 KiB block."""
KEPT = math.ceil(LABEL * BAND_SHARE)
"""Bytes of each label `resume` keeps: the band attack's share, rounded up to a byte (1)."""
SEEDS = 5
"""Whole labels `resume` keeps, as the band attack keeps five states."""
UNTIMED_LIMIT_NS = 600 * 10**9
UNTIMED_TRIES = 1 << 22
SERVER = b"wipe-resume-fixture"


def params(n: int, k: int, limit_ns: int = 1) -> sw.WipeParams:
    """A session's parameters: `n` labels in host memory, declared exactly, `k` sequential reads."""
    return sw.WipeParams(n, k, 1, limit_ns, (("host-dram", n * LABEL),))


def dropped(n: int, k: int) -> int:
    """The labels `naive` gives up for prior content: one label past the `foreign_bits` published at `k` reads
    (`wipe_period`'s `fM`)."""
    return W.foreign_labels("fM", params(n, k))


class Honest:
    """Keeps every label; its prior content was the whole store, and the fill overwrites it."""

    def __init__(self, n: int, c: sw.Challenger) -> None:
        self.store = bytearray(os.urandom(n * LABEL))
        self.z = bytes(self.store)

    def receive(self, start: int, data: bytes) -> None:
        self.store[LABEL * start:LABEL * start + len(data)] = data

    def answer(self, idx) -> bytes:
        return b"".join(bytes(self.store[LABEL * i:LABEL * (i + 1)]) for i in idx)

    def keep(self) -> bytes:
        return bytes(self.store)


class Naive(Honest):
    """Keeps prior content in the room of the last `dropped(N, at_k)` labels (`at_k` the session's reads unless
    given) and answers those reads from it."""

    def __init__(self, n: int, c: sw.Challenger, at_k: int | None = None) -> None:
        self.cut = n - dropped(n, c.params.k if at_k is None else at_k)
        self.store = bytearray(n * LABEL)
        self.store[LABEL * self.cut:] = os.urandom(LABEL * (n - self.cut))
        self.z = bytes(self.store[LABEL * self.cut:])

    def receive(self, start: int, data: bytes) -> None:
        end = min(start + len(data) // LABEL, self.cut)
        if end > start:
            self.store[LABEL * start:LABEL * end] = data[:LABEL * (end - start)]

    def keep(self) -> bytes:
        return bytes(self.store[LABEL * self.cut:])


def candidate(c: int) -> bytes:
    return hashlib.sha512(b"verity-pous/wipe-resume/candidate-key\0" + c.to_bytes(8, "big")).digest()[:32]


class Resume(Honest):
    """The band attack's strategy on the streamed fill (module docstring): `KEPT` bytes of each label and 63 of prior
    content a slot, `SEEDS` whole labels, and a key search, the only way to the state labels resume from, at a read."""

    def __init__(self, n: int, c: sw.Challenger, *, clock: Callable[[], int], search_ns: int, tries: int) -> None:
        self.n, self.clock, self.search_ns, self.tries = n, clock, search_ns, tries
        self.store = bytearray(os.urandom(n * LABEL))
        self.z = self.keep()
        self.key: bytes | None = None
        self.searched = 0
        self.search_spent_ns = 0

    def receive(self, start: int, data: bytes) -> None:
        for j in range(len(data) // LABEL):
            i, lab = start + j, data[LABEL * j:LABEL * (j + 1)]
            if i < SEEDS:
                self.store[LABEL * i:LABEL * (i + 1)] = lab
            else:
                self.store[LABEL * i:LABEL * i + KEPT] = lab[:KEPT]

    def seed(self, i: int) -> bytes:
        return bytes(self.store[LABEL * i:LABEL * (i + 1)])

    def search(self) -> None:
        """Keys tried against the kept whole labels until `search_ns` of this read is spent or `tries` are made."""
        start, target = self.clock(), self.seed(0)
        while self.key is None and self.searched < self.tries and self.clock() - start < self.search_ns:
            k = candidate(self.searched)
            self.searched += 1
            if sw.label(k, 0) == target and all(sw.label(k, s) == self.seed(s) for s in range(1, SEEDS)):
                self.key = k
        self.search_spent_ns += self.clock() - start

    def one(self, i: int) -> bytes:
        if i < SEEDS:
            return self.seed(i)
        if self.key is None:
            self.search()
        if self.key is not None:
            return sw.label(self.key, i)
        return bytes(self.store[LABEL * i:LABEL * i + KEPT]) + bytes(LABEL - KEPT)

    def answer(self, idx) -> bytes:
        return b"".join(self.one(i) for i in idx)

    def keep(self) -> bytes:
        return b"".join(bytes(self.store[LABEL * i + KEPT:LABEL * (i + 1)]) for i in range(SEEDS, self.n))


class KeyLeak(Honest):
    """Handed `K`: keeps SHA-512's state after `LABEL_TAG ‖ K`, the challenger's own resume point (32 bytes of `K`),
    and prior content in the whole store; a read is one compression from that state."""

    def __init__(self, n: int, c: sw.Challenger) -> None:
        super().__init__(n, c)
        self.head = hashlib.sha512(sw.LABEL_TAG + c.key)

    def receive(self, start: int, data: bytes) -> None:
        pass

    def answer(self, idx) -> bytes:
        out = b""
        for i in idx:
            h = self.head.copy()
            h.update(i.to_bytes(8, "big"))
            out += h.digest()
        return out


class Untiered(Honest):
    """Every label in `outside`, memory the commitment doesn't declare; prior content in the whole declared store."""

    def __init__(self, n: int, c: sw.Challenger) -> None:
        super().__init__(n, c)
        self.outside = bytearray(n * LABEL)

    def receive(self, start: int, data: bytes) -> None:
        self.outside[LABEL * start:LABEL * start + len(data)] = data

    def answer(self, idx) -> bytes:
        return b"".join(bytes(self.outside[LABEL * i:LABEL * (i + 1)]) for i in idx)


def _wipe_verdict():
    spec = importlib.util.spec_from_file_location("streamed_wipe_lean_verdict", WIPE_VERDICT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _stats(xs: list[int]) -> dict[str, int] | None:
    if not xs:
        return None
    s = sorted(xs)
    return {"n": len(s), "p50": s[len(s) // 2], "p99": s[min(len(s) - 1, (99 * len(s)) // 100)], "max": s[-1]}


def session(name: str, make: Callable[[int, sw.Challenger], Any], *, n: int, k: int, limit_ns: int, signer, log,
            judge: Callable[..., Any], decide: str, clock: Callable[[], int], tag: bytes = b"",
            coin_secret: bytes | None = None) -> dict[str, Any]:
    """One whole session against the prover `make(n, challenger)` builds; its record (module docstring). The coins
    are the OS's, drawn when the fill completes, unless `coin_secret` fixes them."""
    sid = hashlib.sha512(b"verity-pous/wipe-resume/session\0" + name.encode() + b"\0" + tag).digest()[:32]
    c = sw.Challenger(os.urandom(32), SERVER, sid, params(n, k, limit_ns), signer, log=log, clock=clock,
                      coin_secret=coin_secret)
    prover = make(n, c)
    z_before = hashlib.sha512(prover.z).hexdigest()
    for start, data in c.fill(1 << 14):
        prover.receive(start, data)
    c.run(prover.answer)
    signed = c.close(egress_bits=0)
    v, sides = judge(signed, c.commitment, trusted={signer.public_key}, schemes=ts.SCHEMES, decide=decide)
    kept = prover.keep()
    out = {"prover": name, "labels": n, "k": k, "limit_ns": limit_ns, "ended": signed.ended,
           "reads": len(signed.batches), "accepted": v.accepted, "reasons": list(v.reasons)[:3], "verdicts": sides,
           "prior_bytes": len(prover.z), "prior_share": len(prover.z) / (n * LABEL),
           "kept": kept == prover.z, "prior_sha512_before": z_before,
           "prior_sha512_after": hashlib.sha512(kept).hexdigest(),
           "read_ns": _stats([r.elapsed_ns for r in signed.batches])}
    if isinstance(prover, Resume):
        out |= {"kept_label_bytes": KEPT, "seeds": SEEDS, "keys_searched": prover.searched,
                "search_ns": prover.search_spent_ns, "key_found": prover.key is not None}
    return out


EXPECT = {"honest": True, "naive": False, "resume": False, "resume-untimed": False, "key-leak": True, "untiered": True}


def provers(clock: Callable[[], int], limit_ns: int, untimed_tries: int = UNTIMED_TRIES):
    """`(name, make, limit_ns)` for each prover of the module docstring but `sampling`."""
    return [
        ("honest", Honest, limit_ns),
        ("naive", Naive, limit_ns),
        ("resume", lambda n, c: Resume(n, c, clock=clock, search_ns=limit_ns * 3 // 4, tries=1 << 62), limit_ns),
        ("resume-untimed", lambda n, c: Resume(n, c, clock=clock, search_ns=UNTIMED_LIMIT_NS // 2, tries=untimed_tries),
         UNTIMED_LIMIT_NS),
        ("key-leak", KeyLeak, limit_ns),
        ("untiered", Untiered, limit_ns),
    ]


def sampling(*, n: int, k: int, at_k: int, sessions: int, signer, log, judge, decide: str, clock,
             limit_ns: int) -> dict[str, Any]:
    """`naive` as it is at `at_k` reads over `sessions` sessions at `k`: accepted count against `(g/N)^k`."""
    accepted = 0
    for r in range(sessions):
        rec = session("sampling", lambda n, c: Naive(n, c, at_k), n=n, k=k, limit_ns=limit_ns, signer=signer, log=log,
                      judge=judge, decide=decide, clock=clock, tag=r.to_bytes(4, "big"))
        accepted += rec["accepted"]
    g = n - dropped(n, at_k)
    p = (g / n) ** k
    sd = math.sqrt(sessions * p * (1 - p))
    return {"labels": n, "k": k, "at_k": at_k, "g": g, "sessions": sessions, "accepted": accepted,
            "expected": sessions * p, "p": p, "z_score": (accepted - sessions * p) / sd if sd else None}


def run(*, n: int, k: int, limit_ns: int, decide: str, scratch: Path, clock: Callable[[], int] = time.perf_counter_ns,
        untimed_tries: int = UNTIMED_TRIES, sample_n: int = 4096, sample_k: int = 12, sample_sessions: int = 600,
        judge: Callable[..., Any] | None = None, coin_secret: bytes | None = None) -> dict[str, Any]:
    """Every prover's session at `N = n`, `k` reads, then `sample_sessions` of `sampling` (none at 0)."""
    judge = judge or _wipe_verdict().judge
    signer = ts.TestSigner(b"wipe-resume provers", height=3, directory=scratch)
    log = ts.session_log(scratch)
    out: dict[str, Any] = {"schema": "verity-pous/wipe-resume/v1", "label_bytes": LABEL, "kept_label_bytes": KEPT,
                           "band_share": float(BAND_SHARE), "seeds": SEEDS, "decide": decide, "sessions": []}
    for name, make, lim in provers(clock, limit_ns, untimed_tries):
        rec = session(name, make, n=n, k=k, limit_ns=lim, signer=signer, log=log, judge=judge, decide=decide,
                      clock=clock, coin_secret=coin_secret)
        rec["expect_accept"] = EXPECT[name]
        rec["as_expected"] = rec["accepted"] == EXPECT[name]
        out["sessions"].append(rec)
    g, cert = n - dropped(n, k), W.certify(params(n, k), 0)
    out["bound"] = {"labels": n, "memory_bits": cert["memory_bits"], "k": k, "r": cert["r"],
                    "foreign_bits": cert["foreign_bits"], "naive_log2_accept_at_k": k * math.log2(g / n) if g else None}
    if sample_sessions:
        s_signer = ts.TestSigner(b"wipe-resume sampling", height=10, directory=scratch)
        out["sampling"] = sampling(n=sample_n, k=sample_k, at_k=k, sessions=sample_sessions, signer=s_signer, log=log,
                                   judge=judge, decide=decide, clock=clock, limit_ns=limit_ns)
    out["as_expected"] = all(s["as_expected"] for s in out["sessions"]) and (
        "sampling" not in out or abs(out["sampling"]["z_score"]) < 4)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--labels", type=int, default=1 << 19, help="N (default 2^19: one 32 MiB segment)")
    ap.add_argument("--k", type=int, default=W.READS, help=f"reads a session (default {W.READS})")
    ap.add_argument("--limit-ns", type=int, default=20_000_000)
    ap.add_argument("--decide", choices=("lean", "both", "python"), default="both")
    ap.add_argument("--sample-sessions", type=int, default=600)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="wipe-resume-") as scratch:
        res = run(n=args.labels, k=args.k, limit_ns=args.limit_ns, decide=args.decide, scratch=Path(scratch),
                  sample_sessions=args.sample_sessions)
    (args.out / "resume.json").write_text(json.dumps(res, indent=1) + "\n")
    for s in res["sessions"]:
        print(f"{s['prover']:15} accepted={s['accepted']!s:5} ended={s['ended']:8} reads={s['reads']:5} "
              f"prior={s['prior_share']:.2%} kept={s['kept']} expected={s['as_expected']}")
    if "sampling" in res:
        print("sampling", json.dumps(res["sampling"]))
    sys.exit(0 if res["as_expected"] else 1)


if __name__ == "__main__":
    main()
