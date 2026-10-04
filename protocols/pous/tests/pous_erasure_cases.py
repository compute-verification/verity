"""The secure-erasure difftest's cases (`test_pous_erasure_difftest.py`): each one a prover's script, the verifier
`verity_pous.erasure.ErasureVerifier` run on it, and its line for the Lean model's accept rule over distinct reads of
the page fill, `Pous.eraseWinsDraw` with `Pous.distinctReads` on `Pous.PageFill`'s fill
(`tests/lean/PousErasureCheck.lean`).

A case fixes the audit (`M`, `s`, the page size, `T_fill`, `T_read`, `T_round`, the verifier's secret) and what the
prover does at each step: answer right, answer wrong, or not answer, after a delay on the verifier's clock, and how long
the caller waits after each answer before it reveals the next read.  `run` drives the verifier with the prover
computing `F_K` from the fill challenge, over a transport that either gives up at each deadline or delivers a late
answer anyway (`eager`): through `run_erasure` when the caller never waits, else by hand with the waits.  `line` writes
the same case as Lean reads it:

    Tfill Tread Tround s M pageBytes g z fAnswered fTime n d_1 … d_n K_0 … K_31 R (T_0 … T_39 X_0 … X_(pageBytes−1))×R
      (i a v t r)×s

with `(g, z)` the digest key, `d` the `n` bytes of the prover's digest, `K` the fill key, the `R` pages of the fill
each as its tweak `T` (`KeyedFill.tweak`) and its whole page `X`, the `s` reads at the indices the verifier draws
(`spot_indices`) with the prover's answer `v` and a missing answer as `a = 0`, and `r` the time from the read before's
reveal to this one's (0 for the first).  The fill and the digest it must have are not in the line: Lean computes them,
with `PageFill` from the pages and `Pous.Digest.digest`.

`uv run python tests/pous_erasure_cases.py` (from `protocols/pous`, with Lean's toolchain, elan, on the PATH) builds
the Lean modules the check imports, runs it on every line, and rewrites `fixtures/lean_erasure_verdicts.json` with
Lean's verdicts and the digests of the cases and of the Lean files that define the rule.
"""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from verity.randomness import Key
from verity_pous import erasure
from verity_pous.erasure import (
    KeyedFill,
    fill_key,
    page_keccak_f,
    spot_indices,
    spot_key,
)
from verity_pous.primitives import KECCAK_F_NS

HERE = Path(__file__).resolve().parent
FIXTURE = HERE / "fixtures" / "lean_erasure_verdicts.json"
LEAN_CHECK = HERE / "lean" / "PousErasureCheck.lean"
LEAN_PACKAGE = HERE.parent / "lean"
LEAN_MODULES = ("Pous.Protocol.Model.SecureErasure", "Pous.Protocol.Model.ReadDraw", "Pous.Protocol.Model.PageFill",
                "Pous.Protocol.Model.Digest")
"""The modules whose definitions the check evaluates: `eraseWinsDraw`, `distinctReads` and the game's types, the page
fill's `tweak`, `page`, `byte` and `pageScheme`, and `Digest.digest`."""
LEAN_FILES = (LEAN_CHECK, *(LEAN_PACKAGE / (m.replace(".", "/") + ".lean") for m in LEAN_MODULES))
"""The Lean files that define the verdicts, pinned by digest in the fixture: a change to the rule fails the difftest
until the verdicts are regenerated."""

SEED = "pous-erasure-difftest/v2"
COUNT = 12_000
PAGES = (16, 32, 64, 128)

RIGHT, WRONG, DROP = "right", "wrong", "drop"
DIGEST_FLIP, DIGEST_SHORT, DIGEST_LONG, DIGEST_ZERO = 1, 2, 3, 4
"""A wrong digest: its first bit flipped, its last byte dropped, a zero byte appended, or all zeros."""


@dataclass(frozen=True)
class Step:
    """The prover's answer to one step: `RIGHT`, `WRONG` (changed by `wrong`) or `DROP`, `delay` after the send; and
    for a read after the first, `gap`, how long the caller waits after the read before closes to reveal this one."""

    kind: str
    delay: int = 0
    wrong: int = 0
    gap: int = 0


@dataclass(frozen=True)
class Case:
    source: bytes
    size: int
    spot_reads: int
    page_bytes: int
    fill_deadline: int
    read_deadline: int
    round_deadline: int
    eager: bool
    fill: Step
    reads: tuple[Step, ...]


def wrong_digest(wrong: int, d: bytes) -> bytes:
    return {DIGEST_FLIP: bytes([d[0] ^ 1]) + d[1:], DIGEST_SHORT: d[:-1], DIGEST_LONG: d + b"\x00",
            DIGEST_ZERO: bytes(len(d))}[wrong]


def wrong_byte(wrong: int, v: int) -> int:
    """`v` with its bits flipped by `wrong` below 256; from 256 on, `v + 256·(wrong − 255)`, equal to `v` mod 256."""
    return v ^ wrong if wrong < 256 else v + 256 * (wrong - 255)


def answer(step: Step, right, change):
    return right if step.kind == RIGHT else change(step.wrong, right)


def reveal_gap(case: Case, j: int) -> int:
    """The time from read `j − 1`'s reveal to read `j`'s, when read `j − 1` was answered in time: its delay and then the
    caller's wait.  0 for the first read."""
    return 0 if j == 0 else case.reads[j - 1].delay + case.reads[j].gap


def _step(rng: random.Random, deadline: int, faulty: bool, digest: bool) -> Step:
    if not faulty:
        return Step(RIGHT, rng.choice([0, deadline, rng.randrange(deadline + 1)]))
    late = rng.choice([deadline + 1, deadline + 1 + rng.randrange(10**6)])
    kind = rng.choice([RIGHT, WRONG, WRONG, DROP])
    if kind == RIGHT:
        return Step(RIGHT, late)
    if kind == DROP:
        return Step(DROP)
    wrong = rng.choice([DIGEST_FLIP, DIGEST_SHORT, DIGEST_LONG, DIGEST_ZERO]) if digest else \
        rng.choice([rng.randrange(1, 256), 256 + rng.randrange(3)])
    return Step(WRONG, rng.choice([0, deadline, rng.randrange(deadline + 1), late]), wrong)


def _read(rng: random.Random, t_read: int, t_round: int, faulty: bool, prev: Step | None) -> Step:
    """A read's step and the caller's wait before its reveal: none, up to, exactly at, or (a fault) past `T_round`."""
    if prev is None:
        return _step(rng, t_read, faulty, False)
    slack = max(0, t_round - prev.delay)
    if faulty and rng.random() < 0.25:
        on_time = rng.choice([0, t_read, rng.randrange(t_read + 1)])
        return Step(RIGHT, on_time, gap=slack + 1 + rng.choice([0, rng.randrange(10**6)]))
    step = _step(rng, t_read, faulty, False)
    gap = rng.choice([0, 0, 0, slack, rng.randrange(slack + 1)])
    return Step(step.kind, step.delay, step.wrong, gap)


def _case(rng: random.Random, n: int) -> Case:
    page = rng.choice(PAGES)
    size = rng.randrange(1, 81) if rng.random() < 0.5 else page * rng.randrange(1, 4)
    s = rng.randrange(1, min(size, 8) + 1)
    recompute = page_keccak_f(page) * KECCAK_F_NS
    half = (recompute - 1) // 2
    t_read = rng.choice([0, 1, half, rng.randrange(half + 1)])
    t_round = rng.choice([t_read, recompute - 1 - t_read, rng.randrange(t_read, recompute - t_read)])
    t_fill = rng.choice([0, 1, t_read, rng.randrange(10**3), rng.randrange(10**7)])
    mode = rng.choices(["honest", "one-fault", "random"], [3, 4, 3])[0]
    bad = rng.randrange(s + 1) if mode == "one-fault" else None

    def faulty(k):
        return k == bad if mode == "one-fault" else mode == "random" and rng.random() < 0.15

    fill = _step(rng, t_fill, faulty(s), True)
    reads: list[Step] = []
    for j in range(s):
        reads.append(_read(rng, t_read, t_round, faulty(j), reads[-1] if j else None))
    return Case(b"case-%d" % n, size, s, page, t_fill, t_read, t_round, rng.random() < 0.5, fill, tuple(reads))


def cases(count: int = COUNT, seed: str = SEED) -> list[Case]:
    rng = random.Random(seed)
    return [_case(rng, n) for n in range(count)]


def truth(case: Case) -> tuple[KeyedFill, tuple[int, ...]]:
    """The verifier's `F_K` and all `s` of its indices, from its secret."""
    key = fill_key(case.source, case.size, case.spot_reads)
    return KeyedFill(key, case.size, case.page_bytes), spot_indices(spot_key(case.source, key), case.spot_reads,
                                                                   case.size)


def line(case: Case) -> str:
    fill, indices = truth(case)
    whole = KeyedFill(fill.key, fill.pages * fill.page_bytes, fill.page_bytes)
    f = case.fill
    digest = answer(f, fill.digest(), wrong_digest) if f.kind != DROP else b""
    out = [case.fill_deadline, case.read_deadline, case.round_deadline, case.spot_reads, case.size, case.page_bytes,
           *fill.digest_key, int(f.kind != DROP), f.delay, len(digest), *digest, *fill.key.value, fill.pages]
    for p in range(fill.pages):
        out += [*fill.tweak(p), *whole.page(p)]
    for j, (i, step) in enumerate(zip(indices, case.reads, strict=True)):
        v = answer(step, fill.at(i), wrong_byte) if step.kind != DROP else 0
        out += [i, int(step.kind != DROP), v, step.delay, reveal_gap(case, j)]
    return " ".join(map(str, out))


class Clock:
    def __init__(self) -> None:
        self.now = 0

    def __call__(self) -> int:
        return self.now


def run(case: Case, module: ModuleType = erasure) -> tuple[bool, tuple[int, ...]]:
    """`module.ErasureVerifier` on `case`: whether it accepts, and the indices it revealed.  `module` is
    `verity_pous.erasure` or a mutant of it; the prover's `F_K` is always the real one."""
    clock, held, queue = Clock(), [], []
    v = module.ErasureVerifier(case.size, case.spot_reads, fill_deadline_ns=case.fill_deadline,
                               read_deadline_ns=case.read_deadline, round_deadline_ns=case.round_deadline,
                               source=case.source, page_bytes=case.page_bytes, clock=clock, research=True)

    def send(msg):
        if isinstance(msg, module.FillChallenge):
            held.append(KeyedFill(Key(msg.key), msg.size, msg.page_bytes))
            step = case.fill
            reply = lambda: module.DigestReply(answer(step, held[0].digest(), wrong_digest))
        else:
            step = case.reads[msg.j]
            reply = lambda: module.ReadReply(msg.j, answer(step, held[0].at(msg.index), wrong_byte))
        if step.kind != DROP:
            queue.append((clock.now + step.delay, reply()))

    def receive(until):
        if queue and (case.eager or queue[0][0] <= until):
            t, reply = queue.pop(0)
            clock.now = max(clock.now, t)
            return reply
        clock.now = max(clock.now, until + 1)
        return None

    if any(r.gap for r in case.reads):
        result = _run_with_waits(case, v, module, clock, send, receive)
    else:
        result = module.run_erasure(v, send, receive)
    return result.accepted, tuple(r.index for r in result.reads if r.index is not None)


def _run_with_waits(case, v, module, clock, send, receive):
    """`run_erasure`'s loop, with the caller waiting `case.reads[j].gap` before it reveals read `j`."""
    send(v.fill_challenge())
    while v.closes_at is not None:
        reply = receive(v.closes_at)
        if isinstance(reply, module.DigestReply):
            v.receive_digest(reply.digest)
        elif reply is None or v.overdue:
            v.expire()
    while not v.done:
        clock.now += case.reads[len(v.reads)].gap
        try:
            j, i = v.challenge()
        except module.Lapsed:
            break
        send(module.ReadChallenge(j, i))
        while v.pending == j:
            reply = receive(v.closes_at)
            if isinstance(reply, module.ReadReply) and reply.j == j:
                v.respond(j, reply.value)
            elif reply is None or v.overdue:
                v.expire()
    return v.result()


def cases_digest(lines: list[str]) -> str:
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def lean_files_digests() -> dict[str, str]:
    return {str(p.relative_to(HERE.parent)): hashlib.sha256(p.read_bytes()).hexdigest() for p in LEAN_FILES}


def lean_verdicts(lines: list[str]) -> str:
    """Lean's verdicts on `lines`, one `0` or `1` per line: the check run in the Lean package, on its modules built."""
    subprocess.run(("lake", "build", *LEAN_MODULES), cwd=LEAN_PACKAGE, check=True, capture_output=True)
    out = subprocess.run(("lake", "env", "lean", "--run", str(LEAN_CHECK)), input="\n".join(lines) + "\n",
                         capture_output=True, text=True, cwd=LEAN_PACKAGE, check=True).stdout.strip()
    if len(out) != len(lines) or set(out) - {"0", "1"}:
        raise RuntimeError(f"Lean printed {out[:80]!r}, not {len(lines)} verdicts")
    return out


def main() -> None:
    lines = [line(c) for c in cases()]
    verdicts = lean_verdicts(lines)
    fixture = {"source": "tests/lean/PousErasureCheck.lean via tests/pous_erasure_cases.py", "seed": SEED,
               "count": COUNT, "cases_sha256": cases_digest(lines), "lean_files": lean_files_digests(),
               "verdicts": verdicts}
    FIXTURE.write_text(json.dumps(fixture, indent=1) + "\n")
    print(f"{FIXTURE}: {verdicts.count('1')} of {len(verdicts)} accepted", file=sys.stderr)


if __name__ == "__main__":
    main()
