"""The secure-erasure difftest's cases (`test_pous_erasure_difftest.py`): each one a prover's script, the verifier
`verity_pous.erasure.ErasureVerifier` run on it, and its line for Lean's accept rule (`lean/PousErasureCheck.lean`).

A case fixes the audit (`M`, `s`, the page size, `T_fill`, `T_read`, the verifier's secret) and what the prover does
at each step: answer right, answer wrong, or not answer, after a delay on the verifier's clock.  `run` drives the
verifier through `run_erasure` with the prover computing `F_K` from the fill challenge, over a transport that either
gives up at each deadline or delivers a late answer anyway (`eager`).  `line` writes the same case as Lean reads it:

    Tfill Tread s M expected fAnswered fDigest fTime r F_0 … F_(M−1) (i a v t)×r

with each digest as the integer of `0x01 ‖ bytes` (so that bytes of different lengths stay different), the `r = s`
reads at the indices the verifier draws (`spot_indices`), and a missing answer as `a = 0`.

`uv run python tests/pous_erasure_cases.py` (from `protocols/pous`, with Lean's toolchain, elan, on the PATH) runs
Lean on every line and rewrites `fixtures/lean_erasure_verdicts.json` with Lean's verdicts and the digests of the cases
and of the Lean files it read.
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
LEAN_FILES = (LEAN_CHECK,)
"""The Lean files whose definitions the verdicts come from, pinned by digest in the fixture.  BINDING: once
`cursor/pous-lean-secure-erasure-3cf5` lands, its module joins this list."""
LEAN_PACKAGE = HERE.parent / "lean"
LEAN_COMMAND = ("lean", "--run", str(LEAN_CHECK))
"""How to run the check, from `lean/` so that elan picks the package's toolchain.  BINDING: once the check imports the
Lean lane's module, it runs inside the package, `lake env lean --run`, after `lake build` of that module."""

SEED = "pous-erasure-difftest/v1"
COUNT = 12_000
PAGES = (16, 32, 64, 128)

RIGHT, WRONG, DROP = "right", "wrong", "drop"
DIGEST_FLIP, DIGEST_SHORT, DIGEST_LONG, DIGEST_ZERO = 1, 2, 3, 4
"""A wrong digest: its first bit flipped, its last byte dropped, a zero byte appended, or all zeros."""


@dataclass(frozen=True)
class Step:
    """The prover's answer to one step: `RIGHT`, `WRONG` (changed by `wrong`) or `DROP`, `delay` after the send."""

    kind: str
    delay: int = 0
    wrong: int = 0


@dataclass(frozen=True)
class Case:
    source: bytes
    size: int
    spot_reads: int
    page_bytes: int
    fill_deadline: int
    read_deadline: int
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


def _case(rng: random.Random, n: int) -> Case:
    page = rng.choice(PAGES)
    size = rng.randrange(1, 81)
    s = rng.randrange(1, min(size, 8) + 1)
    recompute = page_keccak_f(page) * KECCAK_F_NS
    t_read = rng.choice([0, 1, recompute - 1, rng.randrange(recompute)])
    t_fill = rng.choice([0, 1, t_read, rng.randrange(10**3), rng.randrange(10**7)])
    mode = rng.choices(["honest", "one-fault", "random"], [3, 4, 3])[0]
    bad = rng.randrange(s + 1) if mode == "one-fault" else None

    def faulty(k):
        return k == bad if mode == "one-fault" else mode == "random" and rng.random() < 0.15

    fill = _step(rng, t_fill, faulty(s), True)
    reads = tuple(_step(rng, t_read, faulty(j), False) for j in range(s))
    return Case(b"case-%d" % n, size, s, page, t_fill, t_read, rng.random() < 0.5, fill, reads)


def cases(count: int = COUNT, seed: str = SEED) -> list[Case]:
    rng = random.Random(seed)
    return [_case(rng, n) for n in range(count)]


def truth(case: Case) -> tuple[KeyedFill, tuple[int, ...]]:
    """The verifier's `F_K` and all `s` of its indices, from its secret."""
    key = fill_key(case.source, case.size, case.spot_reads)
    return KeyedFill(key, case.size, case.page_bytes), spot_indices(spot_key(case.source, key), case.spot_reads,
                                                                   case.size)


def _nat(b: bytes) -> int:
    return int.from_bytes(b"\x01" + b, "big")


def line(case: Case) -> str:
    fill, indices = truth(case)
    memory = b"".join(fill.chunks())
    expected = fill.digest()
    f = case.fill
    out = [case.fill_deadline, case.read_deadline, case.spot_reads, case.size, _nat(expected), int(f.kind != DROP),
           _nat(answer(f, expected, wrong_digest)) if f.kind != DROP else 0, f.delay, len(case.reads), *memory]
    for i, step in zip(indices, case.reads, strict=True):
        out += [i, int(step.kind != DROP), answer(step, memory[i], wrong_byte) if step.kind != DROP else 0, step.delay]
    return " ".join(map(str, out))


class Clock:
    def __init__(self) -> None:
        self.now = 0

    def __call__(self) -> int:
        return self.now


def run(case: Case, module: ModuleType = erasure) -> tuple[bool, tuple[int, ...]]:
    """`module.ErasureVerifier` through `module.run_erasure` on `case`: whether it accepts, and the indices it
    revealed.  `module` is `verity_pous.erasure` or a mutant of it; the prover's `F_K` is always the real one."""
    clock, held, queue = Clock(), [], []
    v = module.ErasureVerifier(case.size, case.spot_reads, fill_deadline_ns=case.fill_deadline,
                               read_deadline_ns=case.read_deadline, source=case.source, page_bytes=case.page_bytes,
                               clock=clock)

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

    result = module.run_erasure(v, send, receive)
    return result.accepted, tuple(r.index for r in result.reads)


def cases_digest(lines: list[str]) -> str:
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def lean_files_digests() -> dict[str, str]:
    return {str(p.relative_to(HERE.parent)): hashlib.sha256(p.read_bytes()).hexdigest() for p in LEAN_FILES}


def lean_verdicts(lines: list[str]) -> str:
    """Lean's verdicts on `lines`: one `0` or `1` per line."""
    out = subprocess.run(LEAN_COMMAND, input="\n".join(lines) + "\n", capture_output=True, text=True,
                         cwd=LEAN_PACKAGE, check=True).stdout.strip()
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
