"""The suite's one injected clock and one entropy source (AGENTS.md "Tests"), in place of `time.monotonic_ns` and the
operating system's generator."""

from __future__ import annotations

import hashlib


class Clock:
    """The verifier's clock in nanoseconds from `now`, each reading `step` after the last (at `step = 0` time moves only
    when a test sets `now`); waiting until `t` (`period.run`'s `sleep_until`) moves it there."""

    def __init__(self, now: int = 0, step: int = 0) -> None:
        self.now, self.step = now, step

    def __call__(self) -> int:
        self.now += self.step
        return self.now

    def sleep_until(self, _clock, t: int) -> None:
        self.now = max(self.now, t)


class Entropy:
    """SHA-512 in counter mode under `label`, in place of the operating system's generator."""

    def __init__(self, label: bytes) -> None:
        self.label, self.count = label, 0

    def __call__(self, n: int) -> bytes:
        out = b""
        while len(out) < n:
            out += hashlib.sha512(self.label + self.count.to_bytes(8, "big")).digest()
            self.count += 1
        return out[:n]
