"""SERVE-H1 interface W-a: a unit's seed is (salt, position, roots), so positions the verifier doesn't assign let one
computed unit be credited twice.

Two calls with one A and one B (a prompt served twice; the prover keys the hiding salts, so it can give both calls one
root) at one position under one salt derive one seed, and the second unit's receipt is a copy of the first's. Distinct
positions give distinct seeds. Runs on main (7d291b14d): ``uv run python benchmarks/rate_redteam/h1_position_replay.py``.
"""
import random

from verity.core.protocols.pouw.circuit import h1tile as H

rng = random.Random(3)
R, C, K = 4, 8, 64
a = [[rng.randrange(256) for _ in range(K)] for _ in range(R)]
b = [[rng.randrange(256) for _ in range(K)] for _ in range(C)]
salt, roots = bytes(range(64)), (b"\x0a" * 64, b"\x0b" * 64)

first = H.reference(a, b, salt, 5, roots)[1]
replayed = H.reference(a, b, salt, 5, roots)[1]
moved = H.reference(a, b, salt, 6, roots)[1]
print("same (salt, position, roots): second receipt is the first's:", first == replayed)
print("next position: a fresh receipt:", first != moved)
