"""Checks of the reference step and the difference tables.  `python3 selfcheck.py`; a few seconds, standard library only.

  - the round, iterated 24 times with the standard constants, is Keccak-f[1600] (SHA3-256 against hashlib);
  - theta's and L's inverses invert, at every width;
  - Ketje v2's twisted step, absorbing into plane y = 0 of the twisted state, is the plain step absorbing into the
    diagonal lanes (the `diag` placement), up to the fixed relabelling pi;
  - the round constant is an offset on the inputs: it changes no collision;
  - chi's row table: the weights by input difference.
"""

from __future__ import annotations

import hashlib
import random

import diff
import keccak as K


def sha3_256(msg: bytes) -> bytes:
    rate = 136
    p = bytearray(msg) + b"\x06" + b"\x00" * ((-len(msg) - 2) % rate) + b"\x80"
    if len(p) % rate:
        p = bytearray(msg) + b"\x86"
    s = [0] * 25
    for off in range(0, len(p), rate):
        blk = p[off:off + rate]
        for i in range(rate // 8):
            s[i] ^= int.from_bytes(blk[8 * i:8 * i + 8], "little")
        s = K.keccak_f(s, 64)
    return b"".join(v.to_bytes(8, "little") for v in s[:4])


def main() -> None:
    rng = random.Random(1)

    for msg in (b"", b"abc", bytes(range(200))):
        assert sha3_256(msg) == hashlib.sha3_256(msg).digest(), "the round is not Keccak-f[1600]'s"
    assert K.round_constants(64)[23] == 0x8000000080008008 == K.last_round_constant(64)
    print("round: 24 of them are Keccak-f[1600] (SHA3-256 matches hashlib on 3 messages)")

    for w in K.WIDTHS:
        for _ in range(20):
            s = [rng.getrandbits(w) for _ in range(25)]
            assert K.theta_inv(K.theta(s, w), w) == s and K.linear_inv(K.linear(s, w), w) == s
            assert K.pi_inv(K.pi(s)) == s
    print("theta, L: inverses invert at w = 1..64")

    for w in K.WIDTHS:
        for _ in range(10):
            s = [rng.getrandbits(w) for _ in range(25)]
            xs = [rng.getrandbits(4 * w) for _ in range(5)]
            t = s
            for x in xs:
                t = K.twisted_step(t, x, w)
            assert K.pi_inv(t) == K.chain(K.pi_inv(s), xs, w, "diag")
    assert K.pi_inv(K.place((1 << 256) - 1, 64, "plane")) == K.place((1 << 256) - 1, 64, "diag")
    print("placement: Ketje v2's twisted step with a plane-0 absorb = the plain step with the diagonal absorb")

    for w in K.WIDTHS:
        for pl in K.PLACEMENTS:
            for _ in range(10):
                s = [rng.getrandbits(w) for _ in range(25)]
                xs = [rng.getrandbits(4 * w) for _ in range(6)]
                rc = rng.getrandbits(w)
                with_rc = K.chain(s, xs, w, pl, rc)
                shifted = [xs[0]] + [x ^ rc for x in xs[1:]]   # lane (0, 0) is the first absorbed lane
                without = K.chain(s, shifted, w, pl, 0)
                without[0] ^= rc
                assert with_rc == without
    print("round constant: a chain with constant rc is the chain with constant 0 on inputs offset by rc")

    by_weight: dict[int, list[str]] = {}
    for d in range(1, 32):
        by_weight.setdefault(diff.ROW_WEIGHT[d], []).append(f"{d:05b}"[::-1])
    assert diff.ROW_WEIGHT[0] == 0 and diff.ROW_OUT[0] == frozenset({0})
    assert sorted(by_weight) == [2, 3, 4]
    assert all(len(diff.ROW_OUT[d]) == 1 << diff.ROW_WEIGHT[d] for d in range(32))
    assert all(0 not in diff.ROW_OUT[d] for d in range(1, 32))
    for wt in sorted(by_weight):
        print(f"chi rows: weight {wt}: {len(by_weight[wt])} input differences (bits x = 0..4): {' '.join(by_weight[wt])}")
    print("ok")


if __name__ == "__main__":
    main()
