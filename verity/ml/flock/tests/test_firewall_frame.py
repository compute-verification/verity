"""The outer firewall's release (`Flock/Firewall/Frame.lean`, `flock-firewall frame`, live PROTOCOL.md §10.1) on a hand-made rep:
each value absorbed in a masked phase leaves as `v + h_i` under the next pad of `zk_veil::Layout::absorbed`'s order (written
out here from the layout's index formulas), the pads' label and root follow the zerocheck's label, the domain frame stays
home, everything outside the masked phases leaves as the device framed it, and a round that breaks the masking challenger's
phases or the transcript's framing is refused, after which nothing leaves. `flock-circuit` holds its own masking challenger to
this release byte for byte on every round it sends (`firewall::Gate`)."""
from __future__ import annotations

import json
import shutil
import struct
import subprocess
from pathlib import Path

import pytest

LEAN = Path(__file__).resolve().parents[3] / "core" / "service" / "flock"
BIN = LEAN / ".lake" / "build" / "bin" / "flock-firewall"
SKIP = 64
LAY = {"n_mlv": 2, "n_lc": 1, "n_ood0": 1, "k0": 1}
ROOT = bytes(range(64))


def hdr(op: int, kind: int, n: int) -> bytes:
    return bytes([op, kind, 0, 0, 0, 0, 0, 0]) + struct.pack("<Q", n)


def pad16(b: bytes) -> bytes:
    return b + bytes(-len(b) % 16)


def label(s: str) -> bytes:
    return hdr(2, 0, len(s)) + pad16(s.encode())


def msg(b: bytes) -> bytes:
    return hdr(5, 0, len(b)) + pad16(b)


def obs(vals: list[int], kind: int = 2) -> bytes:
    return hdr(3, kind, len(vals)) + b"".join(struct.pack("<QQ", v & (2**64 - 1), v >> 64) for v in vals)


def squeeze(kind: int, n: int) -> bytes:
    return hdr(4, kind, n)


def order(n_mlv: int, n_lc: int, n_ood0: int, k0: int) -> list[int]:
    """`Layout::absorbed` from the layout's index formulas (`zk_veil.rs`)."""
    fa = 2 * SKIP + 2 * n_mlv
    lc = [fa + 3 + 2 * j + t for j in range(n_lc) for t in range(2)]
    zp0 = fa + 3 + 2 * n_lc
    lg0 = zp0 + SKIP
    lig = [lg0 + i for i in range(n_ood0)] + [lg0 + n_ood0 + 4 * j + t for j in range(k0) for t in range(4)] \
        + [lg0 + n_ood0 + 4 * k0 + c for c in range(4)] if k0 else []
    return [*range(2 * SKIP), *(2 * SKIP + 2 * j + t for j in range(n_mlv) for t in range(2)), fa, fa + 1, *lc,
            *range(zp0, zp0 + SKIP), *lig]


PADS = [(0x9E3779B97F4A7C15 * (i + 1)) % 2**64 | ((i * 0x5BD1E995 + 7) << 64) for i in range(max(order(**LAY)) + 5)]


class Frame:
    def __init__(self, lay=LAY, pads=PADS, root=ROOT):
        self.p = subprocess.Popen([str(BIN), "frame"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        start = {"layout": lay, "pads": [struct.pack("<QQ", v % 2**64, v >> 64).hex() for v in pads], "root": root.hex()}
        self.start = self.ask(json.dumps(start))

    def ask(self, line: str, body: bytes = b""):
        try:
            self.p.stdin.write(line.encode() + b"\n" + body)
            self.p.stdin.flush()
        except BrokenPipeError:
            return None
        r = self.p.stdout.readline().decode().rstrip("\n")
        return (r, self.p.stdout.read(int(r[3:]))) if r.startswith("ok ") else r

    def round(self, content: bytes, n: int):
        return self.ask(f"round {n} {len(content)}", content)

    def close(self) -> int:
        try:
            self.p.stdin.close()
        except BrokenPipeError:
            pass
        self.p.stdout.close()
        return self.p.wait()


@pytest.fixture(scope="module", autouse=True)
def built():
    if shutil.which("lake") is None:
        pytest.skip("no Lean toolchain")
    subprocess.run(["lake", "build", "flock-firewall"], cwd=LEAN, check=True, capture_output=True)


def masked(vals: list[int], at: int) -> list[int]:
    o = order(**LAY)
    return [v ^ PADS[o[at + i]] for i, v in enumerate(vals)]


def test_a_rep_leaves_masked_in_the_layouts_order():
    f = Frame()
    assert f.start == "ok"
    r0 = hdr(1, 0, 3) + pad16(b"dom") + label("flock-r1cs-v0") + msg(b"\1" * 64) + label("flock-zerocheck-v0") + squeeze(2, 6)
    want = label("flock-r1cs-v0") + msg(b"\1" * 64) + label("flock-zerocheck-v0") + label("verity-flock-zk-pads-v0") + msg(ROOT) \
        + squeeze(2, 6)
    assert f.round(r0, 6) == (f"ok {len(want)}", want)
    vals = list(range(1, 2 * SKIP + 1))
    r1 = obs(vals[:SKIP]) + obs(vals[SKIP:]) + squeeze(1, 1)
    assert f.round(r1, 1)[1] == obs(masked(vals[:SKIP], 0)) + obs(masked(vals[SKIP:], SKIP)) + squeeze(1, 1)
    at = 2 * SKIP
    # the multilinear rounds, final_a and final_b as scalars, the lincheck round, z_partial; a zero squeeze inside
    ml, fab, lc, zp = [11, 12, 13, 14], [21, 22], [31, 32], list(range(100, 100 + SKIP))
    r2 = obs(ml[:2]) + squeeze(2, 0) + obs(ml[2:]) + obs([fab[0]], 1) + obs([fab[1]], 1) + obs(lc) + obs(zp) + squeeze(2, 2)
    want = obs(masked(ml[:2], at)) + squeeze(2, 0) + obs(masked(ml[2:], at + 2)) + obs(masked([fab[0]], at + 4), 1) \
        + obs(masked([fab[1]], at + 5), 1) + obs(masked(lc, at + 6)) + obs(masked(zp, at + 8)) + squeeze(2, 2)
    assert f.round(r2, 2)[1] == want
    at += 8 + SKIP
    # the opening: unmasked but for level 0's OOD value, lane messages and the pair's sums
    lg = [41, 42, 43, 44, 45, 46, 47, 48, 49]
    r3 = label("flock-pcs-open-batch-v0") + obs([5]) + label("verity-flock-zk-l0-open-v0") + obs(lg[:1]) + obs(lg[1:5]) \
        + obs(lg[5:]) + label("verity-flock-zk-l0-close-v0") + obs([6]) + squeeze(2, 3)
    want = label("flock-pcs-open-batch-v0") + obs([5]) + label("verity-flock-zk-l0-open-v0") + obs(masked(lg[:1], at)) \
        + obs(masked(lg[1:5], at + 1)) + obs(masked(lg[5:], at + 5)) + label("verity-flock-zk-l0-close-v0") + obs([6]) + squeeze(2, 3)
    assert f.round(r3, 3)[1] == want
    # a second zerocheck label after the opening is not a masked phase
    r4 = label("flock-zerocheck-v0") + obs([7]) + squeeze(1, 1)
    assert f.round(r4, 1)[1] == r4
    assert f.ask("finish") == "ok"
    assert f.close() == 0


@pytest.mark.parametrize("rounds, why", [
    ([label("flock-zerocheck-v0") + label("flock-pcs-open-batch-v0") + squeeze(1, 1)],
     "err the opening began after 0 of the algebraic phase's 200 masked values"),
    ([label("flock-zerocheck-v0") + obs(list(range(300))) + squeeze(1, 1)], "err more masked values than the layout's 209"),
    ([label("flock-r1cs-v0") + squeeze(1, 1) + label("x") + squeeze(1, 1)], "err a squeeze before the round's end"),
    ([label("flock-r1cs-v0") + squeeze(2, 2)], "err the round squeezes 2 coins, the caller asked for 1"),
    ([label("flock-r1cs-v0")], "err a round without its squeeze"),
    ([label("flock-r1cs-v0") + hdr(3, 1, 2) + bytes(32) + squeeze(1, 1)], "err an observe of kind 1 and length 2"),
    ([hdr(9, 0, 0) + squeeze(1, 1)], "err unknown frame op 9"),
    ([hdr(3, 2, 4) + bytes(16)], "err a frame overruns its round"),
])
def test_a_round_off_the_masking_phases_or_the_framing_is_refused_and_ends_the_rep(rounds, why):
    f = Frame()
    for r in rounds[:-1]:
        assert f.round(r, 1)[0].startswith("ok ")
    assert f.round(rounds[-1], 1) == why
    assert f.round(squeeze(1, 1), 1) in (None, ""), "nothing leaves after a refusal"
    assert f.close() == 1


def test_a_rep_that_ends_with_pads_unused_is_refused():
    f = Frame()
    assert f.round(label("flock-zerocheck-v0") + obs([1, 2]) + squeeze(1, 1), 1)[0].startswith("ok ")
    assert f.ask("finish") == "err the rep ended after 2 of the layout's 209 masked values"
    assert f.close() == 1


def test_a_start_with_too_few_pads_or_a_short_root_is_refused():
    for f, why in [(Frame(pads=PADS[:200]), "pad 200 is outside the rep's 200 pads"), (Frame(root=bytes(32)), "the pads' root has 32 bytes, not 64")]:
        assert f.start == f"err the rep's start: {why}"
        assert f.close() == 1
