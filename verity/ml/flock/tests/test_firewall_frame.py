"""The outer firewall's release (`Flock/Firewall/Frame.lean`, `flock-firewall frame`, live PROTOCOL.md §10.1) on a hand-made rep:
each value absorbed in a masked phase leaves as `v + h_i` under the next pad of `zk_veil::Layout::absorbed`'s order (written
out here from the layout's index formulas), the pads' label and root follow the zerocheck's label, the domain frame stays
home, everything outside the masked phases leaves as the device framed it, and a round that breaks the masking challenger's
phases or the transcript's framing is refused, releasing nothing. A rep stops only at its end: after a round marked `fail` or
refused, every value of the algebraic phase leaves as its pad alone, whatever the round carries, and the rep's `finish` is
`stop`. So what a stopped rep releases under pads `h + v` is byte for byte what the honest rep with values `v` releases under
`h`, and the rounds, their lengths and their order are the same: before `Finish`, nothing depends on whether a check failed.
`flock-circuit` holds its own masking challenger to this release byte for byte on every round it sends (`firewall::Gate`)."""
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

    def round(self, content: bytes, n: int, fail: bool = False):
        return self.ask(f"round {n} {len(content)}" + (" fail" if fail else ""), content)

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


#: the hand-made rep's values: the algebraic phase's 200 in absorbed order (round 1's, then round 2's), level 0's 9
ALG = list(range(1, 2 * SKIP + 1)) + [11, 12, 13, 14, 21, 22, 31, 32] + list(range(100, 100 + SKIP))
LG = [41, 42, 43, 44, 45, 46, 47, 48, 49]


def rep_rounds(alg: list[int]) -> list[tuple[bytes, int]]:
    """The hand-made rep's rounds (each with its coin count) with algebraic values `alg`, as `test_a_rep_leaves_masked_...`
    frames them."""
    a = alg
    r0 = hdr(1, 0, 3) + pad16(b"dom") + label("flock-r1cs-v0") + msg(b"\1" * 64) + label("flock-zerocheck-v0") + squeeze(2, 6)
    r1 = obs(a[:SKIP]) + obs(a[SKIP:2 * SKIP]) + squeeze(1, 1)
    at = 2 * SKIP
    r2 = obs(a[at:at + 2]) + squeeze(2, 0) + obs(a[at + 2:at + 4]) + obs([a[at + 4]], 1) + obs([a[at + 5]], 1) \
        + obs(a[at + 6:at + 8]) + obs(a[at + 8:at + 8 + SKIP]) + squeeze(2, 2)
    r3 = label("flock-pcs-open-batch-v0") + obs([5]) + label("verity-flock-zk-l0-open-v0") + obs(LG[:1]) + obs(LG[1:5]) \
        + obs(LG[5:]) + label("verity-flock-zk-l0-close-v0") + obs([6]) + squeeze(2, 3)
    r4 = label("flock-zerocheck-v0") + obs([7]) + squeeze(1, 1)
    return [(r0, 6), (r1, 1), (r2, 2), (r3, 3), (r4, 1)]


def released(f: Frame, rounds, fail_at: int | None = None) -> list:
    """Each round's answer, the rep stopped at round `fail_at` (marked `fail`), then the rep's `finish`."""
    out = [f.round(r, n, fail=(i == fail_at)) for i, (r, n) in enumerate(rounds)]
    return out + [f.ask("finish")]


def test_a_failed_round_stops_the_rep_whose_algebraic_values_then_leave_as_their_pads_and_which_ends_in_stop():
    f = Frame()
    garbage = [(7 * i + 3) * 0x1_0000_0001 for i in range(len(ALG))]
    out = released(f, rep_rounds(garbage), fail_at=1)
    o = order(**LAY)
    pads = [PADS[o[i]] for i in range(len(ALG))]
    assert out[1][1] == obs(pads[:SKIP]) + obs(pads[SKIP:2 * SKIP]) + squeeze(1, 1)
    at = 2 * SKIP
    assert out[2][1] == obs(pads[at:at + 2]) + squeeze(2, 0) + obs(pads[at + 2:at + 4]) + obs([pads[at + 4]], 1) \
        + obs([pads[at + 5]], 1) + obs(pads[at + 6:at + 8]) + obs(pads[at + 8:at + 8 + SKIP]) + squeeze(2, 2), \
        "a stopped rep stays stopped"
    assert out[3][1] == released(Frame(), rep_rounds(ALG))[3][1], "level 0's values are the firewall's own, masked as an honest rep's"
    assert out[-1] == "stop" and f.close() == 0


def test_a_stopped_rep_releases_under_pads_h_plus_v_what_the_honest_rep_with_values_v_releases_under_h():
    """`h ↦ h + v` is a bijection of the uniform pads, so before `Finish` a stopped rep's release is distributed as an honest
    rep's, whatever the device sent and wherever its check failed; only `finish` differs."""
    honest = released(Frame(), rep_rounds(ALG))
    assert honest[-1] == "ok"
    o = order(**LAY)
    garbage = [(5 * i + 1) << 70 for i in range(len(ALG))]
    #: each round's first absorbed position: the device's honest values before the failed round, anything from it on
    first = [0, 0, 2 * SKIP, len(ALG), len(ALG)]
    for fail_at, s in enumerate(first):
        shifted = list(PADS)
        for i in range(s, len(ALG)):
            shifted[o[i]] ^= ALG[i]
        stopped = released(Frame(pads=shifted), rep_rounds(ALG[:s] + garbage[s:]), fail_at=fail_at)
        assert stopped[:-1] == honest[:-1] and stopped[-1] == "stop", fail_at


def test_a_refused_round_releases_nothing_and_stops_the_rep_which_goes_on_with_the_scheduled_round_in_its_place():
    f = Frame()
    rounds = rep_rounds(ALG)
    assert f.round(*rounds[0])[0].startswith("ok ")
    assert f.round(label("flock-r1cs-v0") + hdr(3, 1, 2) + bytes(32) + squeeze(1, 1), 1) == "err an observe of kind 1 and length 2"
    out = [f.round(r, n, fail=(i == 1)) for i, (r, n) in enumerate(rounds) if i > 0]
    o = order(**LAY)
    assert out[0][1] == obs([PADS[o[i]] for i in range(SKIP)]) + obs([PADS[o[i]] for i in range(SKIP, 2 * SKIP)]) + squeeze(1, 1)
    assert f.ask("finish") == "stop" and f.close() == 0
    f = Frame()
    assert f.round(*rounds[0])[0].startswith("ok ")
    assert f.round(label("x") + squeeze(1, 1) + label("y") + squeeze(1, 1), 1) == "err a squeeze before the round's end"
    out = [f.round(r, n) for r, n in rounds[1:]]
    assert out[0][1] == obs([PADS[o[i]] for i in range(SKIP)]) + obs([PADS[o[i]] for i in range(SKIP, 2 * SKIP)]) + squeeze(1, 1), \
        "a refused round stops the rep without a `fail` after it"
    assert f.ask("finish") == "stop" and f.close() == 0


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
def test_a_round_off_the_masking_phases_or_the_framing_is_refused_and_releases_nothing(rounds, why):
    f = Frame()
    for r in rounds[:-1]:
        assert f.round(r, 1)[0].startswith("ok ")
    assert f.round(rounds[-1], 1) == why
    assert f.ask("finish").startswith("err the rep ended after"), "the rep stays open, stopped, for the scheduled round"
    assert f.close() == 1


def test_a_request_the_frame_cannot_read_ends_the_rep():
    f = Frame()
    assert f.round(squeeze(1, 1), 1)[0] == "ok 16"
    assert f.ask("round 1 16 later", squeeze(1, 1)) == "err a request `round 1 16 later`"
    assert f.round(squeeze(1, 1), 1) in (None, ""), "nothing leaves after it"
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
