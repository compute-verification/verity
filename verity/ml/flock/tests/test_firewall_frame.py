"""The outer firewall's frame (`Flock/Firewall/Frame.lean`, `flock-firewall frame`, live PROTOCOL.md §10.1) on hand-made reps.
At its start the frame draws the rep's randomness from the OS and commits it: the pads `h` (with `h_ab = h_a h_b` and the
triple's `w = x y`), `μ`, the rows' padding, `τ`'s salt and the pads tree's salts, then the root; a start that brings any of
them is refused. Each value absorbed in a masked phase leaves as `v + h_i` under the next pad of `zk_veil::Layout::absorbed`'s
order (written out here from the layout's index formulas), the pads' label and root follow the zerocheck's label, the domain
frame stays home, everything outside the masked phases leaves as the device framed it, and a round that breaks the masking
challenger's phases or the transcript's framing is refused, releasing nothing. A rep stops only at its end: after a round
marked `fail` or refused, every value of the algebraic phase leaves as its pad alone, whatever the round carries, and the
rep's `finish` is `stop`. So what a stopped rep releases under pads `h` is byte for byte what the honest rep releases under
pads `h + v`, and the rounds, their lengths and their order are the same: before `Finish`, nothing depends on whether a check
failed. The frame decides the rep's refusals from the coins the stream answered each round it released (`coins`): a
zerocheck outer coin `r_j = 1`; at the opening's label, a mask slot short of full rank on the claim points of the table's
reps (`prior`'s and the rep's own); level 0's `ρ_0, ρ_1` dependent over F128; and `β = 0` in the inner proof, whose four
rounds leave through the frame too. Each answers `stop` at once, nothing of the round leaving. `flock-circuit` holds its
own masking challenger to this release byte for byte on every round it sends (`firewall::Gate`), its pads tree over the
frame's draws to the frame's root, and its own refusals to the frame's (`live/src/coin_stop_vectors.json` holds Rust's
`ρ` and `β` checks to the frame's). The prover's every other OS read is `flock-firewall tap`'s: a line `N` answered with
`N` fresh bytes."""
from __future__ import annotations

import json
import random
import re
import shutil
import struct
import subprocess
from pathlib import Path

import pytest

LEAN = Path(__file__).resolve().parents[3] / "core" / "service" / "flock"
BIN = LEAN / ".lake" / "build" / "bin" / "flock-firewall"
VECTORS = Path(__file__).resolve().parents[1] / "live" / "src" / "coin_stop_vectors.json"
SKIP = 64
K_PAD = 192
LAY = {"n_mlv": 2, "n_lc": 1, "n_ood0": 1, "k0": 1}
MOD = (1 << 128) | 0x87


def gmul(a: int, b: int) -> int:
    r = 0
    while b:
        if b & 1:
            r ^= a
        b >>= 1
        a <<= 1
        if a >> 128:
            a ^= MOD
    return r


def hdr(op: int, kind: int, n: int) -> bytes:
    return bytes([op, kind, 0, 0, 0, 0, 0, 0]) + struct.pack("<Q", n)


def pad16(b: bytes) -> bytes:
    return b + bytes(-len(b) % 16)


def label(s: str) -> bytes:
    return hdr(2, 0, len(s)) + pad16(s.encode())


def msg(b: bytes) -> bytes:
    return hdr(5, 0, len(b)) + pad16(b)


def wire(vals: list[int]) -> bytes:
    return b"".join(struct.pack("<QQ", v & (2**64 - 1), v >> 64) for v in vals)


def obs(vals: list[int], kind: int = 2) -> bytes:
    return hdr(3, kind, len(vals)) + wire(vals)


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


def slots(n_mlv: int, n_lc: int, n_ood0: int, k0: int) -> dict[str, int]:
    """The triple's slots and the pads' count, from the same formulas."""
    fa = 2 * SKIP + 2 * n_mlv
    hab = fa + 3 + 2 * n_lc + SKIP + n_ood0 + 4 * k0 + 4
    return {"fa": fa, "fb": fa + 1, "hab": hab, "tx": hab + 1, "ty": hab + 2, "tw": hab + 3, "len": hab + 4}


class Frame:
    def __init__(self, lay=LAY, mask=None, start=None):
        self.p = subprocess.Popen([str(BIN), "frame"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if start is None:
            start = json.dumps({"layout": lay} | ({"mask": mask} if mask else {}))
        self.rng = random.Random(1)
        self.start = self.ask(start)
        if isinstance(self.start, tuple):
            b = self.start[1]
            self.h = [int.from_bytes(b[16 * i:16 * i + 16], "little") for i in range(slots(**lay)["len"])]
            self.root = b[-64:]

    def ask(self, line: str, body: bytes = b""):
        try:
            self.p.stdin.write(line.encode() + b"\n" + body)
            self.p.stdin.flush()
        except BrokenPipeError:
            return None
        r = self.p.stdout.readline().decode().rstrip("\n")
        return (r, self.p.stdout.read(int(r[3:]))) if r.startswith("ok ") else r

    def coins(self, coins: list[int]):
        return self.ask(f"coins {len(coins)}", wire(coins))

    def round(self, content: bytes, n: int, fail: bool = False, coins: list[int] | None = None):
        """The round's answer; a released round's coins (seeded, never 0 or 1, unless `coins`) are answered back, as the
        firewall does."""
        r = self.ask(f"round {n} {len(content)}" + (" fail" if fail else ""), content)
        if isinstance(r, tuple):
            assert self.coins([self.rng.getrandbits(128) | 2 for _ in range(n)] if coins is None else coins) == "ok"
        return r

    def close(self) -> int:
        try:
            self.p.stdin.close()
        except BrokenPipeError:
            pass
        self.p.stdout.close()
        self.err = self.p.stderr.read().decode()
        self.p.stderr.close()
        return self.p.wait()


@pytest.fixture(scope="module", autouse=True)
def built():
    if shutil.which("lake") is None:
        pytest.skip("no Lean toolchain")
    subprocess.run(["lake", "build", "flock-firewall"], cwd=LEAN, check=True, capture_output=True)


#: the hand-made rep's values: the algebraic phase's 200 in absorbed order (round 1's, then round 2's), level 0's 9
ALG = list(range(1, 2 * SKIP + 1)) + [11, 12, 13, 14, 21, 22, 31, 32] + list(range(100, 100 + SKIP))
LG = [41, 42, 43, 44, 45, 46, 47, 48, 49]


#: the inner proof's four rounds (`zk_veil::prove_inner`), which leave as they are: `final_c'` then `ε`, `(ρ, σ)` then the
#: batching coins, `τ`'s commitment then `β`, `Y` then the queries
INNER = [(label("verity-flock-zk-inner-v0") + obs([8], 1) + squeeze(1, 1), 1), (obs([9, 10]) + squeeze(2, 5), 5),
         (msg(b"\2" * 64) + squeeze(1, 1), 1), (obs([11, 12, 13]) + squeeze(2, 4), 4)]


def rep_rounds(alg: list[int]) -> list[tuple[bytes, int]]:
    """The hand-made rep's rounds (each with its coin count) with algebraic values `alg`: the binding and the zerocheck's
    label, the skip's sent vectors, the multilinear rounds (a zero squeeze inside), `final_a` and `final_b` as scalars, the
    lincheck round and `z_partial`, the opening with level 0's masked values then `ρ_0`, `ρ_1` alone, level 0's close, a
    zerocheck label after the opening, and the inner proof."""
    a = alg
    r0 = hdr(1, 0, 3) + pad16(b"dom") + label("flock-r1cs-v0") + msg(b"\1" * 64) + label("flock-zerocheck-v0") + squeeze(2, 6)
    r1 = obs(a[:SKIP]) + obs(a[SKIP:2 * SKIP]) + squeeze(1, 1)
    at = 2 * SKIP
    r2 = obs(a[at:at + 2]) + squeeze(2, 0) + obs(a[at + 2:at + 4]) + obs([a[at + 4]], 1) + obs([a[at + 5]], 1) \
        + obs(a[at + 6:at + 8]) + obs(a[at + 8:at + 8 + SKIP]) + squeeze(2, 2)
    r3 = label("flock-pcs-open-batch-v0") + obs([5]) + label("verity-flock-zk-l0-open-v0") + obs(LG[:1]) + obs(LG[1:5]) \
        + obs(LG[5:]) + squeeze(2, 2)
    r4 = label("verity-flock-zk-l0-close-v0") + obs([6]) + squeeze(2, 3)
    r5 = label("flock-zerocheck-v0") + obs([7]) + squeeze(1, 1)
    return [(r0, 6), (r1, 1), (r2, 2), (r3, 2), (squeeze(2, 2), 2), (r4, 3), (r5, 1), *INNER]


def release(h: list[int], root: bytes, alg: list[int]) -> list[bytes]:
    """The hand-made rep's rounds as they leave under pads `h` and root `root`, with algebraic values `alg`: the domain frame
    gone, the pads' label and root after the zerocheck's label, each masked phase's value `v` as `v + h_i` in the absorbed
    order, the rest as the device framed it."""
    o = order(**LAY)
    a, at, n = alg, 2 * SKIP, len(ALG)

    def m(vals: list[int], i: int) -> list[int]:
        return [v ^ h[o[i + k]] for k, v in enumerate(vals)]

    return [label("flock-r1cs-v0") + msg(b"\1" * 64) + label("flock-zerocheck-v0") + label("verity-flock-zk-pads-v0") + msg(root)
            + squeeze(2, 6),
            obs(m(a[:SKIP], 0)) + obs(m(a[SKIP:at], SKIP)) + squeeze(1, 1),
            obs(m(a[at:at + 2], at)) + squeeze(2, 0) + obs(m(a[at + 2:at + 4], at + 2)) + obs(m([a[at + 4]], at + 4), 1)
            + obs(m([a[at + 5]], at + 5), 1) + obs(m(a[at + 6:at + 8], at + 6)) + obs(m(a[at + 8:at + 8 + SKIP], at + 8))
            + squeeze(2, 2),
            label("flock-pcs-open-batch-v0") + obs([5]) + label("verity-flock-zk-l0-open-v0") + obs(m(LG[:1], n))
            + obs(m(LG[1:5], n + 1)) + obs(m(LG[5:], n + 5)) + squeeze(2, 2),
            squeeze(2, 2),
            label("verity-flock-zk-l0-close-v0") + obs([6]) + squeeze(2, 3),
            label("flock-zerocheck-v0") + obs([7]) + squeeze(1, 1),
            *(r for r, _ in INNER)]


def released(f: Frame, rounds, fail_at: int | None = None) -> list:
    """Each round's answer, the rep stopped at round `fail_at` (marked `fail`), then the rep's `finish`."""
    out = [f.round(r, n, fail=(i == fail_at)) for i, (r, n) in enumerate(rounds)]
    return out + [f.ask("finish")]


def test_the_frame_draws_the_reps_randomness_and_commits_it_and_refuses_a_start_that_brings_any():
    f, g = Frame(), Frame()
    s = slots(**LAY)
    leaves = 1 << (s["len"] + K_PAD - 1).bit_length() << 3
    for x in (f, g):
        assert x.start[0] == f"ok {16 * (2 * s['len'] + 2 * K_PAD) + 192 + 192 * leaves + 64}"
        assert x.h[s["hab"]] == gmul(x.h[s["fa"]], x.h[s["fb"]]) and x.h[s["tw"]] == gmul(x.h[s["tx"]], x.h[s["ty"]])
        assert x.close() == 0
    assert f.start[1] != g.start[1] and f.root != g.root, "each rep's draws are fresh OS bytes"
    for start, why in [({"layout": LAY, "pads": ["00" * 16]}, "a start with `pads`"), ({"layout": LAY, "root": "00" * 64}, "a start with `root`")]:
        x = Frame(start=json.dumps(start))
        assert x.start == f"err the rep's start: {why}: the frame takes only the layout and the mask slot, and draws the rep's randomness itself"
        assert x.close() == 1
    x = Frame(mask={"w0": 0, "n": 0, "blocks": 1})
    assert x.start == "err the rep's start: a mask slot of no words" and x.close() == 1


def test_a_rep_leaves_masked_in_the_layouts_order():
    f = Frame()
    out = released(f, rep_rounds(ALG))
    assert [r[1] for r in out[:-1]] == release(f.h, f.root, ALG)
    assert out[-1] == "ok" and f.close() == 0


def test_a_failed_round_stops_the_rep_whose_algebraic_values_then_leave_as_their_pads_and_which_ends_in_stop():
    f = Frame()
    garbage = [(7 * i + 3) * 0x1_0000_0001 for i in range(len(ALG))]
    out = released(f, rep_rounds(garbage), fail_at=1)
    assert [r[1] for r in out[:-1]] == release(f.h, f.root, [0] * len(ALG)), \
        "a stopped rep stays stopped, and level 0's values are the firewall's own, masked as an honest rep's"
    assert out[-1] == "stop" and f.close() == 0


def test_a_stopped_rep_releases_under_pads_h_what_the_honest_rep_releases_under_h_plus_v():
    """`h ↦ h + v` is a bijection of the uniform pads, so before `Finish` a stopped rep's release is distributed as an honest
    rep's, whatever the device sent and wherever its check failed; only `finish` differs."""
    o = order(**LAY)
    garbage = [(5 * i + 1) << 70 for i in range(len(ALG))]
    #: each round's first absorbed position: the device's honest values before the failed round, anything from it on
    first = [0, 0, 2 * SKIP] + [len(ALG)] * (len(rep_rounds(ALG)) - 3)
    for fail_at, s in enumerate(first):
        f = Frame()
        stopped = released(f, rep_rounds(ALG[:s] + garbage[s:]), fail_at=fail_at)
        shifted = list(f.h)
        for i in range(s, len(ALG)):
            shifted[o[i]] ^= ALG[i]
        assert [r[1] for r in stopped[:-1]] == release(shifted, f.root, ALG) and stopped[-1] == "stop", fail_at
        assert f.close() == 0


def test_a_refused_round_releases_nothing_and_stops_the_rep_which_goes_on_with_the_scheduled_round_in_its_place():
    o = order(**LAY)
    f = Frame()
    rounds = rep_rounds(ALG)
    assert f.round(*rounds[0])[0].startswith("ok ")
    assert f.round(label("flock-r1cs-v0") + hdr(3, 1, 2) + bytes(32) + squeeze(1, 1), 1) == "err an observe of kind 1 and length 2"
    out = [f.round(r, n, fail=(i == 1)) for i, (r, n) in enumerate(rounds) if i > 0]
    pads = [f.h[o[i]] for i in range(2 * SKIP)]
    assert out[0][1] == obs(pads[:SKIP]) + obs(pads[SKIP:]) + squeeze(1, 1)
    assert f.ask("finish") == "stop" and f.close() == 0
    f = Frame()
    assert f.round(*rounds[0])[0].startswith("ok ")
    assert f.round(label("x") + squeeze(1, 1) + label("y") + squeeze(1, 1), 1) == "err a squeeze before the round's end"
    out = [f.round(r, n) for r, n in rounds[1:]]
    pads = [f.h[o[i]] for i in range(2 * SKIP)]
    assert out[0][1] == obs(pads[:SKIP]) + obs(pads[SKIP:]) + squeeze(1, 1), "a refused round stops the rep without a `fail` after it"
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


def test_each_released_rounds_coins_come_back_before_the_next_round_and_the_reps_end():
    for then, why in [(lambda f: f.ask("round 1 16", squeeze(1, 1)), "err a round before the 1 coins of the last one"),
                      (lambda f: f.ask("finish"), "err the rep ended before the 1 coins of its last round"),
                      (lambda f: f.coins([0, 0]), "err 2 coins for a round that squeezed 1")]:
        f = Frame()
        assert f.ask("round 1 16", squeeze(1, 1))[0] == "ok 16"
        assert then(f) == why
        assert f.close() == 1
    f = Frame()
    assert f.coins([0]) == "err coins for no round" and f.close() == 1


R0 = label("flock-r1cs-v0") + msg(b"\1" * 64) + label("flock-zerocheck-v0") + squeeze(2, 6)


def test_a_zerocheck_outer_coin_of_one_stops_the_rep_at_once_and_no_other_coin_of_one_does():
    lay = {"n_mlv": 8, "n_lc": 1, "n_ood0": 1, "k0": 1}
    f = Frame(lay)
    assert f.round(squeeze(1, 1), 1, coins=[1])[0] == "ok 16", "a coin before the algebraic phase"
    assert f.round(R0, 6, coins=[1] * 6)[0].startswith("ok "), "the skip's coins"
    assert f.ask("round 1 16", squeeze(1, 1))[0] == "ok 16"
    assert f.coins([1]) == "stop PROVER-STOPPED: degenerate coin: zerocheck outer coin r_0 = 1"
    assert f.close() == 0
    f = Frame()
    assert f.round(R0, 6)[0].startswith("ok ")
    assert f.round(squeeze(1, 1), 1, coins=[1])[0] == "ok 16", "with seven multilinear rounds or fewer there is no outer coin"
    assert f.close() == 0


def test_level_0s_rho_dependent_over_f128_stops_the_rep_at_once_and_no_other_level_0_coin_does():
    """`coin_stop_vectors.json`'s `ρ` cases, which Rust's `rho_independent` agrees with (`zk_veil`'s test of them): `ρ_0`
    and `ρ_1` are the two rounds after level 0's last masked value, and a coin of level 0 before it is no `ρ`."""
    rounds = rep_rounds(ALG)
    for v in json.loads(VECTORS.read_text())["rho"]:
        rho = [int(w, 16) for w in v["coins"]]
        f = Frame()
        for r, n in rounds[:3]:
            assert f.round(r, n)[0].startswith("ok ")
        assert f.round(*rounds[3], coins=rho[:2])[0].startswith("ok "), v["case"]
        assert f.ask("round 2 16", squeeze(2, 2))[0] == "ok 16"
        if v["independent"]:
            assert f.coins(rho[2:]) == "ok", v["case"]
            assert all(isinstance(f.round(r, n), tuple) for r, n in rounds[5:]) and f.ask("finish") == "ok" and f.close() == 0
        else:
            assert f.coins(rho[2:]) == "stop PROVER-STOPPED: degenerate coins: rho_0, rho_1 dependent over F128", v["case"]
            assert f.close() == 0
    f = Frame()
    for r, n in rounds[:3]:
        assert f.round(r, n)[0].startswith("ok ")
    early = label("flock-pcs-open-batch-v0") + obs([5]) + label("verity-flock-zk-l0-open-v0") + obs(LG[:1]) + squeeze(2, 4)
    assert f.round(early, 4, coins=[0] * 4)[0].startswith("ok "), "a lane round's coins, before level 0's last masked value"
    assert f.round(obs(LG[1:5]) + obs(LG[5:]) + squeeze(2, 2), 2)[0].startswith("ok ")
    assert all(isinstance(f.round(r, n), tuple) for r, n in rounds[4:]) and f.ask("finish") == "ok" and f.close() == 0


def test_level_0_closes_only_after_its_four_rho_coins():
    rounds = rep_rounds(ALG)
    f = Frame()
    for r, n in rounds[:4]:
        assert f.round(r, n)[0].startswith("ok ")
    assert f.round(*rounds[5]) == "err level 0 closed after 2 of its 4 ρ coins" and f.close() == 0


def test_beta_zero_in_the_inner_proof_stops_the_rep_at_once_and_no_other_inner_coin_does():
    """`coin_stop_vectors.json`'s `β` cases, which Rust's `prove_inner` agrees with (`zk_veil`'s test of them): `β` is the
    coin of the inner proof's third round, counted from its label."""
    rounds = rep_rounds(ALG)
    k = len(rounds) - 2
    for v in json.loads(VECTORS.read_text())["beta"]:
        f = Frame()
        for r, n in rounds[:k]:
            assert f.round(r, n, coins=[0] * n if r in (x for x, _ in INNER) else None)[0].startswith("ok ")
        assert f.ask(f"round 1 {len(rounds[k][0])}", rounds[k][0])[0].startswith("ok ")
        if v["stop"]:
            assert f.coins([int(v["coin"], 16)]) == "stop PROVER-STOPPED: degenerate coin β = 0" and f.close() == 0
        else:
            assert f.coins([int(v["coin"], 16)]) == "ok"
            assert f.round(*rounds[k + 1], coins=[0] * 4)[0].startswith("ok ") and f.ask("finish") == "ok" and f.close() == 0


def test_the_inner_proof_comes_once_after_the_opening_and_the_rep_ends_only_after_its_four_rounds():
    rounds = rep_rounds(ALG)
    f = Frame()
    for r, n in rounds[:3]:
        assert f.round(r, n)[0].startswith("ok ")
    assert f.round(*INNER[0]) == "err the inner proof began before the opening ended" and f.close() == 0
    f = Frame()
    for r, n in rounds[:-1]:
        assert f.round(r, n)[0].startswith("ok ")
    assert f.ask("finish") == "err the rep ended after 3 of the inner proof's 4 rounds" and f.close() == 1
    f = Frame()
    for r, n in rounds[:-1]:
        assert f.round(r, n)[0].startswith("ok ")
    assert f.round(*INNER[0]) == "err a second inner proof" and f.close() == 0
    f = Frame()
    for r, n in rounds:
        assert f.round(r, n)[0].startswith("ok ")
    assert f.round(squeeze(1, 1), 1) == "err a round after the inner proof's 4" and f.close() == 0


#: a layout with three outer coins and no masked level 0, whose claim points have nine word coordinates (512 words)
RANK = {"n_mlv": 10, "n_lc": 1, "n_ood0": 1, "k0": 0}


def algebraic(lay: dict, seed: int) -> list[tuple[bytes, list[int]]]:
    """A rep's algebraic phase on `lay` (every value zero) with the coins a stream answers each round (seeded): the skip's,
    the outer ones, `z`, the multilinear rounds' with `α` and `β`, the lincheck's with `r'`."""
    rng = random.Random(seed)
    nouter = lay["n_mlv"] - 7
    cs = [[rng.getrandbits(128) | 2 for _ in range(n)] for n in (6, nouter, 1, lay["n_mlv"] + 2, lay["n_lc"] + 1)]
    return [(R0, cs[0]), (squeeze(2, nouter), cs[1]), (obs([0] * SKIP) + obs([0] * SKIP) + squeeze(1, 1), cs[2]),
            (obs([0] * 2 * lay["n_mlv"]) + obs([0], 1) + obs([0], 1) + squeeze(2, lay["n_mlv"] + 2), cs[3]),
            (obs([0] * 2 * lay["n_lc"]) + obs([0] * SKIP) + squeeze(2, lay["n_lc"] + 1), cs[4])]


def opening(f: Frame, phase) -> object:
    """The opening's label after the algebraic phase `phase`: its answer."""
    for r, c in phase:
        assert f.round(r, len(c), coins=c)[0].startswith("ok ")
    return f.ask(f"round 1 {len(label('flock-pcs-open-batch-v0') + squeeze(1, 1))}", label("flock-pcs-open-batch-v0") + squeeze(1, 1))


def test_a_mask_slot_short_of_full_rank_on_the_tables_claim_points_stops_the_rep_at_the_openings_label():
    phase = algebraic(RANK, 7)
    f = Frame(RANK, mask={"w0": 0, "n": 1, "blocks": 512})
    assert opening(f, phase)[0] == "ok 64"
    assert f.close() == 0
    assert 'FRAMERANK\t{"points": 2, "mask_words": 512, "rank": 256}' in f.err
    f = Frame(RANK, mask={"w0": 0, "n": 1, "blocks": 1})
    assert re.fullmatch(r"stop PROVER-STOPPED: degenerate coins: the mask slot's weights have rank [01] < 256 over 2 claim points",
                        opening(f, phase))
    assert f.close() == 0
    f = Frame(RANK, mask={"w0": 0, "n": 1, "blocks": 1})
    coins = [c for _, cs in algebraic(RANK, 8) for c in cs]
    assert f.ask(f"prior {len(coins)}", wire(coins)) == "ok"
    assert re.fullmatch(r"stop PROVER-STOPPED: degenerate coins: the mask slot's weights have rank [01] < 512 over 4 claim points",
                        opening(f, phase)), "an earlier rep's claim points count too"
    assert f.close() == 0


def test_an_earlier_reps_coins_come_whole_and_before_the_reps_first_round():
    f = Frame(RANK)
    assert f.ask("prior 3", bytes(48)) == "err an earlier rep's coins: 3 coins in the algebraic phase, not 24" and f.close() == 1
    f = Frame(RANK)
    assert f.round(R0, 6)[0].startswith("ok ")
    assert f.ask("prior 24", bytes(16 * 24)) == "err an earlier rep's coins after the rep's first round" and f.close() == 1


def test_the_tap_serves_fresh_os_bytes_by_the_count_asked_and_ends_on_anything_else():
    p = subprocess.Popen([str(BIN), "tap"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate(b"32\n0\n32\n")
    assert p.returncode == 0 and len(out) == 64 and out[:32] != out[32:], "two draws of 32 bytes, each the OS's own"
    for bad, why in [(b"x\n", b"a request `x`"), (b"16777217\n", b"16777217 bytes in one request, past 16777216")]:
        p = subprocess.Popen([str(BIN), "tap"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate(bad + b"32\n")
        assert (p.returncode, out) == (1, b"") and why in err
