"""The I/O tie's prototype (`io_tie`) on a fake upstream: N2's commitment shape (four hm96 row leaves, the egress head
at leaf 2) with no certificates, a link-window's id frames in request-major order and a token port. The reference
accepts the honest window and refuses each way a prover can make a drawn frame differ from the committed id; the
Boolean parts agree with it."""

import hashlib
import random
from dataclasses import replace

import numpy as np
import pytest

from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation.bits import evaluate_bits
from verity.ml.commitments.gates.lms import Sha512Compress
from verity.ml.commitments.indexed import RangeIndexedDomain
from verity.ml.commitments.merkle import CommitmentDomain
from verity.core.protocols.certifier import head_check as HC, io_tie as IT
from verity.core.protocols.certifier.certificate import Direction, RecordHead, frames_root, open_frame
from verity.core.protocols.certifier.grid import GridParams, Status

PARAMS = GridParams(buckets=4, allocation=40, frame_bytes=256)
D = 8
PORT = IT.TokenPort(F=32, P=32, C=2, D=D)
LINK = "served"


def coin_for(k: int, n: int) -> int:
    """The least coin that draws frame k of n."""
    return -(-(k << 64) // n)


class Window:
    """A fake upstream's egress link-window: its requests' frames, head, N2-shaped commitment and token port."""

    def __init__(self, requests, frames=None, port_rows=None, seed=b"w"):
        self.requests = requests
        self.frames = frames if frames is not None else [IT.id_frame(x, PARAMS.frame_bytes)
                                                         for outputs, _ in requests for x in outputs]
        n, counts = len(self.frames), []
        for _ in range(PARAMS.buckets):
            counts.append(min(PARAMS.allocation, n - sum(counts)))
        self.head = RecordHead(LINK, Direction.EGRESS, 0, PARAMS.buckets, PARAMS.allocation, PARAMS.frame_bytes,
                               PARAMS.bucket_ms, Status(), True, bytes(64), tuple(counts),
                               frames_root(self.frames, PARAMS)).encode()
        n2 = CommitmentDomain(hashlib.sha512(b"n2" + seed).digest(), -1, RangeIndexedDomain(0, 4), hash="sha512")
        self.salt = IT.fake_salt(seed + b"n2", 2)
        rows = [b"egress record", b"ingress record", self.head, b"ingress head"]
        levels = IT.port_levels(n2, rows, [IT.fake_salt(seed + b"n2", i) if i != 2 else self.salt for i in range(4)])
        self.record, self.record_domain, self.head_path = levels[-1][0], n2.domain_id, IT.path(levels, 2)
        self.port_rows = port_rows if port_rows is not None else PORT.rows(requests)
        self.port_salts = [IT.fake_salt(seed + b"port", i) for i in range(PORT.leaves)]
        pd = PORT.domain(hashlib.sha512(b"port" + seed).digest())
        self.port_levels = IT.port_levels(pd, self.port_rows, self.port_salts)
        self.port_root, self.port_domain = self.port_levels[-1][0], pd.domain_id

    def prove(self, coins) -> IT.Instance:
        """The prover's instance for the challenger's coins: each drawn frame and its port row, with their paths."""
        n, draws = len(self.frames), []
        for coin in coins:
            k = IT.draw(coin, n) if n else 0
            row = PORT.at(k)[0] if n else 0
            draws.append(IT.Draw(self.frames[k] if n else bytes(PARAMS.frame_bytes),
                                 open_frame(self.frames, PARAMS, k) if n else (bytes(64),) * D,
                                 self.port_rows[row], self.port_salts[row], IT.path(self.port_levels, row)))
        return IT.Instance(self.record, self.record_domain, self.head, self.salt, self.head_path, len(LINK), PARAMS,
                           PORT, self.port_root, self.port_domain, tuple(coins), tuple(draws))


def requests(seed: int, count: int = 3):
    rng = random.Random(seed)
    return [([rng.randrange(1 << 32) if g == 0 and j < 2 else rng.randrange(151_329) for j in range(PORT.F)],
             [rng.randrange(151_329) for _ in range(rng.randrange(PORT.P + 1))]) for g in range(count)]


@pytest.fixture(scope="module")
def honest():
    return Window(requests(1))


def test_an_id_frame_parses_to_its_id_and_nothing_else_parses():
    for x in (0, 7, 10, 151_328, (1 << 32) - 1):
        f = IT.id_frame(x, 256)
        assert len(f) == 256 and IT.frame_id(f) == x and f[IT.ID_AT:IT.ID_AT + 10] == b"%10d" % x
    f = IT.id_frame(12, 256)
    for digits in (b"0000000012", b"        1 ", b"          ", b"      1 2 ", b"4294967296", b"        -1", b"       +12"):
        assert IT.frame_id(f[:IT.ID_AT] + digits + f[IT.ID_AT + 10:]) is None, digits
    assert IT.frame_id(f[:-1] + b"\x01") is None
    assert IT.frame_id(f[:100] + b"X" + f[101:]) is None
    with pytest.raises(ValueError):
        IT.id_event(1 << 32)


def test_the_port_places_request_g_output_j_at_its_row_and_column():
    assert PORT.DT == 4 and PORT.leaves == 16
    reqs = requests(2)
    rows = PORT.rows(reqs)
    for g, (outputs, prompt) in enumerate(reqs):
        for s, x in enumerate([*outputs, *prompt, *[0xFFFF_FFFF] * (PORT.P - len(prompt))]):
            row, col = PORT.slot(g, s)
            assert int.from_bytes(rows[row][4 * col:4 * col + 4], "big") == x
        for j in range(PORT.F):
            assert PORT.at(g * PORT.F + j) == PORT.slot(g, j)
    assert all(r == bytes(128) for r in rows[len(reqs) * PORT.C:])
    with pytest.raises(ValueError, match="power of two"):
        IT.TokenPort(F=32, P=32, C=3, D=D)


def test_the_honest_window_is_accepted_at_every_frame_and_an_empty_one_accepts(honest):
    n = len(honest.frames)
    inst = honest.prove([coin_for(k, n) for k in range(n)])
    assert [IT.draw(c, n) for c in inst.coins] == list(range(n))
    assert IT.reference_draws(inst) == [True] * n
    assert IT.reference(Window([]).prove([5, 1 << 63]))


def test_each_way_a_drawn_frame_differs_from_the_committed_id_is_refused(honest):
    reqs, n = honest.requests, len(honest.frames)
    every = [coin_for(k, n) for k in range(n)]
    dup = honest.frames[:11] + honest.frames[10:-1]
    shifted = Window(reqs, frames=dup)
    got = IT.reference_draws(shifted.prove(every))
    differs = [k < 11 or honest.frames[k] == dup[k] for k in range(n)]
    assert got == differs and not all(got)

    swapped = Window([reqs[1], reqs[0], reqs[2]], port_rows=honest.port_rows)
    assert not any(IT.reference_draws(swapped.prove(every))[:2 * PORT.F])

    rows = list(honest.port_rows)
    row, col = PORT.at(40)
    rows[row] = rows[row][:4 * col] + ((int.from_bytes(rows[row][4 * col:4 * col + 4], "big") + 1) % (1 << 32)).to_bytes(4, "big") + rows[row][4 * col + 4:]
    assert IT.reference_draws(Window(reqs, port_rows=rows).prove(every)) == [k != 40 for k in range(n)]

    padded = list(honest.frames)
    x = IT.frame_id(padded[5])
    padded[5] = padded[5][:IT.ID_AT] + b"%010d" % x + padded[5][IT.ID_AT + 10:]
    assert IT.reference_draws(Window(reqs, frames=padded).prove(every)) == [k != 5 for k in range(n)]

    inst = honest.prove(every[:3])
    assert not IT.reference(replace(inst, port_root=Window(requests(9)).port_root))
    assert not IT.reference(replace(inst, draws=(replace(inst.draws[0], frame_path=inst.draws[1].frame_path),
                                                 *inst.draws[1:])))
    with pytest.raises(ValueError, match="doesn't open"):
        IT.reference(replace(inst, salt=IT.fake_salt(b"other", 0)))


def _bits(rows):
    return np.asarray(rows, dtype=np.uint8)


def _eval(fn, *args):
    return [int(b) for b in np.asarray(evaluate_bits(fn, *[_bits(a) for a in args])).reshape(-1)]


def test_the_event_and_column_circuits_are_their_references(honest):
    frames = list(honest.frames[:40])
    for digits in (b"0000000012", b"          ", b"      1 2 ", b"4294967296", b"9999999999", b"         0"):
        frames.append(frames[0][:IT.ID_AT] + digits + frames[0][IT.ID_AT + 10:])
    frames.append(frames[1][:-1] + b"\x01")
    ids = [IT.frame_id(f) for f in frames]
    claims = [x if x is not None else 12 for x in ids]
    claims[1] = (claims[1] + 1) % (1 << 32)
    want = [int(x is not None and x == c) for x, c in zip(ids, claims)]
    assert sum(want) >= 40
    got = _eval(bind(IT.IoTieEvent, B=PARAMS.frame_bytes), [HC._bytes_bits(f) for f in frames],
                [HC._int_bits(c, 32) for c in claims])
    assert got == want

    rows = [r for r in honest.port_rows[:6] for _ in range(32)]
    out = np.asarray(evaluate_bits(bind(IT.IoTieColumn), _bits([HC._bytes_bits(r) for r in rows]),
                                   _bits([HC._int_bits(c % 32, 5) for c in range(len(rows))])))
    for i, r in enumerate(rows):
        assert int("".join(map(str, out.reshape(len(rows), 32)[i][::-1])), 2) == int.from_bytes(r[4 * (i % 32):4 * (i % 32) + 4], "big")


def test_the_parts_ands():
    sha = HC.ands(bind(Sha512Compress))
    assert HC.ands(bind(IT.NetworkFrameSwap)) == 512 and HC.ands(bind(IT.IoTieColumn)) == 992
    for B, leaf in ((256, 3), (4096, 33)):
        for d in (1, 13):
            assert HC.ands(bind(IT.NetworkFrameRead, B=B, D=d)) == (leaf + 2 * d) * sha + 512 * d + 511
    event = HC.ands(bind(IT.IoTieEvent, B=4096))
    assert 8 * 4096 - 80 - 1 < event < 8 * 4096 - 80 + 2_000


@pytest.mark.slow
def test_the_draw_circuit_is_the_reference_on_each_draw(honest):
    """`IoTieDraw` on honest draws and on draws wrong in one input each, against the reference's verdict per draw.
    What the parts' tests can't reach is the wiring: the frame into its read and its event, the port row's column and
    position from k's bits, and the row into its hiding read. Building the commitment gadgets' Boolean bodies takes
    most of its time."""
    n = len(honest.frames)
    ks = [0, 1, 31, 32, 33, 63, 64, n - 1]
    inst = honest.prove([coin_for(k, n) for k in ks])
    rows = list(honest.port_rows)
    row, col = PORT.at(33)
    rows[row] = rows[row][:4 * col] + b"\0\0\0\0" + rows[row][4 * col + 4:]
    wrong = Window(honest.requests, port_rows=rows).prove(inst.coins)
    other = Window(requests(5))
    cases = [(k, d, inst.port_root) for k, d in zip(ks, inst.draws)]
    cases += [(k, d, wrong.port_root) for k, d in zip(ks, wrong.draws)]
    cases += [(ks[0], replace(inst.draws[0], frame_path=inst.draws[1].frame_path), inst.port_root),
              (ks[1], replace(inst.draws[1], salt=inst.draws[2].salt), inst.port_root),
              (ks[2], replace(inst.draws[2], frame=other.frames[ks[2]]), inst.port_root)]
    want = [int(IT.reference(replace(inst, coins=(coin_for(k, n),), draws=(d,), port_root=root))) for k, d, root in cases]
    assert want[:len(ks)] == [1] * len(ks) and want[len(ks):] == [int(k != 33) for k in ks] + [0, 0, 0]
    fn = bind(IT.IoTieDraw, B=PARAMS.frame_bytes, D=D, FOUT=PORT.F, C=PORT.C)
    _, frames_root_ = IT.head_counts(honest.head, len(LINK), PARAMS.buckets)
    args = [IT.draw_arguments(frames_root_, k, d, root, inst.port_domain, D) for k, d, root in cases]
    assert _eval(fn, *[list(col_) for col_ in zip(*args)]) == want
