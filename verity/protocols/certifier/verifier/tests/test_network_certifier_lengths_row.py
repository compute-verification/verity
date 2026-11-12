"""The lengths row's proof block (`lengths_row`): its request row is the arrival slots' layout, its bytes are the
canonical id request's whatever the cell size, the Boolean Definition computes its reference's outputs bit for bit on
rows of its form and on rows that are not, and on the network certifier's own releases, signed and registered
(`connection_rows`), the prover's blocks return the signed lengths rows, which the verifier's registered reads accept,
while a request row with another length, a fails row with another cell and a request row with another place in the
release order are refused: by the block's verdict, or by the read of a row that is not the signed one."""

from __future__ import annotations

import dataclasses
import random
import struct

import numpy as np
import pytest

import signers
from verity.protocols.certifier.verifier import connection_rows as CR
from verity.protocols.certifier.verifier import head_check as HC
from verity.protocols.certifier.verifier import replay_chain as RC
from verity.protocols.certifier.verifier import warden_rows as WR
from verity.protocols.certifier.verifier.active import id_request
from verity.protocols.certifier.verifier.certificate import Issuer
from verity.protocols.certifier.verifier.tests.test_network_certifier_lengths import LEN, PLAIN, traffic
from verity.protocols.certifier.verifier.tests.test_network_certifier_warden import run
from verity.circuits.defs import bind
from verity.circuits.evaluation.bits import evaluate_bits
from verity.protocols.certifier.prover import lengths_row as LR
from verity.protocols.certifier.prover.lengths_row import ACCEPT, Request

LINK = "responses"
T, K = LEN.window // LEN.every, LEN.stride_releases
#: the block that reads the lengths tests' schedule: u32 places, counts of 8 bits
SH = LR.Shape.of(LEN, 32, 8)
#: the shapes the circuit is held to its reference at: the tests' schedule, a later cell that leaves one byte (505),
#: a first cell that does (509), the smallest cell, 4,096-byte cells at 18-bit counts and at counts no request of
#: which leaves the first cell, and a place of a few bits
SHAPES = (dict(K=2, B=512, IW=32, LW=8), dict(K=3, B=505, IW=7, LW=10), dict(K=1, B=509, IW=32, LW=12),
          dict(K=2, B=387, IW=16, LW=9), dict(K=2, B=4096, IW=32, LW=18), dict(K=2, B=4096, IW=20, LW=8),
          dict(K=5, B=395, IW=3, LW=17))


def fails_row(*cells: tuple[int, int]) -> bytes:
    return b"".join(struct.pack("<HH", cls, at) for cls, at in cells)


# --- the shape and the rows ------------------------------------------------------------------------------------------


def test_the_shape_and_the_statics_are_checked():
    """A row of K slots is 96·K bits beside the two u16s, within the widest size class, so K is at most 2,730; a cell
    holds an id request's head and first prompt id; a place is at most a u32 and the longest prompt's request below
    2^32 bytes. The shape that reads a schedule is its K and B, and a schedule without lengths rows has none."""
    assert (SH.K, SH.B, SH.request_bits, SH.fails_bits, SH.row_bits) == (2, 512, 112, 64, 192)
    assert LR.Shape(2730, 4096, 32, 18).row_bits + 32 <= 262_144 < 2731 * 96 + 32
    assert LR.Shape(1, 4096, 32, 28).LW == 28
    assert tuple(SH.form) == (503, 367, 12, 4, 45, 8) and tuple(LR.form(4096)) == (4087, 367, 338, 2, 371, 6)
    refused = ((dict(K=0), "1 <= K <= 2730"), (dict(K=2731), "1 <= K <= 2730"), (dict(K=True), "integers"),
               (dict(B=256), "doesn't fit its first cell"), (dict(IW=0), "IW <= 32"),
               (dict(IW=33), "IW <= 32"), (dict(LW=0), "LW <= 32"), (dict(LW=33), "LW <= 32"),
               (dict(B=4096, LW=29), "2\\^32 bytes or more"))
    for kw, match in refused:
        statics = SH.statics() | kw
        with pytest.raises(ValueError, match=match):
            LR.Shape(**statics)
        with pytest.raises(ValueError, match=match):
            bind(LR.LengthsRow, **statics).signature()
    with pytest.raises(ValueError, match="no lengths rows"):
        LR.Shape.of(PLAIN, 32, 8)


def test_a_schedule_whose_stride_releases_more_than_a_row_holds_is_refused_when_it_is_built():
    """One rule (`connection_rows.check_slots`) holds a lengths row to the slots its proof block returns. The shape
    calls it, and so do a schedule's parameters with lengths rows, so a network certifier is never set to a stride
    whose rows no proof block returns: the refusal names the longest stride of the window that fits, or the most
    lanes where none does. A schedule without lengths rows is not held to it."""
    CR.check_slots(2730)
    for K in (0, 2731):
        with pytest.raises(ValueError, match="1 <= K <= 2730 slots.*fewer lanes or a shorter stride"):
            CR.check_slots(K)
    # 1,000 lanes: a stride of 16 buckets releases up to 4,000 requests and one of 8 up to 2,000
    assert dataclasses.replace(LEN, lanes=1000, every=8).stride_releases == 2000
    assert LR.Shape.of(dataclasses.replace(LEN, lanes=1000, every=8), 32, 8).K == 2000
    with pytest.raises(ValueError, match="1 <= K <= 2730 slots, not K=4000.*stride of 16 buckets releases up to 4000 "
                                         "requests.*`every` is at most 8 with"):
        dataclasses.replace(LEN, lanes=1000, every=16)
    for every in (1, 16):
        with pytest.raises(ValueError, match="no stride of 3000 lanes releases as few, so at most 2730 lanes"):
            dataclasses.replace(LEN, lanes=3000, every=every)
    for change in (dict(lanes=1000, every=16), dict(lanes=3000)):
        assert dataclasses.replace(PLAIN, **change).lengths is None


def test_the_block_reads_two_rows_and_returns_the_lengths_row():
    """The proof block's inputs are the request row and the fails row, whole 16-bit words each, and nothing else
    (every switch computed, none witness); its outputs are two u16s and the lengths row, gates of the circuit."""
    for sh in (SH, LR.Shape(16, 4096, 32, 18), LR.Shape(3, 505, 7, 10)):
        params, out = LR.block(sh).signature()
        assert [(name, t.leaves) for name, t in params] == [("requests", sh.request_bits), ("fails", 32 * sh.K)]
        assert sh.request_bits % 16 == 0 and 0 <= sh.request_bits - sh.K * (1 + sh.IW + 2 * sh.LW) < 16
        assert [t.leaves for t in out.fields] == [16, 16, 96 * sh.K] and out.width_set() == frozenset({1})


def test_the_request_row_is_the_arrival_slots_layout():
    """The request row's fields are `replay_chain`'s K arrival slots without a cap bit (valid, id, P, O as a struct of
    arrays, LW the bit length of the longest request), each element least significant bit first, zero-padded to whole
    words."""
    sh = LR.Shape(3, 512, 7, 10)
    chain = RC.Shape(R=1, W=1, A=1, L=1023, NB=1, IW=7, Q=1, D=1, P=1, K=3)
    assert RC.widths(chain).LW == sh.LW
    assert [(name.removeprefix("a_"), n, w) for name, n, w in RC.arrival_fields(chain)] == LR.request_fields(sh)
    slots = [Request(1, 5, 700, 3), Request(0, 127, 1023, 1023), Request(1, 0, 1, 0)]
    bits = LR.request_row(sh, slots)
    assert len(bits) == sh.request_bits == 96

    def field(at, w):
        return sum(b << i for i, b in enumerate(bits[at:at + w]))

    assert [field(k, 1) for k in range(3)] == [1, 0, 1]
    assert [field(3 + 7 * k, 7) for k in range(3)] == [5, 127, 0]
    assert [field(24 + 10 * k, 10) for k in range(3)] == [700, 1023, 1]
    assert [field(54 + 10 * k, 10) for k in range(3)] == [3, 1023, 0]
    assert not any(bits[84:])
    assert LR.request_row(sh, slots[:1]) == LR.request_row(sh, [slots[0], LR.EMPTY, LR.EMPTY])
    for bad in ([Request(2, 0, 0, 0)], [Request(1, 128, 0, 0)], [Request(1, 0, 1024, 0)], [Request(1, 0, 0, -1)],
                [LR.EMPTY] * 4):
        with pytest.raises(ValueError, match="slot|slots"):
            LR.request_row(sh, bad)
    with pytest.raises(ValueError, match="fails row"):
        LR.arguments(sh, slots, bytes(8))


@pytest.mark.parametrize("B", (387, 395, 505, 509, 512, 4096))
def test_request_bytes_is_the_id_requests_length(B):
    """`request_bytes` is the length of the canonical id request in B-byte cells (`active.id_request` at offset 0) at
    every prompt length around each cell's end and at lengths between, whatever the request's other fields, and it
    is increasing in the prompt's length, so a request's bytes hold its prompt's length."""
    f = LR.form(B)
    near = {p for c in range(4) for p in range(f.first + c * f.each - 2, f.first + c * f.each + 4) if p >= 1}
    rng = random.Random(B)
    lengths = sorted(near | {1, 2, 3} | {rng.randrange(1, f.first + 6 * f.each) for _ in range(40)})
    for p in lengths:
        made = id_request(model="m" * (p % 5), user=f"u{p}", prompt=[rng.randrange(1 << 32) for _ in range(p)],
                          max_tokens=p, seed=B, temperature=0.5, top_p=0.25, frame_bytes=B, offset=0)
        assert LR.request_bytes(B, p) == len(made), (B, p)
    sizes = [LR.request_bytes(B, p) for p in range(f.first + 3 * f.each + 2)]
    assert all(b - a >= 11 for a, b in zip(sizes, sizes[1:])) and sizes[0] == f.start + 2
    with pytest.raises(ValueError, match="number of ids"):
        LR.request_bytes(B, -1)


# --- the circuit -----------------------------------------------------------------------------------------------------


def _rows(sh: LR.Shape, rng: random.Random, n: int) -> list[tuple[list[Request], bytes]]:
    """Request rows and fails rows, of their form and not: prompts at each cell's end and anywhere, slots that are
    not valid with fields set, and fails cells of every class, with `at` at, below and above the request's outputs."""
    f, top = sh.form, (1 << sh.LW) - 1
    ends = sorted({p for c in range(3) for p in (f.first + c * f.each - 1, f.first + c * f.each, f.first + c * f.each + 1)
                   if 0 <= p <= top} | {0, 1, top - 1, top})
    out = []
    for _ in range(n):
        requests, cells = [], []
        for _ in range(sh.K):
            count = rng.choice((0, 1, rng.randrange(top + 1), min(top, rng.randrange(40)), top))
            order = rng.randrange(1 << sh.IW) if rng.random() < 0.7 else rng.choice((0, (1 << sh.IW) - 2, (1 << sh.IW) - 1))
            requests.append(Request(int(rng.random() < 0.8), order,
                                    rng.choice(ends) if rng.random() < 0.6 else rng.randrange(top + 1), count))
            how = rng.random()
            if how < 0.4:
                cells.append((0, 0))
            elif how < 0.7:
                cells.append((1, rng.randrange(min(count, 0xFFFF) + 1)))
            else:
                cells.append((rng.choice((0, 1, 2, 3, 0x8001, 0x100)), rng.choice((0, 1, count, count + 1, 0xFFFF)) & 0xFFFF))
        out.append((requests, fails_row(*cells)))
    return out


@pytest.mark.parametrize("statics", SHAPES, ids=lambda s: "-".join(f"{k}{v}" for k, v in s.items()))
def test_the_circuit_is_its_reference(statics):
    """The Boolean Definition's three outputs are the reference's on every row: the verdict, the failures and the
    lengths row it still computes where the verdict refuses, with rows it accepts and rows it refuses both among
    them."""
    sh = LR.Shape(**statics)
    rows = _rows(sh, random.Random(sh.B * 31 + sh.LW), 200)
    want = [LR.outputs(sh, requests, fails) for requests, fails in rows]
    assert LR.evaluate(sh, rows) == want
    assert {verdict for verdict, _, _ in want} == {0, ACCEPT}
    leaves = evaluate_bits(LR.block(sh), *[np.asarray([a], dtype=np.uint8) for a in LR.arguments(sh, *rows[0])])
    assert np.asarray(leaves).reshape(-1).tolist() == LR.output_bits(sh, *rows[0])


def test_the_verdict_and_the_row_slot_by_slot():
    """A slot that is not valid is an empty slot of the table whatever its fields hold; a valid one is its place plus
    one, its request's bytes, and its outputs, or where its connection failed the cells sent before the `FAIL`. The
    verdict refuses a class past 1, a slot without a failure whose `at` isn't 0, a failure on a slot that isn't
    valid or after more cells than the request has outputs, and a place whose successor is no u32; the failures
    count the slots whose connection failed."""
    sh = LR.Shape(3, 512, 32, 8)
    slots = [Request(1, 6, 3, 9), Request(0, 77, 200, 13), Request(1, 7, 40, 5)]
    verdict, failures, row = LR.outputs(sh, slots, fails_row((0, 0), (0, 0), (1, 5)))
    assert (verdict, failures) == (ACCEPT, 1)
    assert struct.unpack("<9I", row) == (7, LR.request_bytes(512, 3), 9, 0, 0, 0, 8, LR.request_bytes(512, 40), 5)
    assert LR.refusals(sh, slots, fails_row((0, 0), (0, 0), (1, 0))) == []
    for cells, why in ((((2, 0), (0, 0), (0, 0)), "class is 2"), (((0, 4), (0, 0), (0, 0)), "no failure and at = 4"),
                       (((0, 0), (1, 0), (0, 0)), "no request"), (((0, 0), (0, 0), (1, 6)), "after 6 outputs"),
                       (((3, 10), (0, 0), (0, 0)), "class is 3")):
        assert why in LR.refusals(sh, slots, fails_row(*cells))[0]
        assert LR.outputs(sh, slots, fails_row(*cells))[0] == 0
    last = [Request(1, (1 << 32) - 1, 3, 9)]
    assert "no u32" in LR.refusals(sh, last, bytes(12))[0] and LR.outputs(sh, last, bytes(12))[2][:4] == bytes(4)
    assert LR.outputs(sh, [Request(1, (1 << 32) - 2, 3, 9)], bytes(12))[:2] == (ACCEPT, 0)
    assert LR.evaluate(sh, [(last, bytes(12)), (slots, fails_row((1, 9), (0, 0), (1, 5)))]) == \
        [LR.outputs(sh, last, bytes(12)), (ACCEPT, 2, LR.outputs(sh, slots, fails_row((1, 9), (0, 0), (1, 5)))[2])]


def test_the_blocks_ands():
    """The proof block's ANDs at the bindings circuit-check pins: a row of the tests' schedule, and a slot and
    sixteen at 4,096-byte cells with u32 places and 18-bit counts, a slot's gates their difference's fifteenth."""
    memo: dict = {}
    assert HC.ands(LR.block(SH), memo) == ANDS_TESTS
    one, sixteen = (HC.ands(LR.block(LR.Shape(k, 4096, 32, 18)), memo) for k in (1, 16))
    assert (one, sixteen) == (ANDS_ONE, ANDS_SIXTEEN) and (sixteen - one) % 15 == 0
    assert (sixteen - one) // 15 == ANDS_A_SLOT


#: `LengthsRow_v1{K=2,B=512,IW=32,LW=8}`'s
ANDS_TESTS = 402
#: `LengthsRow_v1{K=1,B=4096,IW=32,LW=18}`'s and `LengthsRow_v1{K=16,B=4096,IW=32,LW=18}`'s, and a slot's among them
ANDS_ONE, ANDS_SIXTEEN, ANDS_A_SLOT = 349, 5_614, 351


# --- the network certifier's releases --------------------------------------------------------------------------------

#: each of the traffic's five requests as the compute side holds it, by its place in the release order: its prompt's
#: ids, and its outputs: what the network certifier sent for the three it ended, and more than it sent before the
#: `FAIL` for the two the server failed (none of four, two of six)
PROMPTS, OUTPUTS = (3, 40, 100, 5, 7), (3, 8, 2, 4, 6)


@pytest.fixture(scope="module")
def signed(tmp_path_factory):
    """The lengths tests' traffic through the network certifier, each window's connection rows signed and their
    registration read as the verifier reads it: by window, the opening, the head and domains the verifier derived,
    and the compute side's request rows, one a slot stride."""
    _, arrivals, answers = traffic()
    warden, _, _ = run(answers, arrivals, params=LEN, windows=3)
    issuer = Issuer(signers.signer(tmp_path_factory.mktemp("signer")), signers.SESSION)
    out = {}
    for window, strides in dict(warden.lengths).items():
        opened, cert = CR.attest(issuer, LINK, window, strides, LEN, None)
        head, domains = CR.check_registration(CR.registration(opened.head, cert, LEN), issuer.signer.public_key(),
                                              signers.SESSION, LEN, LINK, window)
        requests = [[Request(1, r.order, PROMPTS[r.order], OUTPUTS[r.order]) for r in stride] for stride in strides]
        out[window] = (opened, head, domains, requests)
    assert [[len(s) for s in requests] for _, _, _, requests in out.values()] == \
        [[2, 0, 0, 0, 0, 1, 0, 0], [0, 2, 0, 0, 0, 0, 0, 0]]
    return out


def _refusal(opened, head, domains, table: str, rows, positions=range(T), reads_from=0) -> str | None:
    """The verifier's registered read of `rows` as the rows of `table` at `positions`, each under the network
    certifier's salt and path for its position (`warden_rows.refusal` on `connection_rows.port`): None, or why not."""
    reads = [LR.reads(opened.table(table), p) for p in positions]
    entry = {"value": reads[0].entry["value"], "positions": list(positions),
             "paths": [r.entry["paths"][0] for r in reads]}
    return WR.refusal(CR.port(head, table, reads_from), domains[table], entry,
                      [WR.bc(row, r.salt) for row, r in zip(rows, reads)])


def _computed(requests, fails) -> list[tuple[int, int, bytes]]:
    """The circuit's outputs on each slot stride's rows."""
    return LR.evaluate(SH, list(zip(requests, fails)))


def test_an_honest_row_passes(signed):
    """Every slot stride's proof block of both windows, loaded or idle: the block accepts, its failures are the
    stride's connections that ended in a `FAIL`, and the lengths row its gates compute is the row the network
    certifier signed, so the verifier's registered reads accept it as the lengths table's row j and the fails row as
    the fails table's, each under the head the verifier checked."""
    for window, (opened, head, domains, requests) in signed.items():
        blocks = LR.blocks(SH, opened, requests)
        lengths, fails = opened.table(CR.LENGTHS).rows, opened.table(CR.FAILS).rows
        assert [b.position for b in blocks] == list(range(T))
        assert [(b.fails, b.lengths) for b in blocks] == list(zip(fails, lengths))
        got = _computed([b.requests for b in blocks], [b.fails for b in blocks])
        assert got == [(ACCEPT, b.failures, b.lengths) for b in blocks]
        assert [b.failures for b in blocks] == [[0] * T, [0, 2] + [0] * (T - 2)][window]
        assert _refusal(opened, head, domains, CR.LENGTHS, [row for _, _, row in got]) is None
        assert _refusal(opened, head, domains, CR.FAILS, [b.fails for b in blocks]) is None
        assert _refusal(opened, head, domains, CR.LENGTHS, [row for _, _, row in got], reads_from=3) is None
        for b in blocks:
            assert b.arguments(SH) == LR.arguments(SH, b.requests, b.fails) and \
                b.output_bits(SH)[:32] == HC._int_bits(ACCEPT, 16) + HC._int_bits(b.failures, 16)
            read = LR.reads(opened.table(CR.LENGTHS), b.position)
            assert read.words == WR.words(b.lengths) and read.entry["value"] == f"{LINK}/lengths/{window}"
    opened, _, _, requests = signed[0]
    with pytest.raises(ValueError, match="one a slot stride"):
        LR.blocks(SH, opened, requests[:-1])
    with pytest.raises(ValueError, match="doesn't read the schedule's rows"):
        LR.blocks(LR.Shape(3, 512, 32, 8), opened, requests)
    with pytest.raises(ValueError, match="outside the table"):
        LR.reads(opened.table(CR.FAILS), T)
    with pytest.raises(ValueError, match="not a table"):
        CR.port(signed[0][1], "ends")


def _changed(requests, j: int, k: int, **fields):
    """The request rows with stride j's slot k changed."""
    row = list(requests[j])
    row[k] = row[k]._replace(**fields)
    return [*requests[:j], row, *requests[j + 1:]]


@pytest.mark.parametrize("j,k,field,value,names", [
    (0, 1, "outputs", 7, "n = 7 and the network certifier signed 8"),
    (0, 1, "outputs", 9, "n = 9 and the network certifier signed 8"),
    (5, 0, "outputs", 0, "n = 0 and the network certifier signed 2"),
    (0, 0, "prompt", 4, "bytes = 413 and the network certifier signed 402"),
    (0, 1, "prompt", 39, "bytes = 802 and the network certifier signed 813"),
    (5, 0, "prompt", 103, "bytes = 1522 and the network certifier signed 1481"),
])
def test_another_length_is_refused(signed, j, k, field, value, names):
    """A request row that states another output count or another prompt length than the released request's (a
    prompt an id longer or shorter, within a cell and into the next): the prover's builder refuses it by row, slot
    and field, and the block's own gates still accept and return a lengths row that is not the signed one, which the
    verifier's registered read refuses at that row."""
    opened, head, domains, requests = signed[0]
    lengths, fails = opened.table(CR.LENGTHS).rows, opened.table(CR.FAILS).rows
    other = _changed(requests, j, k, **{field: value})
    with pytest.raises(LR.Refused, match=f"row {j}: slot {k}: the request row gives {names}"):
        LR.blocks(SH, opened, other)
    got = _computed(other, fails)
    assert [verdict for verdict, _, _ in got] == [ACCEPT] * T
    assert [row != want for (_, _, row), want in zip(got, lengths)] == [i == j for i in range(T)]
    assert _refusal(opened, head, domains, CR.LENGTHS, [row for _, _, row in got]) == \
        f"registered reads: table row {j} does not open the registered root at position {j}"


def test_an_output_count_below_the_cells_sent_before_a_fail_is_refused(signed):
    """Where the connection failed the lengths row holds the cells sent, whatever the request row's count, so a count
    below them would open the signed root: the block's verdict refuses it."""
    opened, _, _, requests = signed[1]
    fails = opened.table(CR.FAILS).rows
    assert struct.unpack("<4H", fails[1]) == (1, 0, 1, 2)
    below = _changed(requests, 1, 1, outputs=1)
    with pytest.raises(LR.Refused, match="row 1: slot 1: the connection failed after 2 outputs and its request has 1"):
        LR.blocks(SH, opened, below)
    verdict, failures, row = _computed(below, fails)[1]
    assert (verdict, failures, row) == (0, 2, opened.table(CR.LENGTHS).rows[1])
    assert [verdict for verdict, _, _ in _computed(_changed(requests, 1, 1, outputs=2), fails)] == [ACCEPT] * T


@pytest.mark.parametrize("cells,names,accepts", [
    (((0, 0), (1, 2)), "slot 0: the request row gives n = 4 and the network certifier signed 0", True),
    (((1, 0), (0, 0)), "slot 1: the request row gives n = 6 and the network certifier signed 2", True),
    (((1, 0), (1, 3)), "slot 1: the request row gives n = 3 and the network certifier signed 2", True),
    (((1, 1), (1, 2)), "slot 0: the request row gives n = 1 and the network certifier signed 0", True),
    (((1, 0), (1, 7)), "slot 1: the connection failed after 7 outputs and its request has 6", False),
    (((1, 0), (2, 2)), "slot 1: the fails slot's class is 2, not 0 or 1", False),
    (((1, 0), (0, 2)), "slot 1: the fails slot has no failure and at = 2, not 0", False),
])
def test_another_fail_cell_is_refused(signed, cells, names, accepts):
    """A fails row with one cell changed, in the window's slot stride of two failed connections: a failure dropped, a
    failure moved, and a cell that is of no form. The verifier's read of the changed row as the fails table's row
    refuses it; the block on it refuses by its verdict, or returns a lengths row that is not the signed one; and the
    prover's builder, given an opening that holds it, refuses it by row and slot."""
    opened, head, domains, requests = signed[1]
    lengths, fails = opened.table(CR.LENGTHS).rows, opened.table(CR.FAILS).rows
    assert struct.unpack("<4H", fails[1]) == (1, 0, 1, 2)
    other = [fails[0], fails_row(*cells), *fails[2:]]
    assert _refusal(opened, head, domains, CR.FAILS, other) == \
        "registered reads: table row 1 does not open the registered root at position 1"
    verdict, _, row = _computed(requests, other)[1]
    assert (verdict == ACCEPT) is accepts and (not accepts or row != lengths[1])
    if accepts:
        got = [lengths[0], row, *lengths[2:]]
        assert "table row 1 does not open" in _refusal(opened, head, domains, CR.LENGTHS, got)
    table = dataclasses.replace(opened.table(CR.FAILS), rows=tuple(other))
    held = dataclasses.replace(opened, tables=(opened.table(CR.LENGTHS), table))
    with pytest.raises(LR.Refused, match=f"row 1: {names}"):
        LR.blocks(SH, held, requests)


def test_another_release_index_is_refused(signed):
    """A request row whose slot states another place in the release order, two requests of a slot stride in each
    other's slots, a request in another slot stride's row, and a row read at another slot stride's position: the
    builder refuses each by row, slot and field, the gates still accept and return a row that is not the signed one,
    and the verifier's read refuses it where it differs, or names the position its own program reads."""
    opened, head, domains, requests = signed[0]
    lengths, fails = opened.table(CR.LENGTHS).rows, opened.table(CR.FAILS).rows

    def refused(other, names, rows_differ):
        with pytest.raises(LR.Refused, match=names):
            LR.blocks(SH, opened, other)
        got = _computed(other, fails)
        assert [verdict for verdict, _, _ in got] == [ACCEPT] * T
        assert [i for i, ((_, _, row), want) in enumerate(zip(got, lengths)) if row != want] == rows_differ
        assert _refusal(opened, head, domains, CR.LENGTHS, [row for _, _, row in got]) == \
            f"registered reads: table row {rows_differ[0]} does not open the registered root at position {rows_differ[0]}"

    refused(_changed(requests, 0, 1, order=2), "row 0: slot 1: the request row gives N \\+ 1 = 3 and the network "
            "certifier signed 2", [0])
    refused(_changed(requests, 5, 0, order=1), "row 5: slot 0: the request row gives N \\+ 1 = 2 and the network "
            "certifier signed 3", [5])
    refused([requests[0][::-1], *requests[1:]], "row 0: slot 0: the request row gives N \\+ 1 = 2 and the network "
            "certifier signed 1", [0])
    moved = [requests[0], requests[5], *requests[2:5], requests[1], *requests[6:]]
    refused(moved, "row 1: slot 0: the request row gives N \\+ 1 = 3 and the network certifier signed 0", [1, 5])
    refused([requests[0][:1], *requests[1:]], "row 0: slot 1: the request row gives N \\+ 1 = 0 and the network "
            "certifier signed 2", [0])
    # the signed rows themselves, two slot strides' in each other's place: under their own salts and paths the
    # verifier's program reads other positions, and under the positions' they open nothing
    swapped = [0, 5, 2, 3, 4, 1, 6, 7]
    rows = [lengths[p] for p in swapped]
    assert _refusal(opened, head, domains, CR.LENGTHS, rows, swapped) == \
        "registered reads: unit 1 reads position 5, and the verifier's program reads 1"
    assert "table row 1 does not open" in _refusal(opened, head, domains, CR.LENGTHS, rows)
    # and no row of one table opens the other's root, nor another window's
    other_opened, other_head, other_domains, _ = signed[1]
    assert "does not open" in _refusal(other_opened, other_head, other_domains, CR.LENGTHS, lengths)
    with pytest.raises(ValueError, match="another domain"):
        WR.refusal(CR.port(head, CR.LENGTHS), domains[CR.FAILS], {}, [])
