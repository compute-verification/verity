"""The network certifier's content rows (`content_rows`): each of a connection's three rows is its class's size and
whole 16-bit words, a decoder gives back what its encoder was given, and bytes that are not a row of the form are
refused by the field that says so. A request row's room is the canonical id request's length at its class's prompt
ids, the end row has a fails slot's layout, and a response row's ids are bounded by the widest public size class.
A request's words are read from its bytes cell by cell, as the device reads a cell before it releases it, and a
words row is those three as u32 words."""

from __future__ import annotations

import random
import struct

import pytest

from verity.circuits.partition_v2 import UNIT_CLASSES
from verity.protocols.certifier.prover import lengths_row as LR
from verity.protocols.certifier.verifier import connection_rows as CR
from verity.protocols.certifier.verifier import content_rows as CT
from verity.protocols.certifier.verifier import warden_rows as WR
from verity.protocols.certifier.verifier.active import HEADER, id_request


def test_a_class_is_two_powers_of_two():
    """A class's size, of prompt ids or of ids released, is a power of two of at least 2; a response row's ids are
    at most 4,096, the most whose row fits the widest public size class behind its tie's verdict."""
    assert CT.MIN_SIZE == 2 and CT.most_ids() == 4096
    assert 16 + 8 * CT.response_bytes(4096) == 131_120 <= UNIT_CLASSES[-1] < 16 + 32 * (8192 + 1)
    for size in (2, 4, 1024, 4096):
        CT.check_ids(size)
        CT.check_size(size, "prompt ids")
    CT.check_size(1 << 17, "prompt ids")
    for size in (0, 1, 3, 12, -4, True, 2.0, "2"):
        with pytest.raises(ValueError, match="a power of two of at least 2"):
            CT.check_ids(size)
        with pytest.raises(ValueError, match="size of prompt ids is a power of two"):
            CT.check_size(size, "prompt ids")
    with pytest.raises(ValueError, match="at most 4096 ids, not O=8192.*fewer egress_cells or a lower max_tokens_cap"):
        CT.check_ids(8192)


def test_the_most_ids_follow_the_widest_class(monkeypatch):
    """The bound on a response row's ids is computed from the widest public size class, not stated beside it: at
    the 16-bit limit no row fits, and at a class of 2^20 bits a row holds 16,384 ids."""
    monkeypatch.setattr(CT, "UNIT_CLASS_MAX", 1 << 20)
    assert CT.most_ids() == 16_384 and CT.response_bytes(16_384) == 65_540
    monkeypatch.setattr(CT, "UNIT_CLASS_MAX", 16 + 32 * 3)
    assert CT.most_ids() == 2
    monkeypatch.setattr(CT, "UNIT_CLASS_MAX", 16)
    assert CT.most_ids() == 0
    with pytest.raises(ValueError, match="at most 0 ids"):
        CT.check_ids(2)


def test_every_row_is_whole_words_of_its_classes_size():
    """A response row is its count and O ids, an end row the four bytes of a fails slot, and a request row its count and
    its class's room to a whole word, so each is whole 16-bit words (`warden_rows.words` reads it)."""
    assert [CT.response_bytes(O) for O in (2, 16, 4096)] == [12, 68, 16_388]
    assert CT.END_BYTES == 4 and CT.END_SLOT.format == CR.SLOT[CR.FAILS].format
    for B, P in ((512, 2), (512, 64), (387, 4), (4096, 512)):
        room = CT.request_room(B, P)
        assert 0 <= CT.request_bytes(B, P) - 4 - room < 2
        for row in (CT.request_row(b"x" * room, B, P), CT.response_row([7] * P, P), CT.end_row(True, P)):
            assert len(WR.words(row)) * 2 == len(row)
    assert CT.request_bytes(512, 2) == 4 + CT.request_room(512, 2) + CT.request_room(512, 2) % 2


@pytest.mark.parametrize("B", (387, 505, 512, 4096))
def test_a_request_rows_room_is_the_id_requests_length(B):
    """A class of P prompt ids holds the canonical id request of P ids in B-byte cells, whatever its other fields,
    and so every id request of fewer: the room is the prover's closed form (`lengths_row.request_bytes`), which
    grows with the prompt."""
    rng = random.Random(B)
    for P in (2, 4, 32, 64, 256, 2048):
        room = CT.request_room(B, P)
        assert room == LR.request_bytes(B, P)
        longest = id_request(model="m" * 60, user="u" * 60, prompt=[rng.randrange(1 << 32) for _ in range(P)],
                             max_tokens=P, seed=B, temperature=0.5, top_p=0.25, frame_bytes=B, offset=0)
        assert len(longest) == room
        row = CT.request_row(longest, B, P)
        assert len(row) == CT.request_bytes(B, P) and CT.request_of(row, B, P) == longest
        for fewer in (1, P - 1):
            short = id_request(model="m", user="u", prompt=[5] * fewer, max_tokens=1, frame_bytes=B, offset=0)
            assert len(short) < room and CT.request_of(CT.request_row(short, B, P), B, P) == short
    with pytest.raises(ValueError, match="no room for an id request"):
        CT.request_room(40, 2)
    for P in (0, 1, 3):
        with pytest.raises(ValueError, match="size of prompt ids"):
            CT.request_room(B, P)


def test_a_request_row_is_its_count_its_bytes_and_zeros():
    """The request row is the request's length as a u32, its bytes, and zero bytes to the class's size; a request of
    no byte or past the class's room, a row of another size, a count of 0 or past the room, and a byte past the
    count are each refused."""
    B, P = 512, 4
    room, size = CT.request_room(B, P), CT.request_bytes(B, P)
    row = CT.request_row(b"abc", B, P)
    assert row == b"\x03\0\0\0abc" + bytes(size - 7) and CT.request_of(row, B, P) == b"abc"
    assert CT.request_of(CT.request_row(b"\0a\0", B, P), B, P) == b"\0a\0"
    for request in (b"", b"x" * (room + 1), "abc", None):
        with pytest.raises(ValueError, match=f"is 1 to {room} bytes"):
            CT.request_row(request, B, P)
    for bad, why in ((row[:-2], f"is {size} bytes, not {size - 2}"), (row + b"\0\0", f"is {size} bytes"),
                     (bytes(size), "bytes is 0, not 1 to"),
                     (struct.pack("<I", room + 1) + row[4:], f"bytes is {room + 1}"),
                     (row[:7] + b"\x01" + row[8:], "not zero past its 3 bytes"),
                     (row[:-1] + b"\x01", "not zero past its 3 bytes")):
        with pytest.raises(ValueError, match=why):
            CT.request_of(bad, B, P)


def test_a_response_row_is_its_count_and_its_ids():
    """The response row is the count of ids as a u32, the ids, and zero words to O: an id of 0 inside the count is
    an id, which the count alone tells from the zeros past it. More ids than the class holds, an id that is no u32,
    a row of another size, a count past O (the count no row has among them) and a word past the count are each
    refused."""
    assert CT.response_row([5, 0, 9], 4) == struct.pack("<5I", 3, 5, 0, 9, 0)
    assert CT.response_row([], 2) == bytes(12) and CT.response_of(bytes(12), 2) == ()
    rng = random.Random(4)
    for O in (2, 16, 256, 4096):
        for m in (0, 1, O - 1, O):
            ids = tuple(rng.choice((0, 1, 0xFFFF_FFFF, rng.randrange(1 << 32))) for _ in range(m))
            row = CT.response_row(ids, O)
            assert len(row) == CT.response_bytes(O) and CT.response_of(row, O) == ids
    assert CT.response_of(CT.response_row([0, 0], 4), 4) == (0, 0) != CT.response_of(CT.response_row([0], 4), 4)
    for ids in ([1, 2, 3], [1 << 32], [-1], [True], [1.0]):
        with pytest.raises(ValueError, match="at most 2 ids, each a u32"):
            CT.response_row(ids, 2)
    with pytest.raises(ValueError, match="at most 4096 ids"):
        CT.response_row([], 8192)
    row = CT.response_row([5, 6], 4)
    for bad, why in ((row[:-4], "is 20 bytes, not 16"), (row, "is 12 bytes, not 20")):
        with pytest.raises(ValueError, match=why):
            CT.response_of(bad, 4 if len(bad) == 16 else 2)
    for count in (5, 1 << 16, CT.NO_COUNT):
        with pytest.raises(ValueError, match=f"count is {count}, past its 4 ids"):
            CT.response_of(struct.pack("<I", count) + row[4:], 4)
    with pytest.raises(ValueError, match="not zero past its 2 ids"):
        CT.response_of(row[:-1] + b"\x01", 4)
    with pytest.raises(ValueError, match="not zero past its 1 ids"):
        CT.response_of(struct.pack("<I", 1) + row[4:], 4)


def test_an_end_row_is_how_the_connection_ended_and_its_cells():
    """A connection that ended in its `END` has the end row (0, the cells it released), and one that ended in a
    `FAIL` (1, the cells before it), in the fails table's slot layout: `at` counts the cells either way, where a
    fails slot counts them before a `FAIL` alone. A class past 1 and a row of another size are refused."""
    assert CT.end_row(False, 0) == bytes(4) and CT.end_row(False, 77) == struct.pack("<HH", 0, 77)
    assert CT.end_row(True, 0) == struct.pack("<HH", 1, 0) and CT.end_row(True, 0xFFFF) == b"\x01\0\xff\xff"
    assert CT.end_row(False, 0xFFFF) == b"\0\0\xff\xff" and CT.END_SLOT.format == CR.SLOT[CR.FAILS].format
    for failed in (False, True):
        for cells in (0, 1, 300, 0xFFFF):
            assert CT.end_of(CT.end_row(failed, cells)) == (failed, cells)
    for failed, cells in ((1, 0), (True, -1), (True, 1 << 16), (False, 1 << 16), (False, True), (None, 0)):
        with pytest.raises(ValueError, match="after a u16 of DATA cells"):
            CT.end_row(failed, cells)
    for bad, why in ((struct.pack("<HH", 2, 0), "class is 2, not 0 or 1"),
                     (struct.pack("<HH", 0x101, 3), "class is 257"),
                     (bytes(3), "is 4 bytes, not 3"), (bytes(8), "is 4 bytes, not 8")):
        with pytest.raises(ValueError, match=why):
            CT.end_of(bad)


@pytest.mark.parametrize("B", (387, 512, 4096))
def test_a_requests_words_are_its_max_tokens_zero_and_its_prompts_count_read_cell_by_cell(B):
    """The rule reads a canonical id request in the network certifier's cells whatever its other fields: its
    `max_tokens` at its place in the first cell, 0 for `ignore_eos`, which the form has no key for, and the count
    of its prompt's ids over its cells, from a prompt of one id to one of many cells. Bytes that are not such a
    request are refused, naming the first cell that doesn't read as the device reads it: a request cut short or
    with a byte after its end, one with another byte in its head or between two ids, and no byte at all."""
    rng = random.Random(B)
    room = B - HEADER.size
    for prompt in (1, 2, 3, 40, 100, 2048):
        for max_tokens in (0, 1, 8, 4096, 0xFFFF_FFFF):
            request = id_request(model="m" * rng.randrange(1, 60), user="u" * rng.randrange(1, 60),
                                 prompt=[rng.randrange(1 << 32) for _ in range(prompt)], max_tokens=max_tokens,
                                 seed=rng.randrange(1 << 32), temperature=0.5, top_p=0.25, frame_bytes=B, offset=0)
            assert CT.request_words(request, B) == (max_tokens, 0, prompt)
            assert CT.request_words(bytearray(request), B) == (max_tokens, 0, prompt)
    assert len(request) > 5 * room
    last = (len(request) - 1) // room
    for bad, cell in ((request[:-1], last), (request + b" ", last if len(request) % room else last + 1),
                      (request[:room], 0), (request[:2 * room + 7], 2), (b"x" + request[1:], 0),
                      (request[:room] + b"x" + request[room + 1:], 1), (bytes(40), 0)):
        with pytest.raises(ValueError, match=f"cell {cell} of the request's [0-9]+ is not that cell of a canonical id "
                                             f"request .*fails such a connection as malformed and releases nothing"):
            CT.request_words(bad, B)
    for bad in (b"", None, "request"):
        with pytest.raises(ValueError, match="a released request has a byte"):
            CT.request_words(bad, B)


def test_a_words_row_is_three_u32_words():
    """The words row is the request's `max_tokens`, its `ignore_eos` and the count of its prompt's ids, a
    little-endian u32 each: six 16-bit words, a field's low word first. A `max_tokens` or a count that is no u32, a
    prompt of no id, an `ignore_eos` that is neither 0 nor 1 and a row of another size are each refused."""
    assert CT.WORDS_BYTES == 12 and CT.WORDS_ROW.format == "<III"
    row = CT.words_row(0x0001_0002, 0, 0x0003_0004)
    assert row == b"\x02\0\x01\0" + bytes(4) + b"\x04\0\x03\0" and len(WR.words(row)) == 6
    assert CT.words_row(8, 0, 3) == struct.pack("<III", 8, 0, 3) and CT.words_of(CT.words_row(8, 1, 3)) == (8, 1, 3)
    for words in ((0, 0, 1), (0xFFFF_FFFF, 0, 0xFFFF_FFFF), (7, 1, 4096)):
        assert CT.words_of(CT.words_row(*words)) == words
    for bad in ((-1, 0, 1), (1 << 32, 0, 1), (8, 2, 1), (8, -1, 1), (8, 0, 0), (8, 0, 1 << 32), (8.0, 0, 1),
                (True, 0, 1), (8, False, 1), (8, 0, None)):
        with pytest.raises(ValueError, match="a words row is a u32 max_tokens, an ignore_eos of 0 or 1 and a u32 of "
                                             "at least 1 prompt id"):
            CT.words_row(*bad)
    for bad, why in ((row[:-1], "a words row is 12 bytes, not 11"), (row + b"\0", "a words row is 12 bytes, not 13"),
                     (struct.pack("<III", 8, 2, 3), "ignore_eos is 2, not 0 or 1"),
                     (struct.pack("<III", 8, 0x100, 3), "ignore_eos is 256, not 0 or 1"),
                     (struct.pack("<III", 8, 0, 0), "prompt has no id"), (bytes(12), "prompt has no id")):
        with pytest.raises(ValueError, match=why):
            CT.words_of(bad)
