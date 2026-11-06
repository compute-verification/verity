"""Lin's matrix–vector statement end to end at test size: the honest session is accepted, and each way a prover can be
wrong or a record tampered with is refused by name."""
import dataclasses
import random

import pytest

from challenger import SeededChallenger, salts
from verity_lin import field as F, matvec
from verity_lin.session import Coin, Message
from verity_lin.table import Table

ROWS, WIDTH, OUT = 16, 32, 4
OFFSETS = (0, 4, 8, 12, 4)
PARAMS = matvec.Params(blowup=4, queries=12)


def _int8(rng, n):
    return [rng.randrange(-128, 128) for _ in range(n)]


def _matvec(w, a):
    return [[sum(w[o + j][k] * x[k] for k in range(WIDTH)) for j in range(OUT)] for o, x in zip(OFFSETS, a)]


def _case(seed=0, wrong_y=None, wrong_a=None):
    rng = random.Random(seed)
    w = [_int8(rng, WIDTH) for _ in range(ROWS)]
    a = [_int8(rng, WIDTH) for _ in OFFSETS]
    y = _matvec(w, a)
    if wrong_y is not None:
        y[wrong_y[0]][wrong_y[1]] += 1
    if wrong_a is not None:
        a[wrong_a[0]][wrong_a[1]] ^= 1
    return _statement(Table(_shapes()[0], w, salt=salts(seed + 1)), a, y, seed)


def _statement(wt, a, y, seed):
    st = matvec.Statement(ROWS, WIDTH, OUT, OFFSETS, wt.root, b"\0" * 64, b"\0" * 64, PARAMS)
    at, yt = matvec.input_tables(st, a, y, salt=salts(seed + 2))
    return dataclasses.replace(st, a_root=at.root, y_root=yt.root), wt, at, yt


def _shapes():
    return matvec.Statement(ROWS, WIDTH, OUT, OFFSETS, b"", b"", b"", PARAMS).shapes()


def _run(st, wt, at, yt, seed=7):
    ch = SeededChallenger(seed)
    return matvec.prove(st, wt, at, yt, ch), ch.log


def test_an_honest_session_is_accepted():
    st, *tables = _case()
    messages, log = _run(st, *tables)
    assert matvec.verify(st, messages, log) is None


def test_a_wrong_output_is_refused():
    st, *tables = _case(wrong_y=(2, 1))
    messages, log = _run(st, *tables)
    assert matvec.verify(st, messages, log) == "sumcheck-round"


def test_a_wrong_input_is_refused():
    st, *tables = _case(wrong_a=(1, 3))
    messages, log = _run(st, *tables)
    assert matvec.verify(st, messages, log) == "sumcheck-round"


class _Posing(Table):
    """Another weights table that answers to the registered root."""

    def __init__(self, other: Table, root: bytes) -> None:
        self.__dict__.update(other.__dict__)
        self._posed = root

    @property
    def root(self) -> bytes:
        return self._posed


def test_outputs_of_other_weights_are_refused_at_the_weights_opening():
    st, wt, _, _ = _case()
    rng = random.Random(99)
    w2 = [_int8(rng, WIDTH) for _ in range(ROWS)]
    a = [_int8(rng, WIDTH) for _ in OFFSETS]
    other = Table(wt.shape, w2, salt=salts(5))
    st, _, at, yt = _statement(wt, a, _matvec(w2, a), 3)
    messages, log = _run(st, _Posing(other, wt.root), at, yt)
    assert matvec.verify(st, messages, log) == "column-path"


def test_a_tampered_column_is_refused():
    st, *tables = _case()
    messages, log = _run(st, *tables)
    assert messages[-1].label == "weights/open"
    data = bytearray(messages[-1].data)
    data[4 + 8 + 64 + 3] ^= 1
    messages[-1] = Message(messages[-1].label, bytes(data))
    assert matvec.verify(st, messages, log) == "column-path"


def test_columns_other_than_the_drawn_ones_are_refused():
    """A valid multiproof of columns the coin didn't draw (the session's last message, so no coin follows it)."""
    st, wt, at, yt = _case()
    messages, log = _run(st, wt, at, yt)
    messages[-1] = Message(messages[-1].label, wt.open(range(PARAMS.queries)))
    assert matvec.verify(st, messages, log) == "column-position"


def test_ends_whose_product_is_not_the_last_claim_are_refused():
    st, *tables = _case()
    messages, log = _run(st, *tables)
    i = next(i for i, m in enumerate(messages) if m.label == "ends")
    g_at, a_at = F.ext_from_bytes(messages[i].data)
    messages[i] = Message("ends", F.ext_bytes(F.add(g_at, F.ONE)) + F.ext_bytes(a_at))
    assert matvec.verify(st, messages, log) == "sumcheck-final"


def test_a_message_changed_after_its_coin_is_refused():
    """A first round moved so its check still passes (v0 + 1, v1 - 1) differs from what the coin after it followed."""
    st, *tables = _case()
    messages, log = _run(st, *tables)
    i = next(i for i, m in enumerate(messages) if m.label == "sumcheck/round")
    v0, v1, v2 = F.ext_from_bytes(messages[i].data)
    moved = F.add(v0, F.ONE), F.sub(v1, F.ONE), v2
    messages[i] = Message(messages[i].label, b"".join(F.ext_bytes(v) for v in moved))
    assert matvec.verify(st, messages, log) == "coin-order"


def test_a_coin_flipped_before_its_message_is_refused():
    st, *tables = _case()
    messages, log = _run(st, *tables)
    log[3] = Coin(log[2].after, log[3].data)
    assert matvec.verify(st, messages, log) == "coin-order"


def test_a_dropped_message_is_refused():
    st, *tables = _case()
    messages, log = _run(st, *tables)
    assert matvec.verify(st, messages[:-1], log) == "missing"


def test_a_statement_with_other_offsets_is_refused():
    st, *tables = _case()
    messages, log = _run(st, *tables)
    other = dataclasses.replace(st, offsets=(0, 4, 8, 12, 8))
    assert matvec.verify(other, messages, log) == "coin-order"


@pytest.mark.parametrize("width", [1 << 10, 1 << 16])
def test_the_default_queries_give_100_bits_per_linear_claim(width):
    assert matvec.queries(100, width, 4) == matvec.Params().queries == 241
