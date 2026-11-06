"""The linear-claim argument against a prover that lies about g·M or γ·M: each lie meets every check the verifier can make
without columns, so only the columns can catch it."""
import random

from challenger import SeededChallenger, salts
from verity_lin import field as F, linear
from verity_lin.session import ProverSession, Reader, VerifierSession, ext_coins, word_bytes
from verity_lin.table import Shape, Table

SHAPE = Shape("test", 8, 16, 4)
QUERIES = 24


def _setup(seed):
    rng = random.Random(seed)
    table = Table(SHAPE, [[rng.randrange(-128, 128) for _ in range(SHAPE.width)] for _ in range(SHAPE.rows)],
                  salt=salts(seed))
    g = [tuple(rng.randrange(F.P) for _ in range(3)) for _ in range(SHAPE.rows)]
    e = [tuple(rng.randrange(F.P) for _ in range(3)) for _ in range(SHAPE.width)]
    true_w = [F.dot_f(g, [r[k] for r in table.rows]) for k in range(SHAPE.width)]
    return table, g, e, true_w


def _prover(s, table, w, bend_z=False):
    """The protocol's messages with a chosen w in place of g·M, and z off by one in its first coordinate if `bend_z`."""
    s.message("t/w", b"".join(F.ext_bytes(x) for x in w))
    gamma = ext_coins(s, SHAPE.rows)
    z = [F.dot_f(gamma, [r[k] for r in table.rows]) for k in range(SHAPE.width)]
    if bend_z:
        z[0] = F.add(z[0], F.ONE)
    s.message("t/z", b"".join(F.ext_bytes(x) for x in z))
    reader = Reader(s, word_bytes(SHAPE.n) * QUERIES)
    s.message("t/open", table.open([reader.uniform(SHAPE.n) for _ in range(QUERIES)]))


def _verdict(table, g, e, v, w, seed, bend_z=False):
    ch = SeededChallenger(seed)
    p = ProverSession(b"statement", ch)
    _prover(p, table, w, bend_z)
    s = VerifierSession(b"statement", p.messages, ch.log)
    try:
        linear.verify(s, SHAPE, table.root, g, e, v, "t", QUERIES)
        s.finish()
    except linear.Refused as r:
        return r.code
    return None


def test_the_true_claim_is_accepted():
    table, g, e, w = _setup(1)
    assert _verdict(table, g, e, F.dot(w, e), w, 2) is None


def test_a_false_claim_with_a_matching_w_is_caught_by_the_columns():
    table, g, e, w = _setup(1)
    v = F.add(F.dot(w, e), F.ONE)
    lie = [F.add(w[0], F.inv(e[0])), *w[1:]]
    assert F.dot(lie, e) == v
    assert _verdict(table, g, e, v, lie, 2) == "column-claim"


def test_the_honest_w_of_a_false_claim_is_refused_at_once():
    table, g, e, w = _setup(1)
    assert _verdict(table, g, e, F.add(F.dot(w, e), F.ONE), w, 2) == "claim"


def test_a_z_that_is_not_gamma_m_is_caught_by_the_columns():
    table, g, e, w = _setup(1)
    assert _verdict(table, g, e, F.dot(w, e), w, 2, bend_z=True) == "column-proximity"
