"""Sampled proofs alone, on a Program whose VUs are finer than its RUs (so interiors are opened)."""

import hashlib
from fractions import Fraction

import pytest

from verity.evaluation import evaluate_region
from verity.ir import Array, Program, Value, bind, composite, primitive
from verity_sampled_proofs import INTERIORS, TRANSCRIPT, Record, Rejected, SampledProofs, interiors_digest

V16 = Value(16)
R = 32


@primitive("SpTestMulAdd", 1, [("a", V16), ("b", V16)], V16, conformance="exact-model-tested")
def MulAdd(a, b):
    return (a * b + a) & 0xFFFF


@composite("SpTestStep", 1, [], lambda S: ((("a", V16), ("b", V16)), V16))
def Step(B, S, a, b):
    return B.call(MulAdd, a, b)


@composite("SpTestChain", 1, [], lambda S: ((("x", Array(2, V16)),), V16))
def Chain(B, S, x):
    return B.call(Step, B.call(Step, x[0], x[1]), x[1])


@composite("SpTestBatch", 1, ["R"], lambda S: ((("X", Array(S.R, Array(2, V16))),), Array(S.R, V16)))
def Batch(B, S, X):
    return B.batch(Chain, X, axes=(0,))


class Log(dict):
    def register(self, scope, digest):
        self[scope] = (len(self) + 1, digest)
        return len(self)

    def __call__(self, scope):
        return self[scope]


def beacon(t):
    return hashlib.sha256(b"beacon" + t.to_bytes(8, "big")).digest()


@pytest.fixture(scope="module")
def setup():
    program = Program(bind(Batch, R=R))
    sp = SampledProofs(program, ru="//SpTestChain", vu="//SpTestStep", p=Fraction(1, 2), k=1, delta=0.01)
    refs = program.input_coll(0).refs
    inputs = {refs[i][2]: (7 * i + 3) & 0xFFFF for i in range(refs.count)}
    values = dict(inputs)
    evaluate_region(program.circuit, program.circuit[0:program.circuit.size], values)
    return sp, inputs, values


def prove(sp, values, log):
    boundary = sp.commit_boundary(values)
    rus = sp.ru_sample(beacon(log.register(TRANSCRIPT, boundary.commitment.root)), boundary.commitment.root)
    interiors = sp.commit_interiors(values, boundary, rus)
    digest = interiors_digest({i: t.commitment for i, t in interiors.items()})
    return sp.open(boundary, interiors, sp.vu_sample(beacon(log.register(INTERIORS, digest)), digest, rus))


def test_honest_transcript_yields_the_configured_allowance(setup):
    sp, inputs, values = setup
    log = Log()
    record = prove(sp, values, log)
    assert any(g not in sp.boundary for g in record.openings)  # the chain's middle value is an interior
    a = sp.verify(record, beacon=beacon, log=log, prescribed=inputs)
    assert (a.n_r, a.n_v, a.p, a.k) == (R, 2, Fraction(1, 2), 1)


def test_corrupted_interiors_are_caught(setup):
    sp, inputs, values = setup
    bad = dict(values)
    for first, _ in sp.vus:
        bad[first.returns[0]] ^= 1  # an interior in every RU; the second step now disagrees with it
    log = Log()
    with pytest.raises(Rejected):
        sp.verify(prove(sp, bad, log), beacon=beacon, log=log, prescribed=inputs)


def test_inputs_must_match_what_is_prescribed(setup):
    sp, inputs, values = setup
    log = Log()
    record = prove(sp, values, log)
    with pytest.raises(Rejected, match="prescribed"):
        sp.verify(record, beacon=beacon, log=log, prescribed={g: v ^ 1 for g, v in inputs.items()})


def test_the_registered_root_is_the_only_one_accepted(setup):
    sp, inputs, values = setup
    log = Log()
    record = prove(sp, values, log)
    log[TRANSCRIPT] = (log[TRANSCRIPT][0], b"\0" * 32)
    with pytest.raises(Rejected, match="registered"):
        sp.verify(record, beacon=beacon, log=log, prescribed=inputs)


def test_reader_authenticates_boundary_values_only(setup):
    sp, _, values = setup
    tree = sp.commit_boundary(values)
    read = sp.reader(Record(tree.commitment, {}, {}))
    g = sp.boundary[-1]
    assert read(g, tree.open(g)) == values[g]
    assert read(sp.boundary[-2], tree.open(g)) is None
