"""Sampled proofs alone, over two linked steps of a Program whose VUs are finer than its RUs."""

import hashlib
from fractions import Fraction

import pytest

from verity.evaluation import evaluate_region
from verity.ir import Array, Program, Value, bind, composite, primitive
from verity_sampled_proofs import Prover, Rejected, SampledProofs

V16 = Value(16)
R = 16


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


PROGRAM = Program(bind(Batch, R=R))
C = PROGRAM.circuit
INPUTS = [PROGRAM.input_coll(0).refs[i][2] for i in range(2 * R)]
OUTPUTS = C.root_outputs()


class Log(dict):
    def register(self, scope, digest, now):
        self[scope] = (now, digest)

    def __call__(self, scope):
        return self[scope]


def beacon(t):
    i = int(t // 1.0) + 1
    return float(i), hashlib.sha256(b"beacon" + i.to_bytes(8, "big")).digest()


def run(sample, tamper=None):
    """Two steps: step s1's first operand of each chain is s0's output (linked), its second is prescribed."""
    sp = SampledProofs(ru="//SpTestChain", vu=("//SpTestStep",), p=Fraction(1, 2), k=1, delta=0.01, sample=sample)
    log = Log()
    prover = Prover(sp, beacon=beacon, register=log.register)
    prescribed, links, peak = {"prompt": {}}, {"s1": {}}, 0
    x = {g: (7 * i + 3) & 0xFFFF for i, g in enumerate(INPUTS)}
    for n, step in enumerate(("s0", "s1")):
        values = dict(x)
        evaluate_region(C, C[0:C.size], values)
        if tamper:
            tamper(sp, values)
        prover.commit(step, PROGRAM, values, now=2.0 * n)
        peak = max(peak, prover.retained_words)
        prover.poll(2.0 * n + 1.5)
        prescribed["prompt"][step] = {g: v for i, (g, v) in enumerate(x.items()) if n == 0 or i % 2}
        if n == 0:
            x = {g: (values[OUTPUTS[i // 2]] if i % 2 == 0 else v) for i, (g, v) in enumerate(x.items())}
            links["s1"] = {g: ("s0", OUTPUTS[i // 2]) for i, g in enumerate(INPUTS) if i % 2 == 0}
    evidence = prover.close(now=4.0)
    return sp, evidence, log, prescribed, links, peak


def verify(sp, evidence, log, prescribed, links):
    return sp.verify(evidence, beacon=beacon, log=log, programs={"s0": PROGRAM, "s1": PROGRAM},
                     prescribed=prescribed, links=links)


@pytest.mark.parametrize("sample", ["audit", "serving"])
def test_honest_epoch_yields_a_profile(sample):
    sp, evidence, *rest, peak = run(sample)
    assert any(evidence["s0"].openings)  # the chain's middle value is an interior
    profile = verify(sp, evidence, *rest)
    assert (profile.n_r, profile.n_v, profile.linked, profile.prescribed) == (2 * R, 2, True, {"prompt"})


def test_sampling_during_serving_retains_less():
    assert run("serving")[-1] < run("audit")[-1]


def test_corrupted_interiors_are_caught():
    def tamper(sp, values):
        for first, _ in sp.layout(PROGRAM).vus:
            values[first.returns[0]] ^= 1  # an interior; the second step now disagrees with it
    with pytest.raises(Rejected, match="replay"):
        verify(*run("audit", tamper)[:-1])


def test_every_input_must_be_prescribed_or_linked():
    sp, evidence, log, prescribed, links, _ = run("audit")
    with pytest.raises(Rejected, match="neither prescribed nor linked"):
        verify(sp, evidence, log, prescribed, {"s1": {}})
    bad = {"s1": {g: (s, (o + 1) % C.size) for g, (s, o) in links["s1"].items()}}
    with pytest.raises(Rejected, match="neither prescribed nor linked"):
        verify(sp, evidence, log, prescribed, bad)


def test_the_registered_root_is_the_only_one_accepted():
    sp, evidence, log, prescribed, links, _ = run("audit")
    log["sp/step/s1"] = (log["sp/step/s1"][0], b"\0" * 32)
    with pytest.raises(Rejected, match="registered"):
        verify(sp, evidence, log, prescribed, links)
