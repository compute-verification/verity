"""The power certifier's proof block (`power_check`) on four block spans of one channel: each part against its reference
on honest spans and on spans wrong in one input (a tampered certificate, a tampered energy, an over-threshold span),
the threshold's rounding against the prover, and the whole Definition against its reference."""

import random
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation.bits import evaluate_bits
from verity.ml.commitments.gates.frame_v3 import leaf_ref, node_ref
from verity.ml.commitments.identity import tagged_sha512
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as SCHEMA
from verity.core.protocols.certifier import head_check as HC
from verity.core.protocols.certifier import power_check as PC
from verity.core.protocols.certifier.power import (ENERGY_BITS, UJ, Meter, PowerHead, PowerStatus, Threshold,
                                                   counts_row, passes_quantized)

CHANNEL = "outlet-3"
DEPTH = 2
INTERVAL_MS = 1000
METER = Meter("stand-in/nebius-bmc", rate_hz=10, accuracy_ppm=10_000, offset_mw=2_000, stand_in=True)
#: test readings, not the card's: declared MACs at dfa1's rand floor, a compression and a forming unit at round values
THRESHOLD = Threshold(
    reading="test", floors=(Fraction(684, 10 ** 15), Fraction(2, 10 ** 9), Fraction(1, 10 ** 9)),
    idle_w=Fraction(808, 10), hidden_j=Fraction(173, 10 ** 15), capacity=Fraction(989 * 10 ** 12), bar=Fraction(3, 100),
    offset_w=Fraction(2), gain_error=Fraction(1, 100))
COUNTS = [(500 * 10 ** 12, 10 ** 9, 10 ** 8), (600 * 10 ** 12, 2 * 10 ** 9, 0), (0, 0, 0), (871 * 10 ** 12, 1, 2)]


def _tree(domain: bytes, values: list[bytes]) -> tuple[bytes, list[tuple[bytes, ...]]]:
    """The frame-v3-sha512 root over hiding values at positions 0 … 2^DEPTH − 1, and each position's path."""
    levels = [[leaf_ref(domain, SCHEMA, v, x) for x, v in enumerate(values)]]
    for k in range(DEPTH):
        below = levels[-1]
        levels.append([node_ref(domain, k, i, below[2 * i], below[2 * i + 1]) for i in range(len(below) // 2)])
    paths = [tuple(levels[k][(x >> k) ^ 1] for k in range(DEPTH)) for x in range(len(values))]
    return levels[-1][0], paths


def _honest_energy(counts: tuple[int, ...], margin_uj: int = 0) -> int:
    """An honest span's energy in µJ: idle and the counted work at their floors, plus `margin_uj`."""
    t = THRESHOLD
    joules = t.idle_w * Fraction(INTERVAL_MS, 1000) + sum(f * n for f, n in zip(t.floors, counts))
    return int(joules * UJ) + margin_uj


def _spans(heads: list[PowerHead], rows: list[bytes], seed: int = 0) -> list[PC.Instance]:
    """Each span's instance: the heads and the counts rows committed under hiding salts in their own trees, and
    each head's certificate's signed digest under a certifier salt."""
    rnd = random.Random(seed)
    salts = [rnd.randbytes(HC.SALT.leaves // 8) for _ in heads]
    counts_salts = [rnd.randbytes(HC.SALT.leaves // 8) for _ in rows]
    salts_c = [rnd.randbytes(64) for _ in heads]
    power_domain, counts_domain = tagged_sha512("test/power", b""), tagged_sha512("test/counts", b"")
    power, paths = _tree(power_domain, [HC.hiding_value(h.encode(), s) for h, s in zip(heads, salts)])
    counts_root, counts_paths = _tree(counts_domain, [HC.hiding_value(r, s) for r, s in zip(rows, counts_salts)])
    out = []
    for i, h in enumerate(heads):
        floors, gain, slack = THRESHOLD.quantized(INTERVAL_MS)
        out.append(PC.Instance(
            power=power, power_domain=power_domain, power_pos=i, channel=CHANNEL.encode(), interval=h.interval,
            interval_ms=INTERVAL_MS, parameters=METER.digest(), counts_root=counts_root, counts_domain=counts_domain,
            counts_pos=i, signed=h.signed_digest(salts_c[i]), floors=floors, gain=gain, slack=slack, head=h.encode(),
            salt=salts[i], path=paths[i], counts=rows[i], counts_salt=counts_salts[i], counts_path=counts_paths[i],
            salt_c=salts_c[i]))
    return out


def _head(i: int, energy_uj: int, status: PowerStatus = PowerStatus.COMPLETE) -> PowerHead:
    return PowerHead(CHANNEL, 100 + i, INTERVAL_MS, status, METER.digest(), energy_uj, samples=10)


def _honest(margins=(0, 1_000_000, 0, 5_000_000)) -> list[PC.Instance]:
    return _spans([_head(i, _honest_energy(c, m)) for i, (c, m) in enumerate(zip(COUNTS, margins))],
                   [counts_row(c) for c in COUNTS])


@pytest.fixture(scope="module")
def cases():
    """Instances by name: (instance, the part that refuses it, or None)."""
    honest = _honest()
    out = {f"honest {i}": (inst, None) for i, inst in enumerate(honest)}
    h0, h1 = honest[0], honest[1]
    over_budget = _honest_energy(COUNTS[0]) + 2 * int(THRESHOLD.slack_j(INTERVAL_MS) * UJ)
    over = _spans([_head(0, over_budget)] + [_head(i, _honest_energy(c)) for i, c in enumerate(COUNTS[1:], 1)],
                   [counts_row(c) for c in COUNTS])[0]
    tampered_head = _head(0, _honest_energy(COUNTS[0]) - 1_000_000)
    tampered = _spans([tampered_head] + [_head(i, _honest_energy(c)) for i, c in enumerate(COUNTS[1:], 1)],
                       [counts_row(c) for c in COUNTS])[0]
    gap = _spans([_head(0, _honest_energy(COUNTS[0]), PowerStatus.SAMPLE_GAP)]
                  + [_head(i, _honest_energy(c)) for i, c in enumerate(COUNTS[1:], 1)], [counts_row(c) for c in COUNTS])[0]
    return out | {
        "an over-threshold span": (over, "threshold"),
        "a tampered energy under the honest certificate": (replace(tampered, signed=h0.signed), "signed"),
        "a tampered certificate": (replace(h0, signed=h1.signed), "signed"),
        "another certifier salt": (replace(h0, salt_c=h1.salt_c), "signed"),
        "a certificate that signs the unsalted record digest": (
            replace(h0, signed=tagged_sha512(PC.RECORD_TAG, h0.head)), "signed"),
        "another interval": (replace(h0, interval=101), "fields"),
        "another channel": (replace(h0, channel=b"outlet-4"), "fields"),
        "another meter": (replace(h0, parameters=bytes(64)), "fields"),
        "another length": (replace(h0, interval_ms=999), "fields"),
        "a sample gap": (gap, "fields"),
        "another head salt": (replace(h0, salt=h1.salt), "opening"),
        "another head position": (replace(h0, power_pos=1), "opening"),
        "another counts salt": (replace(h0, counts_salt=h1.counts_salt), "counts opening"),
        "another span's counts": (replace(h0, counts=h1.counts, counts_salt=h1.counts_salt), "counts opening"),
    }


def _args(inst):
    return {n: np.asarray([bits], dtype=np.uint8) for (n, _), bits in zip(inst.unit.signature()[0], PC.arguments(inst))}


def _out(fn, *args) -> bool:
    return bool(np.asarray(evaluate_bits(fn, *args)).reshape(-1)[0])


def _parts(inst):
    """The fields' and the threshold's circuits on the instance, each evaluated alone."""
    sh, a = PC.Shape(len(inst.channel)), _args(inst)
    head = a["head"]
    return {"fields": _out(bind(PC.PowerHeadFields, L=sh.L), head[:, :8 * sh.E], a["channel"], a["interval"],
                           a["interval_ms"], a["parameters"]),
            "threshold": _out(bind(PC.PowerThreshold, J=len(inst.floors)), head[:, 8 * sh.E:8 * sh.N], a["counts"],
                              a["floors"], a["gain"], a["slack"])}


def test_each_part_is_its_reference_and_each_wrong_input_is_refused_where_it_belongs(cases):
    for name, (inst, refused) in cases.items():
        want = PC.reference_parts(inst)
        assert _parts(inst) == {k: want[k] for k in ("fields", "threshold")}, name
        assert [k for k, ok in want.items() if not ok] == ([] if refused is None else [refused]), name


def test_the_threshold_circuit_decides_the_boundary_exactly():
    """At the largest energy that passes and one µJ above it, and at counts and floors at their widths' extremes."""
    floors, gain, slack = THRESHOLD.quantized(INTERVAL_MS)
    J = len(floors)
    fn = bind(PC.PowerThreshold, J=J)
    for counts in [COUNTS[0], (0, 0, 0), ((1 << 64) - 1,) * J]:
        rhs = sum(f * n for f, n in zip(floors, counts)) + slack
        top = min(rhs // (gain << (ENERGY_BITS - 20)), (1 << 64) - 1)
        for energy in {top, min(top + 1, (1 << 64) - 1), 0}:
            want = passes_quantized(energy, counts, floors, gain, slack)
            args = [np.asarray([HC._bytes_bits(energy.to_bytes(8, "big"))], dtype=np.uint8),
                    np.asarray([HC._bytes_bits(counts_row(counts))], dtype=np.uint8),
                    np.asarray([[b for f in floors for b in HC._int_bits(f, 48)]], dtype=np.uint8),
                    np.asarray([HC._int_bits(gain, 24)], dtype=np.uint8),
                    np.asarray([HC._int_bits(slack, 128)], dtype=np.uint8)]
            assert _out(fn, *args) == want, (counts, energy)
        assert passes_quantized(top, counts, floors, gain, slack) or top == (1 << 64) - 1
        assert top == (1 << 64) - 1 or not passes_quantized(top + 1, counts, floors, gain, slack)


def test_the_quantized_threshold_never_passes_an_interval_the_exact_one_refuses():
    """Rounding is against the prover: over random spans near the boundary, a quantized pass is an exact pass. The
    gain's rounding up moves the boundary by under 10⁻⁶ of the energy and the floors' down by under 2⁻⁴⁰ µJ a count, so
    energies are drawn within 3·10⁻⁶ of it."""
    rnd = random.Random(7)
    t = THRESHOLD
    verdicts = set()
    for _ in range(2000):
        counts = (rnd.randrange(10 ** 15), rnd.randrange(10 ** 10), rnd.randrange(10 ** 10))
        budget = sum(f * n for f, n in zip(t.floors, counts)) + t.slack_j(INTERVAL_MS)
        top = int(budget / (1 - t.gain_error) * UJ)
        energy = top + rnd.randrange(-3 * top // 10 ** 6 - 10, 10)
        passed = t.passes(_head(0, energy), counts)
        verdicts.add(passed)
        if passed:
            assert (1 - t.gain_error) * Fraction(energy, UJ) <= budget
    assert verdicts == {True, False}


def test_a_head_of_another_length_is_no_instance_of_the_unit(cases):
    inst, _ = cases["honest 0"]
    with pytest.raises(ValueError, match="the head is 98 bytes"):
        PC.reference(replace(inst, head=inst.head[:-1]))
    with pytest.raises(ValueError, match="argument head has"):
        PC.arguments(replace(inst, head=inst.head[:-1]))


def test_a_stand_in_meter_says_so():
    with pytest.raises(ValueError, match="stand-in"):
        Meter("stand-in/nebius-bmc", 10, 10_000, 2_000, stand_in=False)
    with pytest.raises(ValueError, match="stand-in"):
        Meter("pdu/outlet-3", 10, 10_000, 2_000, stand_in=True)
    assert Meter("pdu/outlet-3", 10, 10_000, 2_000, stand_in=False).digest() != METER.digest()


@pytest.mark.slow
def test_the_power_check_is_its_reference(cases):
    """The whole Definition on every case against its reference, and its ANDs by part. What the parts' tests can't
    reach is the wiring: the head's and the counts' bytes into their openings at their positions, the head's energy
    into the threshold, and the head's record digest and the certifier's salt into the signed digest. Building the
    commitment gadgets' Boolean bodies takes about a minute, once."""
    insts = [i for i, _ in cases.values()]
    assert PC.evaluate(*insts) == [int(PC.reference(i)) for i in insts]
    size = PC.size(PC.unit(len(CHANNEL), 3, DEPTH, DEPTH))
    assert size["other"] == 4
    assert size["threshold"] < 50_000 < size["openings"]
