"""The power certifier's issuer on the device's channels: a span's readings integrated against the prover, its
certificate verifying on the host only for its own outlet, span, key, session, head and salt, a refused signer leaving
the span without one, and an issued span feeding `power_check`'s reference."""

from dataclasses import replace

import pytest

import signers
from signers import OTHER_SESSION, SESSION
from test_power_certifier_check import COUNTS, DEPTH, THRESHOLD, _honest_energy, _tree

from verity.ml.commitments.identity import tagged_sha512
from verity.core.protocols.certifier import head_check as HC
from verity.core.protocols.certifier import power_check as PC
from verity.core.protocols.certifier.certificate import Direction, Issuer, Link, channel_message
from verity.core.protocols.certifier.power import (CertifiedPower, Meter, Outlet, PowerIssuer, PowerStatus, counts_row,
                                                   integrate)

METER = Meter("stand-in/nebius-dcgm", rate_hz=10, accuracy_ppm=10_000, offset_mw=2_000, stand_in=True)
SPAN_MS = 1000


def _flat(mw: int, n: int = 10) -> list[tuple[int, int]]:
    return [(100 * k + 50, mw) for k in range(n)]


def _coins(seed: int):
    def coins(n: int) -> bytes:
        nonlocal seed
        seed += 1
        return tagged_sha512("test/coins", seed.to_bytes(8, "big"))[:n]
    return coins


@pytest.fixture
def issuer(tmp_path):
    s = signers.signer(tmp_path)
    return PowerIssuer(Issuer(s, SESSION, coins=_coins(0)), METER), s.public_key()


def test_integration_is_the_upper_sum_in_microjoules_and_flags_gaps():
    assert integrate(_flat(500_000), SPAN_MS, METER) == (PowerStatus.COMPLETE, 500_000 * SPAN_MS, 10)
    assert integrate([(0, 100), (200, 300), (400, 200)], 400, Meter("m", 5, 0, 0, False)) == (
        PowerStatus.COMPLETE, 300 * 200 + 300 * 200, 3)
    assert integrate([], SPAN_MS, METER) == (PowerStatus.METER_UNAVAILABLE, 0, 0)
    gapped = [s for s in _flat(1) if s[0] not in (350, 450)]
    assert integrate(gapped, SPAN_MS, METER)[0] is PowerStatus.SAMPLE_GAP
    assert integrate([(800, 1)], SPAN_MS, METER)[0] is PowerStatus.SAMPLE_GAP
    for bad in ([(5, 1), (5, 1)], [(6, 1), (5, 1)], [(SPAN_MS + 1, 1)], [(5, -1)]):
        with pytest.raises(ValueError, match="readings"):
            integrate(bad, SPAN_MS, METER)


def test_an_issued_certificate_verifies_only_for_its_outlet_span_key_session_head_and_salt(issuer, tmp_path):
    power, key = issuer
    c = power.certify("outlet-3", 7, SPAN_MS, _flat(500_000))
    assert c is not None and c.verify(key, SESSION) and c.head.parameters == METER.digest()
    assert c.certificate.channel == Outlet("outlet-3") and c.certificate.window == 7
    (tmp_path / "other").mkdir()
    other_key = signers.signer(tmp_path / "other", b"other").public_key()
    moved = {
        "another session": (c, OTHER_SESSION),
        "another key": (c, None),
        "another span": (replace(c, certificate=replace(c.certificate, window=8)), SESSION),
        "another outlet": (replace(c, certificate=replace(c.certificate, channel=Outlet("outlet-4"))), SESSION),
        "a link": (replace(c, certificate=replace(c.certificate, channel=Link("outlet-3", Direction.EGRESS))), SESSION),
        "another energy": (replace(c, head=replace(c.head, energy_uj=c.head.energy_uj - 1)), SESSION),
        "another salt": (replace(c, salt=bytes(64)), SESSION),
        "a short salt": (replace(c, salt=c.salt[:-1]), SESSION),
        "a tampered signature": (replace(c, certificate=replace(c.certificate, signature=bytes(
            len(c.certificate.signature)))), SESSION),
    }
    for name, (case, session) in moved.items():
        assert not case.verify(other_key if session is None else key, session or SESSION), name


def test_a_power_message_is_never_a_link_message(issuer):
    power, _ = issuer
    key_id, digest = power.device.signer.key_id, bytes(64)
    power_message = channel_message(Outlet("l"), key_id, SESSION, 1, digest)
    assert all(power_message != channel_message(Link("l", d), key_id, SESSION, 1, digest) for d in Direction)
    with pytest.raises(ValueError, match="64 bytes"):
        channel_message(Outlet("l"), key_id, SESSION, 1, bytes(63))


def test_a_refused_signer_leaves_the_span_without_a_certificate(tmp_path):
    first = signers.signer(tmp_path)
    assert PowerIssuer(Issuer(first, SESSION), METER).certify("outlet-3", 6, SPAN_MS, _flat(1)) is not None
    held = PowerIssuer(Issuer(signers.LmsSigner(first.key, first.state), OTHER_SESSION), METER)
    assert held.certify("outlet-3", 7, SPAN_MS, _flat(1)) is None
    assert held.device.refused == [("outlet-3", 7)] and signers.used(first) == [0]


def test_an_issued_span_feeds_the_power_check(issuer):
    power, key = issuer
    spans = [power.certify("outlet-3", 100 + i, SPAN_MS, _flat(_honest_energy(c) // SPAN_MS + 1))
             for i, c in enumerate(COUNTS)]
    assert all(isinstance(c, CertifiedPower) and c.verify(key, SESSION) for c in spans)
    salts = [bytes([i]) * (HC.SALT.leaves // 8) for i in range(len(spans))]
    power_domain, counts_domain = tagged_sha512("test/power", b""), tagged_sha512("test/counts", b"")
    rows = [counts_row(c) for c in COUNTS]
    root, paths = _tree(power_domain, [HC.hiding_value(c.head.encode(), s) for c, s in zip(spans, salts)])
    counts_root, counts_paths = _tree(counts_domain, [HC.hiding_value(r, s) for r, s in zip(rows, salts)])
    floors, gain, slack = THRESHOLD.quantized(SPAN_MS)
    for i, c in enumerate(spans):
        inst = PC.Instance(
            power=root, power_domain=power_domain, power_pos=i, channel=b"outlet-3", interval=c.head.interval,
            interval_ms=SPAN_MS, parameters=METER.digest(), counts_root=counts_root, counts_domain=counts_domain,
            counts_pos=i, signed=c.certificate.digest, floors=floors, gain=gain, slack=slack, head=c.head.encode(),
            salt=salts[i], path=paths[i], counts=rows[i], counts_salt=salts[i], counts_path=counts_paths[i],
            salt_c=c.salt)
        assert len(paths[i]) == DEPTH and PC.reference(inst), i
