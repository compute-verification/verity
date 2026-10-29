"""`verity.ml.commitments.gates.attest`: the period unit accepts a party's signature of its period root
(`lms.period_root_message`) with the period tree's domain id over its count, at the toy binding, on both leaves of a
height-1 key, and refuses it with one thing wrong (the count, the domain id, the root, the deployment, the attachment, the
link, the period, the lower key's identifier or root, C) exactly where `lms.verify` of the role digest and `merkle`'s
range-u64 domain id do. The message's layout is N's alone through I, so the toy binding checks all of it;
`test_boolean_lms` holds the verifier it feeds at N = 64."""
from __future__ import annotations

import numpy as np
import pytest

from verity.ml.circuits import evaluation as E
from verity.ml.commitments.gates import attest as G
from verity.ml.commitments.indexed import RangeU64Domain
from verity.ml.commitments.merkle import CommitmentDomain
from verity.ml.crypto import lms as R

TOY = R.Params(R.TOY_N, 1, 1, toy=True)
DEPLOYMENT, PARTY, ATTACHMENT, LINK, PERIOD = bytes(range(64)), "memory-challenger", bytes(range(32)), "server-3", 7
ROOT = bytes(range(64, 128))
L = 161


def _flip(b: bytes) -> bytes:
    return bytes([b[0] ^ 1]) + b[1:]


def _domain(count: int) -> bytes:
    return CommitmentDomain(G.BINDING, -1, RangeU64Domain(0, count), hash="sha512").domain_id


def _cases() -> tuple[list, list[int]]:
    k = R.keygen(TOY, bytes(range(R.TOY_N)))
    lower = R.public_key(k)
    rows, want = [], []
    for q, count in enumerate((5, 2 ** 64 - 1)):
        s = R.sign_at(k, q, R.period_root_message(k.I, DEPLOYMENT, PARTY, ATTACHMENT, LINK, PERIOD, count, ROOT))
        honest = dict(lower=lower, deployment=DEPLOYMENT, attachment=ATTACHMENT, link=LINK, period=PERIOD, count=count,
                      domain=_domain(count), root=ROOT, sig=s)
        for change in ({}, {"count": count ^ 1}, {"count": count ^ 1, "domain": _domain(count ^ 1)},
                       {"domain": _flip(_domain(count))}, {"domain": _domain(count ^ 2)}, {"root": _flip(ROOT)},
                       {"deployment": _flip(DEPLOYMENT)}, {"attachment": _flip(ATTACHMENT)}, {"link": "server-4"},
                       {"period": PERIOD + 1}, {"lower": _flip(lower)}, {"lower": lower[:16] + _flip(lower[16:])},
                       {"sig": R.Signature(q, _flip(s.C), s.y, s.path)}):
            c = honest | change
            ctx = G.context(c["deployment"], PARTY, c["attachment"], c["link"], c["period"])
            rows.append(G.arguments(TOY, c["lower"], ctx, c["count"], c["domain"], c["root"], c["sig"]))
            m = R.period_root_message(c["lower"][:16], c["deployment"], PARTY, c["attachment"], c["link"], c["period"],
                                      c["count"], c["root"])
            want.append(int(R.verify(TOY, c["lower"], m, c["sig"]) and c["domain"] == _domain(c["count"])))
    return rows, want


@pytest.mark.slow
def test_the_period_unit_agrees_with_the_reference() -> None:
    rows, want = _cases()
    assert want == [1] + [0] * 12 + [1] + [0] * 12
    cols = [np.array([r[k] for r in rows], dtype=np.uint8) for k in range(10)]
    got = np.asarray(E.evaluate_batch(G.period(R.TOY_N, 1, 1, L), *cols)).reshape(-1).tolist()
    assert got == want


def test_the_period_unit_is_typed_by_its_shape() -> None:
    assert len(G.context(DEPLOYMENT, PARTY, ATTACHMENT, LINK, PERIOD)) == L
    v = G.period(64, 1, 3, L)
    assert v is G.period(64, 1, 3, L)
    assert [t.leaves for _, t in v.signature()[0]] == [128, 512, 8 * L, 64, 512, 512, 3, 512, 522 * 512, 3 * 512]
    for bad in (0, -1, True, 1.0):
        with pytest.raises(ValueError):
            G.period(64, 1, 1, bad).signature()


def test_malformed_arguments_are_refused() -> None:
    k = R.keygen(TOY, bytes(range(R.TOY_N)))
    s = R.sign_at(k, 0, bytes(64))
    ctx = G.context(DEPLOYMENT, PARTY, ATTACHMENT, LINK, PERIOD)
    d = _domain(1)
    for count, domain, root in ((-1, d, ROOT), (2 ** 64, d, ROOT), (True, d, ROOT), (1, d, ROOT[:-1]), (1, d[:-1], ROOT)):
        with pytest.raises(ValueError, match="a u64, and its domain id and root 64 bytes"):
            G.arguments(TOY, R.public_key(k), ctx, count, domain, root, s)
    with pytest.raises(ValueError, match="a positive int"):
        G.arguments(TOY, R.public_key(k), b"", 1, d, ROOT, s)
