"""PoUS through the proof service (`service/spec.py`, `service/driver.py`): the Spec is one a `Stream` runs, C's
commitment opens where the claims check needs it, and the CPU path end to end accepts the honest audit with a
verdict-only output and rejects both controls."""
from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import pytest
from verity.experimental.protocols.pous import setup
from verity.experimental.protocols.pous.schemes import p2

BENCH = Path(__file__).resolve().parents[1]
NB = p2.P2().params.block_bytes
"""The unit's width: P2's registered block."""


@pytest.fixture
def svc(monkeypatch):
    monkeypatch.syspath_prepend(str(BENCH))
    from service import driver, spec

    return spec, driver


def _blocks(n: int, nb: int = NB) -> list[bytes]:
    return [hashlib.shake_256(b"C%d" % i).digest(nb) for i in range(n)]


def test_the_spec_is_all_and_a_stream_registers_a_session(svc):
    """`All` (every round proved), the check's Program the unit's, and a session's registration through the public-items
    rule and one-stage's R1-R7 against the challenger's own ports."""
    SP, _ = svc
    from verity.experimental.protocols.sampled_proofs import service as S

    sp = SP.spec(NB)
    assert isinstance(sp.selection, S.All) and sp.check.public_outputs == ()
    assert [p.item for p in sp.public if p.kind == "output"] == ["verdict"]
    session = bytes(range(32))
    stream = S.Stream(sp, SP.unit(NB), budget=1, delta=2.0 ** -40, challenger=SP.CHALLENGER)
    rec = SP.record(NB, session, 5, {"block": "01" * 64, "o0": "02" * 64})
    assert S.check_public(sp, rec).ok
    reg = stream.register(rec, SP.ports(session, 5))
    assert stream.select(reg).units == [0, 1, 2, 3, 4]
    other = S.Stream(sp, SP.unit(NB), budget=1, delta=2.0 ** -40, challenger=SP.CHALLENGER)
    with pytest.raises(S.Refused, match="R3-domain"):
        other.register(rec, SP.ports(bytes(32), 5))


def test_c_commitment_opens_each_leaf_and_binds_the_header(svc):
    SP, _ = svc
    from verity.ml.commitments.merkle import Commitment, Opening, verify_opening
    from verity.experimental.protocols.pous import memory_challenger as mc

    blocks, key = _blocks(5), b"\x11" * 32
    t = SP.commit_c([mc.block_digest(b) for b in blocks], key, b"setup")
    dom = SP.c_domain(b"setup", 5)
    for i, b in enumerate(blocks):
        bc = SP.HM.commit_string(SP.KEY, mc.block_digest(b), SP.c_salt(key, i))
        assert verify_opening(dom, Commitment(t.root, 5), Opening(i, SP.HM.tree_leaf(SP.KEY, bc), tuple(t.path(i))), SP.SCHEMA, SP.LIMITS)
        assert not verify_opening(dom, Commitment(t.root, 5), Opening((i + 1) % 5, SP.HM.tree_leaf(SP.KEY, bc), tuple(t.path(i))),
                                  SP.SCHEMA, SP.LIMITS)
    assert SP.session_binding(t.root, 5) != SP.session_binding(SP.commit_c([mc.block_digest(b) for b in blocks], b"\x12" * 32,
                                                                           b"setup").root, 5)


@pytest.mark.slow
def test_the_cpu_path_accepts_the_honest_audit_verdict_only_and_rejects_the_controls(svc, tmp_path):
    """The setup of P2's registered codec (its expander, and the setup's whole commitment to `W`) at two blocks, a
    three-round session on the logical clock under the test signer, its answers staged for C-Flock and decided through a
    `Stream` (a stub verdict of record: no binaries here). The output file is the verdict alone; the tampered block fails
    C's opening and the edited record its signature, each in a stream of its own."""
    _, driver = svc
    assert driver.main(["--out", str(tmp_path), "--k", "3", "--bps", "2"]) == 0
    assert json.loads((tmp_path / "verdict.json").read_text()) == {"accepted": True}
    doc = json.loads((tmp_path / "summary.json").read_text())
    assert doc["verifier_of_record"] == "stub (no binaries)" and doc["run"]["scheme"] == p2.P2().name
    assert doc["run"]["block_bytes"] == NB
    params = dataclasses.replace(p2.P2().params, blocks_per_segment=2)
    assert doc["run"]["manifest"] == params.manifest()
    assert len(bytes.fromhex(doc["run"]["w_commitment"])) == len(setup.commit(params, bytes(params.w_bytes)))
    h = doc["honest"]
    assert h["symbol"] == "accept" and h["exhaustive"] and h["claims"] == {"accepted": True, "reasons": [], "domains": True}
    assert h["k"] == 3 and h["ands"] == 0 and h["stream"]["public"] == {}
    why = {c["tag"]: (c["symbol"], c["claims"]["reasons"]) for c in doc["controls"]}
    assert why == {"tampered-block": ("reject", ["1 answered blocks are not C's leaves at their rounds' indices"]),
                   "tampered-record": ("reject", ["the signature does not verify"])}
    assert "witness" not in json.dumps(doc["honest"]["stream"])
