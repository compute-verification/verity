"""The served table reads what a listed scheme's declaration states, never a scheme's name: what a party's start writes
(`Launch.starts`, `served_table.started`), what a tied application states over a scheme's defaults (`Scheme.tie`,
`served_table.tied`), and the device premise a piece's answer places (`served_table.attested`)."""
from __future__ import annotations

import dataclasses
import json
import sys

from verity_cli import run as R

REG = R.registry()
T = sys.modules["served_table"]


def test_a_start_writes_what_each_piece_that_reaches_its_party_declares(monkeypatch):
    stand_in = T.Piece("stand-in", T.WORKERS, T.SERVED_WINDOW, "", reaches=T.CHALLENGER, optional=("gate", "x"),
                       starts=("x", "gate"), starts_optional=("gate",))
    monkeypatch.setitem(T.PIECES, "stand-in", {T.WORKERS: stand_in})
    start = T.started(T.CHALLENGER, serve=lambda cfg, state: None, stop=lambda cfg, state: None)
    assert start.writes["stand-in"] == ("x", "gate") and ("stand-in", "gate") in start.optional
    assert ("stand-in", "x") not in start.optional
    assert {c: k for c, k in start.writes.items() if c != "stand-in"} == dict(T.STARTS[T.CHALLENGER].writes)
    monkeypatch.setitem(T.STARTS, T.CHALLENGER, start)
    made = {"x": "1", "gate": "g", "loopback": "l", "main-file": "m", "peer": "p", "lane": "v", "key-sha512": "k"}
    assert T._written(T.CHALLENGER, made)["stand-in"] == {"x": "1", "gate": "g"}


def test_the_declared_starts_are_what_the_challenger_and_the_network_attester_write():
    assert {p: {c: list(k) for c, k in s.writes.items()} for p, s in T.STARTS.items()} == {
        T.CHALLENGER: {"sampled-proofs": ["loopback", "main-file"],
                       "pearl-plus": ["gate", "peer", "lane", "key-sha512", "main-file"]},
        T.NETWORK_ATTESTER: {"network-attester": ["gate", "peer"]}}
    assert T.STARTS[T.CHALLENGER].optional == {("sampled-proofs", "loopback"), ("sampled-proofs", "main-file"),
                                               ("pearl-plus", "lane"), ("pearl-plus", "main-file")}
    assert not T.STARTS[T.NETWORK_ATTESTER].optional


def test_a_tied_application_states_each_listed_schemes_declared_tie(monkeypatch):
    real = T.schemes.get
    monkeypatch.setattr(T.schemes, "get", lambda n: dataclasses.replace(real(n), tie={"order": "fifo"})
                        if n == "network-attester" else real(n))
    got = T.tied(("sampled-proofs", "pearl-plus", "network-attester"))
    assert got[T.MANIFESTS["network-attester"].id] == {"order": "fifo"}
    assert got[T.MANIFESTS["pearl-plus"].id] == {"class": "4,27,2", "tie-uniform": True}
    assert T.MANIFESTS["sampled-proofs"].id not in got


def test_a_piece_answer_places_the_listed_component_whose_id_it_names(tmp_path, monkeypatch):
    monkeypatch.setattr(T.mc, "premise", lambda where: f"premise of {where}")
    pieces = tmp_path / "out" / "pieces"
    for d, protocol, profile in (("certifier", T.MANIFESTS["network-attester"].id, {"placement": "here"}),
                                 ("work", T.MANIFESTS["pearl-plus"].id, {}),
                                 ("other", "verity/nothing/v0", {"placement": "there"})):
        (pieces / d).mkdir(parents=True)
        (pieces / d / "answer.json").write_text(json.dumps({"protocol": protocol, "profile": profile}))
    cfg = {"protocols": ["sampled-proofs", "pearl-plus", "network-attester"]}
    assert T.attested(cfg, tmp_path) == {"network-attester": "premise of here"}
    assert T.attested({"protocols": ["pearl-plus"]}, tmp_path) == {}
