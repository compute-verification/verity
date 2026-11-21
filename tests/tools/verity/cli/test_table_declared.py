"""The served table reads what a listed scheme's declaration states, never a scheme's name: what a party's start writes
(`Launch.starts`, `served_table.started`) and the device premise a piece's answer places (`served_table.attested`)."""
from __future__ import annotations

import dataclasses
import json
import sys

import pytest

from verity_cli import run as R
from verity_vllm.inputs import serves

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


def test_each_declared_read_is_a_run_of_a_party_its_scheme_declares():
    reads = {(n, p, k): run for n, c in T.MANIFESTS.items() if n not in T.UNDECLARED
             and not isinstance(said := T.schemes.get(n).reads, T.Lacks) for p, r in said.items() for k, run in r.items()}
    assert reads == {("bandwidth", T.BANDWIDTH_VERIFIER, "challenger"): ("pous", T.MEMORY_ATTESTER)}
    for scheme, party in reads.values():
        assert party in {r.name for r in T.MANIFESTS[scheme].parties}


def test_the_declared_components_are_the_schemes_whose_record_names_a_row():
    rowed = {n for n in T.schemes.DECLARATIONS if not isinstance(T.schemes.get(n).record, T.Lacks)}
    assert set(T.MANIFESTS) == rowed | set(T.UNDECLARED)
    assert rowed == {"sampled-proofs", "inference-only", "pearl-plus", "pous", "bandwidth", "wipe", "network-attester"}


def test_only_the_sampled_proofs_carry_the_served_windows_rule_and_its_default():
    designs = {n: (c.default_sampling_design, c.sampling_design) for n, c in T.MANIFESTS.items() if c.sampling_design}
    assert designs == {"sampled-proofs": ("rate:1/3481648", T._design)}


def test_a_tied_schemes_parameters_are_what_its_tie_names_and_nothing_else(monkeypatch):
    s = dataclasses.replace(T.schemes.get("pearl-plus"), tie={"class": "1,2,3", "schedule": "serial", "kernel": ""})
    cls = serves.SERVES.window_class
    assert T._tied(s, "x", {"class": cls, "kernel": "k"}) == {
        "class": cls, "schedule": T.schemes.get("pearl-plus").settings["schedule"], "kernel": "k"}
    with pytest.raises(T.Refused, match=r"lane \(known: class, kernel, schedule\)"):
        T._tied(s, "x", {"lane": "v"})


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
