"""REDTEAM phase 8, target 1: #3597's fail-closed guard in `one_stage.proofs.verify` (PRF-147, commit fa52b3a74b).

`verify` refuses `registered=None` when the verifier's own program descriptor reads registered values
(`reads_registered`: a `RegisteredInput<w>_v1` among `definitions`, or no `definitions` object to tell). The lead's
question: can a statement with registered reads pass through `registered=None` anyway — a descriptor that doesn't name
the port, a port under another id pattern, or a partition trick?

These are executable red-team cases, NOT a merge-gate suite. They are evidence on a branch off the tests branch, run with
the n2-zk-record-pins head's code on the path. Findings (ranked) in the phase-8 note.

What they establish:
- An honestly-encoded descriptor ALWAYS names the port: `encode_program` emits every reachable primitive (including
  `RegisteredInput<w>_v1`) as a top-level `definitions` key, so the guard fires and `registered=None` is refused.
- The guard fails closed on a descriptor it can't read (not JSON, no `definitions`, `definitions` not an object).
- The guard's regex `RegisteredInput[1-9][0-9]*_v1` (fullmatch) is EXACTLY Lean's value-binding recognizer
  `Program.inputId?` (`widthAfter? "RegisteredInput"`), so no id pattern Lean binds as a registered value slips past it.
- The partition is not read by the guard, and cannot smuggle a registered read through `registered=None`.
- The RESIDUAL gap the guard cannot close: it keys on the descriptor's `definitions` keys, never on the circuit. A
  descriptor that references a registered port only in a body (not a top-level key) — the shape a descriptor/circuit
  divergence (PRF-133) would take — is NOT flagged. That is where Lean's own fail-open (`Registered.check` with
  `own=None` trusts the prover's public header; see the note and the Lean fixture) would bite.
"""
from __future__ import annotations

import json
import random
import string

import pytest

from verity.core.protocols.sampled_proofs.one_stage import proofs as PF
from verity.ml.circuits import codec, program as P
from verity.ml.circuits.program import Program
from verity.ml.circuits.tests import test_constants as TC


def _never():
    raise AssertionError("the verifier ran: the guard let a registered-reading program through registered=None")


def _reg_program() -> bytes:
    """A real program that reads a registered input (`eps`), encoded as the verifier's own copy would be."""
    return codec.canonical_json(codec.encode_program(Program(TC.New, registered=("eps",))))


def _run_only_program() -> bytes:
    return codec.canonical_json(codec.encode_program(Program(TC.New)))


def _verify(program: bytes, *, registered, partition=None, tmp_path):
    return PF.verify(
        PF.Proof(tmp_path / "circuit.txt", tmp_path / "pub.bin", (tmp_path / "session",)),
        circuit_sha512="ab" * 64, program=program, partition=partition or {}, registered=registered)


def test_an_honestly_encoded_registered_descriptor_names_the_port_and_is_refused(tmp_path, monkeypatch):
    """A Program that reads a registered input is encoded with `RegisteredInput<w>_v1` as a top-level definitions key,
    so the guard fires before the verifier runs and refuses registered=None."""
    monkeypatch.setattr(PF, "executable", _never)
    prog = _reg_program()
    keys = json.loads(prog)["definitions"].keys()
    assert any(P.REGISTERED_ID.fullmatch(k) for k in keys), list(keys)
    assert PF.reads_registered(prog)
    got, why = _verify(prog, registered=None, tmp_path=tmp_path)
    assert got is None and why[0] == "registered", why


def test_a_run_only_descriptor_is_not_flagged_and_goes_on(tmp_path, monkeypatch):
    """A Program whose inputs are all per-run reads registered none; the guard allows registered=None and the call goes
    on to its proof files (refused there, the verifier still untouched)."""
    monkeypatch.setattr(PF, "executable", _never)
    prog = _run_only_program()
    assert not any(P.REGISTERED_ID.fullmatch(k) for k in json.loads(prog)["definitions"])
    assert not PF.reads_registered(prog)
    got, why = _verify(prog, registered=None, tmp_path=tmp_path)
    assert got is None and why[0] == "proof", why


@pytest.mark.parametrize("program", [
    b"not json at all", b"{}", b'{"definitions": []}', b'{"definitions": "RegisteredInput16_v1"}',
    b'{"definitions": null}', b'[1,2,3]', b'{"schema":"x"}'],
    ids=["not-json", "no-definitions", "definitions-list", "definitions-string", "definitions-null",
         "top-level-list", "no-definitions-key"])
def test_the_guard_fails_closed_on_a_descriptor_it_cannot_read(tmp_path, monkeypatch, program):
    monkeypatch.setattr(PF, "executable", _never)
    assert PF.reads_registered(program), program
    got, why = _verify(program, registered=None, tmp_path=tmp_path)
    assert got is None and why[0] == "registered", why


def _lean_width_after(pre: str, s: str):
    """Port of Flock/Program.lean `widthAfter?`."""
    if not (s.startswith(pre) and s.endswith("_v1")):
        return None
    w = s[len(pre):len(s) - 3]
    if len(w) > 0 and all(c in "0123456789" for c in w) and w[0] != "0":
        return int(w)
    return None


def _lean_binds_registered(s: str) -> bool:
    """Port of Flock/Program.lean `inputId?`: Input first, then RegisteredInput; True iff it binds a REGISTERED value."""
    if _lean_width_after("Input", s) is not None:
        return False
    return _lean_width_after("RegisteredInput", s) is not None


def test_the_guard_regex_matches_leans_registered_value_recognizer():
    """`reads_registered`'s regex is EXACTLY Lean's `inputId?` registration branch, so no id Lean binds as a registered
    value slips past the guard (the lead's "a port under another id pattern"). Hand cases + fuzz over the id alphabet
    and over `RegisteredInput`-prefixed near-misses; zero disagreements."""
    hand = ["RegisteredInput1_v1", "RegisteredInput16_v1", "RegisteredInput01_v1", "RegisteredInput0_v1",
            "RegisteredInput_v1", "RegisteredInput1_v11", "RegisteredInput1_v2", "RegisteredInput1_V1",
            "registeredInput1_v1", "RegisteredInput 1_v1", "Input1_v1", "Input16_v1", "Input0_v1",
            "RegisteredRegisteredInput1_v1", "RegisteredInput12345678901234567890_v1", "RegisteredInput+1_v1",
            "RegisteredInput-1_v1", "XRegisteredInput1_v1", "RegisteredInput1_v10", " RegisteredInput1_v1",
            "RegisteredInput1_v1 "]
    alphabet = string.ascii_letters + string.digits + "_ +-"
    rng = random.Random(20260124)
    rand = ["".join(rng.choice(alphabet) for _ in range(rng.randint(0, 24))) for _ in range(40000)]
    rand += ["RegisteredInput" + "".join(rng.choice("0123456789_v ") for _ in range(rng.randint(0, 6)))
             for _ in range(40000)]
    bad = []
    for s in hand + rand:
        guard = PF.reads_registered(json.dumps({"definitions": {s: {}}}).encode())
        pyrx = P.REGISTERED_ID.fullmatch(s) is not None
        if not (_lean_binds_registered(s) == guard == pyrx):
            bad.append((s, _lean_binds_registered(s), guard, pyrx))
    assert bad == [], bad[:20]


def test_the_partition_cannot_smuggle_a_registered_read_past_the_guard(tmp_path, monkeypatch):
    """The guard reads `program` only, never `partition`. A run-only program with a partition that names a
    registered-reading program's digest still passes registered=None (reaches the proof files); and a registered
    program is refused whatever the partition says. So the partition is no lever on the guard."""
    monkeypatch.setattr(PF, "executable", _never)
    foreign = {"program": "cafe" * 32, "units": "derived"}
    got, why = _verify(_run_only_program(), registered=None, partition=foreign, tmp_path=tmp_path)
    assert got is None and why[0] == "proof", why
    got, why = _verify(_reg_program(), registered=None, partition=foreign, tmp_path=tmp_path)
    assert got is None and why[0] == "registered", why


def test_the_guard_keys_on_definitions_not_the_circuit_a_body_only_port_is_missed(tmp_path, monkeypatch):
    """The residual gap: `reads_registered` scans the top-level `definitions` dict keys. A descriptor that references a
    registered port only inside another definition's body (or a `callees` list) — the shape a descriptor/circuit
    divergence (PRF-133) would take — is NOT flagged, so registered=None is allowed. `encode_program` never produces
    this shape (it hoists every primitive to a top-level key), so an HONEST descriptor is safe; the exposure is a
    descriptor whose definitions disagree with the committed circuit's gates, which the guard cannot see."""
    monkeypatch.setattr(PF, "executable", _never)
    body_only = json.dumps({
        "schema": "verity-ir/descriptor/v1",
        "definitions": {"Root_v1": {"body": {"nodes": [{"fn": "RegisteredInput32_v1"}]}}},
        "callees": ["RegisteredInput32_v1"],
    }).encode()
    assert not PF.reads_registered(body_only), "the guard saw a body-only registered reference it should miss"
    got, why = _verify(body_only, registered=None, tmp_path=tmp_path)
    assert got is None and why[0] == "proof", why
