"""The drawn-unit replay bundle: the checked-in fixture checks clean and fits its schema, each kind of tampering is refused by
name, and a fresh bundle from the torch loop's window agrees with the fixture and refuses a tampered loop's outputs."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import jsonschema
import pytest

from verity_e2e_demo import replay_bundle as B

HERE = Path(__file__).parent
FIXTURE = HERE / "fixtures" / "drawn-unit-bundle.json"
SCHEMA = Path(B.__file__).parent / "schemas" / "drawn-unit-bundle.v1.schema.json"


@pytest.fixture(scope="module")
def bundle() -> tuple[dict, bytes]:
    return B.read(FIXTURE)


def _repay(doc: dict, payload: bytes) -> dict:
    doc = copy.deepcopy(doc)
    doc["payload"]["sha512"] = hashlib.sha512(payload).hexdigest()
    return doc


def _row(doc: dict, unit: int, port: str) -> dict:
    return next(r for u in doc["units"] if u["unit"] == unit for r in u["rows"] if r["port"] == port)


def test_fixture_checks_clean_and_fits_its_schema(bundle) -> None:
    doc, payload = bundle
    jsonschema.Draft202012Validator(json.loads(SCHEMA.read_text())).validate(doc)
    assert B.check(doc, payload) == {}
    assert [u["unit"] for u in doc["units"]] == list(B.FIXTURE_DRAW)
    assert {(u["coords"]["slot"], u["coords"]["layer"]) for u in doc["units"]} == {(0, 0), (0, 1), (1, 0), (1, 1)}
    for u in doc["units"]:
        kinds = {r["port"]: ("registered" if r["registered"] else "slot") for r in u["rows"]}
        assert kinds == {"x": "slot", "sx": "slot", "wrow": "registered", "sw": "registered", "o0": "slot"}
    assert all(B.replay(u, payload) for u in doc["units"])


def test_a_flipped_payload_byte_is_refused(bundle) -> None:
    doc, payload = bundle
    bad = bytearray(payload)
    bad[7] ^= 1
    assert B.check(doc, bytes(bad)) == {"payload": "its bytes are not the ones the bundle names"}
    assert B.check(_repay(doc, bytes(bad)), bytes(bad)) == {f"unit:{doc['units'][0]['unit']}/x": "its row and salt are not its string"}


def test_a_wrong_window_path_is_refused(bundle) -> None:
    doc, payload = bundle
    doc = copy.deepcopy(doc)
    r = _row(doc, 300, "x")
    r["window"]["path"][0] = "00" * 64
    assert B.check(doc, payload) == {"unit:300/x": "its window path does not reach its port's registered root"}


def test_a_wrong_slot_path_is_refused(bundle) -> None:
    doc, payload = bundle
    doc = copy.deepcopy(doc)
    _row(doc, 200, "sx")["slot"]["step_path"].append("00" * 64)
    assert B.check(doc, payload) == {"unit:200/sx": "its slot path does not reach its slot, step and block roots"}


def test_a_swapped_registered_row_is_refused(bundle) -> None:
    doc, payload = bundle
    doc = copy.deepcopy(doc)
    _row(doc, 5, "wrow")["registered"]["position"] = 6
    assert B.check(doc, payload) == {"unit:5/wrow": "its registered path does not reach the value's root"}


def test_a_forged_slot_domain_is_refused(bundle) -> None:
    doc, payload = bundle
    doc = copy.deepcopy(doc)
    doc["slot_trees"]["blocks"][0]["steps"][0]["slots"][0]["leaves"] += 1
    assert "slot:0/0/0" in B.check(doc, payload)


def test_a_tampered_output_is_refused_by_replay(bundle) -> None:
    """The replay alone (in `check` a changed output row is refused first by its string)."""
    doc, payload = bundle
    u = copy.deepcopy(doc["units"][1])
    o = next(r for r in u["rows"] if r["kind"] == "out")
    bad = bytearray(payload)
    bad[o["row"][0]] ^= 1
    assert B.replay(u, payload) and not B.replay(u, bytes(bad))


@pytest.fixture(scope="module")
def fresh(tmp_path_factory) -> tuple[Path, dict]:
    d = tmp_path_factory.mktemp("window")
    _, weights = B.torch_loop_window(d)
    return d, weights


def test_a_fresh_bundle_is_the_fixture(fresh) -> None:
    d, weights = fresh
    doc, payload = B.build(d, B.FIXTURE_DRAW, reads=B.torch_loop_reads, weights=weights,
                           draw={"source": "fixture", "law": {"law": "subset", "k": len(B.FIXTURE_DRAW)}})
    assert B.check(doc, payload, registration=json.loads((d / "registration.json").read_text())) == {}
    want, want_payload = B.read(FIXTURE)
    assert payload == want_payload and json.loads(json.dumps(doc)) == want


def test_another_registration_is_refused(fresh, bundle) -> None:
    d, _ = fresh
    doc, payload = bundle
    reg = json.loads((d / "registration.json").read_text())
    for r in reg["roots"]:
        r["root"] = "00" * 64
    bad = B.check(doc, payload, registration=reg)
    assert bad and all(v == "not the registration's root" for v in bad.values())


def test_weights_under_other_salts_are_refused(fresh) -> None:
    d, _ = fresh
    from verity_torch_loop import loop as L
    other = B.torch_loop_weights(L.Model(), hashlib.sha256(b"another test key").digest(), "torch-loop/mlp-w8a8-h128/seed0")
    with pytest.raises(ValueError, match="committed under its own salt"):
        B.build(d, [5], reads=B.torch_loop_reads, weights=other)


def test_a_tampered_loop_fails_replay(tmp_path) -> None:
    _, weights = B.torch_loop_window(tmp_path, tamper=True)
    doc, payload = B.build(tmp_path, [5, 300], reads=B.torch_loop_reads, weights=weights)
    assert B.check(doc, payload) == {u: "its committed outputs are not its Definition's on its committed inputs"
                                     for u in ("unit:5", "unit:300")}
