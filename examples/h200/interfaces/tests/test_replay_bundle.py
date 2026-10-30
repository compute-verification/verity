"""The drawn-unit replay bundle: the fixture is its pinned digests, checks clean and fits its schema, each kind of tampering
is refused by name, and kernels' commit call reproduces their fixture."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import jsonschema
import pytest

from verity_e2e_demo import replay_bundle as B

S = B.S

SCHEMA = Path(B.__file__).parent / "schemas" / "drawn-unit-bundle.v2.schema.json"
#: the bundle of units 5, 200, 300 and 511 (slot 0 layer 0, slot 0 layer 1, slot 1 layer 0, slot 1 layer 1) of a W8A8 MLP's
#: window (two requests, one step, width 128: 512 units in members of 8) on the RTX PRO 6000's dot, under the window's
#: registered subset:4, and that registration; `prove_bundle.py make` wrote them at bc24cf2713 from the retired torch loop,
#: and the window itself is in the evidence store (benchmarks/one_stage/prove_bundle.sh's BUNDLE_WINDOW)
FIXTURES = Path(__file__).parent / "fixtures"
DRAW = (5, 200, 300, 511)
PATHS = "its paths do not reach its unit, slot and step roots"
#: `B.digests` of the fixture: a change to the bundle's bytes changes these
GOLDEN = {"json": "fb509ff3a87ce874cc7fa2bbf5536e3cb7365c0d88153f81ae024fba511b169369e45c43fffa3e669189eeee76dcc6429eb45b820328ada7cbe4008961642cd7",
          "bin": "183e9e504664b5d831135407b51246829dc10eedf9587917cd88cef516599d2c9685bb19e9b013f403bba1aba68e270495c2e830139fea7efa961f0c823b3db3"}
#: SHA-512 of kernels' fixtures/commit-slots-v1.json, which `S.fixture()` writes as ``json.dumps(fixture, indent=1) + "\n"``;
#: its roots and digests are those of the commit `B.REFERENCE` pins
SLOTS_FIXTURE_SHA512 = ("56114dd7ddaaed0cc2a0dba4eed2dedf6660a5462c8696cfa4a5b66e0b0eb54b"
                        "dddf1ba6713756579099f58dc6671f49000b75dea7461359106f3a6ae336c48a")


@pytest.fixture(scope="module")
def bundle() -> tuple[dict, bytes]:
    return B.read(FIXTURES / "drawn-unit-bundle.json")


@pytest.fixture(scope="module")
def registration() -> dict:
    return json.loads((FIXTURES / "drawn-unit-registration.json").read_text())


@pytest.fixture(scope="module")
def validator() -> jsonschema.Draft202012Validator:
    return jsonschema.Draft202012Validator(json.loads(SCHEMA.read_text()))


def _repay(doc: dict, payload: bytes) -> dict:
    doc = copy.deepcopy(doc)
    doc["payload"]["sha512"] = hashlib.sha512(payload).hexdigest()
    return doc


def _row(doc: dict, unit: int, port: str) -> dict:
    return next(r for u in doc["units"] if u["unit"] == unit for r in u["rows"] if r["port"] == port)


def test_the_vendored_commit_call_is_kernels_fixture() -> None:
    assert hashlib.sha512((json.dumps(S.fixture(), indent=1) + "\n").encode()).hexdigest() == SLOTS_FIXTURE_SHA512


def test_fixture_checks_clean_and_fits_its_schema(bundle, validator) -> None:
    doc, payload = bundle
    validator.validate(doc)
    assert B.check(doc, payload) == {}
    assert doc["target"] == {"binding": "GEMM_SM120_S8", "card": "sm_120", "dot": "BlackwellS8ImmaDot32_v1"}
    assert doc["commitment"]["reference"] == B.REFERENCE
    assert [u["unit"] for u in doc["units"]] == list(DRAW)
    assert {(u["coords"]["slot"], u["coords"]["layer"]) for u in doc["units"]} == {(0, 0), (0, 1), (1, 0), (1, 1)}
    for u in doc["units"]:
        kinds = {r["port"]: ("registered" if r["registered"] else "slot") for r in u["rows"]}
        assert kinds == {"x": "slot", "sx": "slot", "wrow": "registered", "sw": "registered", "o0": "slot"}
    assert all(B.replay(u, payload) for u in doc["units"])


def test_inputs_open_at_their_producer_and_outputs_at_their_own_unit(bundle) -> None:
    doc, _ = bundle
    for u in doc["units"]:
        c, base = u["coords"], u["coords"]["layer"] * 129
        rows = {r["port"]: r["slot"] for r in u["rows"]}
        producer = {"block": c["block"], "step": c["step"], "slot": c["slot"], "layer": c["layer"],
                    "op": f"{c['op']}.input_quant", "unit": base}
        assert [(rows[p]["unit"], rows[p]["rank"], rows[p]["producer"]) for p in ("x", "sx")] == [(base, 0, producer),
                                                                                                (base, 1, producer)]
        assert (rows["o0"]["unit"], rows["o0"]["rank"], rows["o0"]["producer"]) == (base + 1 + c["unit"], 0, None)


def test_a_flipped_payload_byte_is_refused(bundle) -> None:
    doc, payload = bundle
    bad = bytearray(payload)
    bad[7] ^= 1
    assert B.check(doc, bytes(bad)) == {"payload": "its bytes are not the ones the bundle names"}
    assert B.check(_repay(doc, bytes(bad)), bytes(bad)) == {f"unit:{doc['units'][0]['unit']}/x": "its row and salt are not its string"}


def test_a_wrong_window_path_is_refused(bundle) -> None:
    doc, payload = bundle
    doc = copy.deepcopy(doc)
    _row(doc, 300, "x")["window"]["path"][0] = "00" * 64
    assert B.check(doc, payload) == {"unit:300/x": "its window path does not reach its port's registered root"}


@pytest.mark.parametrize("edit", [
    lambda s: s["path"].append("00" * 64),
    lambda s: s["slot_path"]["proof"].__setitem__(0, "00" * 64),
    lambda s: s["step_path"]["proof"].__setitem__(0, "00" * 64),
    lambda s: s["step_path"]["roots"].__setitem__(0, "00" * 64),
], ids=["unit-path", "slot-path", "step-path", "slot-root"])
def test_a_wrong_step_commitment_path_is_refused(bundle, edit) -> None:
    doc, payload = bundle
    doc = copy.deepcopy(doc)
    edit(_row(doc, 200, "sx")["slot"])
    assert B.check(doc, payload) == {"unit:200/sx": PATHS}


def test_an_opening_in_another_slot_is_refused(bundle) -> None:
    doc, payload = bundle
    doc = copy.deepcopy(doc)
    _row(doc, 5, "o0")["slot"]["slot"] = 1
    assert B.check(doc, payload) == {"unit:5/o0": "it opens outside its unit's slot and step"}


def test_a_producer_not_where_the_input_opens_is_refused(bundle) -> None:
    doc, payload = bundle
    doc = copy.deepcopy(doc)
    _row(doc, 5, "x")["slot"]["producer"]["unit"] = 1
    _row(doc, 5, "sx")["slot"]["producer"] = None
    _row(doc, 200, "o0")["slot"]["producer"] = dict(_row(doc, 200, "x")["slot"]["producer"])
    assert B.check(doc, payload) == {"unit:5/x": "its producer's coordinates are not where it opens",
                                     "unit:5/sx": "an output opens in its own unit, an input names its producer",
                                     "unit:200/o0": "an output opens in its own unit, an input names its producer"}


def test_a_swapped_registered_row_is_refused(bundle) -> None:
    doc, payload = bundle
    doc = copy.deepcopy(doc)
    _row(doc, 5, "wrow")["registered"]["position"] = 6
    assert B.check(doc, payload) == {"unit:5/wrow": "its registered path does not reach the value's root"}


def test_a_forged_layout_digest_or_block_is_refused(bundle) -> None:
    doc, payload = bundle
    forged = copy.deepcopy(doc)
    forged["commitment"]["layout_digest"] = "00" * 64
    assert B.check(forged, payload) == {"commitment": "its layout digest is not its layout's"}
    forged = copy.deepcopy(doc)
    forged["commitment"]["blocks"][0]["steps"][0]["root"] = "00" * 64
    bad = B.check(forged, payload)
    assert bad["block:0"] == "its root is not the block tree's over its steps' roots"
    assert {k for k, v in bad.items() if v == PATHS} == {f"unit:{u}/{p}" for u in DRAW for p in ("x", "sx", "o0")}


def test_another_context_is_refused(bundle) -> None:
    doc, payload = bundle
    doc = copy.deepcopy(doc)
    doc["commitment"]["context"] = "11" * 64
    bad = B.check(doc, payload)
    assert bad["block:0"] and bad["step:0/0"] == "its domain is not the commit call's"
    assert {k for k, v in bad.items() if v == PATHS} == {f"unit:{u}/{p}" for u in DRAW for p in ("x", "sx", "o0")}


def test_a_live_draw_carries_its_receipt(bundle, validator) -> None:
    doc, payload = bundle
    live = copy.deepcopy(doc)
    live["draw"]["source"] = "live"
    assert B.check(live, payload) == {"draw": "a live draw carries the challenger's live-coin receipt, and a draw is live or fixture"}
    assert not validator.is_valid(live)
    live["draw"]["receipt"] = {"challenger": "test", "coins": "00" * 64}
    assert B.check(live, payload) == {} and validator.is_valid(live)


@pytest.mark.parametrize("binding", ["GEMM_SM90_S8", "GEMM_HOPPER", "GEMM_SM80"])
def test_the_target_is_its_binding_as_circuits_resolves_it(bundle, binding) -> None:
    doc, payload = bundle
    doc = copy.deepcopy(doc)
    doc["target"]["binding"] = binding
    assert "target" in B.check(doc, payload)


def test_a_target_whose_dot_the_templates_do_not_bind_is_refused(bundle) -> None:
    doc, payload = bundle
    doc = copy.deepcopy(doc)
    doc["target"] = B.target("GEMM_HOPPER")
    assert doc["target"]["dot"] == "HopperBF16WgmmaDot16_v1"
    assert B.check(doc, payload) == {f"unit:{u}": "its template binds another dot than the target's HopperBF16WgmmaDot16_v1"
                                     for u in DRAW}


def test_a_tampered_output_is_refused_by_replay(bundle) -> None:
    """The replay alone (in `check` a changed output row is refused first by its string)."""
    doc, payload = bundle
    u = copy.deepcopy(doc["units"][1])
    o = next(r for r in u["rows"] if r["kind"] == "out")
    bad = bytearray(payload)
    bad[o["row"][0]] ^= 1
    assert B.replay(u, payload) and not B.replay(u, bytes(bad))


def test_a_boolean_template_replays_its_bits_in_word_order() -> None:
    """A Boolean-profile template's rows are its ports' bits packed little-endian into u16 words: an argmax over bf16 words
    replays to the index numpy finds, and a flipped low or high index bit is refused."""
    import numpy as np

    logits = np.array([0.5, -3.0, 2.25, 7.5, 1.0, -0.25, 6.0, 0.0, 3.5, -7.0, 4.0, 5.5, 1.5, 2.0, -1.0, 0.75], np.float32)
    words = (logits.view(np.uint32) >> 16).astype("<u2").tobytes()
    unit = {"template": {"id": "Argmax_v2{V=16}"},
            "rows": [{"port": "logits", "row": [0, 32]}, {"port": "o0", "row": [32, 2]}]}
    payload = words + int(np.argmax(logits)).to_bytes(2, "little")
    assert B.replay(unit, payload)
    for bit in (0, 15):
        bad = bytearray(payload)
        bad[32 + bit // 8] ^= 1 << (bit % 8)
        assert not B.replay(unit, bytes(bad))


def test_the_fixture_is_its_digests(bundle, registration) -> None:
    doc, payload = bundle
    assert B.digests(doc, payload) == GOLDEN
    assert B.check(doc, payload, registration=registration) == {}


def test_another_registration_is_refused(bundle, registration) -> None:
    doc, payload = bundle
    reg = copy.deepcopy(registration)
    for r in reg["roots"]:
        r["root"] = "00" * 64
    bad = B.check(doc, payload, registration=reg)
    assert bad and all(v == "not the registration's root" for v in bad.values())
