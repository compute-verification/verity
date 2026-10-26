"""The drawn-unit replay bundle: the fixture, built from the torch loop's window, is its pinned digests, checks clean and fits
its schema, each kind of tampering is refused by name, the vendored commit call reproduces kernels' fixture, and a tampered
loop's outputs are refused."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import jsonschema
import pytest

from verity_e2e_demo import commit_slots as S, replay_bundle as B

SCHEMA = Path(B.__file__).parent / "schemas" / "drawn-unit-bundle.v2.schema.json"
PATHS = "its paths do not reach its unit, slot and step roots"
#: `B.digests` of `B.fixture_bundle`: a change to the bundle's bytes changes these (`python -m verity_e2e_demo.replay_bundle
#: fixture` writes it to look at)
GOLDEN = {"json": "a9e10ec969ab5ebcf41b00fa68cfb5a292dc634e4b01e7d0c23f399ede8a77b31ffc9e91a808fadf2c986e2b84dd7df137745e52a29c8047269841d20b933a18",
          "bin": "183e9e504664b5d831135407b51246829dc10eedf9587917cd88cef516599d2c9685bb19e9b013f403bba1aba68e270495c2e830139fea7efa961f0c823b3db3"}
#: SHA-512 of kernels' fixtures/commit-slots-v1.json at the commit `B.REFERENCE` pins, which `S.fixture()` writes as
#: ``json.dumps(fixture, indent=1) + "\n"``
SLOTS_FIXTURE_SHA512 = ("45f61de8c4c0d28ab0f21802b6586e60d8d45a9661a87271376b845a36801530"
                        "b7c782ffbe5e96cb7154ab33276b81da008facdc6022913a86a1569dc7a3f6b4")


@pytest.fixture(scope="module")
def fresh(tmp_path_factory) -> tuple[Path, dict, B.Commitment]:
    d = tmp_path_factory.mktemp("window")
    _, weights, com = B.torch_loop_window(d)
    return d, weights, com


@pytest.fixture(scope="module")
def bundle(fresh) -> tuple[dict, bytes]:
    doc, payload = _build(fresh, B.FIXTURE_DRAW, draw={"source": "fixture", "law": {"law": "subset", "k": len(B.FIXTURE_DRAW)}})
    return json.loads(json.dumps(doc)), payload


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
    assert [u["unit"] for u in doc["units"]] == list(B.FIXTURE_DRAW)
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
    assert {k for k, v in bad.items() if v == PATHS} == {f"unit:{u}/{p}" for u in B.FIXTURE_DRAW for p in ("x", "sx", "o0")}


def test_another_context_is_refused(bundle) -> None:
    doc, payload = bundle
    doc = copy.deepcopy(doc)
    doc["commitment"]["context"] = "11" * 64
    bad = B.check(doc, payload)
    assert bad["block:0"] and bad["step:0/0"] == "its domain is not the commit call's"
    assert {k for k, v in bad.items() if v == PATHS} == {f"unit:{u}/{p}" for u in B.FIXTURE_DRAW for p in ("x", "sx", "o0")}


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
                                     for u in B.FIXTURE_DRAW}


def test_a_tampered_output_is_refused_by_replay(bundle) -> None:
    """The replay alone (in `check` a changed output row is refused first by its string)."""
    doc, payload = bundle
    u = copy.deepcopy(doc["units"][1])
    o = next(r for r in u["rows"] if r["kind"] == "out")
    bad = bytearray(payload)
    bad[o["row"][0]] ^= 1
    assert B.replay(u, payload) and not B.replay(u, bytes(bad))


def _build(fresh, drawn, binding: str = "GEMM_SM120_S8", **kw) -> tuple[dict, bytes]:
    d, weights, com = fresh
    return B.build(d, drawn, commitment=com, locate=B.torch_loop_locate(128), binding=binding, reads=B.torch_loop_reads,
                   weights=kw.pop("weights", weights), **kw)


def test_the_fixture_is_its_digests(fresh, bundle) -> None:
    doc, payload = bundle
    assert B.digests(doc, payload) == GOLDEN
    assert B.check(doc, payload, registration=json.loads((fresh[0] / "registration.json").read_text())) == {}


def test_the_build_names_the_card_binding(fresh) -> None:
    """The H200's INT8 binding resolves to the s8 dot the templates bind; its BF16 binding does not, and is refused."""
    doc, payload = _build(fresh, [5], binding="GEMM_SM90_S8")
    assert doc["target"] == {"binding": "GEMM_SM90_S8", "card": "sm_90", "dot": "BlackwellS8ImmaDot32_v1"}
    assert B.check(doc, payload) == {}
    with pytest.raises(ValueError, match="binds BlackwellS8ImmaDot32_v1, not GEMM_HOPPER's HopperBF16WgmmaDot16_v1"):
        _build(fresh, [5], binding="GEMM_HOPPER")


def test_a_live_draw_without_its_receipt_is_not_built(fresh) -> None:
    with pytest.raises(ValueError, match="live-coin receipt"):
        _build(fresh, [5], draw={"source": "live"})
    doc, _ = _build(fresh, [5], draw={"source": "live", "receipt": {"challenger": "test"}})
    assert doc["draw"]["receipt"] == {"challenger": "test"}


def test_another_registration_is_refused(fresh, bundle) -> None:
    doc, payload = bundle
    reg = json.loads((fresh[0] / "registration.json").read_text())
    for r in reg["roots"]:
        r["root"] = "00" * 64
    bad = B.check(doc, payload, registration=reg)
    assert bad and all(v == "not the registration's root" for v in bad.values())


def _subset(k: int, units: list[int]) -> dict:
    return {"law": "subset", "population": 512, "k": k, "units": units}


def test_a_draw_under_another_law_than_the_registered_one_is_refused(fresh, tmp_path) -> None:
    """The law is the window's, registered before any coin: a subset:8 draw of a window registered subset:4 is refused,
    and a subset:2 draw of one registered subset:8, as is a draw of more units than its k or a record under another law."""
    reg4 = json.loads((fresh[0] / "registration.json").read_text())
    assert reg4["law"] == {"law": "subset", "k": len(B.FIXTURE_DRAW)}
    eight = [5, 60, 130, 200, 300, 390, 450, 511]
    doc, payload = _build(fresh, eight, draw={"source": "fixture", "law": {"law": "subset", "k": 8}})
    assert set(B.check(doc, payload, registration=reg4)) == {"draw:law"}

    d8 = tmp_path / "window-8"
    _, weights8, com8 = B.torch_loop_window(d8, law="subset:8")
    reg8 = json.loads((d8 / "registration.json").read_text())
    assert reg8["law"] == {"law": "subset", "k": 8}
    two = [5, 200]
    receipt = {"format": "verity/registration-receipt/v1", "challenger": "test"}
    doc, payload = _build((d8, weights8, com8), two, draw={"source": "live", "law": {"law": "subset", "k": 2},
                                                          "receipt": receipt, "record": _subset(2, two)})
    assert set(B.check(doc, payload, registration=reg8)) == {"draw:law", "draw:record"}
    doc, payload = _build((d8, weights8, com8), eight, draw={"source": "live", "law": {"law": "subset", "k": 8},
                                                            "receipt": receipt, "record": _subset(8, eight)})
    assert B.check(doc, payload, registration=reg8) == {}
    doc, payload = _build((d8, weights8, com8), eight[:7], draw={"source": "fixture", "law": {"law": "subset", "k": 8}})
    assert set(B.check(doc, payload, registration=reg8)) == {"draw:law"}


def test_weights_under_other_salts_are_refused(fresh) -> None:
    from verity_torch_loop import loop as L
    other = B.torch_loop_weights(L.Model(), hashlib.sha256(b"another test key").digest(), "torch-loop/mlp-w8a8-h128/seed0")
    with pytest.raises(ValueError, match="committed under its own salt"):
        _build(fresh, [5], weights=other)


def test_rows_under_other_step_salts_are_refused(fresh) -> None:
    d, weights, com = fresh
    other = B.Commitment(com.context, com.layout, com.slots,
                         {k: B.StepRecord(hashlib.sha256(b"another step key").digest(), r.rows) for k, r in com.steps.items()})
    with pytest.raises(ValueError, match="not the step commitment's row"):
        _build((d, weights, other), [5])


def test_a_tampered_loop_fails_replay(tmp_path) -> None:
    _, weights, com = B.torch_loop_window(tmp_path, tamper=True)
    doc, payload = _build((tmp_path, weights, com), [5, 300])
    assert B.check(doc, payload) == {u: "its committed outputs are not its Definition's on its committed inputs"
                                     for u in ("unit:5", "unit:300")}


#: unit 5 is slot 0, layer 0, output coordinate 5: layout unit 0 * 129 + 1 + 5 of slot 0 (`B.torch_loop_layout`)
STEP_INDEX = 6


@pytest.fixture(scope="module")
def step(fresh) -> tuple[dict, dict, bytes]:
    """The fixture's step registered (`B.step_registration`, its draw over every (slot, layout unit)), and the one-block
    bundle of drawn index `STEP_INDEX`."""
    _, weights, com = fresh
    (b, s), = com.steps
    reg = B.step_registration(com, weights, {"law": "subset", "k": 8}, run=B.RUN_ID, program="torch-loop/mlp-w8a8-h128/seed0",
                              block=b, step=s)
    doc, payload = _build(fresh, [5], draw={"source": "fixture", "law": {"law": "subset", "k": 1}})
    return reg, json.loads(json.dumps(doc)), payload


def test_a_drawn_block_holds_to_its_step_registration(step) -> None:
    reg, doc, payload = step
    assert reg["population"] == 2 * 258 and len(reg["slot_roots"]) == 2
    assert B.block_of(reg, STEP_INDEX) == (0, 6)
    assert B.check_step(doc, payload, reg, STEP_INDEX) == {}
    assert len(B.step_digest(reg)) == 128 and B.step_digest(reg) != B.step_digest({**reg, "population": reg["population"] + 1})


@pytest.mark.parametrize("port", ["x", "o0"])
@pytest.mark.parametrize("edit", [
    lambda s: s["path"].append("00" * 64),
    lambda s: s["slot_path"]["proof"].__setitem__(0, "00" * 64),
    lambda s: s["step_path"]["proof"].__setitem__(0, "00" * 64),
], ids=["unit-path", "slot-path", "step-path"])
def test_a_drawn_block_with_a_tampered_path_is_refused(step, edit, port) -> None:
    reg, doc, payload = step
    doc = copy.deepcopy(doc)
    edit(_row(doc, 5, port)["slot"])
    assert B.check_step(doc, payload, reg, STEP_INDEX) == {f"unit:5/{port}": PATHS}


def test_a_drawn_block_under_another_slot_root_is_refused(step) -> None:
    reg, doc, payload = step
    doc = copy.deepcopy(doc)
    _row(doc, 5, "sx")["slot"]["step_path"]["roots"][0] = reg["slot_roots"][1]
    assert B.check_step(doc, payload, reg, STEP_INDEX) == {"unit:5/sx": PATHS,
                                                          "step:sx": "it does not open under the registered step's slot root"}


@pytest.mark.parametrize("index", [STEP_INDEX + 1, STEP_INDEX + 258, 0])
def test_a_block_at_another_index_than_its_draw_is_refused(step, index) -> None:
    reg, doc, payload = step
    assert set(B.check_step(doc, payload, reg, index)) == {"step:index"}


def test_an_index_outside_the_registered_population_is_refused(step) -> None:
    reg, doc, payload = step
    assert set(B.check_step(doc, payload, reg, reg["population"])) == {"step"}
    assert set(B.check_step(doc, payload, {**reg, "population": reg["population"] - 1}, STEP_INDEX)) == {"step"}


def test_another_registered_step_is_refused(step) -> None:
    """A registration whose step root or slot roots are not the bundle's: the bundle's paths reach its own commitment, which
    is not the registered one."""
    reg, doc, payload = step
    assert B.check_step(doc, payload, {**reg, "step_root": "00" * 64}, STEP_INDEX) == {
        "step:root": "its commitment does not hold the registered step root"}
    other = {**reg, "slot_roots": ["00" * 64] * reg["slots"]}
    assert set(B.check_step(doc, payload, other, STEP_INDEX)) == {"step:x", "step:sx", "step:o0"}
    assert set(B.check_step(doc, payload, {**reg, "context": "11" * 64}, STEP_INDEX)) == {"step:commitment"}
    assert set(B.check_step(doc, payload, {**reg, "extra": 1}, STEP_INDEX)) == {"step"}


def test_a_registered_read_the_step_does_not_name_is_refused(step) -> None:
    reg, doc, payload = step
    wrong = [{**w, "root": "00" * 64} if w["value"] == "l0.wup" else w for w in reg["weights"]]
    assert B.check_step(doc, payload, {**reg, "weights": wrong}, STEP_INDEX) == {
        "step:wrow": "its registered value's root is not one the step registration names"}


def test_a_bundle_of_more_than_one_block_is_refused(fresh, step) -> None:
    reg, _, _ = step
    doc, payload = _build(fresh, [5, 6], draw={"source": "fixture", "law": {"law": "subset", "k": 2}})
    assert set(B.check_step(doc, payload, reg, STEP_INDEX)) == {"step:units"}


def test_a_step_whose_proof_blocks_are_some_of_its_units(fresh, step) -> None:
    """With `proof_units` (the coordinate units; each layer's quantizer unit is committed, not proved) the population is
    the slots' proof blocks only, and index b of a slot names its b-th proof unit."""
    _, weights, com = fresh
    (b, s), = com.steps
    pu = [u for u in range(com.layout.unit_count) if u % 129 != 0]
    reg = B.step_registration(com, weights, {"law": "subset", "k": 8}, run=B.RUN_ID, program="torch-loop/mlp-w8a8-h128/seed0",
                              block=b, step=s, proof_units=pu)
    _, doc, payload = step
    assert reg["population"] == 2 * 256 and B.block_of(reg, 5) == (0, 6) and B.block_of(reg, 256) == (1, 1)
    assert B.check_step(doc, payload, reg, 5) == {}
    assert set(B.check_step(doc, payload, reg, STEP_INDEX)) == {"step:index"}
    assert set(B.check_step(doc, payload, {**reg, "proof_units": pu[::-1]}, 5)) == {"step"}
    assert set(B.check_step(doc, payload, {**reg, "population": 2 * 258}, 5)) == {"step"}
