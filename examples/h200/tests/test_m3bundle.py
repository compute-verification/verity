"""M3 proper's sampled proof blocks (``m3bundle.py``) at the small mini-rows shape: a sample of one block of every family the step
registers, through the weights, replay and build phases, bundles and checks every family but the gathers (references in
the Program, with no Definition: visible failures), among them PV past n (V rows of no reads), the norms (their gain one
registered row a word) and the embedding (its token the argmax its sequence committed a step before); a registered piece
off its value's root, a value the step registration doesn't name, and a nonzero word in a row of no reads are refused."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import m3bundle as MB
import m3draw as M3
import m3stepreg as SR
import pytest
from e2e_h200 import weights_commit as WC
from e2e_h200.shape import SHAPES
from verity_e2e_demo import replay_bundle as RB

SHAPE = "mini-rows"
FLAGS = ["--shape", SHAPE, "--batch", "2", "--steps", "3", "--prompt-bytes", "2", "--workers", "1"]
PROGRAM = "ab" * 64


@pytest.fixture(scope="module")
def tree(tmp_path_factory) -> tuple[Path, dict]:
    d = tmp_path_factory.mktemp("m3")
    I = lambda n: str(d / n)  # noqa: E731
    assert M3.main(["commit", "--out", I("c"), "--secret", I("secret"), "--new-secret", *FLAGS]) == 0
    weights = {"program": PROGRAM, "registered": WC.commit(SHAPES[SHAPE], WC.seed_of("test"), PROGRAM, d / "w")["registered"]}
    fams = MB._families(SHAPES[SHAPE])
    reg = SR.build(json.loads((d / "c" / "roots.json").read_text()), weights, PROGRAM, "test", len(fams))
    units = sorted((j % 2) * reg["units_per_slot"] + base + (count - 1) // 2 for j, (base, _, count) in enumerate(fams))
    draw = {"law": "subset", "k": len(fams), "population": reg["population"], "units": units}
    receipt = {"format": "verity/registration-receipt/v1", "registration": RB.step_digest(reg), "verifier": "test",
               "received": "2026-10-27T00:00:00Z"}
    for n, doc in (("reg.json", reg), ("draw.json", draw), ("receipt.json", receipt)):
        (d / n).write_text(json.dumps(doc))
    common = ["--registration", I("reg.json"), "--draw", I("draw.json")]
    assert MB.main(["weights", "--state", I("w"), *common, "--out", I("open.json")]) == 0
    assert MB.main(["replay", "--private", I("priv"), "--secret", I("secret"), "--registered", I("c/roots.json"),
                    *common, *FLAGS]) == 0
    assert MB.main(["build", "--private", I("priv"), *common, "--receipt", I("receipt.json"), "--weights", I("open.json"),
                    "--out", I("tree")]) == 3
    return d / "tree", reg


def _block(tree: Path, summary: dict, family: str) -> tuple[dict, bytes, dict, int]:
    (b,) = [b for b in summary["blocks"] if b["family"] == family]
    d = tree / "blocks" / str(b["g"])
    return (json.loads((d / "bundle.json").read_text()), (d / "bundle.bin").read_bytes(),
            json.loads((d / "window" / "registration.json").read_text()), b["g"])


@pytest.mark.slow
def test_every_family_but_the_gathers_bundles_and_checks(tree) -> None:
    """The one test that builds every family's bundle: a family whose ports the bundle can't express (PV past n, a norm's
    gain, the embedding's token, all visible failures before) fails only here; it costs about 90 s, a third of it the
    softmax's replays."""
    t, _ = tree
    s = json.loads((t / "summary.json").read_text())
    gathers = {g for g, why in s["visible_failures"].items() if "gather" in why}
    assert gathers and set(s["visible_failures"]) == gathers
    assert len(s["blocks"]) + len(gathers) == len(s["draw"])
    assert all(b["check"] == {} and b["check_step"] == {} and all(b["refused"].values()) for b in s["blocks"])
    by = {b["map_template"]: b for b in s["blocks"]}
    assert set(by) >= {"PvRow_v1", "RmsNormOutSums_v1", "EmbeddingCoordinate_v1", "QkSoftmaxRowPad_v1",
                       "ScaledMmS8CoordinateRows_v1", "QuantChunkS8_v1", "RowAbsMax_v1", "Bf16AddRow_v2",
                       "RmsSumSqChunks_v1", "Argmax_v2"}
    assert by["PvRow_v1"]["ports"]["v31"] == "0 reads" and by["PvRow_v1"]["read_steps"] == [0, 1, 2]
    assert by["RmsNormOutSums_v1"]["ports"]["w"] == "32 reads" and len(by["RmsNormOutSums_v1"]["registered"]) == 32
    assert by["EmbeddingCoordinate_v1"]["ports"]["tok"] == "1 reads" and by["EmbeddingCoordinate_v1"]["read_steps"] == [1]


def _repay(doc: dict, pay: bytes) -> dict:
    doc = copy.deepcopy(doc)
    doc["payload"]["sha512"] = hashlib.sha512(pay).hexdigest()
    return doc


@pytest.mark.slow
def test_a_registered_piece_and_a_row_of_no_reads_are_held_to_their_rules(tree) -> None:
    t, reg = tree
    s = json.loads((t / "summary.json").read_text())
    doc, pay, rec, g = _block(t, s, "l0.attn_norm")
    w = next(r for r in doc["units"][0]["rows"] if r["port"] == "w")
    assert {q["kind"] for q in w["reads"]} == {"registered"}
    bad = copy.deepcopy(doc)
    q = next(r for r in bad["units"][0]["rows"] if r["port"] == "w")["reads"][3]
    q["path"] = q["path"][:-2] + ("00" if q["path"][-2:] != "00" else "01")
    assert set(RB.check(bad, pay, registration=rec)) == {f"unit:{doc['units'][0]['unit']}/w"}
    other = copy.deepcopy(reg)
    other["weights"] = [e for e in reg["weights"] if e["value"] != "l0.attn_norm"]
    assert "step:w" in RB.check_step(doc, pay, other, g, registration=rec)

    doc, pay, rec, _ = _block(t, s, "l0.pv")
    v = next(r for r in doc["units"][0]["rows"] if r["port"] == "v31")
    assert v["reads"] == [] and not any(pay[v["row"][0]:v["row"][0] + v["row"][1]])
    tampered = bytearray(pay)
    tampered[v["row"][0]] ^= 1
    assert f"unit:{doc['units'][0]['unit']}/v31" in RB.check(_repay(doc, bytes(tampered)), bytes(tampered), registration=rec)
