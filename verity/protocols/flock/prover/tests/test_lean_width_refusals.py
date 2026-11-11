"""The Lean decoder's width refusals against the codec's, on the NCI difftest's mistyped descriptors
(`desc_difftest_vectors.mistyped`) and a PoUW window's shape (`_tile_window`)."""
from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import pytest
from test_lean_verifier import LEAN, VERIFIER


NCI_SCRIPTS = Path(__file__).resolve().parents[3] / "inference_only" / "verifier" / "nci" / "lean" / "scripts"


def _tile_window(bits: bool) -> str:
    """A PoUW window's shape (`pouw_credit_cases.window_desc`): the root's 16-bit input batch, then 8 bits, and one call
    of a template over a row of 8 bits and a word, returning the word; the row's argument names the 8 bits (`bits`) or
    node 0's first word 8 times, reading a word as bits."""
    word, row = ["Value", 16], ["Array", 8, ["Value", 1]]
    root = {"params": [], "ret_type": word, "ret": {"strided": ["n", 2, 0, 1, 0, 0, 1]}, "nodes": [
        {"form": "batch", "fn": 0, "n": 2, "args": [], "axes": [], "out": 2},
        {"form": "batch", "fn": 2, "n": 8, "args": [], "axes": [], "out": 1},
        {"form": "call", "fn": 1, "out": 0, "args": [{"type": 1, "refs": {"strided": ["n", int(bits), 0, 1, 0, 0, 8]}},
                                                     {"type": 0, "refs": {"strided": ["n", 0, 0, 1, 0, 0, 1]}}]}]}
    tile = {"params": [["a", row], ["zy", word]], "ret_type": word, "ret": {"strided": ["p", 1, 0, 1, 0, 0, 1]},
            "nodes": []}
    return json.dumps({"annotations": {}, "callees": ["Input16_v1", "Tile_v1", "Input1_v1"], "definitions": {
        "Input16_v1": {"form": "primitive", "id": "Input16_v1", "params": [], "ret": word},
        "Input1_v1": {"form": "primitive", "id": "Input1_v1", "params": [], "ret": ["Value", 1]},
        "R_v1": {"form": "composite", "id": "R_v1", "bindings": {}, "body": root},
        "Tile_v1": {"form": "composite", "id": "Tile_v1", "bindings": {}, "body": tile}},
        "program_fn": "R_v1", "root": "R_v1", "schema": "verity-ir/descriptor/v1",
        "types": [word, row, ["Array", 2, word]]}, sort_keys=True, separators=(",", ":"))


@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_width_refusals_agree_with_the_codec(tmp_path):
    """The NCI difftest's 335 descriptors whose references name leaves of other widths than their types give them, and
    their well-typed neighbours (`desc_difftest_vectors.mistyped`), and a PoUW window's shape reading a word as a
    template's bits and its bits as them (`_tile_window`): the Lean decoder, which Flock's verifier, `NciCheck` and
    PoUW's `Credit.programOf` share, refuses as `E_REINTERPRET` exactly those the codec does, with the codec's message
    (`verity/protocols/flock/agreement/README.md`'s D22; `qword_program_agree.py --widths`)."""
    spec = importlib.util.spec_from_file_location("desc_difftest_vectors", NCI_SCRIPTS / "desc_difftest_vectors.py")
    V = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(V)
    with open(tmp_path / "widths.jsonl", "w") as f:
        for inst in [*V.mistyped(), {"program": _tile_window(False)}, {"program": _tile_window(True)}]:
            desc = json.loads(inst["program"])
            try:
                V.reference.codec.decode_program(desc, V.reference.declared(desc))
                why = None
            except ValueError as e:
                why = str(e)
            f.write(json.dumps({"program": inst["program"], "codec": why}) + "\n")
    subprocess.run(["lake", "build"], cwd=LEAN, check=True, capture_output=True)
    p = subprocess.run(["python3", str(VERIFIER / "qword_program_agree.py"), "--lean", str(LEAN / ".lake/build/bin/flock-verify"),
                        "--out", str(tmp_path / "out"), "--widths", str(tmp_path / "widths.jsonl")],
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    assert json.loads(p.stdout.strip().splitlines()[-1])["widths"] == {"both_refuse": 211, "neither": 126}
