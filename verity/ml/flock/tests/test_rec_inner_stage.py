"""The recursive session's inner statement as `pod/rec_inner.py stage` writes it: bound to the `verity/partition/v1` object
beside it (`partition.json`, over `program.json`), which the Lean verifier needs to set the inner statement up at all
(`verify --rec-inner-partition`, V*'s check 5; `rec-inner --partition`)."""
from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest
from test_rec_outer import BIN
from verity.core.protocols.sampled_proofs.one_stage import partition as P
from verity.ml.circuits.codec import program_digest

POD = Path(__file__).resolve().parents[1] / "pod"


def _rec_inner():
    spec = importlib.util.spec_from_file_location("rec_inner", POD / "rec_inner.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.slow
def test_the_inner_statement_binds_the_partition_object_beside_it(tmp_path):
    out = tmp_path / "stage"
    res = _rec_inner().stage(2, tmp_path / "sets", out, plain_leaves=True)
    meta = json.loads((out / "circuit.txt").read_text().split("\n", 2)[1][5:])
    part = json.loads((out / "partition.json").read_text())
    program = json.loads((out / "program.json").read_text())
    assert meta["partition"] == {"rule": P.FORMAT, "digest": P.digest(part)} and meta["program_sha512"] == part["program"]
    assert program_digest(program) == part["program"]
    assert part["query"] == {"name": "Q_template_instance", "version": 0, "params": {"template": next(iter(meta["program_digests"]))}}
    assert (res["partition_digest"], res["program_sha512"]) == (P.digest(part), part["program"])
    if not BIN.exists():
        pytest.skip("the Lean verifier is not built")
    inner = [str(BIN), "rec-inner", "--circuit", str(out / "circuit.txt"), "--public", str(out / "pub-2.bin")]
    p = subprocess.run([*inner, "--partition", str(out / "partition.json"), "--program", str(out / "program.json")],
                       capture_output=True, text=True)
    assert p.returncode == 0 and json.loads(p.stdout)["format"] == "verity/rec-inner/v2", p.stderr
    p = subprocess.run(inner, capture_output=True, text=True)
    assert p.returncode == 1 and "the verifier's partition object is required" in p.stderr
