"""The recursive session's inner statement as `pod/rec_inner.py stage` writes it: bound to the `verity/partition/v1` object
beside it (`partition.json`, over `program.json`), which the Lean verifier needs to set the inner statement up at all
(`verify --rec-inner-partition`, V*'s check 5; `rec-inner --partition`)."""
from __future__ import annotations

import hashlib
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


def _window(src: Path, n: int, part: dict, d: Path) -> bytes:
    """A one-member window (`stage_window`'s input) whose bytes are a staged statement's own: its rows and salts as the
    member's ``m0-private.bin``, its ``b || c`` as ``m0-public.bin``."""
    meta = json.loads((src / "circuit.txt").read_text().split("\n", 2)[1][5:])
    head, body = (src / f"inst-{n}.bin").read_bytes().split(b"\n", 1)
    public = (src / f"pub-{n}.bin").read_bytes().split(b"\n", 1)[1]
    assert body.endswith(public)
    d.mkdir()
    (d / "m0-private.bin").write_bytes(body[:-len(public)])
    (d / "m0-public.bin").write_bytes(public)
    mem = {"k": 0, "template": next(iter(meta["program_digests"])), "n": n, "base": 0,
           "ports": [list(p[:2]) for p in meta["ports"]], "input_ports": meta["input_ports"],
           "out_ports": [list(p) for p in meta["out_ports"]], "frame_v3": json.loads(head)["frame_v3"],
           "public_sha512": hashlib.sha512(public).hexdigest(),
           **({"row_schemas": meta["row_schemas"]} if "row_schemas" in meta else {})}
    (d / "window.json").write_text(json.dumps({"run": "r-test", "registration_digest": "0" * 128, "members": [mem],
                                               "partition": part, "partition_digest": P.digest(part)}))
    (d / "registration.json").write_text(json.dumps({"window": {"run": "r-test"}}))
    return body


@pytest.mark.slow
def test_a_served_windows_member_stages_as_the_inner_statement_over_its_own_bytes(tmp_path):
    """`stage_window` composes the member's circuit itself (plain leaves, and hm96 for ``--zk-out``) and takes the bodies
    from the window's files alone: the same rows, salts and ``b || c`` the window commits, so its public body is the window's
    ``m0-public.bin``; a window whose partition is not the member's population's is refused."""
    ri, sub, n = _rec_inner(), "silu-mul-v2/i256/bf16", 3
    src, sets = tmp_path / "src", tmp_path / "sets"
    ri.stage(n, sets, src, plain_leaves=True, subcircuit=sub)
    part = json.loads((src / "partition.json").read_text())
    body = _window(src, n, part, tmp_path / "window")
    out, zk = tmp_path / "stage", tmp_path / "stage-zk"
    res = ri.stage_window(tmp_path / "window", 0, sub, sets, out, zk)
    for d in (out, zk):
        assert (d / f"inst-{n}.bin").read_bytes().split(b"\n", 1)[1] == body
        assert json.loads((d / "partition.json").read_text()) == part
        assert json.loads((d / "draw.json").read_text()) == json.loads((src / "draw.json").read_text())
    assert (out / "circuit.txt").read_text() == (src / "circuit.txt").read_text()
    assert res["leaf_scheme"] == "flock-leaf/sha512-unsalted" and res["n"] == n and res["subcircuit"] == sub
    assert res["public_body"] == res["window"]["public_sha512"] and res["zk_circuit_sha512"] != res["circuit_sha512"]
    with pytest.raises(ValueError, match="not the window's"):
        ri.stage_window(tmp_path / "window", 0, "silu-mul-v2/i512/bf16", sets, tmp_path / "other")
    if not BIN.exists():
        pytest.skip("the Lean verifier is not built")
    p = subprocess.run([str(BIN), "rec-inner", "--circuit", str(out / "circuit.txt"), "--public", str(out / f"pub-{n}.bin"),
                        "--partition", str(out / "partition.json"), "--program", str(out / "program.json")],
                       capture_output=True, text=True)
    assert p.returncode == 0 and json.loads(p.stdout)["format"] == "verity/rec-inner/v1", p.stderr
