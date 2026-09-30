"""`pearlc_activations.py --replay` reads a capture without torch: BF16 safetensors bit for bit, the manifest's sha256s, and
--split's unit (S moved out, the rest zero-padded to a multiple of 128) on a spiked channel."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("pouw_pearlc_activations", Path(__file__).parents[1] / "pearlc_activations.py")
A = importlib.util.module_from_spec(spec)
sys.modules["pouw_pearlc_activations"] = A
spec.loader.exec_module(A)

K = 128
SPIKE = 37


def bf16(x: np.ndarray) -> tuple[np.ndarray, bytes]:
    """x truncated to bf16 (as float32) and its little-endian bf16 bytes."""
    bits = (x.astype(np.float32).view(np.uint32) >> 16).astype("<u2")
    return (bits.astype(np.uint32) << 16).view(np.float32), bits.tobytes()


def write_safetensors(path: Path, tensors: dict[str, tuple[str, list[int], bytes]]) -> None:
    header, off = {}, 0
    for name, (dtype, shape, raw) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [off, off + len(raw)]}
        off += len(raw)
    h = json.dumps(header).encode()
    path.write_bytes(len(h).to_bytes(8, "little") + h + b"".join(raw for _, _, raw in tensors.values()))


def capture(tmp: Path) -> Path:
    rng = np.random.default_rng(7)
    X = rng.standard_normal((64, K)).astype(np.float32)
    X[:, SPIKE] = 1000.0
    W, wraw = bf16(rng.standard_normal((8, K)))
    snap = tmp / "snap"
    snap.mkdir()
    write_safetensors(snap / "shard.safetensors", {"model.layers.0.mlp.down_proj.weight": ("BF16", [8, K], wraw)})
    (snap / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"model.layers.0.mlp.down_proj.weight": "shard.safetensors"}}))
    rd = tmp / "cap"
    rd.mkdir()
    np.save(rd / "L000-down.npy", X)
    man = {"model": "tiny", "revision": "r0", "snapshot": str(snap), "window": 0, "tokens": 64,
           "inputs": {"model.layers.0.mlp.down_proj": {"file": "L000-down.npy", "sha256": A.sha256(rd / "L000-down.npy")}}}
    (rd / "manifest.json").write_text(json.dumps(man))
    return rd


def test_safetensor_reads_bf16_and_f32_exactly(tmp_path):
    x, raw = bf16(np.array([[1.0, -2.5, 3.140625], [1e-30, -0.0, 65280.0]]))
    y = np.arange(6, dtype="<f4").reshape(3, 2)
    write_safetensors(tmp_path / "t.safetensors", {"x": ("BF16", [2, 3], raw), "y": ("F32", [3, 2], y.tobytes())})
    assert A.safetensor(tmp_path / "t.safetensors", "x").tobytes() == x.tobytes()
    assert np.array_equal(A.safetensor(tmp_path / "t.safetensors", "y"), y)


def test_replay_split_moves_the_spike_and_pads(tmp_path, monkeypatch):
    rd = capture(tmp_path)
    out = tmp_path / "out.json"
    monkeypatch.setattr(sys, "argv", ["pearlc_activations.py", "--replay", str(rd), "--split", "--atom", "sm120", "--G", "4",
                                      "--tiles", "1", "--cross-words", "1", "--out", str(out)])
    A.main()
    doc = json.loads(out.read_text())
    (rec,) = doc["layers"]
    assert doc["revision"] == "r0" and doc["model"] == "tiny" and doc["split"]
    assert rec["A_rows_fail"] == 64 and not rec["unit_admissible_unsplit"]
    assert rec["split_budget"] == 1 and rec["split_channels"] == 1
    assert rec["unit_admissible_split"] and rec["k_split"] == K - 1 and rec["k_padded"] == K
    assert len(rec["tiles"]) == 1 and "rules_ok_padded" in rec["tiles"][0]


def test_replay_refuses_a_changed_capture(tmp_path):
    rd = capture(tmp_path)
    X = np.load(rd / "L000-down.npy")
    X[0, 0] += 1
    np.save(rd / "L000-down.npy", X)
    with pytest.raises(SystemExit, match="sha256"):
        list(A.replay(rd, ""))
