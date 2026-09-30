"""benchmarks/pouw/pearl_c4/real.py, the verifier on a `pearlc_capture.py` directory, on a synthetic one-linear capture: the
stand-in split removes the massive-activation channel that fails D-NF, a chunk out of budget exits 99 with its tasks'
checkpoints, and the next run finishes the tile's replayed debit under the cap."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location("pearl_c4_real", Path(__file__).resolve().parents[1] / "pearl_c4" / "real.py")
R = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = R
_spec.loader.exec_module(R)

NAME, K, M, N = "model.layers.0.self_attn.q_proj", 1024, 128, 128


def bf16(a: np.ndarray) -> np.ndarray:
    """float32 values rounded to BF16 (nearest even), as BF16 codes."""
    bits = a.astype(np.float32).view(np.uint32).astype(np.uint64)
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def capture(root: Path) -> Path:
    rng = np.random.default_rng(3)
    x = bf16(rng.standard_normal((M, K)))
    x[:, 5] = bf16(np.full(M, 2.0 ** 16))                    # a massive activation at an unsampled position
    cap, snap = root / "cap", root / "snap"
    cap.mkdir()
    snap.mkdir()
    np.save(cap / "L000-attn.npy", (x.astype(np.uint32) << 16).view(np.float32))
    w = bf16(rng.standard_normal((N, K)) * 0.02).tobytes()
    header = json.dumps({NAME + ".weight": {"dtype": "BF16", "shape": [N, K], "data_offsets": [0, len(w)]}}).encode()
    (snap / "model.safetensors").write_bytes(len(header).to_bytes(8, "little") + header + w)
    (snap / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {NAME + ".weight": "model.safetensors"}}))
    entry = {"file": "L000-attn.npy", "sha256": hashlib.sha256((cap / "L000-attn.npy").read_bytes()).hexdigest(), "shape": [M, K]}
    (cap / "manifest.json").write_text(json.dumps({"model": "synthetic", "revision": "0", "snapshot": str(snap),
                                                   "inputs": {NAME: entry}}))
    return cap


def test_real_activation_replay(tmp_path):
    cap, out = capture(tmp_path), tmp_path / "out"
    args = [str(cap), str(out), "--tiles", "1", "--workers", "2"]
    assert R.main(args + ["--budget-min", "0"]) == 99                      # one task started, then out of budget
    assert len(list(out.glob("admit-*.json")) + list(out.glob("tile-*.json"))) == 1
    split = json.loads((out / "split-L000-attn.json").read_text())
    assert len(split["side"]) == K // 100 and 5 in split["side"]
    assert R.main(args) == 0
    s = json.loads((out / "summary.json").read_text())
    x, w = s["inputs"]["L000-attn.npy"], s["weights"][NAME]
    assert x["rows"] == M and x["dnf_fail_full"] == M and x["dnf_fail_split"] == 0 and x["non_finite"] == 0
    assert w["rows"] == N and w["dnf_fail_split"] == 0
    (t,) = s["tiles"][NAME]
    assert t["rejected"] == "" and t["credited"] == [64, 64] and t["k_kept"] == K - K // 100 and t["k_unit"] == K
    assert s["tiles_debited"] == 1 and s["over_cap_tiles"] == 0 and 0 <= t["ratio"] < 1
