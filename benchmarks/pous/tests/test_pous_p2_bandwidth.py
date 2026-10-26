"""The remote bandwidth session's prover over the P2 store (`p2_v1/bandwidth.py`): SHA-512's midstate and the device's
receipts held to hashlib and `p2.Sketcher`, the sketch of stored rows held to the unit sketch, the receipt tree built in
worker processes held to `receipts.tree`, PoUS's vk paths from row digests held to the verifier's tree, and the toy's
store served through kernels' staging decode (P2's reference on the CPU) to `p2_party`'s verifier in its own process:
accepted honest, rejected when the prover skips a layer's reads."""
from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from p2_v1 import bandwidth as B  # noqa: E402

from verity.core.protocols.pous import verifier as V  # noqa: E402
from verity.core.protocols.pous.schemes.p2 import P2Params  # noqa: E402
from verity.core.protocols.remote_bandwidth import p2, receipts  # noqa: E402

SEED = hashlib.sha512(b"test_pous_p2_bandwidth/seed").digest()


def test_the_midstate_and_one_block_are_sha512() -> None:
    assert B.IV512[0] == 0x6A09E667F3BCC908 and B.K512[79] == 0x6C44198C4A475817
    prefix = receipts.sketch_prefix(SEED)
    tail = bytes(range(72))
    block = tail + b"\x80" + bytes(128 - 72 - 1 - 16) + (8 * (len(prefix) + 72)).to_bytes(16, "big")
    state = B.compress(B.midstate(prefix), block)
    assert b"".join(x.to_bytes(8, "big") for x in state) == hashlib.sha512(prefix + tail).digest()


def test_the_devices_receipts_are_the_sketchers() -> None:
    rng = np.random.default_rng(1)
    sigma = torch.from_numpy(rng.integers(-(1 << 20), 1 << 20, (37, 16), dtype=np.int32))
    ys = B.receipts_of(torch, SEED, sigma, first=1000, chunk=8)
    sk = p2.Sketcher(SEED, 16)
    assert all(bytes(ys[k]) == sk.receipt(1000 + k, sigma[k].numpy().astype("<i4").tobytes()) for k in range(37))


def test_the_sketch_of_stored_rows_is_the_unit_sketch() -> None:
    rng = np.random.default_rng(2)
    units = 5
    rows = rng.integers(0, 256, (units * 4 + 3, p2.BLOCK_BYTES), dtype=np.uint8)
    sk = B.Sketch(torch, SEED, 16, units, torch.device("cpu"))
    t = torch.from_numpy(rows)
    sk.add(t[:7], 0)
    sk.add(t[7:], 7)
    assert sk.covered() == units and sk.blocks == 4 * units
    ref = p2.Sketcher(SEED, 16)
    for g in range(units):
        assert sk.sigma[g].numpy().astype("<i4").tobytes() == ref.sketch(rows[4 * g:4 * g + 4].tobytes())


@pytest.mark.parametrize("units", [8, 16 * 40 + 5])
def test_the_receipt_tree_is_receipts_tree(units, monkeypatch) -> None:
    rng = np.random.default_rng(units)
    ys = rng.integers(0, 256, (units, 64), dtype=np.uint8)
    n = receipts.leaves(units)
    salts = rng.integers(0, 256, n * receipts.SALT_BYTES, dtype=np.uint8).tobytes()
    plan = receipts.Plan.whole(units)
    ref = receipts.tree(SEED, plan, [bytes(y) for y in ys],
                        [salts[192 * c:192 * (c + 1)] for c in range(n)])
    monkeypatch.setattr(B.ReceiptTree, "CHUNK", 4)
    with ProcessPoolExecutor(2, mp_context=mp.get_context("fork")) as pool:
        got = B.ReceiptTree(SEED, units, ys, salts, pool)
    assert got.root == ref.commitment.root
    assert all(got.path(c) == ref.open(c).path for c in range(n))


def test_vk_paths_from_row_digests_are_the_verifiers(tmp_path) -> None:
    params = P2Params(16448, 21065, 4, 3, "sha512-ctr-v4")
    salt = bytes(range(24))
    rng = np.random.default_rng(3)
    rows = [rng.integers(0, 256, 64, dtype=np.uint8).tobytes() for _ in range(12)]
    (tmp_path / "seg").mkdir()
    (tmp_path / "rows").mkdir()
    from p2_v1 import store as ST

    for s in range(3):
        data = b"".join(rows[4 * s:4 * s + 4])
        (tmp_path / "rows" / f"{s:05d}.rows").write_bytes(data)
        (tmp_path / "seg" / f"{s:05d}.json").write_text(json.dumps({"root": ST.segment_root(params, salt, s, data).hex()}))
    tree = V.MerkleTree([V.row_leaf(params, salt, j, rows[j]) for j in range(12)])
    vk = B.VkPaths(tmp_path, params, salt, tree.root)
    assert vk.perfect and vk.paths(range(12)) == {j: tree.path(j) for j in range(12)}
    with pytest.raises(ValueError, match="vk root"):
        B.VkPaths(tmp_path, params, salt, bytes(64))


def test_the_toy_served_through_the_staging_decode_is_accepted_and_a_skipped_layer_rejected(tmp_path) -> None:
    reg = B.toy(tmp_path / "toy")
    assert B.registration_of_store(tmp_path / "toy" / "store", reg.node) == reg
    verifier = (f"{sys.executable} -m verity.core.protocols.remote_bandwidth.p2_party "
                f"--registration {tmp_path / 'toy' / 'registration.json'} --stdio")
    base = ["session", "--store", str(tmp_path / "toy" / "store"), "--layout", str(tmp_path / "toy" / "layout.json"),
            "--node", reg.node.decode(), "--registration", str(tmp_path / "toy" / "registration.json"),
            "--verifier", verifier, "--device", "cpu", "--ref-decode", "--samples", "64", "--window-s", "0.001",
            "--deadline-s", "0.001", "--workers", "2", "--chunk", "5"]
    assert B.main(base + ["--out", str(tmp_path / "honest"), "--expect", "accept"]) == 0
    doc = json.loads((tmp_path / "honest" / "session.json").read_text())
    a = doc["answer"]
    assert a["verdict"] == "accept" and a["receipts"] == 2 * 2 * reg.blocks
    assert a["q_bytes"] == (a["receipts"] - a["wrong_at_most"]) * p2.UNIT_BYTES
    assert [s["covered_units"] for s in doc["steps"]] == [reg.blocks] * 4
    assert doc["verdict_line"].startswith("BW VERDICT accept") and doc["verdict_line"].endswith("PASS")
    assert B.main(base + ["--out", str(tmp_path / "cheat"), "--skip-layer", "1", "--windows", "1", "--passes", "1",
                          "--expect", "reject"]) == 0
    a = json.loads((tmp_path / "cheat" / "session.json").read_text())["answer"]
    assert a["verdict"] == "reject" and a["failure"] == "receipt"


def test_the_cli_runs_as_a_script(tmp_path) -> None:
    p = subprocess.run([sys.executable, str(HERE / "p2_v1" / "bandwidth.py"), "toy", str(tmp_path)], capture_output=True,
                       text=True, cwd=ROOT)
    assert p.returncode == 0, p.stderr
    assert json.loads(p.stdout)["units"] == 8
