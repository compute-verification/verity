"""The remote bandwidth session's prover over the P2 store (`p2_v1/bandwidth.py`): SHA-512's midstate and the device's
receipts held to hashlib and `p2.Sketcher`, the sketch of stored rows held to the unit sketch, the receipt tree built in
worker processes held to `receipts.tree`, PoUS's vk paths from row digests held to the verifier's tree, the plan of a
layout's tensors' units (the tracked registration's, the real layout's), and the toy's store served through
kernels' staging decode (P2's reference on the CPU) to `p2_party`'s verifier in its own process: accepted honest, under
the toy's plan or one of fewer units, and rejected when the prover skips a layer's reads."""
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
from p2_v1 import bandwidth as B

from verity.core.protocols.pous import verifier as V
from verity.core.protocols.pous.schemes.p2 import P2Params
from verity.core.protocols.remote_bandwidth import p2, p2_party, receipts

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
    assert sk.covered(((0, 2), (3, 2))) == 4
    ref = p2.Sketcher(SEED, 16)
    for g in range(units):
        assert sk.sigma[g].numpy().astype("<i4").tobytes() == ref.sketch(rows[4 * g:4 * g + 4].tobytes())


@pytest.mark.parametrize(("units", "runs"), [(8, None), (16 * 40 + 5, None), (16 * 40 + 5, ((100, 500), (0, 50)))])
def test_the_receipt_tree_is_receipts_tree(units, runs, monkeypatch) -> None:
    plan = receipts.Plan.whole(units) if runs is None else receipts.Plan(runs, units)
    rng = np.random.default_rng(units)
    ys = rng.integers(0, 256, (plan.length, 64), dtype=np.uint8)
    n = receipts.leaves(plan.length)
    salts = rng.integers(0, 256, n * receipts.SALT_BYTES, dtype=np.uint8).tobytes()
    ref = receipts.tree(SEED, plan, [bytes(y) for y in ys],
                        [salts[192 * c:192 * (c + 1)] for c in range(n)])
    monkeypatch.setattr(B.ReceiptTree, "CHUNK", 4)
    with ProcessPoolExecutor(2, mp_context=mp.get_context("fork")) as pool:
        got = B.ReceiptTree(SEED, plan, ys, salts, pool)
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


def test_the_plan_is_the_units_holding_a_tensors_block() -> None:
    cols = ["name", "layer", "n", "k", "offset", "size", "first_block", "blocks", "staging_offset"]
    rows = [[name, 0, 0, 0, 0, 0, first, blocks, 0] for name, first, blocks in
            (("a", 0, 6), ("b", 6, 2), ("c", 8, 3), ("d", 21, 1))]
    plan = B.tensor_plan({"layout": {"tensor_columns": cols, "tensors": rows}}, 10)
    assert plan == receipts.Plan(((0, 3), (5, 1)), 10)


def test_the_tracked_registrations_plan_is_the_real_layouts_tensors_units() -> None:
    """A pass over mem69-v2 reads the units of the 209 tensors' blocks, not the 12,745 blocks of padding past
    lm_head (PoUS's, which the decode never reads): the 3,186 units wholly padding are out, lm_head's last unit, three
    of its blocks and one of padding, in."""
    from p2_v1 import store as ST

    doc = ST._dump(ST.build(ST.MQA16K).to_json())
    assert hashlib.sha512(doc.encode()).hexdigest() == ST.LAYOUT_SHA512
    layout = json.loads(doc)
    tracked = json.loads((HERE / "p2_v1" / "bandwidth-mem69-v2.json").read_text())
    reg = p2_party.registration_of(tracked)
    plan = p2_party.plan_of(tracked, reg)
    assert plan == B.tensor_plan(layout, reg.blocks) == receipts.Plan(((0, 17_003_406),), reg.blocks)
    cols = layout["tensor_columns"]
    end = max(r[cols.index("first_block")] + r[cols.index("blocks")] for r in layout["tensors"])
    assert (len(layout["tensors"]), end, reg.codec.blocks - end) == (209, 68_013_623, 12_745)
    assert (reg.blocks - plan.length, 4 * plan.length - end) == (3_186, 1)


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
    assert a["verdict"] == "accept" and a["receipts"] == 2 * 2 * reg.blocks and a["plan_units"] == reg.blocks
    assert a["q_bytes"] == (a["receipts"] - a["wrong_at_most"]) * p2.UNIT_BYTES
    assert [s["covered_units"] for s in doc["steps"]] == [reg.blocks] * 4
    assert doc["verdict_line"].startswith("BW VERDICT accept") and doc["verdict_line"].endswith("PASS")
    assert B.main(base + ["--out", str(tmp_path / "cheat"), "--skip-layer", "1", "--windows", "1", "--passes", "1",
                          "--expect", "reject"]) == 0
    a = json.loads((tmp_path / "cheat" / "session.json").read_text())["answer"]
    assert a["verdict"] == "reject" and a["failure"] == "receipt"


def test_a_registration_whose_plan_is_fewer_units_is_what_each_pass_reads(tmp_path) -> None:
    """The verifier's plan, sent with its registration, is what the prover reads and commits: the toy's units 0, 1 and
    4 to 6, every receipt drawn, accepted with those receipts alone."""
    reg = B.toy(tmp_path / "toy")
    doc = json.loads((tmp_path / "toy" / "registration.json").read_text())
    assert doc["plan"] == [[0, reg.blocks]]
    path = tmp_path / "fewer.json"
    path.write_text(json.dumps(doc | {"plan": [[0, 2], [4, 3]]}))
    verifier = f"{sys.executable} -m verity.core.protocols.remote_bandwidth.p2_party --registration {path} --stdio"
    assert B.main(["session", "--store", str(tmp_path / "toy" / "store"), "--layout", str(tmp_path / "toy" / "layout.json"),
                   "--node", reg.node.decode(), "--registration", str(path), "--verifier", verifier, "--device", "cpu",
                   "--ref-decode", "--samples", "64", "--window-s", "0.001", "--deadline-s", "0.001", "--workers", "2",
                   "--chunk", "5", "--out", str(tmp_path / "fewer"), "--expect", "accept"]) == 0
    doc = json.loads((tmp_path / "fewer" / "session.json").read_text())
    a = doc["answer"]
    assert a["verdict"] == "accept" and (a["units"], a["plan_units"], a["receipts"]) == (reg.blocks, 5, 2 * 2 * 5)
    assert [s["covered_units"] for s in doc["steps"]] == [5] * 4 and doc["registration"]["plan"] == [[0, 2], [4, 3]]


def test_sessions_reach_a_listening_verifier_which_records_their_answers(tmp_path) -> None:
    """The gate's shape: `p2_party --listen --sessions 2` on the auditor, the honest prover and then the cheat at
    `--connect`, each answer a line of `--answer`."""
    reg = B.toy(tmp_path / "toy")
    verifier = subprocess.Popen([sys.executable, "-m", "verity.core.protocols.remote_bandwidth.p2_party",
                                 "--registration", str(tmp_path / "toy" / "registration.json"), "--listen",
                                 "127.0.0.1:0", "--sessions", "2", "--answer", str(tmp_path / "answers.jsonl")],
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, cwd=ROOT)
    port = json.loads(verifier.stdout.readline())["listening"]
    base = ["session", "--store", str(tmp_path / "toy" / "store"), "--layout", str(tmp_path / "toy" / "layout.json"),
            "--node", reg.node.decode(), "--connect", f"127.0.0.1:{port}", "--device", "cpu", "--ref-decode",
            "--samples", "8", "--windows", "1", "--passes", "1", "--window-s", "0.001", "--deadline-s", "0.001",
            "--workers", "1"]
    assert B.main(base + ["--out", str(tmp_path / "honest"), "--expect", "accept"]) == 0
    assert B.main(base + ["--out", str(tmp_path / "cheat"), "--skip-layer", "1", "--expect", "reject"]) == 0
    assert verifier.wait() == 0
    mine = [json.loads(x) for x in (tmp_path / "answers.jsonl").read_text().splitlines()]
    assert mine == [json.loads((tmp_path / s / "session.json").read_text())["answer"] for s in ("honest", "cheat")]
    assert mine[0]["verdict"] == "accept" and mine[0]["receipts"] == reg.blocks
    assert mine[1]["verdict"] == "reject" and mine[1]["failure"] == "receipt"


def test_a_party_the_gate_starts_says_its_port_and_its_own_record_of_the_answer(tmp_path) -> None:
    """The auditor gate's party shape (`--party CMD`): the party run as the gate runs one, from the tree with no
    site-packages (`-S`: the standard library only, which the gate's `python3 -s -E` has) and its `--device`
    directory; CMD's stdout says the port and then the verifier's record of the answer, which the verdict line holds
    to the session's, and which the device's answers.jsonl keeps."""
    reg = B.toy(tmp_path / "toy")
    party = (f"cd {ROOT} && exec {sys.executable} -S -E -m verity.core.protocols.remote_bandwidth.p2_party "
             f"--device {tmp_path / 'party'} --listen 127.0.0.1:0 "
             f"--registration {tmp_path / 'toy' / 'registration.json'}")
    base = ["session", "--store", str(tmp_path / "toy" / "store"), "--layout", str(tmp_path / "toy" / "layout.json"),
            "--node", reg.node.decode(), "--party", party, "--party-host", "127.0.0.1", "--device", "cpu",
            "--ref-decode", "--samples", "8", "--windows", "1", "--passes", "1", "--window-s", "0.001",
            "--deadline-s", "0.001", "--workers", "1"]
    for name, more, expect in (("honest", [], "accept"), ("cheat", ["--skip-layer", "1"], "reject")):
        assert B.main(base + more + ["--out", str(tmp_path / name), "--expect", expect]) == 0
        doc = json.loads((tmp_path / name / "session.json").read_text())
        assert doc["verifier_record"] == doc["answer"] and doc["answer"]["verdict"] == expect
        assert "verifier_record_agrees=True" in doc["verdict_line"]
    kept = [json.loads(x) for x in (tmp_path / "party" / "answers.jsonl").read_text().splitlines()]
    assert kept == [json.loads((tmp_path / s / "session.json").read_text())["answer"] for s in ("honest", "cheat")]
    record = {"verdict": "accept"}
    assert B.verdict_line(doc["answer"], "reject", record).endswith("verifier_record_agrees=False FAIL")


KILLED = """
import json, socket
with socket.create_server(("127.0.0.1", 0)) as srv:
    print(json.dumps({"listening": srv.getsockname()[1]}), flush=True)
    conn, _ = srv.accept()
    conn.makefile("rb").readline()
    conn.close()
print(json.dumps({"killed": "the session's 1 min ran out"}), flush=True)
"""


def test_a_party_the_gate_kills_fails_the_session_naming_the_kill(tmp_path) -> None:
    """A party whose minutes ran out: the gate closes its session and says so on its stdout (`{"killed"}`), and the
    session fails with that reason, not a bare closed connection."""
    reg = B.toy(tmp_path / "toy")
    (tmp_path / "killed.py").write_text(KILLED)
    out = tmp_path / "killed"
    assert B.main(["session", "--store", str(tmp_path / "toy" / "store"), "--layout", str(tmp_path / "toy" / "layout.json"),
                   "--node", reg.node.decode(), "--party", f"{sys.executable} {tmp_path / 'killed.py'}",
                   "--party-host", "127.0.0.1", "--device", "cpu", "--ref-decode", "--workers", "1", "--out", str(out),
                   "--expect", "accept"]) == 1
    doc = json.loads((out / "session.json").read_text())
    assert doc["answer"]["verdict"] == "prover-error", doc["answer"]
    assert doc["answer"]["reason"] == ("ConnectionError: the auditor's gate killed the party: the session's 1 min ran "
                                       "out")


def test_the_cli_runs_as_a_script(tmp_path) -> None:
    p = subprocess.run([sys.executable, str(HERE / "p2_v1" / "bandwidth.py"), "toy", str(tmp_path)], check=False,
                       capture_output=True, text=True, cwd=ROOT)
    assert p.returncode == 0, p.stderr
    assert json.loads(p.stdout)["units"] == 8
