"""The opt-in serving rows (`commit.serving_rows`): M0's serving row leaf vector byte for byte, core's frame-v3-sha512 / hm96-sha512
reference on the batch and parallel paths, ChaCha20 salts, the agreed domain rule, the registration shape and M0's file layout.
`serving_row_leaf_vector.json` is flock-netlist's vector (notes-asset flock-netlist/serving-row-leaf-vector.json, sha256 92aa7f77...)."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pytest

from verity.commitments import hm96
from verity.commitments.frame_v3 import FrameV3, Port, RowLeaf, WordLeaf
from verity.commitments.identity import identity_digest
from verity.commitments.indexed import RangeIndexedDomain
from verity.commitments.merkle import CommitmentDomain
from verity.commitments.rowleaf import ROLE_X, row_leaf, sha512_row_layout
from verity_vllm.commit import serving_rows as SR

VECTOR = json.loads((Path(__file__).parent / "serving_row_leaf_vector.json").read_text())
PARTITION = {"program": "ab" * 64, "query": {"name": "Q_template_instance", "version": 0, "template": "RoPEHead_v1{D=64}", "width_rule": None}}


def window(n: int, run: str = "r-test") -> SR.Window:
    return SR.Window(PARTITION, "cd" * 64, n, run)


def heads(n: int, seed: int = 3) -> tuple[dict[str, np.ndarray], np.ndarray]:
    rng = np.random.default_rng(seed)
    return {p: rng.integers(0, 1 << 16, size=(n, SR.D), dtype=np.uint16) for p in SR.IN_PORTS}, rng.integers(0, 1 << 16, size=(n, SR.D), dtype=np.uint16)


def reference(c: SR.Committed) -> dict[str, bytes]:
    """Core's slow path: row_leaf over hm96.SHA512.tree_leaf(commit_string(...)), word leaves, FrameV3.root."""
    inst, key = hm96.SHA512, hm96.SHA512.default_key
    fv = FrameV3({**{p: Port(c.domains[p], RowLeaf("sha512", ROLE_X, 16, SR.D)) for p in SR.IN_PORTS},
                  SR.OUT_PORT: Port(c.domains[SR.OUT_PORT], WordLeaf("u16", 2))})
    roots = {}
    for p in SR.IN_PORTS:
        lay = sha512_row_layout(ROLE_X, 16, SR.D)
        bcs = [inst.commit_string(key, lay.digest(c.rows[p][i].astype("<u2").tobytes()), c.salts[p][i * 192:(i + 1) * 192]) for i in range(c.n)]
        assert b"".join(bcs) == c.bc[p]
        roots[p] = fv.root(p, [row_leaf(c.domains[p], i, inst.tree_leaf(key, bcs[i]), SR.ROW_SCHEMA) for i in range(c.n)])
    flat = c.out.reshape(-1)
    roots[SR.OUT_PORT] = fv.root(SR.OUT_PORT, [fv.tree_leaf(SR.OUT_PORT, r, fv.leaf_layout(SR.OUT_PORT, r).digest(int(v).to_bytes(2, "big")))
                                               for r, v in enumerate(flat.tolist())])
    return roots


def test_chacha20_is_rfc8439():
    key, nonce = bytes(range(32)), bytes.fromhex("000000090000004a00000000")
    assert SR.chacha20(key, nonce, 64, counter=1).hex() == (
        "10f1e7e4d13b5915500fdd1fa32071c4c7d1f4c733c068030422aa9ac3d46c4e"
        "d2826446079faa0914c2d705d98b02a2b5129cd1de164eb9cbd083e8a2503c4e")
    crypto = pytest.importorskip("cryptography.hazmat.primitives.ciphers")
    k, nc = random.Random(1).randbytes(32), random.Random(2).randbytes(12)
    enc = crypto.Cipher(crypto.algorithms.ChaCha20(k, (0).to_bytes(4, "little") + nc), mode=None).encryptor()
    assert SR.chacha20(k, nc, 200_003) == enc.update(bytes(200_003))


def test_m0_serving_row_leaf_vector():
    dv = VECTOR["domain"]
    dom = CommitmentDomain(bytes.fromhex(dv["binding"]), dv["owner"], RangeIndexedDomain(0, VECTOR["rows_n"]), hash="sha512")
    assert dom.domain_id.hex() == dv["domain_id"]
    rows = np.array([r["words_u16"] for r in VECTOR["rows"]], dtype=np.uint16)
    ys = b"".join(bytes.fromhex(r["salt_y"]) for r in VECTOR["rows"])
    for chunk in (1, 2, 4):
        roots, bc = SR.commit_ports({"x": dom}, {"x": rows}, {}, {"x": ys}, chunk=chunk)
        assert roots["x"].hex() == VECTOR["root"]
        assert bc["x"].hex() == "".join(r["b_c"] for r in VECTOR["rows"])


@pytest.mark.parametrize("n,workers,chunk", [(1, 1, SR.CHUNK), (37, 1, SR.CHUNK), (37, 2, 8), (64, 3, 16)])
def test_core_reference_on_the_batch_and_parallel_paths(n, workers, chunk):
    rows, out = heads(n)
    c = SR.commit(window(n), rows, out, salt_key=bytes(32), workers=workers, chunk=chunk)
    assert c.roots == reference(c)


def test_salts_are_fresh_and_the_key_fixes_them():
    rows, out = heads(5)
    a = SR.commit(window(5), rows, out, salt_key=b"k" * 32)
    assert a.salts["x"] == SR.salts(b"k" * 32, 0, 5) and a.salts["cs"] == SR.salts(b"k" * 32, 1, 5)
    b, c = SR.commit(window(5), rows, out), SR.commit(window(5), rows, out)
    assert b.salts != c.salts and b.roots["x"] != c.roots["x"] and b.roots[SR.OUT_PORT] == c.roots[SR.OUT_PORT]


def test_population_is_fixed_before_serving():
    rows, out = heads(6)
    with pytest.raises(ValueError, match="population"):
        SR.commit(window(7), rows, out)


def test_domains_follow_the_agreed_rule_from_registration_fields(tmp_path):
    rows, out = heads(9)
    c = SR.commit(window(9, "r-dom"), rows, out, salt_key=bytes(32))
    idx = SR.index_doc([[0, 9, "0", 3, 0, "model.layers.0.self_attn.rotary_emb", "0", 0, 768]], "r-dom")
    SR.write(c, tmp_path, {"row": 101, "run": "r-dom"}, idx)
    rec = json.loads((tmp_path / "registration.json").read_text())
    assert set(rec) == {"format", "commitment", "roots", "leaves", "domains", "partition", "partition_digest", "population", "registered_utc"}
    assert rec["format"] == "verity/one-stage/registration/v0" and rec["commitment"]["source"]["kind"] == "served"
    assert rec["commitment"]["source"]["index"] == hashlib.sha512((tmp_path / "index.json").read_bytes()).hexdigest()
    for p, leaves in rec["leaves"].items():
        schema = SR.WORD_SCHEMA if p == SR.OUT_PORT else SR.ROW_SCHEMA
        binding = identity_digest("verity/one-stage/served-domain/v0", {"program": rec["partition"]["program"], "partition": rec["partition_digest"],
                                                                         "port": p, "schema": schema, "leaves": leaves,
                                                                         "run": rec["commitment"]["source"]["run"]})
        dom = CommitmentDomain(bytes.fromhex(binding), -1, RangeIndexedDomain(0, leaves), hash="sha512")
        assert dom.domain_id.hex() == rec["domains"][p]
    assert rec["leaves"] == {"x": 9, "cs": 9, "out": 9 * 64}
    other = SR.commit(window(9, "r-other"), rows, out, salt_key=bytes(32))
    assert all(other.domains[p].domain_id != c.domains[p].domain_id for p in c.domains)


def test_flock_input_files_wrap_the_window_bodies(tmp_path):
    rows, out = heads(4)
    c = SR.commit(window(4, "r-m0"), rows, out, salt_key=bytes(32))
    SR.write(c, tmp_path / "w", {"row": 101, "run": "r-m0"}, SR.index_doc([], "r-m0"))
    meta = {"scheme": SR.SCHEME, "ports": [["x", 64, 1], ["cs", 64, 1]], "out_ports": [["out", 64]], "vus_per_block": 2, "units_per_vu": 32,
            "partition": {"rule": "verity/partition/v1", "digest": "cd" * 64}, "program_sha512": "ab" * 64, "program_digests": {}}
    text = "flock-circuit v2\nMETA " + json.dumps(meta) + "\nCIRCUIT unit\n..."
    got = SR.flock_input_files(tmp_path / "w", text, tmp_path / "m0", "rope-head/d64")
    pub, inst = Path(got["pub"]).read_bytes(), Path(got["inst"]).read_bytes()
    ph, pbody = pub.split(b"\n", 1)
    ih, ibody = inst.split(b"\n", 1)
    ph, ih = json.loads(ph), json.loads(ih)
    assert ph["rows"] is False and ih["rows"] is True and dict(ph, rows=True) == ih
    assert pbody == c.public_body() and ibody == c.private_body() + c.public_body()
    assert len(pbody) == 4 * (2 * 128 + 64 * 2) and len(c.private_body()) == 4 * (2 * 128 + 2 * 192)
    assert ph["frame_v3"]["domain_ids"] == {p: d.domain_id.hex() for p, d in c.domains.items()}
    assert ph["units"]["indices"] == [0, 1, 2, 3] and ph["units"]["blocks"] == [[0, 1], [2, 3]]
    assert ph["circuit_sha512"] == hashlib.sha512(text.encode()).hexdigest()
    with pytest.raises(ValueError, match="ports"):
        SR.flock_input_files(tmp_path / "w", text.replace('["cs", 64, 1]', '["cs", 128, 1]'), tmp_path / "m1", "rope-head/d64")
