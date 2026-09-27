"""The opt-in serving rows (`commit.serving_rows`): M0's serving row leaf vector byte for byte, core's frame-v3-sha512 / hm96-sha512
reference on the batch and parallel paths for a two-member population, ChaCha20 salts, the agreed domain rule, the registration
shape and M0's per-member file layout.  `serving_row_leaf_vector.json` is flock-netlist's vector (notes-asset
flock-netlist/serving-row-leaf-vector.json, sha256 92aa7f77...)."""

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
TEMPLATES = ["RoPEHead_v1{D=64}", "Norm_v1{N=128}"]
PARTITION = {"format": "verity/partition/v1", "program": "ab" * 64,
             "query": {"name": "Q_template_instances", "version": 0, "params": {"templates": TEMPLATES}}}


def population(n0: int, n1: int) -> tuple[SR.Member, SR.Member]:
    return (SR.Member(TEMPLATES[0], (("x", 64), ("cs", 64)), (("out", 64),), n0, 0),
            SR.Member(TEMPLATES[1], (("in0", 128), ("in1", 64)), (("out.0", 128), ("out.1", 3)), n1, n0))


def window(n0: int, n1: int, run: str = "r-test") -> SR.Window:
    return SR.Window(PARTITION, "cd" * 64, n0 + n1, run, population(n0, n1))


def data(w: SR.Window, seed: int = 3) -> list[dict[str, dict[str, np.ndarray]]]:
    rng = np.random.default_rng(seed)
    gen = lambda n, k: rng.integers(0, 1 << 16, size=(n, k), dtype=np.uint16)  # noqa: E731
    return [{"rows": {p: gen(m.n, k) for p, k in m.rows}, "outs": {p: gen(m.n, k) for p, k in m.outs}} for m in w.members]


def reference(c: SR.Committed) -> dict[str, bytes]:
    """Core's slow path: row_leaf over hm96.SHA512.tree_leaf(commit_string(...)), word leaves, FrameV3.root."""
    inst, key = hm96.SHA512, hm96.SHA512.default_key
    roots = {}
    for p, rows in c.rows.items():
        words = rows.shape[1]
        fv = FrameV3({p: Port(c.domains[p], RowLeaf("sha512", ROLE_X, 16, words))})
        lay = sha512_row_layout(ROLE_X, 16, words)
        bcs = [inst.commit_string(key, lay.digest(rows[i].astype("<u2").tobytes()), c.salts[p][i * 192:(i + 1) * 192]) for i in range(len(rows))]
        assert b"".join(bcs) == c.bc[p]
        roots[p] = fv.root(p, [row_leaf(c.domains[p], i, inst.tree_leaf(key, bcs[i]), SR.ROW_SCHEMA) for i in range(len(rows))])
    for p, outs in c.outs.items():
        fv = FrameV3({p: Port(c.domains[p], WordLeaf("u16", 2))})
        roots[p] = fv.root(p, [fv.tree_leaf(p, r, fv.leaf_layout(p, r).digest(int(v).to_bytes(2, "big"))) for r, v in enumerate(outs.reshape(-1).tolist())])
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


@pytest.mark.parametrize("n0,n1,workers,chunk", [(1, 1, 1, SR.CHUNK), (37, 5, 1, SR.CHUNK), (37, 5, 2, 8), (64, 17, 3, 16)])
def test_core_reference_on_the_batch_and_parallel_paths(n0, n1, workers, chunk):
    w = window(n0, n1)
    c = SR.commit(w, data(w), salt_key=bytes(32), workers=workers, chunk=chunk)
    assert set(c.roots) == {f"{m.template}/{p}" for m in w.members for p, _ in m.rows + m.outs}
    assert c.roots == reference(c)


def test_salts_are_fresh_and_the_key_fixes_them():
    w = window(5, 2)
    d = data(w)
    a = SR.commit(w, d, salt_key=b"k" * 32)
    ports = [m.port(p) for m in w.members for p, _ in m.rows]
    assert [a.salts[p] for p in ports] == [SR.salts(b"k" * 32, i, len(a.rows[p])) for i, p in enumerate(ports)]
    b, c = SR.commit(w, d), SR.commit(w, d)
    out = w.members[0].port("out")
    assert b.salts != c.salts and b.roots[ports[0]] != c.roots[ports[0]] and b.roots[out] == c.roots[out]


def test_population_is_fixed_before_serving():
    w = window(6, 2)
    with pytest.raises(ValueError, match="population"):
        SR.commit(SR.Window(PARTITION, "cd" * 64, 9, "r", w.members), data(w))
    bad = data(w)
    bad[1]["rows"]["in0"] = bad[1]["rows"]["in0"][:1]
    with pytest.raises(ValueError, match="Norm_v1"):
        SR.commit(w, bad)


def test_domains_follow_the_agreed_rule_from_registration_fields(tmp_path):
    w = window(9, 4, "r-dom")
    c = SR.commit(w, data(w), salt_key=bytes(32))
    idx = SR.index_doc(w.members, [[[0, 9, "0", 3, 0, "model.layers.0.self_attn.rotary_emb", "0", 0, 768]], []], "r-dom")
    SR.write(c, tmp_path, {"row": 101, "run": "r-dom"}, idx)
    rec = json.loads((tmp_path / "registration.json").read_text())
    assert set(rec) == {"format", "commitment", "roots", "leaves", "domains", "partition", "partition_digest", "population", "registered_utc"}
    assert rec["commitment"]["source"]["kind"] == "served" and rec["population"] == 13
    assert rec["commitment"]["source"]["index"] == hashlib.sha512((tmp_path / "index.json").read_bytes()).hexdigest()
    for p, leaves in rec["leaves"].items():
        schema = SR.WORD_SCHEMA if p.split("/", 1)[1].startswith("out") else SR.ROW_SCHEMA
        binding = identity_digest("verity/one-stage/served-domain/v0", {"program": rec["partition"]["program"], "partition": rec["partition_digest"],
                                                                         "port": p, "schema": schema, "leaves": leaves,
                                                                         "run": rec["commitment"]["source"]["run"]})
        dom = CommitmentDomain(bytes.fromhex(binding), -1, RangeIndexedDomain(0, leaves), hash="sha512")
        assert dom.domain_id.hex() == rec["domains"][p]
    assert rec["leaves"] == {"RoPEHead_v1{D=64}/x": 9, "RoPEHead_v1{D=64}/cs": 9, "RoPEHead_v1{D=64}/out": 9 * 64,
                             "Norm_v1{N=128}/in0": 4, "Norm_v1{N=128}/in1": 4, "Norm_v1{N=128}/out.0": 4 * 128, "Norm_v1{N=128}/out.1": 4 * 3}
    other = SR.commit(window(9, 4, "r-other"), data(w), salt_key=bytes(32))
    assert all(other.domains[p].domain_id != c.domains[p].domain_id for p in c.domains)


def test_flock_input_files_wrap_each_members_bodies(tmp_path):
    w = window(4, 3, "r-m0")
    d = data(w)
    c = SR.commit(w, d, salt_key=bytes(32))
    SR.write(c, tmp_path / "w", {"row": 101, "run": "r-m0"}, SR.index_doc(w.members, [[], []], "r-m0"))
    meta = {"scheme": SR.SCHEME, "ports": [["in0", 128, 1], ["in1", 64, 1]], "out_ports": [["out.0", 128], ["out.1", 3]], "vus_per_block": 2,
            "units_per_vu": 8, "partition": {"rule": "verity/partition/v1", "digest": "cd" * 64}, "program_sha512": "ab" * 64, "program_digests": {}}
    text = "flock-circuit v2\nMETA " + json.dumps(meta) + "\nCIRCUIT unit\n..."
    got = SR.flock_input_files(tmp_path / "w", 1, text, tmp_path / "m0", "norm/n128")
    assert got["pub"].endswith("m1-pub-3.bin") and got["inst"].endswith("m1-inst-3.bin")
    pub, inst = Path(got["pub"]).read_bytes(), Path(got["inst"]).read_bytes()
    ph, pbody = pub.split(b"\n", 1)
    ih, ibody = inst.split(b"\n", 1)
    ph, ih = json.loads(ph), json.loads(ih)
    assert ph["rows"] is False and ih["rows"] is True and dict(ph, rows=True) == ih
    assert pbody == c.public_body(1) and ibody == c.private_body(1) + c.public_body(1)
    m = w.members[1]
    want_pub = np.concatenate([np.frombuffer(c.bc[m.port(p)], np.uint8).reshape(3, 128) for p in ("in0", "in1")], axis=1).tobytes() + \
        np.concatenate([d[1]["outs"]["out.0"], d[1]["outs"]["out.1"]], axis=1).astype("<u2").tobytes()
    assert pbody == want_pub
    assert ph["frame_v3"]["domain_ids"] == {p: c.domains[m.port(p)].domain_id.hex() for p in ("in0", "in1", "out.0", "out.1")}
    assert ph["units"]["indices"] == [4, 5, 6] and ph["units"]["blocks"] == [[4, 5], [6]]
    assert ph["circuit_sha512"] == hashlib.sha512(text.encode()).hexdigest()
    with pytest.raises(ValueError, match="ports"):
        SR.flock_input_files(tmp_path / "w", 0, text, tmp_path / "m1", "rope-head/d64")
