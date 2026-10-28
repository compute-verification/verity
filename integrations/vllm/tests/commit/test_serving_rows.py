"""The opt-in serving rows (`ml.windows.rows`): Flock's circuit prover's serving row leaf vector byte for byte, core's frame-v3-sha512 / hm96-sha512
reference on the batch and parallel paths for a two-member population, ChaCha20 salts, core's served-domain rule, the registration
(core's v1 record, accepted by `registration.check` against the verifier's own copies) and Flock's circuit prover's per-member file layout, every output
a hidden row (no output word in a public body).  `serving_row_leaf_vector.json` is flock-netlist's vector (notes-asset
flock-netlist/serving-row-leaf-vector.json, sha256 92aa7f77...)."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pytest

from verity.ml.commitments import hm96
from verity.ml.commitments.frame_v3 import FrameV3, Port, RowLeaf
from verity.ml.commitments.identity import identity_digest_sha512
from verity.ml.commitments.indexed import RangeIndexedDomain
from verity.ml.commitments.merkle import CommitmentDomain
from verity.ml.commitments.rowleaf import (ROLE_X, SCHEMA_HM96_SHA512_ROW_SEG as SEG, SCHEMA_HM96_SHA512_ROW_V2 as V2, row_leaf,
                                           sha512_row_layout, sha512_row_seg_digest, sha512_row_v2_digest)
from verity.ml.circuits.defs import REGISTRY, bind
from verity.core.protocols.sampled_proofs.one_stage import draw as D, partition as P, registration as R
from verity.ml.windows import flock_inputs as FI, rows as SR
from verity_vllm.commit import index as IX
from verity.ml.definitions.registry import b1  # noqa: F401  (registers RoPEHead_v1 and SiluMul_v1)

VECTOR = json.loads((Path(__file__).parent / "serving_row_leaf_vector.json").read_text())
DEFS = (bind(REGISTRY["RoPEHead_v1"], D=64), bind(REGISTRY["SiluMul_v1"], I=64))
TEMPLATES = [P.template_id(d) for d in DEFS]
LAW = D.parse_law("subset:1")
RUN = "ab" * 32


def population(n0: int, n1: int) -> tuple[SR.Member, SR.Member]:
    return (SR.Member(TEMPLATES[0], (("x", 64), ("cs", 64)), (("out", 64),), n0, 0),
            SR.Member(TEMPLATES[1], (("in0", 128), ("in1", 64)), (("out.0", 128), ("out.1", 3)), n1, n0))


def window(n0: int, n1: int, run: str = RUN) -> SR.Window:
    program, query = SR.population_program([(DEFS[0], n0), (DEFS[1], n1)])
    return SR.Window(program, query, LAW, run, population(n0, n1))


def data(w: SR.Window, seed: int = 3) -> list[dict[str, dict[str, np.ndarray]]]:
    rng = np.random.default_rng(seed)
    gen = lambda n, k: rng.integers(0, 1 << 16, size=(n, k), dtype=np.uint16)  # noqa: E731
    return [{"rows": {p: gen(m.n, k) for p, k in m.rows}, "outs": {p: gen(m.n, k) for p, k in m.outs}} for m in w.members]


def reference(c: SR.Committed) -> dict[str, bytes]:
    """Core's slow path: row_leaf over hm96.SHA512.tree_leaf(commit_string(...)) for every row, the output rows included, and
    FrameV3.root."""
    inst, key = hm96.SHA512, hm96.SHA512.default_key
    roots = {}
    for p, rows in c.rows.items():
        words = rows.shape[1]
        fv = FrameV3({p: Port(c.domains[p], RowLeaf("sha512", ROLE_X, 16, words))})
        lay = sha512_row_layout(ROLE_X, 16, words)
        bcs = [inst.commit_string(key, lay.digest(rows[i].astype("<u2").tobytes()), c.salts[p][i * 192:(i + 1) * 192]) for i in range(len(rows))]
        assert b"".join(bcs) == c.bc[p]
        roots[p] = fv.root(p, [row_leaf(c.domains[p], i, inst.tree_leaf(key, bcs[i]), SR.ROW_SCHEMA) for i in range(len(rows))])
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
    d = data(w)
    for m, dk in zip(w.members, d):
        for (p, k), (_, padded) in zip(m.outs, m.out_rows):
            assert padded % 64 == 0 and padded >= k
            assert (c.rows[m.port(p)][:, :k] == dk["outs"][p]).all() and not c.rows[m.port(p)][:, k:].any()


def test_salts_are_fresh_and_the_key_fixes_them():
    w = window(5, 2)
    d = data(w)
    a = SR.commit(w, d, salt_key=b"k" * 32)
    ports = [m.port(p) for m in w.members for p, _ in m.all_rows]
    assert [a.salts[p] for p in ports] == [SR.salts(b"k" * 32, i, len(a.rows[p])) for i, p in enumerate(ports)]
    b, c = SR.commit(w, d), SR.commit(w, d)
    out = w.members[0].port("out")
    assert b.salts != c.salts and b.roots[ports[0]] != c.roots[ports[0]] and b.roots[out] != c.roots[out]


def test_population_is_the_partitions():
    w = window(6, 2)
    assert (w.population, [m.base for m in w.members]) == (8, [0, 6])
    with pytest.raises(ValueError, match="partition's units"):
        SR.Window(w.program, w.query, LAW, RUN, population(6, 3))
    with pytest.raises(ValueError, match="partition's units"):
        SR.Window(w.program, w.query, LAW, RUN, population(6, 2)[::-1])
    bad = data(w)
    bad[1]["rows"]["in0"] = bad[1]["rows"]["in0"][:1]
    with pytest.raises(ValueError, match="SiluMul_v1"):
        SR.commit(w, bad)


def test_the_registration_is_cores_v1_record_over_served_domains(tmp_path):
    w = window(9, 4)
    c = SR.commit(w, data(w), salt_key=bytes(32))
    idx = IX.index_doc(w.members, [[[0, 9, "r0", 3, 0, "model.layers.0.self_attn.rotary_emb", "0", 0, 768]], []], RUN)
    win = SR.write(c, tmp_path, {"row": 101, "commitment": "vllm-v1"}, idx, registrar=R)
    rec = json.loads((tmp_path / "registration.json").read_text())
    assert (rec["format"], rec["population"], rec["partition"], rec["window"]["kind"], rec["window"]["run"]) == (
        R.FORMAT, 13, w.partition_digest, "served", RUN)
    assert rec["window"]["index"] == hashlib.sha512((tmp_path / "index.json").read_bytes()).hexdigest() == win["index_sha512"]
    assert win["registration_digest"] == R.digest(rec) and rec["program"] == P.program_digest(w.program)
    leaves = {r["port"]: r["leaves"] for r in rec["roots"]}
    assert leaves == {f"{TEMPLATES[0]}/{p}": 9 for p in ("x", "cs", "out")} | {f"{TEMPLATES[1]}/{p}": 4 for p in ("in0", "in1", "out.0", "out.1")}
    for r in rec["roots"]:
        binding = identity_digest_sha512("verity/one-stage/served-domain/v1", {"program": rec["program"], "partition": rec["partition"],
                                                                                "port": r["port"], "schema": SR.ROW_SCHEMA, "leaves": r["leaves"], "run": RUN})
        assert CommitmentDomain(binding, -1, RangeIndexedDomain(0, r["leaves"]), hash="sha512").domain_id.hex() == r["domain"]
        assert r["root"] == c.roots[r["port"]].hex()
    # the verifier's own copies: its window over its program, the run root of record, and the public bodies it holds
    mine = window(9, 4)
    publics = {m.template: (tmp_path / f"m{k}-public.bin").read_bytes() for k, m in enumerate(mine.members)}
    v = R.check(rec, mine.program, mine.query, LAW, SR.SCHEME, SR.expected_ports(mine), leaf_layer=SR.leaf_layer(publics))
    assert v.ok, v.detail
    other = window(9, 4, "cd" * 32)
    v = R.check(rec, other.program, other.query, LAW, SR.SCHEME, SR.expected_ports(other), leaf_layer=SR.leaf_layer(publics))
    assert not v.ok and "R3-domain" in v.codes
    flipped = dict(publics, **{TEMPLATES[1]: publics[TEMPLATES[1]][:-1] + bytes([publics[TEMPLATES[1]][-1] ^ 1])})
    assert R.check(rec, mine.program, mine.query, LAW, SR.SCHEME, SR.expected_ports(mine), leaf_layer=SR.leaf_layer(flipped)).codes == ("R4-leaf-layer",)
    fewer = window(9, 3)
    assert "R2-population" in R.check(rec, fewer.program, fewer.query, LAW, SR.SCHEME, SR.expected_ports(fewer),
                                      leaf_layer=SR.leaf_layer(publics)).codes
    # a stand-in is the same record but for its kind, which provenance cannot override; a kind core does not know is refused
    stand = SR.registration(c, {"kind": "served", "run": "other", "of": {"run_root": RUN}}, kind="stand-in", registrar=R)
    assert (stand["window"]["kind"], stand["window"]["run"], stand["roots"]) == ("stand-in", RUN, rec["roots"])
    with pytest.raises(ValueError, match="window kind 'replayed'"):
        SR.registration(c, {}, kind="replayed", registrar=R)


def test_flock_input_files_wrap_each_members_bodies(tmp_path):
    w = window(4, 3)
    d = data(w)
    c = SR.commit(w, d, salt_key=bytes(32))
    SR.write(c, tmp_path / "w", {"row": 101}, IX.index_doc(w.members, [[], []], RUN), registrar=R)
    meta = {"scheme": SR.SCHEME, "ports": [["in0", 128, 1], ["in1", 64, 1], ["out.0", 128, 1], ["out.1", 64, 1]], "input_ports": 2,
            "out_ports": [["out.0", 128], ["out.1", 3]], "vus_per_block": 2, "units_per_vu": 8, "partition": {"rule": "verity/partition/v1", "digest": "cd" * 64}, "program_sha512": "ab" * 64, "program_digests": {},
            "statement": "verity/flock-circuit", "subcircuit": "norm/n128"}
    text = "flock-circuit v2\nMETA " + json.dumps(meta) + "\nCIRCUIT unit\n..."
    got = FI.flock_input_files(tmp_path / "w", 1, text, tmp_path / "m0")
    assert got["pub"].endswith("m1-pub-3.bin") and got["inst"].endswith("m1-inst-3.bin")
    pub, inst = Path(got["pub"]).read_bytes(), Path(got["inst"]).read_bytes()
    ph, pbody = pub.split(b"\n", 1)
    ih, ibody = inst.split(b"\n", 1)
    ph, ih = json.loads(ph), json.loads(ih)
    assert ph["rows"] is False and ih["rows"] is True and dict(ph, rows=True) == ih
    assert pbody == c.public_body(1) and ibody == c.private_body(1) + c.public_body(1)
    m = w.members[1]
    want_pub = np.concatenate([np.frombuffer(c.bc[m.port(p)], np.uint8).reshape(3, 128) for p in ("in0", "in1", "out.0", "out.1")], axis=1).tobytes()
    assert pbody == want_pub
    assert "content_digest" not in ph and set(ph["frame_v3"]["schemas"].values()) == {SR.ROW_SCHEMA}
    for p in ("out.0", "out.1"):
        assert d[1]["outs"][p].astype("<u2").tobytes()[:16] not in pbody
    assert ph["frame_v3"]["domain_ids"] == {p: c.domains[m.port(p)].domain_id.hex() for p in ("in0", "in1", "out.0", "out.1")}
    assert ph["units"]["indices"] == [4, 5, 6] and ph["units"]["blocks"] == [[4, 5], [6]]
    assert ph["circuit_sha512"] == hashlib.sha512(text.encode()).hexdigest()
    assert (ph["statement"], ph["subcircuit"]) == ("verity/flock-circuit", "norm/n128")
    with pytest.raises(ValueError, match="ports"):
        FI.flock_input_files(tmp_path / "w", 0, text, tmp_path / "m1")
    public_outputs = dict(meta, ports=meta["ports"][:2])
    del public_outputs["input_ports"]
    with pytest.raises(ValueError, match="ports"):
        FI.flock_input_files(tmp_path / "w", 1, "flock-circuit v2\nMETA " + json.dumps(public_outputs) + "\n", tmp_path / "m2")


def test_a_members_public_file_alone_then_its_provers_file_are_the_files_its_circuits_text_gives(tmp_path):
    """The audit stages every member's public file before the sample and a drawn member's prover's file after it, from the
    circuit read and hashed once (`flock_circuit`) and the window's files read once: byte for byte what the circuit's text gives."""
    w = window(4, 3)
    c = SR.commit(w, data(w), salt_key=bytes(32))
    SR.write(c, tmp_path / "w", {"row": 101}, IX.index_doc(w.members, [[], []], RUN), registrar=R)
    meta = {"scheme": SR.SCHEME, "ports": [["in0", 128, 1], ["in1", 64, 1], ["out.0", 128, 1], ["out.1", 64, 1]], "input_ports": 2,
            "out_ports": [["out.0", 128], ["out.1", 3]], "vus_per_block": 2, "units_per_vu": 8, "partition": {"rule": "verity/partition/v1", "digest": "cd" * 64},
            "program_sha512": "ab" * 64, "program_digests": {"t": "ef" * 64}, "subcircuit": "norm/n128"}
    text = "flock-circuit v2\nMETA " + json.dumps(meta) + "\nCIRCUIT unit µ\n..."
    whole = FI.flock_input_files(tmp_path / "w", 1, text, tmp_path / "whole")
    circ = FI.flock_circuit(text.encode())
    head_line, _, rest = text.split("\n", 2)
    unbound = {x: v for x, v in meta.items() if x not in FI.M0_BINDING_KEYS}
    assert circ == FI.flock_circuit(text) and circ.meta == meta and (circ.pin, circ.cls) == (whole["circuit_sha512"], whole["class"]) == (
        hashlib.sha512(text.encode()).hexdigest(),
        hashlib.sha512((head_line + "\nMETA " + json.dumps(unbound, sort_keys=True, separators=(",", ":")) + "\n" + rest).encode()).hexdigest())
    files = (json.loads((tmp_path / "w" / "window.json").read_text()), json.loads((tmp_path / "w" / "registration.json").read_text()))
    pub = FI.flock_input_files(tmp_path / "w", 1, circ, tmp_path / "staged", write=("pub",), files=files)
    assert "inst" not in pub and not list((tmp_path / "staged").glob("*inst*"))
    inst = FI.flock_input_files(tmp_path / "w", 1, circ, tmp_path / "staged", write=("inst",), files=files)
    assert "pub" not in inst
    for x, got in (("pub", pub), ("inst", inst)):
        assert Path(got[x]).read_bytes() == Path(whole[x]).read_bytes()
    with pytest.raises(ValueError, match="pub and inst, not"):
        FI.flock_input_files(tmp_path / "w", 1, circ, tmp_path / "x", write=("private",))


def test_a_shared_row_member_commits_its_tables_and_writes_m0s_shared_row_body(tmp_path):
    rng = np.random.default_rng(7)
    tables = {"x": rng.integers(0, 1 << 16, size=(3, 64), dtype=np.uint16), "w": rng.integers(0, 1 << 16, size=(5, 64), dtype=np.uint16)}
    tables["x"][2] = tables["x"][0]                                   # a repeated row stays its own table row
    j = np.arange(3 * 5)
    refs = np.stack([j // 5, j % 5], axis=1)
    y = rng.integers(0, 1 << 16, size=(15, 1), dtype=np.uint16)
    gemm = bind(REGISTRY["GemmCoordinate_v1"], K=64)
    m = SR.Member(P.template_id(gemm), (("x", 64), ("w", 64)), (("y", 1),), 15, 0, tables=(("x", 3), ("w", 5)))
    w = SR.Window(*SR.population_program([(gemm, 15)]), LAW, RUN, (m,))
    assert SR.port_leaves(m) == {"x": 3, "w": 5, "y": 15}
    assert {p: e["leaves"] for p, e in SR.expected_ports(w).items()} == {m.port("x"): 3, m.port("w"): 5, m.port("y"): 15}
    c = SR.commit(w, [{"rows": tables, "outs": {"y": y}, "refs": refs}], salt_key=bytes(32))
    assert c.domains[m.port("x")].count == 3 and c.domains[m.port("w")].count == 5 and c.domains[m.port("y")].count == 15
    assert c.roots == reference(c)
    assert c.bc[m.port("x")][:128] != c.bc[m.port("x")][256:]       # fresh salt per table row
    own = np.concatenate([refs, np.arange(15)[:, None]], axis=1)       # an output row is its instance's own
    yrow = np.zeros((15, 64), dtype=np.uint16)
    yrow[:, :1] = y
    assert c.public_body(0) == c.bc[m.port("x")] + c.bc[m.port("w")] + c.bc[m.port("y")] + own.astype("<u4").tobytes()
    assert c.private_body(0) == tables["x"].astype("<u2").tobytes() + tables["w"].astype("<u2").tobytes() + yrow.astype("<u2").tobytes() \
        + c.salts[m.port("x")] + c.salts[m.port("w")] + c.salts[m.port("y")]
    with pytest.raises(ValueError, match="refs"):
        SR.commit(w, [{"rows": tables, "outs": {"y": y}, "refs": refs + [0, 5]}])
    SR.write(c, tmp_path / "w", {"row": 101}, IX.index_doc((m,), [[]], RUN), registrar=R)
    meta = {"scheme": SR.SCHEME, "ports": [["x", 64, 1], ["w", 64, 1], ["y", 64, 1]], "input_ports": 2, "out_ports": [["y", 1]],
            "vus_per_block": 4, "units_per_vu": 1, "subcircuit": "gemm/k64"}
    got = FI.flock_input_files(tmp_path / "w", 0, "flock-circuit v2\nMETA " + json.dumps(meta) + "\n", tmp_path / "m0")
    head = json.loads(Path(got["pub"]).read_bytes().split(b"\n", 1)[0])
    assert head["shared_rows"] == {"x": 3, "w": 5, "y": 15} and head["instances"] == 15


def s8_window(chunks: list[tuple[int, int]], n: int = 5) -> tuple[SR.Window, list[SR.Member]]:
    """``ScaledMmS8Coordinate_v1{K=64}``'s ``n`` units as bit-row members ``(base, size)``, under ``TemplatePopulations``."""
    t = bind(REGISTRY["ScaledMmS8Coordinate_v1"], K=64, DOT=REGISTRY["BlackwellS8ImmaDot32_v1"])
    ms = [SR.Member(P.template_id(t), (("x", 34), ("w", 34)), (("o0", 1),), size, base, name=f"{P.template_id(t)}/c{j}",
                    schemas=(("x", SEG), ("w", V2), ("o0", V2))) for j, (base, size) in enumerate(chunks)]
    return SR.Window(*SR.population_program([(t, n)], several=True), D.parse_law("stratified:2"), RUN, tuple(ms)), ms


def test_bit_row_chunks_commit_each_port_under_its_schema_and_tile_their_template(tmp_path):
    w, ms = s8_window([(0, 3), (3, 2)])
    assert w.query["name"] == "Q_template_instances" and w.population == 5
    rng = np.random.default_rng(5)
    d = [{"rows": {p: rng.integers(0, 1 << 16, size=(m.n, 34), dtype=np.uint16) for p in ("x", "w")},
          "outs": {"o0": rng.integers(0, 1 << 16, size=(m.n, 1), dtype=np.uint16)}} for m in ms]
    c = SR.commit(w, d, salt_key=bytes(32))
    inst, key = hm96.SHA512, hm96.SHA512.default_key
    for m in ms:
        assert m.out_rows == (("o0", 1),) and m.row_schemas == [[SEG, 544], [V2, 544], [V2, 16]]
        for p, schema in (("x", SEG), ("w", V2), ("o0", V2)):
            rows, port = c.rows[m.port(p)], m.port(p)
            digest = sha512_row_seg_digest if schema == SEG else sha512_row_v2_digest
            bcs = [inst.commit_string(key, digest(r.astype("<u2").tobytes(), 16 * len(r)), c.salts[port][i * 192:(i + 1) * 192])
                   for i, r in enumerate(rows)]
            assert b"".join(bcs) == c.bc[port]
            dom = c.domains[port]
            assert dom.binding == w.binding(port, schema, m.n)
            fv = FrameV3({port: Port(dom, RowLeaf("sha512", ROLE_X, 16, 64))})
            assert c.roots[port] == fv.root(port, [row_leaf(dom, i, inst.tree_leaf(key, b), schema) for i, b in enumerate(bcs)])
    win = SR.write(c, tmp_path, {"row": 101}, IX.index_doc(ms, [[], []], RUN), registrar=R)
    for k, m in enumerate(ms):
        units = json.loads((tmp_path / f"m{k}-units.json").read_text())
        assert units["indices"] == list(range(m.base, m.base + m.n)) and units["member"] == m.name
        assert win["members"][k]["units_sha512"] == hashlib.sha512((tmp_path / f"m{k}-units.json").read_bytes()).hexdigest()
        assert win["members"][k]["row_schemas"] == m.row_schemas and win["members"][k]["frame_v3"]["schemas"] == {"x": SEG, "w": V2, "o0": V2}
    rec = json.loads((tmp_path / "registration.json").read_text())
    mine, _ = s8_window([(0, 3), (3, 2)])
    publics = {m.name: (tmp_path / f"m{k}-public.bin").read_bytes() for k, m in enumerate(ms)}
    v = R.check(rec, mine.program, mine.query, mine.law, SR.SCHEME, SR.expected_ports(mine), leaf_layer=SR.leaf_layer(publics))
    assert v.ok, v.detail
    meta = {"scheme": SR.SCHEME, "ports": [["x", 34, 1], ["w", 34, 1], ["o0", 1, 1]], "input_ports": 2, "out_ports": [["o0", 1]],
            "row_schemas": [[SEG, 544], [V2, 544], [V2, 16]], "vus_per_block": 1, "units_per_vu": 1, "subcircuit": "s8/k64"}
    got = FI.flock_input_files(tmp_path, 1, "flock-circuit v2\nMETA " + json.dumps(meta) + "\n", tmp_path / "m1")
    assert json.loads(Path(got["pub"]).read_bytes().split(b"\n", 1)[0])["units"]["indices"] == [3, 4]
    with pytest.raises(ValueError, match="row schemas"):
        FI.flock_input_files(tmp_path, 1, "flock-circuit v2\nMETA " + json.dumps(dict(meta, row_schemas=[[V2, 544]] * 2 + [[V2, 16]])) + "\n",
                             tmp_path / "m2")


@pytest.mark.parametrize("chunks", [[(0, 3), (4, 1)], [(3, 2), (0, 3)], [(0, 3), (2, 3)], [(0, 3)], [(0, 3), (3, 2), (5, 1)]])
def test_chunks_that_do_not_tile_their_template_are_refused(chunks):
    with pytest.raises(ValueError, match="partition's units"):
        s8_window(chunks)


def test_a_bit_row_member_names_a_bit_row_schema_for_every_port():
    with pytest.raises(ValueError, match="bit-row member"):
        SR.Member("t", (("x", 2),), (("o0", 1),), 1, schemas=(("x", V2),))
    with pytest.raises(ValueError, match="bit-row member"):
        SR.Member("t", (("x", 2),), (("o0", 1),), 1, schemas=(("x", V2), ("o0", SR.ROW_SCHEMA)))
