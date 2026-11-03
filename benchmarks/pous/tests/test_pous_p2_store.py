"""The encoded weights store (`p2_v1/store.py`): the real store's layout JSON and the toy fixture
(`p2_v1/store-toy/`) regenerate as pinned, the layout, `W` and the staging buffers by their digests; the toy's `C*` decodes,
layer by layer, to its `W`'s tensors by `reference.decode`, by the registered codec and by `layout.decode_tensor`,
passes the public-encoder setup with the pinned root, and refuses each negative block with its flag; the spot vectors
decode at the real store's indices. The driver (`commit`, `encode`, `finish`, `verify`, `spot`, `tie`) runs on the CPU at a
128-bit width with 16-block segments and the scheme's own codec in the GPU's place, over a small generator with the real
one's tensors: the store it writes is the registered scheme's encoding of `W` in the layout (the public-encoder check
accepts it, and `decode_layer` gives each layer's tensors), its roots are `setup.server_root`'s tree, `verify` catches a
changed byte, a rerun resumes to the same store, and two parts join to it."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("verity")

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))
from p2_v1 import reference  # noqa: E402
from p2_v1 import store as S  # noqa: E402

from verity.core.protocols.pous import protocol, setup, verifier  # noqa: E402
from verity.core.protocols.pous.layout import TensorLayout, decode_tensor  # noqa: E402
from verity.core.protocols.pous.schemes.p2 import P2, P2Params, decode_segment, encode_segment, expand_prefix  # noqa: E402
from verity.core.protocols.pous.verifier import VerifierKey  # noqa: E402
from verity.ml.circuits import requirement  # noqa: E402

TOY = json.loads((S.TOY_DIR / "toy.json").read_text())
SPOT = json.loads((S.TOY_DIR / "spot.json").read_text())
C_STAR = (S.TOY_DIR / "c-star.bin").read_bytes()
STORE = S.toy_store(TOY)
PLAIN = S.toy_plain(STORE)
W, STAGING = PLAIN["w.bin"], PLAIN["staging.bin"]
SALT = bytes.fromhex(TOY["salt"])
NB = STORE.params.block_bytes


def _segments(c_star: bytes = C_STAR) -> list[list[bytes]]:
    n = STORE.params.blocks_per_segment
    return [[c_star[NB * (s * n + v):NB * (s * n + v + 1)] for v in range(n)] for s in range(STORE.params.segments)]


def test_layout_json_regenerates():
    doc = S._dump(S.build(S.MQA16K).to_json())
    assert hashlib.sha512(doc.encode()).hexdigest() == S.LAYOUT_SHA512, "a layout change: pin the new digest"


def test_real_layout():
    """Every tensor from a block boundary, back to back by blocks, so no block holds two; W to whole segments."""
    store, pb = S.build(S.MQA16K), 2054
    at = 0
    for t in store.tensors:
        assert (t.first_block, t.offset, t.blocks) == (at, at * pb, -(-t.size // pb))
        assert store.layout.placement(t.name).offset == t.offset and t.staging_offset % S.STAGING_ALIGN == 0
        at += t.blocks
    p = store.params
    assert (at, p.blocks_per_segment, p.segments, p.blocks) == (68_013_623, 65_536, 1_038, 68_026_368)
    assert (p.w_bytes, p.blocks * p.block_bytes) == (139_726_159_872, 139_862_212_608)
    assert sum(t.size for t in store.tensors) == 139_699_683_328
    for l in range(52):
        qkv, o, up, down = store.layer(l)
        assert (qkv.n, qkv.k, o.n, o.k, up.n, up.k, down.n, down.k) == (16512, 16384, 16384, 16384, 65536, 16384, 16384, 65536)
        assert qkv.first_block == 1_307_915 * l and store.staging_bytes(l) == 2_686_451_712
    (head,) = store.layer(52)
    assert (head.name, head.first_block, head.blocks) == ("lm_head.weight", 68_011_580, 2_043)
    aligned = TensorLayout.of(P2Params(S.W_BITS, S.C0, S.PER_SEGMENT), [(t.name, t.size) for t in store.tensors],
                              align=True)
    assert aligned.params.segments > p.segments + 50


def test_toy_regenerates():
    """The toy's W, commitment, salt and key prefix from the generator's code, the default toy's; its files by their
    pinned hashes."""
    assert TOY["layout"] == json.loads(S._dump(S.build(S.toy_shape(), 11).to_json()))
    assert {t.layer for t in STORE.tensors} == {0, 1, 2} and STORE.layer(2)[0].name == "lm_head.weight"
    straddles = [t.name for t in STORE.tensors if len(STORE.layout.segments(t.name)) > 1]
    assert straddles == ["model.layers.0.mlp.down_proj.weight"] and STORE.layer(1)[0].first_block % 11
    commitment = setup.commit(STORE.params, W)
    assert commitment.hex() == TOY["commitment"]
    assert setup.draw_salt(bytes.fromhex(TOY["source"]), commitment) == SALT
    assert expand_prefix(STORE.params, SALT).hex() == TOY["prefix"]
    for data, key in ((W, "w_sha512"), (C_STAR, "c_star_sha512"), (STAGING, "staging_sha512")):
        assert hashlib.sha512(data).hexdigest() == TOY[key]


def test_toy_decode():
    """Each layer's staging buffer is its tensors' bytes of W, by `reference.decode`, and every tensor and segment decodes
    the same by the registered codec, `layout.decode_tensor` and `protocol.decode_tensor`."""
    def block(i: int) -> bytes:
        return C_STAR[NB * i:NB * (i + 1)]

    for l, start, size, sha in TOY["staging"]:
        buf = S.decode_layer(STORE, C_STAR, SALT, l)
        assert buf == STAGING[start:start + size] and hashlib.sha512(buf).hexdigest() == sha
        for t in STORE.layer(l):
            assert buf[t.staging_offset:t.staging_offset + t.size] == W[t.offset:t.offset + t.size]
            assert decode_tensor(STORE.layout, None, SALT, t.name, block) == W[t.offset:t.offset + t.size]
            assert protocol.decode_tensor(P2(STORE.params), STORE.layout, SALT, t.name, block) == W[t.offset:t.offset + t.size]
    seg = STORE.params.segment_bytes
    for s, blocks in enumerate(_segments()):
        assert decode_segment(STORE.params, None, SALT, s, blocks) == W[s * seg:(s + 1) * seg]


def test_the_layout_reads_as_the_audits_store():
    """`StoreLayout.store` is the toy's `C*` as `requirement.Store`: every block registered and resident at every step,
    since nothing is written or freed after setup, and no payloads."""
    held = STORE.store
    assert held == requirement.Store(STORE.params.blocks, {}, {}, {})
    assert held.resident(0) == held.resident(STORE.params.blocks) == frozenset(range(len(C_STAR) // NB))


def test_toy_public_setup():
    """`Dec(C*)` matches the commitment, so `C* = Enc(W)` (each `D_j` is a bijection on `[0, p)`), with the pinned root."""
    vk = setup.verifier_key_from_encoding(STORE.params, None, bytes.fromhex(TOY["commitment"]), SALT, _segments())
    assert vk.root.hex() == TOY["root"] == VerifierKey.from_segments(STORE.params, SALT, _segments()).root.hex()


def test_toy_refusals():
    assert S.refusal(STORE, C_STAR, SALT, 0) is None
    for neg in TOY["negatives"]:
        bad = S.negative(neg, C_STAR, STORE.params)
        assert S.refusal(STORE, bad, SALT, neg["layer"]) == neg["flag"], neg["id"]
        assert S.refusal(STORE, bad, SALT, 1) is None
    other = bytes(SALT[:-1]) + bytes([SALT[-1] ^ 1])
    assert S.refusal(STORE, C_STAR, other, 0) == 2


def _toy_read():
    toy = S.toy_tensors(STORE)
    return toy, lambda lo, hi: S.w_range(STORE, lambda name, a, b: toy[name][a:b], lo, hi)


def test_w_range_is_the_toy_w():
    """Any byte range of W from the tensors alone: each at its block, zero between and to the last segment's end."""
    toy, read = _toy_read()
    p, (qkv, o) = STORE.params, STORE.tensors[:2]
    assert hashlib.sha512(read(0, p.w_bytes)).hexdigest() == TOY["w_sha512"]
    for lo, hi in [(0, 1), (qkv.size - 3, o.offset + 5), (p.segment_bytes - 7, p.segment_bytes + 9),
                   (STORE.tensors[-1].offset + 1, p.w_bytes), (p.w_bytes - 1, p.w_bytes), (5, 5)]:
        assert read(lo, hi) == W[lo:hi], (lo, hi)
    with pytest.raises(ValueError):
        read(0, p.w_bytes + 1)


def test_commit_is_deterministic():
    """`commit_w` is toy.json's commitment at one and at three workers, and refuses a tensor whose SHA-256 differs."""
    toy, read = _toy_read()
    pieces = [(t.name, t.offset, t.size, hashlib.sha256(toy[t.name]).hexdigest()) for t in STORE.tensors]
    assert S.commit_w(STORE, read, 1, pieces).hex() == TOY["commitment"] == S.commit_w(STORE, read, 3, pieces).hex()
    bad = list(pieces)
    bad[3] = bad[3][:3] + (hashlib.sha256(b"").hexdigest(),)
    with pytest.raises(ValueError, match=r"\['model\.layers\.0\.mlp\.down_proj\.weight'\]"):
        S.commit_w(STORE, read, 3, bad)
    with pytest.raises(ValueError, match="holds 9 of the 10"):
        S.commit_w(STORE, read, 1, pieces + [("past", STORE.params.w_bytes, 1, "")])


def test_spot_vectors():
    """Blocks at the real store's indices (segments up to 1,037, v up to 65,535): keys, then decode."""
    real = S.build(S.MQA16K).params
    salt = bytes.fromhex(SPOT["salt"])
    assert SPOT["params"] == real.manifest() and salt == SALT
    prefix = expand_prefix(real, salt)
    assert prefix.hex() == SPOT["prefix"]
    ref = reference.Params(real.w, real.c0, real.payload_bytes, real.blocks_per_segment, real.segments)
    sha = lambda b: hashlib.sha512(b).hexdigest()  # noqa: E731
    spot_c = (S.TOY_DIR / "spot-c.bin").read_bytes()
    assert sha(spot_c) == SPOT["c_sha512"] and len(spot_c) == NB * len(SPOT["vectors"])
    assert [v["id"] for v in SPOT["vectors"]] == ["w-first", "qkv0-last", "down51-first", "head-last", "w-last"]
    for j, v in enumerate(SPOT["vectors"]):
        assert divmod(v["i"], real.blocks_per_segment) == (v["s"], v["v"])
        k, t, attempt = reference.keys_after(prefix, ref, v["s"], v["v"])
        assert (sha(k.to_bytes(NB, "big")), sha(t.to_bytes(NB, "big")), attempt) == (v["K_sha512"], v["t_sha512"], v["t_attempt"])
        m = S.decode_block(real, prefix, v["i"], spot_c[NB * j:NB * (j + 1)])
        assert m == S.spot_payload(v["i"], v["used"], real.payload_bytes) and sha(m) == v["m_sha512"]


@pytest.mark.slow
@pytest.mark.skipif(not os.environ.get("VERITY_POUS_SLOW"), reason="about two minutes of roots: VERITY_POUS_SLOW=1")
def test_toy_reencodes():
    seg = STORE.params.segment_bytes
    for s, blocks in enumerate(_segments()):
        assert encode_segment(STORE.params, None, SALT, s, W[s * seg:(s + 1) * seg]) == blocks


POINT = "128,173,16"
SOURCE = "11" * 32
GENERATOR = '''
import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class Shape:
    name: str
    width: int
    layers: int
    heads: int
    kv_heads: int
    head_dim: int
    mlp: int
    vocab: int


@dataclass(frozen=True)
class Tensor:
    name: str
    kind: str
    rows: int
    cols: int

    @property
    def nbytes(self):
        return self.rows * self.cols * (1 if self.kind == "s8" else 2)


SHAPES = {"t": Shape("t", 16, 2, 1, 1, 8, 32, 24)}


def tensors(g):
    out = [Tensor("tok", "embed", g.vocab, g.width)]
    for i in range(g.layers):
        hd = g.head_dim
        out += [Tensor(f"l{i}.norm", "norm", 1, g.width), Tensor(f"l{i}.wq", "s8", g.heads * hd, g.width),
                Tensor(f"l{i}.wk", "s8", g.kv_heads * hd, g.width), Tensor(f"l{i}.wv", "s8", g.kv_heads * hd, g.width),
                Tensor(f"l{i}.wo", "s8", g.width, g.heads * hd), Tensor(f"l{i}.wup", "s8", g.mlp, g.width),
                Tensor(f"l{i}.wdown", "s8", g.width, g.mlp), Tensor(f"l{i}.scale", "scale", 1, 4)]
    return out + [Tensor("head", "s8", g.vocab, g.width)]


def stream_bytes(g):
    return sum(t.nbytes for t in tensors(g))


def tensor_bytes(g, seed, name, start=0, stop=None):
    n = next(t.nbytes for t in tensors(g) if t.name == name)
    return hashlib.shake_256(seed + b"/" + name.encode()).digest(n)[start:n if stop is None else stop]


def program(g, seed):
    return "test/" + g.name + "/" + hashlib.sha512(seed).hexdigest()[:8]


def manifest(g, seed):
    return {"program": program(g, seed), "stream_bytes": stream_bytes(g),
            "tensors": [{"name": t.name, "kind": t.kind, "nbytes": t.nbytes,
                         "sha256": hashlib.sha256(tensor_bytes(g, seed, t.name)).hexdigest()} for t in tensors(g)]}
'''


@pytest.fixture()
def gen(tmp_path: Path) -> tuple[str, str]:
    """The small generator's file and its manifest's."""
    p, m = tmp_path / "gen.py", tmp_path / "manifest.json"
    p.write_text(GENERATOR)
    mod = S.load_generator(str(p))
    m.write_text(json.dumps(mod.manifest(mod.SHAPES["t"], b"seed")))
    return str(p), str(m)


def _flags(gen: tuple[str, str], store: Path, out: Path, *extra: str) -> list[str]:
    return ["encode", "--store", str(store), "--out", str(out), "--generator", gen[0], "--manifest", gen[1], "--seed",
            "text:seed", "--shape", "t", "--point", POINT, "--source", f"hex:{SOURCE}", "--workers", "1", *extra]


def _encode(gen: tuple[str, str], store: Path, out: Path) -> int:
    return S.main(_flags(gen, store, out, "--cpu-encoders", "2"))


def _store_segments(store: Path, q: P2Params) -> list[list[bytes]]:
    nb = q.block_bytes
    out = []
    for s in range(q.segments):
        c = (store / "seg" / f"{s:05d}.c").read_bytes()
        out.append([c[i * nb:(i + 1) * nb] for i in range(q.blocks_per_segment)])
    return out


def test_store_is_the_registered_encoding(gen: tuple[str, str], tmp_path: Path) -> None:
    store, out = tmp_path / "store", tmp_path / "out"
    assert _encode(gen, store, out) == 0
    doc = json.loads((out / "store.json").read_text())
    q = S.params_of(doc["manifest"])
    mod = S.load_generator(gen[0])
    g = mod.SHAPES["t"]
    lay = S.build(S.Shape(g.name, g.width, g.layers, g.heads, g.kv_heads, g.head_dim, g.mlp, g.vocab), 16, 128, 173)
    assert lay.params.manifest() == doc["manifest"] and doc["layout_sha512"] == S.layout_sha512(lay)
    assert q.segments == 16 and doc["checked_tensors"] == 13 and doc["tensor_bytes"] == 3456
    w = lay.layout.pack({t.name: b"".join(mod.tensor_bytes(g, b"seed", p) for p in S.generator_parts(t.name))
                         for t in lay.tensors})
    commitment = setup.commit(q, w)
    salt = setup.draw_salt(bytes.fromhex(SOURCE), commitment)
    assert (doc["commitment"], doc["salt"]) == (commitment.hex(), salt.hex())
    vk = setup.verifier_key_from_encoding(q, None, commitment, salt, _store_segments(store, q))
    assert doc["root"] == vk.root.hex()
    pb = q.payload_bytes
    w_leaves = [verifier.row_leaf(q, salt, i, verifier.row_digest(pb, w[i * pb:(i + 1) * pb])) for i in range(q.blocks)]
    assert doc["w_root"] == verifier.MerkleTree(w_leaves).root.hex()
    rows = b"".join((store / "rows" / f"{s:05d}.rows").read_bytes() for s in range(q.segments))
    assert VerifierKey.from_row_digests(q, salt, [rows[64 * i:64 * i + 64] for i in range(q.blocks)]).root == vk.root
    c_star = b"".join((store / "seg" / f"{s:05d}.c").read_bytes() for s in range(q.segments))
    for layer in range(g.layers + 1):
        assert S.decode_layer(lay, c_star, salt, layer) == S.staging_of(lay, w, layer)
    assert doc["tie"] == S.tie(doc) and doc["registration"] is None
    assert all(m["reference_spots"] for m in json.loads((out / "segments.json").read_text()))
    assert S.main(["verify", "--store", str(store), "--out", str(out), "--decode", "--workers", "2"]) == 0
    assert S.main(["spot", "--store", str(store), "--out", str(out), "--generator", gen[0], "--seed", "text:seed",
                   "--count", "16"]) == 0
    assert json.loads((out / "spot.json").read_text())["passed"]
    assert S.main(["tie", "--store", str(store), "--out", str(out), "--registration", "ab" * 64]) == 0
    tied = json.loads((store / "store.json").read_text())
    assert tied["tie"] == S.tie(tied) != doc["tie"]


def test_commit_refuses_another_manifest(gen: tuple[str, str], tmp_path: Path) -> None:
    doc = json.loads(Path(gen[1]).read_text())
    doc["tensors"][2]["sha256"] = "00" * 32
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="l0.wq"):
        S.main(["commit", "--store", str(tmp_path / "s"), "--out", str(tmp_path / "o"), "--generator", gen[0], "--manifest",
                str(bad), "--seed", "text:seed", "--shape", "t", "--point", POINT, "--workers", "1"])


def test_verify_catches_a_changed_byte(gen: tuple[str, str], tmp_path: Path) -> None:
    store, out = tmp_path / "store", tmp_path / "out"
    assert _encode(gen, store, out) == 0
    seg = store / "seg" / "00007.c"
    c = bytearray(seg.read_bytes())
    c[20] ^= 1
    seg.write_bytes(bytes(c))
    assert S.main(["verify", "--store", str(store), "--out", str(out), "--workers", "1"]) == 1
    report = json.loads((out / "verify.json").read_text())
    assert [b["segment"] for b in report["segments"]["bad"]] == [7] and not report["root_ok"]


def test_rerun_resumes_to_the_same_store(gen: tuple[str, str], tmp_path: Path) -> None:
    store, out = tmp_path / "store", tmp_path / "out"
    assert _encode(gen, store, out) == 0
    first = json.loads((out / "store.json").read_text())
    (store / "seg" / "00003.json").unlink()
    assert _encode(gen, store, tmp_path / "out2") == 0
    again = json.loads((tmp_path / "out2" / "store.json").read_text())
    assert again["encode"]["resumed"] == first["manifest"]["segments"] - 1
    assert (again["root"], again["w_root"], again["tie"]) == (first["root"], first["w_root"], first["tie"])
    with pytest.raises(SystemExit):
        S.main(_flags(gen, store, out)[:-4] + ["--source", f"hex:{'22' * 32}", "--workers", "1"])


def test_parts_join_to_the_whole_store(gen: tuple[str, str], tmp_path: Path) -> None:
    assert _encode(gen, tmp_path / "whole", tmp_path / "out") == 0
    whole = json.loads((tmp_path / "out" / "store.json").read_text())
    for name, part, order in (("a", "0:7", "up"), ("b", "7:16", "down")):
        assert S.main(_flags(gen, tmp_path / name, tmp_path / f"out-{name}", "--segments", part, "--order", order)) == 0
    assert S.main(["finish", "--store", str(tmp_path / "a"), "--out", str(tmp_path / "out-a")]) == 1
    for d in ("seg", "rows"):
        for f in (tmp_path / "b" / d).iterdir():
            (tmp_path / "a" / d / f.name).write_bytes(f.read_bytes())
    assert S.main(["finish", "--store", str(tmp_path / "a"), "--out", str(tmp_path / "out-a")]) == 0
    joined = json.loads((tmp_path / "out-a" / "store.json").read_text())
    assert (joined["root"], joined["w_root"], joined["tie"]) == (whole["root"], whole["w_root"], whole["tie"])


def test_rows_below_p() -> None:
    torch = pytest.importorskip("torch")
    p = (1 << 128) - 173
    vals = [0, 1, p - 1, p, p + 1, (1 << 128) - 1, p - (1 << 64)]
    rows = torch.tensor([list(v.to_bytes(16, "big")) for v in vals], dtype=torch.uint8)
    bound = torch.tensor(list(p.to_bytes(16, "big")), dtype=torch.uint8)
    assert S._rows_below(torch, rows, bound).tolist() == [v < p for v in vals]
