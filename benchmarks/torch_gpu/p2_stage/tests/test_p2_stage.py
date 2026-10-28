"""The staging decode's CPU parts: the store's layout as p2_v1's fixture gives it, the stream's bytes laid out as
the store's W, the scatter of decoded rows into a staging buffer, P2's reference decode of one block (pinned to the
spec's deployment vector), the fixture's toy store through the staging path under P2's reference, and the GEMM's twin
against `ScaledMmS8Coordinate_v1{K, GEMM_SM90_S8}`."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import layout as L  # noqa: E402

torch = pytest.importorskip("torch")
import gemm as G  # noqa: E402
import stage as S  # noqa: E402

ROOT = HERE.parents[2]
VECTORS = json.loads((ROOT / "benchmarks/pous/p2_v1/vectors.json").read_text())
TOY = ROOT / "benchmarks/pous/p2_v1/store-toy"
PREFIX = L.registered_prefix(bytes(range(24)))


def model(width: int, layers: int, heads: int, ff: int, vocab: int = 256) -> list[tuple[str, int, int, int]]:
    """`(name, layer, N, K)` in the fixture's order: each layer's QKV (one KV head of 64), O, up and down, then the head."""
    out = []
    for i in range(layers):
        out += [(L.matrix_name(i, "qkv"), i, (heads + 2) * 64, width), (L.matrix_name(i, "o"), i, width, heads * 64),
                (L.matrix_name(i, "up"), i, ff, width), (L.matrix_name(i, "down"), i, width, ff)]
    return out + [("lm_head.weight", layers, vocab, width)]


def layout_doc(tensors: list[tuple[str, int, int, int]], per_segment: int = 1 << 16) -> dict:
    """A layout in the fixture's format by its rules: each tensor from a block boundary, back to back by blocks; each
    layer's staging buffer from 0, every tensor at a multiple of 256."""
    rows, block, layer, staging = [], 0, None, 0
    for name, i, n, k in tensors:
        if i != layer:
            layer, staging = i, 0
        blocks = -(-n * k // L.PAYLOAD)
        rows.append([name, i, n, k, block * L.PAYLOAD, n * k, block, blocks, staging])
        block += blocks
        staging += -(-n * k // 256) * 256
    p = L.S.P2Params(16448, 21065, per_segment, max(1, -(-block // per_segment)))
    return {"schema": L.SCHEMA, "scheme": L.SCHEME, "params": p.manifest(), "block_bytes": p.block_bytes,
            "payload_bytes": p.payload_bytes, "blocks": p.blocks, "tensor_columns": list(L.COLUMNS), "tensors": rows}


def stream_manifest(width: int = 64, mlp: int = 256, layers: int = 2) -> dict:
    """A small weights-stream manifest with the frozen shape's tensor order (one query head, one KV head of 64)."""
    rows, at = [], 0

    def add(name, shape, size):
        nonlocal at
        rows.append({"name": name, "offset": at, "nbytes": size, "shape": shape})
        at += size

    add("tok", [width, 256], 2 * width * 256)
    for i in range(layers):
        add(f"l{i}.attn_norm", [width], 2 * width)
        for t, n, k in (("wq", 64, width), ("wk", 64, width), ("wv", 64, width), ("wo", width, 64)):
            add(f"l{i}.{t}", [n, k], n * k)
            add(f"l{i}.{t}.scale", [n], 4 * n)
        add(f"l{i}.mlp_norm", [width], 2 * width)
        for t, n, k in (("wup", mlp, width), ("wdown", width, mlp)):
            add(f"l{i}.{t}", [n, k], n * k)
            add(f"l{i}.{t}.scale", [n], 4 * n)
    return {"tensors": rows, "stream_bytes": at}


def test_blocks_of():
    assert L.blocks_of(0, 0) == range(0)
    assert L.blocks_of(0, 1) == range(0, 1)
    assert L.blocks_of(2053, 2) == range(0, 2)
    assert L.blocks_of(2054, 2054) == range(1, 2)
    assert L.blocks_of(10, 3 * 2054) == range(0, 4)
    with pytest.raises(ValueError):
        L.blocks_of(-1, 2)


def test_store_layout_at_the_frozen_shape():
    """The frozen layout's numbers: 1,038 segments; layer l from block 1,307,915 l; QKV, O, up and down at blocks 0, 131,711,
    262,401 and 785,158 of it and staging offsets 0, 270,532,608, 538,968,064 and 1,612,709,888; the head at block
    68,011,580 over 2,043 blocks."""
    lay = L.store_layout(layout_doc(model(16384, 52, 256, 65536)), PREFIX)
    assert (lay.params.segments, lay.params.blocks) == (1038, 68_026_368)
    for i in (0, 1, 51):
        ts = lay.layer(i)
        assert [t.matrix for t in ts] == list(L.MATRICES)
        assert [t.first_block - 1_307_915 * i for t in ts] == [0, 131_711, 262_401, 785_158]
        assert [t.blocks for t in ts] == [131_711, 130_690, 522_757, 522_757]
        assert [t.staging_offset for t in ts] == [0, 270_532_608, 538_968_064, 1_612_709_888]
        assert lay.blocks(i) == range(1_307_915 * i, 1_307_915 * (i + 1))
        assert lay.staging_bytes(i) == 2_686_451_712
    qkv = lay.tensor(0, "qkv")
    assert (qkv.n, qkv.k, qkv.name) == (16512, 16384, "model.layers.0.self_attn.qkv_proj.weight")
    assert lay.tensor(0, qkv.name) == qkv
    (head,) = lay.layer(52)
    assert (head.first_block, head.blocks, head.matrix) == (68_011_580, 2_043, None)
    assert lay.spec.blocks_per_segment == 65536 and lay.spec.p == L.SPEC.p
    with pytest.raises(KeyError):
        lay.tensor(0, "gate")


def test_store_layout_reads_the_toy_doc_and_refuses_other_layouts():
    doc = layout_doc(model(64, 2, 1, 64), per_segment=11)
    lay = L.store_layout({"layout": doc}, PREFIX)
    assert lay.params.blocks_per_segment == 11 and lay.layer(2)[0].name == "lm_head.weight"
    for bad in ({"schema": "other"}, {"scheme": "p2-16448/v3"}, {"blocks": doc["blocks"] + 11},
                {"params": {**doc["params"], "c0": 21069}}):
        with pytest.raises(ValueError):
            L.store_layout({**doc, **bad}, PREFIX)
    rows = doc["tensors"]
    for i, col, delta in ((1, 4, 1), (1, 7, 1), (1, 6, -1), (1, 8, -256), (0, 2, 1)):
        changed = [list(r) for r in rows]
        changed[i][col] += delta
        with pytest.raises(ValueError):
            L.store_layout({**doc, "tensors": changed}, PREFIX)
    with pytest.raises(ValueError):
        L.store_layout(doc, PREFIX[:-1])


@pytest.mark.skipif(not (TOY / "toy.json").exists(), reason="p2_v1's toy store (benchmarks/pous/p2_v1/store-toy) isn't here")
def test_layout_doc_is_the_toys():
    toy = json.loads((TOY / "toy.json").read_text())["layout"]
    assert layout_doc(model(64, 2, 1, 64), per_segment=11) == {k: toy[k] for k in layout_doc([]).keys()}


def test_payloads_are_the_streams_bytes_block_aligned():
    doc = stream_manifest()
    lay = L.store_layout(layout_doc(model(64, 2, 1, 256)), PREFIX)
    src = L.Weights.of(doc)
    w = np.random.default_rng(1).integers(0, 256, doc["stream_bytes"], dtype=np.uint8).tobytes()
    read = lambda o, n: w[o:o + n]  # noqa: E731
    assert src.shape(1, "qkv") == (192, 64) and src.shape(1, "down") == (64, 256)
    by = {t["name"]: t for t in doc["tensors"]}
    for t in lay.layer(1):
        want = b"".join(w[by[f"l1.{p}"]["offset"]:][:by[f"l1.{p}"]["nbytes"]] for p in L.PARTS[t.matrix])
        got = L.payloads(lay, src, read, t.first_block, t.blocks)
        assert got[:t.size] == want and not any(got[t.size:])
    whole = L.payloads(lay, src, read, lay.blocks(1).start, len(lay.blocks(1)))
    assert len(whole) == len(lay.blocks(1)) * L.PAYLOAD
    lo, hi = src.span(1)
    assert lo == by["l1.wq"]["offset"] and hi == by["l1.wdown.scale"]["offset"] + by["l1.wdown.scale"]["nbytes"]
    with pytest.raises(KeyError):
        L.payloads(lay, src, read, lay.layer(2)[0].first_block, 1)
    narrow = L.store_layout(layout_doc(model(64, 2, 1, 128)), PREFIX)
    with pytest.raises(ValueError, match="in the stream"):
        L.payloads(narrow, src, read, narrow.tensor(0, "up").first_block, 1)


@pytest.mark.parametrize("case", range(40))
def test_scatter_matches_the_payload_stream(case):
    rng = random.Random(case)
    n = rng.choice([1, 2, 3, 7])
    j0 = rng.randrange(0, 50)
    m = torch.randint(0, 256, (n, L.ROW), dtype=torch.uint8, generator=torch.Generator().manual_seed(case))
    payload = m[:, L.LEAD:].reshape(-1).numpy()
    base = j0 * L.PAYLOAD
    kind = case % 4
    if kind == 0:
        offset, size = base + rng.randrange(0, 2 * L.PAYLOAD), rng.randrange(1, 3 * L.PAYLOAD)
    elif kind == 1:
        offset, size = base - rng.randrange(0, 4000), rng.randrange(1, 9000)
    elif kind == 2:
        offset, size = base + rng.randrange(0, n) * L.PAYLOAD, rng.randrange(1, n + 1) * L.PAYLOAD
    else:
        offset = base + rng.randrange(0, n * L.PAYLOAD)
        size = rng.randrange(1, 40)
    offset = max(0, offset)
    out = torch.full((size,), 0xA5, dtype=torch.uint8)
    copied = S.scatter(m, j0, out, offset, size)
    want = np.full(size, 0xA5, dtype=np.uint8)
    lo, hi = max(offset, base), min(offset + size, base + n * L.PAYLOAD)
    if lo < hi:
        want[lo - offset:hi - offset] = payload[lo - base:hi - base]
    assert copied == max(0, hi - lo)
    assert np.array_equal(out.numpy(), want)


def test_flat_views_any_dtype():
    w = torch.zeros(3, 4, dtype=torch.int8)
    S.flat(w).fill_(0x80)
    assert (w == -128).all()
    s = torch.zeros(5, dtype=torch.float32)
    assert S.flat(s).numel() == 20
    with pytest.raises(ValueError):
        S.flat(torch.zeros(4, 4).t())


def deployment_vector():
    v = next(v for v in VECTORS["vectors"] if v["id"] == "p2-v4/w16448/0")
    prefix = L.R.expand_prefix(bytes.fromhex(v["setup_salt"]), L.SPEC, bytes.fromhex(v["model_commitment"]))
    j = int(v["segment"]) * L.PARAMS.blocks_per_segment + int(v["block"])
    return prefix, j, bytes.fromhex(v["c"]), bytes.fromhex(v["m"])


def test_check_blocks_holds_the_spec_vector():
    prefix, j, c, m = deployment_vector()
    assert m[:L.LEAD] == bytes(L.LEAD)
    assert L.check_blocks(L.PARAMS, prefix, j, c, m[L.LEAD:]) == []


def test_check_blocks_refuses_what_the_store_refuses():
    prefix, j, c, m = deployment_vector()
    [(at, why)] = L.check_blocks(L.PARAMS, prefix, j, S.P_BYTES, m[L.LEAD:])
    assert at == j and "not below p" in why
    bad = L.check_blocks(L.PARAMS, prefix, j + 1, c, m[L.LEAD:])
    assert len(bad) == 1 and bad[0][0] == j + 1
    flipped = bytes([c[0], c[1] ^ 1]) + c[2:]
    assert len(L.check_blocks(L.PARAMS, prefix, j, flipped, m[L.LEAD:])) == 1


def test_twin_is_the_definition():
    gen = torch.Generator().manual_seed(7)
    M, N, K = 5, 4, 64
    x = torch.randint(-128, 128, (M, K), dtype=torch.int8, generator=gen)
    w = torch.randint(-128, 128, (N, K), dtype=torch.int8, generator=gen)
    x[0], w[0], x[1], w[1] = -128, -128, 127, -128
    sx = (torch.rand(M, generator=gen) + 1) * 2.0 ** -8
    sw = (torch.rand(N, generator=gen) + 1) * 2.0 ** -14
    sx[2] = 2.0 ** 100
    words = G.twin_words(x, sx, w, sw).numpy().astype(np.uint16)
    mm, nn = np.meshgrid(np.arange(M), np.arange(N), indexing="ij")
    mm, nn = mm.reshape(-1), nn.reshape(-1)
    want = G.definition_words(K, x.numpy()[mm], sx.numpy()[mm], w.numpy()[nn], sw.numpy()[nn])
    assert np.array_equal(words.reshape(-1), want)


def test_twin_counts_a_changed_word():
    gen = torch.Generator().manual_seed(3)
    M, N, K = 6, 8, 32
    x = torch.randint(-128, 128, (M, K), dtype=torch.int8, generator=gen)
    w = torch.randint(-128, 128, (N, K), dtype=torch.int8, generator=gen)
    sx, sw = torch.rand(M, generator=gen) + 0.5, torch.rand(N, generator=gen) * 1e-3 + 1e-4
    y = G.twin_words(x, sx, w, sw).to(torch.int16).view(torch.bfloat16).clone()
    assert G.twin(x, sx, w, sw, y, block=4)["mismatches"] == 0
    y.view(torch.int16)[3, 5] ^= 1
    got = G.twin(x, sx, w, sw, y, block=4)
    assert got["mismatches"] == 1 and got["first"][0]["m"] == 3 and got["first"][0]["n"] == 5


def test_bf16_rn_is_round_to_nearest_even():
    f = torch.tensor([1.0, 1.00390625, 1.01171875, -2.5, 0.0, 3.0e38, float("inf")], dtype=torch.float32)
    want = f.to(torch.bfloat16).view(torch.int16).to(torch.int32) & 0xFFFF
    assert torch.equal(G.bf16_rn(f), want)


class RefKernels:
    """P2's reference (`reference.decode` under `reference.keys_after`) behind `p2_v1.gpu.Kernels`'s interface, on the
    CPU. A row at or above p decodes to zeros, so refusing it is left to the staging decode's own screen."""

    def __init__(self, lib_dir, build_dir, params, prefix):
        self.params, self.prefix, self.device = params, bytes(prefix), torch.device("cpu")

    def decode(self, c, first=0):
        out = torch.zeros_like(c)
        for i in range(c.shape[0]):
            v = int.from_bytes(bytes(c[i].numpy()), "big")
            if v < self.params.p:
                k, t, _ = L.R.keys_after(self.prefix, self.params, *divmod(first + i, self.params.blocks_per_segment))
                m = L.R.decode(self.params, v, k, t).to_bytes(L.ROW, "big")
                out[i] = torch.frombuffer(bytearray(m), dtype=torch.uint8)
        return out

    def raised(self):
        return 0


@pytest.mark.skipif(not (TOY / "toy.json").exists(), reason="p2_v1's toy store (benchmarks/pous/p2_v1/store-toy) isn't here")
def test_toy_store_stages_as_the_fixture_says(tmp_path, monkeypatch):
    """The toy's C* through `decode_layer` under P2's reference: each layer's staging buffer by its digest, each negative
    block refused with its flag bit, another salt refused, and the spot vectors at the real store's indices."""
    import layer

    monkeypatch.setattr(S.gpu, "Kernels", RefKernels)
    monkeypatch.setattr(layer, "DEVICE", torch.device("cpu"))
    out = layer.Out(tmp_path)
    assert layer.toy(argparse.Namespace(toy=TOY, lib="unused"), out)
    rec = json.loads((tmp_path / "layer.json").read_text().splitlines()[-1])
    assert [n["flags"] for n in rec["negatives"]] == [1 << n["flag"] for n in rec["negatives"]] == [8, 4, 4]
    assert len(rec["staging"]) == 3 and len(rec["spots"]) == 5 and rec["other_salt_flags"] & S.WIDE


class XorKernels:
    """A stand-in for `p2_v1.gpu.Kernels` on the CPU, its own inverse: block j's payload XOR a keystream of (prefix, j),
    its two lead bytes XOR a hash of (prefix, j, the row's payload before and after), so a corrupt row or other keys
    decode wider than the payload, as P2's do. The driver's plumbing (store, scatter, screen, staging, GEMM check) runs
    without the GPU library. Not P2."""

    def __init__(self, lib_dir, build_dir, params, prefix):
        self.prefix, self.device = bytes(prefix), torch.device("cpu")

    def decode(self, c, first=0):
        out = c.clone()
        for i in range(c.shape[0]):
            key = self.prefix + (first + i).to_bytes(8, "big")
            pad = np.frombuffer(hashlib.shake_256(key).digest(L.PAYLOAD), dtype=np.uint8)
            payload = c[i, L.LEAD:].numpy()
            pair = sorted((payload.tobytes(), (payload ^ pad).tobytes()))
            lead = np.frombuffer(hashlib.shake_256(key + pair[0] + pair[1]).digest(L.LEAD), dtype=np.uint8)
            out[i, L.LEAD:] = torch.from_numpy(payload ^ pad)
            out[i, :L.LEAD] = torch.from_numpy(c[i, :L.LEAD].numpy() ^ lead)
        return out

    encode = decode

    def raised(self):
        return 0


def test_decode_layer_zeroes_the_gaps_and_refuses_padding(monkeypatch):
    monkeypatch.setattr(S.gpu, "Kernels", XorKernels)
    lay = L.store_layout(layout_doc(model(64, 2, 1, 256), per_segment=11), PREFIX)
    blocks = lay.blocks(1)
    w = np.random.default_rng(2).integers(0, 256, len(blocks) * L.PAYLOAD, dtype=np.uint8)
    for t in lay.layer(1):
        w[t.offset + t.size - blocks.start * L.PAYLOAD:(t.first_block + t.blocks - blocks.start) * L.PAYLOAD] = 0
    kern = XorKernels(None, None, None, PREFIX)
    rows = kern.encode(torch.cat([torch.zeros(len(blocks), L.LEAD, dtype=torch.uint8),
                                  torch.from_numpy(w).view(-1, L.PAYLOAD)], dim=1), blocks.start)
    buf = torch.full((lay.staging_bytes(1),), 0xA5, dtype=torch.uint8)
    S.decode_layer(S.Store(lay, rows, blocks.start, "unused"), 1, buf)
    at = 0
    for t in lay.layer(1):
        assert not buf[at:t.staging_offset].any()
        rel = t.offset - blocks.start * L.PAYLOAD
        assert bytes(buf[t.staging_offset:t.staging_offset + t.size].numpy()) == w[rel:rel + t.size].tobytes()
        at = t.staging_offset + t.size
    o = lay.tensor(1, "o")
    m = kern.decode(rows[o.first_block + o.blocks - 1 - blocks.start:][:1], o.first_block + o.blocks - 1)
    m[0, -1] ^= 1
    bad = rows.clone()
    bad[o.first_block + o.blocks - 1 - blocks.start] = kern.encode(m, o.first_block + o.blocks - 1)[0]
    with pytest.raises(S.StoreError) as e:
        S.decode_layer(S.Store(lay, bad, blocks.start, "unused"), 1, buf)
    assert e.value.flags == S.PADDING
    with pytest.raises(ValueError):
        S.decode_layer(S.Store(lay, rows, blocks.start, "unused"), 1, buf[:-1])


def test_layer_driver_stages_and_checks_on_a_stand_in_codec(tmp_path, monkeypatch):
    import check
    import layer

    monkeypatch.setattr(S.gpu, "Kernels", XorKernels)
    monkeypatch.setattr(layer.gpu, "Kernels", XorKernels)
    doc = stream_manifest()
    w = np.random.default_rng(5).integers(0, 256, doc["stream_bytes"], dtype=np.uint8)
    for t in doc["tensors"]:
        if t["name"].endswith(".scale"):
            n = t["nbytes"] // 4
            w[t["offset"]:t["offset"] + t["nbytes"]] = ((np.arange(n, dtype=np.float32) + 1) * 2.0 ** -14).view(np.uint8)
    (tmp_path / "m.json").write_text(json.dumps(doc))
    (tmp_path / "w.bin").write_bytes(w.tobytes())
    layout = layout_doc(model(64, 2, 1, 256), per_segment=11)
    (tmp_path / "layout.json").write_text(json.dumps(layout))
    out = tmp_path / "out"
    common = ["--layout", str(tmp_path / "layout.json"), "--weights", str(tmp_path / "m.json"), "--w", str(tmp_path / "w.bin"),
              "--store", str(tmp_path / "store"), "--salt", "00" * 24, "--out", str(out), "--layer", "1"]
    base = common + ["--lib", "unused", "--device", "cpu", "--m", "16,32", "--reps", "1", "--samples", "4",
                     "--encode-chunk", "3"]
    lay = L.store_layout(layout, L.registered_prefix(bytes(24)))
    assert layer.main(base + ["--encode-blocks", str(lay.tensor(1, "qkv").blocks)]) == 0
    recs = [json.loads(x) for x in (out / "layer.json").read_text().splitlines()]
    assert [r["ready"] for r in recs if "ready" in r] == [["qkv"]]
    assert layer.main(base) == 0
    recs = [json.loads(x) for x in (out / "layer.json").read_text().splitlines()]
    staged = [r for r in recs if r["step"] == "stage" and r.get("matrix") in L.MATRICES]
    assert [r["matrix"] for r in staged] == ["qkv", "qkv", "o", "up", "down"]
    assert all(r["weights"]["mismatched_bytes"] == 0 for r in staged)
    whole = [r for r in recs if r.get("matrix") == "layer"]
    assert len(whole) == 1 and whole[0]["matches_matrices"] and whole[0]["nonzero_gap_bytes"] == 0
    gemms = [r for r in recs if r["step"] == "gemm"]
    assert len(gemms) == 2 + 8 and all(r["twin"]["mismatches"] == 0 for r in gemms)
    assert [r["m"] for r in gemms[-8:]] == [16] * 4 + [32] * 4
    assert [r["k"] for r in gemms[-4:]] == [64, 64, 64, 256] and [r["n"] for r in gemms[-4:]] == [192, 64, 256, 64]
    ctl = [r for r in recs if r["step"] == "controls"]
    assert all(c["ok"] for c in ctl) and set(ctl[-1]["results"]) == {"other_keys", "corrupt_row", "row_at_p", "honest",
                                                                       "padding"}
    assert ctl[-1]["results"]["padding"]["flags"] == S.PADDING
    assert [r["m"] for r in recs if r["step"] == "summary"] == [16, 32]
    res = check.check_gemm(out)
    assert set(res) == {f"{m}-m{M}" for m in L.MATRICES for M in (16, 32)}
    assert all(v["mismatches"] == 0 and v["samples"] == 4 for v in res.values())
    with pytest.raises(SystemExit, match="another store"):
        layer.main([a.replace("00" * 24, "01" * 24) for a in base] + ["--steps", "stage"])


def encoded_store(root: Path, rows: np.ndarray, per_segment: int, setup: dict) -> None:
    """An encoded store's directory as `p2_v1/store.py encode` writes it: `setup.json`, and each segment's rows in
    `seg/<s>.c` with their SHA-512 in `seg/<s>.json`."""
    (root / "seg").mkdir(parents=True)
    (root / "setup.json").write_text(json.dumps(setup))
    for s in range(-(-len(rows) // per_segment)):
        data = rows[s * per_segment:(s + 1) * per_segment].tobytes()
        L.segment_path(root, s, "c").write_bytes(data)
        L.segment_path(root, s, "json").write_text(json.dumps({"segment": s, "sha512": hashlib.sha512(data).hexdigest()}))


@pytest.mark.skipif(not (TOY / "toy.json").exists(), reason="p2_v1's toy store (benchmarks/pous/p2_v1/store-toy) isn't here")
def test_the_toys_coin_and_segments_stage_as_an_encoded_store(tmp_path, monkeypatch):
    """The toy's coin and commitment give its salt; its C* as an encoded store's segment files stages each layer to
    toy.json's digest under P2's reference; a setup of another commitment, layout, parameters or salt is refused, and so
    is a segment unfinished or not the SHA-512 its JSON records."""
    import layer

    doc = json.loads((TOY / "toy.json").read_text())
    (tmp_path / "layout.json").write_text(json.dumps(doc["layout"]))
    lsha = hashlib.sha512((tmp_path / "layout.json").read_bytes()).hexdigest()
    setup = {"commitment": doc["commitment"], "layout_sha512": lsha, "manifest": doc["layout"]["params"]}
    coin, com = bytes.fromhex(doc["source"]), bytes.fromhex(doc["commitment"])
    assert L.store_salt(setup, doc["layout"], lsha, coin, com).hex() == doc["salt"]
    full = {**setup, "source": doc["source"], "salt": doc["salt"]}
    assert L.store_salt(full, doc["layout"], lsha, coin, com).hex() == doc["salt"]
    other = coin[:-1] + bytes([coin[-1] ^ 1])
    for bad, c in (({**setup, "commitment": "00" * 64}, coin), ({**setup, "layout_sha512": "00"}, coin),
                   ({k: v for k, v in setup.items() if k != "manifest"}, coin), (full, other)):
        with pytest.raises(ValueError):
            L.store_salt(bad, doc["layout"], lsha, c, com)
    c_star = np.frombuffer((TOY / "c-star.bin").read_bytes(), dtype=np.uint8).reshape(-1, L.ROW)
    bps = doc["layout"]["params"]["blocks_per_segment"]
    root = tmp_path / "store"
    encoded_store(root, c_star, bps, setup)
    lay, got = L.open_store(tmp_path / "layout.json", {}, root, doc["source"], doc["commitment"], None)
    assert got == setup and lay.prefix.hex() == doc["prefix"]
    monkeypatch.setattr(S.gpu, "Kernels", RefKernels)
    monkeypatch.setattr(layer, "DEVICE", torch.device("cpu"))
    for i, _, _, sha in doc["staging"]:
        blocks = lay.blocks(i)
        rows = layer.EncodedStore(root, lay, blocks, setup).rows(blocks)
        assert bytes(rows.numpy()) == L.segment_rows(root, lay.spec, blocks.start, len(blocks)) \
            == c_star[blocks.start:blocks.stop].tobytes()
        buf = torch.full((lay.staging_bytes(i),), 0xA5, dtype=torch.uint8)
        S.decode_layer(S.Store(lay, rows, blocks.start, "unused"), i, buf)
        assert hashlib.sha512(buf.numpy().tobytes()).hexdigest() == sha
    s = (lay.blocks(0).stop - 1) // bps
    L.segment_path(root, s, "c").write_bytes(bytes(bps * L.ROW))
    with pytest.raises(SystemExit, match="SHA-512"):
        layer.EncodedStore(root, lay, lay.blocks(0), setup).rows(lay.blocks(0))
    L.segment_path(root, s, "json").unlink()
    with pytest.raises(SystemExit, match="not done"):
        layer.EncodedStore(root, lay, lay.blocks(0), setup).rows(lay.blocks(0))


def test_layer_driver_reads_an_encoded_store_without_the_stream(tmp_path, monkeypatch):
    """An encoded store's segment files (rows by the stand-in codec under the prefix of `draw_salt(coin, commitment)`)
    through the driver with no stream file: each staged tensor held to the weights manifest's SHA-256, the scales and the
    padding control's last block from the generator. Refused: an encode step, another commitment, a store of other
    weights, an unfinished segment; a tensor whose digest differs fails the run."""
    import layer
    from verity.core.protocols.pous import setup as P

    monkeypatch.setattr(S.gpu, "Kernels", XorKernels)
    monkeypatch.setattr(layer.gpu, "Kernels", XorKernels)
    doc = stream_manifest()
    w = np.random.default_rng(7).integers(0, 256, doc["stream_bytes"], dtype=np.uint8)
    for t in doc["tensors"]:
        if t["name"].endswith(".scale"):
            n = t["nbytes"] // 4
            w[t["offset"]:t["offset"] + t["nbytes"]] = ((np.arange(n, dtype=np.float32) + 1) * 2.0 ** -14).view(np.uint8)
        t["sha256"] = hashlib.sha256(w[t["offset"]:t["offset"] + t["nbytes"]]).hexdigest()
    doc["program"] = "test/program"
    (tmp_path / "m.json").write_text(json.dumps(doc))
    (tmp_path / "w.bin").write_bytes(w.tobytes())
    (tmp_path / "fakegen.py").write_text(
        "from pathlib import Path\nALIASES = {'tiny': 'tiny'}\nSHAPES = ALIASES\n"
        f"W = Path({str(tmp_path / 'w.bin')!r})\n\n\n"
        "def stream(shape, seed, start, stop):\n    assert seed == b'seed'\n    return W.read_bytes()[start:stop]\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    layout = layout_doc(model(64, 2, 1, 256), per_segment=11)
    (tmp_path / "layout.json").write_text(json.dumps(layout))
    coin, com = bytes(range(64)), bytes(range(64, 128))
    lay = L.store_layout(layout, L.registered_prefix(P.draw_salt(coin, com)))
    used = lay.blocks(1).stop
    m = np.zeros((lay.params.blocks, L.ROW), dtype=np.uint8)
    m[:used, L.LEAD:] = np.frombuffer(L.payloads(lay, L.Weights.of(doc), L.reader(tmp_path / "w.bin"), 0, used),
                                      dtype=np.uint8).reshape(used, L.PAYLOAD)
    rows = XorKernels(None, None, None, lay.prefix).encode(torch.from_numpy(m), 0).numpy()
    setup = {"commitment": com.hex(), "layout_sha512": hashlib.sha512((tmp_path / "layout.json").read_bytes()).hexdigest(),
             "manifest": layout["params"], "program": "test/program"}
    root = tmp_path / "store"
    encoded_store(root, rows, 11, setup)
    out = tmp_path / "out"
    args = ["--layout", str(tmp_path / "layout.json"), "--weights", str(tmp_path / "m.json"), "--encoded", str(root),
            "--source", coin.hex(), "--commitment", com.hex(), "--generator", "fakegen", "--shape", "tiny",
            "--weights-seed", "text:seed", "--lib", "unused", "--device", "cpu", "--m", "16", "--reps", "1",
            "--samples", "4", "--out", str(out), "--layer", "1", "--steps", "stage,gemm"]
    assert layer.main(args) == 0
    recs = [json.loads(x) for x in (out / "layer.json").read_text().splitlines()]
    assert recs[0]["kind"] == "encoded" and recs[0]["encoded"] == len(lay.blocks(1))
    staged = [r for r in recs if r["step"] == "stage" and r.get("matrix") in L.MATRICES]
    assert [r["matrix"] for r in staged] == list(L.MATRICES)
    assert all(r["weights"] is None and all(r["manifest_sha256"].values()) for r in staged)
    assert staged[0]["manifest_sha256"] == {"wq": True, "wk": True, "wv": True}
    ctl = [r for r in recs if r["step"] == "controls"][-1]
    assert ctl["ok"] and ctl["results"]["padding"]["flags"] == S.PADDING
    assert [r["m"] for r in recs if r["step"] == "summary"] == [16]
    with pytest.raises(SystemExit):
        layer.main(args[:-1] + ["encode,stage"])
    with pytest.raises(SystemExit, match="commitment"):
        layer.main([a.replace(com.hex(), "00" * 64) for a in args])
    (tmp_path / "m.json").write_text(json.dumps({**doc, "program": "other"}))
    with pytest.raises(SystemExit, match="weights manifest"):
        layer.main(args)
    bad = json.loads(json.dumps(doc))
    next(t for t in bad["tensors"] if t["name"] == "l1.wo")["sha256"] = "00" * 32
    (tmp_path / "m.json").write_text(json.dumps(bad))
    assert layer.main(args + ["--steps", "stage"]) == 1
    (tmp_path / "m.json").write_text(json.dumps(doc))
    L.segment_path(root, lay.blocks(1).start // 11, "json").unlink()
    with pytest.raises(SystemExit, match="not done"):
        layer.main(args)


def test_sample_blocks_takes_each_tensors_ends_and_a_seeded_draw():
    import check

    lay = L.store_layout(layout_doc(model(64, 2, 1, 256), per_segment=11), PREFIX)
    got = check.sample_blocks(list(lay.tensors), 4, 3)
    assert got == sorted(set(got)) == check.sample_blocks(list(lay.tensors), 4, 3)
    for t in lay.tensors:
        mine = [j for j in got if j in t.block_range]
        assert t.first_block in mine and t.block_range.stop - 1 in mine and len(mine) == min(4, t.blocks)
    assert check.sample_blocks(list(lay.tensors), 1000, 3) == list(range(lay.tensors[-1].block_range.stop))
    assert check.sample_blocks(list(lay.tensors), 2, 3) == sorted({j for t in lay.tensors
                                                                   for j in (t.first_block, t.block_range.stop - 1)})


def test_layer_set():
    import full

    assert full.layer_set("0-2,5", 5) == {0, 1, 2, 5} and full.layer_set("3", 3) == {3}
    for bad in ("0-6", "-1", "2-1"):
        with pytest.raises(ValueError):
            full.layer_set(bad, 5)


TOY_W = b"verity-pous/p2-store/toy/w/v1\0"


def toy_stream(doc: dict) -> tuple[dict, bytes]:
    """The toy's W (each tensor `p2.stream(TOY_W, name, size)` at its offset, as `p2_v1/store.py`'s `toy_plain` packs
    it, held to toy.json's digest) and then each stream tensor's f32 scales: a weights stream, and its manifest with
    SHA-256s naming each store tensor's stream tensors (QKV as wq, wk and wv of 64 rows each but the query's) in W."""
    lay = L.store_layout(doc["layout"], PREFIX)
    w = bytearray(lay.params.blocks * L.PAYLOAD)
    for t in lay.tensors:
        w[t.offset:t.offset + t.size] = L.S.stream(TOY_W, t.name.encode(), t.size)
    assert hashlib.sha512(w).hexdigest() == doc["w_sha512"]
    rows, scales = [], bytearray()
    for t in lay.tensors:
        ns = [t.n - 128, 64, 64] if t.matrix == "qkv" else [t.n]
        at = t.offset
        for name, n in zip(t.stream_names, ns):
            rows.append({"name": name, "offset": at, "nbytes": n * t.k, "shape": [n, t.k],
                         "sha256": hashlib.sha256(w[at:at + n * t.k]).hexdigest()})
            s = ((np.arange(n, dtype=np.float32) + 1) * 2.0 ** -14).tobytes()
            rows.append({"name": f"{name}.scale", "offset": len(w) + len(scales), "nbytes": len(s), "shape": [n],
                         "sha256": hashlib.sha256(s).hexdigest()})
            scales += s
            at += n * t.k
    return {"tensors": rows, "stream_bytes": len(w) + len(scales)}, bytes(w + scales)


@pytest.mark.skipif(not (TOY / "toy.json").exists(), reason="p2_v1's toy store (benchmarks/pous/p2_v1/store-toy) isn't here")
def test_full_stages_the_whole_toy_store_and_check_samples_it(tmp_path, monkeypatch):
    """The toy's C* as an encoded store through `full.py` under P2's reference: every tensor of every layer and the head
    staged and held to the manifest's SHA-256, the step's decode summed over them, one layer's GEMM in chunks of output
    channels against the twin and the Definition; `check.py --sample` on each tensor's blocks against P2's reference,
    which finds one changed byte of W. Refused or failed: a tensor whose digest differs, a manifest with no digests or a
    stream tensor of another shape, a segment not its SHA-512 or not done."""
    import check
    import full

    doc = json.loads((TOY / "toy.json").read_text())
    manifest, stream = toy_stream(doc)
    (tmp_path / "layout.json").write_text(json.dumps(doc["layout"]))
    (tmp_path / "m.json").write_text(json.dumps(manifest))
    (tmp_path / "w.bin").write_bytes(stream)
    setup = {"commitment": doc["commitment"], "manifest": doc["layout"]["params"],
             "layout_sha512": hashlib.sha512((tmp_path / "layout.json").read_bytes()).hexdigest()}
    bps = doc["layout"]["params"]["blocks_per_segment"]
    root = tmp_path / "store"
    encoded_store(root, np.frombuffer((TOY / "c-star.bin").read_bytes(), dtype=np.uint8).reshape(-1, L.ROW), bps, setup)
    monkeypatch.setattr(S.gpu, "Kernels", RefKernels)
    store = ["--layout", str(tmp_path / "layout.json"), "--weights", str(tmp_path / "m.json"), "--w", str(tmp_path / "w.bin"),
             "--encoded", str(root), "--source", doc["source"], "--commitment", doc["commitment"]]
    run = store + ["--lib", "unused", "--device", "cpu", "--reps", "1", "--readers", "2", "--hashers", "2"]
    out = tmp_path / "out"
    assert full.main(run + ["--out", str(out), "--gemm-layer", "1", "--m", "16", "--n-chunk", "64", "--samples", "4"]) == 0
    recs = [json.loads(x) for x in (out / "full.json").read_text().splitlines()]
    lay = L.store_layout(doc["layout"], PREFIX)
    staged = [r for r in recs if r["step"] == "stage"]
    assert [r["tensor"] for r in staged] == [t.name for t in lay.tensors]
    assert all(all(r["sha256"].values()) and "refused" not in r for r in staged)
    assert staged[0]["sha256"] == {"l0.wq": True, "l0.wk": True, "l0.wv": True} and staged[-1]["sha256"] == {"head": True}
    summary = next(r for r in recs if r["step"] == "summary")
    assert summary["ok"] and (summary["tensors"], summary["segments"]) == (9, 3)
    assert summary["bytes"] == sum(t.size for t in lay.tensors)
    assert summary["decode_step_s"] == pytest.approx(sum(min(r["decode_s"]) for r in staged), abs=1e-5)
    assert summary["head_s"] > 0 and summary["layer_s"]["min"] > 0
    gemms = [r for r in recs if r["step"] == "gemm"]
    assert [(r["matrix"], r["n"], r["k"]) for r in gemms] == [("qkv", 192, 64), ("o", 64, 64), ("up", 64, 64),
                                                              ("down", 64, 64)]
    assert all(r["twin"] == {"words": 16 * r["n"], "mismatches": 0, "first": []} and all(r["sha256"].values())
               for r in gemms)
    res = check.check_gemm(out)
    assert set(res) == {f"{m}-m16" for m in L.MATRICES} and all(v["mismatches"] == 0 for v in res.values())
    assert res["qkv-m16"]["samples"] == 3 * 4
    assert check.main(store + ["--out", str(out), "--sample", "3", "--workers", "2"]) == 0
    dec = json.loads((out / "check.json").read_text())["decode"]
    assert (dec["checked"], dec["mismatches"], dec["layers"]) == (len(check.sample_blocks(list(lay.tensors), 3, 12)), 0, 3)
    wo = next(r for r in manifest["tensors"] if r["name"] == "l1.wo")
    flipped = bytearray(stream)
    flipped[wo["offset"] + 5] ^= 1
    (tmp_path / "w2.bin").write_bytes(bytes(flipped))
    bad = [a.replace(str(tmp_path / "w.bin"), str(tmp_path / "w2.bin")) for a in store]
    assert check.main(bad + ["--out", str(tmp_path / "c2"), "--sample", "100", "--workers", "2"]) == 1
    dec = json.loads((tmp_path / "c2" / "check.json").read_text())["decode"]
    assert dec["checked"] == lay.tensors[-1].block_range.stop and dec["mismatches"] == 1
    assert dec["first"][0][0] == lay.tensor(1, "o").first_block

    def with_manifest(m: dict) -> list[str]:
        (tmp_path / "m2.json").write_text(json.dumps(m))
        return [a.replace(str(tmp_path / "m.json"), str(tmp_path / "m2.json")) for a in run]

    wrong = json.loads(json.dumps(manifest))
    next(r for r in wrong["tensors"] if r["name"] == "l1.wo")["sha256"] = "00" * 32
    assert full.main(with_manifest(wrong) + ["--out", str(tmp_path / "o2"), "--layers", "1-2"]) == 1
    recs = [json.loads(x) for x in (tmp_path / "o2" / "full.json").read_text().splitlines()]
    assert [r["tensor"] for r in recs if r["step"] == "stage"] == [t.name for t in lay.tensors if t.layer >= 1]
    assert [r["sha256"] for r in recs if r["step"] == "stage"][1] == {"l1.wo": False}
    assert not next(r for r in recs if r["step"] == "summary")["ok"]
    with pytest.raises(SystemExit, match="SHA-256"):
        full.main(with_manifest({"tensors": [{k: v for k, v in r.items() if k != "sha256"} for r in manifest["tensors"]]})
                  + ["--out", str(tmp_path / "o3")])
    other = json.loads(json.dumps(manifest))
    next(r for r in other["tensors"] if r["name"] == "head")["shape"] = [128, 64]
    with pytest.raises(SystemExit, match="lm_head"):
        full.main(with_manifest(other) + ["--out", str(tmp_path / "o3")])
    L.segment_path(root, 1, "c").write_bytes(bytes(bps * L.ROW))
    with pytest.raises(SystemExit, match="SHA-512"):
        full.main(run + ["--out", str(tmp_path / "o4")])
    L.segment_path(root, 1, "json").unlink()
    with pytest.raises(SystemExit, match="not done"):
        full.main(run + ["--out", str(tmp_path / "o4")])


def test_every_kept_path_has_a_reason_in_keep_json():
    """session.sh keeps each node path under keep.json's owner and expiry with that path's reason; a `keep PATH WHAT`
    whose WHAT has no reason would keep nothing."""
    keep = json.loads((HERE / "keep.json").read_text())
    whats = re.findall(r'^\s*keep "\$\w+" (\w+)$', (HERE / "session.sh").read_text(), re.M)
    assert sorted(whats) == sorted(keep["why"])
    assert keep["owner"].startswith("@") and keep["expires"]
    assert all("{layer}" in why for why in keep["why"].values())
