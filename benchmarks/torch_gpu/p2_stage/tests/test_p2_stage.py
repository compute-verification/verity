"""The staging decode's CPU parts: where a layer's matrices lie in W, the scatter of decoded rows into a staging buffer,
P2's reference decode of one block (pinned to the spec's deployment vector), and the GEMM's twin against
`ScaledMmS8Coordinate_v1{K, GEMM_SM90_S8}`."""
from __future__ import annotations

import json
import random
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
PREFIX = L.registered_prefix(bytes(range(24)))


def stream_manifest(width: int = 64, mlp: int = 256, layers: int = 2) -> dict:
    """A small weights-stream manifest with the frozen shape's tensor order (one KV head of `width // 4`)."""
    rows, at = [], 0
    kv = width // 4

    def add(name, shape, size):
        nonlocal at
        rows.append({"name": name, "offset": at, "nbytes": size, "shape": shape})
        at += size

    add("tok", [width, 256], 2 * width * 256)
    for i in range(layers):
        add(f"l{i}.attn_norm", [width], 2 * width)
        for t, n, k in (("wq", width, width), ("wk", kv, width), ("wv", kv, width), ("wo", width, width)):
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


def test_layout_of_both_manifests():
    doc = stream_manifest()
    lay = L.layout_of(doc, PREFIX)
    assert [p.name for p in lay.parts(1, "qkv")] == ["l1.wq", "l1.wk", "l1.wv"]
    assert [p.name for p in lay.parts(0, "down", scales=True)] == ["l0.wdown.scale"]
    assert lay.nbytes(0, "qkv") == 64 * 64 + 2 * 16 * 64
    assert lay.nbytes(0, "qkv", scales=True) == 4 * (64 + 32)
    span = lay.span(0)
    first, last = lay.tensor(0, "wq"), lay.tensor(0, "wdown.scale")
    assert span == range(first.offset // L.PAYLOAD, -(-(last.offset + last.size) // L.PAYLOAD))
    again = L.layout_of(lay.layout.manifest(), PREFIX)
    assert again.layout.tensors == lay.layout.tensors
    with pytest.raises(KeyError):
        lay.parts(0, "gate")


def test_layout_refuses_a_prefix_not_whole_blocks_and_another_point():
    doc = stream_manifest()
    with pytest.raises(ValueError):
        L.layout_of(doc, PREFIX[:-1])
    with pytest.raises(ValueError):
        L.StoreLayout(L.Layout.of(L.S.P2Params(16448, 21065, 1 << 10), [("a", 1)]), PREFIX)


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


def test_decode_block_is_the_spec_vector():
    prefix, j, c, m = deployment_vector()
    assert m[:L.LEAD] == bytes(L.LEAD)
    assert L.decode_block(prefix, j, c) == m[L.LEAD:]
    assert L.check_blocks(prefix, j, c, m[L.LEAD:]) == []


def test_decode_block_refuses_what_decode_segment_refuses():
    prefix, j, c, m = deployment_vector()
    with pytest.raises(ValueError, match="not below p"):
        L.decode_block(prefix, j, S.P_BYTES)
    with pytest.raises(ValueError, match="bytes, not"):
        L.decode_block(prefix, j, c[1:])
    bad = L.check_blocks(prefix, j + 1, c, m[L.LEAD:])
    assert len(bad) == 1 and bad[0][0] == j + 1
    flipped = bytes([c[0], c[1] ^ 1]) + c[2:]
    assert len(L.check_blocks(prefix, j, flipped, m[L.LEAD:])) == 1


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


class XorKernels:
    """A stand-in for `p2_v1.gpu.Kernels` on the CPU: block j's row XOR a keystream of (prefix, j), so the driver's
    plumbing (store, scatter, screen, staging, GEMM check) runs without the GPU library. Not P2."""

    def __init__(self, lib_dir, build_dir, params, prefix):
        self.prefix, self.device = bytes(prefix), torch.device("cpu")

    def _pad(self, first, n):
        import hashlib
        pads = b"".join(hashlib.shake_256(self.prefix + (first + i).to_bytes(8, "big")).digest(L.ROW) for i in range(n))
        return torch.frombuffer(bytearray(pads), dtype=torch.uint8).view(n, L.ROW)

    def decode(self, c, first=0):
        return c ^ self._pad(first, c.shape[0])

    encode = decode

    def raised(self):
        return 0


def test_layer_driver_stages_and_checks_on_a_stand_in_codec(tmp_path, monkeypatch):
    import check
    import layer

    monkeypatch.setattr(S.gpu, "Kernels", XorKernels)
    doc = stream_manifest()
    w = np.random.default_rng(5).integers(0, 256, doc["stream_bytes"] + 4 * L.PAYLOAD, dtype=np.uint8)
    for t in doc["tensors"]:
        if t["name"].endswith(".scale"):
            n = t["nbytes"] // 4
            w[t["offset"]:t["offset"] + t["nbytes"]] = ((np.arange(n, dtype=np.float32) + 1) * 2.0 ** -14).view(np.uint8)
    (tmp_path / "m.json").write_text(json.dumps(doc))
    (tmp_path / "w.bin").write_bytes(w.tobytes())
    out = tmp_path / "out"
    base = ["--manifest", str(tmp_path / "m.json"), "--w", str(tmp_path / "w.bin"), "--store", str(tmp_path / "store"),
            "--lib", "unused", "--salt", "00" * 24, "--out", str(out), "--layer", "1", "--device", "cpu",
            "--m", "32", "--reps", "1", "--samples", "4", "--encode-chunk", "3"]
    lay = L.layout_of(doc, L.registered_prefix(bytes(24)))
    qkv = layer.matrix_blocks(lay, 1, "qkv")
    first = lay.span(1).start
    assert layer.main(base + ["--encode-blocks", str(qkv.stop - first)]) == 0
    recs = [json.loads(x) for x in (out / "layer.json").read_text().splitlines()]
    assert [r["ready"] for r in recs if "ready" in r] == [["qkv"]]
    assert layer.main(base) == 0
    recs = [json.loads(x) for x in (out / "layer.json").read_text().splitlines()]
    staged = [r for r in recs if r["step"] == "stage" and r.get("matrix") in L.MATRICES]
    assert [r["matrix"] for r in staged] == ["qkv", "qkv", "o", "up", "down"]
    assert all(r["weights"]["mismatched_bytes"] == 0 and r["scales"]["mismatched_bytes"] == 0 for r in staged)
    gemms = [r for r in recs if r["step"] == "gemm"]
    assert len(gemms) == 5 and all(r["twin"]["mismatches"] == 0 for r in gemms)
    assert [r["k"] for r in gemms[-4:]] == [64, 64, 64, 256] and [r["n"] for r in gemms[-4:]] == [96, 64, 256, 64]
    ctl = [r for r in recs if r["step"] == "controls"][-1]["refused"]
    assert ctl["other_keys"] and ctl["row_at_p"] and ctl["honest"] is None
    assert any(r["step"] == "summary" for r in recs)
    res = check.check_gemm(out)
    assert set(res) == set(L.MATRICES) and all(v["mismatches"] == 0 and v["samples"] == 4 for v in res.values())
