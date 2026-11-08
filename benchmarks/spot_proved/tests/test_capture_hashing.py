"""How `capture.py` hashes: the weights once per engine, with the same ``params.json`` and rank root byte for byte as hashing
them at every step; a whole cache by `chunks_sha512`, the device's held to the host's; and host copies held within
``--host-mib``."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

HERE = Path(__file__).resolve().parents[1]
RUNNER = SimpleNamespace(model_config=SimpleNamespace(hf_text_config=SimpleNamespace(num_hidden_layers=2, first_k_dense_replace=1)))


def _capture(tmp: Path, monkeypatch, name: str = "spot_capture_hashing", **cfg):
    spec = importlib.util.spec_from_file_location(name, HERE / "capture.py")
    capture = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, capture)
    spec.loader.exec_module(capture)
    S = capture._State({"capture": str(tmp), "steps": [0, 1, 2], "mla_tokens": 0, "open": {"dense": 1, "moe": 1},
                        "keep": "opened", "min_free_gib": 0, **cfg})
    monkeypatch.setattr(capture, "S", S)
    return capture, S


def _weights() -> dict:
    g = torch.Generator().manual_seed(7)
    return {"layers.0.w": torch.randn(64, 32, generator=g), "layers.1.w": torch.randn(48, generator=g).to(torch.bfloat16),
            "norm.w": torch.randn(16, generator=g)}


def _step(capture, S, step: int, weights: dict) -> tuple[bytes, dict]:
    """One recorded step reading every weight: its ``params.json`` bytes and ``meta.json``."""
    S.step, S.t0 = step, 0.0
    S.param_of = weights
    S.used_params = set(weights)
    S.events = [{"seq": 0, "layer": 0, "op": "x", "in": {"w": {"param": "layers.0.w"}}, "out": {"o": capture._ref(torch.arange(4))}}]
    capture._finish(RUNNER)
    d = S.root() / f"step{step}"
    return (d / "params.json").read_bytes(), json.loads((d / "meta.json").read_text())


def test_cached_weight_roots_equal_the_per_step_roots_byte_for_byte(tmp_path, monkeypatch) -> None:
    w = _weights()
    capture, S = _capture(tmp_path / "cached", monkeypatch)
    cached = [_step(capture, S, s, w) for s in range(3)]
    fresh_capture, F = _capture(tmp_path / "fresh", monkeypatch, name="spot_capture_hashing_fresh")
    F.digests = None
    fresh = [_step(fresh_capture, F, s, w) for s in range(3)]
    for (cp, cm), (fp, fm) in zip(cached, fresh):
        assert cp == fp and cm["params_sha512"] == fm["params_sha512"] and cm["rank_root"] == fm["rank_root"]
    want = {n: hashlib.sha512(t.contiguous().view(-1).view(torch.uint8).numpy().tobytes()).hexdigest() for n, t in w.items()}
    assert {n: r["sha512"] for n, r in json.loads(cached[0][0]).items()} == want
    nbytes = sum(t.numel() * t.element_size() for t in w.values())
    assert [m["bytes"].get("params_hashed", 0) for _, m in cached] == [nbytes, 0, 0]
    assert all(m["bytes"].get("params_hashed", 0) == nbytes for _, m in fresh)


def test_a_weight_changed_in_place_is_hashed_again(tmp_path, monkeypatch) -> None:
    w = _weights()
    capture, S = _capture(tmp_path, monkeypatch)
    before, _ = _step(capture, S, 0, w)
    w["norm.w"].add_(1)
    after, meta = _step(capture, S, 1, w)
    got = json.loads(after)["norm.w"]["sha512"]
    assert got != json.loads(before)["norm.w"]["sha512"]
    assert got == hashlib.sha512(w["norm.w"].numpy().tobytes()).hexdigest()
    assert meta["bytes"]["params_hashed"] == w["norm.w"].numel() * 4


def test_chunks_sha512_is_sha512_over_its_pieces(tmp_path, monkeypatch) -> None:
    capture, _ = _capture(tmp_path, monkeypatch)
    for n in (0, 1, 127, 128, 129, capture.CHUNK - 1, capture.CHUNK, 3 * capture.CHUNK + 113):
        raw = bytes(range(256)) * (n // 256) + bytes(range(n % 256))
        pieces = b"".join(hashlib.sha512(raw[i:i + capture.CHUNK]).digest() for i in range(0, n, capture.CHUNK))
        head = b"verity/spot/chunks-sha512\0" + capture.CHUNK.to_bytes(8, "big") + n.to_bytes(8, "big")
        assert capture.chunks_sha512(raw) == hashlib.sha512(head + pieces).hexdigest()


class _Device:
    """`CHUNKS_CU`'s ``chunks`` computed on the host, or wrong in its first piece when ``wrong``."""

    def __init__(self, wrong: bool = False) -> None:
        self.wrong = wrong

    def chunks(self, flat, chunk: int):
        raw = flat.numpy().tobytes()
        out = bytearray(b"".join(hashlib.sha512(raw[i:i + chunk]).digest() for i in range(0, len(raw), chunk)))
        if self.wrong:
            out[0] ^= 1
        return torch.frombuffer(out, dtype=torch.uint8)


def test_a_cache_is_named_on_the_device_and_the_first_are_held_to_the_host(tmp_path, monkeypatch) -> None:
    capture, S = _capture(tmp_path, monkeypatch, gpu_check=1)
    S.dev = _Device()
    cache = torch.arange(3 * capture.CHUNK // 4 + 5, dtype=torch.int32)
    refs = [capture._cache_ref(cache, {"shape": list(cache.shape)}) for _ in range(2)]
    capture._drain()
    want = capture.chunks_sha512(cache.numpy().tobytes())
    assert [r["chunks_sha512"] for r in refs] == [want, want] and all("sha512" not in r for r in refs)
    assert S.gpu_checked == 1 and S.n["hash_check"] == S.n["hash_device"] // 2 and not S.errors


def test_a_device_hash_that_disagrees_with_the_host_is_refused(tmp_path, monkeypatch) -> None:
    capture, S = _capture(tmp_path, monkeypatch)
    S.dev = _Device(wrong=True)
    cache = torch.arange(1000, dtype=torch.int32)
    ref = capture._cache_ref(cache, {})
    capture._drain()
    assert ref["chunks_sha512"] == capture.chunks_sha512(cache.numpy().tobytes())
    assert S.dev is False and len(S.errors) == 1 and "is not the host's" in S.errors[0]


def test_host_copies_stay_within_the_budget(tmp_path, monkeypatch) -> None:
    capture, S = _capture(tmp_path, monkeypatch, host_mib=1, hash_threads=4, keep="none")
    refs = [capture._ref(torch.full((1 << 17,), i, dtype=torch.int32)) for i in range(16)]
    capture._drain()
    assert S.hasher.peak <= 1 << 20 and S.hasher.held == 0
    assert len({r["sha512"] for r in refs}) == 16
