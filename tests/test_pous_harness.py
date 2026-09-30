"""The POUS-only harness (`benchmarks/pous/band_gpu/harness.py`, `bench.py --impl band-gpu`) on CPU: the band's reference codec
stands in for the GPU twin on CPU tensors, a Python thread for the native responder (its wire format), and the verifier is the
harness's own `live.py` process.  The harness's timer and the verifier's clock (`live.py --ticks`) are injected, so no outcome
here depends on a clock or on the machine's load: what is checked is that each step runs and each gate decides, not the
numbers."""
from __future__ import annotations

import hashlib
import importlib.util
import socket
import struct
import sys
import threading
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("verity_pous")

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks" / "pous"
BAND = "band-chain/d12/v1"
WS = {"id": "toy/linears", "bytes": 2560, "tensors": [["q", 16, 32], ["k", 8, 32], ["down", 32, 16]]}


def _bench():
    spec = importlib.util.spec_from_file_location("pous_bench", BENCH / "bench.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _harness():
    sys.path.insert(0, str(BENCH))
    from band_gpu import harness
    return harness


def _small():
    from verity_pous import band
    return band.BandParams(16, 64 * 8, 12)          # 64-byte blocks, 1 KiB segments


class PyResponder:
    """The native responder's wire format over a CPU tensor's blocks, read at answer time."""

    def __init__(self, c) -> None:
        self.blocks = c.reshape(-1, c.shape[-1])
        self.sock = socket.create_server(("127.0.0.1", 0))
        self.counts = {"answered": 0, "pings": 0}
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def port(self) -> int:
        return self.sock.getsockname()[1]

    def answered(self) -> int:
        return self.counts["answered"]

    def pings(self) -> int:
        return self.counts["pings"]

    def _serve(self) -> None:
        try:
            conn, _ = self.sock.accept()
        except OSError:
            return
        with conn:
            while True:
                req = b""
                while len(req) < 16:
                    chunk = conn.recv(16 - len(req))
                    if not chunk:
                        return
                    req += chunk
                op, arg = struct.unpack("<QQ", req)
                if op == 3:
                    return
                body = self.blocks[arg].numpy().tobytes() if op == 1 else bytes(self.blocks.shape[1])
                self.counts["answered" if op == 1 else "pings"] += 1
                conn.sendall(struct.pack("<QQQ", 0, 1000, 2000) + body)

    def stop(self) -> None:
        self.sock.shutdown(socket.SHUT_RDWR)
        self.thread.join()
        self.sock.close()


def _backend(params, flip_last: bool = False):
    """The reference codec as the twin: tags carry each row's segment id and the salt."""
    from verity_pous import schemes
    from verity_pous.labels import salt_value, tag

    harness = _harness()
    sch = schemes.get(BAND, params=params.with_segments(64))

    def tags(salt, segs, device):
        return torch.tensor([[s, *salt] for s in segs], dtype=torch.int64)

    def rows(fn, x, t):
        out = [fn(bytes(int(v) for v in t[g, 1:]), int(t[g, 0]), x[g]) for g in range(x.shape[0])]
        return torch.stack(out)

    def encode(w, t):
        c = rows(lambda salt, s, seg: torch.frombuffer(bytearray(b"".join(sch.encode_segment(salt, s, seg.numpy().tobytes()))),
                                                       dtype=torch.uint8).view(seg.shape).clone(), w, t)
        if flip_last and w.shape[0] > 1:
            c[-1, 3, 0] ^= 1
        return c

    def decode(c, t):
        return rows(lambda salt, s, seg: torch.frombuffer(bytearray(sch.decode_segment(salt, s, [bytes(r) for r in seg.numpy()])),
                                                          dtype=torch.uint8).view(seg.shape).clone(), c, t)

    def timer(fn):
        fn()
        return 1e-3

    return harness.Backend("cpu", encode, decode, tags, lambda salt, n: [tag(salt_value(salt), s).to_bytes(32, "little") for s in range(n)],
                           lambda c, cpu: PyResponder(c), timer)


def _vectors(params):
    """The band's pinned whole-segment vector, at the small geometry, from the reference."""
    from verity_pous import schemes

    harness = _harness()
    b = params.blocks_per_segment * params.block_bytes
    w, salt = harness.vector_bytes("band-segment-w", b), harness.vector_bytes("salt", 24)
    blocks = schemes.get(BAND, params=params.with_segments(1)).encode_segment(salt, 0, w)
    return {"band/64kb/segment": hashlib.sha256(b"".join(blocks)).hexdigest()}


def _args(tmp_path, **kw):
    base = {"scheme": BAND, "max_segments": 0, "kernels": "", "build_dir": str(tmp_path / "build"), "reference_segments": 2,
            "workers": 2, "tokens": "1,8", "reps": 1, "load_tokens": 0, "audits": 2, "pings": 20, "allowance": "max", "k": 0, "deadline_ns": 0, "ticks": True,
            "verifier_cpu": None, "responder_cpu": None, "step_reps": 3, "nvme_dir": "", "nvme_samples": 8, "impl_commit": None}
    return types.SimpleNamespace(**{**base, **kw})


def test_the_harness_runs_every_step_and_its_gates_decide(tmp_path):
    params = _small()
    nvme = tmp_path / "nvme"
    nvme.mkdir()
    body = _bench().BandGpu().run(_args(tmp_path, nvme_dir=str(nvme)), WS, tmp_path, backend=_backend(params), base=params,
                                  vectors=_vectors(params))
    gates = body["gates"]
    assert body["segments"] == 3 and body["blocks"] == 48 and body["weight_bytes"] == 2560
    for g in ("vector", "reference_encode", "round_trip", "timed_audit", "negative_control", "segment_tags"):
        assert gates[g]["passed"], (g, gates[g])
    assert gates["reference_encode"]["segments"] == [0, 2]
    assert not gates["certificate"]["passed"] and "no certificate" in gates["certificate"]["reason"]   # the small point has none
    m = body["metrics"]
    assert m["loop"]["tokens"] == [1, 8] and m["loop"]["tensors"] == 3 and set(m["loop"]["overhead"]) == {1, 8}
    honest, negative = m["audit"], m["negative_audit"]
    from verity_pous import audit

    assert honest["audits_accepted"] == 2 and honest["k"] == audit.CHALLENGES and honest["responder"]["answered"] == 2 * audit.CHALLENGES
    assert set(honest["margin_s"]) >= {"min", "p50", "max"} and honest["limit_ns"] > honest["deadline_ns"]
    assert negative["audits_accepted"] == 0 and negative["wrong"] > 0
    assert m["step_floor"] is None and m["nvme"]["block_read_s"]["n"] == 8 and not list(nvme.iterdir())
    names = {r["name"] for r in body["measurements"]}
    assert {"encode_GBps", "decode_overhead_T1", "decode_overhead_T8", "audit_margin_min_s", "nvme_block_read_p99.9_s"} <= names
    assert body["profile"] is not None and not body["profile"]["certified"]


def test_a_wrong_encoding_fails_the_reference_gate_though_its_own_leaves_audit(tmp_path):
    params = _small()
    body = _bench().BandGpu().run(_args(tmp_path), WS, tmp_path, backend=_backend(params, flip_last=True), base=params,
                                  vectors=_vectors(params))
    assert body["gates"]["vector"]["passed"] and body["gates"]["timed_audit"]["passed"]
    assert not body["gates"]["reference_encode"]["passed"] and body["gates"]["reference_encode"]["equal"] == [True, False]
    assert not body["gates"]["round_trip"]["passed"]


def test_the_step_floor_is_the_fastest_kernel_and_a_call_under_the_certificate_is_flagged():
    harness = _harness()
    times = {0: [7e-4, 6e-4]}

    def step(settings, reps):
        return [4e-4] if settings.get("pair") else times[0]

    fake = harness.Backend("cpu", None, None, None, None, None, None, step)
    got = harness.step_floor(fake, 2, 483_000)
    assert got["fastest"] == "pair" and got["min_s"] == 4e-4 and got["below_certificate"]
    assert set(got["variants"]) == set(harness.STEP_VARIANTS)
    assert not harness.step_floor(fake, 2, 300_000)["below_certificate"]


def test_the_kernel_settings_and_the_loop_layout():
    harness = _harness()
    before = harness.set_kernels({})
    try:
        now = harness.set_kernels(harness.FROZEN_KERNELS)
        assert now["thread_rows"] == 500 and now["whole_chain"] is True
        with pytest.raises(ValueError, match="unknown kernel settings"):
            harness.set_kernels({"turbo": 1})
    finally:
        harness.set_kernels(before)
    pl = harness.placements(WS["tensors"], 2000)
    assert [(name, off, size) for name, _, _, off, size in pl] == [("q", 0, 1024), ("k", 1024, 512)]   # down ends past 2000
