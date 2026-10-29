"""The POUS-only harness (`benchmarks/pous/band_gpu/harness.py`, `bench.py --impl band-gpu`) on CPU: the band's reference codec
stands in for the GPU twin on CPU tensors, a Python thread for the native responder (its wire format), and the memory
challenger is the harness's own `memory_challenger.py` process.  The harness's timer is injected and the memory
challenger runs on its logical clock (`--clock logical`: each answer's time is the responder's reported one), so no
outcome here depends on a clock or on the machine's load: what is checked is that each step runs and each gate and
control decides, not the numbers."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import socket
import struct
import subprocess
import sys
import threading
import types
import weakref
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("verity.core.protocols.pous")

ROOT = Path(__file__).resolve().parents[3]
BENCH = ROOT / "benchmarks" / "pous"
BAND = "band-chain/d12/v1"
WS = {"id": "toy/linears", "bytes": 2560, "tensors": [["q", 16, 32], ["k", 8, 32], ["down", 32, 16]]}
SEED = "5eed" * 8
"""`memory_challenger.py --seed`: the audits' challenges, and so the controls' verdicts, are fixed."""


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
    from verity.core.protocols.pous import band
    return band.BandParams(16, 64 * 8, 12)          # 64-byte blocks, 1 KiB segments


class PyResponder:
    """The native responder's wire format over a CPU tensor's labels, read at answer time."""

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
    from verity.core.protocols.pous import schemes
    from verity.core.protocols.pous.labels import salt_value, tag

    harness = _harness()
    from band_gpu.memory_challenger import scheme_of

    at = lambda d: schemes.Band(type(params)(params.blocks_per_segment, params.m, d, 64))  # noqa: E731
    sch = scheme_of(at(params.d).name, at(params.d).params)

    def tags(salt, segs, device):
        return torch.tensor([[s, *salt] for s in segs], dtype=torch.int64)

    def rows(fn, x, t):
        out = [fn(bytes(int(v) for v in t[g, 1:]), int(t[g, 0]), x[g]) for g in range(x.shape[0])]
        return torch.stack(out)

    def encoder(sch_):
        return lambda w, t: rows(lambda salt, s, seg: torch.frombuffer(bytearray(b"".join(sch_.encode_segment(
            salt, s, seg.numpy().tobytes()))), dtype=torch.uint8).view(seg.shape).clone(), w, t)

    def encode(w, t):
        c = encoder(sch)(w, t)
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
                           lambda c, cpu: PyResponder(c), timer, encode_at=lambda d: encoder(at(d)))


def _vectors(params):
    """The band's pinned whole-segment vector, at the small geometry, from the reference."""
    from verity.core.protocols.pous import schemes

    harness = _harness()
    b = params.blocks_per_segment * params.block_bytes
    w, salt = harness.vector_bytes("band-segment-w", b), harness.vector_bytes("salt", 24)
    blocks = schemes.get(BAND, params=params.with_segments(1)).encode_segment(salt, 0, w)
    return {"band/64kb/segment": hashlib.sha256(b"".join(blocks)).hexdigest()}


def _args(tmp_path, **kw):
    base = {"scheme": BAND, "max_segments": 0, "kernels": "", "build_dir": str(tmp_path / "build"), "reference_segments": 2,
            "workers": 2, "tokens": "1,8", "reps": 1, "load_tokens": 0, "audits": 2, "pings": 20, "allowance": "max", "k": 0, "deadline_ns": 0, "clock": "logical",
            "verifier_cpu": None, "responder_cpu": None, "step_reps": 3, "nvme_dir": "", "nvme_samples": 8, "impl_commit": None,
            "seed": SEED}
    return types.SimpleNamespace(**{**base, **kw})


def test_the_harness_runs_every_step_and_its_gates_decide(tmp_path):
    params = _small()
    nvme = tmp_path / "nvme"
    nvme.mkdir()
    body = _bench().BandGpu().run(_args(tmp_path, nvme_dir=str(nvme)), WS, tmp_path, backend=_backend(params), base=params,
                                  vectors=_vectors(params))
    gates = body["gates"]
    assert body["segments"] == 3 and body["blocks"] == 48 and body["weight_bytes"] == 2560
    for g in ("vector", "reference_encode", "round_trip", "decode_after_timing", "timed_audit", "value_control", "late_control",
              "recompute_control", "segment_tags"):
        assert gates[g]["passed"], (g, gates[g])
    assert set(body["implementation"]["decode_paths"]) == {"gates", "loop", "audits"}   # the CPU stand-in runs no dense decode
    assert body["pous_arm_graphed"] is False and body["metrics"]["loop"]["decode_memory"] is None
    assert gates["reference_encode"]["segments"] == [0, 2]
    assert not gates["certificate"]["passed"] and "no certificate" in gates["certificate"]["reason"]   # the small point has none
    m = body["metrics"]
    assert m["loop"]["tokens"] == [1, 8] and m["loop"]["tensors"] == 3 and set(m["loop"]["overhead"]) == {1, 8}
    honest, value, late, rec = m["audit"], m["value_control"], m["late_control"], m["recompute_control"]
    from verity.core.protocols.pous import audit

    assert honest["audits_accepted"] == 2 and honest["k"] == audit.CHALLENGES and honest["responder"]["answered"] == 2 * audit.CHALLENGES
    assert set(honest["margin_s"]) >= {"min", "p50", "max"} and honest["limit_ns"] > honest["deadline_ns"]
    limit = m["calibration"]["limit_ns"]
    assert honest["limit_fixed_before_load"] and honest["limit_ns"] == limit == m["deadline"]["limit_ns"]      # one limit for all
    assert m["deadline"]["allowance_kind"] == "loopback" and late["limit_ns"] == rec["limit_ns"] == limit
    assert len(honest["records"]) == 2 and all(len(r["source"]) == 64 and len(r["answers"]) == audit.CHALLENGES
                                               for r in honest["records"])                                    # re-derivable
    assert value["audits_accepted"] == 0 and value["wrong"] > 0                                               # rejects on value
    assert late["audits_accepted"] == 0 and late["late"] > 0 and late["wrong"] == 0                           # rejects on time
    assert rec["wrong"] == 0 and rec["late"] > 0 and rec["dropped_blocks"] == 3                               # recomputed right, late
    assert rec["margin_s"]["min"] < 0 and "cliff" in rec["pattern"]
    assert rec["recomputed_answers"] > 0 and rec["device_margin_s"]["max"] == (limit - 1_000_000) / 1e9     # the timer's 1 ms
    assert rec["recompute_margin_kind"].startswith("upper bound on attacker latency")
    assert gates["recompute_control"]["recomputed_answers"] == rec["recomputed_answers"]
    assert m["loop"]["plain_graphed"][1] == {"graphed": False, "reason": "no CUDA graph backend (CPU stand-in)"}
    assert all(len(r) == 1 for r in m["loop"]["plain_reps_s"].values())                                      # every rep kept
    assert m["step_floor"] is None and m["nvme"]["block_read_s"]["n"] == 8 and not list(nvme.iterdir())
    names = {r["name"] for r in body["measurements"]}
    assert {"encode_GBps", "decode_overhead_T1", "decode_overhead_T8", "audit_margin_min_s", "nvme_block_read_p99.9_s"} <= names
    assert body["profile"] is not None and not body["profile"]["certified"]


def test_a_wrong_encoding_stops_the_run_before_anything_is_timed(tmp_path):
    params = _small()
    body = _bench().BandGpu().run(_args(tmp_path), WS, tmp_path, backend=_backend(params, flip_last=True), base=params,
                                  vectors=_vectors(params))
    assert body["gates"]["vector"]["passed"]
    assert not body["gates"]["reference_encode"]["passed"] and body["gates"]["reference_encode"]["equal"] == [True, False]
    assert not body["gates"]["round_trip"]["passed"]
    assert body["metrics"] is None and body["measurements"] == [] and "nothing was timed" in body["stopped"]   # a failed gate stops
    assert "timed_audit" not in body["gates"] and body["profile"] is None


def test_a_decode_wrong_only_after_the_round_trip_publishes_nothing(tmp_path):
    params = _small()
    backend = _backend(params)
    right, seen = backend.decode, []

    def decode(c, t):          # right for the round-trip gate, then wrong, as a race under timing would be
        seen.append(right(c, t) if not seen else seen[0])
        out = seen[-1].clone()
        if len(seen) > 1:
            out.view(-1)[0] ^= 1
        return out

    backend.decode = decode
    body = _bench().BandGpu().run(_args(tmp_path, audits=0), WS, tmp_path, backend=backend, base=params, vectors=_vectors(params))
    assert body["gates"]["round_trip"]["passed"] and not body["gates"]["decode_after_timing"]["passed"]
    assert body["metrics"] is None and body["measurements"] == [] and "nothing is published" in body["stopped"]


def test_the_lane_pair_decode_runs_from_two_segments_and_every_decode_records_its_kernel(monkeypatch):
    harness = _harness()
    from band_gpu import dense

    stub = lambda c, seg, blk, *rest: torch.zeros(len(blk), c.shape[1], dtype=torch.uint8)  # noqa: E731
    monkeypatch.setattr(dense, "_decode_whole", stub)
    monkeypatch.setattr(dense, "_decode_chunked", stub)
    width, bps = 65_536, 512
    c, tags, blocks = torch.zeros(2 * bps, width, dtype=torch.uint8), torch.zeros(2, 32, dtype=torch.uint8), torch.arange(2 * bps)
    before = harness.set_kernels({})
    try:
        harness.take_decode_paths()
        for settings in (harness.KERNELS, {"chain_pairs": False}):
            harness.set_kernels(settings)
            dense.decode_blocks(c, tags, blocks // bps, blocks % bps, bps, 12)
        assert harness.take_decode_paths() == [{"kernel": "band_chain_pairs", "rows": 1024, "calls": 1},
                                               {"kernel": "feistel_chunked per step", "rows": 1024, "calls": 1}]
    finally:
        harness.set_kernels(before)


def test_gates_only_runs_the_kernel_gates_and_times_nothing(tmp_path):
    params = _small()
    body = _bench().BandGpu().run(_args(tmp_path, gates_only=True), WS, tmp_path, backend=_backend(params), base=params,
                                  vectors=_vectors(params))
    assert set(body["gates"]) == {"vector", "reference_encode", "round_trip"} and all(g["passed"] for g in body["gates"].values())
    assert body["metrics"] is None and body["measurements"] == [] and "--gates-only" in body["stopped"]


def test_band_d_builds_the_band_at_that_in_degree_with_its_certificate_cliff_there():
    """`--band-d 6`: the band at in-degree 6, unregistered (`band-chain/d6/v1` is the codec's name, not a registered byte
    format), whose segments round-trip and whose certificate at the frozen point is
    `Pous.SecurityProofs.BandMultiMeetsFamily` with the cliff at 6; the memory challenger process rebuilds it from its manifest;
    d = 4 has no certificate and is refused."""
    from verity.core.protocols.pous import band, schemes

    harness = _harness()
    from band_gpu.memory_challenger import params_of, scheme_of

    small = harness.with_in_degree(_small(), 6)
    sch = scheme_of("band-chain/d6/v1", small)
    assert sch.name == "band-chain/d6/v1" and sch.params.d == 6 and sch.name not in schemes.REGISTRY
    w = hashlib.shake_256(b"verity-pous/test/band-d").digest(small.segment_bytes)
    blocks = sch.encode_segment(harness.SALT, 0, w)
    assert sch.decode_segment(harness.SALT, 0, blocks) == w
    assert blocks != scheme_of(BAND, _small()).encode_segment(harness.SALT, 0, w)            # another d, another encoding
    point = harness.with_in_degree(band.SECURE_FIRST, 6)
    cert = scheme_of("band-chain/d6/v1", point).certificate
    assert (cert.theorem, cert.cliff, cert.k, cert.rounds, cert.call_ns) == (
        "Pous.SecurityProofs.BandMultiMeetsFamily", 6, 111, 5, 241_500)
    assert scheme_of("band-chain/d6/v1", point.with_segments(448)).certificate.segments == 448
    gate, _ = _bench().certificate_gate(scheme_of("band-chain/d6/v1", point), 111, 700_000)
    assert gate["passed"] and gate["effective_rounds"] == 2
    assert params_of("band-chain/d6/v1", point.manifest()) == point
    for d in (4, 1):
        with pytest.raises(ValueError, match=f"d = {d} is refused"):
            harness.with_in_degree(band.SECURE_FIRST, d)


def test_the_harness_runs_the_band_at_d6_and_records_d(tmp_path):
    harness = _harness()
    params = _small()
    body = _bench().BandGpu().run(_args(tmp_path, band_d=6), WS, tmp_path, backend=_backend(harness.with_in_degree(params, 6)),
                                  base=params, vectors=_vectors(params))
    gates, m = body["gates"], body["metrics"]
    assert body["scheme"] == "band-chain/d6/v1" and body["scheme_params"]["d"] == 6 and body["profile"]["params"]["d"] == 6
    for g in ("vector", "reference_encode", "round_trip", "timed_audit", "value_control", "late_control", "recompute_control"):
        assert gates[g]["passed"], (g, gates[g])
    assert gates["vector"]["d"] == 12                                       # the pinned vector, at the registered d
    assert m["recompute_control"]["dropped_blocks"] == 3 * 2 and "every 7-th" in m["recompute_control"]["pattern"]


def test_bench_refuses_a_band_d_without_a_certificate_and_on_another_impl(tmp_path):
    bench = _bench()
    common = ["--weight-set", "qwen2.5-0.5b/linears+head", "--hardware", "cpu", "--out", str(tmp_path)]
    with pytest.raises(SystemExit, match="d = 4 is refused"):
        bench.main(["--impl", "band-gpu", "--band-d", "4", *common])
    with pytest.raises(SystemExit, match="band-gpu's, not reference's"):
        bench.main(["--impl", "reference", "--band-d", "6", *common])


def test_audits_0_runs_the_band_untimed_with_no_limit_audit_or_control_and_says_so(tmp_path, monkeypatch):
    """`--audits 0` (`AUDITS=0`, an untimed placement): the gates, encode, the loop and the step floor run, but no limit,
    audit or control does; the result says `timed: false` and carries no audit, deadline or latency number to cite."""
    import dataclasses

    params = _small()
    harness = _harness()
    monkeypatch.setattr(harness, "audit_round", lambda *a, **k: pytest.fail("an untimed run started an audit"))
    monkeypatch.setattr(harness, "calibrate_limit", lambda *a, **k: pytest.fail("an untimed run calibrated a limit"))
    backend = dataclasses.replace(_backend(params), step=lambda settings, reps: [7e-4] * reps)
    body = _bench().BandGpu().run(_args(tmp_path, audits=0), WS, tmp_path, backend=backend, base=params, vectors=_vectors(params))
    assert body["timed"] is False and body["audits"] == "skipped (untimed placement)" and body["profile"] is None
    gates = body["gates"]
    assert set(gates) == {"vector", "reference_encode", "round_trip", "decode_after_timing", "segment_tags"}
    assert all(g["passed"] for g in gates.values())
    m = body["metrics"]
    assert m["timed"] is False and m["step_floor"]["min_s"] == 7e-4 and set(m["loop"]["overhead"]) == {1, 8}
    assert not {"audit", "deadline", "calibration", "value_control", "late_control", "recompute_control"} & set(m)
    names = {r["name"] for r in body["measurements"]}
    assert {"encode_GBps", "decode_overhead_T1", "decode_overhead_T8", "step_floor_min_s"} <= names
    assert not [name for name in names if name.startswith(("audit", "recompute", "certificate"))]
    assert not (tmp_path / "leaves.bin").exists()


def test_bench_labels_an_untimed_result_and_refuses_audits_0_where_it_would_pass_vacuously(tmp_path, monkeypatch):
    bench = _bench()
    common = ["--weight-set", "qwen2.5-0.5b/linears+head", "--hardware", "cpu", "--out", str(tmp_path)]
    with pytest.raises(SystemExit, match="only band-gpu and the P2 arms run untimed"):
        bench.main(["--impl", "reference", "--audits", "0", *common])

    class Untimed:
        name = "band-gpu"

        def run(self, a, ws, out):
            assert a.audits == 0
            return {"timed": False, "audits": "skipped (untimed placement)", "gates": {"round_trip": {"passed": True}},
                    "metrics": {"ewb": 2e9, "decode_pass_s": 1e-3}, "profile": None,
                    "measurements": [{"name": "encode_GBps", "value": 3.0, "unit": "GB/s"}]}

    monkeypatch.setitem(bench.IMPLS, "band-gpu", Untimed)
    monkeypatch.setattr(bench, "conformance_gate", lambda name: {"passed": True})
    monkeypatch.setenv("RESEARCH_RUN_DIR", str(tmp_path))
    assert bench.main(["--impl", "band-gpu", "--audits", "0", *common]) == 0
    doc = json.loads((tmp_path / "bench-result.json").read_text())
    assert doc["timed"] is False and doc["audits"] == "skipped (untimed placement)"
    fingerprint = json.loads((tmp_path / "result.json").read_text())["workload_fingerprint"]
    assert fingerprint["timed"] is False and fingerprint["audits"] == "skipped (untimed placement)"


def _challenge(conn, i: int, width: int) -> tuple[int, int, bytes]:
    conn.sendall(struct.pack("<QQ", 1, i))
    want, got = 24 + width, b""
    while len(got) < want:
        chunk = conn.recv(want - len(got))
        assert chunk, "the responder closed the connection"
        got += chunk
    _, device_ns, server_ns = struct.unpack_from("<QQQ", got)
    return device_ns, server_ns, got[24:]


def test_the_control_responder_holds_only_the_late_controls_answers():
    """On the wall clock (`real_delay`) the responder holds an answer only when the late control asks it to: a recomputed
    answer has already spent its device time and goes out at once, never waited again (the red team's B2).  The wait is
    injected and recorded, so nothing here reads a clock."""
    harness = _harness()
    width = 8
    c_host = bytes(range(20 * width))
    answer = harness.control_answers(c_host, width, late_every=4, late_ns=999_000, dropped={3},
                                     recompute=lambda i: (b"R" * width, 777_000))
    waits: list[int] = []
    r = harness.ControlResponder(width, answer, real_delay=True, wait=waits.append)
    conn = socket.create_connection(("127.0.0.1", r.port()))
    try:
        got = [_challenge(conn, i, width) for i in (3, 1, 2, 5, 3, 6, 7, 8)]
    finally:
        conn.sendall(struct.pack("<QQ", 3, 0))
        conn.close()
        r.stop()
    assert waits == [999_000, 999_000]                                                 # the 4th and 8th answers alone
    assert got[0] == got[4] == (777_000, 777_000, b"R" * width)                         # recomputed: reported, not held
    assert got[3] == (0, 999_000, c_host[5 * width:6 * width]) and got[7][1] == 999_000
    assert got[1] == (0, 1_000, c_host[width:2 * width])


def test_the_memory_challenger_takes_a_seed_on_the_logical_clock_alone(tmp_path):
    """`--seed` fixes the CPU tests' challenge sources, since a control's gate is a statistical test that real draws fail now
    and then; a timed audit on the wall clock refuses it before reading anything, so its challenges stay `os.urandom`'s."""
    _harness()
    from band_gpu import memory_challenger as mc

    files = ["--ready", str(tmp_path / "ready.json"), "--leaves", str(tmp_path / "leaves.bin"), "--out", str(tmp_path / "out.json")]
    with pytest.raises(SystemExit) as refused:
        mc.main([*files, "--seed", SEED])
    assert refused.value.code == 2 and not (tmp_path / "out.json").exists()
    seed = bytes.fromhex(SEED)
    with pytest.raises(ValueError, match="logical clock alone"):
        mc.audit("127.0.0.1", 0, None, None, audits=1, pings=1, seed=seed)
    assert len({mc.seeded_source(seed, j) for j in range(4)} | {mc.seeded_source(b"other", 0)}) == 5


def test_the_band_recompute_xors_w_in_inside_its_timed_device_call():
    """The band's recompute adversary decodes the dropped label's key and XORs `W`'s label in within the one timed call, so
    its reported device time covers the whole answer and nothing is computed on the host after it."""
    harness = _harness()
    width, bps = 4, 8
    c = torch.arange(2 * bps * width, dtype=torch.uint8).view(2, bps, width)
    w_host = bytes(range(100, 100 + 2 * bps * width))
    state = {"timing": False, "calls": [], "ops": []}

    class Traced(torch.Tensor):                                            # each op on the key, and whether it ran timed
        @classmethod
        def __torch_function__(cls, func, types, args=(), kwargs=None):
            state["ops"].append((getattr(func, "__name__", str(func)), state["timing"]))
            return super().__torch_function__(func, types, args, kwargs or {})

    key = torch.tensor([1, 2, 3, 4], dtype=torch.uint8)

    def recompute(store, tags, i):
        assert int(store.view(-1, width)[i].sum()) == 0 and int(store.view(-1, width)[i - 1].sum()) > 0   # only it is dropped

        def call():
            state["calls"].append(state["timing"])
            return key.clone().as_subclass(Traced)
        return call

    def timer(fn):
        state["timing"] = True
        fn()
        state["timing"] = False
        return 2.5e-6

    rc = harness._recomputer(harness.Backend("cpu", None, None, None, None, None, timer, recompute=recompute), c, None, [5, 13],
                             w_host, bps, width)
    assert rc(13) == (bytes(k ^ w for k, w in zip([1, 2, 3, 4], w_host[13 * width:14 * width])), 2_500)
    assert state["calls"] == [True]
    xors = [timed for name, timed in state["ops"] if "xor" in name]
    assert xors and all(xors), state["ops"]                               # the XOR with W's label is a device op, timed
    untimed = harness._recomputer(harness.Backend("cpu", None, None, None, None, None, lambda fn: 0.0, recompute=recompute), c,
                                  None, [5, 13], w_host, bps, width)
    with pytest.raises(IndexError):                                       # a timer that skips the call leaves no answer
        untimed(5)


def test_the_recompute_margin_is_the_limit_minus_each_recomputed_answers_device_time():
    harness = _harness()
    doc = {"records": [{"answers": [{"index": 1, "device_ns": 0}, {"index": 10, "device_ns": 400_000}]},
                       {"answers": [{"index": 20, "device_ns": 700_000}]}]}
    got = harness.recompute_margins(doc, {10, 20}, 600_000)
    assert got["recomputed_answers"] == 2 and got["recompute_device_s"]["min"] == 4e-4
    assert got["device_margin_s"]["min"] == -1e-4 and got["device_margin_s"]["max"] == 2e-4    # the fastest beats the limit


def test_the_native_build_is_keyed_by_the_nvcc_release_that_compiles_it(tmp_path):
    sys.path.insert(0, str(BENCH))
    from band_gpu import native

    nvcc = tmp_path / "bin" / "nvcc"
    nvcc.parent.mkdir()
    nvcc.write_text("#!/bin/sh\necho 'nvcc: NVIDIA (R) Cuda compiler driver'\necho 'Cuda compilation tools, release 12.8, V12.8.93'\n")
    nvcc.chmod(0o755)
    assert native.nvcc_release(str(tmp_path)) == "12.8" and native.nvcc_release(str(tmp_path / "none")) == "none"
    at = lambda release: native.source_digest("2.7.1+cu128", "12.8", release, "12.0")  # noqa: E731
    assert at("12.8") == at("12.8") != at("13.0")                       # another toolkit's .so is never adopted


def test_every_checkout_of_the_same_native_source_compiles_one_copy_left_untouched(tmp_path, monkeypatch):
    sys.path.insert(0, str(BENCH))
    from band_gpu import native

    copy = native._stable_copy(str(tmp_path))
    assert open(copy, "rb").read() == open(native.SOURCE, "rb").read()
    os.utime(copy, ns=(1, 1))
    assert native._stable_copy(str(tmp_path)) == copy and os.stat(copy).st_mtime_ns == 1   # ninja sees nothing to rebuild
    other = tmp_path / "checkout" / "pous_native.cu"
    other.parent.mkdir()
    other.write_bytes(b"// another source\n")
    monkeypatch.setattr(native, "SOURCE", str(other))
    assert native._stable_copy(str(tmp_path)) == copy and open(copy, "rb").read() == b"// another source\n"


def test_the_recompute_gate_fails_a_run_whose_challenges_missed_every_dropped_block():
    harness = _harness()
    assert harness.recompute_gate({"wrong": 0, "answers": 20, "recomputed_answers": 3})["passed"]
    assert not harness.recompute_gate({"wrong": 0, "answers": 20, "recomputed_answers": 0})["passed"]
    assert not harness.recompute_gate({"wrong": 0, "answers": 20})["passed"]
    assert not harness.recompute_gate({"wrong": 1, "answers": 20, "recomputed_answers": 3})["passed"]


def test_the_step_floor_is_the_fastest_kernel_and_a_call_under_the_certificate_is_flagged():
    harness = _harness()
    times = {0: [7e-4, 6e-4]}

    def step(settings, reps):
        return [4e-4] if settings.get("pair") else times[0]

    fake = harness.Backend("cpu", None, None, None, None, None, None, step=step, clock_mhz=lambda: 2000.0)
    facts = {"before": {"clocks.max.sm": "2500", "nvidia_smi_id": "GPU-b"}}
    got = harness.step_floor(fake, 2, 483_000, facts)
    assert got["fastest"] == "pair" and got["min_s"] == 4e-4 and got["below_certificate"]
    assert set(got["variants"]) == set(harness.STEP_VARIANTS) and "not an upper bound on the adversary" in got["label"]
    assert harness.STEP_VARIANTS["chain_pairs"] == {"chain_pairs": True}                    # the decode's lane-pair kernel
    assert got["sampled_sm_mhz"] == 2000.0 and got["at_max_clock_s"] == 4e-4 * 2000 / 2500                  # scaled to max
    assert got["sampled_nvidia_smi_id"] == "GPU-b"
    assert got["at_plus15pct_overclock_s"] == got["at_max_clock_s"] / 1.15
    assert not harness.step_floor(fake, 2, 300_000, facts)["below_certificate"]


def test_the_plain_arm_warms_the_device_up_before_its_first_rep_and_keeps_no_warmup_rep(monkeypatch):
    harness = _harness()
    ran = []
    monkeypatch.setattr(harness, "forward", lambda flat, pl, x: ran.append(x))

    def timer(fn):
        fn()
        return 0.125

    fake = harness.Backend("cpu", None, None, None, None, None, timer, warmup_s=harness.PLAIN_WARMUP_S)
    reps, how = harness._plain_arm(fake, types.SimpleNamespace(view=lambda *_: None), [], {1: "x1", 8: "x8"}, [1, 8], 3)
    warm = int(harness.PLAIN_WARMUP_S / 0.125)
    assert warm > 0 and ran == ["x1"] * warm + ["x1"] * 4 + ["x8"] * 4           # then each T: one untimed call, 3 reps
    assert reps == {1: [0.125] * 3, 8: [0.125] * 3} and not how[1]["graphed"]
    assert harness.Backend("cpu", None, None, None, None, None, timer).warmup_s == 0           # the CPU stand-in's: none


def test_the_eager_fallback_runs_after_the_failed_capture_and_its_tensors_are_released():
    harness = _harness()
    held = []

    class Tensors:
        pass

    def graphed(fn):
        state = Tensors()                                      # a failed capture's tensors, alive in its frame
        held.append(weakref.ref(state))
        raise RuntimeError("operation not permitted when stream is capturing")

    def fn():
        assert held[0]() is None

    fake = harness.Backend("cpu", None, None, None, None, None, lambda f: (f(), 1e-3)[1], graphed=graphed)
    reps, how = harness._timed_arm(fake, fn, 2)
    assert reps == [1e-3, 1e-3] and not how["graphed"] and how["reason"].startswith("RuntimeError: operation not permitted")


SMI_STUB = """\
import os, sys
args = sys.argv[1:]
with open(os.environ["SMI_LOG"], "a") as f:
    f.write(" ".join(args) + "\\n")
gpus = os.environ["SMI_GPUS"].split()
if "-L" in args:
    for i, u in enumerate(gpus):
        print(f"GPU {i}: NVIDIA RTX PRO 6000 Blackwell Server Edition (UUID: {u})")
    sys.exit(0)
ids = [a[len("--id="):] for a in args if a.startswith("--id=")]
which = [gpus.index(ids[0]) if ids[0] in gpus else int(ids[0])] if ids else range(len(gpus))   # no --id: every GPU, GPU 0 first
query = next((a.split("=", 1)[1] for a in args if a.startswith("--query-gpu=")), None)
for i in which:
    vals = {"index": str(i), "uuid": gpus[i], "pci.bus_id": f"00000000:{i + 1:02X}:00.0", "compute_cap": "12.0",
            "name": "NVIDIA RTX PRO 6000 Blackwell Server Edition", "driver_version": "580.173.02", "clocks.sm": str(2000 + i)}
    print(", ".join(vals.get(q, "1") for q in query.split(",")) if query else f"GPU {vals['pci.bus_id']}\\n    SM : 2000 MHz")
"""


def _smi_stub(tmp_path, gpus: str):
    """A stub nvidia-smi in `tmp_path/bin` on a node with `gpus` (UUIDs, nvidia-smi's index order): it logs every call."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    smi = bin_dir / "nvidia-smi"
    smi.write_text(f"#!{sys.executable}\n{SMI_STUB}")
    smi.chmod(0o755)
    log = tmp_path / "smi.log"
    return bin_dir, log, {"SMI_LOG": str(log), "SMI_GPUS": gpus}


def _queries(log) -> list[str]:
    return [line for line in log.read_text().splitlines() if line != "-L"] if log.exists() else []


def test_the_harness_asks_nvidia_smi_about_the_leased_or_cuda_device_never_gpu_0(tmp_path, monkeypatch):
    harness = _harness()
    bin_dir, log, env = _smi_stub(tmp_path, "GPU-a GPU-b GPU-c")
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("GPU_LEASE_UUID", "GPU-b")
    facts = harness.gpu_facts(0)
    assert facts["nvidia_smi_id"] == "GPU-b" and facts["uuid"] == "GPU-b" and facts["index"] == "1"
    assert facts["clocks.sm"] == "2001" and harness._sm_clock_mhz(harness.device_uuid(0)) == 2001.0
    monkeypatch.delenv("GPU_LEASE_UUID")
    monkeypatch.setattr(harness, "cuda_uuid", lambda index: {0: "GPU-c"}.get(index))       # CUDA's device 0 is GPU 2
    facts = harness.gpu_facts(0)
    assert facts["nvidia_smi_id"] == facts["uuid"] == facts["cuda_uuid"] == "GPU-c" and facts["index"] == "2"
    assert harness.gpu_facts(1) == {"nvidia_smi": None, "nvidia_smi_id": None, "cuda_device": 1, "cuda_uuid": None,
                                    "cuda_visible_devices": None}                            # no id: no query at all
    assert harness._sm_clock_mhz(None) is None
    assert [c.split()[0] for c in _queries(log)] == ["--id=GPU-b", "--id=GPU-b", "--id=GPU-c"]
    monkeypatch.undo()
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index: types.SimpleNamespace(uuid="1cd543c7-ad34"))
    assert harness.cuda_uuid(0) == "GPU-1cd543c7-ad34"                                       # torch's UUID, as nvidia-smi writes it


def _pod_sh(tmp_path, case: str, gpus: str, *args: str, **env: str):
    """pod.sh IMPL=band-gpu AUDITS=0 ARGS on stub nvidia-smi, uv, lscpu, nvcc and torch's venv python (which logs its arguments
    to py-args): its result, the nvidia-smi queries and the run directory."""
    d = tmp_path / case
    d.mkdir()
    bin_dir, log, smi_env = _smi_stub(d, gpus)
    for name, body in (("uv", "exit 0"), ("lscpu", "exit 0")):
        (bin_dir / name).write_text(f"#!/bin/sh\n{body}\n")
        (bin_dir / name).chmod(0o755)
    nvcc = d / "cuda" / "bin" / "nvcc"
    nvcc.parent.mkdir(parents=True)
    nvcc.write_text("#!/bin/sh\necho 'Cuda compilation tools, release 13.0, V13.0.88'\n")
    nvcc.chmod(0o755)
    py = d / "work" / "pous-bench" / "gpu312-torch2.14.0-cu130" / "bin" / "python"     # torch's venv, as if built
    py.parent.mkdir(parents=True)
    py.write_text(f"#!/bin/sh\necho \"$@\" >> {d / 'py-args'}\nexit 0\n")
    py.chmod(0o755)
    (d / "run").mkdir()
    base = {"PATH": f"{bin_dir}:/usr/bin:/bin", "HOME": str(d), "RESEARCH_RUN_DIR": str(d / "run"), "WORK": str(d / "work"),
            "CUDA_HOME": str(d / "cuda")}
    rc = subprocess.run(["bash", str(BENCH / "pod.sh"), "IMPL=band-gpu", "AUDITS=0", *args], cwd=ROOT, capture_output=True, text=True,
                        env={**base, **smi_env, **env})
    return rc, _queries(log), d / "run"


def test_pod_sh_asks_nvidia_smi_about_the_runs_gpu_and_refuses_to_guess_one(tmp_path):
    def run(case: str, gpus: str, **env: str):
        return _pod_sh(tmp_path, case, gpus, **env)

    rc, calls, out = run("lease", "GPU-a GPU-b GPU-c", GPU_LEASE_UUID="GPU-b", CUDA_VISIBLE_DEVICES="1",
                         KERNELS='{"chain_pairs": false}')
    assert rc.returncode == 0, rc.stdout[-2000:]
    assert calls and all(c.startswith("--id=GPU-b ") for c in calls)
    assert ' --kernels {"chain_pairs": false}' in (tmp_path / "lease" / "py-args").read_text()          # KERNELS reaches bench.py
    for name in ("before", "after"):
        assert (out / f"clocks-{name}.txt").read_text().startswith("nvidia-smi --id=GPU-b: 1, GPU-b, 00000000:02:00.0\n")
    rc, calls, _ = run("visible", "GPU-a GPU-b GPU-c", CUDA_VISIBLE_DEVICES="2")
    assert rc.returncode == 0 and calls[0] == "--id=2 --query-gpu=uuid --format=csv,noheader"
    assert calls[1:] and all(c.startswith("--id=GPU-c ") for c in calls[1:])
    assert "--kernels" not in (tmp_path / "visible" / "py-args").read_text()                         # no KERNELS: the defaults
    rc, calls, _ = run("only", "GPU-only")
    assert rc.returncode == 0 and calls and all(c.startswith("--id=GPU-only ") for c in calls)
    rc, calls, _ = run("several", "GPU-a GPU-b GPU-c")
    assert rc.returncode == 2 and "name the run's GPU" in rc.stdout and calls == []


def test_pod_sh_prepare_builds_the_lean_verdicts_and_touches_no_gpu(tmp_path):
    """PREPARE=1, run before the GPU lease: for p2-v3-mc-gpu, `lake build memory-challenger-verdict` in its Lean package
    (lake from the user's elan here, /opt/elan's on the pod image) and exit, with no nvidia-smi query; for p2-v3-gpu,
    `lake build pous-timed-audit` in pous_check where its lakefile declares one; for another IMPL, nothing; a failed build
    exits 3."""
    home = tmp_path / "home"
    lake = home / ".elan" / "bin" / "lake"
    lake.parent.mkdir(parents=True)
    lake.write_text(f"#!/bin/sh\necho \"$PWD $*\" >> {tmp_path / 'lake-args'}\nexit ${{LAKE_RC:-0}}\n")
    lake.chmod(0o755)
    rc, calls, _ = _pod_sh(tmp_path, "mc", "GPU-a GPU-b", "PREPARE=1", "IMPL=p2-v3-mc-gpu", HOME=str(home))
    assert rc.returncode == 0 and calls == []
    package = ROOT / "verity" / "core" / "devices" / "memory_challenger" / "lean"
    assert (tmp_path / "lake-args").read_text() == f"{package} build memory-challenger-verdict\n"
    rc, calls, _ = _pod_sh(tmp_path, "band", "GPU-a GPU-b", "PREPARE=1", HOME=str(home))
    assert rc.returncode == 0 and calls == [] and (tmp_path / "lake-args").read_text().count("\n") == 1
    check = ROOT / "verity" / "core" / "protocols" / "pous" / "check"
    declared = 'name = "pous-timed-audit"' in (check / "lakefile.toml").read_text()
    rc, calls, _ = _pod_sh(tmp_path, "strict", "GPU-a GPU-b", "PREPARE=1", "IMPL=p2-v3-gpu", HOME=str(home))
    assert rc.returncode == 0 and calls == []
    assert (tmp_path / "lake-args").read_text().splitlines()[1:] == ([f"{check} build pous-timed-audit"] if declared else [])
    (tmp_path / "lake-args").unlink()
    rc, calls, _ = _pod_sh(tmp_path, "failed", "GPU-a GPU-b", "PREPARE=1", "IMPL=p2-v3-mc-gpu", HOME=str(home), LAKE_RC="1")
    assert rc.returncode == 3 and calls == []


def test_pod_sh_python_path_loads_the_lean_verdict_without_site_packages():
    """The pod's venv holds numpy, torch and gmpy2 alone, and pod.sh's PYTHONPATH the rest: with that path and no
    site-packages at all, the memory challenger's Lean verdict (`verdict.py`, through `verity_difftest`, Hypothesis
    unimportable) and `live.py` load.  A run's five arms once signed their sessions and then lost them to
    `No module named 'verity_difftest'`."""
    line = next(s for s in (BENCH / "pod.sh").read_text().splitlines() if s.startswith("export PYTHONPATH="))
    path = line.split("=", 1)[1].replace("$REPO", str(ROOT))
    code = ("import importlib.util, sys\n"
            "sys.modules['hypothesis'] = None\n"
            "spec = importlib.util.spec_from_file_location('v', sys.argv[1])\n"
            "spec.loader.exec_module(importlib.util.module_from_spec(spec))\n"
            "sys.path.insert(0, sys.argv[2])\n"
            "from memory_challenger import live\n")
    verdict = ROOT / "verity" / "core" / "devices" / "memory_challenger" / "lean" / "verdict.py"
    rc = subprocess.run([sys.executable, "-S", "-c", code, str(verdict), str(BENCH)], capture_output=True, text=True,
                        env={"PYTHONPATH": path, "PATH": os.environ.get("PATH", "/usr/bin:/bin")})
    assert rc.returncode == 0, rc.stderr[-2000:]


def test_pod_sh_runs_the_placement_gate_first_and_stops_on_its_refusal(tmp_path):
    """GATE_ROLE: placement.py's gate checks the run's machine before anything is built or timed; here the guarded node's record
    is another machine's, so the run exits 4 with the verdict recorded and bench.py never starts."""
    node = tmp_path / "n2.json"
    node.write_text(json.dumps({"boot_id": "another-machine"}))
    rc, _, out = _pod_sh(tmp_path, "gate", "GPU-only", "GATE_ROLE=memory-challenger", "GATE_CORES=124-127",
                         f"GATE_PEERS=worker:gpu-node={node}", "GATE_ABSENT=firewall challenger")
    assert rc.returncode == 4 and "the placement gate refused this run as memory-challenger" in rc.stdout, rc.stdout[-2000:]
    v = json.loads((out / "placement-gate.json").read_text())
    assert v["role"] == "memory-challenger" and v["passed"] is False and v["absent"] == ["challenger", "firewall"]
    assert any(p.startswith("gpu-node (worker): its boot_id ") and p.endswith(" is not the node's another-machine")
               for p in v["problems"]), v["problems"]
    assert v["cores"]["declared"] == "124-127" and not (tmp_path / "gate" / "py-args").exists()
    rc, _, out = _pod_sh(tmp_path, "ungated", "GPU-only")
    assert rc.returncode == 0 and not (out / "placement-gate.json").exists()                      # no GATE_ROLE: no gate


def test_the_kernel_settings_and_the_loop_layout():
    harness = _harness()
    before = harness.set_kernels({})
    try:
        now = harness.set_kernels(harness.FROZEN_KERNELS)
        assert now["thread_rows"] == 500 and now["whole_chain"] is True and now["chain_pairs"] is False
        assert harness.set_kernels(harness.KERNELS) == {**now, "chain_pairs": True}       # the record's, the chain on lane pairs
        assert harness.set_kernels({"chain_pairs": False}) == now                         # selects the record's again
        with pytest.raises(ValueError, match="unknown kernel settings"):
            harness.set_kernels({"turbo": 1})
    finally:
        harness.set_kernels(before)
    pl = harness.placements(WS["tensors"], 2000)
    assert [(name, off, size) for name, _, _, off, size in pl] == [("q", 0, 1024), ("k", 1024, 512)]   # down ends past 2000


def test_the_harness_moves_every_one_of_its_threads_off_the_memory_challengers_and_responders_cpus():
    harness = _harness()
    here = sorted(os.sched_getaffinity(0))
    if len(here) < 3:
        pytest.skip("needs three CPUs")
    started, stop, tid = threading.Event(), threading.Event(), []

    def wait():
        tid.append(threading.get_native_id())
        started.set()
        stop.wait()

    t = threading.Thread(target=wait)
    t.start()
    started.wait()
    try:
        harness.leave_reserved(types.SimpleNamespace(verifier_cpu=here[0], responder_cpu=here[1]))
        assert os.sched_getaffinity(0) == os.sched_getaffinity(tid[0]) == set(here[2:])     # a thread started before, too
    finally:
        stop.set()
        t.join()
        for x in os.listdir("/proc/self/task"):
            try:
                os.sched_setaffinity(int(x), here)
            except ProcessLookupError:
                pass
