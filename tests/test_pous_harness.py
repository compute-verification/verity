"""The POUS-only harness (`benchmarks/pous/band_gpu/harness.py`, `bench.py --impl band-gpu`) on CPU: the band's reference codec
stands in for the GPU twin on CPU tensors, a Python thread for the native responder (its wire format), and the verifier is the
harness's own `live.py` process.  The harness's timer is injected and the verifier runs on its logical clock (`--clock
logical`: each answer's time is the responder's reported one), so no outcome here depends on a clock or on the machine's
load: what is checked is that each step runs and each gate and control decides, not the numbers."""
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
    from band_gpu.live import scheme_of

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
    from verity_pous import schemes

    harness = _harness()
    b = params.blocks_per_segment * params.block_bytes
    w, salt = harness.vector_bytes("band-segment-w", b), harness.vector_bytes("salt", 24)
    blocks = schemes.get(BAND, params=params.with_segments(1)).encode_segment(salt, 0, w)
    return {"band/64kb/segment": hashlib.sha256(b"".join(blocks)).hexdigest()}


def _args(tmp_path, **kw):
    base = {"scheme": BAND, "max_segments": 0, "kernels": "", "build_dir": str(tmp_path / "build"), "reference_segments": 2,
            "workers": 2, "tokens": "1,8", "reps": 1, "load_tokens": 0, "audits": 2, "pings": 20, "allowance": "max", "k": 0, "deadline_ns": 0, "clock": "logical",
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
    for g in ("vector", "reference_encode", "round_trip", "timed_audit", "value_control", "late_control", "recompute_control",
              "segment_tags"):
        assert gates[g]["passed"], (g, gates[g])
    assert gates["reference_encode"]["segments"] == [0, 2]
    assert not gates["certificate"]["passed"] and "no certificate" in gates["certificate"]["reason"]   # the small point has none
    m = body["metrics"]
    assert m["loop"]["tokens"] == [1, 8] and m["loop"]["tensors"] == 3 and set(m["loop"]["overhead"]) == {1, 8}
    honest, value, late, rec = m["audit"], m["value_control"], m["late_control"], m["recompute_control"]
    from verity_pous import audit

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


def test_gates_only_runs_the_kernel_gates_and_times_nothing(tmp_path):
    params = _small()
    body = _bench().BandGpu().run(_args(tmp_path, gates_only=True), WS, tmp_path, backend=_backend(params), base=params,
                                  vectors=_vectors(params))
    assert set(body["gates"]) == {"vector", "reference_encode", "round_trip"} and all(g["passed"] for g in body["gates"].values())
    assert body["metrics"] is None and body["measurements"] == [] and "--gates-only" in body["stopped"]


def test_band_d_builds_the_band_at_that_in_degree_with_its_certificate_cliff_there():
    """`--band-d 6`: the band at in-degree 6, unregistered (`band-chain/d6/v1` is the codec's name, not a registered byte
    format), whose segments round-trip and whose certificate at the frozen point is `band_meets_64` with the cliff at 6;
    the verifier process rebuilds it from its manifest; d = 4 has no certificate and is refused."""
    from verity_pous import band, schemes

    harness = _harness()
    from band_gpu.live import params_of, scheme_of

    small = harness.with_in_degree(_small(), 6)
    sch = scheme_of("band-chain/d6/v1", small)
    assert sch.name == "band-chain/d6/v1" and sch.params.d == 6 and sch.name not in schemes.REGISTRY
    w = hashlib.shake_256(b"verity-pous/test/band-d").digest(small.segment_bytes)
    blocks = sch.encode_segment(harness.SALT, 0, w)
    assert sch.decode_segment(harness.SALT, 0, blocks) == w
    assert blocks != scheme_of(BAND, _small()).encode_segment(harness.SALT, 0, w)            # another d, another encoding
    point = harness.with_in_degree(band.SECURE_FIRST, 6)
    cert = scheme_of("band-chain/d6/v1", point).certificate
    assert (cert.theorem, cert.cliff, cert.k, cert.rounds, cert.call_ns) == ("band_meets_64", 6, 111, 5, 483_000)
    assert scheme_of("band-chain/d6/v1", point.with_segments(448)).certificate.theorem == "band_meets_14"
    gate, _ = _bench().certificate_gate(scheme_of("band-chain/d6/v1", point), 111, 700_000)
    assert gate["passed"] and gate["effective_rounds"] == 1
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


def test_the_band_recompute_xors_w_in_inside_its_timed_device_call():
    """The band's recompute adversary decodes the dropped block's key and XORs `W`'s block in within the one timed call, so
    its reported device time covers the whole answer and nothing is computed on the host after it."""
    harness = _harness()
    width, bps = 4, 8
    c = torch.arange(2 * bps * width, dtype=torch.uint8).view(2, bps, width)
    w_host = bytes(range(100, 100 + 2 * bps * width))
    key = torch.tensor([1, 2, 3, 4], dtype=torch.uint8)
    state = {"timing": False, "calls": []}

    def recompute(store, tags, i):
        assert int(store.view(-1, width)[i].sum()) == 0 and int(store.view(-1, width)[i - 1].sum()) > 0   # only it is dropped

        def call():
            state["calls"].append(state["timing"])
            return key.clone()
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


def test_the_step_floor_is_the_fastest_kernel_and_a_call_under_the_certificate_is_flagged():
    harness = _harness()
    times = {0: [7e-4, 6e-4]}

    def step(settings, reps):
        return [4e-4] if settings.get("pair") else times[0]

    fake = harness.Backend("cpu", None, None, None, None, None, None, step=step, clock_mhz=lambda: 2000.0)
    facts = {"before": {"clocks.max.sm": "2500"}}
    got = harness.step_floor(fake, 2, 483_000, facts)
    assert got["fastest"] == "pair" and got["min_s"] == 4e-4 and got["below_certificate"]
    assert set(got["variants"]) == set(harness.STEP_VARIANTS) and "upper bound" in got["label"]
    assert got["sampled_sm_mhz"] == 2000.0 and got["at_max_clock_s"] == 4e-4 * 2000 / 2500                  # scaled to max
    assert got["at_plus15pct_overclock_s"] == got["at_max_clock_s"] / 1.15
    assert not harness.step_floor(fake, 2, 300_000, facts)["below_certificate"]


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
