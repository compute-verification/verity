"""`wipe_store`, the streamed wipe's prover keeping the fill in device memory, here on its host-memory stand-in: it keeps
each frame where its labels go and answers each read from there, its gate writes both sides of every power-of-two byte
boundary and refuses a store whose addresses wrap, its cheat keeps prior content in the first labels' places and is
rejected one label past f·M, a host without the CUDA driver is refused by name, and the leg runs whole on it against a
challenger side of its own host that streams natively, recording the prover's, the session's and the streamer's figures
and the store in the placement, and against a simulated placement's side on the cheat, rejecting."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import pous_test_signer as ts
from verity.core.protocols.pous import memory_challenger as mc
from verity.core.protocols.pous import streamed_wipe as sw
from verity.core.protocols.pous import wipe_party as WP
from verity.core.protocols.pous import wipe_period as W
from verity.core.protocols.pous import wipe_store as WS

KEY = bytes(range(32))
SMALL = sw.WipeParams(64, W.READS, 1, 1_000, (("gpu", 64 * sw.LABEL_BYTES),))


class Clock:
    """Each reading 7 ns after the last."""

    def __init__(self) -> None:
        self.now = 0

    def __call__(self) -> int:
        self.now += 7
        return self.now


def test_the_gate_writes_each_power_of_two_boundary() -> None:
    n = 139 * 10 ** 9 // sw.LABEL_BYTES
    idx = WS.gate_indices(n)
    boundaries = [(1 << b) // 64 for b in range(25, 38)]
    assert idx == sorted({0, n // 2, n - 1, *boundaries, *(i - 1 for i in boundaries)})
    assert {(1 << 31) // 64 - 1, (1 << 31) // 64, (1 << 32) // 64 - 1, (1 << 32) // 64} <= set(idx)
    assert (1 << 38) // 64 > n and WS.gate_indices(64) == [0, 32, 63]


def test_the_store_keeps_the_fill_and_answers_from_it() -> None:
    p = WS.StoreProver(SMALL, WS.HostMemory(), staging=1000, clock=Clock())
    assert p.gate() == {"labels": 3, "highest": 63, "passed": True}
    p.receive(0, sw.labels(KEY, 0, 40))
    p.receive(40, sw.labels(KEY, 40, 24))
    assert p.answer([63, 0, 17]) == sw.label(KEY, 63) + sw.label(KEY, 0) + sw.label(KEY, 17)
    r = p.record()
    assert r["store"] == {"backend": "host", "host": r["store"]["host"], "tier": "host-dram", "bytes": 64 * 64}
    assert r["fill"]["bytes"] == 64 * 64 and r["fill"]["frames"] == 2 and r["reads"]["n"] == 3
    assert r["reads"]["p50_ns"] == r["reads"]["max_ns"] == 7, "one clock step a read"
    with pytest.raises(ValueError, match="outside the fill"):
        p.receive(60, sw.labels(KEY, 60, 8))
    p.close()


class Wrapping(WS.HostMemory):
    """Host memory whose store wraps at 32 MiB, as a 25-bit offset would."""

    base: int | None = None

    def alloc(self, n: int) -> int:
        a = super().alloc(n)
        self.base = a if self.base is None else self.base
        return a

    def _at(self, addr: int) -> int:
        return self.base + (addr - self.base) % (1 << 25) if addr - self.base >= 1 << 25 else addr

    def htod(self, dst: int, src: int, n: int) -> None:
        super().htod(self._at(dst), src, n)

    def dtoh(self, dst: int, src: int, n: int) -> None:
        super().dtoh(dst, self._at(src), n)


def test_a_store_whose_addresses_wrap_is_refused_at_the_gate() -> None:
    p = WS.StoreProver(W.params_for(2), Wrapping(), clock=Clock())
    with pytest.raises(RuntimeError, match=r"doesn't keep labels \[0, "):
        p.gate()


def test_the_cheat_keeps_prior_content_in_place_of_the_first_labels() -> None:
    p = WS.StoreProver(SMALL, WS.HostMemory(), staging=1000, clock=Clock(), foreign=40)
    p.receive(0, sw.labels(KEY, 0, 64))
    kept = p.answer(range(64))
    assert all(kept[64 * i:64 * (i + 1)] != sw.label(KEY, i) for i in range(40))
    assert kept[64 * 40:] == sw.labels(KEY, 40, 24)
    assert p.record()["foreign"] == {"labels": 40, "bytes": 40 * 64}
    p.close()
    with pytest.raises(ValueError, match="foreign is 65 labels"):
        WS.StoreProver(SMALL, WS.HostMemory(), foreign=65)


def test_a_store_keeping_prior_content_past_f_m_is_rejected() -> None:
    """The GPU store's cheat one label past f·M, on host memory: like `Keeping`, it misses a read with probability at
    least `1 − (18/19)^2152`; on these coins it misses one."""
    params = W.params_for()
    n = W.foreign_labels("fM", params)
    assert (n - 1) * sw.LABEL_BITS <= -(-W.memory_bits(params) // 19) < n * sw.LABEL_BITS
    store = WS.StoreProver(params, WS.HostMemory(), clock=Clock(), foreign=n)
    w = W.run("n0", params, signer=ts.TestSigner(b"store cheat", 2), log=ts.session_log(), schemes=ts.SCHEMES,
              protocol=mc.TEST_PREFIX + W.PROTOCOL, clock=Clock(), coin_secret=b"coins", egress_bits=0,
              prover=lambda _: store)
    store.close()
    assert w.verdict == "reject" and w.report["ended"] == sw.WRONG and "wrong answers" in w.answer["reason"]


def test_a_host_without_the_cuda_driver_is_refused() -> None:
    with pytest.raises(RuntimeError, match="no CUDA driver on this host"):
        WS.Cuda(library="libcuda-absent.so.1")


def test_percentiles() -> None:
    assert WS.percentiles(list(range(100, 0, -1))) == {"n": 100, "p50_ns": 50, "p99_ns": 99, "max_ns": 100,
                                                       "mean_ns": 50}
    assert WS.percentiles([]) == {"n": 0} and WS.gbps(10, 0) is None and WS.gbps(3_000, 1_000) == 3.0


def test_the_leg_on_a_store_with_a_native_fill(tmp_path, monkeypatch, wipe_streamer) -> None:
    """The leg whole as the H200's will run: period.json's wipe on a `wipe_store` prover (host memory standing in for
    the device) gated first, against a challenger side of its own host (`wipe_side.py`) streaming natively. The record
    holds the prover's fill and device reads, the signed session's fill and round trips, and the streamer's gate and
    figures; the placement names the store."""
    period = tmp_path / "periods" / "p0"
    period.mkdir(parents=True)
    (period / "period.json").write_text(json.dumps({"challenger_host": "auditor",
                                                    "wipe": {"store": "host", "segments": 1, "streamer": "native"}}))
    device = tmp_path / "device"
    device.mkdir()
    side = WP.Party("auditor", (sys.executable, str(Path(__file__).parent / "wipe_side.py"), str(device),
                                str(wipe_streamer.path)))
    monkeypatch.setattr(WP.Party, "of", classmethod(lambda cls, doc, node, d, extra=(): side))
    out = tmp_path / "out"
    answer = W.leg(period, "n0", out)
    assert answer["verdict"] == "accept", answer
    assert answer["profile"]["tiers"] == [["host-dram", 1 << 25]]
    assert answer["profile"]["placement"]["prover_store"]["tier"] == "host-dram"
    record = json.loads((out / "wipe.json").read_text())
    assert record["prover"]["gate"]["passed"] and record["prover"]["fill"]["bytes"] == 1 << 25
    assert record["prover"]["reads"]["n"] == W.READS and record["session"]["reads"]["n"] == W.READS
    assert record["session"]["fill"]["bytes"] == 1 << 25
    assert record["report"]["fill"]["labels"] == W.SEGMENT_LABELS and record["report"]["streamer"]["gates"]
    s = W.summary(record)
    assert s["verdict"] == "accept" and s["judge"] == "python" and s["segments"] == 1 and s["tier"] == "host-dram"


def test_the_cheats_leg_rejects_under_a_simulated_placement(tmp_path, monkeypatch) -> None:
    """`main` on period.json's cheat (`foreign_bytes` `fM`) with the challenger a process of its own on the prover's
    node named `simulated`: the wipe rejects, `--expect reject` exits 0, and the record names the placement and what
    the prover kept."""
    period = tmp_path / "periods" / "p0"
    period.mkdir(parents=True)
    (period / "period.json").write_text(json.dumps({"wipe": {"store": "host", "segments": 1, "foreign_bytes": "fM",
                                                             "placement": "simulated", "egress_bits": 0}}))
    device = tmp_path / "device"
    device.mkdir()
    side = WP.Party("n0", (sys.executable, str(Path(__file__).parent / "wipe_side.py"), str(device)))
    extras = []
    monkeypatch.setattr(WP.Party, "of", classmethod(lambda cls, doc, node, d, extra=(): extras.append(extra) or side))
    out = tmp_path / "out"
    assert W.main(["--period", str(period), "--out", str(out), "--node", "n0", "--expect", "reject"]) == 0
    assert extras == [("--egress-bits", "0")], "the local challenger signs the period's egress count"
    record = json.loads((out / "wipe.json").read_text())
    n = W.foreign_labels("fM", W.params_for())
    assert record["answer"]["verdict"] == "reject" and record["prover"]["foreign"]["labels"] == n
    assert record["placement"]["kind"] == "simulated" and record["placement"][WP.PREMISE].startswith("simulated: ")
    assert W.summary(record)["foreign_labels"] == n == record["wipe"]["foreign_labels"]
