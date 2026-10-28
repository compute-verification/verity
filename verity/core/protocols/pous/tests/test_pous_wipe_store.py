"""`wipe_store`, the streamed wipe's prover keeping the fill in device memory, here on its host-memory stand-in: it keeps
each frame where its labels go and answers each read from there, its gate writes both sides of every power-of-two byte
boundary and refuses a store whose addresses wrap, its cheat keeps prior content in the first labels' places and is
rejected one label past the published threshold, a host without the CUDA driver is refused by name, and the leg runs
whole on it against a challenger side of its own host that streams natively, recording the prover's, the session's and
the streamer's figures and the store in the placement, and against a simulated placement's side on the cheat,
rejecting. On a stand-in CUDA driver and NVML (`wipe_store.load`), a device is the larger of the two totals, NVML's
read at the PCI bus id CUDA reports, so the leg's `R` counts the driver's reserve, and without NVML the record names
the reserve uncertified. On a disk (`Disk`, a small fill file in the test's directory) it keeps the fill in a file of
its own, written in whole blocks and patched where a frame isn't, on the file once synced, and removed at close,
refuses a fill it can't keep and never takes an existing file; one session over a disk and host memory (`Joint`)
accepts the honest server and rejects the one keeping its own content on the disk, and the leg runs it whole, the disk
declared."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from pous_fakes import Clock
import pous_test_signer as ts
from verity.core.protocols.pous import memory_challenger as mc
from verity.core.protocols.pous import streamed_wipe as sw
from verity.core.protocols.pous import wipe_party as WP
from verity.core.protocols.pous import wipe_period as W
from verity.core.protocols.pous import wipe_store as WS

KEY = bytes(range(32))
SMALL = sw.WipeParams(64, W.READS, 1, 1_000, (("gpu", 64 * sw.LABEL_BYTES),))


def test_the_gate_writes_each_power_of_two_boundary() -> None:
    n = 139 * 10 ** 9 // sw.LABEL_BYTES
    idx = WS.gate_indices(n)
    boundaries = [(1 << b) // 64 for b in range(25, 38)]
    assert idx == sorted({0, n // 2, n - 1, *boundaries, *(i - 1 for i in boundaries)})
    assert {(1 << 31) // 64 - 1, (1 << 31) // 64, (1 << 32) // 64 - 1, (1 << 32) // 64} <= set(idx)
    assert (1 << 38) // 64 > n and WS.gate_indices(64) == [0, 32, 63]


def test_the_store_keeps_the_fill_and_answers_from_it() -> None:
    p = WS.StoreProver(SMALL, WS.HostMemory(), staging=1000, clock=Clock(step=7))
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
    p = WS.StoreProver(W.params_for(2), Wrapping(), clock=Clock(step=7))
    with pytest.raises(RuntimeError, match=r"doesn't keep labels \[0, "):
        p.gate()


def test_the_cheat_keeps_prior_content_in_place_of_the_first_labels() -> None:
    p = WS.StoreProver(SMALL, WS.HostMemory(), staging=1000, clock=Clock(step=7), foreign=40)
    p.receive(0, sw.labels(KEY, 0, 64))
    kept = p.answer(range(64))
    assert all(kept[64 * i:64 * (i + 1)] != sw.label(KEY, i) for i in range(40))
    assert kept[64 * 40:] == sw.labels(KEY, 40, 24)
    assert p.record()["foreign"] == {"labels": 40, "bytes": 40 * 64}
    p.close()
    with pytest.raises(ValueError, match="foreign is 65 labels"):
        WS.StoreProver(SMALL, WS.HostMemory(), foreign=65)


def test_a_store_keeping_prior_content_past_the_threshold_is_rejected() -> None:
    """The GPU store's cheat one label past the published threshold, on host memory: like `Keeping`, it misses a read
    with probability `1 − (1 − n/N)^2152`, past `1 − 2^-128`; on these coins it misses one."""
    params = W.params_for()
    n, threshold = W.foreign_labels("fM", params), W.certify(params, 0)["foreign_bits"]
    assert (n - 1) * sw.LABEL_BITS <= threshold < n * sw.LABEL_BITS
    store = WS.StoreProver(params, WS.HostMemory(), clock=Clock(step=7), foreign=n)
    w = W.run("n0", params, signer=ts.TestSigner(b"store cheat", 2), log=ts.session_log(), schemes=ts.SCHEMES,
              protocol=mc.TEST_PREFIX + W.PROTOCOL, clock=Clock(step=7), coin_secret=b"coins", egress_bits=0,
              prover=lambda _: store)
    store.close()
    assert w.verdict == "reject" and w.report["ended"] == sw.WRONG and "wrong answers" in w.answer["reason"]


def test_a_host_without_the_cuda_driver_is_refused() -> None:
    with pytest.raises(RuntimeError, match="no CUDA driver on this host"):
        WS.Cuda(library="libcuda-absent.so.1")


H200_CUDA, H200_NVML = 150_109_880_320, 150_754_820_096
"""Demo B's H200: `cuDeviceTotalMem`'s total, and NVML's (`nvidia-smi`'s 143,771 MiB)."""
RESERVE = H200_NVML - H200_CUDA
BUS = "0000:1A:00.0"


def _set(ref, value: int) -> int:
    ref._obj.value = value
    return 0


def _put(buf, value: bytes) -> int:
    buf.value = value
    return 0


def _raw(buf, value: bytes) -> int:
    buf.raw = value
    return 0


def cuda_on_host(total: int, bus: str = BUS) -> SimpleNamespace:
    """A CUDA driver with one H200 of `total` bytes at PCI bus id `bus`, its memory host memory: each function a Python
    one taking what `Cuda` passes, an out-parameter written through its `byref`."""
    host = WS.HostMemory()
    return SimpleNamespace(
        cuInit=lambda flags: 0, cuDriverGetVersion=lambda v: _set(v, 12080), cuDeviceGet=lambda d, i: _set(d, i),
        cuDeviceGetName=lambda buf, n, dev: _put(buf, b"NVIDIA H200"),
        cuDeviceGetUuid=lambda buf, dev: _raw(buf, bytes(range(16))),
        cuDeviceGetAttribute=lambda v, attr, dev: _set(v, {75: 9, 76: 0}[attr]),
        cuDeviceGetPCIBusId=lambda buf, n, dev: _put(buf, bus.encode()),
        cuDeviceTotalMem_v2=lambda t, dev: _set(t, total), cuDevicePrimaryCtxRetain=lambda c, dev: _set(c, 1),
        cuDevicePrimaryCtxRelease_v2=lambda dev: 0, cuCtxSetCurrent=lambda c: 0,
        cuMemGetInfo_v2=lambda f, t: _set(f, total) or _set(t, total),
        cuMemAlloc_v2=lambda p, n: _set(p, host.alloc(n)), cuMemFree_v2=lambda p: host.free(p) or 0,
        cuMemHostAlloc=lambda p, n, flags: _set(p, host.alloc(n)), cuMemFreeHost=lambda p: host.free(p) or 0,
        cuMemcpyHtoD_v2=lambda d, s, n: host.htod(d, s, n) or 0,
        cuMemcpyDtoH_v2=lambda d, s, n: host.dtoh(d, s, n) or 0, cuGetErrorName=lambda r, s: 0)


def nvml_of(totals: dict[str, int], asked: list[str]) -> SimpleNamespace:
    """NVML over the GPUs `{PCI bus id: total bytes}`, each bus id it is asked for appended to `asked`; another is
    `NVML_ERROR_NOT_FOUND` (6)."""
    buses = list(totals)

    def by_bus(bus: bytes, h) -> int:
        asked.append(bus.decode())
        return _set(h, buses.index(bus.decode()) + 1) if bus.decode() in totals else 6

    def memory(h, m) -> int:
        m._obj.total = totals[buses[h.value - 1]]
        return 0

    return SimpleNamespace(nvmlInit_v2=lambda: 0, nvmlShutdown=lambda: 0, nvmlErrorString=lambda r: b"Not Found",
                           nvmlDeviceGetHandleByPciBusId_v2=by_bus, nvmlDeviceGetMemoryInfo=memory)


def _drivers(monkeypatch, libs: dict[str, SimpleNamespace]) -> None:
    """`wipe_store.load` opening `libs` by name, and refusing any other as a host without it does."""

    def load(name: str) -> SimpleNamespace:
        if name not in libs:
            raise OSError(f"{name}: cannot open shared object file: No such file or directory")
        return libs[name]

    monkeypatch.setattr(WS, "load", load)


def _totals(info: dict) -> tuple:
    return info["cuda_total_bytes"], info["nvml_total_bytes"], info["total_bytes"], info["total_source"]


def test_the_device_is_the_larger_of_cudas_and_nvmls_totals(monkeypatch) -> None:
    """An H200's: CUDA's total leaves out the 615.06 MiB its driver reserves and NVML's counts it, so the device is
    NVML's total for the GPU at the bus id CUDA reports, not another GPU NVML lists; a CUDA total past NVML's stands."""
    asked: list[str] = []
    _drivers(monkeypatch, {"libcuda.so.1": cuda_on_host(H200_CUDA),
                           WS.NVML: nvml_of({"0000:0F:00.0": 1 << 40, BUS: H200_NVML}, asked)})
    info = WS.Cuda(0).info()
    assert asked == [BUS] and info["pci_bus_id"] == BUS and "nvml_error" not in info
    assert _totals(info) == (H200_CUDA, H200_NVML, H200_NVML, "nvml")
    assert RESERVE == 644_939_776 and 8 * RESERVE == 5_159_518_208 and round(RESERVE / 2 ** 20, 2) == 615.06
    _drivers(monkeypatch, {"libcuda.so.1": cuda_on_host(H200_CUDA), WS.NVML: nvml_of({BUS: H200_CUDA - 1}, [])})
    assert _totals(WS.Cuda(0).info()) == (H200_CUDA, H200_CUDA - 1, H200_CUDA, "cuda")


def test_without_nvmls_total_the_device_is_cudas_and_says_why(monkeypatch) -> None:
    """A host without NVML, and an NVML without the GPU at CUDA's bus id: CUDA's total, and the reason."""
    _drivers(monkeypatch, {"libcuda.so.1": cuda_on_host(H200_CUDA)})
    info = WS.Cuda(0).info()
    assert _totals(info) == (H200_CUDA, None, H200_CUDA, "cuda")
    assert info["nvml_error"] == ("no NVML on this host (libnvidia-ml.so.1): libnvidia-ml.so.1: cannot open shared "
                                  "object file: No such file or directory")
    asked: list[str] = []
    _drivers(monkeypatch, {"libcuda.so.1": cuda_on_host(H200_CUDA, "0000:3C:00.0"),
                           WS.NVML: nvml_of({BUS: H200_NVML}, asked)})
    other = WS.Cuda(0).info()
    assert asked == ["0000:3C:00.0"] and _totals(other) == (H200_CUDA, None, H200_CUDA, "cuda")
    assert other["nvml_error"] == "nvmlDeviceGetHandleByPciBusId_v2 failed for the GPU at 0000:3C:00.0: Not Found (6)"


@pytest.mark.parametrize("nvml", [True, False], ids=["nvml", "no-nvml"])
def test_the_leg_on_a_gpu_declares_its_driver_reserve(tmp_path, monkeypatch, nvml: bool) -> None:
    """The leg on a GPU (`wipe.store` `cuda`, on `cuda_on_host`) of 64 MiB by CUDA's count and an H200's driver reserve
    more by NVML's: the tier declared is NVML's total, so `R`, and `M` with it, counts the reserve (on Demo B's H200,
    the 5,159,518,208 bits CUDA's count leaves out), and none of the device is undeclared. Without NVML the tier is
    CUDA's, and the record names the reserve uncertified, with the reason."""
    cuda = 64 << 20
    tier = cuda + RESERVE if nvml else cuda
    _drivers(monkeypatch, {"libcuda.so.1": cuda_on_host(cuda)} | ({WS.NVML: nvml_of({BUS: tier}, [])} if nvml else {}))
    period = tmp_path / "periods" / "p0"
    period.mkdir(parents=True)
    (period / "period.json").write_text(json.dumps({"challenger_host": "auditor",
                                                    "wipe": {"store": "cuda", "segments": 1}}))
    device = tmp_path / "device"
    device.mkdir()
    side = WP.Party("auditor", (sys.executable, str(Path(__file__).parent / "wipe_side.py"), str(device)))
    monkeypatch.setattr(WP.Party, "of", classmethod(lambda cls, doc, node, d, extra=(): side))
    out = tmp_path / "out"
    answer = W.leg(period, "n0", out)
    record = json.loads((out / "wipe.json").read_text())
    assert answer["verdict"] == "accept" and record["prover"]["gate"]["passed"], answer
    assert answer["profile"]["tiers"] == [["gpu-hbm", tier]] and answer["public"]["memory_bits"] == 8 * tier
    assert answer["profile"]["certificate"]["outside_bits"] == record["wipe"]["outside_bits"] == 8 * (tier - (1 << 25))
    store = record["placement"]["prover_store"]
    assert _totals(store) == ((cuda, tier, tier, "nvml") if nvml else (cuda, None, cuda, "cuda"))
    assert [u for u in record["placement"]["undeclared_tiers"] if u["tier"] == "gpu-hbm"] == []
    reserve = [g for g in record["uncertified"] if "driver reserves" in g]
    assert reserve == ([] if nvml else [
        f"tier gpu-hbm: the memory NVIDIA H200's driver reserves is undeclared: the tier is CUDA's total ({cuda} bytes), "
        f"which leaves it out, since NVML gave none ({store['nvml_error']}); the fill doesn't cover it and M doesn't "
        "count it"])
    assert W.summary(record)["uncertified"] == record["uncertified"]


@pytest.mark.parametrize("nvml", [True, False], ids=["nvml", "no-nvml"])
def test_the_leg_on_a_disk_and_a_gpu_declares_the_gpus_driver_reserve(tmp_path, monkeypatch, nvml: bool) -> None:
    """The disks ahead of a GPU in one joint store, the whole-node session's shape: the GPU part is declared at NVML's
    total, and without NVML the record names that part's reserve uncertified."""
    cuda = 64 << 20
    tier = cuda + RESERVE if nvml else cuda
    _drivers(monkeypatch, {"libcuda.so.1": cuda_on_host(cuda)} | ({WS.NVML: nvml_of({BUS: tier}, [])} if nvml else {}))
    period = tmp_path / "periods" / "p0"
    period.mkdir(parents=True)
    target = tmp_path / "disk"
    target.mkdir()
    (period / "period.json").write_text(json.dumps({"challenger_host": "auditor", "wipe": {
        "store": "cuda", "segments": 1,
        "disk": {"paths": [str(target)], "bytes": 1 << 25, "reserve_bytes": 0, "direct": None}}}))
    device = tmp_path / "device"
    device.mkdir()
    side = WP.Party("auditor", (sys.executable, str(Path(__file__).parent / "wipe_side.py"), str(device)))
    monkeypatch.setattr(WP.Party, "of", classmethod(lambda cls, doc, node, d, extra=(): side))
    out = tmp_path / "out"
    answer = W.leg(period, "n0", out)
    assert answer["verdict"] == "accept", answer
    record = json.loads((out / "wipe.json").read_text())
    assert record["wipe"]["tier"] == "disk+gpu-hbm" and answer["profile"]["tiers"][1] == ["gpu-hbm", tier]
    reserve = [g for g in record["uncertified"] if "driver reserves" in g]
    assert len(reserve) == (0 if nvml else 1) and all(f"CUDA's total ({cuda} bytes)" in g for g in reserve)
    assert _fills(target) == []


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


def _fills(d: Path) -> list[Path]:
    return sorted(d.glob("verity-wipe-fill-*.bin"))


def test_a_disk_keeps_the_fill_in_a_file_of_its_own_and_answers_from_it(tmp_path) -> None:
    """A frame of whole blocks is written in flight, the rest read, patched and written back; once synced the file
    holds the fill, and close removes it."""
    disk = WS.Disk([tmp_path], reserve_bytes=0, direct=None)
    params = sw.WipeParams(256, W.READS, 1, 1_000, (("disk", disk.total),))
    p = WS.StoreProver(params, disk, staging=2 * WS.BLOCK, clock=Clock(step=7))
    [fill] = _fills(tmp_path)
    assert fill.stat().st_size == 256 * sw.LABEL_BYTES
    assert p.gate() == {"labels": 3, "highest": 255, "passed": True}
    p.receive(0, sw.labels(KEY, 0, 200))
    p.receive(200, sw.labels(KEY, 200, 56))
    p.sync()
    assert fill.read_bytes() == sw.labels(KEY, 0, 256)
    assert p.answer([255, 0, 128, 199, 200]) == b"".join(sw.label(KEY, i) for i in (255, 0, 128, 199, 200))
    r = p.record()
    assert r["store"]["tier"] == "disk" and r["store"]["backend"] == "disk" and r["fill"]["sync_ns"] == 7
    assert r["store"]["targets"] == [{"path": str(tmp_path), "kind": "file", "room": disk.targets[0]["room"],
                                      "file": str(fill), "bytes": 256 * sw.LABEL_BYTES}]
    assert disk.mem_info()[1] == disk.total >= os.statvfs(tmp_path).f_blocks * os.statvfs(tmp_path).f_frsize
    p.close()
    disk.close()
    assert _fills(tmp_path) == []


def test_a_disk_refuses_what_it_cannot_keep(tmp_path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    with pytest.raises(ValueError, match="on a device another target is on"):
        WS.Disk([a, b])
    (tmp_path / "x").write_bytes(b"x")
    with pytest.raises(ValueError, match="neither a directory nor a block device"):
        WS.Disk([tmp_path / "x"])
    with pytest.raises(ValueError, match="at least one target"):
        WS.Disk([])
    disk = WS.Disk([a], reserve_bytes=0, direct=None)
    with pytest.raises(ValueError, match="whole 4096-byte blocks"):
        disk.alloc(100)
    with pytest.raises(RuntimeError, match="room for"):
        disk.alloc(disk.mem_info()[0] + WS.BLOCK)
    taken = a / WS.FILL_FILE.format(pid=os.getpid(), i=0)
    taken.write_bytes(b"someone else's")
    with pytest.raises(FileExistsError):
        disk.alloc(WS.BLOCK)
    disk.close()
    assert taken.read_bytes() == b"someone else's", "an existing file is never taken, written or removed"
    full = WS.Disk([b], reserve_bytes=1 << 62)
    assert full.mem_info()[0] == 0
    full.close()


def _joint(d: Path) -> WS.Joint:
    return WS.Joint([(WS.Disk([d], reserve_bytes=0, direct=None), 256 * sw.LABEL_BYTES),
                     (WS.HostMemory(), 256 * sw.LABEL_BYTES)])


@pytest.mark.parametrize("foreign", [0, 128])
def test_one_session_over_a_disk_and_host_memory(tmp_path, foreign: int) -> None:
    """The disk first: the honest server accepts, and the one keeping its own content in place of the first 128
    labels, all on the disk, misses a read."""
    joint = _joint(tmp_path)
    assert joint.tier == "disk+host-dram"
    assert joint.declared() == (("disk", joint.parts[0][0].total), ("host-dram", 256 * sw.LABEL_BYTES))
    params = sw.WipeParams(512, W.READS, 1, W.LIMIT_NS, joint.declared())
    store = WS.StoreProver(params, joint, staging=4 * WS.BLOCK, clock=Clock(step=7), foreign=foreign)
    store.gate()
    w = W.run("n0", params, signer=ts.TestSigner(b"disk %d" % foreign, 2), log=ts.session_log(), schemes=ts.SCHEMES,
              protocol=mc.TEST_PREFIX + W.PROTOCOL, clock=Clock(step=7), secret=KEY, session_id=bytes(32),
              coin_secret=b"coins", egress_bits=0, prover=lambda _: store)
    [fill] = _fills(tmp_path)
    on_disk, key = fill.read_bytes(), sw.wipe_key(KEY, b"n0", bytes(32))
    assert store.sync_ns == 7 and store.record()["store"]["parts"][0]["tier"] == "disk"
    store.close()
    joint.close()
    assert _fills(tmp_path) == []
    assert on_disk[64 * foreign:] == sw.labels(key, foreign, 256 - foreign)
    assert all(on_disk[64 * i:64 * (i + 1)] != sw.label(key, i) for i in range(foreign))
    if foreign:
        assert w.verdict == "reject" and "wrong answers" in w.answer["reason"]
    else:
        assert w.verdict == "accept", w.answer
        assert w.answer["profile"]["tiers"] == [list(t) for t in joint.declared()]


def test_the_leg_on_a_disk_and_host_memory(tmp_path, monkeypatch) -> None:
    """period.json's wipe on host memory and a disk directory in one session against a challenger side of its own
    host: it accepts, both tiers declared, the disk's filesystems no longer undeclared, and the fill file removed."""
    period = tmp_path / "periods" / "p0"
    period.mkdir(parents=True)
    target = tmp_path / "disk"
    target.mkdir()
    (period / "period.json").write_text(json.dumps({"challenger_host": "auditor", "wipe": {
        "store": "host", "segments": 1, "disk": {"paths": [str(target)], "bytes": 1 << 25, "reserve_bytes": 0,
                                                 "direct": None}}}))
    device = tmp_path / "device"
    device.mkdir()
    side = WP.Party("auditor", (sys.executable, str(Path(__file__).parent / "wipe_side.py"), str(device)))
    monkeypatch.setattr(WP.Party, "of", classmethod(lambda cls, doc, node, d, extra=(): side))
    out = tmp_path / "out"
    answer = W.leg(period, "n0", out)
    assert answer["verdict"] == "accept", answer
    record = json.loads((out / "wipe.json").read_text())
    disk_total = record["prover"]["store"]["parts"][0]["total_bytes"]
    assert answer["profile"]["tiers"] == [["disk", disk_total], ["host-dram", 1 << 25]]
    assert record["wipe"]["tier"] == "disk+host-dram" and record["wipe"]["segments"] == 2
    assert record["prover"]["fill"]["bytes"] == 1 << 26 and record["prover"]["fill"]["sync_ns"] is not None
    part = record["prover"]["store"]["parts"][0]
    assert part["targets"][0]["bytes"] == 1 << 25 and part["direct"] in (True, False)
    dram = [u for u in record["placement"]["undeclared_tiers"] if u["tier"] == "host-dram"]
    assert dram[0]["what"] == "the host's DRAM outside the declared tier"
    assert _fills(target) == []


def test_the_leg_on_a_disk_and_the_whole_dram(tmp_path, monkeypatch) -> None:
    """wipe.dram after a disk store: the DRAM is a tier declared at all the host's physical memory, so none of it is
    undeclared, and the fill's last segment is in it."""
    period = tmp_path / "periods" / "p0"
    period.mkdir(parents=True)
    target = tmp_path / "disk"
    target.mkdir()
    (period / "period.json").write_text(json.dumps({"challenger_host": "auditor", "wipe": {
        "store": "disk", "disk": {"paths": [str(target)], "bytes": 1 << 25, "reserve_bytes": 0, "direct": None},
        "dram": {"bytes": 1 << 25, "reserve_bytes": 0}}}))
    device = tmp_path / "device"
    device.mkdir()
    side = WP.Party("auditor", (sys.executable, str(Path(__file__).parent / "wipe_side.py"), str(device)))
    monkeypatch.setattr(WP.Party, "of", classmethod(lambda cls, doc, node, d, extra=(): side))
    out = tmp_path / "out"
    answer = W.leg(period, "n0", out)
    assert answer["verdict"] == "accept", answer
    record = json.loads((out / "wipe.json").read_text())
    whole = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    disk_total = record["prover"]["store"]["parts"][0]["total_bytes"]
    assert answer["profile"]["tiers"] == [["disk", disk_total], ["host-dram", whole]]
    assert record["wipe"]["tier"] == "disk+host-dram" and record["wipe"]["segments"] == 2
    assert [u for u in record["placement"]["undeclared_tiers"] if u["tier"] == "host-dram"] == []
    assert _fills(target) == []
