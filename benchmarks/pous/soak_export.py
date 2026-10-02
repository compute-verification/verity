"""The honest-latency soak's plot data, `pous-soak/v1`: per cell the survival curves and quantiles of each measured and
modeled latency, the thresholds τ, the server-side breakdown, a per-run and per-minute timeline, vLLM's serving rate, the
slowest challenge answers with their context, and interpolation vectors.

It reads the soak's chunks (`verity_vllm.pipeline.pous_soak`, one npz per chunk) one at a time and never holds two: every
per-cell statistic is accumulated as it goes.  The corrected rerun's chunks export the same way: their response header is
v2 (208 B, the zero-copy wait and completions), which enters the ToR model's response bytes, and a cell's `config` is one of
`CONFIGS` (baseline, tuned, zerocopy).  All of a file's chunks have one header and one set of sampled cores.

    uv run python benchmarks/pous/soak_export.py --out pous-soak.json CHUNK.npz [CHUNK.npz ...] [--art CHUNK.npz=art:… ...]

Definitions (the file's `metrics`, `run.notes` and the spec `pous-soak-plots.md`):

- An answer counts when its status is 0 and it was scheduled at least `WARMUP_NS` after its run's first arrival (`t0_ns`).
- Survival `s[i]` is the share of a row's `n` answers with value ≥ `T_US[i]`, counted exactly.
- Quantiles are nearest-rank order statistics from log histograms of `BINS_PER_DECADE` bins per decade (each bin 0.23 % of
  its value wide), reported as the bin's geometric centre clamped to the exact min and max; values under `HIST_LO_US` or at
  least `HIST_HI_US` are kept exactly.
- `survival_at` is the reference interpolation the vectors pin: 1 below `T_US[0]`, 0 past `max_us`, ln s linear in ln t
  between grid points, and on the segment straddling `max_us` linear from its grid point to (`max_us`, 1/n).
"""
from __future__ import annotations

import argparse
import bisect
import heapq
import json
import math
import subprocess
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "pous-soak/v1"
POINTS_PER_DECADE = 40
T_US = tuple(10 ** (k / POINTS_PER_DECADE) for k in range(5 * POINTS_PER_DECADE + 1))
BINS_PER_DECADE = 1000
HIST_LO_US, HIST_HI_US = 1e-2, 1e6
N_BINS = round(math.log10(HIST_HI_US / HIST_LO_US) * BINS_PER_DECADE)
WARMUP_NS = 1_000_000_000
TOP_STALLS = 200
OP_PING, OP_READ = 2, 4
OPS = {OP_READ: "challenge", OP_PING: "ping"}
LOADED = ("decode", "prefill", "mixed")
#: the responder configs a cell's `config` takes (`pous_soak` ALL_CONFIGS; zerocopy from the corrected rerun on)
CONFIGS = ("baseline", "tuned", "zerocopy")
#: the response header: 24 words (v1, the soak) or 26 (v2, the rerun's zero-copy wait and completions)
HEADER_BYTES = 192
HEADER_BYTES_V2 = 208
NODE0_CPUS = range(96)
BLOCK_BYTES = (65_536, 147_456, 393_216)
ROUNDS = range(6)
QUANTILES = (("p50_us", 0.5), ("p90_us", 0.9), ("p99_us", 0.99), ("p999_us", 0.999), ("p9999_us", 0.9999))
USER_HZ = 100
#: the sampler's thread targets, in order (`pous_soak` sched_targets)
TARGET_RESPONDER, TARGET_VERIFIER = 0, 1

#: node 2 (vy-nebius-2): mlx5_core, eth0 400,000 Mb/s, MTU 1500, adaptive RX/TX coalescing with rx-usecs 8 (the soak's
#: snap and the run doc's Node 2 facts)
MTU = 1500
MSS = MTU - 20 - 20 - 12
FRAME_OVERHEAD = 90
B_PCIE = 25e9
RX_USECS_DEFAULT = 8.0
NIC_RX_BASE_US = 2.0
TX_FIXED_US = 1.0
REQ_FRAME = 106
PROP_NS_PER_M, CABLE_M = 5.0, 3.0
TOR = {"tor_400_tuned": (400e9, "tuned"), "tor_400_default": (400e9, "default"),
       "tor_100_tuned": (100e9, "tuned"), "tor_100_default": (100e9, "default")}

MEASURED = {
    "elapsed": ("verifier send to last byte received", "V1 − V0 (t_recv − t_send0)", False),
    "server_user": ("server user time, request read to last send returned", "S4 − S0", False),
    "server_kernel": ("server kernel time, request RX stamp to response TX stamp", "K_tx_resp − K_rx_req", True),
    "gpu_span": ("device copy span as the server saw it", "S2 − S1", False),
    "schedule_lag": ("verifier send behind its schedule", "V0 − t_sched", False),
}
NON_GPU = ("elapsed", "server_user", "server_kernel", "schedule_lag")

#: breakdown components per op: id, label, formula, and the stamps they subtract (later, earlier); "G" is device_ns
COMPONENTS = {
    "challenge": (
        ("request", "verifier send to server kernel RX", "K_rx_req − V0", ("K_rx_req", "V0")),
        ("socket_pickup", "socket pickup", "S0 − K_rx_req", ("S0", "K_rx_req")),
        ("pre_gpu", "request read to the copy's first event", "S1 − S0", ("S1", "S0")),
        ("gpu_copy", "GPU span: the copy (device events)", "G", ("G", None)),
        ("gpu_overhead", "GPU span: launch and query overhead", "S2 − S1 − G", ("S2", "S1", "G")),
        ("header_to_send", "copy done to the first send", "S3 − S2", ("S3", "S2")),
        ("send", "first send to the last byte's kernel TX stamp", "K_tx_resp − S3", ("K_tx_resp", "S3")),
        ("delivery", "server kernel TX to the verifier's last byte", "V1 − K_tx_resp", ("V1", "K_tx_resp")),
    ),
    "ping": (
        ("request", "verifier send to server kernel RX", "K_rx_req − V0", ("K_rx_req", "V0")),
        ("socket_pickup", "socket pickup", "S0 − K_rx_req", ("S0", "K_rx_req")),
        ("header_to_send", "request read to the first send (no device work)", "S3 − S0", ("S3", "S0")),
        ("send", "first send to the last byte's kernel TX stamp", "K_tx_resp − S3", ("K_tx_resp", "S3")),
        ("delivery", "server kernel TX to the verifier's last byte", "V1 − K_tx_resp", ("V1", "K_tx_resp")),
    ),
}
STAMP_COLS = {"t_sched": "t_sched", "V0": "t_send0", "V1": "t_recv", "K_rx_req": "srv_krx", "S0": "srv_urx", "S1": "srv_s1",
              "S2": "srv_s2", "S3": "srv_utx", "G": "device_ns"}

ATTRIBUTION_RULE = (
    "stall attribution, first rule that holds: 'steal' if the hypervisor stole at least half the answer's elapsed time on the "
    "responder's or verifier's core over the 100 ms samples spanning it; 'vllm-step' if the GPU span (S2 − S1) is at least half "
    "the elapsed and a vLLM step's GPU interval overlaps the answer; 'gpu' if the GPU span is at least half the elapsed with no "
    "step overlapping; 'socket' if the kernel and socket parts (request + socket_pickup + send + delivery) are at least half "
    "the elapsed; else 'unknown'.  Twins (timestamping off) have no socket parts, so they are never 'socket'")


def _dur(us: float) -> str:
    return f"{us / 1000:.1f} ms" if us >= 1000 else f"{us:.0f} µs"


SOCKET_LABELS = {
    "send": "send stall on loopback: TCP send took {x}",
    "delivery": "delivery stall on loopback: the last byte reached the verifier {x} after the server's TX stamp",
    "socket_pickup": "pickup stall: the responder read the request {x} after its kernel RX stamp",
    "request": "request stall on loopback: {x} from the verifier's send to the server's kernel RX",
}


def stall_label(attr: str, comps: dict[str, float], gpu_us: float, elapsed_us: float, steal_role: str,
                steal_ms: float | None, server_user_us: float | None) -> str:
    """A short human label for a stall's attribution, naming its largest part."""
    if attr == "steal":
        return f"hypervisor steal on the {steal_role}'s core: {steal_ms:.0f} ms stolen"
    if attr == "vllm-step":
        return f"GPU copy behind a vLLM step: the GPU span took {_dur(gpu_us)}"
    if attr == "gpu":
        return f"GPU span took {_dur(gpu_us)} with no vLLM step running"
    if attr == "socket":
        k = max(SOCKET_LABELS, key=lambda c: comps.get(c) or 0.0)
        return SOCKET_LABELS[k].format(x=_dur(comps[k]))
    if "request" not in comps:
        su = f", server user time {_dur(server_user_us)}" if server_user_us is not None else ""
        return f"twin without kernel stamps: elapsed {_dur(elapsed_us)}{su}"
    k = max(comps, key=lambda c: comps[c])
    if "send" not in comps:
        return (f"no kernel TX stamp: the largest stamped part, {k.replace('_', ' ')}, took {_dur(comps[k])} of "
                f"{_dur(elapsed_us)}")
    return f"no single cause: the largest part, {k.replace('_', ' ')}, took {_dur(comps[k])} of {_dur(elapsed_us)}"


def survival_at(t_us: list[float] | tuple[float, ...], s: list[float], max_us: float, n: int, t: float) -> float:
    """S(t) from a survival row of `n` answers: 1 below the grid, ln s linear in ln t between grid points, 0 past `max_us`.
    s ≥ 1/n at every grid point up to `max_us`, so the only segment with a zero end straddles it (t_i ≤ max_us < t_{i+1}):
    there ln s runs linearly in ln t from (t_i, s_i) to (max_us, 1/n), and S(max_us) = 1/n exactly.  Past the grid's last
    point the same holds from that point to (max_us, 1/n)."""
    if t > max_us:
        return 0.0
    if t < t_us[0]:
        return 1.0
    if t == max_us:
        return 1.0 / n
    j = bisect.bisect_right(t_us, t) - 1
    if t == t_us[j]:
        return float(s[j])
    if j + 1 < len(t_us) and s[j + 1] > 0:
        x1, y1 = math.log(t_us[j + 1]), math.log(s[j + 1])
    else:
        x1, y1 = math.log(max_us), math.log(1.0 / n)
    x0, y0 = math.log(t_us[j]), math.log(s[j])
    return math.exp(y0 + (y1 - y0) * (math.log(t) - x0) / (x1 - x0))


def call_ns(block_bytes: int) -> int:
    """The adversary's fastest `Π₂` call on `block_bytes` labels (`verity_pous.primitives.pi2_call_ns`, in bits)."""
    from verity_pous.primitives import pi2_call_ns

    return pi2_call_ns(8 * block_bytes)


def tau_us(d: int, call: float, m: float = 1.0) -> float:
    """τ(m) = (D + 1) · call_ns / m / 1000 − 0.001 µs with D = min(d, 5)."""
    return (min(d, 5) + 1) * call / m / 1000 - 0.001


def tor_terms(metric: str, response_bytes: int) -> dict[str, float]:
    """Contract §3's modeled terms (µs) of one ToR variant for a response of `response_bytes` (header + block)."""
    link, setting = TOR[metric]
    frames = -(-response_bytes // MSS)
    wire = response_bytes + FRAME_OVERHEAD * frames
    nic_rx = NIC_RX_BASE_US + (RX_USECS_DEFAULT if setting == "default" else 0.0)
    tx = TX_FIXED_US + max(response_bytes / B_PCIE, wire * 8 / link) * 1e6
    req = REQ_FRAME * 8 / link * 1e6
    prop = PROP_NS_PER_M * CABLE_M / 1000
    return {"t_nic_rx_us": nic_rx, "t_nic_tx_and_wire_us": tx, "t_req_ser_us": req, "t_prop_us": prop,
            "constant_us": nic_rx + tx + req + 2 * prop}


def tor_params(metric: str, header_bytes: int = HEADER_BYTES) -> dict[str, Any]:
    link, setting = TOR[metric]
    per = {str(b): tor_terms(metric, header_bytes + b) for b in BLOCK_BYTES}
    return {"link_bps": link, "setting": setting, "nic": "mlx5_core", "mtu": MTU, "mss_bytes": MSS,
            "frame_overhead_bytes": FRAME_OVERHEAD, "header_bytes": header_bytes, "b_pcie_bytes_per_s": B_PCIE,
            "rx_usecs": RX_USECS_DEFAULT if setting == "default" else 0.0, "nic_rx_base_us": NIC_RX_BASE_US,
            "tx_fixed_us": TX_FIXED_US, "request_frame_bytes": REQ_FRAME, "prop_ns_per_m": PROP_NS_PER_M, "cable_m": CABLE_M,
            "t_nic_rx_us": per[str(BLOCK_BYTES[0])]["t_nic_rx_us"], "t_req_ser_us": per[str(BLOCK_BYTES[0])]["t_req_ser_us"],
            "t_prop_us": per[str(BLOCK_BYTES[0])]["t_prop_us"],
            "t_nic_tx_and_wire_us_by_block": {b: v["t_nic_tx_and_wire_us"] for b, v in per.items()},
            "constant_us_by_block": {b: v["constant_us"] for b, v in per.items()}}


def metrics_table(header_bytes: int = HEADER_BYTES) -> list[dict[str, Any]]:
    out = [{"id": k, "label": lab, "kind": "measured", "formula": f, "params": {"needs_kernel_stamps": kern}}
           for k, (lab, f, kern) in MEASURED.items()]
    for k, (link, setting) in TOR.items():
        out.append({"id": k, "label": f"modeled top-of-rack round trip, {link / 1e9:.0f} Gb/s, {setting} NIC", "kind": "modeled",
                    "formula": "R + t_nic_rx + t_nic_tx_and_wire + t_req_ser + 2·t_prop, R = K_tx_resp − K_rx_req; "
                               "t_nic_tx_and_wire = tx_fixed + max(bytes / B_pcie, wire_bytes · 8 / L), wire_bytes = bytes + "
                               "90 · ⌈bytes / MSS⌉, bytes = header + block",
                    "params": tor_params(k, header_bytes)})
    return out


class Acc:
    """One row's exact count, sum, min, max and survival counts, and its log histogram for quantiles."""

    __slots__ = ("ge", "hi", "hist", "lo", "max", "min", "n", "sum")

    def __init__(self) -> None:
        self.n, self.sum, self.min, self.max = 0, 0.0, math.inf, -math.inf
        self.ge = np.zeros(len(T_US), np.int64)
        self.hist = np.zeros(N_BINS, np.int64)
        self.lo: list[np.ndarray] = []
        self.hi: list[np.ndarray] = []

    def add(self, v: np.ndarray) -> None:
        if not len(v):
            return
        v = np.sort(v)
        self.n += len(v)
        self.sum += float(v.sum())
        self.min, self.max = min(self.min, float(v[0])), max(self.max, float(v[-1]))
        self.ge += len(v) - np.searchsorted(v, np.asarray(T_US), side="left")
        a, b = np.searchsorted(v, HIST_LO_US, side="left"), np.searchsorted(v, HIST_HI_US, side="left")
        if a:
            self.lo.append(v[:a])
        if b < len(v):
            self.hi.append(v[b:])
        mid = v[a:b]
        if len(mid):
            k = np.clip(np.floor(np.log10(mid / HIST_LO_US) * BINS_PER_DECADE).astype(np.int64), 0, N_BINS - 1)
            self.hist += np.bincount(k, minlength=N_BINS)

    def quantile(self, p: float) -> float:
        """The nearest-rank order statistic of rank ⌈p·n⌉ (exact below and above the histogram, else its bin's centre)."""
        k = max(1, math.ceil(p * self.n - 1e-9))
        lo = np.sort(np.concatenate(self.lo)) if self.lo else np.zeros(0)
        if k <= len(lo):
            return float(lo[k - 1])
        k -= len(lo)
        cum = np.cumsum(self.hist)
        if k <= cum[-1]:
            i = int(np.searchsorted(cum, k, side="left"))
            centre = HIST_LO_US * 10 ** ((i + 0.5) / BINS_PER_DECADE)
            return min(max(centre, self.min), self.max)
        hi = np.sort(np.concatenate(self.hi))
        return float(hi[k - int(cum[-1]) - 1])

    def quantiles(self) -> dict[str, float | None]:
        out: dict[str, float | None] = {}
        for name, p in QUANTILES:
            out[name] = None if (name == "p9999_us" and self.n < 10_000) or not self.n else _r(self.quantile(p))
        out["max_us"] = self.max if self.n else None
        return out


def _r(x: float | None, sig: int = 7) -> float | None:
    if x is None or not math.isfinite(x):
        return None
    return float(f"{x:.{sig}g}")


def _utc(ns: int, ms: bool = False) -> str:
    t = datetime.fromtimestamp(ns / 1e9, UTC)
    return t.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z" if ms else t.strftime("%Y-%m-%dT%H:%M:%SZ")


def _nearest_rank(v: np.ndarray, p: float) -> float:
    s = np.sort(v)
    return float(s[max(1, math.ceil(p * len(s) - 1e-9)) - 1])


class Chunk:
    """One chunk's answers with their stamps, metric values and masks; the npz stays open only while it is processed."""

    def __init__(self, path: Path) -> None:
        z = np.load(path, allow_pickle=False)
        self.meta = json.loads(str(z["meta_json"]))
        self.files = set(z.files)
        rec = z["rec"]
        col = {str(n): i for i, n in enumerate(z["rec_columns"])}
        self.header_bytes = HEADER_BYTES_V2 if "srv_zc_wait_ns" in col else HEADER_BYTES
        self.runs = self.meta["runs"]
        self.run = z["rec_run"].astype(np.int64)
        self.cell = z["rec_cell"].astype(np.int64)
        self.op = rec[:, col["op"]]
        self.nbytes = rec[:, col["nbytes"]]
        st = {k: rec[:, col[c]] for k, c in STAMP_COLS.items()}
        st["K_tx_resp"], st["S4"] = z["rec_k_tx_resp"], z["rec_s4"]
        self.st = st
        ts = np.array([bool(r.get("ts", True)) for r in self.runs], bool)
        t0 = np.array([int(r.get("t0_ns") or r.get("start_ns") or 0) for r in self.runs], np.int64)
        self.ts = ts[self.run] if len(self.runs) else np.zeros(len(rec), bool)
        ok = rec[:, col["status"]] == 0
        self.warm_dropped = int(np.count_nonzero(ok & (st["t_sched"] < t0[self.run] + WARMUP_NS)))
        self.keep = ok & (st["t_sched"] >= t0[self.run] + WARMUP_NS) & np.isin(self.op, list(OPS))
        self.status_nonzero = int(np.count_nonzero(~ok))
        self.steps = z["steps"] if "steps" in z.files else np.zeros((0, 8), np.int64)
        self.samples = {k[len("sample_"):]: z[k] for k in z.files if k.startswith("sample_")}
        z.close()

    def diff(self, a: str, b: str | None, c: str | None = None) -> np.ndarray:
        """`a − b (− c)` in µs where every stamp is present (> 0; "G" is a duration), NaN elsewhere."""
        st = self.st
        if a == "G" and b is None:
            return np.where(st["G"] > 0, st["G"] / 1e3, np.nan)
        out = (st[a] - st[b]).astype(np.float64) / 1e3
        good = (st[a] > 0) & (st[b] > 0)
        if c is not None:
            out = out - st[c] / 1e3
            good &= st[c] > 0
        return np.where(good, out, np.nan)

    def values(self) -> dict[str, np.ndarray]:
        v = {"elapsed": self.diff("V1", "V0"), "server_user": self.diff("S4", "S0"),
             "server_kernel": np.where(self.ts, self.diff("K_tx_resp", "K_rx_req"), np.nan),
             "gpu_span": np.where(self.op == OP_READ, self.diff("S2", "S1"), np.nan), "schedule_lag": self.diff("V0", "t_sched")}
        for k in TOR:
            const = np.zeros(len(self.op))
            for b in np.unique(self.nbytes):
                const[self.nbytes == b] = tor_terms(k, self.header_bytes + int(b))["constant_us"]
            v[k] = v["server_kernel"] + const
        return v

    def components(self, op: str) -> dict[str, np.ndarray]:
        return {cid: self.diff(*stamps) for cid, _, _, stamps in COMPONENTS[op]}


class Builder:
    def __init__(self) -> None:
        self.surv: dict[tuple[int, str, str, bool], Acc] = defaultdict(Acc)
        self.brk: dict[tuple[int, str, str], Acc] = defaultdict(Acc)
        self.cells_meta: dict[int, dict[str, Any]] = {}
        self.cell_counts: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.runs_out: list[dict[str, Any]] = []
        self.minutes: dict[int, dict[str, Any]] = {}
        self.serving: dict[Any, dict[str, Any]] = defaultdict(lambda: {"tokens": 0, "steps": 0, "seconds": 0.0, "windows": 0,
                                                                        "gpu_ms": []})
        self.free_checked: dict[str, dict[str, int]] = {m: defaultdict(int) for m in LOADED}
        self.prev_run: tuple[Any, int, str | None] | None = None
        self.stalls: list[tuple[float, int, dict[str, Any]]] = []
        self.calls = {b: call_ns(b) for b in BLOCK_BYTES}
        self.taus = {f"{b}/d{d}": tau_us(d, self.calls[b]) for b in BLOCK_BYTES for d in ROUNDS}
        self.n_answers = self.n_kept = self.n_warm = self.wrong = self.status_nonzero = 0
        self.no_ktx = 0
        self.no_nvml_chunks: list[str] = []
        self.chunk_meta: list[dict[str, Any]] = []
        self.seed: str | None = None
        self.stall_seq = 0
        self.t_first: int | None = None
        self.t_last: int | None = None
        self.header_bytes: int | None = None
        self.cpus: tuple[int, ...] = ()

    def add(self, path: Path) -> None:
        c = Chunk(path)
        m = c.meta
        if self.header_bytes not in (None, c.header_bytes):
            raise SystemExit(f"{path.name}: a {c.header_bytes} B header among {self.header_bytes} B chunks")
        self.header_bytes = c.header_bytes
        cpus = tuple(int(x) for x in c.samples.get("cpus", ()))
        if cpus and self.cpus and cpus != self.cpus:
            raise SystemExit(f"{path.name}: sampled cores {cpus[0]}-{cpus[-1]}, not {self.cpus[0]}-{self.cpus[-1]}")
        self.cpus = self.cpus or cpus
        self.chunk_meta.append({k: m.get(k) for k in ("job", "chunk_index", "chunk_start_utc", "end_utc", "chunk_start_ns", "env")})
        self.seed = self.seed or m.get("schedule_seed")
        for cell in m.get("cells", []):
            if cell["config"] not in CONFIGS:
                raise SystemExit(f"{path.name}: cell {cell['id']} has config {cell['config']!r}, not one of {CONFIGS}")
            self.cells_meta[int(cell["id"])] = cell
        self.n_answers += len(c.op)
        self.n_kept += int(np.count_nonzero(c.keep))
        self.n_warm += c.warm_dropped
        self.wrong += int(m.get("wrong") or 0)
        self.status_nonzero += c.status_nonzero
        vals = c.values()
        self.no_ktx += int(np.count_nonzero(c.keep & c.ts & np.isnan(vals["server_kernel"])))
        self._cells(c, vals)
        self._runs(c, vals)
        self._minutes(c, path.name)
        self._serving(c)
        self._stalls(c, vals)

    def _groups(self, c: Chunk, mask: np.ndarray, key: np.ndarray):
        idx = np.flatnonzero(mask)
        order = idx[np.argsort(key[idx], kind="stable")]
        ks, starts = np.unique(key[order], return_index=True)
        for i, k in enumerate(ks):
            yield int(k), order[starts[i]:starts[i + 1] if i + 1 < len(starts) else len(order)]

    def _cells(self, c: Chunk, vals: dict[str, np.ndarray]) -> None:
        opi = np.where(c.op == OP_READ, 1, 0)
        key = c.cell * 4 + opi * 2 + c.ts.astype(np.int64)
        for k, rows in self._groups(c, c.keep, key):
            cell, op, ts = k // 4, "challenge" if (k // 2) % 2 else "ping", bool(k % 2)
            cc = self.cell_counts[cell]
            cc["answers" if ts else "twin_answers"] += len(rows)
            if ts:
                cc["challenges" if op == "challenge" else "pings"] += len(rows)
            for metric, v in vals.items():
                if (not ts and metric not in MEASURED) or (not ts and MEASURED[metric][2]):
                    continue
                if op == "ping" and metric == "gpu_span":
                    continue
                x = v[rows]
                x = x[np.isfinite(x)]
                if len(x):
                    self.surv[(cell, metric, op, ts)].add(x)
        for op_code, op in OPS.items():
            comps = c.components(op)
            for cell, rows in self._groups(c, c.keep & c.ts & (c.op == op_code), c.cell):
                for cid, v in comps.items():
                    x = v[rows]
                    x = x[np.isfinite(x)]
                    if len(x):
                        self.brk[(cell, op, cid)].add(x)
        for r in c.runs:
            if "error" in r:
                continue
            a, b = int(r.get("start_ns") or 0), int(r.get("end_ns") or 0)
            if a:
                self.t_first = a if self.t_first is None else min(self.t_first, a)
            if b:
                self.t_last = b if self.t_last is None else max(self.t_last, b)
            cc = self.cell_counts[int(r["cell"])]
            cc["runs" if r.get("ts", True) else "twin_runs"] += 1

    def _runs(self, c: Chunk, vals: dict[str, np.ndarray]) -> None:
        sel = c.keep & (c.op == OP_READ)
        rows_by_run = dict(self._groups(c, sel, c.run))
        for j, r in enumerate(c.runs):
            if "error" in r:
                continue
            rows = rows_by_run.get(j, np.zeros(0, np.int64))
            ts = bool(r.get("ts", True))
            out = {"run": len(self.runs_out), "cell": int(r["cell"]), "pass": int(r.get("pass", 0)),
                   "start_utc": r.get("start_utc") or _utc(int(r["start_ns"])), "timestamping": ts, "n": len(rows),
                   "p50_us": {}, "p999_us": {}, "max_us": {}, "over_tau": {}}
            for metric, v in vals.items():
                if not ts and (metric not in MEASURED or MEASURED[metric][2]):
                    continue
                x = v[rows]
                x = x[np.isfinite(x)]
                if not len(x):
                    continue
                out["p50_us"][metric] = _r(_nearest_rank(x, 0.5))
                out["p999_us"][metric] = _r(_nearest_rank(x, 0.999))
                out["max_us"][metric] = _r(float(x.max()))
                if metric in ("elapsed", "tor_400_tuned"):
                    out["over_tau"][metric] = {k: int(np.count_nonzero(x > t)) for k, t in self.taus.items()}
            self.runs_out.append(out)

    def _minutes(self, c: Chunk, name: str) -> None:
        s = c.samples
        if "t_ns" not in s or len(s["t_ns"]) < 2:
            return
        t = s["t_ns"]
        dt = np.diff(t) / 1e9
        minute = (t[1:] // 60_000_000_000).astype(np.int64)
        cpus = list(s["cpus"])
        last = s["lastcpu"][1:] if "lastcpu" in s else None
        steal = np.diff(s["cpustat"][:, :, 7], axis=0) if "cpustat" in s else None
        soft = np.diff(s["softirq"], axis=0).sum(axis=(1, 2)) if "softirq" in s else None
        ctxt = np.diff(s["thread_ctxt"], axis=0).sum(axis=(1, 2)) if "thread_ctxt" in s else None
        clock = s.get("clock_triple")
        off = clock[:, 1] - (clock[:, 0] + clock[:, 2]) // 2 if clock is not None else None
        nv = s.get("nvml")
        nv_ok = nv is not None and len(nv) >= 2
        gpus_ok = "gpus_t_ns" in s and len(s["gpus_t_ns"]) >= 2
        if not nv_ok:
            self.no_nvml_chunks.append(name)
        for mnt in np.unique(minute):
            sel = minute == mnt
            mm = self.minutes.setdefault(int(mnt), {"dt": 0.0, "steal_v": 0, "steal_r": 0, "steal_other": defaultdict(int),
                                                    "soft": 0, "ctxt": 0, "off_min": None, "off_max": None,
                                                    "nvml": [], "gpus": [], "load": []})
            mm["dt"] += float(dt[sel].sum())
            if steal is not None and last is not None:
                ix = np.flatnonzero(sel)
                pos = {cpu: i for i, cpu in enumerate(cpus)}
                for i in ix:
                    r_cpu, v_cpu = int(last[i, TARGET_RESPONDER]), int(last[i, TARGET_VERIFIER])
                    if r_cpu in pos:
                        mm["steal_r"] += int(steal[i, pos[r_cpu]])
                    if v_cpu in pos:
                        mm["steal_v"] += int(steal[i, pos[v_cpu]])
                    for cpu, p in pos.items():
                        if cpu not in (r_cpu, v_cpu):
                            mm["steal_other"][cpu] += int(steal[i, p])
            if soft is not None:
                mm["soft"] += int(soft[sel].sum())
            if ctxt is not None:
                mm["ctxt"] += int(ctxt[sel].sum())
            if off is not None:
                o = off[1:][sel]
                mm["off_min"] = int(o.min()) if mm["off_min"] is None else min(mm["off_min"], int(o.min()))
                mm["off_max"] = int(o.max()) if mm["off_max"] is None else max(mm["off_max"], int(o.max()))
        if nv_ok:
            fields = [str(f) for f in s["nvml_fields"]]
            f = {k: fields.index(k) for k in ("t_ns", "sm_util", "power_mw", "temp_c")}
            for row in nv:
                mm = self.minutes.get(int(row[f["t_ns"]] // 60_000_000_000))
                if mm is not None:
                    mm["nvml"].append((row[f["sm_util"]], row[f["power_mw"]] / 1000, row[f["temp_c"]]))
        if gpus_ok:
            uu = [str(u) for u in s["gpus_uuid"]]
            mine = uu.index(c.meta.get("env", {}).get("gpu_uuid")) if c.meta.get("env", {}).get("gpu_uuid") in uu else 7
            others = [i for i in range(len(uu)) if i != mine]
            for tg, util, load in zip(s["gpus_t_ns"], s["gpus_util"], s["loadavg"], strict=False):
                mm = self.minutes.get(int(tg // 60_000_000_000))
                if mm is not None:
                    mm["gpus"].append(int(util[others].max()))
                    mm["load"].append(float(load[0]))

    def _serving(self, c: Chunk) -> None:
        """Per cell, vLLM's steps in each run's measured window; per loaded load, its steps in the windows with no challenge
        traffic: from the previous run's end to this run's first arrival (the gap and the 2 s settle) when the previous run
        was on the same server (job) and this run did not switch the load, so the load was already running."""
        st = c.steps
        if len(st):
            st = st[np.argsort(st[:, 2], kind="stable")]
        cover = int(c.meta["chunk_start_ns"]) - 5_000_000_000 if c.meta.get("chunk_start_ns") else None
        job = c.meta.get("job")

        def take(sv: dict[str, Any], a: int, b: int) -> None:
            sv["seconds"] += (b - a) / 1e9
            sv["windows"] += 1
            if len(st):
                i, j = np.searchsorted(st[:, 2], a, side="left"), np.searchsorted(st[:, 2], b, side="left")
                w = st[i:j]
                sv["tokens"] += int(w[:, 4].sum())
                sv["steps"] += len(w)
                sv["gpu_ms"].extend(((w[:, 3] - w[:, 2]) / 1e6).tolist())

        for r in c.runs:
            if "error" in r or not r.get("t0_ns") or not r.get("end_ns"):
                self.prev_run = None
                continue
            a, b = int(r["t0_ns"]) + WARMUP_NS, int(r["end_ns"])
            if b > a:
                take(self.serving[int(r["cell"])], a, b)
            load = r.get("load") or {}
            mode = load.get("mode")
            if mode in LOADED:
                chk = self.free_checked[mode]
                prev = self.prev_run
                if prev is None or prev[0] != job:
                    chk["first_on_server"] += 1
                elif load.get("switched", True) or prev[2] != mode:
                    chk["switched"] += 1
                else:
                    lo, hi = max(prev[1], cover) if cover is not None else prev[1], int(r["t0_ns"])
                    if hi > lo:
                        take(self.serving[("free", mode)], lo, hi)
                        chk["used"] += 1
            self.prev_run = (job, int(r["end_ns"]), mode)

    def _stalls(self, c: Chunk, vals: dict[str, np.ndarray]) -> None:
        sel = np.flatnonzero(c.keep & (c.op == OP_READ) & np.isfinite(vals["elapsed"]))
        if not len(sel):
            return
        el = vals["elapsed"][sel]
        top = sel[np.argsort(-el, kind="stable")[:TOP_STALLS]]
        comps = c.components("challenge")
        s = c.samples
        t = s.get("t_ns")
        st = c.steps[np.argsort(c.steps[:, 2], kind="stable")] if len(c.steps) else c.steps
        cpus = list(s["cpus"]) if "cpus" in s else []
        for i in top:
            e = float(vals["elapsed"][i])
            if len(self.stalls) >= TOP_STALLS and e <= self.stalls[0][0]:
                continue
            v0, v1 = int(c.st["V0"][i]), int(c.st["V1"][i])
            steal_ms = softirq = None
            steal_role = "responder"
            if t is not None and len(t) >= 2 and "cpustat" in s and "lastcpu" in s:
                a = max(int(np.searchsorted(t, v0, side="right")) - 1, 0)
                b = min(int(np.searchsorted(t, v1, side="left")), len(t) - 1)
                b = max(b, a + 1) if a + 1 < len(t) else b
                d = s["cpustat"][b, :, 7] - s["cpustat"][a, :, 7]
                by_role = {role: float(d[cpus.index(core)] * 1000 / USER_HZ)
                           for role, core in (("responder", int(s["lastcpu"][b, TARGET_RESPONDER])),
                                              ("verifier", int(s["lastcpu"][b, TARGET_VERIFIER]))) if core in cpus}
                if by_role:
                    steal_role = max(by_role, key=lambda k: by_role[k])
                    steal_ms = by_role[steal_role]
                if "softirq" in s:
                    softirq = int((s["softirq"][b] - s["softirq"][a]).sum())
            overlap = 0.0
            if len(st):
                k = int(np.searchsorted(st[:, 2], v1, side="left"))
                w = st[max(k - 64, 0):k]
                ov = np.clip(np.minimum(w[:, 3], v1) - np.maximum(w[:, 2], v0), 0, None).sum()
                overlap = float(min(ov / max(v1 - v0, 1), 1.0))
            cv = {cid: _r(float(v[i]), 6) for cid, v in comps.items() if np.isfinite(v[i])}
            gpu = float(vals["gpu_span"][i]) if np.isfinite(vals["gpu_span"][i]) else 0.0
            sock = [cv.get(k) for k in ("request", "socket_pickup", "send", "delivery")]
            if steal_ms is not None and steal_ms * 1000 >= e / 2:
                attr = "steal"
            elif gpu >= e / 2:
                attr = "vllm-step" if overlap > 0 else "gpu"
            elif all(x is not None for x in sock) and sum(sock) >= e / 2:
                attr = "socket"
            else:
                attr = "unknown"
            sk = vals["server_kernel"][i]
            su = vals["server_user"][i]
            label = stall_label(attr, cv, gpu, e, steal_role, steal_ms, float(su) if np.isfinite(su) else None)
            row = {"utc": _utc(v0, ms=True), "cell": int(c.cell[i]), "op": "challenge", "elapsed_us": _r(e),
                   "server_kernel_us": _r(float(sk)) if np.isfinite(sk) else None, "components": cv, "steal_ms": steal_ms,
                   "vllm_step_overlap": _r(overlap, 4), "softirq": softirq, "attribution": attr,
                   "attribution_label": label}
            self.stall_seq += 1
            item = (e, -self.stall_seq, row)
            if len(self.stalls) < TOP_STALLS:
                heapq.heappush(self.stalls, item)
            else:
                heapq.heappushpop(self.stalls, item)

    def document(self, *, arts: list[str], commit: str, generated_at: str, host: str, gpu: str,
                 extra_notes: list[str]) -> dict[str, Any]:
        cells = []
        for cid in sorted(self.cells_meta):
            m, cc = self.cells_meta[cid], self.cell_counts.get(cid, {})
            cells.append({"cell": cid, "load": m["load"], "rate_hz": m["rate"], "block_bytes": m["kib"] * 1024,
                          "config": m["config"], **{k: int(cc.get(k, 0)) for k in
                                                    ("runs", "answers", "challenges", "pings", "twin_runs", "twin_answers")}})
        survival, quantiles = [], []
        for (cell, metric, op, ts), acc in sorted(self.surv.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2], not kv[0][3])):
            survival.append({"cell": cell, "metric": metric, "op": op, "timestamping": ts, "n": acc.n, "max_us": acc.max,
                             "s": [_r(g / acc.n) for g in acc.ge.tolist()]})
            quantiles.append({"cell": cell, "metric": metric, "op": op, "timestamping": ts, "n": acc.n, **acc.quantiles()})
        breakdown = []
        for cell in sorted({k[0] for k in self.brk}):
            for op in OPS.values():
                comps = []
                for cid, label, formula, _ in COMPONENTS[op]:
                    acc = self.brk.get((cell, op, cid))
                    if acc is None or not acc.n:
                        continue
                    comps.append({"id": cid, "label": label, "formula": formula, "p50_us": _r(acc.quantile(0.5)),
                                  "p999_us": _r(acc.quantile(0.999)), "mean_us": _r(acc.sum / acc.n)})
                if comps:
                    breakdown.append({"cell": cell, "op": op, "components": comps})
        minutes = []
        for mnt in sorted(self.minutes):
            mm = self.minutes[mnt]
            dt = mm["dt"] or None

            def per_s(x: float, dt: float | None = dt) -> float | None:
                return _r(x / dt, 6) if dt else None

            jiffy_ms = 1000 / USER_HZ
            nv = np.array(mm["nvml"], float) if mm["nvml"] else None
            minutes.append({
                "utc": _utc(mnt * 60_000_000_000),
                "steal_ms_per_s": {"verifier": per_s(mm["steal_v"] * jiffy_ms), "responder": per_s(mm["steal_r"] * jiffy_ms),
                                   "max_other": per_s(max(mm["steal_other"].values(), default=0) * jiffy_ms)},
                "softirq_per_s": per_s(mm["soft"]), "ctx_switch_per_s": per_s(mm["ctxt"]),
                "gpu7_sm_util_pct": _r(float(nv[:, 0].mean()), 5) if nv is not None else None,
                "gpu7_power_w": _r(float(nv[:, 1].mean()), 5) if nv is not None else None,
                "gpu7_temp_c": _r(float(nv[:, 2].mean()), 5) if nv is not None else None,
                "other_gpus_util_pct_max": max(mm["gpus"]) if mm["gpus"] else None,
                "loadavg1": _r(float(np.mean(mm["load"])), 5) if mm["load"] else None,
                "clock_offset_range_ns": (mm["off_max"] - mm["off_min"]) if mm["off_min"] is not None else None})
        serving = []
        keys = [(cid, self.cells_meta[cid]) for cid in sorted(k for k in self.serving if isinstance(k, int))]
        keys += [(("free", m), {"id": None, "load": m, "rate": 0, "kib": None, "config": None}) for m in LOADED]
        for key, m in keys:
            sv = self.serving.get(key) or {"tokens": 0, "steps": 0, "seconds": 0.0, "gpu_ms": []}
            g = np.array(sv["gpu_ms"], float)
            serving.append({"cell": m["id"], "load": m["load"], "rate_hz": m["rate"],
                            "block_bytes": m["kib"] * 1024 if m["kib"] else None, "config": m["config"],
                            "tokens_per_s": _r(sv["tokens"] / sv["seconds"], 6) if sv["seconds"] else None,
                            "steps_per_s": _r(sv["steps"] / sv["seconds"], 6) if sv["seconds"] else None,
                            "step_gpu_ms_p50": _r(_nearest_rank(g, 0.5), 6) if len(g) else None,
                            "step_gpu_ms_p99": _r(_nearest_rank(g, 0.99), 6) if len(g) else None})
        stalls = [{"rank": k + 1, **row} for k, (_, _, row) in enumerate(sorted(self.stalls, key=lambda x: (-x[0], -x[1])))]
        doc = {
            "schema": SCHEMA, "generated_at_utc": generated_at, "source_commit": commit,
            "run": {"host": host, "gpu": gpu,
                    "start_utc": _utc(self.t_first) if self.t_first else None,
                    "end_utc": _utc(self.t_last) if self.t_last else None,
                    "chunks": arts, "answers": self.n_answers, "wrong": self.wrong, "schedule_seed": self.seed,
                    "notes": self.notes(extra_notes, survival)},
            "cells": cells, "metrics": metrics_table(self.header_bytes or HEADER_BYTES), "t_us": [_r(t, 10) for t in T_US], "survival": survival,
            "quantiles": quantiles,
            "thresholds": [{"block_bytes": b, "call_ns": self.calls[b],
                            "rounds": [{"d": d, "D": min(d, 5), "tau_us_at_m1": _r(tau_us(d, self.calls[b]), 10)} for d in ROUNDS]}
                           for b in BLOCK_BYTES],
            "breakdown": breakdown, "timeline": {"runs": self.runs_out, "minutes": minutes}, "serving": serving,
            "stalls": stalls,
        }
        doc["vectors"] = vectors(doc)
        return doc

    def free_note(self) -> str:
        parts = []
        for m in LOADED:
            sv, chk = self.serving.get(("free", m)), self.free_checked[m]
            got = (f"{sv['windows']} windows, {sv['seconds']:.1f} s, {sv['steps']} steps" if sv and sv["seconds"]
                   else "no window, so null")
            parts.append(f"{m}: {got} (of {sum(chk.values())} {m} runs: {chk['used']} used, {chk['switched']} switched the load "
                         f"at their start, {chk['first_on_server']} first on their server)")
        return ("serving's no-challenge rows (cell null, rate_hz 0): vLLM steps whose GPU start falls between a run's end and the "
                "next run's first arrival (the ~1 ms gap and the 2 s settle) on the same server, when the next run kept the "
                "load (load.switched false), so the load ran throughout with no challenge or ping in flight; a run that "
                "switched the load is not used, its new load starting inside the settle; " + "; ".join(parts))

    def notes(self, extra: list[str], survival: list[dict[str, Any]]) -> list[str]:
        out = list(extra)
        cores = f"{self.cpus[0]}-{self.cpus[-1]}" if self.cpus else "80-93"
        header = self.header_bytes or HEADER_BYTES
        node0 = not self.cpus or all(c in NODE0_CPUS for c in self.cpus)
        out += [
            ("NVML (GPU 7's SM utilisation, power, temperature), the 10 s GPU utilisations and loadavg are missing before "
             "07:58Z: the sampler kept only their first row per chunk (fixed in 3ab25fa4); chunks without them: "
             + ", ".join(self.no_nvml_chunks) + "; those minutes carry null") if self.no_nvml_chunks else
            "NVML (GPU 7's SM utilisation, power, temperature), the 10 s GPU utilisations and loadavg: in every chunk",
            f"cross-socket D2H: GPU 7 is on NUMA node 1, the soak's cores {cores} on node 0, so every block copy crosses sockets"
            if node0 else f"same-socket D2H: GPU 7 and the cores {cores} are on NUMA node 1, so no block copy crosses sockets",
            "node 2 is a KVM VM: steal time is the hypervisor's, read from /proc/stat; no CPU frequency (cpu_freq_khz is -1)",
            ("loopback biases R (server_kernel, and every ToR estimate built on it) upward: the server's xmit of each response "
            "skb runs the verifier's TCP receive softirq inline (contract §3)"),
            f"warm-up: the first {WARMUP_NS / 1e9:g} s of each cell run (answers scheduled before the run's first arrival + "
            f"{WARMUP_NS / 1e9:g} s) is left out of cells' counts, survival, quantiles, breakdown, timeline and stalls: "
            f"{self.n_warm:,} answers; {self.n_kept:,} of {self.n_answers:,} answers remain"
            + (f"; {self.status_nonzero:,} with a nonzero status are left out too" if self.status_nonzero else ""),
            f"answers with no K_tx_resp (timestamping on) are left out of server_kernel and the ToR metrics: {self.no_ktx:,}",
            ("twins (cells run with timestamping off) appear as timestamping: false rows for elapsed, server_user, gpu_span and "
            "schedule_lag only; breakdown and stalls' components use timestamping-on answers"),
            ("quantiles: nearest-rank order statistics (rank ⌈p·n⌉) from log histograms of 1000 bins per decade (bin width 0.23 % "
            "of the value), each the bin's geometric centre clamped to the exact min and max (within 0.12 % of the exact value); "
            "values under 0.01 µs or at least 1 s exact; timeline runs' quantiles exact"),
            ("survival counts are exact (values ≥ t_us[i]) and max_us is the exact maximum, unrounded; interpolation (the "
             "vectors): 1 below t_us[0], ln s linear in ln t between grid points, and on the segment straddling max_us "
             "(t_i ≤ max_us < t_{i+1}, s_{i+1} = 0) ln s linear in ln t from (t_i, s_i) to (max_us, 1/n), n the row's n: "
             "exactly 1/n at max_us, 0 past it"),
            ("rows whose max_us passes the grid's last point (100 ms), which the spec's rule does not cover: "
             + (", ".join(f"cell {r['cell']} {r['metric']} {r['op']} (max {r['max_us'] / 1000:.1f} ms)" for r in survival
                          if r["max_us"] > T_US[-1]) or "none")
             + "; there survival_at runs ln s linearly in ln t from (t_us[200], s[200]) to (max_us, 1/n), and the vectors "
             "leave these rows out"),
            ("the ToR model (contract §3) on node 2: mlx5_core, 400 Gb/s, MTU 1500 (MSS 1448, 90 B per frame), B_pcie 25 GB/s, "
            "rx-usecs 8 for default (t_nic_rx 10 µs) and 2 µs for tuned; the 100 Gb/s variants use the same terms at 100 Gb/s; "
            f"response bytes = the {header} B header ({header // 8} words) + the block (pings carry the cell's block from the "
            "staging buffer)"),
            ("thresholds: call_ns from verity_pous.primitives.pi2_call_ns at the explorer's commit 57cddd474 (equal on main); "
            "τ(m) = (D + 1)·call_ns/m/1000 − 0.001 µs, D = min(d, 5), m = 1; an answer is over τ when its value > τ"),
            ("timeline.runs[].over_tau is keyed by metric, then '<block>/d<d>': {\"elapsed\": {\"65536/d0\": count, ...}, "
            "\"tor_400_tuned\": {...}}; every run is counted against every block's thresholds; twins have elapsed only"),
            ("timeline.minutes: steal_ms_per_s on the core the verifier (or responder) last ran on at each 100 ms sample, and the "
            f"largest other core among {cores}; softirq_per_s summed over cores {cores} and all softirq types; ctx_switch_per_s the "
            "voluntary + involuntary switches of the responder thread, verifier and checker; clock_offset_range_ns the range of "
            "CLOCK_REALTIME − CLOCK_MONOTONIC (bracketed) within the minute"),
            ("serving: vLLM steps whose GPU start falls in a run's measured window (first arrival + 1 s to the run's end), per cell "
             "over both twins and timestamping-on runs; step_gpu_ms = gpu_t1 − gpu_t0"),
            self.free_note(),
            "stalls: the 200 slowest timestamping-on or twin challenge answers by elapsed; steal_ms is the stolen time over the "
            "100 ms samples spanning [V0, V1] on the responder's or verifier's core (the larger, 10 ms jiffies); softirq the "
            f"softirqs on cores {cores} over the same samples; vllm_step_overlap the share of [V0, V1] inside vLLM steps' GPU "
            "intervals; " + ATTRIBUTION_RULE + "; attribution_label names the cause and, for socket and unknown, the largest "
            "part",
        ]
        return out


def vectors(doc: dict[str, Any], rows: int = 6) -> list[dict[str, Any]]:
    """Interpolation lookups from `survival_at` over a spread of the file's survival rows: below the grid, on and between
    grid points, three inside the segment straddling the row's max, at the max exactly and just past it."""
    t_us = doc["t_us"]
    cand = sorted((r for r in doc["survival"] if r["n"] >= 2 and r["max_us"] < t_us[-1]), key=lambda r: -r["n"])
    longest = max(cand, key=lambda x: x["max_us"]) if cand else None
    wants = (lambda r: r["op"] == "challenge" and r["timestamping"] and r["metric"] == "elapsed",
             lambda r: r["op"] == "challenge" and r["metric"] == "tor_400_tuned",
             lambda r: r["op"] == "ping",
             lambda r: not r["timestamping"],
             lambda r: r["metric"] == "gpu_span",
             lambda r: r is longest)
    pick: list[dict[str, Any]] = []
    for want in wants:
        r = next((x for x in cand if want(x) and x not in pick), None)
        if r is not None:
            pick.append(r)
    pick += [x for x in cand if x not in pick][: max(0, rows - len(pick))]
    out = []
    for r in pick:
        mx, n = r["max_us"], r["n"]
        last = len(t_us) - 1
        first_lt1 = next((i for i, v in enumerate(r["s"]) if v < 1), last)
        ts = [0.5, t_us[first_lt1], math.sqrt(t_us[first_lt1] * t_us[min(first_lt1 + 1, last)]),
              t_us[min(first_lt1 + 7, last)] * 1.013, mx * 0.5]
        if mx >= t_us[0]:
            i = bisect.bisect_right(t_us, mx) - 1
            if t_us[i] < mx:
                ts += [t_us[i] * (mx / t_us[i]) ** f for f in (0.25, 0.5, 0.75)]
        ts += [mx, mx * (1 + 1e-6), mx * 1.5]
        for t in ts:
            out.append({"inputs": {"cell": r["cell"], "metric": r["metric"], "op": r["op"], "timestamping": r["timestamping"],
                                   "t_us": float(t)},
                        "expected": {"survival_fraction": survival_at(t_us, r["s"], mx, n, float(t))},
                        "tolerance": {"relative": 1e-9}})
    return out


def _commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
                              cwd=Path(__file__).parent).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def headline(doc: dict[str, Any]) -> dict[str, Any]:
    """Per config (responder), the worst cell's p99.9 of elapsed and tor_400_tuned; per block, the share of challenge answers
    over τ(d = 0, m = 1) for both."""
    cfg = {c["cell"]: c for c in doc["cells"]}
    out: dict[str, Any] = {"worst_p999_us": {}, "over_tau_d0": {}}
    for q in doc["quantiles"]:
        if q["op"] != "challenge" or not q["timestamping"] or q["metric"] not in ("elapsed", "tor_400_tuned"):
            continue
        k = f'{cfg[q["cell"]]["config"]}/{q["metric"]}'
        if q["p999_us"] is not None and q["p999_us"] > out["worst_p999_us"].get(k, (-1,))[0]:
            out["worst_p999_us"][k] = (q["p999_us"], q["cell"])
    tau = {t["block_bytes"]: t["rounds"][0]["tau_us_at_m1"] for t in doc["thresholds"]}
    for metric in ("elapsed", "tor_400_tuned"):
        for b, t in tau.items():
            n = sum(r["n"] for r in doc["survival"] if r["metric"] == metric and r["op"] == "challenge" and r["timestamping"]
                    and cfg[r["cell"]]["block_bytes"] == b)
            over = sum(r["over_tau"].get(metric, {}).get(f"{b}/d0", 0) for r in doc["timeline"]["runs"]
                       if r["timestamping"] and cfg[r["cell"]]["block_bytes"] == b)
            out["over_tau_d0"][f"{metric}/{b}"] = {"tau_us": t, "n": n, "over": over, "share": over / n if n else None}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("chunks", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--art", action="append", default=[], help="CHUNK=art:… (default: the chunk's .art file beside it)")
    ap.add_argument("--note", action="append", default=[], help="a run.notes line to add")
    ap.add_argument("--host", default="vy-nebius-2")
    ap.add_argument("--gpu", default="RTX PRO 6000 Blackwell, GPU 7")
    ap.add_argument("--commit", default=None)
    a = ap.parse_args(argv)
    given = dict(x.split("=", 1) for x in a.art)
    b = Builder()
    arts = []
    for p in a.chunks:
        art = given.get(str(p)) or given.get(p.name)
        side = p.with_name(p.name + ".art")
        if not art and side.exists():
            art = side.read_text().split()[0]
        arts.append(art or p.name)
        b.add(p)
        print(f"{p.name}: {b.n_answers:,} answers so far", file=sys.stderr)
    doc = b.document(arts=arts, commit=a.commit or _commit(), generated_at=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                     host=a.host, gpu=a.gpu, extra_notes=a.note)
    a.out.write_text(json.dumps(doc, ensure_ascii=False, allow_nan=False, separators=(",", ":")))
    print(json.dumps(headline(doc), indent=1), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
