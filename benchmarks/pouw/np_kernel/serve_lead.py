"""The lead served by serve-openai on one H200: `serve.cu` (ssn.cu's kernel under the lead's layout, as a library) bound as
the lead's `LeadBackend` (`SsnLead`) and installed on the engine through `verity_vllm.LLM`'s protocols, its coins the
in-process certifier's. Measurement only: what it serves credits nothing.

    python serve_lead.py check --lib LIB [--mode M] --out OUT.jsonl
    python serve_lead.py serve (--stock | --lib LIB [--mode M] [--budget-gb G] [--nbuf B] [--audit N]) [--stats OUT.json] -- ARGS...
    python serve_lead.py report --arm NAME=DIR ... --T T --out REPORT.json

* ``check``: `serve.cu`'s A' and B' against ssn.cu's host tiling (``ssn_lead_check_operands``), then every served output
  of Llama-3.2-1B's four fused linears at 1, 17, 100 and 640 rows against `reference` and against stock's y
  (`int8_exact.dequantize` of the exact product, verity_int8's) bit for bit, its buffers poisoned first, under fresh coins
  (`feed.InProcessCertifier`) and under pooled ones (`PooledCertifier`). Exits 1 on any output that differs from either.
* ``serve``: serve-openai (``ARGS``, its own arguments) with the lead on its engine: ``protocols`` the lead's scheme and
  its ``backend`` this library, added to serve-openai's engine options; the coin link a `PooledCertifier` prepared once
  the linears are loaded. ``--audit N`` checks the first N calls' outputs against `reference` and stock's. ``--stats``
  gets the library's counts, the audits and the lead's details once serve-openai returns. ``--stock`` serves the same
  engine with nothing installed. Every arm schedules synchronously (`ENGINE`), as the B4 row's Build declares, with no
  row read.
* ``report``: each arm's step wall (served_util.py's rule: a step spans the previous step's end to its own, over the
  longest run of consecutive full steps of T tokens that starts at or after the main load's start), and its main load's
  tokens against the first arm's, request by request.

Modes (`serve.cu`): 2 the lead (every word exported and hashed beside the GEMM), 1 exported only, 0 the GEMM serving y
alone. What stands in for what isn't served yet:

* root(A_j) and the words root are hm96 rows and frame trees of the served committer's kernels under stand-in constants
  (`serve.cu`), not the committer's keyed roots; the key leaf is a SHA-512 of the key;
* the coins are this process's (`PooledCertifier`): each (K, N)'s coins are sampled once before serving, for the most
  rows a call takes, and each call is fed their leading rows and tiles, so every call of a shape reads the same coins
  (the release's order and refusals stay the certifier's); the feed is copied to the device from pinned host memory,
  with no link between.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.util
import json
import secrets
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from verity_vllm.protocols.pouw.lead import feed as F

HERE = Path(__file__).resolve().parent
#: the kernel's tiles: 64 rows, 256 columns (a linear's N a whole number of them, its K whole KiB)
ROW_TILE, COL_TILE = 64, 256
#: the most chunk roots the library holds between two `SsnLead.words_root` calls (serve.cu's MAX_CHUNKS)
MAX_ROOTS = 8192
#: the bf16 bits a poisoned output holds until the kernel writes it (0xA5A5)
POISON = 0xA5A5 - (1 << 16)
#: Llama-3.2-1B's fused linears, (N, K): qkv, o, gate_up, down (gate.py's SERVED)
SERVED = ((3072, 2048), (2048, 2048), (16384, 2048), (2048, 8192))
CHECK_ROWS = (1, 17, 100, 640)
#: (m, N, K) of the operand checks: a ragged one-row call, a ragged tile, two column tiles at a whole step's K
OPERANDS = ((1, 256, 1024), (100, 256, 1024), (64, 512, 2048))
FOLD_BOUND = 1 << 22
#: 1.5·2²³, the row source's constant: the epilogue reads a final word plus its row source 0x4B400000 − w_i and its column
#: source β_j − σ, less a1·C and T·G (layout.h's fold_word), as this float plus A_i·B_j
MAGIC = 0x4B400000
#: the engine options every served arm adds: the scheduler the B4 row's Build declares (vLLM's own default is its choice)
ENGINE = {"async_scheduling": False}


class LeadLibError(RuntimeError):
    pass


def _rc(name: str, rc: int) -> None:
    if rc:
        raise LeadLibError(f"ssn_lead_{name} returned {rc}")


class Lib:
    """`serve.cu`'s C API (its header states each function)."""

    def __init__(self, path: str | Path) -> None:
        self.so = so = ctypes.CDLL(str(path))
        i, p = ctypes.c_int, ctypes.c_void_p
        for name, args in {"init": [i, ctypes.c_double, i, i], "load": [i, i, i, p, p, p], "ready": [p],
                           "a_root": [i, i, p, p, p], "call": [i, i, p, p, p, p, p, p, p, p],
                           "words_roots": [p, i], "check_operands": [i, i, i, ctypes.c_ulonglong]}.items():
            f = getattr(so, f"ssn_lead_{name}")
            f.argtypes, f.restype = args, i

    def init(self, max_m: int, budget_gb: float, nbuf: int, mode: int) -> None:
        _rc("init", self.so.ssn_lead_init(max_m, budget_gb, nbuf, mode))


def _stream() -> int:
    import torch
    return torch.cuda.current_stream().cuda_stream


def _host(a: Any) -> tuple[Any, int]:
    """``a`` (a numpy array or a torch tensor in host memory) held contiguous, and its address."""
    if hasattr(a, "data_ptr"):
        a = a.contiguous()
        return a, a.data_ptr()
    a = np.ascontiguousarray(a)
    return a, a.ctypes.data


def admitted_variant() -> str:
    """The lead's variant as this tree names it from the recorded gate (gate.py ``--recorded``), when the gate admits it
    at Llama-3.2-1B's fused linears; SystemExit naming the refusals otherwise."""
    spec = importlib.util.spec_from_file_location("np_kernel_gate", HERE / "gate.py")
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    from verity.core.protocols.pouw.serving import POUW_V2, SERIAL, variant_refusals

    v, g = gate.recorded()
    if refusals := variant_refusals(v, POUW_V2, SERIAL.name, gate.SERVED, g):
        raise SystemExit("the lead's gate doesn't admit this tree's kernel: " + "; ".join(refusals))
    return v.id


@dataclass(frozen=True)
class Pool:
    """One (K, N)'s coins for ``rows`` rows (whole 64-row tiles), as a call's feed holds them, in host tensors: row words
    [rows, H], G words [C], lane words [T, S, C] and column sources [T, 2, C] (i32 bits)."""

    k: int
    n: int
    rows: Any
    g: Any
    lanes: Any
    sources: Any

    @classmethod
    def sample(cls, k: int, n: int, rows: int, rand: F.Rand, pin: bool) -> Pool:
        """Each 64-row tile's coins sampled as a call of 64 rows samples them (`feed.CallCoins.sample`), packed by
        `feed.Feed.of`; G is the first tile's."""
        import torch
        tile = F.Shape(ROW_TILE, k, n)
        t = -(-rows // ROW_TILE)
        out = {"rows": (t * ROW_TILE, tile.H), "g": (tile.C,), "lanes": (t, tile.S, tile.C), "sources": (t, 2, tile.C)}
        got = {name: torch.empty(dims, dtype=torch.int32, pin_memory=pin) for name, dims in out.items()}
        for r in range(t):
            f = F.Feed.of(F.Call(0, r, 0, ROW_TILE, k, n), F.CallCoins.sample(tile, rand))
            got["rows"][r * ROW_TILE:(r + 1) * ROW_TILE] = torch.from_numpy(f.rows.view(np.int32))
            got["lanes"][r] = torch.from_numpy(f.lanes_c[0].view(np.int32))
            got["sources"][r] = torch.from_numpy(f.sources_c[0].astype(np.int32))
            if r == 0:
                got["g"][:] = torch.from_numpy(f.g.view(np.int32))
        return cls(k, n, **got)

    def bytes(self) -> int:
        return sum(x.numel() * 4 for x in (self.rows, self.g, self.lanes, self.sources))


@dataclass(frozen=True)
class PooledFeed:
    """A call's feed from its shape's `Pool`: its R leading row words, the G words, its T leading tiles' lane words and
    column sources, as `feed.Feed` gives them."""

    call: F.Call
    pool: Pool

    @property
    def rows(self) -> Any:
        return self.pool.rows[:self.call.shape.R]

    @property
    def g(self) -> Any:
        return self.pool.g

    def lanes(self, n: int) -> Any:
        return self._tiles(self.pool.lanes, n)

    def sources(self, n: int) -> Any:
        return self._tiles(self.pool.sources, n)

    def _tiles(self, a: Any, n: int) -> Any:
        if n != self.pool.n:
            raise F.FeedError(f"stride {n}: the pool is at its {self.pool.n} columns")
        return a[:self.call.shape.T].reshape(-1)


class PooledCertifier(F.InProcessCertifier):
    """The in-process certifier (its windows, order and refusals) with each (K, N)'s coins sampled once (`Pool`), for
    ``rows`` rows, and every admitted call of that shape fed its leading rows and tiles: a measurement's coins, the same
    for every call of a shape."""

    def __init__(self, rows: int, rand: F.Rand = secrets.token_bytes, pin: bool = True) -> None:
        super().__init__(rand)
        self.rows, self.pin = rows, pin
        self.pools: dict[tuple[int, int], Pool] = {}

    def prepare(self, shapes: Any) -> None:
        """The pools of the (K, N) in ``shapes``, sampled now."""
        for k, n in shapes:
            self._pool(k, n)

    def _pool(self, k: int, n: int) -> Pool:
        if (key := (k, n)) not in self.pools:
            self.pools[key] = Pool.sample(k, n, self.rows, self.rand, self.pin)
        return self.pools[key]

    def coins(self, call: F.Call) -> PooledFeed:
        if call.m > self.rows:
            raise F.FeedError(f"call {call.index}: {call.m} rows, past the pool's {self.rows}")
        return PooledFeed(call, self._pool(call.k, call.n))


def fold(read: Any, s_a: Any, s_w: Any) -> Any:
    """The lead's fold: y [m, N] bf16 from the integer its epilogue reads back, ``read`` (int64 [m, N], taken mod 2³²: the
    final word less every coin term, `MAGIC`'s), and the scales ``s_a`` [m] and ``s_w`` [N]. v is
    the float whose bits are ``read`` and y = bf16(fl(fl(fl(v − 1.5·2²³)·s_i)·s_j)): an FADD, then the scales in
    verity_int8's order, the activation's first. On read = 0x4B400000 + p, v − 1.5·2²³ is p exactly while |p| ≤ 2²²
    (`FOLD_BOUND`), so y is stock's y on p; past it v is another float."""
    import torch
    word = read & 0xFFFFFFFF
    v = torch.where(word >= 1 << 31, word - (1 << 32), word).to(torch.int32).view(torch.float32)
    f = v - torch.tensor(12582912.0, dtype=torch.float32, device=v.device)
    return (f * s_a.reshape(-1, 1).to(torch.float32) * s_w.reshape(1, -1).to(torch.float32)).to(torch.bfloat16)


def reference(xq: Any, s_a: Any, w8: Any, s_w: Any) -> tuple[Any, Any]:
    """The lead's served y [m, N] bf16 for the s8 rows ``xq`` [m, K] (scales ``s_a`` [m]) and weight ``w8`` [N, K] (scales
    ``s_w``), with A·Bᵀ (int64 [m, N]): `fold` on 0x4B400000 + A·Bᵀ, whatever the call's coins, which is stock's y
    (`int8_exact.dequantize` of A·Bᵀ) bit for bit while |A·Bᵀ| ≤ 2²² (the misread past it too)."""
    import torch
    ab = (xq.to(torch.float64) @ w8.to(torch.float64).T).to(torch.int64)
    return fold(ab + MAGIC, s_a, s_w), ab


def compare(xq: Any, s_a: Any, w8: Any, s_w: Any, y: Any) -> dict[str, Any]:
    """The served ``y`` against `reference` and against stock's y (`int8_exact.dequantize` of the exact product), bit for
    bit: the outputs, those that differ from each, the most bf16 steps between y and stock's among outputs of one sign,
    and the outputs whose A·Bᵀ is past 2²²."""
    import torch
    from verity_vllm.engine import int8_exact as E
    ref, ab = reference(xq, s_a, w8, s_w)
    stock = E.dequantize(ab.to(torch.int32), s_a.reshape(-1), s_w)
    yb, sb = y.view(torch.int16).to(torch.int32), stock.view(torch.int16).to(torch.int32)
    same = (yb >= 0) == (sb >= 0)
    return {"outputs": int(y.numel()), "differ_from_reference": int((y.view(torch.int16) != ref.view(torch.int16)).sum()),
            "differ_from_stock": int((yb != sb).sum()), "sign_differs_from_stock": int((~same).sum()),
            "max_steps_from_stock": int(((yb - sb).abs() * same).max()) if y.numel() else 0,
            "ab_past_bound": int((ab.abs() > FOLD_BOUND).sum())}


class SsnLead:
    """The lead's `LeadBackend` on `serve.cu` (``lib``, initialized), its kernel ``variant``. Each call's y is a fresh
    tensor of whole 64-row tiles, poisoned first with ``poison``; the first ``audit`` calls are compared (`compare`)."""

    def __init__(self, lib: Lib, variant: str | None, *, poison: bool = False, audit: int = 0) -> None:
        self.lib, self.variant, self.poison, self.audit = lib, variant, poison, audit
        self.shapes: dict[int, tuple[int, int]] = {}
        self.weights: dict[int, tuple[Any, Any]] = {}
        self.held: int | None = None
        self.audits: list[dict[str, Any]] = []
        self.stats = {"calls": 0, "rows": 0, "feed_bytes": 0, "a_root_s": 0.0, "call_s": 0.0, "chunk_roots": 0, "passes": 0}

    def load(self, w: int, layer: Any) -> None:
        import torch
        w8 = layer.weight.detach().contiguous()
        sw = layer.weight_scale.detach().reshape(-1).to(torch.float32).contiguous()
        n, k = (int(d) for d in w8.shape)
        _rc("load", self.lib.so.ssn_lead_load(w, n, k, w8.data_ptr(), sw.data_ptr(), _stream()))
        self.shapes[w], self.weights[w] = (n, k), (layer.weight, layer.weight_scale)

    def ready(self) -> int:
        if self.held is None:
            got = ctypes.c_ulonglong(0)
            _rc("ready", self.lib.so.ssn_lead_ready(ctypes.byref(got)))
            self.held = got.value
        return self.held

    def key_leaf(self, key: bytes) -> bytes:
        return hashlib.sha512(b"lead/key-leaf/stand-in" + key).digest()

    def a_root(self, w: int, call: F.Call, xq: Any, window: Any) -> bytes:
        self.ready()
        x = xq.contiguous()
        out = ctypes.create_string_buffer(64)
        t0 = time.perf_counter()
        _rc("a_root", self.lib.so.ssn_lead_a_root(w, call.m, x.data_ptr(), ctypes.addressof(out), _stream()))
        self.stats["a_root_s"] += time.perf_counter() - t0
        return out.raw

    def call(self, w: int, layer: Any, xq: Any, s_a: Any, feed: Any) -> Any:
        import torch
        n, _ = self.shapes[w]
        m = int(xq.shape[0])
        y = torch.empty((-(-m // ROW_TILE) * ROW_TILE, n), dtype=torch.bfloat16, device=xq.device)
        if self.poison:
            y.view(torch.int16).fill_(POISON)
        x, sa = xq.contiguous(), s_a.reshape(-1).to(torch.float32).contiguous()
        held = [_host(a) for a in (feed.rows, feed.g, feed.lanes(n), feed.sources(n))]
        t0 = time.perf_counter()
        _rc("call", self.lib.so.ssn_lead_call(w, m, x.data_ptr(), sa.data_ptr(), *(p for _, p in held), y.data_ptr(),
                                              _stream()))
        self.stats["call_s"] += time.perf_counter() - t0
        self.stats["calls"] += 1
        self.stats["rows"] += m
        self.stats["feed_bytes"] += sum(a.nbytes if isinstance(a, np.ndarray) else a.numel() * 4 for a, _ in held)
        if self.audit > 0:
            self.audit -= 1
            self.audits.append({"linear": w, "m": m, "n": n, "k": int(xq.shape[1]),
                                **compare(x, sa, *self.weights[w], y[:m])})
        return y[:m]

    def words_root(self, window: Any) -> bytes:
        buf = ctypes.create_string_buffer(64 * MAX_ROOTS)
        n = self.lib.so.ssn_lead_words_roots(ctypes.addressof(buf), MAX_ROOTS)
        if n < 0:
            raise LeadLibError(f"ssn_lead_words_roots: past {MAX_ROOTS} chunk roots in one window")
        self.stats["chunk_roots"] += n
        self.stats["passes"] += 1
        return hashlib.sha512(b"lead/words-root/stand-in" + window.key + window.salt + buf.raw[:64 * n]).digest()


@dataclass
class _Layer:
    weight: Any
    weight_scale: Any


def check(a: argparse.Namespace) -> int:
    import torch
    from verity_vllm.engine import int8_exact as E
    from verity_vllm.protocols.pouw import lead as PL
    lib = Lib(a.lib)
    lib.init(max(CHECK_ROWS), a.budget_gb, a.nbuf, a.mode)
    lines, bad = [], 0
    for m, n, k in OPERANDS:
        got = lib.so.ssn_lead_check_operands(m, n, k, a.seed)
        bad += got != 0
        lines.append({"check": "operands", "m": m, "n": n, "k": k, "bytes_differ": got})
    backend = SsnLead(lib, None, poison=True)
    gen = torch.Generator(device="cuda").manual_seed(a.seed)
    layers = []
    for w, (n, k) in enumerate(SERVED):
        layers.append(_Layer(torch.randint(-127, 128, (n, k), dtype=torch.int8, device="cuda", generator=gen),
                             (torch.rand((n, 1), device="cuda", generator=gen) * 2e-3 + 1e-5).to(torch.float32)))
        backend.load(w, layers[-1])
    calls = len(CHECK_ROWS) * len(SERVED)
    links = {"fresh": F.InProcessCertifier(np.random.default_rng(a.seed).bytes),
             "pooled": PooledCertifier(max(CHECK_ROWS), np.random.default_rng(a.seed + 1).bytes)}
    for name, link in links.items():
        link.open(0, bytes(64), bytes(64), calls)
    j = 0
    for m in CHECK_ROWS:
        for w, (n, k) in enumerate(SERVED):
            x = torch.randn((m, k), device="cuda", generator=gen).to(torch.bfloat16)
            xq, s_a = E.quantize_rows(x)
            call = F.Call(0, j, w, m, k, n)
            root = backend.a_root(w, call, xq, None)
            for name, link in links.items():
                feed = link.release(call, root)
                y = backend.call(w, layers[w], xq, s_a, feed)
                got = compare(xq, s_a, layers[w].weight, layers[w].weight_scale, y)
                bad += got["differ_from_reference"] != 0 or got["differ_from_stock"] != 0
                lines.append({"check": "call", "coins": name, "mode": a.mode, "m": m, "n": n, "k": k, **got})
            j += 1
        root = backend.words_root(PL.WindowKeys(0, calls, bytes(64), bytes(64)))
        lines.append({"check": "words_root", "m": m, "chunk_roots": backend.stats["chunk_roots"], "root": root.hex()[:16]})
    Path(a.out).write_text("".join(json.dumps(x) + "\n" for x in lines))
    for x in lines:
        print(json.dumps(x), flush=True)
    print(f"CHECK mode {a.mode}: {len(lines)} lines, {bad} failed", flush=True)
    return 1 if bad else 0


def serve(a: argparse.Namespace, rest: list[str]) -> int:
    from verity_vllm.pipeline import cli
    from verity_vllm.pipeline import serve_openai as SO
    from verity_vllm.protocols.pouw import lead as PL
    base_options, base_build, held = SO.engine_options, SO.build, {}
    added: dict[str, Any] = dict(ENGINE)
    if a.stock:
        print(f"LEAD none: the stock arm, engine options {json.dumps(added)} added", flush=True)
    else:
        lib = Lib(a.lib)
        lib.init(a.max_m, a.budget_gb, a.nbuf, a.mode)
        backend = SsnLead(lib, admitted_variant(), audit=a.audit)
        link = PooledCertifier(a.max_m)
        F.InProcessCertifier = lambda: link
        added |= {"protocols": f"pouw:{PL.SCHEME}", "protocols_config": {"pouw": {"backend": backend}}}

    def build(opts: Any, session: Any = None, row: Any = None) -> tuple[Any, Any]:
        llm, com = base_build(opts, session, row)
        held["llm"] = llm
        if not a.stock:
            t0 = time.perf_counter()
            link.prepare(sorted({(k, n) for n, k in backend.shapes.values()}))
            held["prepare_s"] = time.perf_counter() - t0
            print(f"LEAD installed: {len(backend.shapes)} linears, kernel {backend.variant}, mode {a.mode}, pools "
                  f"{sum(p.bytes() for p in link.pools.values()) / 1e9:.2f} GB sampled in {held['prepare_s']:.1f} s",
                  flush=True)
        return llm, com

    SO.engine_options, SO.build = (lambda opts: base_options(opts) | added), build
    sys.argv = ["verity-vllm", "serve-openai", *rest]
    rc = 1
    try:
        rc = cli.main()
    finally:
        llm, opt = held.get("llm"), None
        if llm is not None and llm.protocols is not None:
            opt = llm.protocols.get("pouw")
            llm.close()
        if not a.stock and a.stats:
            Path(a.stats).write_text(json.dumps({
                "mode": a.mode, "budget_gb": a.budget_gb, "nbuf": a.nbuf, "max_m": a.max_m, "kernel": backend.variant,
                "device_bytes": backend.held, "pool_bytes": sum(p.bytes() for p in link.pools.values()),
                "prepare_s": held.get("prepare_s"), "stats": backend.stats, "audits": backend.audits,
                "details": dict(opt.details) if opt is not None else None,
                "passes": len(opt.commits.get("passes", ())) if opt is not None else None,
                "coin_source": opt.commits.get("coin_source") if opt is not None else None}, indent=1))
    return rc


def steady(st: list[dict[str, Any]], T: int, after_s: float) -> list[int]:
    """Indices into ``st`` (the step records by step) of the longest run of consecutive full steps (T tokens) whose span
    (the previous step's ``t`` to its own) starts at or after ``after_s`` (served_util.py's rule, without a capture)."""
    best: list[int] = []
    run: list[int] = []
    for i in range(1, len(st)):
        ok = st[i]["T"] == T and st[i]["step"] == st[i - 1]["step"] + 1 and st[i - 1]["t"] >= after_s
        run = run + [i] if ok else []
        if len(run) > len(best):
            best = list(run)
    return best


def arm_of(d: Path, T: int) -> dict[str, Any]:
    recs = [json.loads(x) for x in (d / "advice_trace.jsonl").read_text().splitlines() if x.strip()]
    st = sorted((r for r in recs if r.get("kind") == "step" and "t" in r), key=lambda r: r["step"])
    timings = json.loads((d / "main.timings.json").read_text())
    idx = steady(st, T, timings["start_unix"])
    walls = [1e3 * (st[i]["t"] - st[i - 1]["t"]) for i in idx]
    out: dict[str, Any] = {"full_steps": len(idx)}
    if walls:
        q = statistics.quantiles(walls, n=10) if len(walls) > 1 else [walls[0]] * 9
        out["step_ms"] = {"median": round(statistics.median(walls), 3), "mean": round(statistics.fmean(walls), 3),
                          "p10": round(q[0], 3), "p90": round(q[-1], 3), "min": round(min(walls), 3), "max": round(max(walls), 3)}
    loads = [json.loads(x) for x in (d / "main.jsonl").read_text().splitlines() if x.strip()]
    out["tokens"] = {r["user"]: (r["response"] or {}).get("token_ids") for r in loads}
    out["ok"] = sum("error" not in (r["response"] or {}) for r in loads)
    if (lead := d / "lead.json").exists():
        out["lead"] = json.loads(lead.read_text())
    return out


def report(a: argparse.Namespace) -> int:
    arms = {name: arm_of(Path(d), a.T) for name, _, d in (x.partition("=") for x in a.arm)}
    base = next(iter(arms))
    doc: dict[str, Any] = {"T": a.T, "base": base, "arms": {}}
    for name, arm in arms.items():
        want = arms[base]["tokens"]
        differ = sorted(u for u, ids in arm["tokens"].items() if ids != want.get(u))
        audits = (arm.get("lead") or {}).get("audits") or []
        doc["arms"][name] = {k: v for k, v in arm.items() if k not in ("tokens", "lead")} | {
            "requests": len(arm["tokens"]), "tokens_equal_to_base": len(arm["tokens"]) - len(differ),
            "requests_differing": differ[:16]}
        if lead := arm.get("lead"):
            doc["arms"][name]["lead"] = {k: v for k, v in lead.items() if k != "audits"} | {
                "audited_calls": len(audits),
                **{f"audit_{k}": sum(x[k] for x in audits) for k in ("outputs", "differ_from_reference", "differ_from_stock",
                                                                      "sign_differs_from_stock", "ab_past_bound")},
                "audit_max_steps_from_stock": max((x["max_steps_from_stock"] for x in audits), default=None)}
        s = doc["arms"][name].get("step_ms", {})
        print(f"ARM {name}: {doc['arms'][name]['full_steps']} full steps, step median {s.get('median')} ms "
              f"(p10 {s.get('p10')}, p90 {s.get('p90')}), tokens equal to {base}'s on "
              f"{doc['arms'][name]['tokens_equal_to_base']} of {doc['arms'][name]['requests']} requests", flush=True)
    Path(a.out).write_text(json.dumps(doc, indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    rest = argv[argv.index("--") + 1:] if "--" in argv else []
    argv = argv[:argv.index("--")] if "--" in argv else argv
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("check", "serve"):
        p = sub.add_parser(name)
        p.add_argument("--lib", required=name == "check", help="serve.cu's shared library")
        p.add_argument("--mode", type=int, default=2, choices=(0, 1, 2))
        p.add_argument("--budget-gb", type=float, default=8.0, help="the exported words a hash chunk holds")
        p.add_argument("--nbuf", type=int, default=4, help="mode 2's word buffers, each its own hash stream")
    c = sub.choices["check"]
    c.add_argument("--seed", type=int, default=1)
    c.add_argument("--out", required=True)
    s = sub.choices["serve"]
    s.add_argument("--stock", action="store_true", help="serve the engine with nothing installed (no --lib)")
    s.add_argument("--max-m", type=int, default=8192, help="the most rows a call takes (serve-openai's T)")
    s.add_argument("--audit", type=int, default=0, help="the first N calls compared with the reference and stock")
    s.add_argument("--stats", default=None)
    r = sub.add_parser("report")
    r.add_argument("--arm", action="append", required=True, help="NAME=DIR, the first the base")
    r.add_argument("--T", type=int, required=True, help="a full step's tokens")
    r.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "serve" and a.stock == bool(a.lib):
        ap.error("serve takes --stock or --lib, one of them")
    if a.cmd == "check":
        return check(a)
    if a.cmd == "serve":
        return serve(a, rest)
    return report(a)


if __name__ == "__main__":
    sys.exit(main())
