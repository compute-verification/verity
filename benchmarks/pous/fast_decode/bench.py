"""The fast-decode prototypes' decode rates on one GPU, each library gated byte-exact against `p2_v1.reference` first.

Libraries: `--lib NAME=PATH` for a prototype (`tcsq.cu`, `nttsq.cu`: `fd_config`, `fd_decode` with mode 0 = D(c) and mode 1
= d(c), and the TC builds' `fd_keys`), `--cgbn NAME=PATH` for a `libp2v1.so` (`p2_v1/p2v1.cu`, the CUDA-core decode in use,
or its generic CGBN path at another width). Every width a library names gets `--bytes` of uniformly random canonical labels
(at least the ten edge values of `c` first) under v4's keys of a fixed setup at that width (`reference.expand_prefix` frames
the width's prime), with the payload profile of the label less two bytes.

For each library, in this order, nothing timed before its gate passed:
- the gate: on the pseudo-random picks and the edge labels, mode 1 equals `reference.d(c)` and mode 0 equals
  `reference.decode` under `reference.keys` (whose K and t rows the device's expansion must equal), every label of mode 0
  equals the first library's at the width, and no flag is raised; a failing pick records which limbs differ and whether the
  output is the expected value's negation, its unreduced form or the square without the parity negation;
- `keys_s` (once a width): the counter-mode SHA-512 expansion of every label's K and t; `sq_s`: mode 1 (one squaring with
  its fold, loads and stores); `kernel_s`: mode 0 on keys expanded beforehand; `decode_s`: keys, then mode 0;
  `one_cta_square_ns` (prototypes): mode 1 on 256 labels by a single CTA, per label, which bounds from above the latency of
  one squaring mod p on one SM (what a sequential root would pay there), its output checked against the full grid's.
GB/s is payload bytes over the median time.

`--contend NAME` runs that library's mode 0 beside a BF16 GEMM (8192^3, on another stream) and records each one's time
alone and together: whether the decode takes the throughput the useful matmul needs. The calibration times an int8 and a
BF16 GEMM (torch) and a read of 8 GiB, on the same GPU. Writes `fast-decode.json`, and with `--result-dir` the run's
`result.json`."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

from verity.core.protocols.pous.schemes.p2 import POINT, P2Params

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import algo  # noqa: E402
from p2_v1 import reference as R  # noqa: E402

SALT = hashlib.sha256(b"verity-pous/bench/fast-decode/salt").digest()
MODEL = hashlib.sha512(b"verity-pous/bench/fast-decode/model").digest()
LATENCY_BLOCKS = 256
CONFIG_FIELDS = ("per", "depth", "limbs", "bytes", "q", "nt", "qb", "nops", "smem", "mmas_per_square", "ctas_per_sm",
                 "registers")


def params_of(per: int) -> R.Params:
    w = algo.Width(per)
    return R.Params.of(P2Params(w.w, w.c0, POINT[2]))


def per_of_bits(bits: int) -> int:
    per = next((p for p in algo.WIDTHS if algo.Width(p).w == bits), None)
    if per is None:
        raise ValueError(f"no prototype width of {bits} bits")
    return per


class Lib:
    """One library through ctypes: `kind` is "fd" (the prototypes' API) or "cgbn" (`p2v1_*`)."""

    def __init__(self, name: str, path: str, kind: str) -> None:
        v, u, q = ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint64
        self.name, self.path, self.kind = name, path, kind
        self.so = ctypes.CDLL(path)
        if kind == "fd":
            cfg = (ctypes.c_int * 12)()
            rc = self.so.fd_config(cfg)
            if rc:
                raise RuntimeError(f"{name}: fd_config error {rc}")
            self.config = dict(zip(CONFIG_FIELDS, list(cfg)))
            self.per = self.config["per"]
            self.so.fd_decode.argtypes = [v, v, v, v, u, u, v, ctypes.c_int, v]
            self.so.fd_decode_grid.argtypes = [v, v, v, v, u, u, v, ctypes.c_int, v, ctypes.c_int]
            self.has_keys = hasattr(self.so, "fd_keys")
            if self.has_keys:
                self.so.fd_keys.argtypes = [ctypes.c_char_p, ctypes.c_int, q, u, q, v, v, v]
                if self.so.fd_keys_version() != 4:
                    raise RuntimeError(f"{name}: keys are not v4's")
        else:
            self.per = per_of_bits(self.so.p2v1_width_bits())
            self.config = {"per": self.per, "width_bits": self.so.p2v1_width_bits()}
            self.so.p2v1_decode.argtypes = [v, v, v, v, u, u, v, v]
            self.so.p2v1_keys.argtypes = [ctypes.c_char_p, ctypes.c_int, q, u, q, v, v, v]
            self.has_keys = self.so.p2v1_keys_version() == 4
        self.mode1 = kind == "fd"

    def decode(self, c, m, k, t, n: int, c0: int, flags, mode: int, stream: int) -> None:
        if self.kind == "fd":
            rc = self.so.fd_decode(c.data_ptr(), m.data_ptr(), k.data_ptr(), t.data_ptr(), n, c0, flags.data_ptr(), mode, stream)
        else:
            assert mode == 0
            rc = self.so.p2v1_decode(c.data_ptr(), m.data_ptr(), k.data_ptr(), t.data_ptr(), n, c0, flags.data_ptr(), stream)
        if rc:
            raise RuntimeError(f"{self.name}: decode error {rc}")

    def decode_one_cta(self, c, m, k, t, n: int, c0: int, flags, mode: int, stream: int) -> None:
        rc = self.so.fd_decode_grid(c.data_ptr(), m.data_ptr(), k.data_ptr(), t.data_ptr(), n, c0, flags.data_ptr(), mode,
                                    stream, 1)
        if rc:
            raise RuntimeError(f"{self.name}: decode error {rc}")

    def keys(self, prefix: bytes, n: int, per_segment: int, k, t, stream: int) -> None:
        fn = self.so.fd_keys if self.kind == "fd" else self.so.p2v1_keys
        rc = fn(prefix, len(prefix), 0, n, per_segment, k.data_ptr(), t.data_ptr(), stream)
        if rc:
            raise RuntimeError(f"{self.name}: keys error {rc}")


def timed(torch, fn, reps: int) -> list[float]:
    fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        out.append(a.elapsed_time(b) / 1e3)
    return out


def edge_values(p: int, w: int) -> list[int]:
    return [0, 1, 2, 3, p - 1, p - 2, (p - 1) // 2, (1 << (w - 1)) - 1, 1 << (w - 1), p - (1 << 32)]


def diagnose(got: bytes, want: int, c: int, P: R.Params) -> dict:
    """How a wrong row differs: the little-endian 32-bit limbs that differ, and the usual suspects."""
    g, p = int.from_bytes(got, "big"), P.p
    diff = g ^ want
    limbs = [i for i in range(P.w // 32) if (diff >> 32 * i) & 0xFFFFFFFF]
    sq = c * c % p
    return {"limbs_differing": len(limbs), "first_limbs": limbs[:8], "last_limbs": limbs[-4:],
            "got_ge_p": g >= p, "equal_mod_p": g % p == want, "negation": (p - g) % p == want,
            "unnegated_square": g == sq, "got_head": got[:8].hex(), "got_tail": got[-8:].hex()}


class Width:
    """One width's labels, keys and reference rows on the device."""

    def __init__(self, torch, per: int, nbytes: int, gate: int, keys_lib: Lib) -> None:
        self.P = P = params_of(per)
        nb = P.width_bytes
        self.n = n = max(64, nbytes // nb)
        P = self.P = P.with_segments(-(-n // P.blocks_per_segment))
        self.prefix = R.expand_prefix(SALT, P, MODEL)
        dev = torch.device("cuda", torch.cuda.current_device())
        g = torch.Generator().manual_seed(20261011 + per)
        c = torch.randint(0, 256, (n, nb), dtype=torch.uint8, generator=g)
        c[:, 0] &= 0x7F
        edges = edge_values(P.p, P.w)
        for i, v in enumerate(edges):
            c[i] = torch.tensor(list(v.to_bytes(nb, "big")), dtype=torch.uint8)
        self.c = c.to(dev)
        host = c.numpy().tobytes()
        self.picks = sorted(set(range(len(edges))) | {n - 1} | {
            int.from_bytes(hashlib.sha256(f"fast-decode/{per}/{j}".encode()).digest()[:8], "big") % n for j in range(gate)})
        self.cval = {i: int.from_bytes(host[i * nb:(i + 1) * nb], "big") for i in self.picks}
        self.ref: dict[int, tuple[int, int, int, int]] = {}
        for i in self.picks:
            k, t, _ = R.keys(SALT, P, MODEL, i // P.blocks_per_segment, i % P.blocks_per_segment)
            self.ref[i] = (R.d(P, self.cval[i]), R.decode(P, self.cval[i], k, t), k, t)
        self.k = torch.empty_like(self.c)
        self.t = torch.empty_like(self.c)
        self.keys_lib = keys_lib
        stream = torch.cuda.current_stream().cuda_stream
        keys_lib.keys(self.prefix, n, P.blocks_per_segment, self.k, self.t, stream)
        torch.cuda.synchronize()
        rows = lambda x, i: int.from_bytes(bytes(x[i].cpu().numpy()), "big")  # noqa: E731
        self.keys_equal = sum(rows(self.k, i) == self.ref[i][2] and rows(self.t, i) == self.ref[i][3] for i in self.picks)
        self.keys_reps = timed(torch, lambda: keys_lib.keys(self.prefix, n, P.blocks_per_segment, self.k, self.t,
                                                            torch.cuda.current_stream().cuda_stream), 5)
        self.base_m = None

    @property
    def payload_gb(self) -> float:
        return self.n * self.P.payload_bytes / 1e9


def run_lib(torch, lib: Lib, W: Width, reps: int) -> dict:
    P, n = W.P, W.n
    flags = torch.zeros(1, dtype=torch.int32, device=W.c.device)
    m = torch.empty_like(W.c)
    stream = lambda: torch.cuda.current_stream().cuda_stream  # noqa: E731
    row = lambda i: bytes(m[i].cpu().numpy())  # noqa: E731
    r: dict = {"name": lib.name, "kind": lib.kind, "path": lib.path, "config": lib.config, "per": lib.per, "w": P.w,
               "blocks": n}
    bad1, bad0 = {}, {}
    m1 = None
    if lib.mode1:
        lib.decode(W.c, m, W.k, W.t, n, P.c0, flags, 1, stream())
        torch.cuda.synchronize()
        m1 = m[:LATENCY_BLOCKS].clone()
        for i in W.picks:
            got = row(i)
            if int.from_bytes(got, "big") != W.ref[i][0]:
                bad1[i] = diagnose(got, W.ref[i][0], W.cval[i], P)
    lib.decode(W.c, m, W.k, W.t, n, P.c0, flags, 0, stream())
    torch.cuda.synchronize()
    for i in W.picks:
        got = row(i)
        if int.from_bytes(got, "big") != W.ref[i][1]:
            bad0[i] = diagnose(got, W.ref[i][1], W.cval[i], P)
    if W.base_m is None and not bad0:
        W.base_m = m.clone()
    same = W.base_m is not None and bool(torch.equal(m, W.base_m))
    fl = int(flags.item())
    passed = not bad1 and not bad0 and same and fl == 0 and W.keys_equal == len(W.picks)
    r["gate"] = {"passed": passed, "picks": len(W.picks), "mode1_failed": {str(i): v for i, v in list(bad1.items())[:6]},
                 "mode1_failures": len(bad1), "mode0_failed": {str(i): v for i, v in list(bad0.items())[:6]},
                 "mode0_failures": len(bad0), "equals_first_at_width": same, "flags": fl,
                 "keys_equal": W.keys_equal}
    if not passed:
        return r
    gb = W.payload_gb
    if lib.mode1:
        sq = timed(torch, lambda: lib.decode(W.c, m, W.k, W.t, n, P.c0, flags, 1, stream()), reps)
        r["sq_s"], r["sq_GBps"], r["sq_reps_s"] = statistics.median(sq), gb / statistics.median(sq), sq
    kr = timed(torch, lambda: lib.decode(W.c, m, W.k, W.t, n, P.c0, flags, 0, stream()), reps)

    def full() -> None:
        W.keys_lib.keys(W.prefix, n, P.blocks_per_segment, W.k, W.t, stream())
        lib.decode(W.c, m, W.k, W.t, n, P.c0, flags, 0, stream())

    dr = timed(torch, full, reps)
    r.update({"kernel_s": statistics.median(kr), "kernel_GBps": gb / statistics.median(kr), "kernel_reps_s": kr,
              "decode_s": statistics.median(dr), "decode_GBps": gb / statistics.median(dr), "decode_reps_s": dr,
              "keys_s": statistics.median(W.keys_reps), "keys_GBps": gb / statistics.median(W.keys_reps)})
    if lib.kind == "fd" and lib.config.get("mmas_per_square"):
        macs = 2 * lib.config["mmas_per_square"] * 16 * 8 * 32
        r["tc_macs_per_block"] = macs
        r["tc_int8_tops_used"] = 2 * macs * n / r["kernel_s"] / 1e12
    if m1 is not None:
        nl = LATENCY_BLOCKS
        lat = timed(torch, lambda: lib.decode_one_cta(W.c, m, W.k, W.t, nl, P.c0, flags, 1, stream()), reps)
        r["one_cta_square_ns"] = statistics.median(lat) / nl * 1e9
        r["one_cta_square_reps_s"] = lat
        r["one_cta_output_equal"] = bool(torch.equal(m[:nl], m1))
    return r


def calibrate(torch) -> dict:
    dev = torch.device("cuda", torch.cuda.current_device())
    out = {}
    s = 8192
    a8 = torch.randint(-128, 127, (s, s), dtype=torch.int8, device=dev)
    b8 = torch.randint(-128, 127, (s, s), dtype=torch.int8, device=dev).t()
    try:
        t = statistics.median(timed(torch, lambda: torch._int_mm(a8, b8), 10))
        out["int8_gemm_tops"] = 2 * s ** 3 / t / 1e12
    except RuntimeError as e:
        out["int8_gemm_error"] = str(e)[:200]
    a = torch.randn(s, s, dtype=torch.bfloat16, device=dev)
    b = torch.randn(s, s, dtype=torch.bfloat16, device=dev)
    o = torch.empty_like(a)
    t = statistics.median(timed(torch, lambda: torch.mm(a, b, out=o), 10))
    out["bf16_gemm_tflops"] = 2 * s ** 3 / t / 1e12
    x = torch.ones(1 << 30, dtype=torch.int64, device=dev)
    t = statistics.median(timed(torch, lambda: x.sum(), 10))
    out["read_8GiB_GBps"] = 8 * (1 << 30) / t / 1e9
    del a8, b8, a, b, o, x
    torch.cuda.empty_cache()
    return out


def contend(torch, lib: Lib, W: Width, mm_reps: int = 40, dec_reps: int = 4) -> dict:
    """The library's mode 0 and a BF16 GEMM, alone and on two streams at once."""
    dev = W.c.device
    s = 8192
    a = torch.randn(s, s, dtype=torch.bfloat16, device=dev)
    b = torch.randn(s, s, dtype=torch.bfloat16, device=dev)
    o = torch.empty_like(a)
    m = torch.empty_like(W.c)
    flags = torch.zeros(1, dtype=torch.int32, device=dev)
    sa, sb = torch.cuda.Stream(), torch.cuda.Stream()

    def mm() -> None:
        for _ in range(mm_reps):
            torch.mm(a, b, out=o)

    def dec(st: int) -> None:
        for _ in range(dec_reps):
            lib.decode(W.c, m, W.k, W.t, W.n, W.P.c0, flags, 0, st)

    t_mm = statistics.median(timed(torch, mm, 3))
    t_dec = statistics.median(timed(torch, lambda: dec(torch.cuda.current_stream().cuda_stream), 3))
    both = []
    for _ in range(3):
        torch.cuda.synchronize()
        e0 = torch.cuda.Event(enable_timing=True)
        ea, eb = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        sa.wait_event(e0)
        sb.wait_event(e0)
        with torch.cuda.stream(sa):
            mm()
            ea.record(sa)
        with torch.cuda.stream(sb):
            dec(sb.cuda_stream)
            eb.record(sb)
        ea.synchronize()
        eb.synchronize()
        both.append((e0.elapsed_time(ea) / 1e3, e0.elapsed_time(eb) / 1e3))
    tm, td = statistics.median(x[0] for x in both), statistics.median(x[1] for x in both)
    flops = 2 * s ** 3 * mm_reps
    pay = W.payload_gb * dec_reps
    return {"lib": lib.name, "gemm": f"bf16 {s}^3 x {mm_reps}", "decode_passes": dec_reps,
            "gemm_alone_s": t_mm, "decode_alone_s": t_dec, "gemm_together_s": tm, "decode_together_s": td,
            "gemm_alone_tflops": flops / t_mm / 1e12, "gemm_together_tflops": flops / tm / 1e12,
            "decode_alone_GBps": pay / t_dec, "decode_together_GBps": pay / td,
            "overlap": (t_mm + t_dec) / max(tm, td), "samples_s": both}


def smi(uuid: str | None) -> str:
    sel = ["-i", uuid.split(",")[0]] if uuid else []
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,clocks.sm,clocks.max.sm,power.draw",
                          "--format=csv,noheader", *sel], capture_output=True, text=True, check=False).stdout
    return out.strip().splitlines()[0] if out.strip() else ""


def result_doc(doc: dict, run_id: str, run_dir: Path, out: Path, argv: list[str]) -> dict:
    files = [str(p.relative_to(run_dir)) for p in [out / "fast-decode.json"] if p.is_relative_to(run_dir)]
    ms = []
    for r in doc["results"]:
        for key, unit in (("sq_GBps", "GB/s"), ("kernel_GBps", "GB/s"), ("decode_GBps", "GB/s"), ("keys_GBps", "GB/s"),
                          ("one_cta_square_ns", "ns")):
            if key in r:
                ms.append({"name": f"{r['name']}.{key}", "value": r[key], "unit": unit})
    for key, value in doc["calibration"].items():
        if isinstance(value, float):
            ms.append({"name": f"calibration.{key}", "value": value, "unit": key.rsplit("_", 1)[-1]})
    failed = [r["name"] for r in doc["results"] if not r["gate"]["passed"]]
    smi_line = [x.strip() for x in doc["smi"].split(",")] + [""] * 3
    return {"schema": "research/result/v0.1", "run_id": run_id,
            "workload_fingerprint": {"harness": "benchmarks/pous/fast_decode/bench.py", "argv": argv,
                                     "source_sha": os.environ.get("RESEARCH_SOURCE_SHA")},
            "validation": {"status": "passed" if doc["passed"] else "failed",
                           "checker": "fast_decode gate (reference.d, reference.decode, reference.keys, cross-library)",
                           "detail": "every library's gate passed" if not failed else f"gate failed: {', '.join(failed)}",
                           "evidence": files},
            "device": {"name": smi_line[2] or doc["device"], "index": int(smi_line[0]) if smi_line[0].isdigit() else None,
                       "uuid": smi_line[1] or None},
            "measurements": ms, "measurement_files": files}


def parse_named(specs: list[str]) -> list[tuple[str, str]]:
    out = []
    for s in specs:
        name, _, path = s.partition("=")
        if not path:
            raise SystemExit(f"expected NAME=PATH, got {s!r}")
        out.append((name, path))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", action="append", default=[], help="NAME=PATH of a prototype library (fd_* API)")
    ap.add_argument("--cgbn", action="append", default=[], help="NAME=PATH of a libp2v1.so")
    ap.add_argument("--bytes", type=int, default=1 << 30, help="stored bytes of blocks at each width")
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--gate", type=int, default=48)
    ap.add_argument("--contend", action="append", default=[], help="a library to run beside a BF16 GEMM")
    ap.add_argument("--out", required=True)
    ap.add_argument("--result-dir")
    a = ap.parse_args()

    import torch

    libs = [Lib(n, p, "cgbn") for n, p in parse_named(a.cgbn)] + [Lib(n, p, "fd") for n, p in parse_named(a.lib)]
    for lib in libs:
        print(json.dumps({"lib": lib.name, "config": lib.config}), flush=True)
    calibration = calibrate(torch)
    print(json.dumps({"calibration": calibration}), flush=True)
    widths: dict[int, Width] = {}
    results, contention = [], []
    for lib in libs:
        if lib.per not in widths:
            keys_lib = next(x for x in libs if x.per == lib.per and x.has_keys)
            widths[lib.per] = Width(torch, lib.per, a.bytes, a.gate, keys_lib)
        r = run_lib(torch, lib, widths[lib.per], a.reps)
        results.append(r)
        print(json.dumps({x: r.get(x) for x in ("name", "w", "sq_GBps", "kernel_GBps", "decode_GBps", "keys_GBps",
                                                "tc_int8_tops_used", "one_cta_square_ns", "one_cta_output_equal")}
                         | {"gate": r["gate"]["passed"]}), flush=True)
        if not r["gate"]["passed"]:
            print(json.dumps({"gate_failure": r["name"], "gate": r["gate"]}), flush=True)
        elif lib.name in a.contend:
            contention.append(contend(torch, lib, widths[lib.per]))
            print(json.dumps({"contention": contention[-1]}), flush=True)
    doc = {"device": torch.cuda.get_device_name(), "smi": smi(os.environ.get("GPU_LEASE_UUID")),
           "calibration": calibration, "results": results, "contention": contention,
           "widths": {str(per): {"w": W.P.w, "c0": W.P.c0, "blocks": W.n, "payload_bytes": W.P.payload_bytes,
                                 "keys_lib": W.keys_lib.name, "keys_equal": W.keys_equal, "picks": len(W.picks)}
                      for per, W in widths.items()},
           "passed": all(r["gate"]["passed"] for r in results)}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "fast-decode.json").write_text(json.dumps(doc, indent=1))
    if a.result_dir:
        run_dir = Path(a.result_dir).resolve()
        res = result_doc(doc, os.environ.get("RESEARCH_RUN_ID", ""), run_dir, out.resolve(), sys.argv)
        tmp = run_dir / ".result.json.tmp"
        tmp.write_text(json.dumps(res, indent=1))
        os.replace(tmp, run_dir / "result.json")
    return 0 if doc["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
