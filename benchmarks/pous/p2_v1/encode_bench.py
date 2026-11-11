"""P2's two encoders on one GPU, side by side, on one segment of fixed payloads under a fixed setup's keys: the library's
default (`p2v1_encode`, CGBN's root; `cgbn` in `gpu.ENCODERS`) and the root on the decode's squaring (`p2v1_encode_sq`,
`p2enc.cuh`; `p2sq`). `encode_rate.sh` runs the three commands in this order.

    encode_bench.py build --lib NAME=DIR ... --out build.json
    encode_bench.py reference --out reference.json
    encode_bench.py run --lib DIR --reference reference.json --out DIR [--a NAME=DIR ...] [--decode-bench]

`build` reads no GPU and imports only the standard library. Of each build (a DIR holding the Makefile's `libp2v1.so`
and `libp2v1.so.ptxas`): every kernel's registers, stack and spilled bytes as ptxas said them, and the SHA-256 of its
disassembly (`cuobjdump -sass`, split at its `Function :` lines). When one build is named `base` and one `default`, it
says which of `base`'s kernels `default` holds instruction for instruction, so a build that adds an encoder shows it
left the others as they were.

`reference` reads no GPU either: `reference.encode` of `--picks` of the segment's blocks under `reference.keys`, on
every CPU the process may use. The first blocks are edge payloads, one of them the block whose first root is of zero.

`run` goes in four steps. A failed gate ends it there: what was measured before it stays, and nothing is measured
after. So does a pass that, at its encoder's last rate, would end later than `--minutes` after the start.

1. Gates, not timed: on the segment's first `--warm` blocks, then on the whole segment. Each encoder writes into a
   buffer filled with 0xA5. `p2sq`'s bytes equal `cgbn`'s, no row is left at 0xA5, the picked rows equal the
   reference's, no kernel flag is raised, and the decode returns the payloads. The same check on a buffer nobody wrote
   must fail.
2. Timing: after a short pass of each, `cgbn`, `p2sq`, `cgbn`, `p2sq` on the segment. A pass is `gpu.Kernels.encode`,
   the key expansion with it, between two CUDA events; its output is held to the gate's bytes.
3. `--decode-bench`: `decode_bench.py` on this library, the squaring's own rate in the same run. Then each encoder
   alone on the card at 1 block and at 1,024.
4. Each `--a` build (the default encoder under another build setting): its bytes on the first `--a-blocks` blocks
   equal the default build's before it is timed, then the default build and each `--a` build in turn, twice. A build
   whose bytes differ is not timed.

nvidia-smi reads the card each second for the whole run (`smi.csv`); each pass gets its mean and peak power and its SM
clock. Writes `encode-bench.json`. With `--say-at RATE`, a `p2sq` rate at or above RATE blocks a second is said on a
line of its own."""
from __future__ import annotations

import argparse
import ctypes
import functools
import hashlib
import json
import multiprocessing as mp
import os
import re
import signal
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

SALT = hashlib.sha256(b"verity-pous/bench/p2-v1/encode-bench/salt").digest()
MODEL = hashlib.sha512(b"verity-pous/bench/p2-v1/encode-bench/model").digest()
SEED = 20261111
POISON = 0xA5
ZERO_ROOT = 5
"""The block whose payload is `p − t`: `m + t` is zero, and so is its first root."""


@functools.cache
def store():
    """`reference`, and the deployment's parameters at one segment."""
    from p2_v1 import reference as R

    return R, R.DEPLOYMENT.with_segments(1)


def stem(name: str) -> str:
    """A kernel's own name in its mangled one: the last length-prefixed identifier before its arguments."""
    i, last = (3 if name.startswith("_ZN") else 2), name
    while i < len(name) and name[i].isdigit():
        j = i
        while name[j].isdigit():
            j += 1
        last, i = name[j:j + int(name[i:j])], j + int(name[i:j])
    return last


def ptxas(path: Path) -> dict[str, dict[str, int]]:
    """Each kernel of an `nvcc -Xptxas -v` report: its registers, and its stack, spill and shared-memory bytes."""
    out: dict[str, dict[str, int]] = {}
    cur: dict[str, int] | None = None
    for line in path.read_text(errors="replace").splitlines():
        if m := re.search(r"Function properties for (\S+)", line):
            cur = out.setdefault(stem(m[1]), {})
        elif cur is not None and (m := re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill", line)):
            cur.update(stack_bytes=int(m[1]), spill_store_bytes=int(m[2]), spill_load_bytes=int(m[3]))
        elif cur is not None and (m := re.search(r"Used (\d+) registers", line)):
            cur["registers"] = int(m[1])
            if s := re.search(r"(\d+) bytes smem", line):
                cur["smem_bytes"] = int(s[1])
    return out


def sass(lib: str) -> dict[str, str]:
    """The SHA-256 of each kernel's disassembly in `lib`, by its mangled name, runs of blanks made one."""
    done = subprocess.run(["cuobjdump", "-sass", lib], capture_output=True, text=True, check=False)
    if done.returncode:
        raise RuntimeError(f"cuobjdump -sass {lib} exits {done.returncode}: {' '.join((done.stderr or done.stdout).split())[:600]}")
    bodies: dict[str, list[str]] = {}
    body = None
    for line in done.stdout.splitlines():
        if "Function :" in line:
            body = bodies.setdefault(line.split("Function :", 1)[1].strip(), [])
        elif body is not None:
            body.append(" ".join(line.split()))
    return {name: hashlib.sha256("\n".join(lines).encode()).hexdigest() for name, lines in bodies.items()}


def build(a: argparse.Namespace) -> int:
    doc: dict[str, dict] = {"ptxas": {}, "sass_sha256": {}}
    hashes: dict[str, dict[str, str]] = {}
    for name, d in (x.split("=", 1) for x in a.lib):
        doc["ptxas"][name] = ptxas(Path(d) / "libp2v1.so.ptxas")
        for kernel, v in sorted(doc["ptxas"][name].items()):
            if "encode" in kernel or "decode_sq" in kernel:
                print(f"{name}: {kernel}: {v.get('registers')} registers, {v.get('spill_store_bytes')} bytes spill stores, "
                      f"{v.get('spill_load_bytes')} bytes spill loads, {v.get('stack_bytes')} bytes stack")
        try:
            hashes[name] = sass(str(Path(d) / "libp2v1.so"))
        except (RuntimeError, OSError) as e:
            doc["sass_sha256"][name] = {"error": str(e)}
            print(f"{name}: no disassembly: {e}")
        else:
            doc["sass_sha256"][name] = {stem(k): v for k, v in sorted(hashes[name].items())}
    if "base" in hashes and "default" in hashes:
        base, new = hashes["base"], hashes["default"]
        changed = sorted(stem(k) for k, v in base.items() if new.get(k) != v)
        doc["same"] = {"base_kernels": len(base), "default_kernels": len(new), "changed": changed,
                       "unchanged": sorted(stem(k) for k, v in base.items() if new.get(k) == v),
                       "only_default": sorted(stem(k) for k in new if k not in base)}
        print(f"{len(base) - len(changed)} of base's {len(base)} kernels are in default instruction for instruction; changed: "
              f"{', '.join(changed) or 'none'}; only in default: {', '.join(doc['same']['only_default']) or 'none'}")
    Path(a.out).write_text(json.dumps(doc, indent=1))
    return 0


def payloads(torch, n: int):
    """The segment's `n` payload rows, on the CPU: uniform bytes under the profile's leading zeros, behind the edge rows."""
    R, P = store()
    nb = P.width_bytes
    m = torch.randint(0, 256, (n, nb), dtype=torch.uint8, generator=torch.Generator().manual_seed(SEED))
    m[:, :nb - P.payload_bytes] = 0
    top = 8 * P.payload_bytes
    _, t, _ = R.keys(SALT, P, MODEL, 0, ZERO_ROOT)
    edges = [0, 1, (1 << top) - 1, 1 << (top - 1), P.p - 1, (P.p - t) % P.p]
    assert edges.index((P.p - t) % P.p) == ZERO_ROOT
    for i, v in enumerate(edges):
        m[i] = torch.frombuffer(bytearray(v.to_bytes(nb, "big")), dtype=torch.uint8)
    return m, len(edges)


def _reference_row(job: tuple[int, bytes]) -> bytes:
    i, m = job
    R, P = store()
    k, t, _ = R.keys(SALT, P, MODEL, 0, i)
    return R.encode(P, int.from_bytes(m, "big"), k, t).to_bytes(P.width_bytes, "big")


def reference(a: argparse.Namespace) -> int:
    import torch

    _, P = store()
    n, nb = a.blocks or P.blocks_per_segment, P.width_bytes
    m, edges = payloads(torch, n)
    host = m.numpy().tobytes()
    picks = sorted(set(range(edges)) | {n - 1} | {
        int.from_bytes(hashlib.sha256(f"encode-bench/{j}".encode()).digest()[:8], "big") % n for j in range(a.picks)})
    with mp.get_context("spawn").Pool(len(os.sched_getaffinity(0))) as pool:
        rows = pool.map(_reference_row, [(i, host[i * nb:(i + 1) * nb]) for i in picks], chunksize=1)
    Path(a.out).write_text(json.dumps({"blocks": n, "payloads_sha512": hashlib.sha512(host).hexdigest(),
                                       "rows": {str(i): c.hex() for i, c in zip(picks, rows, strict=True)}}, indent=1))
    print(f"reference: {len(picks)} of {n} blocks encoded on {len(os.sched_getaffinity(0))} CPUs")
    return 0


class Ended(Exception):
    """`run` ends here: what was measured before stays, and nothing is measured after."""


_LIBC = ctypes.CDLL(None)


def _with_parent() -> None:
    """In a child, before its program starts: a SIGTERM when this process dies, so a killed run leaves no process behind."""
    _LIBC.prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG


class Sampler:
    """nvidia-smi's reading of one card each second, into `path`, until `stop`."""

    def __init__(self, path: Path, uuid: str) -> None:
        self.path = path
        self.file = path.open("w")
        self.proc = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=timestamp,clocks.sm,power.draw,temperature.gpu,utilization.gpu",
             "--format=csv,noheader,nounits", "-lms", "1000", "-i", uuid], stdout=self.file, stderr=subprocess.STDOUT,
            preexec_fn=_with_parent)

    def stop(self) -> list[tuple[float, float, float, float]]:
        """Ends the sampling; the samples as `(unix time, SM MHz, W, deg C)`."""
        self.proc.terminate()
        self.proc.wait()
        self.file.close()
        out = []
        for line in self.path.read_text().splitlines():
            f = [x.strip() for x in line.split(",")]
            try:
                out.append((datetime.strptime(f[0], "%Y/%m/%d %H:%M:%S.%f").timestamp(), float(f[1]), float(f[2]), float(f[3])))
            except (ValueError, IndexError):
                continue
        return out


def card_during(samples: list[tuple[float, float, float, float]], start: float, end: float) -> dict[str, float] | None:
    s = [x for x in samples if start <= x[0] <= end]
    if not s:
        return None
    return {"samples": len(s), "power_w_mean": round(statistics.fmean(x[2] for x in s), 1), "power_w_max": max(x[2] for x in s),
            "sm_mhz_mean": round(statistics.fmean(x[1] for x in s), 1), "sm_mhz_min": min(x[1] for x in s),
            "temperature_c_max": max(x[3] for x in s)}


def run(a: argparse.Namespace) -> int:
    import torch

    from p2_v1.gpu import ENCODERS, Kernels

    R, P = store()
    ref = json.loads(Path(a.reference).read_text())
    n, nb = int(ref["blocks"]), P.width_bytes
    host, _ = payloads(torch, n)
    if hashlib.sha512(host.numpy().tobytes()).hexdigest() != ref["payloads_sha512"]:
        print("the payloads are not the ones the reference rows were made from", file=sys.stderr)
        return 2
    want = {int(i): bytes.fromhex(c) for i, c in ref["rows"].items()}
    dev = torch.device("cuda", torch.cuda.current_device())
    props = torch.cuda.get_device_properties(dev)
    uuid = os.environ.get("GPU_LEASE_UUID", "").split(",")[0] or f"GPU-{props.uuid}"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    prefix = R.expand_prefix(SALT, P, MODEL)
    kern = Kernels(a.lib, a.lib, P, prefix)
    missing = [name for name in ENCODERS.values() if not hasattr(kern.lib, name)]
    if missing:
        print(f"{a.lib} has no {', '.join(missing)}", file=sys.stderr)
        return 2
    ctas = None
    if hasattr(kern.lib, "p2v1_encode_ctas_per_sm"):
        ctas = {e: kern.lib.p2v1_encode_ctas_per_sm(int(e == "p2sq")) for e in ENCODERS}
    doc: dict[str, object] = {
        "device": torch.cuda.get_device_name(dev), "uuid": uuid, "sms": props.multi_processor_count, "ctas_per_sm": ctas,
        "torch": torch.__version__, "blocks": n, "block_bytes": nb, "lib": a.lib, "passed": False, "ended": None,
        "pass": "gpu.Kernels.encode: the key expansion, then the kernel; seconds between two CUDA events"}
    sampler = Sampler(out / "smi.csv", uuid)
    passes: list[dict[str, object]] = []
    rate: dict[tuple[str, str], float] = {}
    began = time.monotonic()
    m = host.to(dev)
    buf = torch.empty_like(m)
    today = None

    def on_sigterm(*_: object) -> None:
        raise Ended("a SIGTERM")

    signal.signal(signal.SIGTERM, on_sigterm)

    def encode(k: Kernels, build: str, encoder: str, count: int):
        """One pass over the first `count` blocks into the buffer, filled with 0xA5 first; its seconds and wall span.
        Ended before it starts when, at this build's and encoder's last rate, it would end after `--minutes`."""
        if (build, encoder) in rate and time.monotonic() - began + count / rate[build, encoder] > 60 * a.minutes:
            raise Ended(f"{a.minutes:g} minutes, before {encoder} of the build {build} on {count} blocks")
        buf[:count].fill_(POISON)
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start = time.time()
        e0.record()
        c = k.encode(m[:count], 0, encoder, out=buf[:count])
        e1.record()
        e1.synchronize()
        t = {"seconds": e0.elapsed_time(e1) / 1e3, "start_unix": start, "end_unix": time.time()}
        rate[build, encoder] = count / t["seconds"]
        return c, t

    def gate(c, bytes_of, flags: int) -> dict[str, object]:
        """`c`, the first rows of an encoder's output, against `bytes_of` (the default encoder's) and the reference's rows."""
        rows = {i: w for i, w in want.items() if i < c.shape[0]}
        equal = bool(torch.equal(c, bytes_of[:c.shape[0]]))
        unwritten = int((c == POISON).all(dim=1).sum())
        same = sum(bytes(c[i].cpu().numpy()) == w for i, w in rows.items())
        return {"passed": equal and unwritten == 0 and same == len(rows) and flags == 0, "equals_default_encoder": equal,
                "rows_left_at_0xA5": unwritten, "reference_rows": len(rows), "reference_rows_equal": same, "flags": flags}

    def gates(count: int):
        """Step 1 on the first `count` blocks: the default encoder's bytes, and the verdicts."""
        c, t0 = encode(kern, "default", "cgbn", count)
        c = c.clone()
        f0 = kern.raised()
        sq, t1 = encode(kern, "default", "p2sq", count)
        f1 = kern.raised()
        back = kern.decode(sq, 0)
        f2 = kern.raised()
        g = {"blocks": count, "cgbn": gate(c, c, f0), "p2sq": gate(sq, c, f1),
             "decode_returns_payloads": bool(torch.equal(back, m[:count])) and f2 == 0,
             "no_write_control": gate(torch.full_like(c, POISON), c, 0),
             "first_pass_s": {"cgbn": t0["seconds"], "p2sq": t1["seconds"]}}
        g["no_write_control_rejected"] = not g["no_write_control"]["passed"]
        g["passed"] = bool(g["cgbn"]["passed"] and g["p2sq"]["passed"] and g["decode_returns_payloads"]
                           and g["no_write_control_rejected"])
        return c, g

    def timed(build: str, k: Kernels, encoder: str, count: int, step: str) -> None:
        """A timed pass, held to the gate's bytes."""
        c, t = encode(k, build, encoder, count)
        flags = k.raised()
        ok = bool(torch.equal(c, today[:count])) and flags == 0
        passes.append({"step": step, "build": build, "encoder": encoder, "blocks": count, **t,
                       "blocks_per_s": count / t["seconds"], "bytes_equal": ok, "flags": flags})
        shown = ("step", "build", "encoder", "blocks", "seconds", "blocks_per_s", "bytes_equal")
        print(json.dumps({x: passes[-1][x] for x in shown}), flush=True)
        save()
        if not ok:
            raise Ended(f"a pass of {encoder} of the build {build} on {count} blocks whose bytes are not the gate's")

    def save() -> None:
        doc["passes"] = passes
        (out / "encode-bench.json").write_text(json.dumps(doc, indent=1))

    def finish(ended: str | None) -> int:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        samples = sampler.stop()
        for p in passes:
            p["card"] = card_during(samples, p["start_unix"], p["end_unix"])
        doc.update(ended=ended, passed=ended is None, passes=passes, smi_samples=len(samples),
                   seconds=round(time.monotonic() - began, 1))
        whole = {e: [p["blocks_per_s"] for p in passes if p["step"] == "timing" and p["encoder"] == e] for e in ENCODERS}
        if all(whole.values()):
            rates = {e: statistics.fmean(v) for e, v in whole.items()}
            seg = P.blocks_per_segment
            doc["timing"] = {
                "blocks_per_s": rates, "p2sq_over_cgbn": rates["p2sq"] / rates["cgbn"],
                "segment_s": {e: seg / r for e, r in rates.items()},
                "segments": a.segments, "minutes": {e: a.segments * seg / r / 60 for e, r in rates.items()},
                "minutes_saved": a.segments * seg * (1 / rates["cgbn"] - 1 / rates["p2sq"]) / 60}
            print(json.dumps(doc["timing"]), flush=True)
            if a.say_at is not None and rates["p2sq"] >= a.say_at:
                doc["timing"]["at_or_over"] = a.say_at
                print(f"AT OR OVER {a.say_at:g} BLOCKS A SECOND: p2sq encodes {rates['p2sq']:.1f}", flush=True)
        save()
        print(f"encode-bench: {'passed' if ended is None else 'ended at ' + ended}", flush=True)
        return 0 if ended is None else 1

    def steps() -> None:
        nonlocal today
        # 1. gates: on the first blocks, then on the segment
        doc["gates"] = {}
        for name, count in (("first_blocks", min(n, a.warm)), ("segment", n)):
            today, g = gates(count)
            doc["gates"][name] = g
            print(json.dumps({"gates": name, **g}), flush=True)
            save()
            if not g["passed"]:
                raise Ended(f"the gates on {count} blocks")
        doc["c_sha512"] = hashlib.sha512(today.cpu().numpy().tobytes()).hexdigest()

        # 2. timing
        for e in ENCODERS:
            encode(kern, "default", e, min(n, a.warm))
        kern.raised()
        for _ in range(a.rounds):
            for e in ENCODERS:
                timed("default", kern, e, n, "timing")

        # 3. the squaring's rate, and each encoder alone on the card
        if a.decode_bench:
            left = 60 * a.minutes - (time.monotonic() - began)
            if left < 60:
                raise Ended(f"{a.minutes:g} minutes, before decode_bench.py")
            cmd = [sys.executable, str(HERE / "decode_bench.py"), "--lib", a.lib, "--build-dir", str(out), "--out", str(out)]
            doc["decode_bench"] = {}
            with (out / "decode-bench.log").open("w") as log:
                try:
                    rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=False, timeout=left,
                                        preexec_fn=_with_parent).returncode
                except subprocess.TimeoutExpired:
                    raise Ended(f"{a.minutes:g} minutes, inside decode_bench.py") from None
            doc["decode_bench"]["exit"] = rc
            try:
                d = json.loads((out / "decode-bench.json").read_text())["results"][0]
                doc["decode_bench"].update(blocks=d["blocks"], kernel_s=d["kernel_s"], kernel_reps_s=d["kernel_reps_s"],
                                           squarings_per_s=2 * d["blocks"] / d["kernel_s"], gate=d["gate"]["passed"])
            except (OSError, KeyError, ValueError, IndexError):
                pass
            if rc:
                raise Ended("decode_bench.py's gate")
        for count in (1, 1024):
            for _ in range(a.small_reps):
                for e in ENCODERS:
                    timed("default", kern, e, min(count, n), "alone")

        # 4. the other builds of the default encoder: one whose bytes are not the default build's is not timed
        count = min(n, a.a_blocks)
        builds, bad = [("default", kern)], []
        doc["a_gates"] = {}
        for name, lib_dir in (x.split("=", 1) for x in a.a):
            k = Kernels(lib_dir, lib_dir, P, prefix)
            c, _ = encode(k, name, "cgbn", count)
            g = doc["a_gates"][name] = gate(c, today, k.raised())
            print(json.dumps({"a_gate": name, **g}), flush=True)
            if g["passed"]:
                builds.append((name, k))
            else:
                bad.append(name)
        if len(builds) > 1:
            for _ in range(a.rounds):
                for name, k in builds:
                    timed(name, k, "cgbn", count, "builds")
        if bad:
            raise Ended(f"the bytes of the build {', '.join(bad)}, which is not timed")

    ended: str | None = "an error"   # whatever is raised, the card's sampler stops and what was measured is written
    try:
        steps()
        ended = None
    except Ended as e:
        ended = str(e)
    finally:
        rc = finish(ended)
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--lib", action="append", default=[], metavar="NAME=DIR")
    b.add_argument("--out", required=True)
    r = sub.add_parser("reference")
    r.add_argument("--blocks", type=int, help="default: one segment's")
    r.add_argument("--picks", type=int, default=57)
    r.add_argument("--out", required=True)
    g = sub.add_parser("run")
    g.add_argument("--lib", required=True, help="the directory of the libp2v1.so that holds both encoders")
    g.add_argument("--a", action="append", default=[], metavar="NAME=DIR", help="another build of the default encoder")
    g.add_argument("--a-blocks", type=int, default=16384)
    g.add_argument("--reference", required=True)
    g.add_argument("--rounds", type=int, default=2)
    g.add_argument("--small-reps", type=int, default=3)
    g.add_argument("--warm", type=int, default=2112)
    g.add_argument("--decode-bench", action="store_true")
    g.add_argument("--segments", type=int, default=1, help="the store's segments, for the minutes a whole encode takes")
    g.add_argument("--minutes", type=float, default=25, help="no pass starts that would end later than this after the start")
    g.add_argument("--say-at", type=float, help="blocks a second of p2sq at or above which the run says so")
    g.add_argument("--out", required=True)
    a = ap.parse_args()
    return {"build": build, "reference": reference, "run": run}[a.command](a)


if __name__ == "__main__":
    sys.exit(main())
