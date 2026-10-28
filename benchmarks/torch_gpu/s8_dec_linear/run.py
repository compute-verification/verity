"""The INT8 linear over the P2 store on one GPU: `p2stage.cu`'s decode (through `stage.py`) of one layer's weights into staging
buffers, checked byte for byte, then the exact INT8 GEMM on them (`verity_vllm.engine.int8_fused.scaled_mm`, which Triton
lowers to wgmma on sm_90a), checked bit for bit against its twin (`int8_exact`: s32 sums by cuBLASLt, then
RN_bf16((f32(acc) * s_a) * s_w)), then timed.

Parts, in order; each writes JSON lines to OUT/run.jsonl and a failed gate stops the run before anything is timed:

  device     the card, its SMs and each staging variant's registers, shared memory and resident CTAs an SM
  encode     memory-accounting's encode (`p2_v1.gpu.Kernels.encode`, v4 keys) of a real INT8 tensor's blocks, timed: the encode
             rate, and that its own decode gives the rows back
  codewords  those codewords decoded by every staging variant into the tensor's staging buffer: equal to the tensor, no flag;
             sampled blocks against the CPU reference
  gate       one layer of `--shape` (QKV, O, up, down back to back from an unaligned W-offset) on random canonical rows, edge
             values first: every staged byte of every variant against the library's decode, on the device; sampled blocks
             against the CPU reference (`fixture.check_rows`); keys given as rows (`decode_keys`) against keys expanded in the
             kernel; grids of 1, 3 and 133 CTAs; a c >= p row flagged and its neighbours still right; and two controls that
             must be refused, no write and the next blocks' keys
  linear     int8_fused on each staged matrix at every `--m`, every output bit against the twin, and int8_fused's activation
             quantization against the twin's
  time       at each `--time-m`, interleaved rep by rep after a warm-up, with each rep's SM clock and energy (NVML's
             counter) where NVML gives them: each variant's staging decode of each matrix alone, and libp2v1's (keys, then
             decode, then the payload bytes copied into staging); the plain GEMM on plaintext (int8_fused, and cuBLASLt's s32
             product); and a layer three ways: plain (GEMMs only), staged (decode, then GEMM, one stream) and overlapped (the
             next matrix's decode on a second stream beside the current GEMM, a two-buffer ring's dependencies); and from the
             energies, the least time any overlap could take at the card's power limit

  python run.py --ship SHIP --out OUT [--parts encode,codewords,gate,linear,time] [--m 1 64 4096 20480]
                [--time-m 20480 4096] [--stage-variant w4]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fixture as F
from stage import Stage

sys.path.insert(0, str(F.ROOT / "integrations" / "vllm"))
from p2_v1.gpu import Kernels
from verity_vllm.engine import int8_exact as X
from verity_vllm.engine import int8_fused as T

SHAPES = {"mqa16k": (16384, 64, 65536)}   # compute-accounting's H200 model: width, KV rows (one head of 64), MLP
SALT = hashlib.sha256(b"verity/s8-dec-linear/salt").digest()[:24]
POISON = 0xA5


class Log:
    def __init__(self, out: Path) -> None:
        out.mkdir(parents=True, exist_ok=True)
        self.f = (out / "run.jsonl").open("a")

    def __call__(self, doc: dict) -> dict:
        doc = {"t": round(time.time(), 3), **doc}
        line = json.dumps(doc)
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()
        return doc


class Device:
    """Slices of a device byte buffer as bytes, for `fixture.check_rows`."""

    def __init__(self, t: torch.Tensor) -> None:
        self.t = t

    def __getitem__(self, s: slice) -> bytes:
        return bytes(self.t[s].cpu().numpy())


def poisoned(n: int, dev: torch.device) -> torch.Tensor:
    return torch.full((n,), POISON, dtype=torch.uint8, device=dev)


def row_of(v: int) -> torch.Tensor:
    return torch.frombuffer(bytearray(v.to_bytes(F.ROW, "big")), dtype=torch.uint8)


def edge_values() -> list[int]:
    p = F.SPEC.p
    return [0, 1, 2, 3, p - 1, p - 2, (p - 1) // 2, (1 << 16447) - 1, 1 << 16447, p - (1 << 32)]


def random_rows(n: int, g: torch.Generator, dev: torch.device) -> torch.Tensor:
    c = torch.randint(0, 256, (n, F.ROW), dtype=torch.uint8, device=dev, generator=g)
    c[:, 0].bitwise_and_(0x7F)
    return c


def mismatches(a: torch.Tensor, b: torch.Tensor) -> tuple[int, list[int]]:
    d = a != b
    n = int(d.sum())
    return n, (torch.nonzero(d)[:4].flatten().tolist() if n else [])


def cpu_check(pfx: bytes, base: int, length: int, c: torch.Tensor, first: int, dst: torch.Tensor, count: int) -> dict:
    picks = F.sample(F.blocks_of(base, length), count)
    rows = {j: bytes(c[j - first].cpu().numpy()) for j in picks}
    bad = F.check_rows(pfx, base, length, rows, Device(dst))
    return {"cpu_blocks": len(picks), "cpu_bad": bad, "cpu_flags": F.refusals(pfx, rows)}


def part_device(st: dict[str, Stage], log: Log) -> None:
    p = torch.cuda.get_device_properties(0)
    log({"part": "device", "name": p.name, "sms": p.multi_processor_count, "cc": f"{p.major}.{p.minor}",
         "memory_gib": round(p.total_memory / 2**30, 1), "torch": torch.__version__, "cuda": torch.version.cuda,
         "variants": {n: {**s.attributes(True), "plain": s.attributes(False)} for n, s in st.items()}})


def part_encode_codewords(kern: Kernels, st: dict[str, Stage], pfx: bytes, dev: torch.device, g: torch.Generator, log: Log,
                          reps: int) -> bool:
    n_rows, k = 256, 16384                   # a 4 MiB INT8 tensor at an unaligned W-offset, others' bytes around it
    base, length = F.PAYLOAD * 40_000 + 777, n_rows * k
    blocks = F.blocks_of(base, length)
    first, n = blocks.start, len(blocks)
    w = torch.randint(-128, 128, (n_rows, k), dtype=torch.int8, device=dev, generator=g)
    payload = torch.randint(0, 256, (n * F.PAYLOAD,), dtype=torch.uint8, device=dev, generator=g)
    off = base - F.PAYLOAD * first
    payload[off:off + length] = w.view(torch.uint8).view(-1)
    m = torch.zeros(n, F.ROW, dtype=torch.uint8, device=dev)
    m[:, 2:] = payload.view(n, F.PAYLOAD)
    kern.raised()
    kern.encode(m[:16].contiguous(), first)
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        c = kern.encode(m, first)
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b) / 1e3)
    enc_flags = kern.raised()
    back = kern.decode(c, first)
    round_trip = bool(torch.equal(back, m))
    s = statistics.median(times)
    log({"part": "encode", "blocks": n, "reps": times, "encode_s": s, "encode_GBps": n * F.PAYLOAD / s / 1e9,
         "encode_MBps": n * F.PAYLOAD / s / 1e6, "flags": enc_flags, "round_trip": round_trip, "keys": "v4"})
    ok = round_trip and enc_flags == 0
    for name, sv in st.items():
        dst = poisoned(length, dev)
        sv.flags()
        sv.decode(c, first, pfx, base, length, dst)
        flags = sv.flags()
        bad, at = mismatches(dst, w.view(torch.uint8).view(-1))
        cpu = cpu_check(pfx, base, length, c, first, dst, 8)
        good = bad == 0 and flags == 0 and not cpu["cpu_bad"] and cpu["cpu_flags"] == 0
        log({"part": "codewords", "variant": name, "N": n_rows, "K": k, "base": base, "blocks": n, "mismatches": bad,
             "first_bad": at, "flags": flags, **cpu, "passed": good})
        ok = ok and good
    return ok


def part_gate(kern: Kernels, st: dict[str, Stage], pfx: bytes, tensors: list, dev: torch.device, g: torch.Generator,
              log: Log) -> tuple[bool, dict]:
    start, total = tensors[0][3], sum(t[4] for t in tensors)
    blocks = F.blocks_of(start, total)
    first, n = blocks.start, len(blocks)
    c = random_rows(n, g, dev)
    for i, v in enumerate(edge_values()):
        c[i] = row_of(v).to(dev)
    kern.raised()
    m = kern.decode(c, first)
    ref_flags = kern.raised()
    plain = m[:, 2:].contiguous().view(-1)
    del m

    def ref(base: int, length: int) -> torch.Tensor:
        o = base - F.PAYLOAD * first
        return plain[o:o + length]

    def rows(base: int, length: int) -> tuple[torch.Tensor, int]:
        b = F.blocks_of(base, length)
        return c[b.start - first:b.stop - first], b.start

    ok, staged = ref_flags == 0, {}
    log({"part": "gate", "what": "reference", "blocks": n, "first": first, "W_bytes": total, "flags": ref_flags})
    for name, sv in st.items():
        for tname, nn, kk, base, length in tensors:
            dst = poisoned(length, dev)
            cr, j0 = rows(base, length)
            sv.flags()
            t0 = time.time()
            sv.decode(cr, j0, pfx, base, length, dst)
            torch.cuda.synchronize()
            wall = time.time() - t0
            flags = sv.flags()
            bad, at = mismatches(dst, ref(base, length))
            cpu = cpu_check(pfx, base, length, cr, j0, dst, 12)
            # random canonical rows: no c >= p, and the device flags every non-codeword the CPU sample found
            good = bad == 0 and not flags & F.FLAG_C and not cpu["cpu_bad"] and \
                bool(flags & F.FLAG_WIDE or not cpu["cpu_flags"] & F.FLAG_WIDE)
            log({"part": "gate", "what": "layer", "variant": name, "tensor": tname, "N": nn, "K": kk, "base": base,
                 "blocks": cr.shape[0], "mismatches": bad, "first_bad": at, "flags": flags, **cpu, "wall_s": round(wall, 4),
                 "passed": good})
            ok = ok and good
            if name == next(iter(st)):
                staged[tname] = dst
            else:
                del dst
    sv = next(iter(st.values()))
    tname, nn, kk, base, length = tensors[1]
    cr, j0 = rows(base, length)
    k, t = kern.keys(j0, cr.shape[0])
    dst = poisoned(length, dev)
    sv.flags()
    sv.decode_keys(cr, k, t, j0, base, length, dst)
    bad, at = mismatches(dst, ref(base, length))
    flags = sv.flags()
    del k, t
    good = bad == 0 and flags & F.FLAG_C == 0
    log({"part": "gate", "what": "keys-as-rows", "tensor": tname, "blocks": cr.shape[0], "mismatches": bad, "first_bad": at,
         "flags": flags, "passed": good})
    ok = ok and good
    base, length = tensors[1][3] + 5, 3000 * F.PAYLOAD + 11
    cr, j0 = rows(base, length)
    for grid in (1, 3, 133):
        dst = poisoned(length, dev)
        sv.decode(cr, j0, pfx, base, length, dst, grid=grid)
        bad, at = mismatches(dst, ref(base, length))
        log({"part": "gate", "what": "grid", "grid": grid, "blocks": cr.shape[0], "mismatches": bad, "first_bad": at,
             "flags": sv.flags(), "passed": bad == 0})
        ok = ok and bad == 0
    j0 = first + 100
    base, length = F.PAYLOAD * j0, 3 * F.PAYLOAD
    cr = c[100:103].clone()
    cr[1] = 0xFF
    dst = poisoned(length, dev)
    sv.decode(cr, j0, pfx, base, length, dst)
    flags = sv.flags()
    r = ref(base, length)
    bad = mismatches(dst[:F.PAYLOAD], r[:F.PAYLOAD])[0] + mismatches(dst[2 * F.PAYLOAD:], r[2 * F.PAYLOAD:])[0]
    good = bool(flags & F.FLAG_C) and bad == 0
    log({"part": "gate", "what": "c>=p", "flags": flags, "neighbour_mismatches": bad, "passed": good})
    ok = ok and good
    tname, nn, kk, base, length = tensors[1]
    cr, j0 = rows(base, length)
    dst = poisoned(length, dev)
    nowrite = mismatches(dst, ref(base, length))[0]
    sv.decode(cr, j0 + 1, pfx, base + F.PAYLOAD, length, dst)
    sv.flags()
    shifted = mismatches(dst, ref(base, length))[0]
    good = nowrite > 0 and shifted > 0
    log({"part": "gate", "what": "controls", "no_write_mismatches": nowrite, "next_keys_mismatches": shifted,
         "refused": good, "passed": good})
    ok = ok and good
    return ok, staged


def part_linear(staged: dict, tensors: list, ms: list[int], dev: torch.device, g: torch.Generator, log: Log) -> bool:
    ok = True
    for tname, nn, kk, base, length in tensors:
        w = staged[tname].view(torch.int8).view(nn, kk)
        s_w = torch.rand(nn, dtype=torch.float32, device=dev, generator=g) * 1e-2 + 1e-4
        for m in ms:
            x = (torch.randn(m, kk, dtype=torch.float32, device=dev, generator=g) * 3).to(torch.bfloat16)
            xq, s_a = T.row_quant(x)
            xq2, s_a2 = X.quantize_rows(x)
            quant_ok = bool(torch.equal(xq, xq2)) and bool(torch.equal(s_a.view(torch.int32), s_a2.view(torch.int32)))
            y = T.product(xq, s_a, w, s_w)
            twin = X.dequantize(X.int_mm(xq, w), s_a, s_w)
            bad, at = mismatches(y.view(torch.int16), twin.view(torch.int16))
            cfg = str(T.pick(m, nn, kk))
            log({"part": "linear", "tensor": tname, "M": m, "N": nn, "K": kk, "cfg": cfg, "mismatches": bad,
                 "first_bad": at, "quant_equal": quant_ok, "passed": bad == 0})
            ok = ok and bad == 0
            del x, xq, xq2, y, twin
        torch.cuda.empty_cache()
    return ok


def sm_clock() -> int | None:
    try:
        return int(torch.cuda.clock_rate(0))
    except Exception:  # noqa: BLE001 (NVML absent: the job's nvidia-smi log has the clocks)
        return None


class Energy:
    """The card's energy counter (NVML's total energy since the driver loaded, in J) and its enforced power limit (W);
    both None without NVML."""

    def __init__(self) -> None:
        self.nv = self.limit_w = self.missing = None
        try:
            import pynvml
            pynvml.nvmlInit()
            uuid = str(torch.cuda.get_device_properties(0).uuid)
            self.h = pynvml.nvmlDeviceGetHandleByUUID(uuid if uuid.startswith("GPU-") else f"GPU-{uuid}")
            pynvml.nvmlDeviceGetTotalEnergyConsumption(self.h)
            self.limit_w = pynvml.nvmlDeviceGetEnforcedPowerLimit(self.h) / 1e3
            self.nv = pynvml
        except Exception as e:  # noqa: BLE001 (no NVML or no energy counter: the arms carry no joules)
            self.missing = repr(e)

    def read(self) -> float | None:
        return self.nv.nvmlDeviceGetTotalEnergyConsumption(self.h) / 1e3 if self.nv else None


def part_time(kern: Kernels, st: dict[str, Stage], sv0: Stage, pfx: bytes, tensors: list, staged: dict, tm: int, reps: int,
              layers: int, dev: torch.device, g: torch.Generator, log: Log) -> None:
    start, total = tensors[0][3], sum(t[4] for t in tensors)
    blocks = F.blocks_of(start, total)
    first = blocks.start
    c = random_rows(len(blocks), g, dev)
    nxt = {t[0]: torch.empty(t[4], dtype=torch.uint8, device=dev) for t in tensors}
    mats = []
    for tname, nn, kk, base, length in tensors:
        b = F.blocks_of(base, length)
        x = (torch.randn(tm, kk, dtype=torch.float32, device=dev, generator=g) * 3).to(torch.bfloat16)
        xq, s_a = T.row_quant(x)
        del x
        mats.append({"name": tname, "N": nn, "K": kk, "base": base, "length": length, "c": c[b.start - first:b.stop - first],
                     "j0": b.start, "off": base - F.PAYLOAD * b.start,
                     "w": staged[tname].view(torch.int8).view(nn, kk), "xq": xq, "s_a": s_a,
                     "s_w": torch.rand(nn, dtype=torch.float32, device=dev, generator=g) * 1e-2 + 1e-4,
                     "y": torch.empty(tm, nn, dtype=torch.bfloat16, device=dev), "plan": T._plan(tm, nn, kk, dev)})
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    s_gemm, s_dec = torch.cuda.Stream(dev), torch.cuda.Stream(dev)
    s_hi = torch.cuda.Stream(dev, priority=-1)
    en = Energy()

    def decode(mt: dict, sv: Stage = sv0, grid: int = 0, stream=None) -> None:
        sv.decode(mt["c"], mt["j0"], pfx, mt["base"], mt["length"], nxt[mt["name"]], grid=grid, stream=stream)

    def decode_p2v1(mt: dict, grid: int = 0, stream=None) -> None:
        # memory-accounting's two kernels (keys, then decode) on the tensor's blocks, then their payload bytes into staging
        with torch.cuda.stream(stream or torch.cuda.current_stream(dev)):
            m = kern.decode(mt["c"], mt["j0"])
            nxt[mt["name"]].copy_(m[:, 2:].reshape(-1)[mt["off"]:mt["off"] + mt["length"]])

    def gemm(mt: dict) -> None:
        T.scaled_mm(mt["xq"], mt["s_a"], mt["w"], mt["s_w"], mt["y"], *mt["plan"])

    def lt(mt: dict) -> None:
        X.int_mm(mt["xq"], mt["w"])

    seq = mats * layers

    def layer_plain() -> None:
        for mt in seq:
            gemm(mt)

    def layer_staged(dec=decode) -> None:
        for mt in seq:
            dec(mt)
            gemm(mt)

    def mark(s: torch.cuda.Stream) -> torch.cuda.Event:
        e = torch.cuda.Event()
        e.record(s)
        return e

    def layer_overlap(grid: int = 0, gemm_stream: torch.cuda.Stream = s_gemm, dec=decode) -> None:
        # a two-buffer ring: decode i + 1 runs beside GEMM i, and may start only once GEMM i - 1 has released its buffer
        cur = torch.cuda.current_stream(dev)
        gemm_stream.wait_stream(cur)
        s_dec.wait_stream(cur)
        dec(seq[0], grid=grid, stream=s_dec)
        dec_done, gemm_done = [mark(s_dec)], []
        for i in range(len(seq)):
            if i + 1 < len(seq):
                if i >= 1:
                    s_dec.wait_event(gemm_done[i - 1])
                dec(seq[i + 1], grid=grid, stream=s_dec)
                dec_done.append(mark(s_dec))
            gemm_stream.wait_event(dec_done[i])
            with torch.cuda.stream(gemm_stream):
                gemm(seq[i])
            gemm_done.append(mark(gemm_stream))
        cur.wait_stream(gemm_stream)
        cur.wait_stream(s_dec)

    arms: dict[str, tuple] = {}
    decoders = {**{name: (lambda mt, sv=sv: decode(mt, sv)) for name, sv in st.items()}, "p2v1": decode_p2v1}
    for name, fn in decoders.items():
        for mt in mats:
            arms[f"decode:{name}:{mt['name']}"] = (lambda mt=mt, fn=fn: fn(mt), "decode", mt["length"])
    for mt in mats:
        arms[f"gemm:{mt['name']}"] = (lambda mt=mt: gemm(mt), "gemm", 2 * tm * mt["N"] * mt["K"])
        arms[f"lt:{mt['name']}"] = (lambda mt=mt: lt(mt), "gemm", 2 * tm * mt["N"] * mt["K"])
    layer_bytes = sum(mt["length"] for mt in mats)
    staged_arms = {"layer:staged": decode, "layer:staged-p2v1": decode_p2v1}
    overlaps = {"layer:overlap": {}, "layer:overlap-g2": {"grid": 2 * sms}, "layer:overlap-hi": {"gemm_stream": s_hi},
                "layer:overlap-p2v1": {"dec": decode_p2v1}}
    arms["layer:plain"] = (layer_plain, "layer", layer_bytes * layers)
    for n, fn in staged_arms.items():
        arms[n] = (lambda fn=fn: layer_staged(fn), "layer", layer_bytes * layers)
    for n, kw in overlaps.items():
        arms[n] = (lambda kw=kw: layer_overlap(**kw), "layer", layer_bytes * layers)
    for fn, _, _ in arms.values():
        fn()
    torch.cuda.synchronize()
    names = list(arms)
    got: dict[str, list] = {n: [] for n in names}
    clocks: dict[str, list] = {n: [] for n in names}
    joules: dict[str, list] = {n: [] for n in names}
    for r in range(reps):
        order = names[r % len(names):] + names[:r % len(names)]
        for n in order:
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0 = en.read()
            a.record()
            arms[n][0]()
            b.record()
            b.synchronize()
            e1 = en.read()
            got[n].append(a.elapsed_time(b) / 1e3)
            clocks[n].append(sm_clock())
            joules[n].append(None if e0 is None else e1 - e0)
    med = {n: statistics.median(v) for n, v in got.items()}
    # NVML's counter steps about every 100 ms (H200, driver 580), so one rep's reading is a whole number of steps: the mean
    # over reps (whose phases against the steps are spread) estimates an arm's energy, and only for arms of several steps
    jmean = {n: statistics.fmean(v) if v[0] is not None else None for n, v in joules.items()}
    for n in names:
        kind, size = arms[n][1], arms[n][2]
        rate = {"decode": {"GBps": size / med[n] / 1e9}, "gemm": {"TOPS": size / med[n] / 1e12},
                "layer": {"W_GBps": size / med[n] / 1e9}}[kind]
        power = {} if jmean[n] is None else {"J": jmean[n], "W": jmean[n] / statistics.fmean(got[n])}
        log({"part": "time", "arm": n, "M": tm, "median_s": med[n], "reps": got[n], "sm_mhz": clocks[n], "joules": joules[n],
             **power, **rate})
    per = {n: med[n] / layers for n in names if n.startswith("layer:")}
    plain = per["layer:plain"]
    lt_layer = sum(med[f"lt:{mt['name']}"] for mt in mats)
    best = min(plain, lt_layer)
    dec = {name: sum(med[f"decode:{name}:{mt['name']}"] for mt in mats) for name in decoders}
    stage_name = next(n for n, sv in st.items() if sv is sv0)
    summary = {"part": "time", "arm": "summary", "M": tm, "pipelined_layers": layers, "model_layers": 52,
               "staging_variant": stage_name, "power_limit_W": en.limit_w, "energy_missing": en.missing,
               "layer_plain_s": plain, "layer_lt_s": lt_layer, "plain_divisor": "int8_fused" if plain <= lt_layer else "cuBLASLt s32",
               "decode_layer_s": dec, "decode_layer_GBps": {k: layer_bytes / v / 1e9 for k, v in dec.items()}}
    for n in (*staged_arms, *overlaps):
        base_arm = "layer:staged-p2v1" if n.endswith("p2v1") else "layer:staged"
        hide = min(plain, dec["p2v1" if n.endswith("p2v1") else stage_name])
        summary[n] = {"layer_s": per[n], "slowdown_vs_plain": per[n] / best, "step_s": 52 * per[n],
                      "hidden": (per[base_arm] - per[n]) / hide}
    if en.limit_w and jmean["layer:plain"] is not None:
        # at the power limit a layer takes at least its GEMMs' and its decode's energy over the limit, however overlapped
        # (an estimate: each arm's energy is at its own clocks, and both carry the card's static power); the decode's
        # energy is the staged layer's less the plain one's, the layer arms being the only ones many counter steps long
        j_plain = jmean["layer:plain"] / layers
        for name, arm in ((stage_name, "layer:staged"), ("p2v1", "layer:staged-p2v1")):
            j_dec = jmean[arm] / layers - j_plain
            floor = max(plain, dec[name], (j_plain + j_dec) / en.limit_w)
            serial = plain + dec[name]
            summary[f"power_floor:{name}"] = {
                "gemm_J": j_plain, "gemm_W": j_plain / plain, "decode_J": j_dec, "decode_W": j_dec / dec[name],
                "decode_J_per_GB": j_dec / (layer_bytes / 1e9), "floor_layer_s": floor, "serial_layer_s": serial,
                "most_hidden": (serial - floor) / min(plain, dec[name])}
    log(summary)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ship", required=True, help="build.sh's tree: lib/<variant>/libp2stage.so and p2v1/libp2v1.so")
    ap.add_argument("--out", required=True)
    ap.add_argument("--parts", default="encode,codewords,gate,linear,time")
    ap.add_argument("--variants", default="", help="comma-separated lib/ variants (default: every one shipped)")
    ap.add_argument("--shape", default="mqa16k", choices=sorted(SHAPES))
    ap.add_argument("--m", type=int, nargs="+", default=[1, 64, 4096, 20480])
    ap.add_argument("--time-m", type=int, nargs="+", default=[20480, 4096])
    ap.add_argument("--stage-variant", default="w4", help="the staging variant the layer arms decode with (default w4, "
                    "else the first shipped)")
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--time-layers", type=int, default=3, help="layers a timed layer arm runs back to back (one unhidden "
                    "prologue decode among them)")
    ap.add_argument("--encode-reps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20261026)
    a = ap.parse_args()
    out, ship = Path(a.out), Path(a.ship)
    log = Log(out)
    dev = torch.device("cuda", 0)
    torch.cuda.set_device(dev)
    g = torch.Generator(device=dev)
    g.manual_seed(a.seed)
    names = a.variants.split(",") if a.variants else sorted(p.name for p in (ship / "lib").iterdir())
    st = {n: Stage(ship / "lib" / n, dev) for n in names}
    pfx = F.prefix(SALT)
    kern = Kernels(str(ship / "p2v1"), str(out), F.SPEC, pfx)
    d, kv, mlp = SHAPES[a.shape]
    tensors = F.packed(F.layer_tensors(d, kv, mlp), start=F.PAYLOAD * 3 + 1001)
    parts = a.parts.split(",")
    part_device(st, log)
    ok = True
    if "encode" in parts or "codewords" in parts:
        ok = part_encode_codewords(kern, st, pfx, dev, g, log, a.encode_reps) and ok
    staged: dict = {}
    if ok and {"gate", "linear", "time"} & set(parts):
        good, staged = part_gate(kern, st, pfx, tensors, dev, g, log)
        ok = ok and good
    if ok and "linear" in parts:
        ok = part_linear(staged, tensors, a.m, dev, g, log) and ok
    if ok and "time" in parts:
        sv0 = st.get(a.stage_variant) or next(iter(st.values()))
        for tm in a.time_m:
            part_time(kern, st, sv0, pfx, tensors, staged, tm, a.reps, a.time_layers, dev, g, log)
            torch.cuda.empty_cache()
    log({"part": "end", "passed": ok, "parts": parts})
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
