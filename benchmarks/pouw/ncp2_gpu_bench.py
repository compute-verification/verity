"""`ncp-v2` on one RTX 4090: every device path gated against `pouw_native`, then timed against BF16, hashing included.

    python benchmarks/pouw/ncp2_gpu_bench.py --dry-run --out DRY.json   (CPU: the hashed-byte accounting and the host gate)
    python benchmarks/pouw/ncp2_gpu_bench.py --out RESULT.json          (on the pod: torch with CUDA, nvcc; honest calls only)

Gates, before any timing (a failing gate stops the run and names its first difference):
- the device's SHA-256, SHA-512, SHAKE256 and TurboSHAKE128 against hashlib and the host build of `pouw_cuda/ncp2.cuh`,
  whose TurboSHAKE128 meets RFC 9861's vectors;
- the device committer (`pouw_cuda/ncp2_kernels.cu`) against `pouw_native`, leaf for leaf, on `pouw_device.gate_calls`
  with Qwen2.5-0.5B's four linears (`pouw_device.first_difference`);
- #218's route-U GEMM (`kernels/pouw_gemm.cu` `ncp_route_u`) on `ncp-v2`'s own X and Y against `pouw_native`'s tile words.

Timings, for each Qwen2.5-0.5B linear at m = 16, 256 and 2,048 rows:
- ``bf16``: ``a @ w.T`` in BF16 (cuBLAS), what serving runs;
- ``route_u``: the (m, 3k, n) u8 × s8 GEMM from c₀ with every 16-deep checkpoint written out: the GEMM floor, unfused;
- ``leaves``: the tile word leaves (3k/16 · 256 words each) and the dequantization word leaves (one word each), hashed from
  device memory under SHA-256 (what the MVP commits), SHA-512 (A22's decided leaf) and TurboSHAKE128 (A22's
  recommendation); and each hash's in-register ceiling (`ncp2_ceiling`), which a kernel hashing its checkpoints as it
  makes them approaches;
- ``f1``: F₁, SHAKE256 for k bytes per weight row, which every call re-derives;
- ``committer``: the correctness-first device committer end to end at m = 16 (its checkpoints go through memory).

The projection per forward (24 layers): ``unfused`` = route_u + every leaf hashed at the from-memory rate + F₁, and
``fused`` = max(route_u, the hashed bytes at the ceiling) + F₁, each over BF16's linears. Hashing's place in the cost model
is settled elsewhere; this measures its throughput.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from verity_vllm.protocol_options import pouw_device as PD
from verity_vllm.protocol_options import pouw_native as PN

LAYERS = 24
ROWS = (16, 256, 2048)
D = 16
UNIT_TAG, NODE_TAG = 27, 33
#: per hash: the block (SHA-2) or rate (sponge) its compression or permutation absorbs, the fewest padding bytes, and
#: `pouw_device`'s kind code
HASH = {"sha256": (64, 9, 0), "sha512": (128, 17, 1), "turboshake128": (168, 1, 3)}
#: the in-register ceiling's kind codes: the compressions, and Keccak-p at SHAKE256's and TurboSHAKE128's round counts
CEILING = {"sha256": 0, "sha512": 1, "shake256": 2, "turboshake128": 3}
SHAKE_RATE = 136


def absorbed(kind: str, length: int) -> int:
    """Bytes a hash's compressions or permutations absorb for one message of ``length`` bytes."""
    block, pad, _code = HASH[kind]
    return -(-(length + pad) // block) * block


def tree_nodes(population: int) -> int:
    """`pouw_circuit.units_root`'s node hashes over ``population`` leaves (an odd last node rises unpaired)."""
    nodes, count = 0, population
    while count > 1:
        nodes += count // 2
        count = (count + 1) // 2
    return nodes


def call_accounting(m: int, k: int, n: int) -> dict:
    """One call's hashed messages, as (count, bytes each), by what they commit; and its XOF bytes."""
    ms, ns, steps = m // D, n // D, 3 * k // D
    population = 3 * ms + ns + ms * ns + m * n
    return {
        "macs": m * k * n,
        "messages": {
            "tile words": (ms * ns, UNIT_TAG + 4 * steps * D * D),
            "dequant words": (m * n, UNIT_TAG + 4),
            "form_y strips (committed per call)": (ns, UNIT_TAG + 4 * (D * 3 * k + D)),
            "form_x strips": (ms, UNIT_TAG + 4 * D * 3 * k),
            "units tree": (tree_nodes(population), NODE_TAG + 64),
        },
        "xof": {"F1": n * k, "E1": m * k},
        "units": population,
    }


def forward_accounting(m: int) -> dict:
    """A 24-layer Qwen2.5-0.5B forward of ``m`` rows: bytes each hash absorbs, by part, and the XOF."""
    out: dict = {"m": m, "macs": 0, "units": 0, "absorbed": {h: {} for h in HASH}, "xof_bytes": 0}
    for k, n in PD.QWEN05:
        acc = call_accounting(m, k, n)
        out["macs"] += LAYERS * acc["macs"]
        out["units"] += LAYERS * acc["units"]
        out["xof_bytes"] += LAYERS * sum(acc["xof"].values())
        for part, (count, length) in acc["messages"].items():
            for h in HASH:
                out["absorbed"][h][part] = out["absorbed"][h].get(part, 0) + LAYERS * count * absorbed(h, length)
    for h in HASH:
        out["absorbed"][h]["total"] = sum(out["absorbed"][h].values())
    out["tile_bytes_per_mac"] = out["absorbed"]["sha256"]["tile words"] / out["macs"]
    return out


def host_gate() -> list[dict]:
    """The host build against `pouw_native` on every gate call: the gate the pod repeats for the device."""
    out = []
    for name, call in PD.gate_calls(qwen=True):
        t0 = time.perf_counter()
        diff = PD.first_difference(PN.commit_call(0, "ncp-v2", call), PD.commit_call(0, "ncp-v2", call))
        out.append({"call": name, "difference": diff, "s": round(time.perf_counter() - t0, 3)})
        if diff:
            raise SystemExit(f"host gate: {name}: {diff}")
    return out


# ---- on the pod --------------------------------------------------------------------------------------------------------------

def gate_hashes() -> dict:
    msgs = [bytes((i * 7 + n) % 256 for i in range(n)) for n in (0, 1, 27, 31, 55, 64, 111, 128, 135, 136, 167, 168, 169, 700)]
    ref = {"sha256": [hashlib.sha256(m).digest() for m in msgs], "sha512": [hashlib.sha512(m).digest() for m in msgs],
           "shake256": [hashlib.shake_256(m).digest(200) for m in msgs], "turboshake128": PD.hash_many("turboshake128", msgs, 200)}
    for kind, want in ref.items():
        got = PD.hash_many(kind, msgs, 200, on="device")
        bad = [len(m) for m, g, w in zip(msgs, got, want) if g != w]
        if bad:
            raise SystemExit(f"device {kind} differs at message lengths {bad}")
    return {kind: "pass" for kind in ref}


def gate_committer() -> list[dict]:
    out = []
    for name, call in PD.gate_calls(qwen=True):
        got, ms = PD.commit_call_timed(0, "ncp-v2", call)
        diff = PD.first_difference(PN.commit_call(0, "ncp-v2", call), got)
        sh = call.shape
        out.append({"call": name, "shape": [sh.m, sh.k, sh.n], "difference": diff, "device_ms": round(ms, 3)})
        if diff:
            raise SystemExit(f"device committer: {name}: {diff}")
    return out


def _cfg(m: int) -> int:
    return 2 if m < 64 else 0


def gate_route_u(ext, torch) -> list[dict]:
    out = []
    for name, call in PD.gate_calls(qwen=True):
        if not name.startswith("qwen05"):
            continue
        nat, sh = PN.Native(call), call.shape
        x = np.concatenate([nat.x_strip(rs) for rs in range(sh.m // D)])
        b, _t = nat.weight(0, sh.n)
        y, c0 = nat.y_block(b, 0)
        want = np.concatenate([nat.tiles(nat._x64(nat.x_strip(rs)), y, c0) for rs in range(sh.m // D)], axis=1)   # [t, m, n]
        cz, ck = ext.ncp_route_u(torch.from_numpy(x).cuda(), torch.from_numpy(y).cuda(), _cfg(sh.m))
        got = ck.cpu().numpy().view(np.uint32).transpose(2, 0, 1)
        ok = np.array_equal(got, want) and np.array_equal(cz.cpu().numpy().view(np.uint32), want[-1])
        out.append({"call": name, "checkpoints_equal": bool(ok)})
        if not ok:
            t, r, c = (int(v) for v in np.argwhere(got != want)[0])
            raise SystemExit(f"route U: {name}: checkpoint {t} of ({r}, {c}) is {got[t, r, c]:#x}, not {want[t, r, c]:#x}")
    return out


#: the most leaves one timing hashes, and the most bytes of words it reads: past a few waves of the GPU's threads the time is
#: linear in the leaf count, and past a few blocks linear in the message length, so a larger call is timed within these and
#: scaled (a call's own count is timed as it is when it is smaller: fewer leaves than the GPU has lanes is part of its time)
TIMED_UNITS = {"tile words": 1 << 15, "dequant words": 1 << 22}
TIMED_BYTES = 2 << 30
#: the saturated from-memory rate: this many leaves of this many words each (64 KiB messages)
RATE_UNITS, RATE_WORDS = 1 << 15, 1 << 14


def _leaf_ms(dev, code: int, count: int, words: int, most: int, reps: int) -> float:
    """``count`` leaves of ``words`` words each, under hash ``code``: timed on at most ``most`` leaves and `TIMED_BYTES` of words,
    and scaled to the call."""
    units = min(count, most)
    timed_words = min(words, max(1, TIMED_BYTES // (4 * units)))
    ms = ctypes.c_float(0)
    if rc := dev.ncp2_leaf_time(code, units, timed_words, reps, ctypes.byref(ms)):
        raise SystemExit(f"leaf hash {code}: CUDA error {rc}")
    return ms.value * count / units * (UNIT_TAG + 4 * words) / (UNIT_TAG + 4 * timed_words)


def timings(ext, torch, reps: int) -> dict:
    from gemm_bench import timed

    dev, out = PD.device(), {"linears": [], "ceiling_bytes_per_s": {}, "from_memory_bytes_per_s": {}}
    for kind, code in CEILING.items():
        rate = ctypes.c_double(0)
        if rc := dev.ncp2_ceiling(code, 128 * 24, 256, 256, ctypes.byref(rate)):
            raise SystemExit(f"ceiling {kind}: CUDA error {rc}")
        out["ceiling_bytes_per_s"][kind] = rate.value
    for h, (_block, _pad, code) in HASH.items():
        ms = _leaf_ms(dev, code, RATE_UNITS, RATE_WORDS, RATE_UNITS, reps)
        out["from_memory_bytes_per_s"][h] = RATE_UNITS * absorbed(h, UNIT_TAG + 4 * RATE_WORDS) / (ms * 1e-3)
    for m in ROWS:
        for k, n in PD.QWEN05:
            row = {"m": m, "k": k, "n": n}
            a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
            row["bf16_ms"] = timed(partial(torch.mm, a, w.T), reps)
            xu = torch.randint(0, 256, (m, 3 * k), device="cuda", dtype=torch.uint8)
            ys = torch.randint(0, 256, (n, 3 * k), device="cuda", dtype=torch.uint8)
            row["route_u_ms"] = timed(partial(ext.ncp_route_u, xu, ys, _cfg(m)), reps)
            del a, w, xu, ys
            torch.cuda.empty_cache()
            acc = call_accounting(m, k, n)
            for h, (_block, _pad, code) in HASH.items():
                for part, most in TIMED_UNITS.items():
                    count, length = acc["messages"][part]
                    row[f"{h} {part} ms"] = _leaf_ms(dev, code, count, (length - UNIT_TAG) // 4, most, reps)
            ms = ctypes.c_float(0)
            if rc := dev.ncp2_f1_time(n, k, reps, ctypes.byref(ms)):
                raise SystemExit(f"F1: CUDA error {rc}")
            row["f1_ms"] = ms.value
            out["linears"].append(row)
    return out


def projection(t: dict) -> dict:
    """Per forward and m, over BF16's linears: ``unfused``, the route-U GEMM plus every message hashed at the rate measured
    from memory for its class (long messages at the tile leaves' bytes per second, short ones at the dequantization leaves'
    time per block) plus F₁; ``fused``, the GEMM or every absorbed byte at the in-register ceiling, whichever is longer,
    plus F₁. Under each leaf hash."""
    out = {}
    for m in ROWS:
        rows = [r for r in t["linears"] if r["m"] == m]
        bf16 = LAYERS * sum(r["bf16_ms"] for r in rows)
        gemm = LAYERS * sum(r["route_u_ms"] for r in rows)
        f1 = LAYERS * sum(r["f1_ms"] for r in rows)
        acc = forward_accounting(m)
        per: dict = {"bf16_ms": bf16, "route_u_ms": gemm, "f1_ms": f1}
        for h in HASH:
            ab = acc["absorbed"][h]
            tile = LAYERS * sum(r[f"{h} tile words ms"] for r in rows)
            deq = LAYERS * sum(r[f"{h} dequant words ms"] for r in rows)
            per_block = deq / (ab["dequant words"] / HASH[h][0])
            rest = (ab["form_y strips (committed per call)"] + ab["form_x strips"]) / ab["tile words"] * tile + ab["units tree"] / HASH[h][0] * per_block
            unfused = gemm + tile + deq + rest + f1
            fused = max(gemm, ab["total"] / t["ceiling_bytes_per_s"][h] * 1e3) + f1
            per[h] = {"hash_ms_unfused": tile + deq + rest, "unfused_x_bf16": unfused / bf16, "fused_x_bf16": fused / bf16}
        out[m] = per
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", action="store_true", help="CPU only: the accounting and the host gate")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    res: dict = {"accounting": [forward_accounting(m) for m in ROWS], "host_gate": host_gate()}
    if not args.dry_run:
        import torch
        from kernels import load

        ext = load()
        res["gates"] = {"hashes": gate_hashes(), "committer": gate_committer(), "route_u": gate_route_u(ext, torch)}
        res["timings"] = timings(ext, torch, args.reps)
        res["projection"] = projection(res["timings"])
        res["device"] = torch.cuda.get_device_name()
    text = json.dumps(res, indent=1, default=str)
    if args.out:
        args.out.write_text(text)
    for a in res["accounting"]:
        gb = {h: round(a["absorbed"][h]["total"] / 1e9, 2) for h in HASH}
        print(f"m={a['m']:>5}: {a['units']:,} units; absorbed GB {gb}; tile words {a['tile_bytes_per_mac']:.3f} B per MAC; "
              f"XOF {a['xof_bytes'] / 1e9:.3f} GB")
    for kind in ("ceiling_bytes_per_s", "from_memory_bytes_per_s"):
        if kind in res.get("timings", {}):
            print(kind, {h: f"{v / 1e9:.0f} GB/s" for h, v in res["timings"][kind].items()})
    for m, p in res.get("projection", {}).items():
        print(f"m={m:>5}: " + "; ".join(f"{h} unfused {p[h]['unfused_x_bf16']:.1f}x fused {p[h]['fused_x_bf16']:.1f}x" for h in HASH))


if __name__ == "__main__":
    main()
