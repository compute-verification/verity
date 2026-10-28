"""τ of the prover's pass on one GPU: the read alone (`rb_read`), the plain pass kernel by kernel (sketch, receipts,
leaves, tree), the tensor-core receipts alone (checked equal to the plain ones), and the pipelined pass (`rs_pass`)
at each chunk size and CTA count asked for, every pass checked to give the plain pass's root, over a store of random words (no codec: a pass's cost does not depend on the bytes).
The store is rounded down to a whole 2^18-block pass chunk, not to a power of two; receipt trees use the driver's
canonical padding. Writes OUT/pass_bench.json.

    python pass_bench.py --lib session.so --gib 16 --out OUT [--chunks 14,15,16,17,18 --ctas 0 --passes 30]
"""
from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

from verity.core.protocols.remote_bandwidth.audit import SAMPLES  # noqa: E402

import device  # noqa: E402
from device import BLOCK, PER_LEAF, DeviceStore, Lib, percentiles, rb, skr  # noqa: E402


def timed(torch, fn) -> float:
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    fn()
    b.record()
    b.synchronize()
    return a.elapsed_time(b)


def main() -> int:
    import torch

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lib", required=True)
    ap.add_argument("--gib", type=float, default=16.0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--chunks", default="14,15,16,17,18", help="log2 of the chunk's blocks")
    ap.add_argument("--ctas", default="0", help="the receipts grid's CTAs, comma-separated (0: one an SM)")
    ap.add_argument("--kept-from", type=int, default=6)
    ap.add_argument("--passes", type=int, default=30)
    ap.add_argument("--open-passes", type=int, default=300)
    ap.add_argument("--openings", type=int, default=SAMPLES)
    ap.add_argument("--contend", default="0:0:5,-10:0:5,-10:0:0.5,-10:0:5:nogc,-10:0:0.5:nogc",
                    help="passes run while a window opens: pass stream priority:open stream priority:GIL switch ms[:nogc]")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    lib = Lib(a.lib, torch)
    copy = tempfile.NamedTemporaryFile(suffix=".so")
    shutil.copyfile(a.lib, copy.name)
    lib_open = Lib(copy.name, torch)
    n = int(a.gib * 2 ** 30) // BLOCK
    blocks = n // (1 << 18) * (1 << 18)
    g = torch.Generator(device="cuda")
    g.manual_seed(1)
    store = DeviceStore(torch, lib, lib_open, None, blocks, b"k" * 32, g, kept_from=a.kept_from)
    N, nl, width = blocks, blocks // PER_LEAF, store.tree_leaves
    xr = torch.zeros(1, dtype=torch.int64, device="cuda")
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    reads = [timed(torch, lambda: lib.call("rb_read", store.data.data_ptr(), N * BLOCK, xr.data_ptr(), 16 * sms))
             for _ in range(8)]
    read_ms = min(reads[2:])
    out = {"device": torch.cuda.get_device_name(0), "blocks": N, "bytes": N * BLOCK, "read_ms": read_ms,
           "read_bytes_per_s": N * BLOCK / read_ms * 1e3}
    print(json.dumps(out), flush=True)

    seed = rb.seed(b"\1" * 32, 0, bytes(64))
    signs = torch.from_numpy(skr.signs(seed)).cuda()
    store.set_pass(lib, seed)
    sk = torch.empty((N, skr.ROWS), dtype=torch.int32, device="cuda")
    ys = torch.empty((N, 8), dtype=torch.int64, device="cuda")
    dig = torch.empty((width, 8), dtype=torch.int64, device="cuda")
    up = torch.empty((width - 1, 8), dtype=torch.int64, device="cuda")
    plain = {k: [] for k in ("sketch", "hash", "leaves", "tree")}
    for _ in range(6):
        plain["sketch"].append(timed(torch, lambda: lib.call("rb_sketch", store.data.data_ptr(), None, N,
                                                            signs.data_ptr(), sk.data_ptr(), 0)))
        plain["hash"].append(timed(torch, lambda: lib.call("rb_sketch_hash", sk.data_ptr(), 0, N, ys.data_ptr())))
        plain["leaves"].append(timed(torch, lambda: lib.call("rb_leaves", ys.data_ptr(), 0, nl, dig.data_ptr(), None)))
        if width > nl:
            lib.call("rb_pads", nl, width - nl, dig[nl:].data_ptr())
        plain["tree"].append(timed(torch, lambda: lib.call("rb_tree", dig.data_ptr(), width, 0, up.data_ptr())))
    out["plain_ms"] = {k: min(v[1:]) for k, v in plain.items()}
    out["plain_ms"]["sum"] = sum(out["plain_ms"].values())
    out["plain_tau"] = out["plain_ms"]["sum"] / read_ms
    plain_root = device.digests(up[-1:].cpu())[0]
    del sk, dig, up
    pk = torch.empty(device.PACK16_WORDS, dtype=torch.int32, device="cuda")
    lib.call("rs_pack16", signs.data_ptr(), pk.data_ptr())
    ys16 = torch.empty((N, 8), dtype=torch.int64, device="cuda")
    tensor = [timed(torch, lambda: lib.call("rs_sketch16_receipts", store.data.data_ptr(), 0, N, pk.data_ptr(),
                                            ys16.data_ptr(), 0)) for _ in range(6)]
    out["tensor_receipts_ms"] = min(tensor[1:])
    out["tensor_receipts_tau"] = out["tensor_receipts_ms"] / read_ms
    out["tensor_receipts_equal"] = bool(torch.equal(ys, ys16))
    out["device_signs_equal"] = all(store.signs_match(rb.seed(b"\4" * 32, i, bytes(64))) for i in range(8))
    del ys, ys16, pk
    torch.cuda.empty_cache()
    print(json.dumps({"plain_ms": out["plain_ms"], "plain_tau": out["plain_tau"],
                      "tensor_receipts_ms": out["tensor_receipts_ms"], "tensor_receipts_tau": out["tensor_receipts_tau"],
                      "tensor_receipts_equal": out["tensor_receipts_equal"],
                      "device_signs_equal": out["device_signs_equal"]}), flush=True)

    out["pipelined"] = []
    for c in [int(x) for x in a.chunks.split(",")]:
        for ctas in [int(x) for x in a.ctas.split(",")]:
            s = store
            s.shape(1 << c, a.kept_from, ctas)
            gpu, host = [], []
            prev = s.run_pass(seed)[0]
            ok = prev == plain_root
            for i in range(a.passes):
                sd = rb.seed(b"\1" * 32, i + 1, prev)
                t0 = time.perf_counter_ns()
                prev, _ = s.run_pass(sd)
                host.append(time.perf_counter_ns() - t0)
                ok = ok and prev == s.plain_root(sd)
            gpu = s.pass_gpu_ms[-a.passes:]
            names = ("head", "set_pass", "alloc", "launch", "wait", "root")
            ph = s.pass_host_phases[-a.passes:]
            phases = {k: sorted(x[j] for x in ph)[len(ph) // 2] for j, k in enumerate(names)}
            row = {"chunk_log2": c, "ctas": ctas, "h0": s.h0, "root_ok": ok, "host_phases_p50_ns": phases,
                   "gpu_ms": percentiles(sorted(gpu)), "host_ns": percentiles(sorted(host)),
                   "tau": sorted(gpu)[len(gpu) // 2] / read_ms, "tau_host": sorted(host)[len(host) // 2] / 1e6 / read_ms}
            challenge, prev_host = b"\x06" * 32, bytes(64)
            s.start_chain(challenge, prev_host)
            chain_gpu, chain_host, chain_seed_ok, chain_signs_ok, chain_root_ok = [], [], True, True, True
            p0 = len(s.chain_host_phases)
            for i in range(a.passes):
                want_seed = rb.seed(challenge, i, prev_host)
                t0 = time.perf_counter_ns()
                prev_host, handle = s.run_chain(i)
                chain_host.append(time.perf_counter_ns() - t0)
                chain_gpu.append(s.pass_gpu_ms[-1])
                handle = s.ready(handle)
                chain_seed_ok = chain_seed_ok and handle[0] == want_seed
                want_pk = torch.empty(device.PACK16_WORDS, dtype=torch.int32, device="cuda")
                want_signs = torch.from_numpy(skr.signs(want_seed)).cuda()
                lib.call("rs_pack16", want_signs.data_ptr(), want_pk.data_ptr())
                chain_signs_ok = chain_signs_ok and bool(torch.equal(handle[1], want_pk.cpu()))
                chain_root_ok = chain_root_ok and prev_host == s.plain_root(want_seed)
            chain_ph = s.chain_host_phases[p0:]
            chain_names = ("alloc", "launch", "wait", "root")
            row.update({"chain_seed_ok": chain_seed_ok, "chain_signs_ok": chain_signs_ok,
                        "chain_root_ok": chain_root_ok,
                        "chain_host_phases_p50_ns": {
                            k: sorted(x[j] for x in chain_ph)[len(chain_ph) // 2] for j, k in enumerate(chain_names)
                        },
                        "chain_gpu_ms": percentiles(sorted(chain_gpu)),
                        "chain_host_ns": percentiles(sorted(chain_host)),
                        "chain_tau": sorted(chain_gpu)[len(chain_gpu) // 2] / read_ms,
                        "chain_tau_host": sorted(chain_host)[len(chain_host) // 2] / 1e6 / read_ms})
            out["pipelined"].append(row)
            print(json.dumps(row), flush=True)
            s.pass_gpu_ms.clear()
            s.chain_host_phases.clear()
    t = []
    for i in range(20):
        t0 = time.perf_counter_ns()
        skr.signs(rb.seed(b"\2" * 32, i, bytes(64)))
        t.append(time.perf_counter_ns() - t0)
    out["signs_host_ns"] = percentiles(sorted(t))
    store.shape(1 << int(a.chunks.split(",")[-1]), a.kept_from, 0)
    store.payload_tree(b"bench")
    kept, prev = [], bytes(64)
    for i in range(a.open_passes):
        prev, h = store.run_pass(rb.seed(b"\3" * 32, i, prev))
        kept.append(h)
    import numpy as np

    rng = np.random.default_rng(5)
    draws = sorted(zip(rng.integers(len(kept), size=a.openings).tolist(), rng.integers(N, size=a.openings).tolist()))
    by = {}
    for i, p in draws:
        by.setdefault(i, []).append(p)
    with torch.cuda.stream(torch.cuda.Stream()):
        for _ in range(2):
            t0 = time.perf_counter_ns()
            store.open_window([(kept[i], ps) for i, ps in sorted(by.items())])
            torch.cuda.synchronize()
            open_ns = time.perf_counter_ns() - t0
    out["open"] = {"openings": a.openings, "passes": len(kept), "ns": open_ns, "stats": store.open_stats[-1]}
    print(json.dumps({"signs_host_ns": out["signs_host_ns"], "open": out["open"]}), flush=True)

    out["contended"] = []
    items = [(kept[i], ps) for i, ps in sorted(by.items())]
    for spec in filter(None, a.contend.split(",")):
        pass_prio, open_prio, switch_ms, *rest = spec.split(":")
        sys.setswitchinterval(float(switch_ms) / 1e3)
        nogc = rest == ["nogc"]
        collections, began = [], {}

        def on_gc(phase: str, info: dict) -> None:
            if phase == "start":
                began[0] = time.perf_counter_ns()
            elif time.perf_counter_ns() - began.get(0, 0) > 10_000_000:
                collections.append({"generation": info["generation"], "ms": (time.perf_counter_ns() - began[0]) / 1e6})
        gc.callbacks.append(on_gc)
        if nogc:
            gc.freeze()
            gc.disable()
        ps, os_ = torch.cuda.Stream(priority=int(pass_prio)), torch.cuda.Stream(priority=int(open_prio))
        done, opened = threading.Event(), {}

        def opener() -> None:
            t0 = time.perf_counter_ns()
            with torch.cuda.stream(os_):
                got = store.open_window(items)
                os_.synchronize()
            while got:
                got.pop()
            opened["ns"] = time.perf_counter_ns() - t0
            done.set()

        stalls = []

        def probe() -> None:
            while not done.is_set():
                t0 = time.perf_counter_ns()
                time.sleep(0.0005)
                if time.perf_counter_ns() - t0 > 20_000_000:
                    stalls.append((t0, time.perf_counter_ns() - t0))

        host, g0, prev = [], len(store.pass_gpu_ms), bytes(64)
        with torch.cuda.stream(ps):
            th, pr = threading.Thread(target=opener), threading.Thread(target=probe)
            th.start()
            pr.start()
            while not done.is_set():
                t0 = time.perf_counter_ns()
                prev, _ = store.run_pass(rb.seed(b"\5" * 32, len(host), prev))
                host.append(time.perf_counter_ns() - t0)
            th.join()
            pr.join()
        gc.callbacks.remove(on_gc)
        if nogc:
            gc.enable()
            gc.unfreeze()
        gpu = store.pass_gpu_ms[g0:]
        st = store.open_stats[-1]
        marks = [st["at_ns"], st["at_ns"] + st["subtrees_ns"], st["at_ns"] + st["subtrees_ns"] + st["gather_ns"]]
        stalls = [{"after_open_ms": (t - marks[0]) / 1e6, "ms": d / 1e6,
                   "phase": ("before", "subtrees", "gather", "assemble")[sum(t >= m for m in marks)]} for t, d in stalls]
        row = {"pass_priority": int(pass_prio), "open_priority": int(open_prio), "switch_ms": float(switch_ms), "gc": not nogc,
               "passes": len(host), "open_ns": opened["ns"], "host_ns": percentiles(sorted(host)),
               "gpu_ms": percentiles(sorted(gpu)), "excess_ms": sum(host) / 1e6 - len(host) * read_ms,
               "open_stats": st, "gil_stalls": stalls, "collections_over_10ms": collections}
        out["contended"].append(row)
        print(json.dumps(row), flush=True)
    sys.setswitchinterval(0.005)
    (a.out / "pass_bench.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
