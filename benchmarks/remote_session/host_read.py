"""How a host segment's bytes are asked for inside a pass (`device.host_read_plan`), each way in turn on one store
with a host tier on the visible GPUs, with no verifier: that every way gives the pass's receipts and tree byte for
byte, then what each costs (`host_read.sh` builds the libraries and runs it under one lease). The store's bytes are
random and not encoded: a pass hashes whatever is stored.

A row is a name of `device.host_read_plan` (`pass`: the host segment read as a shard is), with @N for N receipt CTAs
on a host segment in place of --host-ctas (`ahead2@8`). `pass` runs first whatever --names says: the others are held
to it. For each row in turn, before anything is timed:

    gate     the device's gates with the gate store's host segment read that way (`GpuBackend.gate`): a pass's root
             against the reference's, a session accepted, one that skips a sixteenth and one that skips a quarter of
             the host tier each refused at a receipt
    equal    one pass over the store itself at one seed, every buffer a pass writes filled with 0xA5 first: its root
             and every kept level against the plain pass's at that seed (`DeviceStore.plain_root`, kernels a pass
             does not run), and every receipt, kept level and top level against `pass`'s. `pass`'s own host receipts
             are held to the reference's definition, which runs none of the kernels' code (`reference`)

and once, under `pass` and under the first other row: `control`, the same poisoned pass with the host segments' part
launching nothing (`NoHostWrite`), which must not be the same root, receipts or kept levels. A row whose gate or
equality fails is not timed. Then, the rows in turn rep by rep (--reps), `pass` first, after one untimed rep:

    alone    the host segments' part of that pass with nothing else on the device (`DeviceStore.host_pass`, by CUDA
             events on the host stream), its receipts and kept levels equal to the pass's after every rep
    beside   the same beside bare reads of the shards (`rb_read`), queued behind it for longer than it lasts
    chain    --chain-passes chained passes queued as a session queues them: each pass's interval and its host
             segments' part (`pass_host_ms`); every root in order equal to `pass`'s chain at the same challenge

Writes OUT/host_read.json after each step: the chunk (the tier's step) and the tier's size, each host segment's
address in its page, the bare reads, and each row's gate, equality and times. Nothing new starts after --budget-s
seconds, and the record says what was cut. Exits 0 when every row passed its gate and equality and both controls
were refused, 6 when not.

    python host_read.py --lib session.so --kv kv.so --out OUT --gib 32 --host-gib 1 --gpus 1 --chunk-blocks 65536 \\
        --names ahead1 ahead2 ahead3 ahead4 ahead2-keep ahead2-prefetch feed8 feed8-keep ahead2-feed8
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import device  # noqa: E402
import sketching  # noqa: E402
from device import BLOCK, audit, rb, skr  # noqa: E402


def row_of(word: str, host_ctas: int) -> tuple[str, int]:
    """A row's name and its host receipt CTAs from NAME[@N]."""
    name, _, ctas = word.partition("@")
    device.host_read_plan(name)
    if ctas and not (ctas.isdigit() and int(ctas)):
        raise ValueError(f"a row is NAME or NAME@N, N a host segment's receipt CTAs, not {word!r}")
    return name, int(ctas) if ctas else host_ctas


def use(b: device.GpuBackend, row: tuple[str, int]) -> None:
    """The backend's host segments, its store's and its gate stores', read as `row` says from here on."""
    s = b.store
    s.host_read, s.host_ctas = b.host["host_read"], b.host["host_ctas"] = row


def poison(s: device.DeviceStore) -> None:
    """0xA5 into every byte of every buffer a pass writes: each GPU's ring and its host copies, each segment's
    counters, receipts, tree scratch and kept levels with their host copies, and the roots' host copy."""
    s.offload_jobs.join()
    s.sync()

    def fill(t) -> None:
        t.view(s.torch.uint8).fill_(0xA5)

    for g, gpu in enumerate(s.gpus):
        with s.on(gpu):
            for k, t in s.ring[g].items():
                for x in (t.values() if k == "host" else [t]):
                    fill(x)
    for q, x in enumerate(s.seg_ring):
        with s.on(s.gpus[s.seg_gpu[q]]):
            for t in x.values():
                fill(t)
    fill(s.root_host)
    s.sync()


def poisoned_pass(s: device.DeviceStore, seed: bytes) -> tuple:
    """One pass at `seed` into poisoned buffers: its root, its handle (`DeviceStore.ready`) and, on their GPUs, each
    segment's receipts and the kept levels it wrote."""
    poison(s)
    slot = s.launched % device.SLOTS
    root, handle = s.run_pass(seed)
    handle = s.ready(handle)
    s.offload_jobs.join()
    s.sync()
    return root, handle, [x["ys"].clone() for x in s.seg_ring], [x["kept"][slot].clone() for x in s.seg_ring]


class NoHostWrite:
    """A session library whose host segments' part of a pass launches nothing and says it did: the control. A host
    segment's `rs_pass` is the one with consumer CTAs."""

    def __init__(self, so) -> None:
        self._so = so

    def __getattr__(self, name: str):
        f = getattr(self._so, name)
        if name == "rs_host_pass":
            return lambda *args: 0
        if name == "rs_pass":
            return lambda *args: 0 if args[-1] else f(*args)
        return f


def reference(s: device.DeviceStore, seed: bytes, q: int, step: int = 1 << 15) -> bytes:
    """Segment q's receipts under pass `seed` by the reference's definition and none of the kernels' code, 64 bytes a
    block: σ is torch's float32 product of the blocks' bytes as int8 with the signs of `receipts.sign_bits` (exact:
    `sketching.sketch_torch`), and y is `SketchReceipter.receipt`, SHA-512 on the CPU."""
    torch, dev = s.torch, s.seg_ring[q]["ys"].device
    bits = np.unpackbits(skr.sign_bits(seed), axis=1, bitorder="little")
    receipt, width, out = rb.SketchReceipter(seed).receipt, 4 * skr.ROWS, []
    with s.on(s.gpus[s.seg_gpu[q]]):
        for a in range(0, s.n[q], step):
            e = s.data_g[q][a * BLOCK:min(a + step, s.n[q]) * BLOCK].to(dev).view(torch.int8).view(-1, BLOCK)
            sigma = sketching.sketch_torch(torch, e, bits).cpu().numpy().astype("<i4").tobytes()
            out += [receipt(s.base[q] + a + k, sigma[k * width:(k + 1) * width]) for k in range(e.shape[0])]
    return b"".join(out)


def same(s: device.DeviceStore, got: list, want: list) -> list[bool]:
    """Whether each segment's tensor in `got` is `want`'s, compared where it is."""
    out = []
    for q, (a, b) in enumerate(zip(got, want)):
        with s.on(s.gpus[s.seg_gpu[q]]):
            out.append(bool(s.torch.equal(a, b)))
    return out


def equal(s: device.DeviceStore, seed: bytes, base: tuple | None) -> tuple[dict, tuple]:
    """One poisoned pass at `seed` as the store's host read says, against the plain pass and, given `pass`'s (`base`),
    against it: the record and the pass (`poisoned_pass`)."""
    got = root, handle, ys, kept = poisoned_pass(s, seed)
    plain = s.plain_root(seed, handle)
    out = {"plain_root": plain == root, "plain_levels": all(s.plain_level_matches.values())}
    if base is not None:
        root0, handle0, ys0, kept0 = base
        out.update({"root": root == root0, "receipts": same(s, ys, ys0), "kept": same(s, kept, kept0),
                    "top": bool(s.torch.equal(handle[3], handle0[3])),
                    "kept_host": [bool(s.torch.equal(a, b)) for a, b in zip(handle[2], handle0[2])]})
    out["ok"] = all(all(v) if isinstance(v, list) else v for v in out.values())
    return out, got


def control(b: device.GpuBackend, seed: bytes, base: tuple) -> dict:
    """The poisoned pass with the host segments' part launching nothing: refused when its root is not the pass's and
    no host segment's receipts or kept levels are."""
    s, so = b.store, b.lib.so
    b.lib.so = NoHostWrite(so)
    try:
        root, _, ys, kept = poisoned_pass(s, seed)
    finally:
        b.lib.so = so
    host = [q for q in range(len(s.n)) if s.on_host[q]]
    ys_same, kept_same = same(s, ys, base[2]), same(s, kept, base[3])
    out = {"root_same": root == base[0], "host_receipts_same": [ys_same[q] for q in host],
           "host_kept_same": [kept_same[q] for q in host]}
    out["refused"] = not (out["root_same"] or any(out["host_receipts_same"]) or any(out["host_kept_same"]))
    return out


def host_part(b: device.GpuBackend, base: tuple, slot: int, reads: int) -> tuple[float, bool]:
    """Every host segment's part of the pass whose constants and signs the devices hold (`base`'s, the last one run),
    queued as a pass queues it, with `reads` bare reads of each GPU's shard queued behind it: its milliseconds by the
    host streams' events (the slowest GPU's), and whether it left the pass's receipts and kept levels."""
    s, torch = b.store, b.torch
    last = (s.launched - 1) % device.SLOTS
    spans = []
    for g, gpu in enumerate(s.gpus):
        hbm, *host = s.gpu_segs[g]
        if not host:
            continue
        pk = s.ring[g]["pks"][last] if g == 0 else s.ring[g]["pk"]
        e0, e1 = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        with s.on(gpu, gpu.host_main) as stream:
            e0.record(stream)
            for q in host:
                rc = s.host_pass(q, pk, slot)
                if rc:
                    raise RuntimeError(f"the host pass {s.host_read!r} on GPU {gpu.index}: {rc}")
            e1.record(stream)
        spans.append((e0, e1))
        if reads:
            xr = torch.zeros(1, dtype=torch.int64, device=s.data_g[hbm].device)
            with s.on(gpu):
                for _ in range(reads):
                    b.lib.call("rb_read", s.data_g[hbm].data_ptr(), s.n[hbm] * BLOCK, xr.data_ptr(), 16 * gpu.sms)
    s.sync()
    ms = max(e0.elapsed_time(e1) for e0, e1 in spans)
    ok = all(bool(torch.equal(x["ys"], base[2][q])) and bool(torch.equal(x["kept"][slot], base[3][q]))
             for q, x in enumerate(s.seg_ring) if s.on_host[q])
    return ms, ok


def spread(xs: list[float]) -> dict:
    xs = sorted(xs)
    return {"p50": xs[len(xs) // 2], "min": xs[0], "max": xs[-1], "n": len(xs)}


def unbind() -> None:
    """The device's host tiers from here on with no NUMA policy set (`device.numa_bound` does nothing)."""
    device.numa_bound = lambda node: contextlib.nullcontext()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--names", nargs="+", default=["ahead1", "ahead2", "ahead3"], metavar="NAME[@N]",
                    help="the rows after `pass`, in the order they are gated and timed")
    ap.add_argument("--reps", type=int, default=20, help="the timed reps of `alone` and of `beside`, a row")
    ap.add_argument("--chain-passes", type=int, default=100, help="a chain's passes, a row and round")
    ap.add_argument("--chain-rounds", type=int, default=2, help="the rounds of `chain`, each row once a round")
    ap.add_argument("--budget-s", type=float, default=1080.0, help="nothing new starts this long after the start")
    ap.add_argument("--unbound", action="store_true",
                    help="the host tier's pages wherever the kernel puts them, for a machine that refuses "
                         "set_mempolicy: its gates hold, its times are not those of a tier on the card's NUMA node")
    ap.add_argument("--node", default="host-read")
    ap.add_argument("--seed", type=int, default=20261112)
    device.add_arguments(ap)
    a = ap.parse_args()
    t_start = time.monotonic()
    rows = [("pass", a.host_ctas)] + [row_of(w, a.host_ctas) for w in a.names if w != "pass"]

    def left() -> float:
        return a.budget_s - (time.monotonic() - t_start)

    def label(row: tuple[str, int]) -> str:
        return row[0] if row[1] == a.host_ctas else f"{row[0]}@{row[1]}"

    if a.host_gib <= 0:
        raise SystemExit("--host-gib: the store needs a host tier")
    a.out.mkdir(parents=True, exist_ok=True)
    sys.setswitchinterval(2e-4)

    def log(m: str) -> None:
        print(f"{time.monotonic() - t_start:7.1f}s {m}", flush=True)

    if a.unbound:
        unbind()
    b = device.from_arguments(a, log=log)
    s, torch = b.store, b.torch
    rng = np.random.default_rng(a.seed)
    source, node = rng.bytes(32), a.node.encode()
    b.set_payload_key(rng.bytes(32))
    reg = audit.register(source, node, b.blocks, b.payload_root(node), nonce=rng.bytes(audit.NONCE_BYTES))
    hbm = [q for q in range(len(s.n)) if not s.on_host[q]]
    host = [q for q in range(len(s.n)) if s.on_host[q]]
    host_bytes = sum(s.n[q] for q in host) * BLOCK
    out = {"schema": "verity/remote-session/host-read/v0", "device": [torch.cuda.get_device_name(g.index) for g in s.gpus],
           "chunk_blocks": s.chunk, "step_bytes": s.chunk * BLOCK, "hbm_bytes": s.hbm_blocks * BLOCK,
           "host_bytes": host_bytes, "h0": s.h0, "hc": s.hc, "host_ctas": a.host_ctas, "host_consumers": a.host_consumers,
           "encoded": False, "unbound": a.unbound,
           "host_segments": [{"gpu": s.gpus[s.seg_gpu[q]].index, "numa": s.gpus[s.seg_gpu[q]].numa, "bytes": s.n[q] * BLOCK,
                              "address_in_page": s.data_g[q].data_ptr() % 4096,
                              "address_in_256": s.data_g[q].data_ptr() % 256} for q in host],
           "rows": {}, "cut": []}

    def write() -> None:
        out["seconds"] = time.monotonic() - t_start
        (a.out / "host_read.json").write_text(json.dumps(out, indent=1))

    def rate(ms: float) -> float:
        return host_bytes / (ms / 1e3)

    read = {"hbm_ms": b.read_ms(hbm), "host_ms": b.read_ms(host), "both_ms": b.read_ms(hbm + host)}
    read["host_bytes_per_s"] = rate(read["host_ms"])
    out["read"] = read
    log(f"store: {out['hbm_bytes'] / 2 ** 30:g} GiB on the devices and a host tier of {host_bytes / 2 ** 30:g} GiB in "
        f"steps of {out['step_bytes'] / 2 ** 20:g} MiB; bare reads {json.dumps(read)}; host segments "
        f"{json.dumps(out['host_segments'])}")
    write()

    seed = rb.seed(rng.bytes(rb.CHALLENGE_BYTES), 0, bytes(rb.DIGEST_BYTES))
    base, passed = None, []
    for row in rows:
        if left() <= 0:
            out["cut"].append(f"gate and equal: {label(row)}")
            continue
        r = out["rows"][label(row)] = {"host_read": row[0], "host_ctas": row[1]}
        use(b, row)
        try:
            r["gate"] = b.gate(reg)
            r["equal"], got = equal(s, seed, base)
            if base is None:
                want = [reference(s, seed, q) for q in host]
                r["equal"]["host_receipts_reference"] = [
                    b"".join(device.digests(got[2][q])) == w for q, w in zip(host, want)]
                r["equal"]["ok"] = r["equal"]["ok"] and all(r["equal"]["host_receipts_reference"])
                r["equal"]["host_receipts_sha256"] = [hashlib.sha256(w).hexdigest() for w in want]
                del want
            r["ok"] = bool(r["gate"]["ok"] and r["equal"]["ok"])
        except Exception as error:  # noqa: BLE001
            r["error"], r["ok"] = f"{type(error).__name__}: {error}", False
        log(f"{label(row)}: gate {r.get('gate', {}).get('ok')}, equal {json.dumps(r.get('equal'))}"
            + (f", error {r['error']}" if "error" in r else ""))
        write()
        if "error" in r:
            s.sync()   # a device a kernel faulted on raises here, and the record above is what the run leaves
        if r["ok"]:
            passed.append(row)
            base = base or got
        if base is None:
            log("`pass` itself failed its gate or equality: nothing is timed")
            return 6
    out["control"] = {}
    for row in passed[:2]:
        use(b, row)
        out["control"][label(row)] = control(b, seed, base)
        log(f"control under {label(row)}: {json.dumps(out['control'][label(row)])}")
    write()
    refused = all(c["refused"] for c in out["control"].values())
    if not refused:
        log("a pass whose host part wrote nothing was not refused: nothing is timed")
        return 6

    def timed() -> bool:
        """`alone`, `beside` and `chain` over the rows that passed; whether every rep left the pass's bytes."""
        use(b, rows[0])
        again = poisoned_pass(s, seed)   # the devices hold this pass's constants and signs from here to `chain`
        slot = s.launched % device.SLOTS
        if again[0] != base[0]:
            log("the pass at the same seed is not the same root: nothing is timed")
            return False
        del again
        for arm in ("alone", "beside"):
            times: dict = {label(row): [] for row in passed}
            wrong: dict = {label(row): 0 for row in passed}
            reads = 0
            if arm == "beside":
                # Bare reads of the shard for 1.5 times as long as the slowest row alone lasts.
                reads = int(1.5 * max(x["ms"]["max"] for x in out["alone"].values()) / read["hbm_ms"]) + 2
            for rep in range(a.reps + 1):
                if left() <= 0:
                    out["cut"].append(f"{arm}: after {rep} reps")
                    break
                for row in passed:
                    use(b, row)
                    ms, ok = host_part(b, base, slot, reads)
                    wrong[label(row)] += not ok
                    if rep:
                        times[label(row)].append(ms)
            out[arm] = {k: {"ms": spread(v), "bytes_per_s": rate(spread(v)["p50"]), "wrong": wrong[k], "each_ms": v}
                        for k, v in times.items() if v}
            if reads:
                out[arm + "_reads"] = reads
            log(f"{arm}: " + "; ".join(
                f"{k} {x['ms']['p50']:.2f} ms ({x['bytes_per_s'] / 1e9:.1f} GB/s"
                f"{', WRONG ' + str(x['wrong']) if x['wrong'] else ''})" for k, x in out[arm].items()))
            write()
            if not out[arm]:
                break
        challenge, prev = rng.bytes(rb.CHALLENGE_BYTES), bytes(rb.DIGEST_BYTES)
        chains: dict = {label(row): {"interval_ms": [], "host_ms": [], "roots_sha256": []} for row in passed}
        for rnd in range(a.chain_rounds):
            for row in passed:
                if left() <= 0:
                    out["cut"].append(f"chain: round {rnd}, {label(row)}")
                    continue
                use(b, row)
                s.pass_gpu_ms.clear()
                s.pass_host_ms.clear()
                s.start_chain(challenge, prev)
                roots, seen = hashlib.sha256(), []
                for _, root, _, at in s.chain(a.chain_passes):
                    roots.update(root)
                    seen.append(at)
                s.sync()
                c = chains[label(row)]
                c["interval_ms"] += [(y - x) / 1e6 for x, y in zip(seen[1:], seen[2:])]
                c["host_ms"] += s.pass_host_ms[1:]
                c["roots_sha256"].append(roots.hexdigest())
        want = chains[label(rows[0])]["roots_sha256"][:1]
        out["chain"] = {k: {"pass_ms": spread(c["interval_ms"]), "host_ms": spread(c["host_ms"]),
                            "host_bytes_per_s": rate(spread(c["host_ms"])["p50"]), "passes": len(c["interval_ms"]),
                            "roots_are_pass_s": bool(want) and all(x == want[0] for x in c["roots_sha256"])}
                        for k, c in chains.items() if c["interval_ms"] and c["host_ms"]}
        log("chain: " + "; ".join(
            f"{k} pass {c['pass_ms']['p50']:.2f} ms, host part {c['host_ms']['p50']:.2f} ms "
            f"({c['host_bytes_per_s'] / 1e9:.1f} GB/s){'' if c['roots_are_pass_s'] else ', ROOTS DIFFER'}"
            for k, c in out["chain"].items()))
        return (all(not x["wrong"] for arm in ("alone", "beside") for x in out.get(arm, {}).values())
                and all(c["roots_are_pass_s"] for c in out["chain"].values()))

    gc.collect()
    gc.freeze()
    gc.disable()
    try:
        timed_ok = timed()
    finally:
        gc.enable()
        gc.unfreeze()
    out["ok"] = bool(timed_ok and all(r["ok"] for r in out["rows"].values()))
    write()
    s.close()
    return 0 if out["ok"] else 6


if __name__ == "__main__":
    sys.exit(main())
