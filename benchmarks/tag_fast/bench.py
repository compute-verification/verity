"""ro-blake3/v1 fused into the FP8 decode GEMM on one RTX PRO 6000 (`tf.cu`): gates, then interleaved timings.

Gates; a failed gate times nothing (exit 5):
  key       tf_init's K is `tags.Blake3Tag().key(seed)`
  compress  every BLAKE3 variant's compression is `blake3.compress` on 256 random inputs; Keccak-p[1600, 12] is
            `turboshake.keccak_p` on 16 random states
  small     a 64 x 4096 store: every tag of every kernel form is `tags.Blake3Tag().keyed(seed)` computed here, and the
            leaves and root are `tags.leaf_hash`'s and `layout.tree_root`'s
  shapes    each of Llama-3.1-8B's linears from `reference.py`'s W: every tag of every timed form (buffers poisoned with
            0xFF first) is reference.py's, every leaf too and the root; the sink form's XOR is the reference tags';
            a control under another seed matches no block; at every batch the GEMM's per-span sums with and without the
            tag are equal bit for bit and within FP32 rounding of torch's product, and every timed form's output is too
Timings, every call on cold weights (rings of copies past the L2), each arm queued behind a spin kernel and the arms
interleaved rep by rep (the word codec's method, benchmarks/word_codec/bench.py):
  per shape and batch  cuBLASLt FP8 (torch._scaled_mm), the word codec's plain FP8 GEMM, tf.cu's GEMM without the tag
            (g: two load depths), with the tag to memory (t: variants and depths), with the tag and its leaves (c), with
            the tag XORed into one word (s), and an empty kernel; native is the fastest of cuBLASLt and the plain GEMMs
  per shape  the tag alone (o) and a streaming read of the store
  device    compression chains (BLAKE3's variants; at 4 and 5 rounds, uncertified; Keccak-p[1600, 12]; SHA-512), the
            tree over one step's leaves, and the leaves over one step's tags at once
y_tag is Σ fused / Σ native over the four linears at one batch (layer-weighted), y_commit the same with the leaves.
Writes OUT/tag-fast.json and OUT/bench.log.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import random
import re
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import layout  # noqa: E402
import reference as R  # noqa: E402
from verity.core.protocols.bandwidth import tags  # noqa: E402
from verity.ml.crypto import blake3, turboshake  # noqa: E402

SEED = R.SEED
SEED2 = bytes(range(1, 33))
BATCHES = (1, 2, 4, 8, 16, 32)
L2_DEFEAT_BYTES = 1 << 30
SPIN_CYCLES = 20_000_000
TAG_FORMS = ((0, 1), (1, 1), (3, 1), (1, 2))  # (variant, load depth) of the timed tag forms
MODE_GEMM, MODE_TAGS, MODE_SINK, MODE_TAGONLY = 0, 1, 2, 3
FORM = re.compile(r"(?P<stage>[gtcso])(?P<span>\d+)(?:v(?P<v>\d+))?p(?P<pd>\d+)$")  # an arm of tf.cu's kernels


def forms(arms, stage: str) -> list[str]:
    return [a for a in arms if (m := FORM.match(a)) and m["stage"] == stage]


class Lib:
    def __init__(self, path: str) -> None:
        v, i, u64 = ctypes.c_void_p, ctypes.c_int, ctypes.c_uint64
        so = ctypes.CDLL(path)
        sig = {
            "tf_init": [v, v], "tf_fused": [i, i, i, i, i, v, v, v, v, i, i, i, u64, v, v],
            "tf_plain": [v, v, v, i, i, i, i, v], "tf_leaves": [v, u64, u64, v, v], "tf_tree": [v, u64, v],
            "tf_chain": [i, i, i, u64, v, i, i, v], "tf_b3_one": [i, v, v, v], "tf_keccak_one": [v, v],
            "tf_read": [v, u64, v, i, v], "tf_spin": [ctypes.c_longlong, v], "tf_empty": [i, v],
        }
        for name, args in sig.items():
            getattr(so, name).argtypes = args
            getattr(so, name).restype = i
        so.tf_tree_nodes.argtypes = [u64]
        so.tf_tree_nodes.restype = u64
        self.so, self.path = so, path

    def call(self, fn: str, *args) -> None:
        rc = getattr(self.so, fn)(*args)
        if rc:
            raise RuntimeError(f"{fn}{args[:6]}: error {rc}")


def stream(torch) -> int:
    return torch.cuda.current_stream().cuda_stream


def ptr(t) -> int:
    return t.data_ptr() if t is not None else 0


def timed(torch, fn, reps: int, warm: int = 1) -> list[float]:
    for _ in range(warm):
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


def queued(lib: Lib, torch, fn) -> tuple[float, float]:
    """fn's device time with its launches queued behind a spin (seconds), and the host's time to enqueue them."""
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    lib.call("tf_spin", SPIN_CYCLES, stream(torch))
    a.record()
    h = time.perf_counter()
    fn()
    h = time.perf_counter() - h
    b.record()
    b.synchronize()
    return a.elapsed_time(b) / 1e3, h


def xor_reduce(torch, t) -> int:
    t = t.contiguous().view(torch.int32).clone()
    while t.numel() > 1:
        if t.numel() % 2:
            t = torch.cat([t, torch.zeros(1, dtype=t.dtype, device=t.device)])
        t = torch.bitwise_xor(t[0::2], t[1::2])
    return int(t.item()) & 0xFFFFFFFF


def fused(lib: Lib, torch, v, span, mode, pd, partials, w, x, y, tags_buf, m, n, k, jbase, xr=None):
    lib.call("tf_fused", v, span, mode, pd, int(partials), ptr(w), ptr(x.view(torch.uint8)) if x is not None else 0,
             ptr(y), ptr(tags_buf), m, n, k, jbase, ptr(xr), stream(torch))


def leaf_bytes(nodes, count: int) -> bytes:
    """GPU leaves (u64 words of SHA-512's state) as digests."""
    return nodes[:8 * count].cpu().numpy().astype(">u8").tobytes()


# --- gates -------------------------------------------------------------------------------------------------------------

def gate_compress(lib: Lib, torch, log) -> dict:
    rng = random.Random(11)
    out = {"b3": {}, "keccak": None}
    din = torch.zeros(27, dtype=torch.int32, device="cuda")
    dout = torch.zeros(8, dtype=torch.int32, device="cuda")
    cases = []
    for _ in range(256):
        cv = [rng.getrandbits(32) for _ in range(8)]
        m = bytes(rng.getrandbits(8) for _ in range(64))
        cases.append((cv, m, rng.getrandbits(32), rng.choice([0, 8, 17, 64]), rng.getrandbits(5)))
    for var in range(5):
        ok = True
        for cv, m, ctr, ln, fl in cases:
            words = cv + list(layout._words(m)) + [ctr, ln, fl]
            din.copy_(torch.tensor([w - (1 << 32) if w >= 1 << 31 else w for w in words], dtype=torch.int32))
            lib.call("tf_b3_one", var, ptr(din), ptr(dout), stream(torch))
            got = [w & 0xFFFFFFFF for w in dout.cpu().tolist()]
            ok &= got == blake3.compress(tuple(cv), m, ctr, ln, fl)[:8]
        out["b3"][var] = ok
    st = torch.zeros(25, dtype=torch.int64, device="cuda")
    kok = True
    for _ in range(16):
        s = [rng.getrandbits(64) for _ in range(25)]
        st.copy_(torch.tensor([w - (1 << 64) if w >= 1 << 63 else w for w in s], dtype=torch.int64))
        lib.call("tf_keccak_one", ptr(st), stream(torch))
        want = list(s)
        turboshake.keccak_p(want, 12)
        kok &= [w & ((1 << 64) - 1) for w in st.cpu().tolist()] == want
    out["keccak"] = kok
    out["ok"] = all(out["b3"].values()) and kok
    log(f"gate compress: {'ok' if out['ok'] else 'FAIL'} {out}")
    return out


def gate_store(lib: Lib, torch, name, n, k, w8, ref: dict, batches, log) -> dict:
    """Every form's tags, leaves and root against ref (span -> {tags, leaves, root, jbase, leaf0}), and the GEMM."""
    res = {"ok": True, "spans": {}, "gemm": {}}
    plain = R.plain_layout(w8).contiguous()
    wf = w8.view(torch.float8_e4m3fn)
    gx = torch.Generator(device="cuda").manual_seed(5)
    xs = {mb: torch.randn(mb, k, device="cuda", generator=gx).to(torch.float8_e4m3fn) for mb in batches}
    for span, r in ref.items():
        count, jb = r["blocks"], r["jbase"]
        want = r["tags"]
        tb = torch.empty(count * 32, dtype=torch.uint8, device="cuda")
        e = {}

        def check(label, fn):
            tb.fill_(0xFF)
            fn()
            torch.cuda.synchronize()
            e[label] = bool(torch.equal(tb, want))

        for v, pd in TAG_FORMS:
            check(f"o_v{v}p{pd}", lambda: fused(lib, torch, v, span, MODE_TAGONLY, pd, 0, plain, None, None, tb, 1, n,
                                                k, jb))
            for mb in (1, 16, 32):
                y = torch.zeros(mb, n, dtype=torch.float32, device="cuda")
                check(f"t_v{v}p{pd}_m{mb}", lambda: fused(lib, torch, v, span, MODE_TAGS, pd, 0, plain, xs[mb], y, tb,
                                                          mb, n, k, jb))
        y = torch.zeros(32, n, dtype=torch.float32, device="cuda")
        xr = torch.zeros(1, dtype=torch.int32, device="cuda")
        for mb in (1, 16, 32):
            xr.zero_()
            fused(lib, torch, 1, span, MODE_SINK, 1, 0, plain, xs[mb], y, None, mb, n, k, jb, xr)
            e[f"s_v1p1_m{mb}"] = (int(xr.item()) & 0xFFFFFFFF) == xor_reduce(torch, want)
        # leaves and root
        nleaves = count // 16
        nodes = torch.zeros(int(lib.so.tf_tree_nodes(nleaves)) * 8, dtype=torch.int64, device="cuda")
        fused(lib, torch, 1, span, MODE_TAGONLY, 1, 0, plain, None, None, tb, 1, n, k, jb)
        lib.call("tf_leaves", ptr(tb), count, r["leaf0"], ptr(nodes), stream(torch))
        lib.call("tf_tree", ptr(nodes), nleaves, stream(torch))
        torch.cuda.synchronize()
        e["leaves"] = leaf_bytes(nodes, nleaves) == r["leaves"]
        e["root"] = leaf_bytes(nodes[-8:], 1).hex() == r["root"]
        # control: another seed's tags match no block
        lib.call("tf_init", SEED2, None)
        tb.fill_(0xFF)
        fused(lib, torch, 1, span, MODE_TAGONLY, 1, 0, plain, None, None, tb, 1, n, k, jb)
        torch.cuda.synchronize()
        same = int((tb.view(count, 32) == want.view(count, 32)).all(1).sum())
        e["control_matches"] = same
        lib.call("tf_init", SEED, None)
        ok = all(v for key, v in e.items() if key != "control_matches") and same == 0
        res["spans"][span] = {"ok": ok, "checks": e, "blocks": count}
        res["ok"] &= ok
        log(f"gate {name} span {span}: {'ok' if ok else 'FAIL'} ({count} blocks"
            + ("" if ok else f"; {[key for key, v in e.items() if v is False]}, control {same}") + ")")
    spans = list(ref)
    wt32 = wf.float().t()
    wa32 = wt32.abs()
    for mb in batches:
        x = xs[mb]
        refp = x.float() @ wt32
        scale = (x.float().abs() @ wa32).clamp_min(1e-30)
        e = {}
        for span in spans:
            nsp = k // (32 * span)
            p0 = torch.zeros(nsp * mb * n, dtype=torch.float32, device="cuda")
            p1 = torch.zeros_like(p0)
            tb = torch.empty(ref[span]["blocks"] * 32, dtype=torch.uint8, device="cuda")
            fused(lib, torch, 0, span, MODE_GEMM, 1, 1, plain, x, p0, None, mb, n, k, 0)
            fused(lib, torch, 1, span, MODE_TAGS, 1, 1, plain, x, p1, tb, mb, n, k, 0)
            e[f"partials_equal_{span}"] = bool(torch.equal(p0, p1))
            e[f"partials_rel_{span}"] = float(((p0.view(nsp, mb, n).sum(0) - refp).abs() / scale).max())
            forms = [(f"g{span}p{pd}", (0, MODE_GEMM, pd)) for pd in (1, 2)]
            forms += [(f"t{span}v{v}p{pd}", (v, MODE_TAGS, pd)) for v, pd in TAG_FORMS]
            forms += [(f"s{span}v1p1", (1, MODE_SINK, 1))]
            xr = torch.zeros(1, dtype=torch.int32, device="cuda")
            for label, (v, mode, pd) in forms:
                ya = torch.zeros(mb, n, dtype=torch.float32, device="cuda")
                fused(lib, torch, v, span, mode, pd, 0, plain, x, ya, tb, mb, n, k, 0, xr)
                e[f"{label}_rel"] = float(((ya - refp).abs() / scale).max())
        ya = torch.zeros(mb, n, dtype=torch.float32, device="cuda")
        lib.call("tf_plain", ptr(plain), ptr(x.view(torch.uint8)), ptr(ya), mb, n, k, 0, stream(torch))
        e["plain_rel"] = float(((ya - refp).abs() / scale).max())
        try:
            one = torch.ones((), device="cuda")
            yc = torch._scaled_mm(x, wf.t(), one, one, out_dtype=torch.float32)
            e["cublas_rel"] = float(((yc - refp).abs() / scale).max())
        except Exception as exc:  # noqa: BLE001
            e["cublas_error"] = repr(exc)[:200]
        ok = all(v for key, v in e.items() if key.startswith("partials_equal")) and all(
            v < 1e-5 for key, v in e.items() if key.endswith("_rel") and key != "cublas_rel")
        res["gemm"][mb] = {"ok": ok, "checks": e}
        res["ok"] &= ok
        if not ok:
            log(f"gate {name} gemm M={mb}: FAIL {e}")
    log(f"gate {name} gemm: {'ok' if all(g['ok'] for g in res['gemm'].values()) else 'FAIL'}")
    return res


def small_reference(torch, n: int, k: int, jbase: int) -> tuple:
    w8 = R.weights(torch, n, k, 77)
    plain = R.plain_layout(w8)
    keyed = tags.Blake3Tag().keyed(SEED)
    head = tags.leaf_head(tags.Blake3Tag(), SEED)
    ref = {}
    for span in (32, 64):
        bl = R.block_view(plain, n, k, span)
        count = bl.shape[0]
        ys = [keyed(jbase + j, bytes(bl[j].numpy())) for j in range(count)]
        lv = [tags.leaf_hash(head, jbase // 16 + c, ys[16 * c:16 * c + 16]) for c in range(count // 16)]
        ref[span] = {"blocks": count, "jbase": jbase, "leaf0": jbase // 16, "tags": b"".join(ys), "leaves": b"".join(lv),
                     "root": layout.tree_root(lv).hex()}
    return w8, ref


def load_ref(torch, refdir: Path, name: str, expect: dict) -> tuple:
    ent = expect["shapes"][name]
    n, k = ent["N"], ent["K"]
    w8 = torch.frombuffer(bytearray((refdir / f"w-{name}.bin").read_bytes()), dtype=torch.uint8).view(n, k)
    ref = {}
    for span, s in ent["spans"].items():
        ref[int(span)] = {**s, "tags": (refdir / f"tags-{name}-{span}.bin").read_bytes(),
                          "leaves": (refdir / f"leaves-{name}-{span}.bin").read_bytes()}
    return w8, ref


def to_device_ref(torch, ref: dict) -> dict:
    return {s: {**r, "tags": torch.frombuffer(bytearray(r["tags"]), dtype=torch.uint8).cuda()} for s, r in ref.items()}


# --- timings -----------------------------------------------------------------------------------------------------------

def time_shape(lib: Lib, torch, name, n, k, w8, spans, batches, reps, log) -> dict:
    plain = R.plain_layout(w8).contiguous()
    wf = w8.view(torch.float8_e4m3fn)
    copies = max(2, -(-L2_DEFEAT_BYTES // (n * k)))
    ring = [plain] + [plain.clone() for _ in range(copies - 1)]
    wring = [wf] + [wf.clone() for _ in range(copies - 1)]
    rp = [ptr(b) for b in ring]
    st, so = stream(torch), lib.so
    blocks = {s: layout.block_count(n, k, s) for s in spans}
    tbuf = {s: torch.empty(blocks[s] * 32, dtype=torch.uint8, device="cuda") for s in spans}
    nodes = {s: torch.empty(blocks[s] // 16 * 8, dtype=torch.int64, device="cuda") for s in spans}
    xr = torch.zeros(1, dtype=torch.int32, device="cuda")
    read_grid = 8 * torch.cuda.get_device_properties(0).multi_processor_count
    res = {"N": n, "K": k, "copies": copies, "batches": {}, "standalone": {}}

    def check(rc, what):
        if rc:
            raise RuntimeError(f"{what}: error {rc}")

    # the tag alone and a read, batch-independent
    def arm_o(span, v, pd):
        tp = ptr(tbuf[span])

        def f():
            rc = 0
            for p in rp:
                rc |= so.tf_fused(v, span, MODE_TAGONLY, pd, 0, p, 0, 0, tp, 1, n, k, 0, 0, st)
            check(rc, f"tag-only {span} v{v} p{pd}")
        return f

    def arm_read():
        rc = 0
        for p in rp:
            rc |= so.tf_read(p, n * k, ptr(xr), read_grid, st)
        check(rc, "read")

    def arm_empty_o():
        rc = 0
        for _ in rp:
            rc |= so.tf_empty((n // 64) * (k // 1024), st)
        check(rc, "empty")

    arms = {"read": arm_read, "empty": arm_empty_o}
    for span in spans:
        for v, pd in TAG_FORMS:
            arms[f"o{span}v{v}p{pd}"] = arm_o(span, v, pd)
    res["standalone"] = interleave(lib, torch, arms, reps, copies)
    med = res["standalone"]["median_s"]
    log(f"{name} standalone: read {n * k / med['read'] / 1e9:.0f} GB/s; "
        + ", ".join(f"{a} {n * k / med[a] / 1e9:.0f} GB/s ({med[a] / med['read']:.2f}x read)"
                    for a in med if a.startswith("o")))

    gx = torch.Generator(device="cuda").manual_seed(5)
    for mb in batches:
        x = torch.randn(mb, k, device="cuda", generator=gx).to(torch.float8_e4m3fn)
        xp = ptr(x.view(torch.uint8))
        y = torch.zeros(mb, n, dtype=torch.float32, device="cuda")
        yp = ptr(y)
        one = torch.ones((), device="cuda")
        mpad = mb
        try:
            torch._scaled_mm(x, wf.t(), one, one, out_dtype=torch.bfloat16)
            xc = x
        except Exception:  # noqa: BLE001  cuBLASLt's FP8 path wants M a multiple of 16 here
            mpad = 16 * -(-mb // 16)
            xc = torch.zeros(mpad, k, device="cuda").to(torch.float8_e4m3fn)
            xc[:mb] = x
        wts = [t.t() for t in wring]

        def arm_cublas():
            for wt in wts:
                torch._scaled_mm(xc, wt, one, one, out_dtype=torch.bfloat16)

        def arm_plain():
            rc = 0
            for p in rp:
                rc |= so.tf_plain(p, xp, yp, mb, n, k, 0, st)
            check(rc, "plain")

        def arm_f(span, v, mode, pd, leaves=False):
            tp, npp, cnt = ptr(tbuf[span]), ptr(nodes[span]), blocks[span]
            xrp = ptr(xr)

            def f():
                rc = 0
                for p in rp:
                    rc |= so.tf_fused(v, span, mode, pd, 0, p, xp, yp, tp, mb, n, k, 0, xrp, st)
                    if leaves:
                        rc |= so.tf_leaves(tp, cnt, 0, npp, st)
                check(rc, f"fused {span} v{v} m{mode} p{pd}")
            return f

        def arm_empty():
            rc = 0
            for _ in rp:
                rc |= so.tf_empty((n // 64) * (k // 1024), st)
            check(rc, "empty")

        arms = {"cublas": arm_cublas, "plain": arm_plain, "empty": arm_empty}
        for span in spans:
            for pd in (1, 2):
                arms[f"g{span}p{pd}"] = arm_f(span, 0, MODE_GEMM, pd)
            for v, pd in TAG_FORMS:
                arms[f"t{span}v{v}p{pd}"] = arm_f(span, v, MODE_TAGS, pd)
            arms[f"c{span}v1p1"] = arm_f(span, 1, MODE_TAGS, 1, leaves=True)
            arms[f"s{span}v1p1"] = arm_f(span, 1, MODE_SINK, 1)
        r = interleave(lib, torch, arms, reps, copies)
        med = r["median_s"]
        native_arms = ["cublas", "plain"] + forms(med, "g")
        native = min(med[a] for a in native_arms)
        r.update({"cublas_m": mpad, "native_s": native, "native_arm": min(native_arms, key=lambda a: med[a]),
                  "ratio": {a: med[a] / native for a in med}})
        res["batches"][mb] = r
        best_t = min(forms(med, "t"), key=lambda a: med[a])
        best_c = min(forms(med, "c"), key=lambda a: med[a])
        log(f"{name} M={mb}: native {native * 1e6:.1f} us ({r['native_arm']}; cublas {med['cublas'] * 1e6:.1f}, plain "
            f"{med['plain'] * 1e6:.1f}, g32p1 {med.get('g32p1', 0) * 1e6:.1f}); best tag {best_t} {med[best_t] / native:.3f}x, "
            f"best commit {best_c} {med[best_c] / native:.3f}x; "
            + ", ".join(f"{a} {med[a] / native:.3f}" for a in forms(med, "t") + forms(med, "s"))
            + (" HOST-BOUND" if r["host_bound"] else ""))
    del ring, wring
    torch.cuda.empty_cache()
    return res


def interleave(lib: Lib, torch, arms: dict, reps: int, copies: int) -> dict:
    for f in arms.values():
        f()
    torch.cuda.synchronize()
    names = list(arms)
    times = {a: [] for a in names}
    host = {a: [] for a in names}
    for rep in range(reps):
        order = names[rep % len(names):] + names[:rep % len(names)]
        for a in order:
            t, h = queued(lib, torch, arms[a])
            times[a].append(t / copies)
            host[a].append(h)
    spin_s = SPIN_CYCLES / 2.0e9
    host_max = max(max(v) for v in host.values())
    return {"median_s": {a: statistics.median(v) for a, v in times.items()}, "times_s": times,
            "host_enqueue_max_s": host_max, "host_bound": host_max > 0.8 * spin_s}


def chains(lib: Lib, torch, sms: int, clock_hz: float, reps: int, log) -> dict:
    xr = torch.zeros(1, dtype=torch.int32, device="cuda")
    grid, block = sms * 8, 256
    threads = grid * block
    out = {}
    cases = [(f"b3_v{v}_r7", 0, v, 7, 64, 2000) for v in range(5)]
    cases += [("b3_v0_r4_uncertified", 0, 0, 4, 64, 2000), ("b3_v1_r4_uncertified", 0, 1, 4, 64, 2000),
              ("b3_v1_r5_uncertified", 0, 1, 5, 64, 2000), ("keccak_p1600_12", 1, 0, 12, 168, 600),
              ("sha512", 2, 0, 80, 128, 600)]
    for label, what, v, rounds, nbytes, steps in cases:
        ts = timed(torch, lambda: lib.call("tf_chain", what, v, rounds, steps, ptr(xr), grid, block, stream(torch)),
                   reps)
        t = statistics.median(ts)
        per_s = threads * steps / t
        out[label] = {"steps": steps, "threads": threads, "median_s": t, "compressions_per_s": per_s,
                      "bytes_per_s": per_s * nbytes, "bytes_per_sm_clock": per_s * nbytes / sms / clock_hz,
                      "warp_cycles_per_compression": sms * 4 * clock_hz / (per_s / 32)}
        log(f"chain {label}: {per_s:.3e}/s, {per_s * nbytes / 1e12:.2f} TB/s, "
            f"{out[label]['bytes_per_sm_clock']:.2f} B/SM/clk, {out[label]['warp_cycles_per_compression']:.0f} "
            f"SMSP cycles a warp-compression")
    return out


def commit_costs(lib: Lib, torch, spans, reps: int, log) -> dict:
    """One step's leaves over every tag of the store at once, and its tree, at each span."""
    out = {}
    for span in spans:
        count = layout.store_blocks(span)
        nleaves = count // 16
        tb = torch.randint(0, 1 << 30, (count * 8,), dtype=torch.int32, device="cuda")
        nodes = torch.zeros(int(lib.so.tf_tree_nodes(nleaves)) * 8, dtype=torch.int64, device="cuda")
        tl = statistics.median(timed(torch, lambda: lib.call("tf_leaves", ptr(tb), count, 0, ptr(nodes),
                                                             stream(torch)), reps))
        tt = statistics.median(timed(torch, lambda: lib.call("tf_tree", ptr(nodes), nleaves, stream(torch)), reps))
        out[span] = {"blocks": count, "leaves": nleaves, "leaves_s": tl, "tree_s": tt}
        log(f"commit span {span}: one step's {count} tags -> {nleaves} leaves {tl * 1e3:.2f} ms, tree {tt * 1e3:.2f} ms")
        del tb, nodes
        torch.cuda.empty_cache()
    return out


def summarize(res: dict, batches, spans) -> dict:
    shapes = res["timing"]
    lw = {}
    for mb in batches:
        nat = sum(shapes[s]["batches"][mb]["native_s"] for s in shapes)
        arms = shapes[next(iter(shapes))]["batches"][mb]["median_s"].keys()
        lw[mb] = {a: sum(shapes[s]["batches"][mb]["median_s"][a] for s in shapes) / nat for a in arms}
        lw[mb]["native_s"] = nat
    summary = {"layer_weighted": lw, "y_tag": {}, "y_commit": {}, "y_sink": {}, "tree_share": {}}
    for mb in batches:
        r = lw[mb]
        for key, pre in (("y_tag", "t"), ("y_commit", "c"), ("y_sink", "s")):
            best = min(forms(r, pre), key=lambda a: r[a])
            summary[key][mb] = {"arm": best, "y": r[best]}
        step = layout.LAYERS * r["native_s"]
        summary["tree_share"][mb] = {s: res["commit"][s]["tree_s"] / step for s in spans}
    return summary


def points(res: dict, batches, run: str | None) -> list[dict]:
    """One frontier point per batch and form: the tag stage's slowdown over native, layer-weighted."""
    out = []
    lw = res["summary"]["layer_weighted"]
    for mb in batches:
        for a, y in lw[mb].items():
            if not (m := FORM.match(a)) or m["stage"] not in "tcs":
                continue
            span = int(m["span"])
            out.append({"lane": "ra-tag-fast", "tag": "ro-blake3/v1", "certified": True, "block_bytes": 16 * span,
                        "form": a, "stage": {"t": "tag", "c": "tag+leaves", "s": "tag-sink"}[a[0]], "batch": mb,
                        "y": y, "run": run})
    return out


NCU_FORMS = (("g", 0, MODE_GEMM, 1), ("t", 0, MODE_TAGS, 1), ("t", 1, MODE_TAGS, 1), ("t", 3, MODE_TAGS, 1),
             ("o", 1, MODE_TAGONLY, 1))


def ncu_mode(lib: Lib, torch, shapes, batches, spans) -> int:
    lib.call("tf_init", SEED, None)
    for name in shapes:
        n, k = layout.SHAPES[name]
        w8 = torch.randint(0, 256, (n, k), dtype=torch.uint8, device="cuda") & 0x77
        plain = R.plain_layout(w8).contiguous()
        for mb in batches:
            x = torch.randn(mb, k, device="cuda").to(torch.float8_e4m3fn)
            y = torch.zeros(mb, n, dtype=torch.float32, device="cuda")
            lib.call("tf_plain", ptr(plain), ptr(x.view(torch.uint8)), ptr(y), mb, n, k, 0, stream(torch))
            for span in spans:
                tb = torch.empty(layout.block_count(n, k, span) * 32, dtype=torch.uint8, device="cuda")
                for _, v, mode, pd in NCU_FORMS:
                    fused(lib, torch, v, span, mode, pd, 0, plain, x, y, tb, mb, n, k, 0)
    torch.cuda.synchronize()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lib", required=True)
    ap.add_argument("--ref", required=True, help="reference.py's output directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=9)
    ap.add_argument("--shapes", default=",".join(layout.SHAPES))
    ap.add_argument("--batches", default=",".join(map(str, BATCHES)))
    ap.add_argument("--spans", default="32,64")
    ap.add_argument("--mode", default="all", choices=["all", "ncu"],
                    help="ncu: each profiled form once on random weights, nothing gated or timed (under Nsight Compute)")
    a = ap.parse_args()
    import torch

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        line = f"{time.strftime('%H:%M:%SZ', time.gmtime())} {msg}"
        print(line, flush=True)
        with (out_dir / "bench.log").open("a") as f:
            f.write(line + "\n")

    lib = Lib(a.lib)
    props = torch.cuda.get_device_properties(0)
    sms = props.multi_processor_count
    batches = [int(b) for b in a.batches.split(",")]
    spans = [int(s) for s in a.spans.split(",")]
    shapes = a.shapes.split(",")
    if a.mode == "ncu":
        return ncu_mode(lib, torch, shapes, batches, spans)
    refdir = Path(a.ref)
    expect = json.loads((refdir / "expect.json").read_text())
    res = {"device": {"name": props.name, "sms": sms, "cc": f"{props.major}.{props.minor}"}, "seed": SEED.hex(),
           "lib": a.lib, "ok": True, "gates": {}, "run": os.environ.get("RESEARCH_RUN_ID")}
    log(f"device {props.name}, {sms} SMs")

    key = (ctypes.c_uint8 * 32)()
    lib.call("tf_init", SEED, key)
    res["gates"]["key"] = bytes(key) == tags.Blake3Tag().key(SEED) == bytes.fromhex(expect["key"])
    log(f"gate key: {'ok' if res['gates']['key'] else 'FAIL'}")
    res["gates"]["compress"] = gate_compress(lib, torch, log)
    w8s, refs = small_reference(torch, 64, 4096, R.JBASE + (1 << 20))
    res["gates"]["small"] = gate_store(lib, torch, "small", 64, 4096, w8s.cuda(), to_device_ref(torch, refs),
                                       (1, 8, 16, 32), log)
    res["ok"] = res["gates"]["key"] and res["gates"]["compress"]["ok"] and res["gates"]["small"]["ok"]
    weights = {}
    for name in shapes:
        if not res["ok"]:
            break
        w8, ref = load_ref(torch, refdir, name, expect)
        n, k = expect["shapes"][name]["N"], expect["shapes"][name]["K"]
        w8 = w8.cuda()
        ref = {s: r for s, r in ref.items() if s in spans}
        g = gate_store(lib, torch, name, n, k, w8, to_device_ref(torch, ref), batches, log)
        res["gates"][name] = g
        res["ok"] &= g["ok"]
        weights[name] = (n, k, w8)
    (out_dir / "tag-fast.json").write_text(json.dumps(res, indent=1, default=str))
    if not res["ok"]:
        log("a gate failed: nothing timed")
        return 5

    spin = statistics.median(timed(torch, lambda: lib.call("tf_spin", SPIN_CYCLES, stream(torch)), 3))
    clock_hz = SPIN_CYCLES / spin
    res["clock_hz"] = clock_hz
    log(f"SM clock from the spin: {clock_hz / 1e9:.3f} GHz")
    res["chains"] = chains(lib, torch, sms, clock_hz, 5, log)
    res["commit"] = commit_costs(lib, torch, spans, 5, log)
    res["timing"] = {}
    for name in shapes:
        n, k, w8 = weights[name]
        res["timing"][name] = time_shape(lib, torch, name, n, k, w8, spans, batches, a.reps, log)
        (out_dir / "tag-fast.json").write_text(json.dumps(res, indent=1, default=str))
    if len(shapes) == len(layout.SHAPES):
        res["summary"] = summarize(res, batches, spans)
        res["points"] = points(res, batches, res["run"])
        for mb in batches:
            s = res["summary"]
            log(f"layer-weighted M={mb}: y_tag {s['y_tag'][mb]['y']:.3f} ({s['y_tag'][mb]['arm']}), y_commit "
                f"{s['y_commit'][mb]['y']:.3f} ({s['y_commit'][mb]['arm']}), sink {s['y_sink'][mb]['y']:.3f}, "
                f"tree per step " + ", ".join(f"span {sp} {v:.2%}" for sp, v in s["tree_share"][mb].items()))
    (out_dir / "tag-fast.json").write_text(json.dumps(res, indent=1, default=str))
    log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
