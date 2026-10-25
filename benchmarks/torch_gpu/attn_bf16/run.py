#!/usr/bin/env python3
"""attn_bf16 on one H100/H200 (sm_90a) or RTX PRO 6000 Blackwell (sm_120a): the prebuilt attn_bf16.<arch>.cubin through the
CUDA driver API (ctypes on libcuda.so.1; numpy for the buffers, torch only for the timing's baselines).

0. Identity, before anything runs: exactly one visible device, a card of CARDS (ATTN_EXPECT, a name prefix, narrows it),
   and its UUID equal to GPU_LEASE_UUID when gpu-lease set it.  A mismatch exits 3 before any launch.
1. The check, on every case in CHECK (fixture.py's): S and O are filled with 0xA5 and QK^T and PV run on the case's
   inputs; each buffer must equal the fixture's byte for byte (every word the Definition defines, and 0xA5A5 everywhere
   else).  Each tiling runs at 4 warps a block, then at 1 and at 8, which give the tiles to other warps: the same bytes.
   The tilings are q8 (a block per 8 query rows, head and sequence) and, where 8 divides the query heads a KV head serves,
   h8 (a block per query row, 8 heads and sequence).  A control arm fills the buffers and launches nothing, and must be
   rejected.  The inputs must come back unchanged.  A FULL case (<case>.full.npz) makes its inputs with gen.py, checks
   them against the fixture's digests (exit 5 if the generator disagrees), and compares each sequence's S and O digests.
   Exit 4 if any check fails, before anything is timed.
2. Diagnostic timing (information only, not a panel row), at the H200 model's decode step (TIME_B sequences, TIME_H query
   heads of head dim 64 on TIME_HKV KV heads, every KV length TIME_L): QK^T and PV here in both tilings, each the median of
   REPS after a warm-up, interleaved rep by rep with torch's SDPA (its default choice of backend, and FlashAttention's
   alone; both include the softmax these two leave out) and with torch's two bf16 batched matmuls (cuBLAS), on the card.

One JSON object per line on stdout as each part finishes, a summary on stderr.

  python3 run.py CHECK [CASE ...]   (env: REPS=20, TIME_B=20480, TIME_H=256, TIME_HKV=1, TIME_L=32, ATTN_EXPECT,
                                     ATTN_LIBCUDA, ATTN_ANY_GPU=1 skips the identity check for a dry run on another card)
"""

from __future__ import annotations

import ctypes as C
import glob
import hashlib
import json
import os
import statistics
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gen  # noqa: E402

#: compute capability: (accepted name prefixes, the cubin's arch)
CARDS = {"9.0": (("NVIDIA H100", "NVIDIA H200"), "sm_90a"), "12.0": (("NVIDIA RTX PRO 6000 Blackwell",), "sm_120a")}
CC_MAJOR, CC_MINOR = 75, 76
POISON = 0xA5
WARPS = (4, 1, 8)

u, i32, vp, sz, u64, P = C.c_uint, C.c_int, C.c_void_p, C.c_size_t, C.c_uint64, C.POINTER
CUDA = {
    "cuInit": [u], "cuDeviceGetCount": [P(i32)], "cuDeviceGet": [P(i32), i32], "cuDeviceGetName": [C.c_char_p, i32, i32],
    "cuDeviceGetUuid_v2": [C.c_char_p, i32], "cuDeviceGetAttribute": [P(i32), i32, i32], "cuDriverGetVersion": [P(i32)],
    "cuDevicePrimaryCtxRetain": [P(vp), i32], "cuCtxSetCurrent": [vp], "cuModuleLoadData": [P(vp), C.c_char_p],
    "cuModuleGetFunction": [P(vp), vp, C.c_char_p], "cuMemAlloc_v2": [P(u64), sz], "cuMemFree_v2": [u64],
    "cuMemsetD8_v2": [u64, C.c_ubyte, sz], "cuMemcpyHtoD_v2": [u64, vp, sz], "cuMemcpyDtoH_v2": [vp, u64, sz],
    "cuLaunchKernel": [vp, u, u, u, u, u, u, u, vp, P(vp), P(vp)], "cuCtxSynchronize": [],
    "cuEventCreate": [P(vp), u], "cuEventRecord": [vp, vp], "cuEventSynchronize": [vp],
    "cuEventElapsedTime": [P(C.c_float), vp, vp], "cuGetErrorString": [i32, P(C.c_char_p)],
}


class Args(C.Structure):
    """attn_bf16.cu's AttnArgs."""
    _fields_ = [("x", u64), ("kv", u64), ("out", u64), ("lens", u64), ("nq", u64), ("xs", C.c_int64 * 3),
                ("kvs", C.c_int64 * 3), ("os", C.c_int64 * 3), ("B", C.c_int32), ("H", C.c_int32), ("Mq", C.c_int32),
                ("Lcap", C.c_int32), ("gshift", C.c_int32)]


def emit(obj: dict) -> None:
    print(json.dumps(obj), flush=True)


class Cuda:
    def __init__(self):
        self.lib = C.CDLL(os.environ.get("ATTN_LIBCUDA", "libcuda.so.1"))
        for f, a in CUDA.items():
            getattr(self.lib, f).argtypes, getattr(self.lib, f).restype = a, i32
        self("cuInit", 0)

    def __call__(self, f: str, *args):
        r = getattr(self.lib, f)(*args)
        if r:
            s = C.c_char_p()
            self.lib.cuGetErrorString(r, C.byref(s))
            raise RuntimeError(f"{f}: {r} {s.value.decode() if s.value else ''}")

    def identity(self) -> dict:
        n, dev, maj, mnr, ver = i32(), i32(), i32(), i32(), i32()
        self("cuDeviceGetCount", C.byref(n))
        self("cuDeviceGet", C.byref(dev), 0)
        name, raw = C.create_string_buffer(256), C.create_string_buffer(16)
        self("cuDeviceGetName", name, 256, dev)
        self("cuDeviceGetUuid_v2", raw, dev)
        self("cuDeviceGetAttribute", C.byref(maj), CC_MAJOR, dev)
        self("cuDeviceGetAttribute", C.byref(mnr), CC_MINOR, dev)
        self("cuDriverGetVersion", C.byref(ver))
        h = raw.raw[:16].hex()
        uuid = f"GPU-{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:]}"
        ctx = vp()
        self("cuDevicePrimaryCtxRetain", C.byref(ctx), dev)
        self("cuCtxSetCurrent", ctx)
        return {"devices": n.value, "name": name.value.decode(), "cc": f"{maj.value}.{mnr.value}", "uuid": uuid,
                "driver_api": ver.value, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}

    def alloc(self, nbytes: int) -> int:
        p = u64()
        self("cuMemAlloc_v2", C.byref(p), max(nbytes, 4))
        return p.value

    def put(self, a: np.ndarray) -> int:
        a = np.ascontiguousarray(a)
        p = self.alloc(a.nbytes)
        self("cuMemcpyHtoD_v2", p, a.ctypes.data, a.nbytes)
        return p

    def get(self, p: int, like: np.ndarray) -> np.ndarray:
        out = np.empty_like(like)
        self("cuMemcpyDtoH_v2", out.ctypes.data, p, out.nbytes)
        return out

    def event(self) -> vp:
        e = vp()
        self("cuEventCreate", C.byref(e), 0)
        return e


def strides(shape: tuple) -> tuple:
    """Element strides of batch, head and row of a contiguous [B, H, rows, cols] buffer."""
    return shape[1] * shape[2] * shape[3], shape[2] * shape[3], shape[3]


class Attn:
    """The cubin's kernels over one set of device buffers: Q, P [B, H, *, *], K, V [B, Hkv, *, *] bf16, lens and nq [B] int32."""

    def __init__(self, cu: Cuda, funcs: dict, z: dict):
        self.cu, self.funcs = cu, funcs
        self.D = z["Q"].shape[-1]
        self.B, self.H, self.Mq, self.Lcap = z["P"].shape
        self.Hkv = z["K"].shape[1]
        G = self.H // self.Hkv
        assert self.H % self.Hkv == 0 and G & (G - 1) == 0, (self.H, self.Hkv)
        self.gshift = G.bit_length() - 1
        self.tilings = ("q8", "h8") if G % 8 == 0 else ("q8",)
        self.s_shape, self.o_shape = (self.B, self.H, self.Mq, self.Lcap), (self.B, self.H, self.Mq, self.D)
        self.dev = {k: cu.put(z[k]) for k in ("Q", "K", "V", "P", "lens", "nq")}
        self.S = cu.alloc(int(np.prod(self.s_shape)) * 2)
        self.O = cu.alloc(int(np.prod(self.o_shape)) * 2)
        self.qk_args = self._args("Q", "K", self.S, z["Q"].shape, z["K"].shape, self.s_shape)
        self.pv_args = self._args("P", "V", self.O, z["P"].shape, z["V"].shape, self.o_shape)

    def _args(self, x: str, kv: str, out: int, xs: tuple, kvs: tuple, os_: tuple) -> Args:
        return Args(self.dev[x], self.dev[kv], out, self.dev["lens"], self.dev["nq"], (C.c_int64 * 3)(*strides(xs)),
                    (C.c_int64 * 3)(*strides(kvs)), (C.c_int64 * 3)(*strides(os_)), self.B, self.H, self.Mq, self.Lcap,
                    self.gshift)

    def kernel(self, op: str, tiling: str):
        return self.funcs[f"attn_{op}_d{self.D}" + ("_h8" if tiling == "h8" else "")]

    def launch(self, op: str, tiling: str, warps: int) -> None:
        """q8: a block per (8 query rows, head, sequence); h8: per (query row, 8 heads, sequence); of ``warps`` warps."""
        args = self.qk_args if op == "qk" else self.pv_args
        grid = (-(-self.Mq // 8), self.H, self.B) if tiling == "q8" else (self.Mq, self.H // 8, self.B)
        params = (vp * 1)(C.cast(C.pointer(args), vp))
        self.cu("cuLaunchKernel", self.kernel(op, tiling), *grid, 32 * warps, 1, 1, 0, None, params, None)

    def poison(self) -> None:
        self.cu("cuMemsetD8_v2", self.S, POISON, int(np.prod(self.s_shape)) * 2)
        self.cu("cuMemsetD8_v2", self.O, POISON, int(np.prod(self.o_shape)) * 2)
        self.cu("cuCtxSynchronize")

    def run(self, tiling: str, warps: int) -> None:
        self.launch("qk", tiling, warps)
        self.launch("pv", tiling, warps)
        self.cu("cuCtxSynchronize")

    def outputs(self) -> tuple[np.ndarray, np.ndarray]:
        return (self.cu.get(self.S, np.empty(self.s_shape, np.uint16)), self.cu.get(self.O, np.empty(self.o_shape, np.uint16)))

    def free(self) -> None:
        for p in (*self.dev.values(), self.S, self.O):
            self.cu("cuMemFree_v2", p)


def sha(a: np.ndarray) -> str:
    return hashlib.sha512(np.ascontiguousarray(a).tobytes()).hexdigest()[:32]


def _head(name: str, z: dict, a: Attn) -> dict:
    return {"part": "check", "case": name, "B": a.B, "H": a.H, "Hkv": a.Hkv, "D": a.D, "Mq": a.Mq, "Lcap": a.Lcap,
            "lens": [int(z["lens"].min()), int(z["lens"].max())], "nq": [int(z["nq"].min()), int(z["nq"].max())],
            "tilings": list(a.tilings), "arms": {}}


def check_case(cu: Cuda, funcs: dict, path: str) -> dict:
    name = os.path.basename(path)[: -len(".npz")]
    z = dict(np.load(path))
    a = Attn(cu, funcs, z)
    row = _head(name, z, a)
    row["words"] = {"S": int((z["S"] != 0xA5A5).sum()), "O": int((z["O"] != 0xA5A5).sum())}
    ok = True
    for tiling in a.tilings:
        for warps in WARPS:
            a.poison()
            a.run(tiling, warps)
            S, O = a.outputs()
            arm = {"S_mismatches": int((S != z["S"]).sum()), "O_mismatches": int((O != z["O"]).sum()), "S_sha512": sha(S), "O_sha512": sha(O)}
            if arm["S_mismatches"] or arm["O_mismatches"]:
                ok = False
                arm["examples"] = [{"buffer": k, "at": i, "got": int(got[tuple(i)]), "want": int(z[k][tuple(i)])}
                                   for k, got in (("S", S), ("O", O)) for i in np.argwhere(got != z[k])[:4].tolist()]
            row["arms"][f"{tiling}@{warps}w"] = arm
    a.poison()
    S, O = a.outputs()
    ctl = {"S_mismatches": int((S != z["S"]).sum()), "O_mismatches": int((O != z["O"]).sum())}
    ctl["rejected"] = bool(ctl["S_mismatches"] and ctl["O_mismatches"])
    ok &= ctl["rejected"]
    row["arms"]["control"] = ctl
    row["inputs_unchanged"] = all(np.array_equal(cu.get(a.dev[k], z[k]), z[k]) for k in ("Q", "K", "V", "P", "lens", "nq"))
    ok &= row["inputs_unchanged"]
    row["ok"] = ok
    a.free()
    return row


def full_case(cu: Cuda, funcs: dict, path: str) -> dict:
    """A FULL case: gen.py's inputs, held to the fixture's digests, and each sequence's S and O digests."""
    name = os.path.basename(path)[: -len(".full.npz")]
    f = dict(np.load(path))
    B, H, Hkv, D, Lcap, seed, chunk = (int(v) for v in f["shape"])
    z = {k: gen.array(seed, k, s) for k, s in gen.shapes(B, H, Hkv, D, Lcap).items()}
    z["lens"], z["nq"] = gen.lens_nq(B, Lcap)
    bad_in = {k: int(sum(not np.array_equal(gen.digests(z[k][b0:b0 + chunk][None])[0], f[f"in_{k}"][i])
                         for i, b0 in enumerate(range(0, B, chunk)))) for k in ("Q", "K", "V", "P")}
    a = Attn(cu, funcs, z)
    row = _head(name, z, a)
    row.update({"full": True, "seed": seed, "input_chunks_differing": bad_in, "words": {"S": H * int(z["lens"].sum()), "O": B * H * D}})
    if any(bad_in.values()):
        row["ok"] = False
        row["generator_disagrees"] = True
        a.free()
        return row
    ok = True
    for tiling in a.tilings:
        for warps in WARPS:
            a.poison()
            a.run(tiling, warps)
            S, O = a.outputs()
            sd, od = gen.digests(S), gen.digests(O)
            arm = {"S_sequences_differing": int((sd != f["S"]).any(axis=1).sum()), "O_sequences_differing": int((od != f["O"]).any(axis=1).sum()),
                   "S_sha512": sha(S), "O_sha512": sha(O)}
            if arm["S_sequences_differing"] or arm["O_sequences_differing"]:
                ok = False
                arm["examples"] = {k: np.flatnonzero((d != f[k]).any(axis=1))[:8].tolist() for k, d in (("S", sd), ("O", od))}
            row["arms"][f"{tiling}@{warps}w"] = arm
    a.poison()
    S, O = a.outputs()
    ctl = {"S_sequences_differing": int((gen.digests(S) != f["S"]).any(axis=1).sum()),
           "O_sequences_differing": int((gen.digests(O) != f["O"]).any(axis=1).sum())}
    ctl["rejected"] = bool(ctl["S_sequences_differing"] == B and ctl["O_sequences_differing"] == B)
    ok &= ctl["rejected"]
    row["arms"]["control"] = ctl
    row["inputs_unchanged"] = all(np.array_equal(cu.get(a.dev[k], z[k]), z[k]) for k in ("Q", "K", "V", "P", "lens", "nq"))
    ok &= row["inputs_unchanged"]
    row["ok"] = ok
    a.free()
    return row


def timing(cu: Cuda, funcs: dict, reps: int) -> dict:
    """The H200 model's decode step at every KV length TIME_L: these kernels, SDPA and torch's matmuls, interleaved."""
    B, H, Hkv, L = (int(os.environ.get(k, d)) for k, d in (("TIME_B", 20480), ("TIME_H", 256), ("TIME_HKV", 1), ("TIME_L", 32)))
    D, G = 64, H // Hkv
    rng = np.random.default_rng(0)

    def words(shape: tuple) -> np.ndarray:
        return (rng.standard_normal(shape, np.float32).view(np.uint32) >> 16).astype(np.uint16)

    z = {"Q": words((B, H, 1, D)), "K": words((B, Hkv, L, D)), "V": words((B, Hkv, L, D)), "P": words((B, H, 1, L)),
         "lens": np.full(B, L, np.int32), "nq": np.ones(B, np.int32)}
    a = Attn(cu, funcs, z)
    e0, e1 = cu.event(), cu.event()
    ms = C.c_float()

    def ours(op, tiling, warps):
        cu("cuEventRecord", e0, None)
        a.launch(op, tiling, warps)
        cu("cuEventRecord", e1, None)
        cu("cuEventSynchronize", e1)
        cu("cuEventElapsedTime", C.byref(ms), e0, e1)
        return ms.value

    # a warp per key tile of QK^T and per d tile of PV
    qk_w, pv_w = min(8, -(-L // 16)), D // 16
    arms = {}
    for tiling in a.tilings:
        arms[f"qk_{tiling}"] = lambda tiling=tiling: ours("qk", tiling, qk_w)
        arms[f"pv_{tiling}"] = lambda tiling=tiling: ours("pv", tiling, pv_w)
    out = {"part": "timing", "shape": {"B": B, "H": H, "Hkv": Hkv, "D": D, "L": L, "queries_per_sequence": 1}, "reps": reps,
           "warps": {"qk": qk_w, "pv": pv_w}, "note": "diagnostic only; SDPA includes the softmax qk and pv leave out"}
    try:
        import torch
        import torch.nn.functional as F

        if not torch.cuda.is_available():
            raise ImportError("torch without CUDA")
        dev = torch.device("cuda")
        q = torch.randn((B, H, 1, D), device=dev, dtype=torch.bfloat16)
        k, v = (torch.randn((B, Hkv, L, D), device=dev, dtype=torch.bfloat16) for _ in range(2))
        p = torch.rand((B, H, 1, L), device=dev, dtype=torch.bfloat16)
        ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        gqa = {"enable_gqa": True} if Hkv != H else {}

        def tt(fn):
            ev0.record()
            fn()
            ev1.record()
            ev1.synchronize()
            return ev0.elapsed_time(ev1)

        def sdpa():
            return F.scaled_dot_product_attention(q, k, v, **gqa)

        try:
            sdpa()
            arms["sdpa"] = lambda: tt(sdpa)
        except Exception as e:  # noqa: BLE001 - a torch without GQA in SDPA is reported, not fatal
            out["sdpa"] = f"unavailable: {type(e).__name__}: {str(e)[:200]}"
        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel

            def flash():
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    return sdpa()

            flash()
            arms["sdpa_flash"] = lambda: tt(flash)
        except Exception as e:  # noqa: BLE001 - a backend this torch build lacks is reported, not fatal
            out["sdpa_flash"] = f"unavailable: {type(e).__name__}: {str(e)[:200]}"
        # the G query heads of a KV head as one (G x D) matrix a sequence: S = Q K^T and O = P V as batched GEMMs
        qg, pg, kt = q.view(B, Hkv, G, D), p.view(B, Hkv, G, L), k.transpose(-1, -2)
        arms["torch_qk"] = lambda: tt(lambda: torch.matmul(qg, kt))
        arms["torch_pv"] = lambda: tt(lambda: torch.matmul(pg, v))
        out["torch"] = torch.__version__
    except ImportError as e:
        out["torch"] = f"unavailable: {e}"
    for fn in arms.values():
        for _ in range(3):
            fn()
    got = {arm: [] for arm in arms}
    for _ in range(reps):
        for arm, fn in arms.items():
            got[arm].append(fn())
    out["median_ms"] = {arm: round(statistics.median(ts), 4) for arm, ts in got.items()}
    out["ours_qk_plus_pv_ms"] = {t: round(out["median_ms"][f"qk_{t}"] + out["median_ms"][f"pv_{t}"], 4) for t in a.tilings}
    a.free()
    return out


def main(argv: list[str]) -> int:
    check, names = argv[1], argv[2:]
    cu = Cuda()
    ident = cu.identity()
    want_uuid = os.environ.get("GPU_LEASE_UUID")
    expect = os.environ.get("ATTN_EXPECT")
    names_ok, arch = CARDS.get(ident["cc"], ((), None))
    problems = []
    if not os.environ.get("ATTN_ANY_GPU"):
        if ident["devices"] != 1:
            problems.append(f"{ident['devices']} visible devices, want 1")
        if not arch or not ident["name"].startswith(names_ok) or (expect and not ident["name"].startswith(expect)):
            problems.append(f"{ident['name']} cc {ident['cc']}, want {expect or 'a card of ' + json.dumps(CARDS)}")
        if want_uuid and ident["uuid"] != want_uuid:
            problems.append(f"uuid {ident['uuid']}, the lease's {want_uuid}")
    emit({"part": "identity", **ident, "arch": arch, "expect": expect, "lease_uuid": want_uuid, "problems": problems})
    if problems or not arch:
        print("REFUSED: " + "; ".join(problems or [f"no cubin for cc {ident['cc']}"]), file=sys.stderr)
        return 3
    cubin = open(os.path.join(HERE, f"attn_bf16.{arch}.cubin"), "rb").read()
    mod = vp()
    cu("cuModuleLoadData", C.byref(mod), cubin)
    funcs = {}
    for op in ("qk", "pv"):
        for d in (64, 128):
            for tile in ("", "_h8"):
                f, name = vp(), f"attn_{op}_d{d}{tile}"
                cu("cuModuleGetFunction", C.byref(f), mod, name.encode())
                funcs[name] = f
    paths = [os.path.join(check, f"{n}.npz") for n in names] or sorted(glob.glob(os.path.join(check, "*.npz")))
    rows = []
    for p in paths:
        r = full_case(cu, funcs, p) if p.endswith(".full.npz") else check_case(cu, funcs, p)
        rows.append(r)
        emit(r)
        arms = r["arms"]
        mism = (lambda v: f"{v['S_mismatches']}+{v['O_mismatches']}") if not r.get("full") else \
            (lambda v: f"{v['S_sequences_differing']}+{v['O_sequences_differing']} sequences")
        print(f"{r['case']}: {'ok' if r['ok'] else 'FAILED'}  S {r['words']['S']} O {r['words']['O']} words, mismatches "
              + ", ".join(f"{k} {mism(v)}" for k, v in arms.items() if k != "control")
              + f", control rejected {arms.get('control', {}).get('rejected')}", file=sys.stderr)
        if r.get("generator_disagrees"):
            print(f"REFUSED: {r['case']}: gen.py's inputs differ from the fixture's ({r['input_chunks_differing']})", file=sys.stderr)
            return 5
    if not all(r["ok"] for r in rows):
        print("REFUSED: a case failed; nothing timed", file=sys.stderr)
        return 4
    t = timing(cu, funcs, int(os.environ.get("REPS", "20")))
    emit(t)
    print(f"timing {t['shape']}: {t['median_ms']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
