#!/usr/bin/env python3
"""attn_bf16 on one RTX PRO 6000 Blackwell (sm_120): the prebuilt attn_bf16.cubin through the CUDA driver API (ctypes on
libcuda.so.1; numpy for the buffers, torch only for the timing's baselines).

0. Identity, before anything runs: exactly one visible device, compute capability 12.0, an "NVIDIA RTX PRO 6000
   Blackwell" name, and its UUID equal to GPU_LEASE_UUID when gpu-lease set it.  A mismatch exits 3 before any launch.
1. The check, on every case in CHECK (fixture.py's): S and O are filled with 0xA5 and QK^T and PV run on the case's
   inputs; each buffer must equal the fixture's byte for byte (every word the Definition defines, and 0xA5A5 everywhere
   else).  They run at 4 warps a block, then at 1 and at 8, which give every tile to another warp of another block: the
   same bytes.  A control arm fills the buffers and launches nothing, and must be rejected.  The inputs must come back
   unchanged.  Exit 4 if any check fails, before anything is timed.
2. Diagnostic timing (information only, not a panel row), at the model's decode step (B sequences, H heads of head
   dim 64, every KV length 32): QK^T and PV here, each the median of REPS after a warm-up, interleaved rep by rep with
   torch's SDPA (its default choice of backend, and FlashAttention's alone; both include the softmax these two leave
   out) and with torch's two bf16 matmuls (cuBLAS), all on the same card.

One JSON object per line on stdout as each part finishes, a summary on stderr.

  python3 run.py CHECK [CASE ...]   (env: REPS=20, TIME_B=1023, TIME_H=128, TIME_L=32, ATTN_LIBCUDA, ATTN_ANY_GPU=1 skips
                                     the identity check for a dry run on another card)
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
NAME_PREFIX = "NVIDIA RTX PRO 6000 Blackwell"
CC_MAJOR, CC_MINOR = 75, 76
POISON = 0xA5
WARPS = (4, 1, 8)
KERNELS = {64: ("attn_qk_d64", "attn_pv_d64"), 128: ("attn_qk_d128", "attn_pv_d128")}

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
                ("Lcap", C.c_int32)]


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
    """The cubin's kernels over one set of device buffers: Q, K, V, P [B, H, *, *] bf16, lens and nq [B] int32."""

    def __init__(self, cu: Cuda, funcs: dict, z: dict):
        self.cu, self.z = cu, z
        self.D = z["Q"].shape[-1]
        self.B, self.H, self.Mq, self.Lcap = z["P"].shape
        self.qk_f, self.pv_f = (funcs[k] for k in KERNELS[self.D])
        self.dev = {k: cu.put(z[k]) for k in ("Q", "K", "V", "P", "lens", "nq")}
        self.S = cu.alloc(self.B * self.H * self.Mq * self.Lcap * 2)
        self.O = cu.alloc(self.B * self.H * self.Mq * self.D * 2)
        s_shape, o_shape = (self.B, self.H, self.Mq, self.Lcap), (self.B, self.H, self.Mq, self.D)
        self.qk_args = self._args("Q", "K", self.S, z["Q"].shape, z["K"].shape, s_shape)
        self.pv_args = self._args("P", "V", self.O, z["P"].shape, z["V"].shape, o_shape)
        self.qk_tiles = self.B * self.H * -(-self.Mq // 8) * -(-self.Lcap // 16)
        self.pv_tiles = self.B * self.H * -(-self.Mq // 8) * (self.D // 16)

    def _args(self, x: str, kv: str, out: int, xs: tuple, kvs: tuple, os_: tuple) -> Args:
        return Args(self.dev[x], self.dev[kv], out, self.dev["lens"], self.dev["nq"], (C.c_int64 * 3)(*strides(xs)),
                    (C.c_int64 * 3)(*strides(kvs)), (C.c_int64 * 3)(*strides(os_)), self.B, self.H, self.Mq, self.Lcap)

    def launch(self, f, args: Args, tiles: int, warps: int) -> None:
        params = (vp * 1)(C.cast(C.pointer(args), vp))
        self.cu("cuLaunchKernel", f, -(-tiles // warps), 1, 1, 32 * warps, 1, 1, 0, None, params, None)

    def poison(self) -> None:
        self.cu("cuMemsetD8_v2", self.S, POISON, self.B * self.H * self.Mq * self.Lcap * 2)
        self.cu("cuMemsetD8_v2", self.O, POISON, self.B * self.H * self.Mq * self.D * 2)
        self.cu("cuCtxSynchronize")

    def run(self, warps: int) -> None:
        self.launch(self.qk_f, self.qk_args, self.qk_tiles, warps)
        self.launch(self.pv_f, self.pv_args, self.pv_tiles, warps)
        self.cu("cuCtxSynchronize")

    def outputs(self) -> tuple[np.ndarray, np.ndarray]:
        return self.cu.get(self.S, self.z["S"]), self.cu.get(self.O, self.z["O"])

    def free(self) -> None:
        for p in (*self.dev.values(), self.S, self.O):
            self.cu("cuMemFree_v2", p)


def sha(a: np.ndarray) -> str:
    return hashlib.sha512(np.ascontiguousarray(a).tobytes()).hexdigest()[:32]


def check_case(cu: Cuda, funcs: dict, path: str) -> dict:
    name = os.path.basename(path)[: -len(".npz")]
    z = dict(np.load(path))
    a = Attn(cu, funcs, z)
    row = {"part": "check", "case": name, "B": a.B, "H": a.H, "D": a.D, "Mq": a.Mq, "Lcap": a.Lcap,
           "lens": [int(z["lens"].min()), int(z["lens"].max())], "nq": [int(z["nq"].min()), int(z["nq"].max())],
           "words": {"S": int((z["S"] != 0xA5A5).sum()), "O": int((z["O"] != 0xA5A5).sum())}, "arms": {}}
    ok = True
    for warps in WARPS:
        a.poison()
        a.run(warps)
        S, O = a.outputs()
        arm = {"S_mismatches": int((S != z["S"]).sum()), "O_mismatches": int((O != z["O"]).sum()), "S_sha512": sha(S), "O_sha512": sha(O)}
        if arm["S_mismatches"] or arm["O_mismatches"]:
            ok = False
            arm["examples"] = [{"buffer": k, "at": i, "got": int(got[tuple(i)]), "want": int(z[k][tuple(i)])}
                               for k, got in (("S", S), ("O", O)) for i in np.argwhere(got != z[k])[:4].tolist()]
        row["arms"][f"kernel@{warps}w"] = arm
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


def timing(cu: Cuda, funcs: dict, reps: int) -> dict:
    """The model's decode step at every KV length TIME_L: these kernels, SDPA and torch's matmuls, interleaved."""
    B, H, L = (int(os.environ.get(k, d)) for k, d in (("TIME_B", 1023), ("TIME_H", 128), ("TIME_L", 32)))
    D = 64
    rng = np.random.default_rng(0)

    def words(shape: tuple) -> np.ndarray:
        return (rng.standard_normal(shape, np.float32).view(np.uint32) >> 16).astype(np.uint16)

    z = {"Q": words((B, H, 1, D)), "K": words((B, H, L, D)), "V": words((B, H, L, D)), "P": words((B, H, 1, L)),
         "lens": np.full(B, L, np.int32), "nq": np.ones(B, np.int32), "S": np.zeros((B, H, 1, L), np.uint16),
         "O": np.zeros((B, H, 1, D), np.uint16)}
    a = Attn(cu, funcs, z)
    e0, e1 = cu.event(), cu.event()
    ms = C.c_float()

    def ours(f, args, tiles):
        cu("cuEventRecord", e0, None)
        a.launch(f, args, tiles, WARPS[0])
        cu("cuEventRecord", e1, None)
        cu("cuEventSynchronize", e1)
        cu("cuEventElapsedTime", C.byref(ms), e0, e1)
        return ms.value

    arms = {"qk": lambda: ours(a.qk_f, a.qk_args, a.qk_tiles), "pv": lambda: ours(a.pv_f, a.pv_args, a.pv_tiles)}
    out = {"part": "timing", "shape": {"B": B, "H": H, "D": D, "L": L, "queries_per_sequence": 1}, "reps": reps,
           "note": "diagnostic only; SDPA includes the softmax qk and pv leave out"}
    try:
        import torch
        import torch.nn.functional as F

        if not torch.cuda.is_available():
            raise ImportError("torch without CUDA")
        dev = torch.device("cuda")
        q, k, v = (torch.randn(s, device=dev, dtype=torch.bfloat16) for s in ((B, H, 1, D), (B, H, L, D), (B, H, L, D)))
        p = torch.rand((B, H, 1, L), device=dev, dtype=torch.bfloat16)
        ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

        def tt(fn):
            ev0.record()
            fn()
            ev1.record()
            ev1.synchronize()
            return ev0.elapsed_time(ev1)

        arms["sdpa"] = lambda: tt(lambda: F.scaled_dot_product_attention(q, k, v))
        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel

            def flash():
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    return F.scaled_dot_product_attention(q, k, v)

            flash()
            arms["sdpa_flash"] = lambda: tt(flash)
        except Exception as e:  # noqa: BLE001 - a backend this torch build lacks is reported, not fatal
            out["sdpa_flash"] = f"unavailable: {type(e).__name__}: {str(e)[:200]}"
        kt = k.transpose(-1, -2)
        arms["torch_qk"] = lambda: tt(lambda: torch.matmul(q, kt))
        arms["torch_pv"] = lambda: tt(lambda: torch.matmul(p, v))
        out["torch"] = torch.__version__
    except ImportError as e:
        out["torch"] = f"unavailable: {e}"
    for fn in arms.values():
        for _ in range(3):
            fn()
    got = {k: [] for k in arms}
    for _ in range(reps):
        for k, fn in arms.items():
            got[k].append(fn())
    out["median_ms"] = {k: round(statistics.median(v), 4) for k, v in got.items()}
    out["ours_qk_plus_pv_ms"] = round(out["median_ms"]["qk"] + out["median_ms"]["pv"], 4)
    a.free()
    return out


def main(argv: list[str]) -> int:
    check, names = argv[1], argv[2:]
    cu = Cuda()
    ident = cu.identity()
    want_uuid = os.environ.get("GPU_LEASE_UUID")
    problems = []
    if not os.environ.get("ATTN_ANY_GPU"):
        if ident["devices"] != 1:
            problems.append(f"{ident['devices']} visible devices, want 1")
        if ident["cc"] != "12.0" or not ident["name"].startswith(NAME_PREFIX):
            problems.append(f"{ident['name']} cc {ident['cc']}, want {NAME_PREFIX} cc 12.0")
        if want_uuid and ident["uuid"] != want_uuid:
            problems.append(f"uuid {ident['uuid']}, the lease's {want_uuid}")
    emit({"part": "identity", **ident, "lease_uuid": want_uuid, "problems": problems})
    if problems:
        print("REFUSED: " + "; ".join(problems), file=sys.stderr)
        return 3
    cubin = open(os.path.join(HERE, "attn_bf16.cubin"), "rb").read()
    mod = vp()
    cu("cuModuleLoadData", C.byref(mod), cubin)
    funcs = {}
    for name in (n for pair in KERNELS.values() for n in pair):
        f = vp()
        cu("cuModuleGetFunction", C.byref(f), mod, name.encode())
        funcs[name] = f
    paths = [os.path.join(check, f"{n}.npz") for n in names] or sorted(glob.glob(os.path.join(check, "*.npz")))
    rows = [check_case(cu, funcs, p) for p in paths]
    for r in rows:
        emit(r)
        arms = r["arms"]
        print(f"{r['case']}: {'ok' if r['ok'] else 'FAILED'}  S {r['words']['S']} O {r['words']['O']} words, mismatches "
              + ", ".join(f"{k} {v['S_mismatches']}+{v['O_mismatches']}" for k, v in arms.items())
              + f", control rejected {arms['control']['rejected']}", file=sys.stderr)
    if not all(r["ok"] for r in rows):
        print("REFUSED: a case failed; nothing timed", file=sys.stderr)
        return 4
    t = timing(cu, funcs, int(os.environ.get("REPS", "20")))
    emit(t)
    print(f"timing {t['shape']}: {t['median_ms']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
