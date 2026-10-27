#!/usr/bin/env python3
"""attn_bf16 on one H100/H200 (sm_90a) or RTX PRO 6000 Blackwell (sm_120a): the prebuilt attn_bf16.<arch>.cubin through the
CUDA driver API (ctypes on libcuda.so.1; numpy for the buffers, torch only for the timing's baselines).

0. Identity, before anything runs: exactly one visible device, a card of CARDS (ATTN_EXPECT, a name prefix, narrows it),
   and its UUID equal to GPU_LEASE_UUID when gpu-lease set it.  A mismatch exits 3 before any launch.
1. The check, on every case in CHECK (fixture.py's): S and O are filled with 0xA5 and QK^T and PV run on the case's
   inputs; each buffer must equal the fixture's byte for byte (every word the Definition defines, and 0xA5A5 everywhere
   else).  The tilings are q8 (a CTA per 8 query rows, head and sequence) and ref (the reference kernels: a CTA per 8
   columns, KV head and sequence), each at 4, 1 and 8 warps a CTA, and each head tiling that serves the case
   (attn.serves: h8 and h16 where 8 or 16 divides the query heads a KV head serves, s16 too where no query sees more than
   32 keys and PV's K is at most 32; a CTA per query row and sequence and Y CTAs of head groups), each with one group
   a CTA of 1 warp, every 8th group a CTA of 8 warps, all of them in one CTA of 4 warps, and all of them in one
   warp: each gives the tiles to other warps, for the same bytes.  Each tiling again at 4 warps with S and O starting
   SHIFTS words into their buffers (1: no word pair 32-bit aligned, 2: no 8-word run 16-byte aligned), so the kernels take
   their narrower stores.  The SLACK poisoned words around S and O must stay poison in every arm.  A control arm fills the
   buffers and launches nothing, and must be rejected.  The inputs must come back unchanged.  A FULL case
   (<case>.full.npz) makes its inputs with gen.py, checks them against the fixture's digests (exit 5 if the generator
   disagrees), and compares each sequence's S and O digests: first over its first FULL_B sequences each (M = 4,096
   and 16,384, cases <case>_m<B>: gen.py's words depend only on the flat index, so they are the
   whole batch's first B), then over the whole batch.
2. The torch functions (attn.py's qk and pv, the skeleton's interface), through each path (fast, and @ref the reference
   kernels), on every decode case (one query a sequence) and FULL's batches: over the whole batch with its lens (S past a
   length must be the fill, -inf); again into 0xA5-poisoned out= buffers with fill 0x1234, other warps and
   check_lens=False; once per KV length T over the sequences of that length with k and v sliced to their first T keys, p
   a contiguous (B, H, T) (an odd T's rows are not 32-bit aligned) and no lens, as the skeleton calls them; and, but for
   FULL, with every 5th sequence given no keys (S all fill, O +0).  Every other word must equal the fixture's; for FULL,
   each sequence's digests.  Exit 4 if any check fails, before anything is timed.
3. Diagnostic timing (information only, not a panel row), at the H200 model's decode step for each batch of TIME_B
   sequences (TIME_H query heads of head dim 64 on TIME_HKV KV heads, every KV length TIME_L, PV's K = 32): QK^T and PV here
   at every launch shape of each head tiling that serves (Y and warps; q8 where none does), the reference kernels at 4
   warps (ref_qk, ref_pv, outside best_ms), and through attn.py (one call, one with lens, and the skeleton's TIME_L decode
   steps, a call of each op at each KV length 1..TIME_L), each the median of REPS after a warm-up, interleaved rep by rep
   with torch's SDPA (its default choice of backend, FlashAttention's alone, and its TIME_L steps; all include the softmax
   these two leave out) and with torch's two bf16 batched matmuls (cuBLAS).

One JSON object per line on stdout as each part finishes, a summary on stderr.

  python3 run.py CHECK [CASE ...]   (env: REPS=20, FULL_B=4096,16384, TIME_B=4096,16384,20480, TIME_H=256, TIME_HKV=1,
                                     TIME_L=32, ATTN_EXPECT, ATTN_LIBCUDA, ATTN_ANY_GPU=1 skips the identity check for a
                                     dry run on another card)
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
import attn  # noqa: E402
import gen  # noqa: E402
from attn import Args, grid  # noqa: E402

#: compute capability: (accepted name prefixes, the cubin's arch)
CARDS = {"9.0": (("NVIDIA H100", "NVIDIA H200"), "sm_90a"), "12.0": (("NVIDIA RTX PRO 6000 Blackwell",), "sm_120a")}
CC_MAJOR, CC_MINOR = 75, 76
POISON = 0xA5
WARPS = (4, 1, 8)
#: words of poison past each output buffer: room for the shifted arms, and where no kernel may write
SLACK = 8
#: the shifted arms' offsets of S and O into their buffers, in words (Attn.shift_outputs)
SHIFTS = (1, 2)

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
    """The cubin's kernels over one set of device buffers: Q, P [B, H, *, *], K, V [B, Hkv, *, *] bf16, lens and nq [B] int32,
    and PV's least K (z["kpv"])."""

    def __init__(self, cu: Cuda, funcs: dict, z: dict):
        self.cu, self.funcs = cu, funcs
        self.kpv = int(z["kpv"])
        self.D = z["Q"].shape[-1]
        self.B, self.H, self.Mq, self.Lcap = z["P"].shape
        self.Hkv = z["K"].shape[1]
        G = self.H // self.Hkv
        assert self.H % self.Hkv == 0 and G & (G - 1) == 0, (self.H, self.Hkv)
        self.gshift = G.bit_length() - 1
        self.tilings = ("q8", *(t for t in attn.GROUP if attn.serves(t, G, self.Lcap, self.kpv)), "ref")
        self.s_shape, self.o_shape = (self.B, self.H, self.Mq, self.Lcap), (self.B, self.H, self.Mq, self.D)
        self.dev = {k: cu.put(z[k]) for k in ("Q", "K", "V", "P", "lens", "nq")}
        self.S = cu.alloc((int(np.prod(self.s_shape)) + SLACK) * 2)
        self.O = cu.alloc((int(np.prod(self.o_shape)) + SLACK) * 2)
        self.shift, self.strays = 0, 0
        self.qk_args = self._args("Q", "K", self.S, z["Q"].shape, z["K"].shape, self.s_shape)
        self.pv_args = self._args("P", "V", self.O, z["P"].shape, z["V"].shape, self.o_shape)

    def shift_outputs(self, words: int) -> None:
        """S and O from ``words`` words into their buffers: at 1 no word pair is 32-bit aligned, at 2 no 8-word run is
        16-byte aligned, so the kernels take their narrower stores."""
        self.shift = words
        self.qk_args.out, self.pv_args.out = self.S + 2 * words, self.O + 2 * words

    def _args(self, x: str, kv: str, out: int, xs: tuple, kvs: tuple, os_: tuple) -> Args:
        return Args(self.dev[x], self.dev[kv], out, self.dev["lens"], self.dev["nq"], (C.c_int64 * 3)(*strides(xs)),
                    (C.c_int64 * 3)(*strides(kvs)), (C.c_int64 * 3)(*strides(os_)), self.B, self.H, self.Mq, self.Lcap,
                    self.gshift, self.kpv)

    def kernel(self, op: str, tiling: str):
        return self.funcs[attn.kernel(op, self.D, tiling)]

    def launch(self, op: str, tiling: str, warps: int, Y: int = 0) -> None:
        """attn.grid's CTAs for ``tiling`` (h8, h16, s16: Y CTAs of head groups), of ``warps`` warps."""
        args = self.qk_args if op == "qk" else self.pv_args
        params = (vp * 1)(C.cast(C.pointer(args), vp))
        ctas = grid(tiling, self.B, self.H, self.Mq, Y, self.Lcap if op == "qk" else self.D, self.Hkv)
        self.cu("cuLaunchKernel", self.kernel(op, tiling), *ctas, 32 * warps, 1, 1, 0, None, params, None)

    def shapes(self) -> list[tuple[str, int, int]]:
        """The check's launches, (tiling, warps, Y): q8 and ref at 4, 1 and 8 warps; each head-group tiling one group a
        CTA of 1 warp, every 8th group a CTA of 8 warps, every group in one CTA of 4 warps, and every group in one warp
        (s16's reuse of K and V across a warp's groups, and its reload where they change KV head)."""
        out = [("q8", w, 0) for w in WARPS]
        for t in self.tilings:
            if t in attn.GROUP:
                g = self.H // attn.GROUP[t]
                out += [(t, 1, g), (t, 8, max(1, g // 8)), (t, 4, 1), (t, 1, 1)]
        out += [("ref", w, 0) for w in WARPS]
        return list(dict.fromkeys(out))

    def poison(self) -> None:
        self.cu("cuMemsetD8_v2", self.S, POISON, (int(np.prod(self.s_shape)) + SLACK) * 2)
        self.cu("cuMemsetD8_v2", self.O, POISON, (int(np.prod(self.o_shape)) + SLACK) * 2)
        self.cu("cuCtxSynchronize")

    def run(self, tiling: str, warps: int, Y: int = 0) -> None:
        self.launch("qk", tiling, warps, Y)
        self.launch("pv", tiling, warps, Y)
        self.cu("cuCtxSynchronize")

    def outputs(self) -> tuple[np.ndarray, np.ndarray]:
        """S and O as the kernels left them; self.strays counts the words around them (the slack) that aren't poison."""
        got, self.strays = [], 0
        for p, shape in ((self.S, self.s_shape), (self.O, self.o_shape)):
            n = int(np.prod(shape))
            buf = self.cu.get(p, np.empty(n + SLACK, np.uint16))
            self.strays += int((buf[:self.shift] != 0xA5A5).sum() + (buf[self.shift + n:] != 0xA5A5).sum())
            got.append(buf[self.shift:self.shift + n].reshape(shape))
        return got[0], got[1]

    def free(self) -> None:
        for p in (*self.dev.values(), self.S, self.O):
            self.cu("cuMemFree_v2", p)


def arm_name(tiling: str, warps: int, Y: int) -> str:
    return f"{tiling}@{warps}w" + (f"/y{Y}" if Y else "")


def sha(a: np.ndarray) -> str:
    return hashlib.sha512(np.ascontiguousarray(a).tobytes()).hexdigest()[:32]


def _head(name: str, z: dict, a: Attn) -> dict:
    return {"part": "check", "case": name, "B": a.B, "H": a.H, "Hkv": a.Hkv, "D": a.D, "Mq": a.Mq, "Lcap": a.Lcap, "kpv": a.kpv,
            "lens": [int(z["lens"].min()), int(z["lens"].max())], "nq": [int(z["nq"].min()), int(z["nq"].max())],
            "tilings": list(a.tilings), "arms": {}}


def check_arm(a: Attn, z: dict, tiling: str, warps: int, Y: int) -> tuple[bool, dict]:
    """Both ops once under poison: whether every word of S and O equals the fixture's and none past them was written."""
    a.poison()
    a.run(tiling, warps, Y)
    S, O = a.outputs()
    arm = {"S_mismatches": int((S != z["S"]).sum()), "O_mismatches": int((O != z["O"]).sum()), "strays": a.strays,
           "S_sha512": sha(S), "O_sha512": sha(O)}
    if arm["S_mismatches"] or arm["O_mismatches"]:
        arm["examples"] = [{"buffer": k, "at": i, "got": int(got[tuple(i)]), "want": int(z[k][tuple(i)])}
                           for k, got in (("S", S), ("O", O)) for i in np.argwhere(got != z[k])[:4].tolist()]
    return not (arm["S_mismatches"] or arm["O_mismatches"] or arm["strays"]), arm


def check_case(cu: Cuda, funcs: dict, path: str) -> dict:
    name = os.path.basename(path)[: -len(".npz")]
    z = dict(np.load(path))
    a = Attn(cu, funcs, z)
    row = _head(name, z, a)
    row["words"] = {"S": int((z["S"] != 0xA5A5).sum()), "O": int((z["O"] != 0xA5A5).sum())}
    ok = True
    for tiling, warps, Y in a.shapes():
        good, row["arms"][arm_name(tiling, warps, Y)] = check_arm(a, z, tiling, warps, Y)
        ok &= good
    for shift in SHIFTS:
        a.shift_outputs(shift)
        for tiling in a.tilings:
            Y = 1 if tiling in attn.GROUP else 0
            good, row["arms"][arm_name(tiling, 4, Y) + f"+{shift}"] = check_arm(a, z, tiling, 4, Y)
            ok &= good
    a.shift_outputs(0)
    a.poison()
    S, O = a.outputs()
    ctl = {"S_mismatches": int((S != z["S"]).sum()), "O_mismatches": int((O != z["O"]).sum()), "strays": a.strays}
    ctl["rejected"] = bool(ctl["S_mismatches"] and ctl["O_mismatches"])
    ok &= ctl["rejected"] and not ctl["strays"]
    row["arms"]["control"] = ctl
    row["inputs_unchanged"] = all(np.array_equal(cu.get(a.dev[k], z[k]), z[k]) for k in ("Q", "K", "V", "P", "lens", "nq"))
    ok &= row["inputs_unchanged"]
    row["ok"] = ok
    a.free()
    return row


def full_inputs(f: dict, B: int = 0) -> dict:
    """A FULL case's inputs for its first B sequences (default all), made by gen.py from the seed its file names: gen.py's
    words are a function of the flat index, so they are the whole batch's first B."""
    Bf, H, Hkv, D, Lcap, seed, _, kpv = (int(v) for v in f["shape"])
    B = B or Bf
    z = {k: gen.array(seed, k, s) for k, s in gen.shapes(B, H, Hkv, D, Lcap).items()}
    z["lens"], z["nq"] = gen.lens_nq(B, Lcap)
    z["kpv"] = np.int32(kpv)
    return z


def full_name(path: str, B: int = 0) -> str:
    return os.path.basename(path)[: -len(".full.npz")] + (f"_m{B}" if B else "")


def full_case(cu: Cuda, funcs: dict, path: str, B: int = 0) -> dict:
    """A FULL case over its first B sequences (default all, a multiple of its input chunk): gen.py's inputs, held to the
    fixture's digests, and each sequence's S and O digests against the fixture's first B."""
    f = dict(np.load(path))
    Bf, H, Hkv, D, Lcap, seed, chunk, _ = (int(v) for v in f["shape"])
    B = B or Bf
    assert B <= Bf and B % chunk == 0, (B, Bf, chunk)
    f.update({k: f[k][:B] for k in ("S", "O")})
    z = full_inputs(f, B)
    bad_in = {k: int(sum(not np.array_equal(gen.digests(z[k][b0:b0 + chunk][None])[0], f[f"in_{k}"][i])
                         for i, b0 in enumerate(range(0, B, chunk)))) for k in ("Q", "K", "V", "P")}
    a = Attn(cu, funcs, z)
    row = _head(full_name(path, B if B < Bf else 0), z, a)
    row.update({"full": True, "seed": seed, "of_B": Bf, "input_chunks_differing": bad_in,
                "words": {"S": H * int(z["lens"].sum()), "O": B * H * D}})
    if any(bad_in.values()):
        row["ok"] = False
        row["generator_disagrees"] = True
        a.free()
        return row
    ok = True
    for tiling, warps, Y in a.shapes():
        a.poison()
        a.run(tiling, warps, Y)
        S, O = a.outputs()
        sd, od = gen.digests(S), gen.digests(O)
        arm = {"S_sequences_differing": int((sd != f["S"]).any(axis=1).sum()), "O_sequences_differing": int((od != f["O"]).any(axis=1).sum()),
               "strays": a.strays, "S_sha512": sha(S), "O_sha512": sha(O)}
        if arm["S_sequences_differing"] or arm["O_sequences_differing"]:
            arm["examples"] = {k: np.flatnonzero((d != f[k]).any(axis=1))[:8].tolist() for k, d in (("S", sd), ("O", od))}
        ok &= not (arm["S_sequences_differing"] or arm["O_sequences_differing"] or arm["strays"])
        row["arms"][arm_name(tiling, warps, Y)] = arm
    a.poison()
    S, O = a.outputs()
    ctl = {"S_sequences_differing": int((gen.digests(S) != f["S"]).any(axis=1).sum()),
           "O_sequences_differing": int((gen.digests(O) != f["O"]).any(axis=1).sum()), "strays": a.strays}
    ctl["rejected"] = bool(ctl["S_sequences_differing"] == B and ctl["O_sequences_differing"] == B)
    ok &= ctl["rejected"] and not ctl["strays"]
    row["arms"]["control"] = ctl
    row["inputs_unchanged"] = all(np.array_equal(cu.get(a.dev[k], z[k]), z[k]) for k in ("Q", "K", "V", "P", "lens", "nq"))
    ok &= row["inputs_unchanged"]
    row["ok"] = ok
    a.free()
    return row


def api_case(path: str, Bp: int = 0) -> dict | None:
    """attn.qk and attn.pv through torch on a decode case (a FULL one's first Bp sequences): the whole batch with its lens
    (S past them must be the fill); again into poisoned out= buffers with another fill, other warps and check_lens=False;
    each KV length T's sequences with k and v sliced to their first T keys, p a contiguous (B, H, T) and no lens (the
    skeleton's call); and, but for FULL, every 5th sequence given no keys (S all fill, O +0).  Each through both paths
    (attn.PATHS).  None for a case with prefill rows."""
    import torch

    full = path.endswith(".full.npz")
    f = dict(np.load(path))
    if full:
        Bf = int(f["shape"][0])
        name = full_name(path, Bp if Bp and Bp < Bf else 0)
        f.update({k: f[k][:Bp or Bf] for k in ("S", "O")})
        z = full_inputs(f, Bp)
    else:
        name, z = os.path.basename(path)[: -len(".npz")], f
    if z["Q"].shape[2] != 1:
        return None
    B, H, _, D = z["Q"].shape
    Hkv, Lcap = z["K"].shape[1], z["K"].shape[2]
    kpv, lens = int(z["kpv"]), z["lens"]

    def dev(a: np.ndarray):
        return torch.from_numpy(np.ascontiguousarray(a).view(np.int16)).cuda().view(torch.bfloat16)

    def host(x) -> np.ndarray:
        torch.cuda.synchronize()
        return x.view(torch.int16).cpu().numpy().view(np.uint16)

    q, p, k, v = dev(z["Q"][:, :, 0]), dev(z["P"][:, :, 0]), dev(z["K"]), dev(z["V"])
    if Hkv == 1:
        k, v = k[:, 0], v[:, 0]
    wantS, wantO = (None, None) if full else (f["S"][:, :, 0], f["O"][:, :, 0])
    fills = {}

    def past(s: np.ndarray, n: np.ndarray, arm: str, fill: int) -> np.ndarray:
        """s with the fill past each length n checked (into fills[arm]) and replaced by the fixture's poison."""
        live = np.broadcast_to(np.arange(Lcap)[None, None, :] < n[:, None, None], (B, H, Lcap))
        fills[arm] = bool((s[~live] == fill).all())
        s[~live] = 0xA5A5
        return s

    tl = torch.from_numpy(lens.astype(np.int32)).cuda()
    if not full:  # every 5th sequence with no keys: S all fill, O +0, the rest unchanged
        l0 = lens.copy()
        l0[::5] = 0
        t0 = torch.from_numpy(l0.astype(np.int32)).cuda()
        wS0, wO0 = wantS.copy(), wantO.copy()
        wS0[::5], wO0[::5] = 0xA5A5, 0
    arms = {}
    for via in attn.PATHS:  # each arm through the fast path, and again (@ref) through the reference kernels
        x = "" if via == "fast" else f"@{via}"
        arms["lens" + x] = (past(host(attn.qk(q, k, tl, path=via)), lens, "lens" + x, attn.NEG_INF),
                            host(attn.pv(p, v, tl, kpv=kpv, path=via)))
        S0 = torch.full((B, H, Lcap), 0xA5A5 - 0x10000, dtype=torch.int16, device="cuda")
        O0 = torch.full((B, H, D), 0xA5A5 - 0x10000, dtype=torch.int16, device="cuda")
        attn.qk(q, k, tl, out=S0, fill=0x1234, warps=1, check_lens=False, path=via)
        attn.pv(p, v, tl, out=O0, kpv=kpv, warps=8, check_lens=False, path=via)
        arms["out_unchecked" + x] = (past(host(S0), lens, "out_unchecked" + x, 0x1234), host(O0))
        S2, O2 = np.full((B, H, Lcap), 0xA5A5, np.uint16), np.empty((B, H, D), np.uint16)
        for T in np.unique(lens).tolist():
            sel = np.flatnonzero(lens == T)
            st = torch.from_numpy(sel).cuda()
            S2[sel, :, :T] = host(attn.qk(q[st], k[st][..., :T, :], path=via))
            O2[sel] = host(attn.pv(p[st][..., :T].contiguous(), v[st][..., :T, :], kpv=kpv, path=via))
        arms["per_length" + x] = (S2, O2)
        if not full:
            arms["lens0" + x] = (past(host(attn.qk(q, k, t0, path=via)), l0, "lens0" + x, attn.NEG_INF),
                                 host(attn.pv(p, v, t0, kpv=kpv, path=via)))
    row = {"part": "api", "case": name, "B": B, "H": H, "Hkv": Hkv, "D": D, "Lcap": Lcap, "kpv": kpv,
           "k_form": "(B, T, D)" if Hkv == 1 else "(B, Hkv, T, D)", "arms": {}}
    row["fill_past_lens"] = fills
    for arm, (s, o) in arms.items():
        if full:
            row["arms"][arm] = {"S_sequences_differing": int((gen.digests(s[:, :, None]) != f["S"]).any(axis=1).sum()),
                                "O_sequences_differing": int((gen.digests(o[:, :, None]) != f["O"]).any(axis=1).sum())}
        else:
            ws, wo = (wS0, wO0) if arm.startswith("lens0") else (wantS, wantO)
            row["arms"][arm] = {"S_mismatches": int((s != ws).sum()), "O_mismatches": int((o != wo).sum())}
    row["ok"] = all(fills.values()) and not any(n for a in row["arms"].values() for n in a.values())
    return row


def timing(cu: Cuda, funcs: dict, reps: int) -> list[dict]:
    """The H200 model's decode step for each batch of TIME_B: inputs drawn once at the largest and sliced to each."""
    Bs = [int(b) for b in os.environ.get("TIME_B", "4096,16384,20480").split(",")]
    H, Hkv, L = (int(os.environ.get(k, d)) for k, d in (("TIME_H", 256), ("TIME_HKV", 1), ("TIME_L", 32)))
    D, Bm = 64, max(Bs)
    rng = np.random.default_rng(0)

    def words(shape: tuple) -> np.ndarray:
        return (rng.standard_normal(shape, np.float32).view(np.uint32) >> 16).astype(np.uint16)

    zm = {"Q": words((Bm, H, 1, D)), "K": words((Bm, Hkv, L, D)), "V": words((Bm, Hkv, L, D)), "P": words((Bm, H, 1, L)),
          "lens": np.full(Bm, L, np.int32), "nq": np.ones(Bm, np.int32)}
    return [timing_at(cu, funcs, reps, {**{k: x[:B] for k, x in zm.items()}, "kpv": attn.PAD_K}) for B in Bs]


def timing_at(cu: Cuda, funcs: dict, reps: int, z: dict) -> dict:
    """One batch at every KV length L: these kernels at each launch shape, attn.py's functions (as one call, with lens,
    and as the skeleton's L decode steps, one call a KV length 1..L), SDPA and torch's matmuls, interleaved rep by rep."""
    B, H, _, D = z["Q"].shape
    Hkv, L = z["K"].shape[1], z["K"].shape[2]
    G = H // Hkv
    a = Attn(cu, funcs, z)
    e0, e1 = cu.event(), cu.event()
    ms = C.c_float()

    def ours(op, tiling, warps, Y=0):
        cu("cuEventRecord", e0, None)
        a.launch(op, tiling, warps, Y)
        cu("cuEventRecord", e1, None)
        cu("cuEventSynchronize", e1)
        cu("cuEventElapsedTime", C.byref(ms), e0, e1)
        return ms.value

    arms = {}
    heads = [t for t in a.tilings if t in attn.GROUP]
    for t in heads:  # every launch shape of the head-group kernels: Y CTAs of head groups, of 1 to 8 warps
        g = H // attn.GROUP[t]
        for Y in sorted({max(1, g >> s) for s in range(0, 8)}, reverse=True):
            for w in (1, 2, 4, 8):
                for op in ("qk", "pv"):
                    arms[f"{op}_{t}@{w}w/y{Y}"] = lambda op=op, t=t, w=w, Y=Y: ours(op, t, w, Y)
    if not heads:  # a warp per key tile of QK^T and per d tile of PV
        arms["qk_q8"] = lambda: ours("qk", "q8", min(8, -(-L // 16)))
        arms["pv_q8"] = lambda: ours("pv", "q8", D // 16)
    for op in ("qk", "pv"):  # the reference path, named outside best_ms's "qk_" and "pv_"
        arms[f"ref_{op}"] = lambda op=op: ours(op, "ref", 4)
    out = {"part": "timing", "shape": {"B": B, "H": H, "Hkv": Hkv, "D": D, "L": L, "queries_per_sequence": 1, "kpv": attn.PAD_K},
           "reps": reps, "attn_py_launch": attn.LAUNCH, "note": "diagnostic only; SDPA includes the softmax qk and pv leave out"}
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
        # attn.py's functions as the skeleton calls them: each launch with its output's allocation
        q3, p3, k3, v3 = q[:, :, 0], p[:, :, 0], k[:, 0] if Hkv == 1 else k, v[:, 0] if Hkv == 1 else v
        tl = torch.full((B,), L, dtype=torch.int32, device=dev)
        arms["api_qk"] = lambda: tt(lambda: attn.qk(q3, k3))
        arms["api_pv"] = lambda: tt(lambda: attn.pv(p3, v3))
        arms["api_qk_lens"] = lambda: tt(lambda: attn.qk(q3, k3, tl, check_lens=False))
        arms["api_pv_lens"] = lambda: tt(lambda: attn.pv(p3, v3, tl, check_lens=False))

        def steps():
            for T in range(1, L + 1):
                attn.qk(q3, k3[..., :T, :])
                attn.pv(p3[..., :T], v3[..., :T, :])

        def sdpa_steps():
            for T in range(1, L + 1):
                F.scaled_dot_product_attention(q, k[:, :, :T], v[:, :, :T], **gqa)

        arms["api_steps"] = lambda: tt(steps)
        if "sdpa" in arms:
            arms["sdpa_steps"] = lambda: tt(sdpa_steps)
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
    med = out["median_ms"] = {arm: round(statistics.median(ts), 4) for arm, ts in got.items()}
    out["best_ms"] = {op: min(((t, arm) for arm, t in med.items() if arm.startswith(f"{op}_")), default=None)
                      for op in ("qk", "pv")}
    if all(out["best_ms"].values()):
        out["best_qk_plus_pv_ms"] = round(out["best_ms"]["qk"][0] + out["best_ms"]["pv"][0], 4)
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
    for name in attn.KERNELS:
        f = vp()
        cu("cuModuleGetFunction", C.byref(f), mod, name.encode())
        funcs[name] = f
    paths = [os.path.join(check, f"{n}.npz") for n in names] or sorted(glob.glob(os.path.join(check, "*.npz")))
    prefixes = [int(b) for b in os.environ.get("FULL_B", "4096,16384").split(",") if b]
    runs = [(p, B) for p in paths for B in ([*prefixes, 0] if p.endswith(".full.npz") else [0])]
    rows = []
    for p, B in runs:
        r = full_case(cu, funcs, p, B) if p.endswith(".full.npz") else check_case(cu, funcs, p)
        rows.append(r)
        emit(r)
        arms = r["arms"]
        mism = (lambda v: f"{v['S_mismatches']}+{v['O_mismatches']}") if not r.get("full") else \
            (lambda v: f"{v['S_sequences_differing']}+{v['O_sequences_differing']} sequences")
        print(f"{r['case']}: {'ok' if r['ok'] else 'FAILED'}  S {r['words']['S']} O {r['words']['O']} words, mismatches "
              + ", ".join(f"{k} {mism(v)}" for k, v in arms.items() if k != "control")
              + f", strays {sum(v.get('strays', 0) for v in arms.values())}"
              + f", control rejected {arms.get('control', {}).get('rejected')}", file=sys.stderr)
        if r.get("generator_disagrees"):
            print(f"REFUSED: {r['case']}: gen.py's inputs differ from the fixture's ({r['input_chunks_differing']})", file=sys.stderr)
            return 5
    try:
        import torch

        if not torch.cuda.is_available():
            raise ImportError("torch without CUDA")
        api = [r for r in (api_case(p, B) for p, B in runs) if r is not None]
    except ImportError as e:
        api = [{"part": "api", "ok": False, "unavailable": str(e)}]
    for r in api:
        rows.append(r)
        emit(r)
        print(f"api {r.get('case')}: {'ok' if r['ok'] else 'FAILED'} {json.dumps(r.get('arms', r.get('unavailable')))}", file=sys.stderr)
    if not all(r["ok"] for r in rows):
        print("REFUSED: a case failed; nothing timed", file=sys.stderr)
        return 4
    for t in timing(cu, funcs, int(os.environ.get("REPS", "20"))):
        emit(t)
        print(f"timing {t['shape']}: best {t['best_ms']}; {t['median_ms']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
