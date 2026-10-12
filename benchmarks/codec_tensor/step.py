"""One decode step at HBM exhaustion on one GPU, from the existing kernels: a Llama-3.1 model's FP8 linears (`--model`)
held as p512 words (`wt.cu`: the fused decode, ro-sketch-blake3/v0 at k_s = 16 and the FP8 MMA, then the step's leaves
and tree), and a bf16 KV cache that fills the rest of the GPU's memory at `--batch` sequences, held as p512 words and
decoded inside attention (`benchmarks/word_codec/kv.cu`'s `kv_attn`); the step's new tokens' KV is encoded (`step.cu`'s
`kv_append`, all layers in one launch on a second stream, beside the step). The same step on plain data (each linear's
fastest native FP8 GEMM, the faster plain attention form, a plain append) is its denominator, and a native pass (a read of every byte
the step reads) its unit. Writes `step.json` and `step.log` under `--out`, and in a `research run` its `result.json`.

Nothing is timed before its gate passed:
- wt.cu's decode primitives at inputs that take their rare branches (`gate_prims`), which no gate's random words reach;
- the codecs: wt.cu's encode and decode against `wide.py` at each k (`bench.gate_codec`) and its plain fragment layout
  (`bench.gate_layout`); kv.cu's against `kv.py` (`kv_bench.gate`);
- each linear of the model at M = batch (`bench.gemm_case`): the fused kernel's partial sums equal the plain kernel's
  bit for bit at every k and config, the plain kernel's within FP32 rounding of torch's, and every sketch vector, tag,
  leaf and root; its per-linear timings pick the configs the step runs;
- attention at the step's batch × context (two layers): `kv_attn` over the encoded cache, in each encoded form (the
  CUDA-core forms 1 and 2, the tensor-core forms 4 and 5) and at every split count tried, equals its plain form (0 or
  3) bit for bit at each k, and each plain form is within RTOL of float64 attention (on a few sequences);
- the append: encoded, the cache's bytes equal `kv_encode` of the same words at each slot's index; plain, the words;
- the per-pass key (red team F29, `bench.gate_pass`): each sketched step starts with `wt_pass`, which derives pass j's
  seed pass_seed(σ, j, root_(j−1)), signs, key and leaves' prefix on the device from the previous step's root (the
  first from a registration root); `gate_chain` checks two consecutive steps' link.

The step itself runs on random stored words (the decode, the sketch and the MMAs take the same time on any words), so
the weights and the cache fill the memory once. Arms, interleaved over `--reps`, each queued behind a spin:
- `pass`: a read of the weights and the cache (the native pass);
- `plain` and per k `enc{k}`: the whole step (and `_layerwise`: each layer's append on the second stream after its qkv;
  `_noappend`: no append);
- the parts alone: `_W` the linears (encoded: sketched, with the step's commit; `_Wu` untagged), `_A` attention, `_P`
  the append on the main stream.
`model` sets the measured terms (in native passes) beside the mainline model's,
step = 1 + f_KV max(0, c k - h) + f_W (small) + f_KV k x_1 / ctx + tau.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bench as BW  # noqa: E402  (before kv_bench, which puts word_codec, whose bench.py is another, first on the path)
import sketching as SKT  # noqa: E402
import wide as W  # noqa: E402

sys.path.insert(0, str(HERE.parent / "word_codec"))
import kv as K  # noqa: E402
import kv_bench as KB  # noqa: E402
import kv_twin as KT  # noqa: E402

LM_HEAD = {"8b": (128256, 4096), "70b": (128256, 8192), "tiny": (2048, 512)}
TINY = (2, {"qkv": (384, 512), "o": (512, 512), "gate_up": (1024, 512), "down": (512, 1024)})
ORDER = ("qkv", "o", "gate_up", "down")          # a layer's linears; attention runs after qkv
PLAIN_CFGS = "4x4,8x8"
MODEL_TERMS = {"c": (0.54, 0.59), "h": 0.30, "x1": 230.0, "x1_pooled": 175.0, "tau": 0.04}
HEADS = {"8b": 32, "70b": 64}    # query heads; kv.cu's build (KV_QHEADS) must match
FORMS = {0: "fma-plain", 1: "fma-phased", 2: "fma-ws", 3: "mma-plain", 4: "mma-phased", 5: "mma-ws"}
PLAIN_OF = {1: 0, 2: 0, 4: 3, 5: 3}      # each encoded form's output equals its plain form's bit for bit
RTOL = {0: KB.ATTN_RTOL, 3: 8e-3}        # the tensor-core form rounds P to bf16


def model_shapes(name: str):
    layers, shapes = TINY if name == "tiny" else BW.MODELS[name]
    return layers, dict(shapes), LM_HEAD[name]


def splits_for(b: int, ctx: int, tt: int, sms: int) -> list[int]:
    """kv_bench's split counts for a cell: batch × splits about 1, 2, 4 or 8 CTAs an SM, a split at least 1 or 8 tiles."""
    return sorted({max(1, min(-(-ctx // (t * tt)), -(-c * sms // b)))
                   for c in KB.ATTN_CTAS_PER_SM for t in KB.ATTN_SPLIT_TILES})


def forms_of(sh: dict) -> tuple[list[int], list[int]]:
    """kv_attn's plain and encoded forms in this build (the tensor-core ones where the field's words tile a head)."""
    return ([0, 3], [1, 2, 4, 5]) if sh["mma"] else ([0], [1, 2])


def split_cands(sh: dict, b: int, ctx: int, sms: int) -> dict[int, list[int]]:
    """Per form the split counts tried: a CUDA-core CTA is a split of a sequence (all KV heads, tiles of TT tokens), a
    tensor-core CTA a split of one KV head (tiles of MT tokens)."""
    fma = splits_for(b, ctx, sh["tt"], sms)
    mma = splits_for(b * sh["kv_heads"], ctx, sh["mt"], sms) if sh["mma"] else []
    return {fm: (mma if fm >= 3 else fma) for fm in sum(forms_of(sh), [])}


def part_floats(sh: dict, b: int, cands: dict[int, list[int]]) -> int:
    return b * max(sp * (sh["mw"] if fm >= 3 else 1) for fm, c in cands.items() for sp in c) * sh["part"]


# --- gates -------------------------------------------------------------------------------------------------------------

def gate_prims(wt, torch, log) -> dict:
    """wt.cu's decode primitives (`wt_prim`) at inputs that take their rare branches, which random words reach about
    once in 2^23 (`kv_twin.rare_cases` for p512: a carry or borrow past limb 0, canon's subtraction of p, the square's
    rare fold), against `kv_twin.prim_want`."""
    f = K.field("p512")
    cases = KT.rare_cases(f, random.Random(f.n + 7))
    nb, st, out = 4 * wt.nl, BW.stream(torch), {}
    for op, rows in cases.items():
        a = BW.to_dev(torch, b"".join(BW.limbs_bytes(x, wt.nl) for x, _ in rows))
        b = BW.to_dev(torch, b"".join(BW.limbs_bytes(y, wt.nl) for _, y in rows))
        o = torch.zeros(len(rows) * nb, dtype=torch.uint8, device="cuda")
        wt.call("wt_prim", op, BW.ptr(a), BW.ptr(b), BW.ptr(o), len(rows), st)
        torch.cuda.synchronize()
        got = bytes(o.cpu().tolist())
        bad = [i for i, (x, y) in enumerate(rows)
               if int.from_bytes(got[i * nb:(i + 1) * nb], "little") != KT.prim_want(f, op, x, y)]
        out[KT.PRIMS[op]] = {"cases": len(rows), "bad": bad}
    ok = all(not v["bad"] for v in out.values())
    counts = ", ".join(f"{k} {v['cases']}" for k, v in out.items())
    log(f"gate wt primitives at their rare branches: {'ok' if ok else 'FAIL'} ({counts}) {out if not ok else ''}")
    return {"ok": ok, "ops": out}


def gate_append(kv, torch, ks, log) -> dict:
    f = kv.field
    tlw, wb = K.token_layer_words(f), f.word_bytes
    b, ctx, layers, t = 3, 5, 3, 4
    st = KB.stream(torch)
    out = {}
    for k in [0] + list(ks):
        if k:
            KB.set_key(kv, K.key(f.name, KB.SEED, k))
        bad = []
        for l0, nl in ((0, layers), (1, 1), (layers - 1, 1)):
            new = KB.random_words(torch, b * nl * tlw, f, 11 + k + l0)
            cache = torch.zeros(b * ctx * layers * tlw * wb, dtype=torch.uint8, device="cuda")
            want = torch.zeros_like(cache)
            kv.call("kv_append", KB.ptr(new), KB.ptr(cache), b, ctx, layers, l0, nl, t, k, st)
            for s in range(b):
                for j in range(nl):
                    i = ((s * ctx + t) * layers + l0 + j) * tlw
                    src = new[(s * nl + j) * tlw * wb:(s * nl + j + 1) * tlw * wb]
                    dst = want[i * wb:(i + tlw) * wb]
                    if k:
                        kv.call("kv_encode", KB.ptr(src), KB.ptr(dst), tlw, i, k, 0, st)
                    else:
                        dst.copy_(src)
            torch.cuda.synchronize()
            if not torch.equal(cache, want):
                bad.append({"l0": l0, "nl": nl, "bytes_unequal": int((cache != want).sum())})
        out[k] = {"ok": not bad, "bad": bad}
    ok = all(v["ok"] for v in out.values())
    log(f"gate append: {'ok' if ok else 'FAIL'} (k {[0] + list(ks)}; all layers, the first, the last) {out if not ok else ''}")
    return {"ok": ok, "k": out}


def gate_attention(kv, torch, b: int, ctx: int, ks, sms: int, log) -> dict:
    """kv_attn at the step's batch × context over two layers (the step's index rule, layer 1): each encoded form equals
    its plain form (PLAIN_OF) bit for bit at each k and split count, and each plain form is within RTOL of float64
    attention on the first few sequences."""
    sh = KB.attn_shape(kv)
    tl, tle = K.TOKEN_LAYER_BYTES, sh["tle"]
    if tle != tl:
        raise RuntimeError(f"{kv.name}: a token-layer's words take {tle} bytes, not {tl}: the step wants p512")
    plains, encs = forms_of(sh)
    cands = split_cands(sh, b, ctx, sms)
    layers, layer, nref = 2, 1, min(b, 4)
    st = KB.stream(torch)
    g = torch.Generator(device="cuda").manual_seed(b * 7919 + ctx)
    plain = torch.randn(b * ctx * layers * tl // 2, dtype=torch.bfloat16, device="cuda", generator=g)
    scale = math.log2(math.e) / math.sqrt(K.HEAD_DIM)
    q = (torch.randn(b, sh["heads"], K.HEAD_DIM, device="cuda", generator=g) * scale).to(torch.bfloat16).contiguous()
    part = torch.empty(part_floats(sh, b, cands), dtype=torch.float32, device="cuda")
    o_p = {(fm, sp): torch.empty(b, sh["heads"], K.HEAD_DIM, dtype=torch.float32, device="cuda")
           for fm in plains for sp in cands[fm]}
    o_e = torch.empty(b, sh["heads"], K.HEAD_DIM, dtype=torch.float32, device="cuda")

    def run(store, form, o, sp, k):
        kv.call("kv_attn", KB.ptr(store), form, KB.ptr(q), KB.ptr(o), KB.ptr(part), b, ctx, layers, layer, sp, k, st)

    kvl = plain.view(b, ctx, layers, 2, K.KV_HEADS, K.HEAD_DIM)
    kk, vv = kvl[:nref, :, layer, 0].double(), kvl[:nref, :, layer, 1].double()
    qq = q[:nref].double().view(nref, K.KV_HEADS, sh["heads"] // K.KV_HEADS, K.HEAD_DIM)
    s = torch.einsum("bgqd,btgd->bgqt", qq, kk)
    p = torch.exp2(s - s.amax(-1, keepdim=True))
    ref = torch.einsum("bgqt,btgd->bgqd", p / p.sum(-1, keepdim=True), vv).reshape(nref, sh["heads"], K.HEAD_DIM)
    del kk, vv, s, p
    rel = {FORMS[fm]: 0.0 for fm in plains}
    for (fm, sp), o in o_p.items():
        o.fill_(float("nan"))
        run(plain, fm, o, sp, 1)
        torch.cuda.synchronize()
        rel[FORMS[fm]] = max(rel[FORMS[fm]], float((o[:nref].double() - ref).abs().max() / ref.abs().max()))
    del ref
    enc = torch.empty_like(plain.view(torch.uint8))
    n_words = enc.numel() // kv.field.word_bytes
    equal = {}
    for k in ks:
        KB.set_key(kv, K.key(kv.field.name, KB.SEED, k))
        kv.call("kv_encode", KB.ptr(plain), KB.ptr(enc), n_words, 0, k, sms * 8, st)
        for form in encs:
            for sp in cands[form]:
                o_e.fill_(float("nan"))
                run(enc, form, o_e, sp, k)
                torch.cuda.synchronize()
                equal[f"k{k}_form{form}_sp{sp}"] = bool(torch.equal(o_p[(PLAIN_OF[form], sp)], o_e))
    ok = all(rel[FORMS[fm]] <= RTOL[fm] for fm in plains) and all(equal.values())
    log(f"gate attention B={b} ctx={ctx}: {'ok' if ok else 'FAIL'} (plain rel err "
        + ", ".join(f"{n} {v:.1e}" for n, v in rel.items()) + f" on {nref} sequences; {sum(equal.values())}/"
        f"{len(equal)} encoded = their plain form, k {list(ks)}, forms {encs}, splits "
        + ", ".join(f"{fm}: {c}" for fm, c in cands.items()) + ")")
    del plain, enc, q, part, o_p, o_e, kvl
    torch.cuda.empty_cache()
    return {"ok": ok, "rel_err": rel, "equal": equal, "splits": cands, "layers": layers}


def gate_gemm(wt, torch, shapes: dict, b: int, ks, reps: int, sketch_t: int, log) -> tuple[dict, list, list]:
    fused = [c for c in BW.fused_cfgs(wt, "auto", "16", sketch_t) if c[2] in (-1, -2)]
    plain = BW.parse(PLAIN_CFGS)
    out = {}
    for name, (n, kd) in shapes.items():
        out[name] = BW.gemm_case(wt, torch, name, n, kd, [b], ks, reps, log, plain, fused, [], [], sketch_t=sketch_t)
        if not out[name].get("ok"):
            break
    return out, fused, plain


def pick(g: dict, b: int, ks, fused, plain) -> dict:
    """A linear's arms for the step from its gemm_case timings at M = b: native (cuBLASLt or the plain kernel, the
    faster), and per k the fastest sketched (by the decoders or by two sketch warps) and untagged fused configs."""
    med = g["times"][b]["median_s"]
    nat = min(["cublas"] + [f"plain_c{c}s{s}" for c, s in plain], key=lambda a: med[a])
    out = {"native": nat, "native_s": med[nat], "k": {}}
    for k in ks:
        ran = [c for c in fused if BW.fused_name(k, c) in med]
        sk = min((c for c in ran if c[3] == 16), key=lambda c: med[BW.fused_name(k, c)])
        un = min((c for c in ran if c[3] == 0), key=lambda c: med[BW.fused_name(k, c)])
        out["k"][k] = {"sketch": list(sk), "sketch_s": med[BW.fused_name(k, sk)], "untagged": list(un),
                       "untagged_s": med[BW.fused_name(k, un)]}
        dec = [c for c in ran if c[3] == 16 and not BW.sketch_warps(c)]
        if dec:
            best = min(dec, key=lambda c: med[BW.fused_name(k, c)])
            out["k"][k]["sketch_decoders"], out["k"][k]["sketch_decoders_s"] = list(best), med[BW.fused_name(k, best)]
    return out


# --- the step ----------------------------------------------------------------------------------------------------------

class Step:
    """The step's buffers and its launches, each enqueued on the current stream (and the append's stream)."""

    def __init__(self, wt, kv, torch, model: str, b: int, m: int, ctx: int, ks, picks: dict, sketch_t: int,
                 log) -> None:
        self.wt, self.kv, self.torch, self.b, self.ctx, self.ks, self.T = wt, kv, torch, b, ctx, list(ks), sketch_t
        self.m = m
        self.layers, shapes, lm = model_shapes(model)
        self.shapes = {**shapes, "lm_head": lm}
        self.picks = picks
        f = kv.field
        self.tlw, self.wb = K.token_layer_words(f), f.word_bytes
        self.sh = KB.attn_shape(kv)
        self.lins = [(l, name) for l in range(self.layers) for name in ORDER] + [(self.layers, "lm_head")]
        self.off, o = {}, 0
        for l, name in self.lins:
            n, kd = self.shapes[name]
            self.off[(l, name)] = o
            o += n * kd
        self.w_bytes = o
        self.kv_bytes = b * ctx * self.layers * K.TOKEN_LAYER_BYTES
        log(f"step: {model}, B={b} (linears at M={m}), ctx={ctx}: weights {self.w_bytes / 1e9:.2f} GB, KV {self.kv_bytes / 1e9:.2f} GB")
        self.W = torch.empty(self.w_bytes, dtype=torch.uint8, device="cuda")
        self.W.random_(0, 256)
        self.KV = torch.empty(self.kv_bytes, dtype=torch.uint8, device="cuda")
        for c in torch.split(self.KV, 1 << 33):
            c.random_(0, 256)
        g = torch.Generator(device="cuda").manual_seed(17)
        self.x = {kd: torch.randn(m, kd, device="cuda", generator=g).to(torch.float8_e4m3fn)
                  for kd in {kd for _, kd in self.shapes.values()}}
        self.mpad = 16 * -(-m // 16)
        self.xc = {}
        for kd, x in self.x.items():
            xc = torch.zeros(self.mpad, kd, device="cuda").to(torch.float8_e4m3fn)
            xc[:m] = x
            self.xc[kd] = xc
        self.y = {n: torch.zeros(m, n, dtype=torch.float32, device="cuda") for n, _ in self.shapes.values()}
        self.one = torch.ones((), device="cuda")
        self.wv = {}
        for (l, name), o in self.off.items():
            n, kd = self.shapes[name]
            self.wv[(l, name)] = self.W[o:o + n * kd].view(torch.float8_e4m3fn).view(n, kd).t()
        scale = math.log2(math.e) / math.sqrt(K.HEAD_DIM)
        self.q = (torch.randn(b, self.sh["heads"], K.HEAD_DIM, device="cuda", generator=g) * scale).to(
            torch.bfloat16).contiguous()
        self.o_attn = torch.empty(b, self.sh["heads"], K.HEAD_DIM, dtype=torch.float32, device="cuda")
        self.split_cands = split_cands(self.sh, b, ctx, torch.cuda.get_device_properties(0).multi_processor_count)
        self.part = torch.empty(part_floats(self.sh, b, self.split_cands), dtype=torch.float32, device="cuda")
        self.new_all = KB.random_words(torch, b * self.layers * self.tlw, f, 21)
        self.new_one = KB.random_words(torch, b * self.tlw, f, 22)
        self.xr = torch.zeros(1, dtype=torch.int64, device="cuda")
        self.s2 = torch.cuda.Stream()
        self.ev = [torch.cuda.Event() for _ in range(self.layers + 2)]
        self.splits = {}          # (form, k) -> split count; k = 0 the plain forms
        self.form = {}            # k -> the attention form the step runs (k = 0: the faster plain form)
        self._launch_params()

    # launch parameters --------------------------------------------------------------------------------------------
    def _launch_params(self) -> None:
        torch, wt = self.torch, self.wt
        st = BW.stream(torch)
        bits = SKT.sign_bits(BW.TAG_SEED, 16, SKT.vector_bytes(wt.sb, self.T))
        self.signs = torch.from_numpy(SKT.sign_fragments(bits, SKT.steps(wt.sb), self.T).view(np.int32)).cuda()
        self.frags = BW.sign_frags(wt, torch, self.signs, self.T, st)   # the sketch warps' B fragments of the signs
        BW.set_pass(wt, self.T)
        self.pass_j = 0
        self.root0 = BW.to_dev(torch, BW.root_words(hashlib.sha512(b"verity/codec-tensor/step/registration").digest()))
        self.root_ptr = self.root0.data_ptr()       # root_(j−1): the registration's, then each commit's
        self.keys = {k: W.key(wt.field.name, BW.SEED, k) for k in self.ks}
        self.prm = {}
        self.tag_off = {}
        self.nvec = {}
        for k in self.ks:
            tot = 0
            for name, (n, kd) in self.shapes.items():
                for var in ("sketch", "untagged"):
                    cfg = tuple(self.picks[name]["k"][k][var])
                    launched = ctypes.c_int(0)
                    wt.call("wt_fused_sketch", k, None, None, None, self.m, n, kd, 0, 0, cfg[0], cfg[1], 0, 0, None,
                            None, 0, cfg[3], None, None, self.T, BW.sketch_warps(cfg), ctypes.byref(launched), st)
                    gr = launched.value // (kd // 512)
                    steps = BW.tsteps(wt, torch, self.keys[k], cfg[0], gr, kd // 512)
                    nvec = wt.so.wt_sketch_vectors(n, kd, gr, self.T) if cfg[3] else 0
                    self.prm[(name, k, var)] = (cfg, gr, steps, nvec)
            for l, name in self.lins:
                self.tag_off[(k, l, name)] = tot
                tot += self.prm[(name, k, "sketch")][3]
            self.nvec[k] = tot
        top = max(self.nvec.values())
        self.tags = torch.empty(top * 32, dtype=torch.uint8, device="cuda")
        nleaves = -(-top // BW.TG.PER_LEAF)
        self.nodes = torch.zeros((2 * nleaves + 64) * 64, dtype=torch.uint8, device="cuda")
        self.plain_prm = {}
        for name, (n, kd) in self.shapes.items():
            nat = self.picks[name]["native"]
            if nat != "cublas":
                c, s = (int(v) for v in nat[len("plain_c"):].split("s"))
                launched = ctypes.c_int(0)
                wt.call("wt_gemm", 0, None, None, None, self.m, n, kd, 0, c, s, 0, 0, None, ctypes.byref(launched), st)
                self.plain_prm[name] = (c, s, launched.value // (kd // 512))

    # launches -----------------------------------------------------------------------------------------------------
    def set_keys(self, k: int) -> None:
        BW.set_key(self.wt, self.torch, self.keys[k])
        KB.set_key(self.kv, K.key(self.kv.field.name, KB.SEED, k))

    def lin_plain(self, l: int, name: str) -> None:
        n, kd = self.shapes[name]
        if name not in self.plain_prm:
            self.torch._scaled_mm(self.xc[kd], self.wv[(l, name)], self.one, self.one, out_dtype=self.torch.bfloat16)
            return
        c, s, gr = self.plain_prm[name]
        y = self.y[n]
        y.zero_()
        rc = self.wt.so.wt_gemm(0, self.W.data_ptr() + self.off[(l, name)], self.x[kd].data_ptr(), y.data_ptr(),
                                self.m, n, kd, 0, c, s, gr, 0, self.xr.data_ptr(), None, BW.stream(self.torch))
        if rc:
            raise RuntimeError(f"wt_gemm {name}: error {rc}")

    def lin_enc(self, l: int, name: str, k: int, sketch: bool) -> None:
        n, kd = self.shapes[name]
        cfg, gr, steps, _ = self.prm[(name, k, "sketch" if sketch else "untagged")]
        y = self.y[n]
        y.zero_()
        o = self.off[(l, name)]
        to = self.tag_off[(k, l, name)]
        sg = (self.frags if BW.sketch_warps(cfg) else self.signs).data_ptr() if sketch else None
        rc = self.wt.so.wt_fused_sketch(
            k, self.W.data_ptr() + o, self.x[kd].data_ptr(), y.data_ptr(), self.m, n, kd, o // self.wt.s, 0, cfg[0],
            cfg[1], gr, 0, steps.data_ptr(), self.tags.data_ptr() + 32 * to if sketch else None,
            BW.TAG_BASE + to, 16 if sketch else 0, sg, None, self.T,
            BW.sketch_warps(cfg), None, BW.stream(self.torch))
        if rc:
            raise RuntimeError(f"wt_fused_sketch {name} k={k}: error {rc}")

    def begin_pass(self) -> None:
        """Pass j's signs (and their B fragments), key and leaves' prefix from root_(j−1) (wt_pass, on the stream
        behind the last commit)."""
        st = BW.stream(self.torch)
        self.wt.call("wt_pass", self.pass_j, self.root_ptr, self.signs.data_ptr(), self.T, st)
        self.wt.call("wt_sign_frags", self.signs.data_ptr(), self.frags.data_ptr(), self.T, st)

    def commit(self, k: int) -> None:
        st = BW.stream(self.torch)
        n = self.nvec[k]
        self.wt.call("wt_tag_leaves", self.tags.data_ptr(), n, self.nodes.data_ptr(), st)
        ri = ctypes.c_ulonglong(0)
        self.wt.call("wt_tag_tree", self.nodes.data_ptr(), -(-n // BW.TG.PER_LEAF), ctypes.addressof(ri), st)
        self.root_ptr, self.pass_j = self.nodes.data_ptr() + 64 * ri.value, self.pass_j + 1

    def gate_chain(self, k: int, log) -> dict:
        """Two sketched passes (the linears and the commit): pass j + 1's seed, key and signs on the device are
        pass_seed(σ, j + 1, root_j) of pass j's root as committed, and its first leaves are at that seed."""
        torch, wt = self.torch, self.wt
        self.set_keys(k)
        self.step(k, "none", "W")
        torch.cuda.synchronize()
        off = self.root_ptr - self.nodes.data_ptr()
        root = BW.digests(BW.host_bytes(self.nodes[off:off + 64]))[0]
        j = self.pass_j
        self.step(k, "none", "W")
        torch.cuda.synchronize()
        got = BW.pass_state(wt)
        seed = SKT.pass_seed(BW.TAG_SEED, j, root)
        n = SKT.vector_bytes(wt.sb, self.T)
        want = SKT.sign_fragments(SKT.sign_bits(seed, 16, n), SKT.steps(wt.sb), self.T)
        ys = BW.host_bytes(self.tags[:64 * 32])
        leaves = SKT.leaves(seed, [ys[32 * i:32 * i + 32] for i in range(64)])
        out = {"j": j, "seed": got["seed"] == seed, "key": got["key"] == SKT.key(seed),
               "signs": bool(np.array_equal(self.signs.cpu().numpy().view(np.uint32), want)),
               "leaves": BW.digests(BW.host_bytes(self.nodes[:4 * 64])) == leaves}
        out["ok"] = all(v for a, v in out.items() if a != "j")
        log(f"chain gate k={k}: {out}")
        return out

    def attn(self, l: int, form: int, k: int, splits: int | None = None) -> None:
        sp = splits if splits is not None else self.splits[(form, k)]
        rc = self.kv.so.kv_attn(self.KV.data_ptr(), form, self.q.data_ptr(), self.o_attn.data_ptr(),
                                self.part.data_ptr(), self.b, self.ctx, self.layers, l, sp, k,
                                BW.stream(self.torch))
        if rc:
            raise RuntimeError(f"kv_attn form {form} k={k}: error {rc}")

    def append(self, k: int, l0: int, nl: int, s) -> None:
        src = self.new_all if nl > 1 else self.new_one
        rc = self.kv.so.kv_append(src.data_ptr(), self.KV.data_ptr(), self.b, self.ctx, self.layers, l0, nl,
                                  self.ctx - 1, k, s)
        if rc:
            raise RuntimeError(f"kv_append k={k}: error {rc}")

    def read_pass(self, grid: int, bufs: str = "WK") -> None:
        st = BW.stream(self.torch)
        for buf in [b for n, b in (("W", self.W), ("K", self.KV)) if n in bufs]:
            self.kv.call("kv_read", buf.data_ptr(), buf.numel(), self.xr.data_ptr(), grid, st)

    def decode_alone(self, codec: str, k: int, grid: int) -> None:
        """The weights' words decoded (to the sink, nothing stored) by wt.cu's decoder or kv.cu's, nothing else."""
        st, n = BW.stream(self.torch), self.w_bytes // self.wb
        if codec == "wt":
            self.wt.call("wt_decode", self.W.data_ptr(), None, n, 0, k, self.xr.data_ptr(), grid, st)
        else:
            self.kv.call("kv_decode", self.W.data_ptr(), None, n, 0, k, grid, self.xr.data_ptr(), st)

    # arms ---------------------------------------------------------------------------------------------------------
    def step(self, k: int, append: str = "async", parts: str = "WA") -> None:
        """k = 0: the plain step. append: async (one launch on the second stream, beside the step), layerwise (each
        layer's after its qkv, on the second stream), serial (one launch first, on the main stream) or none. parts: W
        (the linears), A (attention), u (the linears untagged)."""
        torch = self.torch
        main = torch.cuda.current_stream()
        enc = k > 0
        sketch = enc and "u" not in parts
        if append == "async":
            self.ev[-1].record(main)
            self.s2.wait_event(self.ev[-1])
            self.append(k, 0, self.layers, self.s2.cuda_stream)
        elif append == "serial":
            self.append(k, 0, self.layers, main.cuda_stream)
        if sketch and ("W" in parts):
            self.begin_pass()
        for l in range(self.layers):
            for name in ORDER:
                if "W" in parts or "u" in parts:
                    self.lin_enc(l, name, k, sketch) if enc else self.lin_plain(l, name)
                if name == "qkv":
                    if append == "layerwise":
                        self.ev[l].record(main)
                        self.s2.wait_event(self.ev[l])
                        self.append(k, l, 1, self.s2.cuda_stream)
                    if "A" in parts:
                        self.attn(l, self.form[k], k)
        if "W" in parts or "u" in parts:
            self.lin_enc(self.layers, "lm_head", k, sketch) if enc else self.lin_plain(self.layers, "lm_head")
            if sketch:
                self.commit(k)
        if append in ("async", "layerwise"):
            self.ev[-2].record(self.s2)
            main.wait_event(self.ev[-2])


def queued(wt, torch, fn, cycles: int) -> tuple[float, float]:
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    wt.call("wt_spin", cycles, BW.stream(torch))
    a.record()
    h = time.perf_counter()
    fn()
    h = time.perf_counter() - h
    b.record()
    b.synchronize()
    return a.elapsed_time(b) / 1e3, h


def choose_splits(S: Step, torch, reps: int, cycles: int, log) -> dict:
    """Per attention form (each plain form, and each encoded form at each k) the split count whose pass over every layer
    is the fastest, at the step's shape on the step's cache; per k the fastest form (k = 0: plain)."""
    out = {}
    plains, encs = forms_of(S.sh)
    for k in [0] + S.ks:
        if k:
            S.set_keys(k)
        forms = encs if k else plains
        for form in forms:
            per = {}
            for sp in S.split_cands[form]:
                def fn(sp=sp, form=form, k=k):
                    for l in range(S.layers):
                        S.attn(l, form, max(k, 1), sp)
                fn()
                per[sp] = statistics.median(queued(S.wt, torch, fn, cycles)[0] for _ in range(reps))
            best = min(per, key=per.get)
            S.splits[(form, k)] = best
            out[f"form{form}_k{k}"] = {"splits_s": per, "best": best, "best_s": per[best]}
        S.form[k] = min(forms, key=lambda fm: out[f"form{fm}_k{k}"]["best_s"])
        out[f"k{k}_form"] = S.form[k]
        log(f"splits k={k}: " + ", ".join(f"{FORMS[fm]} {out[f'form{fm}_k{k}']['best']} "
                                          f"({out[f'form{fm}_k{k}']['best_s'] * 1e3:.2f} ms)" for fm in forms)
            + f"; form {FORMS[S.form[k]]}")
    return out


def decode_alone(S: Step, torch, ks, sms: int, grid: int, log) -> dict:
    """The decode's own cost, apart from the GEMM and attention it is fused into: each codec's decoder over the weights'
    words at each k (its best grid), in reads of the same bytes."""
    rd = statistics.median(queued(S.wt, torch, lambda: S.read_pass(grid, "W"), 20_000_000)[0] for _ in range(3))
    out = {"read_s": rd, "bytes": S.w_bytes}
    for k in ks:
        S.set_keys(k)
        for codec in ("wt", "kv"):
            per = {}
            for mult in (4, 8, 16):
                def fn(codec=codec, k=k, mult=mult):
                    S.decode_alone(codec, k, sms * mult)
                fn()
                per[mult] = statistics.median(queued(S.wt, torch, fn, 20_000_000)[0] for _ in range(3))
            best = min(per, key=per.get)
            out[f"{codec}_k{k}"] = {"per_mult_s": per, "best": best, "best_s": per[best], "over_read": per[best] / rd}
    log(f"decode alone over the weights ({S.w_bytes / 1e9:.2f} GB, read {rd * 1e3:.2f} ms), in reads: " + ", ".join(
        f"k={k} wt.cu {out[f'wt_k{k}']['over_read']:.2f} kv.cu {out[f'kv_k{k}']['over_read']:.2f}" for k in ks))
    return out


def predicted(fkv: float, ctx: int, k: int, c: float) -> float:
    m = MODEL_TERMS
    return 1 + fkv * max(0.0, c * k - m["h"]) + fkv * k * m["x1"] / ctx + m["tau"]


def model_terms(res: dict) -> dict:
    """The measured terms in native passes beside the mainline model's."""
    t = res["times"]
    p = t["pass"]
    fkv, fw, ctx = res["f_kv"], res["f_w"], res["ctx"]
    out = {"pass_s": p, "plain_over_pass": t["plain"] / p, "k": {}}
    for k in res["ks"]:
        e = f"enc{k}"
        row = {
            "step_over_plain": t[e] / t["plain"],
            "d": t[e] / p,
            "d_layerwise": t[f"{e}_layerwise"] / p,
            "weights": (t[f"{e}_W"] - t["plain_W"]) / p,
            "weights_decode": (t[f"{e}_Wu"] - t["plain_W"]) / p,
            "weights_tag": (t[f"{e}_W"] - t[f"{e}_Wu"]) / p,
            "attention": (t[f"{e}_A"] - t["plain_A"]) / p,
            "append_async": (t[e] - t[f"{e}_noappend"]) / p,
            "append_serial": t[f"{e}_P"] / p,
            "append_plain_async": (t["plain"] - t["plain_noappend"]) / p,
            "parts_sum": (t[f"{e}_W"] + t[f"{e}_A"]) / p,
            "noappend": t[f"{e}_noappend"] / p,
            "model_attention": [fkv * max(0.0, c * k - MODEL_TERMS["h"]) for c in MODEL_TERMS["c"]],
            "model_append": fkv * k * MODEL_TERMS["x1"] / ctx,
            "model_tau": MODEL_TERMS["tau"],
            "model_step": [predicted(fkv, ctx, k, c) for c in MODEL_TERMS["c"]],
            "mu_rebuild_pooled": t[e] / p / (k * MODEL_TERMS["x1_pooled"]),
            "mu_rebuild_locked": t[e] / p / (k * MODEL_TERMS["x1"]),
        }
        row["attention_per_kv_pass"] = row["attention"] / fkv
        row["weights_per_w_pass"] = row["weights"] / fw
        row["o_w_per_w_pass"] = row["weights_decode"] / fw
        row["tau_w_per_w_pass"] = row["weights_tag"] / fw
        out["k"][k] = row
    ks = res["ks"]
    if len(ks) >= 2:
        xs = np.array(ks, dtype=float)
        ys = np.array([out["k"][k]["attention_per_kv_pass"] for k in ks])
        c, mh = np.polyfit(xs, ys, 1)
        out["attention_fit"] = {"c": float(c), "h": float(-mh)}
    return out


POINTS_SCHEMA = "verity/hbm-step-points/v0"


def step_points(res: dict, run_id: str | None) -> dict:
    """The step's points for ra-overhead-frontier's model, one a k: d and the step over plain, and the weights' decode
    o_W and receipt τ_W per weight pass (its `o_w`, `tau_w` at the linears' M), attention per KV pass, the append."""
    pts = []
    for k, r in res["model"]["k"].items():
        pts.append({"model": res["model_name"], "batch": res["batch"], "gemm_m": res["gemm_m"], "ctx": res["ctx"],
                    "k": int(k), "word_bits": 512, "k_s": 16, "receipt": "ro-sketch-blake3/v0, re-keyed every pass",
                    "f_kv": res["f_kv"], "f_w": res["f_w"], "stored_bytes": res["stored_bytes"],
                    "step_over_plain": r["step_over_plain"], "d": r["d"], "plain_over_pass": res["model"]["plain_over_pass"],
                    "o_w_per_w_pass": r["o_w_per_w_pass"], "tau_w_per_w_pass": r["tau_w_per_w_pass"],
                    "attention_per_kv_pass": r["attention_per_kv_pass"], "append_async": r["append_async"],
                    "mu_rebuild_pooled": r["mu_rebuild_pooled"],
                    "sketch_configs": {n: p["k"].get(k, p["k"].get(str(k)))["sketch"]
                                       for n, p in res["picks"].items()}})
    return {"schema": POINTS_SCHEMA, "run": run_id, "device": res["device"], "clock_hz": res.get("clock_hz"),
            "points": pts}


def write_result(res: dict, out_dir: Path, libs) -> None:
    run_dir, run_id = os.environ.get("RESEARCH_RUN_DIR"), os.environ.get("RESEARCH_RUN_ID")
    if "model" in res:
        (out_dir / "step-points.json").write_text(json.dumps(step_points(res, run_id), indent=1))
    if not run_dir or not run_id:
        return
    ms = []
    for a, v in res.get("times", {}).items():
        ms.append({"name": f"{a}_s", "value": v, "unit": "s"})
    for k, row in res.get("model", {}).get("k", {}).items():
        for name in ("step_over_plain", "d", "weights", "weights_tag", "attention", "append_async"):
            ms.append({"name": f"k{k}_{name}", "value": row[name], "unit": "1"})
    rel = os.path.relpath(out_dir, run_dir)
    files = [f"{rel}/step.json", f"{rel}/step.log"] + ([f"{rel}/step-points.json"] if "model" in res else [])
    doc = {"schema": "research/result/v0.1", "run_id": run_id,
           "workload_fingerprint": {"libs": {Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                                             for p in libs},
                                    "model": res["model_name"], "batch": res["batch"], "ctx": res.get("ctx")},
           "validation": {"status": "passed" if res["ok"] else "failed", "checker": "hbm step gates",
                          "detail": "codecs = wide.py and kv.py; fused partial sums = plain, sketch tags and root; "
                                    "encoded attention = plain; append = kv_encode", "evidence": files},
           "measurements": ms, "measurement_files": files, "artifacts": files}
    tmp = Path(run_dir) / "result.json.tmp"
    tmp.write_text(json.dumps(doc, indent=1))
    os.replace(tmp, Path(run_dir) / "result.json")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wt-lib", required=True, help="wt.cu built at -DWT_FIELD=512")
    ap.add_argument("--kv-lib", required=True, help="step.cu built at -DKV_FIELD=512")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="8b", choices=["8b", "70b", "tiny"])
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--gemm-m", type=int, default=0,
                    help="the linears' M (0: the batch; the fused kernel takes at most 32): past 32, attention and the "
                         "append run at the batch and the linears at this M, a proxy where both native and fused "
                         "linears are bound by the weights' read and decode, not the MMA")
    ap.add_argument("--ctx", type=int, default=0, help="0: as many tokens as fill the memory left after the weights")
    ap.add_argument("--margin-gib", type=float, default=2.5, help="memory left free for the step's other buffers")
    ap.add_argument("--ks", default="1,2,3")
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--gemm-reps", type=int, default=5)
    ap.add_argument("--sketch-t", type=int, default=8)
    ap.add_argument("--gate-words", type=int, default=1 << 16)
    ap.add_argument("--gate-ctx", type=int, default=0, help="the attention gate's context (0: the step's)")
    a = ap.parse_args()
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    logf = (out_dir / "step.log").open("a")

    def log(msg: str) -> None:
        line = f"{time.strftime('%H:%M:%S', time.gmtime())}Z {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    import torch

    ks = [int(v) for v in a.ks.split(",")]
    m = a.gemm_m or a.batch
    if m > 32:
        raise SystemExit(f"the fused kernel's M is at most 32, not {m}: pass --gemm-m 32")
    if a.model == "tiny":   # the sanitizer's smoke run: gemm_case's L2-defeating ring would be thousands of copies
        BW.L2_DEFEAT_BYTES = 0
    wt = BW.Lib(a.wt_lib)
    kv = KB.Lib("p512", a.kv_lib)
    v, i = ctypes.c_void_p, ctypes.c_int
    kv.so.kv_append.argtypes = [v, v, i, i, i, i, i, i, i, v]
    wt.so.wt_prim.argtypes = [i, v, v, v, i, v]
    if wt.n != 512 or kv.field.n != 512 or max(ks) > min(wt.kmax, kv.kmax):
        raise SystemExit(f"want p512 builds with k up to {max(ks)}: wt {wt.n} bits kmax {wt.kmax}, kv {kv.field.n}")
    ash = KB.attn_shape(kv)
    if HEADS.get(a.model, ash["heads"]) != ash["heads"]:
        raise SystemExit(f"{a.kv_lib} has {ash['heads']} query heads, {a.model} {HEADS[a.model]}: build KV_QHEADS")
    BW.set_tag(wt)
    props = torch.cuda.get_device_properties(0)
    sms = props.multi_processor_count
    layers, shapes, lm = model_shapes(a.model)
    all_shapes = {**shapes, "lm_head": lm}
    res = {"device": props.name, "sms": sms, "total_memory": props.total_memory, "model_name": a.model,
           "batch": a.batch, "gemm_m": m, "ks": ks, "args": vars(a), "ok": True, "gate": {}}

    def save() -> None:
        (out_dir / "step.json").write_text(json.dumps(res, indent=1, default=str))

    res["attn_shape"] = ash
    log(f"{props.name}: {sms} SMs, {props.total_memory / 2**30:.1f} GiB; {a.model} B={a.batch} (linears at M={m}) k {ks}; "
        f"kv_attn {ash}")
    res["gate"]["layout"] = BW.gate_layout(torch, log)
    res["gate"]["wt_prims"] = gate_prims(wt, torch, log)
    res["gate"]["wt_codec"] = {k: BW.gate_codec(wt, torch, k, 256, log) for k in ks}
    res["gate"]["kv_codec"] = KB.gate(kv, torch, ks, 48, a.gate_words, log)
    res["gate"]["append"] = gate_append(kv, torch, ks, log)
    res["gate"]["pass"] = BW.gate_pass(wt, torch, a.sketch_t, log)
    BW.set_tag(wt)
    res["ok"] = (res["gate"]["layout"]["ok"] and res["gate"]["wt_prims"]["ok"]
                 and all(g["ok"] for g in res["gate"]["wt_codec"].values())
                 and all(g["ok"] for g in res["gate"]["kv_codec"].values()) and res["gate"]["append"]["ok"]
                 and res["gate"]["pass"]["ok"])
    save()
    if not res["ok"]:
        log("a codec gate failed: nothing timed")
        write_result(res, out_dir, [a.wt_lib, a.kv_lib])
        return 5

    gemm, fused, plain = gate_gemm(wt, torch, all_shapes, m, ks, a.gemm_reps, a.sketch_t, log)
    res["gemm"] = gemm
    res["ok"] = all(g.get("ok") for g in gemm.values()) and len(gemm) == len(all_shapes)
    save()
    if not res["ok"]:
        log("a GEMM gate failed: nothing timed")
        write_result(res, out_dir, [a.wt_lib, a.kv_lib])
        return 5
    picks = {name: pick(g, m, ks, fused, plain) for name, g in gemm.items()}
    res["picks"] = picks
    for name, p in picks.items():
        log(f"pick {name}: native {p['native']}; " + ", ".join(
            f"k{k} sketch {BW.fused_name(k, v['sketch'])[7:]} {v['sketch_s'] / p['native_s']:.3f}x, untagged "
            f"{BW.fused_name(k, v['untagged'])[7:]} {v['untagged_s'] / p['native_s']:.3f}x" for k, v in p["k"].items()))
    torch.cuda.empty_cache()

    w_bytes = layers * sum(n * kd for n, kd in shapes.values()) + lm[0] * lm[1]
    free, total = torch.cuda.mem_get_info()
    per_ctx = a.batch * layers * K.TOKEN_LAYER_BYTES
    ctx = a.ctx or (int(free - a.margin_gib * 2**30 - w_bytes) // per_ctx) // 64 * 64
    if ctx < 64:
        raise SystemExit(f"no room for a KV cache: {free / 1e9:.1f} GB free, weights {w_bytes / 1e9:.1f} GB")
    res.update({"ctx": ctx, "free_before": free, "w_bytes": w_bytes, "kv_bytes": ctx * per_ctx})
    res["f_kv"] = res["kv_bytes"] / (res["kv_bytes"] + w_bytes)
    res["f_w"] = 1 - res["f_kv"]
    log(f"fill: {free / 1e9:.2f} GB free; weights {w_bytes / 1e9:.2f} GB, KV {res['kv_bytes'] / 1e9:.2f} GB at ctx "
        f"{ctx} (f_KV {res['f_kv']:.3f})")
    res["gate"]["attention"] = gate_attention(kv, torch, a.batch, a.gate_ctx or ctx, ks, sms, log)
    save()
    if not res["gate"]["attention"]["ok"]:
        res["ok"] = False
        log("the attention gate failed: nothing timed")
        write_result(res, out_dir, [a.wt_lib, a.kv_lib])
        return 5

    S = Step(wt, kv, torch, a.model, a.batch, m, ctx, ks, picks, a.sketch_t, log)
    free_after, _ = torch.cuda.mem_get_info()
    res["free_after"] = free_after
    res["stored_bytes"] = S.w_bytes + S.kv_bytes
    log(f"allocated: {res['stored_bytes'] / 1e9:.2f} GB stored, {free_after / 1e9:.2f} GB left free of "
        f"{total / 1e9:.2f}")
    res["gate"]["chain"] = S.gate_chain(ks[0], log)
    save()
    if not res["gate"]["chain"]["ok"]:
        res["ok"] = False
        log("the pass chain's gate failed: nothing timed")
        write_result(res, out_dir, [a.wt_lib, a.kv_lib])
        return 5
    clock = BW.sm_clock(wt, torch)
    res["clock_hz"] = clock

    best = None
    for mult in (4, 8, 16, 32):
        S.read_pass(sms * mult)
        ts = [queued(wt, torch, lambda m=mult: S.read_pass(sms * m), 20_000_000)[0] for _ in range(3)]
        if best is None or statistics.median(ts) < best[1]:
            best = (mult, statistics.median(ts))
    grid = sms * best[0]
    log(f"pass: {res['stored_bytes'] / best[1] / 1e9:.0f} GB/s at {best[0]} CTAs an SM ({best[1] * 1e3:.2f} ms)")

    BW.set_sketch_tag(wt)
    S.set_keys(ks[0])
    res["splits"] = choose_splits(S, torch, 2, 40_000_000, log)
    save()
    res["decode_alone"] = decode_alone(S, torch, ks, sms, grid, log)
    save()

    arms = {"pass": lambda: S.read_pass(grid),
            "plain": lambda: S.step(0),
            "plain_noappend": lambda: S.step(0, "none"),
            "plain_W": lambda: S.step(0, "none", "W"),
            "plain_A": lambda: S.step(0, "none", "A"),
            "plain_P": lambda: S.step(0, "serial", "")}
    for k in ks:
        arms.update({f"enc{k}": lambda k=k: S.step(k),
                     f"enc{k}_layerwise": lambda k=k: S.step(k, "layerwise"),
                     f"enc{k}_noappend": lambda k=k: S.step(k, "none"),
                     f"enc{k}_W": lambda k=k: S.step(k, "none", "W"),
                     f"enc{k}_Wu": lambda k=k: S.step(k, "none", "u"),
                     f"enc{k}_A": lambda k=k: S.step(k, "none", "A"),
                     f"enc{k}_P": lambda k=k: S.step(k, "serial", "")})
    host = {}
    for name, fn in arms.items():
        if name.startswith("enc"):
            S.set_keys(int(name[3]))
        h = time.perf_counter()
        fn()
        host[name] = time.perf_counter() - h
        torch.cuda.synchronize()
    cycles = {n: int(clock * (2 * h + 0.005)) for n, h in host.items()}
    times = {n: [] for n in arms}
    outran = {}
    names = list(arms)
    for rep in range(a.reps):
        for name in names[rep % len(names):] + names[:rep % len(names)]:
            if name.startswith("enc"):
                S.set_keys(int(name[3]))
            t, h = queued(wt, torch, arms[name], cycles[name])
            if h > cycles[name] / clock:
                outran[name] = outran.get(name, 0) + 1
            times[name].append(t)
    if outran:
        log(f"enqueue outran the spin (reps): {outran}")
    res["times_all_s"] = times
    res["host_enqueue_s"] = host
    res["enqueue_outran"] = outran
    res["times"] = {n: statistics.median(v) for n, v in times.items()}
    res["model"] = model_terms(res)
    t, p = res["times"], res["times"]["pass"]
    log(f"times (ms): " + ", ".join(f"{n} {v * 1e3:.2f}" for n, v in t.items()))
    log(f"plain step {t['plain'] / p:.3f} passes; " + "; ".join(
        f"k={k}: step {r['step_over_plain']:.3f}x plain, d {r['d']:.3f} (layerwise append {r['d_layerwise']:.3f}); "
        f"weights {r['weights']:+.3f} (decode {r['weights_decode']:+.3f}, tag {r['weights_tag']:+.3f}), attention "
        f"{r['attention']:+.3f} (model {r['model_attention'][0]:+.3f}..{r['model_attention'][1]:+.3f}), append "
        f"{r['append_async']:+.3f} async / {r['append_serial']:.3f} serial (model {r['model_append']:.3f}); model step "
        f"{r['model_step'][0]:.3f}..{r['model_step'][1]:.3f}; mu_rebuild {r['mu_rebuild_pooled']:.2%} at x1 175"
        for k, r in res["model"]["k"].items()))
    if "attention_fit" in res["model"]:
        log(f"attention fit: (c k - h) per KV pass with c {res['model']['attention_fit']['c']:.3f}, h "
            f"{res['model']['attention_fit']['h']:.3f}")
    save()
    write_result(res, out_dir, [a.wt_lib, a.kv_lib])
    log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
