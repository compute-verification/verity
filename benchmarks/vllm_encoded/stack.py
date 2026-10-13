"""The encoded stack inside vLLM (Llama-3.1-8B, FP8, one GPU): the remote bandwidth audit's own kernels in place of the
model's linears, its lm_head, its attention and its KV cache's append, so a decode step reads every weight and KV byte as
codec words, and the plain stack, the same engine over the same kernels' plain forms (its denominator, bit for bit).

~~~text
linears   stack: wt.cu (WT_RECEIPT, full slot, masked) fused decode + FP8 MMA + sketch-sha512 receipts on two sketch
          warps, at M <= 128 rows in one launch; the chunks' sums reduced in order (paged.cu's pg_reduce). pstack:
          wt_gemm over the plain fragments in launches of at most 64 rows, the same sums bit for bit (gated). M > 128
          (prefill): the layer
          decoded straight into an ephemeral scratch as its N × K matrix (stack, wt_decode_rows; gated against the
          weight) or unpermuted (pstack) for vLLM's CUTLASS GEMM.
receipts  a decode step is a pass: before the model's forward (eager, the runner's _model_forward) wt_pass derives pass
          j's seed, signs and receipt prefix from root_(j−1) on the device; each linear writes its vectors' receipts
          at the step's running offset, the lm_head's last; after it (in the lm_head's CUDA graph) reader.cu's leaves,
          pads and tree, and the root copied to the buffer the next pass reads. A forward with no lm_head after it
          (vLLM's warm-ups) leaves its pass uncommitted, and the next pass starts from the last root committed.
lm_head   BF16 payloads (Daniel's ruling: its logits are BF16's) in the linears' plain layout (the BF16 matrix's bytes,
          which mma.m16n8k16.bf16 takes as e4m3 m16n8k32 takes FP8), as words: wt.cu's wt_fused_sketch_bf decodes,
          multiplies (BF16 MMA, FP32 sums) and receipts them in one launch, the chunks' sums reduced in order to BF16;
          pstack: wt_gemm's BF16 mode over the plain fragments, the same sums. One CUDA graph a batch, its partial sums
          and logits in buffers the graphs share.
KV        vLLM's own paged cache tensors, a layer's words at their place in the pool (paged.cu): the append encodes (all
          layers' in one launch on a second stream after the step's forward; the step reads its own tokens from vLLM's
          rows), a decode-only step's attention decodes inside paged.cu's tensor-core attention (CUDA-graph safe), and
          a step with prefill gathers its sequences' pages decoded into a scratch for vLLM's FlashAttention.
~~~
"""
from __future__ import annotations

import ctypes
import dataclasses
import hashlib
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
CT = HERE.parent / "codec_tensor"
sys.path.insert(0, str(CT))

import bench as BW  # noqa: E402  (codec_tensor's; before paged, which puts word_codec, whose bench.py is another, first)
import reader as RD  # noqa: E402
import sketching as SKT  # noqa: E402

sys.path.insert(0, str(HERE))
import kernels as K  # noqa: E402
import paged as P  # noqa: E402

SR = BW.SR
T = 8                                   # tasks a sketch vector
PLAIN_CFGS = ((4, 4), (8, 8))           # wt_gemm's (consumers, stages): step.py's PLAIN_CFGS
MAX_ROWS = 128                          # a decode's rows on the fused kernel (wt_fused_sketch's most a launch)
PLAIN_ROWS = 64                         # wt_gemm's most a launch (pstack, the gates' reference)
CAPTURE = [1, 4, 8, 16, 32, 48, 64, 96, 128]
# past 64 rows: 128 rows of x leave 3 stages, and the receipt warps take groups of 2 tasks there (wt.cu receipt_tasks)
WIDE_CFGS = [(d, s, -2, 16) for d in (5, 4, 3) for s in (3, 2)]
ROOT0 = hashlib.sha512(b"verity/vllm-encoded/stack/registration").digest()   # root_(−1)
GATE_TREE = (1, 32, 128)                # batches whose whole step tree is checked against the driver's


def stream() -> int:
    return torch.cuda.current_stream().cuda_stream


def new_stream() -> torch.cuda.Stream:
    """The stack's side streams (the deferred append's, the lm_head's graph capture). A stream is in the context it was
    made in, and so is a graph captured on it: an SM partition (kv_filler's `serve.py --partition`) replaces this with
    one that makes them on the step's green context."""
    return torch.cuda.Stream()


def pieces(m: int, most: int) -> list[int]:
    """A decode of m rows as launches of at most `most`, their sizes as even as they go."""
    p = -(-m // most)
    return [m // p + (i < m % p) for i in range(p)]


def unplain(f: torch.Tensor, n: int, kd: int) -> torch.Tensor:
    """bench.plain_layout's inverse: the fragments' bytes back to the N × K matrix."""
    return f.view(n // 16, kd // 512, 16, 8, 4, 2, 2, 4).permute(0, 6, 3, 1, 2, 5, 4, 7).reshape(n, kd)


def bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int16) if t.element_size() == 2 else t.view(torch.uint8)


class Stack:
    """The libraries, the pass state and the step's receipt buffers of one arm (encoded: stack, else pstack)."""

    def __init__(self, args, encoded: bool) -> None:
        self.args, self.encoded, self.k = args, encoded, args.k if encoded else 0
        wt = self.wt = BW.Lib(args.wt_lib)
        if not (wt.receipt and wt.pb == wt.s and wt.tw * wt.pb == BW.W.TASK_BYTES and 4 * wt.nl == wt.pb):
            raise RuntimeError(f"{wt.name}: the stack takes a WT_RECEIPT build at a full-slot field (task = its words)")
        self.sms = torch.cuda.get_device_properties(0).multi_processor_count
        st = stream()
        if encoded:
            self.key = BW.lib_key(wt, self.k)
            BW.set_key(wt, torch, self.key)
        BW.set_sketch_tag(wt)
        BW.set_pass(wt, T)
        sb = BW.sketch_bits(wt, BW.window_seed(wt), 16, SKT.vector_bytes(wt.sb, T))
        self.signs = torch.from_numpy(BW.pass_signs(wt, sb, T).view(np.int32)).cuda()
        self.frags = BW.sign_frags(wt, torch, self.signs, T, st)
        self.root = BW.to_dev(torch, BW.root_words(ROOT0)).view(torch.int64)
        self.reader = BW.READER = RD.Reader(args.reader_lib, torch)
        self.j, self.at, self.pass_m, self.open = 0, 0, None, False
        self.pass_ptrs = (ctypes.c_void_p * 2)()
        self.wt.call("wt_pass_ptrs", ctypes.addressof(self.pass_ptrs))
        self.rows = MAX_ROWS if encoded else PLAIN_ROWS   # a launch's most rows
        self.fused = [m for m in CAPTURE if m <= self.rows]   # the launch sizes whose configs are selected and gated
        self.xr = torch.zeros(1, dtype=torch.int64, device="cuda")
        self.plib = P.PagedLib(f"p{wt.n}", args.paged_lib)
        if self.plib.field.n != wt.n or self.plib.mask != wt.mask:
            raise RuntimeError(f"paged.cu's field {self.plib.field.name} (mask {self.plib.mask}) is not wt.cu's")
        if encoded:
            P.KB.set_key(self.plib, P.KB.lib_key(self.plib, self.k))
        self.words = 0                       # the store's next word index (linears, then the lm_head)
        self.linears: list[StackLinear] = []   # the FP8 linears, then the lm_head
        self.lm: LmHead | None = None
        self.nvec: dict[int, int] = {}       # a step's receipts at each batch
        self.passes = 0
        self.attention: Attention | None = None

    # the step's pass --------------------------------------------------------------------------------------------
    def buffers(self) -> None:
        top = max(self.nvec.values())
        nl = -(-top // SR.rb.PER_LEAF)
        width = 1 << max(0, (nl - 1).bit_length())
        self.tags = torch.zeros(top * self.wt.tag_bytes // 8, dtype=torch.int64, device="cuda")   # not uint8, which the plaintext scan counts
        self.dig = torch.zeros((width, 8), dtype=torch.int64, device="cuda")
        self.up = torch.zeros((max(width - 1, 1), 8), dtype=torch.int64, device="cuda")

    def begin_pass(self, m: int) -> None:
        """Pass j at batch m: wt_pass from root_(j−1), then the weight and paged-KV kernels' shared pass state."""
        if not self.encoded or m not in self.nvec:
            return
        s = stream()
        self.wt.call("wt_pass", self.j & 0xFFFFFFFF, self.root.data_ptr(), self.signs.data_ptr(), T, s)
        self.wt.call("wt_sign_frags", self.signs.data_ptr(), self.frags.data_ptr(), T, s)
        if self.attention is not None:
            if self.attention.epoch is not None:
                self.attention.epoch.fill_(self.j + 1)
            self.plib.call("kv_rc_pass", self.signs.data_ptr(), self.pass_ptrs[1], s)
        self.j, self.pass_m, self.passes, self.open = self.j + 1, m, self.passes + 1, True

    def step_plan(self, m: int, slots=()):
        """The registered blocks a step's receipts at m credit (`receipts.plan_of` under store-order/v0): a WT_RECEIPT
        launch writes its store's block v's receipt at tag_base + v (`sketch_receipt.gemm_vectors`), so receipt
        lin.at[m] + v is the store's block lin.base · s / 4096 + v, each linear's launches one (m <= MAX_ROWS)."""
        bpw, runs = self.wt.s, []
        for lin in self.linears:
            if len(lin.plan[m]) != 1 or lin.receipts(m) != lin.nbytes // SR.BLOCK or (lin.base * bpw) % SR.BLOCK:
                raise RuntimeError(f"{lin.name} at m = {m}: its receipts are not its store's blocks, once")
            first, n = lin.base * bpw // SR.BLOCK, lin.receipts(m)
            if runs and runs[-1][0] + runs[-1][1] == first:
                runs[-1] = (runs[-1][0], runs[-1][1] + n)
            else:
                runs.append((first, n))
        weights = self.words * bpw // SR.BLOCK
        blocks = weights
        if self.attention is not None:
            blocks += self.attention.pool_slots * self.attention.nl
            for slot in slots:
                first, n = weights + int(slot) * self.attention.nl, self.attention.nl
                if runs and runs[-1][0] + runs[-1][1] == first:
                    runs[-1] = (runs[-1][0], runs[-1][1] + n)
                else:
                    runs.append((first, n))
        return SR.rb.Plan(tuple(runs), blocks)

    def commit(self, m: int, kv_tags=None, slots=()) -> None:
        nw, nk = self.nvec[m], 0 if kv_tags is None else kv_tags.numel() // 8
        n = nw + nk
        if self.at != nw:
            raise RuntimeError(f"a step at m = {m} wrote {self.at} weight receipts, its plan {nw}")
        plan = self.step_plan(m, slots)
        if plan.length != n:
            raise RuntimeError(f"a step at m = {m} has {n} receipts, its plan has {plan.length}")
        self.reader.begin(BW.RECEIPT_SEED, plan, BW.STORE_KEY)
        so, s = self.reader.so, stream()
        nl = -(-n // SR.rb.PER_LEAF)
        width = 1 << max(0, (nl - 1).bit_length())
        if width > self.dig.shape[0]:
            self.dig = torch.zeros((width, 8), dtype=torch.int64, device="cuda")
            self.up = torch.zeros((max(width - 1, 1), 8), dtype=torch.int64, device="cuda")
        dig = self.dig.data_ptr()
        if nk:
            if nw % SR.rb.PER_LEAF or nk % SR.rb.PER_LEAF:
                raise RuntimeError(f"split receipt buffers need whole leaves: weights {nw}, KV {nk}")
            nlw, nlk = nw // SR.rb.PER_LEAF, nk // SR.rb.PER_LEAF
            leaves = (so.rb_leaves(self.tags.data_ptr(), 0, nlw, dig, None, s),
                      so.rb_leaves(kv_tags.data_ptr(), nlw, nlk, dig + 64 * nlw, None, s))
        else:
            leaves = (so.rb_leaves(self.tags.data_ptr(), 0, nl, dig, None, s),)
        for rc in (*leaves, so.rb_pads(nl, width - nl, dig + 64 * nl, s) if width > nl else 0,
                   so.rb_tree(dig, width, 0, self.up.data_ptr(), s) if width > 1 else 0):
            if rc:
                raise RuntimeError(f"the step's commit: error {rc}")
        self.root.copy_(self.dig[0] if width == 1 else self.up[width - 2])
        self.open = False


class StackLinear:
    """One linear: its words (stack, int16 so no uint8 tensor of a linear's size is in the process) at word indices
    base … or its plain fragments (pstack), and each batch's launch. FP8 (an N × K weight, a per-tensor scale), or BF16
    (bf: the lm_head, N × K BF16 as an N × 2K byte matrix; kd counts bytes, and x is BF16 rows)."""

    def __init__(self, st: Stack, name: str, w: torch.Tensor, ws: torch.Tensor | None, bf: bool = False) -> None:
        self.st, self.name, self.idx, self.bf = st, name, len(st.linears), bf
        wb = w.view(torch.uint8)
        self.n, self.kd = wb.shape
        self.nch = self.kd // 512
        self.ws = torch.ones(1, dtype=torch.float32, device="cuda") if bf else ws.reshape(1).float()
        wt = st.wt
        plain = BW.plain_layout(wb)
        self.words = plain.numel() // wt.s
        self.k = st.k
        if st.encoded:
            self.base = st.words
            st.words += self.words
            self.store = torch.empty(plain.numel() // 2, dtype=torch.int16, device="cuda")
            wt.call("wt_encode", plain.data_ptr(), self.store.data_ptr(), self.words, self.base, st.k, 8 * st.sms,
                    stream())
        else:
            self.base, self.store = 0, plain
        self.cfgs: dict[int, tuple] = {}     # capture size -> config
        self.plan: dict[int, list] = {}      # m -> its launches' (rows, (cfg, groups, tweak steps, receipts))
        self.at: dict[int, int] = {}         # m -> the step's offset of its receipts
        st.linears.append(self)

    @property
    def nbytes(self) -> int:
        return self.store.numel() * self.store.element_size()

    @property
    def fused(self):
        so = self.st.wt.so
        return so.wt_fused_sketch_bf if self.bf else so.wt_fused_sketch

    def dry(self, m: int, cfg) -> tuple[int, int]:
        """The launch at m and cfg, not run: its rc and CTA groups."""
        launched = ctypes.c_int(0)
        if self.st.encoded:
            rc = self.fused(self.k, None, None, None, m, self.n, self.kd, 0, 0, cfg[0], cfg[1], 0, 0, None, None, 0,
                            16, None, None, T, 2, ctypes.byref(launched), stream())
        else:
            rc = self.st.wt.so.wt_gemm(3 if self.bf else 0, None, None, None, m, self.n, self.kd, 0, cfg[0], cfg[1],
                                       0, 0, None, ctypes.byref(launched), stream())
        return rc, launched.value // self.nch

    def candidates(self, m: int) -> list[tuple]:
        if not self.st.encoded:
            return list(PLAIN_CFGS)
        cfgs = [c for c in BW.fused_cfgs(self.st.wt, "auto", "16", T) if c[2] == -2]
        cfgs += [c for c in WIDE_CFGS if m > PLAIN_ROWS and c not in cfgs]
        return [c for c in cfgs if self.dry(m, c)[0] == 0]

    def probe(self, m: int, cfg) -> int:
        """The launch's CTA groups."""
        rc, gr = self.dry(m, cfg)
        if rc:
            raise RuntimeError(f"{self.name} m={m} cfg={cfg}: error {rc}")
        return gr

    def planned(self, m: int, cfg, cache: dict) -> tuple:
        at = (self.n, self.kd, m, tuple(cfg))
        if at not in cache:
            gr = self.probe(m, cfg)
            if self.st.encoded:
                steps = BW.tsteps(self.st.wt, torch, self.st.key, cfg[0], gr, self.nch)
                nvec = self.st.wt.so.wt_sketch_vectors(self.n, self.kd, gr, T)
            else:
                steps, nvec = None, 0
            cache[at] = (tuple(cfg), gr, steps, nvec)
        return cache[at]

    def launch(self, x8: torch.Tensor, part: torch.Tensor, plan: tuple, at: int, tags: torch.Tensor | None = None) -> None:
        """The linear's sums into part; its receipts into tags (the step's buffer when None) from receipt `at`."""
        cfg, gr, steps, _ = plan
        st, wt, m = self.st, self.st.wt, x8.shape[0]
        if st.encoded:
            tags = st.tags if tags is None else tags
            rc = self.fused(self.k, self.store.data_ptr(), x8.data_ptr(), part.data_ptr(), m, self.n, self.kd,
                            self.base, 1, cfg[0], cfg[1], gr, 0, steps.data_ptr(), tags.data_ptr() + wt.tag_bytes * at,
                            wt.jbase + at, 16, st.frags.data_ptr(), None, T, 2, None, stream())
        else:
            rc = wt.so.wt_gemm(3 if self.bf else 0, self.store.data_ptr(), x8.data_ptr(), part.data_ptr(), m, self.n,
                               self.kd, 1, cfg[0], cfg[1], gr, 0, st.xr.data_ptr(), None, stream())
        if rc:
            raise RuntimeError(f"{self.name} m={m}: {'the fused kernel' if st.encoded else 'wt_gemm'} error {rc}")

    def sums(self, x8: torch.Tensor, plan: tuple, at: int | None = None) -> torch.Tensor:
        """The sums; at None (the gates, before the step's buffers), the receipts into a buffer of their own."""
        part = torch.empty((self.nch, x8.shape[0], self.n), dtype=torch.float32, device="cuda")
        own = self.own_tags(plan) if at is None else None
        self.launch(x8, part, plan, at or 0, own)
        return part

    def own_tags(self, plan: tuple) -> torch.Tensor | None:
        if not self.st.encoded:
            return None
        return torch.empty(max(plan[3], 1) * self.st.wt.tag_bytes // 8, dtype=torch.int64, device="cuda")

    def receipts(self, m: int) -> int:
        return sum(p[3] for _, p in self.plan[m])

    def reduce(self, part: torch.Tensor, xs: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        m = part.shape[1]
        if out is None:
            out = torch.empty((m, self.n), dtype=torch.bfloat16, device="cuda")
        self.st.plib.call("pg_reduce", part.data_ptr(), part.shape[0], m, self.n, xs.data_ptr(), int(xs.numel() > 1),
                          self.ws.data_ptr(), out.data_ptr(), stream())
        return out

    def fragments(self, scratch: torch.Tensor | None = None) -> torch.Tensor:
        """The plain fragments: the store decoded (into scratch, int32) or the store itself."""
        if not self.st.encoded:
            return self.store
        out = scratch[: self.store.numel() // 2] if scratch is not None else torch.empty(
            self.store.numel() // 2, dtype=torch.int32, device="cuda")
        self.st.wt.call("wt_decode", self.store.data_ptr(), out.data_ptr(), self.words, self.base, self.k,
                        self.st.xr.data_ptr(), 8 * self.st.sms, stream())
        return out.view(torch.uint8)

    def rows(self, scratch: torch.Tensor | None = None) -> torch.Tensor:
        """The N × K matrix (uint8), row major: the store decoded straight into it (into scratch, int32), or the plain
        fragments unpermuted."""
        if not self.st.encoded:
            return unplain(self.store, self.n, self.kd)
        out = scratch[: self.store.numel() // 2] if scratch is not None else torch.empty(
            self.store.numel() // 2, dtype=torch.int32, device="cuda")
        self.st.wt.call("wt_decode_rows", self.store.data_ptr(), out.data_ptr(), self.words, self.base, self.k, self.kd,
                        self.st.xr.data_ptr(), 8 * self.st.sms, stream())
        return out.view(torch.uint8).view(self.n, self.kd)

    def forward(self, x8: torch.Tensor, xs: torch.Tensor) -> torch.Tensor:
        m, st = x8.shape[0], self.st
        xs = xs.reshape(-1).float().contiguous()
        if m > MAX_ROWS:
            from vllm import _custom_ops as ops
            w = self.rows(st.scratch).view(torch.float8_e4m3fn)
            return ops.cutlass_scaled_mm(x8, w.t(), out_dtype=torch.bfloat16, scale_a=xs, scale_b=self.ws, bias=None)
        if self.idx == 0:
            st.at = 0
        x8 = x8.contiguous()
        y = torch.empty((m, self.n), dtype=torch.bfloat16, device="cuda")
        r0 = 0
        for r, plan in self.plan[m]:
            self.reduce(self.sums(x8[r0:r0 + r], plan, st.at), xs if xs.numel() == 1 else xs[r0:r0 + r], y[r0:r0 + r])
            st.at += plan[3]
            r0 += r
        return y


class LmHead:
    """The lm_head: a BF16 StackLinear, the pass's last weight receipts after the FP8 linears, into partial sums and
    logits the batches' CUDA graphs share. The forward hook commits them with the KV receipts. Outside a pass,
    a prefill step's logits,
    vLLM's warm-ups) its receipts go to a buffer of their own and nothing is committed."""

    def __init__(self, st: Stack, layer) -> None:
        w = layer.weight
        if w.dtype != torch.bfloat16 or not w.is_contiguous():
            raise RuntimeError(f"lm_head weight {w.dtype} {tuple(w.shape)}: the stack takes contiguous BF16")
        self.st = st
        self.v, self.h = w.shape
        self.lin = StackLinear(st, "lm_head", w, None, bf=True)
        self.part = torch.empty(self.lin.nch * MAX_ROWS * self.v, dtype=torch.float32, device="cuda")
        self.out = torch.empty((MAX_ROWS, self.v), dtype=torch.bfloat16, device="cuda")
        self.ones = torch.ones(1, dtype=torch.float32, device="cuda")
        self.graphs: dict[tuple, tuple] = {}
        self.cap = None

    @property
    def nbytes(self) -> int:
        return self.lin.nbytes

    @property
    def scratch_bytes(self) -> int:
        return self.part.numel() * 4 + self.out.numel() * 2

    def run(self, x: torch.Tensor, commit: bool) -> torch.Tensor:
        """Logits of x (m <= MAX_ROWS BF16 rows, contiguous) into self.out[:m]; commit writes at the pass's offset."""
        st, lin, m = self.st, self.lin, x.shape[0]
        y = self.out[:m]
        at = lin.at[st.pass_m] if commit else 0
        r0 = 0
        for r, plan in lin.plan[m]:
            part = self.part[: lin.nch * r * self.v].view(lin.nch, r, self.v)
            lin.launch(x[r0:r0 + r], part, plan, at, None if commit else lin.own_tags(plan))
            lin.reduce(part, self.ones, y[r0:r0 + r])
            at += plan[3] if commit else 0
            r0 += r
        if commit:
            st.at = at
        return y

    def apply(self, layer, x: torch.Tensor, bias=None) -> torch.Tensor:
        st = self.st
        x2 = x.reshape(-1, self.h).contiguous()
        m = x2.shape[0]
        capturing = torch.cuda.is_current_stream_capturing()
        commit = st.encoded and st.open and not capturing and m <= MAX_ROWS
        if m > MAX_ROWS or capturing or (st.encoded and not commit):
            y = torch.cat([self.run(x2[r0:r0 + MAX_ROWS], False).clone() for r0 in range(0, m, MAX_ROWS)])
        else:
            key = (m, commit, self.lin.at[st.pass_m] if commit else 0)
            g = self.graphs.get(key)
            if g is None:
                xs = x2.clone()
                self.run(xs, commit)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                if self.cap is None:
                    self.cap = new_stream()
                # captured mid-serve, beside the filler's thread: under "global" its CUDA calls invalidate the capture
                with torch.cuda.graph(graph, stream=self.cap, capture_error_mode="thread_local"):
                    self.run(xs, commit)
                g = self.graphs[key] = (graph, xs)
            g[1].copy_(x2)
            g[0].replay()
            y = self.out[:m].clone()
        if commit:
            st.attention.finish_pass()
        if bias is not None:
            y = y + bias
        return y.reshape(*x.shape[:-1], self.v)


@dataclasses.dataclass
class _Layer:
    lay: P.Layout
    index: int


class Attention:
    """FlashAttentionImpl's forward and KV append over paged.cu. The append is deferred: a step's tokens reach the pages
    after its forward, every layer's in one launch (pg_append_multi) on a second stream, which the next forward waits
    for (the codec's encode is a long chain a word, ~5 ms at p1024, k = 3: one wave a step instead of one a layer on
    the critical path). Until then a step reads its own tokens' K and V from vLLM's rows: a decode-only step
    (max_query_len 1) in paged.cu's attention with each sequence's last token from them, any other step through vLLM's
    FlashAttention over its sequences' pages gathered (decoded) into a scratch, its own tokens written over their slots
    there, with the block table renumbered. In a captured step the rows are staged (at most MAX_ROWS a layer, in the
    graph); outside one the append reads vLLM's own tensors, held for the second stream."""

    def __init__(self, st: Stack, layers: int) -> None:
        self.st, self.lib = st, st.plib
        self.enc = int(st.encoded)
        self.layers: dict[tuple, _Layer] = {}   # by (layer, the cache's storage and shape): vLLM may reallocate it
        self.splits: dict[int, int] = {}
        top = max(P.part_floats(self.lib, b, self.split(b)) for b in range(1, MAX_ROWS + 1))
        self.part = torch.empty(top, dtype=torch.float32, device="cuda")   # the captured steps' (b <= MAX_ROWS)
        self.part_eager = self.part                                        # an eager decode's at any b, grown
        self._plan: tuple | None = None
        self.calls = {"decode": 0, "gathered": 0, "append": 0, "flush_graph": 0, "flush_eager": 0, "gather_grow": 0}
        self.gather_buf: torch.Tensor | None = None
        self.nl = layers
        row = self.lib.shape["kv_heads"] * self.lib.shape["head_dim"]
        self.stage_k = torch.zeros((layers, MAX_ROWS, row), dtype=torch.bfloat16, device="cuda")
        self.stage_v = torch.zeros_like(self.stage_k)
        self.stage_s = torch.full((layers, MAX_ROWS), -1, dtype=torch.int64, device="cuda")
        self.staged: dict[int, tuple] = {}      # layer -> (cache, store base, layer count, layer index)
        self.graph_tab = None
        self.pending: list[tuple] = []          # an eager step's append-table entries, by layer
        self.slots: dict[int, torch.Tensor] = {}
        self.lay: P.Layout | None = None
        self.s2 = new_stream()
        self.fwd_done = torch.cuda.Event()
        self.app_done = torch.cuda.Event()
        self.outstanding = False
        self.stage_bytes = 2 * self.stage_k.numel() * 2
        self.pool_slots = 0
        self.rc_sk = self.used = self.epoch = self.sparse_tags = None
        self.kv_tags = None
        self.kv_tag_capacity = 0
        # gate_step left a valid wt_pass state; initialize the paged constants before vLLM's warmups and captures.
        if st.encoded:
            self.lib.call("kv_rc_pass", st.signs.data_ptr(), st.pass_ptrs[1], stream())

    def partials(self, b: int) -> torch.Tensor:
        """Attention's split partials at batch b: the captured steps' buffer, or (b > MAX_ROWS, eager) one grown to fit
        and kept apart, since the graphs hold the first's address."""
        if b <= MAX_ROWS:
            return self.part
        need = P.part_floats(self.lib, b, self.split(b))
        if need > self.part_eager.numel():
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(f"a captured decode at batch {b}: the stack captures at most {MAX_ROWS}")
            self.part_eager = torch.empty(need, dtype=torch.float32, device="cuda")
        return self.part_eager

    def gathered(self, n: int, kv_cache: torch.Tensor) -> torch.Tensor:
        """A gathered step's scratch of n blocks in the cache's order within a block (which the gather copies): a view
        of one buffer, grown to the next power of two, since one sized each step reaches cudaMalloc and cudaFree
        (device-wide syncs that stall the filler's relaunches) once the cache fills the device."""
        g = self.gather_buf
        if g is None or g.shape[0] < n or g.shape[1:] != kv_cache.shape[1:] or g.stride() != kv_cache.stride():
            self.gather_buf = None
            g = self.gather_buf = torch.empty_strided((1 << (n - 1).bit_length(), *kv_cache.shape[1:]),
                                                      kv_cache.stride(), dtype=kv_cache.dtype, device="cuda")
            self.calls["gather_grow"] += 1
        return g[:n]

    def split(self, b: int) -> int:
        if b not in self.splits:
            self.splits[b] = P.splits_for(b, self.st.sms, self.lib.shape["kv_heads"])
        return self.splits[b]

    def reserve(self, caches) -> None:
        """Reserve sparse receipt state for the cache pool before vLLM captures graphs over that pool."""
        def tensors(x):
            if isinstance(x, torch.Tensor):
                yield x
            elif isinstance(x, dict):
                for y in x.values():
                    yield from tensors(y)
            elif isinstance(x, (list, tuple)):
                for y in x:
                    yield from tensors(y)
            elif isinstance(getattr(x, "tensor", None), torch.Tensor):
                yield x.tensor

        for cache in tensors(caches):
            try:
                lay = P.layout(cache, self.lib.shape["kv_heads"], self.lib.shape["wb"])
            except ValueError:
                continue
            self.reserve_slots(lay.blocks * lay.bs)
            return

    def reserve_slots(self, slots: int) -> None:
        if slots == self.pool_slots:
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(f"the paged receipt pool changed from {self.pool_slots} to {slots} during CUDA graph capture")
        # vLLM profiles with a small temporary cache before allocating the serving pool. No captured graph may retain
        # that temporary cache's receipt buffers or append table.
        self.layers.clear()
        self.staged.clear()
        self.pending.clear()
        self.graph_tab = None
        self.pool_slots = slots
        self.rc_sk = self.used = self.epoch = self.sparse_tags = None
        if self.st.encoded:
            self.rc_sk = torch.empty((slots, self.lib.shape["kv_heads"], 16), dtype=torch.int32, device="cuda")
            self.used = torch.full((slots,), -1, dtype=torch.int32, device="cuda")
            self.epoch = torch.full((1,), self.st.j, dtype=torch.int32, device="cuda")
            self.sparse_tags = torch.empty((slots, self.nl, 8), dtype=torch.int64, device="cuda")

    @staticmethod
    def index(layer) -> int:
        return int(layer.layer_name.split(".layers.")[1].split(".")[0])

    def layer(self, layer, kv_cache: torch.Tensor) -> _Layer:
        idx = self.index(layer)
        key = (idx, kv_cache.data_ptr(), tuple(kv_cache.shape))
        L = self.layers.get(key)
        if L is None:
            lay = P.layout(kv_cache, self.lib.shape["kv_heads"], self.lib.shape["wb"])
            if self.lay is not None and (lay.bs, lay.sb, lay.sh, lay.ss) != (self.lay.bs, self.lay.sb, self.lay.sh,
                                                                               self.lay.ss):
                raise RuntimeError(f"layer {idx}'s cache layout {lay} is not the others' {self.lay}")
            self.lay = lay
            slots = lay.blocks * lay.bs
            if slots != self.pool_slots:
                if idx:
                    raise RuntimeError(f"layer {idx}'s cache has {slots} slots, the others have {self.pool_slots}")
                self.reserve_slots(slots)
            L = self.layers[key] = _Layer(lay, idx)
        return L

    def receipts(self):
        return (self.rc_sk, self.used, self.epoch) if self.st.encoded else None

    def receipt_layer(self, layer: int) -> None:
        if self.st.encoded:
            P.receipts_sparse(self.lib, torch, self.rc_sk, self.sparse_tags, self.used, self.epoch, self.pool_slots,
                              self.nl, layer, self.st.words * self.st.wt.s // SR.BLOCK)

    def finish_pass(self) -> None:
        """Compact the step's selected physical slots in store order, configure its public plan, and commit all receipts."""
        st = self.st
        if not st.encoded or not st.open:
            return
        slots_dev = torch.nonzero(self.used == self.epoch, as_tuple=False).flatten()
        slots = slots_dev.cpu().tolist()
        n = len(slots)
        if n:
            need = n * self.nl
            if need > self.kv_tag_capacity:
                self.kv_tag_capacity = 1 << (need - 1).bit_length()
                self.kv_tags = torch.empty((self.kv_tag_capacity, 8), dtype=torch.int64, device="cuda")
            target = self.kv_tags[:need].view(n, self.nl, 8)
            torch.index_select(self.sparse_tags, 0, slots_dev, out=target)
            tags = self.kv_tags[:need]
        else:
            tags = None
        st.commit(st.pass_m, tags, slots)

    def plan(self, md, bs: int, slots: torch.Tensor) -> tuple:
        """A step's gather: its sequences' blocks in order, the block table renumbered over them and its tokens' slots
        in the scratch (one sync a step)."""
        if self._plan is not None and self._plan[0] is md:
            return self._plan[1:]
        lens = md.seq_lens.cpu().tolist()
        bt = md.block_table.cpu()
        blocks, where_h, bt2 = [], {}, torch.zeros_like(bt)
        for i, n in enumerate(lens):
            nb = -(-n // bs)
            for j, block in enumerate(bt[i, :nb].tolist()):
                if block not in where_h:
                    where_h[block] = len(blocks)
                    blocks.append(block)
                bt2[i, j] = where_h[block]
        dev = torch.tensor(blocks, dtype=torch.int32, device="cuda")
        md2 = dataclasses.replace(md, block_table=bt2.to("cuda", non_blocking=True))
        where = torch.full((int(bt.max()) + 1 if bt.numel() else 1,), -1, dtype=torch.int64, device="cuda")
        where[dev.long()] = torch.arange(len(blocks), dtype=torch.int64, device="cuda")
        ok = slots >= 0
        s2 = torch.where(ok, where[torch.where(ok, slots // bs, 0)] * bs + slots % bs, -1)
        self._plan = (md, dev, md2, s2)
        return dev, md2, s2

    # the deferred append ---------------------------------------------------------------------------------------------
    def join(self) -> None:
        """Before a forward: the last step's appends are in the pages."""
        if self.outstanding:
            torch.cuda.current_stream().wait_event(self.app_done)
            self.outstanding = False

    def _launch(self, tab: torch.Tensor, layers: int, rows: int) -> None:
        main = torch.cuda.current_stream()
        self.fwd_done.record(main)
        self.s2.wait_event(self.fwd_done)
        P.append_multi(self.lib, torch, tab, layers, rows, self.lay, self.enc, self.st.k, self.s2.cuda_stream)
        self.app_done.record(self.s2)
        self.outstanding = True

    def flush_graph(self, rows: int) -> None:
        """After a captured step's replay at `rows` (padded) tokens: its staged rows, every layer's, into the pages."""
        if not self.staged:
            return
        if self.graph_tab is None:
            ents = [(self.stage_k[l], self.stage_v[l], self.stage_s[l], MAX_ROWS, c, base, layers, idx)
                    for l, (c, base, layers, idx) in sorted(self.staged.items())]
            self.graph_tab = P.table(torch, ents).cuda()
            self.graph_layers = len(ents)
        self._launch(self.graph_tab, self.graph_layers, rows)
        self.calls["flush_graph"] += 1

    def flush_eager(self) -> None:
        """After an eager forward: its rows, held until the second stream has read them."""
        if not self.pending:
            return
        rows = max(e[3] for e in self.pending)
        with torch.cuda.stream(self.s2):
            tab = P.table(torch, self.pending).to("cuda", non_blocking=False)
        self._launch(tab, len(self.pending), rows)
        for key, val, slots, *_ in self.pending:
            for t in (key, val, slots):
                t.record_stream(self.s2)
        tab.record_stream(self.s2)
        self.pending = []
        self.calls["flush_eager"] += 1

    # vLLM's hooks ------------------------------------------------------------------------------------------------
    def forward(self, orig):
        att = self

        def fn(impl, layer, query, key, value, kv_cache, attn_metadata, output, output_scale=None,
               output_block_scale=None):
            md = attn_metadata
            if md is None or getattr(md, "use_cascade", False) or impl.kv_cache_dtype != "auto":
                if md is not None and (md.use_cascade or impl.kv_cache_dtype != "auto"):
                    raise RuntimeError("the stack's attention takes a bf16 cache and no cascade")
                return orig(impl, layer, query, key, value, kv_cache, attn_metadata, output, output_scale,
                            output_block_scale)
            L = att.layer(layer, kv_cache)
            k2, v2 = key.view(key.shape[0], -1), value.view(value.shape[0], -1)
            if md.max_query_len == 1:
                b = md.seq_lens.shape[0]
                q2, o2 = query.view(query.shape[0], -1), output.view(output.shape[0], -1)
                P.attn(att.lib, torch, kv_cache, L.lay, att.st.words, att.nl, L.index, att.enc, att.st.k, q2, o2,
                       att.partials(b), md.seq_lens, md.block_table, b, att.split(b), nrows=o2.shape[0], ck=k2, cv=v2,
                       receipts=att.receipts())
                att.receipt_layer(L.index)
                att.calls["decode"] += 1
                return output
            slots = att.slots[att.index(layer)]
            blocks, md2, s2 = att.plan(md, L.lay.bs, slots)
            scratch = att.gathered(blocks.shape[0], kv_cache)
            P.gather(att.lib, torch, kv_cache, L.lay, att.st.words, att.nl, L.index, att.enc, att.st.k, blocks, scratch,
                     receipts=att.receipts())
            att.receipt_layer(L.index)
            lay2 = P.Layout(blocks.shape[0], L.lay.bs, L.lay.sb, L.lay.sh, L.lay.ss)
            P.append(att.lib, torch, scratch, lay2, 0, 1, 0, 0, 0, k2[:s2.shape[0]], v2[:s2.shape[0]], s2)
            att.calls["gathered"] += 1
            return orig(impl, layer, query, key, value, scratch, md2, output, output_scale, output_block_scale)
        return fn

    def update(self, orig):
        att = self

        def fn(impl, layer, key, value, kv_cache, slot_mapping):
            if slot_mapping.dtype != torch.int64:
                raise RuntimeError(f"slot mapping {slot_mapping.dtype}")
            L, idx, n = att.layer(layer, kv_cache), att.index(layer), slot_mapping.shape[0]
            att.slots[idx] = slot_mapping
            k2, v2 = key.view(key.shape[0], -1), value.view(value.shape[0], -1)
            if torch.cuda.is_current_stream_capturing():
                if n > MAX_ROWS:
                    raise RuntimeError(f"a captured step of {n} tokens: the stage holds {MAX_ROWS}")
                att.stage_k[idx, :n].copy_(k2[:n])
                att.stage_v[idx, :n].copy_(v2[:n])
                att.stage_s[idx, :n].copy_(slot_mapping)
                att.stage_s[idx, n:].fill_(-1)
                if idx not in att.staged or att.staged[idx][0].data_ptr() != kv_cache.data_ptr():
                    att.graph_tab = None
                att.staged[idx] = (kv_cache, att.st.words, att.nl, idx)
            else:
                att.pending.append((k2, v2, slot_mapping, n, kv_cache, att.st.words, att.nl, idx))
            att.calls["append"] += 1
        return fn


# --- the install -----------------------------------------------------------------------------------------------------

def rand_fp8(m: int, k: int, gen: torch.Generator) -> torch.Tensor:
    return (torch.randn((m, k), generator=gen, device="cuda") * 0.5).to(torch.float8_e4m3fn)


def rand_x(lin: StackLinear, m: int, gen: torch.Generator) -> torch.Tensor:
    """m rows of the linear's activations: FP8, or BF16 (the lm_head's)."""
    if lin.bf:
        return (torch.randn((m, lin.kd // 2), generator=gen, device="cuda") * 0.5).to(torch.bfloat16)
    return rand_fp8(m, lin.kd, gen)


def select(st: Stack, trials: int, log) -> list[dict]:
    """Each shape's config at each capture size, by CUDA-graph timing over all of the shape's linears in turn."""
    groups: dict[tuple, list[StackLinear]] = {}
    for lin in st.linears:
        groups.setdefault((lin.n, lin.kd), []).append(lin)
    gen = torch.Generator(device="cuda").manual_seed(7)
    cache: dict = {}
    out = []
    for (n, kd), lins in groups.items():
        for m in st.fused:
            x8 = rand_x(lins[0], m, gen)
            part = torch.empty((lins[0].nch, m, n), dtype=torch.float32, device="cuda")
            times = {}
            for cfg in lins[0].candidates(m):
                plan = lins[0].planned(m, cfg, cache)
                tags = lins[0].own_tags(plan)
                for lin in lins:
                    lin.launch(x8, part, plan, 0, tags)
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    for lin in lins:
                        lin.launch(x8, part, plan, 0, tags)
                g.replay()
                best = float("inf")
                for _ in range(trials):
                    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    e0.record()
                    g.replay()
                    g.replay()
                    e1.record()
                    torch.cuda.synchronize()
                    best = min(best, e0.elapsed_time(e1) / 2)
                del g
                times[tuple(cfg)] = best * 1e3 / len(lins)
            if not times:
                raise RuntimeError(f"no config fits n={n} k={kd} m={m}")
            best = min(times, key=times.get)
            for lin in lins:
                lin.cfgs[m] = best
            out.append({"n": n, "k": kd, "m": m, "layers": len(lins), "best": list(best),
                        "us": {"x".join(map(str, c)): round(t, 2) for c, t in times.items()}})
            log(f"select n={n} k={kd} m={m}: {best} {times[best]:.1f} us/layer")
    for lin in st.linears:
        for m in range(1, MAX_ROWS + 1):
            lin.plan[m] = [(r, lin.planned(r, lin.cfgs[min(c for c in st.fused if c >= r)], cache))
                           for r in pieces(m, st.rows)]
    for m in range(1, MAX_ROWS + 1):
        at = 0
        for lin in st.linears:
            lin.at[m] = at
            at += lin.receipts(m)
        st.nvec[m] = at
    torch.cuda.empty_cache()
    return out


def gate_linear(st: Stack, lin: StackLinear, w: torch.Tensor, plain: torch.Tensor, gen, gates: dict) -> None:
    """The store decodes to the plain fragments (wide.py on a few words), every candidate config's sums equal the plain
    kernel's at each capture size bit for bit, the reduced output is within FP32 rounding of the float product."""
    wt = st.wt
    if not torch.equal(unplain(plain, lin.n, lin.kd), w.view(torch.uint8)):
        raise RuntimeError(f"gate: {lin.name}: unplain(plain_layout(W)) is not W")
    if st.encoded:
        if not torch.equal(lin.fragments(), plain):
            raise RuntimeError(f"gate: {lin.name}: the store does not decode to the plain fragments")
        sb, pb = BW.host_bytes(lin.store.view(torch.uint8)), BW.host_bytes(plain)
        rng = random.Random(lin.idx)
        for i in sorted({0, lin.words - 1, *(rng.randrange(lin.words) for _ in range(2))}):
            x = int.from_bytes(pb[wt.s * i:wt.s * (i + 1)], "little")
            if BW.W.word_bytes(BW.W.encode_word(x, lin.base + i, st.key), wt.field) != sb[wt.s * i:wt.s * (i + 1)]:
                raise RuntimeError(f"gate: {lin.name}: word {lin.base + i} is not wide.py's")
            gates["python_words"] += 1
    ref_lib = PlainRef(st, lin, plain)
    cache: dict = {}
    for m in st.fused:
        x8 = rand_x(lin, m, gen)
        ref = ref_lib.sums(x8)
        for cfg in lin.candidates(m):
            got = lin.sums(x8, lin.planned(m, cfg, cache))
            if not torch.equal(got.view(torch.int32), ref.view(torch.int32)):
                d = got.view(torch.int32) != ref.view(torch.int32)
                raise RuntimeError(f"gate: {lin.name} m={m} cfg={cfg}: {int(d.sum())} of {d.numel()} partial sums "
                                   "differ from the plain kernel's")
            gates["configs_checked"] += 1
        if m == 4:
            exact = x8.float() @ w.float().t()
            err = (ref.sum(0) - exact).abs().max().item() / max(exact.abs().max().item(), 1e-30)
            gates["max_rel_err"] = max(gates["max_rel_err"], err)
            if err > 1e-3:
                raise RuntimeError(f"gate: {lin.name}: sums {err:.2e} off the float product")


class PlainRef:
    """wt_gemm over a linear's plain fragments at its first plain config (BF16's mode for the lm_head), in launches of
    at most PLAIN_ROWS rows: the gates' reference sums."""

    def __init__(self, st: Stack, lin: StackLinear, plain: torch.Tensor) -> None:
        self.st, self.lin, self.plain = st, lin, plain

    def sums(self, x8: torch.Tensor) -> torch.Tensor:
        st, lin, m = self.st, self.lin, x8.shape[0]
        if m > PLAIN_ROWS:
            r = pieces(m, PLAIN_ROWS)
            return torch.cat([self.sums(x) for x in x8.split(r)], dim=1)
        c, s = PLAIN_CFGS[0]
        mode = 3 if lin.bf else 0
        launched = ctypes.c_int(0)
        st.wt.call("wt_gemm", mode, None, None, None, m, lin.n, lin.kd, 0, c, s, 0, 0, None, ctypes.byref(launched),
                   stream())
        part = torch.empty((lin.nch, m, lin.n), dtype=torch.float32, device="cuda")
        st.wt.call("wt_gemm", mode, self.plain.data_ptr(), x8.data_ptr(), part.data_ptr(), m, lin.n, lin.kd, 1, c, s,
                   launched.value // lin.nch, 0, st.xr.data_ptr(), None, stream())
        return part


def gate_step(st: Stack, log) -> dict:
    """A decode step's receipts at each capture size (encoded): every linear at batch m after begin_pass, then the
    lm_head and its commit. The pass's seed on the device is pass_seed(σ, j, root_(j−1)) of the previous step's root as
    committed (the first: the registration's). At m = 32 every receipt of the whole step, and at the other sizes the
    edge linears, are sketch_receipt's of their stored blocks at that seed and their offsets. At GATE_TREE's batches
    the step's root is the driver's receipt tree over all of its receipts (`receipts.tree`, salts under STORE_KEY)."""
    if not st.encoded:
        return {}
    wt = st.wt
    gen = torch.Generator(device="cuda").manual_seed(31)
    nper = 4
    out = {}
    for m in CAPTURE:
        root_prev = RD.digest(st.root)
        j = st.j
        st.begin_pass(m)
        torch.cuda.synchronize()
        seed = BW.pass_seed(wt, j, root_prev)
        got = BW.pass_state(wt)
        x8 = {(lin.kd, lin.bf): rand_x(lin, m, gen) for lin in st.linears}
        xs = torch.ones(1, dtype=torch.float32, device="cuda")
        for lin in st.linears[:-1]:
            lin.forward(x8[lin.kd, lin.bf], xs)
        st.lm.run(x8[st.lm.lin.kd, True], True)
        st.commit(m)
        torch.cuda.synchronize()
        full = m == 32
        row = {"j": j, "seed": got["seed"] == seed, "receipts": st.nvec[m], "linears_checked": 0,
               "receipts_checked": 0, "all_receipts": full, "receipts_ok": True}
        nbits = BW.sketch_bits(wt, seed, 16, SKT.vector_bytes(wt.sb, T))
        checked = st.linears if full else st.linears[:nper] + st.linears[-nper:]
        for lin in checked:
            _, (_, _, _, nvec) = lin.plan[m][0]
            first = nvec if full else min(64, nvec)
            vs = lin.store.view(torch.uint8).view(-1, SR.BLOCK).view(torch.int8)[:first]   # store-order/v0
            sk = SKT.sketch_torch(torch, vs, nbits).cpu().numpy()
            del vs
            at = lin.at[m]
            want = b"".join(BW.root_words(y) for y in SR.receipts_of(seed, sk, wt.jbase + at))
            row["receipts_ok"] &= BW.host_bytes(st.tags.view(torch.uint8)[wt.tag_bytes * at:wt.tag_bytes * (at + first)]) == want
            row["linears_checked"] += 1
            row["receipts_checked"] += first
        small = min(st.linears, key=lambda x: x.nbytes)   # plan_of's map is 32 bytes a store byte
        _, (_, gr, _, _) = small.plan[m][0]
        first = small.base * wt.s // SR.BLOCK
        vecs = SR.gemm_vectors(small.n // 16, small.nch, wt.sb, gr) + small.base * wt.s
        # step_plan's premise for each linear: its launch's receipts are its store's blocks, in order
        row["plan_of"] = SR.rb.plan_of(vecs, st.words * wt.s // SR.BLOCK).runs == ((first, small.receipts(m)),)
        del vecs
        if m in GATE_TREE:
            n = st.nvec[m]
            plan = st.step_plan(m)
            ys = st.tags[:8 * n].view(n, 8).cpu().numpy()
            receipts = [RD.digest(y) for y in ys]
            leaves = -(-n // SR.rb.PER_LEAF)
            tree = SR.rb.tree(BW.RECEIPT_SEED, plan, receipts,
                              [RD.store_salt(BW.STORE_KEY, BW.RECEIPT_SEED, c) for c in range(leaves)])
            row["root"] = RD.digest(st.root) == tree.commitment.root
        row["ok"] = row["seed"] and row["receipts_ok"] and row["plan_of"] and row.get("root", True)
        out[m] = row
        log(f"step gate m={m}: {row}")
        if not row["ok"]:
            raise RuntimeError(f"gate: the step's receipts at m = {m}: {row}")
    root_prev, j = RD.digest(st.root), st.j
    st.begin_pass(1)
    torch.cuda.synchronize()
    out["chain"] = BW.pass_state(wt)["seed"] == BW.pass_seed(wt, j, root_prev)
    if not out["chain"]:
        raise RuntimeError("gate: the next pass's seed is not pass_seed of the last step's root")
    return out


def components(st: Stack, lm: LmHead, trials: int = 5) -> dict:
    """Each part of a decode step alone, in ms (the best of `trials` CUDA-event timings, after one warm launch): every
    FP8 linear at m (one CUDA graph, their receipts), the commit alone, the lm_head (its receipts, no commit), the
    pass's start."""
    def best(fn) -> float:
        fn()
        torch.cuda.synchronize()
        out = float("inf")
        for _ in range(trials):
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            fn()
            e1.record()
            torch.cuda.synchronize()
            out = min(out, e0.elapsed_time(e1))
        return out

    gen = torch.Generator(device="cuda").manual_seed(5)
    xs = torch.ones(1, dtype=torch.float32, device="cuda")
    out = {}
    for m in (1, 32):
        x8 = {lin.kd: rand_x(lin, m, gen) for lin in st.linears[:-1]}
        st.begin_pass(m)

        def linears():
            for lin in st.linears[:-1]:
                lin.forward(x8[lin.kd], xs)
        linears()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            linears()
        row = {"linears": best(g.replay)}
        del g
        if st.encoded:
            def commit():
                st.at = st.nvec[m]
                st.commit(m)
            row["commit"] = best(commit)
            row["begin_pass"] = best(lambda: st.begin_pass(m))
        h = torch.randn((m, lm.h), generator=gen, device="cuda").to(torch.bfloat16)
        row["lm_head"] = best(lambda: lm.run(h, False))
        out[m] = row
    return out


def install(runner, args, hooks, fp8_linears, unwrap, plaintext_scan, log) -> dict:
    """The arm's stack, in the runner's load_model: every FP8 linear, the lm_head, the attention and the KV append."""
    t0 = time.perf_counter()
    model = unwrap(runner.model)
    layers = fp8_linears(model)
    if not layers:
        raise RuntimeError("no FP8 linear layers")
    st = Stack(args, encoded=args.arm == "stack")
    torch.cuda.synchronize()
    alloc0 = torch.cuda.memory_allocated()
    gates = {"layers": 0, "configs_checked": 0, "python_words": 0, "max_rel_err": 0.0, "rows_checked": 0}
    gen = torch.Generator(device="cuda").manual_seed(1000)
    plain_bytes = encode_s = 0.0
    for name, layer in layers:
        w = layer.weight.t()
        n, kd = w.shape
        if not w.is_contiguous() or (n, kd) != (layer.output_size_per_partition, layer.input_size_per_partition):
            raise RuntimeError(f"{name}: weight {tuple(layer.weight.shape)} is not the (K, N) view of an N x K matrix")
        if layer.weight_scale.numel() != 1:
            raise RuntimeError(f"{name}: weight scale {tuple(layer.weight_scale.shape)} is not per tensor")
        torch.cuda.synchronize()
        te = time.perf_counter()
        lin = StackLinear(st, name, w, layer.weight_scale)
        torch.cuda.synchronize()
        encode_s += time.perf_counter() - te
        plain = BW.plain_layout(w.view(torch.uint8)) if st.encoded else lin.store
        gate_linear(st, lin, w, plain, gen, gates)
        del plain
        if st.encoded:
            rows = lin.rows()
            if not torch.equal(rows, w.view(torch.uint8)):
                raise RuntimeError(f"gate: {name}'s store does not decode, row major, to its weight")
            del rows
            gates["rows_checked"] += 1
        plain_bytes += w.numel()
        del w
        layer._parameters["weight"] = torch.nn.Parameter(
            torch.empty(0, dtype=torch.float8_e4m3fn, device="cuda"), requires_grad=False)
        K.LINEARS.append(lin)
        hooks.wrap(layer.quant_method, "apply",
                   lambda orig, fn=_apply(len(K.LINEARS) - 1, layer.quant_method.fp8_linear.quant_fp8): fn)
        gates["layers"] += 1
    st.scratch = torch.empty(max(lin.store.numel() * lin.store.element_size() for lin in st.linears) // 4,
                             dtype=torch.int32, device="cuda") if st.encoded else None
    head = model.lm_head
    w = head.weight
    lm = st.lm = LmHead(st, head)
    plain = BW.plain_layout(w.view(torch.uint8)) if st.encoded else lm.lin.store
    gate_linear(st, lm.lin, w, plain, gen, gates)
    del plain, w
    head._parameters["weight"] = torch.nn.Parameter(torch.empty(0, dtype=torch.bfloat16, device="cuda"),
                                                    requires_grad=False)
    hooks.wrap(head.quant_method, "apply", lambda orig: lm.apply)
    selection = select(st, args.select_trials, log)
    st.buffers()
    gates["step"] = gate_step(st, log)
    parts = components(st, lm)
    log(f"components (ms): {parts}")
    att = Attention(st, len(model.model.layers))
    from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl
    hooks.wrap(FlashAttentionImpl, "forward", att.forward)
    hooks.wrap(FlashAttentionImpl, "do_kv_cache_update", att.update)
    passes = hook_passes(st, att, hooks, model)
    import gc
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    alloc1 = torch.cuda.memory_allocated()
    scan = plaintext_scan(min(lin.n * lin.kd for lin in st.linears))
    if st.encoded and (scan["fp8"] or scan["uint8"]):
        raise RuntimeError(f"gate: plaintext left in the process: {scan}")
    store_bytes = sum(lin.nbytes for lin in st.linears if not lin.bf)
    st.attention = att
    wt = st.wt
    return {"stack": st, "record": {
        "arm": args.arm, "k": st.k, "wt_lib": Path(args.wt_lib).name, "paged_lib": Path(args.paged_lib).name,
        "codec": {"field": wt.field.name, "mask": wt.mask, "fold6": wt.fold6, "receipt": "sketch-sha512" if st.encoded
                  else None, "sketch_t": T, "kv_field": st.plib.field.name, "kv_fold6": st.plib.fold6,
                  "receipt_layout": SR.rb.STORE_ORDER if st.encoded else None, "weight_receipts_per_step": st.nvec,
                  "kv_receipt": "one store-order/v0 token-layer block per physical slot and layer" if st.encoded else None,
                  "lm_head": {"rows": lm.v, "fused": True, "receipted": bool(st.encoded),
                                                           "payload": "bf16", "bytes": lm.nbytes}},
        "layers": len(layers), "plain_bytes": plain_bytes, "store_bytes": store_bytes,
        "store_over_plain": store_bytes / plain_bytes, "words": st.words, "encode_s": encode_s,
        "allocated_before": alloc0, "allocated_after": alloc1, "allocated_delta": alloc1 - alloc0,
        "scratch_bytes": {"prefill_linear": st.scratch.numel() * 4 if st.scratch is not None else 0,
                          "lm_head": lm.scratch_bytes,
                          "attention_partials": att.part.numel() * 4,
                          "kv_stage": att.stage_bytes,
                          "receipts": st.tags.numel() * 8 + 64 * (st.dig.shape[0] + st.up.shape[0])},
        "plaintext_scan": scan, "gates": gates, "selection": selection, "passes_hooked": passes, "components_ms": parts,
        "fp8_kernel": type(layers[0][1].quant_method.fp8_linear).__name__,
        "activation_quant": str(layers[0][1].quant_method.activation_quant_key),
        "install_s": time.perf_counter() - t0}}


def _apply(idx: int, quant):
    def fn(layer, x, bias=None):
        if x.dtype != torch.bfloat16:
            raise TypeError(f"activations are {x.dtype}, the linear writes BF16")
        x8, xs = quant(x.reshape(-1, x.shape[-1]), None, None)
        y = torch.ops.ve_enc.linear(x8, xs, idx)
        if bias is not None:
            y = y + bias
        return y.reshape(*x.shape[:-1], y.shape[-1])
    return fn


def hook_passes(st: Stack, att: Attention, hooks, model) -> str:
    """vLLM's V2 runner: before each forward outside a capture the last step's appends (Attention.join) and, at most
    MAX_ROWS (padded) tokens, a pass; after it the step's appends: its full-graph replay's (`run_fullgraph`, its padded
    size desc.num_tokens; staged in the graph) or its eager forward's (the model's own)."""
    from vllm.v1.worker.gpu.cudagraph_utils import ModelCudaGraphManager

    def begin(m: int) -> bool:
        if torch.cuda.is_current_stream_capturing():
            return False
        att.join()
        if 0 < m <= MAX_ROWS:
            st.begin_pass(m)
        else:
            st.open = False
        return True

    def replay(orig):
        def fn(self, desc):
            begin(desc.num_tokens)
            out = orig(self, desc)
            att.flush_graph(desc.num_tokens)
            return out
        return fn

    def forward(orig):
        def fn(input_ids=None, positions=None, *a, **kw):
            ref = input_ids if input_ids is not None else positions
            live = begin(ref.shape[0] if ref is not None else 0)
            out = orig(input_ids, positions, *a, **kw)
            if live:
                att.flush_eager()
            return out
        return fn
    hooks.wrap(ModelCudaGraphManager, "run_fullgraph", replay)
    hooks.wrap(model, "forward", forward)
    return "v2 run_fullgraph + forward, appends deferred"


def summary(st: Stack) -> dict:
    att = st.attention
    return {"passes": st.passes, "j": st.j, "attention_calls": att.calls,
            "lm_head_graphs": sorted({key[0] for key in st.lm.graphs}),
            "kv_store_blocks": att.pool_slots * att.nl,
            "kv_store_bytes": att.pool_slots * att.nl * SR.BLOCK,
            "kv_receipt_buffer_bytes": sum(t.numel() * t.element_size() for t in
                                           (att.rc_sk, att.used, att.sparse_tags) if t is not None)}


__all__ = ["Attention", "LmHead", "Stack", "StackLinear", "install", "summary"]
