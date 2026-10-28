"""The linears a served model runs: `ve.cu` through ctypes, one object per linear layer, holding either the plain FP8 fragments
(`PlainLinear`, the native arm's operand in the fused kernel's own order) or only the codec words (`EncodedLinear`).

vLLM calls a linear through the op ``ve_enc::linear(x8, xs, idx)`` (opaque to torch.compile), on activations its own FP8
kernel's quantization produced (dynamic, per token where CUTLASS FP8 is supported).  At M <= 32 rows (a decode batch) it runs
one of `wc.cu`'s three GEMM forms (the `Config` the arm picked for this shape and M) and reduces the chunks' sums in order to
BF16 (`ve_reduce`), so a plain and an encoded linear over the same weight return the same bits.  At more rows (prefill) it
decodes the layer into a scratch FP8 matrix for vLLM's CUTLASS GEMM: prefill is compute-bound and isn't what's timed.

Kernels are pluggable: a `Lib` is any build of `ve.cu` (packing, k range), and a later fused kernel (a wider codec) is
another `Lib` whose `ve_gemm` takes the same arguments.  `WideLinear` runs the wide codec's fused GEMM instead
(`benchmarks/word_codec/wide.cu`, a `WideLib` a field: Sloth over p = 2^127 − 1, 2^255 − 19, 2^521 − 1, 2^256 − 189 or
2^512 − 569); its chunk sums have the same layout, so `ve_reduce` turns them into the same BF16.  `TensorLinear` runs
the tensor-core codec's (`benchmarks/codec_tensor/wt.cu`, a `TensorLib` a field: decoder warps decoding in shared
memory ahead of one MMA warp), with the same chunk sums.
"""
from __future__ import annotations

import ctypes
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import layout
import torch

HERE = Path(__file__).resolve().parent
WORD_CODEC = HERE.parent / "word_codec"
CODEC_TENSOR = HERE.parent / "codec_tensor"
MAX_ROWS = 32
FORMS = ("simple", "pipe", "ws", "wide", "tensor")
SMEM_PER_BLOCK = 101376  # sm_120's opt-in dynamic shared memory a CTA


def _load(name: str, file: str, where: Path = WORD_CODEC):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, where / file)
        mod = sys.modules[name] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return sys.modules[name]


def _codec():
    return _load("wc_codec", "codec.py")


def _cuda_lib():
    return _load("wc_cuda_lib", "cuda_lib.py")


def _wide():
    return _load("wc_wide", "wide.py")


def _ct():
    return _load("ct_wide", "wide.py", CODEC_TENSOR)


def stream() -> int:
    return torch.cuda.current_stream().cuda_stream


_v, _i, _q = ctypes.c_void_p, ctypes.c_int, ctypes.c_uint64
VE = {   # ve.cu's functions (`cuda_lib.load`)
    "wc_set_key": [_v, _i, _q],
    "wc_encode": [_v, _v, _q, _q, _i, _i, _v],
    "wc_decode": [_v, _v, _q, _q, _i, _v, _i, _v],
    "ve_prepare": [_i, _i],
    "ve_occupancy": [_i, _i, _i, _i, _i],
    "ve_gemm": [_i, _i, _v, _v, _v, _i, _i, _i, _q, _i, _i, _i, _i, _v],
    "ve_reduce": [_v, _i, _i, _i, _v, _i, _v, _v, _v],
    "ve_constants": [_v],
    "wc_spin": [ctypes.c_longlong, _v],
}
WIDE = {   # wide.cu's that `WideLib` calls
    "ww_constants": [_v],
    "ww_set_key": [_v, _i, _v, _v, _v, _v],
    "ww_decode": [_v, _v, _q, _q, _i, _v, _i, _v],
    "ww_encode": [_v, _v, _q, _q, _i, _i, _v],
    "ww_gemm": [_i, _v, _v, _v, _i, _i, _i, _q, _i, _v, _v, _i, _v],
    "ww_gemm_occupancy": [_i, _i, _i, _i],
}
TENSOR = {   # wt.cu's that `TensorLib` calls
    "wt_constants": [_v],
    "wt_set_key": [_v, _i, _v, _v],
    "wt_encode": [_v, _v, _q, _q, _i, _i, _v],
    "wt_decode": [_v, _v, _q, _q, _i, _v, _i, _v],
    "wt_fused": [_i, _v, _v, _v, _i, _i, _i, _q, _i, _i, _i, _i, _i, _v, _v, _q, _v, _v],
}


class Lib:
    """One build of `ve.cu`: its packing (`bits`) and largest k (`kmax`)."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        so = self.so = _cuda_lib().load(self.path, VE)
        cs = (ctypes.c_int * 8)()
        so.ve_constants(cs)
        self.constants = list(cs)
        self.bits, self.kmax = self.constants[6], self.constants[7]
        lw = layout.lane_words(self.bits)
        if self.constants[:5] != [layout.LANES, layout.TASK_TILES, lw, lw * layout.LANES, layout.TASK_COLS]:
            raise RuntimeError(f"{path}: layout constants {self.constants} are not layout.py's")
        self.sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
        self.key = None

    def call(self, fn: str, *args) -> None:
        rc = getattr(self.so, fn)(*args)
        if rc:
            raise RuntimeError(f"{Path(self.path).name}: {fn} error {rc}")

    def set_key(self, seed: bytes, k: int):
        key = _codec().key(seed, k)
        c = (ctypes.c_uint64 * k)(*key.c)
        self.call("wc_set_key", c, k, key.tau)
        self.key = key
        return key


class WideLib:
    """One build of `wide.cu` (one field, `WW_BITS`): its key."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        so = self.so = _cuda_lib().load(self.path, WIDE)
        cs = (ctypes.c_int * 32)()
        so.ww_constants(cs)
        self.constants = list(cs[:21])
        self.bits, self.limbs, self.payload, self.lane_words, self.kmax = (cs[0], cs[1], cs[2], cs[3], cs[7])
        self.field = _wide().field(self.bits)
        f = self.field
        if ((self.limbs, self.payload, self.lane_words, cs[4], cs[5]) !=
                (f.limbs, f.payload, layout.wide_lane_words(f.payload), layout.LANES, layout.TASK_TILES)):
            raise RuntimeError(f"{path}: constants {self.constants} are not wide.py's and layout.py's")
        self.sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
        self.key = None

    def call(self, fn: str, *args) -> None:
        rc = getattr(self.so, fn)(*args)
        if rc:
            raise RuntimeError(f"{Path(self.path).name}: {fn} error {rc}")

    def set_key(self, seed: bytes, k: int):
        wide, f = _wide(), self.field
        key = wide.key(f, seed, k)

        def arr(vals):
            return (ctypes.c_uint32 * len(vals))(*vals)

        c, tau = arr([x for cj in key.c for x in wide.limbs(cj, f)]), arr(wide.limbs(key.tau, f))
        if f.three_mod_four:
            self.call("ww_set_key", c, k, tau, None, None, None)
        else:
            i_, c1 = wide.SQRT_M1, wide.C1
            self.call("ww_set_key", c, k, tau, arr(wide.limbs(i_, f)), arr(wide.limbs(c1, f)),
                      arr(wide.limbs(c1 * i_ % f.p, f)))
        self.key = key
        return key


class TensorLib:
    """One build of the tensor-core codec's `wt.cu` (one field, `WT_FIELD`): its key and each launch's tweak steps."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        so = self.so = _cuda_lib().load(self.path, TENSOR)
        cs = (ctypes.c_int * 15)()
        so.wt_constants(cs)
        self.constants = list(cs)
        (self.bits, self.limbs, self.word_bytes, self.payload_bytes, self.task_words, self.kmax, self.sb,
         self.fused_warps, self.fused_span) = (cs[0], cs[1], cs[2], cs[3], cs[4], cs[10], cs[11], cs[12], cs[13])
        ct = _ct()
        self.field = next(f for f in ct.FIELDS.values() if f.n == self.bits)
        f = self.field
        if ((self.word_bytes, self.payload_bytes, self.task_words, cs[6], self.sb) !=
                (f.word_bytes, f.payload_bytes, ct.TASK_WORDS[f.name], f.c, ct.TASK_WORDS[f.name] * f.word_bytes)):
            raise RuntimeError(f"{path}: constants {self.constants} are not codec_tensor/wide.py's {f}")
        self.sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
        self.key = None
        self._steps: dict[tuple, tuple[int, torch.Tensor]] = {}

    def call(self, fn: str, *args) -> None:
        rc = getattr(self.so, fn)(*args)
        if rc:
            raise RuntimeError(f"{Path(self.path).name}: {fn} error {rc}")

    def _limbs(self, x: int) -> bytes:
        return x.to_bytes(4 * self.limbs, "little")

    def set_key(self, seed: bytes, k: int):
        ct = _ct()
        key = ct.key(self.field.name, seed, k)
        c = ctypes.create_string_buffer(b"".join(self._limbs(x) for x in key.c))
        tau, p = ctypes.create_string_buffer(self._limbs(key.tau)), ctypes.create_string_buffer(self._limbs(self.field.p))
        self.call("wt_set_key", c, k, tau, p)
        self.key, self._steps = key, {}
        return key

    def configs(self) -> list[tuple[int, int]]:
        """(decoders, stages) as codec_tensor's `fused_cfgs("auto")` picks them: the most decoder warps the field's
        registers allow at 4 and 6 stages fewer than fit at M = 32."""
        smax = (SMEM_PER_BLOCK - 4 * 8 * 528 - 32) // (self.sb + 40)
        dmax = self.fused_warps - 2
        need = -(-32 * dmax // self.task_words)
        return list(dict.fromkeys([(dmax, max(need, smax - 4)), (dmax, max(need, smax - 6))]))

    def fused(self, k: int, w, x, y, m: int, n: int, kd: int, base: int, partials: int, cfg: Config, groups: int,
              tsteps, launched=None) -> None:
        self.call("wt_fused", k, w, x, y, m, n, kd, base, partials, cfg.consumers, cfg.stages, groups, 0, tsteps, None,
                  0, launched, stream())

    def steps(self, m: int, n: int, kd: int, cfg: Config) -> tuple[int, torch.Tensor]:
        """The launch's CTA groups (as many as fill the device) and its tweak steps on the device (`bench.tsteps`): a
        lane's next unit is 32·D words on, its task A or A + 1 on, consecutive tasks groups·nchunks apart."""
        nb8 = 1 if m <= 8 else 2 if m <= 16 else 4
        at = (nb8, n, kd, cfg)
        if at not in self._steps:
            launched = ctypes.c_int(0)
            self.fused(self.key.rounds, None, None, None, m, n, kd, 0, 0, cfg, 0, None, ctypes.byref(launched))
            nch = kd // layout.TASK_COLS
            groups = launched.value // nch
            adv, rem = divmod(32 * cfg.consumers, self.task_words)
            gq = groups * nch
            d = (adv * gq * self.task_words + rem, (adv + 1) * gq * self.task_words + rem - self.task_words)
            st = b"".join(self._limbs(self.key.tau * x % self.field.p) for x in d)
            self._steps[at] = (groups, torch.frombuffer(bytearray(st), dtype=torch.uint8).cuda())
        return self._steps[at]


@dataclass(frozen=True)
class Config:
    """One GEMM form and its launch: `groups` CTA groups a chunk for the persistent forms (0: as many as fit), the
    warp-specialized form's consumer warps and ring stages (the tensor GEMM's decoder warps and stages), and the wide
    GEMM's L2 prefetch distance in CTAs."""

    form: int
    groups: int = 0
    consumers: int = 0
    stages: int = 0
    ahead: int = 0

    @property
    def name(self) -> str:
        if self.form == 3:
            return "wide" + (f"-ahead{self.ahead}" if self.ahead else "")
        if self.form == 4:
            return f"tensor-d{self.consumers}s{self.stages}"
        return FORMS[self.form] + (f"-{self.consumers}x{self.stages}" if self.form == 2 else "")


#: the warp-specialized form's (consumers, stages), equal: `wc.cu`'s consumers take tasks i = c, c + C, ... from stage i mod S
#: and wait on its parity alone, which is unambiguous only when each stage has one consumer
WS = ((2, 2), (3, 3), (4, 4), (6, 6), (8, 8))


def candidates(lib: Lib, k: int, m: int, nchunks: int) -> list[Config]:
    out = [Config(0)]
    per = lib.so.ve_occupancy(1, k, m, 0, 0)
    if per > 0:
        out.append(Config(1, max(1, lib.sms * per // nchunks)))
    for c, s in WS:
        lib.call("ve_prepare", k, s)
        per = lib.so.ve_occupancy(2, k, m, c, s)
        if per > 0:
            out.append(Config(2, max(1, lib.sms * per // nchunks), c, s))
    return out


class Linear:
    """One layer's operand on the device.  `k` is the codec's rounds (0: plain fragments)."""

    k = 0

    def __init__(self, lib: Lib, n: int, kd: int, weight_scale: torch.Tensor) -> None:
        layout.check_shape(n, kd)
        self.lib, self.n, self.kd, self.ws = lib, n, kd, weight_scale.reshape(1).float()
        self.nchunks = kd // layout.TASK_COLS
        self.configs: dict[int, Config] = {}
        self.base = 0
        self.store: torch.Tensor

    @property
    def nbytes(self) -> int:
        return self.store.numel() * self.store.element_size()

    def default(self) -> Config:
        return Config(0)

    def candidates(self, m: int) -> list[Config]:
        return candidates(self.lib, self.k, m, self.nchunks)

    def config(self, m: int) -> Config:
        for mm in sorted(self.configs):
            if mm >= m:
                return self.configs[mm]
        return self.default()

    def gemm(self, x8: torch.Tensor, cfg: Config | None = None, partials: bool = True,
             part: torch.Tensor | None = None) -> torch.Tensor:
        m = x8.shape[0]
        cfg = cfg or self.config(m)
        shape = (self.nchunks if partials else 1, m, self.n)
        if part is None:
            part = (torch.empty if partials else torch.zeros)(shape, dtype=torch.float32, device=x8.device)
        elif tuple(part.shape) != shape or part.dtype != torch.float32 or not part.is_contiguous() or not partials:
            raise ValueError(f"partials buffer {tuple(part.shape)} {part.dtype} for {shape}")
        self._launch(cfg, x8, part, partials)
        return part

    def _launch(self, cfg: Config, x8: torch.Tensor, part: torch.Tensor, partials: bool) -> None:
        self.lib.call("ve_gemm", cfg.form, self.k, self.store.data_ptr(), x8.data_ptr(), part.data_ptr(), x8.shape[0], self.n,
                      self.kd, self.base, int(partials), cfg.groups, cfg.consumers, cfg.stages, stream())

    def reduce(self, part: torch.Tensor, xs: torch.Tensor) -> torch.Tensor:
        """BF16 out of the chunks' sums; ``xs`` is one activation scale or one a row (FP32, contiguous)."""
        m = part.shape[1]
        if xs.numel() not in (1, m) or xs.dtype != torch.float32 or not xs.is_contiguous():
            raise ValueError(f"activation scale {tuple(xs.shape)} {xs.dtype} for {m} rows")
        out = torch.empty((m, self.n), dtype=torch.bfloat16, device=part.device)
        self.lib.call("ve_reduce", part.data_ptr(), part.shape[0], m, self.n, xs.data_ptr(), int(xs.numel() > 1),
                      self.ws.data_ptr(), out.data_ptr(), stream())
        return out

    def dense(self) -> torch.Tensor:
        """W (n × k, FP8) rebuilt from the operand (prefill's scratch copy)."""
        raise NotImplementedError

    def forward(self, x8: torch.Tensor, xs: torch.Tensor) -> torch.Tensor:
        """BF16 x W^T from FP8 rows ``x8`` and their activation scale ``xs`` (one, or one a row)."""
        if x8.shape[0] > MAX_ROWS:
            from vllm import _custom_ops as ops
            return ops.cutlass_scaled_mm(x8, self.dense().t(), out_dtype=torch.bfloat16, scale_a=xs, scale_b=self.ws, bias=None)
        return self.reduce(self.gemm(x8.contiguous()), xs.reshape(-1).float().contiguous())


class PlainLinear(Linear):
    def __init__(self, lib: Lib, w: torch.Tensor, weight_scale: torch.Tensor) -> None:
        super().__init__(lib, *w.shape, weight_scale)
        self.store = layout.plain_fragments(w)

    def dense(self) -> torch.Tensor:
        return layout.unplain(self.store, self.n, self.kd).view(torch.float8_e4m3fn)


class EncodedLinear(Linear):
    """Only the codec words: `store` holds W's payloads encoded under the lib's key at word indices base, base + 1, ..."""

    def __init__(self, lib: Lib, w: torch.Tensor, weight_scale: torch.Tensor, k: int, base: int, grid: int = 0) -> None:
        super().__init__(lib, *w.shape, weight_scale)
        if lib.key is None or lib.key.rounds != k or not 1 <= k <= lib.kmax:
            raise ValueError(f"the lib's key is not set for k = {k}")
        self.k, self.base, self.bits = k, base, lib.bits
        payloads = layout.payloads(w, self.bits)
        self.store = torch.empty_like(payloads)
        lib.call("wc_encode", payloads.data_ptr(), self.store.data_ptr(), payloads.numel(), base, k,
                 grid or 4 * lib.sms * 8, stream())
        self.payloads = payloads

    def decoded(self) -> torch.Tensor:
        out = torch.empty_like(self.store)
        xr = torch.zeros(1, dtype=torch.int64, device=self.store.device)
        self.lib.call("wc_decode", self.store.data_ptr(), out.data_ptr(), self.store.numel(), self.base, self.k,
                      xr.data_ptr(), 4 * self.lib.sms * 8, stream())
        return out

    def dense(self) -> torch.Tensor:
        return layout.unpayload(self.decoded(), self.n, self.kd, self.bits).view(torch.float8_e4m3fn)

    def drop_payloads(self) -> None:
        self.payloads = None


class WideLinear(Linear):
    """Only the wide field's words: `store` (int32) holds W's payloads (`layout.wide_payloads`) encoded k rounds under the
    wide lib's key at word indices base, base + 1, ... (k = 0: the payloads themselves, the wide kernel's plain operand).
    `lib` (a `ve.cu` build) only reduces the chunk sums."""

    def __init__(self, lib: Lib, wlib: WideLib, w: torch.Tensor, weight_scale: torch.Tensor, k: int, base: int,
                 grid: int = 0) -> None:
        super().__init__(lib, *w.shape, weight_scale)
        if k and (wlib.key is None or wlib.key.rounds != k or not 1 <= k <= wlib.kmax):
            raise ValueError(f"the wide lib's key is not set for k = {k}")
        if base % (layout.LANES * wlib.lane_words):
            raise ValueError("a wide store starts at a multiple of 32·L words")
        self.wlib, self.k, self.base, self.bits = wlib, k, base, wlib.payload
        self.words = layout.wide_store_words(self.n, self.kd, wlib.payload)
        payloads = layout.wide_payloads(w, wlib.payload, wlib.limbs)
        if k:
            self.store = torch.empty_like(payloads)
            wlib.call("ww_encode", payloads.data_ptr(), self.store.data_ptr(), self.words, base, k, grid or 4 * wlib.sms,
                      stream())
        else:
            self.store = payloads
        self.payloads = payloads

    def default(self) -> Config:
        return Config(3)

    def candidates(self, m: int) -> list[Config]:
        w = self.wlib
        occ = w.so.ww_gemm_occupancy(self.k, m, 0, 1)
        return [Config(3)] + ([Config(3, ahead=occ * w.sms)] if occ > 0 else [])

    def _launch(self, cfg: Config, x8: torch.Tensor, part: torch.Tensor, partials: bool) -> None:
        if cfg.form != 3:
            raise ValueError(f"the wide GEMM has one form, not {cfg.name}")
        self.wlib.call("ww_gemm", self.k, self.store.data_ptr(), x8.data_ptr(), part.data_ptr(), x8.shape[0], self.n,
                       self.kd, self.base, int(partials), None, None, cfg.ahead, stream())

    def decoded(self) -> torch.Tensor:
        if not self.k:
            return self.store
        out = torch.empty_like(self.store)
        xr = torch.zeros(1, dtype=torch.int64, device=self.store.device)
        self.wlib.call("ww_decode", self.store.data_ptr(), out.data_ptr(), self.words, self.base, self.k, xr.data_ptr(),
                       4 * self.wlib.sms, stream())
        return out

    def dense(self) -> torch.Tensor:
        return layout.wide_unpayload(self.decoded(), self.n, self.kd, self.wlib.payload, self.wlib.limbs).view(
            torch.float8_e4m3fn)

    def drop_payloads(self) -> None:
        self.payloads = None


class TensorLinear(Linear):
    """Only `wt.cu`'s words (the tensor-core codec's fused decode GEMM): `store` (S bytes a word, held as int16 so no uint8
    tensor of a linear's size is in the process: `bytes` is its uint8 view) holds W's payloads (`layout.tensor_limbs`)
    encoded k >= 1 rounds under the tensor lib's key at word indices base, base + 1, ...  `lib` (a `ve.cu` build) only
    reduces the chunk sums."""

    def __init__(self, lib: Lib, tlib: TensorLib, w: torch.Tensor, weight_scale: torch.Tensor, k: int, base: int,
                 grid: int = 0) -> None:
        super().__init__(lib, *w.shape, weight_scale)
        if tlib.key is None or tlib.key.rounds != k or not 1 <= k <= tlib.kmax:
            raise ValueError(f"the tensor lib's key is not set for k = {k}")
        self.tlib, self.k, self.base, self.bits = tlib, k, base, 8 * tlib.payload_bytes
        payloads = layout.tensor_limbs(w, tlib.task_words, tlib.payload_bytes, tlib.limbs)
        self.words = payloads.numel() // tlib.limbs
        if tlib.word_bytes % 2:
            raise ValueError(f"{tlib.field.name}: a word of {tlib.word_bytes} bytes is not whole int16s")
        self.store = torch.empty(self.words * tlib.word_bytes // 2, dtype=torch.int16, device=w.device)
        tlib.call("wt_encode", payloads.data_ptr(), self.store.data_ptr(), self.words, base, k, grid or 4 * tlib.sms,
                  stream())
        self.payloads = payloads

    @property
    def bytes(self) -> torch.Tensor:
        return self.store.view(torch.uint8)

    def default(self) -> Config:
        return self.candidates(1)[0]

    def candidates(self, m: int) -> list[Config]:
        return [Config(4, consumers=d, stages=s) for d, s in self.tlib.configs()]

    def _launch(self, cfg: Config, x8: torch.Tensor, part: torch.Tensor, partials: bool) -> None:
        if cfg.form != 4:
            raise ValueError(f"the tensor GEMM has one form, not {cfg.name}")
        t = self.tlib
        groups, steps = t.steps(x8.shape[0], self.n, self.kd, cfg)
        t.fused(self.k, self.store.data_ptr(), x8.data_ptr(), part.data_ptr(), x8.shape[0], self.n, self.kd, self.base,
                int(partials), cfg, groups, steps.data_ptr())

    def decoded(self) -> torch.Tensor:
        t = self.tlib
        out = torch.empty(self.words * t.limbs, dtype=torch.int32, device=self.store.device)
        xr = torch.zeros(1, dtype=torch.int64, device=self.store.device)
        t.call("wt_decode", self.store.data_ptr(), out.data_ptr(), self.words, self.base, self.k, xr.data_ptr(),
               4 * t.sms, stream())
        return out

    def dense(self) -> torch.Tensor:
        t = self.tlib
        return layout.tensor_unlimbs(self.decoded(), self.n, self.kd, t.task_words, t.payload_bytes, t.limbs).view(
            torch.float8_e4m3fn)

    def drop_payloads(self) -> None:
        self.payloads = None


#: the linears ``ve_enc::linear`` dispatches to, by index
LINEARS: list[Linear] = []


@torch.library.custom_op("ve_enc::linear", mutates_args=())
def linear_op(x8: torch.Tensor, xs: torch.Tensor, idx: int) -> torch.Tensor:
    return LINEARS[idx].forward(x8, xs)


@linear_op.register_fake
def _(x8: torch.Tensor, xs: torch.Tensor, idx: int) -> torch.Tensor:
    return x8.new_empty((x8.shape[0], LINEARS[idx].n), dtype=torch.bfloat16)


__all__ = ["FORMS", "LINEARS", "MAX_ROWS", "Config", "EncodedLinear", "Lib", "Linear", "PlainLinear", "TensorLib",
           "TensorLinear", "WideLib", "WideLinear", "candidates", "linear_op", "stream"]
