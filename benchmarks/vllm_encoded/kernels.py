"""The linears a served model runs: `ve.cu` through ctypes, one object per linear layer, holding either the plain FP8 fragments
(`PlainLinear`, the native arm's operand in the fused kernel's own order) or only the codec words (`EncodedLinear`).

vLLM calls a linear through the op ``ve_enc::linear(x8, xs, idx)`` (opaque to torch.compile), on activations its own FP8
kernel's quantization produced (dynamic, per token where CUTLASS FP8 is supported).  At M <= 32 rows (a decode batch) it runs
one of `wc.cu`'s three GEMM forms (the `Config` the arm picked for this shape and M) and reduces the chunks' sums in order to
BF16 (`ve_reduce`), so a plain and an encoded linear over the same weight return the same bits.  At more rows (prefill) it
decodes the layer into a scratch FP8 matrix for vLLM's CUTLASS GEMM: prefill is compute-bound and isn't what's timed.

Kernels are pluggable: a `Lib` is any build of `ve.cu` (packing, k range), and a later fused kernel (a tag stage, a wider
codec) is another `Lib` whose `ve_gemm` takes the same arguments.  `WideLinear` runs ra-codec-wide's fused GEMM instead
(`benchmarks/word_codec/wide.cu`, a `WideLib` a field: Sloth over p = 2^127 − 1, 2^255 − 19 or 2^521 − 1, and with a tag
seed set the bandwidth audit's ro-blake3/v1 tag of every block of encoded words it reads); its chunk sums have the same
layout, so `ve_reduce` turns them into the same BF16.
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
MAX_ROWS = 32
FORMS = ("simple", "pipe", "ws", "wide")


def _load(name: str, file: str):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, WORD_CODEC / file)
        mod = sys.modules[name] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return sys.modules[name]


def _codec():
    return _load("wc_codec", "codec.py")


def _wide():
    return _load("wc_wide", "wide.py")


def stream() -> int:
    return torch.cuda.current_stream().cuda_stream


class Lib:
    """One build of `ve.cu`: its packing (`bits`) and largest k (`kmax`)."""

    def __init__(self, path: str | Path) -> None:
        v, i, q = ctypes.c_void_p, ctypes.c_int, ctypes.c_uint64
        self.path = str(path)
        so = self.so = ctypes.CDLL(self.path)
        so.wc_set_key.argtypes = [v, i, q]
        so.wc_encode.argtypes = [v, v, q, q, i, i, v]
        so.wc_decode.argtypes = [v, v, q, q, i, v, i, v]
        so.ve_prepare.argtypes = [i, i]
        so.ve_occupancy.argtypes = [i, i, i, i, i]
        so.ve_gemm.argtypes = [i, i, v, v, v, i, i, i, q, i, i, i, i, v]
        so.ve_reduce.argtypes = [v, i, i, i, v, i, v, v, v]
        so.ve_constants.argtypes = [v]
        so.wc_spin.argtypes = [ctypes.c_longlong, v]
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
    """One build of `wide.cu` (one field, `WW_BITS`): its key, its tag seed and the tag buffer its GEMMs write."""

    def __init__(self, path: str | Path) -> None:
        v, i, q = ctypes.c_void_p, ctypes.c_int, ctypes.c_uint64
        self.path = str(path)
        so = self.so = ctypes.CDLL(self.path)
        so.ww_constants.argtypes = [v]
        so.ww_set_key.argtypes = [v, i, v, v, v, v]
        so.ww_decode.argtypes = [v, v, q, q, i, v, i, v]
        so.ww_encode.argtypes = [v, v, q, q, i, i, v]
        so.ww_gemm.argtypes = [i, v, v, v, i, i, i, q, i, v, v, i, v]
        so.ww_gemm_occupancy.argtypes = [i, i, i, i]
        cs = (ctypes.c_int * 10)()
        so.ww_constants(cs)
        self.constants = list(cs)
        self.bits, self.limbs, self.payload, self.lane_words, self.kmax = (cs[0], cs[1], cs[2], cs[3], cs[7])
        self.field = _wide().field(self.bits)
        f = self.field
        if ((self.limbs, self.payload, self.lane_words, cs[4], cs[5]) !=
                (f.limbs, f.payload, layout.wide_lane_words(f.payload), layout.LANES, layout.TASK_TILES)):
            raise RuntimeError(f"{path}: constants {self.constants} are not wide.py's and layout.py's")
        self.sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
        self.key = None
        self.tag_seed: bytes | None = None
        self._seed = None
        self.tags: torch.Tensor | None = None

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

    def set_tag(self, seed: bytes | None) -> None:
        """Tag every block the GEMMs read under ro-blake3/v1 with this window seed (None: no tag)."""
        if seed is not None and len(seed) != 32:
            raise ValueError("a tag seed is 32 bytes")
        self.tag_seed, self._seed = seed, None if seed is None else ctypes.create_string_buffer(seed, 32)

    def tag_buffer(self, blocks: int) -> torch.Tensor:
        """The tags of one call's blocks (32 bytes a block), shared by every linear as the step's partial sums are."""
        if self.tags is None or self.tags.numel() < 8 * blocks:
            self.tags = torch.empty(8 * blocks, dtype=torch.int32, device="cuda")
        return self.tags


@dataclass(frozen=True)
class Config:
    """One GEMM form and its launch: `groups` CTA groups a chunk for the persistent forms (0: as many as fit), the
    warp-specialized form's consumer warps and ring stages, and the wide GEMM's L2 prefetch distance in CTAs."""

    form: int
    groups: int = 0
    consumers: int = 0
    stages: int = 0
    ahead: int = 0

    @property
    def name(self) -> str:
        if self.form == 3:
            return "wide" + (f"-ahead{self.ahead}" if self.ahead else "")
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
    With the lib's tag seed set, each GEMM also writes the ro-blake3/v1 tag of block j = base / L + 32·task + lane (that
    lane's L encoded words) to the lib's tag buffer.  `lib` (a `ve.cu` build) only reduces the chunk sums."""

    def __init__(self, lib: Lib, wlib: WideLib, w: torch.Tensor, weight_scale: torch.Tensor, k: int, base: int,
                 grid: int = 0) -> None:
        super().__init__(lib, *w.shape, weight_scale)
        if k and (wlib.key is None or wlib.key.rounds != k or not 1 <= k <= wlib.kmax):
            raise ValueError(f"the wide lib's key is not set for k = {k}")
        if base % (layout.LANES * wlib.lane_words):
            raise ValueError("a wide store starts at a multiple of 32·L words")
        self.wlib, self.k, self.base, self.bits = wlib, k, base, wlib.payload
        self.words = layout.wide_store_words(self.n, self.kd, wlib.payload)
        self.blocks = self.words // wlib.lane_words
        payloads = layout.wide_payloads(w, wlib.payload, wlib.limbs)
        if k:
            self.store = torch.empty_like(payloads)
            wlib.call("ww_encode", payloads.data_ptr(), self.store.data_ptr(), self.words, base, k, grid or 4 * wlib.sms,
                      stream())
        else:
            self.store = payloads
        self.payloads = payloads

    @property
    def tag_block_base(self) -> int:
        return self.base // self.wlib.lane_words

    def default(self) -> Config:
        return Config(3)

    def candidates(self, m: int) -> list[Config]:
        occ = self.wlib.so.ww_gemm_occupancy(self.k, m, int(self.wlib.tag_seed is not None), 1)
        return [Config(3)] + ([Config(3, ahead=occ * self.wlib.sms)] if occ > 0 else [])

    def _launch(self, cfg: Config, x8: torch.Tensor, part: torch.Tensor, partials: bool) -> None:
        if cfg.form != 3:
            raise ValueError(f"the wide GEMM has one form, not {cfg.name}")
        w = self.wlib
        tags = w.tag_buffer(self.blocks).data_ptr() if w.tag_seed is not None else None
        w.call("ww_gemm", self.k, self.store.data_ptr(), x8.data_ptr(), part.data_ptr(), x8.shape[0], self.n, self.kd,
               self.base, int(partials), w._seed, tags, cfg.ahead, stream())

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


#: the linears ``ve_enc::linear`` dispatches to, by index
LINEARS: list[Linear] = []


@torch.library.custom_op("ve_enc::linear", mutates_args=())
def linear_op(x8: torch.Tensor, xs: torch.Tensor, idx: int) -> torch.Tensor:
    return LINEARS[idx].forward(x8, xs)


@linear_op.register_fake
def _(x8: torch.Tensor, xs: torch.Tensor, idx: int) -> torch.Tensor:
    return x8.new_empty((x8.shape[0], LINEARS[idx].n), dtype=torch.bfloat16)


__all__ = ["FORMS", "LINEARS", "MAX_ROWS", "Config", "EncodedLinear", "Lib", "Linear", "PlainLinear", "WideLib",
           "WideLinear", "candidates", "linear_op", "stream"]
