"""The linears a served model runs: `ve.cu` through ctypes, one object per linear layer, holding either the plain FP8 fragments
(`PlainLinear`, the native arm's operand in the fused kernel's own order) or only the codec words (`EncodedLinear`).

A call at M <= 32 rows (a decode batch) quantizes x to FP8 with the activation quantization of vLLM's own FP8 kernel (dynamic,
per token where CUTLASS FP8 is supported), runs one of `wc.cu`'s three GEMM forms (the `Config` the arm picked for this shape
and M), and reduces the chunks' sums in order to BF16 (`ve_reduce`), so a plain and an encoded linear over the same weight
return the same bits.  A call at more rows (prefill) decodes
the layer into a scratch FP8 matrix and hands it to vLLM's own FP8 apply: prefill is compute-bound and isn't what's timed.

Kernels are pluggable: a `Lib` is any build of `ve.cu` (packing, k range), and a later fused kernel (a tag stage, a wider
codec) is another `Lib` whose `ve_gemm` takes the same arguments.
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
FORMS = ("simple", "pipe", "ws")


def _codec():
    if "wc_codec" not in sys.modules:
        spec = importlib.util.spec_from_file_location("wc_codec", WORD_CODEC / "codec.py")
        mod = sys.modules["wc_codec"] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return sys.modules["wc_codec"]


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


@dataclass(frozen=True)
class Config:
    """One GEMM form and its launch: `groups` CTA groups a chunk for the persistent forms (0: as many as fit), and the
    warp-specialized form's consumer warps and ring stages."""

    form: int
    groups: int = 0
    consumers: int = 0
    stages: int = 0

    @property
    def name(self) -> str:
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


class _Weighted:
    """The layer vLLM's own apply sees at prefill: the scratch weight, and every other attribute the layer's."""

    def __init__(self, layer, weight) -> None:
        object.__setattr__(self, "_layer", layer)
        object.__setattr__(self, "weight", weight)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_layer"), name)


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

    def config(self, m: int) -> Config:
        for mm in sorted(self.configs):
            if mm >= m:
                return self.configs[mm]
        return Config(0)

    def gemm(self, x8: torch.Tensor, cfg: Config | None = None, partials: bool = True) -> torch.Tensor:
        m = x8.shape[0]
        cfg = cfg or self.config(m)
        part = (torch.empty if partials else torch.zeros)((self.nchunks if partials else 1, m, self.n), dtype=torch.float32,
                                                          device=x8.device)
        self.lib.call("ve_gemm", cfg.form, self.k, self.store.data_ptr(), x8.data_ptr(), part.data_ptr(), m, self.n, self.kd,
                      self.base, int(partials), cfg.groups, cfg.consumers, cfg.stages, stream())
        return part

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

    def __call__(self, layer, x, bias, original, quant) -> torch.Tensor:
        """vLLM's apply: ``quant(x2d) -> (fp8 x, scale)`` is the FP8 kernel's own activation quantization."""
        if not isinstance(x, torch.Tensor) or x.numel() > MAX_ROWS * x.shape[-1]:
            return original(_Weighted(layer, self.dense().t()), x, bias)
        x2 = x.reshape(-1, x.shape[-1])
        x8, xs = quant(x2)
        y = self.reduce(self.gemm(x8.contiguous()), xs.reshape(-1).float().contiguous())
        if bias is not None:
            y = y + bias
        return y.reshape(*x.shape[:-1], self.n)


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


__all__ = ["FORMS", "MAX_ROWS", "Config", "EncodedLinear", "Lib", "Linear", "PlainLinear", "candidates", "stream"]
