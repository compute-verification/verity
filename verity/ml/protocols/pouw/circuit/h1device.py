"""H1's device records: the hardware facts an H1 window (`h1window`, the `eps1-sm120-h1fx1` tile) is built and served on, one a
GPU architecture. `h1window.h1_tile` takes one by its architecture (``arch``), binds its tiles' exact accumulator with the
record's s8 binding and refuses a contraction outside the record's domain.

* **The step.** Each record names the s8 tensor-core step its kernels' MMAs compute (a `silicon.integer.IntegerMma`, PINNED
  in `silicon.instructions`) and binds it with `ops.boolean.gemm.s8_gemm`, which gives the registered
  `BlackwellS8ImmaDot32_v2` for every card whose step is that function and refuses any other. sm_90's step,
  `HOPPER_SM90_S8_K32`, is sm_120's under Hopper's name (the H200's ``wgmma`` m64nNk32 and ``mma.sync`` m16n8k32, run
  r20261019-043020-f267, 0 mismatches), so an sm_90 window's Program is the sm_120 one's, byte for byte, and a served
  path registers the same trees on either card.
* **The domain** is the contractions the record's kernels serve, beyond what the Definitions refuse themselves (32 | K and
  K < 2^17, `ops.gemm.S8_K_LIMIT`). sm_90's is Dither's h1u units (`benchmarks/hash/dither/h1u.cu`, gated on an H200 at
  GLM-5.3 int8 v3's shapes in run r20261023-195535-8920): 512 | K, as h1u reads each noise block as 16 segments' bits
  (``kh_unit``: ``K % 512`` refused), and K >= 384, `h1zy.LEAST_K`, the least K whose leaf's combine is proved, so the least
  K served is 512. On that domain m = `segment.n_digits`(K / 32) is 2 or 3, the digits h1u's one stream recombines. h1u
  issues ``mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32`` without ``.satfinite``; below 2^17 no partial sum of s8 x s8
  products from a zero accumulator leaves s32 (`integer.exact_k_limit`), so it agrees with the clamped step word for word.
  sm_120's record states no domain: its windows keep the Definitions' own refusals, as before records existed.
* **The sheet.** Each record holds the device's data-sheet facts with their sources: SMs, the clock its tensor peaks are
  quoted at, f_max, the dense s8 MACs an SM clock, and the data sheet's dense INT8 figure (None where no data sheet states
  one). sm_90's are the machine model's DS-90 sheet (`benchmarks/pouw/machine_model/data/devices/sm_90/sheet.toml`, whose
  Lean is `Security/Definitions/Accounting/Sheets/H200.lean`), sm_120's its DS-120 (`Accounting.TS.Headline.Sm120.ds120`).
"""
from __future__ import annotations

from dataclasses import dataclass

from verity.ml.circuits.ops.boolean import gemm as BG
from verity.ml.circuits.silicon.integer import BLACKWELL_SM120_S8_M16N8K32, HOPPER_SM90_S8_K32, IntegerMma

from . import h1zy as Z


@dataclass(frozen=True)
class Device:
    """One architecture's H1 record (module docstring)."""

    arch: str
    gpu: str
    step: IntegerMma                       # the s8 step the kernels' MMAs compute
    instruction: str                       # its `silicon.instructions` entry, the instruction the kernels issue
    domain: tuple[int, int] | None         # (least K, K's divisor) the kernels serve; None: the Definitions' own
    sms: int
    tensor_mhz: int                        # the clock the data sheet's tensor peaks are quoted at
    fmax_mhz: int
    s8_macs: int                           # dense s8 MACs an SM clock
    sheet_s8_tops: float | None            # the data sheet's dense INT8 figure; None: no data sheet states one
    source: str

    def __post_init__(self) -> None:
        if self.step.arch != self.arch:
            raise ValueError(f"{self.arch}'s H1 record names {self.step.name}, an {self.step.arch} step")

    @property
    def gemm(self) -> dict:
        """The tiles' s8 binding on bits (`BG.s8_gemm` of the record's step)."""
        return BG.s8_gemm(self.step)

    @property
    def s8_dense_tops(self) -> float:
        """2 MACs' operations at `s8_macs` an SM clock on every SM at `tensor_mhz`, in TOPS."""
        return 2 * self.s8_macs * self.sms * self.tensor_mhz / 1e6

    def refuse(self, k: int) -> str | None:
        """Why the record's kernels serve no H1 tile at contraction ``k``, or None."""
        if self.domain is None:
            return None
        least, div = self.domain
        if k < least or k % div:
            return (f"K={k} is outside H1's domain on {self.arch} ({self.gpu}): K >= {least} (h1zy.LEAST_K, the least K "
                    f"whose leaf's combine is proved) and {div} | K (h1u reads each noise block as 16 segments' bits)")
        return None


SM120 = Device(
    arch="sm_120", gpu="RTX PRO 6000 Blackwell Server Edition", step=BLACKWELL_SM120_S8_M16N8K32,
    instruction="sm120.mma.m16n8k32.s8", domain=None,
    sms=188, tensor_mhz=2430, fmax_mhz=2430, s8_macs=1024, sheet_s8_tops=None,
    source="DS-120 (Accounting.TS.Headline.Sm120.ds120): f_max 2430 MHz, the driver's reported maximum (SPEC's P1: no data "
           "sheet states it), at which DS-120 quotes every peak; IMMA16832 at 1024 dense MACs an SM clock (census "
           "r20261015-062626-f65b: 1,023.6 measured); 188 SMs, as the device reports (machine model table.toml); no RTX PRO "
           "6000 data sheet figure for INT8 is on file",
)

SM90 = Device(
    arch="sm_90", gpu="H200 SXM (sm_90a)", step=HOPPER_SM90_S8_K32,
    instruction="sm90.mma.m16n8k32.s8", domain=(Z.LEAST_K, 512),
    sms=132, tensor_mhz=1830, fmax_mhz=1980, s8_macs=4096, sheet_s8_tops=1979,
    source="DS-90 (benchmarks/pouw/machine_model/data/devices/sm_90/sheet.toml): H200 data sheet "
           "(nvidia.com/en-us/data-center/h200, Technical Specifications, H200 SXM): 'INT8 Tensor Core 3,958' TOPS, footnote "
           "2 'With sparsity', so 1,979 dense; H100 Tensor Core GPU Architecture whitepaper V1.04, Table 3, H100 SXM5 (the H200 "
           "data sheet states no clock or SM count, its peaks equal H100 SXM5's): 'Peak INT8 Tensor TOPS 1978.9/3957.8', "
           "'GPU Boost Clock for FP8, FP16, BF16, TF32 Tensor Core Ops: 1830 MHz', 'GPU Boost Clock for FP64 Tensor Core Ops, "
           "FP32 and FP64 non Tensor Core Ops: 1980 MHz', 'SMs 132'; so 4096 dense s8 MACs an SM clock "
           "(2 x 4096 x 132 x 1830 MHz = 1,979 TOPS); the device reports 132 SMs and a 1980 MHz maximum (vy-dc-h200)",
)

DEVICES = {d.arch: d for d in (SM120, SM90)}


def device(arch: str) -> Device:
    """The H1 record of architecture ``arch`` (`DEVICES`), else ValueError."""
    try:
        return DEVICES[arch]
    except (KeyError, TypeError):
        raise ValueError(f"H1 has no device record for {arch!r}: its records are {', '.join(DEVICES)}") from None


__all__ = ["DEVICES", "Device", "SM120", "SM90", "device"]
