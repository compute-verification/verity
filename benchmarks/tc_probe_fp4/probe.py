"""Blackwell FP4 tensor-core probe (lanes fp4-5090, fp4-re): one ``mma.sync`` per tile on an RTX 5090, raw bits captured and
compared bit for bit against ``verity.ml.circuits.silicon.models`` (``BLACKWELL_SM120_NVF4`` / ``BLACKWELL_SM120_MXF4``, the primary model and
the validation gate) and against ``model5.py`` (autoproof's five-way align-add, kept as the control whose D16/D17/D18 mismatches
the verity rule must remove).  Families under ``families_fp4_re`` are lane fp4-re's accumulator families; ``specials`` is
record-only (NaN / inf accumulators, sums leaving binary32: the hardware words are summarised, no model is asserted), and so
are mxf4's ``families_mxf4_edges`` (the UE8M0 NaN scale, overflowing blocks that cancel, the RZ edge at 2^128, infinite
accumulators against overflows): what a total mxf4 step must decide.

    python benchmarks/tc_probe_fp4/probe.py --instruction nvf4 --out $RESEARCH_RUN_DIR --sass-check \
        --replay-autoproof $RESEARCH_RUN_DIR/inputs/probe3_1783788526.json --sweep --n-random 4096

Instructions (``--instruction``): ``nvf4`` = ``mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col
.f32.e2m1.e2m1.f32.ue4m3`` (id ``sm120.mma.m16n8k64.e2m1.nvf4``); ``mxf4`` = the same with ``.scale_vec::2X ... .ue8m0``
(``sm120.mma.m16n8k64.e2m1.mxf4``, capture + model5-with-ue8m0-shift as a hypothesis, no hot clip); ``f8f6f4`` = the unscaled
``mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e2m1.e2m1.f32`` (``sm120.mma.m16n8k32.e2m1.f8f6f4``, autoproof probe2
replay only); ``bf16`` = ``mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`` (``sm120.mma.m16n8k16.bf16``), compared with
``verity.ml.circuits.silicon.silicon.tc_dot`` under BOTH ``AMPERE_BF16_M16N8K16`` and ``HOPPER_BF16_M16N8K16`` -- which agrees is the finding.

The kernel (``mma_fp4.cu``) is built with ``-arch=sm_120a``: ptxas refuses ``.block_scale`` / ``.kind::f8f6f4`` on plain
``sm_120`` ("not supported on .target sm_120"), the arch-specific target is required.  ``--sass-check`` runs ``cuobjdump -sass``
and records the number of MMA instructions per tile kernel (must be exactly one each; the run fails otherwise).

``--replay-autoproof FILE`` replays every ``sample_cases`` record of autoproof's ``probes/results/probe{2,3}_*.json`` (read-only
source repo, shipped with ``--send``): each dot at its own tile position (r % 16, (r // 16) % 8), the rest of the tile zero, a zero
accumulator, K/64 (probe3) or K/32 (probe2) dependent launches with the hardware's D fed back as C (ascending K, as autoproof
chained), and requires the hardware word to equal ``silicon_c_bits`` -- the reproduction gate.  Any mismatch is a finding: it is
recorded, nothing is tuned.

``--sweep`` runs the families (fixed seed) and compares every output element with model5: random operands with random scales and
random accumulators; random with zero accumulators; uniform encodings (codes, scales and accumulator bits all uniform); the full
UE4M3 scale-code sweep (every valid code 0x00..0x7E, one and both operands); the hot-group-clip regime (one live group with both
scale codes in 0x78..0x7E, the other groups on subnormal scales with bits at 2^-20, tiny or zero accumulators); all-hot and
one-operand-hot controls; large cancellation (two groups cancel exactly at the top of the window while small terms sit below
E1-41: the window-width probe); accumulator-window probe (large accumulators over tiny terms: W2); tiny accumulators; signed zero.

MEASURED 2026-09-22 on vy-5090 (RTX 5090, driver 570.153.02, nvcc 12.8.93; runs r20260922-182011-88f8 nvf4,
r20260922-182132-d6db f8f6f4, r20260922-182548-e654 mxf4, r20260922-182911-1f10 bf16):
  * autoproof replay: probe3 1601/1601 (K=64/128/256) and probe2 1400/1400 dots reproduced bit for bit -- model5 IS the silicon
    for zero-initialised ascending-K chains, on a different RTX 5090 and a different kernel.
  * nvf4 sweep, 1,736,704 elements, 263,781 hot firings: every family with a +0 accumulator agrees with model5 (0 mismatches over
    hot_clip_regime 131,072 / random_zero_acc 262,144 / scale sweeps / acc_window / tiny_acc).  Two regimes autoproof never probed
    (its accumulators were always +0 or the running sum, i.e. the largest addend) disagree: (1) a -0 accumulator with all-zero
    products returns +0 on silicon, model5 keeps -0 (16,328/16,328 such elements); (2) an incoming accumulator of the OPPOSITE
    sign to the result whose msb lies 37..42 binades below E1 -- at the alignment window's edge -- comes out 1 ulp (up to 8 in the
    cancellation family) larger in magnitude than model5's truncated-toward-zero sum (12,790 of 1,507,328 nonzero-acc elements,
    100% opposite-sign).  Neither a narrower accumulator window nor an anchor computed from scale exponents explains all of them;
    the accumulator's alignment path is the open question, not the 42/36-bit windows, the RZ normalisation or the hot clip.
  * mxf4 (ue8m0, .2X): accepted by ptxas on sm_120a (SASS OMMA.SF...E8); model5 with 4 groups of 16 under a per-32 shift is the
    better structure (2 groups of 32: 32,691/32,768 wrong in large_cancellation vs 126) but not exact: 798/131,072 zero-acc
    mismatches.  Capture only; hypothesis open.
  * bf16 m16n8k16 on the RTX 5090: HOPPER_BF16_M16N8K16 reproduces 204,800/204,800 words; AMPERE_BF16_M16N8K16 fails 48,569.
    The consumer Blackwell part keeps the Hopper BF16 pipeline (one 16-wide group, 26-bit alignment).

Outputs under ``--out``: ``probe_results.json`` (identity, sass counts, replay and per-family counts, first mismatches),
``capture_sample.json`` (records sampled per family: acc, a/b codes, scales, d as hex), ``tiles_<family>.npz`` (the full capture),
``result.json`` (``kind: tc-evidence/v1``; meta per store README §11: instruction, arch, sku, driver, nvcc, families, counts,
model, model_agrees, first_mismatches (<= 20), seed, n; validation passes iff every replay and model mismatch count is zero).

PoUW RTX PRO 6000 (GPU 4, 2026-09-30) -- the K=32 E2M1 rows and the gates:
  * ``f8f6f4_e2m1_e4m3`` / ``f8f6f4_e4m3_e2m1`` (unscaled mixed), ``mxf8f6f4_e2m1`` / ``_e2m1_e4m3`` / ``_e4m3_e2m1`` (UE8M0
    ``.scale_vec::1X``) run through ``mma_fp4.cu``'s ``row_tiles_kernel<kind>``; they and ``f8f6f4`` (now swept with the K=32
    families of ``k32.py``, which also fixes ``--sweep`` for it: it used to generate K=64 operands) are captures whose finding is
    which of ``k32.candidates`` reproduce every word (``candidate_finding``); ``--expect-model NAME`` makes one the gate (a
    fresh-seed confirmation).  All of them lower to QMMA (SASS), where E4M3 x E4M3 is ``GroupSum`` (32,)/26/-133.
  * ``--chain L``: every family's tiles regrouped into chains of L dependent MMAs kept in registers (the generic kernel), against
    the models stepped L times; a chain layout check on exact integers runs first.
  * ``--expect-uuid``: stops before any measurement unless the device CUDA opened has that UUID (identity is read from CUDA, and
    nvidia-smi is queried for that GPU only: on a multi-GPU host its first line is some other GPU).
  * ``--sass-only`` (the offline SASS gate: exactly one MMA per kernel, the row's kernel present), ``--fake-silicon MODEL`` (a CPU
    dry run where MODEL's words stand in for the device; refused under a research run), ``--corrupt-word`` (one flipped bit in the
    first gated family: the run must fail).
  * casts (``--instruction cvt_*``, ``cast.py``); the 2:4 sparse ``mma.sp`` m16n8k128 rows (``--instruction sp_nvf4 | sp_mxf4``,
    ``sparse.py``: a capture over named candidates, the first of which is the dense pinned model on the compressed operands);
    T1's merged k128 route (``--instruction sp_nvf4_t1``, ``t1_merged.py``: the sparse k128 from C = H against two chained dense
    atoms from C = H, on the operands of ``fp4_emulation.py routes`` at 1ad1aaa2, for each ``--hot-bits`` h and C = +0);
    the sparse rows' metadata rule (``--instruction sp_nvf4_meta | sp_mxf4_meta``, ``meta_rule.py``: where each of the 16
    nibbles puts a chunk's stored codes, the pinned step on that gather, and the zero patterns the hardware can write);
    named vectors of a generated Lean vector file (``--instruction nvf4_lean_vectors --vectors F``, ``lean_vectors.py``: each
    selected word against the file's prediction, gated by the file's other in-model vectors, a 0xA5-poisoned D and a no-write
    control; ``--collect DIR`` derives a fill chunk's capture again on the CPU and publishes it).
  * ``scale_bit7`` (nvf4, record only): tile pairs whose UE4M3 scale bytes differ only in bit 7, which the pinned model refuses;
    ``bit7`` counts the words the pair's members differ in on the device, and checks the low-7-bit members against the model.
  * ``scale_split_twins`` (nvf4, mxf4): tile pairs with equal scaled products and accumulators but A's scale halved into its
    nibbles; ``twins`` counts the words the pair's members differ in, and any difference on the device means no model that
    reads only the exact products and the accumulator (no ``GroupSum``) computes the instruction.

Lane tcgen05 -- the datacenter Blackwell backend (``tcgen05_fp4.cu``, ``-arch=sm_100a`` on a B200 / ``sm_103a`` on a B300):
``--instruction tcgen05_nvf4`` = ``tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X`` (UE4M3, registry id
``sm100.tcgen05.mma.e2m1.nvf4``), ``tcgen05_mxf4`` = the same mnemonic with ``.scale_vec::2X`` and the UE8M0 instruction
descriptor (``sm100.tcgen05.mma.e2m1.mxf4``), ``tcgen05_mxf4kind`` = ``kind::mxf4.block_scale.scale_vec::2X`` (CUTLASS's mnemonic
for the same operands).  Each 16x8x64 tile is one 128x8x64 ``tcgen05.mma`` with the 16 rows replicated into all eight 16-row
blocks of A and of the TMEM accumulator (rows 16..127 are a consistency check across the four TMEM sub-partitions:
``replica_disagreements``).  The models compared are the sm_120 ones (``BLACKWELL_SM120_NVF4`` / ``_MXF4``): on sm_100 they are
the HYPOTHESIS under test, not a validated model.  Two accumulator paths are measured per family: ``enable_input_d`` = 0 for tiles
whose accumulator is all +0 bits (D = A.B, the accumulator is not read: a GEMM's first k-block) and 1 otherwise (D = A.B + D
with D preloaded into TMEM by ``tcgen05.st``); ``tmem_path`` forces path 1 for every tile and ``path_disagreements`` counts the
words the two paths disagree on.  ``--chain2`` pairs a family's tiles into K=128 chains (two MMAs back to back on the same TMEM
accumulator, the main-loop path) and compares with the models stepped twice.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import multiprocessing as mp
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
_VERITY_SRC = HERE.parents[1]  # the bf16 comparison imports verity.ml.circuits.silicon.silicon (and verity) from the shipped tree
for _src in (_VERITY_SRC, _VERITY_SRC / "catalog"):
    if _src.is_dir() and str(_src) not in sys.path:
        sys.path.insert(0, str(_src))
import k32  # noqa: E402
import model5  # noqa: E402

MAX_TILES_PER_CALL = 1 << 14

# Families whose accumulators or sums lie outside every model's domain (NaN / inf accumulators, sums leaving binary32): the
# hardware words are recorded and summarised, they never gate validation.
RECORD_ONLY_FAMILIES = {"specials", "scale_bit7", "mxf4_nan_scale", "mxf4_overflow_cancel", "mxf4_overflow_edge",
                        "mxf4_inf_acc_overflow"}

PTX = {
    "nvf4": "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3",
    "mxf4": "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0",
    "f8f6f4": "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e2m1.e2m1.f32",
    "bf16": "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32",
    # sm_100 (lane tcgen05): the mnemonic plus the instruction descriptor it is issued with (types, scale format, M, N, K-major)
    "tcgen05_nvf4": "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X [idesc: e2m1 x e2m1 -> f32, scale ue4m3, M128 N8 K64, K-major]",
    "tcgen05_mxf4": "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::2X [idesc: e2m1 x e2m1 -> f32, scale ue8m0, M128 N8 K64, K-major]",
    "tcgen05_mxf4kind": "tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X [idesc: e2m1 x e2m1 -> f32, scale ue8m0, M128 N8 K64, K-major]",
    # the K=32 E2M1 rows beyond the unscaled E2M1 x E2M1 one (k32.py; generic kernel row_tiles_kernel<kind>)
    "f8f6f4_e2m1_e4m3": "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e2m1.e4m3.f32",
    "f8f6f4_e4m3_e2m1": "mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e4m3.e2m1.f32",
    "mxf8f6f4_e2m1": "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e2m1.e2m1.f32.ue8m0",
    "mxf8f6f4_e2m1_e4m3": "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e2m1.e4m3.f32.ue8m0",
    "mxf8f6f4_e4m3_e2m1": "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e2m1.f32.ue8m0",
}
#: the conversion rows (cast.py; cvt_fp4.cu), dispatched before any MMA machinery
CAST_INSTRUCTIONS = ("cvt_e2m1x2_f32", "cvt_f16x2_e2m1x2", "cvt_ue8m0x2_f32_rz", "cvt_ue8m0x2_f32_rp", "cvt_bf16x2_ue8m0x2",
                     "cvt_e4m3x2_f32_ue4m3")
#: the 2:4 sparse mma.sp rows (sparse.py; mma_fp4.cu's sparse_tiles_kernel), dispatched like the casts
SPARSE_INSTRUCTIONS = ("sp_nvf4", "sp_mxf4")
#: T1's merged k128 route (t1_merged.py): the sparse NVF4 row against the dense one's two-atom chain, from the hot accumulator
T1_INSTRUCTIONS = ("sp_nvf4_t1",)
#: the sparse rows' metadata rule (meta_rule.py): what each of the 16 nibbles gathers, and the pinned step on that gather
META_RULE_INSTRUCTIONS = ("sp_nvf4_meta", "sp_mxf4_meta")
#: named vectors of a generated Lean vector file (lean_vectors.py), captured on the NVF4 row against the file's predicted words
LEAN_VECTOR_INSTRUCTIONS = ("nvf4_lean_vectors",)
#: mma.sync instruction -> RowKind of mma_fp4.cu's generic kernel (single tiles of the new rows, and every in-register chain)
ROW_KIND = {"nvf4": 0, "mxf4": 1, "f8f6f4": 2, "f8f6f4_e2m1_e4m3": 3, "f8f6f4_e4m3_e2m1": 4, "mxf8f6f4_e2m1": 5, "mxf8f6f4_e2m1_e4m3": 6,
            "mxf8f6f4_e4m3_e2m1": 7}
#: the rows only the generic kernel issues
GENERIC_ONLY = {"f8f6f4_e2m1_e4m3", "f8f6f4_e4m3_e2m1", "mxf8f6f4_e2m1", "mxf8f6f4_e2m1_e4m3", "mxf8f6f4_e4m3_e2m1"}
#: rows autoproof's model5 applies to (E2M1 x E2M1 with no scale or the mxf4nvf4 scales)
MODEL5_ROWS = {"nvf4", "mxf4", "f8f6f4", "tcgen05_nvf4", "tcgen05_mxf4", "tcgen05_mxf4kind"}
INSTRUCTION_ID = {
    "nvf4": "sm120.mma.m16n8k64.e2m1.nvf4",
    "mxf4": "sm120.mma.m16n8k64.e2m1.mxf4",
    "f8f6f4": "sm120.mma.m16n8k32.e2m1.f8f6f4",
    "bf16": "sm120.mma.m16n8k16.bf16",
    "tcgen05_nvf4": "sm100.tcgen05.mma.e2m1.nvf4",
    "tcgen05_mxf4": "sm100.tcgen05.mma.e2m1.mxf4",
    "tcgen05_mxf4kind": "sm100.tcgen05.mma.e2m1.mxf4",
    "f8f6f4_e2m1_e4m3": "sm120.mma.m16n8k32.e2m1.e4m3.f8f6f4",
    "f8f6f4_e4m3_e2m1": "sm120.mma.m16n8k32.e4m3.e2m1.f8f6f4",
    "mxf8f6f4_e2m1": "sm120.mma.m16n8k32.e2m1.mxf8f6f4",
    "mxf8f6f4_e2m1_e4m3": "sm120.mma.m16n8k32.e2m1.e4m3.mxf8f6f4",
    "mxf8f6f4_e4m3_e2m1": "sm120.mma.m16n8k32.e4m3.e2m1.mxf8f6f4",
}
KERNEL = {"nvf4": "nvfp4_tiles_kernel", "mxf4": "mxfp4_tiles_kernel", "f8f6f4": "fp4raw_tiles_kernel", "bf16": "bf16_tiles_kernel",
          "tcgen05_nvf4": "tcgen05_nvfp4_tiles_kernel", "tcgen05_mxf4": "tcgen05_mxfp4_tiles_kernel", "tcgen05_mxf4kind": "tcgen05_mxf4kind_tiles_kernel",
          **{instr: f"row_tiles_kernelILi{kind}EE" for instr, kind in ROW_KIND.items() if instr in GENERIC_ONLY}}
K_OF = {"nvf4": 64, "mxf4": 64, "f8f6f4": 32, "bf16": 16, "tcgen05_nvf4": 64, "tcgen05_mxf4": 64, "tcgen05_mxf4kind": 64,
        **{instr: 32 for instr in GENERIC_ONLY}}
NBLK = {"nvf4": 4, "mxf4": 2, "f8f6f4": 0, "bf16": 0, "tcgen05_nvf4": 4, "tcgen05_mxf4": 2, "tcgen05_mxf4kind": 2,
        "f8f6f4_e2m1_e4m3": 0, "f8f6f4_e4m3_e2m1": 0, "mxf8f6f4_e2m1": 1, "mxf8f6f4_e2m1_e4m3": 1, "mxf8f6f4_e4m3_e2m1": 1}
SCALE_KIND = {"nvf4": "ue4m3", "mxf4": "ue8m0", "f8f6f4": "none", "tcgen05_nvf4": "ue4m3", "tcgen05_mxf4": "ue8m0", "tcgen05_mxf4kind": "ue8m0",
              "f8f6f4_e2m1_e4m3": "none", "f8f6f4_e4m3_e2m1": "none", "mxf8f6f4_e2m1": "ue8m0", "mxf8f6f4_e2m1_e4m3": "ue8m0",
              "mxf8f6f4_e4m3_e2m1": "ue8m0"}
#: sm_100 backend: instruction -> kernel kind (tcgen05_fp4.cu) ; the source file and the default arch per backend
TCGEN05_KIND = {"tcgen05_nvf4": 0, "tcgen05_mxf4": 1, "tcgen05_mxf4kind": 2}
SOURCE = {instr: ("tcgen05_fp4.cu" if instr in TCGEN05_KIND else "mma_fp4.cu") for instr in PTX}
DEFAULT_ARCH = {instr: ("sm_100a" if instr in TCGEN05_KIND else "sm_120a") for instr in PTX}
#: canonical K-major no-swizzle shared-memory strides (bytes): LBO = the second 16-byte K chunk of an 8-row group, SBO = the next group
TCGEN05_SMEM = {"lbo_a": 128, "sbo_a": 256, "lbo_b": 128, "sbo_b": 256}
#: hex digits per operand code in the records (E4M3 bytes take two)
HEX_WIDTHS = {instr: tuple(2 if f == "e4m3" else 1 for f in k32.ROWS[instr][:2]) for instr in k32.ROWS}
UE4M3_VALID = np.array([c for c in range(0x80) if c != 0x7F], dtype=np.uint8)  # 127 codes; 0x7F is NaN


# --- build -----------------------------------------------------------------------------------------------------------------------


def uuid_allowed(uuid: str | None, expect: str) -> bool:
    """``--expect-uuid`` is one UUID or a comma list (a lease picks the GPU: every UUID the host's table allows), environment
    variables expanded: ``'$GPU_LEASE_UUID'`` is the GPU node 2's ``gpu-lease`` gave the job (unset, it stays literal and fails)."""
    expect = os.path.expandvars(expect)
    return uuid is not None and uuid in {u.strip() for u in expect.split(",") if u.strip()}


def nvcc_bin() -> str:
    return shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"


def arch_flags(arch: str) -> list[str]:
    """nvcc target flags.  For an arch-specific target (sm_100a, sm_103a, sm_120a) name the compute_XXXa PTX explicitly: nvcc 12.9's
    ``-arch=sm_100a`` also embeds a plain ``compute_100`` PTX, and the tcgen05 mnemonics do not assemble for that target."""
    m = re.fullmatch(r"sm_(\d+)a", arch)
    if m:
        return [f"-gencode=arch=compute_{m.group(1)}a,code=sm_{m.group(1)}a"]
    return [f"-arch={arch}"]


def build(arch: str, out: Path, source: str = "mma_fp4.cu") -> tuple[Path, str]:
    so = out / f"lib{Path(source).stem}_{arch}.so"
    nvcc = subprocess.run([nvcc_bin(), "--version"], capture_output=True, text=True, check=True).stdout.strip().splitlines()[-2:]
    subprocess.run([nvcc_bin(), "-O2", *arch_flags(arch), "-Xcompiler", "-fPIC", "-shared", "-o", str(so), str(HERE / source)], check=True)
    return so, " / ".join(line.strip() for line in nvcc)


def prebuilt(path: str) -> tuple[Path, str]:
    """A library ``build`` produced on another host (``--library``; the cluster's job containers have no CUDA toolkit): the path
    and its toolchain identity (``<so>.nvcc``, else the ``nvcc`` of a ``build.json`` beside it) plus its sha256."""
    so = Path(path)
    sidecar, record = Path(f"{so}.nvcc"), so.parent / "build.json"
    if sidecar.exists():
        ident = sidecar.read_text().strip()
    elif record.exists():
        ident = json.loads(record.read_text()).get("nvcc") or "unknown toolchain"
    else:
        ident = "unknown toolchain"
    return so, f"{ident} (prebuilt {so.name}, sha256 {hashlib.sha256(so.read_bytes()).hexdigest()})"


def sass_check(so: Path) -> dict:
    """MMA instructions per tile kernel from ``cuobjdump -sass``: {kernel: {count, opcodes}} (sm_120: HMMA/OMMA/QMMA; sm_100: the
    UTC*MMA family the tcgen05.mma mnemonics lower to)."""
    cuobjdump = str(Path(nvcc_bin()).parent / "cuobjdump")
    text = subprocess.run([cuobjdump, "-sass", str(so)], capture_output=True, text=True, check=True).stdout
    out: dict = {}
    fn = None
    for line in text.splitlines():
        m = re.search(r"Function : (\S+)", line)
        if m:
            fn = m.group(1)
            for k in KERNEL.values():
                if k in fn:
                    fn = k
            out[fn] = {"mma_count": 0, "opcodes": [], "tcgen05_opcodes": []}
            continue
        if fn and re.search(r"\b[A-Z]*MMA\b|\bOMMA\.|\bQMMA\.|\bHMMA\.|\bUTC[A-Z]*MMA", line):
            op = line.split("*/")[1].split()[0] if "*/" in line else line.strip()
            out[fn]["mma_count"] += 1
            out[fn]["opcodes"].append(op)
        elif fn and re.search(r"\bUTC[A-Z]+\b|\bUTMA\b", line):  # the other tensor-memory instructions (alloc, ld/st, commit ...)
            op = line.split("*/")[1].split()[0] if "*/" in line else line.strip()
            if op not in out[fn]["tcgen05_opcodes"]:
                out[fn]["tcgen05_opcodes"].append(op)
    for rec in out.values():
        if not rec["tcgen05_opcodes"]:
            del rec["tcgen05_opcodes"]
    return out


class Probe:
    def __init__(self, so: Path):
        self.lib = ctypes.CDLL(str(so))
        P = ctypes.c_void_p
        self.tcgen05 = hasattr(self.lib, "tcgen05_tiles")
        if self.tcgen05:
            fn = self.lib.tcgen05_tiles
            fn.restype = ctypes.c_int
            fn.argtypes = [ctypes.c_int] + [P] * 7 + [ctypes.c_int, ctypes.c_int] + [ctypes.c_uint] * 4
            self.lib.tcgen05_idesc.restype = ctypes.c_uint
            self.lib.tcgen05_idesc.argtypes = [ctypes.c_int]
            self.smem = dict(TCGEN05_SMEM)
            self.replica_disagreements = 0  # words of TMEM lanes 16..127 that differ from their replica in lanes 0..15 (reset by the caller)
            self.path_tiles = {"acc_disabled": 0, "acc_tmem": 0}
            self.last_Dall = None  # the last call's full 128-lane TMEM readback (layout-check diagnostics)
        else:
            for name, argc in (("mma_nvfp4_tiles", 6), ("mma_mxfp4_tiles", 6), ("mma_fp4raw_tiles", 4), ("mma_bf16_tiles", 4)):
                fn = getattr(self.lib, name)
                fn.restype = ctypes.c_int
                fn.argtypes = [P] * argc + [ctypes.c_int]
            fn = self.lib.mma_row_tiles
            fn.restype = ctypes.c_int
            fn.argtypes = [ctypes.c_int] + [P] * 6 + [ctypes.c_int, ctypes.c_int]
        if hasattr(self.lib, "device_uuid"):
            self.lib.device_uuid.restype = ctypes.c_int
            self.lib.device_uuid.argtypes = [ctypes.c_char_p, ctypes.c_int]
        if hasattr(self.lib, "set_kernel_launch"):
            self.lib.set_kernel_launch.restype = None
            self.lib.set_kernel_launch.argtypes = [ctypes.c_int]
            self.lib.d_poison_byte.restype = ctypes.c_int
        self.debug_dir = None  # set by main(): where a failing layout check dumps its operands
        self.lib.has_mxf4.restype = ctypes.c_int
        ident = self.lib.device_identity
        ident.restype = ctypes.c_int
        ident.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.c_char_p, ctypes.c_int]

    def set_launch(self, on: bool) -> int:
        """mma_fp4.cu's no-write switch: off, every entry point uploads, poisons D and downloads it without launching a kernel.
        Returns the poison byte; a library built without the switch raises (the control cannot run on it)."""
        if not hasattr(self.lib, "set_kernel_launch"):
            raise RuntimeError("this library has no set_kernel_launch (built before the 0xA5 poison): rebuild mma_fp4.cu")
        self.lib.set_kernel_launch(1 if on else 0)
        return int(self.lib.d_poison_byte())

    def tcgen05_tiles(self, instr: str, A, Bt, SFA, SFB, C, steps: int = 1, force_tmem: bool = False) -> np.ndarray:
        """sm_100 backend: one 128x8x(64 steps) ``tcgen05.mma`` chain per tile.  ``A`` (n,16,64 steps), ``Bt`` (n,8,64 steps),
        ``SFA`` (n,16,nblk steps), ``SFB`` (n,8,nblk steps), ``C`` (n,16,8) -> ``D`` (n,16,8).  The accumulator path per tile:
        enable_input_d = 0 when the tile's C is all +0 bits (unless ``force_tmem``), else 1 with C preloaded into TMEM."""
        n = A.shape[0]
        kind, nblk = TCGEN05_KIND[instr], NBLK[instr]
        k = 64 * steps
        A = np.ascontiguousarray(A, dtype=np.uint8).reshape(n, 16, k)
        Bt = np.ascontiguousarray(Bt, dtype=np.uint8).reshape(n, 8, k)
        SFA = np.ascontiguousarray(SFA, dtype=np.uint8).reshape(n, 16, nblk * steps)
        SFB = np.ascontiguousarray(SFB, dtype=np.uint8).reshape(n, 8, nblk * steps)
        C = np.ascontiguousarray(C, dtype=np.uint32).reshape(n, 16, 8)
        enable = np.ones(n, np.uint8) if force_tmem else (C.reshape(n, -1) != 0).any(axis=1).astype(np.uint8)
        D = np.zeros((n, 16, 8), dtype=np.uint32)
        for s in range(0, n, MAX_TILES_PER_CALL):
            e = min(n, s + MAX_TILES_PER_CALL)
            dall = np.zeros((e - s, 128, 8), dtype=np.uint32)
            a, b, sa, sb, c, en = A[s:e], Bt[s:e], SFA[s:e], SFB[s:e], C[s:e], np.ascontiguousarray(enable[s:e])
            rc = self.lib.tcgen05_tiles(kind, a.ctypes.data, b.ctypes.data, sa.ctypes.data, sb.ctypes.data, c.ctypes.data, en.ctypes.data,
                                        dall.ctypes.data, e - s, steps, self.smem["lbo_a"], self.smem["sbo_a"], self.smem["lbo_b"], self.smem["sbo_b"])
            if rc != 0:
                raise RuntimeError(f"CUDA error {rc} in the {instr} tcgen05 tile kernel")
            D[s:e] = dall[:, :16, :]
            self.replica_disagreements += int(np.count_nonzero(dall[:, 16:, :] != np.tile(dall[:, :16, :], (1, 7, 1))))
            self.last_Dall = dall
        self.path_tiles["acc_disabled"] += int(np.count_nonzero(enable == 0))
        self.path_tiles["acc_tmem"] += int(np.count_nonzero(enable != 0))
        return D

    def identity(self) -> dict:
        major, minor = ctypes.c_int(), ctypes.c_int()
        name = ctypes.create_string_buffer(256)
        rc = self.lib.device_identity(ctypes.byref(major), ctypes.byref(minor), name, 256)
        if rc != 0:
            raise RuntimeError(f"CUDA error {rc} in cudaGetDeviceProperties")
        cuda_uuid = None
        if hasattr(self.lib, "device_uuid"):
            buf = ctypes.create_string_buffer(64)
            if self.lib.device_uuid(buf, 64) == 0:
                cuda_uuid = buf.value.decode()
        # nvidia-smi lists every GPU on the host whatever CUDA_VISIBLE_DEVICES says: query the one CUDA opened, by its UUID
        query = ["nvidia-smi", "--query-gpu=name,driver_version,compute_cap,uuid", "--format=csv,noheader"]
        smi = subprocess.run(query + (["-i", cuda_uuid] if cuda_uuid else []), capture_output=True, text=True)
        fields = [s.strip() for s in smi.stdout.strip().splitlines()[0].split(",")] if smi.returncode == 0 and smi.stdout.strip() else []
        return {"device": name.value.decode(), "compute_capability": f"{major.value}.{minor.value}",
                "nvidia_smi": smi.stdout.strip() if smi.returncode == 0 else None,
                "driver": fields[1] if len(fields) > 1 else None, "gpu_uuid": cuda_uuid or (fields[3] if len(fields) > 3 else None),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "hostname": platform.node(), "has_mxf4": bool(self.lib.has_mxf4()), "backend": "tcgen05" if self.tcgen05 else "mma.sync"}

    def clocks(self, uuid: str | None) -> dict | None:
        """SM / memory clocks, power and throttle reasons of this GPU now (captures are clock-independent; recorded, not locked)."""
        q = "clocks.sm,clocks.mem,clocks.max.sm,power.draw,temperature.gpu,clocks_throttle_reasons.active"
        smi = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"] + (["-i", uuid] if uuid else []),
                             capture_output=True, text=True)
        if smi.returncode != 0 or not smi.stdout.strip():
            return None
        return dict(zip(q.split(","), (s.strip() for s in smi.stdout.strip().splitlines()[0].split(","))))

    def rows(self, instr: str, A, Bt, C, SFA=None, SFB=None, steps: int = 1) -> np.ndarray:
        """The generic kernel: ``steps`` dependent MMAs per tile with D kept in registers.  ``A`` (n,16,K steps) / ``Bt`` (n,8,K steps)
        element codes (the K=32 rows' 8-bit containers are formed here), ``SFA`` (n,16,nblk steps) / ``SFB`` (n,8,nblk steps)."""
        n, kind, k, nblk = A.shape[0], ROW_KIND[instr], K_OF[instr], NBLK[instr]
        A = np.ascontiguousarray(A, dtype=np.uint8).reshape(n, 16, k * steps)
        Bt = np.ascontiguousarray(Bt, dtype=np.uint8).reshape(n, 8, k * steps)
        if k == 32:
            fa, fb, _ = k32.ROWS[instr]
            A, Bt = k32.container(fa, A), k32.container(fb, Bt)
        C = np.ascontiguousarray(C, dtype=np.uint32).reshape(n, 16, 8)
        if nblk:
            SFA = np.ascontiguousarray(SFA, dtype=np.uint8).reshape(n, 16, nblk * steps)
            SFB = np.ascontiguousarray(SFB, dtype=np.uint8).reshape(n, 8, nblk * steps)
        else:
            SFA, SFB = np.zeros((n, 16, 0), np.uint8), np.zeros((n, 8, 0), np.uint8)
        D = np.zeros((n, 16, 8), dtype=np.uint32)
        for s in range(0, n, MAX_TILES_PER_CALL):
            e = min(n, s + MAX_TILES_PER_CALL)
            a, b, sa, sb, c = (np.ascontiguousarray(x[s:e]) for x in (A, Bt, SFA, SFB, C))
            d = np.zeros((e - s, 16, 8), dtype=np.uint32)
            rc = self.lib.mma_row_tiles(kind, a.ctypes.data, b.ctypes.data, sa.ctypes.data, sb.ctypes.data, c.ctypes.data, d.ctypes.data,
                                        e - s, steps)
            if rc != 0:
                raise RuntimeError(f"CUDA error {rc} in the {instr} row kernel (steps={steps})")
            D[s:e] = d
        return D

    def tiles(self, instr: str, A, Bt, C, SFA=None, SFB=None, steps: int = 1, **tcgen05_kw) -> np.ndarray:
        """One MMA per tile.  ``A`` (n,16,K) / ``Bt`` (n,8,K) codes (u8; bf16: u16 words), ``SFA`` (n,16,nblk) / ``SFB`` (n,8,nblk) u8,
        ``C`` (n,16,8) FP32 words -> ``D`` (n,16,8).  ``steps`` > 1: an in-register chain of that many MMAs (generic kernel)."""
        if instr in TCGEN05_KIND:
            return self.tcgen05_tiles(instr, A, Bt, SFA, SFB, C, steps=steps, **tcgen05_kw)
        if instr in GENERIC_ONLY or steps > 1:
            return self.rows(instr, A, Bt, C, SFA, SFB, steps)
        n = A.shape[0]
        k, nblk = K_OF[instr], NBLK[instr]
        dt = np.uint16 if instr == "bf16" else np.uint8
        A = np.ascontiguousarray(A, dtype=dt).reshape(n, 16, k)
        Bt = np.ascontiguousarray(Bt, dtype=dt).reshape(n, 8, k)
        C = np.ascontiguousarray(C, dtype=np.uint32).reshape(n, 16, 8)
        if nblk:
            SFA = np.ascontiguousarray(SFA, dtype=np.uint8).reshape(n, 16, nblk)
            SFB = np.ascontiguousarray(SFB, dtype=np.uint8).reshape(n, 8, nblk)
        D = np.zeros((n, 16, 8), dtype=np.uint32)
        for s in range(0, n, MAX_TILES_PER_CALL):
            e = min(n, s + MAX_TILES_PER_CALL)
            d = np.zeros((e - s, 16, 8), dtype=np.uint32)
            a, b, c = A[s:e], Bt[s:e], C[s:e]
            if instr == "nvf4":
                rc = self.lib.mma_nvfp4_tiles(a.ctypes.data, b.ctypes.data, SFA[s:e].ctypes.data, SFB[s:e].ctypes.data, c.ctypes.data, d.ctypes.data, e - s)
            elif instr == "mxf4":
                rc = self.lib.mma_mxfp4_tiles(a.ctypes.data, b.ctypes.data, SFA[s:e].ctypes.data, SFB[s:e].ctypes.data, c.ctypes.data, d.ctypes.data, e - s)
            elif instr == "f8f6f4":
                rc = self.lib.mma_fp4raw_tiles(a.ctypes.data, b.ctypes.data, c.ctypes.data, d.ctypes.data, e - s)
            else:
                rc = self.lib.mma_bf16_tiles(a.ctypes.data, b.ctypes.data, c.ctypes.data, d.ctypes.data, e - s)
            if rc != 0:
                raise RuntimeError(f"CUDA error {rc} in the {instr} tile kernel")
            D[s:e] = d
        return D


# --- encodings ---------------------------------------------------------------------------------------------------------------------


def f32_bits(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float32).view(np.uint32)


def random_f32_bits(rng, shape, exp_lo=1, exp_hi=254) -> np.ndarray:
    n = int(np.prod(shape))
    e = rng.integers(exp_lo, exp_hi + 1, size=n).astype(np.uint32)
    m = rng.integers(0, 1 << 23, size=n).astype(np.uint32)
    s = rng.integers(0, 2, size=n).astype(np.uint32)
    return ((s << 31) | (e << 23) | m).reshape(shape)


def e2m1_encode_ints(v: np.ndarray) -> np.ndarray:
    """Small integers in {-6..6} that are E2M1 values (0, +-1, +-2, +-3, +-4, +-6) -> codes."""
    table = {0: 0, 1: 2, 2: 4, 3: 5, 4: 6, 6: 7}
    out = np.zeros(v.shape, np.uint8)
    for val, code in table.items():
        out[v == val] = code
        out[v == -val] = code | 8 if val else 0
    return out


def ue4m3_pow2(e_field: int) -> int:
    """UE4M3 code for 2^(e_field-7) (mantissa field 0): the exact power-of-two scales of the layout check."""
    return e_field << 3


def to_bf16_bits(x: np.ndarray) -> np.ndarray:
    u = f32_bits(x).astype(np.uint64)
    lsb = (u >> 16) & 1
    return ((u + 0x7FFF + lsb) >> 16).astype(np.uint16)


def random_bf16_bits(rng, shape, exp_lo=1, exp_hi=254) -> np.ndarray:
    n = int(np.prod(shape))
    e = rng.integers(exp_lo, exp_hi + 1, size=n).astype(np.uint16)
    m = rng.integers(0, 128, size=n).astype(np.uint16)
    s = rng.integers(0, 2, size=n).astype(np.uint16)
    return ((s << 15) | (e << 7) | m).reshape(shape)


# --- layout checks (exact small integers: any fragment / scale-lane error shows up) -----------------------------------------------


def _layout_fail(probe: Probe, instr: str, label: str, want: np.ndarray, got: np.ndarray, **arrays: np.ndarray) -> None:
    """Report a layout-check failure with the first differing words (and, when the probe has a debug dir, every operand as npz)."""
    bad = int(np.count_nonzero(want != got))
    idx = np.argwhere(want != got)[:8]
    ex = "; ".join(f"t{t} r{i} c{j}: want {want[t, i, j]:08x} ({np.array([want[t, i, j]], np.uint32).view(np.float32)[0]:.9g}) "
                   f"got {got[t, i, j]:08x} ({np.array([got[t, i, j]], np.uint32).view(np.float32)[0]:.9g})" for t, i, j in idx)
    debug_dir = getattr(probe, "debug_dir", None)
    if debug_dir is not None:
        extra = {}
        if getattr(probe, "last_Dall", None) is not None:
            extra["Dall"] = probe.last_Dall
        np.savez(Path(debug_dir) / f"layout_debug_{instr}_{label.replace(' ', '_')}.npz", want=want, got=got, **arrays, **extra)
    raise RuntimeError(f"layout check ({label}) failed for {instr}: {bad}/{want.size} elements differ; first: {ex}")


def _exact_operands(rng, instr: str, n: int, steps: int):
    """Exact small-integer operands of ``steps`` chained k-blocks and C + A.B as FP32 words (every partial sum exact in FP32)."""
    k, nblk = K_OF[instr], NBLK[instr]
    fa, fb = k32.ROWS[instr][:2] if instr in k32.ROWS else ("e2m1", "e2m1")
    va, ca = k32.small_operands(rng, fa, (n, 16, k * steps))
    vb, cb = k32.small_operands(rng, fb, (n, 8, k * steps))
    c = rng.integers(-1000, 1001, (n, 16, 8)).astype(np.float64)
    if not nblk:
        return ca, cb, None, None, c, f32_bits(c + np.einsum("nik,njk->nij", va, vb))
    blk = k // nblk
    nb = nblk * steps
    if SCALE_KIND[instr] == "ue4m3":
        ea, eb = rng.integers(5, 11, (n, 16, nb)), rng.integers(5, 11, (n, 8, nb))
        sfa, sfb, pa, pb = (ea << 3).astype(np.uint8), (eb << 3).astype(np.uint8), 2.0 ** (ea - 7), 2.0 ** (eb - 7)
    else:
        lo, hi = (125, 131) if k == 64 else (126, 129)
        ea, eb = rng.integers(lo, hi, (n, 16, nb)), rng.integers(lo, hi, (n, 8, nb))
        sfa, sfb, pa, pb = ea.astype(np.uint8), eb.astype(np.uint8), 2.0 ** (ea - 127), 2.0 ** (eb - 127)
    prod = np.einsum("nigk,njgk->nijg", va.reshape(n, 16, nb, blk), vb.reshape(n, 8, nb, blk))
    return ca, cb, sfa, sfb, c, f32_bits(c + np.einsum("nijg,nig,njg->nij", prod, pa, pb))


def chain_layout_check(probe: Probe, instr: str, steps: int) -> str:
    """The generic kernel's in-register chain of ``steps`` MMAs on exact small integers: operands of step l at k-block l."""
    ca, cb, sfa, sfb, c, want = _exact_operands(np.random.default_rng(2), instr, 64, steps)
    got = probe.tiles(instr, ca, cb, f32_bits(c), sfa, sfb, steps=steps)
    if not np.array_equal(want, got):
        _layout_fail(probe, instr, f"chain of {steps}", want, got, a=ca, b=cb, c=c)
    return "passed"


def layout_check(probe: Probe, instr: str) -> str:
    rng = np.random.default_rng(1)
    n = 64
    k = K_OF[instr]
    if instr in GENERIC_ONLY:
        ca, cb, sfa, sfb, c, want = _exact_operands(rng, instr, n, 1)
        got = probe.tiles(instr, ca, cb, f32_bits(c), sfa, sfb)
        if not np.array_equal(want, got):
            _layout_fail(probe, instr, "single tile", want, got, a=ca, b=cb, c=c)
        return "passed"
    if instr == "bf16":
        a = rng.integers(-8, 9, (n, 16, k)).astype(np.float32)
        b = rng.integers(-8, 9, (n, 8, k)).astype(np.float32)
        c = rng.integers(-1000, 1001, (n, 16, 8)).astype(np.float32)
        want = f32_bits(c + np.einsum("nik,njk->nij", a, b))
        got = probe.tiles(instr, to_bf16_bits(a), to_bf16_bits(b), f32_bits(c))
    else:
        vals = np.array([0, 1, 2, 3, 4, 6, -1, -2, -3, -4, -6])
        a = vals[rng.integers(0, len(vals), (n, 16, k))]
        b = vals[rng.integers(0, len(vals), (n, 8, k))]
        c = rng.integers(-1000, 1001, (n, 16, 8)).astype(np.float64)
        nblk = NBLK[instr]
        if nblk:
            blk = k // nblk
            # distinct power-of-two scales per (row, block) and (col, block) in 2^-2..2^3: any lane/byte misrouting changes the
            # result, and every exact sum stays below 24 significant bits (|term| < 2^17, lowest bit 2^-4) so FP32 holds it exactly
            # (a wider range made 3/8192 mxf4 elements need > 24 bits: RNE reference vs RZ hardware, a false "layout" failure)
            if SCALE_KIND[instr] == "ue4m3":
                ea = rng.integers(5, 11, (n, 16, nblk))
                eb = rng.integers(5, 11, (n, 8, nblk))
                sfa, sfb = (ea << 3).astype(np.uint8), (eb << 3).astype(np.uint8)
                pa, pb = 2.0 ** (ea - 7), 2.0 ** (eb - 7)
            else:
                ea = rng.integers(125, 131, (n, 16, nblk))
                eb = rng.integers(125, 131, (n, 8, nblk))
                sfa, sfb = ea.astype(np.uint8), eb.astype(np.uint8)
                pa, pb = 2.0 ** (ea - 127), 2.0 ** (eb - 127)
            prod = np.einsum("nigk,njgk->nijg", a.reshape(n, 16, nblk, blk), b.reshape(n, 8, nblk, blk)).astype(np.float64)
            want = f32_bits(c + np.einsum("nijg,nig,njg->nij", prod, pa, pb))
            got = probe.tiles(instr, e2m1_encode_ints(a), e2m1_encode_ints(b), f32_bits(c), sfa, sfb)
            if instr in TCGEN05_KIND:
                # the accumulator-disabled path (enable_input_d = 0: D = A.B) on the same operands, and the K=128 chain
                want0 = f32_bits(np.einsum("nijg,nig,njg->nij", prod, pa, pb))
                if not np.array_equal(want, got):
                    _layout_fail(probe, instr, "accumulator enabled", want, got, a=a, b=b, c=c, sfa=sfa, sfb=sfb)
                got0 = probe.tiles(instr, e2m1_encode_ints(a), e2m1_encode_ints(b), np.zeros((n, 16, 8), np.uint32), sfa, sfb)
                if not np.array_equal(want0, got0):
                    _layout_fail(probe, instr, "accumulator disabled", want0, got0, a=a, b=b, sfa=sfa, sfb=sfb)
                h = n // 2
                a2, b2 = np.concatenate([a[:h], a[h:2 * h]], axis=2), np.concatenate([b[:h], b[h:2 * h]], axis=2)
                sfa2, sfb2 = np.concatenate([sfa[:h], sfa[h:2 * h]], axis=2), np.concatenate([sfb[:h], sfb[h:2 * h]], axis=2)
                want2 = f32_bits(c[:h] + np.einsum("nijg,nig,njg->nij", prod[:h], pa[:h], pb[:h]) + np.einsum("nijg,nig,njg->nij", prod[h:2 * h], pa[h:2 * h], pb[h:2 * h]))
                got2 = probe.tiles(instr, e2m1_encode_ints(a2), e2m1_encode_ints(b2), f32_bits(c[:h]), sfa2, sfb2, steps=2)
                if not np.array_equal(want2, got2):
                    _layout_fail(probe, instr, "K=128 chain", want2, got2, a=a2, b=b2, c=c[:h], sfa=sfa2, sfb=sfb2)
                if probe.replica_disagreements:
                    raise RuntimeError(f"layout check for {instr}: {probe.replica_disagreements} replica words (TMEM lanes 16..127) differ from lanes 0..15")
        else:
            want = f32_bits(c + np.einsum("nik,njk->nij", a, b))
            got = probe.tiles(instr, e2m1_encode_ints(a), e2m1_encode_ints(b), f32_bits(c))
    if not np.array_equal(want, got):
        bad = int(np.count_nonzero(want != got))
        raise RuntimeError(f"fragment layout check failed for {instr}: {bad}/{want.size} elements differ from C + A.Bt on exact integer tiles")
    return "passed"


# --- families ------------------------------------------------------------------------------------------------------------------------


def families_nvf4(rng, n_random: int, scale_kind: str, fixed_mult: int = 1):
    """Yield (name, A (n,16,64) codes, Bt (n,8,64), SFA (n,16,nblk), SFB (n,8,nblk), C (n,16,8)).  ``fixed_mult`` multiplies the
    gated families of fixed size (``signed_zero``, ``signed_zero_matrix``, ``anchor_participation``); 1 keeps every earlier stream."""
    k = 64
    nblk = 4 if scale_kind == "ue4m3" else 2

    def codes(shape):
        return rng.integers(0, 16, shape).astype(np.uint8)

    def scales(shape, lo=None, hi=None):
        if scale_kind == "ue4m3":
            return UE4M3_VALID[rng.integers(0, len(UE4M3_VALID), shape)]
        lo, hi = (107, 148) if lo is None else (lo, hi)
        return rng.integers(lo, hi, shape).astype(np.uint8)  # moderate E8M0 exponents (autoproof's random_scale), 0xFF NaN excluded

    def acc_random(n, lo=-30, hi=40):
        return f32_bits((rng.standard_normal((n, 16, 8)) * np.exp2(rng.integers(lo, hi, (n, 16, 8)))).astype(np.float32))

    zeros = lambda n: np.zeros((n, 16, 8), np.uint32)

    n = n_random
    yield "random_random_scales", codes((n, 16, k)), codes((n, 8, k)), scales((n, 16, nblk)), scales((n, 8, nblk)), acc_random(n)

    n = n_random // 2
    yield "random_zero_acc", codes((n, 16, k)), codes((n, 8, k)), scales((n, 16, nblk)), scales((n, 8, nblk)), zeros(n)

    n = n_random // 4
    yield ("uniform_encodings", codes((n, 16, k)), codes((n, 8, k)), scales((n, 16, nblk)), scales((n, 8, nblk)),
           random_f32_bits(rng, (n, 16, 8), 1, 250))

    if scale_kind == "ue4m3":
        # every valid UE4M3 code on every block: A sweeps codes row-wise, B sweeps col-wise (both operands), then one operand only
        n = n_random // 4
        idx = (np.arange(n)[:, None, None] * 64 + np.arange(16)[None, :, None] * 4 + np.arange(nblk)[None, None, :]) % 127
        sfa = UE4M3_VALID[idx]
        idxb = (np.arange(n)[:, None, None] * 32 + np.arange(8)[None, :, None] * 4 + np.arange(nblk)[None, None, :] + 61) % 127
        sfb = UE4M3_VALID[idxb]
        half = zeros(n)
        half[n // 2:] = acc_random(n - n // 2)
        yield "scale_sweep_both", codes((n, 16, k)), codes((n, 8, k)), sfa, sfb, half
        n = n_random // 8
        yield ("scale_sweep_a_only", codes((n, 16, k)), codes((n, 8, k)), UE4M3_VALID[idx[:n]],
               np.full((n, 8, nblk), 0x38, np.uint8), acc_random(n, -10, 10))  # B scale = 1.0

        # hot-group clip regime: block 0 of both operands hot (0x78..0x7E), blocks 1-3 on subnormal scales (codes 1..7, bits at
        # 2^-20); A block 0 sparse (few live products, small gd) so E1 stays below 22 and the clip decides the lsb.
        n = n_random // 4
        A = codes((n, 16, k))
        Bt = codes((n, 8, k))
        keep = rng.random((n, 16, 16)) < 0.15
        A[:, :, :16] = np.where(keep, A[:, :, :16], 0)
        sfa = rng.integers(1, 8, (n, 16, nblk)).astype(np.uint8)
        sfb = rng.integers(1, 8, (n, 8, nblk)).astype(np.uint8)
        sfa[:, :, 0] = rng.integers(0x78, 0x7F, (n, 16))
        sfb[:, :, 0] = rng.integers(0x78, 0x7F, (n, 8))
        acc = zeros(n)
        acc[n // 2:] = random_f32_bits(rng, (n - n // 2, 16, 8), 100, 120)  # tiny: 2^-27 .. 2^-7
        yield "hot_clip_regime", A, Bt, sfa, sfb, acc

        # hot on a random block, with the other blocks spanning the whole code range (mixed regime)
        n = n_random // 8
        sfa, sfb = scales((n, 16, nblk)), scales((n, 8, nblk))
        blk = rng.integers(0, nblk, n)
        sfa[np.arange(n), :, blk] = rng.integers(0x78, 0x7F, (n, 16))
        sfb[np.arange(n), :, blk] = rng.integers(0x78, 0x7F, (n, 8))
        yield "hot_one_block_mixed", codes((n, 16, k)), codes((n, 8, k)), sfa, sfb, acc_random(n, -30, 10)

        n = n_random // 8
        yield ("hot_all_blocks", codes((n, 16, k)), codes((n, 8, k)), rng.integers(0x78, 0x7F, (n, 16, nblk)).astype(np.uint8),
               rng.integers(0x78, 0x7F, (n, 8, nblk)).astype(np.uint8), acc_random(n, -10, 30))

        n = n_random // 8
        sfa = rng.integers(0x78, 0x7F, (n, 16, nblk)).astype(np.uint8)  # A hot, B not: control (no clip)
        yield "hot_a_only", codes((n, 16, k)), codes((n, 8, k)), sfa, rng.integers(1, 0x78, (n, 8, nblk)).astype(np.uint8), acc_random(n, -30, 10)

    # large cancellation (window-width probe): groups 0 and 1 cancel exactly at the top scale, groups 2-3 carry small terms
    n = n_random // 4
    A = codes((n, 16, k))
    Bt = codes((n, 8, k))
    A[:, :, 16:32] = A[:, :, :16]
    Bt[:, :, 16:32] = Bt[:, :, :16] ^ 8  # negate: every product of group 1 is minus the product of group 0
    Bt[:, :, 16:32][Bt[:, :, :16] == 0] = 0
    if scale_kind == "ue4m3":
        sfa, sfb = scales((n, 16, nblk)), scales((n, 8, nblk))
        top = rng.integers(0x60, 0x7F, (n, 16))
        sfa[:, :, 0] = top
        sfa[:, :, 1] = top
        topb = rng.integers(0x60, 0x7F, (n, 8))
        sfb[:, :, 0] = topb
        sfb[:, :, 1] = topb
        sfa[:, :, 2:] = rng.integers(0, 0x40, (n, 16, 2))
        sfb[:, :, 2:] = rng.integers(0, 0x40, (n, 8, 2))
    else:
        sfa, sfb = scales((n, 16, nblk), 137, 148), scales((n, 8, nblk), 137, 148)
        sfa[:, :, 1] = rng.integers(100, 117, (n, 16))
        sfb[:, :, 1] = rng.integers(100, 117, (n, 8))
    acc = zeros(n)
    acc[n // 2:] = acc_random(n - n // 2, -40, -10)
    yield "large_cancellation", A, Bt, sfa, sfb, acc

    # accumulator window (W2): large accumulators over small terms
    n = n_random // 8
    yield ("acc_window", codes((n, 16, k)), codes((n, 8, k)), scales((n, 16, nblk), 100, 120), scales((n, 8, nblk), 100, 120),
           acc_random(n, 20, 60) if scale_kind == "ue4m3" else acc_random(n, 0, 40))

    n = n_random // 8
    yield ("tiny_acc", codes((n, 16, k)), codes((n, 8, k)), scales((n, 16, nblk), 100, 120), scales((n, 8, nblk), 100, 120),
           random_f32_bits(rng, (n, 16, 8), 0, 30))

    n = 256 * fixed_mult
    A = np.where(rng.random((n, 16, k)) < 0.5, 0, 8).astype(np.uint8)  # +-0 operands
    Bt = codes((n, 8, k))
    A[n // 2:, :, :16] = 2  # +1.0
    Bt[n // 2:, :, :16] = np.where(np.arange(16) % 2 == 0, 5, 13).astype(np.uint8)  # +3, -3 alternating: exact cancellation
    acc = np.where(rng.random((n, 16, 8)) < 0.5, np.uint32(0x80000000), np.uint32(0)).astype(np.uint32)
    yield "signed_zero", A, Bt, scales((n, 16, nblk)), scales((n, 8, nblk)), acc

    yield from families_fp4_re(rng, n_random, scale_kind, fixed_mult)


# --- lane fp4-re: the accumulator families (D16 sign of zero, D17 alignment anchor, D18 ue8m0 scales, domain edges) ---------------


def _scale_value(codes: np.ndarray, scale_kind: str) -> np.ndarray:
    """Exact float64 value of every scale code (UE4M3: (8+m) 2^(e-10) / m 2^-9; UE8M0: 2^(code-127))."""
    c = codes.astype(np.int64)
    if scale_kind == "ue4m3":
        e, m = (c >> 3) & 0xF, c & 7
        return np.where(e == 0, m * 2.0**-9, (8 + m) * np.exp2((e - 10).astype(np.float64)))
    return np.exp2((c - 127).astype(np.float64))


def _signed_pow2_acc(rng, shape, e_lo: int, e_hi: int, exact_frac: float = 0.5) -> np.ndarray:
    """FP32 words +-(1.f) 2^e, e uniform in [e_lo, e_hi], f = 0 with probability ``exact_frac`` else uniform 23 bits."""
    n = int(np.prod(shape))
    e = rng.integers(e_lo, e_hi + 1, n) + 127
    e = np.clip(e, 1, 254).astype(np.uint32)
    f = np.where(rng.random(n) < exact_frac, 0, rng.integers(0, 1 << 23, n)).astype(np.uint32)
    s = rng.integers(0, 2, n).astype(np.uint32)
    return ((s << 31) | (e << 23) | f).reshape(shape)


def families_fp4_re(rng, n_random: int, scale_kind: str, fixed_mult: int = 1):
    """The fp4-re families, both scale kinds: yield (name, A, Bt, SFA, SFB, C) like ``families_nvf4``."""
    k, nblk = 64, (4 if scale_kind == "ue4m3" else 2)
    gpb = 4 // nblk  # groups (of 16) per scale block
    ue4 = scale_kind == "ue4m3"

    def codes(shape):
        return rng.integers(0, 16, shape).astype(np.uint8)

    def nz_codes(shape):
        c = rng.integers(1, 16, shape)
        return np.where(c == 8, 1, c).astype(np.uint8)  # never a zero (0x0 / 0x8)

    def scales(shape, lo=None, hi=None):
        if ue4:
            if lo is None:
                return UE4M3_VALID[rng.integers(0, len(UE4M3_VALID), shape)]
            return UE4M3_VALID[rng.integers(lo, hi, shape)]
        lo, hi = (100, 150) if lo is None else (lo, hi)
        return rng.integers(lo, hi, shape).astype(np.uint8)

    def pm_zero(shape):
        return np.where(rng.random(shape) < 0.5, np.uint32(0x80000000), np.uint32(0)).astype(np.uint32)

    zeros = lambda n: np.zeros((n, 16, 8), np.uint32)

    # --- D16: the sign-of-zero matrix.  Accumulator +-0 x (a) all-zero A against random B, (b) random A against all-zero B,
    # (c) +-0 against +-0 operands, (d) every product -0 x -0, (e) pairs cancelling exactly inside every group, (f) group 0
    # cancelled exactly by group 1 under equal scales.  Scales random, including the zero UE4M3 code.
    n = 6 * 96 * fixed_mult
    A, Bt = codes((n, 16, k)), codes((n, 8, k))
    m = np.arange(n) // (96 * fixed_mult)
    A[m == 0] = np.where(rng.random((int((m == 0).sum()), 16, k)) < 0.5, 0, 8)
    Bt[m == 1] = np.where(rng.random((int((m == 1).sum()), 8, k)) < 0.5, 0, 8)
    A[m == 2] = np.where(rng.random((int((m == 2).sum()), 16, k)) < 0.5, 0, 8)
    Bt[m == 2] = np.where(rng.random((int((m == 2).sum()), 8, k)) < 0.5, 0, 8)
    A[m == 3], Bt[m == 3] = 8, 8
    sel = m == 4
    A[sel, :, 1::2] = A[sel, :, 0::2]
    Bt[sel, :, 1::2] = np.where(Bt[sel, :, 0::2] == 0, 0, Bt[sel, :, 0::2] ^ 8)
    sel = m == 5
    A[sel, :, 16:32] = A[sel, :, :16]
    Bt[sel, :, 16:32] = np.where(Bt[sel, :, :16] == 0, 0, Bt[sel, :, :16] ^ 8)
    A[sel, :, 32:] = 0
    sfa, sfb = scales((n, 16, nblk)), scales((n, 8, nblk))
    if ue4:  # group 0 and group 1 share a block only in the ue8m0 layout; equalise the scales for the cross-group cancellation
        sfa[sel, :, 1] = sfa[sel, :, 0]
        sfb[sel, :, 1] = sfb[sel, :, 0]
    yield "signed_zero_matrix", A, Bt, sfa, sfb, pm_zero((n, 16, 8))

    # --- D17: accumulator offset walk.  One nonzero product per row/col in group g at slot s (both random per tile), the
    # accumulator +-(1.f) 2^(E1 - off) with off uniform in 20..60 binades below the term's msb E1, f exact (0) or random;
    # half the tiles ("many") add random small-scale products in the other groups.  Scales random (nonzero) on the leading block.
    n = n_random // 2
    A, Bt = np.zeros((n, 16, k), np.uint8), np.zeros((n, 8, k), np.uint8)
    g = rng.integers(0, 4, n)
    s = rng.integers(0, 16, n)
    kidx = g * 16 + s
    t = np.arange(n)
    A[t[:, None], np.arange(16)[None, :], kidx[:, None]] = nz_codes((n, 16))
    Bt[t[:, None], np.arange(8)[None, :], kidx[:, None]] = nz_codes((n, 8))
    many = rng.random(n) < 0.5
    other = np.ones((n, 16, k), bool)
    other[t[:, None], :, (g[:, None] * 16 + np.arange(16)[None, :])] = False  # keep the leading group clean
    fill_a, fill_b = codes((n, 16, k)), codes((n, 8, k))
    A = np.where(many[:, None, None] & other, fill_a, A)
    Bt = np.where(many[:, None, None] & other[:, :8, :], fill_b, Bt)
    sfa = scales((n, 16, nblk), 1, len(UE4M3_VALID)) if ue4 else scales((n, 16, nblk), 100, 150)
    sfb = scales((n, 8, nblk), 1, len(UE4M3_VALID)) if ue4 else scales((n, 8, nblk), 100, 150)
    small_a = scales((n, 16, nblk), 1, 0x30) if ue4 else scales((n, 16, nblk), 70, 110)
    small_b = scales((n, 8, nblk), 1, 0x30) if ue4 else scales((n, 8, nblk), 70, 110)
    blk = g // gpb
    lead_a = np.zeros((n, 16, nblk), bool)
    lead_a[t, :, blk] = True
    sfa = np.where(lead_a, sfa, np.where(many[:, None, None], small_a, sfa))
    sfb = np.where(lead_a[:, :8], sfb, np.where(many[:, None, None], small_b, sfb))
    va = model5.E2M1_NUM[A[t[:, None], np.arange(16)[None, :], kidx[:, None]].astype(np.int64)] / 2.0  # (n,16)
    vb = model5.E2M1_NUM[Bt[t[:, None], np.arange(8)[None, :], kidx[:, None]].astype(np.int64)] / 2.0  # (n,8)
    sa = _scale_value(sfa[t, :, blk], scale_kind)  # (n,16)
    sb = _scale_value(sfb[t, :, blk], scale_kind)  # (n,8)
    term = (va * sa)[:, :, None] * (vb * sb)[:, None, :]  # (n,16,8) exact in float64
    e1 = np.floor(np.log2(np.abs(term))).astype(np.int64)
    off = rng.integers(20, 61, (n, 16, 8))
    acc_e = np.clip(e1 - off + 127, 1, 254).astype(np.uint32)
    frac = np.where(rng.random((n, 16, 8)) < 0.5, 0, rng.integers(0, 1 << 23, (n, 16, 8))).astype(np.uint32)
    sign = rng.integers(0, 2, (n, 16, 8)).astype(np.uint32)
    yield "acc_offset_walk", A, Bt, sfa, sfb, ((sign << 31) | (acc_e << 23) | frac).astype(np.uint32)

    # --- D17: the widest group mantissas under the largest scales (nvf4: 6*6 x 16 under 448 x 448 = 19-bit group terms at
    # X = 8; model5's msb-anchored window and the scale-exponent anchor differ by 4 binades here) with accumulators around
    # the two grids (2^-24 .. 2^-8, both signs) and random large accumulators.
    n = n_random // 8
    A = np.where(rng.random((n, 16, k)) < 0.8, 0x7, 0xF).astype(np.uint8)
    Bt = np.where(rng.random((n, 8, k)) < 0.8, 0x7, 0xF).astype(np.uint8)
    A = np.where(rng.random((n, 16, k)) < 0.1, codes((n, 16, k)), A)
    if ue4:
        sfa, sfb = rng.integers(0x78, 0x7F, (n, 16, nblk)).astype(np.uint8), rng.integers(0x78, 0x7F, (n, 8, nblk)).astype(np.uint8)
    else:
        sfa, sfb = scales((n, 16, nblk), 128, 141), scales((n, 8, nblk), 128, 141)
    acc = _signed_pow2_acc(rng, (n, 16, 8), -24, -8)
    acc[n // 2:] = _signed_pow2_acc(rng, (n - n // 2, 16, 8), 20, 40)
    yield "wide_mantissa_top_scales", A, Bt, sfa, sfb, acc

    # --- anchor participation.  A tiny accumulator (+-2^e, e in -60..-30, exact or not) against (a) a "dead" group 0: nonzero
    # A nibbles, all-zero B nibbles, the largest scales -- no elementwise product, the accumulator must come back exact;
    # (b) a live group 0 whose dot cancels to zero under the largest scales -- the accumulator is truncated away (+0);
    # (c) nvf4 only: a live nonzero group 0 under a zero scale code on A, then on B -- no participation, exact accumulator.
    n = (3 if ue4 else 2) * 128 * fixed_mult
    A, Bt = np.zeros((n, 16, k), np.uint8), np.zeros((n, 8, k), np.uint8)
    m = np.arange(n) // (128 * fixed_mult)
    A[:, :, :16] = nz_codes((n, 16, 16))
    Bt[m >= 1, :, :16] = nz_codes((int((m >= 1).sum()), 8, 16))
    sel = m == 1
    A[sel, :, 1:16:2] = A[sel, :, 0:16:2]
    Bt[sel, :, 1:16:2] = Bt[sel, :, 0:16:2] ^ 8
    if ue4:
        sfa, sfb = rng.integers(0x60, 0x7F, (n, 16, nblk)).astype(np.uint8), rng.integers(0x60, 0x7F, (n, 8, nblk)).astype(np.uint8)
        sel = m == 2
        half = sel & (np.arange(n) % 2 == 0)
        sfa[half, :, 0] = 0
        sfb[sel & ~half, :, 0] = 0
    else:
        sfa, sfb = scales((n, 16, nblk), 140, 170), scales((n, 8, nblk), 140, 170)
    yield "anchor_participation", A, Bt, sfa, sfb, _signed_pow2_acc(rng, (n, 16, 8), -60, -30)

    # --- FP32 subnormal accumulators (exponent field 0, random mantissa incl. 0) under small terms
    n = n_random // 8
    acc = random_f32_bits(rng, (n, 16, 8), 0, 0)
    if ue4:
        yield "acc_subnormal", codes((n, 16, k)), codes((n, 8, k)), scales((n, 16, nblk), 0, 0x20), scales((n, 8, nblk), 0, 0x20), acc
    else:
        yield "acc_subnormal", codes((n, 16, k)), codes((n, 8, k)), scales((n, 16, nblk), 40, 80), scales((n, 8, nblk), 40, 80), acc

    if not ue4:
        # --- D18 / D13: UE8M0 scale extremes.  Every code 0..254 on one operand against 0..115 on the other (ca + cb <= 369 keeps
        # |sum| < 2^127), then mirrored; random accumulators over 2^-126 .. 2^100 on half the tiles.  Sums below the FP32
        # subnormal grid decide the sign of the underflowed zero (BLACKWELL_SM120_MXF4 vs ..._UNDERFLOW_SIGNED).
        n = n_random // 2
        sfa = rng.integers(0, 255, (n, 16, nblk)).astype(np.uint8)
        sfb = rng.integers(0, 116, (n, 8, nblk)).astype(np.uint8)
        swap = np.arange(n) % 2 == 1
        sfa[swap] = rng.integers(0, 116, (int(swap.sum()), 16, nblk))
        sfb[swap] = rng.integers(0, 255, (int(swap.sum()), 8, nblk))
        acc = zeros(n)
        acc[n // 2:] = _signed_pow2_acc(rng, (n - n // 2, 16, 8), -126, 100, exact_frac=0.2)
        yield "ue8m0_scale_extremes", codes((n, 16, k)), codes((n, 8, k)), sfa, sfb, acc

        # --- D13: nonzero sums entirely below the subnormal grid, both signs, zero and tiny accumulators
        n = n_random // 8
        sfa = rng.integers(0, 60, (n, 16, nblk)).astype(np.uint8)
        sfb = rng.integers(0, 60, (n, 8, nblk)).astype(np.uint8)
        acc = zeros(n)
        acc[n // 2:] = random_f32_bits(rng, (n - n // 2, 16, 8), 0, 2)
        yield "ue8m0_underflow", codes((n, 16, k)), codes((n, 8, k)), sfa, sfb, acc

    # --- domain edges (record only): NaN / +-inf accumulators over random products; +-FLT_MAX / +-2^127 accumulators with
    # the largest same-sign products (nvf4 cannot overflow on its own, mxf4 can: 6*6 x 64 under 2^127 x 2^127 scales);
    # the second half alternates the sign of the whole tile so both overflow directions are observed
    n = 128
    A, Bt = codes((n, 16, k)), codes((n, 8, k))
    special = np.array([0x7FC00000, 0xFFC00000, 0x7F800001, 0x7FBFFFFF, 0x7F800000, 0xFF800000, 0x7F7FFFFF, 0xFF7FFFFF], np.uint32)
    acc = special[rng.integers(0, len(special), (n, 16, 8))]
    sel = np.arange(n) >= n // 2
    negative_tile = (np.arange(n) % 2 == 1) & sel
    A[sel] = 0x7
    A[negative_tile] = 0xF                                   # -6.0 x +6.0: every product -36
    Bt[sel] = 0x7
    acc[sel] = np.where(rng.random((int(sel.sum()), 16, 8)) < 0.5, np.uint32(0x7F7FFFFF), np.uint32(0x7F000000))
    acc[negative_tile] |= np.uint32(0x80000000)
    if ue4:
        sfa, sfb = scales((n, 16, nblk)), scales((n, 8, nblk))
        sfa[sel], sfb[sel] = 0x7E, 0x7E
    else:
        sfa, sfb = scales((n, 16, nblk)), scales((n, 8, nblk))
        sfa[sel] = rng.integers(200, 255, (int(sel.sum()), 16, nblk))
        sfb[sel] = rng.integers(200, 255, (int(sel.sum()), 8, nblk))
    yield "specials", A, Bt, sfa, sfb, acc

    yield ("scale_split_twins", *scale_split_twins(rng, n_random // 8, scale_kind))

    if ue4:
        yield ("scale_bit7", *scale_bit7_twins(rng, max(1, n_random // 32)))
    else:
        yield from families_mxf4_edges(rng)


#: accumulators of the mxf4 edge families: both zeros, +-1, NaNs of both signs (quiet and signalling), both infinities, +-FLT_MAX,
#: the least subnormal
_EDGE_ACC = np.array([0x00000000, 0x80000000, 0x3F800000, 0xBF800000, 0x7FC00000, 0xFFC00000, 0x7F800001, 0x7F800000, 0xFF800000,
                      0x7F7FFFFF, 0xFF7FFFFF, 0x00000001], np.uint32)
#: accumulators at the top of binary32: +-FLT_MAX, +-2^127, +-1.5 2^127, +-(2^127 - 2^103), FLT_MAX - 1 ulp, +0
_TOP_ACC = np.array([0x7F7FFFFF, 0xFF7FFFFF, 0x7F000000, 0xFF000000, 0x7F400000, 0xFF400000, 0x7EFFFFFF, 0xFEFFFFFF, 0x7F7FFFFE,
                     0x00000000], np.uint32)


def families_mxf4_edges(rng):
    """Record-only (UE8M0, two blocks of 32): what a total mxf4 step decides and no finite-domain family reaches.  Yield
    (name, A, Bt, SFA, SFB, C) like ``families_nvf4``.

    ``mxf4_nan_scale``: the UE8M0 NaN code 0xFF on even rows / columns only (odd ones are the controls of the same tile), by block
    and operand, over blocks with nonzero products, a zero A or B block, no product at all, an exactly cancelling block, every
    scale 0xFF, and 0xFF beside a block that overflows.  ``mxf4_overflow_cancel``: block dots that leave binary32 on their own
    (scales 185..254) and cancel exactly -- block 1 = -block 0 under equal scales, group 1 = -group 0 inside block 0, block 1 at
    twice the codes under half the scale -- and a cancellation leaving a residual inside binary32.  ``mxf4_overflow_edge``: one
    or two products of 1 or 1.5 at 2^(100..129) (row i and column j set the exponent) against the accumulators at the top of
    binary32, both signs, so that exact sums fall on both sides of 2^128 and inside (FLT_MAX, 2^128); and random codes whose sums
    straddle it.  ``mxf4_inf_acc_overflow``: infinite and NaN accumulators against positive and negative overflows, an exact
    cancellation of overflowing blocks, and moderate products."""
    k, nblk = 64, 2

    def nz(shape, lo=1):
        return (rng.integers(lo, 8, shape) | (rng.integers(0, 2, shape) << 3)).astype(np.uint8)

    def neg(c):
        return np.where(c & 7, c ^ 8, c).astype(np.uint8)

    def acc_from(pool, n, zero_tiles=None):
        acc = pool[rng.integers(0, len(pool), (n, 16, 8))]
        if zero_tiles is not None:
            acc[zero_tiles] = 0
        return acc.astype(np.uint32)

    n = 8 * 16
    v = np.arange(n) // 16
    A, Bt = nz((n, 16, k)), nz((n, 8, k))
    sfa, sfb = rng.integers(110, 145, (n, 16, nblk)).astype(np.uint8), rng.integers(110, 145, (n, 8, nblk)).astype(np.uint8)
    sel = [v == i for i in range(8)]
    sfa[sel[0], 0::2, 0] = 0xFF                                  # A block 0
    sfb[sel[1], 0::2, 1] = 0xFF                                  # B block 1
    sfa[sel[2], 0::2, 0] = 0xFF
    A[sel[2], :, :32] = 0                                        # ... over A's zero codes
    sfa[sel[3], 0::2, 0] = 0xFF
    Bt[sel[3], :, :32] = 0                                       # ... over B's zero codes
    sfa[sel[4]], sfb[sel[4]] = 0xFF, 0xFF                        # every scale
    A[sel[5]], Bt[sel[5]] = 0, 0
    sfa[sel[5], 0::2, 0] = 0xFF                                  # no product anywhere
    sfa[sel[6], 0::2, 0] = 0xFF                                  # beside a block 1 that overflows
    A[sel[6], :, 32:], Bt[sel[6], :, 32:] = 0x7, 0x7
    sfa[sel[6], :, 1] = rng.integers(230, 255, (int(sel[6].sum()), 16))
    sfb[sel[6], :, 1] = rng.integers(230, 255, (int(sel[6].sum()), 8))
    sfa[sel[7], 0::2, 0] = 0xFF                                  # over a block 0 that cancels exactly
    A[sel[7], :, 1:32:2] = A[sel[7], :, 0:32:2]
    Bt[sel[7], :, 1:32:2] = neg(Bt[sel[7], :, 0:32:2])
    yield "mxf4_nan_scale", A, Bt, sfa, sfb, acc_from(_EDGE_ACC, n)

    n = 128
    v = np.arange(n) % 4
    A, Bt = nz((n, 16, k), 5) & 7, nz((n, 8, k), 5) & 7          # |x| in {3, 4, 6}, positive: every block dot one sign
    ca, cb = rng.integers(200, 255, (n, 16)), rng.integers(200, 255, (n, 8))
    sel = v == 2                                                 # the residual 2^(ca+cb-256) inside binary32, each block outside
    ca[sel], cb[sel] = rng.integers(187, 192, (int(sel.sum()), 16)), rng.integers(188, 193, (int(sel.sum()), 8))
    sfa, sfb = np.repeat(ca[:, :, None], 2, 2).astype(np.uint8), np.repeat(cb[:, :, None], 2, 2).astype(np.uint8)
    for s in (v == 0, v == 2):                                   # block 1 = -block 0
        A[s, :, 32:] = A[s, :, :32]
        Bt[s, :, 32:] = neg(Bt[s, :, :32])
    A[v == 2, :, 0], Bt[v == 2, :, 0], A[v == 2, :, 32] = 0x1, 0x1, 0
    sel = v == 1                                                 # group 1 = -group 0 inside block 0, block 1 empty
    A[sel, :, 16:32] = A[sel, :, :16]
    Bt[sel, :, 16:32] = neg(Bt[sel, :, :16])
    A[sel, :, 32:] = 0
    sel = v == 3                                                 # block 1: twice the codes under half the A scale
    A[sel, :, :32] = rng.integers(1, 6, (int(sel.sum()), 16, 32))
    A[sel, :, 32:] = np.array([0, 2, 4, 5, 6, 7], np.uint8)[A[sel, :, :32]]
    Bt[sel, :, 32:] = neg(Bt[sel, :, :32])
    sfa[sel, :, 1] = sfa[sel, :, 0] - 1
    yield "mxf4_overflow_cancel", A, Bt, sfa, sfb, acc_from(_EDGE_ACC, n, zero_tiles=np.arange(n) < n // 2)

    n = 256
    v = np.arange(n) % 4
    A, Bt = np.zeros((n, 16, k), np.uint8), np.zeros((n, 8, k), np.uint8)
    sfa = np.broadcast_to((190 + np.arange(16))[None, :, None], (n, 16, nblk)).astype(np.uint8)
    sfb = np.broadcast_to((164 + 2 * np.arange(8))[None, None, :].transpose(0, 2, 1), (n, 8, nblk)).astype(np.uint8)
    negative = (np.arange(n) // 4) % 2 == 1
    A[v == 0, :, 0] = 0x2                                        # 1 x 2^(ca+cb-254) = 2^(100 + i + 2j)
    A[v == 1, :, 0], A[v == 1, :, 32] = 0x2, 0x2                 # two of them, one per block
    A[v == 2, :, 0] = 0x3                                        # 1.5 x 2^(100 + i + 2j)
    Bt[v != 3, :, 0], Bt[v == 1, :, 32] = 0x2, 0x2
    Bt[negative & (v != 3), :, 0] = 0xA
    Bt[negative & (v == 1), :, 32] = 0xA
    sel = v == 3                                                 # random codes, sums straddling 2^128
    A[sel], Bt[sel] = nz((int(sel.sum()), 16, k)), nz((int(sel.sum()), 8, k))
    sfa, sfb = sfa.copy(), sfb.copy()
    sfa[sel] = rng.integers(185, 200, (int(sel.sum()), 16, nblk))
    sfb[sel] = rng.integers(170, 186, (int(sel.sum()), 8, nblk))
    yield "mxf4_overflow_edge", A, Bt, sfa, sfb, acc_from(_TOP_ACC, n)

    n = 64
    v = np.arange(n) % 4
    A, Bt = np.full((n, 16, k), 0x7, np.uint8), np.full((n, 8, k), 0x7, np.uint8)
    Bt[v == 1] = 0xF                                             # every product -36
    sfa, sfb = rng.integers(200, 255, (n, 16, nblk)).astype(np.uint8), rng.integers(200, 255, (n, 8, nblk)).astype(np.uint8)
    sel = v == 2                                                 # block 1 = -block 0: overflowing blocks cancelling exactly
    Bt[sel, :, 32:] = 0xF
    sfa[sel, :, 1], sfb[sel, :, 1] = sfa[sel, :, 0], sfb[sel, :, 0]
    sel = v == 3                                                 # moderate products
    A[sel], Bt[sel] = nz((int(sel.sum()), 16, k)), nz((int(sel.sum()), 8, k))
    sfa[sel], sfb[sel] = rng.integers(110, 145, (int(sel.sum()), 16, nblk)), rng.integers(110, 145, (int(sel.sum()), 8, nblk))
    inf_nan = np.array([0x7F800000, 0xFF800000, 0x7FC00000, 0xFFC00000, 0x7F800001, 0xFF800001], np.uint32)
    yield "mxf4_inf_acc_overflow", A, Bt, sfa, sfb, acc_from(inf_nan, n)


#: E2M1 magnitude index -> the index of twice its magnitude (0, .5, 1, 1.5, 2, 3 -> 0, 1, 2, 3, 4, 6); 4 and 6 have none.
_E2M1_DOUBLE = np.array([0, 2, 4, 5, 6, 7, 0, 0], np.uint8)
TWIN_OFFSETS = (27, 28, 29)


def scale_split_twins(rng, n: int, scale_kind: str):
    """The Pipeline-fit discriminator: tile ``2t + 1`` is tile ``2t`` with every A nibble doubled and every A scale halved.

    Every scaled product is the same in both, so any model that reads only the exact products and the accumulator (every
    ``GroupSum``, at any groups, width and floor) gives both twins one word; the scale-exponent anchor of
    ``BlockScaledAlignAdd`` moves one binade down.  The accumulator is ``+-2^(X_max - off)`` with ``off`` = 27, 28, 29 by
    tile pair (``TWIN_OFFSETS``): at 28 it lies on the second twin's grid only, so the words differ wherever the sum is exact
    in binary32 and the accumulator opposes it; at 27 and 29 the twins agree.  Half the pairs keep A's nonzero nibbles in
    group 0, so the sum is often exact.  Returns ``(A, Bt, SFA, SFB, C)``.
    """
    k, nblk = 64, (4 if scale_kind == "ue4m3" else 2)
    h = max(n // 2, 1)
    mag = rng.integers(0, 6, (h, 16, k)).astype(np.uint8)
    mag[: h // 2, :, 16:] = 0
    mag = np.where(rng.random((h, 16, k)) < 0.25, 0, mag).astype(np.uint8)
    sign = (rng.integers(0, 2, (h, 16, k)) << 3).astype(np.uint8)
    a1, a2 = sign | mag, sign | _E2M1_DOUBLE[mag]
    bt = rng.integers(0, 16, (h, 8, k)).astype(np.uint8)
    if scale_kind == "ue4m3":
        sfa1 = (rng.integers(6, 11, (h, 16, nblk)) << 3 | rng.integers(0, 8, (h, 16, nblk))).astype(np.uint8)
        sfb = (rng.integers(6, 11, (h, 8, nblk)) << 3 | rng.integers(0, 8, (h, 8, nblk))).astype(np.uint8)
        sfa2 = sfa1 - np.uint8(8)
        ea, eb = (sfa1 >> 3).astype(np.int64) - 10, (sfb >> 3).astype(np.int64) - 10
    else:
        sfa1 = rng.integers(120, 136, (h, 16, nblk)).astype(np.uint8)
        sfb = rng.integers(120, 136, (h, 8, nblk)).astype(np.uint8)
        sfa2 = sfa1 - np.uint8(1)
        ea, eb = sfa1.astype(np.int64) - 130, sfb.astype(np.int64) - 130
    gpb = 4 // nblk
    a_nz = (model5.E2M1_NUM[a1.astype(np.int64)] != 0).reshape(h, 16, 4, 16).astype(np.int64)
    b_nz = (model5.E2M1_NUM[bt.astype(np.int64)] != 0).reshape(h, 8, 4, 16).astype(np.int64)
    live = np.einsum("nigk,njgk->nijg", a_nz, b_nz) > 0
    blk = np.arange(4) // gpb
    x = -2 + ea[:, :, None, blk] + eb[:, None, :, blk]
    x_max = np.where(live, x, -(1 << 20)).max(-1)
    x_max = np.where(live.any(-1), x_max, -20)
    off = np.array(TWIN_OFFSETS)[np.arange(h) % len(TWIN_OFFSETS)][:, None, None]
    acc = ((rng.integers(0, 2, (h, 16, 8)).astype(np.uint32) << 31) | ((x_max - off + 127).astype(np.uint32) << 23)).astype(np.uint32)

    def twin(u, v):
        out = np.empty((2 * h, *u.shape[1:]), u.dtype)
        out[0::2], out[1::2] = u, v
        return out

    return twin(a1, a2), twin(bt, bt), twin(sfa1, sfa2), twin(sfb, sfb), twin(acc, acc)


#: UE4M3 bytes with bit 7 set that every scale_bit7 pair carries on block 0 of rows 0-2: 0x80 (low 7 bits 0x00, the zero
#: scale), 0xB8 (0x38, 1.0) and 0xFF (0x7F, NaN)
BIT7_PINNED = (0x80, 0xB8, 0xFF)


def scale_bit7_twins(rng, n: int):
    """Tile ``2t`` has bit 7 (the PTX ISA's padding bit, which sm_120 and the pinned model ignore) set on about half its UE4M3
    scale bytes, the ``BIT7_PINNED`` bytes on block 0 of A's and B's rows 0-2; tile ``2t + 1`` is the same tile with every byte's
    low 7 bits.  Equal words across every pair are what a scale decoder that ignores bit 7 produces."""
    k, nblk = 64, 4
    A = rng.integers(0, 16, (n, 16, k)).astype(np.uint8)
    Bt = rng.integers(0, 16, (n, 8, k)).astype(np.uint8)
    lo_a = UE4M3_VALID[rng.integers(0, len(UE4M3_VALID), (n, 16, nblk))]
    lo_b = UE4M3_VALID[rng.integers(0, len(UE4M3_VALID), (n, 8, nblk))]
    hi_a = lo_a | np.where(rng.random(lo_a.shape) < 0.5, 0x80, 0).astype(np.uint8)
    hi_b = lo_b | np.where(rng.random(lo_b.shape) < 0.5, 0x80, 0).astype(np.uint8)
    for r, byte in enumerate(BIT7_PINNED):
        hi_a[:, r, 0] = hi_b[:, r, 0] = byte
        lo_a[:, r, 0] = lo_b[:, r, 0] = byte & 0x7F
    acc = f32_bits((rng.standard_normal((n, 16, 8)) * np.exp2(rng.integers(-10, 10, (n, 16, 8)))).astype(np.float32))

    def twin(x1, x2):
        out = np.empty((2 * n,) + x1.shape[1:], x1.dtype)
        out[0::2], out[1::2] = x1, x2
        return out

    return twin(A, A), twin(Bt, Bt), twin(hi_a, lo_a), twin(hi_b, lo_b), twin(acc, acc)


def bit7_summary(D, SFA, SFB, model, err) -> dict:
    """``scale_bit7``: words where a pair's tiles differ on the device (none: the hardware decodes a UE4M3 byte from its low 7
    bits), and the low-7-bit tiles' mismatches against the pinned model outside its domain errors (the NaN byte 0x7F)."""
    differ = D[0::2] != D[1::2]
    bytes7 = np.unique(np.concatenate([SFA[0::2].ravel(), SFB[0::2].ravel()]))
    rec = {"pairs": int(D.shape[0] // 2), "elements_per_twin": int(D.shape[0] // 2) * 128, "device_words_differ": int(differ.sum()),
           "bit7_bytes_carried": int((bytes7 >= 0x80).sum()), "pinned_bytes": [f"0x{b:02X}" for b in BIT7_PINNED],
           "hardware_ignores_bit7": not bool(differ.any())}
    if model is not None:
        ok = ~err[1::2]
        rec["low7_twin_vs_model"] = {"checked": int(ok.sum()), "mismatches": int(((model[1::2] != D[1::2]) & ok).sum()),
                                     "domain_errors": int((~ok).sum())}
        rec["bit7_tiles_model_domain_errors"] = int(err[0::2].sum())
    first = []
    for t, i, j in list(zip(*np.nonzero(differ)))[:10]:
        first.append({"pair": int(t), "i": int(i), "j": int(j), "word_bit7": f"{int(D[2 * t, i, j]):08x}",
                      "word_low7": f"{int(D[2 * t + 1, i, j]):08x}",
                      "sfa": SFA[2 * t, i].tobytes().hex(), "sfb": SFB[2 * t, j].tobytes().hex()})
    rec["first_differences"] = first
    return rec


def twin_summary(D, models: dict) -> dict:
    """Words where the twins of ``scale_split_twins`` differ, by accumulator offset, on the device and under each model.

    A difference on the device is a pair of inputs with the same exact products and accumulator but different words, so no
    model reading only those (no ``GroupSum``) computes the instruction."""

    off = np.array(TWIN_OFFSETS)[np.arange(D.shape[0] // 2) % len(TWIN_OFFSETS)]

    def differ(words):
        d = words[0::2] != words[1::2]
        return {f"off{o}": int(d[off == o].sum()) for o in TWIN_OFFSETS}

    hw = differ(D)
    return {"pairs": int(D.shape[0] // 2), "elements_per_twin": int(D.shape[0] // 2) * 128, "device_words_differ": hw,
            "model_words_differ": {m: differ(v) for m, v in models.items()},
            "no_product_value_model_fits": any(hw.values())}


def families_bf16(rng, n_random: int):
    k = 16
    cast = lambda x: to_bf16_bits(x.astype(np.float32))
    n = n_random
    yield "randn_random_acc", cast(rng.standard_normal((n, 16, k))), cast(rng.standard_normal((n, 8, k))), None, None, \
        f32_bits((rng.standard_normal((n, 16, 8)) * np.exp2(rng.integers(-20, 21, (n, 16, 8)))).astype(np.float32))
    n = n_random // 2
    yield "uniform_bits", random_bf16_bits(rng, (n, 16, k)), random_bf16_bits(rng, (n, 8, k)), None, None, random_f32_bits(rng, (n, 16, 8))
    n = n_random // 4
    a = rng.standard_normal((n, 16, k)) * 2.0**8
    b = rng.standard_normal((n, 8, k)) * 2.0**8
    a[:, :, 1::2] = a[:, :, 0::2]
    b[:, :, 1::2] = -b[:, :, 0::2]
    a = np.where(rng.random((n, 16, k)) < 0.15, rng.standard_normal((n, 16, k)) * 2.0**-6, a)
    yield "cancellation", cast(a), cast(b), None, None, f32_bits((rng.standard_normal((n, 16, 8)) * 2.0**-4).astype(np.float32))
    n = n_random // 4
    yield "subnormal", random_bf16_bits(rng, (n, 16, k), 0, 3), random_bf16_bits(rng, (n, 8, k), 0, 3), None, None, random_f32_bits(rng, (n, 16, 8), 0, 3)


# --- the model, per tile, in processes ---------------------------------------------------------------------------------------------

_CTX: dict = {}


#: the model whose mismatches gate a run; None: a capture whose finding is which candidates agree (``--expect-model`` sets one)
PRIMARY_MODEL = {"nvf4": "verity", "mxf4": "verity", "f8f6f4": None, "bf16": None,
                 "tcgen05_nvf4": "verity", "tcgen05_mxf4": "verity", "tcgen05_mxf4kind": "verity",
                 **{instr: None for instr in GENERIC_ONLY}}


def verity_models(instr: str) -> dict:
    """The models evaluated for ``instr``: for the mxf4nvf4 rows ``verity`` is the registry's ``BlockScaledAlignAdd`` and the others
    named alternatives (mxf4: the sign of an underflowed zero, log entry D13; for the sm_100 instructions the sm_120 models are the
    hypothesis under test); the K=32 rows (``f8f6f4`` and the rest of ``k32.ROWS``) evaluate ``k32.candidates``, whose
    ``bsaa_g16`` is the scale-free instance earlier runs called ``verity``."""
    from verity.ml.circuits.silicon import models as M
    if instr in ("nvf4", "tcgen05_nvf4"):
        return {"verity": M.BLACKWELL_SM120_NVF4}
    if instr in ("mxf4", "tcgen05_mxf4", "tcgen05_mxf4kind"):
        return {"verity": M.BLACKWELL_SM120_MXF4, "verity_underflow_signed": M.BLACKWELL_SM120_MXF4_UNDERFLOW_SIGNED}
    if instr in k32.ROWS:
        return k32.candidates(instr)
    return {}


def model_words(model, A, Bt, SFA, SFB, C, k_mma: int, nblk: int) -> tuple[np.ndarray, np.ndarray]:
    """One model's word per element, stepped once per k-block with D fed back as C: (D (n,16,8), domain-error mask)."""
    n = A.shape[0]
    steps = A.shape[2] // k_mma
    out = np.zeros((n, 16, 8), np.uint32)
    err = np.zeros((n, 16, 8), np.bool_)
    Al, Bl, Cl = A.tolist(), Bt.tolist(), C.tolist()
    SAl = SFA.tolist() if SFA is not None else None
    SBl = SFB.tolist() if SFB is not None else None
    for t in range(n):
        for i in range(16):
            for j in range(8):
                c = Cl[t][i][j]
                try:
                    for s in range(steps):
                        ks, bs = slice(s * k_mma, (s + 1) * k_mma), slice(s * nblk, (s + 1) * nblk)
                        c = model.step_scaled(c, Al[t][i][ks], Bl[t][j][ks], SAl[t][i][bs] if SAl is not None else (),
                                              SBl[t][j][bs] if SBl is not None else ())
                    out[t, i, j] = c
                except Exception:  # noqa: BLE001 - outside the model's domain (non-finite accumulator, sum leaving binary32)
                    err[t, i, j] = True
    return out, err


class FakeProbe:
    """A CPU stand-in for the device (``--fake-silicon MODEL``): every word is MODEL's, so a dry run exercises the build, the gates,
    the families, the comparison and the verdict without a GPU.  ``main`` refuses it under a research run: it never publishes."""

    tcgen05 = False

    def __init__(self, instr: str, model_name: str):
        models = verity_models(instr)
        if model_name not in models:
            raise SystemExit(f"--fake-silicon: {model_name!r} is not a model of {instr} ({sorted(models)})")
        self.model_name, self.model = model_name, models[model_name]
        self.debug_dir = None

    def identity(self) -> dict:
        return {"device": f"FAKE SILICON ({self.model_name})", "compute_capability": "fake", "nvidia_smi": None, "driver": None,
                "gpu_uuid": "GPU-fake", "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "hostname": platform.node(),
                "has_mxf4": True, "backend": "fake"}

    def clocks(self, uuid):
        return None

    def tiles(self, instr: str, A, Bt, C, SFA=None, SFB=None, steps: int = 1, **_kw) -> np.ndarray:
        D, err = model_words(self.model, np.asarray(A), np.asarray(Bt), SFA, SFB, np.asarray(C, np.uint32), K_OF[instr], NBLK[instr])
        D[err] = 0x7FFFFFFF
        return D


def _init(instr: str, group: int):
    _CTX["instr"] = instr
    _CTX["group"] = group
    try:
        if instr == "bf16":
            from verity.ml.circuits.silicon import silicon as S
            _CTX["pipes"] = {"ampere_bf16_m16n8k16": S.AMPERE_BF16_M16N8K16, "hopper_bf16_m16n8k16": S.HOPPER_BF16_M16N8K16}
            _CTX["tc_dot"] = S.tc_dot
        else:
            _CTX["verity"] = verity_models(instr)
    except Exception as e:  # noqa: BLE001 - a failing initializer would make the pool respawn workers forever; fail the chunk instead
        _CTX["init_error"] = repr(e)


def _eval_chunk(args):
    """Every model's word per element of the chunk: ({model: D (n,16,8)}, {model: domain-error mask})."""
    if "init_error" in _CTX:
        raise RuntimeError(f"model worker failed to initialize: {_CTX['init_error']}")
    A, Bt, SFA, SFB, C = args
    n = A.shape[0]
    instr = _CTX["instr"]
    if instr == "bf16":
        outs = {name: np.zeros((n, 16, 8), np.uint32) for name in _CTX["pipes"]}
        errs = {name: np.zeros((n, 16, 8), np.bool_) for name in _CTX["pipes"]}
        for t in range(n):
            a_rows = [A[t, i].tolist() for i in range(16)]
            b_rows = [Bt[t, j].tolist() for j in range(8)]
            for i in range(16):
                for j in range(8):
                    for name, pipe in _CTX["pipes"].items():
                        try:
                            outs[name][t, i, j] = _CTX["tc_dot"](pipe, int(C[t, i, j]), a_rows[i], b_rows[j])
                        except Exception:  # noqa: BLE001
                            errs[name][t, i, j] = True
        return outs, errs
    # K may be a multiple of the instruction's K (--chain, tcgen05 --chain2): the models are stepped once per k-block, D fed back as C
    k_mma = K_OF[instr]
    steps = A.shape[2] // k_mma
    nblk = NBLK[instr]
    outs, errs = {}, {}
    if instr in MODEL5_ROWS:
        out5 = np.zeros((n, 16, 8), np.uint32)
        err5 = np.zeros((n, 16, 8), np.bool_)
        empty = np.zeros((16, 0), np.uint8)
        for t in range(n):
            c = C[t]
            e = np.zeros((16, 8), np.bool_)
            try:
                for s in range(steps):
                    ks, bs = slice(s * k_mma, (s + 1) * k_mma), slice(s * nblk, (s + 1) * nblk)
                    c, e_s = model5.tile_eval(A[t][:, ks], Bt[t][:, ks], SFA[t][:, bs] if SFA is not None else empty,
                                              SFB[t][:, bs] if SFB is not None else empty[:8], c, SCALE_KIND[instr], _CTX["group"])
                    e |= e_s
            except AssertionError:  # model5 asserts on a scale code outside its domain (scale_bit7): the whole tile is refused
                c, e = np.zeros((16, 8), np.uint32), np.ones((16, 8), np.bool_)
            out5[t], err5[t] = c, e
        outs["model5"], errs["model5"] = out5, err5
    for name, model in _CTX["verity"].items():
        outs[name], errs[name] = model_words(model, A, Bt, SFA, SFB, C, k_mma, nblk)
    return outs, errs


def model_eval(pool, A, Bt, SFA, SFB, C, procs):
    n = A.shape[0]
    chunks = max(1, min(n // 16 + 1, procs * 8))
    idx = np.array_split(np.arange(n), chunks)
    parts = pool.map(_eval_chunk, [(A[i], Bt[i], SFA[i] if SFA is not None else None, SFB[i] if SFB is not None else None, C[i]) for i in idx])
    outs = {name: np.concatenate([p[0][name] for p in parts]) for name in parts[0][0]}
    errs = {name: np.concatenate([p[1][name] for p in parts]) for name in parts[0][1]}
    return outs, errs


def observed_summary(D: np.ndarray, C: np.ndarray, err: np.ndarray) -> dict:
    """What the hardware returned where the model has no answer: word classes and the most frequent words."""
    d = D[err].astype(np.uint32)
    if d.size == 0:
        return {"elements": 0}
    exp = (d >> 23) & 0xFF
    frac = d & 0x7FFFFF
    cls = {"nan": int(((exp == 0xFF) & (frac != 0)).sum()), "pos_inf": int((d == 0x7F800000).sum()), "neg_inf": int((d == 0xFF800000).sum()),
           "finite": int((exp != 0xFF).sum())}
    words, counts = np.unique(d, return_counts=True)
    top = sorted(zip(counts.tolist(), words.tolist()), reverse=True)[:12]
    acc_in = {}
    for a in np.unique(C[err]).tolist()[:16]:
        sel = err & (C == a)
        w, c = np.unique(D[sel], return_counts=True)
        acc_in[f"{a:08x}"] = {f"{int(x):08x}": int(y) for x, y in sorted(zip(w.tolist(), c.tolist()), key=lambda p: -p[1])[:6]}
    return {"elements": int(d.size), "classes": cls, "top_words": {f"{w:08x}": c for c, w in top}, "by_accumulator": acc_in}


# --- records -----------------------------------------------------------------------------------------------------------------------


#: result.json must stay under the store's 1 MiB; a --chain record's operands run to tens of KiB each
RESULT_OPERAND_HEX_MAX = 512


def slim_record(rec: dict) -> dict:
    """``rec`` for result.json: an operand string longer than RESULT_OPERAND_HEX_MAX becomes its length and sha256
    (probe_results.json and tiles_<family>.npz keep it whole)."""

    out = {}
    for k, v in rec.items():
        if k in ("a", "b", "sfa", "sfb") and isinstance(v, str) and len(v) > RESULT_OPERAND_HEX_MAX:
            out[k] = {"hex_chars": len(v), "sha256": hashlib.sha256(v.encode()).hexdigest()}
        else:
            out[k] = v
    return out


def hexrec(acc, a, b, d, sfa=None, sfb=None, bf16=False, widths=(1, 1)) -> dict:
    wa, wb = (4, 4) if bf16 else widths
    rec = {"acc": f"{int(acc):08x}", "a": "".join(f"{int(x):0{wa}x}" for x in a), "b": "".join(f"{int(x):0{wb}x}" for x in b), "d": f"{int(d):08x}"}
    if sfa is not None:
        rec["sfa"] = "".join(f"{int(x):02x}" for x in sfa)
        rec["sfb"] = "".join(f"{int(x):02x}" for x in sfb)
    return rec


# --- autoproof replay ----------------------------------------------------------------------------------------------------------------


def replay_autoproof(probe: Probe, path: Path) -> dict:
    """Every ``sample_cases`` record at its own tile position, zero accumulator, dependent launches in ascending K (D -> C)."""
    data = json.loads(path.read_text())
    probe_no = int(data["probe"])
    instr = "nvf4" if probe_no == 3 else "f8f6f4"
    if probe.tcgen05:
        if probe_no != 3:
            return {"file": path.name, "probe": probe_no, "skipped": "the sm_100 backend has no unscaled kind::f8f6f4 e2m1 kernel", "cases": 0,
                    "matched": 0, "mismatches": 0, "first_mismatches": [], "other_positions_nonzero": 0, "by_k": {}, "gpu_seconds": 0.0}
        instr = "tcgen05_nvf4"
    k_mma = K_OF[instr]
    cases = data["sample_cases"]
    by_k: dict[int, list[int]] = {}
    for r, c in enumerate(cases):
        by_k.setdefault(len(c["a_codes"]), []).append(r)
    mism = []
    other_nonzero = 0
    matched = 0
    t0 = time.time()
    for k_total, rows in sorted(by_k.items()):
        n = len(rows)
        steps = k_total // k_mma
        A = np.zeros((n, 16, k_total), np.uint8)
        Bt = np.zeros((n, 8, k_total), np.uint8)
        scaled = SCALE_KIND.get(instr) == "ue4m3"  # nvf4 on either backend; probe2's f8f6f4 is unscaled
        nblk_total = k_total // 16 if scaled else 0
        SFA = np.zeros((n, 16, max(nblk_total, 1)), np.uint8)
        SFB = np.zeros((n, 8, max(nblk_total, 1)), np.uint8)
        pos = [(r % 16, (r // 16) % 8) for r in rows]
        for t, r in enumerate(rows):
            i, j = pos[t]
            A[t, i] = cases[r]["a_codes"]
            Bt[t, j] = cases[r]["b_codes"]
            if scaled:
                SFA[t, i] = cases[r]["a_scales"]
                SFB[t, j] = cases[r]["b_scales"]
        C = np.zeros((n, 16, 8), np.uint32)
        for s in range(steps):
            ks = slice(s * k_mma, (s + 1) * k_mma)
            if scaled:
                bs = slice(s * 4, (s + 1) * 4)
                D = probe.tiles(instr, A[:, :, ks], Bt[:, :, ks], C, SFA[:, :, bs], SFB[:, :, bs])
            else:
                D = probe.tiles(instr, A[:, :, ks], Bt[:, :, ks], C)
            C = D
        for t, r in enumerate(rows):
            i, j = pos[t]
            want = int(cases[r]["silicon_c_bits"]) & 0xFFFFFFFF
            got = int(D[t, i, j])
            if got == want:
                matched += 1
            elif len(mism) < 20:
                mism.append({"index": r, "k": k_total, "autoproof_silicon": f"{want:08x}", "hardware": f"{got:08x}",
                             "a": "".join(f"{x:x}" for x in cases[r]["a_codes"]), "b": "".join(f"{x:x}" for x in cases[r]["b_codes"]),
                             "sfa": cases[r]["a_scales"], "sfb": cases[r]["b_scales"]})
            rest = D[t].copy()
            rest[i, j] = 0
            other_nonzero += int(np.count_nonzero(rest))
    return {"file": path.name, "probe": probe_no, "instruction": PTX[instr], "autoproof_target": data.get("target"),
            "autoproof_kernel_sha256": data.get("kernel_sha256"), "autoproof_gpu_uuid": data.get("gpu_uuid"),
            "cases": len(cases), "by_k": {str(k): len(v) for k, v in by_k.items()}, "matched": matched,
            "mismatches": len(cases) - matched, "first_mismatches": mism, "other_positions_nonzero": other_nonzero,
            "gpu_seconds": round(time.time() - t0, 2)}


# --- main --------------------------------------------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instruction", default="nvf4",
                    choices=sorted(PTX) + list(CAST_INSTRUCTIONS) + list(SPARSE_INSTRUCTIONS) + list(T1_INSTRUCTIONS)
                    + list(META_RULE_INSTRUCTIONS) + list(LEAN_VECTOR_INSTRUCTIONS))
    ap.add_argument("--arch", default=None, help="nvcc -arch (default sm_120a; sm_100a for the tcgen05 instructions, sm_103a on a B300)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--chain2", action="store_true", help="tcgen05: pair each family's tiles into K=128 chains (two MMAs on one TMEM accumulator)")
    ap.add_argument("--smem-strides", default=None, metavar="LBO_A,SBO_A,LBO_B,SBO_B",
                    help="tcgen05: shared-memory descriptor strides in bytes (default 128,256,128,256: canonical K-major no-swizzle)")
    ap.add_argument("--replay-autoproof", action="append", default=[], metavar="JSON")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--n-random", type=int, default=4096, help="tiles of the largest family (128 elements each)")
    ap.add_argument("--seed", type=int, default=20260922)
    ap.add_argument("--group", type=int, default=16, help="model5 group width hypothesis (16 = the ue4m3 4X structure)")
    ap.add_argument("--procs", type=int, default=max(1, min(32, (os.cpu_count() or 4) - 2)))
    ap.add_argument("--golden-per-family", type=int, default=60)
    ap.add_argument("--families", default=None, help="comma list (default all)")
    ap.add_argument("--sass-check", action="store_true")
    ap.add_argument("--chain", type=int, default=1, metavar="L",
                    help="mma.sync: regroup each family's tiles into chains of L dependent MMAs kept in registers (generic kernel)")
    ap.add_argument("--expect-uuid", default=None,
                    help="stop before any measurement unless CUDA's device has this UUID (or one of a comma list)")
    ap.add_argument("--expect-model", default=None, help="the model whose mismatches gate this run (default: the row's primary model)")
    ap.add_argument("--fake-silicon", default=None, metavar="MODEL", help="dry run on the CPU: MODEL's words stand in for the device")
    ap.add_argument("--corrupt-word", action="store_true", help="flip one bit of the first gated family's first word (the gate must fail)")
    ap.add_argument("--sass-only", action="store_true", help="build and run the SASS gate only (no device)")
    ap.add_argument("--prebuilt", default=None, metavar="SO",
                    help="load this library (an earlier --sass-only run's build of the same source; its sha256 is recorded) instead of "
                         "running nvcc, so a fill job holds no GPU while compiling")
    ap.add_argument("--hot-bits", default="14,15,16", help="sp_nvf4_t1: the h of each hot accumulator H (C = +0 always runs)")
    ap.add_argument("--model-sample", type=int, default=4096,
                    help="sp_nvf4_t1: span words per accumulator checked against the pinned models (and 1/64 as many whole chains)")
    ap.add_argument("--vectors", default=None, help="nvf4_lean_vectors: a generated Fp4Vectors.lean")
    ap.add_argument("--vector-list", default="nvf4P7", help="nvf4_lean_vectors: the Vec list --vector-ids index")
    ap.add_argument("--vector-ids", default=None, help="nvf4_lean_vectors: comma list of 0-based entries of --vector-list")
    ap.add_argument("--chain-ids", default=None, help="nvf4_lean_vectors: comma list of 0-based entries of nvf4Chains")
    ap.add_argument("--special-ids", default=None,
                    help="nvf4_lean_vectors: comma list of 0-based entries of nvf4Specials (outside the model; record-only words)")
    ap.add_argument("--collect", default=None, metavar="DIR",
                    help="nvf4_lean_vectors: no device; re-derive every gate and word on the CPU from DIR, a fill chunk's capture "
                         "(its probe_results.json and dumps.npz), and publish them")
    ap.add_argument("--library", default=None, help="load this library (a build of the same source and arch on a host with the CUDA "
                    "toolkit) instead of compiling")
    args = ap.parse_args(argv)
    if args.instruction in LEAN_VECTOR_INSTRUCTIONS:
        import lean_vectors
        return lean_vectors.main(args)
    if args.instruction in CAST_INSTRUCTIONS:
        import cast
        return cast.main(args)
    if args.instruction in T1_INSTRUCTIONS:
        import t1_merged
        return t1_merged.main(args)
    if args.instruction in META_RULE_INSTRUCTIONS:
        import meta_rule
        return meta_rule.main(args)
    if args.instruction in SPARSE_INSTRUCTIONS:
        import sparse
        return sparse.main(args)
    # the remote runner passes argv without a shell: `$RESEARCH_RUN_DIR` arrives literally and is expanded here
    args.replay_autoproof = [os.path.expandvars(p) for p in args.replay_autoproof]
    out = Path(os.path.expandvars(args.out))
    out.mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    instr = args.instruction
    tcgen05 = instr in TCGEN05_KIND
    if args.arch is None:
        args.arch = DEFAULT_ARCH[instr]
    if args.chain2 and not tcgen05:
        ap.error("--chain2 is a tcgen05 option")
    if args.chain < 1 or (args.chain > 1 and instr not in ROW_KIND):
        ap.error("--chain L (L >= 2) is for the mma.sync rows " + ", ".join(sorted(ROW_KIND)))
    if args.fake_silicon and os.environ.get("RESEARCH_RUN_ID"):
        ap.error("--fake-silicon is a dry run and never publishes: not under a research run")
    if args.sass_only:
        args.sass_check = True

    if args.fake_silicon:
        so, nvcc = (None, "not built (fake silicon)") if not shutil.which(nvcc_bin()) else build(args.arch, out, SOURCE[instr])
        probe = FakeProbe(instr, args.fake_silicon)
    elif args.prebuilt:
        so = Path(os.path.expandvars(args.prebuilt))
        nvcc = f"prebuilt {so.name} sha256 {hashlib.sha256(so.read_bytes()).hexdigest()}"
        probe = None if args.sass_only else Probe(so)
    else:
        so, nvcc = prebuilt(os.path.expandvars(args.library)) if args.library else build(args.arch, out, SOURCE[instr])
        probe = None if args.sass_only else Probe(so)
    if tcgen05 and args.smem_strides and probe is not None:
        probe.smem = dict(zip(("lbo_a", "sbo_a", "lbo_b", "sbo_b"), (int(x) for x in args.smem_strides.split(","))))
    identity = probe.identity() if probe is not None else {"device": None, "gpu_uuid": None, "driver": None, "compute_capability": None,
                                                           "has_mxf4": True, "backend": "none (--sass-only)"}
    identity["nvcc"] = nvcc
    identity["arch"] = args.arch
    if args.expect_uuid and not args.fake_silicon and not args.sass_only and not uuid_allowed(identity.get("gpu_uuid"), args.expect_uuid):
        msg = f"device UUID {identity.get('gpu_uuid')} != expected {args.expect_uuid}: stopped before any measurement"
        (out / "probe_results.json").write_text(json.dumps({"identity": identity, "failures": [msg]}, indent=1))
        print("FAILURES", [msg], flush=True)
        return 2
    primary = args.expect_model or PRIMARY_MODEL[instr]
    if instr == "bf16":
        model_desc, model_src = "verity.ml.circuits.silicon.silicon.tc_dot", "verity.ml.circuits.silicon.silicon"
    elif instr in k32.ROWS:
        model_desc = "k32.candidates: " + ", ".join(sorted(verity_models(instr))) + (" (control: autoproof model5)" if instr in MODEL5_ROWS else "")
        model_src = str(HERE / "k32.py") + " + verity.ml.circuits.silicon.models"
    else:
        model_desc = "verity.ml.circuits.silicon.models." + {"nvf4": "BLACKWELL_SM120_NVF4", "mxf4": "BLACKWELL_SM120_MXF4",
                                               "tcgen05_nvf4": "BLACKWELL_SM120_NVF4 (the sm_120 model as the hypothesis for sm_100 tcgen05)",
                                               "tcgen05_mxf4": "BLACKWELL_SM120_MXF4 (the sm_120 model as the hypothesis for sm_100 tcgen05)",
                                               "tcgen05_mxf4kind": "BLACKWELL_SM120_MXF4 (the sm_120 model as the hypothesis for sm_100 tcgen05)"}[instr] + \
                     " (control: autoproof model5)"
        model_src = "verity.ml.circuits.silicon.models + " + str(HERE / "model5.py")
    if args.expect_model and instr != "bf16" and args.expect_model not in {**verity_models(instr), **({"model5": 1} if instr in MODEL5_ROWS else {})}:
        ap.error(f"--expect-model {args.expect_model!r} is not a model of {instr}")
    results: dict = {"schema": "fp4-re/tc-probe-fp4/v1", "instruction": PTX[instr], "instruction_id": INSTRUCTION_ID[instr],
                     "identity": identity, "model": model_desc, "model_source": model_src, "primary_model": primary,
                     "chain": args.chain, "fake_silicon": args.fake_silicon, "corrupt_word": bool(args.corrupt_word)}
    if probe is not None:
        results["clocks_start"] = probe.clocks(identity.get("gpu_uuid"))
    if tcgen05:
        results["tcgen05"] = {"mma_shape": "128x8x64 (cta_group::1)", "tile_rows_replicated": 8, "smem_strides": probe.smem,
                              "instruction_descriptor": f"0x{probe.lib.tcgen05_idesc(TCGEN05_KIND[instr]):08x}", "chain2": bool(args.chain2),
                              "accumulator_paths": "enable_input_d = 0 for all-+0 accumulator tiles, else TMEM preload (tcgen05.st) + enable_input_d = 1; "
                                                   "tmem_path: every tile through the TMEM preload"}
    print("device", json.dumps(identity), flush=True)
    failures = []

    if args.sass_check and so is not None:
        sass = sass_check(so)
        results["sass"] = sass
        for kern, rec in sass.items():
            if rec["mma_count"] != 1:
                failures.append(f"sass: {kern} has {rec['mma_count']} MMA instructions")
        print("sass", json.dumps(sass), flush=True)
        if KERNEL[instr] not in sass:
            failures.append(f"sass: the {instr} kernel {KERNEL[instr]} is missing from the build")
    if args.sass_only:
        results["failures"] = failures
        (out / "probe_results.json").write_text(json.dumps(results, indent=1))
        print("FAILURES" if failures else "SASS GATE PASSED", failures, flush=True)
        return 1 if failures else 0

    if instr == "mxf4" and not identity["has_mxf4"]:
        failures.append("mxf4 kernel not compiled")
    else:
        probe.debug_dir = out
        results["layout_check"] = layout_check(probe, instr)
        print("layout_check", instr, results["layout_check"], flush=True)
        if instr in ROW_KIND and (instr in GENERIC_ONLY or args.chain > 1):
            results["chain_layout_check"] = chain_layout_check(probe, instr, 3)
            print("chain_layout_check", instr, results["chain_layout_check"], flush=True)

    replays = []
    for p in args.replay_autoproof:
        rep = replay_autoproof(probe, Path(p))
        replays.append(rep)
        if rep["mismatches"] or rep["other_positions_nonzero"]:
            failures.append(f"autoproof replay {rep['file']}: mismatches={rep['mismatches']} other_positions_nonzero={rep['other_positions_nonzero']}")
        print("replay", json.dumps({k: v for k, v in rep.items() if k != "first_mismatches"}), flush=True)
    if replays:
        results["autoproof_replay"] = replays

    fams: dict = {}
    if args.sweep and not (instr == "mxf4" and not identity["has_mxf4"]):
        only = set(args.families.split(",")) if args.families else None
        rng = np.random.default_rng(args.seed)
        if instr == "bf16":
            gen = families_bf16(rng, args.n_random)
        elif instr in k32.ROWS:
            gen = k32.families_k32(rng, args.n_random, instr)
        else:
            gen = families_nvf4(rng, args.n_random, SCALE_KIND[instr])
        capture = {"instruction": PTX[instr], "instruction_id": INSTRUCTION_ID[instr], "k": K_OF[instr], "scale_blocks": NBLK[instr],
                   "chain": args.chain,
                   "note": "acc/d FP32 words (hex); a/b operand codes in K order (e2m1 nibbles as hex digits, e4m3 bytes as 2 hex digits; "
                           "bf16 as 4-hex words); sfa/sfb scale codes per K-block (hex bytes), as measured on the device in identity",
                   "identity": identity, "records": []}
        corrupted = False
        with mp.Pool(args.procs, initializer=_init, initargs=(instr, args.group)) as pool:
            for name, A, Bt, SFA, SFB, C in gen:
                if only is not None and name not in only:
                    continue
                steps = 1
                if args.chain > 1:
                    # L tiles -> one chain of L dependent MMAs: tile c + s*h supplies k-block s, tile c's C the accumulator
                    L = args.chain
                    h = A.shape[0] // L
                    if h == 0:
                        continue
                    A = np.concatenate([A[s * h:(s + 1) * h] for s in range(L)], axis=2)
                    Bt = np.concatenate([Bt[s * h:(s + 1) * h] for s in range(L)], axis=2)
                    if SFA is not None:
                        SFA = np.concatenate([SFA[s * h:(s + 1) * h] for s in range(L)], axis=2)
                        SFB = np.concatenate([SFB[s * h:(s + 1) * h] for s in range(L)], axis=2)
                    C = C[:h]
                    steps = L
                if args.chain2:
                    # pairs of tiles -> one K=128 chain: the second tile's operands are the second k-block, the first tile's C the accumulator
                    h = A.shape[0] // 2
                    A, Bt = np.concatenate([A[:h], A[h:2 * h]], axis=2), np.concatenate([Bt[:h], Bt[h:2 * h]], axis=2)
                    SFA, SFB = np.concatenate([SFA[:h], SFA[h:2 * h]], axis=2), np.concatenate([SFB[:h], SFB[h:2 * h]], axis=2)
                    C = C[:h]
                    steps = 2
                t0 = time.time()
                if tcgen05:
                    probe.replica_disagreements = 0
                    probe.path_tiles = {"acc_disabled": 0, "acc_tmem": 0}
                    D = probe.tiles(instr, A, Bt, C, SFA, SFB, steps=steps)
                    paths = dict(probe.path_tiles)
                    replica_mixed = probe.replica_disagreements
                    probe.replica_disagreements = 0
                    D_tmem = probe.tiles(instr, A, Bt, C, SFA, SFB, steps=steps, force_tmem=True)
                    replica_tmem = probe.replica_disagreements
                else:
                    D = probe.tiles(instr, A, Bt, C, SFA, SFB, steps=steps)
                if args.corrupt_word and not corrupted and name not in RECORD_ONLY_FAMILIES:
                    D = D.copy()
                    D[0, 0, 0] ^= 1
                    corrupted = True
                t_gpu = time.time() - t0
                t0 = time.time()
                models, errs = model_eval(pool, A, Bt, SFA, SFB, C, args.procs)
                t_model = time.time() - t0
                fam = {"tiles": int(A.shape[0]), "elements": int(A.shape[0]) * 128, "record_only": name in RECORD_ONLY_FAMILIES,
                       "nonzero_accumulators": int(np.count_nonzero(C & 0x7FFFFFFF)),
                       "model_domain_errors": {m: int(e.sum()) for m, e in errs.items()},
                       "gpu_seconds": round(t_gpu, 2), "model_seconds": round(t_model, 1), "mismatches_vs_model": {}, "first_mismatches": {}}
                if steps > 1:
                    fam["k_per_tile"] = K_OF[instr] * steps
                if SFA is not None and SCALE_KIND[instr] == "ue4m3":
                    ng = A.shape[2] // 16
                    a_nz = (model5.E2M1_NUM[A.astype(np.int64)] != 0).reshape(-1, 16, ng, 16)
                    b_nz = (model5.E2M1_NUM[Bt.astype(np.int64)] != 0).reshape(-1, 8, ng, 16)
                    live = np.einsum("nigk,njgk->nijg", a_nz.astype(np.int64), b_nz.astype(np.int64)) > 0
                    hot = live & ((SFA.astype(np.int64) & 0x78) == 0x78)[:, :, None, :] & ((SFB.astype(np.int64) & 0x78) == 0x78)[:, None, :, :]
                    fam["hot_firings"] = int(np.count_nonzero(hot.any(-1)))
                acc_nz = (C & 0x7FFFFFFF) != 0
                fam["mismatch_split"] = {}
                for mname, model in models.items():
                    err = errs[mname]
                    mism = (model != D) | err
                    fam["mismatches_vs_model"][mname] = int(mism.sum())
                    fam["mismatch_split"][mname] = {"acc_zero": int((mism & ~acc_nz).sum()), "acc_nonzero": int((mism & acc_nz).sum()),
                                                    "acc_negative_zero": int((mism & (C == 0x80000000)).sum()),
                                                    "domain_errors": int(err.sum()), "wrong_word": int((mism & ~err).sum())}
                    first = []
                    for t, i, j in list(zip(*np.nonzero(mism)))[:20]:
                        rec = hexrec(C[t, i, j], A[t, i], Bt[t, j], D[t, i, j], SFA[t, i] if SFA is not None else None,
                                     SFB[t, j] if SFB is not None else None, bf16=(instr == "bf16"), widths=HEX_WIDTHS.get(instr, (1, 1)))
                        rec["model"] = "error" if err[t, i, j] else f"{int(model[t, i, j]):08x}"
                        first.append(rec)
                    fam["first_mismatches"][mname] = first
                if name == "scale_split_twins" and steps == 1:
                    fam["twins"] = twin_summary(D, models)
                if name == "scale_bit7" and steps == 1:
                    fam["bit7"] = bit7_summary(D, SFA, SFB, models.get(primary), errs.get(primary))
                if primary in errs and errs[primary].any():
                    fam["observed_outside_model_domain"] = observed_summary(D, C, errs[primary])
                if tcgen05:
                    # the second accumulator path (every tile through the TMEM preload) against the same model words, and the
                    # replica lanes: both must agree with path one for the instruction to have ONE semantics
                    path_dis = int(np.count_nonzero(D != D_tmem))
                    fam["tcgen05"] = {"acc_path_tiles": paths, "tmem_path_mismatches_vs_model": {m: int(((v != D_tmem) | errs[m]).sum()) for m, v in models.items()},
                                      "path_disagreements": path_dis, "replica_disagreements": {"mixed_path": replica_mixed, "tmem_path": replica_tmem}}
                    if path_dis:
                        first = []
                        for t, i, j in list(zip(*np.nonzero(D != D_tmem)))[:20]:
                            rec = hexrec(C[t, i, j], A[t, i], Bt[t, j], D[t, i, j], SFA[t, i], SFB[t, j])
                            rec["d_tmem_path"] = f"{int(D_tmem[t, i, j]):08x}"
                            first.append(rec)
                        fam["tcgen05"]["first_path_disagreements"] = first
                        failures.append(f"{name}: {path_dis} words differ between the accumulator-disabled and the TMEM-preload paths")
                    if replica_mixed or replica_tmem:
                        failures.append(f"{name}: replica lanes disagree (mixed {replica_mixed}, tmem {replica_tmem})")
                    for m, cnt in fam["tcgen05"]["tmem_path_mismatches_vs_model"].items():
                        if m == primary and cnt != fam["mismatches_vs_model"][m] and name not in RECORD_ONLY_FAMILIES:
                            failures.append(f"{name}: tmem path mismatches vs {m} = {cnt} != mixed path {fam['mismatches_vs_model'][m]}")
                kw = {"A": A, "Bt": Bt, "C": C, "D": D, **{f"model_{m}": v for m, v in models.items()}}
                if SFA is not None:
                    kw["SFA"], kw["SFB"] = SFA, SFB
                if tcgen05:
                    kw["D_tmem_path"] = D_tmem
                np.savez_compressed(out / f"tiles_{name}.npz", **kw)
                flat = rng.choice(A.shape[0] * 128, size=min(args.golden_per_family, A.shape[0] * 128), replace=False)
                for f in sorted(flat.tolist()):
                    t, i, j = f // 128, (f % 128) // 8, f % 8
                    capture["records"].append({**hexrec(C[t, i, j], A[t, i], Bt[t, j], D[t, i, j], SFA[t, i] if SFA is not None else None,
                                                        SFB[t, j] if SFB is not None else None, bf16=(instr == "bf16"), widths=HEX_WIDTHS.get(instr, (1, 1))), "family": name})
                fams[name] = fam
                print(name, json.dumps({k: v for k, v in fam.items() if k != "first_mismatches"}), flush=True)
        model_names = sorted({m for f in fams.values() for m in f["mismatches_vs_model"]})
        gated = {k: f for k, f in fams.items() if not f["record_only"]}
        results["sweep"] = {"seed": args.seed, "n_random": args.n_random, "group_hypothesis": args.group, "families": fams,
                            "total_elements": sum(f["elements"] for f in fams.values()),
                            "gated_elements": sum(f["elements"] for f in gated.values()),
                            "record_only_families": sorted(k for k, f in fams.items() if f["record_only"]),
                            "total_nonzero_accumulators": sum(f["nonzero_accumulators"] for f in fams.values()),
                            "total_hot_firings": sum(f.get("hot_firings", 0) for f in fams.values()),
                            "total_mismatches_vs_model": {m: sum(f["mismatches_vs_model"][m] for f in gated.values()) for m in model_names},
                            "record_only_mismatches_vs_model": {m: sum(f["mismatches_vs_model"][m] for f in fams.values() if f["record_only"])
                                                                for m in model_names}}
        if tcgen05:
            results["sweep"]["tcgen05"] = {
                "chain2": bool(args.chain2),
                "acc_path_tiles": {p: sum(f["tcgen05"]["acc_path_tiles"][p] for f in fams.values()) for p in ("acc_disabled", "acc_tmem")},
                "total_tmem_path_mismatches_vs_model": {m: sum(f["tcgen05"]["tmem_path_mismatches_vs_model"][m] for f in gated.values()) for m in model_names},
                "total_path_disagreements": sum(f["tcgen05"]["path_disagreements"] for f in fams.values()),
                "total_replica_disagreements": {p: sum(f["tcgen05"]["replica_disagreements"][p] for f in fams.values()) for p in ("mixed_path", "tmem_path")}}
        (out / "capture_sample.json").write_text(json.dumps(capture, separators=(",", ":")))
        if instr == "bf16":
            agreeing = [m for m, c in results["sweep"]["total_mismatches_vs_model"].items() if c == 0]
            results["bf16_pipeline_finding"] = {"agreeing_pipelines": agreeing, "mismatches": results["sweep"]["total_mismatches_vs_model"]}
            if not agreeing:
                failures.append("bf16: neither AMPERE nor HOPPER pipeline reproduces every word")
        else:
            totals = results["sweep"]["total_mismatches_vs_model"]
            results["candidate_finding"] = {"agreeing": sorted(m for m, c in totals.items() if c == 0), "mismatches": totals}
            if primary is not None:
                total = totals.get(primary, 0)
                if total:
                    failures.append(f"sweep mismatches vs {primary}={total} (record-only families excluded)")
        print("sweep TOTAL", results["sweep"]["total_elements"], "elements,", results["sweep"]["total_mismatches_vs_model"],
              "record-only:", results["sweep"]["record_only_mismatches_vs_model"], flush=True)

    if probe is not None:
        results["clocks_end"] = probe.clocks(identity.get("gpu_uuid"))
    results["seconds"] = round(time.time() - t_all, 1)
    results["failures"] = failures
    (out / "probe_results.json").write_text(json.dumps(results, indent=1))

    run_id = os.environ.get("RESEARCH_RUN_ID")
    if run_id:
        agree = {}
        first_mism: list = []
        if "sweep" in results:
            for m, c in results["sweep"]["total_mismatches_vs_model"].items():
                agree[m] = c == 0
            for fam in fams.values():
                for m, lst in fam["first_mismatches"].items():
                    for rec in lst:
                        if len(first_mism) < 20:
                            first_mism.append({"model": m, **slim_record(rec)})
        for rep in replays:
            agree[f"autoproof_replay:{rep['file']}"] = rep["mismatches"] == 0 and rep["other_positions_nonzero"] == 0
            for rec in rep["first_mismatches"]:
                if len(first_mism) < 20:
                    first_mism.append({"model": f"autoproof_replay:{rep['file']}", **slim_record(rec)})
        # the verdict: the primary model over the gated families and every autoproof replay; controls/alternatives are recorded
        # (a row with no primary model is a capture: its finding is ``candidate_finding``, the models that reproduce every word)
        verdict_keys = [k for k in agree if k == primary or k.startswith("autoproof_replay:")]
        model_agrees = all(agree[k] for k in verdict_keys) if verdict_keys else None
        result = {
            "schema": "research/result/v0.1", "kind": "tc-evidence/v1", "run_id": run_id,
            "instruction": INSTRUCTION_ID[instr], "ptx": PTX[instr], "arch": args.arch, "sku": identity["device"],
            "compute_capability": identity["compute_capability"], "driver": identity["driver"], "gpu_uuid": identity["gpu_uuid"],
            "nvcc": nvcc, "sass_mma_counts": {k: v["mma_count"] for k, v in results.get("sass", {}).items()},
            "families": {name: {"elements": f["elements"], "mismatches": f["mismatches_vs_model"], "hot_firings": f.get("hot_firings"),
                                "record_only": f["record_only"], "domain_errors": f["model_domain_errors"],
                                "observed_outside_model_domain": f.get("observed_outside_model_domain")} for name, f in fams.items()},
            "counts": {"elements": results.get("sweep", {}).get("total_elements", 0),
                       "gated_elements": results.get("sweep", {}).get("gated_elements", 0),
                       "replay_cases": sum(r["cases"] for r in replays), "replay_matched": sum(r["matched"] for r in replays),
                       "mismatches": results.get("sweep", {}).get("total_mismatches_vs_model", {}),
                       "record_only_mismatches": results.get("sweep", {}).get("record_only_mismatches_vs_model", {})},
            "model": results["model"], "primary_model": primary, "model_agrees": model_agrees, "agreement": agree,
            "first_mismatches": first_mism, "seed": args.seed, "n": args.n_random, "group_hypothesis": args.group,
            "bf16_pipeline_finding": results.get("bf16_pipeline_finding"), "candidate_finding": results.get("candidate_finding"),
            "chain": args.chain, "expected_model": args.expect_model, "cuda_visible_devices": identity.get("cuda_visible_devices"),
            "clocks": {"start": results.get("clocks_start"), "end": results.get("clocks_end"),
                       "note": "recorded, not locked: a capture's words do not depend on the clock"},
            "backend": identity["backend"], "tcgen05": {**results.get("tcgen05", {}), **results.get("sweep", {}).get("tcgen05", {})} if tcgen05 else None,
            "sass_opcodes": {k: sorted(set(v["opcodes"])) for k, v in results.get("sass", {}).items()},
            "workload_fingerprint": {"probe": "benchmarks/tc_probe_fp4/probe.py", "instruction": INSTRUCTION_ID[instr], "arch": args.arch,
                                     "device": identity["device"], "seed": args.seed, "n_random": args.n_random, "chain2": bool(args.chain2),
                                     "chain": args.chain,
                                     "replays": [Path(p).name for p in args.replay_autoproof]},
            "validation": {"status": "passed" if not failures else "failed",
                           "checker": "tc_probe_fp4 vs verity.ml.circuits.silicon.models (control: model5) / tc_dot / autoproof vectors",
                           "detail": "; ".join(failures) or ("every hardware word reproduced" if primary is not None else
                                                             "capture recorded (layout, SASS and identity gates passed); candidates "
                                                             f"reproducing every word: {(results.get('candidate_finding') or {}).get('agreeing')}"),
                           "evidence": ["probe_results.json"]},
            "measurements": [{"name": "wall", "value": results["seconds"], "unit": "s"}],
            "measurement_files": ["probe_results.json"],
            "artifacts": (["capture_sample.json"] if "sweep" in results else []) + sorted(p.name for p in out.glob("tiles_*.npz")),
        }
        tmp = out / "result.json.tmp"
        tmp.write_text(json.dumps(result, indent=1))
        os.replace(tmp, out / "result.json")
    if results.get("candidate_finding") and primary is None:
        print("candidates reproducing every word:", results["candidate_finding"]["agreeing"], flush=True)
    print("FAILURES" if failures else ("ALL HARDWARE WORDS REPRODUCED" if primary is not None or not fams else "CAPTURE RECORDED"), failures,
          f"{results['seconds']}s", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
