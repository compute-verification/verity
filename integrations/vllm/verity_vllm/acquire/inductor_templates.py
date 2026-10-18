"""Inductor's generated Triton kernels in a served-compiled Program, as templates of their source: the only way the served Build
(`acquire.served`) gets from a kernel to Definitions.

A kernel that a served wrapper launches binds to Definitions when its generated source (the text of its `async_compile.triton`
block from `def <name>(` on, with the name read as KERNEL) is one of the templates below, with each `{NAME}` a numeric literal. The
first occurrence binds NAME and every later one must repeat it, and the template's relations among the names must hold. Its
decorator must be the template's heuristic and declare exactly the template's signature (pointer element types included), fp fusion
on, ftz not disabled and compute capability 9.0: the PTX that `registry.inductor` was read from. A reduction kernel's launch config
(XBLOCK, R0_BLOCK, num_warps) gives its norm statics, read from the serving process's record (`ServedCompiledSource.launch_configs`),
which must name exactly one candidate config, the one that ran. Any other source is refused by name, since no Definition models it.

Each template lists its calls: a Definition with its statics, the kernel parameters it reads (whole, a column range of each row,
or the row of a table that each element of another parameter indexes) and the parameters it writes.

  EMBED_NORM       Embedding_v1{V, H=N}(in_ptr0 ids, in_ptr1 table) -> out_ptr0; InductorRMSNorm_v1(out_ptr0, in_ptr2) -> out_ptr2
  ADD_NORM         InductorAddRMSNorm_v1(in_ptr0, in_ptr1, in_ptr2) -> out_ptr1
  ADD2_NORM        InductorAdd2RMSNorm_v1(in_ptr0, in_ptr1, in_ptr2, in_ptr3) -> (out_ptr1, out_ptr2)
  ADD2_NORM_FINAL  InductorAdd2RMSNormFinal_v1(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2) -> in_out_ptr0
  SILU_MUL         InductorSiluMul_v1{I}(in_ptr0) -> out_ptr0
  ROPE_QK          InductorRoPE_v1{NHEADS=KVH, D}(in_ptr0 columns [H*D, (H+KVH)*D), in_ptr2 row in_ptr1) -> out_ptr0, and
                   InductorRoPE_v1{NHEADS=H, D}(in_ptr0 columns [0, H*D), in_ptr2 row in_ptr1) -> out_ptr1
"""
from __future__ import annotations

import ast
import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from verity.ml.circuits.defs import bind
from verity.ml.circuits.types import Array, Tuple, Value
from verity_vllm.program.registry import b1
from verity_vllm.program.registry import inductor as IR


EMBED_NORM = """\
def KERNEL(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = {N}
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    _tmp7 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp1 = tmp0.to(tl.int64)
        tl.device_assert(((0 <= tmp1) & (tmp1 < {V})) | ~(xmask), "index out of bounds: 0 <= tmp1 < {V}")
        tmp3 = tl.load(in_ptr1 + (r0_1 + {N}*tmp1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp4 = tmp3.to(tl.float32)
        tmp5 = tmp4 * tmp4
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(r0_mask & xmask, tmp8, _tmp7)
        tl.store(out_ptr0 + (r0_1 + {N}*x0), tmp3, r0_mask & xmask)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp9 = tl.load(out_ptr0 + (r0_1 + {N}*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp18 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp10 = tmp9.to(tl.float32)
        tmp11 = tl.full([1, 1], {NF}, tl.float32)
        tmp12 = (tmp7 / tmp11)
        tmp13 = tl.full([1, 1], {EPS}, tl.float32)
        tmp14 = tmp12 + tmp13
        tmp15 = libdevice.rsqrt(tmp14)
        tmp16 = tmp10 * tmp15
        tmp17 = tmp16.to(tl.float32)
        tmp19 = tmp17 * tmp18
        tl.store(out_ptr2 + (r0_1 + {N}*x0), tmp19, r0_mask & xmask)
"""

ADD_NORM = """\
def KERNEL(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = {N}
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp7 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + {N}*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (r0_1 + {N}*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tmp1 + tmp3
        tmp5 = tmp4 * tmp4
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(r0_mask & xmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp9 = tl.load(in_ptr0 + (r0_1 + {N}*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp11 = tl.load(in_ptr1 + (r0_1 + {N}*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp21 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp10 = tmp9.to(tl.float32)
        tmp12 = tmp11.to(tl.float32)
        tmp13 = tmp10 + tmp12
        tmp14 = tl.full([1, 1], {NF}, tl.float32)
        tmp15 = (tmp7 / tmp14)
        tmp16 = tl.full([1, 1], {EPS}, tl.float32)
        tmp17 = tmp15 + tmp16
        tmp18 = libdevice.rsqrt(tmp17)
        tmp19 = tmp13 * tmp18
        tmp20 = tmp19.to(tl.float32)
        tmp22 = tmp20 * tmp21
        tl.store(out_ptr1 + (r0_1 + {N}*x0), tmp22, r0_mask & xmask)
"""

ADD2_NORM = """\
def KERNEL(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = {N}
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp12 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + {N}*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (r0_1 + {N}*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp4 = tl.load(in_ptr2 + (r0_1 + {N}*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tmp3 + tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tmp7.to(tl.float32)
        tmp9 = tmp1 + tmp8
        tmp10 = tmp9 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(r0_mask & xmask, tmp13, _tmp12)
        tmp14 = tmp9.to(tl.float32)
        tl.store(out_ptr1 + (r0_1 + {N}*x0), tmp14, r0_mask & xmask)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp15 = tl.load(in_ptr0 + (r0_1 + {N}*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp17 = tl.load(in_ptr1 + (r0_1 + {N}*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp19 = tl.load(in_ptr2 + (r0_1 + {N}*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp32 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp16 = tmp15.to(tl.float32)
        tmp18 = tmp17.to(tl.float32)
        tmp20 = tmp19.to(tl.float32)
        tmp21 = tmp18 + tmp20
        tmp22 = tmp21.to(tl.float32)
        tmp23 = tmp22.to(tl.float32)
        tmp24 = tmp16 + tmp23
        tmp25 = tl.full([1, 1], {NF}, tl.float32)
        tmp26 = (tmp12 / tmp25)
        tmp27 = tl.full([1, 1], {EPS}, tl.float32)
        tmp28 = tmp26 + tmp27
        tmp29 = libdevice.rsqrt(tmp28)
        tmp30 = tmp24 * tmp29
        tmp31 = tmp30.to(tl.float32)
        tmp33 = tmp31 * tmp32
        tl.store(out_ptr2 + (r0_1 + {N}*x0), tmp33, r0_mask & xmask)
"""

ADD2_NORM_FINAL = """\
def KERNEL(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = {N}
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp12 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_out_ptr0 + (r0_1 + {N}*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr0 + (r0_1 + {N}*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp4 = tl.load(in_ptr1 + (r0_1 + {N}*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tmp3 + tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tmp7.to(tl.float32)
        tmp9 = tmp1 + tmp8
        tmp10 = tmp9 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(r0_mask & xmask, tmp13, _tmp12)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp14 = tl.load(in_out_ptr0 + (r0_1 + {N}*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp16 = tl.load(in_ptr0 + (r0_1 + {N}*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp18 = tl.load(in_ptr1 + (r0_1 + {N}*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp31 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp15 = tmp14.to(tl.float32)
        tmp17 = tmp16.to(tl.float32)
        tmp19 = tmp18.to(tl.float32)
        tmp20 = tmp17 + tmp19
        tmp21 = tmp20.to(tl.float32)
        tmp22 = tmp21.to(tl.float32)
        tmp23 = tmp15 + tmp22
        tmp24 = tl.full([1, 1], {NF}, tl.float32)
        tmp25 = (tmp12 / tmp24)
        tmp26 = tl.full([1, 1], {EPS}, tl.float32)
        tmp27 = tmp25 + tmp26
        tmp28 = libdevice.rsqrt(tmp27)
        tmp29 = tmp23 * tmp28
        tmp30 = tmp29.to(tl.float32)
        tmp32 = tmp30 * tmp31
        tl.store(in_out_ptr0 + (r0_1 + {N}*x0), tmp32, r0_mask & xmask)
"""

SILU_MUL = """\
def KERNEL(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % {I})
    x1 = xindex // {I}
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + {I2}*x1), None).to(tl.float32)
    tmp8 = tl.load(in_ptr0 + ({I} + x0 + {I2}*x1), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = -tmp1
    tmp3 = libdevice.exp(tmp2)
    tmp4 = tl.full([1], 1.0, tl.float32)
    tmp5 = tmp3 + tmp4
    tmp6 = (tmp1 / tmp5)
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 * tmp8
    tl.store(out_ptr0 + (x2), tmp9, None)
"""

ROPE_QK = """\
def KERNEL(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel_0, xnumel_1, XBLOCK : tl.constexpr):
    pid = tl.program_id(0)
    num_xblocks_0 = tl.cdiv(xnumel_0, XBLOCK)
    num_xblocks_1 = num_xblocks_0 + tl.cdiv(xnumel_1, XBLOCK)
    if pid < num_xblocks_0:
        pid_offset = pid
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel_0
        x0 = (xindex % {D})
        x1 = ((xindex // {D}) % {KVH})
        x2 = xindex // {KLEN}
        x4 = xindex
        tmp0 = (x0).to(tl.int32)
        tmp1 = tl.full([1], 0, tl.int64)
        tmp2 = tmp0 >= tmp1
        tmp3 = (x0).to(tl.int64)
        tmp4 = (tmp3).to(tl.int64)
        tmp5 = tl.full([1], {HALF}, tl.int64)
        tmp6 = tmp4 < tmp5
        tmp7 = tl.load(in_ptr0 + ({QLEN} + {D}*x1 + {QKV}*x2 + (x0)), tmp6 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp8 = tl.load(in_ptr1 + (x2), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
        tmp9 = (tl.full([XBLOCK], {P}, tl.int32)).to(tl.int32)
        tmp10 = tmp8 + tmp9
        tmp11 = tmp8 < 0
        tmp12 = tl.where(tmp11, tmp10, tmp8)
        tl.device_assert(((0 <= tl.broadcast_to(tmp12, [XBLOCK])) & (tl.broadcast_to(tmp12, [XBLOCK]) < {P})) | ~(tmp6 & xmask), "index out of bounds: 0 <= tl.broadcast_to(tmp12, [XBLOCK]) < {P}")
        tmp14 = tl.load(in_ptr2 + ({D}*tmp12 + (x0)), tmp6 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp15 = tmp7 * tmp14
        tmp16 = tl.load(in_ptr0 + ({KHI} + {D}*x1 + {QKV}*x2 + (x0)), tmp6 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp17 = tl.load(in_ptr2 + ({HALF} + {D}*tmp12 + (x0)), tmp6 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp18 = tmp16 * tmp17
        tmp19 = tmp15 - tmp18
        tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
        tmp21 = tl.where(tmp6, tmp19, tmp20)
        tmp22 = tmp0 >= tmp5
        tmp23 = tl.full([1], {D}, tl.int64)
        tmp24 = tmp0 < tmp23
        tmp25 = tl.load(in_ptr0 + ({KHI} + {D}*x1 + {QKV}*x2 + ((-{HALF}) + x0)), tmp22 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp26 = tl.load(in_ptr1 + (x2), tmp22 & xmask, eviction_policy='evict_last', other=0.0)
        tmp27 = (tl.full([XBLOCK], {P}, tl.int32)).to(tl.int32)
        tmp28 = tmp26 + tmp27
        tmp29 = tmp26 < 0
        tmp30 = tl.where(tmp29, tmp28, tmp26)
        tl.device_assert(((0 <= tl.broadcast_to(tmp30, [XBLOCK])) & (tl.broadcast_to(tmp30, [XBLOCK]) < {P})) | ~(tmp22 & xmask), "index out of bounds: 0 <= tl.broadcast_to(tmp30, [XBLOCK]) < {P}")
        tmp32 = tl.load(in_ptr2 + ({D}*tmp30 + ((-{HALF}) + x0)), tmp22 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp33 = tmp25 * tmp32
        tmp34 = tl.load(in_ptr0 + ({QLEN} + {D}*x1 + {QKV}*x2 + ((-{HALF}) + x0)), tmp22 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp35 = tl.load(in_ptr2 + ({HALF} + {D}*tmp30 + ((-{HALF}) + x0)), tmp22 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp36 = tmp34 * tmp35
        tmp37 = tmp33 + tmp36
        tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
        tmp39 = tl.where(tmp22, tmp37, tmp38)
        tmp40 = tl.where(tmp6, tmp21, tmp39)
        tl.store(out_ptr0 + (x4), tmp40, xmask)
    elif pid < num_xblocks_1:
        pid_offset = pid - num_xblocks_0
        r0_numel = 1
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel_1
        x5 = (xindex % {D})
        x6 = ((xindex // {D}) % {H})
        x7 = xindex // {QLEN}
        x9 = xindex
        tmp41 = (x5).to(tl.int32)
        tmp42 = tl.full([1], 0, tl.int64)
        tmp43 = tmp41 >= tmp42
        tmp44 = (x5).to(tl.int64)
        tmp45 = (tmp44).to(tl.int64)
        tmp46 = tl.full([1], {HALF}, tl.int64)
        tmp47 = tmp45 < tmp46
        tmp48 = tl.load(in_ptr0 + ({D}*x6 + {QKV}*x7 + (x5)), tmp47 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp49 = tl.load(in_ptr1 + (x7), tmp47 & xmask, eviction_policy='evict_last', other=0.0)
        tmp50 = (tl.full([XBLOCK], {P}, tl.int32)).to(tl.int32)
        tmp51 = tmp49 + tmp50
        tmp52 = tmp49 < 0
        tmp53 = tl.where(tmp52, tmp51, tmp49)
        tl.device_assert(((0 <= tl.broadcast_to(tmp53, [XBLOCK])) & (tl.broadcast_to(tmp53, [XBLOCK]) < {P})) | ~(tmp47 & xmask), "index out of bounds: 0 <= tl.broadcast_to(tmp53, [XBLOCK]) < {P}")
        tmp55 = tl.load(in_ptr2 + ({D}*tmp53 + (x5)), tmp47 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp56 = tmp48 * tmp55
        tmp57 = tl.load(in_ptr0 + ({HALF} + {D}*x6 + {QKV}*x7 + (x5)), tmp47 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp58 = tl.load(in_ptr2 + ({HALF} + {D}*tmp53 + (x5)), tmp47 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp59 = tmp57 * tmp58
        tmp60 = tmp56 - tmp59
        tmp61 = tl.full(tmp60.shape, 0.0, tmp60.dtype)
        tmp62 = tl.where(tmp47, tmp60, tmp61)
        tmp63 = tmp41 >= tmp46
        tmp64 = tl.full([1], {D}, tl.int64)
        tmp65 = tmp41 < tmp64
        tmp66 = tl.load(in_ptr0 + ({HALF} + {D}*x6 + {QKV}*x7 + ((-{HALF}) + x5)), tmp63 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp67 = tl.load(in_ptr1 + (x7), tmp63 & xmask, eviction_policy='evict_last', other=0.0)
        tmp68 = (tl.full([XBLOCK], {P}, tl.int32)).to(tl.int32)
        tmp69 = tmp67 + tmp68
        tmp70 = tmp67 < 0
        tmp71 = tl.where(tmp70, tmp69, tmp67)
        tl.device_assert(((0 <= tl.broadcast_to(tmp71, [XBLOCK])) & (tl.broadcast_to(tmp71, [XBLOCK]) < {P})) | ~(tmp63 & xmask), "index out of bounds: 0 <= tl.broadcast_to(tmp71, [XBLOCK]) < {P}")
        tmp73 = tl.load(in_ptr2 + ({D}*tmp71 + ((-{HALF}) + x5)), tmp63 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp74 = tmp66 * tmp73
        tmp75 = tl.load(in_ptr0 + ({D}*x6 + {QKV}*x7 + ((-{HALF}) + x5)), tmp63 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp76 = tl.load(in_ptr2 + ({HALF} + {D}*tmp71 + ((-{HALF}) + x5)), tmp63 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp77 = tmp75 * tmp76
        tmp78 = tmp74 + tmp77
        tmp79 = tl.full(tmp78.shape, 0.0, tmp78.dtype)
        tmp80 = tl.where(tmp63, tmp78, tmp79)
        tmp81 = tl.where(tmp47, tmp62, tmp80)
        tl.store(out_ptr1 + (x9), tmp81, xmask)
    else:
        pass
"""


class Unmodelled(ValueError):
    """A generated kernel no template models, or one whose decorator or launch config the Definitions were not read for."""


@dataclass(frozen=True)
class Read:
    param: str
    cols: Callable[[dict[str, Any]], tuple[int, int]] | None = None
    row_of: str | None = None


@dataclass(frozen=True)
class Call:
    fn: Any
    statics: Callable[[dict[str, Any]], dict[str, Any]]
    reads: tuple[Read, ...]
    writes: tuple[str, ...]


@dataclass(frozen=True)
class Template:
    name: str
    heuristic: str
    signature: dict[str, str]
    text: str
    relations: Callable[[dict[str, Any]], bool]
    calls: tuple[Call, ...]
    scalars: Callable[[dict[str, Any]], dict[str, tuple[str, int]]]
    config: bool = False
    pattern: re.Pattern = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "pattern", _pattern(self.text.rstrip()))


_NUM = r"-?\d+(?:\.\d*)?(?:e[-+]?\d+)?"
_HOLE = re.compile(r"\{([A-Z][A-Z0-9]*)\}")


def _pattern(text: str) -> re.Pattern:
    out, seen, pos = [], set(), 0
    for m in _HOLE.finditer(text):
        out.append(re.escape(text[pos:m.start()]))
        name = m.group(1)
        out.append(f"(?P={name})" if name in seen else f"(?P<{name}>{_NUM})")
        seen.add(name)
        pos = m.end()
    out.append(re.escape(text[pos:]))
    return re.compile("".join(out))


def _num(s: str) -> int | float:
    return int(s) if re.fullmatch(r"-?\d+", s) else float(s)


def _sig(ptrs: dict[str, str], scalars: tuple[str, ...], constexprs: tuple[str, ...]) -> dict[str, str]:
    return {**ptrs, **{s: "i32" for s in scalars}, **{c: "constexpr" for c in constexprs}}


_BF16 = "*bf16"
_RED = ("xnumel", "r0_numel")
_RED_C = ("XBLOCK", "R0_BLOCK")


def _norm_statics(lit: dict[str, Any]) -> dict[str, Any]:
    return {"N": lit["N"], "EPS": float(lit["EPS"]), "XBLOCK": lit["XBLOCK"], "R0_BLOCK": lit["R0_BLOCK"], "NUM_WARPS": lit["NUM_WARPS"]}


def _red_scalars(lit: dict[str, Any]) -> dict[str, tuple[str, int]]:
    return {"xnumel": ("tokens", 1), "r0_numel": ("const", lit["N"])}


def _norm_relations(lit: dict[str, Any]) -> bool:
    return isinstance(lit["N"], int) and lit["N"] > 0 and float(lit["NF"]) == lit["N"]


def _rope_relations(lit: dict[str, Any]) -> bool:
    D, H, KVH = lit["D"], lit["H"], lit["KVH"]
    return (all(isinstance(lit[k], int) and lit[k] > 0 for k in ("D", "H", "KVH", "HALF", "P")) and D == 2 * lit["HALF"]
            and lit["QLEN"] == H * D and lit["KLEN"] == KVH * D and lit["KHI"] == H * D + lit["HALF"] and lit["QKV"] == (H + 2 * KVH) * D)


TEMPLATES = (
    Template("EMBED_NORM", "reduction",
             _sig({"in_ptr0": "*i32", "in_ptr1": _BF16, "in_ptr2": _BF16, "out_ptr0": _BF16, "out_ptr2": _BF16}, _RED, _RED_C),
             EMBED_NORM, lambda lit: _norm_relations(lit) and isinstance(lit["V"], int) and lit["V"] > 0,
             (Call(b1.Embedding, lambda lit: {"V": lit["V"], "H": lit["N"]}, (Read("in_ptr0"), Read("in_ptr1")), ("out_ptr0",)),
              Call(IR.InductorRMSNorm, _norm_statics, (Read("out_ptr0"), Read("in_ptr2")), ("out_ptr2",))), _red_scalars, config=True),
    Template("ADD_NORM", "reduction", _sig({"in_ptr0": _BF16, "in_ptr1": _BF16, "in_ptr2": _BF16, "out_ptr1": _BF16}, _RED, _RED_C),
             ADD_NORM, _norm_relations,
             (Call(IR.InductorAddRMSNorm, _norm_statics, (Read("in_ptr0"), Read("in_ptr1"), Read("in_ptr2")), ("out_ptr1",)),), _red_scalars,
             config=True),
    Template("ADD2_NORM", "reduction",
             _sig({"in_ptr0": _BF16, "in_ptr1": _BF16, "in_ptr2": _BF16, "in_ptr3": _BF16, "out_ptr1": _BF16, "out_ptr2": _BF16}, _RED, _RED_C),
             ADD2_NORM, _norm_relations,
             (Call(IR.InductorAdd2RMSNorm, _norm_statics, (Read("in_ptr0"), Read("in_ptr1"), Read("in_ptr2"), Read("in_ptr3")),
                   ("out_ptr1", "out_ptr2")),), _red_scalars, config=True),
    Template("ADD2_NORM_FINAL", "reduction",
             _sig({"in_out_ptr0": _BF16, "in_ptr0": _BF16, "in_ptr1": _BF16, "in_ptr2": _BF16}, _RED, _RED_C),
             ADD2_NORM_FINAL, _norm_relations,
             (Call(IR.InductorAdd2RMSNormFinal, _norm_statics, (Read("in_out_ptr0"), Read("in_ptr0"), Read("in_ptr1"), Read("in_ptr2")),
                   ("in_out_ptr0",)),), _red_scalars, config=True),
    Template("SILU_MUL", "pointwise", _sig({"in_ptr0": _BF16, "out_ptr0": _BF16}, ("xnumel",), ("XBLOCK",)),
             SILU_MUL, lambda lit: isinstance(lit["I"], int) and lit["I"] > 0 and lit["I2"] == 2 * lit["I"],
             (Call(IR.InductorSiluMul, lambda lit: {"I": lit["I"]}, (Read("in_ptr0"),), ("out_ptr0",)),),
             lambda lit: {"xnumel": ("tokens", lit["I"])}),
    Template("ROPE_QK", "pointwise",
             _sig({"in_ptr0": _BF16, "in_ptr1": "*i64", "in_ptr2": _BF16, "out_ptr0": _BF16, "out_ptr1": _BF16}, ("xnumel_0", "xnumel_1"), ("XBLOCK",)),
             ROPE_QK, _rope_relations,
             (Call(IR.InductorRoPE, lambda lit: {"NHEADS": lit["KVH"], "D": lit["D"]},
                   (Read("in_ptr0", cols=lambda lit: (lit["QLEN"], lit["QLEN"] + lit["KLEN"])), Read("in_ptr2", row_of="in_ptr1")), ("out_ptr0",)),
              Call(IR.InductorRoPE, lambda lit: {"NHEADS": lit["H"], "D": lit["D"]},
                   (Read("in_ptr0", cols=lambda lit: (0, lit["QLEN"])), Read("in_ptr2", row_of="in_ptr1")), ("out_ptr1",))),
             lambda lit: {"xnumel_0": ("tokens", lit["KLEN"]), "xnumel_1": ("tokens", lit["QLEN"])}),
)

_BLOCK = re.compile(r"^\w+ = async_compile\.triton\('(\w+)', '''(.*?)''', device_str='cuda'\)$", re.M | re.S)
_JIT = re.compile(r"^@triton\.jit\ndef (\w+)\(", re.M)
_FLAGS = ("'enable_fp_fusion': True", "'disable_ftz': False", "cc=90")


def kernel_sources(wrapper_src: str) -> dict[str, str]:
    """{launch-site name: generated source} of a wrapper's `async_compile.triton` blocks (a name's last block, the Runner's)."""
    return {m.group(1): m.group(2) for m in _BLOCK.finditer(wrapper_src)}


def source_sha256(source: str) -> str:
    """The kernel module's content hash: Inductor writes the block's text as the module it loads."""
    return hashlib.sha256(source.encode()).hexdigest()


def launch_config(record: dict[str, Any] | None, source: str) -> dict[str, Any]:
    """The one launch config the serving process ran the kernel with: its record's single candidate."""
    rec = (record or {}).get(source_sha256(source))
    if rec is None:
        raise Unmodelled(f"no launch config recorded for kernel source {source_sha256(source)[:16]}")
    cands, ran = rec.get("candidates") or [], rec.get("executed") or []
    if len(cands) != 1 or any(c != cands[0] for c in ran):
        raise Unmodelled(f"kernel source {source_sha256(source)[:16]}: {len(cands)} candidate configs, ran {ran}; one candidate is required "
                         f"(an autotuned choice is not a function of the row config)")
    return cands[0]


def _function_text(rest: str) -> str:
    """The jit function's text: its lines up to the first one at column 0 (a combo kernel's module goes on with its benchmark
    harness, host code that never runs in the served process)."""
    lines = rest.split("\n")
    end = next((i for i, l in enumerate(lines[1:], 1) if l and not l[0].isspace()), len(lines))
    return "\n".join(lines[:end]).rstrip()


def recognize(name: str, source: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """The template a generated kernel source is, its literals, and its calls with bound Definition ids; `Unmodelled` otherwise."""
    m = _JIT.search(source)
    if m is None:
        raise Unmodelled(f"{name}: no @triton.jit def")
    head, body = source[:m.start()], "def KERNEL(" + _function_text(source[m.end():])
    t, got = next(((t, g) for t in TEMPLATES if (g := t.pattern.fullmatch(body))), (None, None))
    if t is None:
        raise Unmodelled(f"{name}: its source is none of the templates {[t.name for t in TEMPLATES]}")
    lit = {k: _num(v) for k, v in got.groupdict().items()}
    heur = re.search(r"^@triton_heuristics\.(\w+)\(", head, re.M)
    sig = re.search(r"'signature': (\{[^{}]*\})", head)
    if heur is None or heur.group(1) != t.heuristic:
        raise Unmodelled(f"{name} ({t.name}): heuristic {heur and heur.group(1)}, the template's is {t.heuristic}")
    if sig is None or ast.literal_eval(sig.group(1)) != t.signature:
        raise Unmodelled(f"{name} ({t.name}): signature {sig and sig.group(1)} is not {t.signature}")
    missing = [f for f in _FLAGS if f not in head]
    if missing:
        raise Unmodelled(f"{name} ({t.name}): its triton_meta lacks {missing}")
    if not t.relations(lit):
        raise Unmodelled(f"{name} ({t.name}): literals {lit} break the template's relations")
    if t.config:
        if config is None:
            raise Unmodelled(f"{name} ({t.name}): its launch config is a static and none was given")
        kw = config.get("kwargs") or {}
        lit.update(XBLOCK=kw.get("XBLOCK"), R0_BLOCK=kw.get("R0_BLOCK"), NUM_WARPS=config.get("num_warps"))
    calls = []
    for c in t.calls:
        try:
            spec = bind(c.fn, **c.statics(lit))
            if "NUM_WARPS" in spec.bindings:
                IR.norm_layout(**{k: spec.bindings[k] for k in ("N", "EPS", "XBLOCK", "R0_BLOCK", "NUM_WARPS")})
        except (TypeError, ValueError) as e:
            raise Unmodelled(f"{name} ({t.name}): {e}") from None
        ins, out = spec.signature()
        reads = []
        for r, (_n, ty) in zip(c.reads, ins):
            d: dict[str, Any] = {"param": r.param, "numel": numel(ty)}
            if r.cols is not None:
                d["cols"] = list(r.cols(lit))
            if r.row_of is not None:
                d["row_of"] = r.row_of
            reads.append(d)
        outs = out.fields if isinstance(out, Tuple) else (out,)
        if len(ins) != len(c.reads) or len(outs) != len(c.writes):
            raise Unmodelled(f"{name} ({t.name}): {spec.id} takes {len(ins)} inputs and gives {len(outs)} outputs, the template "
                             f"reads {len(c.reads)} and writes {len(c.writes)}")
        calls.append({"definition": spec.id, "base": spec.base_id, "statics": c.statics(lit), "reads": reads,
                      "writes": [{"param": p, "numel": numel(ty)} for p, ty in zip(c.writes, outs)]})
    return {"template": t.name, "literals": lit, "config": config, "source_sha256": source_sha256(source),
            "params": [p for p, ty in t.signature.items() if ty != "constexpr"],
            "scalars": {p: list(v) for p, v in t.scalars(lit).items()}, "calls": calls}


def numel(ty: Any) -> int:
    """Elements of a Definition's per-row input or output type (an array of arrays of values)."""
    if isinstance(ty, Array):
        return ty.n * numel(ty.elem)
    if isinstance(ty, Value):
        return 1
    raise Unmodelled(f"type {ty} is not an array of values")
