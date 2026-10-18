"""The generated source of each Inductor kernel `acquire.inductor_templates` models, with `{NAME}` holes for its literals: the
templates themselves, kept apart from the matching that binds them."""


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

QK_NORM_ROPE = """\
def KERNEL(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr3, xnumel_0, xnumel_1, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    pid = tl.program_id(0)
    num_xblocks_0 = tl.cdiv(xnumel_0, XBLOCK)
    num_xblocks_1 = num_xblocks_0 + tl.cdiv(xnumel_1, XBLOCK)
    if pid < num_xblocks_0:
        pid_offset = pid
        r0_numel = {D}
        rnumel = r0_numel
        RBLOCK: tl.constexpr = R0_BLOCK
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
        xmask = xindex < xnumel_0
        r0_base = tl.arange(0, R0_BLOCK)[None, :]
        rbase = r0_base
        x0 = (xindex % {H})
        x1 = xindex // {H}
        _tmp4 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
        x3 = xindex
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_2 = r0_index
            tmp0 = tl.load(in_ptr0 + (r0_2 + {D}*x0 + {QKV}*x1), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
            tmp1 = tmp0.to(tl.float32)
            tmp2 = tmp1 * tmp1
            tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
            tmp5 = _tmp4 + tmp3
            _tmp4 = tl.where(r0_mask & xmask, tmp5, _tmp4)
        tmp4 = tl.sum(_tmp4, 1)[:, None]
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_2 = r0_index
            tmp6 = (r0_2).to(tl.int32)
            tmp7 = tl.full([1, 1], 0, tl.int64)
            tmp8 = tmp6 >= tmp7
            tmp9 = (r0_2).to(tl.int64)
            tmp10 = (tmp9).to(tl.int64)
            tmp11 = tl.full([1, 1], {HALF}, tl.int64)
            tmp12 = tmp10 < tmp11
            tmp13 = tl.load(in_ptr0 + ({D}*x0 + {QKV}*x1 + (r0_2)), r0_mask & tmp12 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
            tmp14 = tmp13.to(tl.float32)
            tmp15 = tl.full([1, 1], {NF}, tl.float32)
            tmp16 = (tmp4 / tmp15)
            tmp17 = tl.full([1, 1], {EPS}, tl.float32)
            tmp18 = tmp16 + tmp17
            tmp19 = libdevice.rsqrt(tmp18)
            tmp20 = tmp14 * tmp19
            tmp21 = tmp20.to(tl.float32)
            tmp22 = tl.load(in_ptr1 + (tl.broadcast_to(r0_2, [XBLOCK, R0_BLOCK])), r0_mask & tmp12 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
            tmp23 = tmp21 * tmp22
            tmp24 = tl.load(in_ptr2 + (tl.broadcast_to(x1, [XBLOCK, R0_BLOCK])), r0_mask & tmp12 & xmask, eviction_policy='evict_last', other=0.0)
            tmp25 = (tl.full([1, 1], {P}, tl.int32)).to(tl.int32)
            tmp26 = tmp24 + tmp25
            tmp27 = tmp24 < 0
            tmp28 = tl.where(tmp27, tmp26, tmp24)
            tl.device_assert(((0 <= tl.broadcast_to(tmp28, [XBLOCK, R0_BLOCK])) & (tl.broadcast_to(tmp28, [XBLOCK, R0_BLOCK]) < {P})) | ~(r0_mask & tmp12 & xmask), "index out of bounds: 0 <= tl.broadcast_to(tmp28, [XBLOCK, R0_BLOCK]) < {P}")
            tmp30 = tl.load(in_ptr3 + (tl.broadcast_to({D}*tmp28 + (r0_2), [XBLOCK, R0_BLOCK])), r0_mask & tmp12 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
            tmp31 = tmp23 * tmp30
            tmp32 = tl.load(in_ptr0 + ({HALF} + {D}*x0 + {QKV}*x1 + (r0_2)), r0_mask & tmp12 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
            tmp33 = tmp32.to(tl.float32)
            tmp34 = tmp33 * tmp19
            tmp35 = tmp34.to(tl.float32)
            tmp36 = tl.load(in_ptr1 + (tl.broadcast_to({HALF} + (r0_2), [XBLOCK, R0_BLOCK])), r0_mask & tmp12 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
            tmp37 = tmp35 * tmp36
            tmp38 = tl.load(in_ptr3 + (tl.broadcast_to({HALF} + {D}*tmp28 + (r0_2), [XBLOCK, R0_BLOCK])), r0_mask & tmp12 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
            tmp39 = tmp37 * tmp38
            tmp40 = tmp31 - tmp39
            tmp41 = tl.full(tmp40.shape, 0.0, tmp40.dtype)
            tmp42 = tl.where(tmp12, tmp40, tmp41)
            tmp43 = tmp6 >= tmp11
            tmp44 = tl.full([1, 1], {D}, tl.int64)
            tmp45 = tmp6 < tmp44
            tmp46 = tl.load(in_ptr0 + ({HALF} + {D}*x0 + {QKV}*x1 + ((-{HALF}) + r0_2)), r0_mask & tmp43 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
            tmp47 = tmp46.to(tl.float32)
            tmp48 = tl.full([1, 1], {NF}, tl.float32)
            tmp49 = (tmp4 / tmp48)
            tmp50 = tl.full([1, 1], {EPS}, tl.float32)
            tmp51 = tmp49 + tmp50
            tmp52 = libdevice.rsqrt(tmp51)
            tmp53 = tmp47 * tmp52
            tmp54 = tmp53.to(tl.float32)
            tmp55 = tl.load(in_ptr1 + (tl.broadcast_to({HALF} + ((-{HALF}) + r0_2), [XBLOCK, R0_BLOCK])), r0_mask & tmp43 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
            tmp56 = tmp54 * tmp55
            tmp57 = tl.load(in_ptr2 + (tl.broadcast_to(x1, [XBLOCK, R0_BLOCK])), r0_mask & tmp43 & xmask, eviction_policy='evict_last', other=0.0)
            tmp58 = (tl.full([1, 1], {P}, tl.int32)).to(tl.int32)
            tmp59 = tmp57 + tmp58
            tmp60 = tmp57 < 0
            tmp61 = tl.where(tmp60, tmp59, tmp57)
            tl.device_assert(((0 <= tl.broadcast_to(tmp61, [XBLOCK, R0_BLOCK])) & (tl.broadcast_to(tmp61, [XBLOCK, R0_BLOCK]) < {P})) | ~(r0_mask & tmp43 & xmask), "index out of bounds: 0 <= tl.broadcast_to(tmp61, [XBLOCK, R0_BLOCK]) < {P}")
            tmp63 = tl.load(in_ptr3 + (tl.broadcast_to({D}*tmp61 + ((-{HALF}) + r0_2), [XBLOCK, R0_BLOCK])), r0_mask & tmp43 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
            tmp64 = tmp56 * tmp63
            tmp65 = tl.load(in_ptr0 + ({D}*x0 + {QKV}*x1 + ((-{HALF}) + r0_2)), r0_mask & tmp43 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
            tmp66 = tmp65.to(tl.float32)
            tmp67 = tmp66 * tmp52
            tmp68 = tmp67.to(tl.float32)
            tmp69 = tl.load(in_ptr1 + (tl.broadcast_to((-{HALF}) + r0_2, [XBLOCK, R0_BLOCK])), r0_mask & tmp43 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
            tmp70 = tmp68 * tmp69
            tmp71 = tl.load(in_ptr3 + (tl.broadcast_to({HALF} + {D}*tmp61 + ((-{HALF}) + r0_2), [XBLOCK, R0_BLOCK])), r0_mask & tmp43 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
            tmp72 = tmp70 * tmp71
            tmp73 = tmp64 + tmp72
            tmp74 = tl.full(tmp73.shape, 0.0, tmp73.dtype)
            tmp75 = tl.where(tmp43, tmp73, tmp74)
            tmp76 = tl.where(tmp12, tmp42, tmp75)
            tl.store(out_ptr1 + (r0_2 + {D}*x3), tmp76, r0_mask & xmask)
    elif pid < num_xblocks_1:
        pid_offset = pid - num_xblocks_0
        r0_numel = {D}
        rnumel = r0_numel
        RBLOCK: tl.constexpr = R0_BLOCK
        xoffset = pid_offset * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
        xmask = xindex < xnumel_1
        r0_base = tl.arange(0, R0_BLOCK)[None, :]
        rbase = r0_base
        x4 = (xindex % {KVH})
        x5 = xindex // {KVH}
        _tmp81 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
        x7 = xindex
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_6 = r0_index
            tmp77 = tl.load(in_ptr0 + ({QLEN} + r0_6 + {D}*x4 + {QKV}*x5), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
            tmp78 = tmp77.to(tl.float32)
            tmp79 = tmp78 * tmp78
            tmp80 = tl.broadcast_to(tmp79, [XBLOCK, R0_BLOCK])
            tmp82 = _tmp81 + tmp80
            _tmp81 = tl.where(r0_mask & xmask, tmp82, _tmp81)
        tmp81 = tl.sum(_tmp81, 1)[:, None]
        for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
            r0_index = r0_offset + r0_base
            r0_mask = r0_index < r0_numel
            roffset = r0_offset
            rindex = r0_index
            r0_6 = r0_index
            tmp83 = (r0_6).to(tl.int32)
            tmp84 = tl.full([1, 1], 0, tl.int64)
            tmp85 = tmp83 >= tmp84
            tmp86 = (r0_6).to(tl.int64)
            tmp87 = (tmp86).to(tl.int64)
            tmp88 = tl.full([1, 1], {HALF}, tl.int64)
            tmp89 = tmp87 < tmp88
            tmp90 = tl.load(in_ptr0 + ({QLEN} + {D}*x4 + {QKV}*x5 + (r0_6)), r0_mask & tmp89 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
            tmp91 = tmp90.to(tl.float32)
            tmp92 = tl.full([1, 1], {NF}, tl.float32)
            tmp93 = (tmp81 / tmp92)
            tmp94 = tl.full([1, 1], {EPS}, tl.float32)
            tmp95 = tmp93 + tmp94
            tmp96 = libdevice.rsqrt(tmp95)
            tmp97 = tmp91 * tmp96
            tmp98 = tmp97.to(tl.float32)
            tmp99 = tl.load(in_ptr4 + (tl.broadcast_to(r0_6, [XBLOCK, R0_BLOCK])), r0_mask & tmp89 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
            tmp100 = tmp98 * tmp99
            tmp101 = tl.load(in_ptr2 + (tl.broadcast_to(x5, [XBLOCK, R0_BLOCK])), r0_mask & tmp89 & xmask, eviction_policy='evict_last', other=0.0)
            tmp102 = (tl.full([1, 1], {P}, tl.int32)).to(tl.int32)
            tmp103 = tmp101 + tmp102
            tmp104 = tmp101 < 0
            tmp105 = tl.where(tmp104, tmp103, tmp101)
            tl.device_assert(((0 <= tl.broadcast_to(tmp105, [XBLOCK, R0_BLOCK])) & (tl.broadcast_to(tmp105, [XBLOCK, R0_BLOCK]) < {P})) | ~(r0_mask & tmp89 & xmask), "index out of bounds: 0 <= tl.broadcast_to(tmp105, [XBLOCK, R0_BLOCK]) < {P}")
            tmp107 = tl.load(in_ptr3 + (tl.broadcast_to({D}*tmp105 + (r0_6), [XBLOCK, R0_BLOCK])), r0_mask & tmp89 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
            tmp108 = tmp100 * tmp107
            tmp109 = tl.load(in_ptr0 + ({KHI} + {D}*x4 + {QKV}*x5 + (r0_6)), r0_mask & tmp89 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
            tmp110 = tmp109.to(tl.float32)
            tmp111 = tmp110 * tmp96
            tmp112 = tmp111.to(tl.float32)
            tmp113 = tl.load(in_ptr4 + (tl.broadcast_to({HALF} + (r0_6), [XBLOCK, R0_BLOCK])), r0_mask & tmp89 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
            tmp114 = tmp112 * tmp113
            tmp115 = tl.load(in_ptr3 + (tl.broadcast_to({HALF} + {D}*tmp105 + (r0_6), [XBLOCK, R0_BLOCK])), r0_mask & tmp89 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
            tmp116 = tmp114 * tmp115
            tmp117 = tmp108 - tmp116
            tmp118 = tl.full(tmp117.shape, 0.0, tmp117.dtype)
            tmp119 = tl.where(tmp89, tmp117, tmp118)
            tmp120 = tmp83 >= tmp88
            tmp121 = tl.full([1, 1], {D}, tl.int64)
            tmp122 = tmp83 < tmp121
            tmp123 = tl.load(in_ptr0 + ({KHI} + {D}*x4 + {QKV}*x5 + ((-{HALF}) + r0_6)), r0_mask & tmp120 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
            tmp124 = tmp123.to(tl.float32)
            tmp125 = tl.full([1, 1], {NF}, tl.float32)
            tmp126 = (tmp81 / tmp125)
            tmp127 = tl.full([1, 1], {EPS}, tl.float32)
            tmp128 = tmp126 + tmp127
            tmp129 = libdevice.rsqrt(tmp128)
            tmp130 = tmp124 * tmp129
            tmp131 = tmp130.to(tl.float32)
            tmp132 = tl.load(in_ptr4 + (tl.broadcast_to({HALF} + ((-{HALF}) + r0_6), [XBLOCK, R0_BLOCK])), r0_mask & tmp120 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
            tmp133 = tmp131 * tmp132
            tmp134 = tl.load(in_ptr2 + (tl.broadcast_to(x5, [XBLOCK, R0_BLOCK])), r0_mask & tmp120 & xmask, eviction_policy='evict_last', other=0.0)
            tmp135 = (tl.full([1, 1], {P}, tl.int32)).to(tl.int32)
            tmp136 = tmp134 + tmp135
            tmp137 = tmp134 < 0
            tmp138 = tl.where(tmp137, tmp136, tmp134)
            tl.device_assert(((0 <= tl.broadcast_to(tmp138, [XBLOCK, R0_BLOCK])) & (tl.broadcast_to(tmp138, [XBLOCK, R0_BLOCK]) < {P})) | ~(r0_mask & tmp120 & xmask), "index out of bounds: 0 <= tl.broadcast_to(tmp138, [XBLOCK, R0_BLOCK]) < {P}")
            tmp140 = tl.load(in_ptr3 + (tl.broadcast_to({D}*tmp138 + ((-{HALF}) + r0_6), [XBLOCK, R0_BLOCK])), r0_mask & tmp120 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
            tmp141 = tmp133 * tmp140
            tmp142 = tl.load(in_ptr0 + ({QLEN} + {D}*x4 + {QKV}*x5 + ((-{HALF}) + r0_6)), r0_mask & tmp120 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
            tmp143 = tmp142.to(tl.float32)
            tmp144 = tmp143 * tmp129
            tmp145 = tmp144.to(tl.float32)
            tmp146 = tl.load(in_ptr4 + (tl.broadcast_to((-{HALF}) + r0_6, [XBLOCK, R0_BLOCK])), r0_mask & tmp120 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
            tmp147 = tmp145 * tmp146
            tmp148 = tl.load(in_ptr3 + (tl.broadcast_to({HALF} + {D}*tmp138 + ((-{HALF}) + r0_6), [XBLOCK, R0_BLOCK])), r0_mask & tmp120 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
            tmp149 = tmp147 * tmp148
            tmp150 = tmp141 + tmp149
            tmp151 = tl.full(tmp150.shape, 0.0, tmp150.dtype)
            tmp152 = tl.where(tmp120, tmp150, tmp151)
            tmp153 = tl.where(tmp89, tmp119, tmp152)
            tl.store(out_ptr3 + (r0_6 + {D}*x7), tmp153, r0_mask & xmask)
    else:
        pass
"""
