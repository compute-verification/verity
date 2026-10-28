"""The `dither-a` arm: option A of the credit form, the provisional credit form.
Under it the noised work is credited on the operands A, B of a served product that is value-checked (on every sampled
proof block z_u = A_u B_u^T, proved and committed before the sample, y from the committed z), so the served product may come from
a plain GEMM beside the receipted one.  Every run is labelled by `forms.form_of` (`Side.label`): form=`forms.FORM_A`
only for -so<D> with -zy (`dither_so`) on a ruled-in form, whose served epilogue commits the per-unit
z/y records under round s's root before coin s (the receipts need no z of their own); form=beside (unadopted arm) for
-so's -ab/-abn and -ov arms; form=beside, receipts beside the served GEMM, for the row z tables or no z, and for every
arm outside the statement's order (-so), which this module's arms all are.

Serving stream, per int8-exact linear (`dithera::mm`, one op opaque to Inductor): exactlin's quantizer, which also stores
the call's live rows and scales (rows < m, unless the step's flags say no-write) into the window's A slot
(`exactlin_quant_rows_dithera`); then exactlin's GEMM at exactlin's config for the call (`exactlin.pick`), whose
epilogue stores z = X W^T into the window's z slot (rows < m; rows m to T zero, from extra row tiles past M) beside
y = dequant(z), exact-fused's y bit for bit (`dithera_mm`, `dithera_mm_tma`).

Side stream, forked after the quantizer (it reads only window-owned buffers): the call's A table (`h1c_a_call`, the
window's form: DITHER_A_FORM, 3 unless set), its seeds (h1r version 0, from the window's salt and root(A_j), root(B_j))
and its receipts of the noised product T = (A + N_A)(B + N_B)^T only, no peel and no z: h1r's block or split form per
call (`unit="h1r"`: versions 3 and 4, wide where the call's column tiles outnumber the SMs, as dither-1s-mix's units)
or h1u.cu's kh_unit_ws (`unit="kh"`), or the K split build's kh_one (`unit="hbk"`, h1k where the call's live units
fit, its z and y to scratch, read under a flagless copy of the step; `unit="hbr"`, the same call in the H1K_RONLY
build, which writes no z or y and skips their MMA), into a staging buffer.  At the forward's last call `dithera_commit` copies the
staging receipts into the plan's, which the receipt trees commit, unless the step's flags say no-write (F_NW_UNITS),
and under F_FAULT flips bit FAULT_SHIFT of word 0 of unit 0's receipt of every call.  The side stream joins the serving
stream at the forward's last call, and under piecewise graphs at every qkv call too (each piece's capture must join its
forks, and a piece ends at an attention).  Between steps dither_one's Window builds the admit and z tables, the step
trees, the round and the receipt trees, as dither-1s's: the A table before the call's units, z before the sample, the
step and receipt trees between steps, and nothing waits on the side stream but those.  With z="zy" (the arms' -zy) z
is committed as the per-unit z/y leaves instead of the window's row z tables: the epilogue writes one record a 32 x 32 unit (its z,
then its y = dequant(z); a slot past m is zero codes under s_a = +0, so its z is 0 and its y h1zy's dq(0, +0, s_w), as
`scaled_mm_zy`'s) and the window builds every call's leaf table between steps in the row z tables'
place, beside the receipt trees, each root the call's leaf of the z-step tree, so the round over them still comes before
the sample (every leaf in its registered root before the sampling coin, nothing written after it).  With ao="a" (the arms' -ao) the
A tables run on a third stream above the units' priority: A_w waits on its quantizer and the units of call w on
A_w, so each call's A table still precedes its seeds and units, and A_(w+1) runs beside call w's units (the same
launches; only the streams they wait on change); with ao="as" (-aos; kh and hbk, whose seeds are a launch of their own)
each call's seeds follow its A table on that stream, and its units wait on both; ao="m" and "ms" (-aom, -aoms) are the
same on NA_MULTI A streams, call w's on stream w % NA_MULTI, since no call's table reads another's (each call's
counter is in its own prefix words), and a join waits on every A stream used since the last; ao="mn" (-aomn; hbk
and hbr) is "ms" with each call's noise there too, after its seeds (kh_one_noise into a ring of NA_MULTI workspaces,
the side stream's kh_one_units reading it), so the side stream runs the units alone.  With pt=True (-pt) the
window builds the receipt trees and the admit tables between steps on streams of their own (`Window.tree_streams`).
With ls=True (-ls) each step's seeds read the live salt of the round before it (`Window.step_salt`).
The round's leaves are a-step, admit-step, z-step, then y-step (under "zy") and receipt-step (`dither_one.ROUND_EXTRA`).
Under "zy" the serving stream copies each call's served y, the bytes the next layer reads, into y records after its GEMM
(`dithera_ycommit`, dq(0, +0, s_w) past m as the records'); between steps the window builds the served y's tables, whose
roots are the y-step leaves, and a y table over the zy records' y halves, and a call whose two roots differ refuses its
step. The
receipt-step leaf is the calls' receipt-table roots, so the round recorded before the next coin fixes the receipts;
the round waits on the receipt trees for it. Open: the z/y leaves still go under the round recorded after the coin the
step's noise read (the statement's order needs the units after their round's coin, beside the next step: dither_so).

    import dither_a as DA
    DA.patch(t=T, h1c="libh1c.so", h1o="libh1o.so", unit="h1r")   # before LLM()
    llm = LLM(model=int8_exact_checkpoint, ...)
    win = DO.hook(llm); win.captures = DA.CAPTURES

`check` on the gate, control and fault captures (each step's buffers 0xA5 before its kernels wrote them): every call's
A, admit and z roots and the round against commit.py; z = A B^T (int_mm, the A the A table committed) on every call and
zero past m; the served y (dumped) = dequant(z); every record's y (-zy) = dequant(z) at s_a, and dq(0, +0, s_w) past m,
as the served y's records; every receipt = h1r v1's on the saved slots under the window's salt and
the CPU's roots (another unit implementation); three units' receipts by H1V3's Definition on the CPU (`h1v3_unit`), and
their receipts and z by H1V3's own code with the peel (`h1v3_itself`: this source's h1tile unless DITHER_H1V3_TREE names a
tree; one without the peel is refused).  The control (nothing written: the A slot, z, receipts) must be refused on z and on the receipts; the
fault must be refused on the receipts at exactly the flipped words, z and y untouched; "fault_y" (bit 0 of every call's
served y[0, 0] flipped in place) must be refused at run time on every call's y root, every other commitment kept; and
"fault_rr" (the round recorded without the receipt roots) must be refused on the receipt-step leaf and the round alone.
With DITHER_LANE_REFS the gate's check also holds the step's drawn leaves' lane partials (served_h1/partials.py, on the
cpu and the served h1u build's kh_lanes) to each other and to the receipts, and to SERVE-REG's gated lane
(`lane_refs.served`): in the gate (=1), or after serving from the records it saves (=PATH, `lane_refs.py PATH`).
"""
from __future__ import annotations

import ctypes as C
import sys
from pathlib import Path

import numpy as np
import torch
import triton
import triton.language as tl

import dither_fused as DF
import dither_one as DO
import forms as F
from dither_fused import CM, CR, H1R, Call, unwords
import h1c as H1CM
from h1c import Dom, Tab
from hb_side import auto_j

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "exactlin"))
import exactlin as EL
from verity_config.settings import of

settings = of(__file__)


def arm_label() -> dict:
    """This process's dither-a arm's {"form", "form_basis"} (`forms.form_of`, through its Side)."""
    return SIDE[0].label() if SIDE else F.form_of("dither-a")


# "fault_y" (the served y tampered after the GEMM) and "fault_rr" (the round recorded without the receipt roots), set
# by `patch`: "fault_y" only under z="zy", which has the units' y to compare
CAPTURES = ("gate", "control", "fault", "fault_rr")
LIB_ENV = DO.LIB_ENV
LIB = "dithera"
UNITS = ("h1r", "kh", "hbk", "hbr"); KONE = ("hbk", "hbr")
# z's commitment: "row" each slot's N int32 a hiding leaf (dither_one's z table, built between steps); "zy" the per-unit
# leaf: one hiding leaf a credited UR x UC unit holding its z (int32) then its y = dequant(z)
# (bf16), past m z 0 and y dq(0, +0, s_w) (h1zy's empty slot), which the epilogue writes as one record a unit and the
# window commits between steps (ZY_KIND, provisional until INT-TIE names the fields); "none" no z stored
# and no z table, y only: the leaves'
# cost floor, never credited
ZMODES = {"none": 0, "row": 1, "zy": 2}
# nothing, the A tables, or the A tables and (kh, hbk) the seeds on the A stream; "m" and "ms" the same on NA_MULTI
# A streams, call w's on stream w % NA_MULTI
AO = ("", "a", "as", "m", "ms", "mn")
AO_SEEDS = ("as", "ms", "mn")
NA_MULTI = 4
ZY_KIND = "zy-unit-table"
# a call's y alone, one hiding leaf a unit (its UR x UC bf16, dq(0, +0, s_w) past m): the served y's table and the units' (the -zy
# records' y halves) in this one domain and label, so their roots are equal exactly when the bytes are
Y_KIND = "y-unit-table"
ZY1_ENV = "DITHER_ZY1"
# "1": the z/y and y tables' rows in a row tile past m take their x from `dead_rows` (hashed once a window, from the
# bytes such a row holds) and hash no segments; the same leaves and roots
ZDEAD_ENV = "DITHER_ZDEAD"
# "1" (-zy under DITHER_ZDEAD): the compare of the served y with the records' y halves made on their bytes in
# `dithera_ycommit` (the rows the two y tables hash), a call that differs refusing its step as unequal roots do; the
# units' y tables are built, and their roots compared, only at a captured step, whose check reads them and whose
# verdict wants both compares to refuse the same calls
YCMP_ENV = "DITHER_YCMP"
# the gate step's option A partials against SERVE-REG's gated lane (served_h1/lane_refs.py): 1 checks them in the gate
# (one process: the arm's worker is a daemon); a path saves their records there for `lane_refs.py PATH` after serving
LANE_REFS_ENV = "DITHER_LANE_REFS"
# "N,K,MMAX=CFG+..." (exactlin's Cfg text): call (N, K)'s GEMM config in place of exactlin.pick's at the M its CONFIGS
# entry (MMAX, ..) takes, a CFG's ":gG" its tile group (`mm_group`).  A split K's int32 partial sums add up to the same z,
# so z and y are the same bits
MM_CFG_ENV = "DITHER_MM_CFG"
# the units' stream priority (torch's: lower runs first; default 0, below the serving stream's -1 under DITHER_SO_PRIO)
UNITS_PRIO_ENV = "DITHER_SO_UNITS_PRIO"
UNIT = (32, 32)     # (UR, UC): the receipts' units; a narrower down_proj unit changes only this
H1R_BLOCK, H1R_SPLIT = 3, 4     # h1r_launch's versions: a unit a block, a unit six warps
M32 = DO.M32


@triton.jit
def exactlin_quant_rows_dithera(x_ptr, q_ptr, s_ptr, slot_ptr, scale_ptr, S_ptr, K, stride_x, stride_q,
                                BLOCK_K: tl.constexpr):
    # exactlin_quant_rows' body, unchanged
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK_K)
    mask = offs < K
    x = tl.load(x_ptr + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)
    a = tl.max(tl.abs(x), axis=0)
    s = tl.div_rn(tl.maximum(a, 1e-30), 127.0)
    sb = s.to(tl.int32, bitcast=True)
    s1 = ((sb & 0x7FFFFF) | 0x3F800000).to(tl.float32, bitcast=True)
    p = ((254 - (sb >> 23)) << 23).to(tl.float32, bitcast=True)
    xs = x * p
    small = tl.abs(xs) < 0.25
    q = EL._rni_s32(tl.div_rn(tl.where(small, 1.0, xs), tl.broadcast_to(s1, (BLOCK_K,))))
    q = tl.where(small, 0, q)
    q = tl.minimum(tl.maximum(q, -127), 127).to(tl.int8)
    tl.store(q_ptr + row * stride_q + offs, q, mask=mask)
    tl.store(s_ptr + row, s)
    # the window's A slot: S = (parity, m, pad, flags), F_NW_TILES = 2
    m = tl.load(S_ptr + 1)
    flags = tl.load(S_ptr + 3)
    if (row < m) & ((flags & 2) == 0):
        tl.store(slot_ptr + row * K + offs, q, mask=mask)
        tl.store(scale_ptr + row, s)


@triton.jit
def _dithera_epilogue(acc, sa_ptr, sb_ptr, c_ptr, z_ptr, y_ptr, d_ptr, offs_m, offs_n, M, N, T, m, flags, stride_cm,
                      ZL: tl.constexpr, UR: tl.constexpr, UC: tl.constexpr):
    mask_m = offs_m < M
    mask_n = offs_n < N
    EL._epilogue(acc, sa_ptr, sb_ptr, c_ptr, offs_m, offs_n, mask_m, mask_n, stride_cm)
    if (flags & 1) != 0:    # F_DUMP: the served y into the window's dump too
        EL._epilogue(acc, sa_ptr, sb_ptr, d_ptr, offs_m, offs_n, mask_m, mask_n, N)
    if ZL != 0:
        if (flags & 2) == 0:    # F_NW_TILES: no z (nor y leaf)
            live = (offs_m < m)[:, None]
            zv = tl.where(live, acc, 0)
            mask = (offs_m < T)[:, None] & mask_n[None, :]
            if ZL == 1:     # row-major, a slot's N int32 a leaf
                tl.store(z_ptr + offs_m[:, None] * N + offs_n[None, :], zv, mask=mask)
            else:
                # one record a UR x UC unit (row tile i // UR, column tile j // UC; record (i // UR) (N / UC) + j // UC):
                # its z int32 row-major, then its y = dequant(z) bf16 row-major, as dither_a.ZY's leaf; a slot past m is
                # zero codes under s_a = +0 (scaled_mm_zy's empty row), so its y is h1zy's dq(0, +0, s_w), not 0
                r = offs_m[:, None] % UR
                c = offs_n[None, :] % UC
                u = (offs_m[:, None] // UR) * (N // UC) + offs_n[None, :] // UC
                tl.store(z_ptr + u * (UR * UC * 3 // 2) + r * UC + c, zv, mask=mask)
                sa = tl.load(sa_ptr + offs_m, mask=mask_m, other=0.0)
                sa = tl.where(offs_m < m, sa, 0.0)
                sb = tl.load(sb_ptr + offs_n, mask=mask_n, other=0.0)
                yv = zv.to(tl.float32) * sa[:, None]
                yv = yv * sb[None, :]
                yb = yv.to(tl.bfloat16).to(tl.int16, bitcast=True)
                tl.store(y_ptr + u * (UR * UC * 3) + 2 * UR * UC + r * UC + c, yb, mask=mask)


@triton.jit
def _tile(pid, T, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, GROUP_M: tl.constexpr):
    # launch pid's (row tile, column tile): row-major, or GROUP_M row tiles a group with its column tiles outer, so the
    # CTAs resident at once share a few weight tiles over GROUP_M activation tiles (the same tiles in another order)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    if GROUP_M == 1:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    else:
        per = GROUP_M * num_pid_n
        first = (pid // per) * GROUP_M
        size = tl.minimum(tl.cdiv(T, BLOCK_M) - first, GROUP_M)
        pid_m = first + (pid % per) % size
        pid_n = (pid % per) // size
    return pid_m, pid_n


@triton.jit
def _rows_written(S_ptr, T, m):
    # the rows whose z/y records a step writes: all T, or with S[2] > 0 (DITHER_DEAD_ONCE) those below max(m, S[2]),
    # the rows at or past S[2] holding a row past m's records already
    c = tl.load(S_ptr + 2)
    return tl.where(c > 0, tl.minimum(T, tl.maximum(m, c)), T)


@triton.jit
def dithera_mm(a_ptr, b_ptr, sa_ptr, sb_ptr, c_ptr, z_ptr, y_ptr, d_ptr, S_ptr, ws_ptr, cnt_ptr, M, N, K, T, stride_am,
               stride_bn, stride_cm, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
               SPLIT_K: tl.constexpr, ZL: tl.constexpr, UR: tl.constexpr, UC: tl.constexpr, GROUP_M: tl.constexpr = 1):
    # exactlin_scaled_mm over T rows: a row tile at or past M multiplies nothing and stores only its zero z rows
    pid = tl.program_id(0)
    pid_m, pid_n = _tile(pid, T, N, BLOCK_M, BLOCK_N, GROUP_M)
    m = tl.load(S_ptr + 1)
    flags = tl.load(S_ptr + 3)
    TW = _rows_written(S_ptr, T, m)
    live = pid_m * BLOCK_M < M
    nk = tl.cdiv(K, BLOCK_K)
    per = tl.cdiv(nk, SPLIT_K)
    k_lo = tl.program_id(1) * per
    k_hi = tl.where(live, tl.minimum(k_lo + per, nk), k_lo)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
    offs_k = tl.arange(0, BLOCK_K).to(tl.int64)
    mask_m = offs_m < M
    mask_n = offs_n < N
    a_ptrs = a_ptr + stride_am * offs_m[:, None] + (k_lo * BLOCK_K + offs_k)[None, :]
    b_ptrs = b_ptr + (k_lo * BLOCK_K + offs_k)[:, None] + stride_bn * offs_n[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k in range(k_lo, k_hi):
        mask_k = offs_k < K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0)
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0)
        acc = tl.dot(a, b, acc, out_dtype=tl.int32)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K
    if SPLIT_K == 1:
        _dithera_epilogue(acc, sa_ptr, sb_ptr, c_ptr, z_ptr, y_ptr, d_ptr, offs_m, offs_n, M, N, TW, m, flags, stride_cm, ZL, UR, UC)
    else:
        if live:
            # exactlin_scaled_mm's split-K: the workspace and the tile's counter zero again after the last split
            mask = mask_m[:, None] & mask_n[None, :]
            ws_ptrs = ws_ptr + offs_m[:, None] * N + offs_n[None, :]
            tl.atomic_add(ws_ptrs, acc, mask=mask, sem="relaxed")
            EL._membar_gl(pid)
            tl.debug_barrier()
            done = tl.atomic_add(cnt_ptr + pid, 1, sem="acq_rel")
            if done == SPLIT_K - 1:
                EL._membar_gl(pid)
                tot = tl.atomic_xchg(ws_ptrs, tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32), mask=mask, sem="relaxed")
                tl.atomic_xchg(cnt_ptr + pid, 0, sem="relaxed")
                _dithera_epilogue(tot, sa_ptr, sb_ptr, c_ptr, z_ptr, y_ptr, d_ptr, offs_m, offs_n, M, N, TW, m, flags, stride_cm, ZL, UR, UC)
        elif tl.program_id(1) == 0:
            _dithera_epilogue(acc, sa_ptr, sb_ptr, c_ptr, z_ptr, y_ptr, d_ptr, offs_m, offs_n, M, N, TW, m, flags, stride_cm, ZL, UR, UC)


@triton.jit
def dithera_mm_tma(a_desc, b_desc, sa_ptr, sb_ptr, c_ptr, z_ptr, y_ptr, d_ptr, S_ptr, M, N, K, T, stride_cm,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, WARP_SPECIALIZE: tl.constexpr,
                   ZL: tl.constexpr, UR: tl.constexpr, UC: tl.constexpr, GROUP_M: tl.constexpr = 1):
    pid_m, pid_n = _tile(tl.program_id(0), T, N, BLOCK_M, BLOCK_N, GROUP_M)
    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N
    m = tl.load(S_ptr + 1)
    flags = tl.load(S_ptr + 3)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    nk = tl.where(off_m < M, tl.cdiv(K, BLOCK_K), 0)
    for k in tl.range(0, nk, warp_specialize=WARP_SPECIALIZE):
        a = a_desc.load([off_m, k * BLOCK_K])
        b = b_desc.load([off_n, k * BLOCK_K])
        acc = tl.dot(a, b.T, acc, out_dtype=tl.int32)
    offs_m = off_m + tl.arange(0, BLOCK_M).to(tl.int64)
    offs_n = off_n + tl.arange(0, BLOCK_N).to(tl.int64)
    _dithera_epilogue(acc, sa_ptr, sb_ptr, c_ptr, z_ptr, y_ptr, d_ptr, offs_m, offs_n, M, N, _rows_written(S_ptr, T, m), m,
                      flags, stride_cm, ZL, UR, UC)


@triton.jit
def dithera_ycommit(y_ptr, sb_ptr, dst_ptr, S_ptr, rec_ptr, bad_ptr, M, N, T, stride_y, UR: tl.constexpr,
                    UC: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, CMP: tl.constexpr):
    # the served y (int16 view; the bytes the next layer reads), rows < m, into the call's y records (unit row tile,
    # then column tile; Y_KIND), and past m the epilogue's empty-slot y, dq(0, +0, s_w); F_NW_TILES = 2: nothing
    # written; F_FAULT_Y = 64: bit 0 of y[0, 0] flipped in place first, so the next layer and the record read the
    # tampered y.  CMP (YCMP_ENV): those bytes against the y halves of the call's z/y records (rec_ptr, int16) on the
    # rows of the live UR-row tiles, the rows the y tables hash under DITHER_ZDEAD, and bad_ptr (int32) set to 1 on any
    # that differs
    m = tl.load(S_ptr + 1)
    flags = tl.load(S_ptr + 3)
    if (flags & 2) == 0:
        offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
        offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
        mask_n = (offs_n < N)[None, :]
        ld = ((offs_m < m) & (offs_m < M))[:, None] & mask_n
        v = tl.load(y_ptr + offs_m[:, None] * stride_y + offs_n[None, :], mask=ld, other=0)
        if (flags & 64) != 0:
            first = (offs_m[:, None] == 0) & (offs_n[None, :] == 0)
            v = tl.where(first, (v ^ 1).to(tl.int16), v)
            tl.store(y_ptr + offs_m[:, None] * stride_y + offs_n[None, :], v, mask=ld & first)
        sb = tl.load(sb_ptr + offs_n, mask=offs_n < N, other=0.0)
        pad = (tl.zeros((BLOCK_N,), dtype=tl.float32) * sb).to(tl.bfloat16).to(tl.int16, bitcast=True)
        v = tl.where(ld, v, pad[None, :])
        u = (offs_m[:, None] // UR) * (N // UC) + offs_n[None, :] // UC
        tl.store(dst_ptr + u * (UR * UC) + (offs_m[:, None] % UR) * UC + offs_n[None, :] % UC, v,
                 mask=(offs_m < _rows_written(S_ptr, T, m))[:, None] & mask_n)
        if CMP:
            cm = (offs_m < tl.minimum(T, (m + UR - 1) // UR * UR))[:, None] & mask_n
            r = tl.load(rec_ptr + u * (UR * UC * 3) + 2 * UR * UC + (offs_m[:, None] % UR) * UC + offs_n[None, :] % UC,
                        mask=cm, other=0)
            if tl.max(tl.max((cm & (r != v)).to(tl.int32), axis=1), axis=0) > 0:
                tl.store(bad_ptr, 1)


@triton.jit
def dithera_commit(src_ptr, dst_ptr, S_ptr, n, BLOCK: tl.constexpr):
    # F_NW_UNITS = 4: the step's receipts stay unwritten
    flags = tl.load(S_ptr + 3)
    if (flags & 4) == 0:
        offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        tl.store(dst_ptr + offs, tl.load(src_ptr + offs, mask=offs < n), mask=offs < n)


@triton.jit
def dithera_fault(dst_ptr, S_ptr, at_ptr, L, bit, BLOCK: tl.constexpr):
    # F_FAULT = 8: word `at[w]` of the committed receipts flipped at `bit`, every call w
    flags = tl.load(S_ptr + 3)
    if (flags & 8) != 0:
        i = tl.arange(0, BLOCK)
        at = tl.load(at_ptr + i, mask=i < L, other=0)
        v = tl.load(dst_ptr + at, mask=i < L)
        tl.store(dst_ptr + at, v ^ bit, mask=i < L)


def mm_group(text: str) -> int:
    """A $DITHER_MM_CFG config's tile group, its ":gG" (`_tile`; exactlin's Cfg.parse ignores it), else 1."""
    return next((int(r[1:]) for r in text.split(":")[1:] if r[:1] == "g"), 1)


def _piecewise() -> bool:
    """Whether vLLM is capturing or running piecewise graphs (its forward context's mode); unknown counts as yes."""
    try:
        from vllm.forward_context import get_forward_context, is_forward_context_available
        if not is_forward_context_available():
            return False
        return getattr(getattr(get_forward_context(), "cudagraph_runtime_mode", None), "name", "PIECEWISE") == "PIECEWISE"
    except Exception:  # noqa: BLE001
        return True


class Side:
    """The serving stream's GEMM state and the side stream's units over a dither_one Window."""

    in_order = False

    def unadopted(self) -> tuple[str, ...]:
        """The arm's parts no ruling has taken in (`forms.UNADOPTED_PARTS`)."""
        return ()

    def adopted(self) -> tuple[str, ...]:
        """The arm's parts a ruling has taken in (`forms.ADOPTED_PARTS`)."""
        return ()

    def label(self) -> dict:
        return F.form_of("dither-a", self.z, self.in_order, self.unadopted(), self.adopted())

    def __init__(self, win: DO.Window, unit: str, h1u: str | None, z: str = "row", ao: str = "", pt: bool = False):
        if unit not in UNITS:
            raise ValueError(f"dither-a's unit: one of {UNITS}, not {unit!r}")
        if ao not in AO or (ao in AO_SEEDS and unit == "h1r") or (ao == "mn" and unit not in KONE):
            raise ValueError(f"dither-a's ao: one of {AO} ({AO_SEEDS} for kh and hbk, whose seeds are a launch of their "
                             f"own), not {ao!r} with unit {unit!r}")
        self.win, self.unit, self.z, self.zl, self.ao, self.pt, dev = win, unit, z, ZMODES[z], ao, pt, win.dev
        self.zdead = z == "zy" and settings.get(ZDEAD_ENV)
        if settings.get(YCMP_ENV) and not (self.zdead and not getattr(win, "overlap", 0)):
            raise RuntimeError(f"{YCMP_ENV}=1 needs -zy and {ZDEAD_ENV}=1, without -ov")
        self.ycmp: torch.Tensor | None = None
        # under YCMP_ENV each capture's calls refused by the byte compare and by the roots' (`y_leaves`)
        self.ycmp_at: dict[str, dict] = {}
        self.side = torch.cuda.Stream(device=dev, priority=settings.get(UNITS_PRIO_ENV))
        self.fork, self.join = torch.cuda.Event(), torch.cuda.Event()
        if pt:
            win.tree_streams = (torch.cuda.Stream(device=dev), torch.cuda.Stream(device=dev))
        if ao:
            # the A tables on a stream of their own above the units' priority: a call's table is a chain of dependent
            # compressions on a few blocks, so it runs beside the previous call's units, which fill the device; under
            # "m" and "ms" on NA_MULTI such streams, since no call's table reads another's
            na = NA_MULTI if ao in ("m", "ms", "mn") else 1
            self.astreams = [torch.cuda.Stream(device=dev, priority=-1) for _ in range(na)]
            self.a_done = [torch.cuda.Event() for _ in range(na)]
            self.a_join = [torch.cuda.Event() for _ in range(na)]
            # the A streams with work since the last join: a capture may wait only on events recorded inside it
            self.a_used: set[int] = set()
            # under "mn" A stream i writes call w's noise into ws_ring[i] (w = i mod NA_MULTI) and the side stream's units
            # read it, so the noise of call w + NA_MULTI waits on the units of call w (u_done[i], when recorded since
            # the last join; a join orders every earlier unit before the next fork)
            self.u_done = [torch.cuda.Event() for _ in range(na)]
            self.u_rec: set[int] = set()
        self.work = EL.Workspace(dev)
        self.arch = EL.arch(dev.index)
        self.mm_cfg = {tuple(int(v) for v in nk.split(",")): (EL.Cfg.parse(c), mm_group(c))
                       for nk, c in (e.split("=") for e in (settings.get(MM_CFG_ENV) or "").split("+") if e)}
        self.stage = torch.empty_like(win.plan.out)
        self.at = torch.tensor([win.plan.offs[w][0] for w in range(win.L)], dtype=torch.int64, device=dev)
        self.shift = int(win.o.h1o_fault_shift())
        self.ks1 = [(C.c_int * 1)(k) for _, k in win.nk]
        self.sms = torch.cuda.get_device_properties(dev).multi_processor_count
        self.versions = [H1R_BLOCK if n // 32 > self.sms else H1R_SPLIT for n, _ in win.nk]
        self.joins = self.calls = 0
        self.cfgs: dict = {}
        self.zy: list[torch.Tensor] = []
        self.ysv: list[torch.Tensor] = []
        if z != "row":
            # the window's row z tables are never built: under "zy" its z/y leaf tables are, between steps in their
            # place
            zset, build = win.pre["z"], win.c.build
            win.c.build = lambda st, step, domains=True: None if st is zset else build(st, step, domains)
            win.z = [torch.empty(0, dtype=torch.int32, device=dev) for _ in win.z]
        if z == "zy":
            self.unit_of = [UNIT] * win.L
            dead = self.dead_rows() if self.zdead else None
            self.zy = [torch.empty(zy_count(win, w, self.unit_of[w]) * zy_bytes(self.unit_of[w]), dtype=torch.uint8,
                                   device=dev) for w in range(win.L)]
            self.zyset = win.c.set([Tab(Dom(ZY_KIND, zy_fields(win, w, 0, self.unit_of[w]), zy_count(win, w, self.unit_of[w]),
                                            {"step": (1, 0)}), zy_count(win, w, self.unit_of[w]), data=self.zy[w],
                                        stride=zy_bytes(self.unit_of[w]), body=zy_bytes(self.unit_of[w]),
                                        length=zy_bytes(self.unit_of[w]), lab=("zy/", f"/{w}"),
                                        out=win.pre["mid"].leaves_addr(2) + 64 * w, **(dead[w]["zy"] if dead else {}))
                                    for w in range(win.L)])
            win.pre["z"] = self.zyset
            # the served y committed under the round (its roots the y-step leaves) and the units' y (the records' y
            # halves, read in place), compared by root between steps
            yi = win.round_kinds.index("y-step")
            yb, ub = [y_bytes(u) for u in self.unit_of], [zy_bytes(u) for u in self.unit_of]
            cnt = [zy_count(win, w, self.unit_of[w]) for w in range(win.L)]
            self.ysv = [torch.empty(cnt[w] * yb[w], dtype=torch.uint8, device=dev) for w in range(win.L)]
            self.ysset = win.c.set([Tab(Dom(Y_KIND, y_fields(win, w, 0, self.unit_of[w]), cnt[w], {"step": (1, 0)}), cnt[w],
                                        data=self.ysv[w], stride=yb[w], body=yb[w], length=yb[w], lab=("y/", f"/{w}"),
                                        out=win.pre["mid"].leaves_addr(yi) + 64 * w, **(dead[w]["y"] if dead else {}))
                                    for w in range(win.L)])
            self.yuset = win.c.set([Tab(Dom(Y_KIND, y_fields(win, w, 0, self.unit_of[w]), cnt[w], {"step": (1, 0)}), cnt[w],
                                        data=self.zy[w][ub[w] - yb[w]:], stride=ub[w], body=yb[w], length=yb[w],
                                        lab=("y/", f"/{w}"), **(dead[w]["y"] if dead else {})) for w in range(win.L)])
            self.ybad = torch.zeros(win.L, dtype=torch.bool, device=dev)
            self.y_refused: list[dict] = []
            # DITHER_ZY1=1: the y tables read nothing the z tables write and are built right after them, so the three
            # sets go through one h1c_build in the z tables' place, each table into its own set's buffers
            self.zy1 = settings.get(ZY1_ENV)
            if self.zy1:
                win.pre["z"] = win.c.union([self.zyset, self.ysset, self.yuset])
            if settings.get(YCMP_ENV):
                self.ycmp = torch.zeros(win.L, dtype=torch.int32, device=dev)
                if self.zy1:
                    # a step no capture reads builds the union without the units' y tables
                    full, lite, zbuild = win.pre["z"], win.c.union([self.zyset, self.ysset]), win.c.build
                    win.c.build = lambda st, step, domains=True: zbuild(lite if st is full and win.cap is None else st,
                                                                        step, domains)
            win.before_mid.append(self.y_leaves)
        if unit in ("kh",) + KONE:
            if not h1u:
                raise RuntimeError(f"dither-a's {unit} unit needs h1u.so (HB_H1U; HB_H1U_KSPLIT for hbk, HB_H1U_KRONLY for hbr)")
            h = self.h = C.CDLL(h1u)
            vp, u32 = C.c_void_p, C.c_uint
            h.kh_unit_ws.argtypes = [vp, vp, u32, u32, u32, vp, vp, vp, vp, u32, vp]
            for f in (h.kh_one, h.kh_one_noise, h.kh_one_units) if unit in KONE else (h.kh_one,):
                f.argtypes = [vp, vp, u32, u32, u32, vp, vp, vp, vp, u32, vp, vp, vp, vp, vp, vp, vp, u32, vp]
            h.kh_ws_bytes.restype = C.c_ulonglong
            h.kh_ws_bytes.argtypes = [u32, u32, u32]
            h.kh_noise_blocks.restype = u32
            h.kh_noise_blocks.argtypes = [u32]
            h.kh_words.restype = u32
            h.kh_words.argtypes = [u32]
            self.seed0, self.nb, need, s0 = [], [], 0, 0
            for w, (n, k) in enumerate(win.nk):
                _, units, nw = win.plan.offs[w]
                if units != (win.t // 32) * (n // 32) or int(h.kh_words(k)) != nw:
                    raise RuntimeError(f"linear {w}: the plan's {units} units of {nw} words, kh_unit_ws' "
                                       f"{(win.t // 32) * (n // 32)} of {int(h.kh_words(k))}")
                if unit in KONE and (win.wt[w].stride(0) != k or win.slot[w].stride(0) != k):
                    raise RuntimeError(f"linear {w}: kh_one takes row stride K")
                self.seed0.append(s0)
                self.nb.append(int(h.kh_noise_blocks(k)))
                need = max(need, int(h.kh_ws_bytes(win.t, n, k)))
                s0 += units
            self.ws = torch.empty((max(need, 64),), dtype=torch.uint8, device=dev)
            if ao == "mn":
                self.ws_ring = [self.ws] + [torch.empty_like(self.ws) for _ in range(NA_MULTI - 1)]
        if unit in KONE:
            # kh_one (the K split's h1k where the call's live units fit) for its receipts only: its z and y go to
            # scratch, and it reads a copy of the step with no flags, since dithera_commit and dithera_fault apply the
            # step's no-write and fault to the receipts as for the other units
            nmax = max(n for n, _ in win.nk)
            self.zs = torch.empty((2, win.t, nmax), dtype=torch.int32, device=dev)
            self.ys = torch.empty((win.t, nmax), dtype=torch.bfloat16, device=dev)
            self.s_unit = torch.zeros_like(win.S)

    def gemm(self, w: int, xq: torch.Tensor, s_a: torch.Tensor, wt: torch.Tensor, s_w: torch.Tensor, y: torch.Tensor):
        win = self.win
        M, k = xq.shape
        n = wt.shape[0]
        cfg, group = None, 1
        if self.mm_cfg:
            mmax = next((m for m, _ in EL.CONFIGS.get(self.arch, {}).get((n, k), ()) if M <= m), None)
            cfg, group = self.mm_cfg.get((n, k, mmax), (None, 1))
        cfg = cfg or EL.pick(M, n, k, self.arch)
        self.cfgs[(n, k, M)] = str(cfg) + (f":g{group}" if group > 1 else "")
        bm, bn, bk, warps, stages, split, tma = cfg
        if self.zy:
            z, yl = self.zy[w].view(torch.int32), self.zy[w].view(torch.int16)
            ur, uc = self.unit_of[w]
        else:
            z, yl, (ur, uc) = win.z[w], y, UNIT
        d = win.ydump[w] if win.ydump is not None else y
        tiles = triton.cdiv(win.t, bm) * triton.cdiv(n, bn)
        if tma:
            from triton.tools.tensor_descriptor import TensorDescriptor
            dithera_mm_tma[(tiles,)](TensorDescriptor.from_tensor(xq, [bm, bk]), TensorDescriptor.from_tensor(wt, [bn, bk]),
                                     s_a, s_w, y, z, yl, d, win.S, M, n, k, win.t, y.stride(0), BLOCK_M=bm, BLOCK_N=bn,
                                     BLOCK_K=bk, WARP_SPECIALIZE=tma == 2, ZL=self.zl, UR=ur, UC=uc, GROUP_M=group,
                                     num_warps=warps, num_stages=stages, **EL.OPTS)
            return
        if split > 1:
            assert M * n <= self.work.ws.numel() and tiles <= self.work.cnt.numel(), (M, n, tiles)
            ws, cnt = self.work.ws, self.work.cnt
        else:
            ws, cnt = y, y
        dithera_mm[(tiles, split)](xq, wt, s_a, s_w, y, z, yl, d, win.S, ws, cnt, M, n, k, win.t, xq.stride(0),
                                   wt.stride(0), y.stride(0), BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, SPLIT_K=split, ZL=self.zl,
                                   UR=ur, UC=uc, GROUP_M=group, num_warps=warps, num_stages=stages, **EL.OPTS)

    def ycommit(self, w: int, y: torch.Tensor, s_w: torch.Tensor) -> None:
        """On the serving stream after call w's GEMM: the served y into the call's y records, dq(0, +0, s_w) past m
        (`dithera_ycommit`)."""
        if not self.ysv:
            return
        n, t = y.shape[1], self.win.t
        ur, uc = self.unit_of[w]
        cmp = self.ycmp is not None
        dithera_ycommit[(triton.cdiv(t, 32), triton.cdiv(n, 128))](y.view(torch.int16), s_w,
                                                                   self.ysv[w].view(torch.int16), self.win.S,
                                                                   self.zy[w].view(torch.int16),
                                                                   self.ycmp[w:] if cmp else self.win.S,
                                                                   y.shape[0], n, t, y.stride(0), UR=ur, UC=uc,
                                                                   BLOCK_M=32, BLOCK_N=128, CMP=cmp, **EL.OPTS)

    def dead_rows(self) -> list[dict]:
        """$DITHER_ZDEAD's `Tab` fields, each call's {"zy": its z/y table's, "y": its y tables'}: a row tile past m (the
        window's `live` 0) holds z 0 and y dq(0, +0, s_w) in every step, so each column tile's x is hashed here once."""
        from verity_vllm.engine import int8_exact as E
        win, out = self.win, []
        for w in range(win.L):
            (ur, uc), n = self.unit_of[w], win.nk[w][0]
            if ur != 32:
                raise RuntimeError(f"${ZDEAD_ENV}: the window's live flags are 32-row blocks, linear {w}'s units {ur} rows")
            pad = E.dequantize(torch.zeros((1, n), dtype=torch.int32, device=win.dev),
                               torch.zeros(1, dtype=torch.float32, device=win.dev), win.sw[w]).view(torch.int16)
            pad = pad.reshape(n // uc, 1, uc).expand(n // uc, ur, uc).cpu().numpy().astype("<i2")
            y, zero = [pad[c].tobytes() for c in range(n // uc)], bytes(4 * ur * uc)
            out.append({"zy": {"xd": H1CM.dead_x([zero + b for b in y], win.dev), "rlive": win.live, "rbw": n // uc},
                        "y": {"xd": H1CM.dead_x(y, win.dev), "rlive": win.live, "rbw": n // uc}})
        return out

    def y_leaves(self) -> None:
        """Between steps (the window's `before_mid`): the served y's tables, whose roots are the round's y-step leaves,
        and the units' y tables (under DITHER_ZY1 built with the z tables); `ybad` the calls whose two roots differ (a
        refused window); under YCMP_ENV, at a step no capture reads, the calls `dithera_ycommit` flagged."""
        win = self.win
        roots = self.ycmp is None or win.cap is not None
        if not self.zy1:
            win.c.build(self.ysset, win.stepv[0:1])
            if roots:
                win.c.build(self.yuset, win.stepv[0:1])
        if roots:
            o = win.pre["mid"].level_off[win.round_kinds.index("y-step")]
            self.ybad = (win.pre["mid"].level[o:o + 8 * win.L].view(win.L, 8) != self.yuset.roots.view(win.L, 8)).any(1)
        if self.ycmp is not None:
            flagged = self.ycmp != 0
            if roots:
                self.ycmp_at[win.cap] = {"bytes": flagged.nonzero().flatten().tolist(),
                                         "roots": self.ybad.nonzero().flatten().tolist()}
            else:
                self.ybad = flagged
            self.ycmp.zero_()

    def seeds(self, w: int, st: int) -> int:
        """kh's, hbk's and hbr's seeds of call w (h1r version 0), from the window's salt and the call's A root."""
        win = self.win
        return win.r.lib.h1r_launch(win.plan.table.data_ptr() + win.call_bytes * w, 1, win.units[w], self.ks1[w],
                                    win.salt.data_ptr(), win.step_base.data_ptr(), win.plan.seeds.data_ptr(),
                                    self.stage.data_ptr(), 0, 0, st)

    def units(self, w: int, M: int, st: int, s_w: torch.Tensor) -> int:
        win = self.win
        n, k = win.nk[w]
        tab = win.plan.table.data_ptr() + win.call_bytes * w
        if self.unit == "h1r":
            return win.r.lib.h1r_launch(tab, 1, win.units[w], self.ks1[w], win.salt.data_ptr(), win.step_base.data_ptr(),
                                        win.plan.seeds.data_ptr(), self.stage.data_ptr(), 0, self.versions[w], st)
        rc = 0 if self.ao in AO_SEEDS else self.seeds(w, st)
        live = -(-min(M, win.t) // 32)
        if self.unit in KONE:
            f = self.h.kh_one_units if self.ao == "mn" else self.h.kh_one
            return rc or f(*self.one_args(w, M, s_w), st)
        return rc or self.h.kh_unit_ws(win.slot[w].data_ptr(), win.wt[w].data_ptr(), win.t, n, k,
                                       win.plan.seeds.data_ptr() + 8 * 16 * self.seed0[w], win.live.data_ptr(),
                                       self.stage.data_ptr() + 4 * win.plan.offs[w][0], self.ws.data_ptr(),
                                       auto_j(live * (n // 32), self.nb[w], self.sms), st)

    def one_args(self, w: int, M: int, s_w: torch.Tensor) -> tuple:
        """kh_one's arguments for call w but its stream (the same to kh_one_noise and kh_one_units)."""
        win = self.win
        n, k = win.nk[w]
        live = -(-min(M, win.t) // 32)
        ws = self.ws_ring[w % NA_MULTI] if self.ao == "mn" else self.ws
        return (win.slot[w].data_ptr(), win.wt[w].data_ptr(), win.t, n, k,
                win.plan.seeds.data_ptr() + 8 * 16 * self.seed0[w], win.live.data_ptr(),
                self.stage.data_ptr() + 4 * win.plan.offs[w][0], ws.data_ptr(),
                auto_j(live * (n // 32), self.nb[w], self.sms), self.zs[0].data_ptr(), self.zs[1].data_ptr(),
                self.ys.data_ptr(), None, win.scale[w].data_ptr(), s_w.data_ptr(), self.s_unit.data_ptr(),
                min(M, win.t))

    def call(self, w: int, x: torch.Tensor, wt: torch.Tensor, s_w: torch.Tensor, y: torch.Tensor) -> None:
        win = self.win
        M, k = x.shape
        main = torch.cuda.current_stream()
        xq = torch.empty((M, k), dtype=torch.int8, device=x.device)
        s_a = torch.empty(M, dtype=torch.float32, device=x.device)
        exactlin_quant_rows_dithera[(M,)](x, xq, s_a, win.slot[w], win.scale[w], win.S, k, x.stride(0), xq.stride(0),
                                          BLOCK_K=triton.next_power_of_2(k), num_warps=EL.quant_warps(k), **EL.OPTS)
        self.fork.record(main)
        st = self.side.cuda_stream
        if self.ao:
            # A_w waits only on its quantizer and A_(w-1), and the units of call w on A_w: the A table still comes
            # before the call's seeds and units, and A_(w+1) overlaps the units of call w
            i = w % len(self.astreams)
            ast = self.astreams[i]
            ast.wait_event(self.fork)
            rc = win.a_call(w, win.slot[w], win.scale[w], ast.cuda_stream)
            if self.ao in AO_SEEDS:
                rc = rc or self.seeds(w, ast.cuda_stream)
            if self.ao == "mn":
                if i in self.u_rec:
                    ast.wait_event(self.u_done[i])
                rc = rc or self.h.kh_one_noise(*self.one_args(w, M, s_w), ast.cuda_stream)
            self.a_done[i].record(ast)
            self.side.wait_event(self.a_done[i])
            self.a_used.add(i)
            self.gemm(w, xq, s_a, wt, s_w, y)
            self.ycommit(w, y, s_w)
            rc = rc or self.units(w, M, st, s_w)
            if self.ao == "mn":
                self.u_done[i].record(self.side)
                self.u_rec.add(i)
        else:
            self.side.wait_event(self.fork)
            self.gemm(w, xq, s_a, wt, s_w, y)
            self.ycommit(w, y, s_w)
            rc = win.a_call(w, win.slot[w], win.scale[w], st) or self.units(w, M, st, s_w)
        if rc:
            raise RuntimeError(f"dither-a call {w} (M={M}): a side launch returned {rc}")
        last = w == win.L - 1
        if last:
            with torch.cuda.stream(self.side):
                n = self.stage.numel()
                dithera_commit[(triton.cdiv(n, 4096),)](self.stage, win.plan.out, win.S, n, BLOCK=4096)
                dithera_fault[(1,)](win.plan.out, win.S, self.at, win.L, 1 << self.shift,
                                    BLOCK=triton.next_power_of_2(win.L))
        if last or (w % 4 == 0 and _piecewise()):
            if self.ao:
                for i in sorted(self.a_used):
                    self.a_join[i].record(self.astreams[i])
                    main.wait_event(self.a_join[i])
                self.a_used.clear()
                self.u_rec.clear()
            self.join.record(self.side)
            main.wait_event(self.join)
            self.joins += 1
        self.calls += 1

    def leaf_bytes(self) -> dict | None:
        """Per step under "zy": the bytes the epilogue writes into the records, and the leaf build reads (the
        records) and writes (segment digests, leaves, levels), from the tables' shapes."""
        if not self.zy:
            return None
        recs = sum(t.count * t.length for t in self.zyset.tabs)
        segs = sum(t.count * ((t.length + 1023) // 1024) for t in self.zyset.tabs)
        leaves = sum(t.count for t in self.zyset.tabs)
        return {"unit": list(UNIT), "record_bytes": zy_bytes(UNIT), "leaves": leaves, "epilogue_writes": recs,
                "build_reads": recs + 64 * segs, "build_writes": 64 * segs + 64 * 2 * leaves}

    def info(self) -> dict:
        return {**self.label(), "unit": self.unit, "z": self.z, "a_overlap": self.ao, "trees_parallel": self.pt,
                "leaf_bytes_per_step": self.leaf_bytes(), "round_leaves": list(self.win.round_kinds),
                "y_root_refused_steps": self.y_refused[:8] if self.zy else None, "z_dead_rows_x_once": self.zdead,
                "y_compared_in_commit": self.ycmp is not None, "a_form": self.win.c.a_form, "calls": self.calls,
                "joins": self.joins,
                "h1r_versions": {DF.SITES[w % 4]: self.versions[w] for w in range(min(4, self.win.L))}
                if self.unit == "h1r" else None, "gemm_cfgs": {f"{n}x{k}@M{m}": c for (n, k, m), c in self.cfgs.items()}}


def zy_bytes(unit: tuple[int, int]) -> int:
    """A unit's record: its UR x UC z int32, then its y bf16."""
    return 6 * unit[0] * unit[1]


def zy_fields_of(window: str, s: int, w: int, module: str, t: int, n: int, unit: tuple[int, int]) -> dict:
    ur, uc = unit
    return {"window": window, "step": s, "linear": w, "module": module, "slots": t, "n": n, "unit": [ur, uc],
            "record": f"z {ur}x{uc} int32 LE row-major, then y = dequant(z) {ur}x{uc} bf16 LE row-major; past m z 0 "
                      "and y dequant(0, +0, s_w)",
            "order": "unit row block, then column block", "schema": CM.SCHEMA}


def zy_fields(win: DO.Window, w: int, s: int, unit: tuple[int, int]) -> dict:
    return zy_fields_of(win.window, s, w, win.names[w], win.t, win.nk[w][0], unit)


def y_bytes(unit: tuple[int, int]) -> int:
    """A unit's y record: its UR x UC bf16."""
    return 2 * unit[0] * unit[1]


def y_fields(win: DO.Window, w: int, s: int, unit: tuple[int, int]) -> dict:
    ur, uc = unit
    return {"window": win.window, "step": s, "linear": w, "module": win.names[w], "slots": win.t, "n": win.nk[w][0],
            "unit": [ur, uc], "record": f"y {ur}x{uc} bf16 LE row-major; past m dequant(0, +0, s_w)",
            "order": "unit row block, then column block",
            "schema": CM.SCHEMA}


def y_rows(rec: torch.Tensor, t: int, n: int, unit: tuple[int, int]) -> torch.Tensor:
    """A call's y records (Y_KIND) as its y (T, N) int16, row-major."""
    ur, uc = unit
    return rec.view(torch.int16).reshape(t // ur, n // uc, ur, uc).permute(0, 2, 1, 3).reshape(t, n)


def zy_count(win: DO.Window, w: int, unit: tuple[int, int]) -> int:
    n = win.nk[w][0]
    if win.t % unit[0] or n % unit[1]:
        raise ValueError(f"linear {w}: T = {win.t}, N = {n}, not whole {unit[0]} x {unit[1]} units")
    return (win.t // unit[0]) * (n // unit[1])


def zy_split(rec: torch.Tensor, t: int, n: int, unit: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    """A call's records as its z (T, N) int32 and its y (T, N) bf16, row-major."""
    ur, uc = unit
    nr, nc = t // ur, n // uc
    v32 = rec.view(torch.int32).reshape(nr * nc, 3 * ur * uc // 2)[:, :ur * uc]
    v16 = rec.view(torch.int16).reshape(nr * nc, 3 * ur * uc)[:, 2 * ur * uc:]
    z = v32.reshape(nr, nc, ur, uc).permute(0, 2, 1, 3).reshape(t, n)
    y = v16.reshape(nr, nc, ur, uc).permute(0, 2, 1, 3).reshape(t, n).view(torch.bfloat16)
    return z, y


def zy_join(z: torch.Tensor, y: torch.Tensor, unit: tuple[int, int]) -> torch.Tensor:
    """`zy_split`'s inverse: z (T, N) int32 and y (T, N) bf16 as the call's records, (units, zy_bytes) uint8."""
    ur, uc = unit
    t, n = z.shape

    def units(x: torch.Tensor) -> torch.Tensor:
        return x.reshape(t // ur, ur, n // uc, uc).permute(0, 2, 1, 3).reshape(-1, ur * uc).contiguous().view(torch.uint8)
    return torch.cat([units(z), units(y.view(torch.int16))], dim=1)


SIDE: list[Side] = []
CONF: dict = {}


def side() -> Side:
    if not SIDE:
        win = DO._window(torch.device("cuda"))
        win.step_salt = CONF["ls"]
        SIDE.append(Side(win, CONF["unit"], CONF["h1u"], CONF["z"], CONF["ao"], CONF["pt"]))
        before = DO.Window.before

        after = DO.Window.after

        def before_a(self: DO.Window, m: int) -> None:
            before(self, m)
            if SIDE[0].unit in KONE:
                self._put(SIDE[0].s_unit, torch.tensor([0, m, 0, 0], dtype=torch.int32))
            if self.cap:
                SIDE[0].stage.view(torch.uint8).fill_(DF.POISON)
                for x in SIDE[0].zy + SIDE[0].ysv:
                    x.fill_(DF.POISON)
                if SIDE[0].zy:
                    for st in (SIDE[0].zyset, SIDE[0].ysset, SIDE[0].yuset):
                        st.level.view(torch.uint8).fill_(DF.POISON)
                torch.cuda.synchronize()

        def after_a(self: DO.Window) -> None:
            cap, sd = self.cap, SIDE[0]
            after(self)
            if not sd.zy:
                return
            bad = sd.ybad.nonzero().flatten().tolist()
            if bad:
                sd.y_refused.append({"s": self.s - 1, "capture": cap, "calls": len(bad)})
            if cap is not None:
                yi = self.round_kinds.index("y-step")
                self.caps[cap].update(zy=[x.clone() for x in sd.zy], ysv=[x.clone() for x in sd.ysv], y_root_bad=bad,
                                      y_roots=[unwords(self.pre["mid"].leaf(yi, w)) for w in range(self.L)],
                                      yu_roots=[unwords(sd.yuset.root(w)) for w in range(self.L)])
        DO.Window.before = before_a
        DO.Window.after = after_a
    return SIDE[0]


_LIB = torch.library.Library(LIB, "DEF")
_LIB.define("mm(Tensor x, Tensor w, Tensor s_w, int idx) -> Tensor")


def _mm_cuda(x: torch.Tensor, w: torch.Tensor, s_w: torch.Tensor, idx: int) -> torch.Tensor:
    y = torch.empty((x.shape[0], w.shape[0]), dtype=torch.bfloat16, device=x.device)
    if x.shape[0]:
        side().call(idx, x if x.stride(1) == 1 else x.contiguous(), w, s_w.reshape(-1), y)
    return y


_LIB.impl("mm", _mm_cuda, "CUDA")


@torch.library.register_fake(f"{LIB}::mm")
def _mm_fake(x, w, s_w, idx):
    return x.new_empty((x.shape[0], w.shape[0]), dtype=torch.bfloat16)


def patch(t: int, h1c: str, h1o: str, unit: str = "h1r", h1u: str | None = None, key: bytes | None = None,
          z: str = "row", ao: str = "", pt: bool = False, ls: bool = False) -> None:
    """dither_one's patch (its weights' registry, the window's libraries), then every int8-exact linear's `apply`
    through ``dithera::mm`` (before the engine is built, in its process)."""
    from verity_vllm.engine import int8_vllm as IV
    if unit not in UNITS or z not in ZMODES or ao not in AO:
        raise ValueError(f"dither-a's unit: one of {UNITS}, not {unit!r}; its z: one of {tuple(ZMODES)}, not {z!r}; "
                         f"its ao: one of {AO}, not {ao!r}")
    CONF.update(unit=unit, h1u=h1u, z=z, ao=ao, pt=pt, ls=ls)
    global CAPTURES
    DO.ROUND_EXTRA = (("y-step",) if z == "zy" else ()) + ("receipt-step",)
    CAPTURES = ("gate", "control", "fault") + (("fault_y",) if z == "zy" else ()) + ("fault_rr",)
    settings.setdefault("DITHER_A_FORM", "3")
    DO.patch(t=t, h1c=h1c, h1o=h1o, key=key, gate=True)

    def apply(self, layer, x, bias=None):
        if bias is not None or x.dtype != torch.bfloat16:
            raise ValueError(f"{self.prefix}: dither-a takes bf16 activations and no bias")
        lead, k = x.shape[:-1], x.shape[-1]
        x2 = x.reshape(-1, k)
        n = layer.weight.shape[0]
        if x2.shape[0] == 0:
            return x.new_empty(*lead, n)
        return torch.ops.dithera.mm(x2, layer.weight, layer.weight_scale, self.dither_idx).reshape(*lead, n)

    IV.Int8PlainLinearMethod.apply = apply


def stats(win: DO.Window):
    def doc() -> dict:
        return {**win.stats(), **arm_label(), "dither_a": SIDE[0].info() if SIDE else None}
    return doc


# -- the gate ----------------------------------------------------------------------------------------------------------------

# N > 0: `check` hashes a table's leaves in a spawned pool of N processes once it has POOL_MIN rows or more (a prefill
# capture's 2.75M z/y units take ~4 min on one core); unset or 0, in this process
CHECK_WORKERS_ENV = "DITHER_CHECK_WORKERS"
POOL_MIN = 4096
_POOL: list = []


def table_root(dom, key: bytes, label: str, rows) -> bytes:
    """`CM.levels`' root over each row's `CM.leaf` (rows: byte strings or a 2-D uint8 array), the leaves hashed in
    CHECK_WORKERS_ENV's pool when there are POOL_MIN or more."""
    n = settings.get(CHECK_WORKERS_ENV)
    if n <= 0 or len(rows) < POOL_MIN:
        return CM.levels(dom, [CM.leaf(dom, key, label, i, bytes(r))[0] for i, r in enumerate(rows)])[-1][0]
    if not _POOL:
        import multiprocessing as mp
        # sbench's arm workers are daemons, which multiprocessing won't let start children; the pool's own workers
        # are daemons, ended with this process
        cfg = mp.current_process()._config
        daemon = cfg.get("daemon")
        cfg["daemon"] = False
        try:
            _POOL.append(mp.get_context("spawn").Pool(n))
        finally:
            cfg["daemon"] = daemon
    step = -(-len(rows) // (4 * n))
    got = _POOL[0].map(CM.leaves, [(dom.binding, dom.count, key, label, at, rows[at:at + step])
                                   for at in range(0, len(rows), step)], chunksize=1)
    return CM.levels(dom, [x for part in got for x in CM.split(part)])[-1][0]


def check(win: DO.Window, lib_h1o: str, which: str = "gate", n_cpu_units: int = 3, g: dict | None = None,
          r_round: dict | None = None) -> dict:
    """`which`'s capture (`g`, else the window's) against references other than the served kernels (the module
    docstring's list).  `r_round` ({"s", "leaves", "root"}: dither_so's round s + D - 1) is the round that carries the
    capture's receipts when that is not the capture's own: its receipt-step leaf and its root are checked there
    (round's "receipt_step" and "receipt_root"), and the capture's own round's receipt-step leaf, an earlier step's
    receipts, is read as recorded; "fault_rr" then wants that round refused."""
    from verity_vllm.engine import int8_exact as E
    g = win.caps.get(which, {}) if g is None else g
    if "receipts" not in g or "a" not in g:
        return {"ok": False, "complete": False, "why": f"{which} step incomplete: {sorted(g)}"}
    m, s, t, L = g["m"], g["s"], win.t, win.L
    unit = SIDE[0].unit if SIDE else None
    doc: dict = {"which": which, **arm_label(), "complete": True, "s": s, "m": m, "t": t,
                 "poison": f"{DF.POISON:#04x}", "epsilon": "seeded", "unit": unit}
    key, W = win.key, win.window
    zy = bool(SIDE) and bool(SIDE[0].zy)
    bits = np.arange(t) < m
    wrows = CR.word_rows(CR.words(bits))
    a_cpu = DO.a_roots_cpu(win, s, range(L), lambda v: g["a"][v].cpu().numpy(), lambda v: g["sa"][v].cpu().numpy())
    s_cpu, z_cpu = {}, {}
    for w in range(L):
        n, k = win.nk[w]
        fs = {"window": W, "step": s, "linear": w, "module": win.names[w], "slots": t, "schema": CM.SCHEMA}
        sd = CM.domain(fs, len(wrows), "admit")
        s_cpu[w] = CM.levels(sd, [CM.leaf(sd, key, f"s/{s}/{w}", i, x)[0] for i, x in enumerate(wrows)])[-1][0]
        if zy:
            u = SIDE[0].unit_of[w]
            zd = CM.domain(zy_fields(win, w, s, u), zy_count(win, w, u), ZY_KIND)
            zb, lab = g["zy"][w].cpu().numpy().reshape(zy_count(win, w, u), zy_bytes(u)), f"zy/{s}/{w}"
        else:
            fz = {"window": W, "step": s, "linear": w, "module": win.names[w], "slots": t, "n": n, "schema": CM.SCHEMA}
            zd = CM.domain(fz, t, "z-table")
            zb, lab = g["z"][w].cpu().numpy().astype("<i4").view(np.uint8).reshape(t, -1), f"z/{s}/{w}"
        z_cpu[w] = table_root(zd, key, lab, zb)
    doc["a_roots_bad"] = [w for w in range(L) if a_cpu[w] != g["a_roots"][w]]
    doc["a_bad_diag"] = DO.a_bad_diag(g, doc["a_roots_bad"])
    doc["admit_roots_bad"] = [w for w in range(L) if s_cpu[w] != g["admit_roots"][w]]
    doc["z_roots_bad"] = [w for w in range(L) if z_cpu[w] != g["z_roots"][w]]
    f_step = {"window": W, "step": s, "linears": L}
    step_cpu = {"a-step": CM.step_root(CM.domain(f_step, L, "a-step"), [a_cpu[w] for w in range(L)]),
                "admit-step": CM.step_root(CM.domain(f_step, L, "admit-step"), [s_cpu[w] for w in range(L)]),
                "z-step": CM.step_root(CM.domain(f_step, L, "z-step"), [z_cpu[w] for w in range(L)])}
    kinds = win.round_kinds
    # an empty slot's record y, past m: dq(0, +0, s_w) (h1zy's; +0 under a finite positive s_w)
    pads = [E.dequantize(torch.zeros((1, win.nk[w][0]), dtype=torch.int32, device=win.dev),
                         torch.zeros(1, dtype=torch.float32, device=win.dev), win.sw[w]).view(torch.int16)
            for w in range(L)] if zy else []
    if zy:
        # the served y's records (what ycommit read from the served y) hashed here, their step root the
        # round's y-step leaf; the run-time check (device roots of the served y against the units' y) against the
        # CPU's byte compare of the two, which under one domain and label is the roots' compare
        ys_cpu, cpu_bad, dump_bad, fy_bad = {}, [], [], []
        for w in range(L):
            u, n = SIDE[0].unit_of[w], win.nk[w][0]
            cnt = zy_count(win, w, u)
            yd_ = CM.domain(y_fields(win, w, s, u), cnt, Y_KIND)
            yrec = g["ysv"][w].cpu().numpy().reshape(cnt, y_bytes(u))
            urec = g["zy"][w].cpu().numpy().reshape(cnt, zy_bytes(u))[:, zy_bytes(u) - y_bytes(u):]
            ys_cpu[w] = table_root(yd_, key, f"y/{s}/{w}", yrec)
            if not np.array_equal(yrec, urec):
                cpu_bad.append(w)
            if g["y"] is not None:
                yr = y_rows(g["ysv"][w].to(win.dev), t, n, u)
                x = yr[:m] ^ g["y"][w].to(win.dev).view(torch.int16)
                past = bool((yr[m:] != pads[w]).any())
                if bool(x.any()) or past:
                    dump_bad.append(w)
                    if which == "fault_y" and not (x.nonzero().tolist() == [[0, 0]] and int(x[0, 0]) == 1 and not past):
                        fy_bad.append(w)
        step_cpu["y-step"] = CM.step_root(CM.domain(f_step, L, "y-step"), [ys_cpu[w] for w in range(L)])
        doc["y_served_roots_bad"] = [w for w in range(L) if ys_cpu[w] != g["y_roots"][w]]
        doc["y_root_check"] = {"cpu_bad": len(cpu_bad), "device_bad": len(g["y_root_bad"]),
                               "agree": cpu_bad == g["y_root_bad"]}
        doc["y_served_vs_dump_bad"] = dump_bad if g["y"] is not None else None
        if which == "fault_y":
            doc["fault_y_bad"] = fy_bad[:4]
    rleaf = None
    if "receipt-step" in kinds:
        # the receipt roots under the round; three calls' roots rehashed from the captured receipts
        r_dev = [unwords(g["r_roots"][8 * w:8 * w + 8]) for w in range(L)]
        rs = s if r_round is None else r_round["s"]
        rleaf = CM.step_root(CM.domain({"window": W, "step": rs, "linears": L}, L, "receipt-step"), r_dev)
        step_cpu["receipt-step"] = rleaf if r_round is None else g["leaves"][kinds.index("receipt-step")]
        rr_bad = []
        for w in sorted({0, 2, L - 1}):
            o_, units, nw = win.plan.offs[w]
            fr = {"window": W, "step": s, "linear": w, "module": win.names[w], "base": s * win.u_step + win.offs[w],
                  "units": units, "words": nw, "schema": CM.SCHEMA}
            rd = CM.domain(fr, units, "receipts")
            rrows = CM.receipt_rows(g["receipts"][o_:o_ + units * nw].view(units, nw).cpu().numpy().view(np.uint32))
            if table_root(rd, key, f"r/{s}/{w}", rrows) != r_dev[w]:
                rr_bad.append(w)
        doc["receipt_roots_bad"] = rr_bad
    rnd = CM.step_root(CM.domain({"window": W, "step": s, "leaves": DO.round_leaves(kinds)}, len(kinds), "round"),
                       [step_cpu[k] for k in kinds])
    doc["round"] = {**{k.replace("-", "_"): step_cpu[k] == g["leaves"][i] for i, k in enumerate(kinds)},
                    "root": rnd == g["root"]}
    doc["step_roots"] = {k: v.hex() for k, v in step_cpu.items()}
    if r_round is not None and rleaf is not None:
        ri, lv = kinds.index("receipt-step"), list(r_round["leaves"])
        doc["round"]["receipt_step"] = rleaf == lv[ri]
        lv[ri] = rleaf
        doc["round"]["receipt_root"] = CM.step_root(CM.domain(
            {"window": W, "step": r_round["s"], "leaves": DO.round_leaves(kinds)}, len(kinds), "round"), lv) == r_round["root"]
        doc["receipt_round"] = r_round["s"]
    # z = A B^T by int_mm on the committed A, zero past m; the served y = dequant(z); under "zy" the leaf's y = dequant
    # of the leaf's z = the served y, and dq(0, +0, s_w) past m
    z_bad, y_bad, yleaf_bad = [], [], []
    zr, yl = [], []
    for w in range(L):
        if zy:
            a, b = zy_split(g["zy"][w], t, win.nk[w][0], SIDE[0].unit_of[w])
            zr.append(a)
            yl.append(b)
        else:
            zr.append(g["z"][w])
    dev = win.dev
    for w in range(L):
        zt = E.int_mm(g["a"][w][:m].to(dev), win.wt[w])
        zw = zr[w].to(dev)
        if not (torch.equal(zw[:m], zt) and not bool(zw[m:].any())):
            z_bad.append(w)
        yd = E.dequantize(zw[:m], g["sa"][w][:m].to(dev), win.sw[w]).view(torch.int16)
        if g["y"] is not None and not torch.equal(yd, g["y"][w].to(dev).view(torch.int16)):
            y_bad.append(w)
        yw = yl[w].to(dev) if zy else None
        if zy and not (torch.equal(yw[:m].view(torch.int16), yd) and not bool((yw[m:].view(torch.int16) != pads[w]).any())):
            yleaf_bad.append(w)
    doc["z_vs_int_mm_bad"], doc["y_vs_dequant_z_bad"] = z_bad, y_bad if g["y"] is not None else None
    doc["y_leaf_vs_dequant_z_bad"] = yleaf_bad if zy else None
    # every receipt against h1r v1 on the saved slots, the step's salt and the CPU's roots
    r = H1R(lib_h1o)
    live = torch.from_numpy((np.arange(t // 32) * 32 < m).astype(np.uint8)).to(win.dev)
    roots = [torch.from_numpy(np.concatenate([np.frombuffer(a_cpu[w], ">u8"), np.frombuffer(g["b_roots"][w], ">u8")])
                              .astype("<u8").view("<i8").copy()).to(win.dev) for w in range(L)]
    plan = r.plan([Call(g["a"][w].to(dev), win.wt[w], s * win.u_step + win.offs[w], roots[w], live) for w in range(L)])
    salt = DO.cap_salt(win, g)
    r.launch(plan, salt[1], version=1)
    torch.cuda.synchronize()
    got = g["receipts"]
    shift = SIDE[0].shift if SIDE else 24

    def got_of(w: int) -> torch.Tensor:
        o_, units, nw = win.plan.offs[w]
        return got[o_:o_ + units * nw].view(units, nw).to(dev)

    rc_bad, fault_bad = [], []
    for w in range(L):
        ref, mine = plan.receipts(w), got_of(w)
        if which == "fault":
            x = (ref ^ mine)
            cells = x.nonzero().tolist()
            if cells != [[0, 0]] or (int(x[0, 0]) & M32) != (1 << shift):
                fault_bad.append({"call": w, "words_off": cells[:4], "xor00": int(x[0, 0]) & M32})
        if (ref != mine).any():
            rc_bad.append(w)
    doc["receipts_vs_h1r_bad"] = rc_bad
    doc["live_units"] = int(sum((min(m, t) + 31) // 32 * (n // 32) for n, _ in win.nk))
    y_ok = not zy or (not doc["y_served_roots_bad"] and doc["y_root_check"]["agree"])
    y_match = not zy or (not doc["y_root_check"]["cpu_bad"] and not doc["y_served_vs_dump_bad"])
    commits_ok = (not doc["a_roots_bad"] and not doc["admit_roots_bad"] and not doc["z_roots_bad"]
                  and all(doc["round"].values()) and not doc.get("receipt_roots_bad") and y_ok)
    if zy:
        doc["y_refused_steps"] = [x for x in SIDE[0].y_refused if x["capture"] not in ("fault_y", "control")][:4]
    rest_ok = not rc_bad and not z_bad and not y_bad and not yleaf_bad
    if which == "fault_y":
        # refused at run time on every call (the served y's root is not the units'), every commitment else kept
        every = list(range(L))
        doc["conforming"] = (not fy_bad and dump_bad == every and cpu_bad == every and commits_ok and rest_ok)
        doc["ok"] = y_match and doc["conforming"]
        return doc
    if which == "fault_rr":
        # the round recorded without the receipt roots: refused on its receipt-step leaf and root alone
        rr = "root" if r_round is None else "receipt_root"
        others = all(v for k, v in doc["round"].items() if k not in ("receipt_step", rr))
        doc["conforming"] = (not doc["round"]["receipt_step"] and not doc["round"][rr] and others and rest_ok
                             and y_match and y_ok and not doc["a_roots_bad"] and not doc["admit_roots_bad"]
                             and not doc["z_roots_bad"] and not doc.get("receipt_roots_bad"))
        doc["ok"] = commits_ok and doc["conforming"]
        return doc
    if which == "fault":
        doc["fault_shift"], doc["fault_bad"] = shift, fault_bad
        doc["conforming"] = not fault_bad and not z_bad and not y_bad and not yleaf_bad and commits_ok
        doc["ok"] = not rc_bad and doc["conforming"]
        return doc
    # three units' receipts by H1V3's Definition on the CPU and by H1V3's own code; its z (peeled: A B^T) against ours
    units_doc, h1v3_in = [], []
    if which == "gate":
        rng = np.random.default_rng(s)
        for w in [0, 3, 2][:n_cpu_units]:
            n, k = win.nk[w]
            u = int(rng.integers(n // 32))
            a32 = g["a"][w][0:32].cpu().numpy()
            b32 = win.wt[w][32 * u:32 * u + 32].cpu().numpy()
            pos = s * win.u_step + win.offs[w] + u
            ref = DO.h1v3_unit(a32, b32, salt[0], pos, (a_cpu[w], g["b_roots"][w]))
            zk = zr[w][0:32, 32 * u:32 * u + 32].cpu().numpy().astype(np.int64)
            rk = (got_of(w)[u].cpu().numpy().astype(np.int64) & M32).tolist()
            units_doc.append({"call": w, "unit": u, "k": k, "receipt_equal": rk == ref["receipt"],
                              "z_equal_today": bool((zk == ref["today"]).all()), "_zk": zk, "_rk": rk})
            h1v3_in.append({"a": (a32.astype(np.int64) & 0xFF).tolist(), "b": (b32.astype(np.int64) & 0xFF).tolist(),
                            "salt": salt[0].hex(), "position": int(pos), "roots": [a_cpu[w].hex(), g["b_roots"][w].hex()]})
        hv = DO.h1v3_itself(h1v3_in)
        if "units" in hv:
            for x, hu in zip(units_doc, hv["units"]):
                x["h1v3_receipt_equal"] = x["_rk"] == hu["receipt"]
                if hv.get("peeled"):
                    x["h1v3_z_equal"] = bool((x["_zk"] == np.array(hu["z"], dtype=np.int64)).all())
        for x in units_doc:
            del x["_zk"], x["_rk"]
        doc["h1v3"] = {k_: v for k_, v in hv.items() if k_ != "units"}
    doc["vs_h1v3_cpu"] = units_doc
    h1v3_ok = which != "gate" or ("units" in hv and bool(hv.get("peeled")) and all(
        x["h1v3_receipt_equal"] and x["h1v3_z_equal"] for x in units_doc))
    doc["ok"] = (not rc_bad and not z_bad and (g["y"] is None or not y_bad) and not yleaf_bad and commits_ok and h1v3_ok
                 and y_match and not (zy and doc["y_refused_steps"])
                 and all(x["receipt_equal"] and x["z_equal_today"] for x in units_doc))
    lr = settings.get(LANE_REFS_ENV)
    if which == "gate" and lr:
        import lane_refs as LR
        doc["lane_refs"] = LR.served(win, g, salt[0], getattr(SIDE[0], "h", None) if SIDE else None,
                                     path=None if lr == "1" else lr)
        doc["ok"] = doc["ok"] and doc["lane_refs"]["ok"]
    return doc


def gate_verdict(gate: dict, control: dict, fault: dict, *more: dict) -> dict:
    """The poisoned gate passes; its no-write control (complete) is refused on z and on the receipts; the fault step is
    refused on the receipts at exactly the flipped words, with z, y and every commitment but the receipts' kept; then
    CAPTURES' "fault_y" (under -zy: the served y tampered, refused at run time on every call's y root, every other
    commitment kept) and "fault_rr" (the round without the receipt roots, refused on that leaf alone)."""
    refused = bool(control.get("complete") and not control.get("ok") and control.get("z_vs_int_mm_bad")
                   and control.get("receipts_vs_h1r_bad"))
    fault_refused = bool(fault.get("complete") and not fault.get("ok") and fault.get("conforming"))
    extra = {f"{x.get('which')}_refused": bool(x.get("complete") and not x.get("ok") and x.get("conforming"))
             for x in more}
    if SIDE and SIDE[0].ycmp is not None:
        # YCMP_ENV: at every capture the byte compare refused the calls the roots' did, at fault_y every call
        at = SIDE[0].ycmp_at
        extra["y_compare_agrees"] = bool(at.get("fault_y", {}).get("bytes") == list(range(SIDE[0].win.L))
                                         and all(x["bytes"] == x["roots"] for x in at.values()))
        y_cmp = {"y_compare_at_captures": {k: {j: len(v) for j, v in x.items()} for k, x in at.items()}}
    else:
        y_cmp = {}
    return {"ok": bool(gate.get("ok")) and refused and fault_refused and all(extra.values()),
            **arm_label(), **y_cmp,
            "gate_ok": bool(gate.get("ok")), "control_refused": refused, "fault_refused": fault_refused, **extra,
            "control_bad": {k: len(control.get(k) or []) for k in ("z_vs_int_mm_bad", "y_vs_dequant_z_bad",
                                                                    "receipts_vs_h1r_bad", "a_roots_bad", "z_roots_bad")},
            "fault_receipt_calls_off": len(fault.get("receipts_vs_h1r_bad") or [])}
