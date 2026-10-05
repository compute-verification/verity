//! The gate sets the checker knows, with the semantics pinned in
//! `veritor/core/gates.py` and `veritor/core/int8.py`, and the identity
//! digest that names them.
//!
//! A gate's semantics are part of its definition: the checker refuses a
//! statement whose gate-set digest is not the digest of the table it is
//! about to apply.  The digest is `tagged_sha256("veritor/gate-set/v1",
//! canonical_json(manifest))`, where the manifest lists the gates sorted by
//! name with their `arg_widths` (only when they differ from the output width
//! repeated), `arity`, `name`, `proof_cost`, `replay_cost`, `source` and
//! `width`, then the set's `name` and `version`.
//!
//! Widths.  The toy families (`veritor.toy-isa`, `veritor.word-arithmetic`)
//! are parameterised by one word width and resolve at any width up to 64.
//! `veritor.int8-q7` mixes 8- and 32-bit gates: a statement over it carries
//! its widest gate (32, `veritor.protocol.proofs.statement_width`) and
//! resolves only there, while every opened value is decoded by its own leaf
//! schema (`u8`, `u32`; see [`decode_schema`]) and every gate argument is
//! range-checked against the gate's own argument width, exactly as
//! `Gate.evaluate` does in Python.
//!
//! Tensor-core families (`crate::tc`).  `veritor.tensor-core.ampere_bf16_m16n8k16@2`
//! is L1's pinned family (`lean/VeritorSpec/TC.lean`, `veritor.core.silicon`):
//! 32-bit sources and the two `mma.sync` step gates `tc_dot16` (an FP32
//! accumulator word and 16 + 16 BF16 operand words, in k order, a and b
//! interleaved as `[acc, a0..a15, b0..b15]`) and `tc_dot16_0` (the same with a
//! zero accumulator).  `veritor.tc-ampere-bf16@1` is the serving-shaped
//! family of the `typed-obligation/v0` interface: the same two step gates,
//! 16-bit BF16 sources, and the casts `f32_to_bf16` (RNE, canonical NaN
//! `0x7FC0`) and `bf16_to_f32`.  `veritor.tc-ampere-bf16@2` (lane L8) is the
//! same family with the hardware NaN canonicalisation of `f32_to_bf16`
//! (`0x7FFF`, the measured `cvt.rn.bf16.f32` word; [`Op::F32ToBf16Hw`]); it is
//! a distinct gate set with its own digest, and `@1` is unchanged.  All resolve
//! only at statement width 32 and the step gates reject non-finite words, as
//! the Python and the Lean do.

use crate::elem;
use crate::embed;
use crate::frame::tagged_sha256;
use crate::ftz;
use crate::pad;
use crate::reduce;
use crate::tc;

pub const GATE_SET_IDENTITY_TAG: &[u8] = b"veritor/gate-set/v1";

/// One scalar gate: how the checker evaluates it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    Add,
    Sub,
    Mul,
    Lt,
    Eq,
    Shr,
    /// `veritor.int8-q7`: the exact signed product of two int8 words as an int32 word.
    Mul8,
    /// `veritor.int8-q7`: addition modulo `2**32` (exact integer addition on the declared domain).
    Add32,
    /// `veritor.int8-q7`: `clip(floor(s / 128), 0, 127)` of an int32 word.
    RequantU7,
    /// `veritor.int8-q7`: `clip(floor(t / 128), -128, 127)` of an int32 word, as an int8 word.
    RequantS8,
    /// Tensor core: one `mma.sync.m16n8k16` BF16 step, `[acc, a0..a15, b0..b15]` -> FP32 word.
    TcDot16,
    /// Tensor core: the same step with a zero accumulator, `[a0..a15, b0..b15]` -> FP32 word.
    TcDot16_0,
    /// `veritor.tc-ampere-bf16`: FP32 word -> BF16 word, round to nearest even, NaN -> `0x7FC0`.
    F32ToBf16,
    /// `veritor.tc-ampere-bf16`: BF16 word -> FP32 word, exact.
    Bf16ToF32,
    /// A source gate: no relation, its value is the public input of its rank.
    In,
    /// A source gate: no relation, its value is a leaf of `kappa_W`.
    Weight,
    /// `veritor.tc-ampere-bf16@2`: FP32 word -> BF16 word, round to nearest even,
    /// NaN -> `0x7FFF` (the measured `cvt.rn.bf16.f32` word).  Declared after
    /// every earlier gate so that the existing entries keep their numbers.
    F32ToBf16Hw,
    /// `veritor.pad-serve@1` (lane PAD-IMPL): `c ? a : b` on BF16 words, `c` a bit.
    /// Declared (with the five below) after every earlier gate: existing entries
    /// keep their numbers, exactly as `F32ToBf16Hw` was added.
    SelectBf16,
    /// `veritor.pad-serve@1`: `c ? a : b` on FP32 words, `c` a bit.
    SelectF32,
    /// `veritor.pad-serve@1`: `c ? a : b` on I32 words, `c` a bit.
    SelectI32,
    /// `veritor.pad-serve@1`: `1` iff two 32-bit words are equal (the stop test).
    EqI32,
    /// `veritor.pad-serve@1`: conjunction of two `Value<1>` words (absorbing chain).
    BitAnd,
    /// `veritor.pad-serve@1`: negation of one `Value<1>` word.
    BitNot,
    /// `veritor.reduce-mufu@1` (`crate::reduce`, lane PROOF-REDUCE): the FP32
    /// scalar gates and MUFU table lookups of the vLLM reduction kernels.
    /// Declared after every earlier gate; nothing above changes.
    /// FADD: IEEE RN-even, SSE NaN rule (`prims.F32Add`).
    RF32Add,
    /// FMUL: IEEE RN-even, SSE NaN rule (`prims.F32Mul`).
    RF32Mul,
    /// `div.rn.f32`: IEEE RN-even, SSE NaN rule (`prims.F32Div`).
    RF32Div,
    /// FFMA: one rounding of the exact `a*b + c`, NaN -> `0x7FC00000` (`prims.F32Fma`).
    RF32Fma,
    /// `F2FP.BF16.F32`: RNE, NaN -> `0x7FFF` (`prims.F2fpBf16`).
    RF2fpBf16,
    /// Fused-norm residual add `bf16(f32(a) + f32(b))`, NaN -> `0x7FFF` (`prims.Bf16AddF2fp`).
    RBf16AddF2fp,
    /// NeoX rotation, first half: `bf16_rn(RN32(x*c - y*s))` (`prims.RopeOut`).
    RRopeOut,
    /// NeoX rotation, second half: `bf16_rn(RN32(y*c + x*s))` (`prims.RopeOutAdd`).
    RRopeOutAdd,
    /// `a > b` on BF16 values, NaN compares false (`prims.Bf16GtStrict`).
    RBf16Gt,
    /// `c ? a : b` on BF16 words (`prims.SelectBf16`).
    RSelectBf16,
    /// `c ? a : b` on i32 words (`prims.SelectI32`).
    RSelectI32,
    /// Wrapping 32-bit addition (`prims.I32Add`).
    RI32Add,
    /// `a == b` on 32-bit words (`prims.I32Eq`).
    RI32Eq,
    /// Signed `a <= b` (`prims.I32Le`).
    RI32Le,
    /// MUFU.SQRT (`sqrt.approx.ftz.f32`) with the table row as the second,
    /// 64-bit argument `(index << 32) | word` (`prims.MufuSqrtFtz`).
    RMufuSqrtTab,
    /// `div.full.f32` reciprocal: range-scale the denominator, MUFU.RCP with
    /// the opened table row (`prims.DivFullRcp`).
    RDivFullRcp,
    /// `div.full.f32` numerator scaling, decided by `|b|` (`prims.DivFullScaleA`).
    RDivFullScaleA,
    /// `rsqrt.approx.f32`: MUFU.RSQ with subnormal range scaling and the
    /// opened table row (`prims.RsqrtApprox`).
    RRsqrtApprox,
    /// `veritor.elem-bf16@1`: `Bf16AddF2fp_v1` — the CUDA `fused_add_rms_norm`
    /// residual add, `bf16_rn(f32(a) + f32(b))`, NaN -> `0x7FFF` (lane
    /// PROOF-ELEM; declared after every earlier gate, numbers unchanged).
    Bf16AddF2fp,
    /// `veritor.elem-bf16@1`: `SiluMulBf16_v1` — vLLM `act_and_mul_kernel<silu>`,
    /// `bf16(f32(bf16(silu_f32(f32(g)))) * f32(u))`, NaN -> `0x7FC0`, `silu_f32`
    /// by the embedded 2^16-entry table (`crate::elem`).
    SiluMulBf16,
    /// `veritor.pad-inactive@1`: `BoolAnd_v1` (serve3-spec §9) — `(a & b) & 1`
    /// on 1-bit encodings (the flag chain `live_{s+1} = live_s ∧ guard`).
    /// (`SelectBf16`/`SelectI32` are shared with `veritor.pad-serve@1` above:
    /// same registered semantics, one variant.)
    BoolAnd,
    /// `veritor.embed-gather@1` (`crate::embed`, lane PROOF-COV): the
    /// committed-table row gather of the embedding relation.  Declared after
    /// every earlier gate; nothing above changes.
    /// Wrapping 32-bit multiply (the row base `tok * H`).
    EI32Mul,
    /// Wrapping 32-bit add (the flat-index chain).
    EI32Add,
    /// The gather: `(flat, (index << 16) | word)` -> the bf16 word; rejects a
    /// row whose embedded index is not the recomputed flat index.
    EGatherWord,
    /// `veritor.ftz-f32@1` (`crate::ftz`, E4 / E7 EOD 2026-09-11): the FTZ scalar
    /// relations of the fast-math FA2 attention kernel.  Declared after every
    /// earlier gate; nothing above changes.
    /// `ftz(RN(ftz(a) + ftz(b)))` (`prims.F32AddFtz`).
    FtzF32Add,
    /// `ftz(RN(ftz(a) * ftz(b) + ftz(c)))`, one rounding (`prims.F32FmaFtz`).
    FtzF32Fma,
    /// `ftz(RN(ftz(a) * ftz(b) - ftz(c)))`, one rounding (`prims.F32FmaSubFtz`).
    FtzF32FmaSub,
    /// `ftz(a) > ftz(b) ? ftz(a) : ftz(b)` (`prims.F32Max`).
    FtzF32Max,
    /// `ftz(RN(ftz(a) * ftz(b)))` (`prims.F32MulFtz`).
    FtzF32Mul,
    /// `ftz(RN(ftz(a) - ftz(b)))` (`prims.F32SubFtz`).
    FtzF32Sub,
    /// `(sum == ±0 || NaN) ? 1.0 : rcp.approx.ftz(sum)` with the opened table row
    /// `(index << 32) | word` as the second, 64-bit argument (`prims.Fa2InvSum`).
    FtzFa2InvSum,
    /// `m == -inf ? +0 : m`, no flush (`prims.GuardNegInfZero`).
    FtzGuardNegInfZero,
    /// `ex2.approx.ftz.f32` with the opened table row as the second, 64-bit
    /// argument (`prims.MufuEx2Ftz`).
    FtzMufuEx2,
}

/// The number of operand words each side of one `mma.sync.m16n8k16` step.
pub const TC_K: usize = 16;

impl Op {
    pub fn is_source(self) -> bool {
        matches!(self, Op::In | Op::Weight)
    }

    pub fn arity(self) -> usize {
        match self {
            Op::In | Op::Weight => 0,
            Op::RequantU7 | Op::RequantS8 | Op::F32ToBf16 | Op::F32ToBf16Hw | Op::Bf16ToF32 | Op::BitNot | Op::RF2fpBf16 => 1,
            Op::EI32Mul | Op::EI32Add | Op::EGatherWord => 2,
            Op::TcDot16 => 1 + 2 * TC_K,
            Op::TcDot16_0 => 2 * TC_K,
            Op::SelectBf16 | Op::SelectF32 | Op::SelectI32 | Op::RF32Fma | Op::RSelectBf16 | Op::RSelectI32 => 3,
            Op::RRopeOut | Op::RRopeOutAdd => 4,
            Op::FtzGuardNegInfZero => 1,
            Op::FtzF32Fma | Op::FtzF32FmaSub => 3,
            _ => 2,
        }
    }

    fn manifest_name(self) -> &'static str {
        match self {
            Op::Add => "add",
            Op::Sub => "sub",
            Op::Mul => "mul",
            Op::Lt => "lt",
            Op::Eq => "eq",
            Op::Shr => "shr",
            Op::Mul8 => "mul8",
            Op::Add32 => "add32",
            Op::RequantU7 => "requant_u7",
            Op::RequantS8 => "requant_s8",
            Op::TcDot16 => "tc_dot16",
            Op::TcDot16_0 => "tc_dot16_0",
            // the same gate name in both versions of the family; the set decides the semantics
            Op::F32ToBf16 | Op::F32ToBf16Hw => "f32_to_bf16",
            Op::Bf16ToF32 => "bf16_to_f32",
            Op::Bf16AddF2fp => "bf16_add_f2fp",
            Op::SiluMulBf16 => "silu_mul_bf16",
            Op::BoolAnd => "bool_and",
            Op::In => "in",
            Op::Weight => "weight",
            Op::SelectBf16 => "select_bf16",
            Op::SelectF32 => "select_f32",
            Op::SelectI32 => "select_i32",
            Op::EqI32 => "eq_i32",
            Op::BitAnd => "bit_and",
            Op::BitNot => "bit_not",
            Op::RF32Add => "f32_add",
            Op::RF32Mul => "f32_mul",
            Op::RF32Div => "f32_div",
            Op::RF32Fma => "f32_fma",
            Op::RF2fpBf16 => "f2fp_bf16",
            Op::RBf16AddF2fp => "bf16_add_f2fp",
            Op::RRopeOut => "rope_out",
            Op::RRopeOutAdd => "rope_out_add",
            Op::RBf16Gt => "bf16_gt",
            Op::RSelectBf16 => "select_bf16",
            Op::RSelectI32 => "select_i32",
            Op::RI32Add => "i32_add",
            Op::RI32Eq => "i32_eq",
            Op::RI32Le => "i32_le",
            Op::RMufuSqrtTab => "mufu_sqrt_tab",
            Op::RDivFullRcp => "div_full_rcp",
            Op::RDivFullScaleA => "div_full_scale_a",
            Op::RRsqrtApprox => "rsqrt_approx",
            Op::EI32Mul => "i32_mul",
            Op::EI32Add => "i32_add",
            Op::EGatherWord => "gather_word",
            Op::FtzF32Add => "f32_add_ftz",
            Op::FtzF32Fma => "f32_fma_ftz",
            Op::FtzF32FmaSub => "f32_fma_sub_ftz",
            Op::FtzF32Max => "f32_max_ftz",
            Op::FtzF32Mul => "f32_mul_ftz",
            Op::FtzF32Sub => "f32_sub_ftz",
            Op::FtzFa2InvSum => "fa2_inv_sum",
            Op::FtzGuardNegInfZero => "guard_neg_inf_zero",
            Op::FtzMufuEx2 => "mufu_ex2_ftz",
        }
    }

    fn costs(self) -> (u32, u32) {
        // (replay_cost, proof_cost) as declared by the Python constructors.
        match self {
            Op::Mul | Op::Mul8 | Op::SiluMulBf16 => (2, 2),
            Op::TcDot16 | Op::TcDot16_0 => (TC_K as u32, TC_K as u32),
            Op::In | Op::Weight => (0, 1),
            _ => (1, 1),
        }
    }

    fn source(self) -> Option<&'static str> {
        match self {
            Op::In => Some("input"),
            Op::Weight => Some("weight"),
            _ => None,
        }
    }
}

/// Which pinned family a [`GateSet`] is (decides the source widths).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family {
    /// `veritor.toy-isa@1`, `veritor.word-arithmetic@2`: one word width.
    Toy,
    /// `veritor.int8-q7@1`: 8-bit sources, 32-bit accumulators.
    Int8Q7,
    /// `veritor.tensor-core.ampere_bf16_m16n8k16@2`: 32-bit sources (L1's pinned family).
    TensorCore,
    /// `veritor.tc-ampere-bf16@1` and `@2`: 16-bit BF16 sources, FP32 step gates, casts.
    TcAmpereBf16,
    /// `veritor.pad-serve@1` (lane PAD-IMPL): the flag/mux gates of the padded serving
    /// circuit — typed selects, `eq_i32`, boolean algebra; 32-bit sources (every
    /// operand encoding — bit, BF16, FP32, I32 — fits; arguments are still checked
    /// against each gate's own widths).
    PadServe,
    /// `veritor.reduce-mufu@1`: 64-bit sources (BF16/FP32/i32 words and MUFU
    /// table rows, each decoded by its own leaf schema), FP32 scalar gates.
    ReduceMufu,
    /// `veritor.elem-bf16@1`: 16-bit BF16 sources, the elementwise BF16 gates
    /// (`bf16_add_f2fp`, `silu_mul_bf16`) and the widening cast (lane PROOF-ELEM).
    ElemBf16,
    /// `veritor.elem-f32@1`: 32-bit FP32 sources and the hardware narrowing
    /// cast `f32_to_bf16` (`F2fpBf16_v1`, NaN -> `0x7FFF`).
    ElemF32,
    /// `veritor.pad-inactive@1`: the inactive step-transition boundary
    /// (serve3-spec §6/§7): bit, BF16, i32 sources (32-bit In) and the mux
    /// gates `select_bf16`, `select_i32`, `bool_and`.
    PadInactive,
    /// `veritor.embed-gather@1` (lane PROOF-COV): 64-bit sources (a token or
    /// constant is a `u32` leaf, an embedding-table row is a
    /// `(flat << 16) | word` `u64` leaf), the flat-index arithmetic and the
    /// index-forced gather.
    EmbedGather,
    /// `veritor.ftz-f32@1` (E4 / E7, EOD 2026-09-11): 64-bit sources (FP32 words
    /// and MUFU table rows, each decoded by its own leaf schema, as
    /// `ReduceMufu`), the FTZ scalar gates of the FA2 kernel.
    FtzF32,
}

/// A gate set family: name, version and the gates it declares (sorted by name).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GateSet {
    pub name: &'static str,
    pub version: &'static str,
    /// The statement's word: the family's parameter for the toy sets, the
    /// widest gate (32) for the mixed-width families.
    pub width: u32,
    pub family: Family,
    gates: &'static [Op],
}

const TOY_ISA_GATES: &[Op] = &[
    Op::Add,
    Op::Eq,
    Op::In,
    Op::Lt,
    Op::Mul,
    Op::Shr,
    Op::Sub,
    Op::Weight,
];
const WORD_GATES: &[Op] = &[Op::Add, Op::In, Op::Mul, Op::Weight];
const INT8_Q7_GATES: &[Op] = &[
    Op::Add32,
    Op::In,
    Op::Mul8,
    Op::RequantS8,
    Op::RequantU7,
    Op::Weight,
];
const TENSOR_CORE_GATES: &[Op] = &[Op::In, Op::TcDot16, Op::TcDot16_0, Op::Weight];
const TC_AMPERE_BF16_GATES: &[Op] = &[
    Op::Bf16ToF32,
    Op::F32ToBf16,
    Op::In,
    Op::TcDot16,
    Op::TcDot16_0,
    Op::Weight,
];
const TC_AMPERE_BF16_HW_GATES: &[Op] = &[
    Op::Bf16ToF32,
    Op::F32ToBf16Hw,
    Op::In,
    Op::TcDot16,
    Op::TcDot16_0,
    Op::Weight,
];
const PAD_SERVE_GATES: &[Op] = &[
    Op::BitAnd,
    Op::BitNot,
    Op::EqI32,
    Op::In,
    Op::SelectBf16,
    Op::SelectF32,
    Op::SelectI32,
    Op::Weight,
];
// sorted by manifest name, as every family's list is
const REDUCE_MUFU_GATES: &[Op] = &[
    Op::RBf16AddF2fp,
    Op::RBf16Gt,
    Op::Bf16ToF32,
    Op::RDivFullRcp,
    Op::RDivFullScaleA,
    Op::RF2fpBf16,
    Op::RF32Add,
    Op::RF32Div,
    Op::RF32Fma,
    Op::RF32Mul,
    Op::RI32Add,
    Op::RI32Eq,
    Op::RI32Le,
    Op::In,
    Op::RMufuSqrtTab,
    Op::RRopeOut,
    Op::RRopeOutAdd,
    Op::RRsqrtApprox,
    Op::RSelectBf16,
    Op::RSelectI32,
    Op::Weight,
];
const ELEM_BF16_GATES: &[Op] = &[
    Op::Bf16AddF2fp,
    Op::Bf16ToF32,
    Op::In,
    Op::SiluMulBf16,
    Op::Weight,
];
const ELEM_F32_GATES: &[Op] = &[Op::F32ToBf16Hw, Op::In, Op::Weight];
const PAD_INACTIVE_GATES: &[Op] = &[
    Op::BoolAnd,
    Op::In,
    Op::SelectBf16,
    Op::SelectI32,
    Op::Weight,
];
// sorted by manifest name, as every family's list is
const EMBED_GATHER_GATES: &[Op] = &[
    Op::EGatherWord,
    Op::EI32Add,
    Op::EI32Mul,
    Op::In,
    Op::Weight,
];
// sorted by manifest name, as every family's list is (E4's `family.rs::GATES` order)
const FTZ_F32_GATES: &[Op] = &[
    Op::FtzF32Add,
    Op::FtzF32Fma,
    Op::FtzF32FmaSub,
    Op::FtzF32Max,
    Op::FtzF32Mul,
    Op::FtzF32Sub,
    Op::FtzFa2InvSum,
    Op::FtzGuardNegInfZero,
    Op::In,
    Op::FtzMufuEx2,
    Op::Weight,
];

/// The identity of L1's pinned tensor-core family (`GateSet.id`).
pub const TENSOR_CORE_ID: &str = "veritor.tensor-core.ampere_bf16_m16n8k16@2";
/// The identity of the serving-shaped tensor-core family of `typed-obligation/v0`
/// (software NaN canonicalisation `0x7FC0` in `f32_to_bf16`).
pub const TC_AMPERE_BF16_ID: &str = "veritor.tc-ampere-bf16@1";
/// The identity of the same family with the hardware NaN canonicalisation
/// (`0x7FFF`) in `f32_to_bf16` (lane L8, `docs/verified-backend/l8-nan-primitive.md`).
pub const TC_AMPERE_BF16_HW_ID: &str = "veritor.tc-ampere-bf16@2";
/// The identity of the padded-serving flag/mux family (lane PAD-IMPL,
/// `docs/vllm-poc/serve3-spec.md`; Python `veritor.core.pad_serve`, Lean
/// `VeritorSpec/PadServe.lean`, semantics `crate::pad`).
pub const PAD_SERVE_ID: &str = "veritor.pad-serve@1";
/// The identity of the reduction-kernel family (lane PROOF-REDUCE,
/// `docs/vllm-poc/proof-families.md` section "reductions").
pub const REDUCE_MUFU_ID: &str = "veritor.reduce-mufu@1";
/// The identity of the elementwise BF16 family (lane PROOF-ELEM,
/// `docs/vllm-poc/proof-families.md`): `Bf16AddF2fp_v1`, `SiluMulBf16_v1`,
/// `Bf16ToF32_v1` over 16-bit sources.
pub const ELEM_BF16_ID: &str = "veritor.elem-bf16@1";
/// The identity of the elementwise FP32 family: `F2fpBf16_v1` (the hardware
/// `cvt.rn.bf16.f32`) over 32-bit sources.
pub const ELEM_F32_ID: &str = "veritor.elem-f32@1";
/// The identity of the inactive step-transition boundary family (lane
/// PROOF-ELEM, serve3-spec §6: `live_in = 0` implies boundary outputs are the
/// canonical constants).  The gates are the registered `SelectBf16_v1` /
/// `SelectI32_v1` and the spec-pinned `BoolAnd_v1`; when PAD-IMPL lands its
/// `veritor.pad-serve@1` family the statements re-issue under that id (the
/// gate semantics are the same words).
pub const PAD_INACTIVE_ID: &str = "veritor.pad-inactive@1";
/// The identity of the embedding-gather family (lane PROOF-COV,
/// `docs/vllm-poc/proof-families.md` section "embedding gather"; Python
/// `veritor.core.embed_gates`, semantics `crate::embed`).
pub const EMBED_GATHER_ID: &str = "veritor.embed-gather@1";
/// The identity of the FTZ scalar family of the fast-math FA2 attention kernel
/// (E4, EOD 2026-09-11; Python `e2e.coverage_v1.ck_ftz.make_ftz_gate_set`,
/// semantics `crate::ftz`).  The digest is E4's `family.rs` pin, reproduced here.
pub const FTZ_F32_ID: &str = "veritor.ftz-f32@1";
/// `GateSet::resolve(FTZ_F32_ID, 32).digest()` -- E4's pin (`ck_ftz.FTZ_DIGEST`).
pub const FTZ_F32_DIGEST_HEX: &str = "21f8f7d2183cad76ae62486445eee48409b3727b236b12329bb8d9b6e2353fef";

/// The widest word a value may have: `decode_schema` and the evaluators work in `u64`.
pub const MAX_WIDTH: u32 = 64;

/// Decode a canonical fixed-width big-endian value at the width its leaf
/// schema `u<width>` names (`TransparentBackend._decode`): exactly
/// `ceil(width / 8)` bytes holding a value below `2**width`.  The schema must
/// be the canonical spelling `u` followed by the decimal width (1 to 64).
#[cfg_attr(feature = "attribution", inline(never))]
pub fn decode_schema(schema: &str, payload: &[u8]) -> Option<u64> {
    let digits = schema.strip_prefix('u')?;
    if digits.is_empty() || digits.len() > 2 || digits.starts_with('0') {
        return None;
    }
    let width: u32 = digits.parse().ok()?;
    if width == 0 || width > MAX_WIDTH {
        return None;
    }
    let length = (width as usize).div_ceil(8);
    if payload.len() != length {
        return None;
    }
    let mut value = 0u64;
    for &byte in payload {
        value = (value << 8) | byte as u64;
    }
    if value > mask(width) {
        return None;
    }
    Some(value)
}

/// The mask of a `width`-bit word (`width <= 64`).
fn mask(width: u32) -> u64 {
    if width >= 64 {
        u64::MAX
    } else {
        (1u64 << width) - 1
    }
}

impl GateSet {
    /// The family named `id` (`name@version`) at `width` bits, if known.
    pub fn resolve(id: &str, width: u32) -> Option<GateSet> {
        let (name, version, family, gates, widths) = match id {
            "veritor.toy-isa@1" => (
                "veritor.toy-isa",
                "1",
                Family::Toy,
                TOY_ISA_GATES,
                1..=MAX_WIDTH,
            ),
            "veritor.word-arithmetic@2" => (
                "veritor.word-arithmetic",
                "2",
                Family::Toy,
                WORD_GATES,
                1..=MAX_WIDTH,
            ),
            "veritor.int8-q7@1" => (
                "veritor.int8-q7",
                "1",
                Family::Int8Q7,
                INT8_Q7_GATES,
                32..=32,
            ),
            TENSOR_CORE_ID => (
                "veritor.tensor-core.ampere_bf16_m16n8k16",
                "2",
                Family::TensorCore,
                TENSOR_CORE_GATES,
                32..=32,
            ),
            TC_AMPERE_BF16_ID => (
                "veritor.tc-ampere-bf16",
                "1",
                Family::TcAmpereBf16,
                TC_AMPERE_BF16_GATES,
                32..=32,
            ),
            TC_AMPERE_BF16_HW_ID => (
                "veritor.tc-ampere-bf16",
                "2",
                Family::TcAmpereBf16,
                TC_AMPERE_BF16_HW_GATES,
                32..=32,
            ),
            PAD_SERVE_ID => (
                "veritor.pad-serve",
                "1",
                Family::PadServe,
                PAD_SERVE_GATES,
                32..=32,
            ),
            REDUCE_MUFU_ID => (
                "veritor.reduce-mufu",
                "1",
                Family::ReduceMufu,
                REDUCE_MUFU_GATES,
                32..=32,
            ),
            ELEM_BF16_ID => (
                "veritor.elem-bf16",
                "1",
                Family::ElemBf16,
                ELEM_BF16_GATES,
                32..=32,
            ),
            ELEM_F32_ID => (
                "veritor.elem-f32",
                "1",
                Family::ElemF32,
                ELEM_F32_GATES,
                32..=32,
            ),
            PAD_INACTIVE_ID => (
                "veritor.pad-inactive",
                "1",
                Family::PadInactive,
                PAD_INACTIVE_GATES,
                32..=32,
            ),
            EMBED_GATHER_ID => (
                "veritor.embed-gather",
                "1",
                Family::EmbedGather,
                EMBED_GATHER_GATES,
                32..=32,
            ),
            FTZ_F32_ID => (
                "veritor.ftz-f32",
                "1",
                Family::FtzF32,
                FTZ_F32_GATES,
                32..=32,
            ),
            _ => return None,
        };
        if !widths.contains(&width) {
            return None;
        }
        Some(GateSet {
            name,
            version,
            width,
            family,
            gates,
        })
    }

    /// The gate named `op` in this set.
    #[cfg_attr(feature = "attribution", inline(never))]
    pub fn op(&self, op: &str) -> Option<Op> {
        self.gates
            .iter()
            .copied()
            .find(|gate| gate.manifest_name() == op)
    }

    /// The gates of this set, sorted by manifest name.
    pub fn ops(&self) -> &'static [Op] {
        self.gates
    }

    /// The output width of `op` in this set (`Gate.width`).
    pub fn op_width(&self, op: Op) -> u32 {
        match op {
            Op::Mul8 | Op::Add32 => 32,
            Op::RequantU7 | Op::RequantS8 => 8,
            Op::TcDot16 | Op::TcDot16_0 | Op::Bf16ToF32 => 32,
            Op::F32ToBf16 | Op::F32ToBf16Hw | Op::SelectBf16 => 16,
            Op::SelectF32 | Op::SelectI32 => 32,
            Op::EqI32 | Op::BitAnd | Op::BitNot => 1,
            Op::RF32Add | Op::RF32Mul | Op::RF32Div | Op::RF32Fma => 32,
            Op::RMufuSqrtTab | Op::RDivFullRcp | Op::RDivFullScaleA | Op::RRsqrtApprox => 32,
            Op::RI32Add | Op::RSelectI32 => 32,
            Op::RF2fpBf16 | Op::RBf16AddF2fp | Op::RRopeOut | Op::RRopeOutAdd | Op::RSelectBf16 => 16,
            Op::RBf16Gt | Op::RI32Eq | Op::RI32Le => 1,
            Op::Bf16AddF2fp | Op::SiluMulBf16 => 16,
            Op::BoolAnd => 1,
            Op::EI32Mul | Op::EI32Add => 32,
            Op::EGatherWord => 16,
            Op::FtzF32Add
            | Op::FtzF32Fma
            | Op::FtzF32FmaSub
            | Op::FtzF32Max
            | Op::FtzF32Mul
            | Op::FtzF32Sub
            | Op::FtzFa2InvSum
            | Op::FtzGuardNegInfZero
            | Op::FtzMufuEx2 => 32,
            Op::In | Op::Weight => match self.family {
                Family::Toy | Family::TensorCore | Family::ElemF32 | Family::PadInactive => self.width,
                Family::Int8Q7 => 8,
                Family::TcAmpereBf16 | Family::ElemBf16 => 16,
                Family::PadServe => 32,
                Family::ReduceMufu | Family::EmbedGather | Family::FtzF32 => 64,
            },
            Op::Add | Op::Sub | Op::Mul | Op::Lt | Op::Eq | Op::Shr => self.width,
        }
    }

    /// The width of argument `index` of `op` (`Gate.arg_widths[index]`);
    /// `index` must be below `op.arity()`.
    pub fn arg_width(&self, op: Op, index: usize) -> u32 {
        match op {
            Op::Mul8 => 8,
            Op::Add32 | Op::RequantU7 | Op::RequantS8 => 32,
            // `[acc, a0..a15, b0..b15]`: an FP32 accumulator and BF16 operands
            Op::TcDot16 => {
                if index == 0 {
                    32
                } else {
                    16
                }
            }
            Op::TcDot16_0 => 16,
            Op::F32ToBf16 | Op::F32ToBf16Hw | Op::RF2fpBf16 => 32,
            Op::Bf16ToF32 | Op::Bf16AddF2fp | Op::SiluMulBf16 => 16,
            Op::BoolAnd => 1,
            // `[c, a, b]`: one condition bit, then the two typed operands
            Op::SelectBf16 => {
                if index == 0 {
                    1
                } else {
                    16
                }
            }
            Op::RF32Add | Op::RF32Mul | Op::RF32Div | Op::RF32Fma => 32,
            Op::RDivFullScaleA | Op::RI32Add | Op::RI32Eq | Op::RI32Le => 32,
            // `(x: f32, table row: (index << 32) | word)`
            Op::RMufuSqrtTab | Op::RDivFullRcp | Op::RRsqrtApprox => {
                if index == 0 {
                    32
                } else {
                    64
                }
            }
            Op::EI32Mul | Op::EI32Add => 32,
            // `(x: f32, table row: (index << 32) | word)` -- the MUFU table contract
            Op::FtzMufuEx2 | Op::FtzFa2InvSum => {
                if index == 0 {
                    32
                } else {
                    64
                }
            }
            Op::FtzF32Add
            | Op::FtzF32Fma
            | Op::FtzF32FmaSub
            | Op::FtzF32Max
            | Op::FtzF32Mul
            | Op::FtzF32Sub
            | Op::FtzGuardNegInfZero => 32,
            // `(flat index, table row: (index << 16) | word)`
            Op::EGatherWord => {
                if index == 0 {
                    32
                } else {
                    64
                }
            }
            Op::RBf16AddF2fp | Op::RBf16Gt | Op::RRopeOut | Op::RRopeOutAdd => 16,
            // `(condition bit, then, else)`
            Op::RSelectBf16 => {
                if index == 0 {
                    1
                } else {
                    16
                }
            }
            Op::SelectF32 | Op::SelectI32 | Op::RSelectI32 => {
                if index == 0 {
                    1
                } else {
                    32
                }
            }
            Op::EqI32 => 32,
            Op::BitAnd | Op::BitNot => 1,
            _ => self.width,
        }
    }

    /// The argument widths of `op` (`Gate.arg_widths`), one per argument.
    pub fn arg_widths(&self, op: Op) -> Vec<u32> {
        (0..op.arity()).map(|i| self.arg_width(op, i)).collect()
    }

    /// The canonical JSON manifest `veritor.core.GateSet.manifest`.
    pub fn manifest(&self) -> String {
        let mut out = String::from("{\"gates\":[");
        for (index, gate) in self.gates.iter().enumerate() {
            if index > 0 {
                out.push(',');
            }
            let (replay, proof) = gate.costs();
            let source = match gate.source() {
                Some(source) => format!("\"{source}\""),
                None => "null".to_owned(),
            };
            let width = self.op_width(*gate);
            let arity = gate.arity();
            let args = self.arg_widths(*gate);
            out.push('{');
            if args.iter().any(|&arg| arg != width) {
                // only a non-default declaration is part of the identity (`Gate.manifest`)
                let listed: Vec<String> = args.iter().map(u32::to_string).collect();
                out.push_str(&format!("\"arg_widths\":[{}],", listed.join(",")));
            }
            out.push_str(&format!(
                "\"arity\":{},\"name\":\"{}\",\"proof_cost\":{},\"replay_cost\":{},\"source\":{},\"width\":{}}}",
                arity,
                gate.manifest_name(),
                proof,
                replay,
                source,
                width
            ));
        }
        out.push_str(&format!(
            "],\"name\":\"{}\",\"version\":\"{}\"}}",
            self.name, self.version
        ));
        out
    }

    /// `GateSet.digest` as raw bytes.
    pub fn digest(&self) -> [u8; 32] {
        tagged_sha256(GATE_SET_IDENTITY_TAG, self.manifest().as_bytes())
    }

    /// The mask of the statement's word.
    pub fn mask(&self) -> u64 {
        mask(self.width)
    }

    /// Evaluate `op` on `args` with the pinned semantics; `None` when the
    /// number of arguments or the width of one is not the gate's
    /// (`Gate._checked_args` raises there), or when a tensor-core gate meets
    /// a non-finite word (`silicon.tc_dot` raises, `TC.Op.eval` is `none`).
    #[cfg_attr(feature = "attribution", inline(never))]
    pub fn evaluate(&self, op: Op, args: &[u64]) -> Option<u64> {
        if args.len() != op.arity() {
            return None;
        }
        if args
            .iter()
            .enumerate()
            .any(|(index, &value)| value > mask(self.arg_width(op, index)))
        {
            return None;
        }
        match op {
            Op::TcDot16 => {
                return tc::tc_dot(
                    &tc::AMPERE_BF16_M16N8K16,
                    args[0],
                    &args[1..1 + TC_K],
                    &args[1 + TC_K..],
                );
            }
            Op::TcDot16_0 => {
                return tc::tc_dot(&tc::AMPERE_BF16_M16N8K16, 0, &args[..TC_K], &args[TC_K..]);
            }
            Op::F32ToBf16 => return Some(tc::f32_to_bf16(args[0])),
            Op::F32ToBf16Hw => return Some(tc::f32_to_bf16_hw(args[0])),
            Op::Bf16ToF32 => return Some(tc::bf16_to_f32(args[0])),
            Op::SelectBf16 | Op::SelectF32 | Op::SelectI32 => {
                return Some(pad::select(args[0], args[1], args[2]));
            }
            Op::EqI32 => return Some(pad::eq_i32(args[0], args[1])),
            Op::BitAnd => return Some(pad::bit_and(args[0], args[1])),
            Op::BitNot => return Some(pad::bit_not(args[0])),
            Op::RF32Add => return Some(reduce::f32_add(args[0], args[1])),
            Op::RF32Mul => return Some(reduce::f32_mul(args[0], args[1])),
            Op::RF32Div => return Some(reduce::f32_div(args[0], args[1])),
            Op::RF32Fma => return Some(reduce::f32_fma(args[0], args[1], args[2])),
            Op::RF2fpBf16 => return Some(reduce::f2fp_bf16(args[0])),
            Op::RBf16AddF2fp => return Some(reduce::bf16_add_f2fp(args[0], args[1])),
            Op::RRopeOut => return Some(reduce::rope_out(args[0], args[1], args[2], args[3])),
            Op::RRopeOutAdd => {
                return Some(reduce::rope_out_add(args[0], args[1], args[2], args[3]));
            }
            Op::RBf16Gt => return Some(reduce::bf16_gt(args[0], args[1])),
            Op::RSelectBf16 | Op::RSelectI32 => {
                return Some(reduce::select(args[0], args[1], args[2]));
            }
            Op::RI32Add => return Some(reduce::i32_add(args[0], args[1])),
            Op::RI32Eq => return Some(reduce::i32_eq(args[0], args[1])),
            Op::RI32Le => return Some(reduce::i32_le(args[0], args[1])),
            // a table gate is `None` -- a rejection -- when the opened row's
            // index is not the one the hardware would read for the operand
            Op::RMufuSqrtTab => return reduce::mufu_sqrt_tab(args[0], args[1]),
            Op::RDivFullRcp => return reduce::div_full_rcp(args[0], args[1]),
            Op::RDivFullScaleA => return Some(reduce::div_full_scale_a(args[0], args[1])),
            Op::RRsqrtApprox => return reduce::rsqrt_approx(args[0], args[1]),
            Op::Bf16AddF2fp => return Some(elem::bf16_add_f2fp(args[0], args[1])),
            Op::SiluMulBf16 => return Some(elem::silu_mul_bf16(args[0], args[1])),
            // (`SelectBf16`/`SelectI32` are handled by the shared `pad::select` arm above --
            // same registered `c ? a : b` bit-pattern select.)
            // `BoolAnd_v1` (serve3-spec §9): logical and on 1-bit encodings
            Op::BoolAnd => return Some(args[0] & args[1] & 1),
            Op::EI32Mul => return Some(embed::i32_mul(args[0], args[1])),
            Op::EI32Add => return Some(embed::i32_add(args[0], args[1])),
            // `None` -- a rejection -- when the opened row's embedded index is
            // not the recomputed flat index (the MUFU table contract)
            Op::EGatherWord => return embed::gather_word(args[0], args[1]),
            Op::FtzF32Add => return Some(ftz::f32_add_ftz(args[0], args[1])),
            Op::FtzF32Sub => return Some(ftz::f32_sub_ftz(args[0], args[1])),
            Op::FtzF32Mul => return Some(ftz::f32_mul_ftz(args[0], args[1])),
            Op::FtzF32Fma => return Some(ftz::f32_fma_ftz(args[0], args[1], args[2])),
            Op::FtzF32FmaSub => return Some(ftz::f32_fma_sub_ftz(args[0], args[1], args[2])),
            Op::FtzF32Max => return Some(ftz::f32_max_ftz(args[0], args[1])),
            Op::FtzGuardNegInfZero => return Some(ftz::guard_neg_inf_zero(args[0])),
            // `None` -- a rejection -- when the opened row is not the index the
            // hardware reads for the operand (the MUFU table contract)
            Op::FtzMufuEx2 => return ftz::mufu_ex2_tab(args[0], args[1]),
            Op::FtzFa2InvSum => return ftz::fa2_inv_sum_tab(args[0], args[1]),
            _ => {}
        }
        let mask = self.mask();
        let (a, b) = (
            args.first().copied().unwrap_or(0),
            args.get(1).copied().unwrap_or(0),
        );
        Some(match op {
            Op::Add => a.wrapping_add(b) & mask,
            Op::Sub => a.wrapping_sub(b) & mask,
            Op::Mul => a.wrapping_mul(b) & mask,
            Op::Lt => (a < b) as u64,
            Op::Eq => (a == b) as u64,
            Op::Shr => {
                if b < self.width as u64 {
                    a >> b
                } else {
                    0
                }
            }
            Op::Mul8 => (i32::from(a as u8 as i8) * i32::from(b as u8 as i8)) as u32 as u64,
            Op::Add32 => (a as u32).wrapping_add(b as u32) as u64,
            // `>> 7` on an `i32` is an arithmetic shift: floor(s / 128), toward -inf.
            Op::RequantU7 => ((a as u32 as i32) >> 7).clamp(0, 127) as u64,
            Op::RequantS8 => ((a as u32 as i32) >> 7).clamp(-128, 127) as i8 as u8 as u64,
            Op::TcDot16 | Op::TcDot16_0 | Op::F32ToBf16 | Op::F32ToBf16Hw | Op::Bf16ToF32
            | Op::SelectBf16 | Op::SelectF32 | Op::SelectI32 | Op::EqI32 | Op::BitAnd
            | Op::BitNot
            | Op::Bf16AddF2fp | Op::SiluMulBf16 | Op::BoolAnd
            | Op::RF32Add | Op::RF32Mul | Op::RF32Div | Op::RF32Fma | Op::RF2fpBf16
            | Op::RBf16AddF2fp | Op::RRopeOut | Op::RRopeOutAdd | Op::RBf16Gt
            | Op::RSelectBf16 | Op::RSelectI32 | Op::RI32Add | Op::RI32Eq | Op::RI32Le
            | Op::RMufuSqrtTab | Op::RDivFullRcp | Op::RDivFullScaleA | Op::RRsqrtApprox
            | Op::EI32Mul | Op::EI32Add | Op::EGatherWord
            | Op::FtzF32Add | Op::FtzF32Fma | Op::FtzF32FmaSub | Op::FtzF32Max | Op::FtzF32Mul
            | Op::FtzF32Sub | Op::FtzFa2InvSum | Op::FtzGuardNegInfZero | Op::FtzMufuEx2 => {
                unreachable!("handled above")
            }
            Op::In | Op::Weight => unreachable!("source gates have no relation"),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_matches_the_python_layout() {
        let set = GateSet::resolve("veritor.word-arithmetic@2", 8).unwrap();
        assert_eq!(
            set.manifest(),
            "{\"gates\":[\
             {\"arity\":2,\"name\":\"add\",\"proof_cost\":1,\"replay_cost\":1,\"source\":null,\"width\":8},\
             {\"arity\":0,\"name\":\"in\",\"proof_cost\":1,\"replay_cost\":0,\"source\":\"input\",\"width\":8},\
             {\"arity\":2,\"name\":\"mul\",\"proof_cost\":2,\"replay_cost\":2,\"source\":null,\"width\":8},\
             {\"arity\":0,\"name\":\"weight\",\"proof_cost\":1,\"replay_cost\":0,\"source\":\"weight\",\"width\":8}\
             ],\"name\":\"veritor.word-arithmetic\",\"version\":\"2\"}"
        );
    }

    #[test]
    fn int8_q7_manifest_declares_mixed_widths_only_where_they_differ() {
        let set = GateSet::resolve("veritor.int8-q7@1", 32).unwrap();
        assert_eq!(
            set.manifest(),
            "{\"gates\":[\
             {\"arity\":2,\"name\":\"add32\",\"proof_cost\":1,\"replay_cost\":1,\"source\":null,\"width\":32},\
             {\"arity\":0,\"name\":\"in\",\"proof_cost\":1,\"replay_cost\":0,\"source\":\"input\",\"width\":8},\
             {\"arg_widths\":[8,8],\"arity\":2,\"name\":\"mul8\",\"proof_cost\":2,\"replay_cost\":2,\"source\":null,\"width\":32},\
             {\"arg_widths\":[32],\"arity\":1,\"name\":\"requant_s8\",\"proof_cost\":1,\"replay_cost\":1,\"source\":null,\"width\":8},\
             {\"arg_widths\":[32],\"arity\":1,\"name\":\"requant_u7\",\"proof_cost\":1,\"replay_cost\":1,\"source\":null,\"width\":8},\
             {\"arity\":0,\"name\":\"weight\",\"proof_cost\":1,\"replay_cost\":0,\"source\":\"weight\",\"width\":8}\
             ],\"name\":\"veritor.int8-q7\",\"version\":\"1\"}"
        );
        // the set resolves only at its statement width, the widest gate
        assert!(GateSet::resolve("veritor.int8-q7@1", 8).is_none());
        assert!(GateSet::resolve("veritor.int8-q7@1", 64).is_none());
    }

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }

    #[test]
    fn tensor_core_manifest_and_digest_match_the_pinned_family() {
        let set = GateSet::resolve(TENSOR_CORE_ID, 32).unwrap();
        let sixteen = |w: u32| std::iter::repeat_n(w.to_string(), 16).collect::<Vec<_>>().join(",");
        let bf16 = sixteen(16);
        assert_eq!(
            set.manifest(),
            format!(
                "{{\"gates\":[\
                 {{\"arity\":0,\"name\":\"in\",\"proof_cost\":1,\"replay_cost\":0,\"source\":\"input\",\"width\":32}},\
                 {{\"arg_widths\":[32,{bf16},{bf16}],\"arity\":33,\"name\":\"tc_dot16\",\"proof_cost\":16,\"replay_cost\":16,\"source\":null,\"width\":32}},\
                 {{\"arg_widths\":[{bf16},{bf16}],\"arity\":32,\"name\":\"tc_dot16_0\",\"proof_cost\":16,\"replay_cost\":16,\"source\":null,\"width\":32}},\
                 {{\"arity\":0,\"name\":\"weight\",\"proof_cost\":1,\"replay_cost\":0,\"source\":\"weight\",\"width\":32}}\
                 ],\"name\":\"veritor.tensor-core.ampere_bf16_m16n8k16\",\"version\":\"2\"}}"
            )
        );
        // the digest `lean/vectors/tc-ampere-bf16.json` and `silicon.tensor_core_gates` pin
        assert_eq!(
            hex(&set.digest()),
            "ce525ee5ed107a1005f4fd95765529ac9b90ac27e895bdaf46a4e1d16a849302"
        );
        assert!(GateSet::resolve(TENSOR_CORE_ID, 16).is_none());
    }

    #[test]
    fn tc_ampere_bf16_manifest_matches_the_python_family() {
        let set = GateSet::resolve(TC_AMPERE_BF16_ID, 32).unwrap();
        let sixteen = |w: u32| std::iter::repeat_n(w.to_string(), 16).collect::<Vec<_>>().join(",");
        let bf16 = sixteen(16);
        assert_eq!(
            set.manifest(),
            format!(
                "{{\"gates\":[\
                 {{\"arg_widths\":[16],\"arity\":1,\"name\":\"bf16_to_f32\",\"proof_cost\":1,\"replay_cost\":1,\"source\":null,\"width\":32}},\
                 {{\"arg_widths\":[32],\"arity\":1,\"name\":\"f32_to_bf16\",\"proof_cost\":1,\"replay_cost\":1,\"source\":null,\"width\":16}},\
                 {{\"arity\":0,\"name\":\"in\",\"proof_cost\":1,\"replay_cost\":0,\"source\":\"input\",\"width\":16}},\
                 {{\"arg_widths\":[32,{bf16},{bf16}],\"arity\":33,\"name\":\"tc_dot16\",\"proof_cost\":16,\"replay_cost\":16,\"source\":null,\"width\":32}},\
                 {{\"arg_widths\":[{bf16},{bf16}],\"arity\":32,\"name\":\"tc_dot16_0\",\"proof_cost\":16,\"replay_cost\":16,\"source\":null,\"width\":32}},\
                 {{\"arity\":0,\"name\":\"weight\",\"proof_cost\":1,\"replay_cost\":0,\"source\":\"weight\",\"width\":16}}\
                 ],\"name\":\"veritor.tc-ampere-bf16\",\"version\":\"1\"}}"
            )
        );
        assert_eq!(hex(&set.digest()), TC_AMPERE_BF16_DIGEST_HEX);
    }

    /// `veritor.core.tc.TC_AMPERE_BF16.digest` (pinned by `veritor/core/tests`).
    const TC_AMPERE_BF16_DIGEST_HEX: &str =
        "cfe960f86e786bbbea42af6dc382342ad26ff280590a071db743e21bda01d307";
    /// `veritor.core.tc.make_tc_ampere_bf16_gate_set("2").digest` (pinned by `tests/veritor/core/test_tc_nan.py`).
    const TC_AMPERE_BF16_HW_DIGEST_HEX: &str =
        "ca39ce505c70125046bcc9e7af49f84ac55b111558cff9dd201dd326c6421f0a";

    #[test]
    fn tc_ampere_bf16_hw_is_the_same_manifest_at_version_2_with_its_own_digest() {
        let old = GateSet::resolve(TC_AMPERE_BF16_ID, 32).unwrap();
        let new = GateSet::resolve(TC_AMPERE_BF16_HW_ID, 32).unwrap();
        // the manifests differ only in the version string: same gate names, widths, costs
        assert_eq!(
            new.manifest(),
            old.manifest()
                .replace("\"version\":\"1\"", "\"version\":\"2\"")
        );
        assert_ne!(new.manifest(), old.manifest());
        assert_eq!(hex(&new.digest()), TC_AMPERE_BF16_HW_DIGEST_HEX);
        assert_eq!(hex(&old.digest()), TC_AMPERE_BF16_DIGEST_HEX);
        assert!(GateSet::resolve(TC_AMPERE_BF16_HW_ID, 16).is_none());
        // the same gate name resolves to a different op in each set
        assert_eq!(old.op("f32_to_bf16"), Some(Op::F32ToBf16));
        assert_eq!(new.op("f32_to_bf16"), Some(Op::F32ToBf16Hw));
        assert_eq!(new.op_width(Op::F32ToBf16Hw), 16);
        assert_eq!(new.arg_widths(Op::F32ToBf16Hw), vec![32]);
        // the delta: NaN words only
        for nan in [0x7FC0_0000u64, 0xFFC0_0000, 0x7F80_0001, 0xFFFF_FFFF, 0x7FC0_0001] {
            assert_eq!(old.evaluate(Op::F32ToBf16, &[nan]), Some(0x7FC0));
            assert_eq!(new.evaluate(Op::F32ToBf16Hw, &[nan]), Some(0x7FFF));
        }
        for finite in [0x3F80_0000u64, 0x3F80_8000, 0x7F7F_FFFF, 0x7F80_0000, 0xFF80_0000, 0] {
            assert_eq!(
                old.evaluate(Op::F32ToBf16, &[finite]),
                new.evaluate(Op::F32ToBf16Hw, &[finite])
            );
        }
        assert_eq!(new.evaluate(Op::F32ToBf16Hw, &[0x1_0000_0000]), None);
        assert_eq!(new.evaluate(Op::F32ToBf16Hw, &[1, 2]), None);
        // the old entry keeps its number: the new variant is declared last
        assert!((Op::F32ToBf16Hw as usize) > (Op::Weight as usize));
    }

    /// `zk/bench/elem_gates.py make_elem_bf16_gate_set().digest`.
    const ELEM_BF16_DIGEST_HEX: &str =
        "061723740a994b6b47148b19a0b5c5a701edcd52398fd10e9a4a41d4575f5b9e";
    /// `zk/bench/elem_gates.py make_elem_f32_gate_set().digest`.
    const ELEM_F32_DIGEST_HEX: &str =
        "574632aeb865498b2eede41be0f75410e0194f078d1aebf18f643248ba442eac";

    #[test]
    fn elem_bf16_manifest_and_digest_match_the_python_family() {
        let set = GateSet::resolve(ELEM_BF16_ID, 32).unwrap();
        assert_eq!(
            set.manifest(),
            "{\"gates\":[\
             {\"arity\":2,\"name\":\"bf16_add_f2fp\",\"proof_cost\":1,\"replay_cost\":1,\"source\":null,\"width\":16},\
             {\"arg_widths\":[16],\"arity\":1,\"name\":\"bf16_to_f32\",\"proof_cost\":1,\"replay_cost\":1,\"source\":null,\"width\":32},\
             {\"arity\":0,\"name\":\"in\",\"proof_cost\":1,\"replay_cost\":0,\"source\":\"input\",\"width\":16},\
             {\"arity\":2,\"name\":\"silu_mul_bf16\",\"proof_cost\":2,\"replay_cost\":2,\"source\":null,\"width\":16},\
             {\"arity\":0,\"name\":\"weight\",\"proof_cost\":1,\"replay_cost\":0,\"source\":\"weight\",\"width\":16}\
             ],\"name\":\"veritor.elem-bf16\",\"version\":\"1\"}"
        );
        assert_eq!(hex(&set.digest()), ELEM_BF16_DIGEST_HEX);
        // resolves only at the statement width (the widest gate, bf16_to_f32's 32)
        assert!(GateSet::resolve(ELEM_BF16_ID, 16).is_none());
        assert_eq!(set.op("bf16_add_f2fp"), Some(Op::Bf16AddF2fp));
        assert_eq!(set.op("silu_mul_bf16"), Some(Op::SiluMulBf16));
        assert_eq!(set.op("tc_dot16"), None);
        assert_eq!(set.op_width(Op::In), 16);
        // 1 + 1 = 2; silu(1) * 1 = bf16(0.7310586)
        assert_eq!(set.evaluate(Op::Bf16AddF2fp, &[0x3F80, 0x3F80]), Some(0x4000));
        assert_eq!(set.evaluate(Op::SiluMulBf16, &[0x3F80, 0x3F80]), Some(0x3F3B));
        assert_eq!(set.evaluate(Op::Bf16ToF32, &[0x3F80]), Some(0x3F80_0000));
        // NaN canonicalisation: F2FP word for the add, torch word for silu*mul
        assert_eq!(set.evaluate(Op::Bf16AddF2fp, &[0x7FC0, 0x3F80]), Some(0x7FFF));
        assert_eq!(set.evaluate(Op::SiluMulBf16, &[0x7FC0, 0x3F80]), Some(0x7FC0));
        // a 17-bit argument or the wrong arity is no evaluation
        assert_eq!(set.evaluate(Op::Bf16AddF2fp, &[0x1_0000, 0]), None);
        assert_eq!(set.evaluate(Op::SiluMulBf16, &[0x3F80]), None);
    }

    /// `zk/bench/elem_gates.py make_pad_inactive_gate_set().digest`.
    const PAD_INACTIVE_DIGEST_HEX: &str =
        "683bd49cfc6af1ae9e388978c32598dfc43e71b0a8a5c6a33f363c61e7f9085c";

    #[test]
    fn pad_inactive_manifest_and_digest_match_the_python_family() {
        let set = GateSet::resolve(PAD_INACTIVE_ID, 32).unwrap();
        assert_eq!(
            set.manifest(),
            "{\"gates\":[\
             {\"arity\":2,\"name\":\"bool_and\",\"proof_cost\":1,\"replay_cost\":1,\"source\":null,\"width\":1},\
             {\"arity\":0,\"name\":\"in\",\"proof_cost\":1,\"replay_cost\":0,\"source\":\"input\",\"width\":32},\
             {\"arg_widths\":[1,16,16],\"arity\":3,\"name\":\"select_bf16\",\"proof_cost\":1,\"replay_cost\":1,\"source\":null,\"width\":16},\
             {\"arg_widths\":[1,32,32],\"arity\":3,\"name\":\"select_i32\",\"proof_cost\":1,\"replay_cost\":1,\"source\":null,\"width\":32},\
             {\"arity\":0,\"name\":\"weight\",\"proof_cost\":1,\"replay_cost\":0,\"source\":\"weight\",\"width\":32}\
             ],\"name\":\"veritor.pad-inactive\",\"version\":\"1\"}"
        );
        assert_eq!(hex(&set.digest()), PAD_INACTIVE_DIGEST_HEX);
        // the boundary constants of an inactive step (serve3-spec §6)
        assert_eq!(set.evaluate(Op::SelectBf16, &[0, 0x3F80, 0x0000]), Some(0x0000));
        assert_eq!(set.evaluate(Op::SelectBf16, &[1, 0x3F80, 0x0000]), Some(0x3F80));
        assert_eq!(set.evaluate(Op::SelectI32, &[0, 0xDEAD_BEEF, 0]), Some(0));
        assert_eq!(set.evaluate(Op::SelectI32, &[1, 0xDEAD_BEEF, 0]), Some(0xDEAD_BEEF));
        assert_eq!(set.evaluate(Op::BoolAnd, &[0, 1]), Some(0));
        assert_eq!(set.evaluate(Op::BoolAnd, &[1, 1]), Some(1));
        // a 2-bit condition or flag is outside Value<1>
        assert_eq!(set.evaluate(Op::SelectBf16, &[2, 0, 0]), None);
        assert_eq!(set.evaluate(Op::BoolAnd, &[2, 1]), None);
        assert_eq!(set.op_width(Op::In), 32);
    }

    #[test]
    fn ftz_f32_manifest_and_digest_match_e4s_family_pin() {
        let set = GateSet::resolve(FTZ_F32_ID, 32).unwrap();
        assert_eq!(hex(&set.digest()), FTZ_F32_DIGEST_HEX);
        assert!(GateSet::resolve(FTZ_F32_ID, 16).is_none());
        assert!(GateSet::resolve(FTZ_F32_ID, 64).is_none());
        assert_eq!(set.op_width(Op::In), 64);
        assert_eq!(set.op("f32_mul_ftz"), Some(Op::FtzF32Mul));
        assert_eq!(set.op("mufu_ex2_ftz"), Some(Op::FtzMufuEx2));
        assert_eq!(set.arg_widths(Op::FtzMufuEx2), vec![32, 64]);
        assert_eq!(set.arg_widths(Op::FtzF32Fma), vec![32, 32, 32]);
        // E4's vectors: 3 * 7 = 21 exact; a subnormal product flushes to +0
        assert_eq!(set.evaluate(Op::FtzF32Mul, &[0x4040_0000, 0x40E0_0000]), Some(0x41A8_0000));
        assert_eq!(set.evaluate(Op::FtzF32Mul, &[0x0000_0001, 0x3F80_0000]), Some(0));
        // a wrong word is a rejection at the caller (evaluate returns the true word)
        assert_ne!(set.evaluate(Op::FtzF32Mul, &[0x3F80_0000, 0x4000_0000]), Some(0x4000_0001));
        // the table gate rejects a row of any other index than the one the operand reads
        let index = ftz::mufu_ex2_index(0x3F80_0000);
        assert!(set.evaluate(Op::FtzMufuEx2, &[0x3F80_0000, index << 32]).is_some());
        assert_eq!(set.evaluate(Op::FtzMufuEx2, &[0x3F80_0000, (index ^ 1) << 32]), None);
        // -inf guard
        assert_eq!(set.evaluate(Op::FtzGuardNegInfZero, &[0xFF80_0000]), Some(0));
        assert_eq!(set.evaluate(Op::FtzGuardNegInfZero, &[0x3F80_0000]), Some(0x3F80_0000));
        // every earlier family's digest is untouched by the appended family
        assert_eq!(hex(&GateSet::resolve(ELEM_F32_ID, 32).unwrap().digest()), ELEM_F32_DIGEST_HEX);
    }

    #[test]
    fn elem_f32_manifest_and_digest_match_the_python_family() {
        let set = GateSet::resolve(ELEM_F32_ID, 32).unwrap();
        assert_eq!(
            set.manifest(),
            "{\"gates\":[\
             {\"arg_widths\":[32],\"arity\":1,\"name\":\"f32_to_bf16\",\"proof_cost\":1,\"replay_cost\":1,\"source\":null,\"width\":16},\
             {\"arity\":0,\"name\":\"in\",\"proof_cost\":1,\"replay_cost\":0,\"source\":\"input\",\"width\":32},\
             {\"arity\":0,\"name\":\"weight\",\"proof_cost\":1,\"replay_cost\":0,\"source\":\"weight\",\"width\":32}\
             ],\"name\":\"veritor.elem-f32\",\"version\":\"1\"}"
        );
        assert_eq!(hex(&set.digest()), ELEM_F32_DIGEST_HEX);
        assert!(GateSet::resolve(ELEM_F32_ID, 16).is_none());
        // the F2fpBf16_v1 semantics: the same op as tc-ampere-bf16@2's cast
        assert_eq!(set.op("f32_to_bf16"), Some(Op::F32ToBf16Hw));
        assert_eq!(set.op_width(Op::In), 32);
        assert_eq!(set.evaluate(Op::F32ToBf16Hw, &[0x7FC0_0001]), Some(0x7FFF));
        assert_eq!(set.evaluate(Op::F32ToBf16Hw, &[0x3F80_8000]), Some(0x3F80));
    }

    #[test]
    fn tensor_core_gates_evaluate_and_reject_non_finite_words() {
        let set = GateSet::resolve(TC_AMPERE_BF16_ID, 32).unwrap();
        let mut args = vec![0x3F80_0000u64];
        args.extend(std::iter::repeat_n(0x3F80u64, 32));
        assert_eq!(set.evaluate(Op::TcDot16, &args), Some(0x4188_0000));
        assert_eq!(set.evaluate(Op::TcDot16_0, &args[1..]), Some(0x4180_0000));
        assert_eq!(set.evaluate(Op::TcDot16, &args[1..]), None); // arity
        assert_eq!(set.evaluate(Op::TcDot16_0, &args), None);
        let mut wide = args.clone();
        wide[1] = 0x1_0000; // a 17-bit "BF16" operand
        assert_eq!(set.evaluate(Op::TcDot16, &wide), None);
        let mut nan = args.clone();
        nan[0] = 0x7FC0_0000;
        assert_eq!(set.evaluate(Op::TcDot16, &nan), None);
        assert_eq!(set.evaluate(Op::F32ToBf16, &[0x7FC0_0001]), Some(0x7FC0));
        assert_eq!(set.evaluate(Op::F32ToBf16, &[0x1_0000_0000]), None);
        assert_eq!(set.evaluate(Op::Bf16ToF32, &[0x3F80]), Some(0x3F80_0000));
        assert_eq!(set.evaluate(Op::Bf16ToF32, &[0x1_0000]), None);
        assert_eq!(set.op_width(Op::In), 16);
        assert_eq!(GateSet::resolve(TENSOR_CORE_ID, 32).unwrap().op_width(Op::In), 32);
        assert!(set.op("f32_to_bf16").is_some());
        assert!(GateSet::resolve(TENSOR_CORE_ID, 32).unwrap().op("f32_to_bf16").is_none());
    }

    #[test]
    fn semantics_are_modular() {
        let set = GateSet::resolve("veritor.toy-isa@1", 16).unwrap();
        assert_eq!(set.evaluate(Op::Add, &[0xFFFF, 1]), Some(0));
        assert_eq!(set.evaluate(Op::Sub, &[0, 1]), Some(0xFFFF));
        assert_eq!(set.evaluate(Op::Mul, &[0x8000, 2]), Some(0));
        assert_eq!(set.evaluate(Op::Lt, &[3, 4]), Some(1));
        assert_eq!(set.evaluate(Op::Eq, &[3, 4]), Some(0));
        assert_eq!(set.evaluate(Op::Shr, &[0x8000, 15]), Some(1));
        assert_eq!(set.evaluate(Op::Shr, &[0x8000, 16]), Some(0));
        // an argument outside the word, or the wrong number of them, is no evaluation
        assert_eq!(set.evaluate(Op::Add, &[0x1_0000, 1]), None);
        assert_eq!(set.evaluate(Op::Add, &[1]), None);
    }

    #[test]
    fn int8_q7_semantics_floor_and_saturate() {
        let set = GateSet::resolve("veritor.int8-q7@1", 32).unwrap();
        let s8 = |v: i32| v as i8 as u8 as u64;
        let s32 = |v: i32| v as u32 as u64;
        assert_eq!(set.evaluate(Op::Mul8, &[s8(-128), s8(-128)]), Some(16384));
        assert_eq!(
            set.evaluate(Op::Mul8, &[s8(127), s8(-128)]),
            Some(s32(-16256))
        );
        assert_eq!(set.evaluate(Op::Mul8, &[s8(127), s8(127)]), Some(16129));
        assert_eq!(set.evaluate(Op::Mul8, &[0, s8(-1)]), Some(0));
        assert_eq!(set.evaluate(Op::Mul8, &[256, 1]), None);
        assert_eq!(set.evaluate(Op::Add32, &[s32(-1), 1]), Some(0));
        assert_eq!(
            set.evaluate(Op::Add32, &[s32(i32::MAX), 1]),
            Some(s32(i32::MIN))
        );
        assert_eq!(set.evaluate(Op::Add32, &[1 << 32, 0]), None);
        for (t, u7, sat) in [
            (0, 0, 0),
            (-1, 0, -1),
            (-128, 0, -1),
            (-129, 0, -2),
            (127, 0, 0),
            (128, 1, 1),
            (16256, 127, 127),
            (16383, 127, 127),
            (16384, 127, 127),
            (-16384, 0, -128),
            (-16385, 0, -128),
            (i32::MIN, 0, -128),
            (i32::MAX, 127, 127),
        ] {
            assert_eq!(
                set.evaluate(Op::RequantU7, &[s32(t)]),
                Some(u7),
                "requant_u7({t})"
            );
            assert_eq!(
                set.evaluate(Op::RequantS8, &[s32(t)]),
                Some(s8(sat)),
                "requant_s8({t})"
            );
        }
        assert_eq!(set.evaluate(Op::RequantU7, &[0, 0]), None);
    }

    /// `veritor.core.pad_serve.make_pad_serve_gate_set().digest` (pinned by
    /// `tests/veritor/core/test_pad_serve.py`).
    const PAD_SERVE_DIGEST_HEX: &str =
        "3aa430e2d7e6cb4dca307acc5defcff84cdb7ce9b79423cd01af12aa3a10c536";

    #[test]
    fn pad_serve_manifest_matches_the_python_family() {
        let set = GateSet::resolve(PAD_SERVE_ID, 32).unwrap();
        assert_eq!(
            set.manifest(),
            "{\"gates\":[\
             {\"arity\":2,\"name\":\"bit_and\",\"proof_cost\":1,\"replay_cost\":1,\"source\":null,\"width\":1},\
             {\"arity\":1,\"name\":\"bit_not\",\"proof_cost\":1,\"replay_cost\":1,\"source\":null,\"width\":1},\
             {\"arg_widths\":[32,32],\"arity\":2,\"name\":\"eq_i32\",\"proof_cost\":1,\"replay_cost\":1,\"source\":null,\"width\":1},\
             {\"arity\":0,\"name\":\"in\",\"proof_cost\":1,\"replay_cost\":0,\"source\":\"input\",\"width\":32},\
             {\"arg_widths\":[1,16,16],\"arity\":3,\"name\":\"select_bf16\",\"proof_cost\":1,\"replay_cost\":1,\"source\":null,\"width\":16},\
             {\"arg_widths\":[1,32,32],\"arity\":3,\"name\":\"select_f32\",\"proof_cost\":1,\"replay_cost\":1,\"source\":null,\"width\":32},\
             {\"arg_widths\":[1,32,32],\"arity\":3,\"name\":\"select_i32\",\"proof_cost\":1,\"replay_cost\":1,\"source\":null,\"width\":32},\
             {\"arity\":0,\"name\":\"weight\",\"proof_cost\":1,\"replay_cost\":0,\"source\":\"weight\",\"width\":32}\
             ],\"name\":\"veritor.pad-serve\",\"version\":\"1\"}"
        );
        assert_eq!(hex(&set.digest()), PAD_SERVE_DIGEST_HEX);
        // resolves only at the statement width (its widest gate), like the mixed-width families
        assert!(GateSet::resolve(PAD_SERVE_ID, 16).is_none());
        assert!(GateSet::resolve(PAD_SERVE_ID, 1).is_none());
        // the new entries are declared after every earlier gate: existing numbering intact
        assert!((Op::SelectBf16 as usize) > (Op::F32ToBf16Hw as usize));
        // and the pre-existing family digests are what they were
        assert_eq!(
            hex(&GateSet::resolve(TC_AMPERE_BF16_HW_ID, 32).unwrap().digest()),
            TC_AMPERE_BF16_HW_DIGEST_HEX
        );
    }

    #[test]
    fn pad_serve_gates_evaluate_and_reject_out_of_width_words() {
        let set = GateSet::resolve(PAD_SERVE_ID, 32).unwrap();
        // typed selects copy the named operand
        assert_eq!(set.evaluate(Op::SelectBf16, &[1, 0x3F80, 0x7FFF]), Some(0x3F80));
        assert_eq!(set.evaluate(Op::SelectBf16, &[0, 0x3F80, 0x7FFF]), Some(0x7FFF));
        assert_eq!(
            set.evaluate(Op::SelectF32, &[1, 0x3F80_0000, 0xFFFF_FFFF]),
            Some(0x3F80_0000)
        );
        assert_eq!(
            set.evaluate(Op::SelectI32, &[0, 7, 0xFFFF_FFFF]),
            Some(0xFFFF_FFFF)
        );
        // the stop test and the boolean algebra
        assert_eq!(set.evaluate(Op::EqI32, &[151645, 151645]), Some(1));
        assert_eq!(set.evaluate(Op::EqI32, &[151645, 0]), Some(0));
        assert_eq!(set.evaluate(Op::BitAnd, &[1, 1]), Some(1));
        assert_eq!(set.evaluate(Op::BitAnd, &[0, 1]), Some(0));
        assert_eq!(set.evaluate(Op::BitNot, &[0]), Some(1));
        assert_eq!(set.evaluate(Op::BitNot, &[1]), Some(0));
        // width checks: a 2-valued "bit", a 17-bit "bf16", a 33-bit "i32", wrong arity
        assert_eq!(set.evaluate(Op::SelectBf16, &[2, 0, 0]), None);
        assert_eq!(set.evaluate(Op::SelectBf16, &[1, 0x1_0000, 0]), None);
        assert_eq!(set.evaluate(Op::EqI32, &[0x1_0000_0000, 0]), None);
        assert_eq!(set.evaluate(Op::BitNot, &[2]), None);
        assert_eq!(set.evaluate(Op::SelectBf16, &[1, 0]), None);
        assert_eq!(set.evaluate(Op::BitAnd, &[1]), None);
        // widths as declared
        assert_eq!(set.op_width(Op::SelectBf16), 16);
        assert_eq!(set.op_width(Op::EqI32), 1);
        assert_eq!(set.op_width(Op::In), 32);
        assert_eq!(set.arg_widths(Op::SelectBf16), vec![1, 16, 16]);
        assert_eq!(set.arg_widths(Op::SelectI32), vec![1, 32, 32]);
        assert_eq!(set.arg_widths(Op::EqI32), vec![32, 32]);
        assert_eq!(set.arg_widths(Op::BitNot), vec![1]);
        // the pad gates are not resolvable in the other families
        assert!(set.op("tc_dot16").is_none());
        assert!(GateSet::resolve(TC_AMPERE_BF16_ID, 32).unwrap().op("select_bf16").is_none());
        assert_eq!(set.op("select_bf16"), Some(Op::SelectBf16));
        assert_eq!(set.op("eq_i32"), Some(Op::EqI32));
    }

    /// `veritor.core.embed_gates.make_embed_gather_gate_set().digest` (pinned by
    /// `tests/veritor/core/test_embed_gates.py`).
    const EMBED_GATHER_DIGEST_HEX: &str =
        "7af70afd60a0e4a1a446c9d6b3b44866b98e9dcc0116353e37dfe97c14e2eb15";

    #[test]
    fn embed_gather_manifest_matches_the_python_family() {
        let set = GateSet::resolve(EMBED_GATHER_ID, 32).unwrap();
        assert_eq!(
            set.manifest(),
            "{\"gates\":[\
             {\"arg_widths\":[32,64],\"arity\":2,\"name\":\"gather_word\",\"proof_cost\":1,\"replay_cost\":1,\"source\":null,\"width\":16},\
             {\"arity\":2,\"name\":\"i32_add\",\"proof_cost\":1,\"replay_cost\":1,\"source\":null,\"width\":32},\
             {\"arity\":2,\"name\":\"i32_mul\",\"proof_cost\":1,\"replay_cost\":1,\"source\":null,\"width\":32},\
             {\"arity\":0,\"name\":\"in\",\"proof_cost\":1,\"replay_cost\":0,\"source\":\"input\",\"width\":64},\
             {\"arity\":0,\"name\":\"weight\",\"proof_cost\":1,\"replay_cost\":0,\"source\":\"weight\",\"width\":64}\
             ],\"name\":\"veritor.embed-gather\",\"version\":\"1\"}"
        );
        assert_eq!(hex(&set.digest()), EMBED_GATHER_DIGEST_HEX);
        // resolves only at the statement width, like the mixed-width families
        assert!(GateSet::resolve(EMBED_GATHER_ID, 16).is_none());
        assert!(GateSet::resolve(EMBED_GATHER_ID, 64).is_none());
        // the new entries are declared after every earlier gate: numbering intact
        assert!((Op::EI32Mul as usize) > (Op::RRsqrtApprox as usize));
        // and the pre-existing family digests are what they were
        assert_eq!(
            hex(&GateSet::resolve(PAD_SERVE_ID, 32).unwrap().digest()),
            PAD_SERVE_DIGEST_HEX
        );
        assert_eq!(
            hex(&GateSet::resolve(TC_AMPERE_BF16_HW_ID, 32).unwrap().digest()),
            TC_AMPERE_BF16_HW_DIGEST_HEX
        );
    }

    #[test]
    fn embed_gather_gates_evaluate_and_reject() {
        let set = GateSet::resolve(EMBED_GATHER_ID, 32).unwrap();
        // the flat-index chain: base = tok * H, then +1 per column
        assert_eq!(set.evaluate(Op::EI32Mul, &[49151, 576]), Some(49151 * 576));
        assert_eq!(set.evaluate(Op::EI32Add, &[49151 * 576, 1]), Some(49151 * 576 + 1));
        assert_eq!(set.evaluate(Op::EI32Mul, &[0xFFFF_FFFF, 2]), Some(0xFFFF_FFFE));
        assert_eq!(set.evaluate(Op::EI32Add, &[0xFFFF_FFFF, 1]), Some(0));
        // the gather: forces the leaf's embedded flat index
        let leaf = (28_311_551u64 << 16) | 0xBEEF;
        assert_eq!(set.evaluate(Op::EGatherWord, &[28_311_551, leaf]), Some(0xBEEF));
        assert_eq!(set.evaluate(Op::EGatherWord, &[28_311_550, leaf]), None);
        assert_eq!(set.evaluate(Op::EGatherWord, &[0, 0xABCD]), Some(0xABCD));
        assert_eq!(set.evaluate(Op::EGatherWord, &[1, 0xABCD]), None);
        // width and arity checks: a 33-bit flat, a 33-bit add operand, wrong arity
        assert_eq!(set.evaluate(Op::EGatherWord, &[0x1_0000_0000, 0]), None);
        assert_eq!(set.evaluate(Op::EI32Add, &[0x1_0000_0000, 0]), None);
        assert_eq!(set.evaluate(Op::EGatherWord, &[0]), None);
        // widths as declared
        assert_eq!(set.op_width(Op::EGatherWord), 16);
        assert_eq!(set.op_width(Op::EI32Mul), 32);
        assert_eq!(set.op_width(Op::In), 64);
        assert_eq!(set.arg_widths(Op::EGatherWord), vec![32, 64]);
        assert_eq!(set.arg_widths(Op::EI32Add), vec![32, 32]);
        // the gates are not resolvable in the other families
        assert!(GateSet::resolve(REDUCE_MUFU_ID, 32).unwrap().op("gather_word").is_none());
        assert_eq!(set.op("gather_word"), Some(Op::EGatherWord));
        assert!(set.op("mufu_sqrt_tab").is_none());
    }

    #[test]
    fn values_decode_by_their_leaf_schema() {
        assert_eq!(decode_schema("u16", &[0x12, 0x34]), Some(0x1234));
        assert_eq!(decode_schema("u16", &[0x12]), None);
        assert_eq!(decode_schema("u4", &[0x1f]), None);
        assert_eq!(decode_schema("u4", &[0x0f]), Some(15));
        assert_eq!(decode_schema("u8", &[0xff]), Some(255));
        assert_eq!(decode_schema("u32", &[0x80, 0, 0, 0]), Some(1 << 31));
        assert_eq!(decode_schema("u32", &[0x80, 0, 0]), None);
        assert_eq!(decode_schema("u64", &[0xff; 8]), Some(u64::MAX));
        for bad in ["", "u", "u0", "u08", "u65", "u128", "x8", "8", "u8 "] {
            assert_eq!(decode_schema(bad, &[0]), None, "schema {bad:?}");
        }
    }
}
