//! The native tail of a cut IR template: the verifier's own evaluation of the gates that read no
//! input word (RMSNorm's warp-aggregate sum and row scalar), from the program `verity_flock.ir_lower.tail_program` emits.
//!
//! A program is ops `[out slot, primitive id, operand slots...]` (constants `[slot, "Const", value]`) over slots
//! `[cut words | temporaries]`. The primitives are the IR's (`verity.ml.definitions.registry.prims`), line for line: IEEE f32
//! add, mul and div; fma with one rounding and every NaN the canonical 0x7FC00000 (`f32_fma_bits`); and the MUFU ops as the
//! IR's C++ model computes them (`rms_triton_model.cpp`: `rsqrt_nonftz`, `mufu_sqrt`, `div_full_parts`, `mufu_rcp`) on the
//! measured tables, which are pinned here by sha256.

use std::collections::BTreeSet;

use sha2::{Digest, Sha256};

/// sha256 of the raw u32 LE tables (`verity_flock.ir_lower.TABLES`, from `rms_relation.tables("cuda")`).
/// `ex2` is FA2's MUFU.EX2 table (`fa2_relation.tables().ex2`, hw ex2.approx.ftz of j / 2^23); its MUFU.RCP table is `rcp`, byte for
/// byte.
pub const TABLE_SHA256: [(&str, &str, usize); 4] = [
    ("rsq", "218fc04d271773269367dedf4cff9ff719081b1218c3d67d248d8d6126040246", 1 << 24),
    ("sqrt", "4e017678de306e0be8bed66193c2ebf09e9c60a9600cad56cd05bb7b6c3f266f", 1 << 24),
    ("rcp", "c408381446132e07fa5581e930dc33fb790bea7ec4276ef3621e322df2492a55", 1 << 23),
    ("ex2", "b2a42c4a626469c986e33e43f16f41bde9d84de94347cd9ceaa1bd68d36bcdf0", 1 << 23),
];

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Prim {
    Const(u32),
    F32Add,
    F32Mul,
    F32Fma,
    F32Div,
    RsqrtApprox,
    MufuSqrtFtz,
    DivFullRcp,
    DivFullScaleA,
    F32AddFtz,
    F32SubFtz,
    F32MulFtz,
    F32FmaFtz,
    F32FmaSubFtz,
    F32Max,
    GuardNegInfZero,
    MufuEx2Ftz,
    Fa2InvSum,
    F2fpBf16,
}

impl Prim {
    fn arity(self) -> usize {
        match self {
            Prim::Const(_) => 0,
            Prim::RsqrtApprox | Prim::MufuSqrtFtz | Prim::DivFullRcp | Prim::GuardNegInfZero | Prim::MufuEx2Ftz | Prim::Fa2InvSum | Prim::F2fpBf16 => 1,
            Prim::F32Add | Prim::F32Mul | Prim::F32Div | Prim::DivFullScaleA | Prim::F32AddFtz | Prim::F32SubFtz | Prim::F32MulFtz | Prim::F32Max => 2,
            Prim::F32Fma | Prim::F32FmaFtz | Prim::F32FmaSubFtz => 3,
        }
    }

    fn table(self) -> Option<&'static str> {
        match self {
            Prim::RsqrtApprox => Some("rsq"),
            Prim::MufuSqrtFtz => Some("sqrt"),
            Prim::DivFullRcp | Prim::Fa2InvSum => Some("rcp"),
            Prim::MufuEx2Ftz => Some("ex2"),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Op {
    pub out: usize,
    pub prim: Prim,
    pub args: Vec<usize>,
}

/// Parse a tail program; every op's operands must be slots defined before it (cut words `< cut_words` count as defined when
/// `produced` has them) and every out slot defined once.
pub fn parse(tail: &serde_json::Value, cut_words: usize, produced: &BTreeSet<usize>) -> Result<Vec<Op>, String> {
    let ops = tail.as_array().ok_or("the tail program is not a list")?;
    let mut defined: BTreeSet<usize> = produced.clone();
    let mut out = Vec::new();
    for op in ops {
        let v = op.as_array().ok_or("a tail op is not a list")?;
        let slot = v.first().and_then(|x| x.as_u64()).ok_or("a tail op without an out slot")? as usize;
        let id = v.get(1).and_then(|x| x.as_str()).ok_or("a tail op without a primitive")?;
        let rest: Vec<u64> = v[2..].iter().map(|x| x.as_u64().ok_or("a tail operand is not an integer")).collect::<Result<_, _>>()?;
        let prim = match id {
            "Const" => Prim::Const(*rest.first().ok_or("a constant without a value")? as u32),
            "F32Add_v1" => Prim::F32Add,
            "F32Mul_v1" => Prim::F32Mul,
            "F32Fma_v1" => Prim::F32Fma,
            "F32Div_v1" => Prim::F32Div,
            "RsqrtApprox_v1" => Prim::RsqrtApprox,
            "MufuSqrtFtz_v1" => Prim::MufuSqrtFtz,
            "DivFullRcp_v1" => Prim::DivFullRcp,
            "DivFullScaleA_v1" => Prim::DivFullScaleA,
            "F32AddFtz_v1" => Prim::F32AddFtz,
            "F32SubFtz_v1" => Prim::F32SubFtz,
            "F32MulFtz_v1" => Prim::F32MulFtz,
            "F32FmaFtz_v1" => Prim::F32FmaFtz,
            "F32FmaSubFtz_v1" => Prim::F32FmaSubFtz,
            "F32Max_v1" => Prim::F32Max,
            "GuardNegInfZero_v1" => Prim::GuardNegInfZero,
            "MufuEx2Ftz_v1" => Prim::MufuEx2Ftz,
            "Fa2InvSum_v1" => Prim::Fa2InvSum,
            "F2fpBf16_v1" => Prim::F2fpBf16,
            other => return Err(format!("the tail primitive {other} has no verifier implementation")),
        };
        let args: Vec<usize> = if let Prim::Const(_) = prim { vec![] } else { rest.iter().map(|&x| x as usize).collect() };
        if args.len() != prim.arity() || (matches!(prim, Prim::Const(_)) && rest.len() != 1) {
            return Err(format!("tail op {id} with {} operands", args.len()));
        }
        if let Some(a) = args.iter().find(|a| !defined.contains(a)) {
            return Err(format!("tail op {id} reads slot {a} before it is defined"));
        }
        if !defined.insert(slot) {
            return Err(format!("tail slot {slot} defined twice"));
        }
        out.push(Op { out: slot, prim, args });
    }
    if let Some(k) = (0..cut_words).find(|k| !defined.contains(k)) {
        return Err(format!("cut word {k} is computed neither by a unit nor by the tail"));
    }
    Ok(out)
}

pub fn tables_needed(ops: &[Op]) -> BTreeSet<&'static str> {
    ops.iter().filter_map(|o| o.prim.table()).collect()
}

#[derive(Default)]
pub struct Tables {
    pub rsq: Vec<u32>,
    pub sqrt: Vec<u32>,
    pub rcp: Vec<u32>,
    pub ex2: Vec<u32>,
}

impl Tables {
    /// Load `<dir>/<name>.u32` for every needed table and check its pinned sha256.
    pub fn load(dir: &str, need: &BTreeSet<&'static str>) -> Result<Self, String> {
        let mut t = Tables::default();
        for (name, sha, n) in TABLE_SHA256 {
            if !need.contains(name) {
                continue;
            }
            let b = std::fs::read(format!("{dir}/{name}.u32")).map_err(|e| format!("MUFU table {name} under {dir:?}: {e}"))?;
            let got: String = Sha256::digest(&b).iter().map(|x| format!("{x:02x}")).collect();
            if got != sha || b.len() != 4 * n {
                return Err(format!("MUFU table {name}: sha256 {got} is not the pinned {sha}"));
            }
            let v: Vec<u32> = b.chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
            match name {
                "rsq" => t.rsq = v,
                "sqrt" => t.sqrt = v,
                "ex2" => t.ex2 = v,
                _ => t.rcp = v,
            }
        }
        Ok(t)
    }
}

fn f(u: u32) -> f32 {
    f32::from_bits(u)
}

fn u(x: f32) -> u32 {
    x.to_bits()
}

const BIG: f32 = 8.507059173023462e37; // 2^126
const TINY: f32 = 1.1754943508222875e-38; // 2^-126

fn mufu_rcp(xb: u32, r: &[u32]) -> u32 {
    let (s, ax) = (xb & 0x8000_0000, xb & 0x7FFF_FFFF);
    let e = ((ax >> 23) & 0xFF) as i32;
    if e == 0xFF {
        return if ax & 0x7F_FFFF != 0 { 0x7FFF_FFFF } else { s };
    }
    if e == 0 {
        return s | 0x7F80_0000;
    }
    let t = r[(ax & 0x7F_FFFF) as usize];
    let re = ((t >> 23) & 0xFF) as i32 - (e - 127);
    if re <= 0 {
        return s;
    }
    if re >= 0xFF {
        return s | 0x7F80_0000;
    }
    s | ((re as u32) << 23) | (t & 0x7F_FFFF)
}

fn mufu_sqrt(xb: u32, t: &[u32]) -> u32 {
    let (s, ax) = (xb & 0x8000_0000, xb & 0x7FFF_FFFF);
    let e = ((ax >> 23) & 0xFF) as i32;
    if e == 0xFF {
        return if ax & 0x7F_FFFF != 0 { 0x7FFF_FFFF } else if s != 0 { 0x7FFF_FFFF } else { 0x7F80_0000 };
    }
    if e == 0 {
        return s;
    }
    if s != 0 {
        return 0x7FFF_FFFF;
    }
    let k = e - 127;
    let par = k & 1;
    let v = t[(((par as u32) << 23) | (ax & 0x7F_FFFF)) as usize];
    let re = ((v >> 23) & 0xFF) as i32 + (k - par) / 2;
    ((re as u32) << 23) | (v & 0x7F_FFFF)
}

fn mufu_rsq(xb: u32, t: &[u32]) -> u32 {
    let (s, ax) = (xb & 0x8000_0000, xb & 0x7FFF_FFFF);
    let e = ((ax >> 23) & 0xFF) as i32;
    if e == 0xFF {
        return if ax & 0x7F_FFFF != 0 { 0x7FFF_FFFF } else if s != 0 { 0x7FFF_FFFF } else { 0 };
    }
    if e == 0 {
        return s | 0x7F80_0000;
    }
    if s != 0 {
        return 0x7FFF_FFFF;
    }
    let k = e - 127;
    let par = k & 1;
    let v = t[(((par as u32) << 23) | (ax & 0x7F_FFFF)) as usize];
    let re = ((v >> 23) & 0xFF) as i32 - (k - par) / 2;
    ((re as u32) << 23) | (v & 0x7F_FFFF)
}

fn rsqrt_nonftz(x: f32, t: &[u32]) -> f32 {
    let p0 = !(x.abs() < TINY);
    let x = if p0 { x } else { x * 16_777_216.0 };
    let r = f(mufu_rsq(u(x), t));
    if p0 { r } else { r * 4096.0 }
}

fn div_full_parts(a: f32, b: f32, r: &[u32]) -> (f32, f32) {
    let ab = b.abs();
    let (p0, p1) = (ab > BIG, !(ab < TINY));
    let (mut a, mut b) = (a, b);
    if p0 {
        a *= 0.25;
        b *= 0.25;
    }
    if !p1 {
        a *= 16_777_216.0;
        b *= 16_777_216.0;
    }
    (a, f(mufu_rcp(u(b), r)))
}

/// FA2's MUFU.EX2 (`fa2_model.cpp` `mufu_ex2_bits`): K = floor(|x| 2^23); for x >= 0, n = K >> 23 and j = K mod 2^23; for x < 0,
/// the ones' complement of the truncated magnitude (carry only when the fraction bits are zero); T[j] * 2^n, overflow +inf,
/// underflow +0; ftz inputs give 1.0.
fn mufu_ex2(xb: u32, t: &[u32]) -> u32 {
    let ax = xb & 0x7FFF_FFFF;
    let e = ((ax >> 23) & 0xFF) as i32;
    if e == 0xFF {
        return if ax & 0x7F_FFFF != 0 { 0x7FFF_FFFF } else if xb >> 31 != 0 { 0 } else { 0x7F80_0000 };
    }
    if e == 0 {
        return 0x3F80_0000;
    }
    let unb = e - 127;
    if unb >= 30 {
        return if xb >> 31 != 0 { 0 } else { 0x7F80_0000 };
    }
    let mant = ((ax & 0x7F_FFFF) | 0x80_0000) as u64;
    let k = if unb >= 0 { mant << unb } else { mant.checked_shr((-unb) as u32).unwrap_or(0) };
    let fixed: i64 = if xb >> 31 == 0 { k as i64 } else if k & 0x7F_FFFF == 0 { -(k as i64) } else { -(k as i64) - 1 };
    let (n, j) = (fixed >> 23, (fixed & 0x7F_FFFF) as usize);
    let tj = t[j];
    let re = ((tj >> 23) & 0xFF) as i64 + n;
    if re >= 255 {
        0x7F80_0000
    } else if re <= 0 {
        0
    } else {
        ((re as u32) << 23) | (tj & 0x7F_FFFF)
    }
}

/// Flush a subnormal to signed zero (the fast-math build's ftz).
fn ftz(u: u32) -> u32 {
    if u & 0x7F80_0000 == 0 && u & 0x7F_FFFF != 0 { u & 0x8000_0000 } else { u }
}

fn fma_bits(a: u32, b: u32, c: u32) -> u32 {
    let y = f(a).mul_add(f(b), f(c));
    if y.is_nan() { 0x7FC0_0000 } else { u(y) }
}

fn is_nan(x: u32) -> bool {
    x & 0x7FFF_FFFF > 0x7F80_0000
}

/// IEEE f32 with the IR's NaN choice (numpy float32 on x86-64), fixed here rather than left to the
/// compiler's operand order: a NaN operand comes back quieted, `b` when both are NaN if `second` (add, mul) else `a` (div); an
/// invalid operation gives 0xFFC00000.
fn ieee(a: u32, b: u32, second: bool, op: fn(f32, f32) -> f32) -> u32 {
    match (is_nan(a), is_nan(b)) {
        (true, true) => (if second { b } else { a }) | 0x40_0000,
        (true, false) => a | 0x40_0000,
        (false, true) => b | 0x40_0000,
        _ => {
            let y = op(f(a), f(b));
            if y.is_nan() { 0xFFC0_0000 } else { u(y) }
        }
    }
}

/// One primitive on encodings.
pub fn apply(p: Prim, a: &[u32], t: &Tables) -> u32 {
    match p {
        Prim::Const(v) => v,
        Prim::F32Add => ieee(a[0], a[1], true, |x, y| x + y),
        Prim::F32Mul => ieee(a[0], a[1], true, |x, y| x * y),
        Prim::F32Div => ieee(a[0], a[1], false, |x, y| x / y),
        Prim::F32Fma => fma_bits(a[0], a[1], a[2]),
        Prim::RsqrtApprox => u(rsqrt_nonftz(f(a[0]), &t.rsq)),
        Prim::MufuSqrtFtz => mufu_sqrt(a[0], &t.sqrt),
        Prim::DivFullRcp => {
            // the IR's DivFullRcp: pre-scale b, then div.full(1.0, b) = a' * rcp(b')
            let fb = f(a[0]);
            let ab = fb.abs();
            let fb = if ab > BIG { fb * 0.25 } else if !(ab < TINY) { fb } else { fb * 16_777_216.0 };
            let (x, r) = div_full_parts(1.0, fb, &t.rcp);
            u(x * r)
        }
        Prim::DivFullScaleA => {
            let ab = f(a[1]).abs();
            if ab > BIG { u(f(a[0]) * 0.25) } else if ab < TINY { u(f(a[0]) * 16_777_216.0) } else { a[0] }
        }
        Prim::F32AddFtz => ftz(ieee(ftz(a[0]), ftz(a[1]), true, |x, y| x + y)),
        Prim::F32SubFtz => ftz(ieee(ftz(a[0]), ftz(a[1]), false, |x, y| x - y)),
        Prim::F32MulFtz => ftz(ieee(ftz(a[0]), ftz(a[1]), true, |x, y| x * y)),
        Prim::F32FmaFtz => ftz(fma_bits(ftz(a[0]), ftz(a[1]), ftz(a[2]))),
        Prim::F32FmaSubFtz => ftz(fma_bits(ftz(a[0]), ftz(a[1]), ftz(a[2]) ^ 0x8000_0000)),
        // `x > y ? x : y` on the flushed words (FA2 MaxOp; a NaN compares false)
        Prim::F32Max => {
            let (x, y) = (ftz(a[0]), ftz(a[1]));
            if f(x) > f(y) { x } else { y }
        }
        Prim::GuardNegInfZero => if a[0] == 0xFF80_0000 { 0 } else { a[0] },
        Prim::MufuEx2Ftz => mufu_ex2(a[0], &t.ex2),
        Prim::Fa2InvSum => if a[0] & 0x7FFF_FFFF == 0 || is_nan(a[0]) { 0x3F80_0000 } else { mufu_rcp(a[0], &t.rcp) },
        Prim::F2fpBf16 => if is_nan(a[0]) { 0x7FFF } else { (a[0].wrapping_add(0x7FFF + ((a[0] >> 16) & 1)) >> 16) & 0xFFFF },
    }
}

/// Evaluate `ops` over `slots` (the units' cut words already set); every out slot is written.
pub fn eval(ops: &[Op], slots: &mut Vec<Option<u32>>, t: &Tables) {
    for op in ops {
        let args: Vec<u32> = op.args.iter().map(|&s| slots[s].expect("tail operand defined")).collect();
        if slots.len() <= op.out {
            slots.resize(op.out + 1, None);
        }
        slots[op.out] = Some(apply(op.prim, &args, t));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `(a, b, add, mul, div)` from `verity.ml.definitions.registry.prims` (numpy float32, x86-64): quiet and signaling NaNs of
    /// both signs, one or both operands, and invalid operations.
    const VECTORS: [(u32, u32, u32, u32, u32); 10] = [
        (0x7FC0_0001, 0x7FC0_0002, 0x7FC0_0002, 0x7FC0_0002, 0x7FC0_0001),
        (0x7F80_0001, 0x7FC0_0002, 0x7FC0_0002, 0x7FC0_0002, 0x7FC0_0001),
        (0x7FC0_0001, 0x7F80_0002, 0x7FC0_0002, 0x7FC0_0002, 0x7FC0_0001),
        (0x7F80_0001, 0x7F80_0002, 0x7FC0_0002, 0x7FC0_0002, 0x7FC0_0001),
        (0xFFC0_0003, 0x7FC0_0004, 0x7FC0_0004, 0x7FC0_0004, 0xFFC0_0003),
        (0x7FA0_0005, 0xFF90_0006, 0xFFD0_0006, 0xFFD0_0006, 0x7FE0_0005),
        (0x7FC0_0001, 0x3F80_0000, 0x7FC0_0001, 0x7FC0_0001, 0x7FC0_0001),
        (0x3F80_0000, 0x7F80_0009, 0x7FC0_0009, 0x7FC0_0009, 0x7FC0_0009),
        (0x7F80_0000, 0xFF80_0000, 0xFFC0_0000, 0xFF80_0000, 0xFFC0_0000),
        (0x0000_0000, 0x7F80_0000, 0x7F80_0000, 0xFFC0_0000, 0x0000_0000),
    ];

    /// Differential against the IR (`verity_flock.ir_lower` vectors): `IR_TAIL_VECTORS` = JSON `{prim id: [[args.., want]..]}`,
    /// `IR_TAIL_TABLES` = a `write_tables` directory; skipped unless both are set.
    #[test]
    fn prims_match_ir_vectors() {
        let (Ok(v), Ok(dir)) = (std::env::var("IR_TAIL_VECTORS"), std::env::var("IR_TAIL_TABLES")) else { return };
        let all: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(v).unwrap()).unwrap();
        let t = Tables::load(&dir, &TABLE_SHA256.iter().map(|x| x.0).collect()).unwrap();
        let mut bad = Vec::new();
        for (id, cases) in all.as_object().unwrap() {
            let op: Vec<serde_json::Value> = [serde_json::json!(0), serde_json::json!(id)].into_iter().chain((1..=parse_arity(id)).map(|k| serde_json::json!(k))).collect();
            let ops = parse(&serde_json::json!([op]), 0, &(1..4).collect()).unwrap();
            for c in cases.as_array().unwrap() {
                let c: Vec<u32> = c.as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect();
                let got = apply(ops[0].prim, &c[..c.len() - 1], &t);
                if got != c[c.len() - 1] {
                    bad.push(format!("{id} {:08x?} -> {got:08x}, want {:08x}", &c[..c.len() - 1], c[c.len() - 1]));
                }
            }
        }
        assert!(bad.is_empty(), "{} mismatches, e.g. {:?}", bad.len(), &bad[..bad.len().min(5)]);
    }

    fn parse_arity(id: &str) -> usize {
        parse(&serde_json::json!([[0, id, 1, 2, 3]]), 0, &(1..4).collect()).map(|o| o[0].args.len())
            .or_else(|_| parse(&serde_json::json!([[0, id, 1, 2]]), 0, &(1..4).collect()).map(|o| o[0].args.len()))
            .or_else(|_| parse(&serde_json::json!([[0, id, 1]]), 0, &(1..4).collect()).map(|o| o[0].args.len()))
            .unwrap()
    }

    #[test]
    fn nan_payloads_are_the_irs() {
        let t = Tables::default();
        for (a, b, add, mul, div) in VECTORS {
            assert_eq!(apply(Prim::F32Add, &[a, b], &t), add, "add {a:08x} {b:08x}");
            assert_eq!(apply(Prim::F32Mul, &[a, b], &t), mul, "mul {a:08x} {b:08x}");
            assert_eq!(apply(Prim::F32Div, &[a, b], &t), div, "div {a:08x} {b:08x}");
        }
    }
}
