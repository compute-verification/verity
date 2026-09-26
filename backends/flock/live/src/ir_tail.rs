//! The native tail of a cut IR template (lane flock-ir-lowering): the verifier's own evaluation of the gates that read no
//! input word (RMSNorm's warp-aggregate sum and row scalar), from the program `verity_flock.ir_lower.tail_program` emits.
//!
//! A program is ops `[out slot, primitive id, operand slots...]` (constants `[slot, "Const", value]`) over slots
//! `[cut words | temporaries]`. The primitives are the IR's (`verity_vllm.program.registry.prims`), line for line: IEEE f32
//! add, mul and div; fma with one rounding and every NaN the canonical 0x7FC00000 (`f32_fma_bits`); and the MUFU ops as the
//! IR's C++ model computes them (`rms_triton_model.cpp`: `rsqrt_nonftz`, `mufu_sqrt`, `div_full_parts`, `mufu_rcp`) on the
//! measured tables, which are pinned here by sha256.

use std::collections::BTreeSet;

use sha2::{Digest, Sha256};

/// sha256 of the raw u32 LE tables (`verity_flock.ir_lower.TABLES`, from `rms_relation.tables("cuda")`).
pub const TABLE_SHA256: [(&str, &str, usize); 3] = [
    ("rsq", "218fc04d271773269367dedf4cff9ff719081b1218c3d67d248d8d6126040246", 1 << 24),
    ("sqrt", "4e017678de306e0be8bed66193c2ebf09e9c60a9600cad56cd05bb7b6c3f266f", 1 << 24),
    ("rcp", "c408381446132e07fa5581e930dc33fb790bea7ec4276ef3621e322df2492a55", 1 << 23),
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
}

impl Prim {
    fn arity(self) -> usize {
        match self {
            Prim::Const(_) => 0,
            Prim::RsqrtApprox | Prim::MufuSqrtFtz | Prim::DivFullRcp => 1,
            Prim::F32Add | Prim::F32Mul | Prim::F32Div | Prim::DivFullScaleA => 2,
            Prim::F32Fma => 3,
        }
    }

    fn table(self) -> Option<&'static str> {
        match self {
            Prim::RsqrtApprox => Some("rsq"),
            Prim::MufuSqrtFtz => Some("sqrt"),
            Prim::DivFullRcp => Some("rcp"),
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

fn fma_bits(a: u32, b: u32, c: u32) -> u32 {
    let y = f(a).mul_add(f(b), f(c));
    if y.is_nan() { 0x7FC0_0000 } else { u(y) }
}

fn is_nan(x: u32) -> bool {
    x & 0x7FFF_FFFF > 0x7F80_0000
}

/// IEEE f32 with the IR's NaN choice (numpy float32 on x86-64, red-team-flock-2 IR5), fixed here rather than left to the
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

    /// `(a, b, add, mul, div)` from `verity_vllm.program.registry.prims` (numpy float32, x86-64): quiet and signaling NaNs of
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
