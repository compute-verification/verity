//! LGTO0001: lane ligerito-relation's batch proof (`backends/direct/ligerito/proof.py`), its statement LGTS0001 and
//! verification key LGVK0001, verified end to end: statement -> public rows -> the relation sumchecks (LGSC0002 or
//! ligerito-sumcheck-2's LGSC0003) -> the J evaluation claims (+ red-team-2 V1's zero claims on the virtual rows) ->
//! the multi-claim Ligerito PCS opening (`prove.pcs_verify`; ZK: `proto/pcs.verify_open` with the column padding).
//!
//! ~~~text
//! LGTS0001  magic | str relation | 32 B sys_id | 32 B vk_sha256 | u32 l | u32 steps | u32 K | u32 S | S x u32 n_vus
//!           | u8 word_bytes | u8 y_bytes | per real sub-batch: l x K a | l x K b | l y
//! LGVK0001  magic | str relation | 32 B sys_id | u32 m | u32 n_i | u32 n_k | u32 K | u32 n_links | n_links x u32 c_row
//!           | u32 n_virt | n_virt x (str name, u32 row) | 3 x (u32 nnz | nnz x (u32 k, u32 row, u32 coef))
//! LGTO0001  magic | blob params JSON (absorbed verbatim) | blob framing JSON | blob sumcheck | L x 32 B root
//!           | per round k'_i x 3 x 6 u32 | per round |S_i| x W_i u32 rows + sib_len[i] B siblings | 2^{k_L} x 6 u32 y
//!           | (zk) L x t_pad x 6 u32 ybar
//! transcript: lgto/params, lgto/stmt = sha256(stmt), lgto/vk = sha256(key), root_1 | sumcheck | z (J points, PCS
//!           order), beta <- J | PCS rounds (ZK: ybar after root_{i+1}, the last after y)
//! ~~~
//!
//! The public rows are never materialised: `StmtRows` evaluates each at the zero-check's `r_c` from the statement words
//! (operand rows through a per-operand histogram of `eq(r_c, .)` over the word values, so the cost is `O(C K)` additions).

use crate::claims::{powers, Term};
use crate::field::{dot_ext_base, mul, pow, Ext, EXT_DEG, P};
use crate::format::Reader;
use crate::hash::{hex, sha256};
use crate::json::{self, Json};
use crate::merkle::Digest;
use crate::params::{canonical_split, Params};
use crate::pcs::{check_opening, row_from_words, Opening};
use crate::relation::{self, aligned_cover, zero_claims, Constraints, EvalClaim, Layout, PublicRows};
use crate::transcript::{ext_bytes, Coins, Live};
use std::cell::RefCell;

pub const MAGIC_PROOF: &[u8; 8] = b"LGTO0001";
pub const MAGIC_STMT: &[u8; 8] = b"LGTS0001";
pub const MAGIC_KEY: &[u8; 8] = b"LGVK0001";
/// ligerito-zk's `T_PAD`: padding coefficients per committed column under ZK.
pub const T_PAD: usize = 256;
/// `RowMaskLayout.b`: the ZK index map interleaves `a = k_L - b` column bits below the row bits.
const ZK_MASK_B: usize = 3;
/// log2 |F| of the challenge field F_{p^6}.
fn ext_log2() -> f64 {
    6.0 * (P as f64).log2()
}

// ------------------------------------------------------------------------------------------------------ relations ----

/// The operand word format (`unit.public_vectors` BF16, `public_vectors_fp8` E4M3; ligero-verify `Operand`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Operand {
    /// sign 1 | exponent 8 | fraction 7, bias 127; the all-ones exponent field is off the domain.
    Bf16,
    /// sign 1 | exponent 4 | fraction 3, bias 7; the NaN codes 0x7F / 0xFF are off the domain.
    E4m3,
    /// NVFP4 units of `nvf4::WORDS` byte words (64 E2M1 codes + 4 UE4M3 scales): the pins are functions of the whole
    /// unit (`nvf4::unit_pins`), not of one word.
    Nvf4,
}

impl Operand {
    pub fn word_bits(self) -> u32 {
        match self {
            Operand::Bf16 => 16,
            Operand::E4m3 | Operand::Nvf4 => 8,
        }
    }

    /// `(s, m, e)` with `e` a signed integer (reduced mod p by the caller); `Err` off the finite domain.
    pub fn decode(self, w: u32) -> Result<(u32, u32, i64), String> {
        match self {
            Operand::Nvf4 => Err("NVFP4 words decode per unit".into()),
            Operand::Bf16 => {
                if w >> 16 != 0 {
                    return Err("x.a/x.b range".into());
                }
                let t = (w >> 7) & 0xFF;
                if t == 0xFF {
                    return Err("operand exponent field off the finite range".into());
                }
                let nz = (t != 0) as u32;
                Ok((w >> 15, (w & 0x7F) + nz * 128, t as i64 - nz as i64 - 126))
            }
            Operand::E4m3 => {
                if w >> 8 != 0 {
                    return Err("x.a/x.b range".into());
                }
                if w & 0x7F == 0x7F {
                    return Err("E4M3 NaN operand".into());
                }
                let t = (w >> 3) & 0xF;
                let nz = (t != 0) as u32;
                Ok((w >> 7, (w & 0x7) + nz * 8, t as i64 - nz as i64 - 6))
            }
        }
    }
}

fn signed_mod_p(x: i64) -> u32 {
    x.rem_euclid(P as i64) as u32
}

/// The NVFP4 unit's public decode (`fp4/witness.py public_vectors_fp4`; ligero-verify `relation::nvf4`, same code).
pub mod nvf4 {
    /// E2M1 codes per operand per step, codes per scale block, blocks, words per operand per step (the statement's K).
    pub const K: usize = 64;
    pub const GROUP: usize = 16;
    pub const GROUPS: usize = K / GROUP;
    pub const WORDS: usize = K + GROUPS;
    /// pins per unit: `a[0..K].n, b[0..K].n, g[0..GROUPS].(sm, X, part), anc.G`
    pub const PINS: usize = 2 * K + 3 * GROUPS + 1;
    /// adder bits below the leading participating group's unit weight, and the grid floor (`relation.py`)
    pub const FRAC_BITS: i64 = 27;
    pub const GRID_FLOOR: i64 = -174;
    /// E2M1 magnitudes in halves: 0, .5, 1, 1.5, 2, 3, 4, 6.
    const NUMERATOR: [i64; 8] = [0, 1, 2, 3, 4, 6, 8, 12];

    pub fn numerator(code: u32) -> Result<i64, String> {
        if code >= 16 {
            return Err("E2M1 code outside 4 bits".into());
        }
        let n = NUMERATOR[(code & 7) as usize];
        Ok(if code & 8 != 0 { -n } else { n })
    }

    /// UE4M3 as `(mantissa, exponent)`: `m 2^-9` at exponent field 0, else `(8 + m) 2^(e - 10)`; bit 7 and 0x7F are off
    /// the domain.
    pub fn scale(byte: u32) -> Result<(i64, i64), String> {
        if byte >= 128 {
            return Err("UE4M3 scale byte with the padding bit set (PTX ISA: invalid)".into());
        }
        if byte == 0x7F {
            return Err("UE4M3 NaN scale".into());
        }
        let (e, m) = (((byte >> 3) & 0xF) as i64, (byte & 7) as i64);
        Ok(if e == 0 { (m, -9) } else { (8 + m, e - 10) })
    }

    /// One unit's pins from its `a` / `b` rows of `WORDS` words, in `pin_index` order.
    pub fn unit_pins(a: &[u32], b: &[u32]) -> Result<[i64; PINS], String> {
        let mut out = [0i64; PINS];
        for i in 0..K {
            out[i] = numerator(a[i])?;
            out[K + i] = numerator(b[i])?;
        }
        let mut anchor = GRID_FLOOR;
        for g in 0..GROUPS {
            let (ma, ea) = scale(a[K + g])?;
            let (mb, eb) = scale(b[K + g])?;
            let part = (GROUP * g..GROUP * (g + 1)).any(|i| out[i] * out[K + i] != 0) && ma != 0 && mb != 0;
            let x = -2 + ea + eb;
            out[2 * K + 3 * g..2 * K + 3 * g + 3].copy_from_slice(&[ma * mb, x, part as i64]);
            if part {
                anchor = anchor.max(x - FRAC_BITS);
            }
        }
        out[PINS - 1] = anchor;
        Ok(out)
    }

    /// `a[i].n`, `b[i].n`, `g[g].sm` / `.X` / `.part`, `anc.G` (canonical decimal indices only)
    pub fn pin_index(name: &str) -> Option<usize> {
        if name == "anc.G" {
            return Some(PINS - 1);
        }
        let (arr, rest) = name.split_at_checked(1)?;
        let rest = rest.strip_prefix('[')?;
        let close = rest.find(']')?;
        let digits = &rest[..close];
        if digits.is_empty() || (digits.len() > 1 && digits.starts_with('0')) || !digits.bytes().all(|c| c.is_ascii_digit()) {
            return None;
        }
        let i: usize = digits.parse().ok()?;
        match (arr, &rest[close + 1..]) {
            ("a", ".n") if i < K => Some(i),
            ("b", ".n") if i < K => Some(K + i),
            ("g", ".sm") if i < GROUPS => Some(2 * K + 3 * i),
            ("g", ".X") if i < GROUPS => Some(2 * K + 3 * i + 1),
            ("g", ".part") if i < GROUPS => Some(2 * K + 3 * i + 2),
            _ => None,
        }
    }
}

/// One relation the reader accepts: statement widths, the chain shape, `params.RELATIONS`' shape (the relation terms of
/// the soundness bound are those of the 4096-VU cell, as in `prove.soundness`), and the pinned system / key digests.
#[derive(Clone, Debug)]
pub struct RelSpec {
    pub name: &'static str,
    pub op: Operand,
    /// operands per unit (`rel.k`)
    pub k: usize,
    /// units per VU (`1536 / k`)
    pub steps: usize,
    pub word_bytes: u8,
    pub y_bytes: u8,
    pub y_bits: u32,
    pub quad: usize,
    pub lin: usize,
    /// `RelationShape.row_vars` (log2 rows_pad)
    pub row_vars: usize,
    /// the claimed word's chain-end components `(shift, width)` (`sys.chain["y_end"]`, rows `y_end0..`); empty: the word
    /// itself is the one end row `y16`
    pub ends: &'static [(u32, u32)],
    /// accepted `(protocol.system_id(sys), sha256(key.bin))` pairs; empty = not pinned yet (only `--allow-any-key`
    /// verifies it)
    pub pins: &'static [(&'static str, &'static str)],
}

impl RelSpec {
    /// `RelationShape.col_vars` / `has_radix3` at 4096 VUs (`cols = steps * 4096`) and `row_vars`.
    fn shape_vars(&self) -> (usize, bool, usize) {
        let cols = self.steps * 4096;
        let mut c = cols;
        while c % 3 == 0 {
            c /= 3;
        }
        ((usize::BITS - c.leading_zeros()) as usize - 1, cols % 3 == 0, self.row_vars)
    }
}

pub static RELATIONS: [RelSpec; 5] = [
    RelSpec {
        name: "fp8-ada",
        op: Operand::E4m3,
        k: 32,
        steps: 48,
        word_bytes: 1,
        y_bytes: 4,
        y_bits: 22,
        quad: 3532,
        lin: 589,
        row_vars: 12,
        ends: &[],
        pins: FP8_ADA_PINS,
    },
    RelSpec {
        name: "bf16-hopper",
        op: Operand::Bf16,
        k: 16,
        steps: 96,
        word_bytes: 2,
        y_bytes: 2,
        y_bits: 16,
        quad: 3164,
        lin: 339,
        row_vars: 12,
        ends: &[],
        pins: BF16_HOPPER_PINS,
    },
    RelSpec {
        name: "fp8-hopper",
        op: Operand::E4m3,
        k: 32,
        steps: 48,
        word_bytes: 1,
        y_bytes: 4,
        y_bits: 22,
        quad: 3164,
        lin: 554,
        row_vars: 12,
        ends: &[],
        pins: FP8_HOPPER_PINS,
    },
    RelSpec {
        name: "bf16-ampere",
        op: Operand::Bf16,
        k: 16,
        steps: 96,
        word_bytes: 2,
        y_bytes: 2,
        y_bits: 16,
        quad: 3387,
        lin: 347,
        row_vars: 12,
        ends: &[],
        pins: BF16_AMPERE_PINS,
    },
    // prove._FP4Relation: 24 chained NVFP4 units per VU, 68 byte words per operand, the claim an FP32 word bound as sign /
    // exponent field / fraction (`fp4/relation.py` y_end, ligero-verify's `compile_fp4_unit` ends)
    RelSpec {
        name: "fp4-nvf4",
        op: Operand::Nvf4,
        k: nvf4::WORDS,
        steps: 24,
        word_bytes: 1,
        y_bytes: 4,
        y_bits: 32,
        quad: 1389,
        lin: 306,
        row_vars: 11,
        ends: &[(31, 1), (23, 8), (0, 23)],
        pins: FP4_NVF4_PINS,
    },
];

// The system ids equal ligero-verify's `relation.rs` pins of the same systems (checked there against `system.bin`); the
// key digests are the LGVK0001 key.bin of ligerito-relation's gate dumps (the key is a function of the system alone).
/// fp8-ada: key of ligerito-relation 3592bd0 (18:55Z / 20:20Z fixtures), unchanged at ligerito-relation-2 32d3d42.
pub const FP8_ADA_PINS: &[(&str, &str)] =
    &[("6b570eef0c7faa07ccd1dddb2508e2699bd5367b108b15920469d46d16073d37", "da27387bb23ff60a60b4b21ef99f180c84c17456edf41efa31e5797aaed9227b")];
/// bf16-hopper: key of ligerito-relation-2 32d3d42's gate dump.
pub const BF16_HOPPER_PINS: &[(&str, &str)] =
    &[("9dcc7cdc1377a00d877440cc01a0dcdb73b3e989d86df8244eccb35a81be0df4", "1726c3be0bd163ebfba38cc8e9fcaed0d74b76f3b592db0748e6be254b946846")];
/// bf16-ampere: key of ligerito-relation-2 32d3d42's gate dump.
pub const BF16_AMPERE_PINS: &[(&str, &str)] =
    &[("44cb05b9e6988219beba2d66582124e67a1893aac39d0b66126ce0b9d4db89cb", "e0059e7009ba368dc1119abdc923dfcf6e06b81d4424e81106fdd01721a30eb6")];
/// fp4-nvf4: key of ligerito-relation-2 32d3d42's gate dump.
pub const FP4_NVF4_PINS: &[(&str, &str)] =
    &[("a825ba0b826a15366bfb8cdd720a5e9978603cbe589491abcbba7346ad35bf3c", "ac61d99e195150a4c23ccc9d3d5ff73cd72248b7236843ae5339a07c4708928e")];
/// fp8-hopper: key of ligerito-relation-2 32d3d42's gate dump.
pub const FP8_HOPPER_PINS: &[(&str, &str)] =
    &[("c7cbebe365cb5f773a679bc13b5382fb45714511b1912e4e4ab7fb48accd4bc3", "ea793f49d4c0015d835d886df3bab65949189f20a8f35222e653482dc7d65d60")];

pub fn relation_spec(name: &str) -> Option<&'static RelSpec> {
    RELATIONS.iter().find(|r| r.name == name)
}

// ------------------------------------------------------------------------------------------------------- parsing ----

fn read_str(r: &mut Reader, what: &str) -> Result<String, String> {
    let n = r.count(what, 1 << 12)?;
    String::from_utf8(r.bytes(n)?.to_vec()).map_err(|_| format!("{what}: not utf-8"))
}

fn read_blob<'a>(r: &mut Reader<'a>, what: &str, max: usize) -> Result<&'a [u8], String> {
    let n = r.count(what, max)?;
    r.bytes(n)
}

fn read_words(r: &mut Reader, n: usize, width: u8) -> Result<Vec<u32>, String> {
    let b = r.bytes(n * width as usize)?;
    Ok(match width {
        1 => b.iter().map(|&x| x as u32).collect(),
        2 => b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]]) as u32).collect(),
        4 => b.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect(),
        _ => return Err(format!("word width {width}")),
    })
}

#[derive(Clone, Debug)]
pub struct Statement {
    pub relation: String,
    pub sys_id: [u8; 32],
    pub vk_sha256: [u8; 32],
    pub l: usize,
    pub steps: usize,
    pub k: usize,
    pub n_vus: Vec<usize>,
    pub word_bytes: u8,
    pub y_bytes: u8,
    /// per real sub-batch: `l * K` words, unit (column) j's operand i at `j * K + i`
    pub a: Vec<Vec<u32>>,
    pub b: Vec<Vec<u32>>,
    /// per real sub-batch: `l` claimed words
    pub y: Vec<Vec<u32>>,
}

pub fn read_statement(buf: &[u8]) -> Result<Statement, String> {
    let mut r = Reader::new(buf);
    if r.bytes(8)? != MAGIC_STMT {
        return Err("not an LGTS0001 statement".into());
    }
    let relation = read_str(&mut r, "statement relation")?;
    let sys_id: [u8; 32] = r.digest()?;
    let vk_sha256: [u8; 32] = r.digest()?;
    let l = r.count("statement l", 1 << 24)?;
    let steps = r.count("statement steps", 1 << 24)?;
    let k = r.count("statement K", 1 << 12)?;
    let s = r.count("statement S", 1 << 16)?;
    let n_vus = (0..s).map(|_| r.count("statement n_vus", 1 << 24)).collect::<Result<Vec<_>, _>>()?;
    let wb = r.bytes(1)?[0];
    let yb = r.bytes(1)?[0];
    if ![1, 2, 4].contains(&wb) || ![1, 2, 4].contains(&yb) {
        return Err(format!("statement: word widths {wb} / {yb}"));
    }
    let (mut a, mut b, mut y) = (Vec::new(), Vec::new(), Vec::new());
    for &nv in &n_vus {
        if nv > 0 {
            if r.remaining() < l * k * 2 * wb as usize {
                return Err("statement: truncated sub-batch words".into());
            }
            a.push(read_words(&mut r, l * k, wb)?);
            b.push(read_words(&mut r, l * k, wb)?);
            y.push(read_words(&mut r, l, yb)?);
        }
    }
    r.finish().map_err(|e| format!("statement: {e}"))?;
    Ok(Statement { relation, sys_id, vk_sha256, l, steps, k, n_vus, word_bytes: wb, y_bytes: yb, a, b, y })
}

#[derive(Clone, Debug)]
pub struct Key {
    pub relation: String,
    pub sys_id: [u8; 32],
    pub m: usize,
    pub n_i: usize,
    pub n_k: usize,
    pub k: usize,
    pub c_rows: Vec<usize>,
    pub virt: Vec<(String, usize)>,
    pub coo: [Vec<(usize, usize, u32)>; 3],
}

pub fn read_key(buf: &[u8]) -> Result<Key, String> {
    let mut r = Reader::new(buf);
    if r.bytes(8)? != MAGIC_KEY {
        return Err("not an LGVK0001 key".into());
    }
    let relation = read_str(&mut r, "key relation")?;
    let sys_id: [u8; 32] = r.digest()?;
    let m = r.count("key m", 1 << 24)?;
    let n_i = r.count("key n_i", 30)?;
    let n_k = r.count("key n_k", 30)?;
    let k = r.count("key K", 1 << 30)?;
    let nl = r.count("key n_links", 1 << 12)?;
    let c_rows = (0..nl).map(|_| r.u32().map(|x| x as usize)).collect::<Result<Vec<_>, _>>()?;
    let nv = r.count("key n_virt", 1 << 16)?;
    let mut virt = Vec::with_capacity(nv);
    for _ in 0..nv {
        let name = read_str(&mut r, "virtual row name")?;
        let row = r.u32()? as usize;
        virt.push((name, row));
    }
    let mut coo: [Vec<(usize, usize, u32)>; 3] = Default::default();
    for side in coo.iter_mut() {
        let nnz = r.count("key nnz", 1 << 26)?;
        if r.remaining() < 12 * nnz {
            return Err("key: truncated constraint triples".into());
        }
        for _ in 0..nnz {
            let (kk, row, coef) = (r.u32()? as usize, r.u32()? as usize, r.u32()?);
            if coef >= P {
                return Err("key: non-canonical coefficient".into());
            }
            side.push((kk, row, coef));
        }
    }
    r.finish().map_err(|e| format!("key: {e}"))?;
    Ok(Key { relation, sys_id, m, n_i, n_k, k, c_rows, virt, coo })
}

/// The absorbed parameter JSON (`prove.Prover.batch`), canonical: compact, sorted keys, the known keys only.
#[derive(Clone, Debug, PartialEq)]
pub struct LgtoParams {
    pub coins: String,
    pub kprime: Vec<usize>,
    pub l: usize,
    pub n: usize,
    pub n_claims: usize,
    pub order: String,
    pub queries: Vec<usize>,
    pub rate_log2: Vec<usize>,
    pub relation: String,
    pub splits: Vec<(usize, usize)>,
    pub zk: bool,
    /// `None`: a proof from before red-team-2 V1 (no zero claims; rejected unless the caller allows legacy proofs)
    pub zero_blocks: Option<Vec<(usize, usize)>>,
    pub t_pad: usize,
}

fn pairs(v: &Json, what: &str) -> Result<Vec<(usize, usize)>, String> {
    v.as_arr()
        .ok_or_else(|| format!("params {what}: not a list"))?
        .iter()
        .map(|x| match x.usize_list().as_deref() {
            Some([a, b]) => Ok((*a, *b)),
            _ => Err(format!("params {what}: entries must be [int, int]")),
        })
        .collect()
}

pub fn parse_params(raw: &[u8]) -> Result<LgtoParams, String> {
    let text = std::str::from_utf8(raw).map_err(|_| "params: not utf-8".to_string())?;
    let doc = json::parse(text)?;
    let kv = match &doc {
        Json::Obj(kv) => kv,
        _ => return Err("params: not an object".into()),
    };
    const KEYS: [&str; 13] = ["coins", "kprime", "l", "n", "n_claims", "order", "queries", "rate_log2", "relation", "splits", "t_pad", "zero_blocks", "zk"];
    for w in kv.windows(2) {
        if w[0].0 >= w[1].0 {
            return Err("params: keys not sorted / repeated".into());
        }
    }
    if let Some((k, _)) = kv.iter().find(|(k, _)| !KEYS.contains(&k.as_str())) {
        return Err(format!("params: unknown key {k:?}"));
    }
    if doc.dump().as_bytes() != raw {
        return Err("params: not the canonical compact JSON".into());
    }
    let get = |k: &str| doc.get(k).ok_or_else(|| format!("params: missing {k:?}"));
    let int = |k: &str| get(k).and_then(|v| v.as_usize().ok_or_else(|| format!("params {k:?}: not an integer")));
    let list = |k: &str| get(k).and_then(|v| v.usize_list().ok_or_else(|| format!("params {k:?}: not an integer list")));
    let string = |k: &str| get(k).and_then(|v| v.as_str().map(str::to_string).ok_or_else(|| format!("params {k:?}: not a string")));
    let zk = get("zk")?.as_bool().ok_or("params \"zk\": not a boolean")?;
    let zero_blocks = match doc.get("zero_blocks") {
        Some(v) => Some(pairs(v, "zero_blocks")?),
        None => None,
    };
    let t_pad = match doc.get("t_pad") {
        Some(v) => v.as_usize().ok_or("params \"t_pad\": not an integer")?,
        None => 0,
    };
    Ok(LgtoParams {
        coins: string("coins")?,
        kprime: list("kprime")?,
        l: int("l")?,
        n: int("n")?,
        n_claims: int("n_claims")?,
        order: string("order")?,
        queries: list("queries")?,
        rate_log2: list("rate_log2")?,
        relation: string("relation")?,
        splits: pairs(get("splits")?, "splits")?,
        zk,
        zero_blocks,
        t_pad,
    })
}

#[derive(Clone, Debug)]
pub struct PcsProof {
    pub roots: Vec<Digest>,
    /// per round: k'_i messages (g(0), g(1), g(2))
    pub sumcheck: Vec<Vec<[Ext; 3]>>,
    /// per round: the opened rows (round 1 base entries; later rounds from the plane-major words)
    pub openings: Vec<Opening>,
    pub final_y: Vec<Ext>,
    /// ZK: per round `t_pad` padding coefficients; empty without
    pub ybar: Vec<Vec<Ext>>,
}

#[derive(Clone, Debug)]
pub struct Proof {
    pub params_raw: Vec<u8>,
    pub params: LgtoParams,
    pub pcs_params: Params,
    pub sumcheck: Vec<u8>,
    pub pcs: PcsProof,
}

pub fn read_proof(buf: &[u8]) -> Result<Proof, String> {
    let mut r = Reader::new(buf);
    if r.bytes(8)? != MAGIC_PROOF {
        return Err("not an LGTO0001 proof".into());
    }
    let params_raw = read_blob(&mut r, "params", 1 << 16)?.to_vec();
    let params = parse_params(&params_raw)?;
    let framing = json::parse(std::str::from_utf8(read_blob(&mut r, "framing", 1 << 16)?).map_err(|_| "framing: not utf-8")?)?;
    let sumcheck = read_blob(&mut r, "sumcheck", 1 << 28)?.to_vec();
    let l = params.kprime.len();
    let sib_len = framing.get("sib_len").and_then(|v| v.usize_list()).ok_or("framing: sib_len")?;
    let final_len = framing.get("final_len").and_then(|v| v.as_usize()).ok_or("framing: final_len")?;
    let fr_t_pad = match framing.get("t_pad") {
        Some(v) => v.as_usize().ok_or("framing: t_pad")?,
        None => 0,
    };
    if l == 0 || l > 16 || params.queries.len() != l || params.rate_log2.len() != l || params.splits.len() != l || sib_len.len() != l {
        return Err("proof: inconsistent round counts".into());
    }
    let mut pp = Params::with_rates(params.n, params.kprime.clone(), params.queries.clone(), params.rate_log2.clone());
    pp.splits = params.splits.clone();
    pp.validate()?;
    for i in 1..=l {
        if pp.splits[i - 1] != canonical_split(pp.m(i)) {
            return Err(format!("proof: non-canonical encoder split in round {i}"));
        }
    }
    if final_len != pp.final_len() {
        return Err("proof: final vector length".into());
    }
    if fr_t_pad != params.t_pad {
        return Err("proof: framing t_pad differs from the parameters'".into());
    }
    let roots = (0..l).map(|_| r.digest()).collect::<Result<Vec<_>, _>>()?;
    let mut sc = Vec::with_capacity(l);
    for &kp in &params.kprime {
        sc.push((0..kp).map(|_| Ok([r.ext()?, r.ext()?, r.ext()?])).collect::<Result<Vec<_>, String>>()?);
    }
    let mut openings = Vec::with_capacity(l);
    for i in 1..=l {
        let (cols, base) = (pp.cols(i), i == 1);
        let w = cols * if base { 1 } else { EXT_DEG };
        let q = pp.queries[i - 1];
        if r.remaining() < 4 * q * w {
            return Err(format!("proof: truncated rows of round {i}"));
        }
        let mut rows = Vec::with_capacity(q);
        for _ in 0..q {
            let words = (0..w).map(|_| r.base()).collect::<Result<Vec<_>, _>>()?;
            rows.push(row_from_words(&words, cols, base).unwrap());
        }
        if sib_len[i - 1] % 32 != 0 {
            return Err(format!("proof: round {i} sibling bytes not a multiple of 32"));
        }
        let siblings = (0..sib_len[i - 1] / 32).map(|_| r.digest()).collect::<Result<Vec<_>, _>>()?;
        openings.push(Opening { rows, siblings });
    }
    let final_y = r.exts(final_len)?;
    let ybar = if params.t_pad > 0 {
        if params.t_pad > 1 << 16 {
            return Err("proof: t_pad".into());
        }
        (0..l).map(|_| r.exts(params.t_pad)).collect::<Result<Vec<_>, _>>()?
    } else {
        Vec::new()
    };
    r.finish().map_err(|e| format!("proof: {e}"))?;
    Ok(Proof { params_raw, params, pcs_params: pp, sumcheck, pcs: PcsProof { roots, sumcheck: sc, openings, final_y, ybar } })
}

/// A `.coins` file: run.py's `u32 count | count x 32 B`, or this crate's repeated `(u32 len, bytes)`.
pub fn read_coins(raw: &[u8]) -> Result<Live, String> {
    if raw.len() >= 4 {
        let n = u32::from_le_bytes(raw[..4].try_into().unwrap()) as usize;
        if n.checked_mul(32).and_then(|x| x.checked_add(4)) == Some(raw.len()) {
            return Ok(Live::new(raw[4..].chunks_exact(32).map(|c| c.to_vec()).collect()));
        }
    }
    Live::parse(raw)
}

// --------------------------------------------------------------------------------------------------- public rows ----

/// The statement's public rows as multilinears in the column variables (`layout.public_rows`), column `c = s l + j`.
pub struct StmtRows<'a> {
    spec: &'a RelSpec,
    stmt: &'a Statement,
    /// sub-batches S (real ones first, then pads: zero words, y = 0, no VUs)
    s: usize,
    /// (eq_c fingerprint, the operand pins at r_c: `(side K + i) 3 + field` for word relations, `nvf4::pin_index`)
    cache: RefCell<Option<((usize, usize, [u32; EXT_DEG]), Vec<Ext>)>>,
}

impl<'a> StmtRows<'a> {
    pub fn new(spec: &'a RelSpec, stmt: &'a Statement, s: usize) -> StmtRows<'a> {
        StmtRows { spec, stmt, s, cache: RefCell::new(None) }
    }

    /// Every word decodes (NVFP4: by its position in the unit), every claimed word fits `y_bits` and is 0 off the chain ends
    /// (`prove._stmt_subs`, ligerito-relation-2 0db857a).
    pub fn validate(&self) -> Result<(), String> {
        let k = self.stmt.k;
        for (s, (a, b)) in self.stmt.a.iter().zip(&self.stmt.b).enumerate() {
            let mut bad: Vec<String> = Vec::new();
            for (side, words) in [("a", a), ("b", b)] {
                for (n, &w) in words.iter().enumerate() {
                    let r = match self.spec.op {
                        Operand::Nvf4 if n % k < nvf4::K => nvf4::numerator(w).map(|_| ()),
                        Operand::Nvf4 => nvf4::scale(w).map(|_| ()),
                        op => op.decode(w).map(|_| ()),
                    };
                    if let Err(e) = r {
                        let e = format!("{side}: {e}");
                        if !bad.contains(&e) {
                            bad.push(e);
                        }
                    }
                }
            }
            if !bad.is_empty() {
                return Err(format!("statement sub-batch {s}: public inputs rejected: {}", bad.join("; ")));
            }
            if self.stmt.y[s].iter().any(|&y| y as u64 >= 1u64 << self.spec.y_bits) {
                return Err(format!("statement sub-batch {s}: claimed word out of range"));
            }
            // the end constraint is masked off the chain ends, so a word there binds nothing: canonical statements carry 0
            let (steps, nv) = (self.stmt.steps, self.stmt.n_vus[s]);
            if self.stmt.y[s].iter().enumerate().any(|(j, &y)| y != 0 && !(j < steps * nv && j % steps == steps - 1)) {
                return Err(format!("statement sub-batch {s}: claimed word off a chain end (non-canonical)"));
            }
        }
        Ok(())
    }

    fn operand_rows(&self, eq_c: &[Ext]) -> Vec<Ext> {
        let (l, k, op) = (self.stmt.l, self.stmt.k, self.spec.op);
        let n_real = self.stmt.a.len();
        let mut out = Vec::with_capacity(6 * k);
        let mut hist = vec![Ext::ZERO; 1 << op.word_bits()];
        // the pad sub-batches' columns all hold word 0
        let pad_mass = eq_c[n_real * l..].iter().fold(Ext::ZERO, |acc, e| acc.add(e));
        for side in [&self.stmt.a, &self.stmt.b] {
            for i in 0..k {
                hist.iter_mut().for_each(|h| *h = Ext::ZERO);
                hist[0] = pad_mass;
                for (s, words) in side.iter().enumerate() {
                    for j in 0..l {
                        let w = words[j * k + i] as usize;
                        hist[w] = hist[w].add(&eq_c[s * l + j]);
                    }
                }
                let mut acc = [Ext::ZERO; 3];
                for (w, h) in hist.iter().enumerate() {
                    if h.is_zero() {
                        continue;
                    }
                    let (sg, m, e) = op.decode(w as u32).expect("validated statement");
                    acc[0] = acc[0].add(&h.mul_base(sg));
                    acc[1] = acc[1].add(&h.mul_base(m));
                    acc[2] = acc[2].add(&h.mul_base(signed_mod_p(e)));
                }
                out.extend(acc);
            }
        }
        out
    }

    /// The NVFP4 pins at r_c, unit by unit (a pin reads the whole unit, so no per-word histogram).
    fn nvf4_rows(&self, eq_c: &[Ext]) -> Vec<Ext> {
        let (l, k) = (self.stmt.l, self.stmt.k);
        debug_assert_eq!(k, nvf4::WORDS);
        let n_real = self.stmt.a.len();
        let mut acc = vec![Ext::ZERO; nvf4::PINS];
        let mut add = |e: &Ext, pins: &[i64; nvf4::PINS]| {
            for (a, &v) in acc.iter_mut().zip(pins) {
                if v != 0 {
                    *a = a.add(&e.mul_base(signed_mod_p(v)));
                }
            }
        };
        for s in 0..n_real {
            let (a, b) = (&self.stmt.a[s], &self.stmt.b[s]);
            for j in 0..l {
                add(&eq_c[s * l + j], &nvf4::unit_pins(&a[j * k..(j + 1) * k], &b[j * k..(j + 1) * k]).expect("validated statement"));
            }
        }
        // the pad sub-batches' units are all-zero words (`fp4/chain._pad_unit`), whose pins are not all zero
        let pad_mass = eq_c[n_real * l..].iter().fold(Ext::ZERO, |acc, e| acc.add(e));
        let zeros = [0u32; nvf4::WORDS];
        add(&pad_mass, &nvf4::unit_pins(&zeros, &zeros).unwrap());
        acc
    }

    fn per_column(&self, eq_c: &[Ext], f: impl Fn(usize, usize) -> u32) -> Ext {
        let l = self.stmt.l;
        let mut acc = Ext::ZERO;
        for s in 0..self.s {
            for j in 0..l {
                let v = f(s, j);
                if v != 0 {
                    acc = acc.add(&eq_c[s * l + j].mul_base(v));
                }
            }
        }
        acc
    }
}

/// `"12"` -> 12 (decimal, no sign, no leading zero)
fn canonical_index(digits: &str) -> Option<usize> {
    if digits.is_empty() || (digits.len() > 1 && digits.starts_with('0')) || !digits.bytes().all(|c| c.is_ascii_digit()) {
        return None;
    }
    digits.parse().ok()
}

/// `pub:a[12].m` -> (0, 12, 1)
fn parse_pub(name: &str, k: usize) -> Option<(usize, usize, usize)> {
    let rest = name.strip_prefix("pub:")?;
    let side = match rest.as_bytes().first()? {
        b'a' => 0,
        b'b' => 1,
        _ => return None,
    };
    let rest = rest[1..].strip_prefix('[')?;
    let close = rest.find(']')?;
    let digits = &rest[..close];
    if digits.is_empty() || (digits.len() > 1 && digits.starts_with('0')) || !digits.bytes().all(|c| c.is_ascii_digit()) {
        return None;
    }
    let i: usize = digits.parse().ok()?;
    let field = match &rest[close + 1..] {
        ".s" => 0,
        ".m" => 1,
        ".e" => 2,
        _ => return None,
    };
    if i >= k {
        return None;
    }
    Some((side, i, field))
}

impl PublicRows for StmtRows<'_> {
    fn mle(&self, name: &str, eq_c: &[Ext]) -> Result<Ext, String> {
        let (l, steps) = (self.stmt.l, self.stmt.steps);
        if eq_c.len() != self.s * l {
            return Err(format!("public rows: {} columns, S l = {}", eq_c.len(), self.s * l));
        }
        let nv = |s: usize| self.stmt.n_vus.get(s).copied().unwrap_or(0);
        let unknown = || format!("public row {name}: not a row this verifier derives");
        match name {
            "const" => Ok(eq_c.iter().fold(Ext::ZERO, |acc, e| acc.add(e))),
            // ligero/chain.py ChainLayout.masks
            "start" => Ok(self.per_column(eq_c, |s, j| (j >= steps * nv(s) || j % steps == 0) as u32)),
            "link" => Ok(self.per_column(eq_c, |s, j| (j < steps * nv(s) && j % steps != steps - 1) as u32)),
            "end" => Ok(self.per_column(eq_c, |s, j| (j < steps * nv(s) && j % steps == steps - 1) as u32)),
            "y16" if self.spec.ends.is_empty() => Ok(self.per_column(eq_c, |s, j| self.stmt.y.get(s).map_or(0, |y| y[j]))),
            _ if name.starts_with("y_end") => {
                let (sh, wd) = canonical_index(&name[5..]).and_then(|e| self.spec.ends.get(e).copied()).ok_or_else(unknown)?;
                let mask = ((1u64 << wd) - 1) as u32;
                Ok(self.per_column(eq_c, |s, j| self.stmt.y.get(s).map_or(0, |y| (y[j] >> sh) & mask)))
            }
            _ => {
                let idx = match self.spec.op {
                    Operand::Nvf4 => name.strip_prefix("pub:").and_then(nvf4::pin_index),
                    _ => parse_pub(name, self.stmt.k).map(|(side, i, field)| (side * self.stmt.k + i) * 3 + field),
                }
                .ok_or_else(unknown)?;
                let fp = (eq_c.len(), eq_c.as_ptr() as usize, eq_c[eq_c.len() - 1].0);
                let mut cache = self.cache.borrow_mut();
                if cache.as_ref().map(|(f, _)| *f != fp).unwrap_or(true) {
                    let rows = match self.spec.op {
                        Operand::Nvf4 => self.nvf4_rows(eq_c),
                        _ => self.operand_rows(eq_c),
                    };
                    *cache = Some((fp, rows));
                }
                Ok(cache.as_ref().unwrap().1[idx])
            }
        }
    }
}

// ----------------------------------------------------------------------------------------------------- soundness ----

/// `proof.FS_QUERIES_LOG2`: the hash queries a Fiat-Shamir prover is granted (red-team F12's honest label).
pub const FS_QUERIES_LOG2: f64 = 64.0;

/// `prove.Prover.soundness`: `LigeritoParams.soundness` (paper (17) per level, the rate `(2^{k_i} + t_pad) / n_i`, and
/// the relation's zero-check / (i, i') / batching terms) + the claim batching over J + the zero claims.  Returns the
/// interactive union (terms summed as Python's `_log2_sum`) and the largest single term.
pub fn soundness_log2(p: &Params, t_pad: usize, spec: &RelSpec, n_i: usize, n_links: usize, blocks: &[(usize, usize)]) -> Result<(f64, f64), String> {
    let f = ext_log2();
    let mut terms: Vec<f64> = Vec::new();
    for i in 1..=p.rounds() {
        let (s, kp, n, tall) = (p.queries[i - 1] as f64, p.kprime[i - 1] as f64, p.m(i) as f64, p.rows(i) as f64);
        if p.queries[i - 1] < 1 || p.queries[i - 1] > p.m(i) || p.m(i) <= p.rows(i) + t_pad || p.m(i) > 1 << 27 {
            return Err(format!("params: level {i} shape (|S|, rate) out of range"));
        }
        let rate = (tall + t_pad as f64) / n;
        terms.push(s * ((1.0 + rate) / 2.0).log2());
        terms.push((kp * n).log2() - f);
        terms.push((2.0 * kp).log2() - f);
        if i >= 2 {
            terms.push((p.queries[i - 2] as f64 + 1.0).log2() - f);
        }
    }
    let (col_vars, radix3, row_vars) = spec.shape_vars();
    let nvars = (col_vars + radix3 as usize) as f64;
    let chain = 3.0;
    terms.push((spec.quad as f64 + spec.lin as f64 + 3.0 * chain + 4.0).log2() - f);
    terms.push(nvars.log2() - f);
    terms.push((3.0 * nvars + 7.0 + 4.0).log2() - f);
    terms.push((4.0 * row_vars as f64).log2() - f);
    terms.push((chain + 8.0).log2() - f);
    terms.push(((1 + n_links + blocks.len()) as f64).log2() - f);
    if !blocks.is_empty() {
        let s: usize = blocks.iter().map(|(b, _)| b + p.n - n_i).sum();
        terms.push((s as f64).log2() - f);
    }
    let m = terms.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    Ok((m + terms.iter().map(|x| (x - m).exp2()).sum::<f64>().log2(), m))
}

/// `proof.soundness_claim` (red-team F12): what an accepted proof establishes, by coin provenance.  Fiat-Shamir: `Q` times
/// the largest round-by-round term (every coin, the query sets included, is a hash output).  Local seeds are known to the
/// prover; live coins read from a file are not authenticated by the issuing verifier (this verifier does not re-derive
/// them from the stream's commitments), so neither establishes anything.
pub fn soundness_claim(coins: &str, max_term_log2: f64) -> (Option<f64>, String) {
    match coins {
        "fiat-shamir" => (
            Some(FS_QUERIES_LOG2 + max_term_log2),
            format!("Fiat-Shamir: Q x the largest round-by-round term, Q = 2^{FS_QUERIES_LOG2}"),
        ),
        "local" => (None, "none: local-seed coins are known to the prover before it commits (diagnostic)".into()),
        "live" => (None, "none: live coin slots replayed from a file, not authenticated by the issuing verifier".into()),
        _ => (None, format!("none: coins {coins:?}")),
    }
}

// ------------------------------------------------------------------------------------------------------------ PCS ----

/// `sum_t eta^{R + t} ybar[t]`: the ZK padding's share of `< U[s, :], rbar >`.
fn pad_correction(eta: u32, r: usize, ybar: &[Ext]) -> Ext {
    let mut acc = Ext::ZERO;
    let mut e = pow(eta, r as u64);
    for yb in ybar {
        acc = acc.add(&yb.mul_base(e));
        e = mul(e, eta);
    }
    acc
}

/// `prove.pcs_verify` (t_pad = 0) / `proto/pcs.verify_open` commit-first with the column padding: the J claims
/// `w~(points[j]) = values[j]` (PCS variable order) against root_1, which the caller has absorbed.  Returns the number of
/// opened rows.
/// A sparse PCS claim `sum_t weights[t] f[cells[t]] = value` on distinct flat PCS indices (ligerito-proto
/// `open(..., sparse=)`; LGSC0004's Libra mask claims).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SparseTerm {
    pub cells: Vec<usize>,
    pub weights: Vec<Ext>,
    pub value: Ext,
}

pub fn pcs_verify(p: &Params, pf: &PcsProof, points: &[Vec<Ext>], values: &[Ext], sparse: &[SparseTerm], t_pad: usize, coins: &mut dyn Coins) -> Result<usize, String> {
    let l = p.rounds();
    if points.is_empty() || points.len() != values.len() || points.iter().any(|z| z.len() != p.n) {
        return Err("PCS: bad claim shape".into());
    }
    for s in sparse {
        let mut seen = std::collections::HashSet::new();
        if s.cells.is_empty() || s.cells.len() != s.weights.len() || s.cells.iter().any(|&c| c >> p.n != 0 || !seen.insert(c)) {
            return Err("PCS: sparse term: cells must be distinct indices in [0, N), one weight each".into());
        }
    }
    if pf.roots.len() != l || pf.openings.len() != l || pf.sumcheck.len() != l || pf.final_y.len() != p.final_len() {
        return Err("PCS: bad proof shape".into());
    }
    if pf.ybar.len() != if t_pad > 0 { l } else { 0 } || pf.ybar.iter().any(|y| y.len() != t_pad) {
        return Err(format!("PCS: expected ZK padding t_pad = {t_pad}"));
    }
    for i in 1..=l {
        if p.rows(i) + t_pad > p.m(i) {
            return Err("PCS: padded message exceeds the codeword".into());
        }
    }
    let mut zb = Vec::with_capacity(points.len() * p.n * 24);
    for z in points {
        zb.extend(ext_bytes(z));
    }
    coins.absorb(b"z", &zb);
    for s in sparse {
        let mut b: Vec<u8> = s.cells.iter().flat_map(|&c| (c as u32).to_le_bytes()).collect();
        b.extend(ext_bytes(&s.weights));
        coins.absorb(b"sparse", &b);
    }
    let m = points.len();
    let beta = coins.challenge(b"beta", m + sparse.len())?;
    let mut claim = Ext::ZERO;
    let mut terms: Vec<Term> = Vec::with_capacity(m);
    for (j, z) in points.iter().enumerate() {
        claim = claim.add(&beta[j].mul(&values[j]));
        terms.push(Term::eq(beta[j], z.clone()));
    }
    // the sparse terms, recursed as ligerito-proto's `_Sparse`: after a round's column challenges r,
    // w_t *= eq(bits(cell_t >> k), r) and cell_t keeps its row
    let (mut sp_idx, mut sp_w): (Vec<usize>, Vec<Ext>) = (Vec::new(), Vec::new());
    for (k, s) in sparse.iter().enumerate() {
        claim = claim.add(&beta[m + k].mul(&s.value));
        sp_idx.extend_from_slice(&s.cells);
        sp_w.extend(s.weights.iter().map(|w| w.mul(&beta[m + k])));
    }
    let mut r_prev: Vec<Ext> = Vec::new();
    let mut opened = 0usize;
    for i in 1..=l {
        if i >= 2 {
            let prev = i - 1;
            coins.absorb(b"root", &pf.roots[i - 1]);
            if t_pad > 0 {
                coins.absorb(b"ybar", &ext_bytes(&pf.ybar[prev - 1]));
            }
            let s_set = coins.indices(b"S", p.m(prev), p.queries[prev - 1])?;
            let mut symbols = check_opening(p, prev, &pf.openings[prev - 1], &s_set, &pf.roots[prev - 1], &r_prev, Some(coins))?;
            if t_pad > 0 {
                for (j, &s) in s_set.iter().enumerate() {
                    symbols[j] = symbols[j].sub(&pad_correction(p.domain_point(prev, s), p.rows(prev), &pf.ybar[prev - 1]));
                }
            }
            opened += s_set.len();
            let alpha = coins.challenge(b"alpha", 1 + s_set.len())?;
            for t in terms.iter_mut() {
                t.coef = t.coef.mul(&alpha[0]);
            }
            for w in sp_w.iter_mut() {
                *w = w.mul(&alpha[0]);
            }
            claim = claim.mul(&alpha[0]);
            for (j, &s) in s_set.iter().enumerate() {
                terms.push(Term::geom(alpha[1 + j], p.domain_point(prev, s), p.k(prev)));
                claim = claim.add(&alpha[1 + j].mul(&symbols[j]));
            }
        }
        let msgs = &pf.sumcheck[i - 1];
        if msgs.len() != p.kprime[i - 1] {
            return Err(format!("PCS round {i}: {} sumcheck messages, k' = {}", msgs.len(), p.kprime[i - 1]));
        }
        let mut draws = Vec::with_capacity(msgs.len());
        for (j, g) in msgs.iter().enumerate() {
            if g[0].add(&g[1]) != claim {
                return Err(format!("PCS sumcheck round {i} failed (message {j})"));
            }
            coins.absorb(b"g", &ext_bytes(g));
            let r = coins.challenge(b"r", 1)?[0];
            claim = crate::claims::interpolate_deg2(g, &r);
            draws.push(r);
        }
        draws.reverse();
        let k = p.k(i);
        for t in terms.iter_mut() {
            *t = t.restrict(k, &draws);
        }
        if !sp_idx.is_empty() {
            let rbar = crate::claims::eq_vector(&draws);
            for (x, w) in sp_idx.iter_mut().zip(sp_w.iter_mut()) {
                *w = w.mul(&rbar[*x >> k]);
                *x &= (1 << k) - 1;
            }
        }
        r_prev = draws;
    }
    coins.absorb(b"y", &ext_bytes(&pf.final_y));
    if t_pad > 0 {
        coins.absorb(b"ybar", &ext_bytes(&pf.ybar[l - 1]));
    }
    let s_set = coins.indices(b"S", p.m(l), p.queries[l - 1])?;
    let symbols = check_opening(p, l, &pf.openings[l - 1], &s_set, &pf.roots[l - 1], &r_prev, None)?;
    opened += s_set.len();
    for (j, &s) in s_set.iter().enumerate() {
        let eta = p.domain_point(l, s);
        let mut want = dot_ext_base(&pf.final_y, &powers(eta, pf.final_y.len()));
        if t_pad > 0 {
            want = want.add(&pad_correction(eta, p.rows(l), &pf.ybar[l - 1]));
        }
        if want != symbols[j] {
            return Err("PCS: final vector is inconsistent with the opened rows".into());
        }
    }
    let total = terms.iter().fold(Ext::ZERO, |acc, t| acc.add(&t.dot(&pf.final_y)));
    let total = sp_idx.iter().zip(&sp_w).fold(total, |acc, (&x, w)| acc.add(&w.mul(&pf.final_y[x])));
    if total != claim {
        return Err("PCS: final claim failed".into());
    }
    Ok(opened)
}

// --------------------------------------------------------------------------------------------------------- verify ----

#[derive(Clone, Debug, Default)]
pub struct Options {
    /// accept a statement / key whose system id and key digest are not the pinned ones (development; the key is then
    /// only as trustworthy as its file)
    pub allow_any_key: bool,
    /// accept proofs from before red-team-2 V1 (no zero claims): UNSOUND, for reading old fixtures only
    pub allow_legacy: bool,
    /// the per-proof soundness target (log2, e.g. -128); `None` waives it
    pub target_log2: Option<f64>,
}

#[derive(Clone, Debug, Default)]
pub struct Report {
    pub relation: String,
    pub coins_kind: String,
    pub coins: usize,
    /// the interactive union of the proof's parameters (the parameter target applies to it)
    pub soundness_log2: f64,
    /// what the run establishes given its coins (`soundness_claim`); `None`: nothing
    pub claimed_log2: Option<f64>,
    pub claim_basis: String,
    pub sumcheck: String,
    pub zk: bool,
    pub vus: usize,
    pub sub_batches: usize,
    pub n: usize,
    pub n_claims: usize,
    pub zero_blocks: usize,
    pub opened_rows: usize,
    pub legacy: bool,
    /// the system id and key digest are a pinned pair (false only under `allow_any_key`)
    pub key_pinned: bool,
    pub key_sha256: String,
    pub stmt_sha256: String,
}

/// The layout of the batch from the key and the statement (`layout.layout_for`), with the key's own consistency checks.
pub fn layout(key: &Key, stmt: &Statement) -> Result<Layout, String> {
    let real: Vec<usize> = stmt.n_vus.iter().copied().take_while(|&nv| nv > 0).collect();
    if real.is_empty() || stmt.n_vus[real.len()..].iter().any(|&nv| nv != 0) || real.iter().any(|&nv| nv * stmt.steps > stmt.l) {
        return Err("statement sub-batch list".into());
    }
    if !stmt.l.is_power_of_two() {
        return Err("statement: l is not a power of two".into());
    }
    if stmt.n_vus.len() != real.len().next_power_of_two() {
        return Err("statement S does not match its sub-batches".into());
    }
    let n_s = stmt.n_vus.len().trailing_zeros() as usize;
    let n_j = stmt.l.trailing_zeros() as usize;
    let rows = 1usize << key.n_i;
    let mut seen = std::collections::HashSet::new();
    for (name, r) in &key.virt {
        if !seen.insert(name.as_str()) {
            return Err(format!("key: virtual row {name} named twice"));
        }
        if *r >= rows {
            return Err(format!("key: virtual row {name} at {r} outside 2^n_i"));
        }
    }
    let mut vrows: Vec<usize> = key.virt.iter().map(|(_, r)| *r).collect();
    vrows.sort_unstable();
    vrows.dedup();
    if vrows.len() != key.virt.len() {
        return Err("key: two virtual rows share a row".into());
    }
    if key.k == 0 || key.k > 1 << key.n_k {
        return Err("key: K exceeds 2^n_k".into());
    }
    let lay = Layout { n_i: key.n_i, n_k: key.n_k, n_s, n_j, m: key.m, virt: key.virt.clone(), c_rows: key.c_rows.clone() };
    if is_zk_layout(&lay) {
        lay.validate_rows()?;
    } else {
        lay.validate()?;
    }
    Ok(lay)
}

/// An LGSC0004 layout (`layout_for(zk=True)`): the linked rows' shifts are committed (`lgsc4::ZkRows`), so the key names
/// no `next:<x>` virtual row.
pub fn is_zk_layout(lay: &Layout) -> bool {
    lay.n_links() > 0 && lay.virt.iter().all(|(n, _)| !n.starts_with("next:"))
}

/// `zk.f_index`: the PCS cell of relation cell `(i, c)` under the ZK interleaving (`permute_point`'s variable order).
pub fn f_index(i: usize, c: usize, n_i: usize, a: usize) -> usize {
    (c & ((1 << a) - 1)) + (i << a) + ((c >> a) << (a + n_i))
}

/// Verify one LGTO0001 proof.  `coins_for(kind)`: the verifier's coins for the proof's declared `"coins"`
/// (`fiat-shamir` / `local` / `live`); the caller decides which kinds it trusts.
pub fn verify(
    stmt_bytes: &[u8],
    key_bytes: &[u8],
    proof_bytes: &[u8],
    coins_for: &mut dyn FnMut(&str) -> Result<Box<dyn Coins>, String>,
    opt: &Options,
) -> Result<Report, String> {
    let key = read_key(key_bytes)?;
    let stmt = read_statement(stmt_bytes)?;
    let proof = read_proof(proof_bytes)?;
    let prm = &proof.params;
    let spec = relation_spec(&stmt.relation).ok_or_else(|| format!("relation {:?}: not one this verifier knows", stmt.relation))?;
    if key.relation != stmt.relation || prm.relation != stmt.relation {
        return Err("relation mismatch".into());
    }
    let key_sha = sha256(key_bytes);
    if stmt.sys_id != key.sys_id {
        return Err("statement names another system".into());
    }
    if stmt.vk_sha256 != key_sha {
        return Err("statement names another verification key".into());
    }
    let (sid, vk) = (hex(&stmt.sys_id), hex(&key_sha));
    let key_pinned = spec.pins.iter().any(|(s, k)| *s == sid && *k == vk);
    if !key_pinned && !opt.allow_any_key {
        return Err(if spec.pins.is_empty() {
            format!("no pinned system / key for {} yet (--allow-any-key to verify against the file's key)", spec.name)
        } else {
            format!("system {sid} / key {vk} is not a pinned {} system and key (--allow-any-key to verify against it)", spec.name)
        });
    }
    if stmt.steps != spec.steps || stmt.k != spec.k || stmt.word_bytes != spec.word_bytes || stmt.y_bytes != spec.y_bytes {
        return Err("statement shape".into());
    }
    if prm.l != stmt.l {
        return Err("statement shape (params l)".into());
    }
    let lay = layout(&key, &stmt)?;
    let cons = Constraints { k_pad: 1 << key.n_k, a: key.coo[0].clone(), b: key.coo[1].clone(), c: key.coo[2].clone() };
    cons.validate(&lay)?;
    let p = &proof.pcs_params;
    let zk = prm.zk;
    let n_virt = lay.virt.len();
    // LGSC0004 (ligerito-sumcheck-3's ZK sumcheck): the verifier derives the ZK rows itself; ZK padding is mandatory
    let zk_rows = if is_zk_layout(&lay) { Some(crate::lgsc4::zk_rows_for(&lay, key.k)?) } else { None };
    if zk_rows.is_some() && !zk {
        return Err("a ZK layout (committed next rows, LGSC0004) needs the ZK PCS".into());
    }
    if zk {
        // prove.zk_layout: the mask rows sit in [m + n_virt, R); RowMaskLayout: 0 <= a = k_L - b <= n_c
        if lay.rows() < lay.m + n_virt + (1 << ZK_MASK_B) + 4 {
            return Err("ZK: not enough free rows for the mask rows".into());
        }
        let k_l = p.k(p.rounds());
        if k_l < ZK_MASK_B || k_l - ZK_MASK_B > lay.n_c() {
            return Err("ZK: k_L - b column bits cannot be interleaved".into());
        }
    }
    let blocks: Vec<(usize, usize)> = match &prm.zero_blocks {
        Some(zb) => {
            let want = zero_blocks(&lay);
            if *zb != want {
                return Err("proof parameters do not match the layout (zero_blocks)".into());
            }
            want
        }
        None if opt.allow_legacy => Vec::new(),
        None => return Err("proof has no zero claims on the virtual rows (pre-V1 format; unsound)".into()),
    };
    let rows_of_zero: usize = blocks.iter().map(|(b, _)| 1usize << b).sum();
    debug_assert!(blocks.is_empty() || rows_of_zero == n_virt);
    if p.n != lay.n_i + lay.n_c()
        || prm.order != if zk { "zk-interleaved" } else { "row-major" }
        || prm.n_claims != if zk_rows.is_some() { 2 } else { 1 + lay.n_links() } + blocks.len()
    {
        return Err("proof parameters do not match the layout".into());
    }
    if zk != (prm.t_pad > 0) || (prm.t_pad > 0 && prm.t_pad != T_PAD) {
        return Err("ZK padding does not match the parameters".into());
    }
    let (snd, max_term) = soundness_log2(p, prm.t_pad, spec, key.n_i, key.c_rows.len(), &blocks)?;
    if let Some(t) = opt.target_log2 {
        if snd > t + 1e-9 {
            return Err(format!("parameters give an interactive union of 2^{snd:.2} > 2^{t}"));
        }
    }
    let (claimed_log2, claim_basis) = soundness_claim(&prm.coins, max_term);
    let rows = StmtRows::new(spec, &stmt, stmt.n_vus.len());
    rows.validate()?;
    let mut coins = coins_for(&prm.coins)?;
    let coins = coins.as_mut();
    coins.absorb(b"lgto/params", &proof.params_raw);
    coins.absorb(b"lgto/stmt", &sha256(stmt_bytes));
    coins.absorb(b"lgto/vk", &key_sha);
    coins.absorb(b"root", &proof.pcs.roots[0]);
    let magic = proof.sumcheck.get(..8).ok_or("sumcheck bytes: truncated")?;
    let a = p.k(p.rounds()).saturating_sub(ZK_MASK_B);
    let mut sparse: Vec<SparseTerm> = Vec::new();
    let (sc_name, claims): (&str, Vec<EvalClaim>) = if let Some(zkr) = &zk_rows {
        // provisional until ligerito-relation-2 emits LGSC0004: eval claims w(r_i, r_c), w(rho_i, rho_c), then the zero
        // claims; the three Libra claims as sparse PCS terms on f_index(row, col), in sumcheck order
        if magic != crate::lgsc4::MAGIC {
            return Err("sumcheck bytes: a ZK layout (committed next rows) needs LGSC0004".into());
        }
        let sp = crate::lgsc4::parse(&proof.sumcheck).map_err(|e| format!("sumcheck bytes: {e}"))?;
        let c4 = crate::lgsc4::verify(&lay, zkr, &cons, &rows, &sp, coins)?;
        sparse = c4
            .sparse
            .iter()
            .map(|s| SparseTerm { cells: s.rows.iter().zip(&s.cols).map(|(&i, &c)| f_index(i, c, lay.n_i, a)).collect(), weights: s.weights.clone(), value: s.value })
            .collect();
        ("LGSC0004", c4.eval)
    } else if magic == crate::lgsc3::MAGIC {
        let sp = crate::lgsc3::parse(&proof.sumcheck).map_err(|e| format!("sumcheck bytes: {e}"))?;
        ("LGSC0003", crate::lgsc3::verify(&lay, &cons, &rows, &sp, coins)?)
    } else if magic == relation::MAGIC_LGSC {
        let sp = relation::parse_lgsc(&proof.sumcheck).map_err(|e| format!("sumcheck bytes: {e}"))?;
        ("LGSC0002", relation::verify(&lay, &cons, &rows, &sp, coins)?)
    } else {
        if magic == crate::lgsc4::MAGIC {
            return Err("sumcheck bytes: LGSC0004 needs a ZK layout (a key without next:* virtual rows)".into());
        }
        return Err(format!("sumcheck bytes: magic {:?} not implemented (LGSC0002 / LGSC0003)", String::from_utf8_lossy(magic)));
    };
    let mut all = claims.clone();
    all.extend(zero_claims(&lay, &claims[0], &blocks));
    let n_i = lay.n_i;
    let points: Vec<Vec<Ext>> = all
        .iter()
        .map(|c| {
            let (pi, pc) = c.point.split_at(n_i);
            if zk {
                // zk.permute_point: (p_c[:a] || p_i || p_c[a:])
                [&pc[..a], pi, &pc[a..]].concat()
            } else {
                [pc, pi].concat()
            }
        })
        .collect();
    let values: Vec<Ext> = all.iter().map(|c| c.value).collect();
    let opened = pcs_verify(p, &proof.pcs, &points, &values, &sparse, prm.t_pad, coins)?;
    if coins.unused() != 0 {
        return Err(format!("coins: {} unused verifier coins after the protocol ended", coins.unused()));
    }
    Ok(Report {
        relation: spec.name.into(),
        coins_kind: prm.coins.clone(),
        coins: coins.rounds(),
        soundness_log2: snd,
        claimed_log2,
        claim_basis,
        sumcheck: sc_name.into(),
        zk,
        vus: stmt.n_vus.iter().sum(),
        sub_batches: stmt.n_vus.len(),
        n: p.n,
        n_claims: all.len(),
        zero_blocks: blocks.len(),
        opened_rows: opened,
        legacy: prm.zero_blocks.is_none(),
        key_pinned,
        key_sha256: vk,
        stmt_sha256: hex(&sha256(stmt_bytes)),
    })
}

/// The V1 zero blocks `(b, pfx)`: the aligned cover of every run of consecutive virtual rows, in row order (both modes;
/// `proof.zero_blocks`, `relation::virtual_zero_claims`).
pub fn zero_blocks(lay: &Layout) -> Vec<(usize, usize)> {
    let mut rows: Vec<usize> = lay.virt.iter().map(|(_, r)| *r).collect();
    rows.sort_unstable();
    rows.dedup();
    let mut out = Vec::new();
    let mut k = 0;
    while k < rows.len() {
        let lo = rows[k];
        while k + 1 < rows.len() && rows[k + 1] == rows[k] + 1 {
            k += 1;
        }
        out.extend(aligned_cover(lo, rows[k] + 1));
        k += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::claims::eq_vector;
    use crate::refprover::random_point;
    use crate::transcript::FiatShamir;

    /// ligerito-relation 3592bd0's 18:55Z gate dump (fp8-ada, 12 VUs, l = 256, Fiat-Shamir): pre-V1, no zero claims.
    const KEY: &[u8] = include_bytes!("../fixtures/lgto/fp8ada_l256_legacy/key.bin");
    const STMT: &[u8] = include_bytes!("../fixtures/lgto/fp8ada_l256_legacy/honest_fiat-shamir.stmt");
    const PROOF: &[u8] = include_bytes!("../fixtures/lgto/fp8ada_l256_legacy/honest_fiat-shamir.lgto");

    fn fs(kind: &str) -> Result<Box<dyn Coins>, String> {
        match kind {
            "fiat-shamir" => Ok(Box::new(FiatShamir::new(FiatShamir::LABEL))),
            _ => Err(format!("{kind}: test replays Fiat-Shamir only")),
        }
    }

    fn legacy() -> Options {
        Options { allow_legacy: true, target_log2: Some(-128.0), ..Default::default() }
    }

    /// `f_index` and the ZK point permutation describe the same interleaving: `f~(permute(p)) = z~(p)` for `f[f_index(i, c)] =
    /// z[i][c]` (sumcheck order: row bits, then column bits), and a sparse claim on `(i, c)` cells equals it on `f_index` cells.
    #[test]
    fn f_index_matches_permute_point() {
        use crate::refprover::random_vector;
        let (n_i, n_c, a) = (3usize, 4usize, 2usize);
        let z = random_vector(n_i + n_c, 3);
        let zi = |i: usize, c: usize| z[i + (c << n_i)];
        let mut f = vec![0u32; z.len()];
        for i in 0..1 << n_i {
            for c in 0..1 << n_c {
                f[f_index(i, c, n_i, a)] = zi(i, c);
            }
        }
        let pt = random_point(n_i + n_c, 9);
        let (pi, pc) = pt.split_at(n_i);
        let perm = [&pc[..a], pi, &pc[a..]].concat();
        assert_eq!(dot_ext_base(&eq_vector(&perm), &f), dot_ext_base(&eq_vector(&pt), &z));
        let cells: Vec<(usize, usize)> = vec![(1, 5), (7, 0), (2, 15), (0, 9)];
        let w = random_point(cells.len(), 4);
        let on_z = cells.iter().zip(&w).fold(Ext::ZERO, |acc, (&(i, c), w)| acc.add(&w.mul_base(zi(i, c))));
        let on_f = cells.iter().zip(&w).fold(Ext::ZERO, |acc, (&(i, c), w)| acc.add(&w.mul_base(f[f_index(i, c, n_i, a)])));
        assert_eq!(on_z, on_f);
    }

    /// Sparse PCS claims (LGSC0004's Libra mask claims) through `pcs_verify`, against `refprover::prove_commit_first`
    /// (ligerito-proto's `open` / `_Sparse` in Rust; the shape of its `test_zk_padding_and_sparse_claims`, no padding): three
    /// evaluation points plus a sparse term inside one round-1 column (the Libra block shape) and one spread over columns.
    /// Rejected: that test's negatives (a wrong value / weight, the cells reordered, a term dropped), malformed terms.
    #[test]
    fn pcs_sparse_claims() {
        use crate::refprover::{prove_commit_first, random_vector};
        fn bind(c: &mut dyn Coins, root: &Digest) {
            c.absorb(b"statement", b"relation statement");
            c.absorb(b"root", root);
        }
        let p = Params::with_rates(12, vec![4, 3, 3], vec![12, 12, 12], vec![2, 2, 2]);
        let f = random_vector(12, 5);
        let points: Vec<Vec<Ext>> = (0..3).map(|j| random_point(12, 10 + j)).collect();
        let r1 = 1usize << p.k(1);
        let col3: Vec<usize> = (0..40).map(|t| 3 * r1 + (t * 37 + 11) % r1).collect();
        let spread: Vec<usize> = (0..25).map(|t| (t * 2731 + 5) % (1 << 12)).collect();
        let sparse = vec![(col3, random_point(40, 1)), (spread, random_point(25, 2))];
        let (values, svals, pf) = prove_commit_first(&p, &f, &points, &sparse, &mut FiatShamir::new(b"sparse-test"), &mut bind);
        let good: Vec<SparseTerm> = sparse.iter().zip(&svals).map(|((c, w), v)| SparseTerm { cells: c.clone(), weights: w.clone(), value: *v }).collect();
        let run = |sp: &[SparseTerm], vals: &[Ext]| {
            let mut c = FiatShamir::new(b"sparse-test");
            bind(&mut c, &pf.roots[0]);
            pcs_verify(&p, &pf, &points, vals, sp, 0, &mut c)
        };
        run(&good, &values).unwrap();
        let mut bad = good.clone();
        bad[0].value = bad[0].value.add(&Ext::ONE);
        assert!(run(&bad, &values).is_err());
        let mut bad = good.clone();
        bad[0].weights[5] = bad[0].weights[5].add(&Ext::ONE);
        assert!(run(&bad, &values).is_err());
        let mut bad = good.clone();
        bad[1].cells.reverse();
        assert!(run(&bad, &values).is_err());
        assert!(run(&good[..1], &values).is_err());
        assert!(run(&[], &values).is_err());
        let mut v2 = values.clone();
        v2[1] = v2[1].add(&Ext::ONE);
        assert!(run(&good, &v2).is_err());
        for f in [|s: &mut SparseTerm| s.cells[1] = s.cells[0], |s: &mut SparseTerm| s.cells[0] = 1 << 12, |s: &mut SparseTerm| { s.weights.pop(); }] {
            let mut bad = good.clone();
            f(&mut bad[1]);
            assert!(run(&bad, &values).unwrap_err().starts_with("PCS: sparse term"));
        }
        // no sparse terms: the same prover and verifier as the relation's eval-only openings
        let (v0, s0, pf0) = prove_commit_first(&p, &f, &points, &[], &mut FiatShamir::new(b"sparse-test"), &mut bind);
        assert!(s0.is_empty() && v0 == values);
        let mut c = FiatShamir::new(b"sparse-test");
        bind(&mut c, &pf0.roots[0]);
        pcs_verify(&p, &pf0, &points, &v0, &[], 0, &mut c).unwrap();
    }

    #[test]
    fn legacy_fixture_verifies_only_as_legacy() {
        let rep = verify(STMT, KEY, PROOF, &mut fs, &legacy()).unwrap();
        assert_eq!((rep.relation.as_str(), rep.sumcheck.as_str(), rep.vus, rep.zero_blocks), ("fp8-ada", "LGSC0002", 12, 0));
        assert!(rep.legacy && rep.key_pinned && !rep.zk);
        // Python (params.LigeritoParams.soundness + proof.soundness_claim on these parameters): union -128.0169, worst term
        // L1 proximity -128.6616, Fiat-Shamir claim 64 - 128.6616
        assert!((rep.soundness_log2 + 128.0169).abs() < 1e-3, "{}", rep.soundness_log2);
        assert!((rep.claimed_log2.unwrap() + 64.6616).abs() < 1e-3, "{:?}", rep.claimed_log2);
        assert_eq!(soundness_claim("live", -128.0).0, None);
        assert_eq!(soundness_claim("local", -128.0).0, None);
        let e = verify(STMT, KEY, PROOF, &mut fs, &Options { target_log2: Some(-128.0), ..Default::default() }).unwrap_err();
        assert!(e.contains("pre-V1"), "{e}");
        // F5: as one of two proofs of a job, 2^-128.02 misses 2^-129
        assert!(verify(STMT, KEY, PROOF, &mut fs, &Options { target_log2: Some(-129.0), ..legacy() }).is_err());
    }

    #[test]
    fn legacy_fixture_byte_flips_rejected() {
        let mut positions: Vec<usize> = (0..64).collect();
        positions.extend((64..PROOF.len()).step_by(4099));
        positions.push(PROOF.len() - 1);
        for pos in positions {
            let mut b = PROOF.to_vec();
            b[pos] ^= 0x10;
            assert!(verify(STMT, KEY, &b, &mut fs, &legacy()).is_err(), "proof byte {pos} flipped: accepted");
        }
        assert!(verify(STMT, KEY, &PROOF[..PROOF.len() - 1], &mut fs, &legacy()).is_err());
        assert!(verify(STMT, KEY, &[PROOF, &[0u8][..]].concat(), &mut fs, &legacy()).is_err());
    }

    #[test]
    fn legacy_fixture_statement_and_key_tampers_rejected() {
        let st = read_statement(STMT).unwrap();
        let tail = STMT.len() - st.y.iter().map(|y| y.len()).sum::<usize>() * st.y_bytes as usize;
        let mut positions = vec![12, 20, 60, tail - 1, tail, tail + 5, STMT.len() - 1];
        positions.extend((100..tail).step_by(5003));
        for pos in positions {
            let mut s = STMT.to_vec();
            s[pos] ^= 0x01;
            assert!(verify(&s, KEY, PROOF, &mut fs, &legacy()).is_err(), "statement byte {pos} flipped: accepted");
        }
        // any key change moves its digest off the statement's and the pin
        let mut k = KEY.to_vec();
        *k.last_mut().unwrap() ^= 1;
        assert!(verify(STMT, &k, PROOF, &mut fs, &legacy()).is_err());
        let any = Options { allow_any_key: true, ..legacy() };
        assert!(verify(STMT, &k, PROOF, &mut fs, &any).unwrap_err().contains("another verification key"));
    }

    /// ligerito-relation-2 32d3d42's gate dumps of the same statement and key with V1's six zero claims: the honest proof,
    /// its `--zk` twin, and the five V1 must-reject negatives (`neg_fiat-shamir_06..10`, each with its own statement).
    /// 06-09 commit `w + forge` with the forge on virtual rows and the forged sumcheck value consistent with it, so only
    /// a zero claim can object; 10 edits a committed `next:0` row, which the shift pins.
    const V1_PROOF: &[u8] = include_bytes!("../fixtures/lgto/fp8ada_l256_v1/honest.lgto");
    const V1_ZK_PROOF: &[u8] = include_bytes!("../fixtures/lgto/fp8ada_l256_v1/zk_honest.lgto");
    const V1_NEGS: [(&str, &[u8], &[u8], &str); 5] = [
        ("end word +1 ulp", include_bytes!("../fixtures/lgto/fp8ada_l256_v1/v1_06.stmt"), include_bytes!("../fixtures/lgto/fp8ada_l256_v1/v1_06.lgto.neg"), "PCS sumcheck round 1"),
        ("operand bit flip", include_bytes!("../fixtures/lgto/fp8ada_l256_v1/v1_07.stmt"), include_bytes!("../fixtures/lgto/fp8ada_l256_v1/v1_07.lgto.neg"), "PCS sumcheck round 1"),
        // the dropped VU's end word is left at a non-end column: non-canonical since 0db857a (Python rejects it there too)
        ("n_vus - 1", include_bytes!("../fixtures/lgto/fp8ada_l256_v1/v1_08.stmt"), include_bytes!("../fixtures/lgto/fp8ada_l256_v1/v1_08.lgto.neg"), "claimed word off a chain end (non-canonical)"),
        ("zeroed VU, y +1", include_bytes!("../fixtures/lgto/fp8ada_l256_v1/v1_09.stmt"), include_bytes!("../fixtures/lgto/fp8ada_l256_v1/v1_09.lgto.neg"), "PCS sumcheck round 1"),
        ("next:0 row edit", include_bytes!("../fixtures/lgto/fp8ada_l256_v1/v1_10.stmt"), include_bytes!("../fixtures/lgto/fp8ada_l256_v1/v1_10.lgto.neg"), "combined final"),
    ];

    fn strict() -> Options {
        Options { target_log2: Some(-128.0), ..Default::default() }
    }

    /// `V1_PROOF` with its parameter JSON's `from` replaced by `to` (the u32 length prefix before the JSON rewritten).
    fn with_params(from: &str, to: &str) -> Vec<u8> {
        let open = V1_PROOF.iter().position(|&b| b == b'{').unwrap();
        let len = u32::from_le_bytes(V1_PROOF[open - 4..open].try_into().unwrap()) as usize;
        let json = std::str::from_utf8(&V1_PROOF[open..open + len]).unwrap();
        assert_eq!(json.matches(from).count(), 1, "{from}");
        let json = json.replace(from, to);
        [&V1_PROOF[..open - 4], &(json.len() as u32).to_le_bytes()[..], json.as_bytes(), &V1_PROOF[open + len..]].concat()
    }

    #[test]
    fn v1_fixture_verifies_with_its_zero_claims() {
        assert_eq!(read_statement(V1_NEGS[4].1).unwrap().y, read_statement(STMT).unwrap().y);
        let rep = verify(STMT, KEY, V1_PROOF, &mut fs, &strict()).unwrap();
        assert_eq!((rep.sumcheck.as_str(), rep.zero_blocks, rep.legacy, rep.zk, rep.key_pinned), ("LGSC0003", 6, false, false, true));
        let rep = verify(STMT, KEY, V1_ZK_PROOF, &mut fs, &strict()).unwrap();
        assert_eq!((rep.zero_blocks, rep.legacy, rep.zk), (6, false, true));
        assert!(rep.soundness_log2 <= -128.0);
    }

    /// ligerito-relation-2 0db857a (verify-rs-3's canonicality ask): a claimed word off the chain ends (a non-end column of
    /// a VU, or a column past the sub-batch's VUs) makes the statement non-canonical; an end column may carry any word.
    #[test]
    fn y_off_a_chain_end_is_non_canonical() {
        let st = read_statement(STMT).unwrap();
        let spec = relation_spec(&st.relation).unwrap();
        let v = |st: &Statement| StmtRows::new(spec, st, st.n_vus.len()).validate();
        v(&st).unwrap();
        let steps = st.steps;
        assert!(steps > 1 && st.n_vus[0] * steps < st.l);
        for j in [0, steps - 2, steps * st.n_vus[0], st.l - 1] {
            let mut bad = st.clone();
            bad.y[0][j] = 1;
            assert!(v(&bad).unwrap_err().ends_with("claimed word off a chain end (non-canonical)"), "column {j}");
        }
        let mut ok = st.clone();
        ok.y[0][steps - 1] ^= 1;
        v(&ok).unwrap();
    }

    #[test]
    fn v1_negatives_rejected() {
        for (what, stmt, proof, why) in V1_NEGS {
            let e = verify(stmt, KEY, proof, &mut fs, &strict()).unwrap_err();
            assert!(e.contains(why), "{what}: {e}");
            assert!(verify(stmt, KEY, V1_PROOF, &mut fs, &strict()).is_err() || stmt == STMT, "{what}: honest proof on the forged statement");
        }
    }

    #[test]
    fn v1_zero_block_list_must_match_the_layout() {
        let blocks = "[[0,3577],[1,1789],[2,895],[7,28],[6,58],[0,3776]]";
        for forged in ["[[0,3577],[1,1789],[2,895],[7,29],[6,58],[0,3776]]", "[[1,1789],[0,3577],[2,895],[7,28],[6,58],[0,3776]]", "[[0,3577],[1,1789],[2,895],[7,28],[6,58]]", "[]"] {
            let e = verify(STMT, KEY, &with_params(blocks, forged), &mut fs, &strict()).unwrap_err();
            assert!(e.contains("zero_blocks"), "{forged}: {e}");
        }
        // dropping the list makes the proof pre-V1: refused unless legacy is allowed, and then the batch no longer matches
        let bare = with_params(&format!(",\"zero_blocks\":{blocks}"), "");
        assert!(verify(STMT, KEY, &bare, &mut fs, &strict()).unwrap_err().contains("pre-V1"));
        assert!(verify(STMT, KEY, &bare, &mut fs, &Options { allow_legacy: true, ..strict() }).is_err());
    }

    /// ligerito-relation-2 32d3d42's fp4-nvf4 gate dump (12 VUs, l = 256, Fiat-Shamir, 9 zero blocks): honest, and the
    /// negatives `wrong y: claimed word bit 31 flipped (proved)` (the sign end row `y_end0`), `off-domain operand 0xff in
    /// a` (a code word), `V1 end row(s): claimed word +1 ulp, true witness`.
    const FP4_KEY: &[u8] = include_bytes!("../fixtures/lgto/fp4nvf4_l256_v1/key.bin");
    const FP4_STMT: &[u8] = include_bytes!("../fixtures/lgto/fp4nvf4_l256_v1/honest.stmt");
    const FP4_PROOF: &[u8] = include_bytes!("../fixtures/lgto/fp4nvf4_l256_v1/honest.lgto");
    const FP4_NEGS: [(&[u8], &[u8], &str); 3] = [
        (include_bytes!("../fixtures/lgto/fp4nvf4_l256_v1/neg_02.stmt"), include_bytes!("../fixtures/lgto/fp4nvf4_l256_v1/neg_02.lgto.neg"), "zero-check round 0"),
        (include_bytes!("../fixtures/lgto/fp4nvf4_l256_v1/neg_04.stmt"), include_bytes!("../fixtures/lgto/fp4nvf4_l256_v1/neg_04.lgto.neg"), "a: E2M1 code outside 4 bits"),
        (include_bytes!("../fixtures/lgto/fp4nvf4_l256_v1/neg_06.stmt"), include_bytes!("../fixtures/lgto/fp4nvf4_l256_v1/neg_06.lgto.neg"), "PCS sumcheck round 1"),
    ];

    #[test]
    fn fp4_fixture_verifies_and_its_negatives_do_not() {
        let rep = verify(FP4_STMT, FP4_KEY, FP4_PROOF, &mut fs, &strict()).unwrap();
        assert_eq!((rep.relation.as_str(), rep.zero_blocks, rep.legacy, rep.key_pinned), ("fp4-nvf4", 9, false, true));
        // Python (gate_fp4-nvf4.json, the FS positive): union -128.02495, claim -64.66162
        assert!((rep.soundness_log2 + 128.02495).abs() < 1e-3, "{}", rep.soundness_log2);
        assert!((rep.claimed_log2.unwrap() + 64.66162).abs() < 1e-3, "{:?}", rep.claimed_log2);
        for (stmt, proof, why) in FP4_NEGS {
            let e = verify(stmt, FP4_KEY, proof, &mut fs, &strict()).unwrap_err();
            assert!(e.contains(why), "{why}: {e}");
        }
        // the honest proof against a claimed word with one component changed: sign, exponent field, fraction
        // (sub-batches are laid out a | b | y, so the last one's claimed words close the file)
        let st = read_statement(FP4_STMT).unwrap();
        let j = st.steps - 1;
        let at = FP4_STMT.len() - st.l * 4 + j * 4;
        assert_eq!(u32::from_le_bytes(FP4_STMT[at..at + 4].try_into().unwrap()), st.y[st.y.len() - 1][j]);
        assert_ne!(st.y[st.y.len() - 1][j], 0, "an end column");
        for bit in [31, 23, 0] {
            let mut b = FP4_STMT.to_vec();
            b[at + bit / 8] ^= 1 << (bit % 8);
            assert!(verify(&b, FP4_KEY, FP4_PROOF, &mut fs, &Options { allow_any_key: true, ..strict() }).is_err(), "y bit {bit}");
        }
    }

    #[test]
    fn nvf4_decode() {
        let zeros = [0u32; nvf4::WORDS];
        let p = nvf4::unit_pins(&zeros, &zeros).unwrap();
        // zero scales are subnormal m = 0 at 2^-9: sm = 0, X = -2 - 9 - 9, no group participates, the anchor at the floor
        assert!(p[..2 * nvf4::K].iter().all(|&v| v == 0));
        assert_eq!(&p[2 * nvf4::K..], &[0, -20, 0, 0, -20, 0, 0, -20, 0, 0, -20, 0, -174]);
        // every code 1.0 (0x2, two halves), every scale 1.0 (0x38 = 8 2^-3); b's code 0 in group 3 -> it does not participate
        let mut a = [0x2u32; nvf4::WORDS];
        a[nvf4::K..].fill(0x38);
        let mut b = a;
        b[48..64].fill(0x8);
        let p = nvf4::unit_pins(&a, &b).unwrap();
        assert_eq!((p[0], p[nvf4::K + 48]), (2, 0), "-0 is 0");
        assert_eq!(&p[2 * nvf4::K..], &[64, -8, 1, 64, -8, 1, 64, -8, 1, 64, -8, 0, -35]);
        assert_eq!((nvf4::numerator(0xF), nvf4::scale(0x7E), nvf4::scale(0x01)), (Ok(-12), Ok((14, 5)), Ok((1, -9))));
        assert!(nvf4::numerator(16).is_err() && nvf4::scale(0x7F).is_err() && nvf4::scale(0x80).is_err());
        for (name, idx) in [("a[0].n", 0), ("b[63].n", 127), ("g[0].sm", 128), ("g[3].part", 139), ("anc.G", 140)] {
            assert_eq!(nvf4::pin_index(name), Some(idx), "{name}");
        }
        for name in ["a[64].n", "a[01].n", "g[4].X", "g[0].s", "c[0].n", "a[].n", "anc.g"] {
            assert_eq!(nvf4::pin_index(name), None, "{name}");
        }
    }

    #[test]
    fn nvf4_statement_words_are_checked_by_position() {
        let st = read_statement(FP4_STMT).unwrap();
        let spec = relation_spec("fp4-nvf4").unwrap();
        assert!(StmtRows::new(spec, &st, 2).validate().is_ok());
        // a scale value in a code slot and vice versa: 0x38 is a fine scale but not a code; 0x7F is neither
        for (pos, w, ok) in [(0, 0x0F, true), (0, 0x10, false), (nvf4::K, 0x7E, true), (nvf4::K, 0x7F, false), (nvf4::K, 0x80, false), (3 * nvf4::WORDS + 1, 0x38, false)] {
            let mut s2 = read_statement(FP4_STMT).unwrap();
            s2.b[1][pos] = w;
            assert_eq!(StmtRows::new(spec, &s2, 2).validate().is_ok(), ok, "b word {pos} = {w:#x}");
        }
        // the pad sub-batch's zero units carry X = -20 and anc.G = -174 (not zero rows)
        let s1 = Statement { a: st.a[..1].to_vec(), b: st.b[..1].to_vec(), y: st.y[..1].to_vec(), n_vus: st.n_vus[..1].to_vec(), ..st.clone() };
        let eq_c = eq_vector(&random_point(st.l.trailing_zeros() as usize + 1, 7));
        let rows = StmtRows::new(spec, &s1, 2);
        let pad = eq_c[st.l..].iter().fold(Ext::ZERO, |acc, e| acc.add(e));
        let real_only = StmtRows::new(spec, &s1, 1).mle("pub:anc.G", &eq_c[..st.l]).unwrap();
        assert_eq!(rows.mle("pub:anc.G", &eq_c).unwrap(), real_only.add(&pad.mul_base(signed_mod_p(-174))));
    }

    #[test]
    fn zero_blocks_cover_exactly_the_virtual_rows() {
        let key = read_key(KEY).unwrap();
        let st = read_statement(STMT).unwrap();
        let lay = layout(&key, &st).unwrap();
        let blocks = zero_blocks(&lay);
        let mut covered: Vec<usize> = blocks.iter().flat_map(|&(b, pfx)| (pfx << b)..((pfx + 1) << b)).collect();
        covered.sort_unstable();
        let mut virt: Vec<usize> = lay.virt.iter().map(|(_, r)| *r).collect();
        virt.sort_unstable();
        assert_eq!(covered, virt);
        assert_eq!(virt[0], lay.m, "virtual rows start at m");
        assert!(blocks.windows(2).all(|w| (w[0].1 << w[0].0) < (w[1].1 << w[1].0)), "blocks in row order: {blocks:?}");
    }

    /// The histogram evaluation of the operand rows equals the direct `sum_c eq(r_c, c) decode(word_c)`.
    #[test]
    fn stmt_rows_match_direct_evaluation() {
        let st = read_statement(STMT).unwrap();
        let spec = relation_spec(&st.relation).unwrap();
        let s = st.n_vus.len();
        let n_c = (s * st.l).trailing_zeros() as usize;
        let eq_c = eq_vector(&random_point(n_c, 7));
        let rows = StmtRows::new(spec, &st, s);
        rows.validate().unwrap();
        let n_real = st.a.len();
        for (side, words) in [("a", &st.a), ("b", &st.b)] {
            for i in [0, 1, 17, st.k - 1] {
                for (f, fname) in ["s", "m", "e"].iter().enumerate() {
                    let mut want = Ext::ZERO;
                    for c in 0..s * st.l {
                        let (sb, j) = (c / st.l, c % st.l);
                        let w = if sb < n_real { words[sb][j * st.k + i] } else { 0 };
                        let (sg, m, e) = spec.op.decode(w).unwrap();
                        let v = [sg, m, signed_mod_p(e)][f];
                        want = want.add(&eq_c[c].mul_base(v));
                    }
                    let got = rows.mle(&format!("pub:{side}[{i}].{fname}"), &eq_c).unwrap();
                    assert_eq!(got, want, "pub:{side}[{i}].{fname}");
                }
            }
        }
        assert!(rows.mle("pub:a[32].s", &eq_c).is_err());
        assert!(rows.mle("pub:a[01].s", &eq_c).is_err());
        assert!(rows.mle("pub:c[0].s", &eq_c).is_err());
    }

    #[test]
    fn operand_decoders() {
        // E4M3: 0x38 = +1.0 (t = 7: m = 8, e = 0), 0x01 = min subnormal (m = 1, e = -6), NaN codes rejected
        assert_eq!(Operand::E4m3.decode(0x38).unwrap(), (0, 8, 0));
        assert_eq!(Operand::E4m3.decode(0x81).unwrap(), (1, 1, -6));
        assert!(Operand::E4m3.decode(0x7F).is_err() && Operand::E4m3.decode(0xFF).is_err() && Operand::E4m3.decode(0x100).is_err());
        // BF16: 0x3F80 = +1.0 (t = 127: m = 128, e = 0), 0x8001 = -min subnormal; the t = 255 codes rejected
        assert_eq!(Operand::Bf16.decode(0x3F80).unwrap(), (0, 128, 0));
        assert_eq!(Operand::Bf16.decode(0x8001).unwrap(), (1, 1, -126));
        for w in [0x7F80, 0xFF80, 0x7FC0, 0x10000] {
            assert!(Operand::Bf16.decode(w).is_err(), "{w:#x}");
        }
    }
}
