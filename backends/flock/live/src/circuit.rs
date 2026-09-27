//! `verity/flock-circuit` (lane flock-circuit): any lowered IR template, its serving rows committed in the circuit, and
//! nothing public but the frame-v3-sha512 roots, each row's hm96 commit string `b || c`, the declared outputs and the pinned
//! layout. `verity_flock.circuit` composes the pinned circuit (`flock-circuit`) and stages the instance files.
//!
//! A row is an `hm96-sha512/row/v1` leaf: `x = SHA-512(sha512_row_prefix || row)`, `b = x XOR M(key) y`, `c = SHA-512(salt_prefix
//! || y)` for its 192-byte salt `y`; the frame leaf binds `hm96.SHA512.tree_leaf(key, b || c)`. A block holds whole instances
//! (VUs) in aligned ranges of slots: `sha512x3` (three SHA-512 compressions, 2^18 bits: a row's blocks after its constant
//! first block), `hm96` (one per input port: the padding compression, then `b || c` from the salt), the template's units
//! (2^unit_log bits), the tail's slots, and a reserved mask slot of free cells. `A = I ⊗ A_0 + Δ` (and B), where A_0 places each
//! slot type's circuit on its ranges and Δ, derived here from the pinned circuit alone, makes inside the block:
//! - each row's first compression read the prefix's midstate and its hm96 slot the padding block (constants);
//! - every later compression read the previous one's output, and the hm96 slot the last one's (the row's wires);
//! - every unit's leaf input bits copies of the message bits that hash them, and a zero leaf a copy of a forced-zero row;
//! - every other wire of the template's slots, and every slot's constant a copy of the pinned column.
//!
//! Public regions, each opened at the verifier's points (drawn after the commitment, both reps), values all the verifier's:
//! every row's `b || c` (its hm96 slot's output) and every unit's returned-output words. The rows, their salts, the chaining
//! values and every word between slots stay private. A padding instance is the all-zero instance with zero salts, whose
//! `b || c` (checked here natively: a zero salt's mask is zero) and outputs (forced by its circuit) are pinned.
//!
//! The verifier evaluates no part of the computation (standing rule): it checks the proof, the frame-v3-sha512 roots against
//! the public `b || c` and outputs, and nothing else. A circuit carrying a verifier-evaluated tail (a `CUT` line, or a tail,
//! public cut words or native words in META) is refused at parse; a template's tail is a slot type of the circuit or the
//! template has no statement here.

use std::{collections::BTreeMap, sync::OnceLock};

use flock_core::{
    lincheck::{LincheckCircuit, SparseMatrixCircuit, pack_z_lincheck_from_packed},
    proof::eq_skip_weights,
    r1cs::{BlockR1cs, SparseBinaryMatrix, WitnessLayout},
    verifier::ExtraClaim,
};
use flock_field::F128;
use flock_prover::r1cs_hashes::blake3 as b3;
use rayon::prelude::*;
use serde_json::{Value, json};
use sha2::{Digest, Sha512};

use crate::{chunk::log2ceil, ir_block::IrUnitNet, lookup::LookupNet};

pub const TAG: &[u8] = b"verity/flock-circuit";
pub const FORMAT: &str = "flock-circuit";
pub const INST_FORMAT: &str = "flock-circuit-inputs";
pub const STATEMENT: &str = "verity/flock-circuit";
pub const SCHEME: &str = "frame-v3-sha512/hm96-sha512";
pub const ROW_SCHEMA: &str = "hm96-sha512/row/v1";
pub const SHA_NET: &str = "sha512x3";
pub const HM_NET: &str = "hm96";
/// compressions per `sha512x3` slot
pub const PER_SLOT: usize = 3;
/// META's bindings (what the statement binds about the circuit): a class leaves them out.
pub const BINDING_KEYS: [&str; 3] = ["partition", "program_sha512", "program_digests"];
pub const PARTITION_V1: &str = "verity/partition/v1";
pub const PARTITION_PROVISIONAL: &str = "flock-circuit/unit-cover/v0";
pub const SALT_BYTES: usize = 192;
pub const BC_BYTES: usize = 128;
const SHA512_ROW_TAG: &[u8] = b"verity/sha512-row/v1\0";
const ROLE_X: u8 = 1;
/// Free in-block coordinates per verifier point (`r[..PT_LOCAL]`); block coordinates follow.
pub const PT_LOCAL: usize = 24;
/// The leaf-commitment scheme this prover and verifier implement for the proof's own Merkle trees, as the composite circuit
/// pins it (`verity_flock.circuit.LEAF_SCHEME`: today's id and the `hm96-sha256/v1` room it reserves).
pub fn leaf_scheme() -> Value {
    let (id, salt) = if cfg!(feature = "sha512") { ("hm96-sha512/v1", 192) } else { ("flock-leaf/sha256-unsalted", 0) };
    json!({"id": id, "inner": "sha512 column digest", "salt_bytes": salt, "salt_source": "chacha20 expansion of a fresh 256-bit os key per proof, never reused or stored",
           "key": "default_key, a statement constant for every tree",
           "key_sha512": crate::hex(&Sha512::digest(hm96_sha512_default_key()))})
}

/// `hm96-sha512/v1`'s pinned key: `SHA-512("verity/hm96-sha512/key/v1\0" || u32be(i))` for i = 0..3, bit 2047 cleared.
pub fn hm96_sha512_default_key() -> [u8; 256] {
    let mut k = [0u8; 256];
    for (i, c) in k.chunks_mut(64).enumerate() {
        let mut h = Sha512::new();
        h.update(b"verity/hm96-sha512/key/v1\0");
        h.update((i as u32).to_be_bytes());
        c.copy_from_slice(&h.finalize());
    }
    k[255] &= 0x7f;
    k
}
const WORD: usize = 128;
const FRAME: &[u8] = b"veritor/protocol/merkle/frame/v3\0";

pub fn hexb(s: &str) -> Result<Vec<u8>, String> {
    (0..s.len() / 2).map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).map_err(|e| e.to_string())).collect()
}

fn us(v: &Value, k: &str) -> Result<usize, String> {
    v[k].as_u64().map(|x| x as usize).ok_or(format!("circuit META: {k} missing"))
}

// --- the pinned circuit ------------------------------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub enum Kind {
    /// index into `Composite::nets`
    Net(usize),
    Mask,
}

#[derive(Clone, Debug)]
pub struct Range {
    pub kind: Kind,
    pub slot_log: usize,
    pub pos0: usize,
    pub count: usize,
    /// aligned to its slot only, spanning its count (no region reads it); else aligned to its power-of-two span
    pub packed: bool,
}

impl Range {
    pub fn start(&self) -> usize {
        self.pos0 << self.slot_log
    }
    pub fn slot(&self, s: usize) -> usize {
        (self.pos0 + s) << self.slot_log
    }
}

#[derive(Clone, Debug)]
pub struct Port {
    pub name: String,
    pub words: usize,
    /// compressions after the constant first block (the padding block is the hm96 slot's)
    pub comps: usize,
}

/// A row's constants (`sha512/row/v1`): its first block, the chaining value after it, the padding block's message words.
#[derive(Clone, Debug)]
pub struct RowConsts {
    pub prefix: Vec<u8>,
    pub midstate: [u64; 8],
    pub pad: [u64; 16],
}

/// A copy of every bit of output port `src_port` of slot `(src_net, src_idx)` into input port `dst_port` of slot
/// `(dst_net, dst_idx)`, both instance-local (per VU).
#[derive(Clone, Copy, Debug)]
pub struct Wire {
    pub dst: (usize, usize, usize),
    pub src: (usize, usize, usize),
}

#[derive(Clone)]
pub struct Composite {
    pub name: String,
    /// SHA-512 of the circuit file: the pin
    pub sha512: [u8; 64],
    /// the class of every instance it proves: SHA-512 of the file without META's bindings (`BINDING_KEYS`), which a
    /// partition names
    pub class_sha512: [u8; 64],
    pub meta: Value,
    pub nets: Vec<(String, IrUnitNet)>,
    pub ranges: Vec<Range>,
    pub ports: Vec<Port>,
    pub out_words: Vec<usize>,
    pub g: usize,
    pub k_log: usize,
    pub upv: usize,
    pub unit_net: usize,
    pub leaves_in: Vec<Vec<i64>>,
    pub leaves_out: Vec<Vec<usize>>,
    pub rows: Vec<RowConsts>,
    pub sha_net: usize,
    pub hm_net: usize,
    /// the all-zero instance's `b || c` per port
    pub dummy_bc: Vec<[u8; BC_BYTES]>,
    pub dummy_outputs: Vec<u16>,
    pub leaf_scheme: Value,
    /// slots per VU of each circuit
    pub per_vu: Vec<usize>,
    pub wires: Vec<Wire>,
    /// the lookup slot types' fast witnesses, by circuit index
    pub lookups: BTreeMap<usize, LookupNet>,
}

fn table_of<'a>(t: &'a crate::ir_tail::Tables, name: &str) -> &'a [u32] {
    match name {
        "rsq" => &t.rsq,
        "sqrt" => &t.sqrt,
        "ex2" => &t.ex2,
        _ => &t.rcp,
    }
}

/// Every input (output) port of a circuit: its first column in the slot and its width, groups in order.
pub fn port_cols(groups: &[crate::ir_block::Group]) -> Vec<(usize, usize)> {
    groups.iter().flat_map(|g| g.bits.iter().scan(g.col * WORD, |a, &w| { let c = *a; *a += w; Some((c, w)) }).collect::<Vec<_>>()).collect()
}

const SHA512_IV: [u64; 8] = [0x6a09e667f3bcc908, 0xbb67ae8584caa73b, 0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1,
                              0x510e527fade682d1, 0x9b05688c2b3e6c1f, 0x1f83d9abfb41bd6b, 0x5be0cd19137e2179];

const SHA512_K: [u64; 80] = [
    0x428a2f98d728ae22, 0x7137449123ef65cd, 0xb5c0fbcfec4d3b2f, 0xe9b5dba58189dbbc,
    0x3956c25bf348b538, 0x59f111f1b605d019, 0x923f82a4af194f9b, 0xab1c5ed5da6d8118,
    0xd807aa98a3030242, 0x12835b0145706fbe, 0x243185be4ee4b28c, 0x550c7dc3d5ffb4e2,
    0x72be5d74f27b896f, 0x80deb1fe3b1696b1, 0x9bdc06a725c71235, 0xc19bf174cf692694,
    0xe49b69c19ef14ad2, 0xefbe4786384f25e3, 0x0fc19dc68b8cd5b5, 0x240ca1cc77ac9c65,
    0x2de92c6f592b0275, 0x4a7484aa6ea6e483, 0x5cb0a9dcbd41fbd4, 0x76f988da831153b5,
    0x983e5152ee66dfab, 0xa831c66d2db43210, 0xb00327c898fb213f, 0xbf597fc7beef0ee4,
    0xc6e00bf33da88fc2, 0xd5a79147930aa725, 0x06ca6351e003826f, 0x142929670a0e6e70,
    0x27b70a8546d22ffc, 0x2e1b21385c26c926, 0x4d2c6dfc5ac42aed, 0x53380d139d95b3df,
    0x650a73548baf63de, 0x766a0abb3c77b2a8, 0x81c2c92e47edaee6, 0x92722c851482353b,
    0xa2bfe8a14cf10364, 0xa81a664bbc423001, 0xc24b8b70d0f89791, 0xc76c51a30654be30,
    0xd192e819d6ef5218, 0xd69906245565a910, 0xf40e35855771202a, 0x106aa07032bbd1b8,
    0x19a4c116b8d2d0c8, 0x1e376c085141ab53, 0x2748774cdf8eeb99, 0x34b0bcb5e19b48a8,
    0x391c0cb3c5c95a63, 0x4ed8aa4ae3418acb, 0x5b9cca4f7763e373, 0x682e6ff3d6b2b8a3,
    0x748f82ee5defb2fc, 0x78a5636f43172f60, 0x84c87814a1f0ab72, 0x8cc702081a6439ec,
    0x90befffa23631e28, 0xa4506cebde82bde9, 0xbef9a3f7b2c67915, 0xc67178f2e372532b,
    0xca273eceea26619c, 0xd186b8c721c0c207, 0xeada7dd6cde0eb1e, 0xf57d4f7fee6ed178,
    0x06f067aa72176fba, 0x0a637dc5a2c898a6, 0x113f9804bef90dae, 0x1b710b35131c471b,
    0x28db77f523047d84, 0x32caab7b40c72493, 0x3c9ebe0a15c9bebc, 0x431d67c49c100d4c,
    0x4cc5d4becb3e42b6, 0x597f299cfc657e2a, 0x5fcb6fab3ad6faec, 0x6c44198c4a475817,
];

/// One SHA-512 compression of a 128-byte block into `cv` (FIPS 180-4 §6.4.2).
pub fn compress(cv: &mut [u64; 8], block: &[u8]) {
    let mut w: [u64; 80] = [0; 80];
    for (t, x) in w.iter_mut().take(16).enumerate() {
        *x = u64::from_be_bytes(block[8 * t..8 * t + 8].try_into().unwrap());
    }
    for t in 16..80 {
        let s0 = w[t - 15].rotate_right(1) ^ w[t - 15].rotate_right(8) ^ (w[t - 15] >> 7);
        let s1 = w[t - 2].rotate_right(19) ^ w[t - 2].rotate_right(61) ^ (w[t - 2] >> 6);
        w[t] = w[t - 16].wrapping_add(s0).wrapping_add(w[t - 7]).wrapping_add(s1);
    }
    let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = *cv;
    for t in 0..80 {
        let t1 = h.wrapping_add(e.rotate_right(14) ^ e.rotate_right(18) ^ e.rotate_right(41)).wrapping_add((e & f) ^ (!e & g))
            .wrapping_add(SHA512_K[t]).wrapping_add(w[t]);
        let t2 = (a.rotate_right(28) ^ a.rotate_right(34) ^ a.rotate_right(39)).wrapping_add((a & b) ^ (a & c) ^ (b & c));
        (h, g, f, e, d, c, b, a) = (g, f, e, d.wrapping_add(t1), c, b, a, t1.wrapping_add(t2));
    }
    for (x, y) in cv.iter_mut().zip([a, b, c, d, e, f, g, h]) {
        *x = x.wrapping_add(y);
    }
}

/// `sha512/row/v1`'s first block for an x row of `words` u16 words (`verity.commitments.rowleaf.sha512_row_prefix`).
pub fn row_prefix(words: usize) -> Vec<u8> {
    let mut v = SHA512_ROW_TAG.to_vec();
    v.extend([ROLE_X, 16]);
    v.extend((words as u32).to_be_bytes());
    v.resize(128, 0);
    v
}

/// The padding block after `total` bytes (a multiple of 128): 0x80, zeros, the 128-bit bit length.
pub fn pad_block(total: usize) -> [u8; 128] {
    let mut b = [0u8; 128];
    b[0] = 0x80;
    b[112..].copy_from_slice(&(8 * total as u128).to_be_bytes());
    b
}

fn words_be<const N: usize>(b: &[u8]) -> [u64; N] {
    std::array::from_fn(|i| u64::from_be_bytes(b[8 * i..8 * i + 8].try_into().unwrap()))
}

/// Per VU: the `sha512x3` slots, and each port's compressions as `(slot, sub)` (three per slot, ports in order).
pub fn row_slots(ports: &[Port]) -> (usize, Vec<Vec<(usize, usize)>>) {
    let mut at = 0;
    let mut where_ = Vec::new();
    for p in ports {
        where_.push((0..p.comps).map(|k| ((at + k) / PER_SLOT, (at + k) % PER_SLOT)).collect());
        at += p.comps;
    }
    (at.div_ceil(PER_SLOT), where_)
}

/// Each row's chain (compression k's `cv` from compression k-1's `out`) and its hm96 slot's `cv` (from the last).
fn row_wires(ports: &[Port], sha: usize, hm: usize) -> Vec<Wire> {
    let (_, where_) = row_slots(ports);
    let mut out = Vec::new();
    for (p, w) in where_.iter().enumerate() {
        for k in 1..w.len() {
            out.push(Wire { dst: (sha, w[k].0, 2 * w[k].1), src: (sha, w[k - 1].0, w[k - 1].1) });
        }
        let last = w[w.len() - 1];
        out.push(Wire { dst: (hm, p, 0), src: (sha, last.0, last.1) });
    }
    out
}

impl Composite {
    /// `tables`: the directory of the pinned MUFU tables (`verity_flock.ir_lower.write_tables`), when the circuit has lookups.
    pub fn parse(text: &str, tables: Option<&str>) -> Result<Self, String> {
        let sha512: [u8; 64] = Sha512::digest(text.as_bytes()).into();
        let mut lines = text.split_inclusive('\n');
        let head = lines.next().ok_or("empty circuit")?.trim_end();
        let name = head.strip_prefix("flock-circuit ").ok_or("not a flock-circuit circuit")?.to_string();
        let meta: Value = serde_json::from_str(lines.next().and_then(|l| l.trim_end().strip_prefix("META ")).ok_or("no META line")?)
            .map_err(|e| format!("META: {e}"))?;
        if meta["format"] != FORMAT || meta["statement"] != STATEMENT {
            return Err("META: not verity/flock-circuit".into());
        }
        let class_sha512: [u8; 64] = {
            let mut m = meta.clone();
            for k in BINDING_KEYS {
                m.as_object_mut().ok_or("META is not an object")?.remove(k);
            }
            let rest = text.splitn(3, '\n').nth(2).unwrap_or("");
            Sha512::digest(format!("{head}\nMETA {}\n{rest}", serde_json::to_string(&sorted(&m)).unwrap()).as_bytes()).into()
        };
        let is_sha512 = |v: &Value| v.as_str().is_some_and(|x| x.len() == 128 && x.bytes().all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c)));
        if !is_sha512(&meta["program_sha512"]) {
            return Err("META: program_sha512 (the Program the partition is over) is not a SHA-512 digest".into());
        }
        let prule = meta["partition"]["rule"].as_str().unwrap_or("");
        if ![PARTITION_V1, PARTITION_PROVISIONAL].contains(&prule) || !is_sha512(&meta["partition"]["digest"]) {
            return Err(format!("META: the partition is not a {PARTITION_V1} (or the provisional {PARTITION_PROVISIONAL}) SHA-512 digest"));
        }
        let mut nets = Vec::new();
        let mut lookups_fast: Vec<(usize, LookupNet)> = Vec::new();
        while let Some(l) = lines.next() {
            let h: Vec<&str> = l.split_whitespace().collect();
            if h.len() != 3 || h[0] != "CIRCUIT" {
                return Err(format!("circuit: expected a CIRCUIT line, found {:?}", l.chars().take(40).collect::<String>()));
            }
            let n: usize = h[2].parse().map_err(|_| "CIRCUIT sections count")?;
            let body: String = (0..n).map(|_| lines.next().unwrap_or("")).collect();
            let net = std::panic::catch_unwind(|| IrUnitNet::parse(&body)).map_err(|_| format!("CIRCUIT {}: not a flock-ir-unit/v2 expanded circuits", h[1]))?;
            if net.cut.is_some() {
                return Err(format!("CIRCUIT {}: a CUT line (a tail the verifier evaluates); the verifier evaluates no part of the computation", h[1]));
            }
            nets.push((h[1].to_string(), net));
        }
        for k in ["tail", "cut", "cut_words", "public_ports", "public", "native", "outputs_by_tail"] {
            if meta.get(k).is_some() {
                return Err(format!("META: {k:?} (a verifier-evaluated part of the computation); the verifier only checks proofs"));
            }
        }
        let lk = meta["lookups"].as_object().cloned().unwrap_or_default();
        if !lk.is_empty() {
            let need: std::collections::BTreeSet<&'static str> = crate::ir_tail::TABLE_SHA256.iter().map(|t| t.0).filter(|t| lk.contains_key(*t)).collect();
            if need.len() != lk.len() {
                return Err("META: a lookup of a table the statement does not pin".into());
            }
            let t = crate::ir_tail::Tables::load(tables.ok_or("the circuit has lookups: pass --tables DIR")?, &need)?;
            for (name, spec) in &lk {
                let pinned = crate::ir_tail::TABLE_SHA256.iter().find(|x| x.0 == name).unwrap();
                if spec["sha256"].as_str() != Some(pinned.1) || spec["table"].as_str() != Some(name.as_str()) || spec["out_bits"] != 31 {
                    return Err(format!("META: lookup {name} is not the pinned table"));
                }
                let n = spec["in_bits"].as_u64().unwrap_or(0) as usize;
                let lo = spec["lo_bits"].as_u64().unwrap_or(0) as usize;
                if 1usize << n != pinned.2 || !(2..n).contains(&lo) {
                    return Err(format!("META: lookup {name}'s index is not its table's"));
                }
                let words: &'static [u32] = Box::leak(table_of(&t, name).to_vec().into_boxed_slice());
                let (net, fast) = LookupNet::build(&format!("lookup:{name}"), words, n, lo, 31);
                nets.push((format!("lookup:{name}"), net));
                lookups_fast.push((nets.len() - 1, fast));
            }
        }
        if meta["scheme"] != SCHEME || meta["row_schema"] != ROW_SCHEMA {
            return Err(format!("META: rows are not {SCHEME} ({ROW_SCHEMA}) leaves"));
        }
        let ports: Vec<Port> = meta["ports"].as_array().ok_or("META: ports")?.iter().map(|p| Port {
            name: p[0].as_str().unwrap_or("").into(), words: p[1].as_u64().unwrap_or(0) as usize, comps: p[2].as_u64().unwrap_or(0) as usize,
        }).collect();
        for p in &ports {
            if p.words == 0 || (2 * p.words) % 128 != 0 || p.comps != 2 * p.words / 128 {
                return Err(format!("port {}: not whole SHA-512 blocks", p.name));
            }
        }
        // each row's constants, recomputed here from the scheme (never taken from META)
        let rows: Vec<RowConsts> = ports.iter().map(|p| {
            let prefix = row_prefix(p.words);
            let mut midstate = SHA512_IV;
            compress(&mut midstate, &prefix);
            RowConsts { prefix, midstate, pad: words_be(&pad_block(128 + 2 * p.words)) }
        }).collect();
        let listed = meta["rows"].as_array().ok_or("META: rows")?;
        if listed.len() != ports.len() || listed.iter().zip(&rows).any(|(m, r)| {
            m["prefix"].as_str() != Some(crate::hex(&r.prefix).as_str())
                || m["midstate"] != json!(r.midstate.iter().map(|w| format!("{w:016x}")).collect::<Vec<_>>())
                || m["pad"].as_str() != Some(crate::hex(&r.pad.iter().flat_map(|w| w.to_be_bytes()).collect::<Vec<u8>>()).as_str())
        }) {
            return Err("META: a row's constants are not sha512/row/v1's".into());
        }
        if meta["row_key_sha512"].as_str() != Some(crate::hex(&Sha512::digest(hm96_sha512_default_key())).as_str()) {
            return Err("META: the rows' hm96 key is not hm96-sha512/v1's default key".into());
        }
        let g = us(&meta, "vus_per_block")?;
        let upv = us(&meta, "units_per_vu")?;
        let unit_net = nets.iter().position(|n| n.0 == "unit").ok_or("no unit's expanded circuit")?;
        let mut ranges = Vec::new();
        for r in meta["ranges"].as_array().ok_or("META: ranges")? {
            let kind = match r["type"].as_str() {
                Some("mask") => Kind::Mask,
                Some("unit") => Kind::Net(nets.iter().position(|n| Some(n.0.as_str()) == r["net"].as_str()).ok_or("a range of an unknown circuit")?),
                _ => return Err("a range of an unknown type".into()),
            };
            let packed = r["packed"].as_bool().unwrap_or(false);
            ranges.push(Range { kind, slot_log: us(r, "slot_log")?, pos0: us(r, "pos0")?, count: us(r, "count")?, packed });
        }
        let ints = |v: &Value| -> Vec<Vec<i64>> { v.as_array().map(|a| a.iter().map(|u| u.as_array().map(|x| x.iter().filter_map(|y| y.as_i64()).collect()).unwrap_or_default()).collect()).unwrap_or_default() };
        let leaves_in = ints(&meta["leaves_in"]);
        let leaves_out: Vec<Vec<usize>> = ints(&meta["leaves_out"]).into_iter().map(|u| u.into_iter().map(|x| x as usize).collect()).collect();
        let dummy_bc = meta["dummy"]["bc"].as_array().ok_or("META: dummy bc")?.iter()
            .map(|d| hexb(d.as_str().unwrap_or("")).and_then(|b| b.try_into().map_err(|_| "b || c length".to_string()))).collect::<Result<Vec<[u8; BC_BYTES]>, String>>()?;
        let dummy_outputs = meta["dummy"]["outputs"].as_array().ok_or("META: dummy outputs")?.iter().map(|x| x.as_u64().unwrap_or(u64::MAX) as u16).collect();
        let out_words = meta["out_ports"].as_array().ok_or("META: out_ports")?.iter().map(|p| p[1].as_u64().unwrap_or(0) as usize).collect();
        let net_of = |nm: &str| nets.iter().position(|n| n.0 == nm).ok_or(format!("META: unknown circuit {nm}"));
        let mut per_vu = vec![0usize; nets.len()];
        per_vu[unit_net] = upv;
        if let Some(m) = meta["per_vu"].as_object() {
            for (k, v) in m {
                per_vu[net_of(k)?] = v.as_u64().unwrap_or(0) as usize;
            }
        }
        let (sha_net, hm_net) = (net_of(SHA_NET)?, net_of(HM_NET)?);
        let mut wires = Vec::new();
        for w in meta["wires"].as_array().map(|a| a.as_slice()).unwrap_or(&[]) {
            let n = |i: usize| w[i].as_str().ok_or("a wire's circuit".to_string()).and_then(net_of);
            let u = |i: usize| w[i].as_u64().map(|x| x as usize).ok_or("a wire's index".to_string());
            wires.push(Wire { dst: (n(0)?, u(1)?, u(2)?), src: (n(3)?, u(4)?, u(5)?) });
        }
        // the rows' wires come first, exactly as the ports imply
        let derived = row_wires(&ports, sha_net, hm_net);
        if wires.len() < derived.len() || wires.iter().zip(&derived).any(|(a, b)| a.dst != b.dst || a.src != b.src)
            || wires[derived.len()..].iter().any(|w| [sha_net, hm_net].contains(&w.dst.0) || [sha_net, hm_net].contains(&w.src.0))
        {
            return Err("META: the rows' wires are not the chains and hm96 inputs the ports imply".into());
        }
        let c = Composite {
            name, sha512, class_sha512, k_log: us(&meta, "k_log")?, leaf_scheme: meta["leaf_scheme"].clone(), meta, nets, ranges, ports,
            out_words, g, upv, unit_net, leaves_in, leaves_out, rows, sha_net, hm_net, dummy_bc, dummy_outputs, per_vu, wires,
            lookups: lookups_fast.into_iter().collect(),
        };
        c.check()?;
        Ok(c)
    }

    /// The layout's shape: aligned, disjoint ranges inside the block; the comp ranges' roles cover every run and parent of
    /// every VU exactly once; the unit ranges hold `g x units_per_vu` units whose leaf maps are the unit's expanded circuit's ports.
    fn check(&self) -> Result<(), String> {
        let mut spans: Vec<(usize, usize)> = Vec::new();
        for r in &self.ranges {
            let (span, align) = if r.packed { (r.count << r.slot_log, 1 << r.slot_log) } else { (r.count.next_power_of_two() << r.slot_log, r.count.next_power_of_two() << r.slot_log) };
            if r.packed && r.kind != Kind::Net(self.sha_net) {
                return Err("a packed range that is not the compressions'".into());
            }
            if r.start() % align != 0 || r.start() + span > 1 << self.k_log {
                return Err("a range is not aligned to its span or overruns the block".into());
            }
            if let Kind::Net(i) = r.kind
                && self.nets[i].1.unit_log != r.slot_log
            {
                return Err("a unit range's slot size is not its circuit's".into());
            }
            if r.kind == Kind::Mask && r.count != 1 {
                return Err("not one mask slot".into());
            }
            spans.push((r.start(), r.start() + span));
        }
        spans.sort();
        if spans.windows(2).any(|w| w[0].1 > w[1].0) {
            return Err("two ranges overlap".into());
        }
        let (n_sha, _) = row_slots(&self.ports);
        if self.per_vu[self.sha_net] != n_sha || self.per_vu[self.hm_net] != self.ports.len().next_power_of_two() {
            return Err("the rows' slots per VU are not the ports' compressions (three per slot) and one hm96 slot per port".into());
        }
        let (sha, hm) = (&self.nets[self.sha_net].1, &self.nets[self.hm_net].1);
        if port_cols(&sha.in_groups) != (0..2 * PER_SLOT).map(|i| (port_cols(&sha.in_groups)[i].0, if i % 2 == 0 { 512 } else { 1024 })).collect::<Vec<_>>()
            || port_cols(&sha.out_groups).iter().map(|x| x.1).collect::<Vec<_>>() != vec![512; PER_SLOT]
            || port_cols(&hm.in_groups).iter().map(|x| x.1).collect::<Vec<_>>() != vec![512, 1024, 8 * SALT_BYTES]
            || port_cols(&hm.out_groups).iter().map(|x| x.1).collect::<Vec<_>>() != vec![8 * BC_BYTES]
        {
            return Err("the sha512x3 and hm96 circuits' ports are not (cv, msg) x 3 -> out x 3 and (cv, pad, y) -> bc".into());
        }
        let unit = &self.nets[self.unit_net].1;
        let ur: Vec<&Range> = self.ranges.iter().filter(|r| r.kind == Kind::Net(self.unit_net)).collect();
        if ur.len() != 1 || ur[0].count != self.g * self.upv || !ur[0].count.is_power_of_two() {
            return Err("not one unit range of vus_per_block x units_per_vu (a power of two) units".into());
        }
        if unit.in_groups.is_empty() || unit.in_groups.len() > 2 || unit.out_groups.is_empty() || unit.out_groups.len() > 2
            || unit.in_groups[0].bits.iter().any(|&b| b != 16) || unit.out_groups[0].bits.iter().any(|&b| b != 16)
        {
            return Err("the unit's ports are not a group of 16-bit leaf inputs (then cut inputs) and one of 16-bit returned outputs (then cut outputs)".into());
        }
        for (i, r) in self.ranges.iter().enumerate() {
            if let Kind::Net(n) = r.kind
                && r.count != self.g * self.per_vu[n]
            {
                return Err(format!("range {i} of {}: {} slots, not {} per VU", self.nets[n].0, r.count, self.per_vu[n]));
            }
        }
        for (n, (nm, _)) in self.nets.iter().enumerate() {
            let ranges = self.ranges.iter().filter(|r| r.kind == Kind::Net(n)).count();
            if ranges != 1 {
                return Err(format!("circuit {nm} has {ranges} ranges, not one"));
            }
        }
        // every input port of every slot is wired exactly once: the unit's leaf ports by its leaf map, the rest by a wire
        let mut wired: BTreeMap<(usize, usize, usize), usize> = BTreeMap::new();
        for w in &self.wires {
            let (dn, di, dp) = w.dst;
            let (sn, si, sp) = w.src;
            let (dc, sc) = (port_cols(&self.nets[dn].1.in_groups), port_cols(&self.nets[sn].1.out_groups));
            if di >= self.per_vu[dn] || si >= self.per_vu[sn] || dp >= dc.len() || sp >= sc.len() || dc[dp].1 != sc[sp].1 {
                return Err(format!("a wire {w:?} outside its slots' ports or of unequal widths"));
            }
            if dn == self.unit_net && dp < unit.in_groups[0].bits.len() {
                return Err("a wire into a unit's leaf port".into());
            }
            *wired.entry(w.dst).or_default() += 1;
        }
        for (n, (nm, net)) in self.nets.iter().enumerate() {
            if n == self.sha_net || n == self.hm_net {
                continue;
            }
            let first = if n == self.unit_net { net.in_groups[0].bits.len() } else { 0 };
            for i in 0..self.per_vu[n] {
                for p in first..port_cols(&net.in_groups).len() {
                    if wired.get(&(n, i, p)).copied().unwrap_or(0) != 1 {
                        return Err(format!("input port {p} of {nm} slot {i} is not wired exactly once"));
                    }
                }
            }
        }
        let n_in: usize = self.ports.iter().map(|p| p.words).sum();
        let n_out: usize = self.out_words.iter().sum();
        if self.leaves_in.len() != self.upv || self.leaves_out.len() != self.upv {
            return Err("the leaf maps are not one per unit".into());
        }
        let mut seen_out = vec![false; n_out];
        for u in 0..self.upv {
            if self.leaves_in[u].len() != unit.in_groups[0].bits.len() || self.leaves_in[u].iter().any(|&x| x < -1 || x >= n_in as i64)
                || self.leaves_out[u].len() != unit.out_groups[0].bits.len()
            {
                return Err(format!("unit {u}'s leaf maps are not its circuit's ports"));
            }
            for &o in &self.leaves_out[u] {
                if o >= n_out || std::mem::replace(&mut seen_out[o], true) {
                    return Err(format!("output leaf {o} is outside the outputs or returned twice"));
                }
            }
        }
        if seen_out.iter().any(|x| !x) {
            return Err("an output leaf no unit returns".into());
        }
        if unit.useful >= 1 << unit.unit_log {
            return Err("the unit fills its slot: no forced-zero row for a zero leaf".into());
        }
        if self.dummy_bc.len() != self.ports.len() || self.dummy_outputs.len() != n_out {
            return Err("the pinned dummy values are not one per port and output leaf".into());
        }
        if self.leaf_scheme != leaf_scheme() {
            return Err(format!("leaf-commitment scheme {} is not the one this verifier implements", self.leaf_scheme));
        }
        Ok(())
    }

    pub fn range_of(&self, n: usize) -> &Range {
        self.ranges.iter().find(|r| r.kind == Kind::Net(n)).unwrap()
    }

    /// Block-local column of slot `i` (of VU `g`) of circuit `n`.
    pub fn slot_col(&self, n: usize, g: usize, i: usize) -> usize {
        self.range_of(n).slot(g * self.per_vu[n] + i)
    }

    pub fn unit(&self) -> &IrUnitNet {
        &self.nets[self.unit_net].1
    }

    pub fn unit_range(&self) -> &Range {
        self.ranges.iter().find(|r| r.kind == Kind::Net(self.unit_net)).unwrap()
    }

    pub fn row_bytes(&self) -> usize {
        self.ports.iter().map(|p| 2 * p.words).sum()
    }

}

/// The pinned constant column: the first slot of the first range that has one.
pub fn pin_of(c: &Composite) -> usize {
    let r = c.ranges.iter().find(|r| r.kind != Kind::Mask).unwrap();
    r.start() + match r.kind {
        Kind::Net(i) => c.nets[i].1.const_pos,
        Kind::Mask => unreachable!(),
    }
}

/// Δ (A and B), derived from the pinned circuit: every slot's constant, the rows' constants (midstate, padding block), the unit
/// leaf wiring (each leaf bit a copy of its message bit), and every wire (the rows' chains first).
pub fn delta(c: &Composite, pin: usize) -> (Vec<(u32, u32)>, Vec<(u32, u32)>) {
    let (mut da, mut db) = (Vec::new(), Vec::new());
    let both = |da: &mut Vec<(u32, u32)>, db: &mut Vec<(u32, u32)>, i: usize, src: usize| {
        for d in [da, db] {
            d.push((i as u32, i as u32));
            d.push((i as u32, src as u32));
        }
    };
    for r in &c.ranges {
        let Kind::Net(i) = r.kind else { continue };
        for s in 0..r.count {
            let k = r.slot(s) + c.nets[i].1.const_pos;
            if k != pin {
                both(&mut da, &mut db, k, pin);
            }
        }
    }
    let (_, where_) = row_slots(&c.ports);
    let (sin, hin) = (port_cols(&c.nets[c.sha_net].1.in_groups), port_cols(&c.nets[c.hm_net].1.in_groups));
    // bit `t` of row byte `byte` of VU `g`'s port `p`: a message bit of its compression (the block's bytes big-endian per word)
    let msg_col = |g: usize, p: usize, byte: usize, t: usize| -> usize {
        let (k, o) = (byte / 128, byte % 128);
        let (slot, sub) = where_[p][k];
        c.slot_col(c.sha_net, g, slot) + sin[2 * sub + 1].0 + 64 * (o / 8) + 56 - 8 * (o % 8) + t
    };
    let zero = c.unit_range().slot(0) + c.unit().useful;
    for g in 0..c.g {
        for (p, w) in where_.iter().enumerate() {
            let cv = c.slot_col(c.sha_net, g, w[0].0) + sin[2 * w[0].1].0;
            for t in 0..512 {
                both(&mut da, &mut db, cv + t, if (c.rows[p].midstate[t / 64] >> (t % 64)) & 1 == 1 { pin } else { zero });
            }
            let pad = c.slot_col(c.hm_net, g, p) + hin[1].0;
            for t in 0..1024 {
                both(&mut da, &mut db, pad + t, if (c.rows[p].pad[t / 64] >> (t % 64)) & 1 == 1 { pin } else { zero });
            }
        }
    }
    let unit = c.unit();
    let ur = c.unit_range();
    let leaf_cols: Vec<usize> = {
        let g0 = &unit.in_groups[0];
        g0.bits.iter().scan(g0.col * WORD, |a, &w| { let x = *a; *a += w; Some(x) }).collect()
    };
    let starts: Vec<usize> = c.ports.iter().scan(0, |a, p| { let s = *a; *a += p.words; Some(s) }).collect();
    for k in 0..ur.count {
        let (g, u) = (k / c.upv, k % c.upv);
        let base = ur.slot(k);
        for (j, &leaf) in c.leaves_in[u].iter().enumerate() {
            for t in 0..16 {
                let src = if leaf < 0 {
                    base + unit.useful
                } else {
                    let leaf = leaf as usize;
                    let p = starts.iter().rposition(|&s| s <= leaf).unwrap();
                    msg_col(g, p, 2 * (leaf - starts[p]) + t / 8, t % 8)
                };
                both(&mut da, &mut db, base + leaf_cols[j] + t, src);
            }
        }
    }
    for g in 0..c.g {
        for w in &c.wires {
            let (dc, sc) = (port_cols(&c.nets[w.dst.0].1.in_groups)[w.dst.2], port_cols(&c.nets[w.src.0].1.out_groups)[w.src.2]);
            let (dst, src) = (c.slot_col(w.dst.0, g, w.dst.1) + dc.0, c.slot_col(w.src.0, g, w.src.1) + sc.0);
            for t in 0..dc.1 {
                both(&mut da, &mut db, dst + t, src + t);
            }
        }
    }
    (da, db)
}

// --- the instance files ------------------------------------------------------------------------------------------------

/// A `flock-circuit-inputs` file: the prover's has the rows and their salts, the verifier's (`rows: false`) only the public part.
pub struct Instances {
    pub header: Value,
    pub n: usize,
    pub rows: Option<Vec<u8>>,
    /// every row's salt, instance-major, ports in order (the prover's file only)
    pub salts: Option<Vec<u8>>,
    /// every row's `b || c`, instance-major, ports in order
    pub bcs: Vec<[u8; BC_BYTES]>,
    pub outputs: Vec<u16>,
    /// SHA-512 of the header line and every public byte after the rows and salts (`b || c`, outputs): what Σ binds
    pub public_sha512: [u8; 64],
    /// the registered population's `b || c` and outputs (a drawn statement: the frame-v3 roots are over them); `None`: the
    /// statement's own
    pub population: Option<(usize, Vec<[u8; BC_BYTES]>, Vec<u16>)>,
}

/// The header's `units`: the instances' global unit indices (the partition's order: ascending), and per block its units.
fn units_entry(c: &Composite, indices: &[usize]) -> Value {
    json!({"vus_per_block": c.g, "units_per_vu": c.upv, "indices": indices, "blocks": indices.chunks(c.g).map(|b| b.to_vec()).collect::<Vec<_>>(),
           "classes": vec![crate::hex(&c.class_sha512); indices.len()]})
}

impl Instances {
    pub fn load(path: &str, c: &Composite) -> Result<Self, String> {
        let b = std::fs::read(path).map_err(|e| format!("instances: {e}"))?;
        let nl = b.iter().position(|&x| x == b'\n').ok_or("no header line")?;
        let mut header: Value = serde_json::from_slice(&b[..nl]).map_err(|e| format!("header: {e}"))?;
        if header["format"] != INST_FORMAT || header["statement"] != STATEMENT {
            return Err("not a flock-circuit-inputs file".into());
        }
        if header["circuit_sha512"].as_str() != Some(crate::hex(&c.sha512).as_str()) {
            return Err("the instance file was staged for another circuit".into());
        }
        let n = header["instances"].as_u64().ok_or("header: instances")? as usize;
        let has_rows = header["rows"].as_bool().ok_or("header: rows")?;
        let rb = c.row_bytes();
        let (n_out, np) = (c.out_words.iter().sum::<usize>(), c.ports.len());
        let body = &b[nl + 1..];
        let (rows_len, salts_len) = if has_rows { (n * rb, n * np * SALT_BYTES) } else { (0, 0) };
        if body.len() != rows_len + salts_len + BC_BYTES * n * np + 2 * n * n_out {
            return Err("instance file length".into());
        }
        let rows = has_rows.then(|| body[..rows_len].to_vec());
        let salts = has_rows.then(|| body[rows_len..rows_len + salts_len].to_vec());
        let pubs = &body[rows_len + salts_len..];
        let bcs = pubs[..BC_BYTES * n * np].chunks(BC_BYTES).map(|x| x.try_into().unwrap()).collect();
        let outputs = pubs[BC_BYTES * n * np..].chunks(2).map(|x| u16::from_le_bytes([x[0], x[1]])).collect();
        let indices: Vec<usize> = header["units"]["indices"].as_array().ok_or("header: units.indices")?.iter()
            .map(|x| x.as_u64().map(|v| v as usize).ok_or("header: a unit index")).collect::<Result<_, _>>()?;
        if indices.len() != n || indices.windows(2).any(|w| w[0] >= w[1]) || header["units"] != units_entry(c, &indices) {
            return Err("header: the instances' unit indices are not ascending, one per instance, blocks of vus_per_block".into());
        }
        if header.get("unit_draw").is_some() {
            return Err("header: a staged file is a registered population; a draw is made in the session".into());
        }
        header["rows"] = Value::Bool(false);
        let public_sha512 = Self::public_digest(&header, pubs);
        Ok(Instances { header, n, rows, salts, bcs, outputs, public_sha512, population: None })
    }

    fn public_digest(header: &Value, pubs: &[u8]) -> [u8; 64] {
        let mut h = Sha512::new();
        h.update(serde_json::to_vec(&sorted(header)).unwrap());
        h.update(pubs);
        h.finalize().into()
    }

    /// The statement of the drawn units (`unit_draw` over this registered population's positions `[0, n)`): instance `i` is the
    /// population's `units[i]`; its header is the population's with the draw, `instances = K` and the drawn units' indices; its
    /// public digest covers that header and the population's public bytes (the frame-v3 roots stay the population's).
    pub fn drawn(&self, c: &Composite, draw: &Value) -> Result<Instances, String> {
        let units = crate::unit_draw::check(draw)?;
        if draw["population"].as_u64() != Some(self.n as u64) || self.population.is_some() {
            return Err("U: the draw's population is not this registered population".into());
        }
        let np = c.ports.len();
        let (rb, n_out) = (c.row_bytes(), c.out_words.iter().sum::<usize>());
        let indices: Vec<usize> = units.iter().map(|&u| self.header["units"]["indices"][u].as_u64().unwrap_or(0) as usize).collect();
        let mut header = self.header.clone();
        header["unit_draw"] = draw.clone();
        header["instances"] = json!(units.len());
        header["population"] = json!(self.n);
        header["units"] = units_entry(c, &indices);
        let mut pubs: Vec<u8> = self.bcs.iter().flatten().copied().collect();
        pubs.extend(self.outputs.iter().flat_map(|w| w.to_le_bytes()));
        let public_sha512 = Self::public_digest(&header, &pubs);
        Ok(Instances {
            header, n: units.len(),
            rows: self.rows.as_ref().map(|r| units.iter().flat_map(|&u| r[u * rb..(u + 1) * rb].to_vec()).collect()),
            salts: self.salts.as_ref().map(|s| units.iter().flat_map(|&u| s[u * np * SALT_BYTES..(u + 1) * np * SALT_BYTES].to_vec()).collect()),
            bcs: units.iter().flat_map(|&u| self.bcs[u * np..(u + 1) * np].to_vec()).collect(),
            outputs: units.iter().flat_map(|&u| self.outputs[u * n_out..(u + 1) * n_out].to_vec()).collect(),
            public_sha512,
            population: Some((self.n, self.bcs.clone(), self.outputs.clone())),
        })
    }

    pub fn bc(&self, i: usize, p: usize, np: usize) -> [u8; BC_BYTES] {
        self.bcs[i * np + p]
    }

    pub fn salt(&self, i: usize, p: usize, np: usize) -> &[u8] {
        let s = self.salts.as_ref().expect("the prover's instance file has the salts");
        &s[(i * np + p) * SALT_BYTES..(i * np + p + 1) * SALT_BYTES]
    }
}

fn sorted(v: &Value) -> Value {
    match v {
        Value::Object(m) => Value::Object(m.iter().map(|(k, x)| (k.clone(), sorted(x))).collect::<std::collections::BTreeMap<_, _>>().into_iter().collect()),
        Value::Array(a) => Value::Array(a.iter().map(sorted).collect()),
        x => x.clone(),
    }
}

fn fv3_hash(tag: &[u8], parts: &[&[u8]]) -> [u8; 64] {
    let mut h = Sha512::new();
    h.update(FRAME);
    h.update((tag.len() as u32).to_be_bytes());
    h.update(tag);
    for p in parts {
        h.update((p.len() as u64).to_be_bytes());
        h.update(p);
    }
    h.finalize().into()
}

fn uint(x: usize) -> Vec<u8> {
    let b = (x as u64).to_be_bytes();
    let z = b.iter().position(|&c| c != 0).unwrap_or(7);
    b[z..].to_vec()
}

/// A frame-v3-sha512 leaf: position `i` of the domain, its schema and its payload.
pub fn fv3_leaf(domain_id: &[u8], i: usize, schema: &[u8], payload: &[u8]) -> [u8; 64] {
    fv3_hash(b"leaf", &[domain_id, &uint(i), &uint(i), schema, payload])
}

pub fn fv3_root(domain_id: &[u8], leaves: Vec<[u8; 64]>) -> [u8; 64] {
    let n = leaves.len();
    let width = n.next_power_of_two();
    let mut level = leaves;
    level.extend((n..width).map(|r| fv3_hash(b"pad", &[domain_id, &uint(r)])));
    let mut depth = 0;
    while level.len() > 1 {
        level = level.par_chunks(2).enumerate().map(|(i, p)| fv3_hash(b"node", &[domain_id, &uint(depth), &uint(i), &p[0], &p[1]])).collect();
        depth += 1;
    }
    level[0]
}

/// hm96-sha512/v1's salt prefix and the default key's leaf prefix (both one block).
#[cfg(feature = "sha512")]
fn hm96_prefixes() -> (Vec<u8>, Vec<u8>) {
    let k = flock_merkle::hm96::consts();
    (k.salt_prefix.to_vec(), k.leaf_prefix.to_vec())
}

#[cfg(not(feature = "sha512"))]
fn hm96_prefixes() -> (Vec<u8>, Vec<u8>) {
    panic!("verity/flock-circuit's rows are hm96-sha512 leaves: build with the sha512 feature")
}

/// `hm96.SHA512.tree_leaf(key, b || c)` for the default key: SHA-512 of the leaf prefix block and `b || c`.
pub fn row_tree_leaf(bc: &[u8; BC_BYTES]) -> [u8; 64] {
    let mut h = Sha512::new();
    h.update(hm96_prefixes().1);
    h.update(bc);
    h.finalize().into()
}

/// The all-zero row's `b || c` under a zero salt: `b = x` (a zero salt's mask is zero), `c = SHA-512(salt_prefix || 0^192)`.
pub fn zero_row_bc(words: usize) -> [u8; BC_BYTES] {
    let mut h = Sha512::new();
    h.update(row_prefix(words));
    h.update(vec![0u8; 2 * words]);
    let x = h.finalize();
    let mut h = Sha512::new();
    h.update(hm96_prefixes().0);
    h.update([0u8; SALT_BYTES]);
    let mut bc = [0u8; BC_BYTES];
    bc[..64].copy_from_slice(&x);
    bc[64..].copy_from_slice(&h.finalize());
    bc
}

/// The frame-v3-sha512 roots recomputed from the file's `b || c` and output words, against its header's; the pinned dummy
/// `b || c` against the zero row's (the commitment scheme's own functions; the circuit is never evaluated here).
pub fn check_public(c: &Composite, inst: &Instances) -> Result<(), String> {
    let fv = &inst.header["frame_v3"];
    let get = |k: &str, p: &str| -> Result<Vec<u8>, String> { hexb(fv[k][p].as_str().ok_or(format!("frame_v3.{k}.{p} missing"))?) };
    if fv["frame"] != "frame-v3-sha512" || fv["row_key_sha512"].as_str() != Some(crate::hex(&Sha512::digest(hm96_sha512_default_key())).as_str()) {
        return Err("the instance file's rows are not frame-v3-sha512 hm96 leaves under the default key".into());
    }
    let np = c.ports.len();
    // the frame-v3 roots are over the registered population (a drawn statement's instances are some of its units)
    let (n, bcs, outputs) = match &inst.population { Some((n, b, o)) => (*n, b, o), None => (inst.n, &inst.bcs, &inst.outputs) };
    if let Some(d) = inst.header.get("unit_draw") {
        let units = crate::unit_draw::check(d)?;
        if inst.header["instances"].as_u64() != Some(units.len() as u64) || inst.n != units.len() {
            return Err("U3: the statement's instances are not the drawn units".into());
        }
    }
    for (p, port) in c.ports.iter().enumerate() {
        let dom = get("domain_ids", &port.name)?;
        if fv["schemas"][&port.name] != ROW_SCHEMA {
            return Err(format!("input port {}: schema is not {ROW_SCHEMA}", port.name));
        }
        let leaves = (0..n).into_par_iter().map(|i| fv3_leaf(&dom, i, ROW_SCHEMA.as_bytes(), &row_tree_leaf(&bcs[i * np + p]))).collect();
        if fv3_root(&dom, leaves).to_vec() != get("roots", &port.name)? {
            return Err(format!("input port {}: the rows' b || c do not hash to the header's frame-v3-sha512 root", port.name));
        }
        if zero_row_bc(port.words) != c.dummy_bc[p] {
            return Err(format!("the pinned dummy b || c of port {} is not the zero row's", port.name));
        }
    }
    let n_out: usize = c.out_words.iter().sum();
    let mut at = 0;
    for (name, w) in c.meta["out_ports"].as_array().unwrap().iter().map(|x| (x[0].as_str().unwrap_or(""), x[1].as_u64().unwrap_or(0) as usize)) {
        let dom = get("domain_ids", name)?;
        let words: Vec<u16> = (0..n).flat_map(|i| outputs[i * n_out + at..i * n_out + at + w].to_vec()).collect();
        let leaves = words.par_iter().enumerate().map(|(r, v)| fv3_leaf(&dom, r, b"u16", &v.to_be_bytes())).collect();
        if fv3_root(&dom, leaves).to_vec() != get("roots", name)? {
            return Err(format!("output port {name}: the output words do not hash to the header's frame-v3-sha512 root"));
        }
        at += w;
    }
    Ok(())
}

// --- the statement -------------------------------------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub enum Region {
    /// port `p`'s `b || c` of each VU (its hm96 slot's output)
    Digest(usize),
    /// every unit slot's returned-output words
    Out,
}

pub struct RegionShape {
    pub region: Region,
    pub free: Vec<usize>,
    pub mask: u32,
    pub fixed: u32,
}

pub struct BlockCircuit {
    pub k_log: usize,
    pub pin: usize,
    /// per slot type: its log size and its circuit
    pub types: Vec<(usize, SparseMatrixCircuit<'static>)>,
    /// (type index, first position, count) per range
    pub ranges: Vec<(usize, usize, usize)>,
    pub da: Vec<(u32, u32)>,
    pub db: Vec<(u32, u32)>,
}

impl LincheckCircuit for BlockCircuit {
    fn n_cols(&self) -> usize {
        1 << self.k_log
    }

    fn fold_alpha_batched(&self, alpha: F128, eq: &[F128]) -> Vec<F128> {
        assert_eq!(eq.len(), 1 << self.k_log);
        let lo = self.types.iter().map(|t| 1usize << t.0).min().unwrap();
        let c0 = eq[..lo].iter().position(|e| *e != F128::ZERO).expect("eq tensor vanishes on the first slot");
        let inv = eq[c0].inv();
        let bases: Vec<Vec<F128>> = self.types.iter().map(|(sl, m)| m.fold_alpha_batched(alpha, &eq[..1 << sl])).collect();
        let mut out = vec![F128::ZERO; 1 << self.k_log];
        for &(t, pos0, count) in &self.ranges {
            let sl = self.types[t].0;
            let base = &bases[t];
            out[pos0 << sl..(pos0 + count) << sl].par_chunks_mut(1 << sl).enumerate().for_each(|(k, o)| {
                let ratio = eq[((pos0 + k) << sl) | c0] * inv;
                for (x, y) in o.iter_mut().zip(base) {
                    *x = ratio * *y;
                }
            });
        }
        for &(i, c) in &self.da {
            out[c as usize] += alpha * eq[i as usize];
        }
        for &(i, c) in &self.db {
            out[c as usize] += eq[i as usize];
        }
        out
    }

    fn const_pin_col(&self) -> Option<usize> {
        Some(self.pin)
    }
}

/// A dishonest prover's row witness (VU 0 of block 0): a compression's chaining value, the first chaining value (not the
/// midstate), the padding block, or a salt, each followed consistently downstream.
#[derive(Clone, Copy, Debug, Default)]
pub struct RowTamper {
    pub cv_flip: bool,
    pub midstate_flip: bool,
    pub pad_flip: bool,
    pub salt_flip: bool,
}

/// Block `b`'s row slots' input words (`Stmt::row_inputs`), as a witness source hands them over.
pub type RowIns<'a> = &'a (dyn Fn(usize) -> (Vec<Vec<F128>>, Vec<Vec<F128>>) + Sync);

/// Everything a session needs, the same on both sides.
pub struct Stmt {
    pub c: Composite,
    pub n: usize,
    pub blocks_real: usize,
    pub nbl: usize,
    pub m: usize,
    pub m_pts: usize,
    pub pin: usize,
    pub regions: Vec<RegionShape>,
    pub circuit: BlockCircuit,
    pub r1cs: BlockR1cs,
    /// SHA-512 over the statement (bound through Σ)
    pub digest: [u8; 64],
    /// its first 32 bytes, for Flock's own 32-byte statement slot (the R1CS digest its transcript absorbs)
    pub flock_digest: [u8; 32],
    /// how the verifier's identity is pinned: profile, reps, zk, leaf scheme, coin mode (backend identity)
    pub identity: Value,
}

fn bit(w: &F128, t: usize) -> u64 {
    (if t < 64 { w.lo >> t } else { w.hi >> (t - 64) }) & 1
}

fn set_bit(w: &mut F128, t: usize) {
    if t < 64 { w.lo |= 1 << t } else { w.hi |= 1 << (t - 64) }
}

fn flip_bit(w: &mut F128, t: usize) {
    if t < 64 { w.lo ^= 1 << t } else { w.hi ^= 1 << (t - 64) }
}

fn leak<T>(x: T) -> &'static T {
    Box::leak(Box::new(x))
}

impl Stmt {
    pub fn new(c: Composite, n: usize, identity: Value) -> Result<Self, String> {
        let k_log = c.k_log;
        let blocks_real = n.div_ceil(c.g);
        let nbl = log2ceil(blocks_real.max(1)).max(3);
        let m = k_log + nbl;
        let pin = pin_of(&c);
        let (da, db) = delta(&c, pin);
        let mut types: Vec<(usize, SparseMatrixCircuit<'static>)> = Vec::new();
        let mut type_of: BTreeMap<String, usize> = BTreeMap::new();
        let mut ranges = Vec::new();
        for r in &c.ranges {
            let key = format!("{:?}", r.kind);
            let t = *type_of.entry(key).or_insert_with(|| {
                let circ = match r.kind {
                    Kind::Net(i) => {
                        let net = &c.nets[i].1;
                        let k = 1usize << net.unit_log;
                        let pad = |rows: &Vec<Vec<usize>>| { let mut x = rows.clone(); x.resize(k, Vec::new()); SparseBinaryMatrix { num_rows: k, num_cols: k, rows: x.into() } };
                        SparseMatrixCircuit::new(leak(pad(&net.a)), leak(pad(&net.b)))
                    }
                    Kind::Mask => {
                        let k = 1usize << r.slot_log;
                        let id = SparseBinaryMatrix { num_rows: k, num_cols: k, rows: (0..k).map(|i| vec![i]).collect::<Vec<_>>().into() };
                        SparseMatrixCircuit::new(leak(id.clone()), leak(id))
                    }
                };
                types.push((r.slot_log, circ));
                types.len() - 1
            });
            ranges.push((t, r.pos0, r.count));
        }
        let regions = Self::regions(&c)?;
        let mut h = Sha512::new();
        h.update(TAG);
        h.update(c.sha512);
        h.update(serde_json::to_vec(&sorted(&identity)).unwrap());
        for x in [n, blocks_real, nbl, m, k_log, pin, regions.len(), c.g] {
            h.update((x as u64).to_le_bytes());
        }
        for d in [&da, &db] {
            h.update((d.len() as u64).to_le_bytes());
            for &(i, cc) in d.iter() {
                h.update(i.to_le_bytes());
                h.update(cc.to_le_bytes());
            }
        }
        let digest: [u8; 64] = h.finalize().into();
        let flock_digest: [u8; 32] = digest[..32].try_into().unwrap();
        let kk = 1usize << k_log;
        let empty = || SparseBinaryMatrix { num_rows: kk, num_cols: kk, rows: vec![Vec::new(); kk].into() };
        let r1cs = BlockR1cs {
            m,
            k_log,
            k_skip: b3::K_SKIP,
            useful_bits: kk,
            a_0: empty(),
            b_0: empty(),
            c_0: SparseBinaryMatrix { num_rows: kk, num_cols: kk, rows: (0..kk).map(|i| vec![i]).collect::<Vec<_>>().into() },
            layout: WitnessLayout::RowMajor,
            const_pin: Some(pin),
            digest_cache: OnceLock::from(flock_digest),
            csc_cache: OnceLock::new(),
        };
        let circuit = BlockCircuit { k_log, pin, types, ranges, da, db };
        Ok(Stmt { c, n, blocks_real, nbl, m, m_pts: PT_LOCAL + nbl, pin, regions, circuit, r1cs, digest, flock_digest, identity })
    }

    fn regions(c: &Composite) -> Result<Vec<RegionShape>, String> {
        let kl = c.k_log;
        let bits = |lo: usize, n: usize| (lo..lo + n).collect::<Vec<usize>>();
        let mk = |region: Region, free: Vec<usize>, fixed: usize| -> Result<RegionShape, String> {
            let mask = ((1u32 << kl) - 1) & !free.iter().fold(0u32, |mm, &b| mm | (1 << b));
            if fixed as u32 & !mask != 0 || free.len() > PT_LOCAL || free.iter().any(|&b| b >= kl) {
                return Err(format!("{region:?}: not an aligned region of at most {PT_LOCAL} free bits"));
            }
            Ok(RegionShape { region, free, mask, fixed: fixed as u32 })
        };
        let mut out = Vec::new();
        let hr = c.range_of(c.hm_net);
        let (hm, lp, lg) = (&c.nets[c.hm_net].1, c.per_vu[c.hm_net].trailing_zeros() as usize, c.g.trailing_zeros() as usize);
        let bc0 = port_cols(&hm.out_groups)[0].0;
        for p in 0..c.ports.len() {
            out.push(mk(Region::Digest(p), [bits(0, 10), bits(hr.slot_log + lp, lg)].concat(), hr.start() + (p << hr.slot_log) + bc0)?);
        }
        let unit = c.unit();
        let ur = c.unit_range();
        let g0 = &unit.out_groups[0];
        out.push(mk(Region::Out, [bits(0, 7 + g0.words.trailing_zeros() as usize), bits(ur.slot_log, ur.count.trailing_zeros() as usize)].concat(),
                    ur.start() + g0.col * WORD)?);
        Ok(out)
    }

    pub fn blocks(&self) -> usize {
        1 << self.nbl
    }

    fn inst_of(&self, b: usize, g: usize) -> Option<usize> {
        let i = b * self.c.g + g;
        (i < self.n).then_some(i)
    }

    /// Block `b`'s row slots' input words (`sha512x3` slots, then hm96 slots, each VU's in order), computed natively from the rows
    /// and salts (zeros for a padding instance): each compression's chaining value and message block, each hm96 slot's last
    /// chaining value, padding block and salt. `t` tampers VU 0 of block 0, the chain continuing from the tampered value.
    pub fn row_inputs(&self, rows: Option<&[u8]>, salts: Option<&[u8]>, b: usize, t: RowTamper) -> (Vec<Vec<F128>>, Vec<Vec<F128>>) {
        let c = &self.c;
        let (sha, hm) = (&c.nets[c.sha_net].1, &c.nets[c.hm_net].1);
        let (sin, hin) = (port_cols(&sha.in_groups), port_cols(&hm.in_groups));
        let (n_sha, where_) = row_slots(&c.ports);
        let (np, per_hm, rb) = (c.ports.len(), c.per_vu[c.hm_net], c.row_bytes());
        let mut si = vec![vec![F128::ZERO; sha.in_words]; c.g * n_sha];
        let mut hi = vec![vec![F128::ZERO; hm.in_words]; c.g * per_hm];
        let put = |w: &mut [F128], col: usize, words: &[u64]| {
            for (i, &x) in words.iter().enumerate() {
                let at = col + 64 * i;
                debug_assert_eq!(at % 64, 0);
                if at % WORD == 0 { w[at / WORD].lo = x } else { w[at / WORD].hi = x }
            }
        };
        let chained = c.ports.iter().position(|p| p.comps > 1);
        for g in 0..c.g {
            let inst = self.inst_of(b, g);
            let first = b == 0 && g == 0;
            let mut off = 0;
            for (p, port) in c.ports.iter().enumerate() {
                let n = 2 * port.words;
                let row: Vec<u8> = match (rows, inst) { (Some(r), Some(i)) => r[i * rb + off..i * rb + off + n].to_vec(), _ => vec![0; n] };
                off += n;
                let mut cv = c.rows[p].midstate;
                if t.midstate_flip && first && p == 0 {
                    cv[0] ^= 1;
                }
                for k in 0..port.comps {
                    if t.cv_flip && first && k == 1 && chained == Some(p) {
                        cv[0] ^= 1;
                    }
                    let (slot, sub) = where_[p][k];
                    let blk = &row[128 * k..128 * k + 128];
                    let w = &mut si[g * n_sha + slot];
                    put(w, sin[2 * sub].0, &cv);
                    put(w, sin[2 * sub + 1].0, &words_be::<16>(blk));
                    compress(&mut cv, blk);
                }
                let mut pad = c.rows[p].pad;
                if t.pad_flip && first && p == 0 {
                    pad[0] ^= 1;
                }
                let mut salt = [0u8; SALT_BYTES];
                if let (Some(s), Some(i)) = (salts, inst) {
                    salt.copy_from_slice(&s[(i * np + p) * SALT_BYTES..(i * np + p + 1) * SALT_BYTES]);
                }
                if t.salt_flip && first && p == 0 {
                    salt[0] ^= 1;
                }
                let h = &mut hi[g * per_hm + p];
                put(h, hin[0].0, &cv);
                put(h, hin[1].0, &pad);
                put(h, hin[2].0, &(0..SALT_BYTES / 8).map(|i| u64::from_le_bytes(salt[8 * i..8 * i + 8].try_into().unwrap())).collect::<Vec<_>>());
            }
        }
        (si, hi)
    }

    /// Unit slot `k` of block `b`'s input words (its leaf ports, from the rows; zeros for a padding instance or a zero leaf).
    pub fn unit_inputs(&self, rows: Option<&[u8]>, b: usize, k: usize) -> Vec<F128> {
        let c = &self.c;
        let unit = c.unit();
        let (g, u) = (k / c.upv, k % c.upv);
        let mut w = vec![F128::ZERO; unit.in_words];
        let (Some(rows), Some(i)) = (rows, self.inst_of(b, g)) else { return w };
        let rb = c.row_bytes();
        let mut col = unit.in_groups[0].col * WORD;
        for (j, &leaf) in c.leaves_in[u].iter().enumerate() {
            if leaf >= 0 {
                let at = i * rb + 2 * leaf as usize;
                let v = u16::from_le_bytes([rows[at], rows[at + 1]]) as u64;
                if col % WORD < 64 { w[col / WORD].lo |= v << (col % WORD) } else { w[col / WORD].hi |= v << (col % WORD - 64) }
            }
            col += unit.in_groups[0].bits[j];
        }
        w
    }

    /// Every slot's output words (`net.out_cols`) for input words `ins` (a lookup: its table word; else the rows, 64 at once).
    fn eval_outputs(&self, n: usize, net: &IrUnitNet, ins: &[Vec<F128>]) -> Vec<Vec<F128>> {
        if let Some(lk) = self.c.lookups.get(&n) {
            return ins.par_iter().map(|w| {
                let idx = (w[0].lo as usize) & ((1 << lk.n) - 1);
                vec![F128::new((lk.table[idx] & 0x7FFF_FFFF) as u64, 0)]
            }).collect();
        }
        let o0 = net.out_cols[0] * WORD;
        let nw = net.out_cols.len();
        ins.par_chunks(64).flat_map_iter(|grp| {
            let (zz, _, _) = net.eval64(|col| grp.iter().enumerate().fold(0u64, |x, (l, r)| x | (bit(&r[col / WORD], col % WORD) << l)));
            (0..grp.len()).map(move |l| {
                let mut o = vec![F128::ZERO; nw];
                for (r, y) in zz[o0..o0 + nw * WORD].iter().enumerate() {
                    if (y >> l) & 1 == 1 {
                        set_bit(&mut o[r / WORD], r % WORD);
                    }
                }
                o
            }).collect::<Vec<_>>()
        }).collect()
    }

    /// Every slot's input words, converged: the units' leaf words from `unit_inputs(b, k)`, then every wire copied from the
    /// slot that computes it, re-evaluating each slot type whose inputs changed until none do (the tail and the lookups
    /// between the units' two passes). `nets` overrides a circuit's evaluation (a tampered witness).
    pub fn converge(&self, unit_inputs: &(dyn Fn(usize, usize) -> Vec<F128> + Sync), row_ins: RowIns<'_>, nets: &BTreeMap<usize, &IrUnitNet>) -> Vec<Vec<Vec<F128>>> {
        self.converge_with(unit_inputs, Some(row_ins), nets, None)
    }

    /// [`Stmt::converge`] with the units' output words given (`unit_outs`, the device's first pass): the units are never
    /// evaluated here, only the tail stages and the lookups.
    pub fn converge_with(&self, unit_inputs: &(dyn Fn(usize, usize) -> Vec<F128> + Sync), row_ins: Option<RowIns<'_>>,
                         nets: &BTreeMap<usize, &IrUnitNet>, unit_outs: Option<Vec<Vec<F128>>>) -> Vec<Vec<Vec<F128>>> {
        let c = &self.c;
        let nb = self.blocks();
        let net_of = |n: usize| -> &IrUnitNet { nets.get(&n).copied().unwrap_or(&c.nets[n].1) };
        let count = |n: usize| c.range_of(n).count;
        let mut ins: Vec<Vec<Vec<F128>>> = (0..c.nets.len()).map(|n| {
            if n == c.unit_net {
                (0..nb * count(n)).into_par_iter().map(|x| unit_inputs(x / count(n), x % count(n))).collect()
            } else {
                vec![vec![F128::ZERO; net_of(n).in_words]; nb * count(n)]
            }
        }).collect();
        if let Some(f) = row_ins {
            let rv: Vec<_> = (0..nb).into_par_iter().map(f).collect();
            ins[c.sha_net] = rv.iter().flat_map(|x| x.0.iter().cloned()).collect();
            ins[c.hm_net] = rv.into_iter().flat_map(|x| x.1).collect();
        }
        // the rows' wires hold by construction (row_inputs chains natively); the rest converge here
        let row_net = |n: usize| n == c.sha_net || n == c.hm_net;
        let wires: Vec<&Wire> = c.wires.iter().filter(|w| !row_net(w.dst.0)).collect();
        if wires.is_empty() {
            return ins;
        }
        let mut outs: Vec<Vec<Vec<F128>>> = vec![Vec::new(); c.nets.len()];
        let mut dirty: Vec<bool> = (0..c.nets.len()).map(|n| !row_net(n)).collect();
        let fixed = unit_outs.is_some();
        if let Some(u) = unit_outs {
            outs[c.unit_net] = u;
            dirty[c.unit_net] = false;
        }
        for _round in 0..64 {
            for n in 0..c.nets.len() {
                if fixed && n == c.unit_net {
                    dirty[n] = false;
                    continue;
                }
                if dirty[n] {
                    outs[n] = self.eval_outputs(n, net_of(n), &ins[n]);
                    dirty[n] = false;
                }
            }
            let mut changed = false;
            for w in &wires {
                let (dn, sn) = (w.dst.0, w.src.0);
                let (dcol, width) = port_cols(&net_of(dn).in_groups)[w.dst.2];
                let scol = port_cols(&net_of(sn).out_groups)[w.src.2].0 - net_of(sn).out_cols[0] * WORD;
                for b in 0..nb {
                    for g in 0..c.g {
                        let xs = b * count(sn) + g * c.per_vu[sn] + w.src.1;
                        let xd = b * count(dn) + g * c.per_vu[dn] + w.dst.1;
                        for t in 0..width {
                            let v = bit(&outs[sn][xs][(scol + t) / WORD], (scol + t) % WORD);
                            let d = &mut ins[dn][xd][(dcol + t) / WORD];
                            if bit(d, (dcol + t) % WORD) != v {
                                flip_bit(d, (dcol + t) % WORD);
                                changed = true;
                                dirty[dn] = true;
                            }
                        }
                    }
                }
            }
            if !changed {
                return ins;
            }
        }
        panic!("the slots' wires did not converge in 64 rounds (a cycle through a slot's own output?)")
    }

    /// `(z, a, b, z_lincheck)`: every block's compressions (`comps(b)`), every slot of every circuit from its input words `ins`
    /// (`converge`; a lookup by its fast witness), the mask slot from `mask`. `nets` overrides a circuit's evaluation.
    #[allow(clippy::type_complexity)]
    pub fn witness(&self, ins: &[Vec<Vec<F128>>], mask: &(dyn Fn(usize) -> Vec<F128> + Sync), nets: &BTreeMap<usize, &IrUnitNet>)
        -> (Vec<F128>, Vec<F128>, Vec<F128>, Vec<u8>) {
        let c = &self.c;
        let kl = c.k_log;
        let n = 1usize << (self.m - 7);
        let wpb = 1usize << (kl - 7);
        let (mut z, mut a, mut bb) = (vec![F128::ZERO; n], vec![F128::ZERO; n], vec![F128::ZERO; n]);
        z.par_chunks_mut(wpb).zip(a.par_chunks_mut(wpb)).zip(bb.par_chunks_mut(wpb)).enumerate().for_each(|(blk, ((zb, ab), bbb))| {
            for r in c.ranges.iter().filter(|r| r.kind == Kind::Mask) {
                let mw = mask(blk);
                let w = r.start() / WORD;
                zb[w..w + mw.len()].copy_from_slice(&mw);
                ab[w..w + mw.len()].copy_from_slice(&mw);
                bbb[w..w + mw.len()].copy_from_slice(&mw);
            }
        });
        for (ni, (_, pinned)) in c.nets.iter().enumerate() {
            let r = c.range_of(ni);
            let uw = 1usize << (pinned.unit_log - 7);
            for (x, uz, ua, ub) in self.slot_zab(ni, nets.get(&ni).copied().unwrap_or(pinned), &ins[ni]) {
                let (blk, k) = (x / r.count, x % r.count);
                let at = blk * wpb + r.slot(k) / WORD;
                z[at..at + uw].copy_from_slice(&uz);
                a[at..at + uw].copy_from_slice(&ua);
                bb[at..at + uw].copy_from_slice(&ub);
            }
        }
        let zl = pack_z_lincheck_from_packed(&z, self.m, kl);
        (z, a, bb, zl)
    }

    /// Every slot of circuit `ni`'s `(instance, z, a, b)` words from its input words (a lookup: its fast witness).
    #[allow(clippy::type_complexity)]
    pub fn slot_zab(&self, ni: usize, net: &IrUnitNet, ins: &[Vec<F128>]) -> Vec<(usize, Vec<F128>, Vec<F128>, Vec<F128>)> {
        let c = &self.c;
        let uw = 1usize << (c.nets[ni].1.unit_log - 7);
        {
            let items: Vec<(usize, Vec<F128>, Vec<F128>, Vec<F128>)> = if let Some(lk) = c.lookups.get(&ni) {
                ins.par_iter().enumerate().map(|(x, w)| {
                    let (zz, aa, bq) = lk.witness((w[0].lo as usize) & ((1 << lk.n) - 1));
                    let f = |v: Vec<u64>| v.chunks(2).map(|p| F128::new(p[0], p[1])).collect::<Vec<F128>>();
                    (x, f(zz), f(aa), f(bq))
                }).collect()
            } else {
                ins.par_chunks(64).enumerate().flat_map_iter(|(gi, grp)| {
                    let (zz, aa, bq) = net.eval64(|col| grp.iter().enumerate().fold(0u64, |x, (l, r)| x | (bit(&r[col / WORD], col % WORD) << l)));
                    (0..grp.len()).map(move |l| {
                        let pack = |v: &Vec<u64>| -> Vec<F128> {
                            let mut o = vec![F128::ZERO; uw];
                            for (row, &y) in v.iter().enumerate() {
                                if (y >> l) & 1 == 1 {
                                    set_bit(&mut o[row / WORD], row % WORD);
                                }
                            }
                            o
                        };
                        (64 * gi + l, pack(&zz), pack(&aa), pack(&bq))
                    }).collect::<Vec<_>>()
                }).collect()
            };
            items
        }
    }

    /// The device's first pass handed back (`cut`: per group of 32 units and per cut output row of the unit, the row's bits,
    /// unit l of the group in bit l): the tail stages and lookups converged on the host; returns every unit's late input rows
    /// in the same layout (`late_rows` order) and every host slot's `(absolute F128 word offset, z, a, b)`.
    #[allow(clippy::type_complexity)]
    pub fn host_tail(&self, unit_inputs: &(dyn Fn(usize, usize) -> Vec<F128> + Sync), cut: &[u32], cut_rows: &[usize], late_rows: &[usize])
        -> (Vec<u32>, Vec<(usize, Vec<F128>, Vec<F128>, Vec<F128>)>) {
        let c = &self.c;
        let unit = c.unit();
        let n_units = self.blocks() * c.unit_range().count;
        let groups = n_units.div_ceil(32);
        let o0 = unit.out_cols[0] * WORD;
        let outs: Vec<Vec<F128>> = (0..n_units).map(|u| {
            let mut o = vec![F128::ZERO; unit.out_cols.len()];
            let (g, l) = (u / 32, u % 32);
            for (i, &r) in cut_rows.iter().enumerate() {
                if (cut[g * cut_rows.len() + i] >> l) & 1 == 1 {
                    set_bit(&mut o[(r - o0) / WORD], (r - o0) % WORD);
                }
            }
            o
        }).collect();
        let ins = self.converge_with(unit_inputs, None, &BTreeMap::new(), Some(outs));
        let mut late = vec![0u32; groups * late_rows.len()];
        for (u, w) in ins[c.unit_net].iter().enumerate() {
            for (k, &r) in late_rows.iter().enumerate() {
                if bit(&w[r / WORD], r % WORD) == 1 {
                    late[(u / 32) * late_rows.len() + k] |= 1 << (u % 32);
                }
            }
        }
        let wpb = 1usize << (c.k_log - 7);
        let mut host = Vec::new();
        for ni in 0..c.nets.len() {
            if ni == c.unit_net || ni == c.sha_net || ni == c.hm_net {
                continue;
            }
            let r = c.range_of(ni);
            for (x, z, a, b) in self.slot_zab(ni, &c.nets[ni].1, &ins[ni]) {
                host.push(((x / r.count) * wpb + r.slot(x % r.count) / WORD, z, a, b));
            }
        }
        (late, host)
    }

    /// Block `b`'s bytes of a region, in free-assignment order.
    fn region_bytes(&self, reg: &Region, inst: &Instances, b: usize) -> Vec<u8> {
        let c = &self.c;
        let np = c.ports.len();
        match reg {
            Region::Digest(p) => (0..c.g).flat_map(|g| match self.inst_of(b, g) { Some(i) => inst.bc(i, *p, np), None => c.dummy_bc[*p] }).collect(),
            Region::Out => {
                let unit = c.unit();
                let ur = c.unit_range();
                let (g0, n_out) = (&unit.out_groups[0], c.out_words.iter().sum::<usize>());
                (0..ur.count).flat_map(|k| {
                    let (g, u) = (k / c.upv, k % c.upv);
                    let mut w = vec![F128::ZERO; g0.words];
                    let mut col = 0;
                    for (j, &o) in c.leaves_out[u].iter().enumerate() {
                        let v = match self.inst_of(b, g) { Some(i) => inst.outputs[i * n_out + o], None => c.dummy_outputs[o] } as u64;
                        if col % WORD < 64 { w[col / WORD].lo |= v << (col % WORD) } else { w[col / WORD].hi |= v << (col % WORD - 64) }
                        col += g0.bits[j];
                    }
                    w.into_iter().flat_map(|x| [x.lo.to_le_bytes(), x.hi.to_le_bytes()].concat())
                }).collect()
            }
        }
    }

    pub fn region_point(&self, reg: &RegionShape, r: &[F128]) -> Vec<F128> {
        let kl = self.c.k_log;
        let mut p = vec![F128::ZERO; self.m];
        for (j, x) in p.iter_mut().enumerate().take(kl) {
            if (reg.mask >> j) & 1 == 1 && (reg.fixed >> j) & 1 == 1 {
                *x = F128::ONE;
            }
        }
        for (t, &j) in reg.free.iter().enumerate() {
            p[j] = r[t];
        }
        p[kl..].copy_from_slice(&r[PT_LOCAL..PT_LOCAL + self.nbl]);
        p
    }

    pub fn region_value(&self, reg: &RegionShape, r: &[F128], inst: &Instances) -> F128 {
        let nf = reg.free.len();
        let wt = eq_skip_weights(&r[..nf]);
        let eb = eq_skip_weights(&r[PT_LOCAL..PT_LOCAL + self.nbl]);
        let nbytes = (1usize << nf) / 8;
        let mut t = vec![[F128::ZERO; 256]; nbytes];
        for (i, tb) in t.iter_mut().enumerate() {
            for x in 1..256usize {
                let rest = x & (x - 1);
                tb[x] = tb[rest] + wt[8 * i + (x ^ rest).trailing_zeros() as usize];
            }
        }
        (0..self.blocks()).into_par_iter().map(|b| {
            let bytes = self.region_bytes(&reg.region, inst, b);
            debug_assert_eq!(bytes.len(), nbytes, "{:?}", reg.region);
            eb[b] * bytes.iter().enumerate().fold(F128::ZERO, |a, (i, &x)| a + t[i][x as usize])
        }).reduce(|| F128::ZERO, |a, b| a + b)
    }

    pub fn extra_points(&self, pts: &[Vec<F128>]) -> Vec<Vec<F128>> {
        self.regions.iter().flat_map(|reg| pts.iter().map(move |r| self.region_point(reg, r))).collect()
    }

    pub fn extra_claims(&self, pts: &[Vec<F128>], inst: &Instances) -> Vec<ExtraClaim> {
        let values: Vec<F128> = self.regions.iter().flat_map(|reg| pts.iter().map(move |r| self.region_value(reg, r, inst))).collect();
        self.extra_points(pts).into_iter().zip(values)
            .map(|(pt, value)| ExtraClaim { value, skip_weights: eq_skip_weights(&pt[..6]), x_full: pt[6..].to_vec() })
            .collect()
    }

    /// A region claim's ring-switch partial evaluations from the public region data alone: `s[v] = Σ_x eq(p[7..], x) z[128 x + v]`
    /// over the region's cells at its point `p` (M1's audit: a region claim's `s_hat_v` reveals nothing but public data).
    pub fn region_s_hat_v(&self, reg: &RegionShape, r: &[F128], inst: &Instances) -> Vec<F128> {
        let p = self.region_point(reg, r);
        let kl = self.c.k_log;
        let fixed = (reg.fixed & reg.mask) as usize;
        (0..self.blocks()).into_par_iter().map(|b| {
            let bytes = self.region_bytes(&reg.region, inst, b);
            let mut s = vec![F128::ZERO; 128];
            for a in 0..(1usize << reg.free.len()) {
                if (bytes[a / 8] >> (a % 8)) & 1 == 0 {
                    continue;
                }
                let i = reg.free.iter().enumerate().fold(fixed, |i, (t, &j)| i | (((a >> t) & 1) << j)) | (b << kl);
                let x = i >> 7;
                s[i & 127] += (7..self.m).fold(F128::ONE, |acc, j| acc * if (x >> (j - 7)) & 1 == 1 { p[j] } else { F128::ONE + p[j] });
            }
            s
        }).reduce(|| vec![F128::ZERO; 128], |a, b| a.iter().zip(&b).map(|(x, y)| *x + *y).collect())
    }

    /// Words per block the device witness needs for the unit slots: every unit's input words, block-major.
    pub fn region_bytes_len(&self, reg: &RegionShape) -> usize {
        (1usize << reg.free.len()) / 8
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_compression_is_sha512() {
        let msg: Vec<u8> = (0..300u32).map(|i| (i * 7 + 3) as u8).collect();
        let mut padded = msg.clone();
        padded.push(0x80);
        while padded.len() % 128 != 112 {
            padded.push(0);
        }
        padded.extend((8 * msg.len() as u128).to_be_bytes());
        let mut cv = SHA512_IV;
        for blk in padded.chunks(128) {
            compress(&mut cv, blk);
        }
        let want: Vec<u8> = Sha512::digest(&msg).to_vec();
        assert_eq!(cv.iter().flat_map(|w| w.to_be_bytes()).collect::<Vec<u8>>(), want);
        let mut p = row_prefix(64);
        p.extend(vec![0u8; 128]);
        p.extend(pad_block(256));
        let mut cv = SHA512_IV;
        for blk in p.chunks(128) {
            compress(&mut cv, blk);
        }
        let mut h = Sha512::new();
        h.update(row_prefix(64));
        h.update([0u8; 128]);
        assert_eq!(cv.iter().flat_map(|w| w.to_be_bytes()).collect::<Vec<u8>>(), h.finalize().to_vec());
    }

    #[test]
    fn hm96_sha512_default_key_is_core_s() {
        // hashlib.sha512(verity.commitments.hm96.SHA512.default_key).hexdigest() (PR #93)
        assert_eq!(super::leaf_scheme()["key_sha512"],
                   "afc5a60c85d8aa75cb4c7f726588d299a1712e1c6a621f09ca64809a5fedd744a0f6b0201d05e82b29cd5b274e6c29670cd4e4ab10831911fb1f9e0c7019df22");
    }

    /// flock_merkle::hm96 (verity's Flock patch) against core's verity.commitments.hm96 vectors_sha512.json: the key, both
    /// prefixes, the default-key masks and the leaves over given inner digests.
    #[cfg(feature = "sha512")]
    #[test]
    fn hm96_leaves_match_core_vectors() {
        use flock_merkle::hm96;
        let repo = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).canonicalize().unwrap().join("../../..");
        let Ok(text) = std::fs::read_to_string(repo.join("packages/verity/src/verity/commitments/hm96/vectors_sha512.json")) else {
            return;
        };
        let v: serde_json::Value = serde_json::from_str(&text).unwrap();
        let hx = |s: &serde_json::Value| super::hexb(s.as_str().unwrap()).unwrap();
        assert_eq!(hm96::default_key().to_vec(), hx(&v["default_key"]));
        assert_eq!(hm96::consts().salt_prefix.to_vec(), hx(&v["salt_prefix"]));
        assert_eq!(hm96::consts().leaf_prefix.to_vec(), hx(&v["leaf_prefix_default_key"]));
        let mut n = 0;
        for m in v["masks"].as_array().unwrap().iter().filter(|m| m["key"] == "default") {
            assert_eq!(hm96::mask(&hx(&m["salt"])).to_vec(), hx(&m["mask"]), "{}", m["salt_case"]);
            n += 1;
        }
        for l in v["leaves"].as_array().unwrap() {
            let x = flock_hash::digest_from(&hx(&l["inner_digest"]));
            assert_eq!(hm96::leaf(&x, &hx(&l["salt"])).as_flattened().to_vec(), hx(&l["leaf"]));
            n += 1;
        }
        assert!(n >= 8, "{n} vectors");
    }
}
