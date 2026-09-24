//! The three files (`backends/direct/ligero/serialize.py` defines them): `ligero-system/v1` (`/v2` = v1 with the
//! chain end in components), `ligero-statement/v2` (`/v3` = v2 + the authentication block, `authentication =
//! included`; `/v4` = the relation-named statement: byte widths of the words, the relation string, then an optional
//! v3 block), `ligero-proof/v1` (`/v2` = v1 + one multiproof per committed tree).  Reject-first: every length is
//! bounded by the statement before a byte of the proof body is read, every field element is checked canonical (< p)
//! at parse.
//!
//! ~~~text
//! ligero-system/v2     magic "LIGSYS02" | as v1, but the chain's trailing `expr y16` is replaced by
//!                      u32 n_end | n_end x (expr Y, u32 lo, u32 bits): the chain end in COMPONENTS (chain.py) --
//!                      the public word's bit fields `(y >> lo) & (2^bits - 1)`, each of at most 30 bits (so
//!                      < 2^30 < p), because a word of more than 30 bits (the FP4 relation's FP32 accumulator) is
//!                      not a BabyBear element
//! ligero-statement/v4  magic "LIGSTM04" | the v2 header through u32 K | u8 word_bytes | u8 y_bytes | str relation
//!                      | l x K a | l x K b | l x y16 (u8 / u16 / u32 per the byte widths) | u8 has_auth | [v3 block]
//! ligero-system/v3     magic "LIGSYS03" | as v2, but the chain block is u32 n_links | n_links x u32 c | n_links x expr y
//!                      | u8 has_y16 | [expr y16] or [u32 n_end | n_end x (expr Y, u32 lo, u32 bits)]: more than the
//!                      accumulator's three linked rows (the hashed relations carry 16 Poseidon2 capacity lanes along
//!                      the chain; chain.py gives the extra links monomial coefficients, see verify.rs)
//! ligero-statement/v5  magic "LIGSTM05" | the v2 header through u32 K (= 1536, the words per committed ROW) | u8 y_bytes
//!                      | str relation | str authentication ("included-hash") | n_vus x 16 u32 digests (per VU the x
//!                      row's 8 Poseidon2 lanes then the W column's) | l x y16 | the hash-auth block: str version
//!                      ("ligero-b/auth/v2h") | str schema ("poseidon2-babybear-w24/row/v2h") | 32 B Poseidon2
//!                      constants digest | u32 n_vus | n_vus x (u32 vu_index, u32 x_index, u32 w_index) | 3 x tree (a,
//!                      b, y) as in v3 | 3 x (u32 n_sib | n_sib x 32 B) SHA-256 multiproofs.  No operand words: the
//!                      proof's public rows are the digest table (is_end, 16 lanes at every VU's last column).
//! ligero-statement/v6  magic "LIGSTM06" | the v5 layout with `l x y16` replaced by `n_vus x y` (y_bytes): the chain-end
//!                      words in VU order (lane tier0-bytes).  The reader rebuilds the l-vector (y16[v*steps+steps-1] =
//!                      y[v], 0 elsewhere -- what the hashed prover's public vector is), so the statement digest and
//!                      every check are those of the v5 file; 91 -> 27 kB per sub-batch at l = 16384.  The same magic with
//!                      str authentication "included-hash-shared" is the row-sharing pair (`PairStatement`).
//! ~~~
//! The words are kept as `u32` whatever their width; the relation the file names decides (relation.rs) which widths
//! are legal, how the words enter the statement digest and how the operands decode.  The authentication block's leaf
//! schema is `u16`, so a v4 file may carry one only with 16-bit words (Python defines nothing else).  A v1 chain end
//! is kept as the one end term `(y16, family 6, lo 0, bits 0)` -- `bits = 0` meaning "the whole word as a residue"
//! (`chain.end_terms`), so the two formats run through one chain test.

use crate::auth::{AuthBlock, HashAuthBlock, TreeRef, HASH_VERSION, MAX_OPENINGS, MAX_POSITIONS, MAX_PROOF_BYTES, SCHEMA,
                  VERSION as AUTH_VERSION};
use crate::leaf::LeafScheme;
use crate::field::P;

pub type Result<T> = std::result::Result<T, String>;

struct Rd<'a> {
    d: &'a [u8],
    at: usize,
}

impl<'a> Rd<'a> {
    fn new(d: &'a [u8]) -> Rd<'a> {
        Rd { d, at: 0 }
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.at.checked_add(n).map_or(true, |e| e > self.d.len()) {
            return Err("truncated file".into());
        }
        let b = &self.d[self.at..self.at + n];
        self.at += n;
        Ok(b)
    }
    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn b32(&mut self) -> Result<[u8; 32]> {
        Ok(self.take(32)?.try_into().unwrap())
    }
    fn usize_bounded(&mut self, max: usize, what: &str) -> Result<usize> {
        let v = self.u32()? as usize;
        if v > max {
            return Err(format!("{what} = {v} exceeds the bound {max}"));
        }
        Ok(v)
    }
    fn f64(&mut self) -> Result<f64> {
        Ok(f64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn str(&mut self) -> Result<String> {
        let n = self.usize_bounded(1 << 16, "string length")?;
        String::from_utf8(self.take(n)?.to_vec()).map_err(|_| "bad utf-8".to_string())
    }
    fn fe(&mut self) -> Result<u32> {
        let v = self.u32()?;
        if v >= P {
            return Err(format!("non-canonical field element {v} >= p"));
        }
        Ok(v)
    }
    fn fe_vec(&mut self, n: usize) -> Result<Vec<u32>> {
        let raw = self.take(4 * n)?;
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let v = u32::from_le_bytes(raw[4 * i..4 * i + 4].try_into().unwrap());
            if v >= P {
                return Err(format!("non-canonical field element {v} >= p at index {i}"));
            }
            out.push(v);
        }
        Ok(out)
    }
    /// `n` little-endian unsigned words of `width` bytes (1, 2 or 4), widened to u32.
    fn words(&mut self, n: usize, width: u8) -> Result<Vec<u32>> {
        let w = width as usize;
        let raw = self.take(n.checked_mul(w).ok_or("word count overflow")?)?;
        Ok((0..n)
            .map(|i| {
                let mut v = 0u32;
                for k in 0..w {
                    v |= (raw[i * w + k] as u32) << (8 * k);
                }
                v
            })
            .collect())
    }
    fn done(&self) -> Result<()> {
        if self.at != self.d.len() {
            return Err(format!("{} trailing bytes", self.d.len() - self.at));
        }
        Ok(())
    }
}

// ------------------------------------------------------------------------------------------------------ system ----

/// c + sum_j coef_j W_j, coefficients reduced mod p.
#[derive(Clone, Debug, Default)]
pub struct Expr {
    pub lin: Vec<(u32, u32)>, // (row, coef), rows strictly increasing
    pub c: u32,
}

#[derive(Clone, Debug)]
pub struct Linear {
    pub e: Expr,
    pub pub_pin: Option<usize>, // index into `pins`: sum coef W + c = pub (None: = 0)
}

/// One chain-end constraint `Y(W)[j] = value_j` at every end column (`chain.end_terms`): the whole public word as a
/// residue (`bits = 0`, a v1 system) or the bit field `(y >> lo) & (2^bits - 1)` of the public word (a v2 system).
#[derive(Clone, Debug)]
pub struct EndTerm {
    pub y: Expr,
    /// The challenge-coefficient family this constraint draws from (`chain.END_FAMILIES = (6, 0, 1)`: the end family
    /// for the single word / component 0, then the link families 0 and 1, which are unused at an end column).
    pub family: usize,
    pub lo: u32,
    pub bits: u32,
}

impl EndTerm {
    /// The public value this end term compares against, mod p.
    pub fn value(&self, y: u32) -> u32 {
        if self.bits == 0 {
            y % P
        } else {
            ((y as u64 >> self.lo) & ((1u64 << self.bits) - 1)) as u32
        }
    }
}

/// `chain.END_FAMILIES`: the coefficient family of end component x.
pub const END_FAMILIES: [usize; 3] = [6, 0, 1];

/// The most linked rows a chain may carry: 3 accumulator rows + the hashed relations' carry lanes (Poseidon2: 16; an
/// Ajtai leaf carries its whole state, hundreds of rows), bounded so a hostile file cannot size the check.
pub const MAX_LINKS: usize = 1024;

#[derive(Clone, Debug)]
pub struct Chain {
    /// The linked rows: 3 (v1 / v2 systems) or more (v3: `c_rows.len() == y.len() >= 3`).
    pub c_rows: Vec<u32>,
    pub y: Vec<Expr>,
    /// One term for a v1 system (`bits = 0`), one per component (1..=3) for a v2 system.
    pub ends: Vec<EndTerm>,
}

impl Chain {
    /// The `(lo, bits)` of every end component, or `None` for a single-word end (v1).
    pub fn components(&self) -> Option<Vec<(u32, u32)>> {
        if self.ends.len() == 1 && self.ends[0].bits == 0 {
            None
        } else {
            Some(self.ends.iter().map(|e| (e.lo, e.bits)).collect())
        }
    }
}

/// The fingerprint constraints of a `ligero-system/v4` chain (chain.py "Fingerprint constraints"; the row-sharing
/// systems of hashchain.py): `lanes[o]` the 16 lane rows of operand slot `o` (the 16-bit words a Poseidon2 sponge would
/// absorb), `coords` the number of coins rho, and the mode -- coin (rho the tail of challenge 1, F a prover message
/// after it: the GEMM side) or public (rho, F statement values: the hash side).  At the END column of every VU `v`,
/// for every `(o, x)`: `sum_{s < steps} sum_{i < 16} rho_x^(16 s + i) lanes[o][i][steps v + s] = F[v][o][x]`, an extra
/// constraint of the general linear test with the monomial coefficient of index `2 n_extra_links + o coords + x`.
#[derive(Clone, Debug)]
pub struct FpSpec {
    pub lanes: Vec<Vec<u32>>,
    pub coords: usize,
    pub public: bool,
}

pub const FP_LANES: usize = 16;
pub const MAX_FP_COORDS: usize = 32;

#[derive(Clone, Debug)]
pub struct System {
    pub m: usize,
    pub pins: Vec<(String, u32)>,
    pub linear: Vec<Linear>,
    pub quadratic: Vec<[Expr; 3]>,
    pub chain: Option<Chain>,
    /// `ligero-system/v4`: the fingerprint constraints (None otherwise).
    pub fp: Option<FpSpec>,
    pub sys_id: [u8; 32],
    /// The file format: 1 (`LIGSYS01`), 2 (`LIGSYS02`, the chain end in components), 3 (`LIGSYS03`, more than
    /// three linked rows) or 4 (`LIGSYS04`, fingerprint constraints).
    pub version: u8,
}

fn read_expr(r: &mut Rd, m: usize) -> Result<Expr> {
    let nnz = r.usize_bounded(m, "expression nnz")?;
    let mut lin = Vec::with_capacity(nnz);
    let mut last: Option<u32> = None;
    for _ in 0..nnz {
        let row = r.u32()?;
        if row as usize >= m {
            return Err(format!("expression row {row} >= m"));
        }
        if last.map_or(false, |l| row <= l) {
            return Err("expression rows not strictly increasing".into());
        }
        last = Some(row);
        let coef = r.fe()?;
        lin.push((row, coef));
    }
    let c = r.fe()?;
    Ok(Expr { lin, c })
}

impl System {
    pub fn parse(d: &[u8]) -> Result<System> {
        let mut r = Rd::new(d);
        let version: u8 = match r.take(8)? {
            b"LIGSYS01" => 1,
            b"LIGSYS02" => 2,
            b"LIGSYS03" => 3,
            b"LIGSYS04" => 4,
            _ => return Err("not a ligero-system/v1, /v2, /v3 or /v4 file".into()),
        };
        let m = r.usize_bounded(1 << 20, "m")?;
        let n_pins = r.usize_bounded(m, "n_pins")?;
        let mut pins = Vec::with_capacity(n_pins);
        for _ in 0..n_pins {
            let name = r.str()?;
            let row = r.u32()?;
            if row as usize >= m {
                return Err("pin row >= m".into());
            }
            pins.push((name, row));
        }
        let nl = r.usize_bounded(1 << 20, "L")?;
        let mut linear = Vec::with_capacity(nl);
        for _ in 0..nl {
            let e = read_expr(&mut r, m)?;
            let pp = r.u32()?;
            let pub_pin = if pp == 0xFFFF_FFFF {
                None
            } else if (pp as usize) < n_pins {
                Some(pp as usize)
            } else {
                return Err("linear constraint names an unknown pin".into());
            };
            linear.push(Linear { e, pub_pin });
        }
        let nq = r.usize_bounded(1 << 20, "Q")?;
        let mut quadratic = Vec::with_capacity(nq);
        for _ in 0..nq {
            quadratic.push([read_expr(&mut r, m)?, read_expr(&mut r, m)?, read_expr(&mut r, m)?]);
        }
        let has_chain = match r.u8()? {
            0 => false,
            1 => true,
            _ => return Err("has_chain flag not 0/1".into()),
        };
        if version >= 2 && !has_chain {
            return Err("a ligero-system/v2 or /v3 file carries no chain (the formats exist for the chain block)".into());
        }
        let chain = if has_chain {
            // v3 and v4 carry the link count (a v4 file does even with three linked rows: the GEMM side of row sharing)
            let n_links = if version >= 3 { r.usize_bounded(MAX_LINKS, "chain links")? } else { 3 };
            if n_links < 3 {
                return Err(format!("chain links {n_links} rows; a chain links at least the three accumulator rows"));
            }
            if version == 3 && n_links == 3 {
                return Err("a ligero-system/v3 file with three linked rows (Python writes those as v1 / v2)".into());
            }
            let mut c_rows = vec![0u32; n_links];
            for c in c_rows.iter_mut() {
                *c = r.u32()?;
                if *c as usize >= m {
                    return Err("chain row >= m".into());
                }
            }
            let mut y = Vec::with_capacity(n_links);
            for _ in 0..n_links {
                y.push(read_expr(&mut r, m)?);
            }
            // v3: u8 has_y16 selects the single-word end (as v1) or the component end (as v2)
            let single_word_end = match version {
                1 => true,
                2 => false,
                _ => match r.u8()? {
                    0 => false,
                    1 => true,
                    _ => return Err("has_y16 flag not 0/1".into()),
                },
            };
            let ends = if single_word_end {
                vec![EndTerm { y: read_expr(&mut r, m)?, family: END_FAMILIES[0], lo: 0, bits: 0 }]
            } else {
                // chain.end_terms: 1..=3 components, each a bit field of the public word that fits a field element
                let n_end = r.usize_bounded(END_FAMILIES.len(), "chain end components")?;
                if n_end == 0 {
                    return Err("chain end has no components".into());
                }
                let mut ends = Vec::with_capacity(n_end);
                for x in 0..n_end {
                    let y = read_expr(&mut r, m)?;
                    let lo = r.u32()?;
                    let bits = r.u32()?;
                    // a component is compared mod p, so it must be an integer below p: at most 30 bits (2^30 < p < 2^31)
                    if bits == 0 || bits > 30 || lo > 31 || lo + bits > 32 {
                        return Err(format!(
                            "chain end component {x}: bit field lo = {lo}, bits = {bits} is not a field of a 32-bit word below p"
                        ));
                    }
                    ends.push(EndTerm { y, family: END_FAMILIES[x], lo, bits });
                }
                ends
            };
            Some(Chain { c_rows, y, ends })
        } else {
            None
        };
        // v4: the fingerprint block -- n_op x 16 lane rows (distinct witness rows), coords, mode
        let fp = if version == 4 {
            let n_op = r.usize_bounded(8, "fingerprint operand slots")?;
            if n_op == 0 {
                return Err("fingerprint block names no operand slot".into());
            }
            let mut lanes = Vec::with_capacity(n_op);
            let mut seen = std::collections::HashSet::new();
            for _ in 0..n_op {
                let mut op = Vec::with_capacity(FP_LANES);
                for _ in 0..FP_LANES {
                    let row = r.u32()?;
                    if row as usize >= m {
                        return Err("fingerprint lane row >= m".into());
                    }
                    if !seen.insert(row) {
                        return Err("fingerprint lane rows repeat".into());
                    }
                    op.push(row);
                }
                lanes.push(op);
            }
            let coords = r.usize_bounded(MAX_FP_COORDS, "fingerprint coordinates")?;
            if coords == 0 {
                return Err("fingerprint block has no coordinates".into());
            }
            let public = match r.u8()? {
                0 => false,
                1 => true,
                _ => return Err("fingerprint mode not 0 (coin) / 1 (public)".into()),
            };
            Some(FpSpec { lanes, coords, public })
        } else {
            None
        };
        let sys_id: [u8; 32] = r.take(32)?.try_into().unwrap();
        r.done()?;
        Ok(System { m, pins, linear, quadratic, chain, fp, sys_id, version })
    }

    /// This verifier's own digest of the tables it will enforce (BLAKE3 of the canonical file bytes without the
    /// trailing sys_id) -- pinned in `verify.rs` for the REAL chain system, beside the Python `sys_id`.
    pub fn table_digest(d: &[u8]) -> [u8; 32] {
        crate::hash::blake3(&d[..d.len().saturating_sub(32)])
    }
}

// --------------------------------------------------------------------------------------------------- statement ----

#[derive(Clone, Debug)]
pub struct Statement {
    pub l: usize,
    pub n: usize,
    pub d: usize,
    pub t: usize,
    pub target_log2: f64,
    pub zk: bool,
    pub t_pad: usize,
    /// PROTOCOL.md 8c (b-zk-fix): `interactive` = step-0 coins committed, challenges from the coins alone;
    /// `false` = Fiat-Shamir, no coins, challenges from the transcript prefix.
    pub interactive: bool,
    pub n_proofs: usize,
    pub n_vus: usize,
    pub steps: usize,
    pub k_ops: usize,   // operands per unit (16 for the BF16 relations, 32 for E4M3)
    /// The file format: 2, 3, 4, 5 (hashed) or 6 (hashed, trimmed y words).
    pub version: u8,
    /// The relation string of a v4 file (`None`: a v2 / v3 file, the Ampere BF16 unit).
    pub relation: Option<String>,
    /// Bytes per operand word / per chain-end word in the file (2 / 2 for v2 and v3).
    pub word_bytes: u8,
    pub y_bytes: u8,
    pub a: Vec<u32>,    // l x k_ops, each < 2^(8 word_bytes)
    pub b: Vec<u32>,
    pub y16: Vec<u32>,  // l, each < 2^(8 y_bytes)
    /// `authentication = included` (v3, or v4 with `has_auth`): the committed trees and the VU indices the words are
    /// claimed under.
    pub auth: Option<AuthBlock>,
    /// `authentication = included-hash` (v5): the digests, the claimed leaf ranks, the roots and the multiproofs.  `a`
    /// is then the (l x (1 + 2 digest_elems)) public table `[is_end | x lanes | W lanes]` (zeros off the chain ends)
    /// and `b` the (n_vus x 3) index rows `(vu, x, w)` -- what the Python prover hashes into the statement digest as
    /// `<u4` words.  The leaf scheme is `hash_auth.leaf` (named by the relation string's `+<leaf>` suffix).
    pub hash_auth: Option<HashAuthBlock>,
    /// `authentication = included-hash-shared` (a v6 file, `PairStatement`): which side of the pair this is.  G: `a` is
    /// the (l x 1) table `[is_end]`, `b` the (n_vus x 3) index rows; H: `a` the (l x 10) table `[is_end | role |
    /// d0..d7]`, `b` the (n_units x 2) units `(role, rank)`, `y16` zeros, and the fingerprint (rho, F_H) enters the
    /// digest.  `hash_auth` is set on both (its digest bytes follow the side's tag).
    pub side: Option<Side>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    G,
    H,
}

/// The paired statement of row sharing (`ligero-statement/v6`, relchain.SharedHashedRunner): the GEMM side's statement,
/// the hash side's, the unit maps and the fingerprint coordinates.
#[derive(Clone, Debug)]
pub struct PairStatement {
    pub g: Statement,
    pub h: Statement,
    /// `hashchain.shared_units`: the unit hashing VU v's x row / W column.
    pub unit_of_x: Vec<usize>,
    pub unit_of_w: Vec<usize>,
    /// `(role, rank)` per unit: the distinct x ranks ascending (role 0), then the distinct W ranks (role 1).
    pub units: Vec<(u32, u32)>,
    /// The units' digests, n_units x 8.
    pub digests_h: Vec<u32>,
    pub coords: usize,
}

/// The public table of a hashed statement: 1 + 2 digest_elems columns per Ligero column (`hashchain.public_table`).
pub fn hash_table_cols(leaf: &LeafScheme) -> usize {
    1 + 2 * leaf.digest_elems
}
/// The hash side's table of a v6 statement: is_end, role, 8 digest lanes (Poseidon2 rows: `hashchain.row_hash_system`).
pub const ROW_TABLE_COLS: usize = 2 + N_ROW_DIGEST;
pub const N_ROW_DIGEST: usize = 8;
pub const AUTH_HASH: &str = "included-hash";
pub const AUTH_SHARED: &str = "included-hash-shared";

/// The longest relation string a v4 file may carry (the names are short identifiers: `fp8-ada`, `bf16-hopper`).
pub const MAX_RELATION_LEN: usize = 64;

fn read_auth_block(r: &mut Rd, n_vus: usize, steps: usize, k_ops: usize) -> Result<AuthBlock> {
    if r.str()? != AUTH_VERSION {
        return Err("unknown authentication block version".into());
    }
    if r.str()? != SCHEMA {
        return Err("unknown authentication leaf schema".into());
    }
    let n = r.usize_bounded(1 << 22, "auth n_vus")?;
    if n != n_vus {
        return Err(format!("authentication block names {n} VUs, the layout has {n_vus}"));
    }
    let raw = r.take(4 * n)?;
    let mut vu_index = Vec::with_capacity(n);
    for i in 0..n {
        let v = u32::from_le_bytes(raw[4 * i..4 * i + 4].try_into().unwrap());
        if i > 0 && v <= vu_index[i - 1] {
            return Err("authentication vu_index not strictly increasing".into());
        }
        vu_index.push(v);
    }
    let mut trees = Vec::with_capacity(3);
    for name in ["a", "b", "y"] {
        let binding = r.b32()?;
        let owner_plus_two = r.u8()?;
        if owner_plus_two > 1 {
            return Err(format!("authentication tree {name}: owner code {owner_plus_two} (only -2 / -1 are defined)"));
        }
        let count = r.u64()?;
        let root = r.b32()?;
        if count == 0 || count > MAX_POSITIONS {
            return Err(format!("authentication tree {name}: count {count} outside (0, {MAX_POSITIONS}]"));
        }
        trees.push(TreeRef { binding, owner_plus_two, count, root });
    }
    // every claimed leaf must exist: the a/b trees hold K words per VU, the y tree one word per VU
    let k = (steps * k_ops) as u64;
    let last = *vu_index.last().unwrap_or(&0) as u64;
    if (last + 1).checked_mul(k).map_or(true, |x| x > trees[0].count || x > trees[1].count) || last >= trees[2].count {
        return Err("authentication block claims leaves beyond the committed count".into());
    }
    if (n as u64) * k > MAX_OPENINGS {
        return Err(format!("authentication openings exceed the limit {MAX_OPENINGS}"));
    }
    Ok(AuthBlock { vu_index, trees: [trees[0].clone(), trees[1].clone(), trees[2].clone()] })
}

fn read_tree_refs(r: &mut Rd) -> Result<[TreeRef; 3]> {
    let mut trees = Vec::with_capacity(3);
    for name in ["a", "b", "y"] {
        let binding = r.b32()?;
        let owner_plus_two = r.u8()?;
        if owner_plus_two > 1 {
            return Err(format!("authentication tree {name}: owner code {owner_plus_two} (only -2 / -1 are defined)"));
        }
        let count = r.u64()?;
        let root = r.b32()?;
        if count == 0 || count > MAX_POSITIONS {
            return Err(format!("authentication tree {name}: count {count} outside (0, {MAX_POSITIONS}]"));
        }
        trees.push(TreeRef { binding, owner_plus_two, count, root });
    }
    Ok([trees[0].clone(), trees[1].clone(), trees[2].clone()])
}

/// The v5 block after the digests and the y words (`serialize._hash_auth_block_r`): the schema and parameters digest
/// must be the leaf scheme's the relation string named.
fn read_hash_auth_block(r: &mut Rd, n_vus: usize, digests: Vec<u32>, leaf: &'static LeafScheme) -> Result<HashAuthBlock> {
    if r.str()? != HASH_VERSION {
        return Err("unknown hash-authentication block version".into());
    }
    if r.str()? != leaf.schema {
        return Err(format!("unknown hash-authentication leaf schema (the relation names leaf scheme {:?})", leaf.name));
    }
    if r.b32()? != crate::hash::unhex32(leaf.params_sha256) {
        return Err(format!("{} parameters differ from the verifier's", leaf.display));
    }
    let n = r.usize_bounded(1 << 22, "hash-auth n_vus")?;
    if n != n_vus {
        return Err(format!("hash-authentication block names {n} VUs, the layout has {n_vus}"));
    }
    let raw = r.take(12 * n)?;
    let (mut vu_index, mut x_index, mut w_index) = (Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n));
    for i in 0..n {
        let at = |k: usize| u32::from_le_bytes(raw[12 * i + 4 * k..12 * i + 4 * k + 4].try_into().unwrap());
        let v = at(0);
        if i > 0 && v <= vu_index[i - 1] {
            return Err("hash-authentication vu_index not strictly increasing".into());
        }
        vu_index.push(v);
        x_index.push(at(1));
        w_index.push(at(2));
    }
    let trees = read_tree_refs(r)?;
    if x_index.iter().any(|&x| x as u64 >= trees[0].count) || w_index.iter().any(|&w| w as u64 >= trees[1].count)
        || vu_index.iter().any(|&v| v as u64 >= trees[2].count)
    {
        return Err("hash-authentication block claims leaves beyond the committed count".into());
    }
    let mut multiproofs = Vec::with_capacity(3);
    for (ti, tree) in trees.iter().enumerate() {
        // a canonical multiproof of `o` leaves in a tree of depth `dep` has at most min(o, 2^dep) * dep siblings
        let dep = crate::auth::merkle_depth(tree.count) as u64;
        let opened = if ti < 2 { n as u64 } else { n as u64 };
        let bound = opened.min(1u64 << dep.min(63)).saturating_mul(dep).min(MAX_PROOF_BYTES / 32).max(0);
        let n_sib = r.usize_bounded(bound as usize, "multiproof siblings")?;
        let mut sibs = Vec::with_capacity(n_sib);
        for _ in 0..n_sib {
            sibs.push(r.b32()?);
        }
        multiproofs.push(sibs);
    }
    Ok(HashAuthBlock { leaf, vu_index, x_index, w_index, trees, digests, multiproofs })
}

impl Statement {
    pub fn parse(d: &[u8]) -> Result<Statement> {
        let mut r = Rd::new(d);
        let magic = r.take(8)?;
        let version: u8 = match magic {
            b"LIGSTM02" => 2,
            b"LIGSTM03" => 3,
            b"LIGSTM04" => 4,
            b"LIGSTM05" => 5,
            b"LIGSTM06" => 6,
            _ => return Err("not a ligero-statement/v2, /v3, /v4, /v5 or /v6 file".into()),
        };
        let hashed = version >= 5;
        if r.u32()? != 1 {
            return Err("statement mode is not chain (1)".into());
        }
        let l = r.usize_bounded(1 << 22, "l")?;
        let n = r.usize_bounded(1 << 27, "n")?;
        let dd = r.usize_bounded(64, "D")?;
        let t = r.usize_bounded(1 << 16, "t")?;
        let target_log2 = r.f64()?;
        let zk = match r.u8()? {
            0 => false,
            1 => true,
            _ => return Err("zk flag not 0/1".into()),
        };
        let t_pad = r.usize_bounded(1 << 16, "t_pad")?;
        let interactive = match r.u8()? {
            0 => true,
            1 => false,
            _ => return Err("mode code not 0 (interactive) / 1 (fiat-shamir)".into()),
        };
        let n_proofs = r.usize_bounded(1 << 20, "n_proofs")?;
        let n_vus = r.usize_bounded(1 << 22, "n_vus")?;
        let steps = r.usize_bounded(1 << 16, "steps")?;
        // v2..v4: K = operands per unit; v5 / v6: K = words per committed row (steps * operands per unit)
        let k_hdr = r.usize_bounded(if hashed { 1 << 16 } else { 1 << 8 }, "K")?;
        let k_ops = if hashed {
            if steps == 0 || k_hdr % steps != 0 || k_hdr / steps == 0 || k_hdr / steps > 256 {
                return Err(format!("v{version} statement: K = {k_hdr} words per row is not steps ({steps}) x operands per unit"));
            }
            k_hdr / steps
        } else {
            k_hdr
        };
        if l == 0 || !l.is_power_of_two() || !n.is_power_of_two() || n <= l {
            return Err("l, n must be powers of two with n > l".into());
        }
        if steps == 0 || n_vus.checked_mul(steps).map_or(true, |x| x > l) {
            return Err("layout does not fit: steps * n_vus > l".into());
        }
        if dd == 0 || t == 0 || n_proofs == 0 || k_ops == 0 {
            return Err("D, t, n_proofs, K must be positive".into());
        }
        if zk != (t_pad > 0) {
            return Err("t_pad must be positive exactly under zk".into());
        }
        if !target_log2.is_finite() || target_log2 >= 0.0 {
            return Err("target_log2 must be a finite negative number".into());
        }
        if hashed {
            return Self::parse_hashed(r, version, l, n, dd, t, target_log2, zk, t_pad, interactive, n_proofs, n_vus, steps, k_ops);
        }
        // v4: the widths and the relation name come before the words; v2 / v3 are 16-bit BF16 words, unnamed
        let (word_bytes, y_bytes, relation) = if version == 4 {
            let wb = r.u8()?;
            let yb = r.u8()?;
            if !matches!(wb, 1 | 2 | 4) || !matches!(yb, 1 | 2 | 4) {
                return Err(format!("statement word widths {wb}/{yb} bytes are not 1, 2 or 4"));
            }
            let n_rel = r.usize_bounded(MAX_RELATION_LEN, "relation name length")?;
            let rel = String::from_utf8(r.take(n_rel)?.to_vec()).map_err(|_| "relation name: bad utf-8".to_string())?;
            if rel.is_empty() {
                return Err("v4 statement names no relation".into());
            }
            (wb, yb, Some(rel))
        } else {
            (2u8, 2u8, None)
        };
        let a = r.words(l * k_ops, word_bytes)?;
        let b = r.words(l * k_ops, word_bytes)?;
        let y16 = r.words(l, y_bytes)?;
        let has_auth = match version {
            2 => false,
            3 => true,
            _ => match r.u8()? {
                0 => false,
                1 => true,
                _ => return Err("has_auth flag not 0/1".into()),
            },
        };
        if has_auth && (word_bytes, y_bytes) != (2, 2) {
            return Err(format!(
                "authentication block (leaf schema {SCHEMA}) is defined for 16-bit words only; the statement has {word_bytes}/{y_bytes}-byte words"
            ));
        }
        let auth = if has_auth { Some(read_auth_block(&mut r, n_vus, steps, k_ops)?) } else { None };
        r.done()?;
        Ok(Statement {
            l, n, d: dd, t, target_log2, zk, t_pad, interactive, n_proofs, n_vus, steps, k_ops,
            version, relation, word_bytes, y_bytes, a, b, y16, auth, hash_auth: None, side: None,
        })
    }

    /// The v5 / v6 body after the header (`serialize._read_v5`): `a` becomes the derived public table, `b` the index
    /// rows.  v6 carries the n_vus chain-end words instead of the l-vector; the vector is rebuilt with zeros off the ends.
    #[allow(clippy::too_many_arguments)]
    fn parse_hashed(mut r: Rd, version: u8, l: usize, n: usize, dd: usize, t: usize, target_log2: f64, zk: bool, t_pad: usize,
                    interactive: bool, n_proofs: usize, n_vus: usize, steps: usize, k_ops: usize) -> Result<Statement> {
        let yb = r.u8()?;
        if !matches!(yb, 1 | 2 | 4) {
            return Err(format!("statement word width {yb} bytes is not 1, 2 or 4"));
        }
        let n_rel = r.usize_bounded(MAX_RELATION_LEN, "relation name length")?;
        let rel = String::from_utf8(r.take(n_rel)?.to_vec()).map_err(|_| "relation name: bad utf-8".to_string())?;
        if rel.is_empty() {
            return Err(format!("v{version} statement names no relation"));
        }
        // LIGSTM06 is two formats told apart here: the trimmed hashed statement (included-hash) and the row-sharing
        // pair (included-hash-shared, `PairStatement::parse`)
        let auth = r.str()?;
        if version == 6 && auth == AUTH_SHARED {
            return Err("a ligero-statement/v6 file (authentication = included-hash-shared) is a PAIR: verify it with `pair` / `batch-pair`".into());
        }
        if auth != AUTH_HASH {
            return Err(format!("v{version} statement names an authentication other than included-hash"));
        }
        // "<rel>" (Poseidon2) or "<rel>+<leaf>": the leaf scheme fixes the digest table's width and the block's framing
        let (_, leaf) = crate::leaf::split_relation(&rel)?;
        let nd = 2 * leaf.digest_elems;
        let digests = r.fe_vec(n_vus * nd)?;
        let y16 = if version == 6 {
            let y_words = r.words(n_vus, yb)?;
            let mut y16 = vec![0u32; l];
            for (v, &y) in y_words.iter().enumerate() {
                y16[v * steps + steps - 1] = y;
            }
            y16
        } else {
            r.words(l, yb)?
        };
        let hash_auth = read_hash_auth_block(&mut r, n_vus, digests, leaf)?;
        r.done()?;
        // the public table: is_end = 1 and the lanes at every VU's last column, 0 elsewhere (hashchain.public_table)
        let cols = hash_table_cols(leaf);
        let mut a = vec![0u32; l * cols];
        for v in 0..n_vus {
            let j = v * steps + steps - 1;
            a[j * cols] = 1;
            a[j * cols + 1..(j + 1) * cols].copy_from_slice(&hash_auth.digests[v * nd..(v + 1) * nd]);
        }
        let mut b = Vec::with_capacity(3 * n_vus);
        for v in 0..n_vus {
            b.extend_from_slice(&[hash_auth.vu_index[v], hash_auth.x_index[v], hash_auth.w_index[v]]);
        }
        Ok(Statement {
            l, n, d: dd, t, target_log2, zk, t_pad, interactive, n_proofs, n_vus, steps, k_ops,
            version, relation: Some(rel), word_bytes: 4, y_bytes: yb, a, b, y16, auth: None, hash_auth: Some(hash_auth), side: None,
        })
    }

    /// The Ligero configuration the header fixes, checked as `parse` does (`k_ops` = K / steps for the hashed layouts).
    fn read_header(r: &mut Rd, k_words_per_row: bool) -> Result<(usize, usize, usize, usize, f64, bool, usize, bool, usize, usize, usize, usize)> {
        if r.u32()? != 1 {
            return Err("statement mode is not chain (1)".into());
        }
        let l = r.usize_bounded(1 << 22, "l")?;
        let n = r.usize_bounded(1 << 27, "n")?;
        let dd = r.usize_bounded(64, "D")?;
        let t = r.usize_bounded(1 << 16, "t")?;
        let target_log2 = r.f64()?;
        let zk = match r.u8()? {
            0 => false,
            1 => true,
            _ => return Err("zk flag not 0/1".into()),
        };
        let t_pad = r.usize_bounded(1 << 16, "t_pad")?;
        let interactive = match r.u8()? {
            0 => true,
            1 => false,
            _ => return Err("mode code not 0 (interactive) / 1 (fiat-shamir)".into()),
        };
        let n_proofs = r.usize_bounded(1 << 20, "n_proofs")?;
        let n_vus = r.usize_bounded(1 << 22, "n_vus")?;
        let steps = r.usize_bounded(1 << 16, "steps")?;
        let k_hdr = r.usize_bounded(if k_words_per_row { 1 << 16 } else { 1 << 8 }, "K")?;
        let k_ops = if k_words_per_row {
            if steps == 0 || k_hdr % steps != 0 || k_hdr / steps == 0 || k_hdr / steps > 256 {
                return Err(format!("statement: K = {k_hdr} words per row is not steps ({steps}) x operands per unit"));
            }
            k_hdr / steps
        } else {
            k_hdr
        };
        Self::check_config(l, n, dd, t, target_log2, zk, t_pad, n_proofs, n_vus, steps, k_ops)?;
        Ok((l, n, dd, t, target_log2, zk, t_pad, interactive, n_proofs, n_vus, steps, k_ops))
    }

    #[allow(clippy::too_many_arguments)]
    fn check_config(l: usize, n: usize, dd: usize, t: usize, target_log2: f64, zk: bool, t_pad: usize, n_proofs: usize,
                    n_vus: usize, steps: usize, k_ops: usize) -> Result<()> {
        if l == 0 || !l.is_power_of_two() || !n.is_power_of_two() || n <= l {
            return Err("l, n must be powers of two with n > l".into());
        }
        if steps == 0 || n_vus.checked_mul(steps).map_or(true, |x| x > l) {
            return Err("layout does not fit: steps * n_vus > l".into());
        }
        if dd == 0 || t == 0 || n_proofs == 0 || k_ops == 0 {
            return Err("D, t, n_proofs, K must be positive".into());
        }
        if zk != (t_pad > 0) {
            return Err("t_pad must be positive exactly under zk".into());
        }
        if !target_log2.is_finite() || target_log2 >= 0.0 {
            return Err("target_log2 must be a finite negative number".into());
        }
        Ok(())
    }

    /// The words as 16-bit values (the authentication leaves): defined only for 2-byte words, which the parser
    /// guarantees whenever an authentication block is present.
    pub fn words_u16(words: &[u32]) -> Vec<u16> {
        words.iter().map(|&w| w as u16).collect()
    }

    pub fn k(&self) -> usize {
        self.l + self.t_pad
    }
    pub fn h_len(&self) -> usize {
        2 * self.k() - 1 - self.l
    }
    pub fn q_len(&self) -> usize {
        self.l + self.k() - 1
    }
    pub fn w_len(&self) -> usize {
        if self.zk { self.k() } else { self.l }
    }
    /// Committed rows: m witness rows + 6 D mask rows under ZK (prox, lin, qa, qb, chain_a, chain_b -- PROTOCOL.md
    /// 8b after b-zk-fix; the pre-fix protocol had one chain mask row family, `Z_H s_chain`).
    pub fn committed_rows(&self, m: usize) -> usize {
        m + if self.zk { 6 * self.d } else { 0 }
    }
    /// The registered relation the statement names, without a v5 `+<leaf>` suffix (`None`: the Ampere BF16 unit).
    pub fn base_relation(&self) -> Option<&str> {
        self.relation.as_deref().map(|r| r.split_once('+').map_or(r, |(b, _)| b))
    }

    pub fn mode_name(&self) -> &'static str {
        if self.interactive { "interactive" } else { "fiat-shamir" }
    }
    pub fn depth(&self) -> usize {
        self.n.trailing_zeros() as usize
    }
}

impl PairStatement {
    /// `serialize._read_v6`.  Both sides' statements are built here: G's table / index rows and H's table / units are
    /// derived from the block (the units from the claimed ranks, `hashchain.shared_units`; the file's roles must agree),
    /// and the per-VU digest expansion the leaf check takes is placed in `hash_auth.digests` (x unit's 8 lanes, then
    /// the W unit's).
    pub fn parse(d: &[u8]) -> Result<PairStatement> {
        let mut r = Rd::new(d);
        if r.take(8)? != b"LIGSTM06" {
            return Err("not a ligero-statement/v6 file".into());
        }
        let (l, n, dd, t, target_log2, zk, t_pad, interactive, n_proofs, n_vus, steps, k_ops) = Statement::read_header(&mut r, true)?;
        let yb = r.u8()?;
        if !matches!(yb, 1 | 2 | 4) {
            return Err(format!("statement word width {yb} bytes is not 1, 2 or 4"));
        }
        let n_rel = r.usize_bounded(MAX_RELATION_LEN, "relation name length")?;
        let rel = String::from_utf8(r.take(n_rel)?.to_vec()).map_err(|_| "relation name: bad utf-8".to_string())?;
        if rel.is_empty() {
            return Err("v6 statement names no relation".into());
        }
        if r.str()? != AUTH_SHARED {
            return Err("v6 statement names an authentication other than included-hash-shared".into());
        }
        // the rows are hashed by the Poseidon2 sponge of `hashchain.row_hash_system` (8 digest lanes): a leaf suffix other
        // than the default names a scheme the hash side does not implement
        let (_, leaf) = crate::leaf::split_relation(&rel)?;
        if leaf.digest_elems != N_ROW_DIGEST || !std::ptr::eq(leaf, &crate::leaf::POSEIDON2) {
            return Err(format!("v6 statement: row sharing hashes with Poseidon2 only, the relation names {:?}", leaf.name));
        }
        let y16 = r.words(l, yb)?;
        let l_h = r.usize_bounded(1 << 22, "l_h")?;
        let n_h = r.usize_bounded(1 << 27, "n_h")?;
        let d_h = r.usize_bounded(64, "D_h")?;
        let t_h = r.usize_bounded(1 << 16, "t_h")?;
        let t_pad_h = r.usize_bounded(1 << 16, "t_pad_h")?;
        let n_units = r.usize_bounded(1 << 22, "n_units")?;
        let coords = r.usize_bounded(MAX_FP_COORDS, "fingerprint coordinates")?;
        if n_units == 0 || coords == 0 {
            return Err("hash side: no units, or no fingerprint coordinates".into());
        }
        Statement::check_config(l_h, n_h, d_h, t_h, target_log2, zk, t_pad_h, n_proofs, n_units, steps, k_ops)?;
        if d_h != dd {
            return Err("hash side: D differs from the GEMM side's (one target, one field extension)".into());
        }
        let raw = r.fe_vec(n_units * (1 + N_ROW_DIGEST))?;
        let roles_file: Vec<u32> = (0..n_units).map(|u| raw[u * (1 + N_ROW_DIGEST)]).collect();
        let mut digests_h = Vec::with_capacity(n_units * N_ROW_DIGEST);
        for u in 0..n_units {
            digests_h.extend_from_slice(&raw[u * (1 + N_ROW_DIGEST) + 1..(u + 1) * (1 + N_ROW_DIGEST)]);
        }
        if roles_file.iter().any(|&x| x > 1) {
            return Err("hash side: a unit role is not 0 (x row) / 1 (W column)".into());
        }
        let hash_auth = read_hash_auth_block(&mut r, n_vus, Vec::new(), leaf)?;
        r.done()?;
        // the units from the block: distinct x ranks ascending (role 0), then distinct W ranks (role 1)
        let distinct = |v: &[u32]| {
            let mut u = v.to_vec();
            u.sort_unstable();
            u.dedup();
            u
        };
        let (xr, wr) = (distinct(&hash_auth.x_index), distinct(&hash_auth.w_index));
        if xr.len() + wr.len() != n_units {
            return Err(format!("hash side: the file has {n_units} units, the authentication block {} distinct rows", xr.len() + wr.len()));
        }
        let mut units: Vec<(u32, u32)> = xr.iter().map(|&x| (0u32, x)).collect();
        units.extend(wr.iter().map(|&w| (1u32, w)));
        if units.iter().zip(&roles_file).any(|(u, &rf)| u.0 != rf) {
            return Err("hash side: unit roles do not match the authentication block".into());
        }
        let unit_of_x: Vec<usize> = hash_auth.x_index.iter().map(|x| xr.binary_search(x).unwrap()).collect();
        let unit_of_w: Vec<usize> = hash_auth.w_index.iter().map(|w| xr.len() + wr.binary_search(w).unwrap()).collect();
        // the per-VU digest expansion (what `auth::check_hashed` opens under the roots)
        let mut digests = Vec::with_capacity(n_vus * 2 * N_ROW_DIGEST);
        for v in 0..n_vus {
            digests.extend_from_slice(&digests_h[unit_of_x[v] * N_ROW_DIGEST..(unit_of_x[v] + 1) * N_ROW_DIGEST]);
            digests.extend_from_slice(&digests_h[unit_of_w[v] * N_ROW_DIGEST..(unit_of_w[v] + 1) * N_ROW_DIGEST]);
        }
        let hash_auth = HashAuthBlock { digests, ..hash_auth };
        // G: [is_end] (l x 1), the index rows (n_vus x 3)
        let mut a_g = vec![0u32; l];
        for v in 0..n_vus {
            a_g[v * steps + steps - 1] = 1;
        }
        let mut b_g = Vec::with_capacity(3 * n_vus);
        for v in 0..n_vus {
            b_g.extend_from_slice(&[hash_auth.vu_index[v], hash_auth.x_index[v], hash_auth.w_index[v]]);
        }
        // H: [is_end | role | d0..d7] (l_h x 10), the units (n_units x 2), y16 = 0
        let mut a_h = vec![0u32; l_h * ROW_TABLE_COLS];
        for (u, &(role, _)) in units.iter().enumerate() {
            for s in 0..steps {
                a_h[(u * steps + s) * ROW_TABLE_COLS + 1] = role;
            }
            let j = u * steps + steps - 1;
            a_h[j * ROW_TABLE_COLS] = 1;
            a_h[j * ROW_TABLE_COLS + 2..(j + 1) * ROW_TABLE_COLS].copy_from_slice(&digests_h[u * N_ROW_DIGEST..(u + 1) * N_ROW_DIGEST]);
        }
        let b_h: Vec<u32> = units.iter().flat_map(|&(role, rank)| [role, rank]).collect();
        let g = Statement {
            l, n, d: dd, t, target_log2, zk, t_pad, interactive, n_proofs, n_vus, steps, k_ops,
            version: 6, relation: Some(rel.clone()), word_bytes: 4, y_bytes: yb, a: a_g, b: b_g, y16, auth: None,
            hash_auth: Some(hash_auth.clone()), side: Some(Side::G),
        };
        let h = Statement {
            l: l_h, n: n_h, d: d_h, t: t_h, target_log2, zk, t_pad: t_pad_h, interactive, n_proofs, n_vus: n_units, steps, k_ops,
            version: 6, relation: Some(rel), word_bytes: 4, y_bytes: yb, a: a_h, b: b_h, y16: vec![0u32; l_h], auth: None,
            hash_auth: Some(hash_auth), side: Some(Side::H),
        };
        Ok(PairStatement { g, h, unit_of_x, unit_of_w, units, digests_h, coords })
    }
}

// ------------------------------------------------------------------------------------------------------- proof ----

#[derive(Clone, Debug)]
pub struct Coins {
    pub c1: [u8; 32],
    pub c2: [u8; 32],
    pub r1: [u8; 32],
    pub s1: [u8; 32],
    pub r2: [u8; 32],
    pub s2: [u8; 32],
}

#[derive(Clone, Debug)]
pub struct Proof {
    pub coins: Option<Coins>,
    pub root: [u8; 32],
    pub w: Vec<u32>,      // D x w_len
    pub h: Vec<u32>,      // D x h_len
    pub q: Vec<u32>,      // D x q_len
    pub v: Vec<u32>,      // D x k (zk) or empty
    pub cols: Vec<u32>,   // t
    pub opened: Vec<u32>, // M x t row-major
    pub paths: Vec<[u8; 32]>, // t x depth
    /// `ligero-proof/v2`: the canonical multiproofs of the a, b, y trees (authentication = included).
    pub auth: Option<Vec<Vec<[u8; 32]>>>,
    /// `ligero-proof/v3`: the fingerprint message F (n_vus x n_op x coords, VU-major) of a coin-mode fingerprinted
    /// system (the GEMM side of row sharing).
    pub fp: Option<Vec<u32>>,
}

impl Proof {
    /// Parse with every size fixed by the statement and the system (`m`); a header that disagrees is rejected
    /// before the body is read.  A v2 transcript is only accepted against a v3 statement and vice versa.
    pub fn parse(d: &[u8], st: &Statement, m: usize) -> Result<Proof> {
        Self::parse_fp(d, st, m, None)
    }

    /// `parse` with the fingerprint message: `fp_len = Some(n)` demands a v3 transcript carrying exactly n words
    /// (the GEMM side of row sharing: n_vus x n_op x coords), `None` refuses one.
    pub fn parse_fp(d: &[u8], st: &Statement, m: usize, fp_len: Option<usize>) -> Result<Proof> {
        let mut r = Rd::new(d);
        let magic = r.take(8)?;
        let (v2, v3) = match magic {
            b"LIGPRF01" => (false, false),
            b"LIGPRF02" => (true, false),
            b"LIGPRF03" => (false, true),
            _ => return Err("not a ligero-proof/v1, /v2 or /v3 file".into()),
        };
        if v3 != fp_len.is_some() {
            return Err(if v3 { "transcript carries a fingerprint message but the system has no coin-mode fingerprint constraints" }
                       else { "the system's fingerprint constraints take a prover message (ligero-proof/v3); the transcript carries none" }.into());
        }
        if v2 != st.auth.is_some() {
            return Err(if v2 { "transcript carries multiproofs but the statement has no authentication block" }
                       else { "statement has an authentication block but the transcript carries no multiproofs" }.into());
        }
        let hash_id = r.u8()?;
        if hash_id != 1 {
            return Err(format!("proof names hash id {hash_id}; this verifier uses BLAKE3 (1)"));
        }
        let has_coins = r.u8()?;
        let has_v = r.u8()?;
        let has_q = r.u8()?;
        if has_coins > 1 || has_v > 1 || has_q > 1 {
            return Err("flag byte not 0/1".into());
        }
        // The two modes are mutually exclusive (PROTOCOL.md 8c): an interactive transcript carries the step-0
        // commitments and both openings; a Fiat-Shamir transcript must not carry coins at all.
        if st.interactive && has_coins != 1 {
            return Err("coin commitments / openings missing (interactive mode)".into());
        }
        if !st.interactive && has_coins != 0 {
            return Err("fiat-shamir transcript must not carry coins".into());
        }
        if (has_v == 1) != st.zk {
            return Err("linear-test message presence disagrees with the zk mode".into());
        }
        if has_q != 1 {
            return Err("chain transcript missing".into());
        }
        let m_rows = st.committed_rows(m);
        let expect = [
            (st.d, "D"),
            (st.w_len(), "w_len"),
            (st.h_len(), "h_len"),
            (st.q_len(), "q_len"),
            (if st.zk { st.k() } else { 0 }, "v_len"),
            (m_rows, "M"),
            (st.t, "t"),
            (st.depth(), "depth"),
        ];
        for (want, name) in expect.iter() {
            let got = r.u32()? as usize;
            if got != *want {
                return Err(format!("malformed proof: header {name} = {got}, the statement fixes {want}"));
            }
        }
        let coins = if has_coins == 1 {
            let mut c = Coins { c1: [0; 32], c2: [0; 32], r1: [0; 32], s1: [0; 32], r2: [0; 32], s2: [0; 32] };
            for slot in [&mut c.c1, &mut c.c2, &mut c.r1, &mut c.s1, &mut c.r2, &mut c.s2] {
                slot.copy_from_slice(r.take(32)?);
            }
            Some(c)
        } else {
            None
        };
        let root: [u8; 32] = r.take(32)?.try_into().unwrap();
        let w = r.fe_vec(st.d * st.w_len())?;
        let h = r.fe_vec(st.d * st.h_len())?;
        let q = r.fe_vec(st.d * st.q_len())?;
        let v = if st.zk { r.fe_vec(st.d * st.k())? } else { Vec::new() };
        let raw = r.take(4 * st.t)?;
        let mut cols = Vec::with_capacity(st.t);
        for i in 0..st.t {
            let c = u32::from_le_bytes(raw[4 * i..4 * i + 4].try_into().unwrap());
            if c as usize >= st.n {
                return Err(format!("column index {c} >= n"));
            }
            cols.push(c);
        }
        let opened = r.fe_vec(m_rows * st.t)?;
        let mut paths = Vec::with_capacity(st.t * st.depth());
        for _ in 0..st.t * st.depth() {
            paths.push(r.take(32)?.try_into().unwrap());
        }
        let auth = if let Some(ab) = &st.auth {
            // a canonical multiproof of `o` leaves in a tree of depth `dep` has at most min(o, 2^dep) * dep siblings
            let mut mps = Vec::with_capacity(3);
            for (ti, tree) in ab.trees.iter().enumerate() {
                let opened_leaves = if ti < 2 { st.n_vus * st.steps * st.k_ops } else { st.n_vus } as u64;
                let dep = crate::auth::merkle_depth(tree.count) as u64;
                let bound = opened_leaves.min(1u64 << dep.min(63)).saturating_mul(dep).min(MAX_PROOF_BYTES / 32);
                let n_sib = r.usize_bounded(bound as usize, "multiproof siblings")?;
                let mut sibs = Vec::with_capacity(n_sib);
                for _ in 0..n_sib {
                    sibs.push(r.b32()?);
                }
                mps.push(sibs);
            }
            Some(mps)
        } else {
            None
        };
        let fp = if let Some(n_fp) = fp_len {
            let got = r.u32()? as usize;
            if got != n_fp {
                return Err(format!("malformed proof: fingerprint message holds {got} words, the statement fixes {n_fp}"));
            }
            Some(r.fe_vec(n_fp)?)
        } else {
            None
        };
        r.done()?;
        Ok(Proof { coins, root, w, h, q, v, cols, opened, paths, auth, fp })
    }
}
