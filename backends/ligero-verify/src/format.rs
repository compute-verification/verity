//! The three files (`backends/direct/ligero/serialize.py` defines them): `ligero-system/v1`, `ligero-statement/v2`
//! (`/v3` = v2 + the authentication block, `authentication = included`; `/v4` = the relation-named statement: byte
//! widths of the words, the relation string, then an optional v3 block), `ligero-proof/v1` (`/v2` = v1 + one
//! multiproof per committed tree).  Reject-first: every length is bounded by the statement before a byte of the
//! proof body is read, every field element is checked canonical (< p) at parse.
//!
//! ~~~text
//! ligero-statement/v4  magic "LIGSTM04" | the v2 header through u32 K | u8 word_bytes | u8 y_bytes | str relation
//!                      | l x K a | l x K b | l x y16 (u8 / u16 / u32 per the byte widths) | u8 has_auth | [v3 block]
//! ~~~
//! The words are kept as `u32` whatever their width; the relation the file names decides (relation.rs) which widths
//! are legal, how the words enter the statement digest and how the operands decode.  The authentication block's leaf
//! schema is `u16`, so a v4 file may carry one only with 16-bit words (Python defines nothing else).

use crate::auth::{AuthBlock, TreeRef, MAX_OPENINGS, MAX_POSITIONS, MAX_PROOF_BYTES, SCHEMA, VERSION as AUTH_VERSION};
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

#[derive(Clone, Debug)]
pub struct Chain {
    pub c_rows: [u32; 3],
    pub y: [Expr; 3],
    pub y16: Expr,
}

#[derive(Clone, Debug)]
pub struct System {
    pub m: usize,
    pub pins: Vec<(String, u32)>,
    pub linear: Vec<Linear>,
    pub quadratic: Vec<[Expr; 3]>,
    pub chain: Option<Chain>,
    pub sys_id: [u8; 32],
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
        if r.take(8)? != b"LIGSYS01" {
            return Err("not a ligero-system/v1 file".into());
        }
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
        let chain = if r.u8()? == 1 {
            let mut c_rows = [0u32; 3];
            for c in c_rows.iter_mut() {
                *c = r.u32()?;
                if *c as usize >= m {
                    return Err("chain row >= m".into());
                }
            }
            let y = [read_expr(&mut r, m)?, read_expr(&mut r, m)?, read_expr(&mut r, m)?];
            let y16 = read_expr(&mut r, m)?;
            Some(Chain { c_rows, y, y16 })
        } else {
            None
        };
        let sys_id: [u8; 32] = r.take(32)?.try_into().unwrap();
        r.done()?;
        Ok(System { m, pins, linear, quadratic, chain, sys_id })
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
    /// The file format: 2, 3 or 4.
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
}

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

impl Statement {
    pub fn parse(d: &[u8]) -> Result<Statement> {
        let mut r = Rd::new(d);
        let magic = r.take(8)?;
        let version: u8 = match magic {
            b"LIGSTM02" => 2,
            b"LIGSTM03" => 3,
            b"LIGSTM04" => 4,
            _ => return Err("not a ligero-statement/v2, /v3 or /v4 file".into()),
        };
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
        let k_ops = r.usize_bounded(1 << 8, "K")?;
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
            version, relation, word_bytes, y_bytes, a, b, y16, auth,
        })
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
    pub fn mode_name(&self) -> &'static str {
        if self.interactive { "interactive" } else { "fiat-shamir" }
    }
    pub fn depth(&self) -> usize {
        self.n.trailing_zeros() as usize
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
}

impl Proof {
    /// Parse with every size fixed by the statement and the system (`m`); a header that disagrees is rejected
    /// before the body is read.  A v2 transcript is only accepted against a v3 statement and vice versa.
    pub fn parse(d: &[u8], st: &Statement, m: usize) -> Result<Proof> {
        let mut r = Rd::new(d);
        let magic = r.take(8)?;
        let v2 = match magic {
            b"LIGPRF01" => false,
            b"LIGPRF02" => true,
            _ => return Err("not a ligero-proof/v1 or /v2 file".into()),
        };
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
        r.done()?;
        Ok(Proof { coins, root, w, h, q, v, cols, opened, paths, auth })
    }
}
