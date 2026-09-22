//! The three files (`backends/direct/ligero/serialize.py` defines them): `ligero-system/v1`, `ligero-statement/v1`,
//! `ligero-proof/v1`.  Reject-first: every length is bounded by the statement before a byte of the proof body is
//! read, every field element is checked canonical (< p) at parse.

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
    fn u16_vec(&mut self, n: usize) -> Result<Vec<u16>> {
        let raw = self.take(2 * n)?;
        Ok((0..n).map(|i| u16::from_le_bytes([raw[2 * i], raw[2 * i + 1]])).collect())
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
    pub coin_commitment: bool,
    pub n_proofs: usize,
    pub n_vus: usize,
    pub steps: usize,
    pub k_ops: usize,   // operands per unit (16)
    pub a: Vec<u16>,    // l x k_ops
    pub b: Vec<u16>,
    pub y16: Vec<u16>,  // l
}

impl Statement {
    pub fn parse(d: &[u8]) -> Result<Statement> {
        let mut r = Rd::new(d);
        if r.take(8)? != b"LIGSTM01" {
            return Err("not a ligero-statement/v1 file".into());
        }
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
        let coin_commitment = match r.u8()? {
            0 => false,
            1 => true,
            _ => return Err("coin_commitment flag not 0/1".into()),
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
        let a = r.u16_vec(l * k_ops)?;
        let b = r.u16_vec(l * k_ops)?;
        let y16 = r.u16_vec(l)?;
        r.done()?;
        Ok(Statement { l, n, d: dd, t, target_log2, zk, t_pad, coin_commitment, n_proofs, n_vus, steps, k_ops, a, b, y16 })
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
    /// Committed rows: m witness rows + 5 D mask rows under ZK (prox, lin, qa, qb, chain).
    pub fn committed_rows(&self, m: usize) -> usize {
        m + if self.zk { 5 * self.d } else { 0 }
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
}

impl Proof {
    /// Parse with every size fixed by the statement and the system (`m`); a header that disagrees is rejected
    /// before the body is read.
    pub fn parse(d: &[u8], st: &Statement, m: usize) -> Result<Proof> {
        let mut r = Rd::new(d);
        if r.take(8)? != b"LIGPRF01" {
            return Err("not a ligero-proof/v1 file".into());
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
        // With the commitment off (HVZK mode) the Python prover still writes c_i / r_i / s_i into the transcript and
        // the Python verifier ignores them (DISCREPANCIES.md D4): they are parsed and never used.
        if st.coin_commitment && has_coins != 1 {
            return Err("coin commitments / openings missing (the verifier's mode commits the coins)".into());
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
        r.done()?;
        Ok(Proof { coins, root, w, h, q, v, cols, opened, paths })
    }
}
