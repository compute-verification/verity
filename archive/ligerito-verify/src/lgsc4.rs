//! LGSC0004: ligerito-sumcheck-3's ZK sumcheck block (`archive/direct/ligerito/sumcheck.py` `_verify_zk` on
//! `lane/ligerito-sumcheck-3`, spec in that lane's report `## Interface (LGSC0004)`).
//!
//! Same zero-check and combined sumcheck as LGSC0003 ([`crate::lgsc3`]), made zero-knowledge:
//! * each sumcheck carries a Libra mask `ghat = sum_R (g_R(x_R) - base_R)` whose coefficients (`3^v - 1` per round group:
//!   every monomial of degree `<= 2` per variable but the constant) sit in the committed `g_rows`; the verifier's only
//!   duty is the linear functional `K = sum_R (g_R(r_R) - base_R)`, a sparse PCS claim on those cells;
//! * the zero-check's final carries `T` (its mask's claim) and no `next:x` tables: the shifted rows are COMMITTED
//!   (`zk.next`, plus the mask link `zk.mask -> zk.mask_next`), so the combined final charges
//!   `n_w~(r_i) = sum_x g3^x eq(r_i, next_x) + g3^L eq(r_i, mask_next)` and one value `V'(rho_c) = s_w~ w~(., rho_c)`;
//! * a third, row-reduction sumcheck over `n_i` variables reduces `V'` to one evaluation `w~(rho_i, rho_c)`.
//!
//! Coins: LGSC0003's, then per row-reduction round `absorb("rb/q/{t}")` + `challenge("rb/r/{t}", v)`, `absorb("values_b")`.
//! Claims: `w~(r_i, r_c)`, `w~(rho_i, rho_c)`, then the three sparse mask claims (zero-check, combined, row reduction).

use crate::claims::eq_vector;
use crate::field::{inv, pow, Ext, EXT_DEG};
use crate::format::Reader;
use crate::lgsc3::{bool_sum_mixed, eq_sum_mixed, eval_mixed, mle_ext, pow3};
use crate::relation::{succ_mle, Constraints, EvalClaim, Layout, PublicRows};
use crate::sumcheck::eq1;
use crate::transcript::{ext_bytes, Coins};

pub const MAGIC: &[u8; 8] = b"LGSC0004";
/// message rounds (sumcheck-3 ef49a7d's coin-lean default: `3^6` ext per message)
pub const MAX_ARITY: usize = 6;
/// the zero-check's final table round (`3 x 2^vf` ext)
pub const MAX_VF: usize = 12;
/// `layout.ZK_B`: `2^ZK_B` uniform rows
pub const ZK_B: usize = 3;
/// `layout.ZK_CELLS_PER_VAR`: Libra cells reserved per sumcheck variable (arity 6: `(3^6 - 1) / 6 < 122` ext)
pub const ZK_CELLS_PER_VAR: usize = 6 * 122;

/// `layout.ZkRows`: the ZK layout's committed rows above the virtual block.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ZkRows {
    /// committed shift of the linked row `c_x`, `x < n_links`
    pub next: Vec<usize>,
    pub mask: usize,
    pub mask_next: usize,
    /// `i1 * i2 = i3` (constraint `k_product`): masks the zero-check's `Az, Bz, Cz` tables
    pub product: [usize; 3],
    pub k_product: usize,
    pub uniform: Vec<usize>,
    /// the Libra coefficients, 6 cells each, row after row
    pub g_rows: Vec<usize>,
    pub g_cells: usize,
}

/// `sum_t weights[t] w[rows[t], cols[t]] = value` on base cells of the committed polynomial (distinct cells, all in
/// `zk.g_rows`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SparseClaim {
    pub rows: Vec<usize>,
    pub cols: Vec<usize>,
    pub weights: Vec<Ext>,
    pub value: Ext,
    pub name: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Claims {
    pub eval: Vec<EvalClaim>,
    pub sparse: Vec<SparseClaim>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Proof {
    pub zc: Vec<usize>,
    pub vf: usize,
    pub n_links: usize,
    pub cmb: Vec<usize>,
    pub rb: Vec<usize>,
    pub zc_rounds: Vec<Vec<Ext>>,
    pub final_abc: [Vec<Ext>; 3],
    /// the zero-check mask's claim
    pub t: Ext,
    pub cmb_rounds: Vec<Vec<Ext>>,
    /// `w~(r_i, r_c)`, `V'(rho_c)`, the combined mask's claim
    pub values: [Ext; 3],
    pub rb_rounds: Vec<Vec<Ext>>,
    /// `w~(rho_i, rho_c)`, the row-reduction mask's claim
    pub values_b: [Ext; 2],
}

impl Proof {
    /// Magic .. rb arities: absorbed verbatim as `lgsc/header`.
    pub fn header(&self) -> Vec<u8> {
        let mut h = MAGIC.to_vec();
        h.push(self.zc.len() as u8);
        h.extend(self.zc.iter().map(|&a| a as u8));
        h.push(self.vf as u8);
        h.push(self.n_links as u8);
        h.push(self.cmb.len() as u8);
        h.extend(self.cmb.iter().map(|&a| a as u8));
        h.push(self.rb.len() as u8);
        h.extend(self.rb.iter().map(|&a| a as u8));
        h
    }

    /// `Schedule.zk_coeffs`: Libra coefficients (ext elements) the three masks use.
    pub fn coeffs(&self) -> usize {
        self.zc.iter().chain(&self.cmb).chain(&self.rb).map(|&v| pow3(v) - 1).sum()
    }
}

/// `SumcheckProof._from_bytes_zk`: message arities in `1..=6`, `vf` in `1..=12`, every coordinate `< p`, no trailing bytes.
pub fn parse(buf: &[u8]) -> Result<Proof, String> {
    let mut r = Reader::new(buf);
    if r.bytes(8)? != MAGIC {
        return Err("not an LGSC0004 sumcheck proof".into());
    }
    let mut u8s = |n: usize| -> Result<Vec<usize>, String> { Ok(r.bytes(n)?.iter().map(|&b| b as usize).collect()) };
    let z = u8s(1)?[0];
    let zc = u8s(z)?;
    let h = u8s(2)?;
    let (vf, n_links) = (h[0], h[1]);
    let q = u8s(1)?[0];
    let cmb = u8s(q)?;
    let b = u8s(1)?[0];
    let rb = u8s(b)?;
    if zc.iter().chain(&cmb).chain(&rb).any(|&a| !(1..=MAX_ARITY).contains(&a)) || !(1..=MAX_VF).contains(&vf) {
        return Err("LGSC0004: arity out of range".into());
    }
    let zc_rounds = zc.iter().map(|&v| r.exts(pow3(v))).collect::<Result<Vec<_>, _>>()?;
    let final_abc = [r.exts(1 << vf)?, r.exts(1 << vf)?, r.exts(1 << vf)?];
    let t = r.ext()?;
    let cmb_rounds = cmb.iter().map(|&v| r.exts(pow3(v))).collect::<Result<Vec<_>, _>>()?;
    let v = r.exts(3)?;
    let rb_rounds = rb.iter().map(|&v| r.exts(pow3(v))).collect::<Result<Vec<_>, _>>()?;
    let vb = r.exts(2)?;
    if r.remaining() != 0 {
        return Err(format!("LGSC0004: {} trailing bytes", r.remaining()));
    }
    Ok(Proof { zc, vf, n_links, cmb, rb, zc_rounds, final_abc, t, cmb_rounds, values: [v[0], v[1], v[2]], rb_rounds, values_b: [vb[0], vb[1]] })
}

pub fn write(p: &Proof) -> Vec<u8> {
    let mut out = p.header();
    for m in p.zc_rounds.iter().chain(p.final_abc.iter()) {
        out.extend(ext_bytes(m));
    }
    out.extend(ext_bytes(&[p.t]));
    for m in &p.cmb_rounds {
        out.extend(ext_bytes(m));
    }
    out.extend(ext_bytes(&p.values));
    for m in &p.rb_rounds {
        out.extend(ext_bytes(m));
    }
    out.extend(ext_bytes(&p.values_b));
    out
}

/// `layout.layout_for(..., zk=True)`'s ZK rows from the rest of the layout, top down: `2^ZK_B` uniform rows, the Libra g rows
/// (`ZK_CELLS_PER_VAR` cells per variable of the three sumchecks), the product triple, the mask link, the committed next
/// rows; `n_i` must leave room for them above the virtual block (`layout_for` grows it; the verifier only checks).  `k`: the
/// number of constraints, the last one the product check.
pub fn zk_rows_for(lay: &Layout, k: usize) -> Result<ZkRows, String> {
    let (n_c, n_links) = (lay.n_c(), lay.n_links());
    let n_vars = lay.n_k + n_c + lay.n_i.max(n_c) + lay.n_i;
    let g_cells = ZK_CELLS_PER_VAR * n_vars;
    let n_g = g_cells.div_ceil(lay.cols()).max(1);
    let free_from = lay.m + lay.virt.len();
    if k == 0 || free_from + n_links + 2 + 3 + (1 << ZK_B) + n_g > lay.rows() {
        return Err("ZK layout: no room for the ZK rows above the virtual block".into());
    }
    let mut top = lay.rows();
    let uniform: Vec<usize> = (top - (1 << ZK_B)..top).collect();
    top -= 1 << ZK_B;
    let g_rows: Vec<usize> = (top - n_g..top).collect();
    top -= n_g;
    let product = [top - 3, top - 2, top - 1];
    top -= 3;
    let (mask, mask_next) = (top - 2, top - 1);
    top -= 2;
    Ok(ZkRows { next: (top - n_links..top).collect(), mask, mask_next, product, k_product: k - 1, uniform, g_rows, g_cells })
}

/// The layout checks LGSC0004 relies on: [`Layout::validate_rows`], no `next:x` virtual rows, every ZK row committed
/// above `m`, off the virtual block and distinct, and constraint `k_product` exactly `i1 * i2 = i3`.
pub fn validate_layout(lay: &Layout, zk: &ZkRows, cons: &Constraints) -> Result<(), String> {
    lay.validate_rows()?;
    if lay.virt.iter().any(|(n, _)| n.starts_with("next:")) {
        return Err("ZK layout: next rows are committed, not virtual".into());
    }
    if zk.next.len() != lay.n_links() {
        return Err("ZK layout: one committed next row per link".into());
    }
    let rows: Vec<usize> = zk.next.iter().chain([&zk.mask, &zk.mask_next]).chain(&zk.product).chain(&zk.uniform).chain(&zk.g_rows).copied().collect();
    let mut seen = std::collections::HashSet::new();
    for &r in &rows {
        if r < lay.m || r >= lay.rows() || lay.virt.iter().any(|(_, v)| *v == r) || !seen.insert(r) {
            return Err(format!("ZK layout: row {r} is not a distinct committed row above the relation"));
        }
    }
    let [i1, i2, i3] = zk.product;
    let k = zk.k_product;
    let only = |side: &[(usize, usize, u32)], row: usize| {
        let at: Vec<_> = side.iter().filter(|e| e.0 == k).collect();
        at.len() == 1 && at[0].1 == row && at[0].2 == 1
    };
    if k >= cons.k_pad || !only(&cons.a, i1) || !only(&cons.b, i2) || !only(&cons.c, i3) {
        return Err(format!("ZK layout: constraint {k} is not the product row check i1 * i2 = i3"));
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum Kind {
    /// zero-check: base = `sum_b eq(tau, b) g(b)`
    Eq,
    /// plain sums: base = `2^-v sum_b g(b)`
    Avg,
}

/// `ZkMask` on the verifier's side: the weights of `K` in the coefficients, group by group.
struct Mask {
    kind: Kind,
    offset: usize,
    w: Vec<Ext>,
}

/// `_outer`: `W[e] = prod_j ws[j][e_j]`, axis 0 outermost.
fn outer(ws: &[[Ext; 3]]) -> Vec<Ext> {
    let mut w = vec![Ext::ONE];
    for a in ws {
        w = w.iter().flat_map(|x| a.iter().map(move |y| x.mul(y))).collect();
    }
    w
}

impl Mask {
    fn new(kind: Kind, offset: usize) -> Mask {
        Mask { kind, offset, w: Vec::new() }
    }

    /// `ZkMask.bind`: `K += g_t(r) - base_t`, i.e. coefficient `e != 0` gets `prod_j r_j^e_j - base weight`.
    fn bind(&mut self, r: &[Ext], taus: &[Ext]) {
        let at = outer(&r.iter().map(|x| [Ext::ONE, *x, x.mul(x)]).collect::<Vec<_>>());
        let base = match self.kind {
            Kind::Eq => outer(&taus.iter().map(|t| [Ext::ONE, *t, *t]).collect::<Vec<_>>()),
            Kind::Avg => {
                let h = Ext::from_base(inv(2));
                outer(&vec![[Ext::ONE, h, h]; r.len()])
            }
        };
        self.w.extend(at.iter().zip(&base).skip(1).map(|(a, b)| a.sub(b)));
    }

    fn n(&self) -> usize {
        self.w.len()
    }

    /// `ZkMask.claim`: coordinate `d` of coefficient `u` is g-block cell `6 (offset + u) + d`, weight `W_u x^d`.
    fn claim(&self, zk: &ZkRows, n_cols: usize, value: Ext, name: &str) -> SparseClaim {
        let n = EXT_DEG * self.w.len();
        let (mut rows, mut cols, mut weights) = (Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n));
        for (u, wu) in self.w.iter().enumerate() {
            for d in 0..EXT_DEG {
                let cell = EXT_DEG * (self.offset + u) + d;
                let mut xd = Ext::ZERO;
                xd.0[d] = 1;
                rows.push(zk.g_rows[cell / n_cols]);
                cols.push(cell % n_cols);
                weights.push(wu.mul(&xd));
            }
        }
        SparseClaim { rows, cols, weights, value, name: name.into() }
    }
}

/// Verify the three sumchecks of an LGSC0004 proof; returns the two evaluation claims and the three sparse mask claims
/// the PCS must then check on the commitment.
pub fn verify(lay: &Layout, zk: &ZkRows, cons: &Constraints, pub_rows: &dyn PublicRows, proof: &Proof, coins: &mut dyn Coins) -> Result<Claims, String> {
    validate_layout(lay, zk, cons)?;
    cons.validate(lay)?;
    let (n_k, n_c, n_i, n_links) = (lay.n_k, lay.n_c(), lay.n_i, lay.n_links());
    let n = n_k + n_c;
    let n_cmb = n_i.max(n_c);
    let vf = proof.vf;
    if proof.zc.iter().sum::<usize>() + vf != n
        || proof.cmb.iter().sum::<usize>() != n_cmb
        || proof.rb.iter().sum::<usize>() != n_i
        || vf > n_c
        || proof.zc.is_empty()
        || proof.rb.is_empty()
        || proof.n_links != n_links
    {
        return Err("schedule does not fit the layout".into());
    }
    if EXT_DEG * proof.coeffs() > zk.g_cells.min(zk.g_rows.len() * lay.cols()) {
        return Err("schedule needs more Libra coefficients than the layout reserves".into());
    }
    let n_zc_w: usize = proof.zc.iter().map(|&v| pow3(v) - 1).sum();
    let n_cmb_w: usize = proof.cmb.iter().map(|&v| pow3(v) - 1).sum();
    let (mut mz, mut mc, mut mr) = (Mask::new(Kind::Eq, 0), Mask::new(Kind::Avg, n_zc_w), Mask::new(Kind::Avg, n_zc_w + n_cmb_w));
    coins.absorb(b"lgsc/header", &proof.header());
    let tau = coins.challenge(b"zc/tau", n)?;
    // ---------------------------------------------------------------- zero-check (+ its mask: T)
    let mut e = Ext::ONE;
    let mut claim = Ext::ZERO;
    let mut rs: Vec<Ext> = Vec::with_capacity(n);
    let mut nrem = n;
    for (t, (&v, m)) in proof.zc.iter().zip(&proof.zc_rounds).enumerate() {
        let taus: Vec<Ext> = (0..v).map(|j| tau[nrem - 1 - j]).collect();
        if e.mul(&eq_sum_mixed(m, &taus)) != claim {
            return Err(format!("zero-check round {t}: sum_b eq(tau, b) q(b) != claim"));
        }
        coins.absorb(format!("zc/q/{t}").as_bytes(), &ext_bytes(m));
        let r = coins.challenge(format!("zc/r/{t}").as_bytes(), v)?;
        mz.bind(&r, &taus);
        for j in 0..v {
            e = e.mul(&eq1(&taus[j], &r[j]));
        }
        claim = e.mul(&eval_mixed(m, &r));
        rs.extend_from_slice(&r);
        nrem -= v;
    }
    let fabc = &proof.final_abc;
    let prod: Vec<Ext> = (0..1 << vf).map(|b| fabc[0][b].mul(&fabc[1][b]).sub(&fabc[2][b])).collect();
    if e.mul(&mle_ext(&prod, &eq_vector(&tau[..vf])).add(&proof.t)) != claim {
        return Err("zero-check final: sum_b eq(tau, b) (Az Bz - Cz)(b) + T != claim".into());
    }
    let mut fin = Vec::new();
    for tab in fabc.iter() {
        fin.extend(ext_bytes(tab));
    }
    fin.extend(ext_bytes(&[proof.t]));
    coins.absorb(b"zc/final", &fin);
    let ch = coins.challenge(b"zc/final", vf + 3)?;
    let (gamma, g3, beta) = (ch[vf], ch[vf + 1], ch[vf + 2]);
    rs.extend_from_slice(&ch[..vf]);
    rs.reverse(); // r_var[t] <-> variable t
    let (r_c, r_k) = (rs[..n_c].to_vec(), rs[n_c..].to_vec());
    let eqf = eq_vector(&rs[..vf]);
    let abc: Vec<Ext> = fabc.iter().map(|tab| mle_ext(tab, &eqf)).collect();
    // ---------------------------------------------------------------- combined rows + committed shift (+ mask)
    let g2 = gamma.mul(&gamma);
    let mut g3p = vec![Ext::ONE];
    for x in 1..=n_links {
        g3p.push(g3p[x - 1].mul(&g3));
    }
    let rep = |d: usize| Ext::from_base(pow(2, d as u64));
    let (rep_i, rep_c) = (rep(n_cmb - n_i), rep(n_cmb - n_c));
    let rows_claim = abc[0].add(&gamma.mul(&abc[1])).add(&g2.mul(&abc[2]));
    let mut claim = beta.mul(&rows_claim).mul(&rep_i).mul(&rep_c);
    let mut rs2: Vec<Ext> = Vec::with_capacity(n_cmb);
    for (t, (&v, m)) in proof.cmb.iter().zip(&proof.cmb_rounds).enumerate() {
        if bool_sum_mixed(m, v) != claim {
            return Err(format!("combined round {t}: sum_b q(b) != claim"));
        }
        coins.absorb(format!("cmb/q/{t}").as_bytes(), &ext_bytes(m));
        let r = coins.challenge(format!("cmb/r/{t}").as_bytes(), v)?;
        mc.bind(&r, &[]);
        claim = eval_mixed(m, &r);
        rs2.extend_from_slice(&r);
    }
    rs2.reverse();
    let (r_i, rho_c) = (rs2[..n_i].to_vec(), rs2[..n_c].to_vec());
    let (eq_k, eq_i) = (eq_vector(&r_k), eq_vector(&r_i));
    let mut coef_r = Ext::ZERO;
    for (side, g) in [(&cons.a, Ext::ONE), (&cons.b, gamma), (&cons.c, g2)] {
        let acc = side.iter().fold(Ext::ZERO, |acc, &(k, row, v)| acc.add(&eq_k[k].mul(&eq_i[row]).mul_base(v)));
        coef_r = coef_r.add(&acc.mul(&g));
    }
    let n_w = (0..n_links).fold(g3p[n_links].mul(&eq_i[zk.mask_next]), |acc, x| acc.add(&g3p[x].mul(&eq_i[zk.next[x]])));
    let coef_r = beta.mul(&coef_r).add(&n_w);
    let eq_c = eq_vector(&r_c);
    let mut z_r = proof.values[0];
    for (name, row) in &lay.virt {
        z_r = z_r.add(&eq_i[*row].mul(&pub_rows.mle(name, &eq_c)?));
    }
    let succ = succ_mle(&rho_c, &r_c, lay.n_j);
    if coef_r.mul(&z_r).mul(&rep_c).sub(&succ.mul(&proof.values[1]).mul(&rep_i)).add(&proof.values[2]) != claim {
        return Err("combined final: (beta coef + n_w)(r_i) z(r_i, r_c) - succ(rho, r_c) V'(rho) + mask != claim".into());
    }
    coins.absorb(b"values", &ext_bytes(&proof.values));
    // ---------------------------------------------------------------- row reduction: V'(rho_c) = sum_i s_w(i) w(i, rho_c)
    let mut claim = proof.values[1];
    let mut rs3: Vec<Ext> = Vec::with_capacity(n_i);
    for (t, (&v, m)) in proof.rb.iter().zip(&proof.rb_rounds).enumerate() {
        if bool_sum_mixed(m, v) != claim {
            return Err(format!("row-reduction round {t}: sum_b q(b) != claim"));
        }
        coins.absorb(format!("rb/q/{t}").as_bytes(), &ext_bytes(m));
        let r = coins.challenge(format!("rb/r/{t}").as_bytes(), v)?;
        mr.bind(&r, &[]);
        claim = eval_mixed(m, &r);
        rs3.extend_from_slice(&r);
    }
    rs3.reverse();
    let rho_i = rs3;
    let eq_rho = eq_vector(&rho_i);
    let s_w = (0..n_links).fold(g3p[n_links].mul(&eq_rho[zk.mask]), |acc, x| acc.add(&g3p[x].mul(&eq_rho[lay.c_rows[x]])));
    let vb = &proof.values_b;
    if s_w.mul(&vb[0]).add(&vb[1]) != claim {
        return Err("row-reduction final: s(rho_i) w(rho_i, rho_c) + mask != claim".into());
    }
    coins.absorb(b"values_b", &ext_bytes(vb));
    debug_assert_eq!(mz.n() + mc.n() + mr.n(), proof.coeffs());
    let mut p0 = r_i;
    p0.extend_from_slice(&r_c);
    let mut p1 = rho_i;
    p1.extend_from_slice(&rho_c);
    let c = lay.cols();
    Ok(Claims {
        eval: vec![
            EvalClaim { point: p0, value: proof.values[0], name: "w(r_i, r_c)".into() },
            EvalClaim { point: p1, value: vb[0], name: "w(rho_i, rho_c)".into() },
        ],
        sparse: vec![mz.claim(zk, c, proof.t, "mask(zc)"), mc.claim(zk, c, proof.values[2], "mask(cmb)"), mr.claim(zk, c, vb[1], "mask(rb)")],
    })
}

/// A fixture's `layout.zk`.
pub fn zk_rows_of_json(j: &crate::json::Json) -> Result<ZkRows, String> {
    let list = |k: &str| j.get(k).and_then(|v| v.usize_list()).ok_or(format!("layout.zk: {k}"));
    let one = |k: &str| j.get(k).and_then(|v| v.as_usize()).ok_or(format!("layout.zk: {k}"));
    let product: [usize; 3] = list("product")?.try_into().map_err(|_| "layout.zk: product needs 3 rows")?;
    Ok(ZkRows {
        next: list("next")?,
        mask: one("mask")?,
        mask_next: one("mask_next")?,
        product,
        k_product: one("k_product")?,
        uniform: list("uniform")?,
        g_rows: list("g_rows")?,
        g_cells: one("g_cells")?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::P;
    use crate::json::Json;
    use crate::lgsc3::fixture_negatives;
    use crate::relation::{ext_of_json, load_fixture};
    use crate::transcript::{Scripted, SumcheckLocal};

    /// ligerito-sumcheck-3 ef49a7d: the coin-lean default schedule, and the `zk-small` preset (21 coins at full size).
    const DEFAULT: &str = include_str!("../fixtures/sumcheck/lgsc0004_fp8ada_l64_S2.json");
    const SMALL: &str = include_str!("../fixtures/sumcheck/lgsc0004_fp8ada_l64_S2_zksmall.json");

    struct Fx {
        text: &'static str,
        base: crate::relation::Fixture,
        zk: ZkRows,
        k: usize,
        sparse: Vec<SparseClaim>,
        g_cells: Vec<u32>,
    }

    fn load(text: &'static str) -> Fx {
        let base = load_fixture(text).unwrap();
        let d = crate::json::parse(text).unwrap();
        let lay = d.get("layout").unwrap();
        let zk = zk_rows_of_json(lay.get("zk").unwrap()).unwrap();
        let k = lay.get("K").unwrap().as_usize().unwrap();
        let sparse = d
            .get("claims")
            .unwrap()
            .as_arr()
            .unwrap()
            .iter()
            .filter(|c| c.get("kind").and_then(Json::as_str) == Some("sparse"))
            .map(|c| SparseClaim {
                rows: c.get("rows").unwrap().usize_list().unwrap(),
                cols: c.get("cols").unwrap().usize_list().unwrap(),
                weights: c.get("weights").unwrap().as_arr().unwrap().iter().map(|w| ext_of_json(w).unwrap()).collect(),
                value: ext_of_json(c.get("value").unwrap()).unwrap(),
                name: c.get("name").unwrap().as_str().unwrap().into(),
            })
            .collect();
        let g_cells = d.get("zk_g_cells").unwrap().usize_list().unwrap().into_iter().map(|x| (x % P as usize) as u32).collect();
        Fx { text, base, zk, k, sparse, g_cells }
    }

    fn run(fx: &Fx, pf: &Proof, seed: &[u8]) -> Result<Claims, String> {
        verify(&fx.base.layout, &fx.zk, &fx.base.constraints, &fx.base.pub_rows, pf, &mut SumcheckLocal::new(seed))
    }

    /// Each fixture (fp8-ada, l = 64, S = 2, `--zk` layout, `LocalCoins(b"fixture")`): the ZK rows are the ones
    /// `zk_rows_for` derives; accepted with the recorded transcript (every absorb byte-checked) and with the local coins; the
    /// two evaluation claims and the three sparse mask claims (rows, cols, weights, value) equal the Python verifier's, and
    /// every sparse claim holds on the fixture's g cells; the 8 recorded negatives rejected in the same stage as Python.
    fn check_fixture(fx: &Fx, schedule: (Vec<usize>, usize, Vec<usize>, Vec<usize>), coins: usize) {
        assert_eq!(zk_rows_for(&fx.base.layout, fx.k).unwrap(), fx.zk);
        let proof = parse(&fx.base.proof).unwrap();
        assert_eq!(write(&proof), fx.base.proof);
        assert_eq!((proof.zc.clone(), proof.vf, proof.cmb.clone(), proof.rb.clone()), schedule);
        assert_eq!(EXT_DEG * proof.coeffs(), fx.g_cells.len());
        let mut sc = Scripted::new(fx.base.transcript.clone());
        let claims = verify(&fx.base.layout, &fx.zk, &fx.base.constraints, &fx.base.pub_rows, &proof, &mut sc).unwrap();
        sc.finish().unwrap();
        assert_eq!(claims.eval, fx.base.claims);
        assert_eq!(claims.sparse, fx.sparse);
        let mut lc = SumcheckLocal::new(b"fixture");
        let again = verify(&fx.base.layout, &fx.zk, &fx.base.constraints, &fx.base.pub_rows, &proof, &mut lc).unwrap();
        assert_eq!(again, claims);
        assert_eq!(lc.rounds(), coins);
        let c = fx.base.layout.cols();
        for s in &claims.sparse {
            let lhs = (0..s.rows.len()).fold(Ext::ZERO, |acc, t| {
                let g = fx.zk.g_rows.iter().position(|&r| r == s.rows[t]).unwrap();
                acc.add(&s.weights[t].mul_base(fx.g_cells[g * c + s.cols[t]]))
            });
            assert_eq!(lhs, s.value, "{} does not hold on the committed g cells", s.name);
        }
        let negs = fixture_negatives(fx.text).unwrap();
        assert_eq!(negs.len(), 8);
        for (bytes, why) in &negs {
            let got = parse(bytes).and_then(|pf| run(fx, &pf, b"fixture"));
            let stage = why.split(':').next().unwrap();
            assert!(got.as_ref().is_err_and(|e| e.starts_with(stage)), "{got:?} vs python {why}");
        }
        assert!(run(fx, &proof, b"other").is_err());
    }

    #[test]
    fn lgsc0004_fixture_default_schedule() {
        check_fixture(&load(DEFAULT), (vec![3, 5, 4, 4], 3, vec![6, 6], vec![6, 6]), 10);
    }

    #[test]
    fn lgsc0004_fixture_zk_small() {
        check_fixture(&load(SMALL), (vec![3; 5], 4, vec![3; 4], vec![4; 3]), 14);
    }

    /// Byte flips anywhere in the proof, a corrupted public row / constraint coefficient, a moved ZK row, a product
    /// constraint that is not `i1 * i2 = i3`, and schedule / parser limits: all rejected.
    #[test]
    fn lgsc0004_tampering() {
        let fx = load(SMALL);
        let proof = parse(&fx.base.proof).unwrap();
        for pos in (0..24).chain((24..fx.base.proof.len()).step_by(13)) {
            let mut b = fx.base.proof.clone();
            b[pos] ^= 1;
            assert!(parse(&b).and_then(|pf| run(&fx, &pf, b"fixture")).is_err(), "accepted a flipped bit at byte {pos}");
        }
        let lay = &fx.base.layout;
        let v = |zk: &ZkRows, cons: &Constraints, pubs: &Vec<(String, Vec<u32>)>| verify(lay, zk, cons, pubs, &proof, &mut SumcheckLocal::new(b"fixture"));
        let mut pubs = fx.base.pub_rows.clone();
        let idx = pubs.iter().position(|(n, _)| n == "y16").unwrap();
        pubs[idx].1[3] ^= 1;
        assert!(v(&fx.zk, &fx.base.constraints, &pubs).is_err());
        let mut cons = fx.base.constraints.clone();
        cons.a[7].2 = (cons.a[7].2 + 1) % P;
        assert!(v(&fx.zk, &cons, &fx.base.pub_rows).is_err());
        // the verifier's own row weights: moving a committed next / mask row changes n_w / s_w
        for f in [|z: &mut ZkRows| z.next[1] = z.uniform[0], |z: &mut ZkRows| z.mask_next = z.uniform[1], |z: &mut ZkRows| z.mask = z.uniform[2]] {
            let mut zk = fx.zk.clone();
            f(&mut zk);
            zk.uniform.retain(|r| *r != zk.next[1] && *r != zk.mask && *r != zk.mask_next);
            let e = v(&zk, &fx.base.constraints, &fx.base.pub_rows).unwrap_err();
            assert!(e.starts_with("combined final") || e.starts_with("row-reduction final"), "{e}");
        }
        let mut zk = fx.zk.clone();
        zk.mask = zk.next[0];
        assert!(v(&zk, &fx.base.constraints, &fx.base.pub_rows).unwrap_err().starts_with("ZK layout"));
        let mut zk = fx.zk.clone();
        zk.g_rows[0] = lay.virt[0].1;
        assert!(v(&zk, &fx.base.constraints, &fx.base.pub_rows).unwrap_err().starts_with("ZK layout"));
        let mut cons = fx.base.constraints.clone();
        let at = cons.c.iter().position(|e| e.0 == fx.zk.k_product).unwrap();
        cons.c[at].1 = fx.zk.product[0];
        assert!(v(&fx.zk, &cons, &fx.base.pub_rows).unwrap_err().starts_with("ZK layout"));
        let mut zk = fx.zk.clone();
        zk.g_cells = EXT_DEG * proof.coeffs() - 1;
        assert!(v(&zk, &fx.base.constraints, &fx.base.pub_rows).unwrap_err().starts_with("schedule needs more Libra"));
        let mut p2 = proof.clone();
        p2.rb.pop();
        p2.rb_rounds.pop();
        assert!(run(&fx, &p2, b"fixture").unwrap_err().starts_with("schedule"));
        let mut p2 = proof.clone();
        p2.n_links += 1;
        assert!(run(&fx, &p2, b"fixture").unwrap_err().starts_with("schedule"));
        let mut b = fx.base.proof.clone();
        b.push(0);
        assert!(parse(&b).is_err());
        assert!(parse(&fx.base.proof[..fx.base.proof.len() - 1]).is_err());
        let rb_arity_at = 8 + 1 + proof.zc.len() + 2 + 1 + proof.cmb.len() + 1;
        for bad in [0u8, 7] {
            let mut b = fx.base.proof.clone();
            b[rb_arity_at] = bad;
            assert!(parse(&b).is_err());
        }
        let vf_at = 8 + 1 + proof.zc.len();
        for bad in [0u8, 13] {
            let mut b = fx.base.proof.clone();
            b[vf_at] = bad;
            assert!(parse(&b).is_err());
        }
        let hdr = proof.header().len();
        let mut b = fx.base.proof.clone();
        b[hdr..hdr + 4].copy_from_slice(&P.to_le_bytes());
        assert!(parse(&b).is_err());
        // the LGSC0003 parser does not take it, and vice versa
        assert!(crate::lgsc3::parse(&fx.base.proof).is_err());
        // zk_rows_for: a layout without room above the virtual block
        let mut tight = lay.clone();
        tight.m = lay.rows() - lay.virt.len() - 10;
        assert!(zk_rows_for(&tight, fx.k).is_err());
    }
}
