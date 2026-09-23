//! The relation side: ligerito-sumcheck's three sumchecks (`backends/direct/ligerito/sumcheck.py`, interface frozen
//! 18:20Z, wire `LGSC0002`) verified from the public data, producing the evaluation claims the PCS must then check.
//!
//! Committed polynomial: `w~` over `N = 2^(n_i + n_c)` cells, `x = i + R * c` (row bits low), `z[i, c]` for the committed
//! rows `i < m`, zero above; the virtual rows (`const`, chain masks, `pub:*`, `y16`, the shifted `next:x`) live in
//! `[m, R)` and are the verifier's (public rows) or the prover's (`next_vals`, tied down by the shift sumcheck).
//! Constraints `A_k(z) B_k(z) = C_k(z)` per column as sparse `(k, row, coef)` triples over `K_pad = 2^n_k` slots.
//!
//! 1. zero-check over `(k, c)`: `sum eq(tau, (k, c)) (Az Bz - Cz)(k, c) = 0`, Gruen form, the two top variables in one
//!    bivariate opening round, then one variable per round (top first); ends in `abc = (Az, Bz, Cz)(r_k, r_c)`.
//! 2. rows: `sum_i coef(i) z~(i, r_c) = abc0 + g abc1 + g^2 abc2`, `coef = eq(r_k, .)^T (A + g B + g^2 C)`; ends in
//!    `coef~(r_i) z~(r_i, r_c)` with `z~(r_i, r_c) = values[0] + sum_v eq(r_i, idx_v) virt_v(r_c)`.
//! 3. shift: `sum_c succ~(r_c, c) V(c) = sum_x g3^x next_vals[x]`, `V(c) = sum_x g3^x z[c_x, c]`; ends in
//!    `succ~(rho, r_c) sum_x g3^x values[1 + x]`.
//! Claims out: `w~(r_i || r_c) = values[0]`, `w~(bits(c_x) || rho) = values[1 + x]`.

use crate::claims::eq_vector;
use crate::field::{Ext, EXT_DEG, P};
use crate::format::Reader;
use crate::sumcheck::{eq1, eval_from_0_1_inf};
use crate::transcript::{ext_bytes, Coins};

pub const MAGIC_LGSC: &[u8; 8] = b"LGSC0002";

#[derive(Clone, Debug)]
pub struct Layout {
    pub n_i: usize,
    pub n_k: usize,
    pub n_s: usize,
    pub n_j: usize,
    /// committed rows
    pub m: usize,
    /// virtual row name -> row index in [m, 2^n_i)
    pub virt: Vec<(String, usize)>,
    /// committed row index of the linked `c` rows, x = 0..n_links
    pub c_rows: Vec<usize>,
}

impl Layout {
    pub fn n_c(&self) -> usize {
        self.n_s + self.n_j
    }
    pub fn n_links(&self) -> usize {
        self.c_rows.len()
    }
    pub fn rows(&self) -> usize {
        1 << self.n_i
    }
    pub fn cols(&self) -> usize {
        1 << self.n_c()
    }
    pub fn virt_row(&self, name: &str) -> Option<usize> {
        self.virt.iter().find(|(n, _)| n == name).map(|(_, r)| *r)
    }
    pub fn validate(&self) -> Result<(), String> {
        if self.n_i == 0 || self.n_i > 30 || self.n_k == 0 || self.n_k > 30 || self.n_c() == 0 || self.n_c() > 40 {
            return Err("layout: bits out of range".into());
        }
        if self.n_k + self.n_c() < 2 {
            return Err("layout: the zero-check needs at least two variables".into());
        }
        if self.m > self.rows() {
            return Err("layout: m exceeds 2^n_i".into());
        }
        for (name, r) in &self.virt {
            if *r < self.m || *r >= self.rows() {
                return Err(format!("layout: virtual row {name} at {r} is not in [m, R)"));
            }
        }
        for x in 0..self.n_links() {
            if self.c_rows[x] >= self.m {
                return Err(format!("layout: linked row {} is not committed", self.c_rows[x]));
            }
            if self.virt_row(&format!("next:{x}")).is_none() {
                return Err(format!("layout: no virtual row next:{x}"));
            }
        }
        Ok(())
    }
}

/// Sparse constraint matrices: `(k, row, coef)` with `k < K_pad`, `row < 2^n_i`, `coef` in the base field.
#[derive(Clone, Debug, Default)]
pub struct Constraints {
    pub k_pad: usize,
    pub a: Vec<(usize, usize, u32)>,
    pub b: Vec<(usize, usize, u32)>,
    pub c: Vec<(usize, usize, u32)>,
}

impl Constraints {
    pub fn validate(&self, lay: &Layout) -> Result<(), String> {
        if self.k_pad != 1 << lay.n_k {
            return Err(format!("constraints: K_pad {} != 2^n_k", self.k_pad));
        }
        for (side, m) in [("a", &self.a), ("b", &self.b), ("c", &self.c)] {
            for &(k, r, v) in m {
                if k >= self.k_pad || r >= lay.rows() || v >= P {
                    return Err(format!("constraints: {side} triple ({k}, {r}, {v}) out of range"));
                }
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumcheckProof {
    /// M[a][b]: the bivariate opening message (a = X1-representation index, b = X2-representation index)
    pub round0: [[Ext; 3]; 3],
    pub rounds1: Vec<[Ext; 3]>,
    pub abc: [Ext; 3],
    pub rounds2: Vec<[Ext; 3]>,
    pub next_vals: Vec<Ext>,
    pub rounds3: Vec<[Ext; 3]>,
    pub values: Vec<Ext>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EvalClaim {
    pub point: Vec<Ext>,
    pub value: Ext,
    pub name: String,
}

fn msg(r: &mut Reader) -> Result<[Ext; 3], String> {
    Ok([r.ext()?, r.ext()?, r.ext()?])
}

/// `SumcheckProof.from_bytes`; rejects non-canonical coordinates and trailing bytes.
pub fn parse_lgsc(buf: &[u8]) -> Result<SumcheckProof, String> {
    let mut r = Reader::new(buf);
    if r.bytes(8)? != MAGIC_LGSC {
        return Err("not an LGSC0002 sumcheck proof".into());
    }
    let hdr = r.bytes(5)?;
    let (n1, n2, n3, nl, nv) = (hdr[0] as usize, hdr[1] as usize, hdr[2] as usize, hdr[3] as usize, hdr[4] as usize);
    let mut round0 = [[Ext::ZERO; 3]; 3];
    for a in 0..3 {
        round0[a] = msg(&mut r)?;
    }
    let rounds1 = (0..n1).map(|_| msg(&mut r)).collect::<Result<Vec<_>, _>>()?;
    let abc = msg(&mut r)?;
    let rounds2 = (0..n2).map(|_| msg(&mut r)).collect::<Result<Vec<_>, _>>()?;
    let next_vals = r.exts(nl)?;
    let rounds3 = (0..n3).map(|_| msg(&mut r)).collect::<Result<Vec<_>, _>>()?;
    let values = r.exts(nv)?;
    if r.remaining() != 0 {
        return Err(format!("LGSC0002: {} trailing bytes", r.remaining()));
    }
    Ok(SumcheckProof { round0, rounds1, abc, rounds2, next_vals, rounds3, values })
}

pub fn write_lgsc(p: &SumcheckProof) -> Vec<u8> {
    let mut out = MAGIC_LGSC.to_vec();
    out.extend_from_slice(&[p.rounds1.len() as u8, p.rounds2.len() as u8, p.rounds3.len() as u8, p.next_vals.len() as u8, p.values.len() as u8]);
    for a in 0..3 {
        out.extend_from_slice(&ext_bytes(&p.round0[a]));
    }
    for m in &p.rounds1 {
        out.extend_from_slice(&ext_bytes(m));
    }
    out.extend_from_slice(&ext_bytes(&p.abc));
    for m in &p.rounds2 {
        out.extend_from_slice(&ext_bytes(m));
    }
    out.extend_from_slice(&ext_bytes(&p.next_vals));
    for m in &p.rounds3 {
        out.extend_from_slice(&ext_bytes(m));
    }
    out.extend_from_slice(&ext_bytes(&p.values));
    out
}

/// `q(r1, r2)` from the mixed representation `M[a][b]`.
fn q2_at(m: &[[Ext; 3]; 3], r1: &Ext, r2: &Ext) -> Ext {
    let g: Vec<Ext> = (0..3).map(|b| eval_from_0_1_inf(&m[0][b], &m[1][b], &m[2][b], r1)).collect();
    eval_from_0_1_inf(&g[0], &g[1], &g[2], r2)
}

fn q_at(m: &[Ext; 3], r: &Ext) -> Ext {
    eval_from_0_1_inf(&m[0], &m[1], &m[2], r)
}

fn flat(m: &[[Ext; 3]; 3]) -> Vec<Ext> {
    m.iter().flat_map(|row| row.iter().copied()).collect()
}

/// `< eq(r, .), vec >` for a base-field vector of length 2^|r|.
pub fn mle_base(vec: &[u32], eq: &[Ext]) -> Ext {
    let mut acc = Ext::ZERO;
    for (v, e) in vec.iter().zip(eq) {
        if *v != 0 {
            acc = acc.add(&e.mul_base(*v));
        }
    }
    acc
}

/// The MLE of the cyclic successor `[a = b + 1 mod 2^n_j on the low n_j bits, a = b on the rest]` at `(a, b)`.
pub fn succ_mle(a: &[Ext], b: &[Ext], n_j: usize) -> Ext {
    let one = Ext::ONE;
    let n = a.len();
    let eqs: Vec<Ext> = (0..n).map(|u| eq1(&a[u], &b[u])).collect();
    let mut total = Ext::ZERO;
    let mut pref = one;
    for t in 0..n_j {
        // bit t flips 0 -> 1 (b_t = 0, a_t = 1); lower bits carried (b_u = 1, a_u = 0); higher bits equal
        let mut term = pref.mul(&one.sub(&b[t]).mul(&a[t]));
        for u in t + 1..n_j {
            term = term.mul(&eqs[u]);
        }
        total = total.add(&term);
        pref = pref.mul(&b[t].mul(&one.sub(&a[t])));
    }
    total = total.add(&pref); // the wrap 1..1 -> 0..0
    for u in n_j..n {
        total = total.mul(&eqs[u]);
    }
    total
}

fn label(s: &str) -> Vec<u8> {
    s.as_bytes().to_vec()
}

/// Verify the three sumchecks.  `pub_rows`: every virtual row that is not `next:*`, as base-field vectors of length
/// `2^n_c`.  Returns the evaluation claims on the committed polynomial (the caller checks them against the commitment).
pub fn verify(lay: &Layout, cons: &Constraints, pub_rows: &[(String, Vec<u32>)], proof: &SumcheckProof, coins: &mut dyn Coins) -> Result<Vec<EvalClaim>, String> {
    lay.validate()?;
    cons.validate(lay)?;
    let (n_k, n_c, n_i, n_links) = (lay.n_k, lay.n_c(), lay.n_i, lay.n_links());
    let n = n_k + n_c;
    if proof.rounds1.len() != n - 2 || proof.rounds2.len() != n_i || proof.rounds3.len() != n_c || proof.next_vals.len() != n_links || proof.values.len() != 1 + n_links {
        return Err("sumcheck proof: shape mismatch with the layout".into());
    }
    let pub_row = |name: &str| -> Result<&Vec<u32>, String> {
        let v = pub_rows.iter().find(|(nm, _)| nm == name).map(|(_, v)| v).ok_or_else(|| format!("public row {name} missing"))?;
        if v.len() != lay.cols() {
            return Err(format!("public row {name}: {} entries, C = {}", v.len(), lay.cols()));
        }
        if v.iter().any(|x| *x >= P) {
            return Err(format!("public row {name}: non-canonical entry"));
        }
        Ok(v)
    };
    let one = Ext::ONE;
    // ---------------------------------------------------------------- 1. zero-check
    let tau = coins.challenge(b"zc/tau", n)?;
    let (tau1, tau2) = (tau[n - 1], tau[n - 2]);
    let m = &proof.round0;
    let lhs = one.sub(&tau1).mul(&one.sub(&tau2)).mul(&m[0][0])
        .add(&one.sub(&tau1).mul(&tau2).mul(&m[0][1]))
        .add(&tau1.mul(&one.sub(&tau2)).mul(&m[1][0]))
        .add(&tau1.mul(&tau2).mul(&m[1][1]));
    if !lhs.is_zero() {
        return Err("zero-check round 0: sum_{x1,x2} eq(tau, x) M[x1][x2] != 0".into());
    }
    coins.absorb(b"zc/q/0", &ext_bytes(&flat(m)));
    let r12 = coins.challenge(b"zc/r01", 2)?;
    let mut rs: Vec<Ext> = vec![r12[0], r12[1]];
    let mut e = eq1(&tau1, &r12[0]).mul(&eq1(&tau2, &r12[1]));
    let mut claim = e.mul(&q2_at(m, &r12[0], &r12[1]));
    for t in 2..n {
        let msg = &proof.rounds1[t - 2];
        let tau_t = tau[n - 1 - t];
        let lhs = e.mul(&one.sub(&tau_t).mul(&msg[0]).add(&tau_t.mul(&msg[1])));
        if lhs != claim {
            return Err(format!("zero-check round {t}: (1 - tau) q(0) + tau q(1) != claim"));
        }
        coins.absorb(&label(&format!("zc/q/{t}")), &ext_bytes(msg));
        let r = coins.challenge(&label(&format!("zc/r/{t}")), 1)?[0];
        rs.push(r);
        e = e.mul(&eq1(&tau_t, &r));
        claim = e.mul(&q_at(msg, &r));
    }
    let abc = &proof.abc;
    if e.mul(&abc[0].mul(&abc[1]).sub(&abc[2])) != claim {
        return Err("zero-check final: eq(tau, r) (Az Bz - Cz)(r) != claim".into());
    }
    coins.absorb(b"zc/abc", &ext_bytes(abc));
    rs.reverse(); // r[t] <-> variable t
    let r_c = rs[..n_c].to_vec();
    let r_k = rs[n_c..].to_vec();
    // ---------------------------------------------------------------- 2. rows
    let gamma = coins.challenge(b"rows/gamma", 1)?[0];
    let g2 = gamma.mul(&gamma);
    let mut claim = abc[0].add(&gamma.mul(&abc[1])).add(&g2.mul(&abc[2]));
    let mut rs2 = Vec::with_capacity(n_i);
    for t in 0..n_i {
        let msg = &proof.rounds2[t];
        if msg[0].add(&msg[1]) != claim {
            return Err(format!("row sumcheck round {t}: q(0) + q(1) != claim"));
        }
        coins.absorb(&label(&format!("rows/q/{t}")), &ext_bytes(msg));
        let r = coins.challenge(&label(&format!("rows/r/{t}")), 1)?[0];
        rs2.push(r);
        claim = q_at(msg, &r);
    }
    rs2.reverse();
    let r_i = rs2;
    // coef~(r_i) = sum over the triples of eq(r_k, k) eq(r_i, row) coef g^side
    let eq_k = eq_vector(&r_k);
    let eq_i = eq_vector(&r_i);
    let mut coef_r = Ext::ZERO;
    for (side, g) in [(&cons.a, one), (&cons.b, gamma), (&cons.c, g2)] {
        let mut acc = Ext::ZERO;
        for &(k, row, v) in side.iter() {
            acc = acc.add(&eq_k[k].mul(&eq_i[row]).mul_base(v));
        }
        coef_r = coef_r.add(&acc.mul(&g));
    }
    let eq_c = eq_vector(&r_c);
    let mut z_r = proof.values[0];
    for (name, row) in &lay.virt {
        if name.starts_with("next:") {
            continue;
        }
        z_r = z_r.add(&eq_i[*row].mul(&mle_base(pub_row(name)?, &eq_c)));
    }
    for x in 0..n_links {
        let row = lay.virt_row(&format!("next:{x}")).unwrap();
        z_r = z_r.add(&eq_i[row].mul(&proof.next_vals[x]));
    }
    if coef_r.mul(&z_r) != claim {
        return Err("row sumcheck final: coef(r_i) z(r_i, r_c) != claim".into());
    }
    coins.absorb(b"rows/next", &ext_bytes(&proof.next_vals));
    // ---------------------------------------------------------------- 3. shift
    let g3 = coins.challenge(b"shift/gamma", 1)?[0];
    let mut g3p = Vec::with_capacity(n_links.max(1));
    g3p.push(one);
    for x in 1..n_links {
        g3p.push(g3p[x - 1].mul(&g3));
    }
    let mut claim = Ext::ZERO;
    for x in 0..n_links {
        claim = claim.add(&g3p[x].mul(&proof.next_vals[x]));
    }
    let mut rs3 = Vec::with_capacity(n_c);
    for t in 0..n_c {
        let msg = &proof.rounds3[t];
        if msg[0].add(&msg[1]) != claim {
            return Err(format!("shift sumcheck round {t}: q(0) + q(1) != claim"));
        }
        coins.absorb(&label(&format!("shift/q/{t}")), &ext_bytes(msg));
        let r = coins.challenge(&label(&format!("shift/r/{t}")), 1)?[0];
        rs3.push(r);
        claim = q_at(msg, &r);
    }
    rs3.reverse();
    let rho = rs3;
    let succ = succ_mle(&rho, &r_c, lay.n_j);
    let mut v_r = Ext::ZERO;
    for x in 0..n_links {
        v_r = v_r.add(&g3p[x].mul(&proof.values[1 + x]));
    }
    if succ.mul(&v_r) != claim {
        return Err("shift sumcheck final: succ(rho, r_c) V(rho) != claim".into());
    }
    coins.absorb(b"values", &ext_bytes(&proof.values));
    // ---------------------------------------------------------------- claims
    let mut claims = Vec::with_capacity(1 + n_links);
    let mut p0 = r_i.clone();
    p0.extend_from_slice(&r_c);
    claims.push(EvalClaim { point: p0, value: proof.values[0], name: "w(r_i, r_c)".into() });
    for (x, &cr) in lay.c_rows.iter().enumerate() {
        let mut pt: Vec<Ext> = (0..n_i).map(|t| Ext::from_base(((cr >> t) & 1) as u32)).collect();
        pt.extend_from_slice(&rho);
        claims.push(EvalClaim { point: pt, value: proof.values[1 + x], name: format!("w(c_{x}, rho)") });
    }
    Ok(claims)
}

// ---------------------------------------------------------------------------------------------------------- fixture

/// ligerito-sumcheck's `sumcheck.fixture()` JSON: layout, constraints, public rows, proof, claims, recorded transcript.
pub struct Fixture {
    pub layout: Layout,
    pub constraints: Constraints,
    pub pub_rows: Vec<(String, Vec<u32>)>,
    pub proof: Vec<u8>,
    pub claims: Vec<EvalClaim>,
    pub transcript: Vec<crate::transcript::Recorded>,
}

pub fn hex_bytes(s: &str) -> Result<Vec<u8>, String> {
    if s.len() % 2 != 0 {
        return Err("hex: odd length".into());
    }
    (0..s.len() / 2).map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).map_err(|e| format!("hex: {e}"))).collect()
}

fn ext_of_json(v: &crate::json::Json) -> Result<Ext, String> {
    let c = v.usize_list().ok_or("fixture: ext element")?;
    if c.len() != EXT_DEG {
        return Err("fixture: ext element has the wrong degree".into());
    }
    let mut e = [0u32; EXT_DEG];
    for (i, x) in c.iter().enumerate() {
        if *x >= P as usize {
            return Err("fixture: non-canonical coordinate".into());
        }
        e[i] = *x as u32;
    }
    Ok(Ext(e))
}

pub fn load_fixture(text: &str) -> Result<Fixture, String> {
    use crate::json::Json;
    use crate::transcript::Recorded;
    let d = crate::json::parse(text)?;
    let get = |o: &Json, k: &str| -> Result<Json, String> { o.get(k).cloned().ok_or_else(|| format!("fixture: missing {k}")) };
    let lay = get(&d, "layout")?;
    let usize_of = |o: &Json, k: &str| -> Result<usize, String> { get(o, k)?.as_usize().ok_or_else(|| format!("fixture: {k}")) };
    let virt = match get(&lay, "virt")? {
        Json::Obj(kv) => kv.iter().map(|(k, v)| v.as_usize().map(|r| (k.clone(), r)).ok_or("fixture: virt row")).collect::<Result<Vec<_>, _>>()?,
        _ => return Err("fixture: virt".into()),
    };
    let layout = Layout {
        n_i: usize_of(&lay, "n_i")?,
        n_k: usize_of(&lay, "n_k")?,
        n_s: usize_of(&lay, "n_s")?,
        n_j: usize_of(&lay, "n_j")?,
        m: usize_of(&lay, "m")?,
        virt,
        c_rows: get(&lay, "c_rows")?.usize_list().ok_or("fixture: c_rows")?,
    };
    let cj = get(&d, "constraints")?;
    let side = |k: &str| -> Result<Vec<(usize, usize, u32)>, String> {
        let arr = get(&cj, k)?;
        let arr = arr.as_arr().ok_or("fixture: coo")?;
        if arr.len() != 3 {
            return Err("fixture: coo needs [ks, rows, coefs]".into());
        }
        let ks = arr[0].usize_list().ok_or("fixture: coo ks")?;
        let rs = arr[1].usize_list().ok_or("fixture: coo rows")?;
        let vs = arr[2].usize_list().ok_or("fixture: coo coefs")?;
        if ks.len() != rs.len() || ks.len() != vs.len() {
            return Err("fixture: coo lengths".into());
        }
        Ok((0..ks.len()).map(|i| (ks[i], rs[i], (vs[i] % P as usize) as u32)).collect())
    };
    let constraints = Constraints { k_pad: usize_of(&cj, "K_pad")?, a: side("a")?, b: side("b")?, c: side("c")? };
    let pub_rows = match get(&d, "public_rows")? {
        Json::Obj(kv) => kv
            .iter()
            .map(|(k, v)| v.usize_list().map(|r| (k.clone(), r.into_iter().map(|x| (x % P as usize) as u32).collect())).ok_or("fixture: public row"))
            .collect::<Result<Vec<_>, _>>()?,
        _ => return Err("fixture: public_rows".into()),
    };
    let proof = hex_bytes(get(&d, "proof_hex")?.as_str().ok_or("fixture: proof_hex")?)?;
    let claims = get(&d, "claims")?
        .as_arr()
        .ok_or("fixture: claims")?
        .iter()
        .map(|c| {
            let point = get(c, "point")?.as_arr().ok_or("fixture: point")?.iter().map(ext_of_json).collect::<Result<Vec<_>, _>>()?;
            Ok(EvalClaim { point, value: ext_of_json(&get(c, "value")?)?, name: get(c, "name")?.as_str().unwrap_or("").to_string() })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let transcript = match d.get("transcript") {
        Some(t) => t
            .as_arr()
            .ok_or("fixture: transcript")?
            .iter()
            .map(|e| {
                if let Some(l) = e.get("absorb") {
                    Ok(Recorded::Absorb { label: l.as_str().ok_or("fixture: absorb label")?.as_bytes().to_vec(), data: hex_bytes(get(e, "bytes")?.as_str().ok_or("fixture: absorb bytes")?)? })
                } else {
                    let l = get(e, "challenge")?;
                    let values = get(e, "values")?.as_arr().ok_or("fixture: challenge values")?.iter().map(ext_of_json).collect::<Result<Vec<_>, _>>()?;
                    Ok(Recorded::Challenge { label: l.as_str().ok_or("fixture: challenge label")?.as_bytes().to_vec(), values })
                }
            })
            .collect::<Result<Vec<_>, String>>()?,
        None => Vec::new(),
    };
    Ok(Fixture { layout, constraints, pub_rows, proof, claims, transcript })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn succ_mle_matches_the_dense_definition() {
        // sum_c eq(a, c) eq(b, pred(c)) over n_c = n_s + n_j bits, pred cyclic within the low n_j bits
        let mut seed = 0x9e3779b97f4a7c15u64;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            Ext::from_base((seed % P as u64) as u32).mul(&Ext([1, 2, 3, 4, 5, 6]))
        };
        let (n_s, n_j) = (2usize, 3usize);
        let a: Vec<Ext> = (0..n_s + n_j).map(|_| rnd()).collect();
        let b: Vec<Ext> = (0..n_s + n_j).map(|_| rnd()).collect();
        let ea = eq_vector(&a);
        let eb = eq_vector(&b);
        let l = 1 << n_j;
        let mut dense = Ext::ZERO;
        for c in 0..(1 << (n_s + n_j)) {
            let (s, j) = (c / l, c % l);
            let pred = s * l + (j + l - 1) % l;
            dense = dense.add(&ea[c].mul(&eb[pred]));
        }
        assert_eq!(succ_mle(&a, &b, n_j), dense);
    }

    /// ligerito-sumcheck's fixture (0bc9d89, `sumcheck.fixture()`: fp8-ada, l = 64, S = 2, `LocalCoins(b"fixture")`):
    /// accept with the claims the Python verifier returned; every byte flip of the proof and a corrupted public row /
    /// constraint coefficient / layout are rejected.
    #[test]
    fn sumcheck_fixture_fp8ada_toy() {
        use crate::transcript::SumcheckLocal;
        let fx = load_fixture(include_str!("../fixtures/sumcheck/fp8ada_l64_S2.json")).unwrap();
        let proof = parse_lgsc(&fx.proof).unwrap();
        assert_eq!(write_lgsc(&proof), fx.proof);
        let run = |lay: &Layout, cons: &Constraints, pubs: &[(String, Vec<u32>)], pf: &SumcheckProof| {
            verify(lay, cons, pubs, pf, &mut SumcheckLocal::new(b"fixture"))
        };
        let claims = run(&fx.layout, &fx.constraints, &fx.pub_rows, &proof).unwrap();
        assert_eq!(claims, fx.claims);
        assert_eq!(claims.len(), 4);
        assert_eq!(claims[0].point.len(), 19);
        // proof byte flips (every 7th byte after the header, plus each header byte)
        for pos in (0..13).chain((13..fx.proof.len()).step_by(7)) {
            let mut b = fx.proof.clone();
            b[pos] ^= 1;
            let r = parse_lgsc(&b).and_then(|pf| run(&fx.layout, &fx.constraints, &fx.pub_rows, &pf));
            assert!(r.is_err(), "accepted a flipped bit at byte {pos}");
        }
        // a public row entry
        let mut pubs = fx.pub_rows.clone();
        let idx = pubs.iter().position(|(n, _)| n == "y16").unwrap();
        pubs[idx].1[3] ^= 1;
        assert!(run(&fx.layout, &fx.constraints, &pubs, &proof).is_err());
        // a constraint coefficient
        let mut cons = fx.constraints.clone();
        cons.b[100].2 = (cons.b[100].2 + 1) % P;
        assert!(run(&fx.layout, &cons, &fx.pub_rows, &proof).is_err());
        // a linked row index only moves the claim's point (the PCS check is what binds it)
        let mut lay = fx.layout.clone();
        lay.c_rows[1] += 1;
        let moved = run(&lay, &fx.constraints, &fx.pub_rows, &proof).unwrap();
        assert_ne!(moved[2].point, fx.claims[2].point);
        assert_eq!(moved[0], fx.claims[0]);
        // the virtual row of a public vector
        let mut lay = fx.layout.clone();
        let k = lay.virt.iter().position(|(n, _)| n == "y16").unwrap();
        lay.virt[k].1 ^= 1;
        assert!(run(&lay, &fx.constraints, &fx.pub_rows, &proof).is_err());
        // another seed = another verifier: rejected
        assert!(verify(&fx.layout, &fx.constraints, &fx.pub_rows, &proof, &mut SumcheckLocal::new(b"other")).is_err());
    }

    #[test]
    fn lgsc_rejects_garbage() {
        assert!(parse_lgsc(b"LGSC0001xxxxx").is_err());
        assert!(parse_lgsc(b"LGSC0002").is_err());
        let mut b = MAGIC_LGSC.to_vec();
        b.extend_from_slice(&[0, 0, 0, 0, 0]);
        b.extend_from_slice(&[0u8; 9 * 24]);
        b.extend_from_slice(&[0u8; 3 * 24]);
        let p = parse_lgsc(&b).unwrap();
        assert_eq!(p.rounds1.len(), 0);
        b.push(0);
        assert!(parse_lgsc(&b).is_err());
        // non-canonical coordinate
        let mut b2 = MAGIC_LGSC.to_vec();
        b2.extend_from_slice(&[0, 0, 0, 0, 0]);
        b2.extend_from_slice(&P.to_le_bytes());
        b2.extend_from_slice(&[0u8; 9 * 24 + 3 * 24 - 4]);
        assert!(parse_lgsc(&b2).is_err());
    }
}
