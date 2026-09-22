//! Data-parallel GKR over the layered checker (PROTOCOL.md 4.1, Libra-style two-phase wiring).
//!
//! Layer claim:  Σ_{g,c} w(g) eq(γ,c) V_out(g,c) = v,  V_out(g,c) = A_g(V_in(·,c)) · B_g(V_in(·,c)).
//! Phase 1 (b copy variables, degree 3): Σ_c eq(γ,c) Σ_g w(g) Ã_g(c) B̃_g(c)          → c*
//! Phase 2a (s_in variables, degree 2): Σ_l Ṽ(l) P(l),  P(l) = E* Σ_g w(g) B̃_g(c*) a_g[l]  → l*, sends Ṽ(l*), P̃(l*)
//! Phase 2b (s_in variables, degree 2): Σ_r Ṽ(r) R(r),  R(r) = E* Σ_g w(g) Ã_g(l*) b_g[r]  → r*, sends Ṽ(r*)
//! where Ṽ(x) = Ṽ_in(x, c*) and E* = eq(γ, c*).  The verifier evaluates R̃(r*) itself from the public wiring.
//! Two claims on V_in at (l*,c*), (r*,c*) merge with a fresh α into the next layer's w.

use crate::circuit::{Gate, Layer};
use crate::field::{eq_table, eval_mle, fold_in_place, interpolate, Ext, Fp};
use crate::transcript::Transcript;

pub struct Proof { pub msgs: Vec<Ext> }

pub struct Reader<'a> { pub msgs: &'a [Ext], pub pos: usize }
impl<'a> Reader<'a> {
    pub fn next(&mut self) -> Result<Ext, String> {
        let v = self.msgs.get(self.pos).copied().ok_or_else(|| "proof truncated".to_string())?;
        self.pos += 1;
        Ok(v)
    }
    pub fn take(&mut self, n: usize) -> Result<Vec<Ext>, String> { (0..n).map(|_| self.next()).collect() }
}

/// The values of a layer's input space for copy `c`: a slice of length >= 2^s_in (entries past the real
/// width are zero by construction of the exporter's padding).
pub trait LayerInput { fn row(&self, c: usize) -> &[Fp]; fn copies(&self) -> usize; }

pub struct Strided<'a> { pub data: &'a [Fp], pub width: usize, pub copies: usize }
impl<'a> LayerInput for Strided<'a> {
    fn row(&self, c: usize) -> &[Fp] { if c < self.copies { &self.data[c * self.width..(c + 1) * self.width] } else { &[] } }
    fn copies(&self) -> usize { self.copies }
}

fn eval_terms_base(terms: &[crate::circuit::Term], row: &[Fp]) -> Fp {
    let mut acc = Fp::ZERO;
    for t in terms { if let Some(v) = row.get(t.idx as usize) { acc += t.coef * *v; } }
    acc
}

fn eval_terms_ext(terms: &[crate::circuit::Term], tab: &[Ext]) -> Ext {
    let mut acc = Ext::ZERO;
    for t in terms { acc += tab[t.idx as usize].scale(t.coef); }
    acc
}

pub fn eq_point(a: &[Ext], b: &[Ext]) -> Ext {
    let mut e = Ext::ONE;
    for (x, y) in a.iter().zip(b) { e = e * (*x * *y + (Ext::ONE - *x) * (Ext::ONE - *y)); }
    e
}

pub struct LayerClaim { pub w: Vec<Ext>, pub copy_point: Vec<Ext>, pub value: Ext }

pub fn threads() -> usize {
    if let Ok(s) = std::env::var("VERITY_GKR_THREADS") { if let Ok(n) = s.parse::<usize>() { return n.max(1); } }
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1).min(16)
}

/// Prover for one layer; returns the next layer's claim (or, for the last layer, the two raw claims).
pub fn prove_layer(layer: &Layer, input: &(dyn LayerInput + Sync), b: usize, claim: &LayerClaim, tr: &mut Transcript, out: &mut Vec<Ext>, last: bool) -> (LayerClaim, Option<(Vec<Ext>, Ext, Vec<Ext>, Ext)>) {
    let g_n = layer.gates.len();
    let copies = 1usize << b;
    // phase 1 tables
    let mut a = vec![Ext::ZERO; copies * g_n];
    let mut bb = vec![Ext::ZERO; copies * g_n];
    {
        let nthreads = threads();
        let chunk = copies.div_ceil(nthreads);
        std::thread::scope(|s| {
            for (ci, (ac, bc)) in a.chunks_mut(chunk * g_n).zip(bb.chunks_mut(chunk * g_n)).enumerate() {
                let gates = &layer.gates;
                s.spawn(move || {
                    for (k, (arow, brow)) in ac.chunks_mut(g_n).zip(brow_chunks(bc, g_n)).enumerate() {
                        let c = ci * chunk + k;
                        let row = input.row(c);
                        if row.is_empty() { continue; }
                        for (g, gate) in gates.iter().enumerate() {
                            arow[g] = Ext::from_base(eval_terms_base(&gate.a, row));
                            brow[g] = Ext::from_base(eval_terms_base(&gate.b, row));
                        }
                    }
                });
            }
        });
    }
    let mut e = eq_table(&claim.copy_point);
    let w = &claim.w;
    let mut c_star = Vec::with_capacity(b);
    let mut len = copies;
    for _round in 0..b {
        let half = len / 2;
        // h(t) = Σ_{c<half} E_t[c] · Σ_g w[g] (a0 + t da)(b0 + t db)
        let nthreads = threads();
        let chunk = half.div_ceil(nthreads).max(1);
        let mut partial = vec![[Ext::ZERO; 4]; nthreads];
        std::thread::scope(|s| {
            for (ti, acc) in partial.iter_mut().enumerate() {
                let (a, bb, e) = (&a, &bb, &e);
                s.spawn(move || {
                    let lo = ti * chunk;
                    let hi = ((ti + 1) * chunk).min(half);
                    for c in lo..hi {
                        let (mut k0, mut k1, mut k2) = (Ext::ZERO, Ext::ZERO, Ext::ZERO);
                        let r0 = c * g_n;
                        let r1 = (c + half) * g_n;
                        for g in 0..g_n {
                            let a0 = a[r0 + g];
                            let da = a[r1 + g] - a0;
                            let b0 = bb[r0 + g];
                            let db = bb[r1 + g] - b0;
                            let wg = w[g];
                            k0 += wg * (a0 * b0);
                            k1 += wg * (a0 * db + da * b0);
                            k2 += wg * (da * db);
                        }
                        let e0 = e[c];
                        let de = e[c + half] - e0;
                        for t in 0..4u64 {
                            let tt = Ext::from_u64(t);
                            let inner = k0 + tt * (k1 + tt * k2);
                            acc[t as usize] += (e0 + tt * de) * inner;
                        }
                    }
                });
            }
        });
        let mut h = [Ext::ZERO; 4];
        for p in &partial { for t in 0..4 { h[t] += p[t]; } }
        out.extend_from_slice(&h);
        tr.absorb_exts(&h);
        let r = tr.challenge();
        c_star.push(r);
        // fold a, bb, e
        fold_rows(&mut a, g_n, half, r);
        fold_rows(&mut bb, g_n, half, r);
        fold_in_place(&mut e, r);
        len = half;
    }
    let e_star = e[0];
    let a_star = &a[..g_n];
    let b_star = &bb[..g_n];
    // Ṽ(·) = Σ_c eq(c*, c) V_in(·, c)
    let n_in = 1usize << layer.s_in;
    let ec = eq_table(&c_star);
    let mut vf = vec![Ext::ZERO; n_in];
    for c in 0..copies {
        let row = input.row(c);
        if row.is_empty() { continue; }
        let ecc = ec[c];
        for (l, v) in row.iter().enumerate().take(n_in) { if v.0 != 0 { vf[l] += ecc.scale(*v); } }
    }
    let wp: Vec<Ext> = w.iter().map(|x| *x * e_star).collect();
    // phase 2a
    let mut p = vec![Ext::ZERO; n_in];
    for (g, gate) in layer.gates.iter().enumerate() {
        let wb = wp[g] * b_star[g];
        for t in &gate.a { p[t.idx as usize] += wb.scale(t.coef); }
    }
    let _ = a_star;
    let mut v2 = vf.clone();
    let l_star = sumcheck_prod(&mut vf, &mut p, tr, out);
    let vl = vf[0];
    let pl = p[0];
    out.push(vl); out.push(pl);
    tr.absorb_exts(&[vl, pl]);
    // phase 2b
    let el = eq_table(&l_star);
    let mut rr = vec![Ext::ZERO; n_in];
    for (g, gate) in layer.gates.iter().enumerate() {
        let wa = wp[g] * eval_terms_ext(&gate.a, &el);
        for t in &gate.b { rr[t.idx as usize] += wa.scale(t.coef); }
    }
    let r_star = sumcheck_prod(&mut v2, &mut rr, tr, out);
    let vr = v2[0];
    out.push(vr);
    tr.absorb_ext(vr);
    if last {
        return (LayerClaim { w: vec![], copy_point: c_star.clone(), value: Ext::ZERO }, Some((l_star, vl, r_star, vr)));
    }
    let alpha = tr.challenge();
    let er = eq_table(&r_star);
    let nw: Vec<Ext> = el.iter().zip(&er).map(|(x, y)| *x + alpha * *y).collect();
    (LayerClaim { w: nw, copy_point: c_star, value: vl + alpha * vr }, None)
}

fn brow_chunks(b: &mut [Ext], g_n: usize) -> std::slice::ChunksMut<'_, Ext> { b.chunks_mut(g_n) }

fn fold_rows(v: &mut Vec<Ext>, width: usize, half: usize, r: Ext) {
    for c in 0..half {
        for g in 0..width {
            let lo = v[c * width + g];
            let hi = v[(c + half) * width + g];
            v[c * width + g] = lo + (hi - lo) * r;
        }
    }
    v.truncate(half * width);
}

/// Degree-2 sumcheck of Σ_x U(x) W(x); folds both in place; returns the bound point.
fn sumcheck_prod(u: &mut Vec<Ext>, w: &mut Vec<Ext>, tr: &mut Transcript, out: &mut Vec<Ext>) -> Vec<Ext> {
    let mut point = Vec::new();
    while u.len() > 1 {
        let half = u.len() / 2;
        let (mut s0, mut s1, mut s2) = (Ext::ZERO, Ext::ZERO, Ext::ZERO);
        for i in 0..half {
            let (u0, u1, w0, w1) = (u[i], u[i + half], w[i], w[i + half]);
            s0 += u0 * w0;
            s1 += u1 * w1;
            s2 += (u1 + u1 - u0) * (w1 + w1 - w0);
        }
        out.extend_from_slice(&[s0, s1, s2]);
        tr.absorb_exts(&[s0, s1, s2]);
        let r = tr.challenge();
        point.push(r);
        fold_in_place(u, r);
        fold_in_place(w, r);
    }
    point
}

/// Verifier for one layer.  Returns the next claim, or the two input claims for the last layer.
pub fn verify_layer(layer: &Layer, b: usize, claim: &LayerClaim, tr: &mut Transcript, rd: &mut Reader, last: bool) -> Result<(LayerClaim, Option<(Vec<Ext>, Ext, Vec<Ext>, Ext)>), String> {
    let mut running = claim.value;
    let mut c_star = Vec::with_capacity(b);
    for round in 0..b {
        let h = rd.take(4)?;
        if h[0] + h[1] != running { return Err(format!("{}: phase-1 round {round} sum mismatch", layer.name)); }
        tr.absorb_exts(&h);
        let r = tr.challenge();
        running = interpolate(&h, r);
        c_star.push(r);
    }
    let e_star = eq_point(&claim.copy_point, &c_star);
    let wp: Vec<Ext> = claim.w.iter().map(|x| *x * e_star).collect();
    let (l_star, after_l) = verify_sumcheck_prod(layer.s_in, running, tr, rd, &format!("{} phase-2a", layer.name))?;
    let vl = rd.next()?;
    let pl = rd.next()?;
    if vl * pl != after_l { return Err(format!("{}: phase-2a final check", layer.name)); }
    tr.absorb_exts(&[vl, pl]);
    let (r_star, after_r) = verify_sumcheck_prod(layer.s_in, pl, tr, rd, &format!("{} phase-2b", layer.name))?;
    let vr = rd.next()?;
    tr.absorb_ext(vr);
    // R̃(r*) = Σ_g w'(g) Ã_g(l*) B̃_g(r*)
    let el = eq_table(&l_star);
    let er = eq_table(&r_star);
    let mut rt = Ext::ZERO;
    for (g, gate) in layer.gates.iter().enumerate() {
        let ga: &Gate = gate;
        rt += wp[g] * eval_terms_ext(&ga.a, &el) * eval_terms_ext(&ga.b, &er);
    }
    if vr * rt != after_r { return Err(format!("{}: phase-2b final check (wiring)", layer.name)); }
    if last {
        return Ok((LayerClaim { w: vec![], copy_point: c_star, value: Ext::ZERO }, Some((l_star, vl, r_star, vr))));
    }
    let alpha = tr.challenge();
    let nw: Vec<Ext> = el.iter().zip(&er).map(|(x, y)| *x + alpha * *y).collect();
    Ok((LayerClaim { w: nw, copy_point: c_star, value: vl + alpha * vr }, None))
}

fn verify_sumcheck_prod(vars: usize, mut running: Ext, tr: &mut Transcript, rd: &mut Reader, what: &str) -> Result<(Vec<Ext>, Ext), String> {
    let mut point = Vec::with_capacity(vars);
    for round in 0..vars {
        let s = rd.take(3)?;
        if s[0] + s[1] != running { return Err(format!("{what}: round {round} sum mismatch")); }
        tr.absorb_exts(&s);
        let r = tr.challenge();
        running = interpolate(&s, r);
        point.push(r);
    }
    Ok((point, running))
}

/// The verifier's evaluation of the input layer (columns in the clear): W̃(x, c) with padding rows zero.
pub fn eval_columns(cols: &[Fp], ncols: usize, units: usize, s_c: usize, x: &[Ext], c: &[Ext]) -> Ext {
    let ec = eq_table(c);
    let ex = eq_table(x);
    let mut acc = Ext::ZERO;
    for u in 0..units {
        let row = &cols[u * ncols..(u + 1) * ncols];
        let mut inner = Ext::ZERO;
        for (i, v) in row.iter().enumerate() { if v.0 != 0 { inner += ex[i].scale(*v); } }
        acc += inner * ec[u];
    }
    debug_assert_eq!(1usize << s_c, ex.len());
    let _ = eval_mle;
    acc
}
