//! LGSC0003: ligerito-sumcheck-2's data-driven round schedule (`backends/direct/ligerito/sumcheck.py` on
//! `lane/ligerito-sumcheck-2` @ 91a9509 / 1fbbe86, spec in that lane's report `## Interface`).
//!
//! Same statement and claims as LGSC0002 ([`crate::relation`]); what changes on the wire and in the checks:
//! * a round of arity `v` binds the `v` TOP remaining variables with ONE coin `challenge(label, v)`; its message is
//!   `M[a_1]..[a_v]` (`3^v` ext elements, row-major, `a_j in {0, 1, inf}`), `q(r) = sum_a M[a] prod_j w(a_j, r_j)` with
//!   `w(0, r) = 1 - r`, `w(1, r) = r`, `w(inf, r) = r (r - 1)`;
//! * the zero-check's last `vf` variables arrive as tables (`Az, Bz, Cz` and the `next:x` rows on them), one coin draws
//!   `r_f`, `gamma`, `g3`, `beta`; `next_vals` are the tables' MLE at `r_f` (not sent);
//! * rows + shift are ONE combined sumcheck over `max(n_i, n_c)` variables.
//! Coins: `absorb("lgsc/header")`, `challenge("zc/tau", n)`, per zero-check round `absorb("zc/q/{t}")` +
//! `challenge("zc/r/{t}", v)`, `absorb("zc/final")` + `challenge("zc/final", vf + 3)`, per combined round
//! `absorb("cmb/q/{t}")` + `challenge("cmb/r/{t}", v)`, `absorb("values")`.

use crate::claims::eq_vector;
use crate::field::{pow, Ext};
use crate::format::Reader;
use crate::relation::{succ_mle, Constraints, EvalClaim, Layout, PublicRows};
use crate::sumcheck::eq1;
use crate::transcript::{ext_bytes, Coins};

pub const MAGIC: &[u8; 8] = b"LGSC0003";
pub const MAX_ARITY: usize = 4;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Proof {
    pub zc: Vec<usize>,
    pub vf: usize,
    pub cmb: Vec<usize>,
    /// per zero-check round: `3^v` ext elements
    pub zc_rounds: Vec<Vec<Ext>>,
    /// `Az, Bz, Cz` on the last `vf` variables (`2^vf` each, bit t of the index = variable t)
    pub final_abc: [Vec<Ext>; 3],
    /// `next:x` on the same variables, `x < n_links`
    pub final_next: Vec<Vec<Ext>>,
    pub cmb_rounds: Vec<Vec<Ext>>,
    /// `w~(r_i, r_c)`, then `w~(bits(c_x), rho_c)`
    pub values: Vec<Ext>,
}

fn pow3(v: usize) -> usize {
    3usize.pow(v as u32)
}

impl Proof {
    /// Magic .. n_values: absorbed verbatim as `lgsc/header`.
    pub fn header(&self) -> Vec<u8> {
        let mut h = MAGIC.to_vec();
        h.push(self.zc.len() as u8);
        h.extend(self.zc.iter().map(|&a| a as u8));
        h.push(self.vf as u8);
        h.push(self.final_next.len() as u8);
        h.push(self.cmb.len() as u8);
        h.extend(self.cmb.iter().map(|&a| a as u8));
        h.push(self.values.len() as u8);
        h
    }
}

/// `SumcheckProof.from_bytes`: arities in `1..=4`, every coordinate `< p`, no trailing bytes.
pub fn parse(buf: &[u8]) -> Result<Proof, String> {
    let mut r = Reader::new(buf);
    if r.bytes(8)? != MAGIC {
        return Err("not an LGSC0003 sumcheck proof".into());
    }
    let mut u8s = |n: usize| -> Result<Vec<usize>, String> { Ok(r.bytes(n)?.iter().map(|&b| b as usize).collect()) };
    let z = u8s(1)?[0];
    let zc = u8s(z)?;
    let (vf, nl) = {
        let h = u8s(2)?;
        (h[0], h[1])
    };
    let q = u8s(1)?[0];
    let cmb = u8s(q)?;
    let nv = u8s(1)?[0];
    if zc.iter().chain(&cmb).chain(std::iter::once(&vf)).any(|&a| !(1..=MAX_ARITY).contains(&a)) {
        return Err("LGSC0003: arity out of range".into());
    }
    let zc_rounds = zc.iter().map(|&v| r.exts(pow3(v))).collect::<Result<Vec<_>, _>>()?;
    let final_abc = [r.exts(1 << vf)?, r.exts(1 << vf)?, r.exts(1 << vf)?];
    let final_next = (0..nl).map(|_| r.exts(1 << vf)).collect::<Result<Vec<_>, _>>()?;
    let cmb_rounds = cmb.iter().map(|&v| r.exts(pow3(v))).collect::<Result<Vec<_>, _>>()?;
    let values = r.exts(nv)?;
    if r.remaining() != 0 {
        return Err(format!("LGSC0003: {} trailing bytes", r.remaining()));
    }
    Ok(Proof { zc, vf, cmb, zc_rounds, final_abc, final_next, cmb_rounds, values })
}

pub fn write(p: &Proof) -> Vec<u8> {
    let mut out = p.header();
    for m in p.zc_rounds.iter().chain(p.final_abc.iter()).chain(&p.final_next).chain(&p.cmb_rounds) {
        out.extend(ext_bytes(m));
    }
    out.extend(ext_bytes(&p.values));
    out
}

/// Contract a `(3,)*v` mixed-representation message axis by axis (axis 0 = `X_1`, outermost) with `ws[j]`.
fn contract(m: &[Ext], ws: &[[Ext; 3]]) -> Ext {
    let mut cur = m.to_vec();
    for w in ws {
        let s = cur.len() / 3;
        cur = (0..s).map(|i| w[0].mul(&cur[i]).add(&w[1].mul(&cur[s + i])).add(&w[2].mul(&cur[2 * s + i]))).collect();
    }
    cur[0]
}

fn eval_mixed(m: &[Ext], r: &[Ext]) -> Ext {
    let ws: Vec<[Ext; 3]> = r.iter().map(|x| [Ext::ONE.sub(x), *x, x.mul(&x.sub(&Ext::ONE))]).collect();
    contract(m, &ws)
}

fn eq_sum_mixed(m: &[Ext], taus: &[Ext]) -> Ext {
    let ws: Vec<[Ext; 3]> = taus.iter().map(|t| [Ext::ONE.sub(t), *t, Ext::ZERO]).collect();
    contract(m, &ws)
}

fn bool_sum_mixed(m: &[Ext], v: usize) -> Ext {
    contract(m, &vec![[Ext::ONE, Ext::ONE, Ext::ZERO]; v])
}

fn mle_ext(vec: &[Ext], eq: &[Ext]) -> Ext {
    vec.iter().zip(eq).fold(Ext::ZERO, |acc, (v, e)| acc.add(&v.mul(e)))
}

/// Verify the zero-check and the combined sumcheck; returns the `1 + n_links` evaluation claims on the committed
/// polynomial (points `(p_i || p_c)`, as LGSC0002) that the PCS must then check.
pub fn verify(lay: &Layout, cons: &Constraints, pub_rows: &dyn PublicRows, proof: &Proof, coins: &mut dyn Coins) -> Result<Vec<EvalClaim>, String> {
    lay.validate()?;
    cons.validate(lay)?;
    let (n_k, n_c, n_i, n_links) = (lay.n_k, lay.n_c(), lay.n_i, lay.n_links());
    let n = n_k + n_c;
    let n_cmb = n_i.max(n_c);
    let vf = proof.vf;
    if proof.zc.iter().sum::<usize>() + vf != n || proof.cmb.iter().sum::<usize>() != n_cmb || vf > n_c || proof.zc.is_empty() {
        return Err("schedule does not fit the layout".into());
    }
    if proof.final_next.len() != n_links || proof.values.len() != 1 + n_links {
        return Err("shape mismatch".into());
    }
    coins.absorb(b"lgsc/header", &proof.header());
    let tau = coins.challenge(b"zc/tau", n)?;
    // ---------------------------------------------------------------- zero-check
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
        for j in 0..v {
            e = e.mul(&eq1(&taus[j], &r[j]));
        }
        claim = e.mul(&eval_mixed(m, &r));
        rs.extend_from_slice(&r);
        nrem -= v;
    }
    let fabc = &proof.final_abc;
    let prod: Vec<Ext> = (0..1 << vf).map(|b| fabc[0][b].mul(&fabc[1][b]).sub(&fabc[2][b])).collect();
    if e.mul(&mle_ext(&prod, &eq_vector(&tau[..vf]))) != claim {
        return Err("zero-check final: sum_b eq(tau, b) (Az Bz - Cz)(b) != claim".into());
    }
    let mut fin = Vec::new();
    for tab in fabc.iter().chain(&proof.final_next) {
        fin.extend(ext_bytes(tab));
    }
    coins.absorb(b"zc/final", &fin);
    let ch = coins.challenge(b"zc/final", vf + 3)?;
    let (gamma, g3, beta) = (ch[vf], ch[vf + 1], ch[vf + 2]);
    rs.extend_from_slice(&ch[..vf]);
    rs.reverse(); // r_var[t] <-> variable t
    let (r_c, r_k) = (rs[..n_c].to_vec(), rs[n_c..].to_vec());
    let eqf = eq_vector(&rs[..vf]);
    let abc: Vec<Ext> = fabc.iter().map(|tab| mle_ext(tab, &eqf)).collect();
    let next_vals: Vec<Ext> = proof.final_next.iter().map(|tab| mle_ext(tab, &eqf)).collect();
    // ---------------------------------------------------------------- combined rows + shift
    let g2 = gamma.mul(&gamma);
    let mut g3p = vec![Ext::ONE];
    for x in 1..n_links.max(1) {
        g3p.push(g3p[x - 1].mul(&g3));
    }
    let rows_claim = abc[0].add(&gamma.mul(&abc[1])).add(&g2.mul(&abc[2]));
    let shift_claim = (0..n_links).fold(Ext::ZERO, |acc, x| acc.add(&g3p[x].mul(&next_vals[x])));
    let rep = |d: usize| Ext::from_base(pow(2, d as u64));
    let mut claim = beta.mul(&rows_claim).mul(&rep(n_cmb - n_i)).add(&shift_claim.mul(&rep(n_cmb - n_c)));
    let mut rs2: Vec<Ext> = Vec::with_capacity(n_cmb);
    for (t, (&v, m)) in proof.cmb.iter().zip(&proof.cmb_rounds).enumerate() {
        if bool_sum_mixed(m, v) != claim {
            return Err(format!("combined round {t}: sum_b q(b) != claim"));
        }
        coins.absorb(format!("cmb/q/{t}").as_bytes(), &ext_bytes(m));
        let r = coins.challenge(format!("cmb/r/{t}").as_bytes(), v)?;
        claim = eval_mixed(m, &r);
        rs2.extend_from_slice(&r);
    }
    rs2.reverse();
    let (r_i, rho_c) = (rs2[..n_i].to_vec(), rs2[..n_c].to_vec());
    // coef~(r_i) = sum over the triples of eq(r_k, k) eq(r_i, row) coef g^side  (the verifier's O(nnz))
    let (eq_k, eq_i) = (eq_vector(&r_k), eq_vector(&r_i));
    let mut coef_r = Ext::ZERO;
    for (side, g) in [(&cons.a, Ext::ONE), (&cons.b, gamma), (&cons.c, g2)] {
        let acc = side.iter().fold(Ext::ZERO, |acc, &(k, row, v)| acc.add(&eq_k[k].mul(&eq_i[row]).mul_base(v)));
        coef_r = coef_r.add(&acc.mul(&g));
    }
    let eq_c = eq_vector(&r_c);
    let mut z_r = proof.values[0];
    for (name, row) in &lay.virt {
        if !name.starts_with("next:") {
            z_r = z_r.add(&eq_i[*row].mul(&pub_rows.mle(name, &eq_c)?));
        }
    }
    for x in 0..n_links {
        let row = lay.virt_row(&format!("next:{x}")).ok_or("layout: next row missing")?;
        z_r = z_r.add(&eq_i[row].mul(&next_vals[x]));
    }
    let succ = succ_mle(&rho_c, &r_c, lay.n_j);
    let v_r = (0..n_links).fold(Ext::ZERO, |acc, x| acc.add(&g3p[x].mul(&proof.values[1 + x])));
    if beta.mul(&coef_r).mul(&z_r).add(&succ.mul(&v_r)) != claim {
        return Err("combined final: beta coef(r_i) z(r_i, r_c) + succ(rho, r_c) V(rho) != claim".into());
    }
    coins.absorb(b"values", &ext_bytes(&proof.values));
    let mut claims = Vec::with_capacity(1 + n_links);
    let mut p0 = r_i.clone();
    p0.extend_from_slice(&r_c);
    claims.push(EvalClaim { point: p0, value: proof.values[0], name: "w(r_i, r_c)".into() });
    for (x, &cr) in lay.c_rows.iter().enumerate() {
        let mut pt: Vec<Ext> = (0..n_i).map(|t| Ext::from_base(((cr >> t) & 1) as u32)).collect();
        pt.extend_from_slice(&rho_c);
        claims.push(EvalClaim { point: pt, value: proof.values[1 + x], name: format!("w(c_{x}, rho)") });
    }
    Ok(claims)
}

/// The fixture's negatives (`{"proof_hex", "reason", ...}`): each must be rejected, with the Python reason's stage.
pub fn fixture_negatives(text: &str) -> Result<Vec<(Vec<u8>, String)>, String> {
    let d = crate::json::parse(text)?;
    let neg = match d.get("negatives") {
        Some(n) => n.as_arr().ok_or("fixture: negatives")?.to_vec(),
        None => return Ok(Vec::new()),
    };
    neg.iter()
        .map(|e| {
            let hex = e.get("proof_hex").and_then(|x| x.as_str()).ok_or("fixture: negative proof_hex")?;
            let why = e.get("reason").and_then(|x| x.as_str()).unwrap_or("").to_string();
            Ok((crate::relation::hex_bytes(hex)?, why))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::relation::load_fixture;
    use crate::transcript::{Scripted, SumcheckLocal};

    const FIXTURE: &str = include_str!("../fixtures/sumcheck/lgsc0003_fp8ada_l64_S2.json");

    /// ligerito-sumcheck-2's fixture (1fbbe86, fp8-ada, l = 64, S = 2, schedule zc 3,3,3,3,3 / vf 4 / cmb 3,3,3,3,
    /// 11 coins, `LocalCoins(b"fixture")`): accepted with the recorded transcript (every absorb byte-checked) and with
    /// the local coins, claims equal to the Python verifier's; the 6 recorded negatives rejected in the same stage; byte
    /// flips and a corrupted public row / coefficient rejected.
    #[test]
    fn lgsc0003_fixture_fp8ada_toy() {
        let fx = load_fixture(FIXTURE).unwrap();
        let proof = parse(&fx.proof).unwrap();
        assert_eq!(write(&proof), fx.proof);
        assert_eq!((proof.zc.clone(), proof.vf, proof.cmb.clone()), (vec![3, 3, 3, 3, 3], 4, vec![3, 3, 3, 3]));
        let mut sc = Scripted::new(fx.transcript.clone());
        let claims = verify(&fx.layout, &fx.constraints, &fx.pub_rows, &proof, &mut sc).unwrap();
        sc.finish().unwrap();
        assert_eq!(claims, fx.claims);
        let mut lc = SumcheckLocal::new(b"fixture");
        assert_eq!(verify(&fx.layout, &fx.constraints, &fx.pub_rows, &proof, &mut lc).unwrap(), fx.claims);
        assert_eq!(lc.rounds(), 11);
        let run = |pf: &Proof| verify(&fx.layout, &fx.constraints, &fx.pub_rows, pf, &mut SumcheckLocal::new(b"fixture"));
        let negs = fixture_negatives(FIXTURE).unwrap();
        assert_eq!(negs.len(), 6);
        for (bytes, why) in &negs {
            let got = parse(bytes).and_then(|pf| run(&pf));
            let stage = why.split(':').next().unwrap();
            assert!(got.as_ref().is_err_and(|e| e.starts_with(stage)), "{got:?} vs python {why}");
        }
        for pos in (0..22).chain((22..fx.proof.len()).step_by(11)) {
            let mut b = fx.proof.clone();
            b[pos] ^= 1;
            assert!(parse(&b).and_then(|pf| run(&pf)).is_err(), "accepted a flipped bit at byte {pos}");
        }
        let mut pubs = fx.pub_rows.clone();
        let idx = pubs.iter().position(|(n, _)| n == "y16").unwrap();
        pubs[idx].1[3] ^= 1;
        assert!(verify(&fx.layout, &fx.constraints, &pubs, &proof, &mut SumcheckLocal::new(b"fixture")).is_err());
        let mut cons = fx.constraints.clone();
        cons.a[7].2 = (cons.a[7].2 + 1) % crate::field::P;
        assert!(verify(&fx.layout, &cons, &fx.pub_rows, &proof, &mut SumcheckLocal::new(b"fixture")).is_err());
        assert!(verify(&fx.layout, &fx.constraints, &fx.pub_rows, &proof, &mut SumcheckLocal::new(b"other")).is_err());
    }

    #[test]
    fn schedule_and_parser_limits() {
        let fx = load_fixture(FIXTURE).unwrap();
        let proof = parse(&fx.proof).unwrap();
        // a schedule that does not add up to the layout's variables, and arity 0 / 5 in the header
        let mut p2 = proof.clone();
        p2.cmb.pop();
        p2.cmb_rounds.pop();
        let e = verify(&fx.layout, &fx.constraints, &fx.pub_rows, &p2, &mut SumcheckLocal::new(b"fixture")).unwrap_err();
        assert!(e.starts_with("schedule"), "{e}");
        for bad in [0u8, 5] {
            let mut b = fx.proof.clone();
            b[9] = bad;
            assert!(parse(&b).is_err());
        }
        let mut b = fx.proof.clone();
        b.push(0);
        assert!(parse(&b).is_err());
        assert!(parse(&fx.proof[..fx.proof.len() - 1]).is_err());
        // a non-canonical coordinate in the first zero-check message
        let mut b = fx.proof.clone();
        b[22..26].copy_from_slice(&crate::field::P.to_le_bytes());
        assert!(parse(&b).is_err());
    }
}
