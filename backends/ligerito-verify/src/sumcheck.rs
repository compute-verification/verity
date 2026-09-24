//! Sumcheck round verification for the relation side (ligerito-sumcheck's three sumchecks: the degree-3 zero-check in
//! Gruen form, the degree-2 rows sumcheck, the degree-2 shift sumcheck).  Only the round arithmetic lives here; the
//! message framing, the labels and the claim wiring follow ligerito-sumcheck's `## Interface` and are added on top.
//!
//! Round forms (all values in F_{p^6}):
//! * `Plain`  — degree-2 round polynomial `g` sent as `(g(0), g(1), g(2))`; check `g(0) + g(1) == claim`,
//!   continue with `g(r)`.
//! * `Gruen`  — round polynomial `eq(tau_t, X) * q(X)` with `q` quadratic sent as `(q(0), q(1), q(inf))` (`q(inf)` = the
//!   leading coefficient); check `(1 - tau_t) q(0) + tau_t q(1) == claim`, continue with `eq(tau_t, r) * q(r)`.
//!
//! Bit convention (ligerito-sumcheck 17:20Z): variable `t` is bit `t` of the flat index, LSB first,
//! `eq(r, x) = prod_t (r_t x_t + (1 - r_t)(1 - x_t))`; the rounds bind the variables in the order the messages arrive
//! (`rs[t]` is the challenge of message `t`), and callers map that draw order onto variable indices.

use crate::field::Ext;

/// A round message in one of the two supported forms.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Round {
    Plain([Ext; 3]),
    Gruen([Ext; 3]),
}

/// `q(X) = q0 + c1 X + qinf X^2` with `c1 = q1 - q0 - qinf`, evaluated at `r`.
pub fn eval_from_0_1_inf(q0: &Ext, q1: &Ext, qinf: &Ext, r: &Ext) -> Ext {
    let c1 = q1.sub(q0).sub(qinf);
    q0.add(&r.mul(&c1.add(&qinf.mul(r))))
}

/// `eq(a, b) = a b + (1 - a)(1 - b)` for two extension scalars.
pub fn eq1(a: &Ext, b: &Ext) -> Ext {
    let one = Ext::ONE;
    a.mul(b).add(&one.sub(a).mul(&one.sub(b)))
}

/// `eq(a, b)` for two points of equal length.
pub fn eq_points(a: &[Ext], b: &[Ext]) -> Ext {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = Ext::ONE;
    for (x, y) in a.iter().zip(b) {
        acc = acc.mul(&eq1(x, y));
    }
    acc
}

/// `eq(r, x)` for a boolean point given as the bits of `x` (bit `t` <-> `r[t]`).
pub fn eq_at_index(r: &[Ext], x: usize) -> Ext {
    let mut acc = Ext::ONE;
    for (t, rt) in r.iter().enumerate() {
        let f = if (x >> t) & 1 == 1 { *rt } else { Ext::ONE.sub(rt) };
        acc = acc.mul(&f);
    }
    acc
}

/// One round.  Returns the next claim.
pub fn round(claim: &Ext, msg: &Round, r: &Ext, tau: Option<&Ext>) -> Result<Ext, String> {
    match msg {
        Round::Plain(g) => {
            if g[0].add(&g[1]) != *claim {
                return Err("sumcheck round: g(0) + g(1) != claim".into());
            }
            Ok(crate::claims::interpolate_deg2(g, r))
        }
        Round::Gruen(q) => {
            let tau = tau.ok_or("sumcheck round: Gruen form needs tau_t")?;
            let one_minus = Ext::ONE.sub(tau);
            if one_minus.mul(&q[0]).add(&tau.mul(&q[1])) != *claim {
                return Err("sumcheck round: (1 - tau) q(0) + tau q(1) != claim".into());
            }
            Ok(eq1(tau, r).mul(&eval_from_0_1_inf(&q[0], &q[1], &q[2], r)))
        }
    }
}

/// All rounds of one sumcheck: `msgs[t]` with challenge `rs[t]` (and `taus[t]` for Gruen rounds).  Returns the final
/// claim, to be matched against the prover's evaluation claims by the caller.
pub fn verify_rounds(claim: &Ext, msgs: &[Round], rs: &[Ext], taus: Option<&[Ext]>) -> Result<Ext, String> {
    if msgs.len() != rs.len() {
        return Err(format!("sumcheck: {} messages but {} challenges", msgs.len(), rs.len()));
    }
    if let Some(t) = taus {
        if t.len() != msgs.len() {
            return Err(format!("sumcheck: {} messages but {} tau coordinates", msgs.len(), t.len()));
        }
    }
    let mut cur = *claim;
    for (t, (m, r)) in msgs.iter().zip(rs).enumerate() {
        cur = round(&cur, m, r, taus.map(|ts| &ts[t])).map_err(|e| format!("{e} (round {t})"))?;
    }
    Ok(cur)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::P;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn ext(&mut self) -> Ext {
            let mut c = [0u32; 6];
            for x in &mut c {
                *x = (self.next() % P as u64) as u32;
            }
            Ext(c)
        }
    }

    /// Multilinear extension of a table over `nvars` bits, evaluated at `r` (bit t <-> r[t]).
    fn mle(table: &[Ext], r: &[Ext]) -> Ext {
        let mut acc = Ext::ZERO;
        for (x, v) in table.iter().enumerate() {
            acc = acc.add(&eq_at_index(r, x).mul(v));
        }
        acc
    }

    /// Honest prover for the degree-3 zero-check sum_x eq(tau, x) a(x) b(x) - eq(tau, x) c(x) in Gruen form, binding
    /// the LOW bit first (variable t = message t).  Returns messages and the final (a, b, c) evaluations.
    fn zero_check_prover(a: &[Ext], b: &[Ext], c: &[Ext], tau: &[Ext], rs: &[Ext]) -> (Vec<Round>, [Ext; 3]) {
        let n = tau.len();
        let (mut a, mut b, mut c) = (a.to_vec(), b.to_vec(), c.to_vec());
        let mut msgs = Vec::new();
        let mut prefix = Ext::ONE; // eq(tau_{<t}, r_{<t}): the bound variables' eq factor, carried inside q
        for t in 0..n {
            let h = a.len() / 2;
            // q(X) = prefix * sum_{x'} eq(tau_{>t}, x') * (a_X b_X - c_X)(x'), a_X = (1 - X) a(0,x') + X a(1,x'), pairs (2y, 2y+1)
            let w: Vec<Ext> = (0..h).map(|y| eq_at_index(&tau[t + 1..], y)).collect();
            let mut q0 = Ext::ZERO;
            let mut q1 = Ext::ZERO;
            let mut qinf = Ext::ZERO;
            for y in 0..h {
                let (a0, a1) = (a[2 * y], a[2 * y + 1]);
                let (b0, b1) = (b[2 * y], b[2 * y + 1]);
                let (c0, c1) = (c[2 * y], c[2 * y + 1]);
                q0 = q0.add(&w[y].mul(&a0.mul(&b0).sub(&c0)));
                q1 = q1.add(&w[y].mul(&a1.mul(&b1).sub(&c1)));
                // leading coefficient of (a0 + (a1-a0)X)(b0 + (b1-b0)X) - (c0 + (c1-c0)X)
                qinf = qinf.add(&w[y].mul(&a1.sub(&a0).mul(&b1.sub(&b0))));
            }
            msgs.push(Round::Gruen([prefix.mul(&q0), prefix.mul(&q1), prefix.mul(&qinf)]));
            let r = rs[t];
            prefix = prefix.mul(&eq1(&tau[t], &r));
            let fold = |v: &[Ext]| -> Vec<Ext> {
                (0..h).map(|y| v[2 * y].add(&r.mul(&v[2 * y + 1].sub(&v[2 * y])))).collect()
            };
            a = fold(&a);
            b = fold(&b);
            c = fold(&c);
        }
        (msgs, [a[0], b[0], c[0]])
    }

    #[test]
    fn eval_from_0_1_inf_matches_interpolation() {
        let mut g = Rng(11);
        let (c0, c1, c2, r) = (g.ext(), g.ext(), g.ext(), g.ext());
        let q = |x: &Ext| c0.add(&c1.mul(x)).add(&c2.mul(&x.mul(x)));
        let q0 = q(&Ext::ZERO);
        let q1 = q(&Ext::ONE);
        assert_eq!(eval_from_0_1_inf(&q0, &q1, &c2, &r), q(&r));
        let q2 = q(&Ext::from_base(2));
        assert_eq!(crate::claims::interpolate_deg2(&[q0, q1, q2], &r), q(&r));
    }

    #[test]
    fn eq_helpers_agree() {
        let mut g = Rng(5);
        let r: Vec<Ext> = (0..4).map(|_| g.ext()).collect();
        let s: Vec<Ext> = (0..4).map(|_| g.ext()).collect();
        // eq(r, s) = sum_x eq(r, x) eq(s, x)
        let mut acc = Ext::ZERO;
        for x in 0..16 {
            acc = acc.add(&eq_at_index(&r, x).mul(&eq_at_index(&s, x)));
        }
        assert_eq!(acc, eq_points(&r, &s));
        // sum_x eq(r, x) = 1
        let mut one = Ext::ZERO;
        for x in 0..16 {
            one = one.add(&eq_at_index(&r, x));
        }
        assert_eq!(one, Ext::ONE);
        // agrees with claims::eq_vector (bit j <-> z[j])
        let v = crate::claims::eq_vector(&r);
        for x in 0..16 {
            assert_eq!(v[x], eq_at_index(&r, x));
        }
    }

    #[test]
    fn gruen_zero_check_roundtrip_and_negatives() {
        let mut g = Rng(99);
        let n = 4;
        let a: Vec<Ext> = (0..16).map(|_| g.ext()).collect();
        let b: Vec<Ext> = (0..16).map(|_| g.ext()).collect();
        let c: Vec<Ext> = a.iter().zip(&b).map(|(x, y)| x.mul(y)).collect(); // satisfied relation
        let tau: Vec<Ext> = (0..n).map(|_| g.ext()).collect();
        let rs: Vec<Ext> = (0..n).map(|_| g.ext()).collect();
        let (msgs, fin) = zero_check_prover(&a, &b, &c, &tau, &rs);
        let out = verify_rounds(&Ext::ZERO, &msgs, &rs, Some(&tau)).unwrap();
        // final claim = eq(tau, r) * (a(r) b(r) - c(r)) with the MLEs at r
        let (ar, br, cr) = (mle(&a, &rs), mle(&b, &rs), mle(&c, &rs));
        assert_eq!(out, eq_points(&tau, &rs).mul(&ar.mul(&br).sub(&cr)));
        assert_eq!([ar, br, cr], fin);
        // unsatisfied relation: honest messages sum to a nonzero value, so the zero claim fails
        let mut c2 = c.clone();
        c2[5] = c2[5].add(&Ext::ONE);
        let (msgs2, _) = zero_check_prover(&a, &b, &c2, &tau, &rs);
        assert!(verify_rounds(&Ext::ZERO, &msgs2, &rs, Some(&tau)).is_err());
        // tampered message / wrong tau / missing tau / wrong count
        let mut m3 = msgs.clone();
        if let Round::Gruen(q) = &mut m3[2] {
            q[1] = q[1].add(&Ext::ONE);
        }
        assert!(verify_rounds(&Ext::ZERO, &m3, &rs, Some(&tau)).is_err());
        let mut tau2 = tau.clone();
        tau2[0] = tau2[0].add(&Ext::ONE);
        assert!(verify_rounds(&Ext::ZERO, &msgs, &rs, Some(&tau2)).is_err());
        assert!(verify_rounds(&Ext::ZERO, &msgs, &rs, None).is_err());
        assert!(verify_rounds(&Ext::ZERO, &msgs[..3], &rs, Some(&tau)).is_err());
        // a tampered q(inf) in the middle breaks the next round's check; in the LAST round it passes the round checks
        // and shifts the final claim (which the caller must match against the evaluation claims)
        let mut m4 = msgs.clone();
        if let Round::Gruen(q) = &mut m4[1] {
            q[2] = q[2].add(&Ext::ONE);
        }
        assert!(verify_rounds(&Ext::ZERO, &m4, &rs, Some(&tau)).is_err());
        let mut m5 = msgs.clone();
        if let Round::Gruen(q) = &mut m5[n - 1] {
            q[2] = q[2].add(&Ext::ONE);
        }
        assert_ne!(verify_rounds(&Ext::ZERO, &m5, &rs, Some(&tau)).unwrap(), out);
    }

    #[test]
    fn plain_deg2_rows_sumcheck_roundtrip() {
        // sum_i alpha(i) z(i) with alpha, z tables of 8 entries: g_t(X) from pairs
        let mut g = Rng(7);
        let alpha: Vec<Ext> = (0..8).map(|_| g.ext()).collect();
        let z: Vec<Ext> = (0..8).map(|_| g.ext()).collect();
        let rs: Vec<Ext> = (0..3).map(|_| g.ext()).collect();
        let mut claim = Ext::ZERO;
        for i in 0..8 {
            claim = claim.add(&alpha[i].mul(&z[i]));
        }
        let (mut a, mut zz) = (alpha.clone(), z.clone());
        let mut msgs = Vec::new();
        for t in 0..3 {
            let h = a.len() / 2;
            let at = |x: u32| -> Ext {
                let xe = Ext::from_base(x);
                let mut s = Ext::ZERO;
                for y in 0..h {
                    let av = a[2 * y].add(&xe.mul(&a[2 * y + 1].sub(&a[2 * y])));
                    let zv = zz[2 * y].add(&xe.mul(&zz[2 * y + 1].sub(&zz[2 * y])));
                    s = s.add(&av.mul(&zv));
                }
                s
            };
            msgs.push(Round::Plain([at(0), at(1), at(2)]));
            let r = rs[t];
            a = (0..h).map(|y| a[2 * y].add(&r.mul(&a[2 * y + 1].sub(&a[2 * y])))).collect();
            zz = (0..h).map(|y| zz[2 * y].add(&r.mul(&zz[2 * y + 1].sub(&zz[2 * y])))).collect();
        }
        let out = verify_rounds(&claim, &msgs, &rs, None).unwrap();
        assert_eq!(out, mle(&alpha, &rs).mul(&mle(&z, &rs)));
        assert!(verify_rounds(&claim.add(&Ext::ONE), &msgs, &rs, None).is_err());
    }
}
