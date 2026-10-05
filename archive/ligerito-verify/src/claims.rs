//! Tensor claims (paper section 6.6): the verifier never materializes the batched vector w~_i; it keeps a list of
//! terms, each `coef . (x)_j (a_j, b_j)` over the current number of variables, with bit j of the vector index <->
//! factor j (bit 0 = least significant = the fastest-varying ROW index; the column bits of the column-major matrix
//! are the HIGH variables).
//!
//! * `Eq` -- the evaluation claim and its descendants: pairs (1 - z_j, z_j), i.e. w[x] = coef . eq(x, z);
//! * `Geom` -- an opened codeword row s of a coefficient-basis Reed-Solomon code: pairs (1, eta_s^(2^j)), i.e.
//!   w[x] = coef . eta_s^x, with eta_s a BASE-field domain point.
//!
//! `restrict` fixes the top k' variables at the sumcheck challenges (paper eq. (3): Mat(w) r-bar), `expand` /
//! `dot` evaluate the remaining tensor for the final check `s_{l-1}(r) = w~_l^T y_l`.

use crate::field::{dot_ext, dot_ext_base, eq_pair, mul, Ext};

#[derive(Clone, Debug)]
pub enum Kind {
    Eq { z: Vec<Ext> },
    Geom { eta: u32 },
}

#[derive(Clone, Debug)]
pub struct Term {
    pub coef: Ext,
    pub nvars: usize,
    pub kind: Kind,
}

impl Term {
    pub fn eq(coef: Ext, z: Vec<Ext>) -> Term {
        let nvars = z.len();
        Term { coef, nvars, kind: Kind::Eq { z } }
    }

    pub fn geom(coef: Ext, eta: u32, nvars: usize) -> Term {
        Term { coef, nvars, kind: Kind::Geom { eta } }
    }

    /// The factor (a_j, b_j) of variable j.
    pub fn pair(&self, j: usize) -> (Ext, Ext) {
        debug_assert!(j < self.nvars);
        match &self.kind {
            Kind::Eq { z } => eq_pair(&z[j]),
            Kind::Geom { eta } => (Ext::ONE, Ext::from_base(pow2k(*eta, j))),
        }
    }

    /// Fix the top variables k .. nvars-1 at `r` (r[j] <-> variable k + j): coef *= prod_j (a (1 - r_j) + b r_j).
    pub fn restrict(&self, k: usize, r: &[Ext]) -> Term {
        assert_eq!(k + r.len(), self.nvars, "restrict: {} + {} != {}", k, r.len(), self.nvars);
        let mut acc = self.coef;
        for (j, rj) in r.iter().enumerate() {
            let (a, b) = self.pair(k + j);
            let one_minus = Ext::ONE.sub(rj);
            acc = acc.mul(&a.mul(&one_minus).add(&b.mul(rj)));
        }
        let kind = match &self.kind {
            Kind::Eq { z } => Kind::Eq { z: z[..k].to_vec() },
            Kind::Geom { eta } => Kind::Geom { eta: *eta },
        };
        Term { coef: acc, nvars: k, kind }
    }

    /// The tensor over variables lo .. hi-1 as a vector of 2^(hi-lo) entries (bit j of the index <-> variable lo + j),
    /// WITHOUT the coefficient.
    pub fn expand(&self, lo: usize, hi: usize) -> Vec<Ext> {
        let mut vec = vec![Ext::ONE];
        for j in lo..hi {
            let (a, b) = self.pair(j);
            let mut next = Vec::with_capacity(2 * vec.len());
            next.extend(vec.iter().map(|v| v.mul(&a)));
            next.extend(vec.iter().map(|v| v.mul(&b)));
            vec = next;
        }
        vec
    }

    /// coef . < tensor over all remaining variables , y >  (y has 2^nvars entries).
    pub fn dot(&self, y: &[Ext]) -> Ext {
        assert_eq!(y.len(), 1usize << self.nvars);
        let inner = match &self.kind {
            Kind::Geom { eta } => dot_ext_base(y, &powers(*eta, y.len())),
            Kind::Eq { .. } => dot_ext(y, &self.expand(0, self.nvars)),
        };
        self.coef.mul(&inner)
    }
}

/// eta^(2^j) by repeated squaring.
pub fn pow2k(eta: u32, j: usize) -> u32 {
    let mut e = eta;
    for _ in 0..j {
        e = mul(e, e);
    }
    e
}

/// [1, eta, eta^2, ..., eta^(count-1)] in the base field.
pub fn powers(eta: u32, count: usize) -> Vec<u32> {
    let mut out = Vec::with_capacity(count);
    let mut cur = 1u32;
    for _ in 0..count {
        out.push(cur);
        cur = mul(cur, eta);
    }
    out
}

/// eq(x, z) over x in {0,1}^k with bit j of x <-> z_j (the paper's r-bar for the challenge vector r).
pub fn eq_vector(z: &[Ext]) -> Vec<Ext> {
    let mut vec = vec![Ext::ONE];
    for zj in z {
        let (a, b) = eq_pair(zj);
        let mut next = Vec::with_capacity(2 * vec.len());
        next.extend(vec.iter().map(|v| v.mul(&a)));
        next.extend(vec.iter().map(|v| v.mul(&b)));
        vec = next;
    }
    vec
}

/// The degree-2 sumcheck message g given by its values at 0, 1, 2, evaluated at r.
pub fn interpolate_deg2(g: &[Ext; 3], r: &Ext) -> Ext {
    // c2 = (g2 - 2 g1 + g0) / 2 ; c1 = g1 - g0 - c2 ; g(r) = g0 + r (c1 + c2 r)
    let inv2 = crate::field::inv(2);
    let c2 = g[2].add(&g[0]).sub(&g[1].mul_base(2)).mul_base(inv2);
    let c1 = g[1].sub(&g[0]).sub(&c2);
    g[0].add(&r.mul(&c1.add(&c2.mul(r))))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::P;

    fn e(v: u32) -> Ext {
        Ext::from_base(v % P)
    }

    #[test]
    fn eq_vector_is_multilinear_eq() {
        let z = vec![e(3), e(5), Ext([1, 2, 3, 4, 5, 6])];
        let v = eq_vector(&z);
        assert_eq!(v.len(), 8);
        for x in 0..8usize {
            let mut expect = Ext::ONE;
            for j in 0..3 {
                let (a, b) = eq_pair(&z[j]);
                expect = expect.mul(if (x >> j) & 1 == 1 { &b } else { &a });
            }
            assert_eq!(v[x], expect);
        }
        // sum over the cube of eq(x, z) is 1
        let s = v.iter().fold(Ext::ZERO, |a, b| a.add(b));
        assert_eq!(s, Ext::ONE);
    }

    #[test]
    fn geom_expand_is_powers_and_restrict_matches_full_partial_evaluation() {
        let eta = 12345u32;
        let t = Term::geom(e(7), eta, 6);
        let full = t.expand(0, 6);
        let pw = powers(eta, 64);
        for x in 0..64 {
            assert_eq!(full[x], Ext::from_base(pw[x]));
        }
        // restrict top 2 variables at r: coef' . tensor(low 4) must equal Mat(w) r-bar, column-major with 16 rows
        let r = vec![Ext([9, 8, 7, 6, 5, 4]), Ext([1, 1, 2, 3, 5, 8])];
        let rt = t.restrict(4, &r);
        let rb = eq_vector(&r);
        for row in 0..16 {
            let mut expect = Ext::ZERO;
            for col in 0..4 {
                expect = expect.add(&full[row + 16 * col].mul(&rb[col]));
            }
            expect = expect.mul(&t.coef);
            assert_eq!(rt.coef.mul(&rt.expand(0, 4)[row]), expect);
        }
        // the same for an eq term
        let z: Vec<Ext> = (0..6).map(|i| Ext([i, 2 * i, 3, 0, 1, 7])).collect();
        let t = Term::eq(Ext([2, 0, 0, 0, 0, 1]), z.clone());
        let full = t.expand(0, 6);
        assert_eq!(full, eq_vector(&z));
        let rt = t.restrict(4, &r);
        for row in 0..16 {
            let mut expect = Ext::ZERO;
            for col in 0..4 {
                expect = expect.add(&full[row + 16 * col].mul(&rb[col]));
            }
            expect = expect.mul(&t.coef);
            assert_eq!(rt.coef.mul(&rt.expand(0, 4)[row]), expect);
        }
        // dot agrees with the expanded inner product
        let y: Vec<Ext> = (0..16).map(|i| Ext([i * 11 % P, 3, 1, 4, 1, 5])).collect();
        let mut expect = Ext::ZERO;
        for i in 0..16 {
            expect = expect.add(&rt.expand(0, 4)[i].mul(&y[i]));
        }
        assert_eq!(rt.dot(&y), rt.coef.mul(&expect));
    }

    #[test]
    fn interpolation_hits_the_nodes() {
        let g = [Ext([1, 2, 3, 4, 5, 6]), Ext([7, 0, 0, 1, 0, 0]), Ext([9, 9, 9, 9, 9, 9])];
        assert_eq!(interpolate_deg2(&g, &e(0)), g[0]);
        assert_eq!(interpolate_deg2(&g, &e(1)), g[1]);
        assert_eq!(interpolate_deg2(&g, &e(2)), g[2]);
    }
}
