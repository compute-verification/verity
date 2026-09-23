//! Round dimensions and the soundness bound (paper section 6.3, eq. (17)) -- the Rust mirror of
//! `backends/direct/ligerito/params.py` (lane ligerito-design).  Until params.py lands the shapes follow
//! `proto/dims.py`: `n = log2 N`; L committed rounds with k'_i column bits each; k_i = k_{i-1} - k'_i (k_0 = n);
//! round i commits R_i = 2^{k_i} rows x C_i = 2^{k'_i} columns, codeword length m_i = 2^{k_i + rate_log2[i]} along the
//! tall axis (the RS rate is per committed round since proto 7bc2fdd: ligerito-design's S point uses 1/4 at round 1
//! and 1/16 afterwards); the final vector y has 2^{k_L} entries.  Field terms are over |F| = p^6 ~ 2^186.

use crate::field::{pow, root_of_unity, P};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Params {
    pub n: usize,
    pub kprime: Vec<usize>,
    pub queries: Vec<usize>,
    /// log2 of the inverse RS rate, per committed round.
    pub rate_log2: Vec<usize>,
    /// Legacy header form: proto b4d7e12 absorbed `"rate": 2` (one integer); 7bc2fdd absorbs `"rate": [2, 2, 2]`.
    /// Set by the proto reader from the header's own type so both generations verify bit-for-bit.
    pub header_rate_scalar: bool,
    /// Multiplicative shift of the evaluation domain: 1 = the subgroup <w_m> (proto), 31 = the Ligero coset g<w_m>.
    pub domain_shift: u32,
    /// Per round, the (n1, n2) factorisation of the codeword length used by the prover's 4-step NTT: codeword
    /// position s holds the evaluation at w_m^e with e = s / n1 + n2 (s mod n1) (`proto/ntt_gpu.Encoder.eta_index`;
    /// n2 = 1 is natural order).  Any such factorisation is a bijection of the domain, so it is a layout choice,
    /// not a soundness parameter; `canonical_split` is what the GPU encoder picks.
    pub splits: Vec<(usize, usize)>,
}

/// `proto/ntt_gpu.split`: (n1, n2) with n1 n2 = m, both <= 4096, n1 >= n2; m <= 4096 -> (m, 1).
pub fn canonical_split(m: usize) -> (usize, usize) {
    if m <= 4096 {
        return (m, 1);
    }
    let logm = m.trailing_zeros() as usize;
    let log1 = (logm + 1) / 2;
    (1 << log1, 1 << (logm - log1))
}

impl Params {
    pub fn new(n: usize, kprime: Vec<usize>, queries: Vec<usize>, rate_log2: usize) -> Params {
        let rates = vec![rate_log2; kprime.len()];
        Params::with_rates(n, kprime, queries, rates)
    }

    pub fn with_rates(n: usize, kprime: Vec<usize>, queries: Vec<usize>, rate_log2: Vec<usize>) -> Params {
        let mut p = Params { n, kprime, queries, rate_log2, header_rate_scalar: false, domain_shift: 1, splits: Vec::new() };
        if p.validate_shape().is_ok() {
            p.splits = (1..=p.rounds()).map(|i| canonical_split(p.m(i))).collect();
        }
        p
    }

    /// The header the Fiat-Shamir transcript absorbs first (`json.dumps` with default separators).
    pub fn header_json(&self) -> String {
        let list = |v: &[usize]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", ");
        let rate = if self.header_rate_scalar {
            self.rate_log2.first().map_or(0, |r| *r).to_string()
        } else {
            format!("[{}]", list(&self.rate_log2))
        };
        format!(
            "{{\"n\": {}, \"kprime\": [{}], \"queries\": [{}], \"rate\": {}}}",
            self.n,
            list(&self.kprime),
            list(&self.queries),
            rate
        )
    }

    /// Number of committed rounds (the paper's l - 1).
    pub fn rounds(&self) -> usize {
        self.kprime.len()
    }

    /// k_i for committed round i (1-based): the row bits; k(0) = n.
    pub fn k(&self, i: usize) -> usize {
        self.n - self.kprime[..i].iter().sum::<usize>()
    }

    pub fn rows(&self, i: usize) -> usize {
        1 << self.k(i)
    }

    pub fn cols(&self, i: usize) -> usize {
        1 << self.kprime[i - 1]
    }

    /// Codeword length (Merkle leaves) of round i.
    pub fn m(&self, i: usize) -> usize {
        1 << self.depth(i)
    }

    /// log2 of the inverse rate of round i (1-based).
    pub fn rate(&self, i: usize) -> usize {
        self.rate_log2[i - 1]
    }

    pub fn depth(&self, i: usize) -> usize {
        self.k(i) + self.rate(i)
    }

    pub fn final_len(&self) -> usize {
        1 << self.k(self.rounds())
    }

    pub fn validate(&self) -> Result<(), String> {
        self.validate_shape()?;
        let l = self.rounds();
        if self.splits.len() != l {
            return Err(format!("params: {} NTT splits for {l} rounds", self.splits.len()));
        }
        for i in 1..=l {
            let (n1, n2) = self.splits[i - 1];
            if n1 == 0 || n2 == 0 || !n1.is_power_of_two() || !n2.is_power_of_two() || n1 * n2 != self.m(i) {
                return Err(format!("params: round {i} split {n1} x {n2} is not a factorisation of m = {}", self.m(i)));
            }
        }
        if self.domain_shift == 0 || self.domain_shift >= P {
            return Err("params: domain shift".into());
        }
        Ok(())
    }

    fn validate_shape(&self) -> Result<(), String> {
        let l = self.rounds();
        if l == 0 {
            return Err("params: no committed rounds".into());
        }
        if self.queries.len() != l {
            return Err(format!("params: {} query counts for {l} rounds", self.queries.len()));
        }
        if self.rate_log2.len() != l {
            return Err(format!("params: {} rates for {l} rounds", self.rate_log2.len()));
        }
        if self.header_rate_scalar && self.rate_log2.iter().any(|r| *r != self.rate_log2[0]) {
            return Err("params: a scalar-rate header with different per-round rates".into());
        }
        if self.n == 0 || self.n > 40 {
            return Err(format!("params: n = {} out of range", self.n));
        }
        let mut k = self.n;
        for (i, &kp) in self.kprime.iter().enumerate() {
            if kp == 0 || kp > k {
                return Err(format!("params: k'_{} = {kp} with k_{} = {k}", i + 1, i));
            }
            k -= kp;
            let rate = self.rate_log2[i];
            if rate == 0 {
                return Err(format!("params: round {} rate 1 has no distance", i + 1));
            }
            if k + rate > crate::field::TWO_ADICITY as usize {
                return Err(format!("params: codeword 2^{} exceeds the 2-adicity", k + rate));
            }
            if self.queries[i] == 0 || self.queries[i] > (1usize << (k + rate)) {
                return Err(format!("params: |S_{}| = {} for m = 2^{}", i + 1, self.queries[i], k + rate));
            }
        }
        Ok(())
    }

    /// The exponent e of codeword position s in round i (eta_s = shift . w_m^e).
    pub fn eta_index(&self, i: usize, s: usize) -> usize {
        let (n1, n2) = self.splits[i - 1];
        s / n1 + n2 * (s % n1)
    }

    /// The evaluation point eta_s of codeword position s in round i.
    pub fn domain_point(&self, i: usize, s: usize) -> u32 {
        let w = root_of_unity(self.m(i));
        crate::field::mul(self.domain_shift, pow(w, self.eta_index(i, s) as u64))
    }

    /// log2 of the soundness error, eq. (17) with the Reed-Solomon unique-decoding term written as
    /// ((1 + rho)/2)^|S| where rho = 2^k / m -- (17) prints (m - 2^k - 1)/(2m) = (1 - rho)/2 for that base, which
    /// contradicts the paper's own numerics (|S| = 148 <-> 2^-100 needs (5/8)^148) and the asymptotics section's
    /// log((1 + rho)/2); the row-sampling argument gives (1 - d/(2m))^|S| = ((1 + rho)/2)^|S| up to the +1/m.
    /// Field terms: (2 k'_i + |S_{i-1}| + 1 + m_i k'_i) / |F| per round, |F| = p^6.  Returns (total, per-term list).
    pub fn soundness_log2(&self) -> (f64, Vec<(String, f64)>) {
        let log2_f = 6.0 * (P as f64).log2();
        let mut terms = Vec::new();
        let mut total = 0f64;
        let l = self.rounds();
        for i in 1..=l {
            let m = self.m(i) as f64;
            let rows = self.rows(i) as f64;
            let s = self.queries[i - 1] as f64;
            let kp = self.kprime[i - 1] as f64;
            // rows agreeing with the (unique) close codeword when the matrix is >= d/2-far
            let p_rs = ((m + rows + 1.0) / (2.0 * m)).log2() * s;
            let prev_s = if i >= 2 { self.queries[i - 2] as f64 } else { 0.0 };
            let field = (2.0 * kp + prev_s + 1.0 + m * kp).log2() - log2_f;
            terms.push((format!("round{i}_rs"), p_rs));
            terms.push((format!("round{i}_field"), field));
            total += p_rs.exp2() + field.exp2();
        }
        (total.log2(), terms)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dims_chain() {
        let p = Params::new(20, vec![6, 5, 4], vec![192, 192, 64], 2);
        p.validate().unwrap();
        assert_eq!(p.k(0), 20);
        assert_eq!(p.k(1), 14);
        assert_eq!(p.k(3), 5);
        assert_eq!(p.rows(1), 1 << 14);
        assert_eq!(p.cols(1), 64);
        assert_eq!(p.m(1), 1 << 16);
        assert_eq!(p.final_len(), 32);
        assert!(Params::new(20, vec![6, 15], vec![1, 1], 2).validate().is_err());
        assert!(Params::new(20, vec![6], vec![1, 1], 2).validate().is_err());
    }

    #[test]
    fn soundness_reproduces_the_paper_query_count() {
        // rate 1/4, |S| = 148 -> ~ 2^-100 per round (paper section 7, lambda = 100)
        let p = Params::new(24, vec![8, 8], vec![148, 148], 2);
        let (_, terms) = p.soundness_log2();
        let rs1 = terms.iter().find(|(k, _)| k == "round1_rs").unwrap().1;
        assert!((rs1 - -100.3).abs() < 0.5, "{rs1}");
        // |S| = 192 -> ~ 2^-130 per round; three rounds ~ 2^-128.6
        let p = Params::new(29, vec![9, 8, 6], vec![192, 192, 192], 2);
        let (tot, _) = p.soundness_log2();
        assert!(tot < -128.0 && tot > -131.0, "{tot}");
    }

    #[test]
    fn domain_points_are_distinct_roots() {
        for split in [(128, 1), (16, 8), (8, 16)] {
            let mut p = Params::new(8, vec![3], vec![4], 2);
            p.splits = vec![split];
            p.validate().unwrap();
            let m = p.m(1);
            let pts: std::collections::HashSet<u32> = (0..m).map(|s| p.domain_point(1, s)).collect();
            assert_eq!(pts.len(), m);
            for s in 0..m {
                assert_eq!(pow(p.domain_point(1, s), m as u64), 1);
            }
        }
        let mut p = Params::new(8, vec![3], vec![4], 2);
        p.splits = vec![(64, 1)];
        assert!(p.validate().is_err());
        assert_eq!(canonical_split(1 << 12), (1 << 12, 1));
        assert_eq!(canonical_split(1 << 13), (1 << 7, 1 << 6));
        assert_eq!(canonical_split(1 << 24), (1 << 12, 1 << 12));
        // eta_index with (n1, n2) = (4, 2), m = 8: s = 5 -> 5/4 + 2 (5 mod 4) = 1 + 2 = 3
        let mut p = Params::new(3, vec![2], vec![2], 2); // k = 1, m = 8
        p.splits = vec![(4, 2)];
        assert_eq!(p.eta_index(1, 5), 3);
    }

    #[test]
    fn header_json_matches_json_dumps() {
        let mut p = Params::new(12, vec![4, 3, 2], vec![16, 16, 16], 2);
        assert_eq!(p.header_json(), "{\"n\": 12, \"kprime\": [4, 3, 2], \"queries\": [16, 16, 16], \"rate\": [2, 2, 2]}");
        p.header_rate_scalar = true;
        assert_eq!(p.header_json(), "{\"n\": 12, \"kprime\": [4, 3, 2], \"queries\": [16, 16, 16], \"rate\": 2}");
        let q = Params::with_rates(12, vec![4, 3, 2], vec![16, 16, 16], vec![2, 4, 4]);
        q.validate().unwrap();
        assert_eq!(q.header_json(), "{\"n\": 12, \"kprime\": [4, 3, 2], \"queries\": [16, 16, 16], \"rate\": [2, 4, 4]}");
        assert_eq!(q.m(1), 1 << 10);
        assert_eq!(q.m(2), 1 << 9);
        assert_eq!(q.m(3), 1 << 7);
    }
}
