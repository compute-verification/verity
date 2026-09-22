//! Radix-2 NTT over BabyBear on extension vectors (the six coordinates share every twiddle), Montgomery twiddles.
//! Semantics of `gpu/ligero.py::ntt`: forward evaluates `sum_m c_m w^{m j}` on `<w>`, `w = 31^((p-1)/n)`; inverse
//! uses `w^-1` and scales by `n^-1`.  Natural order in and out.

use crate::field::{add, inv, mont_mul, mul, pow, root_of_unity, sub, to_mont, Ext, DEG};

pub struct Plan { pub n: usize, tw: Vec<Vec<u32>>, rev: Vec<u32>, scale: Option<u32> }

impl Plan {
    pub fn new(n: usize, inverse: bool) -> Plan {
        assert!(n.is_power_of_two());
        let mut w = root_of_unity(n);
        if inverse { w = inv(w); }
        let bits = n.trailing_zeros();
        let rev: Vec<u32> = (0..n as u32).map(|i| if bits == 0 { 0 } else { i.reverse_bits() >> (32 - bits) }).collect();
        let mut tw = Vec::new();
        let mut len = 2;
        while len <= n {
            let wl = pow(w, (n / len) as u64);
            let mut level = Vec::with_capacity(len / 2);
            let mut cur = 1u32;
            for _ in 0..len / 2 { level.push(to_mont(cur)); cur = mul(cur, wl); }
            tw.push(level);
            len *= 2;
        }
        Plan { n, tw, rev, scale: if inverse { Some(inv(n as u32)) } else { None } }
    }

    pub fn run(&self, v: &mut [Ext]) {
        let n = self.n;
        assert_eq!(v.len(), n);
        for i in 0..n {
            let j = self.rev[i] as usize;
            if j > i { v.swap(i, j); }
        }
        let mut len = 2;
        let mut level = 0;
        while len <= n {
            let half = len / 2;
            let tw = &self.tw[level];
            for start in (0..n).step_by(len) {
                for j in 0..half {
                    let w = tw[j];
                    let u = v[start + j];
                    let x = v[start + j + half];
                    let mut t = [0u32; DEG];
                    for c in 0..DEG { t[c] = mont_mul(x.0[c], w); }
                    let mut lo = [0u32; DEG];
                    let mut hi = [0u32; DEG];
                    for c in 0..DEG { lo[c] = add(u.0[c], t[c]); hi[c] = sub(u.0[c], t[c]); }
                    v[start + j] = Ext(lo);
                    v[start + j + half] = Ext(hi);
                }
            }
            len *= 2;
            level += 1;
        }
        if let Some(s) = self.scale {
            let sm = to_mont(s);
            for x in v.iter_mut() { for c in 0..DEG { x.0[c] = mont_mul(x.0[c], sm); } }
        }
    }

    /// The same transform on `L` interleaved base-field lanes (`v[pos * L + lane]`, e.g. `L = R * DEG` for `R`
    /// extension rows batched together): every butterfly is `L` independent lane butterflies sharing one twiddle, a
    /// loop the compiler vectorises at every stage (the planar layout only vectorises once `half >= 16`).
    pub fn run_lanes<const L: usize>(&self, v: &mut [u32]) {
        let n = self.n;
        assert_eq!(v.len(), n * L);
        for i in 0..n {
            let j = self.rev[i] as usize;
            if j > i {
                let (a, b) = v.split_at_mut(j * L);
                a[i * L..(i + 1) * L].swap_with_slice(&mut b[..L]);
            }
        }
        let mut len = 2;
        let mut level = 0;
        while len <= n {
            let half = len / 2;
            let tw = &self.tw[level][..half];
            for start in (0..n).step_by(len) {
                let blk = &mut v[start * L..(start + len) * L];
                let (lo, hi) = blk.split_at_mut(half * L);
                for (j, &w) in tw.iter().enumerate() {
                    let a: &mut [u32; L] = (&mut lo[j * L..(j + 1) * L]).try_into().unwrap();
                    let b: &mut [u32; L] = (&mut hi[j * L..(j + 1) * L]).try_into().unwrap();
                    for l in 0..L {
                        let t = mont_mul(b[l], w);
                        let u = a[l];
                        a[l] = add(u, t);
                        b[l] = sub(u, t);
                    }
                }
            }
            len *= 2;
            level += 1;
        }
        if let Some(s) = self.scale {
            let sm = to_mont(s);
            for x in v.iter_mut() { *x = mont_mul(*x, sm); }
        }
    }

    /// The same transform on one base-field plane (a coordinate of an extension vector stored planar); the inner
    /// loops are over contiguous `u32`s with contiguous twiddles so they vectorise.
    pub fn run_base(&self, v: &mut [u32]) {
        let n = self.n;
        assert_eq!(v.len(), n);
        for i in 0..n {
            let j = self.rev[i] as usize;
            if j > i { v.swap(i, j); }
        }
        let mut len = 2;
        let mut level = 0;
        while len <= n {
            let half = len / 2;
            let tw = &self.tw[level][..half];
            for start in (0..n).step_by(len) {
                let (lo, hi) = v[start..start + len].split_at_mut(half);
                for ((a, b), &w) in lo.iter_mut().zip(hi.iter_mut()).zip(tw) {
                    let t = mont_mul(*b, w);
                    let u = *a;
                    *a = add(u, t);
                    *b = sub(u, t);
                }
            }
            len *= 2;
            level += 1;
        }
        if let Some(s) = self.scale {
            let sm = to_mont(s);
            for x in v.iter_mut() { *x = mont_mul(*x, sm); }
        }
    }
}

/// `k` values on `H = <w_n^(n/k)>` -> `n` values on `<w_n>` (`gpu/ligero.py::encode_ext`, one row).
pub fn encode(row: &[Ext], n: usize) -> Vec<Ext> {
    let k = row.len();
    let mut c = row.to_vec();
    Plan::new(k, true).run(&mut c);
    c.resize(n, Ext::ZERO);
    Plan::new(n, false).run(&mut c);
    c
}

/// Evaluates rows (given by their `k` values on `H`) at a fixed set of opened positions `j` of the `n`-point domain
/// without the full size-`n` NTT: `j = (n/k) m' + c` lies on coset `c`; `a(w_n^j) = NTT_k(a_hat_d * w_n^{c d})[m']`,
/// and coset 0 is the row itself.
pub struct RowEval { k: usize, ratio: usize, inv_k: Plan, fwd_k: Plan, twists: Vec<Option<Vec<u32>>>, pub targets: Vec<(usize, usize)>, planar: bool }

impl RowEval {
    pub fn new(k: usize, n: usize, opened: &[usize]) -> RowEval {
        let ratio = n / k;
        let wn = root_of_unity(n);
        let targets: Vec<(usize, usize)> = opened.iter().map(|&j| (j % ratio, j / ratio)).collect();
        let twists = (0..ratio).map(|c| {
            if c == 0 || !targets.iter().any(|t| t.0 == c) { return None; }
            let wc = pow(wn, c as u64);
            let mut cur = 1u32;
            let mut v = Vec::with_capacity(k);
            for _ in 0..k { v.push(to_mont(cur)); cur = mul(cur, wc); }
            Some(v)
        }).collect();
        RowEval { k, ratio, inv_k: Plan::new(k, true), fwd_k: Plan::new(k, false), twists, targets, planar: std::env::var("VERITY_GKR_PLANAR").map(|v| v == "1").unwrap_or(false) }
    }

    /// `out[i] = row(w_n^{opened[i]})`; `scratch` is reused across rows (`2 k DEG` words).  Planar: the six
    /// coordinates are transformed as six base-field vectors (identical twiddles), which vectorises.
    pub fn eval(&self, row: &[Ext], scratch: &mut Vec<u32>, out: &mut [Ext]) {
        let k = self.k;
        assert_eq!(row.len(), k);
        if !self.planar { return self.eval_interleaved(row, scratch, out); }
        scratch.clear();
        scratch.resize(2 * k * DEG, 0);
        let (hat, work) = scratch.split_at_mut(k * DEG);
        for q in 0..DEG {
            let plane = &mut hat[q * k..(q + 1) * k];
            for d in 0..k { plane[d] = row[d].0[q]; }
            self.inv_k.run_base(plane);
        }
        for (i, &(c, m)) in self.targets.iter().enumerate() { if c == 0 { out[i] = row[m]; } }
        for c in 1..self.ratio {
            let Some(tw) = &self.twists[c] else { continue };
            for q in 0..DEG {
                let src = &hat[q * k..(q + 1) * k];
                let dst = &mut work[q * k..(q + 1) * k];
                for ((d, s), &w) in dst.iter_mut().zip(src).zip(tw.iter()) { *d = mont_mul(*s, w); }
                self.fwd_k.run_base(dst);
            }
            for (i, &(cc, m)) in self.targets.iter().enumerate() {
                if cc == c {
                    let mut e = [0u32; DEG];
                    for q in 0..DEG { e[q] = work[q * k + m]; }
                    out[i] = Ext(e);
                }
            }
        }
    }

    /// `R` rows at once (`rows` = `R * k` values, row-major; `out` = `R * t`, row-major): the rows are interleaved
    /// into `L = R * DEG` lanes per position so every NTT butterfly and every twist multiply is an `L`-wide SIMD
    /// loop with one twiddle.  Bit-exact with `eval` row by row.
    pub fn eval_batch<const R: usize, const L: usize>(&self, rows: &[Ext], scratch: &mut Vec<u32>, out: &mut [Ext]) {
        let k = self.k;
        let t = self.targets.len();
        assert_eq!(rows.len(), R * k);
        assert_eq!(out.len(), R * t);
        assert_eq!(L, R * DEG);
        scratch.clear();
        scratch.resize(2 * k * L, 0);
        let (hat, work) = scratch.split_at_mut(k * L);
        for r in 0..R {
            let row = &rows[r * k..(r + 1) * k];
            for d in 0..k { hat[d * L + r * DEG..d * L + (r + 1) * DEG].copy_from_slice(&row[d].0); }
        }
        self.inv_k.run_lanes::<L>(hat);
        for (i, &(c, m)) in self.targets.iter().enumerate() {
            if c == 0 { for r in 0..R { out[r * t + i] = rows[r * k + m]; } }
        }
        for c in 1..self.ratio {
            let Some(tw) = &self.twists[c] else { continue };
            for d in 0..k {
                let w = tw[d];
                let src: &[u32; L] = (&hat[d * L..(d + 1) * L]).try_into().unwrap();
                let dst: &mut [u32; L] = (&mut work[d * L..(d + 1) * L]).try_into().unwrap();
                for l in 0..L { dst[l] = mont_mul(src[l], w); }
            }
            self.fwd_k.run_lanes::<L>(work);
            for (i, &(cc, m)) in self.targets.iter().enumerate() {
                if cc == c {
                    for r in 0..R {
                        let mut e = [0u32; DEG];
                        e.copy_from_slice(&work[m * L + r * DEG..m * L + (r + 1) * DEG]);
                        out[r * t + i] = Ext(e);
                    }
                }
            }
        }
    }

    /// Interleaved variant (one `Ext` butterfly = six independent lanes; the scratch is reinterpreted).
    fn eval_interleaved(&self, row: &[Ext], scratch: &mut Vec<u32>, out: &mut [Ext]) {
        let k = self.k;
        scratch.clear();
        scratch.resize(2 * k * DEG, 0);
        // SAFETY-free reinterpretation: build Ext vectors from the u32 scratch by copying (k is small: 4096)
        let mut hat: Vec<Ext> = row.to_vec();
        let mut work: Vec<Ext> = vec![Ext::ZERO; k];
        self.inv_k.run(&mut hat);
        for (i, &(c, m)) in self.targets.iter().enumerate() { if c == 0 { out[i] = row[m]; } }
        for c in 1..self.ratio {
            let Some(tw) = &self.twists[c] else { continue };
            for d in 0..k {
                let mut e = [0u32; DEG];
                for q in 0..DEG { e[q] = mont_mul(hat[d].0[q], tw[d]); }
                work[d] = Ext(e);
            }
            self.fwd_k.run(&mut work);
            for (i, &(cc, m)) in self.targets.iter().enumerate() { if cc == c { out[i] = work[m]; } }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::Ext;

    #[test]
    fn roundtrip_and_coset_eval() {
        let k = 16;
        let n = 64;
        let row: Vec<Ext> = (0..k as u64).map(|i| Ext([(i * i + 3) as u32, (7 * i) as u32, 1, 2, 3, (i % 5) as u32])).collect();
        let mut v = row.clone();
        Plan::new(k, false).run(&mut v);
        Plan::new(k, true).run(&mut v);
        assert_eq!(v, row);
        let enc = encode(&row, n);
        for m in 0..k { assert_eq!(enc[m * 4], row[m]); }
        let opened = vec![0usize, 1, 2, 3, 17, 30, 63, 44];
        let re = RowEval::new(k, n, &opened);
        let mut out = vec![Ext::ZERO; opened.len()];
        let mut scratch = Vec::new();
        re.eval(&row, &mut scratch, &mut out);
        for (i, &j) in opened.iter().enumerate() { assert_eq!(out[i], enc[j], "position {j}"); }
        // lanes: batched rows agree with the row-by-row evaluation
        let rows: Vec<Ext> = (0..4 * k as u64).map(|i| Ext([(i * 7 + 1) as u32, (i * i) as u32 % 1000, 5, (i % 3) as u32, 9, (i * 13) as u32])).collect();
        let mut outb = vec![Ext::ZERO; 4 * opened.len()];
        re.eval_batch::<4, 24>(&rows, &mut scratch, &mut outb);
        for r in 0..4 {
            let mut o = vec![Ext::ZERO; opened.len()];
            re.eval(&rows[r * k..(r + 1) * k], &mut scratch, &mut o);
            assert_eq!(&outb[r * opened.len()..(r + 1) * opened.len()], &o[..], "row {r}");
        }
        let mut v = rows[..k].to_vec();
        let mut lanes = vec![0u32; k * DEG];
        for d in 0..k { lanes[d * DEG..(d + 1) * DEG].copy_from_slice(&v[d].0); }
        Plan::new(k, false).run(&mut v);
        Plan::new(k, false).run_lanes::<DEG>(&mut lanes);
        for d in 0..k { assert_eq!(&lanes[d * DEG..(d + 1) * DEG], &v[d].0); }
    }
}
