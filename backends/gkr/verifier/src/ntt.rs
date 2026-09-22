//! Radix-2 NTT over BabyBear on extension vectors (the six coordinates share every twiddle), Montgomery twiddles.
//! Semantics of `gpu/ligero.py::ntt`: forward evaluates `sum_m c_m w^{m j}` on `<w>`, `w = 31^((p-1)/n)`; inverse
//! uses `w^-1` and scales by `n^-1`.  Natural order in and out.

use crate::field::{add, inv, mont_mul, mul, pow, root_of_unity, sub, to_mont, Ext, DEG};

pub struct Plan { pub n: usize, tw: Vec<Vec<u32>>, rev: Vec<u32>, scale: Option<u32>, w: u32, prune_stages: usize, sub_pows: Vec<u32> }

fn bitrev(x: usize, bits: usize) -> usize { if bits == 0 { 0 } else { x.reverse_bits() >> (usize::BITS - bits as u32) } }

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
        Plan { n, tw, rev, scale: if inverse { Some(inv(n as u32)) } else { None }, w, prune_stages: 0, sub_pows: vec![] }
    }

    /// Prepares `eval_pruned` with `stages` leading stages (sub-transform size `n >> stages`).
    pub fn with_pruning(mut self, stages: usize) -> Plan {
        let m = self.n >> stages;
        let wm = pow(self.w, 1u64 << stages);
        let mut cur = 1u32;
        self.sub_pows = (0..m).map(|_| { let r = to_mont(cur); cur = mul(cur, wm); r }).collect();
        self.prune_stages = stages;
        self
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

    /// Output-pruned forward transform on `L` lanes (natural-order input): `stages` decimation-in-frequency stages
    /// (the first `stages` levels of the full transform, `n/2` butterflies each) leave `2^stages` independent
    /// sub-transforms of size `M = n / 2^stages`; the wanted outputs `X[k]` are then read directly from the
    /// sub-transform `bitrev_stages(k mod 2^stages)` as `M`-term sums (`X[k] = sum_j y[j] w_M^(j (k >> stages))`).
    /// Cheaper than the full transform whenever the wanted set is small (`~64` of `4096` here).  `out[i]` receives the
    /// `L` lanes of `X[wanted[i]]`.  Forward plans only.
    pub fn eval_pruned<const L: usize>(&self, v: &mut [u32], stages: usize, wanted: &[usize], out: &mut [u32]) {
        let n = self.n;
        assert!(self.scale.is_none() && stages <= n.trailing_zeros() as usize);
        assert_eq!(v.len(), n * L);
        let levels = self.tw.len();
        let mut len = n;
        for s in 0..stages {
            let half = len / 2;
            let tw = &self.tw[levels - 1 - s][..half];
            for start in (0..n).step_by(len) {
                let blk = &mut v[start * L..(start + len) * L];
                let (lo, hi) = blk.split_at_mut(half * L);
                for (j, &w) in tw.iter().enumerate() {
                    let a: &mut [u32; L] = (&mut lo[j * L..(j + 1) * L]).try_into().unwrap();
                    let b: &mut [u32; L] = (&mut hi[j * L..(j + 1) * L]).try_into().unwrap();
                    for l in 0..L {
                        let u = a[l];
                        let x = b[l];
                        a[l] = add(u, x);
                        b[l] = mont_mul(sub(u, x), w);
                    }
                }
            }
            len = half;
        }
        let m = len; // sub-transform size
        assert_eq!(stages, self.prune_stages, "plan prepared for a different pruning depth");
        let wm = &self.sub_pows; // w_M^i, i < M, Montgomery
        debug_assert_eq!(wm.len(), m);
        for (i, &k) in wanted.iter().enumerate() {
            let b = bitrev(k & ((1 << stages) - 1), stages);
            let kp = k >> stages;
            let y = &v[b * m * L..(b + 1) * m * L];
            let o: &mut [u32; L] = (&mut out[i * L..(i + 1) * L]).try_into().unwrap();
            *o = [0u32; L];
            let mut e = 0usize;
            for j in 0..m {
                let w = wm[e];
                let yj: &[u32; L] = (&y[j * L..(j + 1) * L]).try_into().unwrap();
                for l in 0..L { o[l] = add(o[l], mont_mul(yj[l], w)); }
                e += kp;
                if e >= m { e -= m; }
            }
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
pub struct RowEval { k: usize, ratio: usize, inv_k: Plan, fwd_k: Plan, twists: Vec<Option<Vec<u32>>>, pub targets: Vec<(usize, usize)>, planar: bool, prune: usize, wanted: Vec<Vec<usize>>, wanted_slot: Vec<Vec<usize>> }

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
        // pruning depth: sub-transforms of ~64 points (`VERITY_GKR_PRUNE=0` disables)
        let prune: usize = std::env::var("VERITY_GKR_PRUNE").ok().and_then(|v| v.parse().ok()).unwrap_or_else(|| (k.trailing_zeros() as usize).saturating_sub(6));
        let mut wanted = vec![Vec::new(); ratio];
        let mut wanted_slot = vec![Vec::new(); ratio];
        for (i, &(c, m)) in targets.iter().enumerate() { wanted[c].push(m); wanted_slot[c].push(i); }
        RowEval { k, ratio, inv_k: Plan::new(k, true), fwd_k: Plan::new(k, false).with_pruning(prune), twists, targets, planar: std::env::var("VERITY_GKR_PLANAR").map(|v| v == "1").unwrap_or(false), prune, wanted, wanted_slot }
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
        scratch.resize(2 * k * L + t * L, 0);
        let (hat, rest) = scratch.split_at_mut(k * L);
        let (work, pruned) = rest.split_at_mut(k * L);
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
            if self.prune > 0 {
                let wl = &self.wanted[c];
                self.fwd_k.eval_pruned::<L>(work, self.prune, wl, &mut pruned[..wl.len() * L]);
                for (wi, &i) in self.wanted_slot[c].iter().enumerate() {
                    for r in 0..R {
                        let mut e = [0u32; DEG];
                        e.copy_from_slice(&pruned[wi * L + r * DEG..wi * L + (r + 1) * DEG]);
                        out[r * t + i] = Ext(e);
                    }
                }
            } else {
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
        // pruned forward transform: every output, every pruning depth
        for stages in 0..=4 {
            let plan = Plan::new(k, false).with_pruning(stages);
            let mut l2 = vec![0u32; k * DEG];
            for d in 0..k { l2[d * DEG..(d + 1) * DEG].copy_from_slice(&rows[d].0); }
            let wanted: Vec<usize> = (0..k).collect();
            let mut o = vec![0u32; k * DEG];
            plan.eval_pruned::<DEG>(&mut l2, stages, &wanted, &mut o);
            for d in 0..k { assert_eq!(&o[d * DEG..(d + 1) * DEG], &v[d].0, "stages {stages} output {d}"); }
        }
        std::env::set_var("VERITY_GKR_PRUNE", "2");
        let re2 = RowEval::new(k, n, &opened);
        let mut outp = vec![Ext::ZERO; 4 * opened.len()];
        re2.eval_batch::<4, 24>(&rows, &mut scratch, &mut outp);
        assert_eq!(outp, outb);
        std::env::remove_var("VERITY_GKR_PRUNE");
    }
}
