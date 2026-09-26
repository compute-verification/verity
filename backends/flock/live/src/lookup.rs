//! A table-lookup slot of `verity/flock-circuit` (lane flock-circuit): `value = table[index]` for a pinned MUFU table
//! (`ir_tail::TABLE_SHA256`), built here from the table itself, the same on the prover's and the verifier's side, as an
//! ordinary unit's expanded circuit (`IrUnitNet`): so the fold, Δ and the regions treat it like any other slot type.
//!
//! Rows: the index bits (the input group, one 128-bit word); the one-hot decoders of its low `lo` and high `n - lo` bits
//! (recursively, each minterm the AND of two half minterms); per output bit j and high minterm h one product row
//! `hi_h * (XOR of the low minterms l with bit j of table[h 2^lo + l] set)`; the output group, bit j the XOR of its products;
//! the constant. So a lookup costs about `2^(n-lo) (bits + 1) + 2^lo` committed bits, and the table lives in the matrix.
//!
//! [`LookupNet::witness`] fills the rows from the index directly (the minterms are one-hot, so every row is a table read):
//! what the prover computes, the same values the rows define.

use sha2::{Digest, Sha256};

use crate::ir_block::{Group, IrUnitNet};

pub struct LookupNet {
    pub n: usize,
    pub unit_log: usize,
    pub const_pos: usize,
    pub lo: usize,
    pub bits: usize,
    pub table: &'static [u32],
    /// per decoder level of the low and high index halves: (first row, bit count) of each committed minterm block
    lo_rows: Vec<(usize, usize)>,
    hi_rows: Vec<(usize, usize)>,
    prod0: usize,
    out0: usize,
}

/// A linear form over committed columns: (columns, constant).
type Form = (Vec<usize>, bool);

struct Builder {
    a: Vec<Vec<usize>>,
    b: Vec<Vec<usize>>,
}

impl Builder {
    fn row(&mut self, x: &Form, y: &Form, konst: usize) -> usize {
        let f = |f: &Form| { let mut v = f.0.clone(); if f.1 { v.push(konst); } v };
        self.a.push(f(x));
        self.b.push(f(y));
        self.a.len() - 1
    }

    /// Minterms of `bits` (index i = the assignment of value i): forms for one bit, committed rows above. Records each
    /// committed block's first row and size in `blocks` (the top one last).
    fn decode(&mut self, bits: &[usize], konst: usize, blocks: &mut Vec<(usize, usize)>) -> Vec<Form> {
        if bits.len() == 1 {
            return vec![(vec![bits[0]], true), (vec![bits[0]], false)];
        }
        let h = bits.len() / 2;
        let lo = self.decode(&bits[..h], konst, blocks);
        let hi = self.decode(&bits[h..], konst, blocks);
        let first = self.a.len();
        let out: Vec<Form> = (0..1usize << bits.len()).map(|i| (vec![self.row(&lo[i & ((1 << h) - 1)], &hi[i >> h], konst)], false)).collect();
        blocks.push((first, bits.len()));
        out
    }
}

impl LookupNet {
    /// `name` names the slot type; `n` index bits; `lo` of them in the low decoder; `bits` output bits of each table word.
    pub fn build(name: &str, table: &'static [u32], n: usize, lo: usize, bits: usize) -> (IrUnitNet, Self) {
        assert!(table.len() == 1 << n && (2..n).contains(&lo) && bits <= 32 && n <= 128);
        // rows are placed after the fact: the constant's column is only known at the end, so build with a placeholder and patch
        const K: usize = usize::MAX;
        let mut bd = Builder { a: Vec::new(), b: Vec::new() };
        for i in 0..128 {
            if i < n { bd.a.push(vec![i]); bd.b.push(vec![i]); } else { bd.a.push(vec![]); bd.b.push(vec![]); }
        }
        let idx: Vec<usize> = (0..n).collect();
        let (mut lo_rows, mut hi_rows) = (Vec::new(), Vec::new());
        let lo_m = bd.decode(&idx[..lo], K, &mut lo_rows);
        let hi_m = bd.decode(&idx[lo..], K, &mut hi_rows);
        let nlo = 1usize << lo;
        let prod0 = bd.a.len();
        for j in 0..bits {
            for (h, hm) in hi_m.iter().enumerate() {
                let base = h * nlo;
                let g: Vec<usize> = (0..nlo).filter(|&l| (table[base + l] >> j) & 1 == 1).map(|l| lo_m[l].0[0]).collect();
                bd.row(hm, &(g, false), K);
            }
        }
        while bd.a.len() % 128 != 0 {
            bd.a.push(vec![]);
            bd.b.push(vec![]);
        }
        let out0 = bd.a.len();
        let nh = hi_m.len();
        for j in 0..128 {
            if j < bits { bd.a.push((0..nh).map(|h| prod0 + j * nh + h).collect()); bd.b.push(vec![K]); } else { bd.a.push(vec![]); bd.b.push(vec![]); }
        }
        let konst = bd.a.len();
        bd.a.push(vec![konst]);
        bd.b.push(vec![konst]);
        let fix = |rows: &mut Vec<Vec<usize>>| for r in rows.iter_mut() { for c in r.iter_mut() { if *c == K { *c = konst; } } r.sort_unstable(); };
        fix(&mut bd.a);
        fix(&mut bd.b);
        let useful = bd.a.len();
        let mut h = Sha256::new();
        h.update(format!("verity/flock-circuit/lookup {name} n={n} lo={lo} bits={bits}\n").as_bytes());
        for w in table {
            h.update(w.to_le_bytes());
        }
        let unit_log = useful.next_power_of_two().trailing_zeros().max(7) as usize;
        let net = IrUnitNet {
            name: name.to_string(), useful, const_pos: konst,
            in_groups: vec![Group { col: 0, words: 1, bits: vec![n] }],
            out_groups: vec![Group { col: out0 / 128, words: 1, bits: vec![bits] }],
            in_words: 1, out_cols: vec![out0 / 128], a: bd.a, b: bd.b, unit_log, sha256: h.finalize().into(), leaves: None, cut: None,
        };
        (net, LookupNet { n, unit_log, const_pos: konst, lo, bits, table, lo_rows, hi_rows, prod0, out0 })
    }

    /// The honest (z, a, b) bits of one lookup of `index`, as 64-bit words over the slot's rows (bit r of word r / 64).
    pub fn witness(&self, index: usize) -> (Vec<u64>, Vec<u64>, Vec<u64>) {
        let w = (1usize << self.unit_log) / 64;
        let (mut z, mut a, mut b) = (vec![0u64; w], vec![0u64; w], vec![0u64; w]);
        let set = |v: &mut Vec<u64>, r: usize| v[r / 64] |= 1 << (r % 64);
        let idx = index & ((1 << self.n) - 1);
        for i in 0..self.n {
            if (idx >> i) & 1 == 1 {
                set(&mut z, i);
                set(&mut a, i);
                set(&mut b, i);
            }
        }
        // a decoder block of k bits at `first` over bits [off, off + k): its minterm i is the AND of the half minterms, so only
        // row (the block's own value) is 1; its a / b are the half minterms' values (a form's value for a one-bit half)
        let decoders = |z: &mut Vec<u64>, a: &mut Vec<u64>, b: &mut Vec<u64>, blocks: &[(usize, usize)], off: usize, total: usize| {
            // blocks are recorded post-order: rebuild the recursion to know each block's bit range
            fn walk(k: usize, off: usize, blocks: &[(usize, usize)], at: &mut usize, out: &mut Vec<(usize, usize, usize)>) {
                if k == 1 {
                    return;
                }
                let h = k / 2;
                walk(h, off, blocks, at, out);
                walk(k - h, off + h, blocks, at, out);
                let (first, kk) = blocks[*at];
                assert_eq!(kk, k);
                *at += 1;
                out.push((first, k, off));
            }
            let mut spans = Vec::new();
            let mut at = 0;
            walk(total, off, blocks, &mut at, &mut spans);
            for (first, k, o) in spans {
                let v = (idx >> o) & ((1 << k) - 1);
                let h = k / 2;
                let (vl, vh) = (v & ((1 << h) - 1), v >> h);
                for i in 0..1usize << k {
                    let (il, ih) = (i & ((1 << h) - 1), i >> h);
                    let r = first + i;
                    let (x, y) = (il == vl, ih == vh);
                    if x { set(a, r); }
                    if y { set(b, r); }
                    if x && y { set(z, r); }
                }
            }
        };
        decoders(&mut z, &mut a, &mut b, &self.lo_rows, 0, self.lo);
        decoders(&mut z, &mut a, &mut b, &self.hi_rows, self.lo, self.n - self.lo);
        let (nlo, nh) = (1usize << self.lo, 1usize << (self.n - self.lo));
        let (l, hsel) = (idx & (nlo - 1), idx >> self.lo);
        for j in 0..self.bits {
            for h in 0..nh {
                let r = self.prod0 + j * nh + h;
                let g = (self.table[h * nlo + l] >> j) & 1 == 1;
                if h == hsel { set(&mut a, r); }
                if g { set(&mut b, r); }
                if h == hsel && g { set(&mut z, r); }
            }
            if (self.table[idx] >> j) & 1 == 1 {
                let r = self.out0 + j;
                set(&mut z, r);
                set(&mut a, r);
            }
            set(&mut b, self.out0 + j);
        }
        let c = self.const_pos;
        set(&mut z, c);
        set(&mut a, c);
        set(&mut b, c);
        (z, a, b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The fast witness is the rows' own evaluation (eval64), and it satisfies every row, on a small random table.
    #[test]
    fn witness_is_the_rows() {
        let n = 10;
        let mut s = 12345u64;
        let table: Vec<u32> = (0..1 << n).map(|_| { s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (s >> 33) as u32 & 0x7FFF_FFFF }).collect();
        let table: &'static [u32] = Box::leak(table.into_boxed_slice());
        let (net, ln) = LookupNet::build("t", table, n, 4, 31);
        for idx in [0usize, 1, 517, 1023, 700] {
            let (z, a, b) = ln.witness(idx);
            let (ez, ea, eb) = net.eval64(|col| if col < n { ((idx >> col) & 1) as u64 } else { 0 });
            for r in 0..net.useful {
                let bit = |v: &Vec<u64>| (v[r / 64] >> (r % 64)) & 1;
                assert_eq!(bit(&z), ez[r] & 1, "z row {r} idx {idx}");
                assert_eq!(bit(&a), ea[r] & 1, "a row {r} idx {idx}");
                assert_eq!(bit(&b), eb[r] & 1, "b row {r} idx {idx}");
                assert_eq!(bit(&a) & bit(&b), bit(&z), "row {r} unsatisfied");
            }
            let out: u32 = (0..31).map(|j| (bit_of(&z, ln.out0 + j) as u32) << j).sum();
            assert_eq!(out, table[idx]);
        }
    }

    fn bit_of(v: &[u64], r: usize) -> u64 {
        (v[r / 64] >> (r % 64)) & 1
    }
}
