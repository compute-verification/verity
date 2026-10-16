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
//!
//! [`LookupNet::build_v2`] is `table/v2`, the mirror of the verifier's `Lookup.buildV2` (the one specification), held to
//! its vectors (`verity/ml/flock/verifier/lookup_v2_vectors.json`) by [`LookupNet::rows_digest`]: product rows only for the
//! `k` = [`width`] output bits, and each output bit `k <= j < bits` its constant.

use sha2::{Digest, Sha256, Sha512};

use crate::ir_block::{Group, IrUnitNet};

/// The library (core's `verity.ml.circuits.library`, the verifier's `Flock/Library.lean`): name, SHA-512 of the raw u32 LE words,
/// index bits, value bits. A table is `library` exactly when its SHA-512 is listed, and `program` otherwise.
pub const LIBRARY: [(&str, &str, usize, usize); 6] = [
    ("rsq", "39f7b912bc25b913618d926725a18974af9eb246f6f36ce06ef63ffdb293d80bc222623c6d182d2c23e16f5ada2b8db9e7e6a53b3e42046763f7794ac3cef378", 24, 31),
    ("sqrt", "33d076106efaa955e6452200d45651a799628e76d1d1cc682168f464a03100b937bee08533761d9fa8f8a114cee2a32d322888d5f08fd320bfeb17805a5c2536", 24, 31),
    ("rcp", "8a317adbcc078bf70ae01959b679d49f97cd9c8a93d6f39b7325c93d0fd550fa4396a4c3a0568d2387c2ff2f33eb76453b33a7b8890d97752d86cc1d971aeacc", 23, 31),
    ("ex2", "5cb768c1725369ad905e725437463729d5b6054ca9adc498e755247e2ba848b510e3325e1c3fecad1242832aca5fa0afd463b16002a7f34050a7b377ed107877", 23, 31),
    ("tanh_mufu", "f83e87737a231de2ec109f5a7ea9f8dac78089ef9978ac19ce540c03cefd356ce542948ba8ffb11952b68a2d0f63fe4c6e2e84d218739a262a9b4bb07dd66985", 27, 31),
    ("lg2", "6f498dcd26e75335dcec0acdbfadae425c206b36d5a071a417c646db018a53b0b5482455c913c9ed94ab09c0aa5ccbc8d2e4188abae767050df4544ec4d3e550", 23, 31),
];

/// The SHA-512 (hex) of a table's raw u32 LE words: its archive key, and what decides its label.
pub fn table_sha512(table: &[u32]) -> String {
    let mut h = Sha512::new();
    for w in table {
        h.update(w.to_le_bytes());
    }
    h.finalize().iter().map(|x| format!("{x:02x}")).collect()
}

/// One more than the highest of the low `bits` output bits that is not the same in every word (0 if none varies).
pub fn varying(table: &[u32], bits: usize) -> usize {
    let mask = ((1u64 << bits) - 1) as u32;
    let d = table.iter().fold(0u32, |acc, w| acc | (w ^ table[0])) & mask;
    32 - d.leading_zeros() as usize
}

/// The output bits a `table/v2` read gives product rows (`Lookup.width`): a library table's up to its varying width, a
/// program table's every one of its `bits`, so a program slot's shape never depends on the table's values.
pub fn width(sha512: &str, table: &[u32], bits: usize) -> usize {
    if LIBRARY.iter().any(|t| t.1 == sha512) { varying(table, bits) } else { bits }
}

#[derive(Clone)]
pub struct LookupNet {
    pub n: usize,
    pub unit_log: usize,
    pub const_pos: usize,
    pub lo: usize,
    pub bits: usize,
    /// the output bits with product rows (`bits` in v1; [`width`] in v2); each bit `k <= j < bits` is its constant
    pub k: usize,
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
        Self::rows(name, table, n, lo, bits, bits, format!("verity/flock-circuit/lookup {name} n={n} lo={lo} bits={bits}\n"))
    }

    /// **`table/v2`** (`Lookup.buildV2`): `sha512` is the table's SHA-512 (hex), which the caller has checked names these
    /// words; it alone decides the label. A library table is read only in the library's shape.
    pub fn build_v2(name: &str, sha512: &str, table: &'static [u32], n: usize, lo: usize, bits: usize) -> Result<(IrUnitNet, Self), String> {
        if n > 128 || table.len() != 1 << n || !(2..n).contains(&lo) || bits > 32 {
            return Err(format!("lookup {name}: not a table of 2^{n} words"));
        }
        if LIBRARY.iter().any(|t| t.1 == sha512 && (t.2, t.3) != (n, bits)) {
            return Err(format!("lookup {name}: a library table read in a shape other than the library's"));
        }
        let k = width(sha512, table, bits);
        Ok(Self::rows(name, table, n, lo, bits, k, format!("verity/flock-circuit/lookup {name} gen=table/v2 n={n} lo={lo} bits={bits} k={k}\n")))
    }

    /// The slot's rows, with product rows for the output bits below `k` (`header` starts the net's SHA-256 identity).
    fn rows(name: &str, table: &'static [u32], n: usize, lo: usize, bits: usize, k: usize, header: String) -> (IrUnitNet, Self) {
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
        for j in 0..k {
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
            if j < k {
                bd.a.push((0..nh).map(|h| prod0 + j * nh + h).collect());
                bd.b.push(vec![K]);
            } else if j < bits && (table[0] >> j) & 1 == 1 {
                bd.a.push(vec![K]);
                bd.b.push(vec![K]);
            } else {
                bd.a.push(vec![]);
                bd.b.push(vec![]);
            }
        }
        let konst = bd.a.len();
        bd.a.push(vec![konst]);
        bd.b.push(vec![konst]);
        let fix = |rows: &mut Vec<Vec<usize>>| for r in rows.iter_mut() { for c in r.iter_mut() { if *c == K { *c = konst; } } r.sort_unstable(); };
        fix(&mut bd.a);
        fix(&mut bd.b);
        let useful = bd.a.len();
        let mut h = Sha256::new();
        h.update(header.as_bytes());
        for w in table {
            h.update(w.to_le_bytes());
        }
        let unit_log = useful.next_power_of_two().trailing_zeros().max(7) as usize;
        let net = IrUnitNet {
            name: name.to_string(), useful, const_pos: konst,
            in_groups: vec![Group { col: 0, words: 1, bits: vec![n] }],
            out_groups: vec![Group { col: out0 / 128, words: 1, bits: vec![bits] }],
            in_words: 1, out_cols: vec![out0 / 128], a: bd.a, b: bd.b, unit_log, sha256: h.finalize().into(), leaves: None, cut: None,
            flat: Default::default(),
        };
        (net, LookupNet { n, unit_log, const_pos: konst, lo, bits, k, table, lo_rows, hi_rows, prod0, out0 })
    }

    /// The first row of the top low-decoder block: low minterm `l` is row `lo_top + l`.
    pub fn lo_top(&self) -> usize {
        self.lo_rows.last().map_or(0, |b| b.0)
    }

    /// The first product row: output bit `j < k`'s product with high minterm `h` is row `prod0 + j 2^(n - lo) + h`.
    pub fn prod0(&self) -> usize {
        self.prod0
    }

    /// The rows digest the verifier's vectors pin (`Lookup.rowsDigest`): `SHA-512("flock-lookup-rows/v2" ‖ 0 ‖ le32[n, lo,
    /// bits, k, useful, const_pos, prod0, lo_top, unit_log] ‖ per row: le32|A_r| ‖ A_r ‖ le32|B_r| ‖ B_r)`. The product rows'
    /// B side counts as empty, as the verifier's fold reads it from the table (`Lookup.foldB`); `product_b_is_the_table`
    /// holds it to the table instead.
    pub fn rows_digest(&self, net: &IrUnitNet) -> [u8; 64] {
        let mut h = Sha512::new();
        h.update(b"flock-lookup-rows/v2\0");
        for x in [self.n, self.lo, self.bits, self.k, net.useful, self.const_pos, self.prod0, self.lo_top(), self.unit_log] {
            h.update((x as u32).to_le_bytes());
        }
        let products = self.prod0..self.prod0 + (self.k << (self.n - self.lo));
        for r in 0..net.useful {
            let b: &[usize] = if products.contains(&r) { &[] } else { &net.b[r] };
            for row in [&net.a[r][..], b] {
                h.update((row.len() as u32).to_le_bytes());
                for &c in row {
                    h.update((c as u32).to_le_bytes());
                }
            }
        }
        h.finalize().into()
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
        // a constant output bit's row is `[const] * [const]` when the bit is 1 and empty when it is 0
        for j in self.k..self.bits {
            if (self.table[0] >> j) & 1 == 1 {
                let r = self.out0 + j;
                set(&mut z, r);
                set(&mut a, r);
                set(&mut b, r);
            }
        }
        for j in 0..self.k {
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
        let table = random_table(n, 12345, 0x7FFF_FFFF, 0);
        let (net, ln) = LookupNet::build("t", table, n, 4, 31);
        rows_hold(&net, &ln, table, n);
    }

    /// `table/v2`: with constant bits above the varying ones (some 1, some 0), as a library table's rows are built, and as
    /// a program table's (every bit kept).
    #[test]
    fn witness_is_the_rows_v2() {
        let n = 10;
        let table = random_table(n, 777, 0x000F_FFFF, (1 << 30) | (1 << 23) | (1 << 21));
        let k = varying(table, 31);
        assert_eq!(k, 20);
        let (net, ln) = LookupNet::rows("t", table, n, 4, 31, k, String::new());
        assert_eq!(ln.k, 20);
        rows_hold(&net, &ln, table, n);
        let (net, ln) = LookupNet::build_v2("t", &table_sha512(table), table, n, 4, 31).unwrap();
        assert_eq!(ln.k, 31, "a table outside the library keeps every bit");
        rows_hold(&net, &ln, table, n);
    }

    /// The product rows' B side, which the digest leaves to the table: row `(j, h)` reads the low minterms `l` with bit `j`
    /// of `table[h 2^lo + l]` set, as the verifier's `Lookup.foldB` does.
    #[test]
    fn product_b_is_the_table() {
        let n = 9;
        let table = random_table(n, 99, 0x00FF_FFFF, 0);
        let (net, ln) = LookupNet::build_v2("t", &table_sha512(table), table, n, 3, 31).unwrap();
        let (nlo, nh) = (1usize << ln.lo, 1usize << (n - ln.lo));
        for j in 0..ln.k {
            for h in 0..nh {
                let want: Vec<usize> = (0..nlo).filter(|&l| (table[h * nlo + l] >> j) & 1 == 1).map(|l| ln.lo_top() + l).collect();
                assert_eq!(net.b[ln.prod0 + j * nh + h], want, "product row ({j}, {h})");
            }
        }
    }

    #[test]
    fn a_library_table_only_in_the_librarys_shape() {
        let table = random_table(4, 1, 0xFF, 0);
        assert!(matches!(LookupNet::build_v2("t", LIBRARY[2].1, table, 4, 2, 31), Err(e) if e.contains("library's")));
    }

    /// The verifier's vectors (`lookup_v2_vectors.json`, written from the compiled Lean): every small table, and the four
    /// MUFU tables when `FLOCK_TABLE_ARCHIVE` names a directory of tables by SHA-512 (an archive's `sha512/`).
    #[test]
    fn v2_reproduces_the_verifiers_vectors() {
        let repo = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).canonicalize().unwrap().join("../../../..");
        let Ok(text) = std::fs::read_to_string(repo.join("verity/ml/flock/verifier/lookup_v2_vectors.json")) else {
            return;
        };
        let v: serde_json::Value = serde_json::from_str(&text).unwrap();
        let hex = |b: &[u8]| b.iter().map(|x| format!("{x:02x}")).collect::<String>();
        let check = |e: &serde_json::Value, bytes: Vec<u8>| {
            let table: Vec<u32> = bytes.chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
            let table: &'static [u32] = Box::leak(table.into_boxed_slice());
            let sha = table_sha512(table);
            assert_eq!(sha, e["table_sha512"].as_str().unwrap());
            let at = |k: &str| e[k].as_u64().unwrap() as usize;
            let (net, ln) = LookupNet::build_v2("lookup", &sha, table, at("n"), at("lo"), at("bits")).unwrap();
            let got = (LIBRARY.iter().any(|t| t.1 == sha), ln.k, net.useful, ln.unit_log, ln.prod0, ln.lo_top(), hex(&ln.rows_digest(&net)));
            let want = (e["library"].as_bool().unwrap(), at("k"), at("useful"), at("unit_log"), at("prod0"), at("lo_top"),
                        e["rows_sha512"].as_str().unwrap().to_string());
            assert_eq!(got, want, "{}", e["label"]);
        };
        for e in v["small"].as_array().unwrap() {
            let s = e["table_hex"].as_str().unwrap();
            check(e, (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap()).collect());
        }
        if let Ok(dir) = std::env::var("FLOCK_TABLE_ARCHIVE") {
            for e in v["mufu"].as_array().unwrap() {
                check(e, std::fs::read(format!("{dir}/{}", e["table_sha512"].as_str().unwrap())).unwrap());
            }
        }
    }

    fn random_table(n: usize, seed: u64, mask: u32, set: u32) -> &'static [u32] {
        let mut s = seed;
        let table: Vec<u32> = (0..1 << n).map(|_| { s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); ((s >> 33) as u32 & mask) | set }).collect();
        Box::leak(table.into_boxed_slice())
    }

    /// The fast witness is the rows' own evaluation (eval64), satisfies every row, and reads `table[idx]` in its output.
    fn rows_hold(net: &IrUnitNet, ln: &LookupNet, table: &[u32], n: usize) {
        for idx in [0usize, 1, 517, 1023, 700].map(|i| i & ((1 << n) - 1)) {
            let (z, a, b) = ln.witness(idx);
            let (ez, ea, eb) = net.eval64(|col| if col < n { ((idx >> col) & 1) as u64 } else { 0 });
            assert!(net.eval64_flat(|col| if col < n { ((idx >> col) & 1) as u64 } else { 0 }) == (ez.clone(), ea.clone(), eb.clone()));
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
