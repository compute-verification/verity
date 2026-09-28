//! A native witness for verity/flock-circuit's SHA-512 slot circuits: `sha512x3` (three compressions) and `hm96` (a row's
//! padding compression, the salt's two compressions and `b || c`). Every row's `z`, `a` and `b` come from SHA-512 computed in
//! 64-bit words, where `IrUnitNet::eval64` (the host) and `fc_unit_witness` (the device, level by level) evaluate the rows.
//!
//! Every row reads bits of a few quantities of a compression, and `z = a & b` on every row:
//! - an addition `x + y` (`sha512_circuit._add2`): bit `i`'s carry row is `(x ^ c)_i & (y ^ c)_i`, with `c = (x + y) ^ x ^ y`
//!   (bit `i` the carry into bit `i`), and a committed carry is `c_{i+1}`;
//! - a committed word (each schedule word, each round's `a` and `e`): its bits;
//! - `Ch`'s rows `e_i & (f ^ g)_i` and `Maj`'s `(a ^ c)_i & (b ^ c)_i`;
//! - the input rows and the output copy rows: the ports' bits.
//!
//! [`Tape`] records those words in the circuit's creation order ([`compression`] follows `compress_gadget` operation for
//! operation). A [`NativeSha`] plan says, per row of a pinned slot circuit, which bit of which tape word its `a` and its `b`
//! are. It is derived from the circuit itself: the rows between the inputs and the outputs are matched in order against the
//! tape's candidate rows (every carry row and every commit a compression could have) on 256 random lanes of `eval64`, so the
//! builder's constant folding (a round constant's low zero bits, the salt's constant midstate, the carry-commit spacing) is
//! read off the circuit, not modelled here: a candidate the circuit folded away is skipped. [`NativeSha::check`] then compares
//! every row's `z`, `a` and `b` with `eval64` on fresh random lanes and on all-zero and all-one inputs. The proof system checks
//! whatever witness it is given, so a wrong plan could cost completeness, never soundness.

use flock_field::F128;
use flock_merkle::hm96;

use crate::circuit::{compress, lanes, port_cols, transpose64, SHA512_IV, SHA512_K};
use crate::ir_block::IrUnitNet;

const WORD: usize = 128;
/// A row source: [`ZERO`], [`ONE`], or `2 + 64 w + i` for bit `i` of tape word `w`.
pub const ZERO: u32 = 0;
pub const ONE: u32 = 1;
/// Random 64-lane batches the plan is matched on (256 lanes), and fresh ones [`NativeSha::check`] compares on.
const MATCH_BATCHES: u64 = 4;
const CHECK_BATCHES: u64 = 4;

fn src(w: usize, i: usize) -> u32 {
    (2 + 64 * w + i) as u32
}

/// A slot's quantities in creation order and, while a plan is derived, its candidate rows `(a source, b source)`.
pub struct Tape {
    pub words: Vec<u64>,
    cands: Option<Vec<(u32, u32)>>,
}

impl Tape {
    pub fn new(record: bool) -> Self {
        Tape { words: Vec::with_capacity(11_000), cands: record.then(Vec::new) }
    }

    fn push(&mut self, w: u64) -> usize {
        self.words.push(w);
        self.words.len() - 1
    }

    /// `x + y`: pushes `x ^ c`, `y ^ c`, `c`, `x + y`. Candidates per bit `i < 63`: its carry row, then its committed carry.
    fn add(&mut self, x: u64, y: u64) -> u64 {
        let s = x.wrapping_add(y);
        let c = s ^ x ^ y;
        let w = self.push(x ^ c);
        self.push(y ^ c);
        self.push(c);
        self.push(s);
        if let Some(k) = &mut self.cands {
            for i in 0..63 {
                k.push((src(w, i), src(w + 1, i)));
                k.push((src(w + 2, i + 1), ONE));
            }
        }
        s
    }

    /// The sum just added, committed.
    fn commit(&mut self) {
        let w = self.words.len() - 1;
        if let Some(k) = &mut self.cands {
            k.extend((0..64).map(|i| (src(w, i), ONE)));
        }
    }

    /// A word of ANDs `x_i & y_i`: pushes `x`, `y`.
    fn and(&mut self, x: u64, y: u64) {
        let w = self.push(x);
        self.push(y);
        if let Some(k) = &mut self.cands {
            k.extend((0..64).map(|i| (src(w, i), src(w + 1, i))));
        }
    }
}

/// One SHA-512 compression with the feed-forward, as `sha512_circuit.compress_gadget` builds it: the schedule words, then each
/// round's `Ch`, `t1`, `Maj`, `t2`, `e` and `a`, then the feed-forward (words little-endian bit lists there, `u64` here).
pub fn compression(t: &mut Tape, cv: [u64; 8], msg: &[u64; 16]) -> [u64; 8] {
    let mut w = [0u64; 80];
    w[..16].copy_from_slice(msg);
    for i in 16..80 {
        let s0 = w[i - 15].rotate_right(1) ^ w[i - 15].rotate_right(8) ^ (w[i - 15] >> 7);
        let s1 = w[i - 2].rotate_right(19) ^ w[i - 2].rotate_right(61) ^ (w[i - 2] >> 6);
        let x = t.add(w[i - 16], s0);
        let x = t.add(x, w[i - 7]);
        w[i] = t.add(x, s1);
        t.commit();
    }
    let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = cv;
    for (&k, &wi) in SHA512_K.iter().zip(&w) {
        let s1 = e.rotate_right(14) ^ e.rotate_right(18) ^ e.rotate_right(41);
        t.and(e, f ^ g);
        let ch = g ^ (e & (f ^ g));
        let x = t.add(h, s1);
        let x = t.add(x, ch);
        let x = t.add(x, k);
        let t1 = t.add(x, wi);
        let s0 = a.rotate_right(28) ^ a.rotate_right(34) ^ a.rotate_right(39);
        t.and(a ^ c, b ^ c);
        let mj = c ^ ((a ^ c) & (b ^ c));
        let t2 = t.add(s0, mj);
        let ne = t.add(d, t1);
        t.commit();
        let na = t.add(t1, t2);
        t.commit();
        (h, g, f, e, d, c, b, a) = (g, f, e, ne, c, b, a, na);
    }
    let st = [a, b, c, d, e, f, g, h];
    std::array::from_fn(|i| t.add(cv[i], st[i]))
}

/// The slot circuits this witness knows, by their input ports (`circuit.py`'s `sha512x3` and `hm96` layouts).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Slot {
    Sha512x3,
    Hm96,
}

const SHA_PORTS: [(usize, usize); 6] = [(0, 512), (512, 1024), (1536, 512), (2048, 1024), (3072, 512), (3584, 1024)];
const HM_PORTS: [(usize, usize); 3] = [(0, 512), (512, 1024), (1536, 1536)];

fn port<const N: usize>(inp: &[u64], col: usize) -> [u64; N] {
    std::array::from_fn(|i| inp[col / 64 + i])
}

/// One slot's tape from its input columns (`inp`: the input group's bits, 64 per word): the input words, the compressions, then
/// the output ports' bits (`out0 .. out2`, or hm96's `b || c` in byte-string order).
pub fn tape(slot: Slot, inp: &[u64], t: &mut Tape) {
    for &x in inp {
        t.push(x);
    }
    match slot {
        Slot::Sha512x3 => {
            let outs: Vec<[u64; 8]> = (0..3).map(|k| compression(t, port(inp, SHA_PORTS[2 * k].0), &port(inp, SHA_PORTS[2 * k + 1].0))).collect();
            for x in outs.into_iter().flatten() {
                t.push(x);
            }
        }
        Slot::Hm96 => {
            // x = the row's digest (its last chaining value and the padding block); c = SHA-512(salt_prefix || y), from the
            // prefix's midstate; b = x XOR M(key) y (`sha512_circuit.hm96_row`)
            let x = compression(t, port(inp, 0), &port(inp, 512));
            let salt: Vec<u8> = inp[1536 / 64..1536 / 64 + 24].iter().flat_map(|w| w.to_le_bytes()).collect();
            let yw: [u64; 24] = std::array::from_fn(|i| u64::from_be_bytes(salt[8 * i..8 * i + 8].try_into().unwrap()));
            let mut mid = SHA512_IV;
            compress(&mut mid, &hm96::consts().salt_prefix);
            let st = compression(t, mid, yw[..16].try_into().unwrap());
            let mut last = [0u64; 16];
            last[..8].copy_from_slice(&yw[16..]);
            last[8] = 0x80 << 56;
            last[15] = 8 * (128 + hm96::SALT_BYTES as u64);
            let c = compression(t, st, &last);
            let m = hm96::mask(&salt);
            let mut bc = [0u8; 128];
            for i in 0..8 {
                bc[8 * i..8 * i + 8].copy_from_slice(&x[i].to_be_bytes());
                bc[64 + 8 * i..72 + 8 * i].copy_from_slice(&c[i].to_be_bytes());
            }
            for (b, m) in bc.iter_mut().zip(m) {
                *b ^= m;
            }
            for i in 0..16 {
                t.push(u64::from_le_bytes(bc[8 * i..8 * i + 8].try_into().unwrap()));
            }
        }
    }
}

/// 64 lanes' tapes, bit-sliced: `bits[64 w + i]` is bit `i` of tape word `w` (lane `l` in bit `l`).
fn sliced_tapes(slot: Slot, lanes_in: &[Vec<u64>]) -> Vec<u64> {
    let tapes: Vec<Vec<u64>> = lanes_in.iter().map(|inp| {
        let mut t = Tape::new(false);
        tape(slot, inp, &mut t);
        t.words
    }).collect();
    let n = tapes[0].len();
    let mut bits = vec![0u64; 64 * n];
    for w in 0..n {
        let mut m = [0u64; 64];
        for (l, t) in tapes.iter().enumerate() {
            m[l] = t[w];
        }
        transpose64(&mut m);
        bits[64 * w..64 * w + 64].copy_from_slice(&m);
    }
    bits
}

fn sliced(bits: &[u64], s: u32) -> u64 {
    match s {
        ZERO => 0,
        ONE => !0,
        s => bits[(s - 2) as usize],
    }
}

/// A batch of 64 lanes: their input words, `eval64`'s `(z, a, b)` and their bit-sliced tapes.
struct Batch {
    z: Vec<u64>,
    a: Vec<u64>,
    b: Vec<u64>,
    tapes: Vec<u64>,
}

impl Batch {
    fn new(slot: Slot, net: &IrUnitNet, lanes_in: &[Vec<u64>]) -> Self {
        let cols: Vec<[u64; 64]> = (0..lanes_in[0].len()).map(|j| {
            let mut m = [0u64; 64];
            for (l, inp) in lanes_in.iter().enumerate() {
                m[l] = inp[j];
            }
            transpose64(&mut m);
            m
        }).collect();
        let (z, a, b) = net.eval64(|col| cols[col / 64][col % 64]);
        Batch { z, a, b, tapes: sliced_tapes(slot, lanes_in) }
    }
}

/// 64 lanes of input words: each port bit from `fill` (a lane and word index to 64 bits), the columns outside the ports zero.
fn lanes_of(slot: Slot, in_words: usize, fill: impl Fn(usize, usize) -> u64) -> Vec<Vec<u64>> {
    let ports: &[(usize, usize)] = match slot {
        Slot::Sha512x3 => &SHA_PORTS,
        Slot::Hm96 => &HM_PORTS,
    };
    (0..64).map(|l| {
        let mut v = vec![0u64; in_words];
        for &(col, width) in ports {
            for j in col / 64..(col + width) / 64 {
                v[j] = fill(l, j);
            }
        }
        v
    }).collect()
}

fn random_lanes(slot: Slot, in_words: usize, seed: u64) -> Vec<Vec<u64>> {
    let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    let mut rows = vec![vec![0u64; in_words]; 64];
    for r in rows.iter_mut() {
        for x in r.iter_mut() {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            *x = s;
        }
    }
    lanes_of(slot, in_words, |l, j| rows[l][j])
}

/// A pinned slot circuit's native witness plan.
pub struct NativeSha {
    pub slot: Slot,
    /// the slot's input words (the input group's bits, 64 per word)
    pub in_words: usize,
    /// its 128-bit witness words per slot (`2^(unit_log - 7)`, as `Stmt::slot_zab` packs a slot)
    pub uw: usize,
    /// per row, the sources of its `a` and its `b`; `z = a & b`
    pub rows: Vec<(u32, u32)>,
}

impl NativeSha {
    /// The plan of `net` (a `sha512x3` or `hm96` slot circuit), checked against `eval64` ([`NativeSha::check`]).
    pub fn derive(net: &IrUnitNet) -> Result<Self, String> {
        let ports = port_cols(&net.in_groups);
        let slot = if ports == SHA_PORTS {
            Slot::Sha512x3
        } else if ports == HM_PORTS {
            Slot::Hm96
        } else {
            return Err(format!("{}: input ports {ports:?} are neither sha512x3's nor hm96's", net.name));
        };
        let (in_bits, in_words) = (net.in_words * WORD, net.in_words * 2);
        let batches: Vec<Batch> = (0..MATCH_BATCHES).map(|k| Batch::new(slot, net, &random_lanes(slot, in_words, 1 + k))).collect();
        let mut t = Tape::new(true);
        tape(slot, &vec![0u64; in_words], &mut t);
        let (n_words, cands) = (t.words.len(), t.cands.take().unwrap());
        let mut rows = vec![(ZERO, ZERO); net.useful];
        for (r, row) in rows.iter_mut().enumerate().take(in_bits) {
            if net.is_input(r) {
                *row = (src(r / 64, r % 64), src(r / 64, r % 64));
            }
        }
        let out_ports = port_cols(&net.out_groups);
        let out_bits: usize = out_ports.iter().map(|p| p.1).sum();
        let out_base = 64 * n_words - out_bits;
        let mut off = 0;
        for &(col, width) in &out_ports {
            for j in 0..width {
                let q = out_base + off + j;
                rows[col + j] = (src(q / 64, q % 64), ONE);
            }
            off += width;
        }
        rows[net.const_pos] = (ONE, ONE);
        let out_start = net.out_groups[0].col * WORD;
        let matches = |c: (u32, u32), r: usize| batches.iter().all(|b| sliced(&b.tapes, c.0) == b.a[r] && sliced(&b.tapes, c.1) == b.b[r]);
        let mut k = 0;
        for r in in_bits..out_start {
            if net.a[r].is_empty() && net.b[r].is_empty() {
                continue;
            }
            while k < cands.len() && !matches(cands[k], r) {
                k += 1;
            }
            if k == cands.len() {
                return Err(format!("{}: row {r} matches none of the native tape's candidate rows left", net.name));
            }
            rows[r] = cands[k];
            k += 1;
        }
        let plan = NativeSha { slot, in_words, uw: 1 << (net.unit_log - 7), rows };
        plan.check(net)?;
        Ok(plan)
    }

    /// Every row's `z`, `a` and `b` equal `eval64`'s on fresh random lanes, all-zero inputs and all-one inputs.
    pub fn check(&self, net: &IrUnitNet) -> Result<(), String> {
        let mut sets: Vec<(String, Vec<Vec<u64>>)> =
            (0..CHECK_BATCHES).map(|k| (format!("random batch {k}"), random_lanes(self.slot, self.in_words, 1000 + k))).collect();
        sets.push(("all-zero inputs".into(), lanes_of(self.slot, self.in_words, |_, _| 0)));
        sets.push(("all-one inputs".into(), lanes_of(self.slot, self.in_words, |_, _| !0)));
        for (what, lanes_in) in sets {
            let bt = Batch::new(self.slot, net, &lanes_in);
            let (z, a, b) = self.eval64_from(&bt.tapes);
            if let Some(r) = (0..z.len()).find(|&r| (z[r], a[r], b[r]) != (bt.z[r], bt.a[r], bt.b[r])) {
                return Err(format!("{}: row {r} differs from eval64 on {what}", net.name));
            }
        }
        Ok(())
    }

    fn eval64_from(&self, bits: &[u64]) -> (Vec<u64>, Vec<u64>, Vec<u64>) {
        let a: Vec<u64> = self.rows.iter().map(|&(s, _)| sliced(bits, s)).collect();
        let b: Vec<u64> = self.rows.iter().map(|&(_, s)| sliced(bits, s)).collect();
        let z = a.iter().zip(&b).map(|(x, y)| x & y).collect();
        (z, a, b)
    }

    /// Up to 64 slots at once, as `IrUnitNet::eval64` (`input(col)`: the lanes' bit at input column `col`), bit-sliced.
    pub fn eval64(&self, input: impl Fn(usize) -> u64) -> (Vec<u64>, Vec<u64>, Vec<u64>) {
        let mut lanes_in = vec![vec![0u64; self.in_words]; 64];
        for j in 0..self.in_words {
            let mut m: [u64; 64] = std::array::from_fn(|i| input(64 * j + i));
            transpose64(&mut m);
            for (l, x) in m.into_iter().enumerate() {
                lanes_in[l][j] = x;
            }
        }
        self.eval64_from(&sliced_tapes(self.slot, &lanes_in))
    }

    /// One slot's `(z, a, b)` words from its input words (`Stmt::row_inputs`' per-slot words), as `Stmt::slot_zab` packs a slot:
    /// what one device thread computes for one slot.
    pub fn slot_zab(&self, ins: &[F128]) -> (Vec<F128>, Vec<F128>, Vec<F128>) {
        let inp: Vec<u64> = ins.iter().flat_map(|w| [w.lo, w.hi]).collect();
        let mut t = Tape::new(false);
        tape(self.slot, &inp, &mut t);
        let bit = |s: u32| -> u64 {
            match s {
                ZERO => 0,
                ONE => 1,
                s => (t.words[((s - 2) / 64) as usize] >> ((s - 2) % 64)) & 1,
            }
        };
        let (mut z, mut a, mut b) = (vec![F128::ZERO; self.uw], vec![F128::ZERO; self.uw], vec![F128::ZERO; self.uw]);
        for (r, &(sa, sb)) in self.rows.iter().enumerate() {
            let (x, y) = (bit(sa), bit(sb));
            let (k, o) = (r / WORD, r % WORD);
            for (v, x) in [(&mut z, x & y), (&mut a, x), (&mut b, y)] {
                if o < 64 { v[k].lo |= x << o } else { v[k].hi |= x << (o - 64) }
            }
        }
        (z, a, b)
    }

    /// Up to 64 slots' `(z, a, b)` words, lane by lane: [`NativeSha::eval64`] packed as `Stmt::slot_zab` packs it.
    pub fn slots_zab(&self, ins: &[Vec<F128>]) -> Vec<(Vec<F128>, Vec<F128>, Vec<F128>)> {
        let bit = |r: &Vec<F128>, col: usize| (if col % WORD < 64 { r[col / WORD].lo >> (col % 64) } else { r[col / WORD].hi >> (col % 64) }) & 1;
        let (z, a, b) = self.eval64(|col| ins.iter().enumerate().fold(0u64, |x, (l, r)| x | (bit(r, col) << l)));
        let (z, a, b) = (lanes(&z, ins.len(), self.uw), lanes(&a, ins.len(), self.uw), lanes(&b, ins.len(), self.uw));
        z.into_iter().zip(a).zip(b).map(|((z, a), b)| (z, a, b)).collect()
    }

    /// The most tape words a streaming emitter keeps behind the newest word a row has read, over the rows between the inputs
    /// and the outputs (a device kernel that computes the tape as it writes rows holds this many words).
    pub fn window(&self, net: &IrUnitNet) -> usize {
        let (mut hi, mut back) = (0usize, 0usize);
        for &(sa, sb) in &self.rows[net.in_words * WORD..net.out_groups[0].col * WORD] {
            for s in [sa, sb].into_iter().filter(|&s| s > ONE) {
                let w = ((s - 2) / 64) as usize;
                hi = hi.max(w);
                back = back.max(hi - w);
            }
        }
        back
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    fn xorshift(s: &mut u64) -> u64 {
        *s ^= *s << 13;
        *s ^= *s >> 7;
        *s ^= *s << 17;
        *s
    }

    #[test]
    fn native_compression_is_sha512() {
        let mut s = 7u64;
        for _ in 0..16 {
            let cv: [u64; 8] = std::array::from_fn(|_| xorshift(&mut s));
            let msg: [u64; 16] = std::array::from_fn(|_| xorshift(&mut s));
            let block: Vec<u8> = msg.iter().flat_map(|w| w.to_be_bytes()).collect();
            let mut want = cv;
            compress(&mut want, &block);
            assert_eq!(compression(&mut Tape::new(false), cv, &msg), want);
        }
    }

    /// The `sha512x3` and `hm96` slot circuits as `verity_flock.circuit.compose` writes them (`CARRY_EVERY` the carry-commit
    /// spacing: 5 is the pinned circuit's), from this repository's builder; `None` without Python and the verity packages.
    fn slot_circuits(carry_every: usize) -> Option<Vec<IrUnitNet>> {
        let repo = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).canonicalize().ok()?.join("../../..");
        let dir = std::env::temp_dir().join(format!("sha512-native-{}-{carry_every}", std::process::id()));
        std::fs::create_dir_all(&dir).ok()?;
        let py = format!(
            "import sys\nsys.path[:0] = ['{r}/backends/flock/python', '{r}/packages/verity/src', '{r}/backends/numerical/python']\n\
             from verity_flock import ir_lower as IL, sha512_circuit as S\nS.CARRY_EVERY = {carry_every}\n\
             open('{d}/sha512x3.txt', 'w').write(IL.netlist(IL.layout(S.compression3(), ['cv0', 'msg0', 'cv1', 'msg1', 'cv2', 'msg2'], ['out0', 'out1', 'out2']), 'sha512x3', None, None))\n\
             open('{d}/hm96.txt', 'w').write(IL.netlist(IL.layout(S.hm96_row(), ['cv', 'pad', 'y'], ['bc']), 'hm96', None, None))\n",
            r = repo.display(), d = dir.display());
        let ok = std::process::Command::new("python3").arg("-c").arg(py).status().ok()?.success();
        let nets = ok.then(|| ["sha512x3", "hm96"].iter().map(|n| IrUnitNet::load(dir.join(format!("{n}.txt")).to_str().unwrap())).collect());
        let _ = std::fs::remove_dir_all(&dir);
        if nets.is_none() {
            eprintln!("skipped: python3 could not build the slot circuits from {}", repo.display());
        }
        nets
    }

    fn plans_match(carry_every: usize) {
        let Some(nets) = slot_circuits(carry_every) else { return };
        for net in &nets {
            let t = Instant::now();
            let plan = NativeSha::derive(net).unwrap_or_else(|e| panic!("{e}"));
            let derive_s = t.elapsed().as_secs_f64();
            // lane by lane (a device thread's view) against the host's eval64 + lanes, on one more random batch
            let lanes_in = random_lanes(plan.slot, plan.in_words, 77);
            let ins: Vec<Vec<F128>> = lanes_in.iter().map(|v| v.chunks(2).map(|p| F128::new(p[0], p[1])).collect()).collect();
            let bit = |r: &Vec<F128>, col: usize| (if col % WORD < 64 { r[col / WORD].lo >> (col % 64) } else { r[col / WORD].hi >> (col % 64) }) & 1;
            let t = Instant::now();
            let (zz, aa, bb) = net.eval64(|col| ins.iter().enumerate().fold(0u64, |x, (l, r)| x | (bit(r, col) << l)));
            let host = (lanes(&zz, 64, plan.uw), lanes(&aa, 64, plan.uw), lanes(&bb, 64, plan.uw));
            let host_s = t.elapsed().as_secs_f64();
            let t = Instant::now();
            let native: Vec<_> = ins.iter().map(|w| plan.slot_zab(w)).collect();
            let native_s = t.elapsed().as_secs_f64();
            for (l, (z, a, b)) in native.iter().enumerate() {
                assert!(*z == host.0[l] && *a == host.1[l] && *b == host.2[l], "{}: slot {l} differs from eval64 + lanes", net.name);
            }
            assert!(plan.slots_zab(&ins).iter().zip(&native).all(|(x, y)| x == y));
            eprintln!("{} (carries every {carry_every}): {} rows, plan derived and checked in {derive_s:.2} s, window {} words; \
                       64 slots: eval64 + lanes {:.1} ms, native slot by slot {:.1} ms", net.name, plan.rows.len(), plan.window(net),
                      host_s * 1e3, native_s * 1e3);
        }
    }

    #[test]
    fn native_witness_matches_eval64_on_the_pinned_slot_circuits() {
        plans_match(5);
    }

    /// The same emitter on the denser slot of `docs/gemm-hash-cost-plan.md` row 2 (carries committed every 16 bits).
    #[test]
    #[ignore = "builds two more slot circuits in Python (about a minute)"]
    fn native_witness_matches_eval64_with_carries_every_16() {
        plans_match(16);
    }
}
