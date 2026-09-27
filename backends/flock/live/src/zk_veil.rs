//! Zero knowledge for `verity/flock-circuit` (lane flock-zk, M1): VEIL's compiler (Dalal–Hemo–Rabinovich–Rothblum, ePrint
//! 2026/683, §4.1 and Remark 4.4) applied to Flock's algebraic phase, over GF(2^128).
//!
//! **Exposed messages.** Every value the zerocheck and the lincheck send (round-1 skip evaluations, the multilinear rounds, the
//! three final evaluations, the lincheck rounds and `z_partial`) is sent as `v' = v + h` for a pad `h` from the prover's
//! randomness ([`Pads`]), laid out by [`Layout`]. The pads are committed ([`InnerCommit`]: a Reed–Solomon codeword with
//! `K_PAD` uniform padding symbols per row, `hm96-sha512/v1` leaves, SHA-512 nodes) before the zerocheck's first coin.
//! Upstream's prover runs unchanged: [`Masking`] adds the pads as the messages are absorbed, and [`mask_proof`] puts the same
//! values in the proof.
//!
//! **The checks.** For fixed coins every zerocheck and lincheck check is affine in the exposed values except the zerocheck's
//! final product `a · b`. [`replay`] re-runs both verifiers on the masked proof with each quantity an affine form in the pads
//! ([`Lin`]), so each check becomes a linear constraint on `h`. The product becomes linear with a committed `h_ab = h_a h_b`,
//! and that product is checked by sacrificing a second committed triple `(x, y, w = x y)` (open `ρ = ε h_a + x`, `σ = h_b + y`,
//! check `ε h_ab + w + σ x + ρ y = ρ σ`). The ring switch's claim check on the two non-region claims (whose values are
//! masked) is a constraint too; its `s_hat_v` stays in the clear, hidden by the mask slot's uniform words (see
//! [`mask_rank_ok`]). All constraints are batched with verifier coins into one `⟨c, h⟩ = t`, proved by [`prove_inner`]:
//! the prover sends `τ = ⟨c, μ⟩` for a committed uniform `μ`, then `Y = h + β μ` (with its padding), and opens `Q` columns
//! of both codewords.
//!
//! **Simulation** ([`simulate_inner`], and the binary's `simulate`): every exposed value is uniform (the pads are), `Y` is
//! uniform, `τ = (⟨c, Y⟩ + t) / β`, and the opened columns are uniform subject to `C(Y)_j = C(h)_j + β C(μ)_j` (the padding
//! makes any `Q ≤ K_PAD` columns of one codeword uniform). The prover refuses the degenerate coins `β = 0` and `ε`-independent
//! cases, and a mask slot whose weights fail to span (`mask_rank_ok`); the simulator refuses the same coins.

use flock_core::{
    lincheck::{LincheckCircuit, QuirkyPoint, SkipPoint, build_quirky_eq_table_from_weights},
    pcs::ring_switch::{build_claim_weights_from_skip, claim_check},
    r1cs::BlockR1cs,
    zerocheck::{
        multilinear::{interpolate_at_z_combined, interpolate_at_z_on_lambda},
        univariate_skip_optimized::{medium_challenges_ghash, small_challenges_ghash},
        ZerocheckProof,
    },
    lincheck::LincheckProof,
};
use flock_field::{F128, F256};
use flock_transcript::challenger::Challenger;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha512};

pub const LABEL_ZC: &[u8] = b"flock-zerocheck-v0";
pub const LABEL_PCS: &[u8] = b"flock-pcs-open-batch-v0";
pub const LABEL_PADS: &[u8] = b"verity-flock-zk-pads-v0";
pub const LABEL_INNER: &[u8] = b"verity-flock-zk-inner-v0";
pub use flock_core::pcs::ligerito::zk::{LABEL_CLOSE as L0_CLOSE, LABEL_OPEN as L0_OPEN};
pub const K_SKIP: usize = 6;
pub const SKIP: usize = 1 << K_SKIP;
const N_INNER: usize = 7;
/// Uniform padding symbols per inner codeword row (≥ `Q`, so any `Q` opened columns of one row are uniform).
pub const K_PAD: usize = 192;
/// Opened columns of the inner codeword.
pub const Q: usize = 168;
/// Inner codeword length over the message length: rate at most 1/8.
pub const INNER_LOG_INV_RATE: usize = 3;
/// ChaCha20 streams of [`crate::zk_hooks::ProverRng`] (1-3 are M0's).
pub const STREAM_PADS: u64 = 4;
pub const STREAM_DOT_MASK: u64 = 5;

/// Ligerito level 0's padding per lane for `q0` queries: both reps' openings of the one root, plus 64 for the OOD value, the
/// lane-round messages and the extra lanes' sums (any that many positions of a lane are uniform).
pub fn level0_t_pad(q0: usize) -> usize {
    2 * q0 + 64
}

/// Level 0's query count under its padding. With `l` message words per lane the code is `RS[2l, l + t_pad]`, of rate
/// `ρ' = (l + t_pad) / 2l`, and Fast100's rule holds at every level: `Q log₂(1/(√ρ' + η)) ≥ 100`, `η = 1/50` (the Johnson
/// slack of the soundness lane's `Accounting/Bound.lean`). The least such `Q ≥ q_min`; `None` when the padding outgrows the
/// lane first.
pub fn level0_queries(l: usize, q_min: usize) -> Option<usize> {
    let bits = |q: usize| {
        let rate = (l + level0_t_pad(q)) as f64 / (2 * l) as f64;
        -(q as f64) * (rate.sqrt() + 0.02).log2()
    };
    (q_min..).take_while(|&q| level0_t_pad(q) <= l).find(|&q| bits(q) >= 100.0)
}

/// Where each pad sits, per rep. Exposed values first, in proof-field order: zerocheck `round1_ab[64]`, `round1_c[64]`, the
/// multilinear rounds `(G(1), G(∞))`, `final_a`, `final_b`, `final_c`; lincheck rounds `(q(1), q(∞))`, `z_partial[64]`;
/// Ligerito's level 0 (when masked): its OOD values, its `k0` lane-round messages `(u_0, u_2)` (F256: two coordinates each),
/// the extra lanes' sums `e_0, e_1`. Then the committed auxiliaries: `h_ab = h_a h_b`, the sacrificed triple `x, y, w = x y`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layout {
    pub n_mlv: usize,
    pub n_lc: usize,
    /// Ligerito level 0's OOD samples and lane rounds (both 0: Ligerito not masked)
    pub n_ood0: usize,
    pub k0: usize,
}

impl Layout {
    pub fn new(m: usize, k_log: usize, n_ood0: usize, k0: usize) -> Self {
        Layout { n_mlv: m - K_SKIP, n_lc: k_log - K_SKIP, n_ood0, k0 }
    }
    pub fn ligerito(&self) -> bool {
        self.k0 > 0
    }
    fn lg0(&self) -> usize {
        self.zp(SKIP)
    }
    /// Level 0's OOD value `i`.
    pub fn lg_ood(&self, i: usize) -> usize {
        self.lg0() + i
    }
    /// Lane round `j`'s message coordinate `t` (`u_0.c0, u_0.c1, u_2.c0, u_2.c1`).
    pub fn lg_msg(&self, j: usize, t: usize) -> usize {
        self.lg0() + self.n_ood0 + 4 * j + t
    }
    /// Extra lane `i`'s sum, coordinate `c`.
    pub fn lg_e(&self, i: usize, c: usize) -> usize {
        self.lg0() + self.n_ood0 + 4 * self.k0 + 2 * i + c
    }
    fn n_lig(&self) -> usize {
        if self.ligerito() { self.n_ood0 + 4 * self.k0 + 4 } else { 0 }
    }
    pub fn r1ab(&self, i: usize) -> usize {
        i
    }
    pub fn r1c(&self, i: usize) -> usize {
        SKIP + i
    }
    pub fn ml(&self, j: usize, t: usize) -> usize {
        2 * SKIP + 2 * j + t
    }
    pub fn fa(&self) -> usize {
        2 * SKIP + 2 * self.n_mlv
    }
    pub fn fb(&self) -> usize {
        self.fa() + 1
    }
    pub fn fc(&self) -> usize {
        self.fa() + 2
    }
    pub fn lc(&self, j: usize, t: usize) -> usize {
        self.fc() + 1 + 2 * j + t
    }
    pub fn zp(&self, i: usize) -> usize {
        self.lc(self.n_lc, 0) + i
    }
    pub fn hab(&self) -> usize {
        self.zp(SKIP) + self.n_lig()
    }
    pub fn tx(&self) -> usize {
        self.hab() + 1
    }
    pub fn ty(&self) -> usize {
        self.hab() + 2
    }
    pub fn tw(&self) -> usize {
        self.hab() + 3
    }
    /// Exposed values (pads that mask a sent value).
    pub fn exposed(&self) -> usize {
        self.hab()
    }
    /// The algebraic phase's exposed values (the first [`Self::piop_exposed`] indices).
    pub fn piop_exposed(&self) -> usize {
        self.zp(SKIP)
    }
    pub fn len(&self) -> usize {
        self.tw() + 1
    }
    /// The pads in the order the prover absorbs the values they mask (`final_c` is never absorbed): the algebraic phase's,
    /// then level 0's (absorbed between `zk::LABEL_OPEN` and `zk::LABEL_CLOSE`).
    pub fn absorbed(&self) -> Vec<usize> {
        let mut v: Vec<usize> = (0..SKIP).map(|i| self.r1ab(i)).collect();
        v.extend((0..SKIP).map(|i| self.r1c(i)));
        for j in 0..self.n_mlv {
            v.extend([self.ml(j, 0), self.ml(j, 1)]);
        }
        v.extend([self.fa(), self.fb()]);
        for j in 0..self.n_lc {
            v.extend([self.lc(j, 0), self.lc(j, 1)]);
        }
        v.extend((0..SKIP).map(|i| self.zp(i)));
        if self.ligerito() {
            v.extend((0..self.n_ood0).map(|i| self.lg_ood(i)));
            v.extend((0..self.k0).flat_map(|j| (0..4).map(move |t| (j, t))).map(|(j, t)| self.lg_msg(j, t)));
            v.extend([self.lg_e(0, 0), self.lg_e(0, 1), self.lg_e(1, 0), self.lg_e(1, 1)]);
        }
        v
    }
    /// How many of [`Self::absorbed`] belong to the algebraic phase.
    pub fn piop_absorbed(&self) -> usize {
        self.absorbed().len() - self.n_lig()
    }
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({"n_mlv": self.n_mlv, "n_lc": self.n_lc, "len": self.len(), "exposed": self.exposed(),
            "zerocheck": {"round1_ab": [self.r1ab(0), self.r1ab(SKIP - 1)], "round1_c": [self.r1c(0), self.r1c(SKIP - 1)],
                          "multilinear": [self.ml(0, 0), self.ml(self.n_mlv - 1, 1)], "final_a": self.fa(), "final_b": self.fb(), "final_c": self.fc()},
            "lincheck": {"rounds": [self.lc(0, 0), self.lc(self.n_lc - 1, 1)], "z_partial": [self.zp(0), self.zp(SKIP - 1)]},
            "ligerito_level0": if self.ligerito() { serde_json::json!({"ood": [self.lg_ood(0), self.n_ood0], "lane_messages": [self.lg_msg(0, 0), self.lg_msg(self.k0 - 1, 3)],
                                                   "e": [self.lg_e(0, 0), self.lg_e(1, 1)]}) } else { serde_json::Value::Null },
            "aux": {"h_ab": self.hab(), "x": self.tx(), "y": self.ty(), "w": self.tw()}})
    }
}

/// One rep's prover randomness for the algebraic phase: the pads `h`, the dot-product mask `μ`, and each row's RS padding.
pub struct Pads {
    pub h: Vec<F128>,
    pub mu: Vec<F128>,
    pub pad_h: Vec<F128>,
    pub pad_mu: Vec<F128>,
}

impl Pads {
    /// Rep `rep`'s pads from the prover's ChaCha20 streams (never from the transcript).
    pub fn draw(rng: &mut crate::zk_hooks::ProverRng, lay: &Layout, rep: usize) -> Self {
        let s = lay.len();
        let base = (rep as u64) << 32;
        let mut h = rng.f128s(STREAM_PADS, base, s);
        h[lay.hab()] = h[lay.fa()] * h[lay.fb()];
        h[lay.tw()] = h[lay.tx()] * h[lay.ty()];
        let mu = rng.f128s(STREAM_DOT_MASK, base, s);
        let pad_h = rng.f128s(STREAM_PADS, base + (1 << 24), K_PAD);
        let pad_mu = rng.f128s(STREAM_DOT_MASK, base + (1 << 24), K_PAD);
        Pads { h, mu, pad_h, pad_mu }
    }
}

// --- affine forms ------------------------------------------------------------------------------------------------------

/// `c + Σ a_i h_i` (characteristic 2: a masked value `v' = v + h_i` is the unmasked `v = v' + h_i`).
#[derive(Clone, Debug, PartialEq)]
pub struct Lin {
    pub c: F128,
    pub a: Vec<F128>,
}

impl Lin {
    pub fn konst(c: F128, s: usize) -> Self {
        Lin { c, a: vec![F128::ZERO; s] }
    }
    /// The unmasked value behind masked `v` under pad `i`.
    pub fn masked(v: F128, i: usize, s: usize) -> Self {
        let mut l = Self::konst(v, s);
        l.a[i] = F128::ONE;
        l
    }
    pub fn add(&self, o: &Lin) -> Lin {
        Lin { c: self.c + o.c, a: self.a.iter().zip(&o.a).map(|(x, y)| *x + *y).collect() }
    }
    pub fn scale(&self, k: F128) -> Lin {
        Lin { c: self.c * k, a: self.a.iter().map(|x| *x * k).collect() }
    }
    pub fn add_const(&self, k: F128) -> Lin {
        Lin { c: self.c + k, a: self.a.clone() }
    }
    pub fn eval(&self, h: &[F128]) -> F128 {
        self.a.iter().zip(h).fold(self.c, |acc, (x, y)| acc + *x * *y)
    }
    /// `Σ w_i l_i`.
    pub fn combo(ws: &[F128], ls: &[Lin]) -> Lin {
        let s = ls[0].a.len();
        let mut out = Self::konst(F128::ZERO, s);
        for (w, l) in ws.iter().zip(ls) {
            out.c += *w * l.c;
            for (o, x) in out.a.iter_mut().zip(&l.a) {
                *o += *w * *x;
            }
        }
        out
    }
}

// --- the prover's masking challenger ----------------------------------------------------------------------------------

/// Wraps the rep's challenger: from the zerocheck's label to the opening's, every absorbed field element gets its pad (in
/// [`Layout::absorbed`] order); the pads' root is absorbed right after the zerocheck label, before its first coin. Every coin
/// drawn in that phase is logged (the prover replays its own checks from them).
pub struct Masking<C: Challenger> {
    pub inner: C,
    order: Vec<usize>,
    n_piop: usize,
    lig: bool,
    h: Vec<F128>,
    at: usize,
    on: bool,
    done: bool,
    done_l0: bool,
    root: [u8; 64],
    pub coins: Vec<F128>,
    /// level 0's coins (between `zk::LABEL_OPEN` and `zk::LABEL_CLOSE`)
    pub coins_l0: Vec<F128>,
    l0: bool,
    /// Called at the opening's label with the phase's coins, before any opening message is absorbed: the prover's refusal
    /// point for degenerate coins (`Err` stops the proof there).
    pub at_open: Option<Box<dyn FnMut(&[F128]) -> Result<(), String> + Send>>,
}

impl<C: Challenger> Masking<C> {
    pub fn new(inner: C, lay: &Layout, pads: &Pads, root: [u8; 64]) -> Self {
        Masking { inner, order: lay.absorbed(), n_piop: lay.piop_absorbed(), lig: lay.ligerito(), h: pads.h.clone(), at: 0, on: false, done: false,
                  done_l0: false, root, coins: Vec::new(), coins_l0: Vec::new(), l0: false, at_open: None }
    }
    pub fn finish(self) -> (C, Vec<F128>, Vec<F128>) {
        assert!(self.done && (self.done_l0 || !self.lig) && self.at == self.order.len(),
                "masking: {} of {} values absorbed masked", self.at, self.order.len());
        (self.inner, self.coins, self.coins_l0)
    }
    fn pad(&mut self, v: F128) -> F128 {
        let i = *self.order.get(self.at).expect("masking: more absorbed values than the layout's");
        self.at += 1;
        v + self.h[i]
    }
}

impl<C: Challenger> Challenger for Masking<C> {
    fn observe_label(&mut self, label: &[u8]) {
        if label == LABEL_PCS && self.on {
            self.on = false;
            self.done = true;
            assert_eq!(self.at, self.n_piop, "masking: the opening began before every exposed value was absorbed");
            if let Some(f) = self.at_open.as_mut()
                && let Err(e) = f(&self.coins)
            {
                panic!("PROVER-STOPPED: {e}");
            }
        }
        if label == L0_CLOSE && self.l0 {
            self.l0 = false;
            self.on = false;
            self.done_l0 = true;
            assert_eq!(self.at, self.order.len(), "masking: level 0's exposed values are not the layout's");
        }
        self.inner.observe_label(label);
        if label == LABEL_ZC && !self.done {
            self.on = true;
            self.inner.observe_label(LABEL_PADS);
            self.inner.observe_bytes(&self.root);
        }
        if label == L0_OPEN && self.done && self.lig && !self.done_l0 {
            self.on = true;
            self.l0 = true;
        }
    }
    fn observe_f128(&mut self, v: F128) {
        let v = if self.on { self.pad(v) } else { v };
        self.inner.observe_f128(v);
    }
    fn observe_f128_slice(&mut self, vs: &[F128]) {
        let m: Vec<F128> = if self.on { vs.iter().map(|v| self.pad(*v)).collect() } else { vs.to_vec() };
        self.inner.observe_f128_slice(&m);
    }
    fn observe_bytes(&mut self, b: &[u8]) {
        self.inner.observe_bytes(b);
    }
    fn sample_f128(&mut self) -> F128 {
        let x = self.inner.sample_f128();
        if self.l0 {
            self.coins_l0.push(x);
        } else if self.on {
            self.coins.push(x);
        }
        x
    }
    fn sample_f128_vec(&mut self, n: usize) -> Vec<F128> {
        let x = self.inner.sample_f128_vec(n);
        if self.l0 {
            self.coins_l0.extend_from_slice(&x);
        } else if self.on {
            self.coins.extend_from_slice(&x);
        }
        x
    }
    /// Live coins need no grinding: the inner challenger absorbs a nonce as bytes (never masked).
    fn grind_pow(&mut self, bits: u32) -> u64 {
        self.inner.grind_pow(bits)
    }
    fn verify_pow(&mut self, nonce: u64, bits: u32) -> bool {
        self.inner.verify_pow(nonce, bits)
    }
    fn fork_from_seed(&self, seed: [F128; 2], label: &'static [u8]) -> Self {
        assert!(!self.on, "masking: a fork inside the masked phase");
        Masking { inner: self.inner.fork_from_seed(seed, label), order: Vec::new(), n_piop: 0, lig: false, h: Vec::new(), at: 0, on: false,
                  done: true, done_l0: true, root: self.root, coins: Vec::new(), coins_l0: Vec::new(), l0: false, at_open: None }
    }
}

/// Logs every coin drawn through it (the inner phase's coins, for the simulator's input).
pub struct Logging<'a, C: Challenger> {
    pub inner: &'a mut C,
    pub coins: Vec<F128>,
}

impl<C: Challenger> Challenger for Logging<'_, C> {
    fn observe_label(&mut self, l: &[u8]) {
        self.inner.observe_label(l)
    }
    fn observe_f128(&mut self, v: F128) {
        self.inner.observe_f128(v)
    }
    fn observe_f128_slice(&mut self, v: &[F128]) {
        self.inner.observe_f128_slice(v)
    }
    fn observe_bytes(&mut self, b: &[u8]) {
        self.inner.observe_bytes(b)
    }
    fn sample_f128(&mut self) -> F128 {
        let x = self.inner.sample_f128();
        self.coins.push(x);
        x
    }
    fn sample_f128_vec(&mut self, n: usize) -> Vec<F128> {
        let x = self.inner.sample_f128_vec(n);
        self.coins.extend_from_slice(&x);
        x
    }
    fn fork_from_seed(&self, _: [F128; 2], _: &'static [u8]) -> Self {
        unreachable!()
    }
}

/// A challenger that hands out logged coins and ignores what it absorbs: the prover's replay of its own checks.
pub struct Tape {
    coins: std::collections::VecDeque<F128>,
}

impl Tape {
    pub fn new(coins: &[F128]) -> Self {
        Tape { coins: coins.iter().copied().collect() }
    }
}

impl Challenger for Tape {
    fn observe_f128(&mut self, _: F128) {}
    fn sample_f128(&mut self) -> F128 {
        self.coins.pop_front().expect("tape: out of coins")
    }
    fn fork_from_seed(&self, _: [F128; 2], _: &'static [u8]) -> Self {
        unreachable!()
    }
}

/// The algebraic phase's exposed values in [`Layout`] order (a proof's masked zerocheck and lincheck fields).
pub fn exposed_values(zc: &ZerocheckProof, lc: &LincheckProof, lay: &Layout) -> Vec<F128> {
    let mut v = vec![F128::ZERO; lay.piop_exposed()];
    for i in 0..SKIP {
        v[lay.r1ab(i)] = zc.round1_ab[i];
        v[lay.r1c(i)] = zc.round1_c[i];
        v[lay.zp(i)] = lc.z_partial[i];
    }
    for (j, r) in zc.multilinear_rounds.iter().enumerate() {
        v[lay.ml(j, 0)] = r.0;
        v[lay.ml(j, 1)] = r.1;
    }
    v[lay.fa()] = zc.final_a_eval;
    v[lay.fb()] = zc.final_b_eval;
    v[lay.fc()] = zc.final_c_eval;
    for (j, r) in lc.rounds.iter().enumerate() {
        v[lay.lc(j, 0)] = r.0;
        v[lay.lc(j, 1)] = r.1;
    }
    v
}

/// A zerocheck and lincheck proof holding the exposed values `v` (the inverse of [`exposed_values`]).
pub fn from_exposed(v: &[F128], lay: &Layout) -> (ZerocheckProof, LincheckProof) {
    let zc = ZerocheckProof {
        round1_ab: (0..SKIP).map(|i| v[lay.r1ab(i)]).collect(),
        round1_c: (0..SKIP).map(|i| v[lay.r1c(i)]).collect(),
        multilinear_rounds: (0..lay.n_mlv).map(|j| (v[lay.ml(j, 0)], v[lay.ml(j, 1)])).collect(),
        final_a_eval: v[lay.fa()], final_b_eval: v[lay.fb()], final_c_eval: v[lay.fc()], grinding_nonces: vec![],
    };
    let lc = LincheckProof {
        rounds: (0..lay.n_lc).map(|j| (v[lay.lc(j, 0)], v[lay.lc(j, 1)])).collect(),
        z_partial: (0..SKIP).map(|i| v[lay.zp(i)]).collect(), matrix_evals: vec![], grinding_nonces: vec![], comb_partial: vec![],
    };
    (zc, lc)
}

/// The proof's zerocheck and lincheck with every exposed value masked (the values [`Masking`] sent).
pub fn mask_proof(zc: &ZerocheckProof, lc: &LincheckProof, lay: &Layout, h: &[F128]) -> (ZerocheckProof, LincheckProof) {
    let mut z = zc.clone();
    let mut l = lc.clone();
    assert!(z.grinding_nonces.is_empty() && l.grinding_nonces.is_empty() && l.matrix_evals.is_empty());
    for i in 0..SKIP {
        z.round1_ab[i] += h[lay.r1ab(i)];
        z.round1_c[i] += h[lay.r1c(i)];
        l.z_partial[i] += h[lay.zp(i)];
    }
    for (j, r) in z.multilinear_rounds.iter_mut().enumerate() {
        r.0 += h[lay.ml(j, 0)];
        r.1 += h[lay.ml(j, 1)];
    }
    z.final_a_eval += h[lay.fa()];
    z.final_b_eval += h[lay.fb()];
    z.final_c_eval += h[lay.fc()];
    for (j, r) in l.rounds.iter_mut().enumerate() {
        r.0 += h[lay.lc(j, 0)];
        r.1 += h[lay.lc(j, 1)];
    }
    (z, l)
}

/// Ligerito level 0's exposed fields masked in the proof (the values [`Masking`] sent between `L0_OPEN` and `L0_CLOSE`).
pub fn mask_ligerito(lp: &mut flock_core::pcs::ligerito::LigeritoProof, lay: &Layout, h: &[F128]) {
    if !lay.ligerito() {
        return;
    }
    for i in 0..lay.n_ood0 {
        lp.ood_values[i] += h[lay.lg_ood(i)];
    }
    for j in 0..lay.k0 {
        let m = &mut lp.sumcheck_transcript_f256[j];
        m.u_0 = m.u_0 + F256::new(h[lay.lg_msg(j, 0)], h[lay.lg_msg(j, 1)]);
        m.u_2 = m.u_2 + F256::new(h[lay.lg_msg(j, 2)], h[lay.lg_msg(j, 3)]);
    }
    let z = lp.zk_level0.as_mut().expect("a masked level 0 carries its hiding messages");
    for i in 0..2 {
        z.e[i] = z.e[i] + F256::new(h[lay.lg_e(i, 0)], h[lay.lg_e(i, 1)]);
    }
}

/// Level 0's exposed values as sent (a proof's masked fields), in [`Layout`] order from `lg_ood(0)`.
pub fn ligerito_exposed(lp: &flock_core::pcs::ligerito::LigeritoProof, lay: &Layout) -> Vec<F128> {
    let mut v: Vec<F128> = lp.ood_values[..lay.n_ood0].to_vec();
    for m in &lp.sumcheck_transcript_f256[..lay.k0] {
        v.extend(m.u_0.coordinates());
        v.extend(m.u_2.coordinates());
    }
    if let Some(z) = &lp.zk_level0 {
        v.extend(z.e.iter().flat_map(|e| e.coordinates()));
    }
    v
}

/// Overwrite level 0's exposed fields with `v` (`ligerito_exposed` order): the simulator's uniform values.
pub fn set_ligerito_exposed(lp: &mut flock_core::pcs::ligerito::LigeritoProof, lay: &Layout, v: &[F128]) {
    assert_eq!(v.len(), lay.n_ood0 + 4 * lay.k0 + 4);
    lp.ood_values[..lay.n_ood0].copy_from_slice(&v[..lay.n_ood0]);
    let at = |k: usize| v[lay.n_ood0 + k];
    for j in 0..lay.k0 {
        lp.sumcheck_transcript_f256[j].u_0 = F256::new(at(4 * j), at(4 * j + 1));
        lp.sumcheck_transcript_f256[j].u_2 = F256::new(at(4 * j + 2), at(4 * j + 3));
    }
    let z = lp.zk_level0.as_mut().expect("level 0's hiding messages");
    for i in 0..2 {
        z.e[i] = F256::new(at(4 * lay.k0 + 2 * i), at(4 * lay.k0 + 2 * i + 1));
    }
}

/// An affine form over F256: two coordinate forms (`x = x_0 + x_1 U`).
pub type Lin2 = [Lin; 2];

fn l2_masked(v: F256, i: [usize; 2], s: usize) -> Lin2 {
    [Lin::masked(v.c0, i[0], s), Lin::masked(v.c1, i[1], s)]
}

fn l2_konst(v: F256, s: usize) -> Lin2 {
    [Lin::konst(v.c0, s), Lin::konst(v.c1, s)]
}

fn l2_add(a: &Lin2, b: &Lin2) -> Lin2 {
    [a[0].add(&b[0]), a[1].add(&b[1])]
}

/// `k · x` for a public F256 `k`: `x_0 k + x_1 (k U)`, coordinate by coordinate.
fn l2_scale(x: &Lin2, k: F256) -> Lin2 {
    let (m0, m1) = (k, k * F256::new(F128::ZERO, F128::ONE));
    [x[0].scale(m0.c0).add(&x[1].scale(m1.c0)), x[0].scale(m0.c1).add(&x[1].scale(m1.c1))]
}

/// Level 0's lane phase as constraints (upstream's `recursive_verifier_with_basis_succinct` up to the blinded claim): the
/// claim `target + Σ β_i y_i` through the `k0` lane rounds' quadratics `u_0 + r (claim + u_2) + r² u_2`, then
/// `claim + ρ_0 e_0 + ρ_1 e_1 = T'`. `sent`: the proof's masked level-0 fields (`ligerito_exposed` order).
pub fn ligerito_constraints(lay: &Layout, rec: &flock_core::pcs::ligerito::zk::Level0ZkRecord, sent: &[F128], t_prime: F256)
    -> Result<Vec<(&'static str, Lin)>, String> {
    let s = lay.len();
    if sent.len() != lay.n_ood0 + 4 * lay.k0 + 4 || rec.ood.len() != lay.n_ood0 || rec.challenges.len() != lay.k0 {
        return Err("zk: level 0's record is not the layout's".into());
    }
    let mut claim0 = Lin::konst(rec.target, s);
    for (i, &(beta, _)) in rec.ood.iter().enumerate() {
        claim0 = claim0.add(&Lin::masked(sent[i], lay.lg_ood(i), s).scale(beta));
    }
    let mut claim: Lin2 = [claim0, Lin::konst(F128::ZERO, s)];
    let at = |k: usize| sent[lay.n_ood0 + k];
    for (j, &r) in rec.challenges.iter().enumerate() {
        let u0 = l2_masked(F256::new(at(4 * j), at(4 * j + 1)), [lay.lg_msg(j, 0), lay.lg_msg(j, 1)], s);
        let u2 = l2_masked(F256::new(at(4 * j + 2), at(4 * j + 3)), [lay.lg_msg(j, 2), lay.lg_msg(j, 3)], s);
        claim = l2_add(&l2_add(&u0, &l2_scale(&l2_add(&claim, &u2), r)), &l2_scale(&u2, r * r));
    }
    let e = |i: usize| l2_masked(F256::new(at(4 * lay.k0 + 2 * i), at(4 * lay.k0 + 2 * i + 1)), [lay.lg_e(i, 0), lay.lg_e(i, 1)], s);
    let lhs = l2_add(&l2_add(&claim, &l2_scale(&e(0), rec.rho[0])), &l2_add(&l2_scale(&e(1), rec.rho[1]), &l2_konst(t_prime, s)));
    let [c0, c1] = lhs;
    Ok(vec![("ligerito lane phase (c0)", c0), ("ligerito lane phase (c1)", c1)])
}

// --- the verifier's replay ---------------------------------------------------------------------------------------------

/// What the masked algebraic phase hands the opening: the two z-claims with their values as affine forms, and the
/// constraints so far (each `Lin = 0`).
pub struct Replayed {
    pub ab_point: QuirkyPoint,
    pub c_point: QuirkyPoint,
    pub ab_value: Lin,
    pub c_value: Lin,
    pub cons: Vec<(&'static str, Lin)>,
    /// the product's two unmasked operands (for the triple)
    pub a_value: Lin,
    pub b_value: Lin,
}

/// Re-run upstream's zerocheck and lincheck verifiers (`zerocheck::verify_with_grinding`, `lincheck::verify_with_grinding`,
/// `verifier::verify_core_inner`; Fast100: grinding disabled) on the masked proof, absorbing exactly what they absorb and
/// drawing exactly their coins, with every message-dependent quantity an affine form in the pads.
pub fn replay<Ch: Challenger>(r1cs: &BlockR1cs, zc: &ZerocheckProof, lc: &LincheckProof, circuit: &dyn LincheckCircuit, lay: &Layout,
                              root: &[u8; 64], ch: &mut Ch) -> Result<Replayed, String> {
    let s = lay.len();
    let m = r1cs.m;
    if !r1cs.c0_is_identity() || r1cs.k_skip != K_SKIP || m < K_SKIP + N_INNER {
        return Err("zk: the statement is not a C = I, k_skip = 6 block R1CS".into());
    }
    if zc.round1_ab.len() != SKIP || zc.round1_c.len() != SKIP || zc.multilinear_rounds.len() != lay.n_mlv || !zc.grinding_nonces.is_empty()
        || lc.rounds.len() != lay.n_lc || lc.z_partial.len() != SKIP || !lc.grinding_nonces.is_empty() || !lc.matrix_evals.is_empty()
    {
        return Err("zk: the masked zerocheck or lincheck has the wrong shape".into());
    }
    let mut cons: Vec<(&'static str, Lin)> = Vec::new();
    // ---- zerocheck
    ch.observe_label(LABEL_ZC);
    ch.observe_label(LABEL_PADS);
    ch.observe_bytes(root);
    let r_skip = ch.sample_f128_vec(K_SKIP);
    let r_outer = ch.sample_f128_vec(m - K_SKIP - N_INNER);
    let mut r = vec![F128::ZERO; m];
    r[..K_SKIP].copy_from_slice(&r_skip);
    r[K_SKIP..K_SKIP + 3].copy_from_slice(&small_challenges_ghash());
    r[K_SKIP + 3..K_SKIP + N_INNER].copy_from_slice(&medium_challenges_ghash());
    r[K_SKIP + N_INNER..].copy_from_slice(&r_outer);
    ch.observe_f128_slice(&zc.round1_ab);
    ch.observe_f128_slice(&zc.round1_c);
    let z = ch.sample_f128();
    let unit = |i: usize| -> Vec<F128> { (0..SKIP).map(|j| if j == i { F128::ONE } else { F128::ZERO }).collect() };
    let w_c: Vec<F128> = (0..SKIP).map(|i| interpolate_at_z_on_lambda(&unit(i), K_SKIP, z)).collect();
    let w_comb: Vec<F128> = (0..SKIP).map(|i| interpolate_at_z_combined(&unit(i), K_SKIP, z)).collect();
    let r1c: Vec<Lin> = (0..SKIP).map(|i| Lin::masked(zc.round1_c[i], lay.r1c(i), s)).collect();
    let r1ab: Vec<Lin> = (0..SKIP).map(|i| Lin::masked(zc.round1_ab[i], lay.r1ab(i), s)).collect();
    let computed_c = Lin::combo(&w_c, &r1c);
    let fc = Lin::masked(zc.final_c_eval, lay.fc(), s);
    cons.push(("zerocheck c_eval", computed_c.add(&fc)));
    let comb: Vec<Lin> = r1ab.iter().zip(&r1c).map(|(a, c)| a.add(c)).collect();
    let mut running = Lin::combo(&w_comb, &comb).add(&computed_c);
    let mut rhos = Vec::with_capacity(lay.n_mlv);
    for (j, &(m1, minf)) in zc.multilinear_rounds.iter().enumerate() {
        let r_eq = r[K_SKIP + j];
        let g1 = Lin::masked(m1, lay.ml(j, 0), s);
        let ginf = Lin::masked(minf, lay.ml(j, 1), s);
        let g0 = running.add(&g1.scale(r_eq)).scale((F128::ONE + r_eq).inv());
        ch.observe_f128(m1);
        ch.observe_f128(minf);
        let rho = ch.sample_f128();
        rhos.push(rho);
        let opr = F128::ONE + rho;
        running = g0.scale(opr).add(&g1.scale(rho)).add(&ginf.scale(rho * opr));
    }
    let a = Lin::masked(zc.final_a_eval, lay.fa(), s);
    let b = Lin::masked(zc.final_b_eval, lay.fb(), s);
    // a · b = a'b' + a' h_b + b' h_a + h_ab
    let mut prod = Lin::konst(zc.final_a_eval * zc.final_b_eval, s);
    prod.a[lay.fb()] = zc.final_a_eval;
    prod.a[lay.fa()] = zc.final_b_eval;
    prod.a[lay.hab()] = F128::ONE;
    cons.push(("zerocheck final a·b", running.add(&prod)));
    ch.observe_f128(zc.final_a_eval);
    ch.observe_f128(zc.final_b_eval);
    let r_rest: Vec<F128> = r[K_SKIP..].to_vec();
    let x_ab = r1cs.x_ab_from_multilinear_values(SkipPoint::Phi8(z), &rhos);
    // ---- lincheck
    ch.observe_label(b"flock-lincheck-v0");
    let alpha = ch.sample_f128();
    // the prover's own replay takes its lincheck's folded comb (`comb_partial`, never serialized); a verifier folds its own
    let own = lc.comb_partial.len() == SKIP;
    let mut comb_vec = if own {
        Vec::new()
    } else {
        let eq_inner = build_quirky_eq_table_from_weights(&x_ab.z_skip.weights(K_SKIP), &x_ab.x_inner_rest);
        circuit.fold_alpha_batched(alpha, &eq_inner)
    };
    let mut running = a.scale(alpha).add(&b);
    if let Some(col) = circuit.const_pin_col() {
        let beta = ch.sample_f128();
        if !own {
            comb_vec[col] += beta;
        }
        running = running.add_const(beta);
    }
    let mut r_rounds = Vec::with_capacity(lay.n_lc);
    for (j, &(e1v, einfv)) in lc.rounds.iter().enumerate() {
        ch.observe_f128(e1v);
        ch.observe_f128(einfv);
        let rr = ch.sample_f128();
        let e1 = Lin::masked(e1v, lay.lc(j, 0), s);
        let einf = Lin::masked(einfv, lay.lc(j, 1), s);
        let e0 = running.add(&e1);
        let c1 = e0.add(&e1).add(&einf);
        running = einf.scale(rr * rr).add(&c1.scale(rr)).add(&e0);
        if !own {
            let half = comb_vec.len() / 2;
            for i in 0..half {
                comb_vec[i] = comb_vec[i] + rr * (comb_vec[i + half] + comb_vec[i]);
            }
            comb_vec.truncate(half);
        }
        r_rounds.push(rr);
    }
    if own {
        comb_vec = lc.comb_partial.clone();
    }
    ch.observe_f128_slice(&lc.z_partial);
    let zp: Vec<Lin> = (0..SKIP).map(|i| Lin::masked(lc.z_partial[i], lay.zp(i), s)).collect();
    cons.push(("lincheck final", running.add(&Lin::combo(&comb_vec, &zp))));
    let r_inner_skip = x_ab.z_skip.sample_fresh(ch);
    let lambda = r_inner_skip.weights(K_SKIP);
    let w = Lin::combo(&lambda, &zp);
    let mut r_inner_rest = r_rounds;
    r_inner_rest.reverse();
    let ab_point = r1cs.ab_claim_point(r_inner_skip, &r_inner_rest, &x_ab.x_outer);
    let c_point = r1cs.c_claim_point(SkipPoint::Phi8(z), &r_rest);
    Ok(Replayed { ab_point, c_point, ab_value: w, c_value: fc, cons, a_value: a, b_value: b })
}

/// The two z-claim points from the algebraic phase's coins alone (in [`replay`]'s draw order; no message is needed).
pub fn points_from_coins(r1cs: &BlockR1cs, lay: &Layout, has_pin: bool, coins: &[F128]) -> Result<(QuirkyPoint, QuirkyPoint), String> {
    let m = r1cs.m;
    let need = K_SKIP + (m - K_SKIP - N_INNER) + 1 + lay.n_mlv + 1 + usize::from(has_pin) + lay.n_lc + 1;
    if coins.len() != need {
        return Err(format!("zk: {} coins in the algebraic phase, not {need}", coins.len()));
    }
    let mut at = 0;
    let mut take = |n: usize| { let v = coins[at..at + n].to_vec(); at += n; v };
    let r_skip = take(K_SKIP);
    let r_outer = take(m - K_SKIP - N_INNER);
    let z = take(1)[0];
    let rhos = take(lay.n_mlv);
    let _alpha = take(1);
    if has_pin {
        take(1);
    }
    let mut r_inner_rest = take(lay.n_lc);
    let r_inner_skip = SkipPoint::Phi8(take(1)[0]);
    r_inner_rest.reverse();
    let mut r_rest = vec![F128::ZERO; m - K_SKIP];
    r_rest[..3].copy_from_slice(&small_challenges_ghash());
    r_rest[3..N_INNER].copy_from_slice(&medium_challenges_ghash());
    r_rest[N_INNER..].copy_from_slice(&r_outer);
    let _ = r_skip;
    let x_ab = r1cs.x_ab_from_multilinear_values(SkipPoint::Phi8(z), &rhos);
    Ok((r1cs.ab_claim_point(r_inner_skip, &r_inner_rest, &x_ab.x_outer), r1cs.c_claim_point(SkipPoint::Phi8(z), &r_rest)))
}

/// The ring switch's claim check on a claim whose value is masked, as a constraint: `claim_check(weights, s_hat_v) = value`.
pub fn ring_switch_constraint(p: &QuirkyPoint, s_hat_v: &[F128], value: &Lin) -> (F128, Lin) {
    let (skip, x) = point_parts(p);
    let cc = claim_check(&build_claim_weights_from_skip(&skip, x[0]), s_hat_v);
    (cc, value.add_const(cc))
}

/// A z-claim point as the opening takes it: the 64 skip weights (`LOG_PACKING - 1` skip bits) and the rest of the point.
pub fn point_parts(p: &QuirkyPoint) -> (Vec<F128>, Vec<F128>) {
    (p.z_skip.weights(K_SKIP), [p.x_inner_rest.as_slice(), p.x_outer.as_slice()].concat())
}

/// The triple's constraints for the sacrifice coin `ε` and the opened `ρ, σ`.
pub fn triple_constraints(lay: &Layout, eps: F128, rho: F128, sigma: F128) -> Vec<(&'static str, Lin)> {
    let s = lay.len();
    let mut t1 = Lin::konst(rho, s);
    t1.a[lay.fa()] = eps;
    t1.a[lay.tx()] = F128::ONE;
    let mut t2 = Lin::konst(sigma, s);
    t2.a[lay.fb()] = F128::ONE;
    t2.a[lay.ty()] = F128::ONE;
    let mut t3 = Lin::konst(rho * sigma, s);
    t3.a[lay.hab()] = eps;
    t3.a[lay.tw()] = F128::ONE;
    t3.a[lay.tx()] = sigma;
    t3.a[lay.ty()] = rho;
    vec![("triple ρ", t1), ("triple σ", t2), ("triple sacrifice", t3)]
}

// --- the inner commitment and the dot-product proof --------------------------------------------------------------------

fn pow2_at_least(x: usize) -> usize {
    x.next_power_of_two()
}

/// The inner code: evaluations of the message polynomial `Σ m_i X^i` at the nonzero points `1, 2, ..., N` (as GF(2^128)
/// elements of the polynomial basis), `N = 2^INNER_LOG_INV_RATE · next_pow2(len)`.
pub struct Code {
    pub msg: usize,
    pub n: usize,
    pts: Vec<F128>,
}

impl Code {
    pub fn new(msg: usize) -> Self {
        let n = pow2_at_least(msg) << INNER_LOG_INV_RATE;
        Code { msg, n, pts: (1..=n as u64).map(|j| F128::new(j, 0)).collect() }
    }
    pub fn at(&self, m: &[F128], j: usize) -> F128 {
        let x = self.pts[j];
        m.iter().rev().fold(F128::ZERO, |acc, c| acc * x + *c)
    }
    pub fn encode(&self, m: &[F128]) -> Vec<F128> {
        use rayon::prelude::*;
        assert_eq!(m.len(), self.msg);
        (0..self.n).into_par_iter().map(|j| self.at(m, j)).collect()
    }
}

fn f128_bytes(v: &[F128]) -> Vec<u8> {
    v.iter().flat_map(|x| x.lo.to_le_bytes().into_iter().chain(x.hi.to_le_bytes())).collect()
}

type Hash = flock_hash::Digest;

fn node(l: &Hash, r: &Hash) -> Hash {
    let mut h = Sha512::new();
    h.update(l.as_flattened());
    h.update(r.as_flattened());
    flock_hash::digest_from(&h.finalize())
}

fn leaf_digest(v: &[F128]) -> Hash {
    flock_hash::digest_from(&Sha512::digest(f128_bytes(v)))
}

/// The pads' commitment: the two rows `(h || pad_h)` and `(μ || pad_μ)` encoded, leaf `j` = column `j` (32 bytes) under an
/// `hm96-sha512/v1` salt (a fresh tree id of the proof's salt key), SHA-512 nodes.
pub struct InnerCommit {
    pub code: Code,
    pub ch: Vec<F128>,
    pub cm: Vec<F128>,
    pub salts: Vec<u8>,
    pub tree: Vec<Hash>,
}

impl InnerCommit {
    pub fn new(p: &Pads, salts: Vec<u8>) -> Self {
        let code = Code::new(p.h.len() + K_PAD);
        let ch = code.encode(&[p.h.as_slice(), p.pad_h.as_slice()].concat());
        let cm = code.encode(&[p.mu.as_slice(), p.pad_mu.as_slice()].concat());
        Self::from_columns(code, ch, cm, salts)
    }

    fn from_columns(code: Code, ch: Vec<F128>, cm: Vec<F128>, salts: Vec<u8>) -> Self {
        assert_eq!(salts.len(), code.n * crate::zk_hooks::SALT_BYTES);
        let mut tree = vec![Hash::default(); 2 * code.n - 1];
        for j in 0..code.n {
            tree[j] = flock_merkle::hm96::leaf(&leaf_digest(&[ch[j], cm[j]]), &salts[j * crate::zk_hooks::SALT_BYTES..(j + 1) * crate::zk_hooks::SALT_BYTES]);
        }
        let (mut rd, mut len, mut wr) = (0, code.n, code.n);
        while len > 1 {
            for i in 0..len / 2 {
                tree[wr + i] = node(&tree[rd + 2 * i], &tree[rd + 2 * i + 1]);
            }
            rd += len;
            wr += len / 2;
            len /= 2;
        }
        InnerCommit { code, ch, cm, salts, tree }
    }
    pub fn root(&self) -> [u8; 64] {
        self.tree.last().unwrap().as_flattened().try_into().unwrap()
    }
    fn path(&self, mut j: usize) -> Vec<Hash> {
        let (mut rd, mut len) = (0, self.code.n);
        let mut out = Vec::new();
        while len > 1 {
            out.push(self.tree[rd + (j ^ 1)]);
            rd += len;
            len /= 2;
            j /= 2;
        }
        out
    }
}

/// The inner proof: the triple's openings, `τ`, `Y`, and the opened columns with their salts and paths.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct InnerProof {
    pub root: Vec<u8>,
    pub rho: F128,
    pub sigma: F128,
    pub tau: F128,
    pub y: Vec<F128>,
    pub opened: Vec<(F128, F128)>,
    pub salts: Vec<flock_merkle::hm96::Salt>,
    pub paths: Vec<Vec<Hash>>,
}

/// Batch the constraints with the verifier's coins: `(c, t)` with `⟨c, h⟩ = t`.
pub fn batch(cons: &[(&'static str, Lin)], gammas: &[F128]) -> (Vec<F128>, F128) {
    let l = Lin::combo(gammas, &cons.iter().map(|c| c.1.clone()).collect::<Vec<_>>());
    (l.a, l.c)
}

fn queries<Ch: Challenger>(ch: &mut Ch, n: usize) -> Vec<usize> {
    ch.sample_f128_vec(Q).iter().map(|x| (x.lo as usize) & (n - 1)).collect()
}

/// The prover's inner proof, on the rep's challenger after the opening. `cons` are the algebraic phase's and the ring
/// switch's constraints (the triple's are added here). Refuses `β = 0` (the simulator refuses the same coins).
pub fn prove_inner<Ch: Challenger>(ic: &InnerCommit, p: &Pads, lay: &Layout, mut cons: Vec<(&'static str, Lin)>, ch: &mut Ch)
    -> Result<InnerProof, String> {
    ch.observe_label(LABEL_INNER);
    let eps = ch.sample_f128();
    let rho = eps * p.h[lay.fa()] + p.h[lay.tx()];
    let sigma = p.h[lay.fb()] + p.h[lay.ty()];
    ch.observe_f128_slice(&[rho, sigma]);
    cons.extend(triple_constraints(lay, eps, rho, sigma));
    let gammas = ch.sample_f128_vec(cons.len());
    let (c, _t) = batch(&cons, &gammas);
    let tau = c.iter().zip(&p.mu).fold(F128::ZERO, |a, (x, y)| a + *x * *y);
    ch.observe_f128(tau);
    let beta = ch.sample_f128();
    if beta == F128::ZERO {
        return Err("PROVER-STOPPED: degenerate coin β = 0".into());
    }
    let y: Vec<F128> = p.h.iter().zip(&p.mu).map(|(a, b)| *a + beta * *b)
        .chain(p.pad_h.iter().zip(&p.pad_mu).map(|(a, b)| *a + beta * *b)).collect();
    ch.observe_f128_slice(&y);
    let qs = queries(ch, ic.code.n);
    Ok(InnerProof {
        root: ic.root().to_vec(), rho, sigma, tau, y,
        opened: qs.iter().map(|&j| (ic.ch[j], ic.cm[j])).collect(),
        salts: qs.iter().map(|&j| flock_merkle::hm96::salt_of(&ic.salts[j * crate::zk_hooks::SALT_BYTES..])).collect(),
        paths: qs.iter().map(|&j| ic.path(j)).collect(),
    })
}

/// The verifier's side of [`prove_inner`].
pub fn verify_inner<Ch: Challenger>(pr: &InnerProof, lay: &Layout, mut cons: Vec<(&'static str, Lin)>, ch: &mut Ch) -> Result<(), String> {
    let code = Code::new(lay.len() + K_PAD);
    if pr.y.len() != code.msg || pr.opened.len() != Q || pr.salts.len() != Q || pr.paths.len() != Q || pr.root.len() != 64 {
        return Err("zk inner: malformed proof".into());
    }
    ch.observe_label(LABEL_INNER);
    let eps = ch.sample_f128();
    ch.observe_f128_slice(&[pr.rho, pr.sigma]);
    cons.extend(triple_constraints(lay, eps, pr.rho, pr.sigma));
    let gammas = ch.sample_f128_vec(cons.len());
    let (c, t) = batch(&cons, &gammas);
    ch.observe_f128(pr.tau);
    let beta = ch.sample_f128();
    if beta == F128::ZERO {
        return Err("zk inner: degenerate coin β = 0".into());
    }
    ch.observe_f128_slice(&pr.y);
    let qs = queries(ch, code.n);
    let cy = c.iter().zip(&pr.y).fold(F128::ZERO, |a, (x, y)| a + *x * *y);
    if cy != t + beta * pr.tau {
        return Err("zk inner: the batched constraint fails (⟨c, Y⟩ ≠ t + β τ)".into());
    }
    let depth = code.n.trailing_zeros() as usize;
    for (k, &j) in qs.iter().enumerate() {
        let (a, b) = pr.opened[k];
        if code.at(&pr.y, j) != a + beta * b {
            return Err(format!("zk inner: column {j} is not consistent with Y"));
        }
        if pr.paths[k].len() != depth {
            return Err("zk inner: a path of the wrong depth".into());
        }
        let mut hsh = flock_merkle::hm96::leaf(&leaf_digest(&[a, b]), pr.salts[k].as_flattened());
        let mut idx = j;
        for sib in &pr.paths[k] {
            hsh = if idx & 1 == 0 { node(&hsh, sib) } else { node(sib, &hsh) };
            idx /= 2;
        }
        if hsh.as_flattened() != pr.root.as_slice() {
            return Err(format!("zk inner: column {j}'s path does not reach the pads' root"));
        }
    }
    Ok(())
}

// --- the mask slot's rank condition ------------------------------------------------------------------------------------

/// `eq(p, x)` over the bits of word index `x` (LSB first).
pub fn eq_word(p: &[F128], x: usize) -> F128 {
    p.iter().enumerate().fold(F128::ONE, |acc, (i, &pi)| acc * if (x >> i) & 1 == 1 { pi } else { F128::ONE + pi })
}

/// The mask slot hides the non-region claims' `s_hat_v`: for claim points `p_1..p_k` (their word coordinates, `x_outer[1..]`),
/// `s_hat_v^{p}[v]` gets `Σ_x eq(p, x) u_{v,x}` over the mask words `x`, with `u` uniform bits. That is uniform on
/// `GF(2^128)^k`, independently for each `v`, exactly when the vectors `(eq(p_1, x), ..., eq(p_k, x))` span `GF(2)^{128k}`.
/// Returns the GF(2)-rank and whether it is full.
pub fn mask_rank_ok(points: &[Vec<F128>], mask_words: &[usize]) -> (usize, bool) {
    let k = points.len();
    let limbs = 2 * k;
    let mut rows: Vec<Vec<u64>> = mask_words.iter().map(|&x| {
        points.iter().flat_map(|p| {
            let e = eq_word(p, x);
            [e.lo, e.hi]
        }).collect()
    }).collect();
    let mut rank = 0;
    for col in 0..64 * limbs {
        let (w, b) = (col / 64, col % 64);
        let Some(piv) = (rank..rows.len()).find(|&r| (rows[r][w] >> b) & 1 == 1) else { continue };
        rows.swap(rank, piv);
        let pr = rows[rank].clone();
        for (r, row) in rows.iter_mut().enumerate() {
            if r != rank && (row[w] >> b) & 1 == 1 {
                for (x, y) in row.iter_mut().zip(&pr) {
                    *x ^= y;
                }
            }
        }
        rank += 1;
    }
    (rank, rank == 64 * limbs)
}

/// The constraints the honest pads fail (none for an honest prover; a dishonest prover's proof carries on to the verifier).
pub fn failing(cons: &[(&'static str, Lin)], h: &[F128]) -> Vec<&'static str> {
    cons.iter().filter(|(_, l)| l.eval(h) != F128::ZERO).map(|c| c.0).collect()
}

/// The simulator's inner proof, drawing the verifier's coins from `coins` in the verifier's order (`ε`, the batching coins,
/// `β`, the queries), for the public constraints `cons` (the triple's added here). `ρ, σ, Y` are uniform, `τ` is solved, and
/// the opened columns are uniform subject to the column check. The tree holds the simulated columns at the opened positions
/// and all-zero columns elsewhere (hm96 hides every leaf).
pub fn simulate_inner<Ch: Challenger>(lay: &Layout, rng: &mut crate::zk_hooks::ProverRng, index: u64, mut cons: Vec<(&'static str, Lin)>,
                                      salts: Vec<u8>, coins: &mut Ch) -> Result<InnerProof, String> {
    let code = Code::new(lay.len() + K_PAD);
    let base = (index + 1) << 44;
    let eps = coins.sample_f128();
    let two = rng.f128s(STREAM_PADS, base, 2);
    let (rho, sigma) = (two[0], two[1]);
    cons.extend(triple_constraints(lay, eps, rho, sigma));
    let gammas = coins.sample_f128_vec(cons.len());
    let (c, t) = batch(&cons, &gammas);
    let beta = coins.sample_f128();
    if beta == F128::ZERO {
        return Err("PROVER-STOPPED: degenerate coin β = 0".into());
    }
    let y = rng.f128s(STREAM_PADS, base + (1 << 20), code.msg);
    let cy = c.iter().zip(&y).fold(F128::ZERO, |a, (x, yy)| a + *x * *yy);
    let tau = (cy + t) * beta.inv();
    let qs = queries(coins, code.n);
    let hs = rng.f128s(STREAM_PADS, base + (1 << 21), qs.len());
    let (mut ch, mut cm) = (vec![F128::ZERO; code.n], vec![F128::ZERO; code.n]);
    for (&j, a) in qs.iter().zip(hs) {
        ch[j] = a;
        cm[j] = (code.at(&y, j) + a) * beta.inv();
    }
    let ic = InnerCommit::from_columns(code, ch, cm, salts);
    Ok(InnerProof {
        root: ic.root().to_vec(), rho, sigma, tau, y,
        opened: qs.iter().map(|&j| (ic.ch[j], ic.cm[j])).collect(),
        salts: qs.iter().map(|&j| flock_merkle::hm96::salt_of(&ic.salts[j * crate::zk_hooks::SALT_BYTES..])).collect(),
        paths: qs.iter().map(|&j| ic.path(j)).collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zk_hooks::{ProverRng, ProverSeed};

    #[test]
    fn layout_is_a_bijection_on_the_exposed_values() {
        let lay = Layout::new(25, 22, 1, 6);
        let mut seen = vec![false; lay.len()];
        for i in lay.absorbed() {
            assert!(!std::mem::replace(&mut seen[i], true));
        }
        assert!(!seen[lay.fc()], "final_c is never absorbed");
        seen[lay.fc()] = true;
        assert!(seen[..lay.exposed()].iter().all(|x| *x));
        assert_eq!(lay.len(), lay.exposed() + 4);
    }

    #[test]
    fn triple_sacrifice_holds_and_catches_a_bad_product() {
        let lay = Layout::new(20, 16, 0, 0);
        let mut rng = ProverRng::from_seed(ProverSeed::injected([3; 32]));
        let p = Pads::draw(&mut rng, &lay, 0);
        let eps = F128::new(0x1234, 0x5678);
        let rho = eps * p.h[lay.fa()] + p.h[lay.tx()];
        let sigma = p.h[lay.fb()] + p.h[lay.ty()];
        for (nm, c) in triple_constraints(&lay, eps, rho, sigma) {
            assert_eq!(c.eval(&p.h), F128::ZERO, "{nm}");
        }
        let mut bad = p.h.clone();
        bad[lay.hab()] += F128::ONE;
        assert_ne!(triple_constraints(&lay, eps, rho, sigma)[2].1.eval(&bad), F128::ZERO);
    }

    #[test]
    fn inner_proof_round_trips_and_rejects_a_false_constraint() {
        let lay = Layout::new(20, 16, 0, 0);
        let mut rng = ProverRng::from_seed(ProverSeed::injected([5; 32]));
        let p = Pads::draw(&mut rng, &lay, 0);
        let code = Code::new(lay.len() + K_PAD);
        let salts = crate::zk_hooks::LeafSalts::for_seed(&ProverSeed::injected([6; 32])).expand(9, code.n);
        let ic = InnerCommit::new(&p, salts);
        // one true constraint: h_0 + h_1 = (h_0 + h_1)
        let s = lay.len();
        let mut l = Lin::konst(p.h[0] + p.h[1], s);
        l.a[0] = F128::ONE;
        l.a[1] = F128::ONE;
        let coins: Vec<F128> = (1..2000u64).map(|i| F128::new(i * 7919, i)).collect();
        let pr = prove_inner(&ic, &p, &lay, vec![("t", l.clone())], &mut Tape::new(&coins)).unwrap();
        verify_inner(&pr, &lay, vec![("t", l.clone())], &mut Tape::new(&coins)).unwrap();
        let mut bad = l.clone();
        bad.c += F128::ONE;
        assert!(verify_inner(&pr, &lay, vec![("t", bad)], &mut Tape::new(&coins)).is_err());
        let mut pr2 = pr.clone();
        pr2.opened[3].0 += F128::ONE;
        assert!(verify_inner(&pr2, &lay, vec![("t", l)], &mut Tape::new(&coins)).is_err());
    }

    #[test]
    fn level0_queries_are_the_soundness_lanes_counts() {
        for (l, q) in [(1 << 12, Some(277)), (1 << 13, Some(242)), (1 << 14, Some(229)), (1 << 17, Some(219)), (1 << 18, Some(219)),
                       (1 << 19, Some(218)), (1 << 9, None), (1 << 10, None), (1 << 11, None)] {
            assert_eq!(level0_queries(l, 218), q, "l = {l}");
        }
    }

    #[test]
    fn rank_of_random_points_is_full() {
        let mut rng = ProverRng::from_seed(ProverSeed::injected([8; 32]));
        let pts: Vec<Vec<F128>> = (0..2).map(|i| rng.f128s(1, 100 + i, 18)).collect();
        let words: Vec<usize> = (0..8).flat_map(|b| (0..128).map(move |j| b * (1 << 15) + 12288 + j)).collect();
        let (rank, ok) = mask_rank_ok(&pts, &words);
        assert!(ok, "rank {rank}");
        let (_, ok1) = mask_rank_ok(&pts, &words[..200]);
        assert!(!ok1, "200 words cannot span 256 dimensions");
    }
}
