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
//! `final_c'` (a constant of two constraints, which upstream never absorbs) is absorbed before the coins, the prover sends
//! `τ = ⟨c, μ⟩` for a committed uniform `μ`, then `Y = h + β μ` (with its padding), and opens `Q` columns of both codewords.
//!
//! **Simulation** ([`simulate_inner`], and the binary's `simulate`): every exposed value is uniform (the pads are), `Y` is
//! uniform, `τ = (⟨c, Y⟩ + t) / β`, and the opened columns are uniform subject to `C(Y)_j = C(h)_j + β C(μ)_j` (the padding
//! makes any `Q ≤ K_PAD` columns of one codeword uniform). The prover refuses the degenerate coins `β = 0` and `ε`-independent
//! cases, a zerocheck outer coin `r_j = 1` ([`Masking`], as it opens), and a mask slot whose weights fail to span
//! (`mask_rank_ok`); the simulator refuses the same coins. The prover also stops when its own batched constraint fails
//! ([`SelfCheck`]), before `τ`'s commitment: the simulation covers runs whose constraint holds, and `Y` of a failing one
//! gives away `⟨c, h⟩ + t`.

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
/// [`crate::zk_hooks::ProverRng`]'s purposes (an injected seed's ChaCha20 streams; 1-3 are M0's).
pub const STREAM_PADS: u64 = 4;
pub const STREAM_DOT_MASK: u64 = 5;

/// Ligerito level 0's padding per lane for `q0` queries per rep: every distinct level-0 position the session opens (both
/// reps' queries), so those positions of a lane are uniform. Nothing else evaluates the padded lanes: the OOD value, the
/// lane-round messages and each pair's sums are masked by the algebraic pads, and each rep's `ybar` by its own pair's padding.
pub fn level0_t_pad(q0: usize) -> usize {
    2 * q0
}

/// Level 0's query bits per run: with `l` message words per lane the code is `RS[2l, l + t_pad]`, of rate
/// `ρ' = (l + t_pad) / 2l`, and `q` queries give `q log₂(1/(√ρ' + η))`, `η = 1/50` (the Johnson slack of the soundness lane's
/// `Accounting/Bound.lean`).
pub fn level0_bits(l: usize, q: usize) -> f64 {
    let rate = (l + level0_t_pad(q)) as f64 / (2 * l) as f64;
    -(q as f64) * (rate.sqrt() + 0.02).log2()
}

/// Goldreich–Kahan's estimate of a verifier's completion probability `p` (M2's simulator): fresh runs (`run`: whether one
/// reached the proofs) until `t` of them do, with no cap and without the extraction run; returns the runs, so
/// `p̃ = t / runs`. `t = Θ(λ)` keeps `p̃` within `[p/2, 2p]` except with probability `2^-Ω(λ)`. The estimation follows an
/// extraction run that reached the proofs (probability `p`), so it costs `t` runs in expectation.
pub fn gk_estimate(t: usize, mut run: impl FnMut() -> bool) -> usize {
    let (mut ok, mut runs) = (0, 0);
    while ok < t {
        runs += 1;
        ok += run() as usize;
    }
    runs
}

/// The simulator's rewinds after an estimate from `runs` runs: `4t / p̃ = 4·runs`.
pub fn gk_rewinds(runs: usize) -> usize {
    4 * runs
}

/// The least `q ≥ q_min` with Fast100's 100 bits per run at level 0 ([`level0_bits`]); `None` when the padding outgrows the
/// lane first.
pub fn level0_queries(l: usize, q_min: usize) -> Option<usize> {
    (q_min..).take_while(|&q| level0_t_pad(q) <= l).find(|&q| level0_bits(l, q) >= 100.0)
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
    /// The pads in the order the prover absorbs the values they mask (`final_c`, which upstream never absorbs, is the inner
    /// proof's first item): the algebraic phase's, then level 0's (absorbed between `zk::LABEL_OPEN` and `zk::LABEL_CLOSE`).
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
    /// the salt of `τ`'s commitment (M2)
    pub tau_salt: Vec<u8>,
}

impl Pads {
    /// Rep `rep`'s pads from the prover's randomness (OS bytes; never from the transcript), drawn once and kept in the rep's state.
    pub fn draw(rng: &mut crate::zk_hooks::ProverRng, lay: &Layout, rep: usize) -> Self {
        let s = lay.len();
        let base = (rep as u64) << 32;
        let mut h = rng.f128s(STREAM_PADS, base, s);
        h[lay.hab()] = h[lay.fa()] * h[lay.fb()];
        h[lay.tw()] = h[lay.tx()] * h[lay.ty()];
        let mu = rng.f128s(STREAM_DOT_MASK, base, s);
        let pad_h = rng.f128s(STREAM_PADS, base + (1 << 24), K_PAD);
        let pad_mu = rng.f128s(STREAM_DOT_MASK, base + (1 << 24), K_PAD);
        let tau_salt = rng.f128s(STREAM_PADS, base + (1 << 25), TAU_SALT_BYTES / 16).iter().flat_map(|x| [x.lo.to_le_bytes(), x.hi.to_le_bytes()].concat()).collect();
        Pads { h, mu, pad_h, pad_mu, tau_salt }
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
    /// the zerocheck's outer coins: `coins[K_SKIP..K_SKIP + n_outer]` ([`points_from_coins`]' `r_outer`)
    n_outer: usize,
}

impl<C: Challenger> Masking<C> {
    pub fn new(inner: C, lay: &Layout, pads: &Pads, root: [u8; 64]) -> Self {
        Masking { inner, order: lay.absorbed(), n_piop: lay.piop_absorbed(), lig: lay.ligerito(), h: pads.h.clone(), at: 0, on: false, done: false,
                  done_l0: false, root, coins: Vec::new(), coins_l0: Vec::new(), l0: false, at_open: None, n_outer: lay.n_mlv - N_INNER }
    }
    /// The prover stops at a zerocheck outer coin `r_j = 1` among `coins[from..]`, as the coin opens and before it sends
    /// anything that depends on it, reading the coin alone. At `r_j = 1` the replay's round `j` divides by zero, and the
    /// honest pads miss the inner proof's constraint by a witness-dependent amount that the view would show. Under uniform
    /// coins this refuses with probability at most `n_outer · 2^-128` per rep.
    fn refuse_outer(&self, from: usize) {
        let outer = K_SKIP..K_SKIP + self.n_outer;
        if let Some(i) = (from..self.coins.len()).find(|i| outer.contains(i) && self.coins[*i] == F128::ONE) {
            panic!("PROVER-STOPPED: degenerate coin: zerocheck outer coin r_{} = 1", i - K_SKIP);
        }
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
            self.refuse_outer(self.coins.len() - 1);
        }
        x
    }
    fn sample_f128_vec(&mut self, n: usize) -> Vec<F128> {
        let x = self.inner.sample_f128_vec(n);
        if self.l0 {
            self.coins_l0.extend_from_slice(&x);
        } else if self.on {
            self.coins.extend_from_slice(&x);
            self.refuse_outer(self.coins.len() - n);
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
                  done: true, done_l0: true, root: self.root, coins: Vec::new(), coins_l0: Vec::new(), l0: false, at_open: None, n_outer: 0 }
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
    l.comb_partial = Vec::new();
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
/// drawing exactly their coins, with every message-dependent quantity an affine form in the pads. `own_comb`: the prover's own
/// replay passes its lincheck's folded `comb` (upstream's `comb_partial`) instead of folding again; a verifier passes `None`
/// and folds its own (a proof's field is never read).
#[allow(clippy::too_many_arguments)]
pub fn replay<Ch: Challenger>(r1cs: &BlockR1cs, zc: &ZerocheckProof, lc: &LincheckProof, circuit: &dyn LincheckCircuit, lay: &Layout,
                              root: &[u8; 64], own_comb: Option<&[F128]>, ch: &mut Ch) -> Result<Replayed, String> {
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
    if own_comb.is_some_and(|c| c.len() != SKIP) {
        return Err("zk: the prover's folded comb has the wrong length".into());
    }
    let own = own_comb.is_some();
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
    if let Some(c) = own_comb {
        comb_vec = c.to_vec();
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
    /// M2: `τ` is sent as this commitment before `β` and opened (with `tau_salt`) only in the proof, after the last coin, so
    /// no relation among the prover's messages can be tested before every coin is out (the single-rewind simulator's
    /// extraction run relies on it)
    pub tau_commit: Vec<u8>,
    pub tau_salt: Vec<u8>,
    pub tau: F128,
    pub y: Vec<F128>,
    pub opened: Vec<(F128, F128)>,
    pub salts: Vec<flock_merkle::hm96::Salt>,
    pub paths: Vec<Vec<Hash>>,
}

pub const TAU_SALT_BYTES: usize = 192;
const TAU_TAG: &[u8] = b"verity/flock-zk/tau/v1\0";

/// `τ`'s commitment: the `hm96-sha512/v1` leaf of `SHA-512(TAU_TAG ‖ τ)` under `salt`.
pub fn tau_commitment(tau: F128, salt: &[u8]) -> Vec<u8> {
    let x = flock_hash::digest_from(&Sha512::digest([TAU_TAG, &f128_bytes(&[tau])].concat()));
    flock_merkle::hm96::leaf(&x, salt).as_flattened().to_vec()
}

/// Batch the constraints with the verifier's coins: `(c, t)` with `⟨c, h⟩ = t`.
pub fn batch(cons: &[(&'static str, Lin)], gammas: &[F128]) -> (Vec<F128>, F128) {
    let l = Lin::combo(gammas, &cons.iter().map(|c| c.1.clone()).collect::<Vec<_>>());
    (l.a, l.c)
}

fn queries<Ch: Challenger>(ch: &mut Ch, n: usize) -> Vec<usize> {
    query_positions(&ch.sample_f128_vec(Q), n)
}

/// The inner proof's opened columns for its query coins (drawn with replacement, so a position may repeat).
pub fn query_positions(words: &[F128], n: usize) -> Vec<usize> {
    words.iter().map(|x| (x.lo as usize) & (n - 1)).collect()
}

/// Whether [`prove_inner`] checks the batched constraint on its own pads before it sends anything after the batching coins.
/// A failing constraint's `Y` gives away `⟨c, h⟩ + t`, a value of the witness, so the prover checks before it speaks
/// ([`SelfCheck::Stop`], every honest plan's). [`SelfCheck::Skip`] belongs to a negative control, whose witness fails by
/// design and whose proof the verifier must refuse, and to the simulator's dummy runs, which must look like the prover's
/// runs until the proofs whatever the dummy witness is.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SelfCheck {
    #[default]
    Stop,
    Skip,
}

/// The prover's inner proof, on the rep's challenger after the opening. `cons` are the algebraic phase's and the ring
/// switch's constraints (the triple's are added here). `final_c`: the proof's `final_c'`, a constant of two of `cons` that
/// nothing before absorbs, absorbed before `ε` so that it is fixed before the batching coins. With [`SelfCheck::Stop`] it
/// stops when `⟨c, h⟩ ≠ t` after the batching coins, before `τ`'s commitment goes out. Refuses `β = 0` (the simulator
/// refuses the same coins).
pub fn prove_inner<Ch: Challenger>(ic: &InnerCommit, p: &Pads, lay: &Layout, final_c: F128, mut cons: Vec<(&'static str, Lin)>,
                                   check: SelfCheck, ch: &mut Ch) -> Result<InnerProof, String> {
    ch.observe_label(LABEL_INNER);
    ch.observe_f128(final_c);
    let eps = ch.sample_f128();
    let rho = eps * p.h[lay.fa()] + p.h[lay.tx()];
    let sigma = p.h[lay.fb()] + p.h[lay.ty()];
    ch.observe_f128_slice(&[rho, sigma]);
    cons.extend(triple_constraints(lay, eps, rho, sigma));
    let gammas = ch.sample_f128_vec(cons.len());
    let (c, t) = batch(&cons, &gammas);
    if check == SelfCheck::Stop && c.iter().zip(&p.h).fold(t, |a, (x, y)| a + *x * *y) != F128::ZERO {
        return Err("PROVER-STOPPED: the batched constraint fails (⟨c, h⟩ ≠ t)".into());
    }
    let tau = c.iter().zip(&p.mu).fold(F128::ZERO, |a, (x, y)| a + *x * *y);
    let tau_salt = p.tau_salt.clone();
    let tau_commit = tau_commitment(tau, &tau_salt);
    ch.observe_bytes(&tau_commit);
    let beta = ch.sample_f128();
    if beta == F128::ZERO {
        return Err("PROVER-STOPPED: degenerate coin β = 0".into());
    }
    let y: Vec<F128> = p.h.iter().zip(&p.mu).map(|(a, b)| *a + beta * *b)
        .chain(p.pad_h.iter().zip(&p.pad_mu).map(|(a, b)| *a + beta * *b)).collect();
    ch.observe_f128_slice(&y);
    let qs = queries(ch, ic.code.n);
    Ok(InnerProof {
        root: ic.root().to_vec(), rho, sigma, tau_commit, tau_salt, tau, y,
        opened: qs.iter().map(|&j| (ic.ch[j], ic.cm[j])).collect(),
        salts: qs.iter().map(|&j| flock_merkle::hm96::salt_of(&ic.salts[j * crate::zk_hooks::SALT_BYTES..])).collect(),
        paths: qs.iter().map(|&j| ic.path(j)).collect(),
    })
}

/// The verifier's side of [`prove_inner`].
pub fn verify_inner<Ch: Challenger>(pr: &InnerProof, lay: &Layout, final_c: F128, mut cons: Vec<(&'static str, Lin)>, ch: &mut Ch)
    -> Result<(), String> {
    let code = Code::new(lay.len() + K_PAD);
    if pr.y.len() != code.msg || pr.opened.len() != Q || pr.salts.len() != Q || pr.paths.len() != Q || pr.root.len() != 64
        || pr.tau_commit.len() != 64 || pr.tau_salt.len() != TAU_SALT_BYTES
    {
        return Err("zk inner: malformed proof".into());
    }
    ch.observe_label(LABEL_INNER);
    ch.observe_f128(final_c);
    let eps = ch.sample_f128();
    ch.observe_f128_slice(&[pr.rho, pr.sigma]);
    cons.extend(triple_constraints(lay, eps, pr.rho, pr.sigma));
    let gammas = ch.sample_f128_vec(cons.len());
    let (c, t) = batch(&cons, &gammas);
    ch.observe_bytes(&pr.tau_commit);
    let beta = ch.sample_f128();
    if beta == F128::ZERO {
        return Err("zk inner: degenerate coin β = 0".into());
    }
    ch.observe_f128_slice(&pr.y);
    let qs = queries(ch, code.n);
    if tau_commitment(pr.tau, &pr.tau_salt) != pr.tau_commit {
        return Err("zk inner: τ does not open its commitment".into());
    }
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
///
/// The rows are inserted one at a time into an echelon basis (one vector per pivot column, its lowest set bit), each reduced
/// against it lowest bit first, and evaluated only as they are inserted: the rank can't exceed the `128 k` columns, so once
/// the basis is full the remaining words are not read. Honest coins fill it within a few blocks' words.
pub fn mask_rank_ok(points: &[Vec<F128>], mask_words: &[usize]) -> (usize, bool) {
    let limbs = 2 * points.len();
    let full = 64 * limbs;
    let mut basis: Vec<Option<Vec<u64>>> = vec![None; full];
    let mut rank = 0;
    for &x in mask_words {
        if rank == full {
            break;
        }
        let mut row: Vec<u64> = points.iter().flat_map(|p| {
            let e = eq_word(p, x);
            [e.lo, e.hi]
        }).collect();
        let mut w = 0;
        while let Some(at) = (w..limbs).find(|&i| row[i] != 0) {
            w = at;
            let col = 64 * w + row[w].trailing_zeros() as usize;
            match &basis[col] {
                Some(b) => row[w..].iter_mut().zip(&b[w..]).for_each(|(r, v)| *r ^= v),
                None => {
                    basis[col] = Some(row);
                    rank += 1;
                    break;
                }
            }
        }
    }
    (rank, rank == full)
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
    let tau_salt: Vec<u8> = rng.f128s(STREAM_PADS, base + (1 << 22), TAU_SALT_BYTES / 16).iter().flat_map(|x| [x.lo.to_le_bytes(), x.hi.to_le_bytes()].concat()).collect();
    let tau_commit = tau_commitment(tau, &tau_salt);
    let qs = queries(coins, code.n);
    let hs = rng.f128s(STREAM_PADS, base + (1 << 21), qs.len());
    let (mut ch, mut cm) = (vec![F128::ZERO; code.n], vec![F128::ZERO; code.n]);
    for (&j, a) in qs.iter().zip(hs) {
        ch[j] = a;
        cm[j] = (code.at(&y, j) + a) * beta.inv();
    }
    let ic = InnerCommit::from_columns(code, ch, cm, salts);
    Ok(InnerProof {
        root: ic.root().to_vec(), rho, sigma, tau_commit, tau_salt, tau, y,
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
        for i in lay.absorbed().into_iter().chain([lay.fc()]) {
            assert!(!std::mem::replace(&mut seen[i], true));
        }
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
        let pr = prove_inner(&ic, &p, &lay, F128::ZERO, vec![("t", l.clone())], SelfCheck::Stop, &mut Tape::new(&coins)).unwrap();
        verify_inner(&pr, &lay, F128::ZERO, vec![("t", l.clone())], &mut Tape::new(&coins)).unwrap();
        let mut bad = l.clone();
        bad.c += F128::ONE;
        assert!(verify_inner(&pr, &lay, F128::ZERO, vec![("t", bad)], &mut Tape::new(&coins)).is_err());
        let mut pr2 = pr.clone();
        pr2.opened[3].0 += F128::ONE;
        assert!(verify_inner(&pr2, &lay, F128::ZERO, vec![("t", l)], &mut Tape::new(&coins)).is_err());
    }

    /// `final_c'` is a constant of two rows (`replay`'s c_eval, the ring switch's c claim), so a prover who knew `γ` before
    /// fixing it could cancel any failing row. Coins are committed in advance (M2) and only the rounds' bytes bind a message:
    /// the verifier accepts when the inner proof checks and its rounds are the ones the prover sent before each coin.
    #[test]
    fn final_c_solved_after_the_batching_coins_is_refused() {
        let lay = Layout::new(20, 16, 0, 0);
        let s = lay.len();
        let mut rng = ProverRng::from_seed(ProverSeed::injected([7; 32]));
        let p = Pads::draw(&mut rng, &lay, 0);
        let code = Code::new(s + K_PAD);
        let ic = InnerCommit::new(&p, crate::zk_hooks::LeafSalts::for_seed(&ProverSeed::injected([8; 32])).expand(9, code.n));
        let v = F128::new(0xc0ffee, 0x5eed);
        let fc = v + p.h[lay.fc()];
        let cons = |fc: F128, broken: bool| -> Vec<(&'static str, Lin)> {
            let ab = p.h[lay.fa()] + if broken { F128::ONE } else { F128::ZERO };
            vec![("zerocheck c_eval", Lin::masked(fc, lay.fc(), s).add_const(v)), ("zerocheck final a·b", Lin::masked(ab, lay.fa(), s)),
                 ("ring switch c", Lin::masked(fc, lay.fc(), s).add_const(v))]
        };
        let coin = |k: u64| F128::new(k.wrapping_mul(0x9e37_79b9_7f4a_7c15), k + 1);
        let schedule = vec![vec![coin(1)], (10..16).map(coin).collect(), vec![coin(99)], (100..100 + Q as u64).map(coin).collect()];
        let run = |broken: bool, solve: bool| -> bool {
            let mut pch = crate::RecordingChallenger::new(b"rep", schedule.clone());
            let check = if broken { SelfCheck::Skip } else { SelfCheck::Stop };
            let pr = prove_inner(&ic, &p, &lay, fc, cons(fc, broken), check, &mut pch).unwrap();
            let mut sent = fc;
            if solve {
                let mut all = cons(fc, broken);
                all.extend(triple_constraints(&lay, schedule[0][0], pr.rho, pr.sigma));
                let g = &schedule[1];
                let (c, t) = batch(&all, g);
                let sum = c.iter().zip(&p.h).fold(t, |a, (x, y)| a + *x * *y);
                sent += sum * (g[0] + g[2]).inv();
            }
            let mut vch = crate::RecordingChallenger::new(b"rep", schedule.clone());
            let ok = verify_inner(&pr, &lay, sent, cons(sent, broken), &mut vch).is_ok();
            assert!(pch.mismatch.is_none() && vch.mismatch.is_none());
            ok && vch.rounds == pch.rounds
        };
        assert!(run(false, false), "the honest proof");
        assert!(!run(true, false), "a failing row");
        assert!(!run(true, true), "a failing row cancelled by final_c' solved after γ");
    }

    /// The prover checks before it speaks: on a failing row it stops after the batching coins, with the rounds an honest run
    /// sends up to them and nothing observed after them. On the honest witness the check changes nothing: the same rounds and
    /// the same proof bytes as the unchecked prover's, pinned from the prover before the check.
    #[test]
    fn the_prover_stops_before_tau_on_a_failing_row() {
        let lay = Layout::new(20, 16, 0, 0);
        let s = lay.len();
        let p = Pads::draw(&mut ProverRng::from_seed(ProverSeed::injected([7; 32])), &lay, 0);
        let code = Code::new(s + K_PAD);
        let ic = InnerCommit::new(&p, crate::zk_hooks::LeafSalts::for_seed(&ProverSeed::injected([8; 32])).expand(9, code.n));
        let v = F128::new(0xc0ffee, 0x5eed);
        let fc = v + p.h[lay.fc()];
        let cons = |broken: bool| -> Vec<(&'static str, Lin)> {
            let ab = p.h[lay.fa()] + if broken { F128::ONE } else { F128::ZERO };
            vec![("zerocheck c_eval", Lin::masked(fc, lay.fc(), s).add_const(v)), ("zerocheck final a·b", Lin::masked(ab, lay.fa(), s)),
                 ("ring switch c", Lin::masked(fc, lay.fc(), s).add_const(v))]
        };
        let coin = |k: u64| F128::new(k.wrapping_mul(0x9e37_79b9_7f4a_7c15), k + 1);
        let schedule = vec![vec![coin(1)], (10..16).map(coin).collect(), vec![coin(99)], (100..100 + Q as u64).map(coin).collect()];
        let run = |broken: bool, check: SelfCheck| {
            let mut pch = crate::RecordingChallenger::new(b"rep", schedule.clone());
            let pr = prove_inner(&ic, &p, &lay, fc, cons(broken), check, &mut pch);
            assert!(pch.mismatch.is_none());
            (pr.map(|pr| bincode::serialize(&pr).unwrap()), pch)
        };
        let (honest, hch) = run(false, SelfCheck::Stop);
        let (unchecked, uch) = run(false, SelfCheck::Skip);
        let honest = honest.unwrap();
        assert_eq!(Ok(&honest), unchecked.as_ref());
        assert_eq!(hch.rounds, uch.rounds);
        assert_eq!(hch.rounds.len(), 4);
        let (stopped, sch) = run(true, SelfCheck::Stop);
        assert_eq!(stopped, Err("PROVER-STOPPED: the batched constraint fails (⟨c, h⟩ ≠ t)".to_string()));
        assert_eq!(sch.rounds, hch.rounds[..2], "ε's round and the batching coins' round, as the honest run sends them");
        assert!(sch.pending.is_empty(), "nothing observed after the batching coins");
        let (sent, bch) = run(true, SelfCheck::Skip);
        assert!(sent.is_ok() && bch.rounds.len() == 4, "the negative control's prover sends its proof");
        let digest: String = Sha512::digest(&honest).iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(digest, HONEST_INNER_PROOF_SHA512);
    }

    /// SHA-512 of the honest inner proof's bincode in [`the_prover_stops_before_tau_on_a_failing_row`], from `prove_inner`
    /// before the self-check existed (main at 5049de02f).
    const HONEST_INNER_PROOF_SHA512: &str =
        "a817dca1ba04991788bf6b362c6d15a80dc1fc68bba6a66cdf2bfaf0977bb45e7c8ad12b43c554e04501990345833130dfa6e9625e48d89f1bc112c13108e35f";

    /// The prover stops at a zerocheck outer coin `r_j = 1` as the coin opens, drawn as one vector or one at a time, and at
    /// no other coin equal to 1 (a skip coin, a later one).
    #[test]
    fn outer_coin_one_is_refused_as_it_opens() {
        let lay = Layout::new(20, 16, 0, 0);
        let p = Pads::draw(&mut ProverRng::from_seed(ProverSeed::injected([9; 32])), &lay, 0);
        let n_outer = lay.n_mlv - N_INNER;
        let run = |outer_at: Option<usize>, singly: bool| -> Result<usize, String> {
            let mut outer: Vec<F128> = (0..n_outer as u64).map(|i| F128::new(i + 2, 7)).collect();
            if let Some(j) = outer_at {
                outer[j] = F128::ONE;
            }
            let mut schedule = vec![vec![F128::ONE; K_SKIP]];
            if singly {
                schedule.extend(outer.into_iter().map(|x| vec![x]));
            } else {
                schedule.push(outer);
            }
            schedule.push(vec![F128::ONE]);
            let mut ch = Masking::new(crate::RecordingChallenger::new(b"rep", schedule), &lay, &p, [0; 64]);
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
                ch.observe_label(LABEL_ZC);
                ch.sample_f128_vec(K_SKIP);
                if singly {
                    (0..n_outer).for_each(|_| { ch.sample_f128(); });
                } else {
                    ch.sample_f128_vec(n_outer);
                }
                ch.sample_f128();
                assert!(ch.inner.mismatch.is_none());
                ch.coins.len()
            }))
            .map_err(|e| e.downcast_ref::<String>().cloned().unwrap_or_default())
        };
        for singly in [false, true] {
            assert_eq!(run(None, singly), Ok(K_SKIP + n_outer + 1));
            for j in [0, n_outer / 2, n_outer - 1] {
                assert_eq!(run(Some(j), singly), Err(format!("PROVER-STOPPED: degenerate coin: zerocheck outer coin r_{j} = 1")));
            }
        }
    }

    #[test]
    fn level0_query_counts() {
        // the least counts at t_pad = 2q (the red team's table), and none where the padding outgrows the lane at rate 1/2
        for (l, q) in [(1 << 12, Some(268)), (1 << 13, Some(239)), (1 << 14, Some(227)), (1 << 17, Some(219)), (1 << 18, Some(219)),
                       (1 << 19, Some(218)), (1 << 9, None), (1 << 10, None), (1 << 11, None)] {
            assert_eq!(level0_queries(l, 218), q, "l = {l}");
        }
        // the agreed counts at RoPE's sizes keep 100 bits per run
        for (l, q) in [(1 << 12, 277), (1 << 13, 242), (1 << 14, 229)] {
            assert!(level0_bits(l, q) >= 100.0 && level0_bits(l, q - 10) < 100.0, "l = {l}");
        }
    }

    /// Level 0 and level 1 of a two-rep session at toy size, exactly: for fixed coins the view (every row either rep opens,
    /// each rep's `ybar`, the pair's masked sums `e'` and, conservatively, all of `y_1'`) is F128-linear in the witness lanes
    /// and the prover's randomness (every lane's padding, the extra lanes, `e`'s pads), so it is independent of the witness
    /// exactly when the witness's images lie in the span of the randomness's. Built from the protocol's own `lane_codeword`,
    /// `pad_codeword`, `ybar` and `enforced_extra`; the verifier's row check holds on it.
    mod exact {
        use flock_core::pcs::ligerito::zk::{self as z0, EXTRA_LANES, Level0Zk};
        use flock_field::{F128, F256};

        use crate::zk_hooks::{ProverRng, ProverSeed};

        const LOG_MSG: usize = 4;
        const L: usize = 1 << LOG_MSG;
        const W: usize = 4;

        struct Coins {
            eq: [Vec<F256>; 2],
            rho: [[F256; 2]; 2],
            b: [Vec<F256>; 2],
            q: [Vec<usize>; 2],
        }

        fn coins(q: [Vec<usize>; 2]) -> Coins {
            let mut rng = ProverRng::from_seed(ProverSeed::injected([21; 32]));
            let mut f256 = |i: u64, n: usize| -> Vec<F256> { rng.f128s(1, i << 20, 2 * n).chunks(2).map(|c| F256::new(c[0], c[1])).collect() };
            let eq = [0, 1].map(|r| {
                let c = f256(r, 2);
                (0..W).map(|l| (0..2).fold(F256::ONE, |a, i| a * if (l >> i) & 1 == 1 { c[i] } else { F256::ONE + c[i] })).collect()
            });
            let rho = [0, 1].map(|r| {
                let v = f256(10 + r, 2);
                assert!(z0::rho_independent([v[0], v[1]]));
                [v[0], v[1]]
            });
            Coins { eq, rho, b: [f256(20, L), f256(21, L)], q }
        }

        fn enc(v: &[F128]) -> Vec<F128> {
            z0::lane_codeword(&[], Some(v), LOG_MSG, 1)
        }

        /// `x`: the witness lanes (`W·L`), then every lane's padding (`(W + 4)·t_pad`), the extra lanes (`4·L`), `e`'s pads (8).
        fn vars(t_pad: usize) -> (usize, usize) {
            (W * L, (W + EXTRA_LANES) * t_pad + EXTRA_LANES * L + 8)
        }

        /// The session's view, and whether the verifier's row check holds at every rep's queries. `pair`: each rep's pair.
        fn view(x: &[F128], t_pad: usize, pair: [usize; 2], c: &Coins) -> (Vec<F128>, bool) {
            let p: Vec<&[F128]> = x[..W * L].chunks(L).collect();
            let at = W * L;
            let mu: Vec<Vec<F128>> = x[at..at + (W + EXTRA_LANES) * t_pad].chunks(t_pad).map(|m| m.to_vec()).collect();
            let at = at + (W + EXTRA_LANES) * t_pad;
            let r: Vec<Vec<F128>> = x[at..at + EXTRA_LANES * L].chunks(L).map(|m| m.to_vec()).collect();
            let h = &x[at + EXTRA_LANES * L..];
            let cw: Vec<Vec<F128>> = (0..W + EXTRA_LANES).map(|l| if l < W {
                enc(p[l]).iter().zip(z0::lane_codeword(&mu[l], None, LOG_MSG, 1)).map(|(a, b)| *a + b).collect()
            } else {
                z0::lane_codeword(&mu[l], Some(&r[l - W]), LOG_MSG, 1)
            }).collect();
            let z = Level0Zk { queries0: 0, t_pad, rep: 0, extra: std::array::from_fn(|j| r[j].clone()), pads: mu.clone(), tamper: 0 };
            let mut positions: Vec<usize> = c.q.concat();
            positions.sort();
            positions.dedup();
            let mut v: Vec<F128> = positions.iter().flat_map(|&q| cw.iter().map(move |l| l[q])).collect();
            let mut ok = true;
            for rep in 0..2 {
                let zr = Level0Zk { rep: pair[rep], ..z.clone() };
                let [a, b] = z0::rep_lanes(pair[rep]);
                let (eq, rho) = (&c.eq[rep], c.rho[rep]);
                let y1: Vec<F256> = (0..L).map(|i| (0..W).fold(rho[0] * r[a][i] + rho[1] * r[b][i], |acc, l| acc + eq[l] * p[l][i])).collect();
                let ybar = z0::ybar(&zr, W, eq, rho);
                let padcw = z0::pad_codeword(&ybar, LOG_MSG, 1);
                let [e0, e1] = [enc(&y1.iter().map(|y| y.c0).collect::<Vec<_>>()), enc(&y1.iter().map(|y| y.c1).collect::<Vec<_>>())];
                for &q in &c.q[rep] {
                    let row: Vec<F128> = cw.iter().map(|l| l[q]).collect();
                    let lhs = (0..W).fold(z0::enforced_extra(&[row.clone()], &[q], &[F128::ONE], W, pair[rep], rho, &padcw), |acc, l| acc + eq[l] * row[l]);
                    ok &= lhs == F256::new(e0[q], e1[q]);
                }
                let e = [a, b].map(|j| (0..L).fold(F256::ZERO, |acc, i| acc + c.b[rep][i] * r[j][i]));
                v.extend(y1.iter().chain(&ybar).flat_map(|y| y.coordinates()));
                v.extend(e.iter().flat_map(|y| y.coordinates()).zip(&h[4 * rep..4 * rep + 4]).map(|(a, b)| a + *b));
            }
            (v, ok)
        }

        fn rank(mut m: Vec<Vec<F128>>) -> usize {
            let cols = m.first().map_or(0, |r| r.len());
            let mut r = 0;
            for col in 0..cols {
                let Some(p) = (r..m.len()).find(|&i| m[i][col] != F128::ZERO) else { continue };
                m.swap(r, p);
                let iv = m[r][col].inv();
                let pivot: Vec<F128> = m[r].iter().map(|x| *x * iv).collect();
                for row in m.iter_mut() {
                    let f = row[col];
                    if f != F128::ZERO {
                        row.iter_mut().zip(&pivot).for_each(|(x, y)| *x += f * *y);
                    }
                }
                m[r] = pivot;
                r += 1;
            }
            r
        }

        /// The dimension of the witness's images outside the randomness's span (0: the view hides the witness).
        fn leak(t_pad: usize, pair: [usize; 2], c: &Coins) -> usize {
            let (nw, nr) = vars(t_pad);
            let image = |i: usize| {
                let mut x = vec![F128::ZERO; nw + nr];
                x[i] = F128::ONE;
                view(&x, t_pad, pair, c).0
            };
            let rand: Vec<Vec<F128>> = (nw..nw + nr).map(image).collect();
            let all: Vec<Vec<F128>> = rand.iter().cloned().chain((0..nw).map(image)).collect();
            rank(all) - rank(rand)
        }

        /// Coins and query sets a malicious verifier could choose (red-team-m1-zk's variant): with a pair per rep nothing
        /// leaks, and the shared-pair control leaks wherever the reps' coins differ.
        #[test]
        fn adversarial_coins_and_queries_leak_nothing() {
            let unit = |i: usize| -> Vec<F256> { (0..W).map(|l| if l == i { F256::ONE } else { F256::ZERO }).collect() };
            for (name, q, same, boolean, shared_leaks) in [
                ("identical queries", [vec![1, 7, 20], vec![1, 7, 20]], false, false, true),
                ("disjoint, t_pad = distinct", [vec![1, 2, 3], vec![4, 5, 6]], false, false, true),
                ("identical coins in both reps", [vec![1, 7, 20], vec![3, 7, 30]], true, false, false),
                ("identical coins and queries", [vec![2, 9, 17], vec![2, 9, 17]], true, false, false),
                ("boolean lane challenges", [vec![1, 7, 20], vec![3, 7, 30]], false, true, true),
            ] {
                let mut c = coins(q);
                if same {
                    c.eq[1] = c.eq[0].clone();
                    c.rho[1] = c.rho[0];
                    c.b[1] = c.b[0].clone();
                }
                if boolean {
                    c.eq = [unit(0), unit(3)];
                }
                assert_eq!(leak(6, [0, 1], &c), 0, "{name}: a pair per rep leaks");
                assert_eq!(leak(6, [0, 0], &c) > 0, shared_leaks, "{name}: the shared-pair control");
            }
        }

        #[test]
        fn a_pair_per_rep_hides_the_witness_and_a_shared_pair_does_not() {
            // 3 queries per rep, 5 distinct positions: t_pad = 2 * 3 covers them
            let c = coins([vec![1, 7, 20], vec![3, 7, 30]]);
            let mut rng = ProverRng::from_seed(ProverSeed::injected([22; 32]));
            let (nw, nr) = vars(6);
            let (_, ok) = view(&rng.f128s(2, 0, nw + nr), 6, [0, 1], &c);
            assert!(ok, "the verifier's row check fails on the model");
            assert_eq!(leak(6, [0, 1], &c), 0, "a pair per rep leaks");
            assert!(leak(6, [0, 0], &c) > 0, "a shared pair should leak (red-team-m1-zk F1)");
            // padding below the distinct opened positions leaks too
            assert!(leak(4, [0, 1], &c) > 0, "t_pad below the opened positions should leak");
        }
    }

    /// The estimate runs until `t` successes whatever `p` (red-team-m2-zk G2: a cap of 64t runs let a tiny `p` stop it at one
    /// success), lands within `[p/2, 2p]`, and the rewinds it allows all fail with probability `(1 − p)^(4·runs) ≈ e^-4t`.
    #[test]
    fn the_gk_estimate_is_uncapped_and_its_rewinds_suffice() {
        let mut rng = ProverRng::from_seed(ProverSeed::injected([31; 32]));
        let t = 32;
        for (i, p) in [1.0, 0.5, 1.0 / 64.0, 1.0 / 4096.0].into_iter().enumerate() {
            let mut k = 0u64;
            let mut coin = || {
                k += 1;
                (rng.f128s(9, (i as u64) << 40 | k, 1)[0].lo as f64 / u64::MAX as f64) < p
            };
            let runs = gk_estimate(t, &mut coin);
            let est = t as f64 / runs as f64;
            assert!(runs >= t && est >= p / 2.0 && est <= 2.0 * p, "p = {p}: p̃ = {est} from {runs} runs");
            if p < 1.0 {
                let fail = (1.0 - p).powi(gk_rewinds(runs) as i32);
                assert!(fail < 2f64.powi(-(t as i32)), "p = {p}: the rewinds all fail with probability {fail}");
            }
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

    /// The oracle: `mask_rank_ok` as it was before the echelon basis (main at 378453fb3), Gauss–Jordan over every word's row.
    fn mask_rank_gauss_jordan(points: &[Vec<F128>], mask_words: &[usize]) -> (usize, bool) {
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

    /// The mask slot's words as the binary's `mask_words` lays them out: `blocks` blocks of `2^block_log` words, the slot's
    /// 128 words at `w0` in each.
    fn slot_words(blocks: usize, block_log: usize, w0: usize) -> Vec<usize> {
        (0..blocks).flat_map(|b| (0..128).map(move |j| (b << block_log) + w0 + j)).collect()
    }

    /// The echelon basis gives the oracle's rank and verdict on random claim points (1 to 4, each with every word, with one
    /// word short of the full rank, and with no words), on degenerate ones (a repeated point, the zero point, a Boolean point,
    /// a point repeated four times), on repeated words, and with no points at all.
    #[test]
    fn mask_rank_matches_gauss_jordan() {
        let mut rng = ProverRng::from_seed(ProverSeed::injected([12; 32]));
        let d = 19;
        let words = slot_words(16, 15, 12288);
        let pt = |rng: &mut ProverRng, i: u64| rng.f128s(1, 200 + i, d);
        let (p, q, r, s) = (pt(&mut rng, 0), pt(&mut rng, 1), pt(&mut rng, 2), pt(&mut rng, 3));
        let zero = vec![F128::ZERO; d];
        let boolean: Vec<F128> = (0..d).map(|i| if i % 3 == 0 { F128::ONE } else { F128::ZERO }).collect();
        let few = |k: usize| words[..128 * k - 1].to_vec();
        let cases: Vec<(&str, Vec<Vec<F128>>, Vec<usize>, Option<bool>)> = vec![
            ("one point", vec![p.clone()], words.clone(), Some(true)),
            ("two points", vec![p.clone(), q.clone()], words.clone(), Some(true)),
            ("three points", vec![p.clone(), q.clone(), r.clone()], words.clone(), Some(true)),
            ("four points", vec![p.clone(), q.clone(), r.clone(), s.clone()], words.clone(), Some(true)),
            ("one point, one word short", vec![p.clone()], few(1), Some(false)),
            ("two points, one word short", vec![p.clone(), q.clone()], few(2), Some(false)),
            ("four points, one word short", vec![p.clone(), q.clone(), r.clone(), s.clone()], few(4), Some(false)),
            ("four points, one block", vec![p.clone(), q.clone(), r.clone(), s.clone()], words[..128].to_vec(), Some(false)),
            ("two points, no words", vec![p.clone(), q.clone()], vec![], Some(false)),
            ("a repeated point", vec![p.clone(), p.clone()], words.clone(), Some(false)),
            ("a repeated pair", vec![p.clone(), q.clone(), p.clone(), q.clone()], words.clone(), Some(false)),
            ("one point four times", vec![p.clone(), p.clone(), p.clone(), p.clone()], words.clone(), Some(false)),
            ("the zero point", vec![zero.clone()], words.clone(), Some(false)),
            ("the zero point beside a random one", vec![p.clone(), zero.clone()], words.clone(), Some(false)),
            ("a Boolean point", vec![boolean.clone()], words.clone(), Some(false)),
            ("a Boolean point beside random ones", vec![p.clone(), boolean, q.clone()], words.clone(), Some(false)),
            ("one word repeated", vec![p.clone()], vec![words[5]; 300], Some(false)),
            ("every word twice", vec![p.clone(), q.clone()], words.iter().flat_map(|&w| [w, w]).collect(), Some(true)),
            ("no points", vec![], words.clone(), Some(true)),
            ("no points, no words", vec![], vec![], Some(true)),
        ];
        for (name, pts, w, want) in cases {
            let got = mask_rank_ok(&pts, &w);
            assert_eq!(got, mask_rank_gauss_jordan(&pts, &w), "{name}");
            if let Some(ok) = want {
                assert_eq!(got.1, ok, "{name}: rank {} of {}", got.0, 128 * pts.len());
            }
        }
        for k in 1..=4 {
            for trial in 0..4u64 {
                let pts: Vec<Vec<F128>> = (0..k as u64).map(|i| pt(&mut rng, 1000 + 16 * trial + i)).collect();
                for n in [128 * k - 1, 128 * k, 128 * k + 64, words.len()] {
                    assert_eq!(mask_rank_ok(&pts, &words[..n]), mask_rank_gauss_jordan(&pts, &words[..n]), "{k} points, {n} words, trial {trial}");
                }
            }
        }
    }

    /// At the proved sizes, the mask slots of K = 4096 (2,048 blocks of 2^17 words: 262,144 mask words) and K = 14,336 (512
    /// blocks of 2^19: 65,536), with 28-coordinate claim points: the rank check's two calls (2 points, then 4), and a repeated
    /// pair, which reads every word.
    #[test]
    #[cfg_attr(debug_assertions, ignore = "the oracle at the proved sizes takes minutes unoptimised: cargo test --release (check_build.sh test)")]
    fn mask_rank_matches_gauss_jordan_at_the_proved_sizes() {
        let mut rng = ProverRng::from_seed(ProverSeed::injected([13; 32]));
        let pts: Vec<Vec<F128>> = (0..4).map(|i| rng.f128s(1, 300 + i, 28)).collect();
        for (blocks, block_log) in [(2048, 17), (512, 19)] {
            let words = slot_words(blocks, block_log, 3 << (block_log - 3));
            for (name, p) in [("two points", pts[..2].to_vec()), ("four points", pts.clone()),
                              ("a repeated pair", vec![pts[0].clone(), pts[1].clone(), pts[0].clone(), pts[1].clone()])] {
                let got = mask_rank_ok(&p, &words);
                assert_eq!(got, mask_rank_gauss_jordan(&p, &words), "{blocks} blocks: {name}");
                assert_eq!(got.1, name != "a repeated pair", "{blocks} blocks: {name}");
            }
        }
    }
}
