//! Network emulation for the latency axis (lane latency, 2026-09-22).
//!
//! The campaign's proof model is interactive (`backends/ZK_CONSTRUCTION.md` 3.3: the verifier's coins travel over
//! the network), so every verifier message on the critical path is one round trip.  In a Fiat-Shamir transcript a
//! verifier message is exactly a `sample` that follows an `observe`: the prover sent something (observed) and the
//! next coin depends on it.  [`RttChallenger`] wraps any Plonky3 challenger, counts those observe -> sample
//! boundaries (`rounds.sequential_depth`, measured rather than derived) and, when an RTT is set (`--rtt-ms`),
//! sleeps one RTT there -- the prover waiting for the coin.  Coins sampled back to back without a prover message
//! in between (alpha's extension coordinates, the query indices) cost no extra trip.  The proof-of-work grind
//! (`GrindingChallenger::grind`) is one prover message (the witness) followed by the query coins: one trip; the
//! grind's search clones never sleep or count.  The protocol and the bytes are unchanged: the hook is a delay.
//!
//! State is process-global so `Backend::challenger()` needs no parameter: [`begin`] before the prover, [`end`]
//! after it (the verifier's challenger, built later, neither sleeps nor counts).

use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use p3_challenger::{CanObserve, CanSample, CanSampleBits, FieldChallenger, GrindingChallenger};
use p3_field::Field;

static ACTIVE: AtomicBool = AtomicBool::new(false);
static RTT_NANOS: AtomicU64 = AtomicU64::new(0);
static TRIPS: AtomicU64 = AtomicU64::new(0);
static SLEPT_NANOS: AtomicU64 = AtomicU64::new(0);

/// Start counting critical-path verifier messages on every challenger built from now on; sleep `rtt_ms` at each.
pub fn begin(rtt_ms: f64) {
    RTT_NANOS.store((rtt_ms * 1e6).round() as u64, Ordering::SeqCst);
    TRIPS.store(0, Ordering::SeqCst);
    SLEPT_NANOS.store(0, Ordering::SeqCst);
    ACTIVE.store(true, Ordering::SeqCst);
}

/// Stop; returns `(trips, slept_seconds)`.
pub fn end() -> (u64, f64) {
    ACTIVE.store(false, Ordering::SeqCst);
    (TRIPS.load(Ordering::SeqCst), SLEPT_NANOS.load(Ordering::SeqCst) as f64 * 1e-9)
}

#[derive(Debug)]
pub struct RttChallenger<Inner> {
    inner: Inner,
    /// A prover message was observed since the last coin: the next coin is a new verifier message.
    dirty: bool,
    /// Counts and sleeps only on the challenger the prover drives (never on grind's search clones).
    active: bool,
}

impl<Inner> RttChallenger<Inner> {
    pub fn new(inner: Inner) -> Self {
        Self { inner, dirty: false, active: ACTIVE.load(Ordering::SeqCst) }
    }
    fn trip(&mut self) {
        if !self.dirty {
            return;
        }
        self.dirty = false;
        if !self.active {
            return;
        }
        TRIPS.fetch_add(1, Ordering::SeqCst);
        let ns = RTT_NANOS.load(Ordering::SeqCst);
        if ns > 0 {
            let t = Instant::now();
            std::thread::sleep(Duration::from_nanos(ns));
            SLEPT_NANOS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::SeqCst);
        }
    }
}

impl<Inner: Clone> Clone for RttChallenger<Inner> {
    fn clone(&self) -> Self {
        Self { inner: self.inner.clone(), dirty: self.dirty, active: false }
    }
}

impl<T, Inner: CanObserve<T>> CanObserve<T> for RttChallenger<Inner> {
    fn observe(&mut self, value: T) {
        self.dirty = true;
        self.inner.observe(value);
    }
    fn observe_slice(&mut self, values: &[T])
    where
        T: Clone,
    {
        self.dirty = true;
        self.inner.observe_slice(values);
    }
}

impl<T, Inner: CanSample<T>> CanSample<T> for RttChallenger<Inner> {
    fn sample(&mut self) -> T {
        self.trip();
        self.inner.sample()
    }
}

impl<Inner: CanSampleBits<usize>> CanSampleBits<usize> for RttChallenger<Inner> {
    fn sample_bits(&mut self, bits: usize) -> usize {
        self.trip();
        self.inner.sample_bits(bits)
    }
}

impl<F: Field, Inner: FieldChallenger<F>> FieldChallenger<F> for RttChallenger<Inner> {}

impl<Inner: GrindingChallenger> GrindingChallenger for RttChallenger<Inner> {
    type Witness = Inner::Witness;
    fn grind(&mut self, bits: usize) -> Self::Witness {
        // the witness is a prover message and the coins that follow it depend on it: one trip, then the inner
        // grind (whose search clones the inner challenger; they never reach this wrapper)
        self.dirty = true;
        self.trip();
        self.inner.grind(bits)
    }
}
