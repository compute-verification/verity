import MemoryChallenger.WipeProtocol
import Definitions.Accounting.Wipe

/-!
# An accepted wipe meets the survivor bound it publishes

For a session it accepts and certifies, `MemoryChallenger.WipeProtocol.check` (`verity/wipe/v0`, whose verdict is
`StreamedWipe.verdict`) publishes `foreign_bits = 512·(⌈n/r⌉ − 1) + n + ⌊log₂ k⌋ + 130 + R + E` at `eps = 2^-128`
(`certify`): `n` labels, `k` timed reads, `r = ratio k`, `R` the declared memory past the fill, `E` the network
certifiers' count of the egress, `egress_mib` whole MiB. `WipeSurvivorBound` says that every prover strategy `τ` of the
device model (`Accounting.Wipe.Strategy`), with the labels `prf K` of the challenger's key `K`, meets the survivor bound
(`Verity.Wipe.SurvivorBound`) at `foreign_bits` and

  `eps + prfAdvantage prf (τ.distinguisher …) + τ.deltaNet …`,

on the event the auditor's devices observe (`Accounting.Wipe.Strategy.Observed`): every spot read answered with its
label, as the memory challenger signs it, and the certifiers' count of the egress at most `E`. No term is zero:

* the labels' PRF advantage (`Pous.StreamedWipe.prfAdvantage`) against the distinguisher the adversary's moves make
  (`Accounting.Wipe.Strategy.distinguisher`), SHA-512's under `prf/sha-512`;
* `δ_net` (`Accounting.Wipe.Strategy.deltaNet`), the probability that the observed event holds while the egress is
  outside a set `A` of at most `2^E` strings, which the network certifier's guarantee bounds
  (`NetTiming.Guarantees.SampledLeakAccepted`).

The strategy is arbitrary, its fill any bits of the store up to `memory_bits`. The state the fill leaves out
(`Verity.Wipe.unfilled`) is counted beside `foreign_bits`, as `SurvivorBound` counts it. The commitment is the one its
bytes decode to (`WipeProtocol.decodeCommitment`, the protocol's decoder), whose `Params.check` refuses fewer reads than
a batch (`1 ≤ batch ≤ k`): `check` alone accepts a commitment of no reads, where `ratio 0 = 1` doesn't meet the sampling
condition and the bound fails. Proved from `Pous.Guarantees.StreamedWipeMeetsReads` at `r = ratio k`
(`Proofs.Properties.Pous.Wipe`).

* `WipeSurvivorBoundHonest`: an honest session's parameters of 1 to `MAX_READS` reads with no egress are certified
  at the published pair, and the honest strategy (`honest`: the labels written into the store, each read answered from
  it) meets the observed event on every run, with its fill the `n · 512` bits of the labels and its egress in a set of
  `2^0` strings.
* `WipeSurvivorBoundNontrivial`: at one segment (`segmentParams` on `segmentDevice`), a strategy that keeps the whole
  store (`cheat`) fails the bound under a verdict that accepts every run, for every PRF.
-/

namespace MemoryChallenger.Properties

open Verity
open scoped ENNReal

/-- **An accepted, certified wipe meets the survivor bound at the pair it publishes.** When the bytes `bytes` decode to
the commitment `c` and `check` accepts the signed session `e` of `c` and publishes it certified, with `egress_mib`,
`eps`, `foreign_bits` and `memory_bits`, every strategy `τ` of a device for `c`'s labels and reads whose fill is at most
`memory_bits`, with the labels `prf` of the challenger's key, every count `count` of the egress by the network
certifiers and every set `A` of at most `2^(egress_mib MiB)` egress strings, meets `SurvivorBound foreign_bits` on the
observed event at `eps` plus the PRF's advantage against the adversary's distinguisher plus `δ_net` at `A`. -/
def WipeSurvivorBound : Prop :=
  ∀ (schemes : Schemes) (p : WipeProtocol.Params) (bytes : ByteArray) (c : StreamedWipe.Commitment)
    (e : WipeProtocol.Evidence) (egressMib : Int) (eps : Core.Ratio) (foreignBits k memoryBits r : Nat)
    (profile : StreamedWipe.Params × Int),
    WipeProtocol.decodeCommitment bytes = .ok c →
    WipeProtocol.check schemes p c e = .accept ⟨true, egressMib, eps, foreignBits, k, memoryBits, r⟩ profile →
    ∀ {ι : Type} (d : Accounting.Device ι) {C : Circuit} {B : ℕ} (s : Blocks C B) (b : Fin B)
      [NeZero c.params.labels] (τ : Accounting.Wipe.Strategy d s b c.params.labels c.params.k) {κ : ℕ}
      (prf : Pous.Bits κ → Accounting.Wipe.Labels c.params.labels) (count : (Fin τ.egress → Bool) → ℕ)
      (A : Finset (Fin τ.egress → Bool)),
      τ.fill ≤ memoryBits → A.card ≤ 2 ^ (egressMib.toNat * WipeProtocol.UNIT_BITS) →
      (τ.wipe (τ.Observed count (egressMib.toNat * WipeProtocol.UNIT_BITS)) prf).SurvivorBound foreignBits
        fun _ hold keep => (eps.num : ℝ≥0∞) / eps.den +
          Pous.StreamedWipe.prfAdvantage prf
            (τ.distinguisher (τ.Observed count (egressMib.toNat * WipeProtocol.UNIT_BITS)) A hold keep) +
          τ.deltaNet (τ.Observed count (egressMib.toNat * WipeProtocol.UNIT_BITS)) prf A hold

/-! ## The honest strategy and a cheating one -/

section Strategies

variable {ι : Type} {d : Accounting.Device ι} {n : ℕ}

/-- The honest fill: label `i`'s 512 bits at store bits `512 i` to `512 i + 511`, the rest of the state as it was. -/
def honestFill (σ : Device.State d) (L : Accounting.Wipe.Labels n) : Device.State d :=
  ⟨σ.chip, fun a => if ha : a.val < n * 512 then L ⟨a.val / 512, by omega⟩ ⟨a.val % 512, by omega⟩ else σ.store a⟩

/-- The honest read of label `i`: store bits `512 i` to `512 i + 511`. -/
def honestRead (h : n * 512 ≤ d.mem) (σ : Device.State d) (i : Fin n) : Pous.Bits 512 :=
  fun t => σ.store ⟨i.val * 512 + t.val, by have := i.isLt; have := t.isLt; omega⟩

/-- **The honest strategy**: it writes the labels into the first `n · 512` bits of the store, sends nothing, answers
each read from the store and leaves the filled state to the block. -/
def honest (d : Accounting.Device ι) {C : Circuit} {B : ℕ} (s : Blocks C B) (b : Fin B) (n k : ℕ)
    (h : n * 512 ≤ d.mem) (carry : Device.State d → s.Carried b) : Accounting.Wipe.Strategy d s b n k where
  fill := n * 512
  fill_le := h
  egress := 0
  carry := carry
  filled σ L := (honestFill σ L, fun i => i.elim0)
  answer σ' _ _ i := honestRead h σ' i
  after σ' _ _ := σ'

/-- **A cheating strategy**: it counts the whole store as filled but writes nothing, keeping the device's state, sends
nothing and answers every read with zeros. -/
def cheat (d : Accounting.Device ι) {C : Circuit} {B : ℕ} (s : Blocks C B) (b : Fin B) (n k : ℕ)
    (carry : Device.State d → s.Carried b) : Accounting.Wipe.Strategy d s b n k where
  fill := d.mem
  fill_le := le_rfl
  egress := 0
  carry := carry
  filled σ _ := (σ, fun i => i.elim0)
  answer _ _ _ _ := fun _ => false
  after σ' _ _ := σ'

end Strategies

instance instNeZeroNatHPowOfNat_properties : NeZero (2 ^ 19 : ℕ) := ⟨(Nat.two_pow_pos 19).ne'⟩

/-- One segment's device: no on-chip state, a store of `2^19` labels, `2^28` bits, and no side channel. -/
def segmentDevice : Accounting.Device Unit := ⟨0, 2 ^ 19 * 512, fun _ => 0, 0, fun _ => 0⟩

/-- One segment's session parameters: `2^19` labels, `2152` reads, one a batch, as `WipeProtocol`'s own check of
`certify` takes them. -/
def segmentParams : StreamedWipe.Params := ⟨2 ^ 19, 2152, 1, 1000, #[("gpu", 2 ^ 25)]⟩

/-- **The honest lemma of `WipeSurvivorBound`: its hypotheses hold for an honest session.** The parameters of a session
of 1 to `MAX_READS` reads, with no egress, are certified at `eps = 2^-128` and `foreign_bits` the threshold
`foreignBits` at `r = ratio k`, with their reads, `memory_bits` and ratio. The honest strategy on any device whose store
holds the fill has its fill the labels' `n · 512` bits, its egress in a set of `2^0` strings, and meets the observed
event on every run at the count of no egress. -/
def WipeSurvivorBoundHonest : Prop :=
  (∀ q : StreamedWipe.Params, 1 ≤ q.k → q.k ≤ WipeProtocol.MAX_READS →
      WipeProtocol.certify q 0 = ⟨true, 0, WipeProtocol.EPS, WipeProtocol.foreignBits q (WipeProtocol.ratio q.k), q.k,
        WipeProtocol.memoryBits q, WipeProtocol.ratio q.k⟩) ∧
    ∀ {ι : Type} (d : Accounting.Device ι) {C : Circuit} {B : ℕ} (s : Blocks C B) (b : Fin B) (n k : ℕ)
      (h : n * 512 ≤ d.mem) (carry : Device.State d → s.Carried b),
      (honest d s b n k h carry).fill = n * 512 ∧
        (Finset.univ : Finset (Fin (honest d s b n k h carry).egress → Bool)).card ≤ 2 ^ 0 ∧
        ∀ σ L c, (honest d s b n k h carry).Observed (fun _ => 0) 0 σ L c

/-- **The nontrivial lemma of `WipeSurvivorBound`: its conclusion fails under a verdict that accepts every run.** At one
segment, the cheating strategy, with every PRF, every egress string in `A` and a verdict that accepts every run, fails
`SurvivorBound` at the pair `certify` publishes for the segment with no egress: it keeps the published `foreign_bits`
(11,709,069 bits, at `r = 24`) of foreign content in the store with probability 1, while the bound is `2^-128` plus a
PRF advantage and a `δ_net` that are 0. -/
def WipeSurvivorBoundNontrivial : Prop :=
  ∀ {C : Circuit} {B : ℕ} (s : Blocks C B) (b : Fin B) (carry : Device.State segmentDevice → s.Carried b) {κ : ℕ}
    (prf : Pous.Bits κ → Accounting.Wipe.Labels (2 ^ 19)),
    ¬ ((cheat segmentDevice s b (2 ^ 19) 2152 carry).wipe (fun _ _ _ => True) prf).SurvivorBound
        (WipeProtocol.certify segmentParams 0).foreignBits fun _ hold keep =>
          ((WipeProtocol.certify segmentParams 0).eps.num : ℝ≥0∞) / (WipeProtocol.certify segmentParams 0).eps.den +
            Pous.StreamedWipe.prfAdvantage prf
              ((cheat segmentDevice s b (2 ^ 19) 2152 carry).distinguisher (fun _ _ _ => True)
                Finset.univ hold keep) +
            (cheat segmentDevice s b (2 ^ 19) 2152 carry).deltaNet (fun _ _ _ => True) prf
              Finset.univ hold

end MemoryChallenger.Properties
