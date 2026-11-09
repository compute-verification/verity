module

public import Properties.Pous.Wipe
import Mathlib.Algebra.Order.Module.Field
import Mathlib.Basic.ENNReal.Real

/-!
# The survivor bound's honest and nontrivial lemmas

The two lemmas beside `Properties.Pous.Wipe`'s `WipeSurvivorBound`: its hypotheses hold for an honest session, and its
conclusion fails under a verdict that accepts every run. Neither is a property, nor a witness of one: a witness
constructs its instance, and each quantifies over every session's parameters, device and PRF. Each is a `def G : Prop`
under the properties' namespace, proved by the `SecurityProofs` theorem of its name in lower camel case, and audited
with the proofs: no record pins it.

* `WipeSurvivorBoundHonest`: an honest session's parameters of at most `MAX_READS` reads with no egress are certified
  at the published pair, and the honest strategy (`honest`: the labels written into the store, each read answered from
  it) meets the observed event on every run, with its fill the `n · 512` bits of the labels and its egress in a set of
  `2^0` strings.
* `WipeSurvivorBoundNontrivial`: at one segment (`segmentParams` on `segmentDevice`), a strategy that keeps the whole
  store (`cheat`) fails the bound under a verdict that accepts every run, for every PRF.
-/

@[expose] public section

namespace MemoryChallenger.Properties

open Verity
open scoped ENNReal

/-- **The honest lemma of `WipeSurvivorBound`: its hypotheses hold for an honest session.** The parameters of a session
of at most `MAX_READS` reads, with no egress, are certified at `eps = 2^-128` and `foreign_bits` the threshold
`foreignBits` at `r = ratio k`, with their reads, `memory_bits` and ratio. The honest strategy on any device whose store
holds the fill has its fill the labels' `n · 512` bits, its egress in a set of `2^0` strings, and meets the observed
event on every run at the count of no egress. -/
def WipeSurvivorBoundHonest : Prop :=
  (∀ q : StreamedWipe.Params, q.k ≤ WipeProtocol.MAX_READS →
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
