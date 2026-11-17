module

public import Properties.Pous.Wipe
import Mathlib.Algebra.Order.Module.Field
import Mathlib.Basic.ENNReal.Real

/-! The survivor bound's honest and nontrivial lemmas, which no record pins: neither is a witness, because each
quantifies over every session's parameters, device and PRF. -/

@[expose] public section

namespace MemoryChallenger.Properties

open Verity
open scoped ENNReal

/-- **The honest lemma of `WipeSurvivorBound`**: an honest session of 1 to `MAX_READS` reads with no egress is certified
by `certify`, and the honest strategy meets the observed event on every run. Its hypothesis on `check` holds of no
session, because `check` publishes `certify` only at `EGRESS_MAX_UNITS` (`uncounted`). -/
def WipeSurvivorBoundHonest : Prop :=
  (∀ (q : StreamedWipe.Params) [NeZero q.k], q.k ≤ WipeProtocol.MAX_READS →
      WipeProtocol.certify q 0 = ⟨true, 0, WipeProtocol.EPS, WipeProtocol.foreignBits q (WipeProtocol.ratio q.k), q.k,
        WipeProtocol.memoryBits q, WipeProtocol.ratio q.k⟩) ∧
    ∀ {ι : Type} (d : ResourceAccounting.Device ι) {C : Circuit} {B : ℕ} (s : Blocks C B) (b : Fin B) (n k : ℕ)
      (h : n * 512 ≤ d.mem) (carry : Device.State d → s.Carried b),
      (honest d s b n k h carry).fill = n * 512 ∧
        (Finset.univ : Finset (Fin (honest d s b n k h carry).egress → Bool)).card ≤ 2 ^ 0 ∧
        ∀ σ L c, (honest d s b n k h carry).Observed (fun _ => 0) 0 σ L c

/-- **The nontrivial lemma of `WipeSurvivorBound`**: at one segment, under a verdict that accepts every run, the
cheating strategy keeps the published `foreign_bits` of foreign content with probability 1, against a bound of
`2^-128`. -/
def WipeSurvivorBoundNontrivial : Prop :=
  ∀ {C : Circuit} {B : ℕ} (s : Blocks C B) (b : Fin B) (carry : Device.State segmentDevice → s.Carried b) {κ : ℕ}
    (prf : Pous.Bits κ → ResourceAccounting.Wipe.Labels (2 ^ 19)),
    ¬ ((cheat segmentDevice s b (2 ^ 19) 2152 carry).wipe (fun _ _ _ => True) prf).SurvivorBound
        (WipeProtocol.certify segmentParams 0).foreignBits fun _ hold keep =>
          ((WipeProtocol.certify segmentParams 0).eps.num : ℝ≥0∞) / (WipeProtocol.certify segmentParams 0).eps.den +
            Pous.StreamedWipe.prfAdvantage prf
              ((cheat segmentDevice s b (2 ^ 19) 2152 carry).distinguisher (fun _ _ _ => True)
                Finset.univ hold keep) +
            (cheat segmentDevice s b (2 ^ 19) 2152 carry).deltaNet (fun _ _ _ => True) prf
              Finset.univ hold

end MemoryChallenger.Properties
