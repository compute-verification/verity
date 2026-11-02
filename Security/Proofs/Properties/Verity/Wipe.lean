import Proofs.Properties.Verity.Subprotocols
import Proofs.Properties.Pous.Wipe
import Proofs.Core.Blocks

/-!
# The wipe's target from its survivor bound

`Vocab.WipeSound k` is `Vocab.Target k .wipe`: the probability that the wipe's verdict at composite `k`
(`MemoryChallenger.WipeProtocol.protocol`, `verity/wipe/v0`, at onsite) accepts while the run, read at the wipe's boundary
as the wipe's game (`Vocab.wipeRun`), keeps foreign content past the survivor threshold the composite publishes
(`Verity.Wipe.OfRun.Kept` at `Vocab.foreignBits`) is at most the wipe's error. It follows from its component-level bound
(`Vocab.ComponentBound k .wipe`, `Vocab.sound_of`), and that bound, at each `θ F c σ`, from four things
(`Vocab.wipe_bound`):

* **The survivor bound.** The wipe that opens the period (`Vocab.wipe θ (Vocab.circuitFile θ F)`) meets
  `Verity.Wipe.SurvivorBound` at the published threshold and an `ε` of the adversary's moves. For `verity/wipe/v0` that
  is `MemoryChallenger.Properties.WipeSurvivorBound` at the device's strategy (`eps` plus the labels' PRF advantage plus
  `δ_net`, on the event the auditor's devices observe), at the `foreign_bits` the accepted case publishes, and at any
  threshold above it (`Verity.Wipe.SurvivorBound.mono`, `wipeSurvivorBound_atLeast`).
* **The run's coins at the boundary are the wipe's** (`Verity.Wipe.OfRun.Preserves`, a named assumption): on the run,
  the wipe's verdict holds while foreign content past the threshold survives with probability at most `ε` at the run's
  moves (`Verity.Wipe.OfRun.prob_kept_le`).
* **The verdict at the code** (`Vocab.WipeRefines`): where the wipe's component accepts its own case in what the prover
  sent, every release its check read is the device's at the round `pos` assigns it (`Vocab.Logged`), and the break `brk`
  is absent, the run's wipe verdict holds (`Verity.Wipe.OfRun.accepts`). It is a hypothesis until it is proved of the
  code, as the period audits' refinements are (`Subprotocol.PoubRefines` and its siblings). The rest of the event, a
  release out of place or a break, is answered for by the wipe's error beyond `δ` (`Vocab.WipeBreakBound`, the form of
  `Vocab.Coupling.BreakBound` in the game): a forgery of the device's LMS signature, or a collision.
* **Within `δ`**: `ε` at the run's moves is at most the wipe's published `δ` at the strategy's SHA-512 calls.

Beside them, rule 3's checks at `.wipe` (`Vocab.wipeSound_of_honest`, `Vocab.wipeSound_of_nontrivial`), from the generic
`Vocab.sound_of_honest` and `Vocab.sound_of_nontrivial`.
-/

namespace Verity

open Accounting Core
open scoped ENNReal

/-- **The survivor bound at a higher threshold**: a wipe that meets its survivor bound at `foreignBits` meets it, at the
same `ε`, at every threshold `foreignBits'` at least `foreignBits`, since the bound asks less of fewer `S_φ`. -/
theorem Wipe.SurvivorBound.mono {ι : Type} {d : Accounting.Device ι} {C : Verity.Circuit} {B : ℕ}
    {s : Verity.Blocks C B} {b : Fin B} {Ω Ch : Type} [Fintype Ω] [Nonempty Ω] [Fintype Ch] [Nonempty Ch]
    {w : Verity.Wipe d s b Ω Ch}
    {foreignBits foreignBits' : ℕ}
    {ε : (S_φ : ℕ) → ((Fin S_φ → Bool) → Device.State d) →
      (Device.State d → (Fin w.egress → Bool) → Ch → Fin S_φ → Bool) → ℝ≥0∞}
    (h : foreignBits ≤ foreignBits') (hw : w.SurvivorBound foreignBits ε) : w.SurvivorBound foreignBits' ε :=
  fun S_φ hS => hw S_φ (le_trans (Nat.add_le_add_right h _) hS)

namespace Vocab

variable {ι : Type} {d : Device ι} {α : Type} [Fintype α] (V : Vocab d α)

/-- **The wipe's refinement: its compiled component accepts only where the run's wipe verdict holds.** At every outcome
`o` where the wipe's component at composite `k` (onsite's `MemoryChallenger.WipeProtocol.protocol`) accepts its own case
in what the prover sent, every release its check read is the device's at the round `pos` assigns it (`Logged`), and the
break `brk` is absent, the memory challenger's verdict on the run at the wipe's boundary holds
(`Verity.Wipe.OfRun.accepts` at `wipeRun`). -/
def WipeRefines (k : Verity.Registration.Composite) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Game.Strategy (V.game θ F c))
    (pos : ℕ → ℕ) (brk : Exchange → Prop) : Prop :=
  ∀ o, ComponentAccepts k .wipe o.sent → V.Logged pos .wipe o → ¬ brk o → (V.wipeRun θ F c σ).accepts o

/-- **The wipe's error beyond `δ` answers for the rest of its event**: in the play of `σ`, the probability that the
wipe's component at composite `k` accepts while a release its check read is not the device's at its round (`pos`), or
the break `brk` holds, is at most the probability that the wipe's forger forges under the compiled LMS verifier
(`Vocab.forger`, `lmsVerify`) plus the wipe's collision term at the strategy (`Vocab.Coupling.BreakBound`'s form). -/
def WipeBreakBound (k : Verity.Registration.Composite) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Game.Strategy (V.game θ F c))
    (pos : ℕ → ℕ) (brk : Exchange → Prop) : Prop :=
  Game.prob (fun o => ComponentAccepts k .wipe o.sent ∧ (¬ V.Logged pos .wipe o ∨ brk o)) (V.game θ F c) σ ≤
    ENNReal.ofReal ((V.forger .wipe θ F c σ).Pr lmsVerify) +
      V.loss .wipe θ (ENNReal.ofReal ((V.finder .wipe θ F c σ).Pr sha512Msg))

/-- **The wipe's component-level event is within its error.** At `θ F c` and the strategy `σ`: where the wipe that opens
the period meets its survivor bound at the published threshold and `ε` (`hw`), the run's coins at the wipe's boundary are
the wipe's at `σ` (`hR`), the component refines the run's wipe verdict (`hA`), `ε` at the run's moves is within the
published `δ` at the strategy's SHA-512 calls (`hε`), and the error beyond `δ` answers for the rest of the event where
the finder meets `FinderCR` (`hb`), the probability that the wipe's component at `k` accepts its own case while the
run keeps foreign content past the threshold is at most the wipe's error at `σ`. -/
theorem wipe_bound [Fintype V.Seed] [Nonempty V.Seed] [Fintype V.Spots] [Nonempty V.Spots]
    (k : Verity.Registration.Composite) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Game.Strategy (V.game θ F c)) (pos : ℕ → ℕ)
    (brk : Exchange → Prop)
    {ε : (S_φ : ℕ) → ((Fin S_φ → Bool) → Device.State d) →
      (Device.State d → (Fin (V.wipe θ (V.circuitFile θ F)).egress → Bool) → V.Spots → Fin S_φ → Bool) →
      ℝ≥0∞}
    (hw : (V.wipe θ (V.circuitFile θ F)).SurvivorBound (V.foreignBits θ) ε)
    (hR : (V.wipeRun θ F c σ).Preserves σ)
    (hA : V.WipeRefines k θ F c σ pos brk)
    (hε : ε (V.wipeRun θ F c σ).S_φ (V.wipeRun θ F c σ).hold (V.wipeRun θ F c σ).keep ≤
      V.delta θ .wipe (V.cost θ F c σ))
    (hb : V.FinderCR .wipe θ F c σ → V.WipeBreakBound k θ F c σ pos brk) :
    Game.prob (fun o => ComponentAccepts k .wipe o.sent ∧ ¬ V.Holds k θ F c σ .wipe o) (V.game θ F c) σ ≤
      V.error .wipe θ F c σ := by
  unfold Vocab.error Accounting.condError
  split_ifs with hcr
  · obtain ⟨hcr, hun⟩ := hcr
    have hf : ENNReal.ofReal ((V.forger .wipe θ F c σ).Pr lmsVerify) ≤
        ENNReal.ofReal (V.epsForge θ (V.callsMul θ .wipe * V.cost θ F c σ + V.callsAdd θ .wipe) (V.signatures θ)) :=
      ENNReal.ofReal_le_ofReal (hun (V.forger_calls _ θ F c σ) (V.forger_signed _ θ F c σ))
    show _ ≤ V.delta θ .wipe (V.cost θ F c σ) +
      ENNReal.ofReal (V.epsForge θ (V.callsMul θ .wipe * V.cost θ F c σ + V.callsAdd θ .wipe) (V.signatures θ)) +
      V.loss .wipe θ (ENNReal.ofReal ((V.finder .wipe θ F c σ).Pr sha512Msg))
    rw [add_assoc]
    refine le_trans (Game.prob_mono
      (p := fun o => ComponentAccepts k .wipe o.sent ∧ ¬ V.Holds k θ F c σ .wipe o)
      (q := fun o => ((V.wipeRun θ F c σ).accepts o ∧ (V.wipeRun θ F c σ).Kept (V.foreignBits θ) o) ∨
        (ComponentAccepts k .wipe o.sent ∧ (¬ V.Logged pos .wipe o ∨ brk o))) (fun o ho => ?_) _ σ)
      (le_trans (Game.prob_or_le _ _ _ σ) (add_le_add ?_ ((hb hcr).trans (add_le_add hf le_rfl))))
    · have hk : (V.wipeRun θ F c σ).Kept (V.foreignBits θ) o := by simpa [Vocab.Holds] using ho.2
      by_cases hl : V.Logged pos .wipe o
      · by_cases hbk : brk o
        · exact Or.inr ⟨ho.1, Or.inr hbk⟩
        · exact Or.inl ⟨hA o ho.1 hl hbk, hk⟩
      · exact Or.inr ⟨ho.1, Or.inl hl⟩
    · exact (Wipe.OfRun.prob_kept_le hw (V.wipeRun θ F c σ) σ hR).trans hε
  · exact le_top

section

variable {k : Verity.Registration.Composite}

/-- **The wipe's component-level bound from its survivor bound** (`wipe_bound` at every `θ F c σ`): the wipe that opens
the period meets its survivor bound at the published threshold (`hw`), the run's coins at the boundary are the wipe's at
every strategy (`hR`), the component refines the run's wipe verdict (`hA`), `ε` at the run's moves is within the
published `δ` (`hε`), and the error beyond `δ` answers for the rest of the event (`hb`). -/
theorem componentBound_wipe [Fintype V.Seed] [Nonempty V.Seed] [Fintype V.Spots] [Nonempty V.Spots]
    (pos : (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → Game.Strategy (V.game θ F c) →
      ℕ → ℕ)
    (brk : (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → Game.Strategy (V.game θ F c) →
      Exchange → Prop)
    (ε : (θ : ByteArray) → (F : Array ByteArray) → (S_φ : ℕ) → ((Fin S_φ → Bool) → Device.State d) →
      (Device.State d → (Fin (V.wipe θ (V.circuitFile θ F)).egress → Bool) → V.Spots → Fin S_φ → Bool) →
      ℝ≥0∞)
    (hw : ∀ θ F, (V.wipe θ (V.circuitFile θ F)).SurvivorBound (V.foreignBits θ) (ε θ F))
    (hR : ∀ θ F c σ, (V.wipeRun θ F c σ).Preserves σ)
    (hA : ∀ θ F c σ, V.WipeRefines k θ F c σ (pos θ F c σ) (brk θ F c σ))
    (hε : ∀ θ F c σ, ε θ F (V.wipeRun θ F c σ).S_φ (V.wipeRun θ F c σ).hold (V.wipeRun θ F c σ).keep ≤
      V.delta θ .wipe (V.cost θ F c σ))
    (hb : ∀ θ F c σ, V.FinderCR .wipe θ F c σ → V.WipeBreakBound k θ F c σ (pos θ F c σ) (brk θ F c σ)) :
    V.ComponentBound k .wipe := fun θ F c σ =>
  V.wipe_bound k θ F c σ (pos θ F c σ) (brk θ F c σ) (hw θ F) (hR θ F c σ) (hA θ F c σ) (hε θ F c σ) (hb θ F c σ)

/-- **The wipe's target from its survivor bound** (`componentBound_wipe`, `sound_of`). -/
theorem wipeSound_of_survivor [Fintype V.Seed] [Nonempty V.Seed] [Fintype V.Spots] [Nonempty V.Spots]
    (pos : (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → Game.Strategy (V.game θ F c) →
      ℕ → ℕ)
    (brk : (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → Game.Strategy (V.game θ F c) →
      Exchange → Prop)
    (ε : (θ : ByteArray) → (F : Array ByteArray) → (S_φ : ℕ) → ((Fin S_φ → Bool) → Device.State d) →
      (Device.State d → (Fin (V.wipe θ (V.circuitFile θ F)).egress → Bool) → V.Spots → Fin S_φ → Bool) →
      ℝ≥0∞)
    (hw : ∀ θ F, (V.wipe θ (V.circuitFile θ F)).SurvivorBound (V.foreignBits θ) (ε θ F))
    (hR : ∀ θ F c σ, (V.wipeRun θ F c σ).Preserves σ)
    (hA : ∀ θ F c σ, V.WipeRefines k θ F c σ (pos θ F c σ) (brk θ F c σ))
    (hε : ∀ θ F c σ, ε θ F (V.wipeRun θ F c σ).S_φ (V.wipeRun θ F c σ).hold (V.wipeRun θ F c σ).keep ≤
      V.delta θ .wipe (V.cost θ F c σ))
    (hb : ∀ θ F c σ, V.FinderCR .wipe θ F c σ → V.WipeBreakBound k θ F c σ (pos θ F c σ) (brk θ F c σ)) :
    V.WipeSound k :=
  V.sound_of k .wipe (V.componentBound_wipe pos brk ε hw hR hA hε hb)

/-- **The wipe's target from its component-level bound** -/
theorem wipeSound_of (h : V.ComponentBound k .wipe) : V.WipeSound k := V.sound_of k .wipe h

/-- **At the honest deployment, the wipe**: `sound_of_honest` at `.wipe`. -/
theorem wipeSound_of_honest (h : V.HonestHolds k .wipe) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (hreg : V.Registered θ F) :
    Game.prob (fun o => ComponentAccepts k .wipe o.sent ∧ ¬ V.Holds k θ F c (V.honest θ F c) .wipe o)
        (V.game θ F c) (V.honest θ F c) ≤ V.error .wipe θ F c (V.honest θ F c) ∧
      Game.prob (fun o => V.Verdict k .wipe o ∧ ¬ V.Holds k θ F c (V.honest θ F c) .wipe o) (V.game θ F c)
        (V.honest θ F c) = 0 :=
  V.sound_of_honest k .wipe h θ F c hreg

/-- **The wipe's target fails for a verifier that accepts everything**: `sound_of_nontrivial` at `.wipe`. -/
theorem wipeSound_of_nontrivial (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Game.Strategy (V.game θ F c))
    (hfail : ∀ o, ¬ V.Holds k θ F c σ .wipe o) (hlt : V.error .wipe θ F c σ < 1) :
    ¬ V.TargetFor (fun _ => True) k .wipe :=
  V.sound_of_nontrivial k .wipe θ F c σ hfail hlt

end

end Vocab

end Verity

namespace MemoryChallenger.Properties.SecurityProofs

open Verity
open scoped ENNReal

/-- **An accepted, certified `verity/wipe/v0` session's survivor bound at any threshold at least the one it
publishes**: `wipeSurvivorBound` at the published `foreign_bits`, raised to `foreignBits'` by
`Verity.Wipe.SurvivorBound.mono`. A composite whose published threshold is at least every accepted case's (its
`Vocab.foreignBits`) reads the survivor bound there. -/
theorem wipeSurvivorBound_atLeast (schemes : Schemes) (p : WipeProtocol.Params) (bytes : ByteArray)
    (c : StreamedWipe.Commitment) (e : WipeProtocol.Evidence) (egressMib : Int) (eps : Core.Ratio)
    (foreignBits k memoryBits r : Nat) (profile : StreamedWipe.Params × Int)
    (hdec : WipeProtocol.decodeCommitment bytes = .ok c)
    (h : WipeProtocol.check schemes p c e = .accept ⟨true, egressMib, eps, foreignBits, k, memoryBits, r⟩ profile)
    {ι : Type} (d : Accounting.Device ι) {C : Verity.Circuit} {B : ℕ} (s : Verity.Blocks C B) (b : Fin B)
    [NeZero c.params.labels] (τ : Accounting.Wipe.Strategy d s b c.params.labels c.params.k) {κ : ℕ}
    (prf : Pous.Bits κ → Accounting.Wipe.Labels c.params.labels) (count : (Fin τ.egress → Bool) → ℕ)
    (A : Finset (Fin τ.egress → Bool)) (hfill : τ.fill ≤ memoryBits)
    (hA : A.card ≤ 2 ^ (egressMib.toNat * WipeProtocol.UNIT_BITS)) {foreignBits' : ℕ}
    (hle : foreignBits ≤ foreignBits') :
    (τ.wipe (τ.Observed count (egressMib.toNat * WipeProtocol.UNIT_BITS)) prf).SurvivorBound foreignBits'
      fun _ hold keep => (eps.num : ℝ≥0∞) / eps.den +
        Pous.StreamedWipe.prfAdvantage prf
          (τ.distinguisher (τ.Observed count (egressMib.toNat * WipeProtocol.UNIT_BITS)) A hold keep) +
        τ.deltaNet (τ.Observed count (egressMib.toNat * WipeProtocol.UNIT_BITS)) prf A hold :=
  Wipe.SurvivorBound.mono hle
    (wipeSurvivorBound schemes p bytes c e egressMib eps foreignBits k memoryBits r profile hdec h d s b τ prf count A
      hfill hA)

end MemoryChallenger.Properties.SecurityProofs
