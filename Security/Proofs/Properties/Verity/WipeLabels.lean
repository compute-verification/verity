import Proofs.Properties.Verity.Wipe
import Definitions.Accounting.Assumptions.WipeLabels

/-!
# The wipe's target from an accepted session, on the deployed labels

`Vocab.wipeSound_of_session` derives the wipe's target (`Vocab.WipeSound k`) through `Vocab.wipeSound_of_survivor` from a
`verity/wipe/v0` session that `check` accepts certified, with the labels resting on their own premise
(`Accounting.Assumptions.WipeLabelsPrf`):

* **The wipe is a session's** (`Vocab.WipeSession`): at each `θ F`, the wipe that opens the period is a device
  strategy's for the session's commitment (`Accounting.Wipe.Strategy.wipe`), its verdict the event the auditor's
  devices observe and its labels the deployed ones under the compiled SHA-512 (`Accounting.Wipe.deployedLabels` at
  `sha512Msg`), read through a bijection of its seeds with the challenger's 256-bit key and one of its spot reads with
  the commitment's (`Verity.Wipe.comap`).
* **The session's survivor bound is the wipe's** (`Vocab.WipeSession.survivorBound`): the accepted session's bound
  (`MemoryChallenger.Properties.SecurityProofs.wipeSurvivorBound_atLeast`) at the composite's threshold carries over
  through the bijections (`Verity.Wipe.SurvivorBound.of_comap`).
* **The labels' premise bounds their advantage** (`Vocab.WipeSession.epsilon_le`): `WipeLabelsPrf` at the
  distinguisher the run's moves make (`Vocab.WipeSession.distinguisher`) and the strategy's SHA-512 calls
  (`Vocab.cost`) bounds the labels' PRF advantage in that bound by its `ε`, so the bound at the run's moves is at most
  `eps + ε + δ_net`, and within the published `δ` where that sum is.

The run's coins at the boundary (`Verity.Wipe.OfRun.Preserves`), the refinement (`Vocab.WipeRefines`) and the error
beyond `δ` (`Vocab.WipeBreakBound`) are `wipeSound_of_survivor`'s.

**One device strategy per `θ F`.** `Vocab.wipe θ f` takes no prover strategy `σ`, so the session at `θ F` fixes one
device strategy `τ` for every `σ` the adversary plays there: the target this file derives is for that fixed strategy,
not for one per `σ`. It follows `Vocab.wipe` when that takes `σ` (lean's parameter in `Security/Properties/Verity/Game.lean`).
-/

namespace Verity

open Accounting Core
open scoped ENNReal

namespace Wipe

variable {ι : Type} {d : Accounting.Device ι} {C : Verity.Circuit} {B : ℕ} {s : Verity.Blocks C B} {b : Fin B}
  {Ω Ch Ω' Ch' : Type}

/-- **A wipe read through bijections**: `w` with its seed `ω` read as `e₁ ω` and its spot reads `c` as `e₂ c`. The
fill, the egress, the carried state and the device's states are `w`'s. -/
def comap (w : Verity.Wipe d s b Ω' Ch') (e₁ : Ω ≃ Ω') (e₂ : Ch ≃ Ch') : Verity.Wipe d s b Ω Ch where
  fill := w.fill
  fill_le := w.fill_le
  egress := w.egress
  carry := w.carry
  filled σ ω := w.filled σ (e₁ ω)
  accepts σ ω c := w.accepts σ (e₁ ω) (e₂ c)
  after σ ξ c := w.after σ ξ (e₂ c)

/-- **A survivor bound carries over through bijections**: where `w` is `w'` read through `e₁` and `e₂`, and `w'` meets
its survivor bound at `foreignBits` and `ε'`, `w` meets it at `foreignBits` and `ε'` at the moves read back: `keep` on
`w'`'s egress, which is `w`'s, and on the spot reads through `e₂`. -/
theorem SurvivorBound.of_comap [Fintype Ω] [Nonempty Ω] [Fintype Ch] [Nonempty Ch] [Fintype Ω'] [Nonempty Ω']
    [Fintype Ch'] [Nonempty Ch'] {w : Verity.Wipe d s b Ω Ch} {w' : Verity.Wipe d s b Ω' Ch'} (e₁ : Ω ≃ Ω')
    (e₂ : Ch ≃ Ch') (h : w = w'.comap e₁ e₂) {foreignBits : ℕ}
    {ε' : (S_φ : ℕ) → ((Fin S_φ → Bool) → Device.State d) →
      (Device.State d → (Fin w'.egress → Bool) → Ch' → Fin S_φ → Bool) → ℝ≥0∞}
    (hw : w'.SurvivorBound foreignBits ε') :
    w.SurvivorBound foreignBits fun S_φ hold keep =>
      ε' S_φ hold fun σ ξ c => keep σ (ξ ∘ Fin.cast (congrArg Verity.Wipe.egress h)) (e₂.symm c) := by
  subst h
  intro S_φ hS hold keep
  refine le_trans (le_of_eq ?_) (hw S_φ hS hold _)
  rw [MemoryChallenger.Properties.SecurityProofs.value_draw_pr, MemoryChallenger.Properties.SecurityProofs.value_draw_pr]
  exact MemoryChallenger.Properties.SecurityProofs.pr_congr_equiv ((Equiv.refl _).prodCongr (e₁.prodCongr e₂))
    fun x => by simp [comap]; exact fun _ => Iff.rfl

end Wipe

namespace Vocab

variable {ι : Type} {d : Device ι} {α : Type} [Fintype α] (V : Vocab d α)

/-- **The wipe that opens the period is an accepted session's, on the deployed labels.** At `θ F`: the bytes `bytes`
decode to the commitment `c`, and `verity/wipe/v0`'s check under its deployed schemes (`MemoryChallenger.SCHEMES`)
accepts the signed session `e` of `c` under the parameters `p` and publishes it certified, with `egress_mib`, `eps`,
`foreign_bits` (`published`) at most the composite's threshold (`Vocab.foreignBits`), and `memory_bits`. The wipe is the
strategy `τ`'s for `c`'s labels and reads (`Accounting.Wipe.Strategy.wipe`), its fill at most `memory_bits`, its
verdict the event the auditor's devices observe at the certifiers' count `count` of the egress and the published
egress, and its labels the deployed ones under the compiled SHA-512 (`Accounting.Wipe.deployedLabels` at `sha512Msg`),
read through the bijection `seed` of its seeds with the challenger's 256-bit key and `spots` of its spot reads with
`c`'s (`Verity.Wipe.comap`). `A`, a set of at most `2^(egress_mib MiB)` egress strings, is where `δ_net` is read.

`τ` is one device strategy for `θ F`, whatever prover strategy `σ` runs there: `Vocab.wipe` takes no `σ`. A session per
`σ` follows when `Vocab.wipe` takes `σ` (lean's parameter). -/
structure WipeSession (θ : ByteArray) (F : Array ByteArray) where
  p : MemoryChallenger.WipeProtocol.Params
  bytes : ByteArray
  c : MemoryChallenger.StreamedWipe.Commitment
  e : MemoryChallenger.WipeProtocol.Evidence
  egressMib : Int
  eps : _root_.Core.Ratio
  published : ℕ
  reads : ℕ
  memoryBits : ℕ
  r : ℕ
  profile : MemoryChallenger.StreamedWipe.Params × Int
  decodes : MemoryChallenger.WipeProtocol.decodeCommitment bytes = .ok c
  certified : MemoryChallenger.WipeProtocol.check MemoryChallenger.SCHEMES p c e =
    .accept ⟨true, egressMib, eps, published, reads, memoryBits, r⟩ profile
  within : published ≤ V.foreignBits θ
  [labels_ne : NeZero c.params.labels]
  τ : Accounting.Wipe.Strategy d (V.schedule (V.circuitFile θ F)) (V.boundary (V.circuitFile θ F)) c.params.labels
    c.params.k
  count : (Fin τ.egress → Bool) → ℕ
  A : Finset (Fin τ.egress → Bool)
  fill_le : τ.fill ≤ memoryBits
  card_le : A.card ≤ 2 ^ (egressMib.toNat * MemoryChallenger.WipeProtocol.UNIT_BITS)
  seed : V.Seed ≃ Pous.Bits 256
  spots : V.Spots ≃ Accounting.Wipe.Reads c.params.labels c.params.k
  realizes : V.wipe θ (V.circuitFile θ F) =
    (τ.wipe (τ.Observed count (egressMib.toNat * MemoryChallenger.WipeProtocol.UNIT_BITS))
      (Accounting.Wipe.deployedLabels sha512Msg c.params.labels)).comap seed spots

attribute [instance] WipeSession.labels_ne

namespace WipeSession

variable {V} {θ : ByteArray} {F : Array ByteArray} (W : V.WipeSession θ F)

/-- The session's labels: the deployed ones of its commitment's `n`, under the compiled SHA-512. -/
def labels : Pous.Bits 256 → Accounting.Wipe.Labels W.c.params.labels :=
  Accounting.Wipe.deployedLabels sha512Msg W.c.params.labels

/-- The verdict the wipe reads: the observed event at the certifiers' count and the published egress. -/
def verdict : Device.State d → Accounting.Wipe.Labels W.c.params.labels →
    Accounting.Wipe.Reads W.c.params.labels W.c.params.k → Prop :=
  W.τ.Observed W.count (W.egressMib.toNat * MemoryChallenger.WipeProtocol.UNIT_BITS)

/-- The wipe that opens the period sends the strategy's egress. -/
theorem egress_eq : (V.wipe θ (V.circuitFile θ F)).egress = W.τ.egress := congrArg Verity.Wipe.egress W.realizes

/-- **A read-back of the wipe that opens the period, as the strategy's**: on the strategy's egress, which is the wipe's,
and the commitment's spot reads through `spots`. -/
noncomputable def keepOf {S : ℕ}
    (keep : Device.State d → (Fin (V.wipe θ (V.circuitFile θ F)).egress → Bool) → V.Spots → Fin S → Bool) :
    Device.State d → (Fin W.τ.egress → Bool) → Accounting.Wipe.Reads W.c.params.labels W.c.params.k → Pous.Bits S :=
  fun σ ξ c => keep σ (ξ ∘ Fin.cast W.egress_eq) (W.spots.symm c)

/-- **The distinguisher the moves make** (`Accounting.Wipe.Strategy.distinguisher`), at the device's state `hold` from
the foreign content and the read-back `keep` of the wipe that opens the period. -/
noncomputable def distinguisher {S : ℕ} (hold : Pous.Bits S → Device.State d)
    (keep : Device.State d → (Fin (V.wipe θ (V.circuitFile θ F)).egress → Bool) → V.Spots → Fin S → Bool) :
    (Pous.Bits S × Accounting.Wipe.Reads W.c.params.labels W.c.params.k) × Accounting.Wipe.Labels W.c.params.labels →
      Prop :=
  W.τ.distinguisher W.verdict W.A hold (W.keepOf keep)

/-- **`δ_net` at the moves** (`Accounting.Wipe.Strategy.deltaNet` at `A`). -/
noncomputable def deltaNet {S : ℕ} (hold : Pous.Bits S → Device.State d) : ℝ≥0∞ :=
  W.τ.deltaNet W.verdict W.labels W.A hold

/-- **The session's survivor bound at the moves**: `eps`, plus the labels' PRF advantage against the distinguisher the
moves make, plus `δ_net`. -/
noncomputable def epsilon (S : ℕ) (hold : (Fin S → Bool) → Device.State d)
    (keep : Device.State d → (Fin (V.wipe θ (V.circuitFile θ F)).egress → Bool) → V.Spots → Fin S → Bool) : ℝ≥0∞ :=
  (W.eps.num : ℝ≥0∞) / W.eps.den + Pous.StreamedWipe.prfAdvantage W.labels (W.distinguisher hold keep) +
    W.deltaNet hold

/-- **The wipe that opens the period meets the session's survivor bound** at the composite's threshold:
`wipeSurvivorBound_atLeast` at the session, through `seed` and `spots` (`Verity.Wipe.SurvivorBound.of_comap`). -/
theorem survivorBound [Fintype V.Seed] [Nonempty V.Seed] [Fintype V.Spots] [Nonempty V.Spots] :
    (V.wipe θ (V.circuitFile θ F)).SurvivorBound (V.foreignBits θ) W.epsilon :=
  Wipe.SurvivorBound.of_comap W.seed W.spots W.realizes
    (MemoryChallenger.Properties.SecurityProofs.wipeSurvivorBound_atLeast MemoryChallenger.SCHEMES W.p W.bytes W.c
      W.e W.egressMib W.eps W.published W.reads W.memoryBits W.r W.profile W.decodes W.certified d _ _ W.τ W.labels
      W.count W.A W.fill_le W.card_le W.within)

/-- **The labels' premise bounds the session's survivor bound at the moves**: where `WipeLabelsPrf` holds at the
distinguisher the moves make, at `ε` and `t` calls, the bound is at most `eps + ε t n + δ_net`. -/
theorem epsilon_le {ε : ℕ → ℕ → ℝ≥0∞} {t S : ℕ} {hold : (Fin S → Bool) → Device.State d}
    {keep : Device.State d → (Fin (V.wipe θ (V.circuitFile θ F)).egress → Bool) → V.Spots → Fin S → Bool}
    (h : Accounting.Assumptions.WipeLabelsPrf sha512Msg ε (W.distinguisher hold keep) t) :
    W.epsilon S hold keep ≤ (W.eps.num : ℝ≥0∞) / W.eps.den + ε t W.c.params.labels + W.deltaNet hold :=
  add_le_add (add_le_add le_rfl (tsub_le_iff_left.mpr h)) le_rfl

end WipeSession

section

variable {k : Verity.Registration.Composite}

/-- **The wipe's target from accepted sessions on the deployed labels** (`wipeSound_of_survivor` at
`WipeSession.epsilon`). At every `θ F` the wipe that opens the period is an accepted, certified session's (`W`), and at
every `θ F c σ`: the run's coins at the boundary are the wipe's (`hR`), the component refines the run's wipe verdict
(`hA`), the labels' premise holds at the distinguisher the run's moves make, at `ε` and the strategy's SHA-512 calls
(`hprf`), `eps`, `ε` there and `δ_net` at the run's moves add up to at most the published `δ` (`hδ`), and the error
beyond `δ` answers for the rest of the event where the finder meets `FinderCR` (`hb`).

The wipe target is for one device strategy per `θ F` (`W θ F`'s `τ`), not one per prover strategy `σ`: `Vocab.wipe`
takes no `σ`, so every `σ` at `θ F` meets the same wipe. It follows lean's `σ` parameter of `Vocab.wipe` when that
lands. -/
theorem wipeSound_of_session [Fintype V.Seed] [Nonempty V.Seed] [Fintype V.Spots] [Nonempty V.Spots]
    (W : ∀ θ F, V.WipeSession θ F) (ε : ℕ → ℕ → ℝ≥0∞)
    (pos : (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → Game.Strategy (V.game θ F c) →
      ℕ → ℕ)
    (brk : (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → Game.Strategy (V.game θ F c) →
      Exchange → Prop)
    (hR : ∀ θ F c σ, (V.wipeRun θ F c σ).Preserves σ)
    (hA : ∀ θ F c σ, V.WipeRefines k θ F c σ (pos θ F c σ) (brk θ F c σ))
    (hprf : ∀ θ F c σ, Accounting.Assumptions.WipeLabelsPrf sha512Msg ε
      ((W θ F).distinguisher (V.wipeRun θ F c σ).hold (V.wipeRun θ F c σ).keep) (V.cost θ F c σ))
    (hδ : ∀ θ F c σ, ((W θ F).eps.num : ℝ≥0∞) / (W θ F).eps.den + ε (V.cost θ F c σ) (W θ F).c.params.labels +
      (W θ F).deltaNet (V.wipeRun θ F c σ).hold ≤ V.delta θ .wipe (V.cost θ F c σ))
    (hb : ∀ θ F c σ, V.FinderCR .wipe θ F c σ → V.WipeBreakBound k θ F c σ (pos θ F c σ) (brk θ F c σ)) :
    V.WipeSound k :=
  V.wipeSound_of_survivor pos brk (fun θ F => (W θ F).epsilon) (fun θ F => (W θ F).survivorBound) hR hA
    (fun θ F c σ => ((W θ F).epsilon_le (hprf θ F c σ)).trans (hδ θ F c σ)) hb

end

end Vocab

end Verity
