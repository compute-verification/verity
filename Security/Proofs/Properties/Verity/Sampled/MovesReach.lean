module

public import Proofs.Properties.Verity.Sampled.RoundsRead
public import Proofs.Properties.Verity.Sampled.RecordDraw

/-!
# `MovesReach`, clause by clause

On the joint law's support off the held-off events (`ReachesOff`), each outcome is the writer's at the widened sample's
draw, the decoded registration and a live session's outcome (`reachesOff_out`). What v2's files decode to in the play on
the tape beside it, clause by clause of `MovesOf`:

* **The play from the first stop** (`ReachesOff.out_reaches`): the play beside an outcome is one the first stop
  reaches (`firstStop_playReaches`), at the widened sample and the link points the outcome is the writer's at.
* **The record's draw** (`ReachesOff.recordDraw`, clause (i)): at the sample-coupled coin map, where the play's sampled
  verdict accepts, the log v2 read is the device's, v2 reads the sample's coins of the family `label` answering `after`,
  and every release of that family precedes every opening, the record decodes to the session's draw file at the
  outcome's sample (`recordDraw_firstStop`, `sample_extend`).
-/

@[expose] public section

namespace Verity.Sampled

open Verity.Game
open FlockSoundness.Discharge.Tables (LiveOutJ)

local notation "𝔽" => FlockLevel3.GF128

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-- **An outcome on the joint law's support off `BL` and `B` is the writer's, beside a play the first stop reaches**
(`reachesOff_out`, `firstStop_playReaches`). -/
theorem ReachesOff.out_reaches {V : Vocab d α} {θ : ByteArray} {F : Array ByteArray} {c : ByteArray} {A : Live}
    {rd : Reading A.rounds} {σ : Strategy (V.game θ F c)} {D : Decoders A.rounds A.Reg}
    {M : CoinMap (V.game θ F c) σ A.Ω A.rounds A.mP} {BL : LinkHold A.mP} {B : RoundHold}
    {o : A.Out × CoinTape (V.game θ F c) σ} (h : ReachesOff V θ F c A rd σ D M BL B o) :
    ∃ (ω : (A.extend M.widening).L.Ω) (pts : Fin 2 → Fin A.mP → 𝔽)
      (reps : Fin (A.J ((A.extend M.widening).L.draw ω) (regOf V θ F c A rd σ D)) → Fin 2 → List Flock.Round),
      ¬ BL (firstStop V θ F c A rd σ M ω).g' pts ∧
      o.1 = (regOf V θ F c A rd σ D, (A.extend M.widening).L.draw ω,
        A.wr ((A.extend M.widening).L.draw ω) (regOf V θ F c A rd σ D)
          ⟨D.root (firstStop V θ F c A rd σ M ω).modes, pts, reps, D.proofs (CoinTape.run (V.game θ F c) σ o.2)⟩) ∧
      PlayReaches (firstStop V θ F c A rd σ M ω).g' (firstStop V θ F c A rd σ M ω).s'
        (CoinTape.run (V.game θ F c) σ o.2) := by
  obtain ⟨ω, pts, hL, hr⟩ := h
  obtain ⟨_, h1, -⟩ := hr.out
  exact ⟨ω, pts, _, hL, h1, firstStop_playReaches V θ F c A rd σ M ω hr⟩

/-- **The record's draw on the joint law's support** (`MovesOf`'s clause (i)): at the sample-coupled coin map, an
outcome on the joint law's support off `BL` and `B` is the writer's at the widened sample `ω`, and where the play beside
it ends where the sampled verdict accepts, the log v2 read is the device's, v2 reads the sample's coins of the family
`label` answering `after`, and every release of that family precedes every opening, its record decodes to the session's
draw file at `ω`'s sample (`ReachesOff.out_reaches`, `recordDraw_firstStop`). -/
theorem ReachesOff.recordDraw (e : 𝔽 ≃ (Fin 128 → Bool)) {V : Vocab d α} {θ : ByteArray} {F : Array ByteArray}
    {c : ByteArray} {k : Verity.Registration.Composite} {A : Live} {rd : Reading A.rounds}
    {σ : Strategy (V.game θ F c)} {D : Decoders A.rounds A.Reg} {BL : LinkHold A.mP} {B : RoundHold}
    {label : String} {after : ByteArray} {m : ℕ} {κ : A.Ω → Fin m → Fin 512 → Bool}
    (hA : SetsUp V θ F A) (ht : PopTied θ A) (hR : DrawReadAll k coinsRead) (hH : DrawHeld k)
    (hrd : PicksOpenings rd) (hm : _root_.SampledProofs.Challenger.MAX_COINS ≤ m)
    (hsrc : ∀ ω, _root_.Properties.SampledProofs.bytesOf (A.src ω 0) =
      _root_.Properties.SampledProofs.bytesOf (coinBytes m 64 (κ ω) 0))
    (hfix : ∀ S, Flock.norm (A.dj S) = A.dj S) {o : A.Out × CoinTape (V.game θ F c) σ}
    (h : ReachesOff V θ F c A rd σ D (sampleMap e V θ F c A rd σ label after κ) BL B o)
    (hv : V.Verdict k .sampled (CoinTape.run (V.game θ F c) σ o.2))
    (hl : Binding.Logged Binding.readsV2 (CoinTape.run (V.game θ F c) σ o.2))
    (hN : SampleNamed k (CoinTape.run (V.game θ F c) σ o.2).sent label after)
    (hP : ∀ pre r post, (CoinTape.run (V.game θ F c) σ o.2).log = pre ++ r :: post → ∀ i s n,
      r.frame = .opening i s n → ∀ q ∈ post, ¬ (q.frame = .fresh label ∧ q.answers = after)) :
    ∃ (ω : (A.extend (sampleMap e V θ F c A rd σ label after κ).widening).L.Ω) (pts : Fin 2 → Fin A.mP → 𝔽)
      (reps : Fin (A.J ((A.extend (sampleMap e V θ F c A rd σ label after κ).widening).L.draw ω)
        (regOf V θ F c A rd σ D)) → Fin 2 → List Flock.Round),
      ¬ BL (firstStop V θ F c A rd σ (sampleMap e V θ F c A rd σ label after κ) ω).g' pts ∧
      o.1 = (regOf V θ F c A rd σ D, (A.extend (sampleMap e V θ F c A rd σ label after κ).widening).L.draw ω,
        A.wr ((A.extend (sampleMap e V θ F c A rd σ label after κ).widening).L.draw ω) (regOf V θ F c A rd σ D)
          ⟨D.root (firstStop V θ F c A rd σ (sampleMap e V θ F c A rd σ label after κ) ω).modes, pts, reps,
            D.proofs (CoinTape.run (V.game θ F c) σ o.2)⟩) ∧
      ∀ record, sentRecord (CoinTape.run (V.game θ F c) σ o.2).sent = some record →
        FlockSoundness.Discharge.Exec.recordDraw record =
          some (A.dj ((A.extend (sampleMap e V θ F c A rd σ label after κ).widening).L.draw ω)) := by
  obtain ⟨ω, pts, reps, hL, h1, hreach⟩ := h.out_reaches
  refine ⟨ω, pts, reps, hL, h1, fun record hrec => ?_⟩
  rw [sample_extend]
  exact recordDraw_firstStop e hA ht hR hH hrd hm hsrc hfix ω hreach hv hl hN hP hrec

end Verity.Sampled
