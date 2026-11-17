module

public import Proofs.Properties.Verity.SampledTwo.CutOrder
public import Proofs.Properties.Verity.SampledTwo.Device

/-! `PrintsInOrder` at the two-session history, from `LedgerInOrder`, which is taken as a hypothesis, not
proved, because `dev₂` takes each setup's ledger lines as given. -/

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

namespace Verity.SampledTwo.Sessions

open Lean (Json)
open Properties.Challenger Spot.Challenger
open Verity.Sampled.Device (RunSetup)

theorem ledgerOpens_mono {L L' : List Json} {w a : String} (h : LedgerOpens L w a) (hL : ∀ x ∈ L, x ∈ L') :
    LedgerOpens L' w a := by
  unfold LedgerOpens at h ⊢
  obtain ⟨line, hl, rest⟩ := h
  exact ⟨line, hL line hl, rest⟩

theorem endsBefore_mono {L L' : List Json} {a b : String} (h : EndsBefore L a b) (hL : ∀ x ∈ L, x ∈ L') :
    EndsBefore L' a b := by
  unfold EndsBefore at h ⊢
  obtain ⟨x, hx, y, hy, rest⟩ := h
  exact ⟨x, hL x hx, y, hL y hy, rest⟩

theorem ledgerOpens_of_opensOn {h : History} {pk : ByteArray} {w id : String} (ho : OpensOn h pk w id) :
    LedgerOpens (h.ledger pk) w id := by
  unfold OpensOn at ho
  obtain ⟨line, hl, e, he, hk, hs, hid, hr, -⟩ := ho
  unfold LedgerOpens
  refine ⟨line, hl, e, he, hk, hid, ?_⟩
  have hb : (e.schema == Spot.Challenger.LEDGER_SCHEMA) = true := beq_iff_eq.2 hs
  have h1 : Properties.Pouw.LiveSalt.openRoot e = Json.str w := by
    unfold Properties.Pouw.LiveSalt.openRoot
    rw [if_pos hb]
    exact hr
  unfold openedRoot
  rw [h1]
  rfl

/-- A run is opened on one window root, the lines open only the two setups' runs, and they hold no `end` of
the second setup's run whose `seq` is below a `reserve` of the first's; none names a window. It is taken as a
hypothesis, not proved. -/
structure LedgerInOrder (S₁ S₂ : RunSetup) : Prop where
  once : ∀ w w' a, LedgerOpens (S₁.ledger ++ S₂.ledger) w a → LedgerOpens (S₁.ledger ++ S₂.ledger) w' a → w = w'
  own : ∀ w a, LedgerOpens (S₁.ledger ++ S₂.ledger) w a → a = S₁.id ∨ a = S₂.id
  order : ¬ EndsBefore (S₁.ledger ++ S₂.ledger) S₂.id S₁.id

/-- `LedgerInOrder` is taken, not proved. -/
theorem printsInOrder_dev₂ (S₁ S₂ : Exchange → RunSetup) (cut : List Release → ℕ) (o : Exchange) {w₁ w₂ : String}
    (hw : executionPhasesOf o.params = some (w₁, w₂)) (hL : w₁ ≠ w₂ → LedgerInOrder (S₁ o) (S₂ o)) :
    PrintsInOrder (dev₂ S₁ S₂ cut o) w₁ w₂ o.log := by
  unfold PrintsInOrder
  by_cases hww : w₁ = w₂
  · rw [if_pos hww]
    have h := printsTwo_dev₂ S₁ S₂ cut o hw (fun hne => absurd hww hne)
    unfold PrintsTwo at h
    rw [if_pos hww] at h
    exact h
  · rw [if_neg hww]
    intro pk a b ha hb hab
    have h2 := oneExecutionPhase_false hw hww
    obtain ⟨once, own, order⟩ := hL hww
    have hsub := ledger_two S₁ S₂ cut h2
    have ha' := ledgerOpens_mono ha (hsub pk)
    have hb' := ledgerOpens_mono hb (hsub pk)
    have hab' := endsBefore_mono hab (hsub pk)
    have hne : a ≠ b := by
      intro e
      rw [e] at ha'
      exact hww (once w₁ w₂ b ha' hb')
    have hid : a = (S₁ o).id ∧ b = (S₂ o).id := by
      rcases own w₁ a ha' with h1 | h1
      · rcases own w₂ b hb' with h3 | h3
        · exact absurd (h1.trans h3.symm) hne
        · exact ⟨h1, h3⟩
      · rcases own w₂ b hb' with h3 | h3
        · rw [h1, h3] at hab'
          exact absurd hab' order
        · exact absurd (h1.trans h3.symm) hne
    rw [hid.1] at ha'
    rw [hid.2] at hb'
    refine ⟨_, _, (List.take_append_drop (cut o.log) o.log).symm, ?_, ?_⟩
    · refine printsOn_part (S := S₁ o) (S' := S₂ o) (log' := o.log.drop (cut o.log)) (fun pk' r hr => ?_)
        fun pk' ho => ?_
      · rw [runs_two S₁ S₂ cut h2] at hr
        exact List.mem_append.1 hr
      · exact hww (once w₁ w₂ (S₂ o).id (ledgerOpens_mono (ledgerOpens_of_opensOn ho) (hsub pk')) hb')
    · refine printsOn_part (S := S₂ o) (S' := S₁ o) (log' := o.log.take (cut o.log)) (fun pk' r hr => ?_)
        fun pk' ho => ?_
      · rw [runs_two S₁ S₂ cut h2] at hr
        exact (List.mem_append.1 hr).symm
      · exact hww (once w₁ w₂ (S₁ o).id ha' (ledgerOpens_mono (ledgerOpens_of_opensOn ho) (hsub pk')))

variable {ι : Type} {d : ResourceAccounting.Device ι} {α : Type} [Fintype α]

/-- `LedgerInOrder` is taken, not proved. -/
theorem cutOrder_dev₂ (V : Vocab d α) (k : Verity.Registration.Composite) (P : Deployment)
    (fs₁ fs₂ : ByteArray → List (String × ℕ)) (cut : List Release → ℕ) (θ : ByteArray) (F : Array ByteArray)
    (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)) (hw : Wired V k θ) (hF : FirstReads θ)
    (hO : OrderShown θ)
    (hL : ∀ o : Exchange, o.params = θ → ∀ w₁ w₂, executionPhasesOf θ = some (w₁, w₂) → w₁ ≠ w₂ →
      LedgerInOrder (P.setup₁ fs₁ cut o) (P.setup₂ fs₂ cut o))
    (hD : V.signer θ = P.signer fs₁ fs₂ cut θ) :
    CutBound V k θ F c σ ReadIsDevice (terms V (devOf P fs₁ fs₂ cut) θ F c σ) :=
  cutBound_order V k (devOf P fs₁ fs₂ cut) θ F c σ hw hF hO
    (fun o hθ w₁ w₂ hw' =>
      printsInOrder_dev₂ (P.setup₁ fs₁ cut) (P.setup₂ fs₂ cut) cut o (by rw [hθ]; exact hw') (hL o hθ w₁ w₂ hw'))
    (signsPlay_dev₂ V P fs₁ fs₂ cut θ F c hD)

end Verity.SampledTwo.Sessions
