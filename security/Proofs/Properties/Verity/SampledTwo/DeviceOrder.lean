module

public import Proofs.Properties.Verity.SampledTwo.CutOrder
public import Proofs.Properties.Verity.SampledTwo.Device

/-!
# The two-session history printed the play's log in its ledger's order

`CutOrder.cutBound_order` takes, in `PrintsTwo`'s place, a fact of the device alone (`PrintsInOrder`): on two windows,
where the device's own ledger opens the first window for a run `a` and the second for a run `b` and ends `a` before it
reserves `b`, the play's log is the releases of the runs on the first window and then of those on the second. This
file proves it at the two-session history (`Sessions.dev₂`), and states the cut at the deployed history.

* **What it takes of the two setups' ledger lines** (`LedgerInOrder`), three facts, none naming a window:
  a run is opened on one window root (`once`); the lines open only the two setups' runs (`own`); and they hold no
  `end` of the second setup's run whose `seq` is below a `reserve` of the first's (`order`).
* **Why they give the order** (`printsInOrder_dev₂`): the runs `a` and `b` are the two setups' (`own`), and not one run,
  which would be opened on both windows (`once`). They are not the second setup's and then the first's (`order`). So
  the ledger opens the first setup's run on the first window and the second's on the second, and neither on the
  other's (`once`): each window's runs print their part of the log (`printsOn_part`).
* **The swap.** Where the prover's request names the two windows the other way round, a device that opens each window
  as the request names it opens the first window for the second setup's run. The three facts still hold of its ledger,
  and `PrintsInOrder` holds with them: the condition it asks would be an `end` of the second setup's run before a
  `reserve` of the first's.

`dev₂` takes each setup's ledger lines as given (`RunSetup.ledger`, a list of lines; at a deployment, its `opens` and
`closes`, functions of the play), so no theorem here proves the three facts. `LedgerInOrder` is taken as a hypothesis,
not proved: it is a lemma of the modelled device once the model builds that device's whole ledger line by line, and
no statement takes it then.
-/

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

namespace Verity.SampledTwo.Sessions

open Lean (Json)
open Properties.Challenger Spot.Challenger
open Verity.Sampled.Device (RunSetup)

/-! ## The ledger's opens and order, at more lines -/

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

/-- **A window the device's ledger opens withheld for a run is one its ledger opens for it** (`OpensOn`, which asks the
v2 schema and a withheld run, gives `LedgerOpens`, which asks neither): under v2 the root an open names is its
window's `root`. -/
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

/-! ## The two setups' ledger lines -/

/-- **The two setups' ledger lines are two sessions' in the device's order**: a run is opened on one window root; the
lines open only the two setups' runs; and they hold no `end` of the second setup's run whose `seq` is below a `reserve`
of the first's. None names a window, so each holds of a device that opens the windows as a prover's request names
them. It is taken as a hypothesis, not proved: it is a lemma of the modelled device once the model builds that
device's whole ledger line by line, and no statement takes it then. -/
structure LedgerInOrder (S₁ S₂ : RunSetup) : Prop where
  once : ∀ w w' a, LedgerOpens (S₁.ledger ++ S₂.ledger) w a → LedgerOpens (S₁.ledger ++ S₂.ledger) w' a → w = w'
  own : ∀ w a, LedgerOpens (S₁.ledger ++ S₂.ledger) w a → a = S₁.id ∨ a = S₂.id
  order : ¬ EndsBefore (S₁.ledger ++ S₂.ledger) S₂.id S₁.id

/-- **The two-session history printed the play's log in its ledger's order** (`PrintsInOrder`), at every play: on one
window, the one run prints the whole log; on two, where the setups' ledger lines are two sessions' in the device's
order (`LedgerInOrder`) and the ledger opens the first window for a run and the second for a run and ends the first
before it reserves the second, those runs are the first setup's and the second's, and the log is the first
`cut o.log` releases and the rest, each printed by its setup's run. `LedgerInOrder` is taken, not proved. -/
theorem printsInOrder_dev₂ (S₁ S₂ : Exchange → RunSetup) (cut : List Release → ℕ) (o : Exchange) {w₁ w₂ : String}
    (hw : windowsOf o.params = some (w₁, w₂)) (hL : w₁ ≠ w₂ → LedgerInOrder (S₁ o) (S₂ o)) :
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
    have h2 := oneWindow_false hw hww
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

/-! ## The cut at the deployed history -/

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-- **The cut where the device's ledger orders the two sessions, at the deployed two-session history**
(`CutOrder.cutBound_order`): where the sampled verdict accepts only where the form does (`Wired`), the first stage's
coins are read under the parameters' key (`FirstReads`), an accepted play on two windows shows the order in the ledger
the second stage's coin evidence carries (`OrderShown`), the two setups' ledger lines are two sessions' in the device's
order (`LedgerInOrder`) and the vocabulary's signer is the deployment's, an accepted play whose logs the check didn't
read, or read other than the device's, is at most as likely as the read finder's collision plus the forger's forgery
over both sessions. `LedgerInOrder` is taken, not proved. -/
theorem cutOrder_dev₂ (V : Vocab d α) (k : Verity.Registration.Composite) (P : Deployment)
    (fs₁ fs₂ : ByteArray → List (String × ℕ)) (cut : List Release → ℕ) (θ : ByteArray) (F : Array ByteArray)
    (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c)) (hw : Wired V k θ) (hF : FirstReads θ)
    (hO : OrderShown θ)
    (hL : ∀ o : Exchange, o.params = θ → ∀ w₁ w₂, windowsOf θ = some (w₁, w₂) → w₁ ≠ w₂ →
      LedgerInOrder (P.setup₁ fs₁ cut o) (P.setup₂ fs₂ cut o))
    (hD : V.signer θ = P.signer fs₁ fs₂ cut θ) :
    CutBound V k θ F c σ ReadIsDevice (terms V (devOf P fs₁ fs₂ cut) θ F c σ) :=
  cutBound_order V k (devOf P fs₁ fs₂ cut) θ F c σ hw hF hO
    (fun o hθ w₁ w₂ hw' =>
      printsInOrder_dev₂ (P.setup₁ fs₁ cut) (P.setup₂ fs₂ cut) cut o (by rw [hθ]; exact hw') (hL o hθ w₁ w₂ hw'))
    (signsPlay_dev₂ V P fs₁ fs₂ cut θ F c hD)

end Verity.SampledTwo.Sessions
