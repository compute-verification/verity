module

public import Proofs.Properties.Verity.SampledTwo.CutRead
public import Proofs.Properties.Verity.SampledTwo.Steps
public import Proofs.Flock.Claims.Sampling.Select.Code
public import Proofs.Flock.SpotFlock.Jeq
public import Proofs.Properties.Challenger.Limits
public import Proofs.Properties.Challenger.Coins
public import Proofs.Properties.Spot
import all Lean.Data.Json.Basic
import all Init.Data.Ord.String
import all Init.Data.Array.Basic
import all Init.Data.String.Search
import all Init.Data.String.Slice

/-!
# The first stage's coins read under the parameters' key, from the form's acceptance

`CutRead` takes `FirstReads θ` as a premise: on every accepted play, the parameters name a challenger key at the
deployed LMS parameters, the case names the first session's window and family, and `drawCoins` under that key reads
the first stage's coin evidence. This module derives it from the two-stage form's acceptance (`firstReads_of_shown`)
on every play whose first stage's coin evidence is not `null` (`FirstShown`), in four steps:

* **The key** (`keyOf_some`): the form reads the interior audit's parameters with `OneStageZk.membersParamsOfJson`
  (`form_audit`), which reads them as `OneStageZk.paramsOfJson` does, and that refuses a challenger key at any other
  LMS parameters (`members_deployed`, `Limits.paramsOfJson_deployed`). So `keyOf θ` is the interior audit's key.
* **The coin check ran** (`run_coins`): `Spot.Run.run` returns a refusal (`{"step", "codes"}`) at every step before
  its last, and reaches its last only past its coin check (`Spot.Run.coinsCode`), which then gave no code. A refusal
  has no `symbol` (`refused_symbol`), and the form accepts only where the run of record's symbol is `accept`
  (`accepted_steps`), so on an accepted play the coin check gave no code.
* **What the coin check read** (`coinsCode_none`): no code is either `null` evidence where the first stage read no
  byte, or `drawCoins` reading the evidence under the run's key, window and family after the record's digest
  (`coinsReason_none`).
* **The run's inputs** (`inputAt_ok`, `inputOf`): the run's key is the interior audit's, its window and family the
  boundary registration's, its coin evidence the form's `coins` (`firstCoins`), its record the registration's, whose
  digest is the commitment's (`Properties.Spot.digestIsRecord's`). These are `firstDraws`'.

`FirstShown θ` is a hypothesis here, and `CutAccept.firstShown` proves it from the form's refusal of `null` first-stage
coin evidence. The run of record alone passes a first stage that read no byte with `null` coin evidence, and of such a
play the cut reads no first session (`readsLogs` is empty), so its event would fail there.

As in `Cut`, at a history holding one play's signatures a genuine signature of an earlier period that the evidence
carries is none the history signed, so it is the forger's, and the forger's term bounds nothing there.
-/

@[expose] public section

namespace Verity.SampledTwo

open Lean (Json)
open Properties.Challenger Spot.Challenger
open scoped ENNReal

/-! ## The parameters' key -/

/-- The per-member form's parameters name a challenger key at the deployed LMS parameters, as the one statement's
do (`Limits.paramsOfJson_deployed`). -/
theorem members_deployed {j : Json} {a : Spot.OneStageZk.Params}
    (h : Spot.OneStageZk.membersParamsOfJson j = .ok a) :
    ∃ key label, a.base.challenger = some (key, label) ∧ key.params = Lms.deployed := by
  unfold Spot.OneStageZk.membersParamsOfJson at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  repeat' split at h
  all_goals first
    | (cases h; done)
    | (cases h; exact Limits.paramsOfJson_deployed (by assumption))

-- `paramsOfJson` unfolds to one `match` or `if` per decoded field, hence the heartbeat budget.
set_option maxHeartbeats 4000000 in
/-- The two-stage form's interior audit parameters are the per-member form's. -/
theorem form_audit {j : Json} {p : VStar.Recursive.SampledTwo.Params}
    (h : VStar.Recursive.SampledTwo.paramsOfJson j = .ok p) :
    ∃ j', Spot.OneStageZk.membersParamsOfJson j' = .ok p.audit := by
  unfold VStar.Recursive.SampledTwo.paramsOfJson at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  repeat' split at h
  all_goals first
    | (cases h; done)
    | (cases h; exact ⟨_, by assumption⟩)

/-- The sampled component's parameters are the two-stage form's `paramsOfJson`'s. -/
theorem paramsAt_form {θ : ByteArray} {p : VStar.Recursive.SampledTwo.Params} (h : paramsAt θ = .ok p) :
    ∃ j, VStar.Recursive.SampledTwo.paramsOfJson j = .ok p := by
  unfold paramsAt at h
  simp only [bind, Except.bind] at h
  repeat' split at h
  all_goals first
    | (cases h; done)
    | exact ⟨_, h⟩
    | exact ⟨_, by assumption⟩

/-- **The parameters' key is the interior audit's challenger's**, at the deployed LMS parameters. -/
theorem keyOf_some {θ : ByteArray} {p : VStar.Recursive.SampledTwo.Params} (h : paramsAt θ = .ok p) :
    ∃ key label, p.audit.base.challenger = some (key, label) ∧ keyOf θ = some key := by
  obtain ⟨j, hj⟩ := paramsAt_form h
  obtain ⟨j', ha⟩ := form_audit hj
  obtain ⟨key, label, hk, hd⟩ := members_deployed ha
  refine ⟨key, label, hk, ?_⟩
  have hD : atDeployed key = true := by simp [atDeployed, hd]
  simp [keyOf, h, hk, Except.toOption, hD]

/-! ## The run of record's coin check -/

theorem none_of_forall {o : Option String} (h : ∀ a, o = some a → False) : o = none := by
  cases o with
  | none => rfl
  | some a => exact (h a rfl).elim

-- `run` unfolds to about 50 decided leaves under one `match` or `if` per step, hence the heartbeat budget.
set_option maxHeartbeats 4000000 in
/-- **A run that returned refused at a step, or passed its coin check**: the coin check (`coinsCode`) at the run's
key, main's history, coin evidence, record, window and family on the first `read` bytes of its source gave no code. -/
theorem run_coins (i : Spot.Run.Input) (form : Bool) (j : Json) (h : Spot.Run.run i form = .ok j) :
    (∃ (step : String) (codes : Array String),
      j = Json.mkObj [("step", Json.str step), ("codes", Json.arr (codes.map Json.str))]) ∨
    ∃ read, Spot.Run.coinsCode i.key i.main i.coins i.record i.params.verifier.windowRoot i.family
      (i.source.extract 0 read) = none := by
  unfold Spot.Run.run at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  repeat'
    first
    | split at h
    | focus (cases h; done)
    | (obtain ⟨_, h⟩ | ⟨_, h⟩ := Spot.Run.ite_eq_ok h; exact Or.inl ⟨_, _, h.symm⟩)
    | exact Or.inl ⟨_, _, (Except.ok.inj h).symm⟩
    | exact Or.inr ⟨_, by assumption⟩
    | exact Or.inr ⟨_, none_of_forall (by assumption)⟩

/-- A refusal names no symbol. -/
theorem refused_symbol (step : String) (codes : Array String) :
    Spot.getD (Json.mkObj [("step", Json.str step), ("codes", Json.arr (codes.map Json.str))]) "symbol" =
      Json.null :=
  (rfl)

/-- **What a coin check that gave no code read**: `null` evidence, or `drawCoins` reading the evidence under the key,
window and family after the record's digest. -/
theorem coinsCode_none {key : Option ChallengerKey} {main coins record : Json} {window family : Option String}
    {read : ByteArray} (h : Spot.Run.coinsCode key main coins record window family read = none) :
    coins = Json.null ∨ ∃ k w label after ξ, key = some k ∧ window = some w ∧ family = some label ∧
      Spot.Registration.digest record = .ok after ∧ drawCoins k w label after coins = .ok ξ := by
  unfold Spot.Run.coinsCode at h
  by_cases hr : read.isEmpty = true
  · rw [if_pos hr] at h
    left
    cases coins <;> first | rfl | (simp at h)
  · rw [if_neg hr] at h
    right
    cases key with
    | none => simp at h
    | some k =>
      cases window with
      | none => simp at h
      | some w =>
        cases family with
        | none => simp at h
        | some label =>
          cases hd : Spot.Registration.digest record with
          | error _ => rw [hd] at h; simp at h
          | ok after =>
            rw [hd] at h
            cases hc : Spot.Challenger.coinsReason k main coins w label after read with
            | some _ => simp [hc] at h
            | none =>
              obtain ⟨ξ, hξ, -⟩ := Properties.Challenger.Coins.coinsReason_none hc
              exact ⟨k, w, label, after, ξ, rfl, rfl, rfl, rfl, hξ⟩

/-! ## The first stage's coins on an accepted play -/

/-- The two-stage case's registration is a commitment the form decoded. -/
theorem commitment_of_caseOf {s : Verity.Sent} {p : VStar.Recursive.SampledTwo.Params}
    {c : Spot.OneStage.Commitment} {e : VStar.Recursive.SampledTwo.Evidence} (h : caseOf s = .ok (p, c, e)) :
    ∃ b, Spot.OneStage.commitmentOf b = .ok c := by
  unfold caseOf at h
  obtain ⟨p', -, h⟩ := bind_ok h
  obtain ⟨ce, hce, h⟩ := bind_ok h
  have hc : ce.1 = c := by
    first
      | exact congrArg (fun x => x.2.1) (Except.ok.inj h)
      | (simp only [pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h; exact h.2.1)
  subst hc
  unfold caseIn at hce
  simp only [bind, Except.bind, pure, Except.pure] at hce
  repeat' split at hce
  all_goals first
    | (cases hce; done)
    | (cases hce; exact ⟨_, by assumption⟩)

/-- **The first stage's coin evidence is not `null`**, on every accepted play after the parameters `θ`. -/
def FirstShown (θ : ByteArray) : Prop :=
  ∀ s : Verity.Sent, s.params = θ → Accepts s → ∀ p c e, caseOf s = .ok (p, c, e) → e.coins ≠ Json.null

/-- **The first stage's coins are read under the parameters' key on every accepted play** whose first stage's coin
evidence is not `null` (module docstring). -/
theorem firstReads_of_shown (θ : ByteArray) (hS : FirstShown θ) : FirstReads θ := by
  intro s hθ ha p c e h1
  have ha' := ha
  obtain ⟨p', c', e', pub, prof, h1', hacc⟩ := ha'
  rw [h1] at h1'
  obtain ⟨hp, hc, he⟩ : p = p' ∧ c = c' ∧ e = e' := by
    simp only [Except.ok.injEq, Prod.mk.injEq] at h1'
    exact h1'
  rw [← hp, ← hc, ← he] at hacc
  obtain ⟨-, c₂, ξ₂, r, i, run, hi, hrun, -, -, hsym, -⟩ := VStar.Recursive.SampledTwo.accepted_steps hacc
  obtain ⟨-, -, -, -, vs, received₂, -, rfl⟩ := VStar.Recursive.SampledTwo.inputAt_ok hi
  rcases run_coins _ true run hrun with ⟨step, codes, rfl⟩ | ⟨read, hcc⟩
  · have hs := SpotFlock.str_of_jeq (SpotFlock.jeq_symm hsym)
    rw [refused_symbol] at hs
    cases hs
  rcases coinsCode_none hcc with hn | ⟨k, w, label, after, ξ, hk, hw, hl, hd, hξ⟩
  · exact (hS s hθ ha p c e h1 hn).elim
  have hpθ : paramsAt θ = .ok p := by
    rw [← hθ]
    exact paramsAt_of_caseOf h1
  obtain ⟨key, _, hch, hK⟩ := keyOf_some hpθ
  have hkk : key = k := by
    have hk' : (p.audit.base.challenger.map (·.1)) = some k := hk
    rw [hch] at hk'
    exact Option.some.inj hk'
  rw [hkk] at hK
  obtain ⟨b, hb⟩ := commitment_of_caseOf h1
  have hdc : Spot.Registration.digest c.record = .ok c.digest := Properties.Spot.digestIsRecord's b c hb
  have hac : after = c.digest := by
    have hd' : Spot.Registration.digest c.record = .ok after := hd
    rw [hdc] at hd'
    exact (Except.ok.inj hd').symm
  subst hac
  have hw' : p.boundary.verifier.windowRoot = some w := hw
  have hl' : firstLabel p = some label := hl
  have hb : boundaryOf p = .ok p.boundary := rfl
  refine ⟨k, ⟨k, w, label, c.digest⟩, ξ, hK, ?_, hξ⟩
  simp [firstDraws, hb, hw', hl', Except.toOption]

/-! ## The cut -/

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-- **The cut with the read in its event** (`cutBound_read`), at `FirstShown θ` in `FirstReads θ`'s place. -/
theorem cutBound_shown (V : Vocab d α) (k : Verity.Registration.Composite) (dev : Exchange → History)
    (θ : ByteArray) (F : Array ByteArray) (c : ByteArray) (σ : Verity.Game.Strategy (V.game θ F c))
    (hw : Wired V k θ) (hS : FirstShown θ)
    (hP : ∀ o : Exchange, o.params = θ → ∀ w₁ w₂, windowsOf θ = some (w₁, w₂) → PrintsTwo (dev o) w₁ w₂ o.log)
    (hs : SignsPlay V dev θ F c) : CutBound V k θ F c σ ReadIsDevice (terms V dev θ F c σ) :=
  cutBound_read V k dev θ F c σ hw (firstReads_of_shown θ hS) hP hs

end Verity.SampledTwo
