module

public import VStarRecursive.SampledTwo

/-!
# What an accepted two-stage check ran

The two-stage form's check (`VStar.Recursive.SampledTwo.checkBlobsAtWith`) accepts only through its steps. Where it
accepts at the switch `form` (`accepted_steps`):

* the switch is on;
* the run of record's input is `inputAt`'s (`inputAt_ok`): the interior registration's commitment is its record's, its
  sample read the bytes `ξ₂` (`OneStageZk.coinsOf`), every session `OneStageZk.membersRun` ran accepted (`r.all`), and
  the input is `inputOf` at those sessions' verdicts stamped (`stamped`);
* `Spot.Run.run`, at the switch on, returned a run of record whose symbol is `accept`, and the first stage's bytes `ξ`
  run no further than what it read (`sourceRead`, read as `ξ.size ≤ read` by `sourceRead_le`), and on two windows the
  first session's log shows no second-stage coin after the interior's digest (`interiorUnshown`, read by
  `interiorUnshown_roots`);
* the interior audit's verdict at its coins, over the sessions' verdicts, is the accept, with the check's public
  outputs and profile.

The pieces read an accepted play (`Verity.SampledTwo.AcceptsOn`) through these.
-/

@[expose] public section

namespace VStar.Recursive.SampledTwo

open Lean (Json)
open Spot

private theorem bind_ok {ε α β : Type} {x : Except ε α} {f : α → Except ε β} {b : β} (h : x >>= f = .ok b) :
    ∃ a, x = .ok a ∧ f a = .ok b := by
  cases x with
  | error => cases h
  | ok a => exact ⟨a, rfl, h⟩

/-- **The last step accepts** only with the switch on, the run of record's symbol `accept` and the interior audit
accepting with the same public outputs and profile. -/
theorem finishWith_accept {form : Bool} {two : Core.Outcome OneStage.Public Core.Integrity} {run : Json}
    {pub : OneStage.Public} {prof : Core.Integrity} (h : finishWith form two run = .accept pub prof) :
    form = true ∧ Spot.jeq (Spot.getD run "symbol") (Json.str "accept") = true ∧ two = .accept pub prof := by
  unfold finishWith at h
  cases two with
  | reject why => cases h
  | undecided why => cases h
  | accept pub' prof' =>
    cases hj : Spot.jeq (Spot.getD run "symbol") (Json.str "accept") <;> cases form <;> simp_all

/-- **What `inputAt` computed** where it returned. -/
theorem inputAt_ok {p : Params} {c : OneStage.Commitment} {ξ : ByteArray} {e : Evidence} {bs : Core.Blobs}
    {c₂ : OneStage.Commitment} {ξ₂ : ByteArray} {r : OneStageZk.Run} {i : Spot.Run.Input}
    (h : inputAt p c ξ e bs = .ok (c₂, ξ₂, r, i)) :
    (∃ s, Flock.canon e.record = .ok s ∧ OneStage.commitmentOf s.toUTF8 = .ok c₂) ∧
      OneStageZk.coinsOf p.audit c₂ e.interior = .ok ξ₂ ∧
      OneStageZk.membersRun p.audit p.members c₂ e.interior e.statements bs = .ok r ∧ r.all = true ∧
      ∃ vs received₂, stamped p e bs r.out = .ok vs ∧ i = inputOf p c ξ ξ₂ e received₂ vs := by
  unfold inputAt at h
  obtain ⟨s, h1, h⟩ := bind_ok h
  obtain ⟨c₂', h2, h⟩ := bind_ok h
  obtain ⟨ξ₂', h3, h⟩ := bind_ok h
  obtain ⟨r', h4, h⟩ := bind_ok h
  cases h5 : r'.all
  · simp [h5] at h
    cases h
  · simp only [h5, Bool.not_true, Bool.false_eq_true, ↓reduceIte] at h
    obtain ⟨vs, h6, h⟩ := bind_ok h
    split at h
    · simp only [pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, rfl, rfl, rfl⟩ := h
      exact ⟨⟨s, h1, h2⟩, h3, h4, h5, vs, _, h6, rfl⟩
    · cases h

/-- **An accepted two-stage check ran every step** (module docstring). -/
theorem accepted_steps {form : Bool} {p : Params} {c : OneStage.Commitment} {ξ : ByteArray} {e : Evidence}
    {bs : Core.Blobs} {pub : OneStage.Public} {prof : Core.Integrity}
    (h : checkBlobsAtWith form p c ξ e bs = .accept pub prof) :
    form = true ∧ ∃ c₂ ξ₂ r i run, inputAt p c ξ e bs = .ok (c₂, ξ₂, r, i) ∧ Spot.Run.run i true = .ok run ∧
      sourceRead ξ run = .ok () ∧ interiorUnshown p e = .ok () ∧
      Spot.jeq (Spot.getD run "symbol") (Json.str "accept") = true ∧
      OneStage.checkAt p.audit.base c₂ (some ξ₂) { e.interior.base with verdicts := r.out } = .accept pub prof := by
  unfold checkBlobsAtWith at h
  cases hp : prepare form p c ξ e bs with
  | error err => simp [hp] at h
  | ok v =>
    obtain ⟨two, run⟩ := v
    simp only [hp] at h
    obtain ⟨hf, hj, htwo⟩ := finishWith_accept h
    subst hf
    unfold prepare at hp
    obtain ⟨⟨c₂, ξ₂, r, i⟩, hi, hp⟩ := bind_ok hp
    obtain ⟨run', hr, hp⟩ := bind_ok hp
    obtain ⟨u, hs, hp⟩ := bind_ok hp
    obtain ⟨u', hu, hp⟩ := bind_ok hp
    simp only [pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at hp
    obtain ⟨rfl, rfl⟩ := hp
    exact ⟨rfl, c₂, ξ₂, r, i, run', hi, hr, hs, hu, hj, htwo⟩

/-- **The evidence's first-stage bytes are no more than the run read**, where the run reports what it read and
`sourceRead` passed. -/
theorem sourceRead_le {ξ : ByteArray} {run : Json} (h : sourceRead ξ run = .ok ()) {read : Nat}
    (hr : Spot.nat? (Spot.getD run "read") = some read) : ξ.size ≤ read := by
  unfold sourceRead at h
  rw [hr] at h
  by_cases hlt : read < ξ.size
  · simp [hlt] at h
  · omega

/-- **On two windows the first session's log shows no second-stage coin after the interior's digest**: where the
check's `interiorUnshown` passed, the boundary's window is `w` and not the interior's, the second stage's family is
`label`, the interior registration's digest is `after` and the first session `e.coins` names on `w` has the roots
`roots`, those roots show no coin of `label` released after `after`. -/
theorem interiorUnshown_roots {p : Params} {e : Evidence} (h : interiorUnshown p e = .ok ()) {w label : String}
    {after : ByteArray} {k : Spot.Challenger.ChallengerKey} {c : Spot.Challenger.LiveCommitment}
    {roots : Spot.Challenger.Roots} (hw : p.boundary.verifier.windowRoot = some w)
    (hne : some w ≠ p.interior.verifier.windowRoot)
    (hl : Spot.getD (Spot.getD e.interior.base.coins "segment") "label" = .str label)
    (ha : Spot.Registration.digest e.record = .ok after)
    (hk : Spot.Challenger.ChallengerKey.ofJson (Spot.getD e.coins "key") = .ok k)
    (ht : Spot.Challenger.trustedSessionOf (Spot.getD e.coins "ledger") (Json.mkObj [("root", Json.str w)]) k
      (Spot.getD e.coins "pins") (Spot.getD e.coins "main") (Spot.getD e.coins "session") = .ok (c, roots)) :
    interiorShown roots label after = false := by
  have hb : (some w == p.interior.verifier.windowRoot) = false := by simpa using hne
  unfold interiorUnshown at h
  rw [hw, hl] at h
  simp only [hb, Bool.false_eq_true, ↓reduceIte, ha, hk, ht, bind, Except.bind] at h
  cases hc : interiorShown roots label after
  · rfl
  · rw [hc] at h
    simp at h

end VStar.Recursive.SampledTwo
