import SampledProofs.OneStageZk

/-!
# v2 accepts only the public file and the sizes its registration names

`OneStageZk.checkBlobs` refuses evidence whose public file is not the one the registration names (`publicWhy`), then a
window whose sizes are not the ones v2 reads (`sizesWhy`), before it runs `checkProofs`. So where it accepts:

* the evidence's public digest is the registration's `window.public_file` (`public_of_accept`): every file a session
  reads as its public file is fetched under the digest the statement registers, never one the evidence alone chose;
* the draw opens exactly the positive count of coins the registration's `window.draw_coins` names (`draw_of_accept`);
* session 0 holds at most `MAX_ROUNDS` rounds after its link points, each use of fewer than `2 ^ 64` F128s
  (`rounds_of_accept`);
* `checkProofs` accepts the same outcome (`checkProofs_of_accept`).
-/

namespace SampledProofs.OneStageZk

/-- `publicWhy` names no reason only where the registration names the public file `pub`. -/
theorem registeredPublic_of_publicWhy {c : OneStage.Commitment} {pub : String} (h : publicWhy c pub = none) :
    registeredPublic c = some pub := by
  unfold publicWhy at h
  cases hd : registeredPublic c with
  | none =>
    rw [hd] at h
    cases h
  | some d =>
    rw [hd] at h
    by_cases hp : pub = d
    · rw [hp]
    · simp [hp] at h

/-- **v2 accepts only the public file the registration names**: where `checkBlobs` accepts, the evidence's public
digest is the registration's `window.public_file`. -/
theorem public_of_accept {p : Params} {c : OneStage.Commitment} {e : Evidence} {bs : Core.Blobs}
    {x : OneStage.Public} {y : Core.Integrity} (h : checkBlobs p c e bs = .accept x y) :
    registeredPublic c = some e.pub := by
  unfold checkBlobs at h
  cases hw : publicWhy c e.pub with
  | some w =>
    rw [hw] at h
    cases h
  | none => exact registeredPublic_of_publicWhy hw

/-- Where `checkBlobs` accepts, `sizesWhy` names no reason. -/
theorem sizesWhy_of_accept {p : Params} {c : OneStage.Commitment} {e : Evidence} {bs : Core.Blobs}
    {x : OneStage.Public} {y : Core.Integrity} (h : checkBlobs p c e bs = .accept x y) :
    sizesWhy c e bs = none := by
  unfold checkBlobs at h
  cases hw : publicWhy c e.pub with
  | some w =>
    rw [hw] at h
    cases h
  | none =>
    rw [hw] at h
    cases hs : sizesWhy c e bs with
    | none => rfl
    | some w =>
      rw [hs] at h
      cases h

/-- Where `checkBlobs` accepts, `checkProofs` accepts the same outcome. -/
theorem checkProofs_of_accept {p : Params} {c : OneStage.Commitment} {e : Evidence} {bs : Core.Blobs}
    {x : OneStage.Public} {y : Core.Integrity} (h : checkBlobs p c e bs = .accept x y) :
    checkProofs p c e bs = .accept x y := by
  have hs := sizesWhy_of_accept h
  unfold checkBlobs at h
  cases hw : publicWhy c e.pub with
  | some w =>
    rw [hw] at h
    cases h
  | none =>
    rw [hw] at h
    rw [hs] at h
    exact h

/-- A count the registration names is positive. -/
theorem registeredDraw_pos {c : OneStage.Commitment} {m : Nat} (h : registeredDraw c = some m) : 0 < m := by
  unfold registeredDraw at h
  split at h
  next k _ =>
    split at h
    next hk =>
      cases h
      exact hk
    next => cases h
  next => cases h

/-- **v2's draw opens the coins the registration names**: where `checkBlobs` accepts, the registration's
`window.draw_coins` is a positive count `m`, and the draw's segment opens exactly `m` coins. -/
theorem draw_of_accept {p : Params} {c : OneStage.Commitment} {e : Evidence} {bs : Core.Blobs}
    {x : OneStage.Public} {y : Core.Integrity} (h : checkBlobs p c e bs = .accept x y) :
    ∃ m, registeredDraw c = some m ∧ 0 < m ∧ ∀ opened : Array Json,
      getD (getD e.base.coins "segment") "openings" = .arr opened → opened.size = m := by
  have hs := sizesWhy_of_accept h
  have hd : drawWhy c e = none := by
    unfold sizesWhy at hs
    cases hd : drawWhy c e with
    | none => rfl
    | some w =>
      rw [hd] at hs
      cases hs
  unfold drawWhy at hd
  cases hr : registeredDraw c with
  | none =>
    exfalso
    split at hd
    · unfold drawCountWhy at hd
      rw [hr] at hd
      cases hd
    · cases hd
  | some m =>
    refine ⟨m, rfl, registeredDraw_pos hr, fun opened ho => ?_⟩
    rw [ho] at hd
    unfold drawCountWhy at hd
    rw [hr] at hd
    by_cases hn : opened.size = m
    · exact hn
    · simp [hn] at hd

/-- `sessionsSizesWhy` names no reason for the sessions `s0 :: rest` only where it names none for `s0`. -/
theorem sessionSizesWhy_head {bs : Core.Blobs} {s0 : Session} {rest : List Session} {i : Nat}
    (h : sessionsSizesWhy bs (s0 :: rest) i = none) : sessionSizesWhy bs s0 i = none := by
  cases h0 : sessionSizesWhy bs s0 i with
  | none => rfl
  | some w =>
    exfalso
    revert h
    simp [sessionsSizesWhy, h0]

/-- `roundsCapWhy` names no reason only for uses within its caps. -/
theorem roundsCap_of_none {us : List (Challenger.Use × Array String)} (h : roundsCapWhy us = none) :
    us.length ≤ MAX_ROUNDS + 1 ∧ ∀ x ∈ us, x.2.size < 2 ^ 64 := by
  unfold roundsCapWhy at h
  by_cases hlen : MAX_ROUNDS + 1 < us.length
  · rw [if_pos hlen] at h
    cases h
  · rw [if_neg hlen] at h
    refine ⟨Nat.le_of_not_lt hlen, fun x hx => ?_⟩
    cases hf : us.find? tooWide with
    | some z =>
      rw [hf] at h
      cases h
    | none =>
      have := List.find?_eq_none.1 hf x hx
      simpa [tooWide] using this

/-- **v2's sessions are within its caps**: where `checkBlobs` accepts, session 0's uses of the challenger's coins, as
its record lists them (`sessionUses`), are at most `MAX_ROUNDS` rounds after its link points, each of fewer than
`2 ^ 64` F128s. -/
theorem rounds_of_accept {p : Params} {c : OneStage.Commitment} {e : Evidence} {bs : Core.Blobs}
    {x : OneStage.Public} {y : Core.Integrity} (h : checkBlobs p c e bs = .accept x y) :
    ∀ (s0 : Session) rest, e.sessions.toList = s0 :: rest →
      ∀ b record us0, bs.get s0.record s!"session {s0.name}'s record" = .ok b →
        String.fromUTF8? b = some record → sessionUses 0 record = .ok us0 →
        us0.length ≤ MAX_ROUNDS + 1 ∧ ∀ x ∈ us0, x.2.size < 2 ^ 64 := by
  intro s0 rest hl b record us0 hb hrec hus
  have hs := sizesWhy_of_accept h
  have hz : sessionsSizesWhy bs e.sessions.toList 0 = none := by
    unfold sizesWhy at hs
    cases hd : drawWhy c e with
    | some w =>
      rw [hd] at hs
      cases hs
    | none =>
      rw [hd] at hs
      exact hs
  rw [hl] at hz
  have h0 := sessionSizesWhy_head hz
  unfold sessionSizesWhy at h0
  simp only [hb, hrec, hus] at h0
  cases hc : roundsCapWhy us0 with
  | some w =>
    rw [hc] at h0
    cases h0
  | none => exact roundsCap_of_none hc

end SampledProofs.OneStageZk
