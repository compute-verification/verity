import Proofs.Flock.Soundness.Discharge.Exec.VerifyK

/-!
# flock-verify's acceptance through `keep`, at the record's parsed value

`VerifyK.lean`'s walkers read `Record.decodeWithK keep` and `verifyWithMK keep` on the record's text. The recursion's
last step at the parsed value (`FlockDeploy.nextJ`) runs the verdict on a parsed value `j`, never on a text, so the KS
and Bits route reads the same walkers at `j` (`Record.decodeJWithK`, `verifyJWithMK`, `Flock/With/VerifyE.lean`):

* `verifyJWithMK_tables_cap`, `verifyJK_ok_tables`, `verifyJK_one_ok`, `decodeJWithK_points`, `decodeJWithK_roundsWF`,
  `decodeJWithK_retained`, `decodeJWithK_allRetained`: each text walker at `j`, with the same proof less the parse.
* **The transfer** (`decodeWithK_ok_iff`, `verifyWithMK_ok_iff`): the text forms accept exactly where the record parses
  and the forms at its parsed value accept. So each walker here gives its text form, and a statement over every parsed
  value covers every record text.

At `keep := id` and SHA-512, `Record.decodeJWithK` on a record's parsed value is `Record.decode` on the record
(`decode_eq_decodeJWithK_id`).
-/

set_option Elab.async false

namespace FlockSoundness.Discharge.Exec

open Flock Refine
open Lean (Json)

/-! ## The tables at the parsed value -/

/-- **`verifyWithMK_tables_cap` at the record's parsed value `j`.** -/
theorem verifyJWithMK_tables_cap (keep : Bool → Bool) (hs : Hashes Flock.V) {spec : SessionSpec} {tabs : Array Setup}
    {j : Json} {proofs : Array ByteArray} (h : verifyJWithMK keep hs spec tabs j proofs = .ok ()) :
    ∃ (sess : RunRecord) (schs : Array Flock.Schedule), Record.decodeJWithK keep hs spec j = .ok sess ∧
      tabs.size = spec.tables.size ∧ 0 < tabs.size ∧ schs.size = tabs.size ∧
      ∀ (i : ℕ) (hi : i < tabs.size), ∃ (hs' : i < schs.size) (c : ByteArray),
        Flock.fast100 tabs[i].m = some schs[i] ∧
        verifyRepWith hs spec tabs[i] schs[i] sess i 0 proofs[2 * i]! = .ok c ∧
        verifyRepWith hs spec tabs[i] schs[i] sess i 1 proofs[2 * i + 1]! = .ok c := by
  unfold verifyJWithMK at h
  dsimp only at h
  by_cases c0 : (tabs.size != spec.tables.size || tabs.size == 0) = true
  · rw [ite_eq_left c0] at h; obtain ⟨_, h0, _⟩ := ebind_ok h; cases h0
  rw [ite_eq_right c0] at h
  obtain ⟨sess, hsess, h⟩ := ebind_ok h
  obtain ⟨_, _, h⟩ := ebind_ok h
  obtain ⟨schs, hschs, h⟩ := ebind_ok h
  split at h
  · obtain ⟨_, h0, _⟩ := ebind_ok h; cases h0
  split at h
  · obtain ⟨_, h0, _⟩ := ebind_ok h; cases h0
  obtain ⟨_, _, h⟩ := ebind_ok h
  obtain ⟨_, hloop, _⟩ := ebind_ok h
  rw [Std.Legacy.Range.forIn_eq_forIn_range'] at hloop
  simp only [Std.Legacy.Range.size, Nat.sub_zero, Nat.add_one_sub_one, Nat.div_one] at hloop
  have c0' : tabs.size = spec.tables.size ∧ 0 < tabs.size := by
    simp only [Bool.or_eq_true, bne_iff_ne, ne_eq, beq_iff_eq, not_or, not_not] at c0
    exact ⟨c0.1, Nat.pos_of_ne_zero c0.2⟩
  obtain ⟨hsz, hsch⟩ := array_mapM_ok_getElem hschs
  have hP := range'_forIn_each tabs.size _
    (fun i => ∃ c, verifyRepWith hs spec tabs[i]! schs[i]! sess i 0 proofs[2 * i]! = .ok c ∧
      verifyRepWith hs spec tabs[i]! schs[i]! sess i 1 proofs[2 * i + 1]! = .ok c)
    (fun a s y hy => by
      obtain ⟨c₀, h₀, hy⟩ := ebind_ok hy
      obtain ⟨c₁, h₁, hy⟩ := ebind_ok hy
      by_cases c3 : ((c₀ : ByteArray) != c₁) = true
      · rw [ite_eq_left c3] at hy; obtain ⟨_, h0, _⟩ := ebind_ok hy; cases h0
      rw [ite_eq_right c3] at hy
      simp only [pure, Except.pure, Except.ok.injEq] at hy
      have hb : (c₀ == c₁) = true := by
        cases hb : (c₀ == c₁)
        · exact absurd (by simp [bne, hb]) c3
        · rfl
      rw [bytes_eq_of_beq hb] at h₀
      exact ⟨⟨c₁, h₀, h₁⟩, _, hy.symm⟩) _ _ hloop
  refine ⟨sess, schs, hsess, c0'.1, c0'.2, hsz, fun i hi => ?_⟩
  have hs' : i < schs.size := hsz ▸ hi
  obtain ⟨c, h₀, h₁⟩ := hP i hi
  rw [getElem!_pos tabs i hi, getElem!_pos schs i hs'] at h₀ h₁
  refine ⟨hs', c, ?_, h₀, h₁⟩
  have hf := hsch i hi hs'
  cases hm : Flock.fast100 tabs[i].m with
  | none => rw [hm] at hf; cases hf
  | some s => rw [hm] at hf; cases hf; rfl

/-- **`verifyK_ok_tables` at the record's parsed value `j`.** -/
theorem verifyJK_ok_tables (keep : Bool → Bool) {spec : SessionSpec} {tabs : Array Setup} {j : Json}
    {proofs : Array ByteArray} (hN : ∀ st ∈ tabs, st.merkle.Named)
    (h : verifyJWithMK keep With.std spec tabs j proofs = .ok ()) :
    ∃ (sess : RunRecord) (schs : Array Flock.Schedule), Record.decodeJWithK keep With.std spec j = .ok sess ∧
      tabs.size = spec.tables.size ∧ 0 < tabs.size ∧ schs.size = tabs.size ∧
      ∀ (i : ℕ) (hi : i < tabs.size), ∃ (hs : i < schs.size) (c : ByteArray),
        Flock.fast100 tabs[i].m = some schs[i] ∧
        verifyRep spec tabs[i] schs[i] sess i 0 proofs[2 * i]! = .ok c ∧
        verifyRep spec tabs[i] schs[i] sess i 1 proofs[2 * i + 1]! = .ok c := by
  obtain ⟨sess, schs, hd, hsz, hpos, hsch, hall⟩ := verifyJWithMK_tables_cap keep With.std h
  refine ⟨sess, schs, hd, hsz, hpos, hsch, fun i hi => ?_⟩
  obtain ⟨hs, c, hf, h₀, h₁⟩ := hall i hi
  rw [With.verifyRepWith_std _ _ (hN _ (Array.getElem_mem hi))] at h₀ h₁
  exact ⟨hs, c, hf, h₀, h₁⟩

/-- **`verifyK_one_ok` at the record's parsed value `j`.** -/
theorem verifyJK_one_ok (keep : Bool → Bool) {spec : SessionSpec} {st : Setup} {j : Json}
    {proofs : Array ByteArray} (hN : st.merkle.Named)
    (h : verifyJWithMK keep With.std spec #[st] j proofs = .ok ()) :
    ∃ sess sch c, Record.decodeJWithK keep With.std spec j = .ok sess ∧ Flock.fast100 st.m = some sch ∧
      verifyRep spec st sch sess 0 0 proofs[0]! = .ok c ∧ verifyRep spec st sch sess 0 1 proofs[1]! = .ok c := by
  obtain ⟨sess, schs, hd, -, -, -, hall⟩ := verifyJK_ok_tables keep (fun s hs => by simp at hs; exact hs ▸ hN) h
  obtain ⟨hs, c, hf, h₀, h₁⟩ := hall 0 (by simp)
  exact ⟨sess, schs[0], c, hd, hf, h₀, h₁⟩

/-! ## What the decoder reads through `keep`, at the parsed value -/

set_option maxHeartbeats 1000000 in
/-- **`decodeWithK_points` at the record's parsed value `j`.** -/
theorem decodeJWithK_points (keep : Bool → Bool) (hs : Hashes Flock.V) {spec : SessionSpec} {j : Json}
    {sess : RunRecord} (h : Record.decodeJWithK keep hs spec j = .ok sess) :
    sess.points.size = spec.points ∧ ∀ p ∈ sess.points.toList, p.size = spec.mPts := by
  unfold Record.decodeJWithK at h
  dsimp only at h
  repeat' (first | (obtain ⟨_, _, h⟩ := ebind_ok h) | (split at h))
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    exact check_pts (by assumption)

set_option maxHeartbeats 1000000 in
/-- **`decodeWithK_roundsWF` at the record's parsed value `j`.** -/
theorem decodeJWithK_roundsWF (keep : Bool → Bool) (hs : Hashes Flock.V) {spec : SessionSpec} {j : Json}
    {sess : RunRecord} (h : Record.decodeJWithK keep hs spec j = .ok sess) :
    ∀ s ∈ sess.streams.toList, RoundsWF s.rounds := by
  unfold Record.decodeJWithK at h
  dsimp only at h
  repeat' (first | (obtain ⟨_, _, h⟩ := ebind_ok h) | (split at h))
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    intro s hs
    try dsimp only at hs
    obtain ⟨x, hx⟩ := mapM_ok_mem (by assumption) s hs
    try dsimp only at hx
    repeat' (first | (obtain ⟨_, _, hx⟩ := ebind_ok hx) | (split at hx))
    all_goals
      simp only [pure, Except.pure, Except.ok.injEq] at hx
      subst hx
      apply roundsWF_of
      intro r hr
      obtain ⟨k, y, hg⟩ := mapIdxM_ok_mem (by assumption) r hr
      try dsimp only at hg
      repeat' (first | (obtain ⟨_, _, hg⟩ := ebind_ok hg) | (split at hg))
      all_goals
        simp only [pure, Except.pure, Except.ok.injEq] at hg
        subst hg
        exact check_coins (by assumption)

set_option maxHeartbeats 1000000 in
/-- **`decodeWithK_retained` at the record's parsed value `j`.** -/
theorem decodeJWithK_retained (keep : Bool → Bool) (hs : Hashes Flock.V) {spec : SessionSpec} {j : Json}
    {sess : RunRecord} (hr : spec.retainRounds = true) (h : Record.decodeJWithK keep hs spec j = .ok sess) :
    ∀ s ∈ sess.streams.toList, ∀ r ∈ s.rounds.toList, ∃ b, r.bytes = some b := by
  unfold Record.decodeJWithK at h
  dsimp only at h
  repeat' (first | (obtain ⟨_, _, h⟩ := ebind_ok h) | (split at h))
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    intro s hs
    try dsimp only at hs
    obtain ⟨x, hx⟩ := mapM_ok_mem (by assumption) s hs
    try dsimp only at hx
    repeat' (first | (obtain ⟨_, _, hx⟩ := ebind_ok hx) | (split at hx))
    all_goals
      simp only [pure, Except.pure, Except.ok.injEq] at hx
      subst hx
      intro r hr'
      obtain ⟨k, y, hg⟩ := mapIdxM_ok_mem (by assumption) r hr'
      try dsimp only at hg
      iterate 9 obtain ⟨_, _, hg⟩ := ebind_ok hg
      obtain ⟨_, hret, hg⟩ := ebind_ok hg
      have hn : Record.isNull y "msg" = false := by
        have := check_ok_iff.1 hret
        rw [hr] at this
        simpa using this
      rw [hn] at hg
      simp only [Bool.false_eq_true, ↓reduceIte] at hg
      obtain ⟨_, _, hg⟩ := ebind_ok hg
      obtain ⟨b, _, hg⟩ := ebind_ok hg
      iterate 3 obtain ⟨_, _, hg⟩ := ebind_ok hg
      obtain ⟨x, hx, hg⟩ := ebind_ok hg
      simp only [pure, Except.pure, Except.ok.injEq] at hx hg
      subst hx hg
      exact ⟨b, rfl⟩

/-- **`decodeWithK_allRetained` at the record's parsed value `j`.** -/
theorem decodeJWithK_allRetained (keep : Bool → Bool) (hs : Hashes Flock.V) {spec : SessionSpec} {j : Json}
    {sess : RunRecord} (hr : spec.retainRounds = true) (h : Record.decodeJWithK keep hs spec j = .ok sess) :
    ∀ s ∈ sess.streams.toList, AllRetained s.rounds := fun s hs' i hi => by
  rw [getElem!_pos s.rounds i hi]
  exact decodeJWithK_retained keep hs hr h s hs' _ (Array.getElem_mem_toList hi)

/-! ## The transfer -/

/-- **The decoder read through `keep` accepts a record text exactly where the record parses and the decoder accepts
its parsed value** (`Record.decodeWithK_of_parse`; a record that doesn't parse is refused). -/
theorem decodeWithK_ok_iff (keep : Bool → Bool) (hs : Hashes Flock.V) (spec : SessionSpec) (text : String)
    (sess : RunRecord) :
    Record.decodeWithK keep hs spec text = .ok sess ↔
      ∃ j, Json.parse text = .ok j ∧ Record.decodeJWithK keep hs spec j = .ok sess := by
  cases hp : Json.parse text with
  | error e =>
    refine ⟨fun h => ?_, fun ⟨_, hj, _⟩ => by cases hj⟩
    unfold Record.decodeWithK at h
    rw [hp] at h
    obtain ⟨_, h0, _⟩ := ebind_ok h
    cases h0
  | ok j =>
    rw [Record.decodeWithK_of_parse keep hs spec hp]
    exact ⟨fun h => ⟨j, rfl, h⟩, fun ⟨_, hj, h⟩ => by cases hj; exact h⟩

/-- **The verdict read through `keep` accepts a record text exactly where the record parses and the verdict accepts
its parsed value** (`verifyWithMK_of_parse`). -/
theorem verifyWithMK_ok_iff (keep : Bool → Bool) (hs : Hashes Flock.V) (spec : SessionSpec) (tabs : Array Setup)
    (record : String) (proofs : Array ByteArray) :
    verifyWithMK keep hs spec tabs record proofs = .ok () ↔
      ∃ j, Json.parse record = .ok j ∧ verifyJWithMK keep hs spec tabs j proofs = .ok () := by
  cases hp : Json.parse record with
  | error e =>
    refine ⟨fun h => ?_, fun ⟨_, hj, _⟩ => by cases hj⟩
    obtain ⟨_, _, hd, -⟩ := verifyWithMK_tables_cap keep hs h
    obtain ⟨_, hj, -⟩ := (decodeWithK_ok_iff keep hs spec record _).1 hd
    rw [hp] at hj
    cases hj
  | ok j =>
    rw [verifyWithMK_of_parse keep hs spec tabs proofs hp]
    exact ⟨fun h => ⟨j, rfl, h⟩, fun ⟨_, hj, h⟩ => by cases hj; exact h⟩

/-- **At `keep := id` and SHA-512, the decoder at a record's parsed value is `Record.decode` on the record.** -/
theorem decode_eq_decodeJWithK_id {spec : SessionSpec} {text : String} {j : Json} (h : Json.parse text = .ok j) :
    Record.decode spec text = Record.decodeJWithK id With.std spec j :=
  (decodeWithK_id_std spec text).symm.trans (Record.decodeWithK_of_parse id With.std spec h)

end FlockSoundness.Discharge.Exec
