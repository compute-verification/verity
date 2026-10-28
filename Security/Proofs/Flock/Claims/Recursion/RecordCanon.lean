import Proofs.Flock.Claims.Recursion.VerdictRows
import Proofs.Flock.Claims.Bridge.Custody

/-!
# Row B at the decision: what the record's decode gives

Row B (`RecDeploy.Rows.RowB`) asks four things of rep `r` of table `i` at a decision of the last step. Three of them
are what the verdict's decoder checks of every record it returns (`DecAt.decode`), whatever the record's value:

* **S11 finds the rep's stream** (`DecAt.finds`): the decoder refuses a record without a stream at each rep stream's
  name (`decodeJWith_finds`), and table `i`'s rep `r` is the rep stream `<t>/rep<r>` (`repStreams_getElem`).
* **S12 makes its rounds well formed and none of its coin rounds empty** (`DecAt.roundsWF`, `DecAt.noEmpty`): every
  decoded round has `1 ≤ n` and exactly `n` coins (`decodeJWith_rounds`).

The fourth, `framed` (the rounds hold the rep's frames), is about the record's values, not its form: it holds where
each round retained its chunk's bytes and counts its chunk's closing squeeze (`framedAt_of_rounds`).
-/

namespace RecDeploy.Rows

open Flock
open FlockDeploy (Coin FinalP)
open FlockSoundness.Refine (ebind_ok check_ok_iff mapM_ok_mem mapIdxM_ok_mem roundsWF_of RoundsWF)
open Flock.RecAlgebra (NoEmpty)
open FlockSoundness.Discharge.Recursion (HClass)
open Lean (Json)

/-! ## What the decoder checks -/

theorem check_round {n : ℕ} {coins : Array F128} {b c : Bool} {why : String} {u : Unit}
    (h : Record.check (decide (1 ≤ n) && b && coins.size == n && c) why = .ok u) : 1 ≤ n ∧ coins.size = n := by
  have := check_ok_iff.1 h
  simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at this
  exact ⟨this.1.1.1, this.1.2⟩

set_option maxHeartbeats 1000000 in
/-- **S12: every decoded round has at least one coin, and as many as its count.** -/
theorem decodeJWith_rounds (hs : Hashes Flock.V) {spec : SessionSpec} {j : Json} {sess : RunRecord}
    (h : Record.decodeJWith hs spec j = .ok sess) :
    ∀ s ∈ sess.streams.toList, ∀ rd ∈ s.rounds.toList, 1 ≤ rd.n ∧ rd.coins.size = rd.n := by
  rw [Record.decodeJWith_eq_decodeJWithK] at h
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
      intro rd hrd
      obtain ⟨k, y, hg⟩ := mapIdxM_ok_mem (by assumption) rd hrd
      try dsimp only at hg
      repeat' (first | (obtain ⟨_, _, hg⟩ := ebind_ok hg) | (split at hg))
      all_goals
        simp only [pure, Except.pure, Except.ok.injEq] at hg
        subst hg
        exact check_round (by assumption)

theorem find_of_any {ss : Array StreamRec} {w : String} (h : ss.any (·.name == w) = true) :
    ∃ s, ss.find? (·.name == w) = some s := by
  cases hf : ss.find? (·.name == w) with
  | some s => exact ⟨s, rfl⟩
  | none =>
    rw [Array.find?_eq_none] at hf
    obtain ⟨s, hs, hw⟩ := Array.any_eq_true'.1 h
    exact absurd hw (hf s hs)

set_option maxHeartbeats 1000000 in
/-- **S11: the decoder finds a stream at each rep stream's name.** -/
theorem decodeJWith_finds (hs : Hashes Flock.V) {spec : SessionSpec} {j : Json} {sess : RunRecord}
    (h : Record.decodeJWith hs spec j = .ok sess) :
    ∀ x ∈ spec.repStreams.toList, ∃ s, sess.streams.find? (·.name == x.2.2) = some s := by
  rw [Record.decodeJWith_eq_decodeJWithK] at h
  unfold Record.decodeJWithK at h
  dsimp only at h
  repeat' (first | (obtain ⟨_, _, h⟩ := ebind_ok h) | (split at h))
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    intro x hx
    have h11 := check_ok_iff.1 ‹Record.check (_ == spec.repStreams.size && _) _ = _›
    simp only [Bool.and_eq_true, Array.all_eq_true_iff_forall_mem] at h11
    obtain ⟨i, r, w⟩ := x
    exact find_of_any (h11.2 _ (Array.mem_toList_iff.1 hx))

/-! ## Frames from the rounds' values -/

open FlockSoundness.Refine (Frame encs Retains) in
open FlockSoundness.Discharge.Exec (chunks) in
open FlockVBridge.Family (FramedAt sqCount) in
/-- **The rounds hold the frames** where round `k` retained chunk `k`'s bytes and counts its closing squeeze's coins. -/
theorem framedAt_of_rounds {rs : Array Round} {fs : List Frame}
    (hb : ∀ k < (chunks fs).1.length, (rs[k]!).bytes = some ⟨(encs ((chunks fs).1.getD k [])).toArray⟩)
    (hn : ∀ k < (chunks fs).1.length, (rs[k]!).n = sqCount ((chunks fs).1.getD k [])) :
    FramedAt rs 0 fs := by
  intro k hk
  rw [Nat.zero_add]
  refine ⟨?_, hn k hk⟩
  unfold Retains
  rw [hb k hk]
  simp

/-! ## Row B's form at a decision -/

namespace DecAt

variable {rd : Claims.Msg → Option FinalP} {x : Claims.Msg} {ms : List Claims.Msg} {cs : List Coin}
  (D : DecAt rd x ms cs) (cls : HClass) {i : ℕ}

/-- **S11 finds rep `r`'s stream** of a table of the statement. -/
theorem finds (hi : i < D.spec.tables.size) (r : Fin 2) : (D.files cls i).Finds r := by
  obtain ⟨hq, e⟩ := Claims.Bridge.Custody.repStreams_getElem D.spec hi r.isLt
  obtain ⟨s, hs⟩ := decodeJWith_finds _ D.decode _ (Array.getElem_mem_toList hq)
  rw [e] at hs
  dsimp only at hs
  show D.sess.streams.find? (fun y => y.name == toString D.spec.tables[i]! ++ toString "/rep" ++ toString (r : ℕ)) =
    some ((D.sess.streams.find? fun y => y.name == toString D.spec.tables[i]! ++ toString "/rep" ++
      toString (r : ℕ)).getD default)
  rw [hs]
  rfl

theorem stream_mem (hi : i < D.spec.tables.size) (r : Fin 2) :
    (D.files cls i).stream r ∈ D.sess.streams.toList :=
  Array.mem_toList_iff.2 (Array.mem_of_find?_eq_some (D.finds cls hi r))

/-- **S12: rep `r`'s rounds are well formed.** -/
theorem roundsWF (hi : i < D.spec.tables.size) (r : Fin 2) : RoundsWF ((D.files cls i).stream r).rounds :=
  roundsWF_of fun rd h => (decodeJWith_rounds _ D.decode _ (D.stream_mem cls hi r) rd h).2

/-- **S12: none of rep `r`'s coin rounds is empty.** -/
theorem noEmpty (acc : ℕ → Array F128) (hi : i < D.spec.tables.size) (r : Fin 2) :
    NoEmpty (D.rep cls acc i r).rounds := by
  intro c hc
  change c ∈ ((D.files cls i).stream r).rounds.map (·.coins) at hc
  obtain ⟨rd, hrd, rfl⟩ := Array.mem_map.1 hc
  obtain ⟨h1, h2⟩ := decodeJWith_rounds _ D.decode _ (D.stream_mem cls hi r) rd (Array.mem_toList_iff.2 hrd)
  show rd.coins.size ≠ 0
  omega

end DecAt

end RecDeploy.Rows
