import Proofs.Flock.C3.Reps.RepsCounted
import Proofs.Flock.C3.Compose.HoloEncoded

/-!
# `DrawnOf` at the verifier's map, from the reps' coin rounds

A rep's round `q` of coins takes one history round, so the reps' held rounds before `q` (`heldBefore`) count the
schedule's coin rounds before it (`heldBefore_rep`): all of them at the schedule's end (`heldBefore_rep_size`), and `p`
at the `p`-th (`heldBefore_of_coinRound`, by `filter_range_getElem?`). `holoSelOne_coinRound` puts rep `i`'s `p`-th
coin round at history round `repStart c₀ i + p`, and `drawnOf_holoMap` is `DrawnOf` at `holoMap`'s draws where each
rep's consumed rounds are the schedule's coin rounds (`repIdxB_of_size`, `repIdxB_of_rounds`).
-/

namespace FlockC3.Reps

open Flock Flock.RecVStar Flock.Guarantees FlockSoundness
open FlockSoundness.Discharge.Recursion (VStmt HoloParams)

theorem sum_min_one (f : ℕ → ℕ) :
    ∀ l : List ℕ, (l.map fun q => min (f q) 1).sum = (l.filter fun q => 0 < f q).length
  | [] => rfl
  | a :: l => by
    rw [List.map_cons, List.sum_cons, sum_min_one f l]
    rcases Nat.eq_zero_or_pos (f a) with h | h
    · have hl : (a :: l).filter (fun q => 0 < f q) = l.filter fun q => 0 < f q :=
        List.filter_cons_of_neg (by simp [h])
      rw [hl, h, Nat.zero_min, Nat.zero_add]
    · have hl : (a :: l).filter (fun q => 0 < f q) = a :: l.filter fun q => 0 < f q :=
        List.filter_cons_of_pos (by simp [h])
      have hm : min (f a) 1 = 1 := Nat.min_eq_right h
      rw [hl, List.length_cons, hm, Nat.add_comm]

/-- **The `p`-th of the first `n` numbers passing `P` is `q`**: then `q < n`, `q` passes, and `p` of them are below
`q`. -/
theorem filter_range_getElem? (P : ℕ → Bool) : ∀ (n p q : ℕ), ((List.range n).filter P)[p]? = some q →
    q < n ∧ P q = true ∧ ((List.range q).filter P).length = p
  | 0, p, q, h => by simp at h
  | n + 1, p, q, h => by
    rw [List.range_succ, List.filter_append] at h
    by_cases hp : p < ((List.range n).filter P).length
    · rw [List.getElem?_append_left hp] at h
      obtain ⟨hq, hP, hl⟩ := filter_range_getElem? P n p q h
      exact ⟨Nat.lt_succ_of_lt hq, hP, hl⟩
    · rw [List.getElem?_append_right (Nat.le_of_not_lt hp)] at h
      cases hn : P n
      · have hl : List.filter P [n] = [] := List.filter_cons_of_neg (by simp [hn])
        rw [hl] at h
        simp at h
      · have hl : List.filter P [n] = [n] := List.filter_cons_of_pos hn
        rw [hl] at h
        obtain ⟨h1, -⟩ := List.getElem?_eq_some_iff.1 h
        have h0 : p - ((List.range n).filter P).length = 0 := by
          have h1' : p - ((List.range n).filter P).length < 1 := h1
          omega
        rw [h0, List.getElem?_cons_zero, Option.some.injEq] at h
        rw [← h]
        exact ⟨Nat.lt_succ_self n, hn, by omega⟩

/-- **A rep's held rounds before `k` are its schedule's coin rounds before `k`**: each round of coins takes one. -/
theorem heldBefore_rep (i : Fin 2) (rs : Array SchedRound) (k : ℕ) :
    heldBefore (.rep i) rs k = ((List.range (min k rs.size)).filter fun q => 0 < (rs.getD q default).1).length := by
  unfold heldBefore
  exact sum_min_one (fun q => (rs.getD q default).1) _

theorem heldBefore_rep_size (i : Fin 2) (rs : Array SchedRound) :
    heldBefore (.rep i) rs rs.size = (coinRounds rs).length := by
  unfold coinRounds
  rw [heldBefore_rep, Nat.min_self]

/-- **The `p`-th coin round has `p` held rounds before it**, and draws coins. -/
theorem heldBefore_of_coinRound (i : Fin 2) {rs : Array SchedRound} {p q : ℕ} (h : (coinRounds rs)[p]? = some q) :
    heldBefore (.rep i) rs q = p ∧ 0 < (rs.getD q default).1 := by
  obtain ⟨hq, hP, hl⟩ := filter_range_getElem? (fun k => 0 < (rs.getD k default).1) rs.size p q h
  have hm : min q rs.size = q := Nat.min_eq_left (Nat.le_of_lt hq)
  refine ⟨?_, of_decide_eq_true hP⟩
  rw [heldBefore_rep, hm]
  exact hl

/-- **Rep `i`'s `p`-th coin round is drawn at history round `repStart c₀ i + p`**, `c₀` the schedule's coin rounds,
where the session's streams are both reps' rounds `rs` first (`holoStreams`' order). -/
theorem holoSelOne_coinRound {ish : RecVStar.Inner} {tbl : String} {rs : Array SchedRound}
    {L : List (HoloStream × Array SchedRound)}
    (hS : holoStreams ish.m ish.kLog ish.claims ish.sparse ish.stacked = some ((.rep 0, rs) :: (.rep 1, rs) :: L))
    (i : Fin 2) {p q : ℕ} (h : (coinRounds rs)[p]? = some q) :
    FlockC3.Compose.holoSelOne ish tbl (repStart (coinRounds rs).length i + p) ((HoloStream.rep i).name tbl) q := by
  obtain ⟨hp, hq⟩ := heldBefore_of_coinRound i h
  have hs := FlockC3.Compose.holoSelOne_repName (tbl := tbl) (i := i) hS hq
  rwa [heldBefore_rep_size, hp] at hs

/-- **`DrawnOf` at `holoMap`'s draws**: where each rep's consumed rounds are the reps' schedule's coin rounds (`hb`),
the session's streams are both reps' first, and the history has room for both reps' rounds (`hr`). -/
theorem drawnOf_holoMap {S : ℕ} {Vs : Fin S → VStmt} {st : Flock.Stmt} {hp : HoloParams} {Cm : Type}
    {ish : RecVStar.Inner} {tbl : String} {bc : Cm → String → ℕ → List UInt8} {rs : Array SchedRound}
    {L : List (HoloStream × Array SchedRound)}
    (hS : holoStreams ish.m ish.kLog ish.claims ish.sparse ish.stacked = some ((.rep 0, rs) :: (.rep 1, rs) :: L))
    {S₀ : Model.Statement} {b : ((j : Fin S) → (Vs j).Bits) → Fin 2 → ValsSrc}
    (hb : ∀ Lb i, repIdxB S₀ (b Lb i) = coinRounds rs) {r : ℕ} (hr : 1 + 2 * (coinRounds rs).length ≤ r) :
    DrawnOf S₀ r b (holoMap st hp Cm ish tbl bc).selC fun i => (HoloStream.rep i).name tbl := by
  intro Lb i
  rw [hb Lb 0, hb Lb i]
  refine ⟨?_, fun p q h => FlockC3.Compose.selC_of_holoSelOne (holoSelOne_coinRound hS i h)⟩
  unfold repStart
  split <;> omega

end FlockC3.Reps

#print axioms FlockC3.Reps.sum_min_one
#print axioms FlockC3.Reps.filter_range_getElem?
#print axioms FlockC3.Reps.heldBefore_rep
#print axioms FlockC3.Reps.heldBefore_rep_size
#print axioms FlockC3.Reps.heldBefore_of_coinRound
#print axioms FlockC3.Reps.holoSelOne_coinRound
#print axioms FlockC3.Reps.drawnOf_holoMap
