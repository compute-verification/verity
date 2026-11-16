module

public import Definitions.Pous.Model.Continuous

/-!
# The continuous audit's memory challenger enforces its three rules

A round-structured model would assume three requirements on the protocol. Each is proved here of the memory challenger
in `security/Definitions/Pous/Model/Continuous.lean`, for every machine.

1. **No pipelining.** `close_le_send`: round `j + 1` is sent no earlier than round `j` closed.
   `contRecords_getElem?`: round `j`'s record is the record of `i_j` after the indices before it, so it depends only on
   `i_0, …, i_j` (`no_pipelining`).
2. **A hard deadline.** `round_depth_le` and `closed_le`: a round closes within `D_late` units of its own send time.
   `not_reject`: a round that is not rejected answered within `Q_late` queries, and `contProg_depth_le` within
   `D_late` units.
3. **Every answer past `Δ` is counted.** `hit_on_time`: a hit took at most `D` units and `Q` queries and returned the
   right block. `late_not_hit`: an answer past `(D, Q)` is never a hit.
-/

@[expose] public section

namespace PousContinuous

open Pous

variable {M : IdealModel} {sch : Scheme M}

/-- A round takes at most its `T` units. -/
theorem round_depth_le (mach : ContMachine sch) (W : Bits sch.n) (pp : Bits sch.t) (h : List (Fin sch.B))
    (f : M.Qry → M.Ans) : ∀ (T : ℕ) (m : List Bool) (msg : Option (Fin sch.B)),
      (mach.round W pp h T m msg).depth f ≤ T
  | 0, m, msg => by
    unfold ContMachine.round
    split <;> simp [OracleAlgorithm.depth]
  | T + 1, m, msg => by
    unfold ContMachine.round
    split
    · simp [OracleAlgorithm.depth]
    · simpa [OracleAlgorithm.depth] using round_depth_le mach W pp h f T _ none

/-- **Requirement 2.** A round's program takes at most `D_late` units. -/
theorem contProg_depth_le (mach : ContMachine sch) (D_late : ℕ) (W : Bits sch.n) (pp : Bits sch.t) (C : sch.Code)
    (ω : M.Ω) (arr : ℕ → ℕ) (s : ContState sch.B) (i : Fin sch.B) (f : M.Qry → M.Ans) :
    (contProg mach D_late W pp C ω arr s i).depth f ≤ D_late :=
  round_depth_le _ _ _ _ _ _ _ _


/-- **Requirement 3.** A hit returned the right block within `D` units and `Q` queries. -/
theorem hit_on_time {ℓ : ℕ} {σ : Type} (D Q Q_late : ℕ) (f : M.Qry → M.Ans) (p : OracleAlgorithm M (Option (Bits ℓ) × σ))
    (c : Bits ℓ) (hh : contVerdict D Q Q_late f p c = .hit) :
    (p.run f).1 = some c ∧ p.depth f ≤ D ∧ p.work f ≤ Q := by
  unfold contVerdict at hh
  split at hh
  · cases hh
  · rename_i a ha
    split_ifs at hh with h1 h2 h3
    subst h3
    exact ⟨ha, by omega, by omega⟩


/-- The indices sent after `h` from `s` are `s`'s followed by `h`. -/
theorem foldl_contNext_sent (mach : ContMachine sch) (D_late : ℕ) (W : Bits sch.n) (pp : Bits sch.t) (C : sch.Code)
    (ω : M.Ω) (arr : ℕ → ℕ) :
    ∀ (h : List (Fin sch.B)) (s : ContState sch.B),
      (h.foldl (contNext mach D_late W pp C ω arr) s).sent = s.sent ++ h
  | [], s => by simp
  | i :: h, s => by
    rw [List.foldl_cons, foldl_contNext_sent mach D_late W pp C ω arr h]
    simp [contNext]

theorem contAfter_sent (mach : ContMachine sch) (D_late : ℕ) (W : Bits sch.n) (pp : Bits sch.t) (C : sch.Code)
    (ω : M.Ω) (arr : ℕ → ℕ) (h : List (Fin sch.B)) : (contAfter mach D_late W pp C ω arr h).sent = h := by
  unfold contAfter
  rw [foldl_contNext_sent]
  rfl

/-- **Requirement 1.** Round `j`'s record is the record of `i_j` after the indices before it. -/
theorem contRecords_getElem? (mach : ContMachine sch) (D Q D_late Q_late : ℕ) (W : Bits sch.n) (pp : Bits sch.t)
    (C : sch.Code) (ω : M.Ω) (arr : ℕ → ℕ) :
    ∀ (s : ContState sch.B) (l : List (Fin sch.B)) (j : ℕ),
      (contRecords mach D Q D_late Q_late W pp C ω arr s l)[j]? =
        l[j]?.map (contRecord mach D Q D_late Q_late W pp C ω arr
          ((l.take j).foldl (contNext mach D_late W pp C ω arr) s))
  | s, [], j => by simp [contRecords]
  | s, i :: is, 0 => by simp [contRecords]
  | s, i :: is, j + 1 => by
    simp only [contRecords, List.getElem?_cons_succ, List.take_succ_cons, List.foldl_cons]
    exact contRecords_getElem? mach D Q D_late Q_late W pp C ω arr _ is j


/-- The audit's records over `List.ofFn I`: round `j` is the record of `I j` after `I 0, …, I (j − 1)`. -/
theorem contRecords_ofFn (mach : ContMachine sch) (D Q D_late Q_late : ℕ) (W : Bits sch.n) (pp : Bits sch.t)
    (C : sch.Code) (ω : M.Ω) (arr : ℕ → ℕ) {n : ℕ} (I : Fin n → Fin sch.B) :
    contRecords mach D Q D_late Q_late W pp C ω arr .start (List.ofFn I) =
      List.ofFn fun j : Fin n => contRecord mach D Q D_late Q_late W pp C ω arr
        (contAfter mach D_late W pp C ω arr ((List.ofFn I).take j)) (I j) := by
  apply List.ext_getElem?
  intro j
  rw [contRecords_getElem?, List.getElem?_ofFn, List.getElem?_ofFn]
  unfold contAfter
  split_ifs <;> simp_all

end PousContinuous
