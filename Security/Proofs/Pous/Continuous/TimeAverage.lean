import Mathlib.Data.Fintype.Fin
import Proofs.Pous.Continuous.PerAnswer

/-!
# The continuous audit's time-averaged bound: `Pous.Guarantees.ContinuousTimeAverage`

The proof, pinned as `Pous.SecurityProofs.ContinuousTimeAverage`. The memory challenger and the
responder are `Security/Definitions/Pous/Model/Continuous.lean`'s; the memory challenger draws one coin and one
index per slot, and a challenge arrives at slot `t` iff its coin is below `a`.

1. **The arrivals before `t`** (`pre`) are a prefix of all the arrivals, because the arrivals are sorted
   (`arrivals_split`). If a challenge arrives at `t`, it is the next one (`arrAt_pre`, `histAt_get`).
2. **The view at slot `t`.** Replace the arrival schedule by `arrS a x t`: the arrivals before `t`, then `t`. The state
   after the challenges that arrived before `t` is the same under both (`contAfter_slot`), and it depends only on the
   draws before `t` (`pre_congr`, `histAt_congr`). This is where queueing matters: the indices sent before the
   challenge that arrives at `t` are `histAt a x t`, whatever the rounds did.
3. **The bridge.** A challenge arriving at `t` is challenge `|pre a x t|`, and its record is the record of index
   `(x t).2` in the view at `t` (`record_slot`). At an idle slot the view's memory at the send time is `contHeldAt`
   (`contMem_slot`). The arriving slots inject into the records (`misses_le`), so an accepting run has at most `f`
   arriving slots that miss.
4. **One slot.** In the view at `t` the history `histAt a x t` and the schedule are fixed by the earlier draws, so
   `PerAnswer.lean`'s `pr_bad_le` and `miss_ge` apply as they are: off `bad`, the slot's draw `(c, i)` is an arrival
   that misses with probability at least `(a/b)·(ε_t − ε_proof)` (`slotQ_le`, with `pr_coin` and `pr_and_prod`).
5. **Counting** (`pr_misses_gt_le`) over the `T` slot draws, with `q_t = (a/b)·max(ε_t − ε_proof, 0)` at an idle slot
   off `bad` (`slotQ_eq`) and `0` elsewhere; the history `pr_misses_gt_le` indexes by is the list of earlier draws,
   read back as a schedule by `pad`. The stopped union over the `T` slots costs `T·(β + 1)·π`
   (`continuous_time_average`).
-/

namespace PousContinuous

open Pous PousThm1
open scoped ENNReal

/-! ## The arrivals before a time -/

section Arrivals

variable {T b B : ℕ}

/-- The arrivals before time `t`. -/
def pre (a : ℕ) (x : Fin T → Fin b × Fin B) (t : ℕ) : List (Fin T) :=
  (arrivals a x).filter fun s : Fin T => (s : ℕ) < t

/-- The arrival schedule as seen at time `t`: the arrivals before `t`, then `t` for every later challenge. -/
def arrS (a : ℕ) (x : Fin T → Fin b × Fin B) (t : ℕ) : ℕ → ℕ :=
  fun j => ((pre a x t).map Fin.val).getD j t

theorem histAt_eq (a : ℕ) (x : Fin T → Fin b × Fin B) (t : ℕ) :
    histAt a x t = (pre a x t).map fun s => (x s).2 := rfl

theorem length_histAt (a : ℕ) (x : Fin T → Fin b × Fin B) (t : ℕ) :
    (histAt a x t).length = (pre a x t).length := by
  rw [histAt_eq, List.length_map]

theorem arrS_length (a : ℕ) (x : Fin T → Fin b × Fin B) (t : ℕ) : arrS a x t (pre a x t).length = t := by
  simp [arrS, List.getD_eq_getElem?_getD]

/-- A sorted list is its elements below `t` followed by the rest. -/
theorem sorted_split {l : List (Fin T)} (hl : l.Pairwise (· < ·)) (t : ℕ) :
    l = l.filter (fun s : Fin T => (s : ℕ) < t) ++ l.filter (fun s : Fin T => ¬ (s : ℕ) < t) := by
  induction l with
  | nil => rfl
  | cons s l ih =>
    rw [List.pairwise_cons] at hl
    by_cases hs : (s : ℕ) < t
    · rw [List.filter_cons_of_pos (by simpa using hs), List.filter_cons_of_neg (by simpa using hs),
        List.cons_append]
      exact congrArg (s :: ·) (ih hl.2)
    · have hy : ∀ y ∈ l, ¬ (y : ℕ) < t := fun y hy => by
        have := Fin.lt_def.1 (hl.1 y hy)
        omega
      have h1 : l.filter (fun s : Fin T => (s : ℕ) < t) = [] :=
        List.filter_eq_nil_iff.2 fun y hm => by simpa using hy y hm
      have h2 : l.filter (fun s : Fin T => ¬ (s : ℕ) < t) = l :=
        List.filter_eq_self.2 fun y hm => by simpa using hy y hm
      rw [List.filter_cons_of_neg (by simpa using hs), List.filter_cons_of_pos (by simpa using hs), h1, h2,
        List.nil_append]

/-- In a sorted list that contains `t`, the first element not below `t` is `t`. -/
theorem sorted_head {l : List (Fin T)} (hl : l.Pairwise (· < ·)) (t : Fin T) (ht : t ∈ l) :
    (l.filter fun s : Fin T => ¬ (s : ℕ) < t)[0]? = some t := by
  induction l with
  | nil => simp at ht
  | cons s l ih =>
    rw [List.pairwise_cons] at hl
    by_cases hs : (s : ℕ) < t
    · have htl : t ∈ l := by
        rcases List.mem_cons.1 ht with h | h
        · subst h; omega
        · exact h
      rw [List.filter_cons_of_neg (by simpa using hs)]
      exact ih hl.2 htl
    · have hst : s = t := by
        rcases List.mem_cons.1 ht with h | h
        · exact h.symm
        · have := Fin.lt_def.1 (hl.1 t h)
          omega
      rw [List.filter_cons_of_pos (by simpa using hs), hst]
      rfl

theorem arrivals_sorted (a : ℕ) (x : Fin T → Fin b × Fin B) : (arrivals a x).Pairwise (· < ·) :=
  (List.pairwise_lt_finRange T).filter _

theorem mem_arrivals {a : ℕ} {x : Fin T → Fin b × Fin B} {t : Fin T} (ht : ((x t).1 : ℕ) < a) :
    t ∈ arrivals a x := by
  simp [arrivals, ht]

/-- The arrivals are those before `t`, then the rest. -/
theorem arrivals_split (a : ℕ) (x : Fin T → Fin b × Fin B) (t : ℕ) :
    arrivals a x = pre a x t ++ (arrivals a x).filter fun s : Fin T => ¬ (s : ℕ) < t :=
  sorted_split (arrivals_sorted a x) t

/-- Before the arrivals at or after `t`, the two schedules agree. -/
theorem arrAt_lt (a : ℕ) (x : Fin T → Fin b × Fin B) (t j : ℕ) (hj : j < (pre a x t).length) :
    arrAt a x j = arrS a x t j := by
  have hj' : j < ((pre a x t).map Fin.val).length := by rwa [List.length_map]
  unfold arrAt arrS
  rw [arrivals_split a x t, List.map_append, List.getD_eq_getElem?_getD, List.getElem?_append_left hj',
    List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hj']
  rfl

/-- **A challenge arriving at `t` is the next one**: its arrival time is `t`. -/
theorem arrAt_pre (a : ℕ) (x : Fin T → Fin b × Fin B) (t : Fin T) (ht : ((x t).1 : ℕ) < a) :
    arrAt a x (pre a x t).length = t := by
  unfold arrAt
  rw [arrivals_split a x t, List.map_append, List.getD_eq_getElem?_getD,
    List.getElem?_append_right (by rw [List.length_map]), List.length_map, Nat.sub_self, List.getElem?_map,
    sorted_head (arrivals_sorted a x) t (mem_arrivals ht)]
  rfl

theorem histAt_T (a : ℕ) (x : Fin T → Fin b × Fin B) : histAt a x T = (arrivals a x).map fun s => (x s).2 := by
  unfold histAt
  congr 1
  exact List.filter_eq_self.2 fun s _ => by simp [s.isLt]

theorem histAt_T_split (a : ℕ) (x : Fin T → Fin b × Fin B) (t : ℕ) :
    histAt a x T = histAt a x t ++ ((arrivals a x).filter fun s : Fin T => ¬ (s : ℕ) < t).map fun s => (x s).2 := by
  rw [histAt_T]
  conv_lhs => rw [arrivals_split a x t]
  rw [List.map_append]
  rfl

/-- The indices sent before the challenge that arrives at `t` are `histAt a x t`. -/
theorem take_histAt (a : ℕ) (x : Fin T → Fin b × Fin B) (t : ℕ) :
    (histAt a x T).take (pre a x t).length = histAt a x t := by
  rw [histAt_T_split a x t]
  exact List.take_left' (length_histAt a x t)

/-- The challenge that arrives at `t` carries the index drawn at `t`. -/
theorem histAt_get (a : ℕ) (x : Fin T → Fin b × Fin B) (t : Fin T) (ht : ((x t).1 : ℕ) < a) :
    (histAt a x T)[(pre a x t).length]? = some (x t).2 := by
  rw [histAt_T_split a x t, List.getElem?_append_right (by rw [length_histAt]), length_histAt, Nat.sub_self,
    List.getElem?_map, sorted_head (arrivals_sorted a x) t (mem_arrivals ht)]
  rfl

/-- **The arrivals before `t` depend only on the draws before `t`.** -/
theorem pre_congr (a : ℕ) {x x' : Fin T → Fin b × Fin B} {t : ℕ}
    (hx : ∀ s : Fin T, (s : ℕ) < t → x s = x' s) : pre a x t = pre a x' t := by
  unfold pre arrivals
  rw [List.filter_filter, List.filter_filter]
  refine List.filter_congr fun s _ => ?_
  by_cases hs : (s : ℕ) < t <;> simp [hs, hx]

theorem histAt_congr (a : ℕ) {x x' : Fin T → Fin b × Fin B} {t : ℕ}
    (hx : ∀ s : Fin T, (s : ℕ) < t → x s = x' s) : histAt a x t = histAt a x' t := by
  rw [histAt_eq, histAt_eq, pre_congr a hx]
  refine List.map_congr_left fun s hs => ?_
  have hst : (s : ℕ) < t := by
    unfold pre at hs
    simpa using (List.mem_filter.1 hs).2
  rw [hx s hst]

theorem arrS_congr (a : ℕ) {x x' : Fin T → Fin b × Fin B} {t : ℕ}
    (hx : ∀ s : Fin T, (s : ℕ) < t → x s = x' s) : arrS a x t = arrS a x' t := by
  unfold arrS
  rw [pre_congr a hx]

end Arrivals

/-! ## The memory challenger reads the schedule only up to the next challenge -/

section Congr

variable {M : IdealModel} {sch : Scheme M} (mach : ContMachine sch) (D Q D_late Q_late : ℕ) (W : Bits sch.n)
  (pp : Bits sch.t) (C : sch.Code) (ω : M.Ω)

theorem contAfter_append (arr : ℕ → ℕ) (h : List (Fin sch.B)) (i : Fin sch.B) :
    contAfter mach D_late W pp C ω arr (h ++ [i]) =
      contNext mach D_late W pp C ω arr (contAfter mach D_late W pp C ω arr h) i := by
  unfold contAfter
  rw [List.foldl_append]
  rfl

/-- The state after `h` reads the schedule only at the challenges of `h`. -/
theorem contAfter_congr {arr arr' : ℕ → ℕ} (h : List (Fin sch.B)) (hag : ∀ j < h.length, arr j = arr' j) :
    contAfter mach D_late W pp C ω arr h = contAfter mach D_late W pp C ω arr' h := by
  induction h using List.reverseRecOn with
  | nil => rfl
  | append_singleton h i ih =>
    have hl : arr (contAfter mach D_late W pp C ω arr' h).sent.length =
        arr' (contAfter mach D_late W pp C ω arr' h).sent.length := by
      rw [contAfter_sent]
      exact hag _ (by simp)
    rw [contAfter_append, contAfter_append, ih fun j hj => hag j (by simp; omega)]
    unfold contNext contProg contMem contSend
    rw [hl]

/-- The record of the next round reads the schedule only at its own challenge. -/
theorem contRecord_congr {arr arr' : ℕ → ℕ} (s : ContState sch.B) (i : Fin sch.B)
    (hl : arr s.sent.length = arr' s.sent.length) :
    contRecord mach D Q D_late Q_late W pp C ω arr s i = contRecord mach D Q D_late Q_late W pp C ω arr' s i := by
  unfold contRecord contProg contMem contSend
  rw [hl]

theorem length_contRecords (arr : ℕ → ℕ) :
    ∀ (s : ContState sch.B) (l : List (Fin sch.B)),
      (contRecords mach D Q D_late Q_late W pp C ω arr s l).length = l.length
  | _, [] => rfl
  | s, i :: l => by simp [contRecords, length_contRecords arr _ l]

end Congr

/-! ## The view at slot `t` -/

section Slot

variable {M : IdealModel} {sch : Scheme M} (mach : ContMachine sch) (D Q D_late Q_late : ℕ) (W : Bits sch.n)
  (pp : Bits sch.t) (C : sch.Code) (ω : M.Ω) (a : ℕ) {T b : ℕ} (x : Fin T → Fin b × Fin sch.B)

/-- The state after the challenges that arrived before `t` is the same in the view at `t`. -/
theorem contAfter_slot (t : ℕ) :
    contAfter mach D_late W pp C ω (arrAt a x) (histAt a x t) =
      contAfter mach D_late W pp C ω (arrS a x t) (histAt a x t) :=
  contAfter_congr mach D_late W pp C ω _ fun j hj => arrAt_lt a x t j (by rwa [length_histAt] at hj)

theorem contIdle_iff (t : ℕ) :
    contIdle mach D_late W pp C ω a x t ↔
      (contAfter mach D_late W pp C ω (arrS a x t) (histAt a x t)).closed ≤ t := by
  unfold contIdle
  rw [contAfter_slot]

/-- **At an idle slot, the view's memory at the send time is the memory at `t`.** -/
theorem contMem_slot (t : ℕ) (hi : contIdle mach D_late W pp C ω a x t) :
    (contMem mach W pp C ω (arrS a x t) (contAfter mach D_late W pp C ω (arrS a x t) (histAt a x t))).length =
      contHeldAt mach D_late W pp C ω a x t := by
  rw [contIdle_iff] at hi
  unfold contMem contSend contHeldAt
  simp only [contAfter_sent, length_histAt, arrS_length, contAfter_slot]
  rw [max_eq_left hi]

/-- **The bridge.** The challenge that arrives at `t` is round `|pre a x t|`, and its record is the record of index
`(x t).2` in the view at `t`. -/
theorem record_slot (t : Fin T) (ht : ((x t).1 : ℕ) < a) :
    (contRecords mach D Q D_late Q_late W pp C ω (arrAt a x) .start (histAt a x T))[(pre a x t).length]? =
      some (contRecord mach D Q D_late Q_late W pp C ω (arrS a x t)
        (contAfter mach D_late W pp C ω (arrS a x t) (histAt a x t)) (x t).2) := by
  rw [contRecords_getElem?, histAt_get a x t ht, take_histAt, Option.map_some]
  congr 1
  change contRecord mach D Q D_late Q_late W pp C ω (arrAt a x)
    (contAfter mach D_late W pp C ω (arrAt a x) (histAt a x t)) (x t).2 = _
  rw [contAfter_slot]
  apply contRecord_congr
  rw [contAfter_sent, length_histAt, arrS_length, arrAt_pre a x t ht]

theorem length_filter_eq_card {α : Type} (l : List α) (p : α → Prop) [DecidablePred p] :
    (l.filter fun r => decide (p r)).length =
      ((Finset.range l.length).filter fun j => ∃ r ∈ l[j]?, p r).card := by
  conv_lhs => rw [← List.ofFn_getElem (xs := l)]
  rw [length_filter_ofFn, ← Nat.Iio_eq_range, ← Fin.map_valEmbedding_univ, Finset.filter_map, Finset.card_map]
  congr 1
  refine Finset.filter_congr fun j _ => ?_
  simp

/-- **The arriving slots that miss are at most the rounds that miss.** -/
theorem misses_le :
    (Finset.univ.filter fun t : Fin T => ((x t).1 : ℕ) < a ∧
        (contRecord mach D Q D_late Q_late W pp C ω (arrS a x t)
          (contAfter mach D_late W pp C ω (arrS a x t) (histAt a x t)) (x t).2).verdict ≠ .hit).card ≤
      ((contRecords mach D Q D_late Q_late W pp C ω (arrAt a x) .start (histAt a x T)).filter
        fun r => r.verdict ≠ .hit).length := by
  rw [length_filter_eq_card]
  refine Finset.card_le_card_of_injOn (fun t => (pre a x t).length) (fun t ht => ?_) (fun t₁ ht₁ t₂ ht₂ he => ?_)
  · obtain ⟨ha, hn⟩ := (Finset.mem_filter.1 ht).2
    have hr := record_slot mach D Q D_late Q_late W pp C ω a x t ha
    refine Finset.mem_filter.2 ⟨Finset.mem_range.2 ?_, _, hr ▸ rfl, hn⟩
    exact (List.getElem?_eq_some_iff.1 hr).1
  · have h₁ := arrAt_pre a x t₁ (Finset.mem_filter.1 ht₁).2.1
    have h₂ := arrAt_pre a x t₂ (Finset.mem_filter.1 ht₂).2.1
    exact Fin.ext (h₁.symm.trans ((congrArg (arrAt a x) he).trans h₂))

end Slot

/-! ## One slot's draw -/

/-- The coin is below `a` with probability `a/b`. -/
theorem pr_coin (a b : ℕ) (hab : a ≤ b) : (pr fun c : Fin b => (c : ℕ) < a).toReal = a / b := by
  rw [pr_toReal, Fin.card_filter_val_lt, min_eq_right hab, Fintype.card_fin]

/-- The two parts of a slot's draw are independent. -/
theorem pr_and_prod {α β : Type} [Fintype α] [Fintype β] (P : α → Prop) (R : β → Prop) :
    (pr fun d : α × β => P d.1 ∧ R d.2).toReal = (pr P).toReal * (pr R).toReal := by
  classical
  have hf : (Finset.univ.filter fun d : α × β => P d.1 ∧ R d.2) = Finset.univ.filter P ×ˢ Finset.univ.filter R := by
    ext d
    simp
  rw [pr_toReal, pr_toReal, pr_toReal, hf, Finset.card_product, Fintype.card_prod, Nat.cast_mul, Nat.cast_mul,
    mul_div_mul_comm]

/-- A history of draws, oldest first, as a draw at every slot: `d₀` past its end. -/
def pad {α : Type} {T : ℕ} (d₀ : α) (h : List α) : Fin T → α := fun s => h.getD s d₀

theorem pad_take {α : Type} {T : ℕ} (d₀ : α) (y : Fin T → α) (j : ℕ) (s : Fin T) (hs : (s : ℕ) < j) :
    pad d₀ ((List.ofFn y).take j) s = y s := by
  simp [pad, List.getD_eq_getElem?_getD, List.getElem?_take_of_lt hs]

/-! ## The slot's verdict, its bad event and its share of the bound, in the view at `t` -/

section SlotQ

variable {M : IdealModel} (sch : Scheme M) (D Q D_late Q_late β : ℕ) (W : Bits sch.n) (mach : ContMachine sch)
  (a : ℕ) {T b : ℕ}

/-- In the view at `t`, a challenge with index `i` arriving at `t` is a hit. -/
def slotHit (u : M.Ω × sch.Coins) (x : Fin T → Fin b × Fin sch.B) (t : ℕ) (i : Fin sch.B) : Prop :=
  hitAt sch D Q D_late Q_late W mach (arrS a x t) u (histAt a x t) i

/-- In the view at `t`, the memory and the non-hits fit in `β` bits. -/
def slotBad (u : M.Ω × sch.Coins) (x : Fin T → Fin b × Fin sch.B) (t : ℕ) : Prop :=
  bad sch D Q D_late Q_late W mach (arrS a x t) β u (histAt a x t)

open Classical in
/-- The slot's share of the bound: `(a/b)·max(ε_t − ε_proof, 0)` at an idle slot off `slotBad`, else `0`. -/
noncomputable def slotQ (u : M.Ω × sch.Coins) (x : Fin T → Fin b × Fin sch.B) (t : ℕ) : ℝ :=
  if slotBad sch D Q D_late Q_late β W mach a u x t then 0
  else if (contAfter mach D_late W (sch.enc W u.1 u.2).1 (sch.enc W u.1 u.2).2 u.1 (arrS a x t)
      (histAt a x t)).closed ≤ t then
    (a / b : ℝ) * max (contFreed sch (heldAt sch D_late W mach (arrS a x t) u (histAt a x t)).length -
      epsProof sch β) 0
  else 0

variable {a}

theorem slotHit_congr (u : M.Ω × sch.Coins) {x x' : Fin T → Fin b × Fin sch.B} {t : ℕ}
    (hx : ∀ s : Fin T, (s : ℕ) < t → x s = x' s) :
    slotHit sch D Q D_late Q_late W mach a u x t = slotHit sch D Q D_late Q_late W mach a u x' t := by
  unfold slotHit
  rw [arrS_congr a hx, histAt_congr a hx]

theorem slotBad_congr (u : M.Ω × sch.Coins) {x x' : Fin T → Fin b × Fin sch.B} {t : ℕ}
    (hx : ∀ s : Fin T, (s : ℕ) < t → x s = x' s) :
    slotBad sch D Q D_late Q_late β W mach a u x t ↔ slotBad sch D Q D_late Q_late β W mach a u x' t := by
  unfold slotBad
  rw [arrS_congr a hx, histAt_congr a hx]

theorem slotQ_congr (u : M.Ω × sch.Coins) {x x' : Fin T → Fin b × Fin sch.B} {t : ℕ}
    (hx : ∀ s : Fin T, (s : ℕ) < t → x s = x' s) :
    slotQ sch D Q D_late Q_late β W mach a u x t = slotQ sch D Q D_late Q_late β W mach a u x' t := by
  unfold slotQ slotBad
  rw [arrS_congr a hx, histAt_congr a hx]

/-- **One slot.** The slot's share of the bound is at most the probability that its draw is an arrival that misses:
`(a/b)·(ε_t − ε_proof)` off `slotBad`, by `miss_ge`. -/
theorem slotQ_le (hab : a ≤ b) (u : M.Ω × sch.Coins) (x : Fin T → Fin b × Fin sch.B) (t : ℕ) :
    slotQ sch D Q D_late Q_late β W mach a u x t ≤
      (pr fun d : Fin b × Fin sch.B => (d.1 : ℕ) < a ∧ ¬ slotHit sch D Q D_late Q_late W mach a u x t d.2).toReal := by
  unfold slotQ
  split_ifs with hb hi
  · exact ENNReal.toReal_nonneg
  · refine le_of_le_of_eq ?_ (pr_and_prod (fun c : Fin b => (c : ℕ) < a)
      fun i => ¬ slotHit sch D Q D_late Q_late W mach a u x t i).symm
    rw [pr_coin a b hab]
    exact mul_le_mul_of_nonneg_left
      (max_le (miss_ge sch D Q D_late Q_late W mach (arrS a x t) β u (histAt a x t) hb) ENNReal.toReal_nonneg)
      (by positivity)
  · exact ENNReal.toReal_nonneg

/-- At an idle slot off `slotBad`, the slot's share is `(a/b)·max(ε_t − ε_proof, 0)` at the memory at `t`. -/
theorem slotQ_eq (u : M.Ω × sch.Coins) (x : Fin T → Fin b × Fin sch.B) (t : ℕ)
    (hb : ¬ slotBad sch D Q D_late Q_late β W mach a u x t) :
    slotQ sch D Q D_late Q_late β W mach a u x t =
      (a / b : ℝ) * if contIdle mach D_late W (sch.enc W u.1 u.2).1 (sch.enc W u.1 u.2).2 u.1 a x t then
        max (contFreed sch (contHeldAt mach D_late W (sch.enc W u.1 u.2).1 (sch.enc W u.1 u.2).2 u.1 a x t) -
          epsProof sch β) 0
      else 0 := by
  unfold slotQ
  rw [ite_eq_right hb]
  by_cases hi : contIdle mach D_late W (sch.enc W u.1 u.2).1 (sch.enc W u.1 u.2).2 u.1 a x t
  · rw [ite_eq_left ((contIdle_iff ..).1 hi), ite_eq_left hi]
    unfold heldAt
    rw [contMem_slot _ _ _ _ _ _ _ _ _ hi]
  · rw [ite_eq_right (mt (contIdle_iff ..).2 hi), ite_eq_right hi, mul_zero]

end SlotQ

open Classical in
/-- **The continuous audit's time-averaged bound**, `Pous.Guarantees.ContinuousTimeAverage`, from `TimedINC`. -/
theorem continuous_time_average : Pous.Guarantees.ContinuousTimeAverage := by
  intro M sch D Q D_late Q_late β T a b π hb hab hinc W mach f βe hβe
  have : Nonempty (Fin sch.B) := ⟨⟨0, sch.B_pos⟩⟩
  have : Nonempty (Fin b) := ⟨⟨0, hb⟩⟩
  set G := ⨅ θ : Set.Ioi (0 : ℝ), ((θ : ℝ) * f + Real.log (1 / βe)) / (1 - Real.exp (-(θ : ℝ))) with hG
  let d₀ : Fin b × Fin sch.B := (⟨0, hb⟩, ⟨0, sch.B_pos⟩)
  let X : M.Ω × sch.Coins → List (Fin b × Fin sch.B) → Fin b × Fin sch.B → Prop := fun u h d =>
    (d.1 : ℕ) < a ∧ ¬ slotHit sch D Q D_late Q_late W mach a u (pad d₀ h : Fin T → _) h.length d.2
  let q : M.Ω × sch.Coins → List (Fin b × Fin sch.B) → ℝ := fun u h =>
    slotQ sch D Q D_late Q_late β W mach a u (pad d₀ h : Fin T → _) h.length
  let ev' : M.Ω × sch.Coins → (Fin T → Fin b × Fin sch.B) → Prop := fun u y =>
    ((Finset.univ.filter fun j : Fin T => X u ((List.ofFn y).take j) (y j)).card : ℝ) ≤ f ∧
      G < ∑ j : Fin T, q u ((List.ofFn y).take j)
  let stop : M.Ω × sch.Coins → (Fin T → Fin b × Fin sch.B) → Prop := fun u y =>
    ∃ t ∈ (Finset.univ : Finset (Fin T)), slotBad sch D Q D_late Q_late β W mach a u y t
  have hpad : ∀ (y : Fin T → Fin b × Fin sch.B) (j : Fin T),
      ((List.ofFn y).take j).length = j ∧ ∀ s : Fin T, (s : ℕ) < ((List.ofFn y).take j).length →
        (pad d₀ ((List.ofFn y).take j) : Fin T → _) s = y s := by
    intro y j
    have hl : ((List.ofFn y).take j).length = j := by simp
    exact ⟨hl, fun s hs => pad_take d₀ y j s (hl ▸ hs)⟩
  have hX : ∀ (u : M.Ω × sch.Coins) (y : Fin T → Fin b × Fin sch.B) (j : Fin T),
      X u ((List.ofFn y).take j) (y j) ↔
        ((y j).1 : ℕ) < a ∧ ¬ slotHit sch D Q D_late Q_late W mach a u y j (y j).2 := by
    intro u y j
    obtain ⟨hl, hs⟩ := hpad y j
    change _ ∧ ¬ slotHit sch D Q D_late Q_late W mach a u _ _ _ ↔ _
    rw [slotHit_congr sch D Q D_late Q_late W mach u hs, hl]
  have hq : ∀ (u : M.Ω × sch.Coins) (y : Fin T → Fin b × Fin sch.B) (j : Fin T),
      q u ((List.ofFn y).take j) = slotQ sch D Q D_late Q_late β W mach a u y j := by
    intro u y j
    obtain ⟨hl, hs⟩ := hpad y j
    change slotQ sch D Q D_late Q_late β W mach a u _ _ = _
    rw [slotQ_congr sch D Q D_late Q_late β W mach u hs, hl]
  have himp : ∀ x : M.Ω × sch.Coins × (Fin T → Fin b × Fin sch.B),
      (let pp := (sch.enc W x.1 x.2.1).1
       let C := (sch.enc W x.1 x.2.1).2
       let rs := contRecords mach D Q D_late Q_late W pp C x.1 (arrAt a x.2.2) .start (histAt a x.2.2 T)
       contAccepts f rs ∧ G < ((a : ℝ) / b) * ∑ t : Fin T,
         if contIdle mach D_late W pp C x.1 a x.2.2 t then
           max (contFreed sch (contHeldAt mach D_late W pp C x.1 a x.2.2 t) - epsProof sch β) 0
         else 0) →
      stop (x.1, x.2.1) x.2.2 ∨ ev' (x.1, x.2.1) x.2.2 := by
    rintro ⟨ω, r, y⟩ ⟨hcount, hsum⟩
    by_cases hs : stop (ω, r) y
    · exact Or.inl hs
    refine Or.inr ⟨le_trans ?_ hcount, hsum.trans_eq ?_⟩
    · refine le_trans (Nat.cast_le.2 (Finset.card_le_card fun j => ?_))
        (Nat.cast_le.2 (misses_le mach D Q D_late Q_late W (sch.enc W ω r).1 (sch.enc W ω r).2 ω a y))
      simp only [Finset.mem_filter, Finset.mem_univ, true_and]
      rw [hX]
      exact id
    · rw [Finset.mul_sum]
      refine Finset.sum_congr rfl fun t _ => ?_
      have hb' : ¬ slotBad sch D Q D_late Q_late β W mach a (ω, r) y t := fun h => hs ⟨t, Finset.mem_univ _, h⟩
      rw [hq, slotQ_eq sch D Q D_late Q_late β W mach (ω, r) y t hb']
  have hre : ∀ R : M.Ω × sch.Coins → (Fin T → Fin b × Fin sch.B) → Prop,
      pr (fun x : M.Ω × sch.Coins × (Fin T → Fin b × Fin sch.B) => R (x.1, x.2.1) x.2.2) =
        pr (fun y : (M.Ω × sch.Coins) × (Fin T → Fin b × Fin sch.B) => R y.1 y.2) := fun R =>
    pr_equiv (Equiv.prodAssoc _ _ _).symm (fun y : (M.Ω × sch.Coins) × (Fin T → Fin b × Fin sch.B) => R y.1 y.2)
  have hstop : pr (fun y : (M.Ω × sch.Coins) × (Fin T → Fin b × Fin sch.B) => stop y.1 y.2) ≤ T * (β + 1) * π := by
    refine (pr_exists_le _ _).trans ?_
    calc ∑ t : Fin T, pr (fun y : (M.Ω × sch.Coins) × (Fin T → Fin b × Fin sch.B) =>
            slotBad sch D Q D_late Q_late β W mach a y.1 y.2 t)
        ≤ ∑ _t : Fin T, ((β : ℝ≥0∞) + 1) * π := by
          refine Finset.sum_le_sum fun t _ => ?_
          rw [← pr_equiv (Equiv.prodComm _ _)]
          simp only [Equiv.prodComm_apply, Prod.fst_swap, Prod.snd_swap]
          exact pr_prod_le_of_forall (fun y u => slotBad sch D Q D_late Q_late β W mach a u y t) _
            fun y => pr_bad_le sch D Q D_late Q_late W mach (arrS a y t) β π hinc (histAt a y t)
      _ = T * (β + 1) * π := by
          rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, mul_assoc]
  have hmain : pr (fun y : (M.Ω × sch.Coins) × (Fin T → Fin b × Fin sch.B) => ev' y.1 y.2) ≤
      ENNReal.ofReal βe := by
    refine pr_prod_le_of_forall ev' _ fun u => ?_
    convert pr_misses_gt_le T (X u) (q u)
      (fun h _ => slotQ_le sch D Q D_late Q_late β W mach hab u (pad d₀ h) h.length) f βe hβe
  calc _ ≤ pr (fun x : M.Ω × sch.Coins × (Fin T → Fin b × Fin sch.B) =>
          stop (x.1, x.2.1) x.2.2 ∨ ev' (x.1, x.2.1) x.2.2) := pr_mono himp
    _ = pr (fun y : (M.Ω × sch.Coins) × (Fin T → Fin b × Fin sch.B) => stop y.1 y.2 ∨ ev' y.1 y.2) :=
        hre fun u y => stop u y ∨ ev' u y
    _ ≤ pr (fun y : (M.Ω × sch.Coins) × (Fin T → Fin b × Fin sch.B) => stop y.1 y.2) +
          pr (fun y : (M.Ω × sch.Coins) × (Fin T → Fin b × Fin sch.B) => ev' y.1 y.2) := pr_or_le _ _
    _ ≤ T * (β + 1) * π + ENNReal.ofReal βe := add_le_add hstop hmain
    _ = ENNReal.ofReal βe + T * (β + 1) * π := add_comm _ _

end PousContinuous
