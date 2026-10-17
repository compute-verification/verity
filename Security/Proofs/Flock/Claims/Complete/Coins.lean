import Proofs.Flock.Claims.Complete.Run
import Proofs.Flock.Claims.Prob
import Proofs.Flock.Claims.NumbersDeploy
import Proofs.Flock.Level3.ArithFacts

/-!
# δc: where the honest prover can fail

Plain Flock is perfectly complete but for one step. Zerocheck's round `i` recovers `g(0) = (claim + r·g(1))·(1 + r)⁻¹`
at the outer coin `r = r_i` (`bindAndZerocheckWith`), and `F128.inv 0 = 0`: at `r = 1` the verifier's `g(0)` is `0`
whatever the honest prover's is, and the next claim need not hold. So the honest prover can fail only where some
zerocheck outer coin is `1` (`Bad`), at most `2^-128` per coin element (`pr_slot_le`).

* **The bound** (`pr_bad_le`): `Pr[Bad] ≤ ε`, a union bound over the outer coins (`outerSlots`, `Complete.Ready`).
* **The composition** (`complete_of`): if the honest prover is accepted at every coin outside `Bad` on every statement in
  scope, ready (`ready H`), with a witness in the relation, then `Claims.Flock.Complete H (deployPlain rel)`, at the
  record's error `δc H` (`Complete.Ready`, which states where it is `1`), whose prover is the record's (`honestP`).
* **The error is below `1`** where the verifier can accept, at the sessions the numbers are stated at (`PlainIn`,
  `ZkIn`: at most `200` tables at `m ≤ 35`): at most `44` outer coins per table (`Claims.Flock.Complete_meets_plain`),
  and `50` coins at the masked verifier's one table (`Claims.Flock.Complete_meets_zk`).
-/

namespace FlockComplete

open Flock FlockDeploy

/-! ## The probability of the bad coins -/

theorem Pr_or_le {α : Type} [Fintype α] (p q : α → Prop) :
    Claims.Pr (fun a => p a ∨ q a) ≤ Claims.Pr p + Claims.Pr q := by
  classical
  unfold Claims.Pr
  rw [← add_div]
  refine div_le_div_of_nonneg_right ?_ (by positivity)
  norm_cast
  refine le_trans (Finset.card_le_card ?_) (Finset.card_union_le _ _)
  intro a
  simp only [Finset.mem_filter, Finset.mem_union, Finset.mem_univ, true_and]
  exact id

/-- **The union bound** over a list of events. -/
theorem Pr_exists_le {α β : Type} [Fintype α] (E : β → α → Prop) :
    ∀ L : List β, Claims.Pr (fun a => ∃ b ∈ L, E b a) ≤ (L.map fun b => Claims.Pr (E b)).sum
  | [] => by simp [Claims.Pr_false]
  | b :: L => by
    refine le_trans (le_of_eq (Claims.Pr_congr (q := fun a => E b a ∨ ∃ b' ∈ L, E b' a) fun a => ?_))
      (le_trans (Pr_or_le (E b) _) ?_)
    · simp
    · simp only [List.map_cons, List.sum_cons]
      exact (add_le_add_iff_left _).2 (Pr_exists_le E L)

theorem Pr_equiv {α β : Type} [Fintype α] [Fintype β] (e : α ≃ β) (p : β → Prop) :
    Claims.Pr (fun a => p (e a)) = Claims.Pr p := by
  classical
  rw [Claims.Pr_eq_sum (fun a => p (e a)), Claims.Pr_eq_sum p, Fintype.card_congr e]
  congr 1
  exact e.sum_comp (fun b => if p b then (1 : ℝ) else 0)

theorem Pr_eq_const {α : Type} [Fintype α] (c : α) : Claims.Pr (fun a : α => a = c) = 1 / Fintype.card α := by
  classical
  unfold Claims.Pr
  congr 1
  rw [Finset.filter_eq' Finset.univ c]
  simp

theorem Pr_not {α : Type} [Fintype α] [Nonempty α] (p : α → Prop) :
    Claims.Pr (fun a => ¬ p a) = 1 - Claims.Pr p := by
  classical
  rw [Claims.Pr_eq_sum (fun a => ¬ p a), Claims.Pr_eq_sum p]
  have hc : (0 : ℝ) < Fintype.card α := by exact_mod_cast Fintype.card_pos
  have e : ∀ a, (if ¬ p a then (1 : ℝ) else 0) = 1 - (if p a then 1 else 0) := by
    intro a; split_ifs <;> simp_all
  rw [Finset.sum_congr rfl fun a _ => e a, Finset.sum_sub_distrib, Finset.sum_const, Finset.card_univ, nsmul_eq_mul,
    mul_one]
  field_simp

/-- **A coin element is `1` with probability at most `2^-128`**, at any index (`0` past the coins or the elements). -/
theorem pr_slot_le {k : ℕ} (j i : ℕ) :
    Claims.Pr (fun r : Fin k → Coin => (Claims.coinSeq r j).at i = 1) ≤ 1 / 2 ^ 128 := by
  by_cases hj : j < k
  · by_cases hi : i < 65536
    · let e1 : (Fin k → Coin) ≃ Coin × ({j' // j' ≠ (⟨j, hj⟩ : Fin k)} → Coin) :=
        Equiv.piSplitAt (⟨j, hj⟩ : Fin k) fun _ => Coin
      let e2 : Coin ≃ FlockLevel3.GF128 × ({i' // i' ≠ (⟨i, hi⟩ : Fin 65536)} → FlockLevel3.GF128) :=
        Equiv.piSplitAt (⟨i, hi⟩ : Fin 65536) fun _ => FlockLevel3.GF128
      have : Nonempty FlockLevel3.GF128 := ⟨(0 : F128)⟩
      have hev : ∀ r : Fin k → Coin, (Claims.coinSeq r j).at i = 1 ↔
          (e1.trans (Equiv.prodCongr e2 (Equiv.refl _)) r).1.1 = (1 : F128) := by
        intro r
        simp only [Claims.coinSeq, hj, dite_true, Coin.at, hi]
        rfl
      rw [Claims.Pr_congr hev, Pr_equiv (e1.trans (Equiv.prodCongr e2 (Equiv.refl _)))
        (fun s => s.1.1 = (1 : F128)), Claims.Pr_fst (fun s : FlockLevel3.GF128 × _ => s.1 = (1 : F128)),
        Claims.Pr_fst (fun g : FlockLevel3.GF128 => g = (1 : F128))]
      exact le_of_eq ((Pr_eq_const (α := FlockLevel3.GF128) (1 : F128)).trans
        (by rw [FlockLevel3.card_F]; norm_num))
    · refine le_trans (le_of_eq (Claims.Pr_congr (q := fun _ => False) fun r => ?_)) ?_
      · simp only [Coin.at, hi, dite_false, iff_false]
        decide
      · rw [Claims.Pr_false]; positivity
  · refine le_trans (le_of_eq (Claims.Pr_congr (q := fun _ => False) fun r => ?_)) ?_
    · simp only [Claims.coinSeq, hj, dite_false, iff_false]
      show ¬ (default : Coin).at i = 1
      unfold Coin.at
      split
      · show ¬ (0 : F128) = 1
        decide
      · decide
    · rw [Claims.Pr_false]; positivity

/-! ## The bad coins -/

/-- **The bad coins**: a zerocheck outer coin is `1`, where the verifier's `(1 + r)⁻¹` is `0⁻¹ = 0`. -/
def Bad (x : Claims.Msg) {k : ℕ} (r : Fin k → Coin) : Prop := ∃ p ∈ outerSlots x, (Claims.coinSeq r p.1).at p.2 = 1

/-- **`Pr[Bad] ≤ ε`**: the union bound over the outer coins. -/
theorem pr_bad_le (x : Claims.Msg) {k : ℕ} : Claims.Pr (fun r : Fin k → Coin => Bad x r) ≤ ε x := by
  have h1 := Pr_exists_le (fun p (r : Fin k → Coin) => (Claims.coinSeq r p.1).at p.2 = 1) (outerSlots x)
  have h2 : ((outerSlots x).map fun p =>
      Claims.Pr (fun (r : Fin k → Coin) => (Claims.coinSeq r p.1).at p.2 = 1)).sum ≤ ε x := by
    refine le_trans (List.sum_le_length_nsmul _ (1 / 2 ^ 128) ?_) (le_of_eq ?_)
    · intro y hy
      obtain ⟨p, -, rfl⟩ := List.mem_map.1 hy
      exact pr_slot_le p.1 p.2
    · rw [List.length_map, nsmul_eq_mul, ε]
      ring
  exact le_trans h1 h2

/-! ## The composition -/

/-- **Completeness from acceptance at the good coins**: if, on every statement in scope that the verifier can accept
and every witness in the relation, the honest prover is accepted at every tape and every coin outside `Bad`, then Flock's
plain verifier is complete for `honestP` at the record's error, `δc H`. -/
theorem complete_of (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation)
    (hacc : ∀ x w, Claims.Flock.Scope (deployPlain rel) x → Claims.Flock.R (deployPlain rel).parse x w →
      ready H x = true → ∀ (ω : Tape) (r : Fin ((V false).k x) → Coin), ¬ Bad x r →
        (Claims.run H (honestP w) (V false) x ω r).out = .acc) :
    Claims.Flock.Complete H (deployPlain rel) := by
  intro _ x w hs hr ω
  show 1 - δc H x ≤ _
  unfold δc
  split
  · rename_i hready
    calc 1 - ε x ≤ 1 - Claims.Pr (fun r : Fin ((V false).k x) → Coin => Bad x r) := by
          linarith [pr_bad_le x (k := (V false).k x)]
      _ = Claims.Pr (fun r : Fin ((V false).k x) → Coin => ¬ Bad x r) := (Pr_not _).symm
      _ ≤ Claims.Pr (fun r : Fin ((V false).k x) → Coin =>
            (Claims.run H (honestP w) (V false) x ω r).out = .acc) :=
          Claims.Pr_mono fun r hr' => hacc x w hs hr hready ω r hr'
  · rw [sub_self]
    exact Claims.Pr_nonneg _

/-! ## The error is below `1` -/

theorem length_flatMap_range_le {β : Type} (n c : ℕ) (f : ℕ → List β) (h : ∀ q < n, (f q).length ≤ c) :
    ((List.range n).flatMap f).length ≤ n * c := by
  rw [List.length_flatMap]
  refine (List.sum_le_length_nsmul _ c ?_).trans (by simp)
  intro y hy
  obtain ⟨q, hq, rfl⟩ := List.mem_map.1 hy
  exact h q (List.mem_range.1 hq)

theorem m_sub_le {sts : Array Flock.Stmt} (hm : ∀ st ∈ sts.toList, st.m ≤ 35) {q : ℕ} (hq : q < 2 * sts.size) :
    sts[q / 2]!.m - 13 ≤ 22 := by
  have hi : q / 2 < sts.size := by omega
  have := hm sts[q / 2]! (by rw [getElem!_pos sts (q / 2) hi]; exact Array.getElem_mem_toList hi)
  omega

/-- **At most `22` outer coins per rep**, two reps per table, at tables of `m ≤ 35`. -/
theorem outerSlots_length_le {x : Claims.Msg} {s : Ins} {spec : SessionSpec} {sts : Array Flock.Stmt}
    (hs : setupOf false x = some (s, spec, sts)) (hm : ∀ st ∈ sts.toList, st.m ≤ 35) :
    (outerSlots x).length ≤ 2 * sts.size * 22 := by
  unfold outerSlots
  rw [hs]
  exact length_flatMap_range_le _ _ _ fun q hq => by simpa using m_sub_le hm hq

theorem outerSlotsZk_length_le {x : Claims.Msg} {s : Ins} {spec : SessionSpec} {sts : Array Flock.Stmt}
    (hs : setupOf true x = some (s, spec, sts)) (hm : ∀ st ∈ sts.toList, st.m ≤ 35) :
    (outerSlotsZk x).length ≤ 2 * sts.size * 22 := by
  unfold outerSlotsZk
  rw [hs]
  exact length_flatMap_range_le _ _ _ fun q hq => by simpa using m_sub_le hm hq

theorem rhoSlotsZk_length_le {x : Claims.Msg} {s : Ins} {spec : SessionSpec} {sts : Array Flock.Stmt}
    (hs : setupOf true x = some (s, spec, sts)) : (rhoSlotsZk x).length ≤ 2 * sts.size * 1 := by
  unfold rhoSlotsZk
  rw [hs]
  exact length_flatMap_range_le _ _ _ fun q _ => by split <;> simp

theorem betaSlotsZk_length_le {x : Claims.Msg} {s : Ins} {spec : SessionSpec} {sts : Array Flock.Stmt}
    (hs : setupOf true x = some (s, spec, sts)) : (betaSlotsZk x).length ≤ 2 * sts.size * 1 := by
  unfold betaSlotsZk
  rw [hs]
  exact length_flatMap_range_le _ _ _ fun q _ => by split <;> simp

end FlockComplete

open Flock FlockDeploy FlockComplete in
/-- **The plain record's error is below `1`** on every statement the verifier can accept (`ready H`) at a session the
numbers are stated at (`PlainIn`): at most `44` outer coins per table over `2^128`, at most `200` tables. -/
theorem Claims.Flock.Complete_meets_plain (rel : Readable.Parsed → Readable.Relation) (H : Claims.Msg → Claims.Msg)
    {x : Claims.Msg} {s : Ins} {spec : SessionSpec} {sts : Array _root_.Flock.Stmt} (hin : PlainIn x s spec sts)
    (hready : ready H x = true) :
    (deployPlain rel).δc H x ≤ 44 * sts.size / 2 ^ 128 ∧ (deployPlain rel).δc H x < 1 := by
  have hlen := outerSlots_length_le hin.setup fun st hst => (hin.tab st hst).2.1
  have hle : (deployPlain rel).δc H x ≤ 44 * sts.size / 2 ^ 128 := by
    have e : δc H x = ε x := by simp [δc, hready]
    show δc H x ≤ _
    rw [e, ε]
    gcongr
    exact_mod_cast hlen.trans (le_of_eq (by ring))
  refine ⟨hle, lt_of_le_of_lt hle ?_⟩
  have h200 : (sts.size : ℝ) ≤ 200 := by exact_mod_cast hin.size_le
  rw [div_lt_one (by positivity)]
  calc 44 * (sts.size : ℝ) ≤ 44 * 200 := by linarith
    _ < 2 ^ 128 := by norm_num

open Flock FlockDeploy FlockComplete in
/-- **The masked record's error is below `1`** on every statement the masked verifier can accept (`readyZk H`, one
table) at a session the numbers are stated at (`ZkIn`): `44` outer coins, two `β`s and two pairs counted twice, over
`2^128`. -/
theorem Claims.Flock.Complete_meets_zk (rel : Readable.Parsed → Readable.Relation) (H : Claims.Msg → Claims.Msg)
    {x : Claims.Msg} {s : Ins} {spec : SessionSpec} {sts : Array _root_.Flock.Stmt} (hin : ZkIn x s spec sts)
    (hready : readyZk H x = true) :
    (deployZk rel).δc H x ≤ 50 / 2 ^ 128 ∧ (deployZk rel).δc H x < 1 := by
  have h1 : sts.size = 1 := by
    have h := hready
    unfold readyZk at h
    rw [hin.setup] at h
    simp only [Bool.and_eq_true] at h
    exact beq_iff_eq.1 h.1.1.1.1.1.1
  have ho := outerSlotsZk_length_le hin.setup fun st hst => (hin.tab st hst).2.1
  have hr := rhoSlotsZk_length_le hin.setup
  have hb := betaSlotsZk_length_le hin.setup
  rw [h1] at ho hr hb
  have hle : (deployZk rel).δc H x ≤ 50 / 2 ^ 128 := by
    have e : δcZk H x = εZk x := by simp [δcZk, hready]
    show δcZk H x ≤ _
    rw [e, εZk]
    gcongr
    have ho' : ((outerSlotsZk x).length : ℝ) ≤ 44 := by exact_mod_cast ho
    have hr' : ((rhoSlotsZk x).length : ℝ) ≤ 2 := by exact_mod_cast hr
    have hb' : ((betaSlotsZk x).length : ℝ) ≤ 2 := by exact_mod_cast hb
    linarith
  exact ⟨hle, lt_of_le_of_lt hle (by norm_num)⟩
