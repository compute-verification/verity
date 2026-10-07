import Definitions.NetworkCertifier.Grid
import Mathlib.Data.Sym.Card
import Mathlib.Data.Set.Card
import Mathlib.Order.Interval.Finset.Fin
import Mathlib.Algebra.Order.BigOperators.Group.Finset

/-!
# Proposition 2: timing is occupancy, its counts, and tightness

`docs/network-transparency/timing-channel.md` §2.3.
-/

namespace NetTiming

open Finset

variable {T S : ℕ} {F : Type}

/-- Free slots: the sender addresses slots, and there are `2^(T·S)` occupancies. -/
theorem card_free (T S : ℕ) : Fintype.card (Fin T → Fin S → Bool) = 2 ^ (T * S) := by
  rw [Fintype.card_fun, Fintype.card_fun]
  simp [Fintype.card_bool, ← pow_mul, mul_comm]

/-! ### Left-packed rows are prefix patterns -/

/-- The prefix row with `k` occupied slots. -/
def prefixRow (S : ℕ) (k : ℕ) : Fin S → Bool := fun s => decide ((s : ℕ) < k)

theorem card_filter_val_lt (S k : ℕ) (hk : k ≤ S) : (univ.filter fun s : Fin S => (s : ℕ) < k).card = k := by
  have : (univ.filter fun s : Fin S => (s : ℕ) < k).map Fin.valEmbedding = range k := by
    ext x; simp only [mem_map, mem_filter, mem_univ, true_and, Fin.valEmbedding_apply, mem_range]
    constructor
    · rintro ⟨s, hs, rfl⟩; exact hs
    · intro hx; exact ⟨⟨x, by omega⟩, hx, rfl⟩
  rw [← card_map, this, card_range]

/-- A left-packed row is the prefix row of its count. -/
theorem eq_prefixRow_of_rowLeftPacked (r : Fin S → Bool) (h : RowLeftPacked r) :
    r = prefixRow S (univ.filter fun s => r s = true).card := by
  funext s
  simp only [prefixRow]
  by_cases hs : r s = true
  · rw [hs]; symm; rw [decide_eq_true_iff]
    have hsub : Iic s ⊆ univ.filter fun s => r s = true := by
      intro s' hs'; simp only [mem_Iic] at hs'; simp only [mem_filter, mem_univ, true_and]; exact h s' s hs' hs
    have := card_le_card hsub
    rw [Fin.card_Iic] at this; omega
  · have hs' : r s = false := by simpa using hs
    rw [hs']; symm; rw [decide_eq_false_iff_not, not_lt]
    have hsub : (univ.filter fun s => r s = true) ⊆ Iio s := by
      intro s' hs''; simp only [mem_filter, mem_univ, true_and] at hs''; simp only [mem_Iio]
      by_contra hlt; exact hs (h s s' (not_lt.mp hlt) hs'')
    have := card_le_card hsub
    rw [Fin.card_Iio] at this; exact this

theorem rowLeftPacked_prefixRow (k : ℕ) : RowLeftPacked (prefixRow S k) := by
  intro s s' hle h; simp only [prefixRow, decide_eq_true_eq] at *
  exact lt_of_le_of_lt (Fin.le_def.mp hle) h

theorem card_prefixRow (k : ℕ) (hk : k ≤ S) : (univ.filter fun s => prefixRow S k s = true).card = k := by
  simp only [prefixRow, decide_eq_true_eq]; exact card_filter_val_lt S k hk

/-- Left-packed rows are counted by `Fin (S + 1)`. -/
def rowEquiv (S : ℕ) : {r : Fin S → Bool // RowLeftPacked r} ≃ Fin (S + 1) where
  toFun r := ⟨(univ.filter fun s => r.1 s = true).card,
    Nat.lt_succ_of_le ((card_filter_le _ _).trans (by simp))⟩
  invFun k := ⟨prefixRow S k, rowLeftPacked_prefixRow k⟩
  left_inv r := by
    apply Subtype.ext; simp only; exact (eq_prefixRow_of_rowLeftPacked r.1 r.2).symm
  right_inv k := by
    apply Fin.ext; simp only; exact card_prefixRow k (Nat.lt_succ_iff.mp k.2)

/-- Left-packed: `(S + 1)^T` occupancies. -/
theorem card_leftPacked (T S : ℕ) :
    Fintype.card {P : Fin T → Fin S → Bool // LeftPacked P} = (S + 1) ^ T := by
  have e : {P : Fin T → Fin S → Bool // LeftPacked P} ≃ (Fin T → Fin (S + 1)) :=
    (Equiv.subtypePiEquivPi (p := fun _ r => RowLeftPacked r)).trans (Equiv.piCongrRight fun _ => rowEquiv S)
  rw [Fintype.card_congr e, Fintype.card_fun, Fintype.card_fin, Fintype.card_fin]

/-! ### Fill -/

/-- The occupancy of a fill is the prefix pattern of its counts. -/
theorem occ_fill (f : ℕ → F) (n : Fin T → Fin (S + 1)) :
    occ (fill f n) = fun (t : Fin T) (s : Fin S) => decide ((s : ℕ) < (n t : ℕ)) := by
  funext t s; simp only [occ, fill]; split_ifs with h <;> simp [h]

/-- A fill is left-packed. -/
theorem leftPacked_occ_fill (f : ℕ → F) (n : Fin T → Fin (S + 1)) : LeftPacked (occ (fill f n)) := by
  rw [occ_fill]; intro t; exact rowLeftPacked_prefixRow (S := S) (n t)

/-- **Timing is occupancy, (i).** For a fixed frame sequence, the grid determines the counts. -/
theorem fill_injective (f : ℕ → F) : Function.Injective (fill (T := T) (S := S) f) := by
  intro n n' h
  have ho := congrArg occ h
  rw [occ_fill, occ_fill] at ho
  funext t
  have hrow : prefixRow S (n t) = prefixRow S (n' t) := congrFun ho t
  have h1 := card_prefixRow (S := S) (n t) (Nat.lt_succ_iff.mp (n t).2)
  have h2 := card_prefixRow (S := S) (n' t) (Nat.lt_succ_iff.mp (n' t).2)
  rw [hrow] at h1
  exact Fin.ext (h1.symm.trans h2)

/-- `offset` is strictly increasing across buckets: bucket `t`'s frames all come before bucket `t'`'s. -/
theorem offset_add_le (n : Fin T → Fin (S + 1)) {t t' : Fin T} (h : t < t') :
    offset n t + (n t : ℕ) ≤ offset n t' := by
  unfold offset
  have hsub : insert t (univ.filter fun u => u < t) ⊆ univ.filter fun u => u < t' := by
    intro u hu; simp only [mem_insert, mem_filter, mem_univ, true_and] at hu ⊢
    rcases hu with rfl | hu
    · exact h
    · exact hu.trans h
  have hnot : t ∉ univ.filter fun u => u < t := by simp
  calc (∑ u ∈ univ.filter fun u => u < t, (n u : ℕ)) + (n t : ℕ)
      = ∑ u ∈ insert t (univ.filter fun u => u < t), (n u : ℕ) := by rw [sum_insert hnot, add_comm]
    _ ≤ ∑ u ∈ univ.filter fun u => u < t', (n u : ℕ) := sum_le_sum_of_subset hsub

/-- Occupied positions have distinct global indices `offset + s`. -/
theorem offset_inj (n : Fin T → Fin (S + 1)) {t t' : Fin T} {s s' : Fin S}
    (hs : (s : ℕ) < n t) (hs' : (s' : ℕ) < n t') (heq : offset n t + s = offset n t' + s') : t = t' ∧ s = s' := by
  rcases lt_trichotomy t t' with hlt | rfl | hgt
  · have := offset_add_le n hlt; omega
  · exact ⟨rfl, Fin.ext (by omega)⟩
  · have := offset_add_le n hgt; omega

/-- **Timing is occupancy, (ii).** A grid whose occupancy is left-packed is the fill of some frame sequence at some
counts. -/
theorem exists_fill_of_leftPacked [Inhabited F] (g : Grid T S F) (h : LeftPacked (occ g)) :
    ∃ (f : ℕ → F) (n : Fin T → Fin (S + 1)), g = fill f n := by
  classical
  let n : Fin T → Fin (S + 1) := fun t => rowEquiv S ⟨occ g t, h t⟩
  have hrow : ∀ t, occ g t = prefixRow S (n t) := fun t => eq_prefixRow_of_rowLeftPacked _ (h t)
  let f : ℕ → F := fun k =>
    if hk : ∃ p : Fin T × Fin S, (p.2 : ℕ) < n p.1 ∧ offset n p.1 + p.2 = k then
      (g hk.choose.1 hk.choose.2).getD default else default
  refine ⟨f, n, ?_⟩
  funext t s
  have hocc : (g t s).isSome = decide ((s : ℕ) < n t) := by
    have := congrFun (hrow t) s; simpa [occ, prefixRow] using this
  simp only [fill]
  split_ifs with hs
  · have hk : ∃ p : Fin T × Fin S, (p.2 : ℕ) < n p.1 ∧ offset n p.1 + p.2 = offset n t + s := ⟨(t, s), hs, rfl⟩
    simp only [f, hk, ↓reduceDIte]
    obtain ⟨h1, h2⟩ := offset_inj n hk.choose_spec.1 hs hk.choose_spec.2
    rw [h1, h2]
    have hsome : (g t s).isSome = true := by rw [hocc]; simpa using hs
    obtain ⟨x, hx⟩ := Option.isSome_iff_exists.mp hsome
    simp [hx]
  · have : (g t s).isSome = false := by rw [hocc]; simpa using hs
    exact Option.not_isSome_iff_eq_none.mp (by simp [this])

/-! ### Fixed M -/

/-- The multiset of buckets, bucket `t` repeated `n t` times. -/
def countsMultiset (n : Fin T → Fin (S + 1)) : Multiset (Fin T) := ∑ t, Multiset.replicate (n t) t

theorem count_countsMultiset (n : Fin T → Fin (S + 1)) (t : Fin T) :
    (countsMultiset n).count t = n t := by
  classical
  simp [countsMultiset, Multiset.count_sum', Multiset.count_replicate]

theorem card_countsMultiset (n : Fin T → Fin (S + 1)) :
    Multiset.card (countsMultiset n) = ∑ t, (n t : ℕ) := by
  simp [countsMultiset, Multiset.card_sum]

/-- **Fixed M, all released:** at most `binom(T + M − 1, M)` count vectors, which is the doc's `binom(M+T−1, T−1)`
for `T ≥ 1`. -/
theorem card_counts_eq_le (T S M : ℕ) :
    (univ.filter fun n : Fin T → Fin (S + 1) => ∑ t, (n t : ℕ) = M).card ≤ (T + M - 1).choose M := by
  classical
  rw [← Fintype.card_subtype]
  let g : {n : Fin T → Fin (S + 1) // ∑ t, (n t : ℕ) = M} → Sym (Fin T) M :=
    fun n => Sym.mk (countsMultiset n.1) (by rw [card_countsMultiset]; exact n.2)
  have hg : Function.Injective g := by
    intro n n' h
    apply Subtype.ext; funext t
    have := congrArg (fun m : Sym (Fin T) M => (m : Multiset (Fin T)).count t) h
    simp only [g, Sym.coe_mk, count_countsMultiset] at this
    exact Fin.ext this
  have := Fintype.card_le_of_injective g hg
  rwa [Sym.card_sym_eq_choose, Fintype.card_fin] at this

/-- **At most M released:** at most `binom(T + M, M)`, which is the doc's `binom(M+T, T)`. -/
theorem card_counts_le_le (T S M : ℕ) :
    (univ.filter fun n : Fin T → Fin (S + 1) => ∑ t, (n t : ℕ) ≤ M).card ≤ (T + M).choose M := by
  classical
  rw [← Fintype.card_subtype]
  let g : {n : Fin T → Fin (S + 1) // ∑ t, (n t : ℕ) ≤ M} → Sym (Option (Fin T)) M :=
    fun n => Sym.mk ((countsMultiset n.1).map some + Multiset.replicate (M - ∑ t, (n.1 t : ℕ)) none) (by
      rw [Multiset.card_add, Multiset.card_map, Multiset.card_replicate, card_countsMultiset]; have := n.2; omega)
  have hg : Function.Injective g := by
    intro n n' h
    apply Subtype.ext; funext t
    have := congrArg (fun m : Sym (Option (Fin T)) M => (m : Multiset (Option (Fin T))).count (some t)) h
    simp only [g, Sym.coe_mk, Multiset.count_add, Multiset.count_replicate, reduceCtorEq, ↓reduceIte, add_zero] at this
    rw [Multiset.count_map_eq_count' _ _ (Option.some_injective _), Multiset.count_map_eq_count' _ _
      (Option.some_injective _), count_countsMultiset, count_countsMultiset] at this
    exact Fin.ext this
  have := Fintype.card_le_of_injective g hg
  rw [Sym.card_sym_eq_choose, Fintype.card_option, Fintype.card_fin] at this
  simpa using this

/-! ### Tightness -/

theorem queue_self (S : ℕ) (n : ℕ → Fin (S + 1)) (t : ℕ) : queue S (fun t => (n t : ℕ)) t = 0 := by
  induction t with
  | zero => rfl
  | succ t ih =>
    simp only [queue, ih, zero_add]
    have := Nat.lt_succ_iff.mp (n t).2
    rw [min_eq_left this, Nat.sub_self]

/-- **Tightness, the schedule.** If the sender enqueues `n t ≤ S` frames in bucket `t`, the queue stays empty and the
FIFO certifier releases exactly `n t`. Over `T` buckets this uses `∑ n t ≤ T·S` frames. -/
theorem released_self (S : ℕ) (n : ℕ → Fin (S + 1)) : released S (fun t => (n t : ℕ)) = n := by
  funext t
  apply Fin.ext
  simp only [released, queue_self, zero_add]
  exact min_eq_left (Nat.lt_succ_iff.mp (n t).2)

/-- Conservation: what is still queued, plus what was released, is what was enqueued. -/
theorem queue_add_sum_released (S : ℕ) (enq : ℕ → ℕ) (T : ℕ) :
    queue S enq T + ∑ t ∈ range T, (released S enq t : ℕ) = ∑ t ∈ range T, enq t := by
  induction T with
  | zero => simp [queue]
  | succ T ih =>
    rw [sum_range_succ, sum_range_succ, ← ih]
    simp only [queue, released]
    have := min_le_left (queue S enq T + enq T) S
    omega

/-- **The certifier never releases more than was enqueued.** -/
theorem sum_released_le (T S : ℕ) (enq : ℕ → ℕ) :
    ∑ t ∈ Finset.range T, (released S enq t : ℕ) ≤ ∑ t ∈ Finset.range T, enq t := by
  have := queue_add_sum_released S enq T
  omega

/-- **Fixed M, at the certifier.** When the sender enqueues at most `M` frames over the window, the FIFO certifier can release
at most `binom(T + M, M)` distinct grids: the doc's `binom(M+T, T)`. -/
theorem encard_fifoGrids_le_of_available (T S M : ℕ) (f : ℕ → F) :
    {g | ∃ enq : ℕ → ℕ, ∑ t ∈ Finset.range T, enq t ≤ M ∧ g = fifoGrid T S f enq}.encard ≤
      ((T + M).choose M : ℕ) := by
  classical
  let cs : Finset (Fin T → Fin (S + 1)) := univ.filter fun n => ∑ t, (n t : ℕ) ≤ M
  have hsub : {g | ∃ enq : ℕ → ℕ, ∑ t ∈ Finset.range T, enq t ≤ M ∧ g = fifoGrid T S f enq} ⊆
      ((cs.image (fill f) : Finset (Grid T S F)) : Set (Grid T S F)) := by
    rintro g ⟨enq, hM, rfl⟩
    simp only [coe_image, Set.mem_image, mem_coe, cs, mem_filter, mem_univ, true_and]
    refine ⟨fun t => released S enq t, ?_, rfl⟩
    show ∑ t : Fin T, (released S enq t : ℕ) ≤ M
    have h := sum_released_le T S enq
    rw [← Fin.sum_univ_eq_sum_range (fun t => (released S enq t : ℕ)) T] at h
    omega
  calc _ ≤ ((cs.image (fill f)).card : ℕ∞) := by
        rw [← Set.encard_coe_eq_coe_finsetCard]; exact Set.encard_le_encard hsub
    _ ≤ ((T + M).choose M : ℕ) := by
        exact_mod_cast (card_image_le.trans (card_counts_le_le T S M))

/-- The FIFO certifier realizes W4: every grid it releases is left-packed. -/
theorem leftPacked_fifoGrid (f : ℕ → F) (enq : ℕ → ℕ) : LeftPacked (occ (fifoGrid T S f enq)) :=
  leftPacked_occ_fill f _

/-- **Tightness.** Over all enqueue schedules, with the frames leaving in queue order `f`, the FIFO certifier can release
exactly `(S + 1)^T` distinct grids. So `T·log2(S+1)` bits are attained with zero error. -/
theorem encard_fifoGrids (T S : ℕ) (f : ℕ → F) :
    (Set.range fun enq : ℕ → ℕ => fifoGrid T S f enq).encard = ((S + 1) ^ T : ℕ) := by
  have hrange : (Set.range fun enq : ℕ → ℕ => fifoGrid T S f enq) = Set.range (fill (T := T) (S := S) f) := by
    ext g; constructor
    · rintro ⟨enq, rfl⟩; exact ⟨_, rfl⟩
    · rintro ⟨n, rfl⟩
      let n' : ℕ → Fin (S + 1) := fun u => if h : u < T then n ⟨u, h⟩ else 0
      refine ⟨fun u => (n' u : ℕ), ?_⟩
      simp only [fifoGrid, released_self]
      congr 1; funext t; simp [n', t.2]
  rw [hrange, ← Set.image_univ, (fill_injective f).encard_image, Set.encard_univ, ENat.card_eq_coe_fintype_card,
    Fintype.card_fun, Fintype.card_fin, Fintype.card_fin]

end NetTiming
