module

public import Proofs.Flock.Bridge.FilePart.Uniform
public import Proofs.Flock.Bridge.Lowers

/-!
# The proof blocks of `k` input gates and `n` units of `m` rows

On a circuit whose first `k` gates are inputs and whose gate `k + m·u + i` is unit `u`'s row, unit `u`'s lowered gates
are `k + 3m·u, …, k + 3m·(u + 1) − 1` (`unitGates_of`). On `k = 2 + n·B` input gates, where unit `v` reads input gate
`2 + v·B + b` exactly when `b` is a source some row reads, the inputs unit `u` owns sit at `2 + u·B + b`, in order
(`insOf_of`).
-/

@[expose] public section

namespace Flock.Bridge.FilePart

open FlockSoundness FlockSoundness.Audit

/-! ## A partition of `k` input gates and `n` units of `m` rows -/

section abstract

variable {C : Verity.Boolean.Circuit} {n : ℕ} (P : Audit.Partition C n) {k m : ℕ}

theorem off_of (hsize : ∀ g : Fin C.N, gateSize (C.op g) = if g.val < k then 1 else 3) :
    ∀ j, j ≤ C.N → off C j = min j k + 3 * (j - k)
  | 0, _ => by simp [off]
  | j + 1, hj => by
    classical
    have ih := off_of hsize j (by omega)
    have hs : off C (j + 1) = off C j + gateSize (C.op ⟨j, by omega⟩) := by
      rw [off_eq_sum, off_eq_sum]
      have : (Finset.univ.filter fun h : Fin C.N => h.val < j + 1) =
          insert ⟨j, by omega⟩ (Finset.univ.filter fun h : Fin C.N => h.val < j) := by
        ext h
        simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_insert, Fin.ext_iff]
        omega
      rw [this, Finset.sum_insert (by simp), add_comm]
    rw [hs, ih, hsize]
    simp only
    split <;> omega

theorem posOf_of (hsize : ∀ g : Fin C.N, gateSize (C.op g) = if g.val < k then 1 else 3) (g : Fin C.N)
    (hg : g.val < k) : posOf C g = g.val := by
  rw [posOf_eq, off_of hsize g.val g.isLt.le, hsize, if_pos hg]
  omega

/-- **Unit `u`'s lowered gates** are positions `k + 3m·u` to `k + 3m·(u + 1) − 1`. -/
theorem mem_unitGates_of (hN : C.N = k + n * m)
    (hsize : ∀ g : Fin C.N, gateSize (C.op g) = if g.val < k then 1 else 3)
    (hunit : ∀ g : Fin C.N, (P.unit g).map Fin.val = if g.val < k then none else some ((g.val - k) / m))
    (u : Fin n) (i : ℕ) :
    i ∈ unitGates P u ↔ k + 3 * (m * u.val) ≤ i ∧ i < k + 3 * (m * u.val) + 3 * m := by
  rw [mem_unitGates]
  constructor
  · rintro ⟨g, hu, j, hj, rfl⟩
    have hv := hunit g
    rw [hu] at hv
    simp only [Option.map_some] at hv
    split at hv
    · cases hv
    · rename_i hk
      have hvu := (Option.some.inj hv).symm
      rw [hsize, if_neg hk] at hj
      rw [off_of hsize g.val g.isLt.le]
      have hm : 0 < m := Nat.pos_of_ne_zero fun h0 => by
        have := g.isLt; rw [h0] at hN; omega
      have h1 := Nat.div_mul_le_self (g.val - k) m
      have h2 := Nat.lt_mul_div_succ (g.val - k) hm
      rw [hvu] at h1 h2
      rw [Nat.mul_comm] at h1
      rw [Nat.mul_add, Nat.mul_one] at h2
      omega
  · rintro ⟨h1, h2⟩
    have hm : 0 < m := by omega
    have hu := u.isLt
    have hun : m * u.val + m ≤ n * m := by
      rw [Nat.mul_comm n, ← Nat.mul_succ]; exact Nat.mul_le_mul_left _ hu
    have hgl : k + (i - k) / 3 < C.N := by omega
    have hk : ¬ k + (i - k) / 3 < k := by omega
    refine ⟨⟨k + (i - k) / 3, hgl⟩, ?_, (i - k) % 3, ?_, ?_⟩
    · have hv := hunit ⟨k + (i - k) / 3, hgl⟩
      rw [if_neg hk] at hv
      obtain ⟨w, hw, hwv⟩ := Option.map_eq_some_iff.1 hv
      rw [hw]
      congr 1
      apply Fin.ext
      rw [hwv]
      show (k + (i - k) / 3 - k) / m = u.val
      apply Nat.div_eq_of_lt_le
      · rw [Nat.mul_comm]; omega
      · rw [Nat.add_mul, Nat.one_mul, Nat.mul_comm]; omega
    · rw [hsize, if_neg hk]; omega
    · rw [off_of hsize _ hgl.le]
      show i = min (k + (i - k) / 3) k + 3 * (k + (i - k) / 3 - k) + (i - k) % 3
      omega

theorem unitGates_pairwise (u : Fin n) : (unitGates P u).Pairwise (· < ·) := by
  unfold unitGates
  rw [List.pairwise_flatMap]
  refine ⟨fun g _ => ?_, ?_⟩
  · split
    · exact List.pairwise_lt_range.map _ fun a b h => by omega
    · exact List.Pairwise.nil
  · refine (List.pairwise_lt_finRange _).imp fun {g h} hgh => ?_
    intro x hx y hy
    split at hx
    · split at hy
      · obtain ⟨a, ha, rfl⟩ := List.mem_map.1 hx
        obtain ⟨b, hb, rfl⟩ := List.mem_map.1 hy
        have := off_add_le (C := C) hgh
        rw [List.mem_range] at ha
        omega
      · simp at hy
    · simp at hx

/-- **Unit `u`'s lowered gates, in order.** -/
theorem unitGates_of (hN : C.N = k + n * m)
    (hsize : ∀ g : Fin C.N, gateSize (C.op g) = if g.val < k then 1 else 3)
    (hunit : ∀ g : Fin C.N, (P.unit g).map Fin.val = if g.val < k then none else some ((g.val - k) / m))
    (u : Fin n) : unitGates P u = (List.range (3 * m)).map (k + 3 * m * u.val + ·) := by
  refine (unitGates_pairwise P u).eq_of_mem_iff (List.pairwise_lt_range.map _ fun a b h => by omega) fun i => ?_
  rw [mem_unitGates_of P hN hsize hunit, List.mem_map]
  simp only [List.mem_range, Nat.mul_assoc]
  constructor
  · rintro ⟨h1, h2⟩
    exact ⟨i - (k + 3 * (m * u.val)), by omega, by omega⟩
  · rintro ⟨a, ha, rfl⟩
    omega

/-- On a list without repeats, a predicate true only at `u` is found at `u`. -/
theorem find?_finRange_unique {p : Fin n → Bool} {u : Fin n} (hu : p u = true) (h : ∀ v, p v = true → v = u) :
    (List.finRange n).find? p = some u := by
  obtain ⟨x, hx⟩ := Option.isSome_iff_exists.1 (List.find?_isSome.2 ⟨u, List.mem_finRange u, hu⟩)
  rw [hx, h x (List.find?_some hx)]

/-- **Unit `u`'s owned inputs' positions, in order**: on `k = 2 + n·B` input gates, the constant, the zero and `B`
sources a unit, where unit `u'` reads input gate `2 + v·B + b` exactly when `u' = v` and `b` is a source some row reads
(`Rd`), they are `2 + u·B + b` for the `b ∈ S`, any ascending list of those sources. -/
theorem insOf_of (κ : Fin C.N → Option Bool) {B : ℕ} (hk : k = 2 + n * B) (hkN : k ≤ C.N)
    (hsize : ∀ g : Fin C.N, gateSize (C.op g) = if g.val < k then 1 else 3)
    (hin : ∀ g : Fin C.N, (C.op g).isInput = true ↔ g.val < k)
    (hκ : ∀ g : Fin C.N, g.val < k → (κ g = none ↔ 2 ≤ g.val))
    (Rd : ℕ → Prop)
    (hreads : ∀ (g : Fin C.N) (v : Fin n), 2 ≤ g.val → g.val < k →
      ((∃ h, P.unit h = some v ∧ g ∈ (C.op h).args) ↔ (g.val - 2) / B = v.val ∧ Rd ((g.val - 2) % B)))
    (S : List ℕ) (hS : S.Pairwise (· < ·)) (hSm : ∀ b, b ∈ S ↔ b < B ∧ Rd b) (u : Fin n) :
    (insOf P κ u).map (posOf C) = S.map (2 + u.val * B + ·) := by
  have hpos : ∀ a g : Fin C.N, a < g → posOf C a < posOf C g := fun a g h =>
    lt_of_lt_of_le (posOf_lt_off h) (by rw [posOf_eq g]; omega)
  have hl : ((insOf P κ u).map (posOf C)).Pairwise (· < ·) :=
    ((List.pairwise_lt_finRange _).filter _).map _ hpos
  refine hl.eq_of_mem_iff (hS.map _ fun a b h => by omega) fun i => ?_
  simp only [List.mem_map, mem_insOf]
  constructor
  · rintro ⟨g, ⟨hgi, hgκ, ho⟩, rfl⟩
    have hgk : g.val < k := (hin g).1 (by rw [hgi]; rfl)
    have h2 := (hκ g hgk).1 hgκ
    obtain ⟨hv, hrd⟩ := (hreads g u h2 hgk).1 (reads_of_owner ho)
    have hB : 0 < B := Nat.pos_of_ne_zero fun h0 => by rw [h0] at hk; omega
    refine ⟨(g.val - 2) % B, (hSm _).2 ⟨Nat.mod_lt _ hB, hrd⟩, ?_⟩
    rw [posOf_of hsize g hgk]
    have := Nat.div_add_mod' (g.val - 2) B
    rw [hv] at this
    omega
  · rintro ⟨b, hb, rfl⟩
    obtain ⟨hbB, hrd⟩ := (hSm b).1 hb
    have hu := u.isLt
    have hun : u.val * B + B ≤ n * B := by rw [← Nat.succ_mul]; exact Nat.mul_le_mul_right _ hu
    have hgk : 2 + u.val * B + b < k := by omega
    let g : Fin C.N := ⟨2 + u.val * B + b, by omega⟩
    have hdiv : (g.val - 2) / B = u.val := by
      simp only [g]
      rw [show 2 + u.val * B + b - 2 = b + u.val * B by omega, Nat.add_mul_div_right _ _ (by omega),
        Nat.div_eq_of_lt hbB, Nat.zero_add]
    have hmod : (g.val - 2) % B = b := by
      simp only [g]
      rw [show 2 + u.val * B + b - 2 = b + u.val * B by omega, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hbB]
    refine ⟨g, ⟨input_of_isInput ((hin g).2 hgk), (hκ g hgk).2 (by simp only [g]; omega), ?_⟩,
      posOf_of hsize g hgk⟩
    unfold owner
    refine find?_finRange_unique (decide_eq_true ((hreads g u (by simp only [g]; omega) hgk).2
      ⟨hdiv, by rw [hmod]; exact hrd⟩)) fun v hv => ?_
    have := ((hreads g v (by simp only [g]; omega) hgk).1 (of_decide_eq_true hv)).1
    exact Fin.ext (by omega)

end abstract

end Flock.Bridge.FilePart
