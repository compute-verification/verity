import Models.Accounting.Guarantees.Actual
import Proofs.Accounting.Actual.Physical

/-!
# The capacity form: the windows divide the developer's region

A computed gate of the developer's region over `[0, t)` lies at exactly one location `X i` (they are disjoint) and in
exactly one window `k = ⌊t g / ω⌋ < ⌈t/ω⌉`, which names its unit `(i, k)`. A window's region is
`A[X i, [kω, min((k + 1)ω, t)))`: compute rate bounds its gates by `Φ_{X i} ω`; what it reads in the cut is what it
reads in `A`, which the split bounds by `mem_{X i}(kω) + in_{X i}([kω, …))`, and memory and links by
`M_{X i} + β_{X i} ω`.
-/

namespace Accounting.SecurityProofs.Actual

open Verity Nci Finset

variable {L : Type*} (A : Placed L)

theorem mem_into_iff (C : Circuit) {R : Finset (Fin C.N)} {a : Fin C.N} :
    a ∈ C.into R ↔ a ∉ R ∧ ∃ g ∈ R, a ∈ C.reads g := by
  simp [Circuit.into]

theorem cut_reads_of_mem {S : Finset (Fin A.N)} {g : Fin A.N} (h : g ∈ S) : (A.cut S).reads g = A.reads g := by
  simp [Placed.cut, h]

theorem cut_into {S R : Finset (Fin A.N)} (hR : R ⊆ S) : (A.cut S).into R = A.into R := by
  refine Finset.ext fun a => (mem_into_iff (A.cut S)).trans (Iff.trans ?_ (mem_into A).symm)
  refine and_congr_right fun _ => ⟨fun ⟨g, hg, h⟩ => ⟨g, hg, ?_⟩, fun ⟨g, hg, h⟩ => ⟨g, hg, ?_⟩⟩
  · rwa [cut_reads_of_mem A (hR hg)] at h
  · rwa [cut_reads_of_mem A (hR hg)]

theorem cut_isInput_of_mem {S : Finset (Fin A.N)} {g : Fin A.N} (h : g ∈ S) : (A.cut S).isInput g = A.isInput g := by
  simp [Placed.cut, h]

theorem cut_isInput_of_not_mem {S : Finset (Fin A.N)} {g : Fin A.N} (h : g ∉ S) : (A.cut S).isInput g = true := by
  simp [Placed.cut, h]

theorem gates_eq {C : Circuit} {n : ℕ} (P : Partition C n) (u : Fin n) (R : Finset (Fin C.N))
    (h : ∀ g, P.unit g = some u ↔ g ∈ R) : P.gates u = R := by
  ext g
  simp [Partition.gates, h]

theorem window_le {k ω t : ℕ} : min ((k + 1) * ω) t - k * ω ≤ ω := by
  rw [Nat.sub_le_iff_le_add]
  exact (min_le_left _ _).trans (by rw [add_one_mul]; omega)

theorem region_subset {X Y : Finset L} {I J : Set ℕ} (hX : X ⊆ Y) (hI : I ⊆ J) : A.region X I ⊆ A.region Y J :=
  fun g hg => by
    obtain ⟨hi, hx, ht⟩ := (mem_region A).1 hg
    exact (mem_region A).2 ⟨hi, hX hx, hI ht⟩

theorem lt_ceil_of_lt {a t ω : ℕ} (hω : 0 < ω) (h : a < t) : a / ω < (t + ω - 1) / ω := by
  rw [Nat.div_lt_iff_lt_mul hω]
  have := Nat.lt_div_mul_add (a := t + ω - 1) hω
  omega

theorem div_eq_iff_mem {a k ω : ℕ} (hω : 0 < ω) : a / ω = k ↔ a ∈ Set.Ico (k * ω) ((k + 1) * ω) := by
  rw [Set.mem_Ico, ← Nat.le_div_iff_mul_le hω, ← Nat.div_lt_iff_lt_mul hω]
  omega

variable [DecidableEq L] {n : ℕ} (X : Fin n → Finset L)

open Classical in
/-- The window of each gate: its location and `⌊t g / ω⌋`, for the computed gates of the region `S` whose window is
below `m`. -/
noncomputable def window (S : Finset (Fin A.N)) (ω m : ℕ) (g : Fin A.N) : Option (Fin (n * m)) :=
  if h : g ∈ S ∧ (∃ i, A.x g ∈ X i) ∧ A.t g / ω < m then
    some (finProdFinEquiv (Classical.choose h.2.1, ⟨A.t g / ω, h.2.2⟩))
  else none

theorem window_eq_some {S : Finset (Fin A.N)} {ω m : ℕ} (hX : ∀ i j, i ≠ j → Disjoint (X i) (X j)) {g : Fin A.N}
    {i : Fin n} {k : Fin m} :
    window A X S ω m g = some (finProdFinEquiv (i, k)) ↔ g ∈ S ∧ A.x g ∈ X i ∧ A.t g / ω = k := by
  unfold window
  split_ifs with h
  · rw [Option.some_inj, finProdFinEquiv.injective.eq_iff, Prod.mk.injEq]
    have hc := Classical.choose_spec h.2.1
    constructor
    · rintro ⟨hi, hk⟩
      exact ⟨h.1, hi ▸ hc, congrArg Fin.val hk⟩
    · rintro ⟨-, hi, hk⟩
      refine ⟨?_, Fin.ext hk⟩
      by_contra hne
      exact Finset.disjoint_left.1 (hX _ _ hne) hc hi
  · simp only [false_iff]
    rintro ⟨hS, hi, hk⟩
    exact h ⟨hS, ⟨i, hi⟩, hk ▸ k.2⟩

theorem Windows : Guarantees.Actual.Windows := by
  intro L E _ _ A ℓ Φ M ends β link n X θ hθ t ω hω hX hC hΦ hM hL
  classical
  set S := A.region (univ.biUnion X) (Set.Iio t)
  set m := (t + ω - 1) / ω
  have hSm : ∀ g ∈ S, A.t g / ω < m := fun g hg => lt_ceil_of_lt hω ((mem_region A).1 hg).2.2
  have hSX : ∀ g ∈ S, ∃ i, A.x g ∈ X i := fun g hg => by
    simpa using ((mem_region A).1 hg).2.1
  let P : Partition (A.declaration (univ.biUnion X) t θ hθ).D (n * m) :=
    { unit := window A X S ω m
      input_iff := fun (g : Fin A.N) => by
        change window A X S ω m g = none ↔ (A.cut S).isInput g = true
        unfold window
        by_cases hg : g ∈ S
        · rw [cut_isInput_of_mem A hg, ((mem_region A).1 hg).1]
          simp [hg, hSX g hg, hSm g hg]
        · rw [cut_isInput_of_not_mem A hg]
          simp [hg] }
  refine ⟨P, fun i k => ?_⟩
  have key : ∀ g : Fin A.N, window A X S ω m g = some (finProdFinEquiv (i, k)) ↔
      g ∈ A.region (X i) (Set.Ico (k * ω) ((k + 1) * ω) ∩ Set.Iio t) := fun g => by
    rw [window_eq_some A X hX, mem_region, mem_region, div_eq_iff_mem hω]
    simp only [mem_univ, true_and, mem_biUnion, Set.mem_inter_iff, Set.mem_Iio]
    constructor
    · rintro ⟨⟨hi, -, ht⟩, hx, hk⟩
      exact ⟨hi, hx, hk, ht⟩
    · rintro ⟨hi, hx, hk, ht⟩
      exact ⟨⟨hi, ⟨i, hx⟩, ht⟩, hx, hk⟩
  have hgates := gates_eq P (finProdFinEquiv (i, k)) (A.region (X i) (Set.Ico (k * ω) ((k + 1) * ω) ∩ Set.Iio t)) key
  refine ⟨hgates, ?_, ?_⟩
  · rw [hgates, Set.Ico_inter_Iio]
    change (A.region (X i) (Set.Ico (k * ω) (min ((k + 1) * ω) t))).card ≤ Φ (X i) * ω
    have h := hΦ (X i) ⟨i, rfl⟩ (k * ω) (min ((k + 1) * ω) t)
    rw [Placed.work, ← card_eq_sum_ones] at h
    exact h.trans (Nat.mul_le_mul_left _ window_le)
  · rw [hgates, Set.Ico_inter_Iio]
    set R := A.region (X i) (Set.Ico (k * ω) (min ((k + 1) * ω) t))
    have hRS : R ⊆ S := region_subset A (subset_biUnion_of_mem X (mem_univ i)) fun s hs => by
      simp only [Set.mem_Ico, lt_min_iff] at hs
      exact Set.mem_Iio.2 hs.2.2
    unfold novelReads
    change A.width ((A.cut S).into R \ θ) ≤ _
    rw [cut_into A hRS]
    refine (width_mono A sdiff_subset).trans ((width_into_region_le A _ _ _ (forward_of_causal A hC _)).trans ?_)
    exact Nat.add_le_add (hM (X i) ⟨i, rfl⟩ _) ((ingress_le A hL _ _ _).trans (Nat.mul_le_mul_left _ window_le))

end Accounting.SecurityProofs.Actual
