import Definitions.Accounting.Guarantees.Actual
import Proofs.Accounting.Actual.Physical
import Proofs.Nci.ReadsIn

/-!
# The capacity form: the windows divide the developer's region

`WindowsAt`: a computed gate of the developer's region over `[0, t)` lies at exactly one location `X i` (they are
disjoint) and, for cuts `τ` of `[0, t)`, in exactly one window `[τ j, τ (j + 1))` (`cuts_exists_mem`, `cuts_eq_of_mem`),
which names its unit `(i, j)`. Compute rate bounds a window's gates by `Φ_{X i} (τ (j + 1) − τ j)`; what it reads in the
cut is what it reads in `A`, and the split in `G` (`width_into_region_inter_le`) and links bound its reads in `G` by
`stateIn_{X i}(τ j, G) + β_{X i} (τ (j + 1) − τ j)`. No memory constraint is used.

`Windows` is `WindowsAt` at the cuts `min(kω, t)` (`uniformCuts`) and `G = univ`: a window of them is
`A[X i, [kω, (k + 1)ω) ∩ [0, t)]`, at most `ω` long, its novel reads are at most its reads
(`Nci.SecurityProofs.novelReads_le_readsIn`), and memory bounds the state at its start by `M_{X i}`.
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

theorem div_eq_iff_mem {a k ω : ℕ} (hω : 0 < ω) : a / ω = k ↔ a ∈ Set.Ico (k * ω) ((k + 1) * ω) := by
  rw [Set.mem_Ico, ← Nat.le_div_iff_mul_le hω, ← Nat.div_lt_iff_lt_mul hω]
  omega

/-- A window below `⌈t/ω⌉` starts before `t`. -/
theorem mul_lt_of_lt_ceil {k t ω : ℕ} (hω : 0 < ω) (hk : k < (t + ω - 1) / ω) : k * ω < t := by
  have h := (Nat.le_div_iff_mul_le hω).1 (Nat.lt_iff_add_one_le.1 hk)
  rw [add_one_mul] at h
  omega

/-- Every tick before `t` lies in a window of the cuts. -/
theorem cuts_exists_mem {t : ℕ} (κ : Cuts t) {s : ℕ} (hs : s < t) :
    ∃ j : Fin κ.J, s ∈ Set.Ico (κ.τ j.castSucc) (κ.τ j.succ) := by
  classical
  set F := univ.filter fun k : Fin (κ.J + 1) => κ.τ k ≤ s
  have hF : F.Nonempty := ⟨0, mem_filter.2 ⟨mem_univ _, by rw [κ.zero]; exact Nat.zero_le s⟩⟩
  have hm : κ.τ (F.max' hF) ≤ s := (mem_filter.1 (F.max'_mem hF)).2
  have hml : F.max' hF ≠ Fin.last κ.J := fun h => by
    rw [h, κ.last] at hm
    omega
  refine ⟨(F.max' hF).castPred hml, ?_, ?_⟩
  · rwa [Fin.castSucc_castPred]
  · by_contra hle
    have hmem := F.le_max' _ (mem_filter.2 ⟨mem_univ _, Nat.le_of_not_lt hle⟩)
    have h2 : (((F.max' hF).castPred hml).succ : ℕ) ≤ (F.max' hF : ℕ) := Fin.le_def.1 hmem
    rw [Fin.val_succ, Fin.coe_castPred] at h2
    omega

/-- A tick lies in at most one window of the cuts. -/
theorem cuts_eq_of_mem {t : ℕ} (κ : Cuts t) {s : ℕ} {j j' : Fin κ.J}
    (h : s ∈ Set.Ico (κ.τ j.castSucc) (κ.τ j.succ)) (h' : s ∈ Set.Ico (κ.τ j'.castSucc) (κ.τ j'.succ)) : j = j' := by
  have key : ∀ a b : Fin κ.J, a < b → κ.τ a.succ ≤ κ.τ b.castSucc := fun a b hab =>
    κ.mono (Fin.succ_le_castSucc_iff.2 hab)
  rcases lt_trichotomy j j' with hlt | heq | hgt
  · exact absurd (h.2.trans_le ((key j j' hlt).trans h'.1)) (lt_irrefl s)
  · exact heq
  · exact absurd (h'.2.trans_le ((key j' j hgt).trans h.1)) (lt_irrefl s)

variable [DecidableEq L] {n : ℕ} (X : Fin n → Finset L)

open Classical in
/-- The window of each gate under the cuts `κ`: its location and the window of its tick, for the gates of `S` at the
locations. -/
noncomputable def cutWindow (S : Finset (Fin A.N)) {t : ℕ} (κ : Cuts t) (g : Fin A.N) : Option (Fin (n * κ.J)) :=
  if h : g ∈ S ∧ (∃ i, A.x g ∈ X i) ∧ ∃ j : Fin κ.J, A.t g ∈ Set.Ico (κ.τ j.castSucc) (κ.τ j.succ) then
    some (finProdFinEquiv (Classical.choose h.2.1, Classical.choose h.2.2))
  else none

theorem cutWindow_eq_some {S : Finset (Fin A.N)} {t : ℕ} {κ : Cuts t} (hX : ∀ i j, i ≠ j → Disjoint (X i) (X j))
    {g : Fin A.N} {i : Fin n} {j : Fin κ.J} :
    cutWindow A X S κ g = some (finProdFinEquiv (i, j)) ↔
      g ∈ S ∧ A.x g ∈ X i ∧ A.t g ∈ Set.Ico (κ.τ j.castSucc) (κ.τ j.succ) := by
  unfold cutWindow
  split_ifs with h
  · rw [Option.some_inj, finProdFinEquiv.injective.eq_iff, Prod.mk.injEq]
    have hc := Classical.choose_spec h.2.1
    have hc' := Classical.choose_spec h.2.2
    constructor
    · rintro ⟨hi, hj⟩
      exact ⟨h.1, hi ▸ hc, hj ▸ hc'⟩
    · rintro ⟨-, hi, hj⟩
      refine ⟨?_, cuts_eq_of_mem κ hc' hj⟩
      by_contra hne
      exact Finset.disjoint_left.1 (hX _ _ hne) hc hi
  · simp only [false_iff]
    rintro ⟨hS, hi, hj⟩
    exact h ⟨hS, ⟨i, hi⟩, ⟨j, hj⟩⟩

theorem WindowsAt : Guarantees.Actual.WindowsAt := by
  intro L E _ _ A ℓ Φ ends β link n X θ hθ t κ G hX hC hΦ hL
  classical
  set S := A.region (univ.biUnion X) (Set.Iio t)
  have hSt : ∀ g ∈ S, A.t g < t := fun g hg => Set.mem_Iio.1 ((mem_region A).1 hg).2.2
  have hSX : ∀ g ∈ S, ∃ i, A.x g ∈ X i := fun g hg => by
    simpa using ((mem_region A).1 hg).2.1
  let P : Partition (A.declaration (univ.biUnion X) t θ hθ).D (n * κ.J) :=
    { unit := cutWindow A X S κ
      input_iff := fun (g : Fin A.N) => by
        change cutWindow A X S κ g = none ↔ (A.cut S).isInput g = true
        unfold cutWindow
        by_cases hg : g ∈ S
        · have hc : g ∈ S ∧ (∃ i, A.x g ∈ X i) ∧
              ∃ j : Fin κ.J, A.t g ∈ Set.Ico (κ.τ j.castSucc) (κ.τ j.succ) :=
            ⟨hg, hSX g hg, cuts_exists_mem κ (hSt g hg)⟩
          rw [cut_isInput_of_mem A hg, ((mem_region A).1 hg).1, dite_eq_left hc]
          simp
        · rw [cut_isInput_of_not_mem A hg]
          simp [hg] }
  refine ⟨P, fun i j => ?_⟩
  have hτ : κ.τ j.succ ≤ t := (κ.mono (Fin.le_last _)).trans κ.last.le
  have key : ∀ g : Fin A.N, cutWindow A X S κ g = some (finProdFinEquiv (i, j)) ↔
      g ∈ A.region (X i) (Set.Ico (κ.τ j.castSucc) (κ.τ j.succ)) := fun g => by
    rw [cutWindow_eq_some A X hX, mem_region, mem_region]
    constructor
    · rintro ⟨⟨hi, -, -⟩, hx, hk⟩
      exact ⟨hi, hx, hk⟩
    · rintro ⟨hi, hx, hk⟩
      exact ⟨⟨hi, mem_biUnion.2 ⟨i, mem_univ _, hx⟩, Set.mem_Iio.2 (hk.2.trans_le hτ)⟩, hx, hk⟩
  have hgates := gates_eq P (finProdFinEquiv (i, j)) (A.region (X i) (Set.Ico (κ.τ j.castSucc) (κ.τ j.succ))) key
  refine ⟨hgates, ?_, ?_⟩
  · rw [hgates]
    change (A.region (X i) (Set.Ico (κ.τ j.castSucc) (κ.τ j.succ))).card ≤ Φ (X i) * (κ.τ j.succ - κ.τ j.castSucc)
    have h := hΦ (X i) ⟨i, rfl⟩ (κ.τ j.castSucc) (κ.τ j.succ)
    rw [Placed.work, ← card_eq_sum_ones] at h
    exact h
  · rw [hgates]
    set R := A.region (X i) (Set.Ico (κ.τ j.castSucc) (κ.τ j.succ))
    have hRS : R ⊆ S :=
      region_subset A (subset_biUnion_of_mem X (mem_univ i)) fun s hs => Set.mem_Iio.2 (hs.2.trans_le hτ)
    unfold readsIn
    change A.width ((A.cut S).into R ∩ G) ≤ _
    rw [cut_into A hRS]
    exact (width_into_region_inter_le A _ _ _ G (forward_of_causal A hC _)).trans
      (Nat.add_le_add_left (ingress_le A hL _ _ _) _)

/-- **The uniform cuts** of `[0, t)` at the window `ω`: `τ k = min(kω, t)` for `k ≤ ⌈t/ω⌉`. -/
def uniformCuts (t ω : ℕ) (hω : 0 < ω) : Cuts t where
  J := (t + ω - 1) / ω
  τ k := min ((k : ℕ) * ω) t
  mono a b hab := by
    have : (a : ℕ) * ω ≤ (b : ℕ) * ω := Nat.mul_le_mul_right ω (Fin.le_def.1 hab)
    change min ((a : ℕ) * ω) t ≤ min ((b : ℕ) * ω) t
    omega
  zero := by simp
  last := by
    refine min_eq_right ?_
    have := Nat.lt_div_mul_add (a := t + ω - 1) hω
    change t ≤ (t + ω - 1) / ω * ω
    omega

theorem uniformCuts_castSucc {t ω : ℕ} (hω : 0 < ω) (k : Fin (uniformCuts t ω hω).J) :
    (uniformCuts t ω hω).τ k.castSucc = k * ω :=
  min_eq_left (mul_lt_of_lt_ceil hω k.2).le

theorem uniformCuts_succ {t ω : ℕ} (hω : 0 < ω) (k : Fin (uniformCuts t ω hω).J) :
    (uniformCuts t ω hω).τ k.succ = min ((k + 1) * ω) t :=
  rfl

theorem Windows : Guarantees.Actual.Windows := by
  intro L E _ _ A ℓ Φ M ends β link n X θ hθ t ω hω hX hC hΦ hM hL
  obtain ⟨P, hP⟩ := WindowsAt A ℓ Φ ends β link n X θ hθ t (uniformCuts t ω hω) univ hX hC hΦ hL
  refine ⟨P, fun i k => ?_⟩
  obtain ⟨hgates, hsize, hreads⟩ := hP i k
  have e : min ((k : ℕ) * ω) t = k * ω := min_eq_left (mul_lt_of_lt_ceil hω k.2).le
  have hst : A.stateIn (X i) (k * ω) univ ≤ M (X i) :=
    stateIn_le_of_memoryAt A (memoryAt_of_memory A hM Set.univ) ⟨i, rfl⟩ (Set.mem_univ _) _
  refine ⟨hgates.trans ?_, hsize.trans ?_, (Nci.SecurityProofs.novelReads_le_readsIn _ _).trans (hreads.trans ?_)⟩
  · show A.region (X i) (Set.Ico (min ((k : ℕ) * ω) t) (min (((k : ℕ) + 1) * ω) t)) = _
    rw [e]
    exact (congrArg (A.region (X i)) Set.Ico_inter_Iio).symm
  · show Φ (X i) * (min (((k : ℕ) + 1) * ω) t - min ((k : ℕ) * ω) t) ≤ _
    rw [e]
    exact Nat.mul_le_mul_left _ window_le
  · show A.stateIn (X i) (min ((k : ℕ) * ω) t) univ +
        linkCapacity ends β (X i) * (min (((k : ℕ) + 1) * ω) t - min ((k : ℕ) * ω) t) ≤ _
    rw [e]
    exact Nat.add_le_add hst (Nat.mul_le_mul_left _ window_le)

end Accounting.SecurityProofs.Actual
