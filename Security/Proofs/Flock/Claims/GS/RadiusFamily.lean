import Proofs.Flock.Claims.GS.ZkRadius

/-!
# A family of radius decoders read as one, and the masked verifier's

`Deploy` fixes one column type, `List F`, and one trace type `Σ i, T i` for every statement (`IRS.famEnc`,
`DecodeFamily`). A family of decoders `dec i`, each meeting `ListWithin` for its code at its length and radius, is one
decoder (`famDecodeR`): each runs on the opened columns read at its width (`IRS.colsAt`), and every result is listed.

* `Flock.instZeroListCol`: the zero-filled word of a family's code is filled with the empty column `[]` where nothing
  was opened, and an empty column agrees with no trace's encoding.
* `fill_colsAt`: the zero-filled word read at a width is the read of the zero-filled word.
* `listWithin_famDecodeR`: the family's decoder meets `ListWithin` at the family's code read through any `t` that keeps
  a column's lanes (the trim of trailing zeros, `FlockDeploy.trimCol`, among them), with the length and the radius of
  a trace `⟨i, M⟩` its index's: a column a trace's encoding agrees with reads back as its code's column, so the
  positions it agrees with only grow under the read.
* `famDecodeR_complexity`: the family's count (`famCountedR`) is the sum over its finitely many indices of each one's
  read (`os.length·s i`) and count; at a size `Q` above every index's size and read, it is at most
  `card ι · (p + X)` at `Q`, `p` the indices' common polynomial.
* `length_famDecodeR_le`: the family lists at most `card ι · L` where each index lists at most `L`.

At the masked verifier's code (`Flock.zkEncD`, its traces `Flock.ZkTrace`, `m = 25..35`): `zkDecodeRD` meets
`ListWithin` at `Flock.zkLenD`, `zkEncD` read through such a `t`, and `Flock.zkThetaD`, the length `2^d` and the radius
`1 − √ρ_zk − 1/200` read off the trace's `m` (`listWithin_zkDecodeRD`), with its count at most `11·(zkPolyR + X)` at
the schedules' sizes summed (`zkSizeRD`, `zkDecodeRD_complexity`), and its lists at most that long
(`zkDecodeRD_length_le`).
-/

namespace Claims.Flock

/-- **The empty column**: the zero-filled word of a family's code (columns `List F`) has `[]` where no column was
opened. -/
instance instZeroListCol {F : Type} : Zero (List F) := ⟨[]⟩

theorem zero_list_eq {F : Type} : (0 : List F) = [] := rfl

end Claims.Flock

namespace Claims.GS

open Polynomial FlockSoundness FlockSoundness.Model FlockLevel3

/-- **The zero-filled word read at a width** is the read of the zero-filled word. -/
theorem fill_colsAt {F : Type} [Zero F] (s n : ℕ) (os : List (ℕ × List F)) (q : Fin n) :
    Flock.fill n (IRS.colsAt s os) q = fun ℓ : Fin s => (Flock.fill n os q).getD ℓ 0 := by
  have hp : ((fun p : ℕ × (Fin s → F) => p.1 == q.val) ∘ fun p : ℕ × List F => (p.1, fun ℓ : Fin s => p.2.getD ℓ 0)) =
      fun p => p.1 == q.val := rfl
  unfold Flock.fill IRS.colsAt
  rw [List.find?_map, hp]
  cases os.find? (fun p => p.1 == q.val) with
  | none =>
    funext ℓ
    simp [Flock.zero_list_eq]
  | some p => rfl

/-- **A family's radius decoder**: each index's decoder on the opened columns read at its width, every result
listed. -/
noncomputable def famDecodeR {F ι : Type} [Zero F] [Fintype ι] {s : ι → ℕ} {T : ι → Type}
    (dec : (i : ι) → List (ℕ × (Fin (s i) → F)) → List (T i)) (os : List (ℕ × List F)) : List (Σ i, T i) :=
  (Finset.univ : Finset ι).toList.flatMap fun i => (dec i (IRS.colsAt (s i) os)).map fun M => ⟨i, M⟩

/-- `famDecodeR` with its count: over the indices, each one's read of the opened columns at its width
(`os.length·s i`, one step per entry) and its decoder's count, summed. -/
noncomputable def famCountedR {F ι : Type} [Zero F] [Fintype ι] {s : ι → ℕ} {T : ι → Type}
    (counted : (i : ι) → List (ℕ × (Fin (s i) → F)) → List (T i) × ℕ) (os : List (ℕ × List F)) :
    List (Σ i, T i) × ℕ :=
  ((Finset.univ : Finset ι).toList.flatMap fun i => (counted i (IRS.colsAt (s i) os)).1.map fun M => ⟨i, M⟩,
    ∑ i, (os.length * s i + (counted i (IRS.colsAt (s i) os)).2))

/-- **A family's radius decoder meets `ListWithin`** at the family's code read through `t`, the length and radius of a
trace `⟨i, M⟩` its index's, when each index's decoder meets it for its code and `t` keeps a column's lanes: the
decoders read the opened columns lane by lane, `0` past a column's end. -/
theorem listWithin_famDecodeR {F ι : Type} [DecidableEq F] [Zero F] [Fintype ι] {s : ι → ℕ} {T : ι → Type}
    (len : (i : ι) → T i → ℕ) (enc : (i : ι) → T i → ℕ → Fin (s i) → F) (θ : (i : ι) → T i → ℝ)
    (dec : (i : ι) → List (ℕ × (Fin (s i) → F)) → List (T i)) (t : List F → List F)
    (ht : ∀ u ℓ, (t u).getD ℓ 0 = u.getD ℓ 0) (h : ∀ i, Flock.ListWithin (len i) (enc i) (θ i) (dec i)) :
    Flock.ListWithin (fun M : Σ i, T i => len M.1 M.2) (fun M q => t (List.ofFn (enc M.1 M.2 q)))
      (fun M => θ M.1 M.2) (famDecodeR dec) := by
  rintro os ⟨i, M⟩ hM
  simp only [famDecodeR, List.mem_flatMap, Finset.mem_toList, Finset.mem_univ, true_and, List.mem_map]
  refine ⟨i, M, h i (IRS.colsAt (s i) os) M (hM.trans ?_), rfl⟩
  have hsub : ((Finset.univ : Finset (Fin (len i M))).filter fun q => t (List.ofFn (enc i M q.val)) =
        Flock.fill (len i M) os q) ⊆
      (Finset.univ : Finset (Fin (len i M))).filter fun q => enc i M q.val =
        Flock.fill (len i M) (IRS.colsAt (s i) os) q := by
    refine Finset.monotone_filter_right _ fun q _ hq => ?_
    rw [fill_colsAt, ← hq]
    simp only [ht]
    exact (IRS.getD_of_ofFn _ _ rfl).symm
  exact_mod_cast Finset.card_le_card hsub

/-- **A family's radius decoder's complexity**: when each index's decoder has `DecoderComplexity` at its size and the
common polynomial `p`, the family's count is at most `card ι · (p + X)` at any size `Q` above every index's size and
read. -/
theorem famDecodeR_complexity {F ι : Type} [Zero F] [Fintype ι] {s : ι → ℕ} {T : ι → Type}
    (dec : (i : ι) → List (ℕ × (Fin (s i) → F)) → List (T i))
    (counted : (i : ι) → List (ℕ × (Fin (s i) → F)) → List (T i) × ℕ)
    (size : (i : ι) → List (ℕ × (Fin (s i) → F)) → ℕ) (p : Polynomial ℕ)
    (h : ∀ i, DecoderComplexity (dec i) (counted i) (size i) p) (Q : List (ℕ × List F) → ℕ)
    (hQ : ∀ i os, size i (IRS.colsAt (s i) os) ≤ Q os) (hR : ∀ i (os : List (ℕ × List F)), os.length * s i ≤ Q os) :
    DecoderComplexity (famDecodeR dec) (famCountedR counted) Q (C (Fintype.card ι) * (p + X)) := by
  intro os
  have e : ∀ i, (counted i (IRS.colsAt (s i) os)).1 = dec i (IRS.colsAt (s i) os) := fun i => (h i _).1
  refine ⟨by simp only [famCountedR, famDecodeR, e], ?_⟩
  simp only [famCountedR, eval_mul, eval_C, eval_add, eval_X]
  calc ∑ i, (os.length * s i + (counted i (IRS.colsAt (s i) os)).2) ≤ ∑ _i : ι, (Q os + p.eval (Q os)) :=
        Finset.sum_le_sum fun i _ => Nat.add_le_add (hR i os) ((h i _).2.trans (eval_mono_nat p (hQ i os)))
    _ = Fintype.card ι * (p.eval (Q os) + Q os) := by
        rw [Finset.sum_const, Finset.card_univ, smul_eq_mul, add_comm (Q os)]

/-- **A family's radius decoder lists at most `card ι · L`** where each index's decoder lists at most `L` on the opened
columns read at its width. -/
theorem length_famDecodeR_le {F ι : Type} [Zero F] [Fintype ι] {s : ι → ℕ} {T : ι → Type}
    (dec : (i : ι) → List (ℕ × (Fin (s i) → F)) → List (T i)) (os : List (ℕ × List F)) (L : ℕ)
    (h : ∀ i, (dec i (IRS.colsAt (s i) os)).length ≤ L) : (famDecodeR dec os).length ≤ Fintype.card ι * L := by
  unfold famDecodeR
  rw [List.length_flatMap]
  refine (List.sum_le_sum (g := fun _ => L) fun i _ => ?_).trans ?_
  · rw [List.length_map]
    exact h i
  · simp

/-! ## The masked verifier's radius decoder -/

/-- **The masked verifier's radius decoder**: each `m`'s masked radius decoder (`zkDecodeR`) on the opened columns read
at its width. -/
noncomputable def zkDecodeRD (os : List (ℕ × List GF128)) : List Flock.ZkTrace :=
  famDecodeR zkDecodeR os

/-- `zkDecodeRD` with its count. -/
noncomputable def zkCountedRD (os : List (ℕ × List GF128)) : List Flock.ZkTrace × ℕ :=
  famCountedR zkCountedR os

/-- **The masked family's size**: its schedules' sizes summed, `Σ_i (os.length + 2^{d_i})·s_i·128`. -/
def zkSizeRD (os : List (ℕ × List GF128)) : ℕ :=
  ∑ i : Flock.ZkIdx, (os.length + 2 ^ Flock.zkLen i) * Flock.zkWidth i * 128

/-- **The masked verifier's radius decoder meets `ListWithin`** at the deployed masked code read through any `t` that
keeps a column's lanes, with the length `2^d` and the radius `1 − √ρ_zk − 1/200` read off the trace's `m`. -/
theorem listWithin_zkDecodeRD (t : List GF128 → List GF128) (ht : ∀ u ℓ, (t u).getD ℓ 0 = u.getD ℓ 0) :
    Flock.ListWithin Flock.zkLenD (fun M q => t (Flock.zkEncD M q)) Flock.zkThetaD zkDecodeRD :=
  listWithin_famDecodeR (s := Flock.zkWidth)
    (T := fun i => Flock.ZkMessage GF128 (Flock.zkWidth i) (Flock.zkCols i) (Flock.zkPad i))
    (fun i _ => 2 ^ Flock.zkLen i) (fun i => (Flock.flockZkLevelZero i).enc)
    (fun i _ => 1 - Real.sqrt (((2 : ℝ) ^ Flock.zkCols i + Flock.zkPad i) / 2 ^ Flock.zkLen i) - 1 / 200) zkDecodeR
    t ht fun i => by
      rw [Flock.flockZkLevelZero, Flock.zkLevelZero_enc]
      exact listWithin_zkDecodeR i

/-- **The masked verifier's radius decoder's complexity**: at most `11·(zkPolyR + X)` at the schedules' sizes
summed. -/
theorem zkDecodeRD_complexity : DecoderComplexity zkDecodeRD zkCountedRD zkSizeRD (C 11 * (zkPolyR + X)) := by
  have hle : ∀ (i : Flock.ZkIdx) (os : List (ℕ × List GF128)),
      (os.length + 2 ^ Flock.zkLen i) * Flock.zkWidth i * 128 ≤ zkSizeRD os := fun i os =>
    Finset.single_le_sum (f := fun i : Flock.ZkIdx => (os.length + 2 ^ Flock.zkLen i) * Flock.zkWidth i * 128)
      (fun _ _ => Nat.zero_le _) (Finset.mem_univ i)
  have := famDecodeR_complexity zkDecodeR zkCountedR
    (fun i os => (os.length + 2 ^ Flock.zkLen i) * Flock.zkWidth i * 128) zkPolyR zkDecodeR_complexity zkSizeRD
    (fun i os => by
      rw [IRS.length_colsAt]
      exact hle i os)
    (fun i os => le_trans (calc os.length * Flock.zkWidth i
        _ ≤ (os.length + 2 ^ Flock.zkLen i) * Flock.zkWidth i := Nat.mul_le_mul_right _ (Nat.le_add_right _ _)
        _ ≤ (os.length + 2 ^ Flock.zkLen i) * Flock.zkWidth i * 128 := Nat.le_mul_of_pos_right _ (by norm_num))
      (hle i os))
  rw [Fintype.card_fin] at this
  exact this

/-- **The masked verifier's radius decoder lists at most `11·(zkPolyR + X)` at `zkSizeRD`**: each `m`'s decoder lists
at most `2^d`, one per point of the filled word (`length_gsDecode_le`), and `2^d` is within the size. -/
theorem zkDecodeRD_length_le (os : List (ℕ × List GF128)) :
    (zkDecodeRD os).length ≤ (C 11 * (zkPolyR + X)).eval (zkSizeRD os) := by
  have hi : ∀ i : Flock.ZkIdx, (zkDecodeR i (IRS.colsAt (Flock.zkWidth i) os)).length ≤ zkSizeRD os := by
    intro i
    unfold zkDecodeR
    refine (length_gsDecode_le _ _ _ _ _).trans ?_
    rw [length_fillPts, max_eq_right Nat.one_le_two_pow]
    refine le_trans ?_ (Finset.single_le_sum
      (f := fun i : Flock.ZkIdx => (os.length + 2 ^ Flock.zkLen i) * Flock.zkWidth i * 128)
      (fun _ _ => Nat.zero_le _) (Finset.mem_univ i))
    have hw : 1 ≤ Flock.zkWidth i * 128 := Nat.one_le_iff_ne_zero.2 (by positivity)
    calc 2 ^ Flock.zkLen i ≤ os.length + 2 ^ Flock.zkLen i := Nat.le_add_left _ _
      _ ≤ (os.length + 2 ^ Flock.zkLen i) * (Flock.zkWidth i * 128) := Nat.le_mul_of_pos_right _ hw
      _ = (os.length + 2 ^ Flock.zkLen i) * Flock.zkWidth i * 128 := by ring
  refine (length_famDecodeR_le zkDecodeR os _ hi).trans ?_
  simp only [Fintype.card_fin, eval_mul, eval_C, eval_add, eval_X]
  exact Nat.mul_le_mul_left 11 (Nat.le_add_left _ _)

end Claims.GS
