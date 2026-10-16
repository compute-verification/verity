import Proofs.Flock.Claims.Decode

/-!
# A finite family of interleaved Reed–Solomon codes as one code, and its decoder

`Deploy` fixes one column type and one message type for every statement, while Flock's level-0 code depends on the
statement's `m` (its lanes, columns and padding). A finite family of `IRS` codes `C i`, `i : ι`, of `s i` lanes is one
code with columns `List F` and messages `Σ i, T i` (`IRS.famEnc`): message `⟨i, M⟩`'s column at `q` is `C i`'s, as a
list. Its decoder (`IRS.famDecode`) runs each code's decoder on the opened columns of its length (`IRS.colsOfLen`) and
lists every result, so it meets `ListComplete` at every `agree` (`IRS.listComplete_famDecode`): a column a message's
encoding agrees with has that code's length, so the message's code sees every column it agrees with, among no more
columns.
-/

namespace Claims

namespace IRS

/-- **A family's column**: message `⟨i, M⟩`'s column at `q`, `C i`'s `s i` lanes in order. -/
def famEnc {F ι : Type} [Field F] {s : ι → ℕ} {T : ι → Type} (C : (i : ι) → IRS F (T i) (s i))
    (M : Σ i, T i) (q : ℕ) : List F :=
  List.ofFn ((C M.1).enc M.2 q)

/-- The opened columns of length `s`, as columns of `s` lanes. -/
def colsOfLen {F : Type} [Zero F] (s : ℕ) (os : List (ℕ × List F)) : List (ℕ × (Fin s → F)) :=
  (os.filter fun p => decide (p.2.length = s)).map fun p => (p.1, fun ℓ => p.2.getD ℓ 0)

open Classical in
/-- **A family's decoder**: each code's decoder on the opened columns of its length, every result listed. -/
noncomputable def famDecode {F ι : Type} [Field F] [DecidableEq F] [Fintype ι] {s : ι → ℕ} {T : ι → Type}
    [∀ i, Fintype (T i)] (C : (i : ι) → IRS F (T i) (s i)) (agree : ℝ) (os : List (ℕ × List F)) :
    List (Σ i, T i) :=
  (Finset.univ : Finset ι).toList.flatMap fun i => ((C i).decode agree (colsOfLen (s i) os)).map fun M => ⟨i, M⟩

theorem colsOfLen_nodup {F : Type} [Zero F] (s : ℕ) (os : List (ℕ × List F)) (hnd : (os.map Prod.fst).Nodup) :
    ((colsOfLen s os).map Prod.fst).Nodup := by
  unfold colsOfLen
  rw [List.map_map]
  exact (List.Sublist.map _ List.filter_sublist).nodup hnd

theorem length_colsOfLen_le {F : Type} [Zero F] (s : ℕ) (os : List (ℕ × List F)) :
    (colsOfLen s os).length ≤ os.length := by
  unfold colsOfLen
  rw [List.length_map]
  exact List.length_filter_le _ _

/-- A column equal to a list of `s` values is that list read back as `s` lanes. -/
theorem getD_of_ofFn {F : Type} [Zero F] {s : ℕ} (f : Fin s → F) (l : List F) (h : List.ofFn f = l) :
    l.length = s ∧ (fun ℓ : Fin s => l.getD ℓ 0) = f := by
  subst h
  refine ⟨List.length_ofFn, funext fun ℓ => ?_⟩
  rw [List.getD_eq_getElem _ _ (by simp), List.getElem_ofFn]

/-- **A family's decoder meets `ListComplete`** for the family's columns, at every `agree`. -/
theorem listComplete_famDecode {F ι : Type} [Field F] [DecidableEq F] [Fintype ι] {s : ι → ℕ} {T : ι → Type}
    [∀ i, Fintype (T i)] (C : (i : ι) → IRS F (T i) (s i)) (agree : ℝ) :
    Flock.ListComplete (famEnc C) agree (famDecode C agree) := by
  classical
  rintro os ⟨i, M⟩ hnd hM
  simp only [famDecode, List.mem_flatMap, Finset.mem_toList, Finset.mem_univ, true_and, List.mem_map]
  refine ⟨i, M, (C i).listComplete_decode agree _ M (colsOfLen_nodup (s i) os hnd) ?_, rfl⟩
  set os' := colsOfLen (s i) os
  have hcnt : (os.filter fun p => decide (famEnc C ⟨i, M⟩ p.1 = p.2)).length ≤
      (os'.filter fun p => decide ((C i).enc M p.1 = p.2)).length := by
    simp only [os', colsOfLen, List.filter_map, List.length_map, List.filter_filter]
    refine (List.monotone_filter_right os fun p hp => ?_).length_le
    simp only [famEnc] at hp
    obtain ⟨hl, hf⟩ := getD_of_ofFn _ _ (of_decide_eq_true hp)
    simp only [Function.comp_apply, Bool.and_eq_true, decide_eq_true_eq]
    exact ⟨hf.symm, hl⟩
  have hN := length_colsOfLen_le (s i) os
  rcases le_or_gt 0 agree with ha | ha
  · calc agree * os'.length ≤ agree * os.length := mul_le_mul_of_nonneg_left (by exact_mod_cast hN) ha
      _ ≤ _ := hM
      _ ≤ _ := by exact_mod_cast hcnt
  · calc agree * os'.length ≤ 0 := mul_nonpos_of_nonpos_of_nonneg ha.le (Nat.cast_nonneg _)
      _ ≤ _ := Nat.cast_nonneg _

end IRS

end Claims
