import Proofs.Flock.Claims.Decode

/-!
# A finite family of interleaved Reed–Solomon codes as one code, and its decoder

`Deploy` fixes one column type and one message type for every statement, while Flock's level-0 code depends on the
statement's `m` (its lanes, columns and padding). A finite family of `IRS` codes `C i`, `i : ι`, of `s i` lanes is one
code with columns `List F` and messages `Σ i, T i` (`IRS.famEnc`): message `⟨i, M⟩`'s column at `q` is `C i`'s, as a
list. Its Johnson count is the message's code's (`IRS.famNMin`). Its decoder (`IRS.famDecode`) runs each code's decoder
on every opened column read at that code's width (`IRS.colsAt`), and lists every result, so it meets `ListComplete` at
the family's Johnson count, at every `agree > 0` (`IRS.listComplete_famDecode`): reading keeps the number of columns
and their positions, and a column a message's encoding agrees with reads back as its code's column.
-/

namespace Claims

namespace IRS

/-- **A family's column**: message `⟨i, M⟩`'s column at `q`, `C i`'s `s i` lanes in order. -/
def famEnc {F ι : Type} [Field F] {s : ι → ℕ} {T : ι → Type} (C : (i : ι) → IRS F (T i) (s i))
    (M : Σ i, T i) (q : ℕ) : List F :=
  List.ofFn ((C M.1).enc M.2 q)

/-- **A family's Johnson count** at `agree`: message `⟨i, M⟩`'s code's, `C i`'s `nMin`. -/
noncomputable def famNMin {F ι : Type} [Field F] {s : ι → ℕ} {T : ι → Type} (C : (i : ι) → IRS F (T i) (s i))
    (agree : ℝ) (M : Σ i, T i) : ℕ :=
  (C M.1).nMin agree

/-- The opened columns read as columns of `s` lanes: lane `ℓ` is entry `ℓ`, `0` past the column's end. -/
def colsAt {F : Type} [Zero F] (s : ℕ) (os : List (ℕ × List F)) : List (ℕ × (Fin s → F)) :=
  os.map fun p => (p.1, fun ℓ => p.2.getD ℓ 0)

open Classical in
/-- **A family's decoder**: each code's decoder on the opened columns read at its width, every result listed. -/
noncomputable def famDecode {F ι : Type} [Field F] [DecidableEq F] [Fintype ι] {s : ι → ℕ} {T : ι → Type}
    [∀ i, Fintype (T i)] (C : (i : ι) → IRS F (T i) (s i)) (agree : ℝ) (os : List (ℕ × List F)) :
    List (Σ i, T i) :=
  (Finset.univ : Finset ι).toList.flatMap fun i => ((C i).decode agree (colsAt (s i) os)).map fun M => ⟨i, M⟩

theorem colsAt_fst {F : Type} [Zero F] (s : ℕ) (os : List (ℕ × List F)) :
    (colsAt s os).map Prod.fst = os.map Prod.fst := by
  unfold colsAt
  rw [List.map_map]
  rfl

theorem length_colsAt {F : Type} [Zero F] (s : ℕ) (os : List (ℕ × List F)) : (colsAt s os).length = os.length :=
  List.length_map _

/-- A column equal to a list of `s` values reads back as those `s` lanes. -/
theorem getD_of_ofFn {F : Type} [Zero F] {s : ℕ} (f : Fin s → F) (l : List F) (h : List.ofFn f = l) :
    (fun ℓ : Fin s => l.getD ℓ 0) = f := by
  subst h
  funext ℓ
  rw [List.getD_eq_getElem _ _ (by simp), List.getElem_ofFn]

/-- **A family's decoder meets `ListComplete`** for the family's columns, at the family's Johnson count, at every
`agree > 0`. -/
theorem listComplete_famDecode {F ι : Type} [Field F] [DecidableEq F] [Fintype ι] {s : ι → ℕ} {T : ι → Type}
    [∀ i, Fintype (T i)] (C : (i : ι) → IRS F (T i) (s i)) (agree : ℝ) (ha : 0 < agree) :
    Flock.ListComplete (famEnc C) agree (famNMin C agree) (famDecode C agree) := by
  classical
  rintro os ⟨i, M⟩ hnd hN hM
  simp only [famDecode, List.mem_flatMap, Finset.mem_toList, Finset.mem_univ, true_and, List.mem_map]
  have hlen := length_colsAt (s i) os
  refine ⟨i, M, (C i).listComplete_decode agree ha _ M (by rwa [colsAt_fst]) (by rw [hlen]; exact hN) ?_, rfl⟩
  have hcnt : (os.filter fun p => decide (famEnc C ⟨i, M⟩ p.1 = p.2)).length ≤
      ((colsAt (s i) os).filter fun p => decide ((C i).enc M p.1 = p.2)).length := by
    simp only [colsAt, List.filter_map, List.length_map]
    refine (List.monotone_filter_right os fun p hp => ?_).length_le
    simp only [famEnc] at hp
    simp only [Function.comp_apply, decide_eq_true_eq]
    exact (getD_of_ofFn _ _ (of_decide_eq_true hp)).symm
  rw [hlen]
  exact hM.trans (by exact_mod_cast hcnt)

end IRS

end Claims
