import Proofs.Flock.EndToEnd.Joint.Drawn

namespace Flock.Guarantees

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit
open scoped ENNReal

variable {W : ℕ → StratifiedBlock}

-- A
example {δ : ℝ} (l : List (BlockRec W)) (hl : ENNReal.ofReal δ ≤ (l.map BlockRec.escape).prod) :
    ENNReal.ofReal δ ≤ ∏ i : Fin l.length, BlockRec.escape l[i.1] := by
  rw [Fin.prod_univ_fun_getElem]; exact hl

-- A2
example {δ : ℝ} (l : List (BlockRec W)) (hl : ENNReal.ofReal δ ≤ ∏ i : Fin l.length, BlockRec.escape l[i.1]) :
    ENNReal.ofReal δ ≤ ∏ i : Fin l.length, (W l[i].idx).ideal.escape l[i].wrong :=
  hl.trans_eq (Finset.prod_congr rfl fun _ _ => rfl)

-- A3
example {δ : ℝ} (l : List (BlockRec W)) (hl : ENNReal.ofReal δ ≤ (l.map BlockRec.escape).prod) :
    ENNReal.ofReal δ ≤ ∏ i : Fin l.length, (W l[i].idx).ideal.escape l[i].wrong :=
  hl.trans_eq (Fin.prod_univ_fun_getElem l BlockRec.escape).symm

-- B
example (h : (t : ℕ) → Fin (W t).n → ℕ) (l : List (BlockRec W)) :
    (l.map (BlockRec.harm h)).sum = ∑ i : Fin l.length, BlockRec.harm h l[i.1] := by
  rw [Fin.sum_univ_fun_getElem]

-- B2
example (h : (t : ℕ) → Fin (W t).n → ℕ) (l : List (BlockRec W)) :
    ∑ i : Fin l.length, BlockRec.harm h l[i.1] = ∑ i : Fin l.length, ∑ u ∈ l[i].wrong, h l[i].idx u :=
  Finset.sum_congr rfl fun _ _ => rfl

-- C
example (h : (t : ℕ) → Fin (W t).n → ℕ) (δ : ℝ) (l : List (BlockRec W)) :
    BddAbove (Set.range fun B : {B : (i : Fin l.length) → Finset (Fin (W l[i].idx).n) //
      ENNReal.ofReal δ ≤ ∏ i, (W l[i].idx).ideal.escape (B i)} => ((∑ i, ∑ u ∈ B.1 i, h l[i].idx u : ℕ) : ℝ)) :=
  (Set.finite_range _).bddAbove

-- D
example (h : (t : ℕ) → Fin (W t).n → ℕ) (δ : ℝ) (l : List (BlockRec W))
    (B₀ : {B : (i : Fin l.length) → Finset (Fin (W l[i].idx).n) //
      ENNReal.ofReal δ ≤ ∏ i, (W l[i].idx).ideal.escape (B i)})
    (hb : BddAbove (Set.range fun B : {B : (i : Fin l.length) → Finset (Fin (W l[i].idx).n) //
      ENNReal.ofReal δ ≤ ∏ i, (W l[i].idx).ideal.escape (B i)} => ((∑ i, ∑ u ∈ B.1 i, h l[i].idx u : ℕ) : ℝ))) :
    ((∑ i, ∑ u ∈ B₀.1 i, h l[i].idx u : ℕ) : ℝ) ≤ Stratified.maxHarm W h δ l :=
  le_ciSup hb B₀

-- E0: the round-4 proof with a larger budget
set_option maxHeartbeats 1000000 in
example (h : (t : ℕ) → Fin (W t).n → ℕ) {δ : ℝ} {l : List (BlockRec W)}
    (hl : ENNReal.ofReal δ ≤ (l.map BlockRec.escape).prod) :
    (((l.map (BlockRec.harm h)).sum : ℕ) : ℝ) ≤ Stratified.maxHarm W h δ l := by
  have hB : ENNReal.ofReal δ ≤ ∏ i : Fin l.length, (W l[i].idx).ideal.escape l[i].wrong :=
    hl.trans_eq (Fin.prod_univ_fun_getElem l BlockRec.escape).symm
  have e : (l.map (BlockRec.harm h)).sum = ∑ i : Fin l.length, ∑ u ∈ l[i].wrong, h l[i].idx u :=
    (Fin.sum_univ_fun_getElem l (BlockRec.harm h)).symm
  rw [e]
  exact le_ciSup (f := fun B : {B : (i : Fin l.length) → Finset (Fin (W l[i].idx).n) //
      ENNReal.ofReal δ ≤ ∏ i, (W l[i].idx).ideal.escape (B i)} => ((∑ i, ∑ u ∈ B.1 i, h l[i].idx u : ℕ) : ℝ))
    (Set.finite_range _).bddAbove ⟨fun i => l[i].wrong, hB⟩

end Flock.Guarantees
