module

public import Proofs.Flock.Soundness.ExecRows
public import Properties.Flock.Circuit.Keyed
public import Spot.Bits

/-!
# Bits' rows are the audit's (`Spot.Bits.Rows`)

`Spot.Bits` reads a circuit file's unit net as `Bits.Rows`, its computed rows as lists of columns; the audit
reads it as `FlockSoundness.Rows`, its columns `Fin`-indexed. `RowsAgree R R'`: the same inputs and computed rows, each
computed row reading the same columns in A and in B.

* `rowsAgree_ofNet`: a built net's rows agree, through `Rows.ofNet_a` and `Rows.ofNet_b`.
* `RowsAgree.stack`: `k` stacked copies of agreeing rows agree, copy column by copy column (`stackCol_val`) and row by
  row (`stackRow_val`).
* `rowsAgree_circuitRows`: so a circuit's template instance rows (`Placed.circuitRows`) are Bits' stacked unit rows.
-/

@[expose] public section

namespace Flock.Bridge

open FlockSoundness

/-- **Bits' rows `R` are the audit's `R'`**: the same inputs and computed rows, each computed row reading the same
columns in A and in B. -/
def RowsAgree (R : Spot.Bits.Rows) (R' : Rows) : Prop :=
  R.nIn = R'.nIn ∧ R.nComp = R'.nComp ∧
    ∀ i : Fin R'.nComp, R.a.getD i.val [] = (R'.a i).map Fin.val ∧ R.b.getD i.val [] = (R'.b i).map Fin.val

theorem getD_map_range {α : Type} (f : ℕ → α) {n i : ℕ} (hi : i < n) (d : α) :
    ((Array.range n).map f).getD i d = f i := by
  simp [Array.getD, hi]

/-- **A built net's rows agree.** -/
theorem rowsAgree_ofNet {net : Flock.Net} (h : Built net) (hu : net.unitLog ≤ 32) :
    RowsAgree (Spot.Bits.Rows.ofNet net) (Rows.ofNet h) :=
  ⟨rfl, rfl, fun i =>
    ⟨(getD_map_range (fun i => (net.a.row (min (net.inWords * Flock.WORD) net.constPos + i)).toList) i.isLt _).trans
        (Rows.ofNet_a h hu i).symm,
      (getD_map_range (fun i => (net.b.row (min (net.inWords * Flock.WORD) net.constPos + i)).toList) i.isLt _).trans
        (Rows.ofNet_b h hu i).symm⟩⟩

/-- Copy `u`'s column `c` is the same column in both stacks. -/
theorem stackCol_val {R : Spot.Bits.Rows} {R' : Rows} (hR : RowsAgree R R') (k : ℕ) {u : ℕ} (hu : u < k)
    (c : Fin R'.w) : (R'.stackCol k hu c).val = R.stackCol k u c.val := by
  unfold Rows.stackCol Spot.Bits.Rows.stackCol
  rw [hR.1, hR.2.1]
  split_ifs <;> rfl

/-- Row `i` of both stacks reads the same columns, from rows `f` and `F` that do. -/
theorem stackRow_val {R : Spot.Bits.Rows} {R' : Rows} (hR : RowsAgree R R') (k : ℕ)
    {f : Fin R'.nComp → List (Fin R'.w)} {F : Array (List ℕ)}
    (hf : ∀ j : Fin R'.nComp, F.getD j.val [] = (f j).map Fin.val) (i : Fin (k * (R'.nComp + 1))) :
    (R'.stackRow k f i).map Fin.val = R.stackRow k F i.val := by
  unfold Rows.stackRow Spot.Bits.Rows.stackRow
  rw [hR.1, hR.2.1]
  split_ifs with hj
  · rfl
  · have hlt : i.val % (R'.nComp + 1) - 1 < R'.nComp := by
      have := Nat.mod_lt i.val (show 0 < R'.nComp + 1 by omega)
      omega
    rw [show F.getD (i.val % (R'.nComp + 1) - 1) [] = _ from hf ⟨_, hlt⟩, List.map_map, List.map_map]
    exact List.map_congr_left fun c _ => stackCol_val hR k _ c

/-- **Stacking keeps agreement.** -/
theorem RowsAgree.stack {R : Spot.Bits.Rows} {R' : Rows} (hR : RowsAgree R R') (k : ℕ) :
    RowsAgree (R.stack k) (R'.stack k) := by
  refine ⟨by simp only [Spot.Bits.Rows.stack, Rows.stack, hR.1],
    by simp only [Spot.Bits.Rows.stack, Rows.stack, hR.2.1], fun i => ⟨?_, ?_⟩⟩
  · have hi : i.val < k * (R.nComp + 1) := by rw [hR.2.1]; exact i.isLt
    exact (getD_map_range (R.stackRow k R.a) hi _).trans
      (stackRow_val hR k (fun j => (hR.2.2 j).1) i).symm
  · have hi : i.val < k * (R.nComp + 1) := by rw [hR.2.1]; exact i.isLt
    exact (getD_map_range (R.stackRow k R.b) hi _).trans
      (stackRow_val hR k (fun j => (hR.2.2 j).2) i).symm

/-- **A circuit's template instance rows are Bits' stacked unit rows.** -/
theorem rowsAgree_circuitRows (c : Flock.Circuit) (hU : Built c.unit) (hu : c.unit.unitLog ≤ 32) :
    RowsAgree ((Spot.Bits.Rows.ofNet c.unit).stack c.upv)
      (FlockSoundness.Discharge.Placed.circuitRows c hU) :=
  (rowsAgree_ofNet hU hu).stack c.upv

end Flock.Bridge
