/-
What a laid-out net's rows force: if its computed rows (from the end of its input rows to its constant) hold under the
column values `v`, a row's `A` and `B` the XOR of `v` over its columns, and its constant column is 1 (`LaidRows`), then
every row of the builder holds under `v ∘ col` (`laid_rows`), and output `j`'s column holds output `j`'s form
(`laid_outs`).
-/
import Proofs.Flock.Soundness.Discharge.Hm.Fresh
import Proofs.Flock.Verifier.HmNets

namespace FlockSoundness.Discharge.Hm

open Flock.HmNets hiding and

/-- The XOR of the column values `v` over a row's columns. -/
def xorRow (v : ℕ → Bool) (cs : Array ℕ) : Bool := cs.toList.foldr (fun j b => v j ^^ b) false

theorem xorRow_cols (v : ℕ → Bool) {nIn inRows constPos : ℕ} (hc : v constPos = true) (f : Lin) :
    xorRow v (cols nIn inRows constPos f) = ev (fun x => v (col nIn inRows x)) f := by
  unfold xorRow cols ev
  rw [Array.toList_append, List.foldr_append, List.toList_toArray, List.foldr_map]
  cases f.c <;> simp [hc]

theorem xorRow_single (v : ℕ → Bool) (j : ℕ) : xorRow v #[j] = v j := by simp [xorRow]

/-- A laid-out net's input rows: `groupWords nIn` words. -/
abbrev inRowsOf (inBits : Array ℕ) : ℕ := groupWords (inBits.foldl (· + ·) 0) * Flock.WORD

/-- A laid-out net's output group's rows. -/
abbrev spanOf (outBits : Array ℕ) : ℕ := groupWords (outBits.foldl (· + ·) 0) * Flock.WORD

/-- The column of a laid-out net's first output: past the input rows and the ANDs, aligned to the output group. -/
def outStart (st : St) (inBits outBits : Array ℕ) : ℕ :=
  (inRowsOf inBits + st.rows.length + spanOf outBits - 1) / spanOf outBits * spanOf outBits

theorem layout_outGroups (st : St) (inBits outBits : Array ℕ) (outs : Array Lin) :
    (layout st inBits outBits outs).outGroups =
      #[⟨outStart st inBits outBits / Flock.WORD, groupWords (outBits.foldl (· + ·) 0), outBits⟩] := by
  simp [layout, outStart, inRowsOf, spanOf]

theorem layout_inGroups (st : St) (inBits outBits : Array ℕ) (outs : Array Lin) :
    (layout st inBits outBits outs).inGroups = #[⟨0, groupWords (inBits.foldl (· + ·) 0), inBits⟩] := rfl

theorem outStart_word (st : St) (inBits outBits : Array ℕ) :
    outStart st inBits outBits / Flock.WORD * Flock.WORD = outStart st inBits outBits := by
  unfold outStart spanOf
  rw [← Nat.mul_assoc, Nat.mul_div_cancel _ (by decide : 0 < Flock.WORD)]

/-- **A net laid out by `layout` holds at the column values `v`**: its constant column is 1, and every row from the end
of its input rows to its constant holds, `A·v` times `B·v` is `v i`, each the XOR of `v` over the row's columns. -/
def LaidRows (l : Laid) (nRows : ℕ) (v : ℕ → Bool) : Prop :=
  v l.constPos = true ∧
    ∀ i, nRows ≤ i → i < l.constPos → (xorRow v (l.ra.getD i #[]) && xorRow v (l.rb.getD i #[])) = v i

/-- **A laid-out net's rows force its builder's rows.** Let `layout` lay out the builder's state `st` (`nIn` input bits,
rows outputting `nIn, nIn + 1, …`: `Good`). If its rows hold at `v`, every row of `st` holds under `z = v ∘ col`. -/
theorem laid_rows (st : St) (inBits outBits : Array ℕ) (outs : Array Lin) (v : ℕ → Bool)
    (hg : Good (inBits.foldl (· + ·) 0) st) (h : LaidRows (layout st inBits outBits outs) (inRowsOf inBits) v) :
    stHolds (fun x => v (col (inBits.foldl (· + ·) 0) (inRowsOf inBits) x)) st := by
  obtain ⟨h1, -, h3, -⟩ := layout_rows st inBits outBits outs
  have hspan := groupWords_pos (outBits.foldl (· + ·) 0)
  have hst := le_roundUp (inRowsOf inBits + st.rows.length) (spanOf outBits) hspan
  intro r hr
  obtain ⟨k, hk, rfl⟩ := List.getElem_of_mem (List.mem_reverse.2 hr)
  have hk' : k < st.rows.length := by simpa using hk
  have hout := hg.2 k _ (List.getElem?_eq_getElem hk)
  obtain ⟨ha, hb⟩ := h3 k hk
  have := h.2 (inRowsOf inBits + k) (by omega) (by rw [h1]; unfold inRowsOf spanOf at *; omega)
  rw [ha, hb, xorRow_cols v h.1, xorRow_cols v h.1] at this
  unfold rowHolds
  rw [hout]
  dsimp only
  rw [col_and, this]

/-- **A laid-out net's rows force its outputs**: with no more outputs than output bits, output `j`'s column
(`outStart + j`) holds `outs[j]`'s value under `z = v ∘ col`. -/
theorem laid_outs (st : St) (inBits outBits : Array ℕ) (outs : Array Lin) (v : ℕ → Bool)
    (hout : outs.size ≤ outBits.foldl (· + ·) 0) (h : LaidRows (layout st inBits outBits outs) (inRowsOf inBits) v)
    (j : ℕ) (hj : j < outs.size) :
    v (outStart st inBits outBits + j) = ev (fun x => v (col (inBits.foldl (· + ·) 0) (inRowsOf inBits) x)) outs[j] := by
  obtain ⟨h1, -, -, h4⟩ := layout_rows st inBits outBits outs
  have hspan := groupWords_pos (outBits.foldl (· + ·) 0)
  have hst := le_roundUp (inRowsOf inBits + st.rows.length) (spanOf outBits) hspan
  have hob := le_groupWords (outBits.foldl (· + ·) 0)
  obtain ⟨ha, hb⟩ := h4 j hj
  have := h.2 (outStart st inBits outBits + j) (by unfold outStart; omega)
    (by rw [h1]; unfold outStart inRowsOf spanOf at *; omega)
  unfold outStart at this
  rw [ha, hb, xorRow_cols v h.1, xorRow_single, h.1, Bool.and_true] at this
  exact this.symm

end FlockSoundness.Discharge.Hm
