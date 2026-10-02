/-
What a laid-out net's rows force (`laid_holds`): if its computed rows (from the end of its input rows to its constant)
hold under the column values `v`, a row's `A` and `B` the XOR of `v` over its columns, and its constant column is 1, then
every row of the builder holds under `v ∘ col` (`stHolds`), and output `j`'s column holds output `j`'s form.
-/
import FlockSoundness.Discharge.Hm.Fresh
import FlockProofs.HmNets

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

/-- **A laid-out net's rows force its builder's rows and outputs.** Let `l` lay out the builder's state `st` (`nIn` input
bits, rows outputting `nIn, nIn + 1, …`: `Good`), with no more outputs than output bits. If `v` is 1 at `l`'s constant
column and every row `i` from the end of the input rows to the constant holds (`A·v` times `B·v` is `v i`), then every
row of `st` holds under `z = v ∘ col`, and output `j`'s column (`start + j`) holds `outs[j]`'s value under `z`. -/
theorem laid_holds (st : St) (inBits outBits : Array ℕ) (outs : Array Lin) (v : ℕ → Bool)
    (hg : Good (inBits.foldl (· + ·) 0) st) (hout : outs.size ≤ outBits.foldl (· + ·) 0)
    (hc : v (layout st inBits outBits outs).constPos = true)
    (hrow : ∀ i, groupWords (inBits.foldl (· + ·) 0) * Flock.WORD ≤ i → i < (layout st inBits outBits outs).constPos →
      (xorRow v ((layout st inBits outBits outs).ra.getD i #[]) &&
        xorRow v ((layout st inBits outBits outs).rb.getD i #[])) = v i) :
    stHolds (fun x => v (col (inBits.foldl (· + ·) 0) (groupWords (inBits.foldl (· + ·) 0) * Flock.WORD) x)) st ∧
      ∀ j (hj : j < outs.size),
        v ((groupWords (inBits.foldl (· + ·) 0) * Flock.WORD + st.rows.length +
            groupWords (outBits.foldl (· + ·) 0) * Flock.WORD - 1) /
          (groupWords (outBits.foldl (· + ·) 0) * Flock.WORD) * (groupWords (outBits.foldl (· + ·) 0) * Flock.WORD) + j) =
        ev (fun x => v (col (inBits.foldl (· + ·) 0) (groupWords (inBits.foldl (· + ·) 0) * Flock.WORD) x)) outs[j] := by
  obtain ⟨h1, -, h3, h4⟩ := layout_rows st inBits outBits outs
  have hspan := groupWords_pos (outBits.foldl (· + ·) 0)
  have hst := le_roundUp (groupWords (inBits.foldl (· + ·) 0) * Flock.WORD + st.rows.length)
    (groupWords (outBits.foldl (· + ·) 0) * Flock.WORD) hspan
  have hob := le_groupWords (outBits.foldl (· + ·) 0)
  refine ⟨fun r hr => ?_, fun j hj => ?_⟩
  · obtain ⟨k, hk, rfl⟩ := List.getElem_of_mem (List.mem_reverse.2 hr)
    have hk' : k < st.rows.length := by simpa using hk
    have hout := hg.2 k _ (List.getElem?_eq_getElem hk)
    obtain ⟨ha, hb⟩ := h3 k hk
    have := hrow (groupWords (inBits.foldl (· + ·) 0) * Flock.WORD + k) (by omega) (by rw [h1]; omega)
    rw [ha, hb, xorRow_cols v hc, xorRow_cols v hc] at this
    unfold rowHolds
    rw [hout]
    dsimp only
    rw [col_and, this]
  · obtain ⟨ha, hb⟩ := h4 j hj
    have := hrow ((groupWords (inBits.foldl (· + ·) 0) * Flock.WORD + st.rows.length +
        groupWords (outBits.foldl (· + ·) 0) * Flock.WORD - 1) / (groupWords (outBits.foldl (· + ·) 0) * Flock.WORD) *
        (groupWords (outBits.foldl (· + ·) 0) * Flock.WORD) + j) (by omega) (by rw [h1]; omega)
    rw [ha, hb, xorRow_cols v hc, xorRow_single, hc, Bool.and_true] at this
    exact this.symm

end FlockSoundness.Discharge.Hm
