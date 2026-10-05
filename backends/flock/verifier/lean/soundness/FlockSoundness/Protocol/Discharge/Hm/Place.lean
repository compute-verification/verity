/-
A row net's slot in the accepted template statement. At a row of a `sha512x3` or `hm96` slot past the net's input rows,
the block matrices' row is the net's row placed in the slot plus Δ's first-loop pairs (`rowNet_entry`): Δ's other flat
pairs are at the row nets' input bits (`TemplateLayout.deltaIn`), and the typed tail's at unit columns, which are in the
root's and the parts' slots. So in every satisfying witness the slot holds its net's rows (`laidRows_of_sat`): the
computed rows hold, and the constant column is 1 (its row is `pin · pin`, or the pin itself). With `HmNets.check`, VU
`g`'s `sha512x3` and `hm96` slots hold the verifier's layouts past their 8192 and 4096 input rows (`shaSlot_of_sat`,
`hmSlot_of_sat`), `port_commit_rows`'s `hsha` and `hhm`.
-/
import FlockSoundness.Discharge.Hm.Nets
import FlockSoundness.Protocol.ExecTemplatePlace

open Flock FlockLevel3

namespace FlockSoundness.Discharge.Hm

open Compose Flock.DeriveAll

/-- The block's bit at column `y` of block `o` (0 past the block). -/
def blockBit (S : Model.Statement) (z : Model.Witness S.m) (o : Fin (2 ^ S.nbl)) (y : ℕ) : Bool :=
  FlockSoundness.bit (if h : y < 2 ^ S.kLog then S.block z o ⟨y, h⟩ else 0)

theorem sum_count_nat {k : ℕ} (L : List ℕ) (hL : ∀ y ∈ L, y < k) (v : Fin k → ZMod 2) :
    ∑ j : Fin k, ((L.count j.val : ℕ) : ZMod 2) * v j = (L.map fun y => if h : y < k then v ⟨y, h⟩ else 0).sum := by
  induction L with
  | nil => simp
  | cons c L ih =>
      have hc : c < k := hL c List.mem_cons_self
      simp only [List.count_cons, Nat.cast_add, add_mul, Finset.sum_add_distrib,
        ih fun y hy => hL y (List.mem_cons_of_mem _ hy), List.map_cons, List.sum_cons, dite_eq_left hc]
      rw [add_comm]
      congr 1
      rw [Finset.sum_eq_single ⟨c, hc⟩]
      · simp
      · intro j _ hj
        have : (c == j.val) = false := by
          simp only [beq_eq_false_iff_ne]
          exact fun h => hj (Fin.ext h.symm)
        simp [this]
      · simp

theorem bit_sum_row (f : ℕ → ZMod 2) (b : ℕ) (r : Array ℕ) :
    FlockSoundness.bit ((r.toList.map (b + ·)).map f).sum = xorRow (fun j => FlockSoundness.bit (f (b + j))) r := by
  unfold xorRow
  induction r.toList with
  | nil => rfl
  | cons y l ih => simp only [List.map_cons, List.sum_cons, bit_add, ih, List.foldr_cons]

/-- A row of `Sparse.ofRows`'s rows, its columns below the width. -/
theorem ofRows_cols_lt {rows : Array (Array ℕ)} {width : ℕ} {m : Sparse} (h : Sparse.ofRows rows width = .ok m)
    (i : ℕ) : ∀ y ∈ (rows.getD i #[]).toList, y < width := by
  intro y hy
  by_cases hi : i < rows.size
  · have hri : rows.getD i #[] = rows[i] := by simp [Array.getD_eq_getD_getElem?, hi]
    rw [hri] at hy
    exact (ofRows_spec h).2.2.2 _ (Array.getElem_mem_toList hi) y hy
  · have hri : rows.getD i #[] = #[] := by simp [Array.getD_eq_getD_getElem?, hi]
    rw [hri] at hy
    cases hy

theorem shaNet_inRows {c : Circuit} (h : Flock.HmNets.check c = .ok ()) :
    (c.nets.getD c.shaNet default).2.inWords * Flock.WORD = 8192 := by
  obtain ⟨-, -, -, -, -, hi, -, -, -, -, -, -, -, hw, -⟩ := Net.ofRows_ok (shaNet_ofRows h)
  rw [hw, ← hi, (shaNet_groups h).1]
  decide

theorem hmNet_inRows {c : Circuit} (h : Flock.HmNets.check c = .ok ()) :
    (c.nets.getD c.hmNet default).2.inWords * Flock.WORD = 4096 := by
  obtain ⟨-, -, -, -, -, hi, -, -, -, -, -, -, -, hw, -⟩ := Net.ofRows_ok (hmNet_ofRows h)
  rw [hw, ← hi, (hmNet_groups h).1]
  decide

section Slots

variable {types : String → Option CircuitType} {ls : Array Flock.Layout} {info : String → Option TableInfo}
  {words : String → Option (Array ℕ)} {unit : ℕ} {done : Array D} {st : Stmt} {tb : TypedBlock}

set_option hygiene false in
local notation "𝒖" => done.getD unit default

theorem rowNet_lt (T : TemplateLayout st tb words done 𝒖) {n : ℕ} (hn : n = st.c.shaNet ∨ n = st.c.hmNet) :
    n < st.c.nets.size := by
  rcases hn with rfl | rfl
  · exact T.shaNet.1
  · exact T.hmNet.1

theorem slot_lt_count (T : TemplateLayout st tb words done 𝒖) {n g i : ℕ} (hn : n = st.c.shaNet ∨ n = st.c.hmNet)
    (hg : g < st.c.g) (hi : i < st.c.perVu.getD n 0) : g * st.c.perVu.getD n 0 + i < (st.c.rangeOf n).count := by
  rw [(T.range (rowNet_lt T hn)).2.2.1]
  calc g * st.c.perVu.getD n 0 + i < g * st.c.perVu.getD n 0 + st.c.perVu.getD n 0 := by omega
    _ = (g + 1) * st.c.perVu.getD n 0 := (Nat.succ_mul _ _).symm
    _ ≤ st.c.g * st.c.perVu.getD n 0 := Nat.mul_le_mul_right _ hg

end Slots

end FlockSoundness.Discharge.Hm
