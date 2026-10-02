/-
A row net's slot in the accepted template statement. At a row of a `sha512x3` or `hm96` slot past the net's input rows,
the block matrices' row is the net's row placed in the slot plus Δ's first-loop pairs (`rowNet_entry`): Δ's other flat
pairs are at the row nets' input bits (`TemplateLayout.deltaIn`), and the typed tail's at unit columns, which are in the
root's and the parts' slots. So in every satisfying witness the slot holds its net's rows (`laidRows_of_sat`): the
computed rows hold, and the constant column is 1 (its row is `pin · pin`, or the pin itself).
-/
import FlockSoundness.Discharge.Hm.Nets
import FlockSoundness.ExecTemplatePlace

open Flock FlockLevel3

namespace FlockSoundness.Discharge.Hm

open Compose Flock.DeriveAll Types.Dag

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

/-- **The block rows at a row net's row past its input rows**: the net's row, placed in the slot, and Δ's first-loop
pairs there. -/
theorem rowNet_entry (hd : deriveChecked types ls info words unit = .ok done) (hul : unit < ls.size)
    (T : TemplateLayout st tb words done 𝒖) (U : UnitShape done 𝒖) {n s off x : ℕ}
    (hn : n = st.c.shaNet ∨ n = st.c.hmNet) (hat : AtSlot st.c n s off x)
    (hlo : (st.c.nets.getD n default).2.inWords * Flock.WORD ≤ off) {l : Flock.HmNets.Laid} {name : String}
    (hl : Net.ofRows name l.useful l.constPos l.inGroups l.outGroups l.ra l.rb = .ok (st.c.nets.getD n default).2)
    (j : ℕ) :
    rangesEntry st (slotEntryA st) x j + deltaEntry st.da x j =
        ((((l.ra.getD off #[]).toList.map ((st.c.rangeOf n).slot s + ·)).count j +
          (constPairs st.c st.pin).count (x, j) : ℕ) : ZMod 2) ∧
      rangesEntry st (slotEntryB st) x j + deltaEntry st.db x j =
        ((((l.rb.getD off #[]).toList.map ((st.c.rangeOf n).slot s + ·)).count j +
          (constPairs st.c st.pin).count (x, j) : ℕ) : ZMod 2) := by
  have hnn := rowNet_lt T hn
  obtain ⟨hm, hk, -, hsl⟩ := T.range hnn
  have h32 : (st.c.nets.getD n default).2.unitLog ≤ 32 := by
    rw [← hsl]; exact le_trans (hat.slotLog_le T.bound hm) T.kLog
  have hw : 2 ^ (st.c.nets.getD n default).2.unitLog ≤ 2 ^ 32 := Nat.pow_le_pow_right (by norm_num) h32
  obtain ⟨-, ha, hb, -, -, -, -, -, -, -, -, -, hlk, -⟩ := Net.ofRows_ok hl
  have hx : ∀ y, DeltaIn st.c y → y ≠ x := by
    intro y hy hyx
    obtain ⟨n', hn', s', off', hat', hoff'⟩ := T.deltaIn y hy
    have hnn' := rowNet_lt T hn'
    obtain ⟨rfl, -, rfl⟩ := hat'.inj T.disj (T.range hnn').1 (T.range hnn').2.1 hm hk (hyx ▸ hat)
    omega
  have hy : ∀ g' < st.c.g, ∀ u, InUnit done 𝒖 u → NotConst 𝒖 u → Typed.col st.c tb g' u ≠ x := by
    intro g' hg' u hu _ hux
    rcases T.at_inUnit hd hg' hu with ⟨-, hat'⟩ | ⟨i, hi, hiu, -, -, -, hat'⟩
    · obtain ⟨h1, -⟩ := hat'.inj T.disj (T.range T.unitNet).1 (T.range T.unitNet).2.1 hm hk (hux ▸ hat)
      rcases hn with rfl | rfl
      · exact T.shaNet.2 h1.symm
      · exact T.hmNet.2 h1.symm
    · obtain ⟨-, hpn, -, hps, hph, -⟩ := T.part i hi hiu
      obtain ⟨h1, -⟩ := hat'.inj T.disj (T.range hpn).1 (T.range hpn).2.1 hm hk (hux ▸ hat)
      rcases hn with rfl | rfl
      · exact hps h1
      · exact hph h1
  obtain ⟨hda, hdb⟩ := T.delta_count hx j
  have hre := fun e => rangesEntry_at T.disj hm hk hat e j
  have hla := ofRows_cols_lt ha off
  have hlb := ofRows_cols_lt hb off
  refine ⟨?_, ?_⟩
  · rw [hre, hda, T.tail_zero hd hul U hy true j, Nat.add_zero]
    show (if _ then sparseEntry (st.c.nets.getD n default).2.a off _ else 0) + _ = _
    rw [sparseEntry_ofRows ha, ofRows_row ha hw, count_slot _ (by rw [hsl]; exact hla) j]
    push_cast; ring
  · rw [hre, hdb, T.tail_zero hd hul U hy false j, Nat.add_zero]
    show (if _ then sparseEntry (st.c.nets.getD n default).2.b off _ +
      lookupEntry (st.c.nets.getD n default).2.lookup off _ else 0) + _ = _
    rw [hlk]
    show (if _ then sparseEntry (st.c.nets.getD n default).2.b off _ + 0 else 0) + _ = _
    rw [add_zero, sparseEntry_ofRows hb, ofRows_row hb hw, count_slot _ (by rw [hsl]; exact hlb) j]
    push_cast; ring

end Slots

end FlockSoundness.Discharge.Hm
