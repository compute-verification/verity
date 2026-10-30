/-
A row net's slot in the accepted template statement. At a row of a `sha512x3` or `hm96` slot past the net's input rows,
the block matrices' row is the net's row placed in the slot plus Δ's first-loop pairs (`rowNet_entry`): Δ's other flat
pairs are at the row nets' input bits (`TemplateLayout.deltaIn`), and the typed tail's at unit columns, which are in the
root's and the parts' slots. So in every satisfying witness the slot holds its net's rows (`laidRows_of_sat`): the
computed rows hold, and the constant column is 1 (its row is `pin · pin`, or the pin itself). With `HmNets.check`,
template instance `g`'s `sha512x3` and `hm96` slots hold the verifier's layouts past their 8192 and 4096 input rows
(`shaSlot_of_sat`, `hmSlot_of_sat`), `port_commit_rows`'s `hsha` and `hhm`.
-/
import Proofs.Flock.Soundness.Discharge.Hm.Nets
import Proofs.Flock.Soundness.ExecTemplatePlace

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

/-- **A row net's slot holds its net's rows** in every block of a satisfying witness of the accepted template
statement: each computed row past the input rows holds, and the slot's constant column is 1. -/
theorem laidRows_of_sat (hd : deriveChecked types ls info words unit = .ok done) (hul : unit < ls.size)
    (T : TemplateLayout st tb words done 𝒖) (U : UnitShape done 𝒖)
    {regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))} {z : Model.Witness (T.model regs).m}
    (hz : (T.model regs).Satisfies z) (o : Fin (2 ^ (T.model regs).nbl)) {n s : ℕ}
    (hn : n = st.c.shaNet ∨ n = st.c.hmNet) (hs : s < (st.c.rangeOf n).count) {l : Flock.HmNets.Laid}
    {name : String}
    (hl : Net.ofRows name l.useful l.constPos l.inGroups l.outGroups l.ra l.rb = .ok (st.c.nets.getD n default).2) :
    LaidRows l ((st.c.nets.getD n default).2.inWords * Flock.WORD)
      fun j => blockBit (T.model regs) z o ((st.c.rangeOf n).slot s + j) := by
  have hnn := rowNet_lt T hn
  obtain ⟨hm, hk, -, hsl⟩ := T.range hnn
  obtain ⟨-, ha, hb, -, hcp, -, -, hin, hca, hcb, -, -, -, -⟩ := Net.ofRows_ok hl
  have hfit : l.constPos < 2 ^ (st.c.rangeOf n).slotLog := by rw [hsl, ← hcp]; exact T.fit n
  have hat : ∀ off < 2 ^ (st.c.rangeOf n).slotLog, AtSlot st.c n s off ((st.c.rangeOf n).slot s + off) :=
    fun off ho => ⟨hs, ho, rfl⟩
  have hxlt : ∀ off < 2 ^ (st.c.rangeOf n).slotLog, (st.c.rangeOf n).slot s + off < 2 ^ st.c.kLog :=
    fun off ho => (hat off ho).lt T.bound hm
  have hbit : ∀ y (hy : y < 2 ^ (T.model regs).kLog), blockBit (T.model regs) z o y =
      FlockSoundness.bit ((T.model regs).block z o ⟨y, hy⟩) := by
    intro y hy
    unfold blockBit
    split
    · rfl
    · exact absurd hy ‹_›
  refine ⟨?_, fun off hlo hoff => ?_⟩
  · have hx := hxlt _ hfit
    show blockBit (T.model regs) z o ((st.c.rangeOf n).slot s + l.constPos) = true
    rw [hbit _ hx]
    by_cases hp : (st.c.rangeOf n).slot s + l.constPos = st.pin
    · have hpin : (⟨(st.c.rangeOf n).slot s + l.constPos, hx⟩ : Fin (2 ^ (T.model regs).kLog)) = (T.model regs).pin :=
        Fin.ext hp
      rw [hpin, hz.2.1 o]
      rfl
    · have hlo' : (st.c.nets.getD n default).2.inWords * Flock.WORD ≤ l.constPos := hin
      have hE := rowNet_entry hd hul T U hn (hat _ hfit) hlo' hl
      have hC := fun j => constPairs_count (pin := st.pin) T.disj T.slotLog T.fit hm hk (hat _ hfit) j
      have hrow : ∀ (r : Array (Array ℕ)), r.getD l.constPos #[] = #[l.constPos] →
          ∀ j : Fin (2 ^ (T.model regs).kLog),
          ((((r.getD l.constPos #[]).toList.map ((st.c.rangeOf n).slot s + ·)).count j.val +
            (constPairs st.c st.pin).count ((st.c.rangeOf n).slot s + l.constPos, j.val) : ℕ) : ZMod 2) =
            if j = (T.model regs).pin then 1 else 0 := by
        intro r hr j
        rw [hr, hC j.val, ite_eq_left ⟨hcp.symm, hp⟩]
        have hj : (j = (T.model regs).pin) ↔ j.val = st.pin := Fin.ext_iff
        by_cases h1 : j.val = (st.c.rangeOf n).slot s + l.constPos
        · have h2 : ¬ j.val = st.pin := fun h => hp (h1.symm.trans h)
          rw [ite_eq_right (fun h => h2 (hj.1 h)), ite_eq_left h1, ite_eq_right h2]
          simp only [List.map_cons, List.map_nil, List.count_singleton, h1, beq_self_eq_true, ite_true]
          decide
        · rw [ite_eq_right h1]
          have h3 : List.count j.val [(st.c.rangeOf n).slot s + l.constPos] = 0 :=
            List.count_eq_zero.2 (by simp [h1])
          simp only [List.map_cons, List.map_nil, h3, zero_add]
          by_cases h2 : j.val = st.pin
          · rw [ite_eq_left h2, ite_eq_left (hj.2 h2)]; rfl
          · rw [ite_eq_right h2, ite_eq_right (fun h => h2 (hj.1 h))]; rfl
      have hcopy : CopyRow (T.model regs) ⟨(st.c.rangeOf n).slot s + l.constPos, hx⟩ (T.model regs).pin :=
        ⟨fun j => ((hE j.val).1.trans (hrow l.ra hca j)), fun j => ((hE j.val).2.trans (hrow l.rb hcb j))⟩
      rw [hcopy.block_eq hz o, hz.2.1 o]
      rfl
  · have ho : off < 2 ^ (st.c.rangeOf n).slotLog := lt_trans hoff hfit
    have hx := hxlt _ ho
    have hE := rowNet_entry hd hul T U hn (hat _ ho) hlo hl
    have hC : ∀ j, (constPairs st.c st.pin).count ((st.c.rangeOf n).slot s + off, j) = 0 := fun j => by
      rw [constPairs_count T.disj T.slotLog T.fit hm hk (hat _ ho) j, ite_eq_right (fun h => by have := h.1; omega)]
    have hcols : ∀ (r : Array (Array ℕ)) (m : Sparse), Sparse.ofRows r (2 ^ (st.c.nets.getD n default).2.unitLog) = .ok m →
        ∀ y ∈ (r.getD off #[]).toList.map ((st.c.rangeOf n).slot s + ·), y < 2 ^ (T.model regs).kLog := by
      intro r m hr y hy
      obtain ⟨y', hy', rfl⟩ := List.mem_map.1 hy
      exact hxlt _ (by rw [hsl]; exact ofRows_cols_lt hr off y' hy')
    have hmv : ∀ (r : Array (Array ℕ))
        (M : Matrix (Fin (2 ^ (T.model regs).kLog)) (Fin (2 ^ (T.model regs).kLog)) (ZMod 2)),
        (∀ y ∈ (r.getD off #[]).toList.map ((st.c.rangeOf n).slot s + ·), y < 2 ^ (T.model regs).kLog) →
        (∀ j : Fin (2 ^ (T.model regs).kLog), M ⟨(st.c.rangeOf n).slot s + off, hx⟩ j =
          ((((r.getD off #[]).toList.map ((st.c.rangeOf n).slot s + ·)).count j.val +
            (constPairs st.c st.pin).count ((st.c.rangeOf n).slot s + off, j.val) : ℕ) : ZMod 2)) →
        FlockSoundness.bit (M.mulVec ((T.model regs).block z o) ⟨(st.c.rangeOf n).slot s + off, hx⟩) =
          xorRow (fun j => blockBit (T.model regs) z o ((st.c.rangeOf n).slot s + j)) (r.getD off #[]) := by
      intro r M hlt hM
      simp only [Matrix.mulVec, dotProduct, hM, hC, Nat.add_zero]
      rw [sum_count_nat _ hlt]
      exact bit_sum_row _ _ _
    have h := congrArg FlockSoundness.bit (hz.1 o ⟨(st.c.rangeOf n).slot s + off, hx⟩)
    rw [bit_mul] at h
    show _ = blockBit (T.model regs) z o ((st.c.rangeOf n).slot s + off)
    rw [hbit _ hx]
    refine Eq.trans ?_ h
    exact congrArg₂ (· && ·) (hmv l.ra (T.model regs).A₀ (hcols _ _ ha) fun j => (hE j.val).1).symm
      (hmv l.rb (T.model regs).B₀ (hcols _ _ hb) fun j => (hE j.val).2).symm

theorem slot_lt_count (T : TemplateLayout st tb words done 𝒖) {n g i : ℕ} (hn : n = st.c.shaNet ∨ n = st.c.hmNet)
    (hg : g < st.c.g) (hi : i < st.c.perVu.getD n 0) : g * st.c.perVu.getD n 0 + i < (st.c.rangeOf n).count := by
  rw [(T.range (rowNet_lt T hn)).2.2.1]
  calc g * st.c.perVu.getD n 0 + i < g * st.c.perVu.getD n 0 + st.c.perVu.getD n 0 := by omega
    _ = (g + 1) * st.c.perVu.getD n 0 := (Nat.succ_mul _ _).symm
    _ ≤ st.c.g * st.c.perVu.getD n 0 := Nat.mul_le_mul_right _ hg

/-- **Template instance `g`'s `sha512x3` slot `i` holds the verifier's `sha512x3` rows**, past its 8192 input rows, in
every block of a satisfying witness of the accepted template statement whose circuit passed `HmNets.check`. -/
theorem shaSlot_of_sat (hd : deriveChecked types ls info words unit = .ok done) (hul : unit < ls.size)
    (T : TemplateLayout st tb words done 𝒖) (U : UnitShape done 𝒖)
    {regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))} {z : Model.Witness (T.model regs).m}
    (hz : (T.model regs).Satisfies z) (o : Fin (2 ^ (T.model regs).nbl)) (hck : Flock.HmNets.check st.c = .ok ())
    {g i : ℕ} (hg : g < st.c.g) (hi : i < st.c.perVu.getD st.c.shaNet 0) :
    LaidRows (Flock.HmNets.shaLaid ()) 8192 fun j => blockBit (T.model regs) z o (st.c.slotCol st.c.shaNet g i + j) := by
  have h := laidRows_of_sat hd hul T U hz o (Or.inl rfl) (slot_lt_count T (Or.inl rfl) hg hi) (shaNet_ofRows hck)
  rw [shaNet_inRows hck] at h
  exact h

/-- **Template instance `g`'s `hm96` slot `i` holds the verifier's `hm96` rows**, past its 4096 input rows. -/
theorem hmSlot_of_sat (hd : deriveChecked types ls info words unit = .ok done) (hul : unit < ls.size)
    (T : TemplateLayout st tb words done 𝒖) (U : UnitShape done 𝒖)
    {regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))} {z : Model.Witness (T.model regs).m}
    (hz : (T.model regs).Satisfies z) (o : Fin (2 ^ (T.model regs).nbl)) (hck : Flock.HmNets.check st.c = .ok ())
    {g i : ℕ} (hg : g < st.c.g) (hi : i < st.c.perVu.getD st.c.hmNet 0) :
    LaidRows (Flock.HmNets.hmLaid ()) 4096 fun j => blockBit (T.model regs) z o (st.c.slotCol st.c.hmNet g i + j) := by
  have h := laidRows_of_sat hd hul T U hz o (Or.inr rfl) (slot_lt_count T (Or.inr rfl) hg hi) (hmNet_ofRows hck)
  rw [hmNet_inRows hck] at h
  exact h

end Slots

end FlockSoundness.Discharge.Hm
