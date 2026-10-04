import FlockSoundness.Discharge.HiddenExec.Placed
import FlockSoundness.Discharge.ScopeG.Keys

set_option Elab.async false

/-!
# The VU program's places, from columns that put its inputs on their unit slots

A VU input of `vuProg` is a VU-stacked unit input no wire copies (`vuIns`), so at VU `g` its column is that input's unit
slot column (`vuInPos`), as `AcceptedH.col` places it in the unit-net program. `VuPlace` is what `ProgPlaces` needs
beyond that: at each drawn unit's site, the columns of the VU's rows (`col`), its inputs at `vuInPos` (`hcol`), and the
placement of the VU's rows (`placed`). From any `VuPlace`:

* `VuPlace.pp`: `ProgPlaces` for `vuProg`, its instance the program's own; its aliasing is Δ's, from keys: inputs on one
  gate have one key (`vuProg_wire_key`), and inputs with one key carry one value (`acc_key_block`: one copy position, or
  two forced-zero rows, `copyPos_of_key`);
* `VuPlace.hConst`: only a unit's constant column sits on the one gate;
* `VuPlace.hZero`: every column on the zero gate carries 0 (`acc_zero_block`).

The input groups past the leaf group enter only through keys: a leaf cut's key is a row's message bit, an unwired cut
input's the zero or a padding row (`Aliased.srcAt`), and a wired one is no VU input (`vuIns`).
-/

namespace FlockSoundness.Discharge.ScopeG

open Flock FlockLevel3 Model Audit Compose
open FlockSoundness.Discharge.HiddenExec

/-! ## Inputs with one key carry one value -/

/-- **Stacked inputs with one key copy one position, or each a forced-zero row** (`Aliased.copies_of_key`, pointwise). -/
theorem copyPos_of_key {tags : Tags} {st : Stmt} (L : Layout st) (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog)))
    (F : Aliased.UnitFacts tags st) {g x x' : ℕ} (hg : g < st.c.g) (hx : x < st.c.upv * (st.c.unit.inWords * WORD))
    (hx' : x' < st.c.upv * (st.c.unit.inWords * WORD)) (hkey : Aliased.srcKey st.c x = Aliased.srcKey st.c x') :
    Aliased.copyPos L regs F hg x = Aliased.copyPos L regs F hg x' ∨
      (Aliased.IsZeroPos L regs F (Aliased.copyPos L regs F hg x) ∧
        Aliased.IsZeroPos L regs F (Aliased.copyPos L regs F hg x')) := by
  by_cases hz : Aliased.srcKey st.c x = .zero
  · exact Or.inr ⟨Aliased.copyPos_zero L regs F hg hx hz, Aliased.copyPos_zero L regs F hg hx' (hkey ▸ hz)⟩
  by_cases hp : ∃ k off, Aliased.srcKey st.c x = .pad k off
  · obtain ⟨k, off, hpad⟩ := hp
    obtain ⟨hk1, ho1⟩ := Aliased.srcKey_pad hpad
    obtain ⟨hk2, ho2⟩ := Aliased.srcKey_pad (hkey ▸ hpad)
    left
    have : x = x' := by
      rw [← Nat.div_add_mod x (st.c.unit.inWords * WORD), ← Nat.div_add_mod x' (st.c.unit.inWords * WORD),
        ← hk1, ← ho1, ← hk2, ← ho2]
    rw [this]
  simp only [not_exists] at hp
  rcases Aliased.copyPos_key L regs F hg hx hz hp with h1 | ⟨hlt, h1⟩ <;>
    rcases Aliased.copyPos_key L regs F hg hx' (hkey ▸ hz) (fun k off => hkey ▸ hp k off) with h2 | ⟨hlt', h2⟩
  · exact Or.inl (Fin.ext (h1.trans (hkey ▸ h2.symm)))
  · exact absurd (h1 ▸ (Aliased.copyPos L regs F hg x).isLt) (hkey ▸ hlt')
  · exact absurd (h2 ▸ (Aliased.copyPos L regs F hg x').isLt) (hkey ▸ hlt)
  · exact Or.inr ⟨h1, h2⟩

section acc

variable {T : Model.Statement} (a : AcceptedH T) {c : Flock.Circuit} (hc : a.st.c = c) {g : ℕ} (hg : g < c.g)

include a hc hg in
/-- **Stacked unit inputs with one key carry one value** at VU `g`'s unit slots of an accepted hidden statement, in
every block of a satisfying witness. -/
theorem acc_key_block {x x' : ℕ} (hx : x < c.upv * unitW c) (hx' : x' < c.upv * unitW c)
    (hk : Aliased.srcKey c x = Aliased.srcKey c x') {r r' : Fin (2 ^ T.kLog)}
    (hr : r.val = c.unitRange.slot (g * c.upv + x / unitW c) + x % unitW c)
    (hr' : r'.val = c.unitRange.slot (g * c.upv + x' / unitW c) + x' % unitW c)
    {z : Model.Witness T.m} (hsat : T.Satisfies z) (o : Fin (2 ^ T.nbl)) :
    T.block z o r = T.block z o r' := by
  obtain ⟨tags, cf, pf, coins, tables, draw, partition, program, st, ht, h, L, hR, regs, hT⟩ := a
  subst hT
  subst hc
  unfold unitW at hx hx' hr hr'
  have Lf := flat_layout ht h
  have F := flat_unitFacts ht h
  have hW : 0 < st.c.unit.inWords * WORD := Nat.pos_of_ne_zero fun h0 => by rw [h0, Nat.mul_zero] at hx; omega
  have e1 : (L.model regs).block z o r = (L.model regs).block z o (Aliased.copyPos Lf regs F hg x) := by
    rw [copyAt_blockH L Lf regs F hR (Aliased.stack_copy_lt L hg hx) (Nat.mod_lt _ hW) r hr hsat o]
    unfold Aliased.copyPos
    rw [dite_eq_left hx]
  have e2 : (L.model regs).block z o r' = (L.model regs).block z o (Aliased.copyPos Lf regs F hg x') := by
    rw [copyAt_blockH L Lf regs F hR (Aliased.stack_copy_lt L hg hx') (Nat.mod_lt _ hW) r' hr' hsat o]
    unfold Aliased.copyPos
    rw [dite_eq_left hx']
  rw [e1, e2]
  rcases copyPos_of_key Lf regs F hg hx hx' hk with he | ⟨⟨s, hs, he⟩, ⟨s', hs', he'⟩⟩
  · rw [he]
  · rw [he, he', zeroPos_blockH L Lf regs F hs hsat o, zeroPos_blockH L Lf regs F hs' hsat o]

include a hc hg in
/-- **A stacked unit input with the zero key carries 0** at VU `g`'s unit slots of an accepted hidden statement. -/
theorem acc_zero_block {x : ℕ} (hx : x < c.upv * unitW c) (hz : Aliased.srcKey c x = .zero) {r : Fin (2 ^ T.kLog)}
    (hr : r.val = c.unitRange.slot (g * c.upv + x / unitW c) + x % unitW c)
    {z : Model.Witness T.m} (hsat : T.Satisfies z) (o : Fin (2 ^ T.nbl)) : T.block z o r = 0 := by
  obtain ⟨tags, cf, pf, coins, tables, draw, partition, program, st, ht, h, L, hR, regs, hT⟩ := a
  subst hT
  subst hc
  unfold unitW at hx hr
  have Lf := flat_layout ht h
  have F := flat_unitFacts ht h
  have hW : 0 < st.c.unit.inWords * WORD := Nat.pos_of_ne_zero fun h0 => by rw [h0, Nat.mul_zero] at hx; omega
  rw [copyAt_blockH L Lf regs F hR (Aliased.stack_copy_lt L hg hx) (Nat.mod_lt _ hW) r hr hsat o]
  have e : Aliased.copyAt Lf regs F (Aliased.stack_copy_lt L hg hx) (x % (st.c.unit.inWords * WORD)) =
      Aliased.copyPos Lf regs F hg x := by
    unfold Aliased.copyPos
    rw [dite_eq_left hx]
  rw [e]
  obtain ⟨s, hs, he⟩ := Aliased.copyPos_zero Lf regs F hg hx hz
  rw [he]
  exact zeroPos_blockH L Lf regs F hs hsat o

end acc

/-! ## The VU's inputs at their unit slots -/

/-- Every VU input is a stacked unit input of the VU. -/
theorem vuIns_lt (c : Flock.Circuit) {s : ℕ} (hs : s < (vuIns c).length) : (vuIns c).getD s 0 < c.upv * unitW c := by
  rw [List.getD_eq_getElem _ _ hs]
  have hm : (vuIns c)[s] ∈ vuIns c := List.getElem_mem hs
  generalize (vuIns c)[s] = v at hm ⊢
  unfold vuIns at hm
  simp only [List.mem_flatMap, List.mem_range, List.mem_map, List.mem_filter] at hm
  obtain ⟨k, hk, x, ⟨hx, -⟩, rfl⟩ := hm
  have hxW : x < unitW c := lt_of_lt_of_le hx (Nat.min_le_left _ _)
  calc k * unitW c + x < (k + 1) * unitW c := by rw [Nat.succ_mul]; omega
    _ ≤ c.upv * unitW c := Nat.mul_le_mul_right _ hk

theorem vuB_eq (c : Flock.Circuit) (hs : scopeOkGT c = true) : vuB c hs = (vuIns c).length := by
  show ((vuIns c).map (vuInCol c)).length = _
  rw [List.length_map]

/-- **VU input `s`'s column at VU `g`**: its stacked unit input's column in VU `g`'s unit slots (`AcceptedH.col`'s). -/
def vuInPos (c : Flock.Circuit) (g s : ℕ) : ℕ :=
  c.unitRange.slot (g * c.upv + (vuIns c).getD s 0 / unitW c) + (vuIns c).getD s 0 % unitW c

section place

variable {c : Flock.Circuit} {hs : scopeOkGT c = true} {n : ℕ} {Reg : Type} {mPts : ℕ}
  {plan : Finset (Fin n) → Reg → List (TabSpec mPts)} {tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length}

variable (c hs) in
/-- **The VU's rows placed at each drawn unit's site**: their columns in the block of the unit's table (`col`), each VU
input at its stacked unit input's column (`hcol`), and the placement there (`placed`). -/
structure VuPlace (si : SitesH plan tab) where
  col : ∀ S R u, u ∈ S → Fin (vuRows c hs).w → Fin (2 ^ (Partition.tabOf plan tab S R u).S.kLog)
  hcol : ∀ S R u (h : u ∈ S) (x : Fin (vuRows c hs).w), x.val < vuB c hs →
    (col S R u h x).val = vuInPos c (si.vu S R u h) x.val
  placed : ∀ S R u (h : u ∈ S), Placement (Partition.tabOf plan tab S R u).S (vuRows c hs) (col S R u h)

variable {si : SitesH plan tab} (P : VuPlace c hs si)
  (hparse : ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S),
    HmRow.parse (si.acc S R u h).tags (si.acc S R u h).cf (si.acc S R u h).tables = .ok c)
  (outs : Finset (Fin (vuN c hs n)))

/-- A drawn unit's columns of `vuProg`: `P.col` along `rows_vuProg`. -/
def VuPlace.colP (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S) :
    Fin ((vuProg c hs n).rows u).w → Fin (2 ^ (Partition.tabOf plan tab S R u).S.kLog) :=
  fun x => P.col S R u h (Fin.cast (congrArg Rows.w (rows_vuProg c hs n u)) x)

include hparse in
/-- **`aliased` at each drawn unit's site**: two inputs the program's own instance puts on one gate have one key
(`vuProg_wire_key`), so they carry one value (`acc_key_block`). -/
theorem VuPlace.aliased (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S) :
    ∀ x x', (Const.Prog.instC (vuProg c hs n) outs u).wire x = (Const.Prog.instC (vuProg c hs n) outs u).wire x' →
      x.val < ((vuProg c hs n).rows u).nIn → x'.val < ((vuProg c hs n).rows u).nIn →
      ∀ z : Witness (Partition.tabOf plan tab S R u).S.m, (Partition.tabOf plan tab S R u).S.Satisfies z →
        (Partition.tabOf plan tab S R u).S.block z (si.o S R u h) (P.colP S R u h x) =
          (Partition.tabOf plan tab S R u).S.block z (si.o S R u h) (P.colP S R u h x') := by
  intro x x' hw hx hx' z hsat
  have hk := vuProg_wire_key hs outs u x x' hx hx' hw
  have e := congrArg Rows.nIn (rows_vuProg c hs n u)
  have h1 : x.val < vuB c hs := lt_of_lt_of_eq hx e
  have h2 : x'.val < vuB c hs := lt_of_lt_of_eq hx' e
  have hc := (si.acc S R u h).c_eq (hparse S R u h)
  have hg : si.vu S R u h < c.g := by rw [← hc]; exact si.hvu S R u h
  have hx1 : (vuIns c).getD x.val 0 < c.upv * unitW c := vuIns_lt c (by rw [← vuB_eq c hs]; exact h1)
  have hx2 : (vuIns c).getD x'.val 0 < c.upv * unitW c := vuIns_lt c (by rw [← vuB_eq c hs]; exact h2)
  have hk' : Aliased.srcKey c ((vuIns c).getD x.val 0) = Aliased.srcKey c ((vuIns c).getD x'.val 0) := hk
  have hr1 : (P.colP S R u h x).val = c.unitRange.slot (si.vu S R u h * c.upv + (vuIns c).getD x.val 0 / unitW c) +
      (vuIns c).getD x.val 0 % unitW c := P.hcol S R u h _ h1
  have hr2 : (P.colP S R u h x').val = c.unitRange.slot (si.vu S R u h * c.upv + (vuIns c).getD x'.val 0 / unitW c) +
      (vuIns c).getD x'.val 0 % unitW c := P.hcol S R u h _ h2
  exact acc_key_block (si.acc S R u h) hc hg hx1 hx2 hk' hr1 hr2 hsat _

include hparse in
/-- **`ProgPlaces` for the VU's program**: each drawn unit placed by `P`, its instance the program's own
(`Const.Prog.instC`), aliased as Δ copies (`VuPlace.aliased`). -/
noncomputable def VuPlace.pp : ProgPlaces (vuProg c hs n) outs plan tab where
  inst := Const.Prog.instC _ outs
  o := si.o
  col := P.colP
  placed S R u h := Placed.placement_rows (rows_vuProg c hs n u) (P.placed S R u h)
  aliased := P.aliased hparse outs

/-- **`hConst` at `VuPlace.pp`**: only a unit's constant column sits on the one gate. -/
theorem VuPlace.hConst :
    ((vuProg c hs n).partition outs).ConstCols plan tab (P.pp hparse outs).derived.place {vuOneG c hs n} :=
  Const.constCols_of_instC _ (Zero.oneGate_uniform n) (fun _ => rfl)

/-- **`hZero` at `VuPlace.pp`**: an input on the zero gate has the zero key (`vuKeyIdx_eq_none`), so it carries 0
(`acc_zero_block`). -/
theorem VuPlace.hZero :
    ((vuProg c hs n).partition outs).ZeroCols plan tab (P.pp hparse outs).derived.place {vuZeroG c hs n} := by
  refine Zero.Prog.zeroCols_singleton (P.pp hparse outs) (zer := vuZeroG c hs n) (one := vuOneG c hs n)
    (Zero.unit_uniform n (vuZeroG c hs n) (Placed.zeroGate n (vuB c hs)).isLt)
    (fun he => by
      have := congrArg Fin.val he
      simp [vuZeroG, vuOneG, Placed.zeroGate, Placed.oneGate] at this)
    (fun u => ?_) ?_
  · show (Const.Prog.instC (vuProg c hs n) outs u).wire _ = _
    rw [Const.Prog.instC_wire_one]
    exact Zero.oneGate_uniform n u
  · intro S R u h x hx hw z hsat
    change (Const.Prog.instC (vuProg c hs n) outs u).wire x = vuZeroG c hs n at hw
    rw [Const.Prog.instC_wire_input _ _ _ x hx] at hw
    have hx1 : x.val < vuB c hs := lt_of_lt_of_eq hx (congrArg Rows.nIn (rows_vuProg c hs n u))
    have e1 : (Const.Prog.insAt (vuProg c hs n) u x.val).val =
        (Placed.insOf n (vuB c hs) (vuKeyIdx c (vuB c hs)) u.val ⟨x.val, hx1⟩).val :=
      Placed.insAt_uniform n u x.val hx1
    have hv := e1.symm.trans (congrArg Fin.val hw)
    have hnone : vuKeyIdx c (vuB c hs) ⟨x.val, hx1⟩ = none :=
      (Placed.insOf_eq_zero_iff u.isLt _).1 (Fin.ext hv)
    have hc := (si.acc S R u h).c_eq (hparse S R u h)
    have hg : si.vu S R u h < c.g := by rw [← hc]; exact si.hvu S R u h
    have hxs : (vuIns c).getD x.val 0 < c.upv * unitW c := vuIns_lt c (by rw [← vuB_eq c hs]; exact hx1)
    have hz : Aliased.srcKey c ((vuIns c).getD x.val 0) = .zero := (vuKeyIdx_eq_none (B := vuB c hs) ⟨x.val, hx1⟩).1 hnone
    have hr : ((P.pp hparse outs).col S R u h x).val =
        c.unitRange.slot (si.vu S R u h * c.upv + (vuIns c).getD x.val 0 / unitW c) +
          (vuIns c).getD x.val 0 % unitW c := P.hcol S R u h _ hx1
    exact acc_zero_block (si.acc S R u h) hc hg hxs hz hr hsat _

end place

end FlockSoundness.Discharge.ScopeG
