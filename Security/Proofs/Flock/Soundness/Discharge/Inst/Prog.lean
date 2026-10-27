import Proofs.Flock.Soundness.Discharge.Inst.Place
import Proofs.Flock.Soundness.Discharge.Placed.Keyed

/-!
# The program of the whole template instance, and its `ProgPlaces`

`instProg c n` is `n` units of the template instance's composite rows (`instRows`): the `upv` unit copies, the stage nets'
slots and the lookup slots, joined by Δ's wires. Each composite input is a stacked unit input no wire fills, and sits on
the gate of its Δ source key (`keyOf`, `Aliased.srcKey` of the stacked input it is): two composite inputs of one unit
share a gate exactly when Δ copies one column into them, or both read the zero (`instProg_wire_key`). It is `keyProg`
with the unit's rows replaced by the template instance's, beside it.

At every drawn unit's site in an accepted untyped statement:
* the statement places the program's rows at the unit's template instance (`placement_inst`, moved along the parse);
* a composite input carries what its stacked unit input's copy position carries (`inst_block`), so inputs on one gate
  carry one value (`Accepted.inst_aliased`, from `Aliased.copyPos_of_key`).

So `Sites.progPlaces_instProg` is a `ProgPlaces` that takes only the sites and the parse.
-/

open Flock FlockLevel3

namespace FlockSoundness.Discharge.Inst

open Model Audit Compose Placed

/-! ## Keys and gates -/

/-- **Composite input `ℓ`'s key**: Δ's source key of the stacked unit input it is. -/
def keyOf (c : Flock.Circuit) (ℓ : ℕ) : Aliased.Src := Aliased.srcKey c ((inX c).getD ℓ 0)

/-- `n` composite inputs' keys: `none` at the zero, else the first input with the same key. -/
def instIdx (c : Flock.Circuit) (n : ℕ) (x : Fin n) : Option (Fin n) :=
  if keyOf c x.val = .zero then none else (List.finRange n).find? fun i => keyOf c i.val = keyOf c x.val

theorem instIdx_eq_none {c : Flock.Circuit} {n : ℕ} (x : Fin n) : instIdx c n x = none ↔ keyOf c x.val = .zero := by
  unfold instIdx
  split
  · exact iff_of_true rfl ‹_›
  · refine iff_of_false (fun h => ?_) ‹_›
    rw [List.find?_eq_none] at h
    exact h x (List.mem_finRange x) (by simp)

/-- **Equal codes are equal keys.** -/
theorem instIdx_eq {c : Flock.Circuit} {n : ℕ} {x x' : Fin n} (h : instIdx c n x = instIdx c n x') :
    keyOf c x.val = keyOf c x'.val := by
  by_cases hz : keyOf c x.val = .zero
  · have := (instIdx_eq_none x).2 hz
    rw [h] at this
    rw [hz, (instIdx_eq_none x').1 this]
  · have hz' : keyOf c x'.val ≠ .zero := fun hz' => hz ((instIdx_eq_none x).1 (h.trans ((instIdx_eq_none x').2 hz')))
    unfold instIdx at h
    rw [ite_eq_right hz, ite_eq_right hz'] at h
    rcases hf : (List.finRange n).find? (fun i => keyOf c i.val = keyOf c x.val) with _ | i
    · rw [List.find?_eq_none] at hf
      exact absurd (by simp) (hf x (List.mem_finRange x))
    · rw [hf] at h
      have h1 := List.find?_some hf
      have h2 := List.find?_some h.symm
      simp only [decide_eq_true_eq] at h1 h2
      exact h1.symm.trans h2

/-- **The program of the whole template instance on Δ's source keys**: `n` units of the template instance's composite
rows (`instRows`), composite input `x` of each on the gate of its key (`instIdx`), the zero's gate when it reads the
zero. -/
noncomputable abbrev instProg (c : Flock.Circuit) (n : ℕ) : Prog (gates (inGates n (instRows c).nIn) (instRows c) n) n :=
  uniform (inGates n (instRows c).nIn) (instRows c) (insOf n (instRows c).nIn (instIdx c (instRows c).nIn))
    (oneGate n _) (insOf_ne_one n _ _) n

/-- **Two composite inputs of one unit of `instProg` on one gate have one key**, in the program's own instance. -/
theorem instProg_wire_key (c : Flock.Circuit) {n : ℕ} (outs : Finset (Fin (gates (inGates n (instRows c).nIn) (instRows c) n)))
    (u : Fin n) (x x' : Fin ((instProg c n).rows u).w) (hx : x.val < ((instProg c n).rows u).nIn)
    (hx' : x'.val < ((instProg c n).rows u).nIn)
    (hw : (Const.Prog.instC (instProg c n) outs u).wire x = (Const.Prog.instC (instProg c n) outs u).wire x') :
    keyOf c x.val = keyOf c x'.val := by
  have e := congrArg Rows.nIn (rows_uniform _ _ _ _ (insOf_ne_one n _ (instIdx c (instRows c).nIn)) n u)
  have h1 : x.val < (instRows c).nIn := lt_of_lt_of_eq hx e
  have h2 : x'.val < (instRows c).nIn := lt_of_lt_of_eq hx' e
  rw [Const.Prog.instC_wire_input _ _ _ x hx, Const.Prog.instC_wire_input _ _ _ x' hx'] at hw
  have hv : (Const.Prog.insAt (instProg c n) u x.val).val = (Const.Prog.insAt (instProg c n) u x'.val).val :=
    congrArg Fin.val hw
  have e1 : (Const.Prog.insAt (instProg c n) u x.val).val =
      (insOf n (instRows c).nIn (instIdx c (instRows c).nIn) u.val ⟨x.val, h1⟩).val :=
    insAt_uniform n u x.val h1
  have e2 : (Const.Prog.insAt (instProg c n) u x'.val).val =
      (insOf n (instRows c).nIn (instIdx c (instRows c).nIn) u.val ⟨x'.val, h2⟩).val :=
    insAt_uniform n u x'.val h2
  exact instIdx_eq (n := (instRows c).nIn) (x := ⟨x.val, h1⟩) (x' := ⟨x'.val, h2⟩)
    ((insOf_eq_iff u.isLt ⟨x.val, h1⟩ ⟨x'.val, h2⟩).1 (Fin.ext (e1.symm.trans (hv.trans e2))))

/-! ## A composite input's value -/

section block

variable {tags : Tags} {st : Stmt} (L : FlockSoundness.Layout st) (F : Aliased.UnitFacts tags st)
  (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog)))

/-- **A composite input carries what its stacked unit input's copy position carries** (`Aliased.copyPos`), in every
block of a satisfying witness: composite input `x` is stacked input `(inX c)[x]` of template instance `g`. -/
theorem inst_block {g : ℕ} (hg : g < st.c.g) (x : Fin (instRows st.c).w) (hx : x.val < (instRows st.c).nIn)
    {z : Model.Witness (L.model regs).m} (hz : (L.model regs).Satisfies z) (o : Fin (2 ^ (L.model regs).nbl)) :
    (L.model regs).block z o
        (composePlace (L.model regs) (logical st.c) (logical_ordered st.c) (pos st.c g) (inst_hb L F regs hg) x) =
      (L.model regs).block z o (Aliased.copyPos L regs F hg ((inX st.c).getD x.val 0)) := by
  have hx' : x.val < (inCols st.c).length := hx
  obtain ⟨he, hm⟩ := inCols_getD hx'
  have hX : (inX st.c).getD x.val 0 < st.c.upv * (st.c.unit.inWords * WORD) := inX_lt hm
  have hW : 0 < st.c.unit.inWords * WORD := Nat.pos_of_ne_zero fun h => by rw [h, Nat.mul_zero] at hX; omega
  have hg0 : 0 < st.c.g := by omega
  have hpv := perVu_unit F.check L.unit_mem L.unit_kind L.unit_count hg0
  have hle : (inCols st.c).length ≤ (order st.c).length := (inv_order st.c).1.length_le
  unfold Aliased.copyPos
  rw [dite_eq_left hX]
  refine Aliased.copyAt_block L regs F _ (Nat.mod_lt _ hW) _ ?_ hz o
  have hv : (composePlace (L.model regs) (logical st.c) (logical_ordered st.c) (pos st.c g) (inst_hb L F regs hg) x).val =
      pos st.c g ((order st.c).getD x.val 0) :=
    (composePlace_val (S := L.model regs) (L := logical st.c) (h := logical_ordered st.c) (pos := pos st.c g)
      (hb := inst_hb L F regs hg) x).trans (ite_eq_left (by show x.val < (order st.c).length; omega))
  rw [hv, order_getD_in hx', he]
  unfold inCode
  rw [pos_eq, hpv]
  rfl

end block

/-- **Composite inputs with one key carry one value**, at template instance `g` of an accepted untyped statement, in every
block of a satisfying witness. -/
theorem Accepted.inst_aliased {T : Model.Statement} (a : Accepted T) {g : ℕ} (hg : g < a.st.c.g) (o : Fin (2 ^ T.nbl))
    (x x' : Fin (instRows a.st.c).w) (hx : x.val < (instRows a.st.c).nIn) (hx' : x'.val < (instRows a.st.c).nIn)
    (hk : keyOf a.st.c x.val = keyOf a.st.c x'.val) {z : Model.Witness T.m} (hz : T.Satisfies z) :
    T.block z o (Fin.cast a.kLog_eq (composePlace (a.layout.model a.regs) (logical a.st.c) (logical_ordered a.st.c)
        (pos a.st.c g) (inst_hb a.layout (Aliased.setupH_unitFacts a.ht a.h) a.regs hg) x)) =
      T.block z o (Fin.cast a.kLog_eq (composePlace (a.layout.model a.regs) (logical a.st.c) (logical_ordered a.st.c)
        (pos a.st.c g) (inst_hb a.layout (Aliased.setupH_unitFacts a.ht a.h) a.regs hg) x')) := by
  obtain ⟨tags, cf, pf, coins, tables, draw, partition, program, st, ht, h, regs, hT⟩ := a
  subst hT
  simp only [Fin.cast_eq_self]
  rw [inst_block (setupH_layout ht h) (Aliased.setupH_unitFacts ht h) regs hg x hx hz o,
    inst_block (setupH_layout ht h) (Aliased.setupH_unitFacts ht h) regs hg x' hx' hz o]
  obtain ⟨-, hm⟩ := inCols_getD (c := st.c) hx
  obtain ⟨-, hm'⟩ := inCols_getD (c := st.c) hx'
  rcases Aliased.copyPos_of_key (setupH_layout ht h) regs (Aliased.setupH_unitFacts ht h) hg (inX_lt hm) (inX_lt hm')
      hk with he | ⟨⟨s, hs, he⟩, ⟨s', hs', he'⟩⟩
  · rw [he]
  · rw [he, he', Aliased.zeroPos_block _ regs _ hs hz o, Aliased.zeroPos_block _ regs _ hs' hz o]

/-! ## `ProgPlaces` -/

section prog

variable {n : ℕ} {Reg : Type} {mPts : ℕ} {plan : Finset (Fin n) → Reg → List (TabSpec mPts)}
  {tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length}
  {c : Flock.Circuit} {si : Sites plan tab}
  (hparse : ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S),
    HmRow.parse (si.acc S R u h).tags (si.acc S R u h).cf (si.acc S R u h).tables = .ok c)

include hparse in
/-- **W6 for the template instance's program, from the parse**: each drawn unit's rows are the composite rows of its
table's accepted circuit. -/
theorem Sites.rows_instProg (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S) :
    (instProg c n).rows u = instRows (si.acc S R u h).st.c :=
  (rows_uniform _ _ _ _ _ n u).trans (congrArg instRows ((si.acc S R u h).c_eq (hparse S R u h))).symm

/-- **A drawn unit's columns in its table's block**: its template instance's composite columns (`composePlace` at
`pos`). -/
noncomputable def Sites.instCol (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S) :
    Fin ((instProg c n).rows u).w → Fin (2 ^ (Partition.tabOf plan tab S R u).S.kLog) :=
  fun x => Fin.cast (si.acc S R u h).kLog_eq
    (composePlace ((si.acc S R u h).layout.model (si.acc S R u h).regs) (logical (si.acc S R u h).st.c)
      (logical_ordered _) (pos (si.acc S R u h).st.c (si.vu S R u h))
      (inst_hb (si.acc S R u h).layout (Aliased.setupH_unitFacts (si.acc S R u h).ht (si.acc S R u h).h)
        (si.acc S R u h).regs (si.hvu S R u h))
      (Fin.cast (congrArg Rows.w (Sites.rows_instProg hparse S R u h)) x))

variable (outs : Finset (Fin (gates (inGates n (instRows c).nIn) (instRows c) n)))

/-- **`ProgPlaces` for the program of the whole template instance**, from the sites and the parse alone: each drawn unit
placed at its template instance in its table's accepted statement (`placement_inst`), its instance the program's own
(`Const.Prog.instC`), its composite inputs on one gate carrying one value (`Accepted.inst_aliased`). -/
noncomputable def Sites.progPlaces_instProg : ProgPlaces (instProg c n) outs plan tab where
  inst := Const.Prog.instC (instProg c n) outs
  o := si.o
  col := Sites.instCol hparse
  placed S R u h := Placement.sat <| placement_rows (Sites.rows_instProg hparse S R u h)
    (Accepted.placement_cast (si.acc S R u h).hT
      (placement_inst (si.acc S R u h).layout (Aliased.setupH_unitFacts (si.acc S R u h).ht (si.acc S R u h).h)
        (si.acc S R u h).regs (si.hvu S R u h)))
  aliased S R u h x x' hw hx hx' z hz := by
    have hc := (si.acc S R u h).c_eq (hparse S R u h)
    have e := congrArg Rows.nIn (Sites.rows_instProg hparse S R u h)
    have hk : keyOf (si.acc S R u h).st.c x.val = keyOf (si.acc S R u h).st.c x'.val := by
      rw [hc]; exact instProg_wire_key c outs u x x' hx hx' hw
    exact Accepted.inst_aliased (si.acc S R u h) (si.hvu S R u h) (si.o S R u h)
      (Fin.cast (congrArg Rows.w (Sites.rows_instProg hparse S R u h)) x)
      (Fin.cast (congrArg Rows.w (Sites.rows_instProg hparse S R u h)) x') (lt_of_lt_of_eq hx e)
      (lt_of_lt_of_eq hx' e) hk hz

/-- `progPlaces_instProg`'s instance is the program's own. -/
theorem Sites.progPlaces_instProg_inst (u : Fin n) :
    (Sites.progPlaces_instProg hparse outs).inst u = Const.Prog.instC (instProg c n) outs u :=
  rfl

end prog

end FlockSoundness.Discharge.Inst
