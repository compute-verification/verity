import Proofs.Flock.Soundness.Discharge.Zero.Cols
import Proofs.Flock.Soundness.Discharge.Zero.Stack
import Proofs.Flock.Soundness.Discharge.Aliased.Keys
import Proofs.Flock.Soundness.Discharge.Placed.Wiring
import Proofs.Flock.Soundness.Discharge.Placed.Keyed
import Proofs.Flock.Soundness.Discharge.Const.Inst

/-!
# `hZero` at the circuit's program (e2e, untyped statements)

The untyped statement `Stmt.setupH` accepted places a template instance (`upv` stacked copies of the unit net) at each drawn unit's site
(`Placed.Sites`, `Placed.progPlaces`), and Δ copies into each stacked input the column `Aliased.copyPos` names.
- `copyPos_zero`: at a stacked input that reads the zero (`ZeroIn`), that column is a unit slot's forced-zero row.
- `stack_zero_block`, `accepted_zero_block`: so in every block of a satisfying witness the input carries 0
  (`Aliased.copyPos_block`, `Aliased.zeroPos_block`).
- `zeroCols_uniform`: `hZero` for the program of `n` units of the circuit's template instance rows (`Placed.uniform`) at the sites, each
  unit's instance the program's own (`Const.Prog.instC`), when every input the program wires to its zero gate reads the
  zero. `zeroCols_circuitProg`: the same on source-indexed input gates (`Placed.circuitProg`), when every input with no
  source reads the zero.
- `srcKey_zero_iff`: Δ's source key (`Aliased.srcKey`) at a stacked input is the zero exactly when the input reads the
  zero, so `zeroCols_keyIdx`, on the gates indexed by those keys (`Aliased.keyIdx`), assumes nothing about the wiring.
  `zeroCols_progPlaces` is the same at `Placed.progPlaces`, and `zeroCols_keyProg` at `Placed.Sites.progPlaces_keyProg`,
  which takes only the sites and the parse.
-/

open Flock FlockLevel3

namespace FlockSoundness.Discharge.Zero

open Model Audit

/-! ## Δ's copy positions at the zero -/

section copies

variable {tags : Tags} {st : Stmt} (L : Layout st) (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog)))
  (F : Aliased.UnitFacts tags st)

/-- **Δ's copy position at a stacked input that reads the zero is a forced-zero row**: template instance `g`'s input
`x`'s (`Aliased.copyPos`) is its slot's forced-zero row (a zero leaf) or the first slot's (a cut bit past 16). -/
theorem copyPos_zero {g : ℕ} (hg : g < st.c.g) {x : ℕ} (hx : x < st.c.upv * (st.c.unit.inWords * WORD))
    (hz : ZeroIn st.c x) : Aliased.IsZeroPos L regs F (Aliased.copyPos L regs F hg x) := by
  have hs := Aliased.stack_copy_lt L hg hx
  unfold Aliased.copyPos Aliased.copyAt
  rw [dite_eq_left hx, ite_eq_left (show x % (st.c.unit.inWords * WORD) ∈ portBits st.c.unit.inGroups from hz.1)]
  split
  · rcases flatSrcAt_zeroIn hz g with he | he
    · exact ⟨_, hs, Fin.ext he⟩
    · exact ⟨0, Nat.lt_of_le_of_lt (Nat.zero_le _) hs, Fin.ext he⟩
  · exact ⟨_, hs, rfl⟩

include F in
/-- **A stacked input that reads the zero carries 0**: at template instance `g`'s place (`Layout.col`), in every block
of a satisfying witness. -/
theorem stack_zero_block {g : ℕ} (hg : g < st.c.g) (x : Fin ((unitRows st L.unit_built).stackW st.c.upv))
    (hx : x.val < ((unitRows st L.unit_built).stack st.c.upv).nIn) (hz : ZeroIn st.c x.val)
    {z : Model.Witness (L.model regs).m} (hsat : (L.model regs).Satisfies z) (o : Fin (2 ^ (L.model regs).nbl)) :
    (L.model regs).block z o (L.col regs hg x) = 0 := by
  have hx' : x.val < st.c.upv * (st.c.unit.inWords * WORD) := by
    have : x.val < st.c.upv * (unitRows st L.unit_built).nIn := hx
    rwa [L.unitRows_nIn] at this
  rw [Aliased.copyPos_block L regs F hg x hx hsat o]
  obtain ⟨s, hs, he⟩ := copyPos_zero L regs F hg hx' hz
  rw [he]
  exact Aliased.zeroPos_block L regs F hs hsat o

end copies

/-- **At an accepted untyped statement, a template instance's stacked input that reads the zero carries 0**, at the
template instance's columns (`Placed.Accepted.col`), in every block of a satisfying witness. -/
theorem accepted_zero_block {T : Model.Statement} (a : Placed.Accepted T) {g : ℕ} (hg : g < a.st.c.g)
    (x : Fin a.rows.w) (hx : x.val < a.rows.nIn) (hz : ZeroIn a.st.c x.val) {z : Model.Witness T.m}
    (hsat : T.Satisfies z) (o : Fin (2 ^ T.nbl)) : T.block z o (a.col hg x) = 0 := by
  obtain ⟨tags, cf, pf, coins, tables, draw, partition, program, st, ht, h, regs, hT⟩ := a
  subst hT
  exact stack_zero_block (setupH_layout ht h) regs (Aliased.setupH_unitFacts ht h) hg x hx hz hsat o

/-- An accepted untyped statement's circuit is the one its circuit file parses to. -/
theorem accepted_c_eq {T : Model.Statement} (a : Placed.Accepted T) {c : Flock.Circuit}
    (hp : HmRow.parse a.tags a.cf a.tables = .ok c) : a.st.c = c := by
  have := (setupH_spec a.ht a.h).1
  rw [hp] at this
  exact (Except.ok.inj this).symm

/-! ## Δ's source keys at the zero -/

/-- **Δ's source key at a port bit is the zero exactly when the bit reads the zero** (`ZeroPort`). -/
theorem srcAt_zero_iff (c : Flock.Circuit) (s₀ p t : ℕ) : Aliased.srcAt c (s₀ % c.upv) p t = .zero ↔ ZeroPort c s₀ p t := by
  unfold Aliased.srcAt ZeroPort
  by_cases hp : p < (c.unit.inGroups.getD 0 default).bits.size
  · have hp' : ¬ (c.unit.inGroups.getD 0 default).bits.size ≤ p := by omega
    simp only [hp, hp', ↓reduceIte, true_and, false_and, or_false]
    split <;> simp_all
  · have hp' : (c.unit.inGroups.getD 0 default).bits.size ≤ p := by omega
    rw [ite_eq_right hp]
    have hR : ∀ (P Q : Prop), ((p < (c.unit.inGroups.getD 0 default).bits.size ∧ P) ∨
        ((c.unit.inGroups.getD 0 default).bits.size ≤ p ∧ Q)) ↔ Q :=
      fun _ _ => ⟨fun h => h.elim (fun h => absurd h.1 hp) (fun h => h.2), fun h => Or.inr ⟨hp', h⟩⟩
    rw [hR]
    split
    next e l hf =>
      simp only [hf, List.head?_cons, Option.mem_def, Option.some.injEq, exists_eq_left', Bool.and_eq_true,
        decide_eq_true_eq]
      split <;> simp_all
    next hf =>
      refine iff_of_false ?_ (by rw [hf]; simp)
      split <;> simp

/-- **Δ's source key at a stacked input is the zero exactly when the input reads the zero** (`ZeroIn`). -/
theorem srcKey_zero_iff {c : Flock.Circuit} {x : ℕ} (hx : x < c.upv * (c.unit.inWords * WORD)) :
    Aliased.srcKey c x = .zero ↔ ZeroIn c x := by
  have hn : 0 < c.unit.inWords * WORD := Nat.pos_of_ne_zero fun h => by rw [h, Nat.mul_zero] at hx; omega
  have hk : x / (c.unit.inWords * WORD) < c.upv := (Nat.div_lt_iff_lt_mul hn).2 hx
  have hmod : x / (c.unit.inWords * WORD) % c.upv = x / (c.unit.inWords * WORD) := Nat.mod_eq_of_lt hk
  unfold Aliased.srcKey ZeroIn InPort ZeroAt
  by_cases hm : x % (c.unit.inWords * WORD) ∈ portBits c.unit.inGroups
  · rw [ite_eq_left hm, ← srcAt_zero_iff, hmod]
    exact ⟨fun h => ⟨hm, h⟩, fun h => h.2⟩
  · rw [ite_eq_right hm]
    exact iff_of_false (fun h => by cases h) (fun h => hm h.1)

/-! ## The program of `n` units -/

section uniform

variable {k : ℕ} {R : Rows} {ins : ℕ → Fin R.nIn → Fin k} {one : Fin k} {hone : ∀ u c, ins u c ≠ one}

/-- A program input of `uniform` is no unit's gate. -/
theorem unit_uniform : ∀ (n : ℕ) (g : Fin (Placed.gates k R n)), g.val < k →
    (Placed.uniform k R ins one hone n).unit g = none
  | 0, _, _ => rfl
  | n + 1, g, hg => by
    have h' : g.val < Placed.gates k R n := lt_of_lt_of_le hg (Placed.le_gates k R n)
    change (Prog.snoc (Placed.uniform k R ins one hone n) R _ _ _).unit g = none
    simp only [Prog.unit, h', ↓reduceDIte, unit_uniform n ⟨g.val, h'⟩ hg, Option.map_none]

/-- `uniform` wires unit `u`'s input `i` to the input gate `ins u i`. -/
theorem insAt_uniform : ∀ (n : ℕ) (u : Fin n) (i : ℕ) (hi : i < R.nIn),
    (Const.Prog.insAt (Placed.uniform k R ins one hone n) u i).val = (ins u.val ⟨i, hi⟩).val
  | 0, u, _, _ => u.elim0
  | n + 1, u, i, hi => by
    induction u using Fin.lastCases with
    | last =>
      change (Const.Prog.insAt (Prog.snoc (Placed.uniform k R ins one hone n) R _ _ _) (Fin.last n) i).val = _
      rw [Const.Prog.insAt_last _ _ _ _ _ i hi]
      simp
    | cast v =>
      change (Const.Prog.insAt (Prog.snoc (Placed.uniform k R ins one hone n) R _ _ _) v.castSucc i).val = _
      rw [Const.Prog.insAt_castSucc]
      simp [insAt_uniform n v i hi]

/-- `uniform` wires every unit's constant to the input gate `one`. -/
theorem oneAt_uniform : ∀ (n : ℕ) (u : Fin n),
    (Const.Prog.oneAt (Placed.uniform k R ins one hone n) u).val = one.val
  | 0, u => u.elim0
  | n + 1, u => by
    induction u using Fin.lastCases with
    | last =>
      change (Const.Prog.oneAt (Prog.snoc (Placed.uniform k R ins one hone n) R _ _ _) (Fin.last n)).val = _
      rw [Const.Prog.oneAt_last]
      simp
    | cast v =>
      change (Const.Prog.oneAt (Prog.snoc (Placed.uniform k R ins one hone n) R _ _ _) v.castSucc).val = _
      rw [Const.Prog.oneAt_castSucc]
      simp [oneAt_uniform n v]

/-- **Every unit of `uniform` has its constant on `one`** (`Const.Prog.OneGate`). -/
theorem oneGate_uniform (n : ℕ) :
    Const.Prog.OneGate (Placed.uniform k R ins one hone n) (Fin.castLE (Placed.le_gates k R n) one) :=
  fun u => Fin.ext (oneAt_uniform n u)

end uniform

/-! ## `hZero` at the circuit's program -/

section circuit

variable {n : ℕ} {Reg : Type} {mPts : ℕ} {plan : Finset (Fin n) → Reg → List (TabSpec mPts)}
  {tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length}
  {k : ℕ} {c : Flock.Circuit} {hU : Built c.unit} {ins : ℕ → Fin (Placed.circuitRows c hU).nIn → Fin k}
  {one : Fin k} {hone : ∀ u x, ins u x ≠ one}

/-- **`hZero` for the circuit's program at the accepted statements' sites.** The program of `n` units of the circuit's template instance
rows (`Placed.uniform`), each drawn unit placed at its site (`si`: its template instance and block in its table's
accepted untyped statement, whose circuit file parses to `c`) with the program's own instance (`Const.Prog.instC`):
every column on the zero gate carries 0 in every satisfying witness, when every input the program wires to the zero gate
reads the zero (`hzer`). -/
theorem zeroCols_uniform (si : Placed.Sites plan tab)
    (hrows : si.RowsOf (Placed.uniform k (Placed.circuitRows c hU) ins one hone n))
    (hparse : ∀ S R u (h : u ∈ S),
      HmRow.parse (si.acc S R u h).tags (si.acc S R u h).cf (si.acc S R u h).tables = .ok c)
    {outs : Finset (Fin (Placed.gates k (Placed.circuitRows c hU) n))}
    (pp : ProgPlaces (Placed.uniform k (Placed.circuitRows c hU) ins one hone n) outs plan tab)
    (hinst : ∀ u, pp.inst u = Const.Prog.instC _ outs u) (ho : ∀ S R u h, pp.o S R u h = si.o S R u h)
    (hcol : ∀ S R u h, pp.col S R u h = si.col hrows S R u h)
    {zer : Fin k} (hz1 : zer ≠ one) (hzer : ∀ (u : Fin n) x, ins u.val x = zer → ZeroIn c x.val) :
    ((Placed.uniform k (Placed.circuitRows c hU) ins one hone n).partition outs).ZeroCols plan tab pp.derived.place
      {Fin.castLE (Placed.le_gates k _ n) zer} := by
  refine Prog.zeroCols_singleton pp (one := Fin.castLE (Placed.le_gates k _ n) one) (unit_uniform n _ zer.isLt)
    (fun he => hz1 (Fin.castLE_injective _ he)) (fun u => ?_) ?_
  · rw [hinst u, Const.Prog.instC_wire_one]
    exact oneGate_uniform n u
  · intro S R u h x hx hw z hsat
    rw [hinst u, Const.Prog.instC_wire_input _ _ _ x hx] at hw
    have hx' : x.val < (Placed.circuitRows c hU).nIn :=
      lt_of_lt_of_eq hx (congrArg Rows.nIn (Placed.rows_uniform _ _ _ _ _ n u))
    have hv := congrArg Fin.val hw
    rw [insAt_uniform n u x.val hx'] at hv
    have hzi : ZeroIn c x.val := hzer u ⟨x.val, hx'⟩ (Fin.ext hv)
    rw [ho, hcol]
    have hxa : x.val < (si.acc S R u h).rows.nIn := lt_of_lt_of_eq hx (congrArg Rows.nIn (hrows S R u h))
    have hza : ZeroIn (si.acc S R u h).st.c x.val := by
      rw [accepted_c_eq (si.acc S R u h) (hparse S R u h)]
      exact hzi
    exact accepted_zero_block (si.acc S R u h) (si.hvu S R u h) (Fin.cast (congrArg Rows.w (hrows S R u h)) x) hxa
      hza hsat _

/-- **`hZero` for the circuit's program on source-indexed input gates** (`Placed.circuitProg`): an input with no source
is on the zero gate, so it suffices that every input with no source reads the zero (`hsrc`). -/
theorem zeroCols_circuitProg {B : ℕ} {src : Fin (Placed.circuitRows c hU).nIn → Option (Fin B)}
    (si : Placed.Sites plan tab) (hrows : si.RowsOf (Placed.circuitProg c hU n B src))
    (hparse : ∀ S R u (h : u ∈ S),
      HmRow.parse (si.acc S R u h).tags (si.acc S R u h).cf (si.acc S R u h).tables = .ok c)
    {outs : Finset (Fin (Placed.gates (Placed.inGates n B) (Placed.circuitRows c hU) n))}
    (pp : ProgPlaces (Placed.circuitProg c hU n B src) outs plan tab)
    (hinst : ∀ u, pp.inst u = Const.Prog.instC _ outs u) (ho : ∀ S R u h, pp.o S R u h = si.o S R u h)
    (hcol : ∀ S R u h, pp.col S R u h = si.col hrows S R u h) (hsrc : ∀ x, src x = none → ZeroIn c x.val) :
    ((Placed.circuitProg c hU n B src).partition outs).ZeroCols plan tab pp.derived.place
      {Fin.castLE (Placed.le_gates _ _ n) (Placed.zeroGate n B)} :=
  zeroCols_uniform si hrows hparse pp hinst ho hcol (fun h => by simp [Placed.zeroGate, Placed.oneGate, Fin.ext_iff] at h)
    fun u x hx => hsrc x ((Placed.insOf_eq_zero_iff u.isLt x).1 hx)

/-- **`hZero` for the circuit's program with its input gates indexed by Δ's source keys** (`Aliased.keyIdx`, the wiring
that gives `pp.aliased`): an input's key is the zero exactly when it reads the zero (`srcKey_zero_iff`), so nothing is
left to assume about the wiring. -/
theorem zeroCols_keyIdx (si : Placed.Sites plan tab)
    (hrows : si.RowsOf (Placed.circuitProg c hU n (Placed.circuitRows c hU).nIn
      (Aliased.keyIdx c (Placed.circuitRows c hU).nIn)))
    (hparse : ∀ S R u (h : u ∈ S),
      HmRow.parse (si.acc S R u h).tags (si.acc S R u h).cf (si.acc S R u h).tables = .ok c)
    {outs : Finset (Fin (Placed.gates (Placed.inGates n (Placed.circuitRows c hU).nIn) (Placed.circuitRows c hU) n))}
    (pp : ProgPlaces (Placed.circuitProg c hU n (Placed.circuitRows c hU).nIn
      (Aliased.keyIdx c (Placed.circuitRows c hU).nIn)) outs plan tab)
    (hinst : ∀ u, pp.inst u = Const.Prog.instC _ outs u) (ho : ∀ S R u h, pp.o S R u h = si.o S R u h)
    (hcol : ∀ S R u h, pp.col S R u h = si.col hrows S R u h) :
    ((Placed.circuitProg c hU n (Placed.circuitRows c hU).nIn
        (Aliased.keyIdx c (Placed.circuitRows c hU).nIn)).partition outs).ZeroCols plan tab pp.derived.place
      {Fin.castLE (Placed.le_gates _ _ n) (Placed.zeroGate n (Placed.circuitRows c hU).nIn)} :=
  zeroCols_circuitProg si hrows hparse pp hinst ho hcol fun x hx => by
    have h1 : x.val < c.upv * min (c.unit.inWords * WORD) c.unit.constPos := x.isLt
    exact (srcKey_zero_iff (lt_of_lt_of_le h1 (Nat.mul_le_mul_left _ (Nat.min_le_left _ _)))).1
      ((Aliased.keyIdx_eq_none x).1 hx)

/-- **`hZero` at `Placed.progPlaces`** with the program's own instance, for the program whose input gates are indexed by
Δ's source keys: `zeroCols_keyIdx` with its place hypotheses discharged. -/
theorem zeroCols_progPlaces (si : Placed.Sites plan tab)
    (hrows : si.RowsOf (Placed.circuitProg c hU n (Placed.circuitRows c hU).nIn
      (Aliased.keyIdx c (Placed.circuitRows c hU).nIn)))
    (hparse : ∀ S R u (h : u ∈ S),
      HmRow.parse (si.acc S R u h).tags (si.acc S R u h).cf (si.acc S R u h).tables = .ok c)
    (outs : Finset (Fin (Placed.gates (Placed.inGates n (Placed.circuitRows c hU).nIn) (Placed.circuitRows c hU) n)))
    {hal : _} :
    ((Placed.circuitProg c hU n (Placed.circuitRows c hU).nIn
        (Aliased.keyIdx c (Placed.circuitRows c hU).nIn)).partition outs).ZeroCols plan tab
      (Placed.progPlaces si hrows outs (Const.Prog.instC _ outs) hal).derived.place
      {Fin.castLE (Placed.le_gates _ _ n) (Placed.zeroGate n (Placed.circuitRows c hU).nIn)} :=
  zeroCols_keyIdx si hrows hparse _ (fun _ => rfl) (fun _ _ _ _ => rfl) (fun _ _ _ _ => rfl)

/-- **`hZero` at `Placed.Sites.progPlaces_keyProg`**, the places of the circuit's program on Δ's source keys that take
only the sites and the parse: every column on the zero gate carries 0 in every satisfying witness. -/
theorem zeroCols_keyProg (hU : Built c.unit) {si : Placed.Sites plan tab}
    (hparse : ∀ S R u (h : u ∈ S),
      HmRow.parse (si.acc S R u h).tags (si.acc S R u h).cf (si.acc S R u h).tables = .ok c)
    (outs : Finset (Fin (Placed.gates (Placed.inGates n (Placed.circuitRows c hU).nIn) (Placed.circuitRows c hU) n))) :
    ((Placed.keyProg c hU n).partition outs).ZeroCols plan tab
      (Placed.Sites.progPlaces_keyProg hU hparse outs).derived.place
      {Fin.castLE (Placed.le_gates _ _ n) (Placed.zeroGate n (Placed.circuitRows c hU).nIn)} :=
  zeroCols_progPlaces si (Placed.Sites.rowsOf_keyProg hU hparse) hparse outs

end circuit

end FlockSoundness.Discharge.Zero
