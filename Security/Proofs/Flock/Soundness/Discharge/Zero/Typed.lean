import Proofs.Flock.Soundness.Discharge.Zero.Cols
import Proofs.Flock.Soundness.Discharge.Zero.Flat
import Proofs.Flock.Soundness.Discharge.Zero.Uniform
import Proofs.Flock.Soundness.Discharge.Placed.Typed
import Proofs.Flock.Soundness.Types.ProgramE2E

/-!
# `hZero` on the typed path (e2e, typed flat classes)

On a typed flat-class statement, `Placed.UProg.progPlaces_of_classes` places each drawn unit of a program of units
(`UProg`) at a slot of its table's class, for the program's rows circuit `p.toRows hk`.
- `UProg.zeroCols_progPlaces_of_classes`: `hZero` there for a program input `zer` other than the constant, when each slot
  input that reads `zer` copies a forced-zero row of its class (`hzr`, as in `UProg.zeroCols_of_classes`).
- `typedAccepted_copyPos_mem_zeros`: at an accepted typed flat-class statement (`Placed.TypedAccepted`), an input whose
  slot offset reads the zero (`ZeroAt`) copies a forced-zero row (`flatCopyPos_mem_zeros`).
- `UProg.zeroCols_progPlaces_of_setupH`: so `hZero` at `Placed.UProg.progPlaces_of_setupH`, when each unit input that the
  program wires to `zer` is one whose slot offset reads the zero in its table's circuit (`hza`).
- `flatKey_zero_iff`: a flat class's unit input key (`Aliased.flatKey`) is the zero exactly when its slot offset reads the
  zero, so `UProg.zeroCols_progPlaces_of_setupH_key` states `hza` on the keys `Aliased.flatCopies_of_key` reads.
-/

open Flock FlockLevel3

namespace FlockSoundness.Discharge.Zero

open Model Audit

theorem castInst_wire {C : Audit.Circuit} {n : ℕ} {P : Audit.Partition C n} {u : Fin n} {R₁ R₂ : Rows} (e : R₁ = R₂)
    (i : P.IsRowsUnit u R₁) (c : Fin R₂.w) :
    (Placed.castInst e i).wire c = i.wire (Fin.cast (congrArg Rows.w e).symm c) := by
  subst e; rfl

theorem castInst_one {C : Audit.Circuit} {n : ℕ} {P : Audit.Partition C n} {u : Fin n} {R₁ R₂ : Rows} (e : R₁ = R₂)
    (i : P.IsRowsUnit u R₁) : (Placed.castInst e i).wire R₂.one = i.wire R₁.one := by
  subst e; rfl

/-- **At an accepted typed flat-class statement, an input that reads the zero copies a forced-zero row** of the
statement's class (`Placed.TypedAccepted.cls`). -/
theorem typedAccepted_copyPos_mem_zeros {T : Model.Statement} (a : Placed.TypedAccepted T) (g : Fin a.cls.slots)
    (i : ℕ) (hz : ZeroAt a.st.c g.val ((a.k.done.getD a.k.unit default).inCols.getD i 0)) :
    a.cls.copyPos g i ∈ a.cls.zeros := by
  obtain ⟨tags, cf, pf, coins, tables, draw, partition, program, st, ht, h, k, net, hr, regs, hT⟩ := a
  subst hT
  exact flatCopyPos_mem_zeros ht h hr regs g i hz

namespace UProg

variable {k one : ℕ} (hk : one < k) {outs : List ℕ} (p : FlockSoundness.UProg k one outs) {Reg : Type} {mPts : ℕ}
  (plan : Finset (Fin p.n) → Reg → List (TabSpec mPts))
  (tab : (S : Finset (Fin p.n)) → (R : Reg) → Fin p.n → Fin (plan S R).length)

/-- **`hZero` at the typed path's places, from the tables' classes**: for a program input `zer` other than the constant,
every column on `zer` carries 0 in every satisfying witness, when each slot input that reads `zer` copies a forced-zero
row of its class (`hzr`). -/
theorem zeroCols_progPlaces_of_classes
    (cls : ∀ S R u, u ∈ S → TableClass (Partition.tabOf plan tab S R u).S)
    (slot : ∀ S R u (h : u ∈ S), Fin (cls S R u h).slots)
    (o : ∀ S R u, u ∈ S → Fin (2 ^ (Partition.tabOf plan tab S R u).S.nbl))
    (hr : ∀ S R u (h : u ∈ S), (p.spec hk u).Reads (cls S R u h))
    (hcp : ∀ S R u (h : u ∈ S), (cls S R u h).Copies (slot S R u h) (p.srcOf u))
    {zer : ℕ} (hzk : zer < k) (hz1 : zer ≠ one)
    (hzr : ∀ S R u (h : u ∈ S) (i : ℕ), i < (p.spec hk u).t.inBits → p.srcOf u i = .input zer →
      (cls S R u h).copyPos (slot S R u h) i ∈ (cls S R u h).zeros) :
    ((p.toRows hk).partition p.outsC).ZeroCols plan tab
      (Placed.UProg.progPlaces_of_classes hk p plan tab cls slot o hr hcp).derived.place
      {⟨zer, Nat.lt_of_lt_of_le hzk p.k_le_rowsN⟩} := by
  refine Prog.zeroCols_singleton _ (one := p.oneOf hk) (FlockSoundness.UProg.toRows_unit_of_lt hk p _ hzk)
    (fun he => hz1 (congrArg Fin.val he)) (fun u => ?_) ?_
  · change (Placed.castInst _ (p.instOf hk u)).wire _ = _
    rw [castInst_one]
    exact Fin.ext (p.pairs_oneGate hk u)
  · intro S R u h c hc hw z hsat
    have e := Placed.UProg.rows_toRows hk p u
    have hi : (p.rowsOf hk u).nIn = (p.spec hk u).t.inBits := (p.spec hk u).nIn_eq
    have hc1 : c.val < (p.spec hk u).t.inBits := by
      rw [← hi, ← e]; exact hc
    change (Placed.castInst _ (p.instOf hk u)).wire c = _ at hw
    rw [castInst_wire] at hw
    have hv := congrArg Fin.val hw
    rw [p.instOf_wire hk u _ (by rw [Fin.val_cast]; exact hc1), Fin.val_cast] at hv
    have hs : p.srcOf u c.val = .input zer := p.srcGate_inj (p.srcOf_valid hk u hc1) ⟨hzk, hz1⟩ hv
    have hn : (p.rowsOf hk u).nIn = (cls S R u h).rows.nIn :=
      congrArg Rows.nIn (UnitSpec.R_eq_of_reads (hr S R u h))
    change (Partition.tabOf plan tab S R u).S.block z
        ((((cls S R u h).atSlot (slot S R u h) (o S R u h)).cast (UnitSpec.R_eq_of_reads (hr S R u h)).symm).cast
          e.symm).o
        (((((cls S R u h).atSlot (slot S R u h) (o S R u h)).cast (UnitSpec.R_eq_of_reads (hr S R u h)).symm).cast
          e.symm).col c) = 0
    rw [RowsAt.cast_o, RowsAt.cast_o, RowsAt.cast_col, RowsAt.cast_col]
    change (Partition.tabOf plan tab S R u).S.block z (o S R u h) ((cls S R u h).col (slot S R u h) _) = 0
    rw [(cls S R u h).copy (slot S R u h) _ (by rw [Fin.val_cast, Fin.val_cast, ← hn, hi]; exact hc1) z hsat
      (o S R u h), Fin.val_cast, Fin.val_cast]
    exact (cls S R u h).zero _ (hzr S R u h c.val hc1 hs) z hsat (o S R u h)

/-- **`hZero` at the typed path's places, from setupH's accepts**: each drawn unit at slot `slot` of block `o` of its
table's accepted class (`Placed.UProg.progPlaces_of_setupH`), for a program input `zer` other than the constant, when each
unit input that the program wires to `zer` is one whose slot offset reads the zero in its table's circuit (`hza`). -/
theorem zeroCols_progPlaces_of_setupH
    (acc : ∀ S R u, u ∈ S → Placed.TypedAccepted (Partition.tabOf plan tab S R u).S)
    (slot : ∀ S R u (h : u ∈ S), Fin (acc S R u h).cls.slots)
    (o : ∀ S R u, u ∈ S → Fin (2 ^ (Partition.tabOf plan tab S R u).S.nbl))
    (hr : ∀ S R u (h : u ∈ S), (p.spec hk u).Reads (acc S R u h).cls)
    (hcp : ∀ S R u (h : u ∈ S), (acc S R u h).cls.Copies (slot S R u h) (p.srcOf u))
    {zer : ℕ} (hzk : zer < k) (hz1 : zer ≠ one)
    (hza : ∀ S R u (h : u ∈ S) (i : ℕ), i < (p.spec hk u).t.inBits → p.srcOf u i = .input zer →
      ZeroAt (acc S R u h).st.c (slot S R u h).val
        (((acc S R u h).k.done.getD (acc S R u h).k.unit default).inCols.getD i 0)) :
    ((p.toRows hk).partition p.outsC).ZeroCols plan tab
      (Placed.UProg.progPlaces_of_setupH hk p plan tab acc slot o hr hcp).derived.place
      {⟨zer, Nat.lt_of_lt_of_le hzk p.k_le_rowsN⟩} :=
  zeroCols_progPlaces_of_classes hk p plan tab (fun S R u h => (acc S R u h).cls) slot o hr hcp hzk hz1
    fun S R u h i hi hs => typedAccepted_copyPos_mem_zeros (acc S R u h) (slot S R u h) i (hza S R u h i hi hs)

end UProg

/-- **A flat class's unit input key is the zero exactly when its slot offset reads the zero**: input `i` (port bit
`inCols[i]`) at slot `g`, whose copy index is `g % upv`. -/
theorem flatKey_zero_iff (c : Flock.Circuit) (inCols : Array ℕ) (g i : ℕ) :
    Aliased.flatKey c inCols (g % c.upv) i = .zero ↔ ZeroAt c g (inCols.getD i 0) :=
  srcAt_zero_iff c g _ _

namespace UProg

variable {k one : ℕ} (hk : one < k) {outs : List ℕ} (p : FlockSoundness.UProg k one outs) {Reg : Type} {mPts : ℕ}
  (plan : Finset (Fin p.n) → Reg → List (TabSpec mPts))
  (tab : (S : Finset (Fin p.n)) → (R : Reg) → Fin p.n → Fin (plan S R).length)

/-- **`hZero` at the typed path's places, on Δ's keys**: `zeroCols_progPlaces_of_setupH`, with each unit input that the
program wires to `zer` one whose key at its slot's copy index is the zero (`Aliased.flatKey`). -/
theorem zeroCols_progPlaces_of_setupH_key
    (acc : ∀ S R u, u ∈ S → Placed.TypedAccepted (Partition.tabOf plan tab S R u).S)
    (slot : ∀ S R u (h : u ∈ S), Fin (acc S R u h).cls.slots)
    (o : ∀ S R u, u ∈ S → Fin (2 ^ (Partition.tabOf plan tab S R u).S.nbl))
    (hr : ∀ S R u (h : u ∈ S), (p.spec hk u).Reads (acc S R u h).cls)
    (hcp : ∀ S R u (h : u ∈ S), (acc S R u h).cls.Copies (slot S R u h) (p.srcOf u))
    {zer : ℕ} (hzk : zer < k) (hz1 : zer ≠ one)
    (hkey : ∀ S R u (h : u ∈ S) (i : ℕ), i < (p.spec hk u).t.inBits → p.srcOf u i = .input zer →
      Aliased.flatKey (acc S R u h).st.c ((acc S R u h).k.done.getD (acc S R u h).k.unit default).inCols
        ((slot S R u h).val % (acc S R u h).st.c.upv) i = .zero) :
    ((p.toRows hk).partition p.outsC).ZeroCols plan tab
      (Placed.UProg.progPlaces_of_setupH hk p plan tab acc slot o hr hcp).derived.place
      {⟨zer, Nat.lt_of_lt_of_le hzk p.k_le_rowsN⟩} :=
  zeroCols_progPlaces_of_setupH hk p plan tab acc slot o hr hcp hzk hz1
    fun S R u h i hi hs => (flatKey_zero_iff _ _ _ _).1 (hkey S R u h i hi hs)

end UProg

end FlockSoundness.Discharge.Zero
