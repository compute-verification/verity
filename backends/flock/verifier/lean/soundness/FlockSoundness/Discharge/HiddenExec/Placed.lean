import FlockSoundness.Discharge.HiddenExec.Copies
import FlockSoundness.Discharge.HiddenExec.Setup
import FlockSoundness.Protocol.Discharge.HiddenExec.Placed
import FlockSoundness.Discharge.HiddenExec.Flat

/-!
# `ProgPlaces`, `hConst` and `hZero` from `Stmt.setupHidden`'s accept

`Placed/Setup.lean`, `Placed/Keyed.lean`, `Const/Keyed.lean` and `Zero/Uniform.lean`'s circuit section, for untyped
hidden-output statements:

* `AcceptedH T`: a run of `Stmt.setupHidden` that accepted, with a layout `L` of its statement (`Layout`, which
  `Hidden.hidden_layout` gives), whose unit slots' rows are its flat twin's (`UnitRowsFlat`, which `unitRowsFlat` gives),
  and whose block R1CS `L.model regs` is `T`. Its VU rows are the unit net's rows stacked `units_per_vu` times, placed at
  every VU (`AcceptedH.placement`); its aliasing comes from Δ's source keys (`AcceptedH.aliased_of_key`, through
  `aliased_of_copiesH`), and a stacked input that reads the zero carries 0 (`AcceptedH.zero_block`).
* `SitesH plan tab`: each drawn unit's acceptance, VU and block, as `Placed.Sites`.
* `SitesH.progPlaces_keyProg`: `ProgPlaces` for the circuit's program on Δ's source keys (`Placed.keyProg`), from the
  sites and the parse alone; `constCols_keyProgH` and `zeroCols_keyProgH` are its `hConst` and `hZero`.
-/

set_option Elab.async false

namespace FlockSoundness.Discharge.HiddenExec

open Flock FlockLevel3
open Model Audit
open FlockSoundness.Discharge.Zero (ZeroIn)

/-! ## An accepted hidden-output statement -/

namespace AcceptedH

variable {T : Model.Statement} (a : AcceptedH T)

/-- **`Layout.Aliased` in `ProgPlaces`' form**, as `Placed.Accepted.aliased_of_layout`. -/
theorem aliased_of_layout {C : Audit.Circuit} {n : ℕ} {P : Audit.Partition C n} {u : Fin n} {R : Rows}
    (hR : R = a.rows) (inst : P.IsRowsUnit u R) (o : Fin (2 ^ T.nbl)) {g : ℕ} (hg : g < a.st.c.g)
    (hal : a.L.Aliased a.regs (hR ▸ inst) (Fin.cast a.nbl_eq.symm o) hg) :
    ∀ c c', inst.wire c = inst.wire c' → c.val < R.nIn → c'.val < R.nIn → ∀ z : Model.Witness T.m, T.Satisfies z →
      T.block z o (a.col hg (Fin.cast (congrArg Rows.w hR) c)) =
        T.block z o (a.col hg (Fin.cast (congrArg Rows.w hR) c')) := by
  subst hR
  obtain ⟨tags, cf, pf, coins, tables, draw, partition, program, st, ht, h, L, hR, regs, hT⟩ := a
  subst hT
  exact hal

end AcceptedH

/-! ## Each drawn unit's place -/

section Sites

variable {N n : ℕ} {Reg : Type} {mPts : ℕ} (plan : Finset (Fin n) → Reg → List (TabSpec mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length)

variable {plan tab}

variable {p : Prog N n} (si : SitesH plan tab) (hrows : si.RowsOf p)

end Sites

/-! ## The circuit's program on Δ's source keys -/

section keyed

variable {n : ℕ} {Reg : Type} {mPts : ℕ} {plan : Finset (Fin n) → Reg → List (TabSpec mPts)}
  {tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length}
  {c : Flock.Circuit} (hU : Built c.unit) {si : SitesH plan tab}
  (hparse : ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S),
    HmRow.parse (si.acc S R u h).tags (si.acc S R u h).cf (si.acc S R u h).tables = .ok c)

variable (outs : Finset (Fin (Placed.gates (Placed.inGates n (Placed.circuitRows c hU).nIn) (Placed.circuitRows c hU) n)))

end keyed

/-! ## `hZero` at the circuit's program -/

section circuit

variable {n : ℕ} {Reg : Type} {mPts : ℕ} {plan : Finset (Fin n) → Reg → List (TabSpec mPts)}
  {tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length}
  {k : ℕ} {c : Flock.Circuit} {hU : Built c.unit} {ins : ℕ → Fin (Placed.circuitRows c hU).nIn → Fin k}
  {one : Fin k} {hone : ∀ u x, ins u x ≠ one}

end circuit

end FlockSoundness.Discharge.HiddenExec
