import Proofs.Flock.Soundness.Discharge.PlainJ.Defs
import Proofs.Flock.Soundness.Discharge.ZkHidden.DefsJ

/-!
# One plain sampled session of `J` hidden-output tables: the plan, the draws and the openings

`flock-verify verify --plain-tables --session-tables J` (no `--zk`) at hidden outputs: `Stmt.setupTables` sets table `j`
up with `Stmt.setupHidden` under `tagsAt` at its draw `Composed.drawJ`. These are `PlainJ/Defs.lean`'s plan pieces with
the one-table hidden-output tables (`HiddenExec.TableH`, `DrawSetupH`) in place of `Integrate.Table`, at the plain inputs
`PlainJ.inputsPJ I j`, as `ZkHidden.tableShHJ` takes them at the `--zk` inputs:

* `tablePHJ`, `planPHJ`, `tabPHJ`: table `j`'s hidden-output table with the inputs it is set up on (its `DrawSetupH`'s
  table at table `j`'s inputs, else a reference setup `x₀`'s refused table at table 0's), the plan of `nTab I` tables,
  and each drawn unit's table (`partOf`).
* `DrawOkPHJ`, `opnPHJ`: the coin server's draw at each table, and table `j`'s openings, with `setupHidden` for
  `setupH`.
-/

namespace FlockSoundness.Discharge.PlainHJ

open Game Model Accounting Refine Audit Audit.Partition Flock
open FlockSoundness.Discharge.Exec FlockSoundness.Discharge.Integrate FlockSoundness.Discharge.HiddenExec
open FlockSoundness.Discharge.Composed (nTab nTab_pos drawJ partOf partS mem_partS)
open FlockSoundness.Discharge.PlainJ (inputsPJ)

local notation "𝔽" => FlockLevel3.GF128

variable {n mPts : ℕ} {Reg : Type}

section Plan

variable {I : Inputs} (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} {d₀ : Lean.Json}
  (x₀ : DrawSetupH (inputsPJ I 0) mPts S₀ d₀)

open Classical in
/-- **Table `j` at draw `S`, under hidden outputs**, with the inputs it is set up on: the table flock-verify sets up at
table `j`'s inputs and draw when it sets one up, and otherwise the reference setup `x₀`'s refused table at table 0's
inputs. -/
noncomputable def tablePHJ (S : Finset (Fin n)) (j : Fin (nTab I)) :
    (I' : Inputs) × TableH I' mPts (partS I (dj S) S j) :=
  if h : Nonempty (DrawSetupH (inputsPJ I j) mPts (partS I (dj S) S j) (drawJ I (dj S) j)) then
    ⟨inputsPJ I j, (Classical.choice h).table⟩
  else ⟨inputsPJ I 0, x₀.refused _⟩

theorem tablePHJ_set {S : Finset (Fin n)} {j : Fin (nTab I)}
    (h : Nonempty (DrawSetupH (inputsPJ I j) mPts (partS I (dj S) S j) (drawJ I (dj S) j))) :
    tablePHJ dj x₀ S j = ⟨inputsPJ I j, (Classical.choice h).table⟩ := by
  unfold tablePHJ
  rw [dite_eq_left_of_eq_true (eq_true h)]

theorem tablePHJ_refused {S : Finset (Fin n)} {j : Fin (nTab I)}
    (h : ¬ Nonempty (DrawSetupH (inputsPJ I j) mPts (partS I (dj S) S j) (drawJ I (dj S) j))) :
    tablePHJ dj x₀ S j = ⟨inputsPJ I 0, x₀.refused _⟩ := by
  unfold tablePHJ
  rw [dite_eq_right_of_eq_false (eq_false h)]

/-- **Table `j` where flock-verify sets it up**: at a `DrawSetupH` of table `j`'s inputs and draw, `tablePHJ` is the
table of one with the same statement. -/
theorem tablePHJ_of {S : Finset (Fin n)} {j : Fin (nTab I)}
    (x : DrawSetupH (inputsPJ I j) mPts (partS I (dj S) S j) (drawJ I (dj S) j)) :
    ∃ y : DrawSetupH (inputsPJ I j) mPts (partS I (dj S) S j) (drawJ I (dj S) j), y.st = x.st ∧
      tablePHJ dj x₀ S j = ⟨inputsPJ I j, y.table⟩ :=
  ⟨Classical.choice ⟨x⟩, Except.ok.inj ((Classical.choice ⟨x⟩).h.symm.trans x.h), tablePHJ_set dj x₀ ⟨x⟩⟩

/-- **The hidden-output plan of `J` tables**: at draw `S`, table `j` is `tablePHJ … S j`'s table, for every
registration. -/
noncomputable def planPHJ : Finset (Fin n) → Reg → List (TabSpec mPts) :=
  fun S _ => List.ofFn fun j => (tablePHJ dj x₀ S j).2.T

theorem planPHJ_length (S : Finset (Fin n)) (R : Reg) : (planPHJ (Reg := Reg) dj x₀ S R).length = nTab I :=
  List.length_ofFn

/-- Each drawn unit is in the table it is drawn into (`partOf`). -/
noncomputable def tabPHJ : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (planPHJ (Reg := Reg) dj x₀ S R).length :=
  fun S _ u => Fin.cast List.length_ofFn.symm (partOf I (dj S) u.val)

theorem tabOfPHJ_eq (S : Finset (Fin n)) (R : Reg) (u : Fin n) :
    Partition.tabOf (planPHJ dj x₀) (tabPHJ dj x₀) S R u = (tablePHJ dj x₀ S (partOf I (dj S) u.val)).2.T := by
  simp only [Partition.tabOf, planPHJ, tabPHJ, List.get_ofFn]
  rfl

end Plan

/-! ## The coin server's draws, and table `j`'s openings -/

/-- **The coin server's draw at `S`, at `J` hidden-output tables** (`PlainJ.DrawOkPJ` with `setupHidden` for
`setupH`): each table's draw proves the drawn units it is given (`partS`), and setupHidden sets each table up at it
with `mPts` link points. -/
def DrawOkPHJ (I : Inputs) (dj : Finset (Fin n) → Lean.Json) (mPts : ℕ) (S : Finset (Fin n)) : Prop :=
  ∀ j : Fin (nTab I),
    (∀ D, Draw.UnitDraw.ofJson (drawJ I (dj S) j) = .ok D → ∀ u ∈ partS I (dj S) S j, u.val ∈ D.proved) ∧
    ∀ st, Stmt.setupHidden (tagsAt I.tags I.count j) I.circuitFile I.publicFile I.coins I.tables
      (some (drawJ I (dj S) j)) I.partition I.program = .ok st → st.mPts = mPts

/-- **Table `j`'s openings at hidden outputs** (`PlainJ.opnPJ` with `setupHidden`): `opnOf` at the statement
setupHidden sets up under table `j`'s tags at its draw, and its `fast100` schedule, on proof files `2j` and `2j + 1`. -/
noncomputable def opnPHJ (I : Inputs) (plan : Finset (Fin n) → Reg → List (TabSpec mPts))
    (dj : Finset (Fin n) → Lean.Json) (S : Finset (Fin n)) (R : Reg) (proofs : Array ByteArray)
    (j : Fin (plan S R).length) : Opens 𝔽 Dig × Salts (List UInt8) :=
  match Stmt.setupHidden (tagsAt I.tags I.count j.val) I.circuitFile I.publicFile I.coins I.tables
      (some (drawJ I (dj S) j.val)) I.partition I.program with
  | .ok st => match Flock.fast100 st.m with
    | some sch => opnOf (Setup.ofCircuit st) sch (DecodeZLJ.nXT ((plan S R).get j))
        #[proofs[2 * j.val]!, proofs[2 * j.val + 1]!]
    | none => default
  | .error _ => default

end FlockSoundness.Discharge.PlainHJ
