import Proofs.Flock.Claims.ZK.Session
import Proofs.Flock.Claims.Scope.Split
import Proofs.Flock.Claims.Scope.Tables

/-!
# The deployed masked session's table data

`CoinLayer.TableData` at the deployed masked session (`FlockDeploy.setupOf true`), with what the tree fixes made concrete
and the rest named:

* concrete: the tables (`jOf`, `stAt`: `setupOf true x`'s tables, none where it does not set up), each table as
  `Coined` tables at its draw (`TabPlan.C`: `Composed.coinedOf` at the table's `TabZK`), and the witness's table
  witnesses (`tableData`'s `wt`: each table's share, `Readable.tableWit`, as its model witness `Readable.zOf`);
* `TabPlan`: each table's `TabZK` and the Ligerito prover, a plan the tree has at the hidden-output form only
  (`ZkHidden.planZHJ`, `TabPlan.HiddenForm`), not at the single-input-group form;
* `SessionPlan`: the finite tape and its reading as the masked prover's (`tapeZk`) and as the tables' randomness
  (`tape`), each table's coins at the session's coins (`cz`, `sessionPZk`'s per-table derivation), the public-region
  dummy (`wt₀`), and the round encodings (`enc`, `fin`, `off`, `sessionPZk`'s ask messages and `Final.encode`).

`tables_hidden`: `RefSession.Tables` at `tableData` where each table is the hidden-output plan's
(`setupTable_planZHJ`). `OneTableLowers`: the relation at a one-table parse makes the witness satisfy the table's
lowered statement (`stmtOf`), the converse of `good_of_committed_plain`'s reading, which `RefSession.Complete` needs.
-/

namespace Claims.ZKMal.Masked

open Claims FlockSoundness.Merkle FlockSoundness.ZK
open FlockSoundness.Discharge
open CoinLayer

/-- **The masked session's tables at statement `x`**: `setupOf true x`'s, none where it does not set up. -/
def stsOf (x : Msg) : Array _root_.Flock.Stmt :=
  match FlockDeploy.setupOf true x with
  | some (_, _, sts) => sts
  | none => #[]

/-- The masked session's table count at `x`. -/
def jOf (x : Msg) : ℕ := (stsOf x).size

/-- Table `j` of the masked session at `x`. -/
def stAt (x : Msg) (j : Fin (jOf x)) : _root_.Flock.Stmt := (stsOf x)[j.val]'j.isLt

/-- **Each table's `--zk` plan**: the claim points' size, the Ligerito prover, and each table's `TabZK`. -/
structure TabPlan where
  mPts : Msg → ℕ
  P : Composed.LigProver
  Tz : (x : Msg) → Fin (jOf x) → ZkSession.TabZK FlockLevel3.GF128 (mPts x)

/-- **Table `j` at `x` as `Coined` tables**: `coinedOf` at its `TabZK` and its statement. -/
noncomputable def TabPlan.C (L : TabPlan) (x : Msg) (j : Fin (jOf x)) : Composed.CoinedSetup :=
  Composed.coinedOf L.P (L.Tz x j) (stAt x j)

/-- **Every table is the hidden-output plan's**: at each statement and table, `coinedOf` at some table of
`ZkHidden.planZHJ` at a draw `verify --zk` sets up, at that draw's statement (`setupTable_planZHJ`'s tables). -/
def TabPlan.HiddenForm (L : TabPlan) : Prop :=
  ∀ x (j : Fin (jOf x)), ∃ (n : ℕ) (Reg : Type) (I : Exec.Inputs) (dj : Finset (Fin n) → Lean.Json)
    (S₀ : Finset (Fin n)) (d₀ : Lean.Json)
    (y₀ : ZkHidden.DrawSetupZHJ (Composed.inputsJ I 0) (L.mPts x) S₀ d₀) (_ : I.tags.typed = false)
    (S : Finset (Fin n)) (R : Reg) (k : Fin (Composed.nTab I))
    (y : ZkHidden.DrawSetupZHJ (Composed.inputsJ I k) (L.mPts x) (Composed.partS I (dj S) S k)
      (Composed.drawJ I (dj S) k)),
    L.C x j = Composed.coinedOf L.P ((ZkHidden.planZHJ (Reg := Reg) dj y₀ S R).get
      (Fin.cast (ZkHidden.planZHJ_length dj y₀ S R).symm k)) y.x.st

/-- **The rest of the masked session's table data**: the finite tape, its reading as the masked prover's and as the
tables' randomness, each table's coins at the session's coins, the public-region dummy, and the round encodings. -/
structure SessionPlan (L : TabPlan) where
  Tape : Type
  [finTape : Fintype Tape]
  [neTape : Nonempty Tape]
  tapeZk : Tape → FlockComplete.TapeZk
  cz : (x : Msg) → (Fin (refK x) → FlockDeploy.Coin) → ∀ j, Composed.CoinsZK (L.C x j).Rest
  wt₀ : (x : Msg) → ∀ j, FlockSoundness.Model.Witness (L.C x j).S.m
  enc : (x : Msg) → (c : Fin (refK x) → FlockDeploy.Coin) → ℕ → (∀ j, ((L.C x j).T (cz x c j)).tab.Cut) → Msg
  fin : (x : Msg) → (c : Fin (refK x) → FlockDeploy.Coin) → (∀ j, ((L.C x j).T (cz x c j)).tab.View) → Msg
  off : (x : Msg) → (c : Fin (refK x) → FlockDeploy.Coin) → (∀ j, ((L.C x j).T (cz x c j)).tab.Cut) → Msg
  tape : (x : Msg) → Tape → (∀ j, ZkView.Base (L.C x j).z (L.C x j).Hi) ×
    (ZkView.Leaves (fun j => (L.C x j).z) → ZkView.Salt) ×
      (ZkView.Leaves (fun j => (L.C x j).z) → ZkView.Bits × ZkView.Salt)

attribute [instance] SessionPlan.finTape SessionPlan.neTape

/-- **The deployed masked session's table data** at the plans `L` and `Q`: the tables `setupOf true x`'s, each
`L.C`, and the witness's table witnesses each table's share as its model witness. -/
noncomputable def tableData (L : TabPlan) (Q : SessionPlan L) : TableData where
  Tape := Q.Tape
  tapeZk := Q.tapeZk
  J := jOf
  C := L.C
  cz := Q.cz
  wt x w j := _root_.Flock.Readable.zOf 1 (_root_.Flock.Readable.tableWit (jOf x) w j.val) 0 (L.C x j).S.m
  wt₀ := Q.wt₀
  enc := Q.enc
  fin := Q.fin
  off := Q.off
  tape := Q.tape

/-- **`Tables` at the hidden-output form**: where every table is the hidden-output plan's, each is a table the session
sets up (`setupTable_planZHJ`). -/
theorem tables_hidden (L : TabPlan) (Q : SessionPlan L) (nr : Msg → ℕ → ℕ) (hH : L.HiddenForm) :
    (session nr (tableData L Q)).Tables := by
  intro x j
  obtain ⟨_, Reg, I, dj, _, _, y₀, ht, S, R, k, y, h⟩ := hH x j
  show SetupTable (L.C x j)
  rw [h]
  exact setupTable_planZHJ L.P I dj y₀ ht S R k y

/-- **The relation at a one-table parse makes the witness satisfy the table's lowered statement**: at a parse whose
one table is `st`, a witness in the readable relation is, as its model witness (`Readable.zOf`), a witness of
`stmtOf st` at its own regions. The converse of `good_of_committed_plain`'s reading, which `RefSession.Complete` needs
at the masked session's tables (each table's share in its one-table parse, `Case.tableT`). -/
def OneTableLowers : Prop :=
  ∀ (p : _root_.Flock.Readable.Parsed) (st : _root_.Flock.Stmt) (w : Claims.Flock.Wit)
    (hwf : FlockSoundness.Refine.StmtWF st), p.sts = #[st] →
    Claims.Flock.RelS (_root_.Flock.Readable.toReadable _root_.Flock.Readable.relation p) w →
      (FlockSoundness.Refine.stmtOf st hwf (FlockSoundness.Refine.regionsOf st)).Satisfies
        (_root_.Flock.Readable.zOf 1 w 0 st.m)

end Claims.ZKMal.Masked
