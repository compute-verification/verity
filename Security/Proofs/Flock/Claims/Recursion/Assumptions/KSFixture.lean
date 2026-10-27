import Proofs.Flock.Recursive.KSStatement
import Proofs.Flock.Claims.Recursion.LamDeployed
import Proofs.Flock.Claims.Scope.Tables

/-!
# `RecursiveKS`'s nontrivial fixture: what the deployed readers make of it

`RecursiveKS`'s nontrivial lemma (`Flock.Guarantees.recursiveKS_nontrivial`) is about one compiled statement whose inner
statement is false. What makes it false is built here by hand, at the level the deployed relation reads
(`Flock.Readable.relation`: the inner statement's tables as one R1CS system, `Flock.Readable.sysOf`): a row of the first
table whose A side is zero, so its bit is `0` in every block, while one of the table's fixed bits sets that bit to `1` in
block `0` (`rowView`, `rowView₀`). That is the shape a gate `w ⊕ w` claimed to output `1` lowers to: its row's A side
`e_w + e_w` vanishes over 𝔽₂. The inner statement's scope is stated as the checks `PlainIn` reads (`plainView`). What
the kernel can't build is the reading: the deployed reader reads a statement only through `jsonOf`, whose
`Lean.Json.Parser.anyCore` is a `partial def` the kernel can't unfold, and only where every V\* part's circuit is V\*'s
byte for byte (`RecDeploy.partsVStar`). So the named assumption `DeployedParses x i₀` says only that the deployed readers
map the statement's bytes `x` to those hand-built values, and `PartsInScope` states the V\* parts' scope at the run's
coin-server record, the premises `RecDeploy.lamLin2_deployed_ge_dep` takes at every statement. Wherever no statement
reads, `RecursiveKS` holds at the verifier that accepts everything (`Flock.Guarantees.recursiveKSAtAll_of_readX_none`),
so some such reading is needed.
-/

namespace RecDeploy

open Claims (Msg Out)

/-- **The inner statement's first table at row `i₀`**: its `k_log`, whether it fits the model (`Tab.Ok`), whether
`i₀` is in its block, whether row `i₀`'s A side is zero, and block `0`'s value of the first fixed bit at `i₀`. -/
def rowView (i₀ : ℕ) (p : _root_.Flock.Readable.Parsed) : Option (Bool × Bool × Bool × Option Bool) :=
  p.sts[0]?.map fun st =>
    (decide (_root_.Flock.Readable.Tab.ofStmt st).Ok, decide (i₀ < 2 ^ st.c.kLog),
      (List.range (2 ^ st.c.kLog)).all (fun r => !(_root_.Flock.Readable.Tab.ofStmt st).a i₀ r),
      (((_root_.Flock.Readable.Tab.ofStmt st).fixed.find? fun f => f.1 == i₀).map fun f => f.2 0))

/-- **The fixture's false row**: the table fits, row `i₀` is in the block, its A side is zero, and block `0` fixes its
bit to `1`. -/
def rowView₀ : Bool × Bool × Bool × Option Bool := (true, true, true, some true)

/-- **The checks `FlockDeploy.PlainIn` reads of a plain setup**: one to 200 tables, at most two link points, and each
table at `22 ≤ m ≤ 35`, `k_log ≤ 27`, at most 1026 regions and at most 64 link points (`InRangeWide`). -/
def plainView (t : FlockDeploy.Ins × _root_.Flock.SessionSpec × Array _root_.Flock.Stmt) : Bool :=
  decide (0 < t.2.2.size) && decide (t.2.2.size ≤ 200) && decide (t.2.1.points ≤ 2) &&
    t.2.2.toList.all fun st => decide (22 ≤ st.m) && decide (st.m ≤ 35) && decide (st.c.kLog ≤ 27) &&
      decide (st.regions.size ≤ 1026) && decide (t.2.1.mPts ≤ 64)

/-- **The deployed readers map the bytes `x` to the hand-built fixture**: the compiled statement's parse
(`RecDeploy.parse`) has the dense form (no holo inner circuits, no RecSparse phase); the deployed plain verifier's
reading of its inner statement (`xIOf x`) has the false row `rowView₀` at `i₀`; and Flock's plain setup of that inner
statement passes `PlainIn`'s checks.

Assumption record. No claim: three evaluations of deterministic deployed code on public bytes, which say nothing about
the prover or the world; the values they evaluate to are built by hand, and everything the nontrivial lemma needs of the
statement is proved from them. It is witnessed outside Lean by a run that feeds the fixture's bytes to the compiled
verifier's reader and Flock's setup and records the three readings. The fixture is the deployed compiled statement over
an inner statement with a gate `w ⊕ w` whose output is claimed `1`; no such fixture or run exists yet, and this
docstring is to name its art id and run id when they do. -/
def DeployedParses (x : Msg) (i₀ : ℕ) : Prop :=
  (parse x).map (fun s => (s.form.holo, s.form.sparse)) = some (false, none) ∧
  (_root_.Flock.Readable.parseBytes (FlockDeploy.readOf false) (xIOf x)).bind (rowView i₀) = some rowView₀ ∧
  (FlockDeploy.setupOf false (xIOf x)).map plainView = some true

/-- **The V\* parts are in scope at the run's coin-server record**: each part `j < nS x`, staged at the transcript
`tr`, is in the parts' scope (`PartIn`), there are at most `2^16` parts and at most `2^56` inner setup inputs.

Assumption record. No claim: `RecDeploy.lamLin2_deployed_ge_dep`'s scope premises (`hO`, `hn`, `hS`), which every
statement `λ₂`'s numbers are stated at takes; the parts are V\*'s at the deployed shape, fixed by the deployment, not by
the fixture's inner statement. Witnessed outside Lean by the fixture's run, which records each part's setup. -/
def PartsInScope (x : Msg) (tr : Msg → List (Msg × Out) → ByteArray) : Prop :=
  (∀ j < nS x, ∃ i spec' sts', PartIn (staged (Stage.ofMsg tr) x (pick (Stage.ofMsg tr) x j) j) i spec' sts') ∧
    nS x ≤ 2 ^ 16 ∧ (innerSetupIns (xIOf x)).length ≤ 2 ^ 56

/-- **`RecursiveKS`'s nontrivial fixture**: some bytes `x` the deployed readers map to the hand-built fixture at some
row `i₀` (`DeployedParses`), with a coin-server record at whose transcript its parts are in scope (`PartsInScope`).
The bytes, the row and the record are the fixture run's: the bytes hold V\*'s part circuits, the row is where the
inner circuit's gate `w ⊕ w` sits, and the parts' staging reads the record's transcript. -/
def KSFixture : Prop :=
  ∃ (x : Msg) (i₀ : ℕ) (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8),
    DeployedParses x i₀ ∧ PartsInScope x (recTr aux nOf hd recRest)

end RecDeploy
