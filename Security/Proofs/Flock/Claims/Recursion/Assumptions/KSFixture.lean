import Proofs.Flock.Recursive.KSStatement
import Proofs.Flock.Claims.Recursion.LamDeployed

/-!
# `RecursiveKS`'s nontrivial fixture: what the deployed readers make of it

`RecursiveKS`'s nontrivial lemma (`Flock.Guarantees.recursiveKS_nontrivial`) is about one compiled statement whose inner
statement is false. Everything it needs of that statement is built here by hand, at the model level: the inner
statement's block-0 view `innerView₀` (a one-gate circuit `x ⊕ x` claimed to output `1` on input `0`, so it has no
witness), its setup's tables `innerTables₀` (one table at `m = 22`, `k_log = 6`, one region, two link points), and a
dense form. What the kernel can't build is the reading: the deployed reader reads a statement only through `jsonOf`,
whose `Lean.Json.Parser.anyCore` is a `partial def` the kernel can't unfold, and only where every V\* part's circuit is
V\*'s byte for byte (`RecDeploy.partsVStar`, 22 MB at the deployed shape). So the named assumption `DeployedParses x`
says only that the deployed readers map the statement's bytes `x` to those hand-built values, and `PartsInScope` states
the V\* parts' scope at the run's coin-server record, the premises `RecDeploy.lamLin2_deployed_ge_dep` takes at every
statement. Wherever no statement reads, `RecursiveKS` holds at the verifier that accepts everything
(`Flock.Guarantees.recursiveKSAtAll_of_readX_none`), so some such reading is needed.
-/

namespace RecDeploy

open Claims (Msg Out)

/-- The fixture's inner circuit: one input wire `0` and one gate, `w₁ = w₀ ⊕ w₀`, its output. It outputs `0` on every
input. -/
def innerCircuit₀ : Claims.Circuit := ⟨1, [.xor 0 0], [1]⟩

/-- **The fixture's inner statement, block 0**: the circuit `innerCircuit₀`, block 0's input the public value `0` and its
output the public value `1`, and the public values `[false, true]`: the statement claims that the circuit outputs `1` on
`0`. -/
def innerView₀ : Claims.Circuit × List Claims.Flock.Src × List Claims.Flock.Src × List Bool :=
  (innerCircuit₀, [.pub 0], [.pub 1], [false, true])

/-- **The fixture's inner setup**: its link points (`points`, `mPts`) and each table's `(m, k_log, regions)`: one table
at `m = 22`, `k_log = 6` and one region, two link points. -/
def innerTables₀ : ℕ × ℕ × List (ℕ × ℕ × ℕ) := (2, 2, [(22, 6, 1)])

/-- **The deployed readers map the bytes `x` to the hand-built fixture**: the compiled statement's parse
(`RecDeploy.parse`) has the dense form (no holo inner circuits, no RecSparse phase); the deployed bound record's inner
Flock parses its inner statement (`xIOf x`) to one whose circuit, block-0 input and output and public values are
`innerView₀`'s; and Flock's plain setup sets that inner statement up at `innerTables₀`'s link points and tables.

Assumption record. No claim: three evaluations of deterministic deployed code on public bytes, which say nothing about
the prover or the world; the values they evaluate to are built by hand, and everything the nontrivial lemma needs of the
statement is proved from them. It is witnessed outside Lean by a run that feeds the fixture's bytes to the compiled
verifier's reader and Flock's setup and records the three readings. The fixture is the deployed compiled statement over
the inner statement `innerView₀`; no such fixture or run exists yet, and this docstring is to name its art id and run id
when they do. -/
def DeployedParses (x : Msg) : Prop :=
  (parse x).map (fun s => (s.form.holo, s.form.sparse)) = some (false, none) ∧
  ((bdDeployed _root_.Flock.Readable.relation (FlockDeploy.nextR DeployedR.rdR)
      (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation DeployedR.rdR).cols DeployedR.wER FlockDeploy.joinR
      rowInDeployed DeployedR.finalR DeployedR.wR).deploy.parse (xIOf x)).map
    (fun s => (s.C, s.inp 0, s.out 0, s.pub)) = some innerView₀ ∧
  (FlockDeploy.setupOf false (xIOf x)).map
    (fun t => (t.2.1.points, t.2.1.mPts, t.2.2.toList.map fun st => (st.m, st.c.kLog, st.regions.size))) =
    some innerTables₀

/-- **The V\* parts are in scope at the run's coin-server record**: each part `j < nS x`, staged at the transcript
`tr`, is in the parts' scope (`PartIn`), there are at most `2^16` parts and at most `2^56` inner setup inputs.

Assumption record. No claim: `RecDeploy.lamLin2_deployed_ge_dep`'s scope premises (`hO`, `hn`, `hS`), which every
statement `λ₂`'s numbers are stated at takes; the parts are V\*'s at the deployed shape, fixed by the deployment, not by
the fixture's inner statement. Witnessed outside Lean by the fixture's run, which records each part's setup. -/
def PartsInScope (x : Msg) (tr : Msg → List (Msg × Out) → ByteArray) : Prop :=
  (∀ j < nS x, ∃ i spec' sts', PartIn (staged (Stage.ofMsg tr) x (pick (Stage.ofMsg tr) x j) j) i spec' sts') ∧
    nS x ≤ 2 ^ 16 ∧ (innerSetupIns (xIOf x)).length ≤ 2 ^ 56

/-- **`RecursiveKS`'s nontrivial fixture**: some bytes `x` the deployed readers map to the hand-built fixture
(`DeployedParses`), with a coin-server record at whose transcript its parts are in scope (`PartsInScope`). The bytes
and the record are the fixture run's: the bytes hold V\*'s 22 MB part circuits, and the parts' staging reads the
record's transcript. -/
def KSFixture : Prop :=
  ∃ (x : Msg) (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ) (hd : Msg → Flock.RecRecord.Record)
    (recRest : Msg → List (Msg × Out) → List UInt8),
    DeployedParses x ∧ PartsInScope x (recTr aux nOf hd recRest)

end RecDeploy
