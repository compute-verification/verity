module

public import Proofs.Flock.Claims.Recursion.LamDeployed
public import Proofs.Flock.Claims.Recursion.RecRecord
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Topology.Sheaves.Init

/-!
# `RecursiveKS`'s nontrivial fixture: what the deployed readers make of it

`RecursiveKS`'s nontrivial lemma (`Flock.Guarantees.recursiveKS_nontrivial`) is about one compiled statement that the
deployed readers read: dense, with at least one V\* session, its inner statement in the plain scope (the checks
`PlainIn` reads, `setupView`), and its V\* parts in scope at a live-coin record. Its inner statement need not be
false: at the verifier that accepts everything the record stages no V\* statement, so `E*` extracts nothing from the
silent prover, whatever the inner statement. What the kernel can't build is the reading: the deployed reader reads a
statement only through `jsonOf`, whose `Lean.Json.Parser.anyCore` is a `partial def` the kernel can't unfold, and only
where every V\* part's circuit is V\*'s byte for byte (`RecDeploy.partsVStar`). So the named assumption
`DeployedParses x` says only what the deployed readers make of the statement's bytes `x`, and `PartsInScope` states
the V\* parts' scope at the run's live-coin record, the premises `RecDeploy.lamLin2_deployed_ge_dep` takes at every
statement. Wherever no statement reads, `RecursiveKS` holds at the verifier that accepts everything
(`Flock.Guarantees.recursiveKSAtAll_of_readX_none`), so some such reading is needed.
-/

@[expose] public section

namespace RecDeploy

open Claims (Msg Out)

/-- **The checks `FlockDeploy.PlainIn` reads of a plain setup**: one to 200 tables, at most two link points, and each
table at `22 ≤ m ≤ 35`, `k_log ≤ 27`, at most 1026 regions and at most 64 link points (`InRangeWide`). -/
def setupView (t : FlockDeploy.Ins × _root_.Flock.SessionSpec × Array _root_.Flock.Stmt) : Bool :=
  decide (0 < t.2.2.size) && decide (t.2.2.size ≤ 200) && decide (t.2.1.points ≤ 2) &&
    t.2.2.toList.all fun st => decide (22 ≤ st.m) && decide (st.m ≤ 35) && decide (st.c.kLog ≤ 27) &&
      decide (st.regions.size ≤ 1026) && decide (t.2.1.mPts ≤ 64)

/-- **What the deployed readers make of the bytes `x`**: the compiled statement's parse (`RecDeploy.parse`) has the
dense form (no holo inner circuits, no RecSparse phase); it has at least one V\* session (`nS`); and Flock's plain
setup of its inner statement (`xIOf x`) passes `PlainIn`'s checks.

Assumption record. No claim: three evaluations of deterministic deployed code on public bytes, which say nothing about
the prover or the world. It is witnessed outside Lean on a session that stages V\* over an honest inner statement with
the deployed recursive driver, by feeding the compiled statement's bytes to the compiled verifier's reader and Flock's
setup and reading the three off.

How the witness is reproduced. The session is one `benchmarks/recursion_e2e` generates; the kept one is the toy
deployment at K = 64 and N = 4 (`RecDeploy.HonestFixture`'s docstring has the generator's flags and the file that names
the kept session). On its inner statement's files the compiled verifier's `rec-inner` writes the inner shape (`m`,
`k_log`, regions) and neither a `holo` nor a `sparse` key (dense); its `rec-statements` at that shape lists V\*'s
statements, at least one; and its `statement` on the same files derives the plain inner statement's setup. The
whole-session tests of `verity/protocols/proofs/recursion/verifier/vstar_cli/tests/test_vstar_recursive.py` (opt-in,
under `VERITY_VSTAR_RECURSIVE_DATA`) run the first two on the session they read and hold a proved session to each
statement listed; no test compares the setup's readings with `setupView`'s ranges. It is weaker than the Prop in that
no Lean term `x` is formed, so the readings are the compiled verifier's on the session's files rather than `parse`,
`nS` and `xIOf` evaluated at an `x`. -/
def DeployedParses (x : Msg) : Prop :=
  (parse x).map (fun s => (s.form.holo, s.form.sparse)) = some (false, none) ∧
  0 < nS x ∧
  (FlockDeploy.setupOf false (xIOf x)).map setupView = some true

/-- **The V\* parts are in scope at the run's live-coin record**: each part `j < nS x`, staged at the transcript
`tr`, is in the parts' scope (`PartIn`), there are at most `2^16` parts and at most `2^56` inner setup inputs.

Assumption record. No claim: `RecDeploy.lamLin2_deployed_ge_dep`'s scope premises (`hO`, `hn`, `hS`), which every
statement `λ₂`'s numbers are stated at takes; the parts are V\*'s at the deployed shape, fixed by the deployment, not by
the fixture's inner statement. Witnessed outside Lean on the fixture's session, each of whose parts is set up.

How the witness is reproduced. The session is `DeployedParses`'s, at its live-coin record (the inner session and its
`public.json` transcript). The whole-session tests of
`verity/protocols/proofs/recursion/verifier/vstar_cli/tests/test_vstar_recursive.py` hold that the compiled verifier
accepts each part under `--zk` at that part's flags, so each part's setup passed; the parts are as many as
`rec-statements` lists; and `vstar-recursive statement` holds the statement the session's inputs name under `2^56`
bytes. No test compares a part's setup (`mPts`, `m`, `k_log`, regions) with `PartIn`'s ranges; the compiled
verifier's `statement` at a part's flags derives that part's. It is weaker than the Prop in that the parts are those
the generator's driver staged, not `pick`'s choice, which is `Classical.choose`. -/
def PartsInScope (x : Msg) (tr : Msg → List (Msg × Out) → ByteArray) : Prop :=
  (∀ j < nS x, ∃ i spec' sts', PartIn (staged (Stage.ofMsg tr) x (pick (Stage.ofMsg tr) x j) j) i spec' sts') ∧
    nS x ≤ 2 ^ 16 ∧ (innerSetupIns (xIOf x)).length ≤ 2 ^ 56

/-- **`RecursiveKS`'s nontrivial fixture**: some bytes `x` the deployed readers read as `DeployedParses` says, with a
live-coin record at whose transcript its parts are in scope (`PartsInScope`). The bytes and the record are the
fixture run's: the bytes hold V\*'s part circuits, and the parts' staging reads the record's transcript. -/
def KSFixture : Prop :=
  ∃ (x : Msg) (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8),
    DeployedParses x ∧ PartsInScope x (recTr aux nOf hd recRest)

end RecDeploy
