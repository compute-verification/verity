import Proofs.Flock.Recursive.KSStatement
import Proofs.Flock.Claims.Recursion.LamDeployed
import Proofs.Flock.Claims.Scope.Tables

/-!
# `RecursiveKS`'s nontrivial fixture: what the deployed readers make of it

`RecursiveKS`'s nontrivial lemma (`Flock.Guarantees.recursiveKS_nontrivial`) is about one compiled statement that the
deployed readers read: dense, with at least one V\* session, its inner statement in the plain scope (the checks
`PlainIn` reads, `setupView`), and its V\* parts in scope at a coin-server record. Its inner statement need not be
false: at the verifier that accepts everything the record stages no V\* statement, so `E*` extracts nothing from the
silent prover, whatever the inner statement. What the kernel can't build is the reading: the deployed reader reads a
statement only through `jsonOf`, whose `Lean.Json.Parser.anyCore` is a `partial def` the kernel can't unfold, and only
where every V\* part's circuit is V\*'s byte for byte (`RecDeploy.partsVStar`). So the named assumption
`DeployedParses x` says only what the deployed readers make of the statement's bytes `x`, and `PartsInScope` states
the V\* parts' scope at the run's coin-server record, the premises `RecDeploy.lamLin2_deployed_ge_dep` takes at every
statement. Wherever no statement reads, `RecursiveKS` holds at the verifier that accepts everything
(`Flock.Guarantees.recursiveKSAtAll_of_readX_none`), so some such reading is needed.
-/

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
the prover or the world. It is witnessed outside Lean by a run that stages V\* over an honest inner statement with the
deployed recursive driver, feeds the compiled statement's bytes to the compiled verifier's reader and Flock's setup, and
records the three readings.

The run is r20261027-030140-0e78 (cpu-3), evidence art:476851730535363877c1ec51d6a6558a939de0842b245a6a4ec8ef2dcde6f71a
(its `summary.json`), over the bytes HonestFixture's run r20261026-223243-c2a7 staged (outer tree
art:d872affc203376ec503fbbc949b6f2ab3d191b866e6a8cbd7c13656a836734c8). `flock-verify` built at 53a30ae7bc
read: `rec-inner` wrote neither a `holo` nor a `sparse` key (dense); `rec-statements` listed 10 V\* statements; and the
plain inner statement's setup gave 1 table, 2 points, `mPts = 27`, `(m, k_log, regions) = (25, 22, 3)`, inside
`setupView`, and was accepted. The setup numbers come from a `dbgTrace` added to a scratch copy of `verifyCmd`'s setup
(the art's `probe.diff`). The run is weaker than the Prop in two ways: no Lean term `x` is formed, so the readings are
the compiled verifier's on the run's files rather than `parse`, `nS` and `xIOf` evaluated at an `x`; and the staging is
c2a7's, at main 4331ec922a. -/
def DeployedParses (x : Msg) : Prop :=
  (parse x).map (fun s => (s.form.holo, s.form.sparse)) = some (false, none) ∧
  0 < nS x ∧
  (FlockDeploy.setupOf false (xIOf x)).map setupView = some true

/-- **The V\* parts are in scope at the run's coin-server record**: each part `j < nS x`, staged at the transcript
`tr`, is in the parts' scope (`PartIn`), there are at most `2^16` parts and at most `2^56` inner setup inputs.

Assumption record. No claim: `RecDeploy.lamLin2_deployed_ge_dep`'s scope premises (`hO`, `hn`, `hS`), which every
statement `λ₂`'s numbers are stated at takes; the parts are V\*'s at the deployed shape, fixed by the deployment, not by
the fixture's inner statement. Witnessed outside Lean by the fixture's run, which records each part's setup.

The run is `DeployedParses`'s, r20261027-030140-0e78 (art:476851730535363877c1ec51d6a6558a939de0842b245a6a4ec8ef2dcde6f71a),
at c2a7's coin-server record (the inner session and its `public.json` transcript). Each of the 10 parts' `--zk` setup
gave 1 table, 2 points, `mPts` from 27 to 33 and `(m, k_log, regions)` within `m` 26–32, `k_log` 22–26 and at most 48
regions, inside `PartIn`, and each was accepted. 10 ≤ `2^16`, and the whole tree is 1.8 GB, under `2^56` inputs. It is
weaker than the Prop in that the parts are those c2a7's driver staged, not `pick`'s choice, which is `Classical.choose`. -/
def PartsInScope (x : Msg) (tr : Msg → List (Msg × Out) → ByteArray) : Prop :=
  (∀ j < nS x, ∃ i spec' sts', PartIn (staged (Stage.ofMsg tr) x (pick (Stage.ofMsg tr) x j) j) i spec' sts') ∧
    nS x ≤ 2 ^ 16 ∧ (innerSetupIns (xIOf x)).length ≤ 2 ^ 56

/-- **`RecursiveKS`'s nontrivial fixture**: some bytes `x` the deployed readers read as `DeployedParses` says, with a
coin-server record at whose transcript its parts are in scope (`PartsInScope`). The bytes and the record are the
fixture run's: the bytes hold V\*'s part circuits, and the parts' staging reads the record's transcript. -/
def KSFixture : Prop :=
  ∃ (x : Msg) (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8),
    DeployedParses x ∧ PartsInScope x (recTr aux nOf hd recRest)

end RecDeploy
