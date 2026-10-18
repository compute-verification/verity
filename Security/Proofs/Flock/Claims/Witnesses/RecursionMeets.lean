import Proofs.Flock.Claims.Witnesses.Recursion
import Proofs.Flock.Claims.DeployEfficient
import Proofs.Flock.Claims.Recursion.DeployComplete

/-!
# Antecedent witnesses: the ZK-recursion compiler's claims at the deployed record

Each of the recursion's claims (`Definitions.Claims.Recursion`) has its checkable antecedents met at the deployed record,
beside the toy record it fails at (`Witnesses.Recursion`). The tree has no deployed record of the compiled verifier
itself: its claims are parametric in it, and what they read of the deployment is the inner Flock (the deployed plain
verifier, `FlockDeploy.deployPlain`, as the inner session's tags are plain), the outer `--zk` Flock
(`FlockDeploy.deployZk`), each at its radius decoder, and the readable scope. At every relation the deployed records
read:

* `KS_meets`, `KSPriv_meets`, `Bits_meets`, `Efficient_meets`: the outer radius decoder meets the list-decoding spec at
  the deployed `--zk` record (`decOK`, `FlockDeploy.listWithin_zk`), the spec those claims quantify the outer decoder
  over (`Efficient` is stated at one decoder, this one);
* `KSFlock_meets`, `BitsFlock_meets`: so do the inner radius decoder at the deployed plain record and the outer one;
* `KSBound_meets`, `KSPrivBound_meets`: the outer radius decoder meets the spec at the deployed `--zk` record. The tree
  has no deployed bound record (`Claims.Recursion.Bound`: its element-level Flock has no code counterpart yet), so the
  inner decoder's antecedent waits for one;
* `Inner.KS_meets`, `Inner.KSAbort_meets`: the inner radius decoder meets the spec at the deployed plain record, the
  inner protocol `Inner.ofFlock` reads there;
* `ZK_meets`, `ZKPriv_meets`: the deployed outer record is masked;
* `ScopeMatches_meets`: the deployed inner Flock's readable scope is its scope check (`FlockDeploy.scopeMatches_plain`),
  which `Flock.RecScope.scopeMatches_deployed` reads;
* `Complete_meets`, `Nontrivial_meets`: the readable scope holds a statement of each kind the compiled verifier accepts,
  dense, holo and sparse (candidate 3's shape: `m = 25`, `k_log = 22`, three regions, RecSparse `(18, 13, 6, 0)` and R
  at `(23, 2)`), each with a V* session that draws every unit (`inScope_kinds`);
* `Complete_meets_deployed`: the compiled verifier's record (`RecDeploy.deployRec`) states an error below `1` on every
  statement it can accept (`RecDeploy.readyRec`) whose inner session is one the numbers are stated at
  (`FlockDeploy.PlainIn`), whose V* statements are of tables at `m ≤ 35`, at most `2^64` of them
  (`RecDeploy.δcRec_lt_one`).
-/

namespace Claims.Witness.Recursion

open Claims Claims.Recursion

/-! ## The readable scope's kinds -/

/-- A V* session's full draw over four units. -/
def fullDraw4 : Recursion.Draw := ⟨some 4, 4, [0, 1, 2, 3], []⟩

/-- A statement at candidate 3's inner table (`m = 25`, `k_log = 22`, three regions, eight claims), its circuits private
or not (`holo`), with no RecSparse phase and no R, and one V* session that draws every unit. -/
def denseStmt (holo : Bool) : Recursion.Stmt :=
  ⟨[], none, none, ⟨25, 22, 3, 8, holo, none, none, false⟩, [some fullDraw4]⟩

/-- Candidate 3's sparse statement: its inner table, the RecSparse phase `(18, 13, 6, 0)`, R at `(23, 2)`, the inner
statement's dimensions the form's, its unit range inside the block, and one V* session that draws every unit. -/
def sparseStmt (holo : Bool) : Recursion.Stmt :=
  ⟨[], some (25, 22, 3), none, ⟨25, 22, 3, 8, holo, some (18, 13, 6, 0), some (23, 2), true⟩, [some fullDraw4]⟩

theorem draws_full : ∀ d ∈ [some fullDraw4], ∃ e, d = some e ∧ e.Full := by
  intro d hd
  rw [List.mem_singleton] at hd
  exact ⟨fullDraw4, hd, rfl, rfl, rfl⟩

theorem denseStmt_inScope (holo : Bool) : InScope (denseStmt holo) :=
  ⟨rfl, show 22 ≤ 25 by decide, show 25 ≤ 35 by decide, show 6 ≤ 22 by decide, rfl, Or.inl ⟨rfl, rfl⟩, draws_full⟩

theorem sparseStmt_inScope (holo : Bool) : InScope (sparseStmt holo) := by
  have hp : pLog 13 = 5 := by decide
  refine ⟨rfl, show 22 ≤ 25 by decide, show 25 ≤ 35 by decide, show 6 ≤ 22 by decide, rfl, Or.inr ⟨18, 13, 6, rfl,
    show 1 ≤ 18 by decide, show 6 ≤ 13 by decide, show 13 ≤ 22 by decide, rfl, rfl, ?_, ?_⟩, draws_full⟩
  · simp [sparseStmt, hp]
  · rw [hp]
    decide

/-- **The readable scope holds each kind of inner shape the compiled verifier accepts**: dense, holo, and candidate 3's
sparse phase with public or private circuits, each with a V* session that draws every unit. -/
theorem inScope_kinds :
    InScope (denseStmt false) ∧ InScope (denseStmt true) ∧ InScope (sparseStmt false) ∧ InScope (sparseStmt true) :=
  ⟨denseStmt_inScope false, denseStmt_inScope true, sparseStmt_inScope false, sparseStmt_inScope true⟩

/-! ## The witnesses -/

/-- `KS`'s antecedent at the deployment: the outer radius decoder meets the list-decoding spec at the deployed `--zk`
record. -/
theorem KS_meets (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) :
    decOK (FlockDeploy.deployZk rel) Claims.GS.zkDecodeRD :=
  FlockDeploy.listWithin_zk rel

/-- `KSPriv`'s antecedent at the deployment: the outer radius decoder meets the spec at the deployed `--zk` record. -/
theorem KSPriv_meets (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) :
    decOK (FlockDeploy.deployZk rel) Claims.GS.zkDecodeRD :=
  FlockDeploy.listWithin_zk rel

/-- `Bits`' checkable antecedent at the deployment: the outer radius decoder meets the spec at the deployed `--zk`
record. -/
theorem Bits_meets (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) :
    decOK (FlockDeploy.deployZk rel) Claims.GS.zkDecodeRD :=
  FlockDeploy.listWithin_zk rel

/-- `Efficient`'s decoder at the deployment, the outer radius decoder, meets the spec at the deployed `--zk` record. -/
theorem Efficient_meets (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) :
    decOK (FlockDeploy.deployZk rel) Claims.GS.zkDecodeRD :=
  FlockDeploy.listWithin_zk rel

/-- `KSFlock`'s antecedents at the deployment: the inner radius decoder meets the spec at the deployed plain record, and
the outer one at the deployed `--zk` record. -/
theorem KSFlock_meets (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) :
    decOK (FlockDeploy.deployPlain rel) Claims.GS.plainDecodeRD ∧
      decOK (FlockDeploy.deployZk rel) Claims.GS.zkDecodeRD :=
  ⟨FlockDeploy.listWithin_plain rel, FlockDeploy.listWithin_zk rel⟩

/-- `BitsFlock`'s checkable antecedents at the deployment: the inner and outer radius decoders meet the spec at the
deployed plain and `--zk` records. -/
theorem BitsFlock_meets (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) :
    decOK (FlockDeploy.deployPlain rel) Claims.GS.plainDecodeRD ∧
      decOK (FlockDeploy.deployZk rel) Claims.GS.zkDecodeRD :=
  ⟨FlockDeploy.listWithin_plain rel, FlockDeploy.listWithin_zk rel⟩

/-- `KSBound`'s antecedent at the deployment: the outer radius decoder meets the spec at the deployed `--zk` record. -/
theorem KSBound_meets (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) :
    decOK (FlockDeploy.deployZk rel) Claims.GS.zkDecodeRD :=
  FlockDeploy.listWithin_zk rel

/-- `KSPrivBound`'s antecedent at the deployment: the outer radius decoder meets the spec at the deployed `--zk`
record. -/
theorem KSPrivBound_meets (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) :
    decOK (FlockDeploy.deployZk rel) Claims.GS.zkDecodeRD :=
  FlockDeploy.listWithin_zk rel

/-- `Inner.KS`'s antecedent at the deployment: the inner radius decoder meets the spec at the deployed plain record. -/
theorem Inner.KS_meets (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) :
    decOK (FlockDeploy.deployPlain rel) Claims.GS.plainDecodeRD :=
  FlockDeploy.listWithin_plain rel

/-- `Inner.KSAbort`'s antecedent at the deployment: the inner radius decoder meets the spec at the deployed plain
record. -/
theorem Inner.KSAbort_meets (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) :
    decOK (FlockDeploy.deployPlain rel) Claims.GS.plainDecodeRD :=
  FlockDeploy.listWithin_plain rel

/-- `ZK`'s checkable antecedent at the deployment: the deployed outer record is masked. -/
theorem ZK_meets (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) :
    (FlockDeploy.deployZk rel).zk = true :=
  rfl

/-- `ZKPriv`'s checkable antecedent at the deployment: the deployed outer record is masked. -/
theorem ZKPriv_meets (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) :
    (FlockDeploy.deployZk rel).zk = true :=
  rfl

/-- `ScopeMatches`' antecedent at the deployment: the deployed inner Flock's readable scope is its scope check, which
the compiled verifier's scope check reads (`Flock.RecScope.scopeMatches_deployed`). -/
theorem ScopeMatches_meets (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) :
    Claims.Flock.ScopeMatches (FlockDeploy.deployPlain rel) :=
  FlockDeploy.scopeMatches_plain rel

/-- `Complete`'s antecedent: the readable scope holds a statement of each kind the compiled verifier accepts. -/
theorem Complete_meets :
    InScope (denseStmt false) ∧ InScope (denseStmt true) ∧ InScope (sparseStmt false) ∧ InScope (sparseStmt true) :=
  inScope_kinds

/-- **`Complete`'s error is below `1` at the deployed record**: `deployRec`'s error (`RecDeploy.δcRec`, the inner
session's plain `δc` plus each V* session's masked one) on every statement the compiled verifier can accept, at an inner
session the numbers are stated at, V* statements of tables at `m ≤ 35`, and at most `2^64` parts: at most `44` coins per
inner table and `50` per V* session, over `2^128`. -/
theorem Complete_meets_deployed (IV : Verifier FlockDeploy.Coin) (ivScope : Msg → Bool) (b : RecDeploy.Brings)
    (H : Msg → Msg) {x : Msg} {s : FlockDeploy.Ins} {spec : _root_.Flock.SessionSpec}
    {sts : Array _root_.Flock.Stmt} (hin : FlockDeploy.PlainIn (RecDeploy.xIOf x) s spec sts)
    (hready : RecDeploy.readyRec b.st H x = true)
    (hm : ∀ j < RecDeploy.nS x, ∀ s' spec' sts',
      RecDeploy.setupO (RecDeploy.staged b.st x (RecDeploy.pick b.st x j) j) = some (s', spec', sts') →
        ∀ t ∈ sts'.toList, t.m ≤ 35) (hn : RecDeploy.nS x ≤ 2 ^ 64) :
    (RecDeploy.deployRec IV ivScope b).δc H x ≤ (44 * sts.size + 50 * RecDeploy.nS x) / 2 ^ 128 ∧
      (RecDeploy.deployRec IV ivScope b).δc H x < 1 :=
  ⟨RecDeploy.δcRec_le b.st H hin hready hm, RecDeploy.δcRec_lt_one b.st H hin hready hm hn⟩

/-- `Nontrivial`'s antecedent: the readable scope holds a statement of each kind the compiled verifier accepts. -/
theorem Nontrivial_meets :
    InScope (denseStmt false) ∧ InScope (denseStmt true) ∧ InScope (sparseStmt false) ∧ InScope (sparseStmt true) :=
  inScope_kinds

end Claims.Witness.Recursion
