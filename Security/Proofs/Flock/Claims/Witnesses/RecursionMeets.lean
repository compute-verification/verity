import Proofs.Flock.Claims.Witnesses.Recursion
import Proofs.Flock.Claims.DeployEfficient
import Proofs.Flock.Claims.Recursion.DeployComplete
import Proofs.Flock.Claims.Recursion.DeployStop
import Proofs.Flock.Claims.Recursion.DeployChecked
import Proofs.Flock.Claims.Recursion.BoundWeightDeploy

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
* `KSBound_meets`, `KSPrivBound_meets`: at the deployed bound record (`RecDeploy.bdCover`, `RecDeploy.bdDeployed` at
  `γ`'s weight `RecDeploy.wCover` at `IVStar`, `coverW` of the trimmed plain record's weight; its element-level
  Flock is the trimmed plain record `FlockDeploy.deployPlainTrim` with only its verifier's steps, columns and weight
  replaced), the inner radius decoder meets the spec, and so does the outer one at the deployed `--zk` record;
* `Inner.KS_meets`, `Inner.KSAbort_meets`: the inner radius decoder meets the spec at the deployed plain record, the
  inner protocol `Inner.ofFlock` reads there;
* `KSFlock_meets_trim`, `Inner.KS_meets_trim`, `Inner.KSAbort_meets_trim`: the same at the trimmed records
  (`FlockDeploy.deployPlainTrim`, `FlockDeploy.deployZkTrim`, `FlockDeploy.listWithin_plainTrim`,
  `FlockDeploy.listWithin_zkTrim`), where the recursion's KS theorems sit;
* `ZK_meets`, `ZKPriv_meets`: the deployed outer record is masked;
* `ZKStop_meets`: the deployed outer record is masked, and the compiled verifier's record with its stop
  (`RecDeploy.deployRecStop`) is one whose prover `ZKStop` reads is `refPStop` on the tape each statement draws
  (`RecDeploy.prover_deployRecStop`), which sends the empty message wherever its stop has fired
  (`RecDeploy.refPStop_halted`), and whose stop fires at the first round on every statement out of scope or whose
  inner statement the inner verifier can't accept (`RecDeploy.halted_of_statement`), and nowhere before the staging
  round at an inner-ready statement in scope where every answer is a coin's bytes and none a bad inner coin
  (`RecDeploy.halted_live`);
* `ZKStop_meets_ready`: where that leaves the stop live. The inner statement `RecDeploy.PlainLeavesAdmitted` names
  is in Flock's scope and true, and at every statement in scope carrying it the reference prover runs every
  round before the staging round where every answer is a coin's bytes and none a bad inner coin
  (`RecDeploy.ready_of_admitted`); so it does wherever the inner statement's size, its regions and its honest record
  fit, which in scope are the inner readiness the stop reads (`RecDeploy.ready_inScope_iff`);
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
def fullDraw4 : Recursion.DrawRecord := ⟨some 4, 4, [0, 1, 2, 3], []⟩

/-- A statement at candidate 3's inner table (`m = 25`, `k_log = 22`, three regions, eight claims), its circuits private
or not (`holo`), with no RecSparse phase and no R, and one V* session that draws every unit. -/
def denseStmt (holo : Bool) : Recursion.CompiledStatement :=
  ⟨[], none, none, ⟨25, 22, 3, 8, holo, none, none, false⟩, [some fullDraw4]⟩

/-- Candidate 3's sparse statement: its inner table, the RecSparse phase `(18, 13, 6, 0)`, R at `(23, 2)`, the inner
statement's dimensions the form's, its unit range inside the block, and one V* session that draws every unit. -/
def sparseStmt (holo : Bool) : Recursion.CompiledStatement :=
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

/-- `KSFlock`'s antecedents at the trimmed records: the inner radius decoder meets the spec at the trimmed plain record,
and the outer one at the trimmed `--zk` record. -/
theorem KSFlock_meets_trim (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) :
    decOK (FlockDeploy.deployPlainTrim rel) Claims.GS.plainDecodeRD ∧
      decOK (FlockDeploy.deployZkTrim rel) Claims.GS.zkDecodeRD :=
  ⟨FlockDeploy.listWithin_plainTrim rel, FlockDeploy.listWithin_zkTrim rel⟩

/-- `BitsFlock`'s checkable antecedents at the deployment: the inner and outer radius decoders meet the spec at the
deployed plain and `--zk` records. -/
theorem BitsFlock_meets (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) :
    decOK (FlockDeploy.deployPlain rel) Claims.GS.plainDecodeRD ∧
      decOK (FlockDeploy.deployZk rel) Claims.GS.zkDecodeRD :=
  ⟨FlockDeploy.listWithin_plain rel, FlockDeploy.listWithin_zk rel⟩

/-- `KSBound`'s antecedents at the deployment: the inner radius decoder meets the spec at the deployed bound record's
element-level Flock (`RecDeploy.deE`, which keeps `deployPlainTrim`'s code), and the outer one at the deployed `--zk`
record. -/
theorem KSBound_meets (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (nextE : Msg → List Msg → List FlockDeploy.Coin → Claims.OC Claims.Out)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) (join : List Msg → Msg)
    (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × RecDeploy.Salt)) × Msg)) :
    decOK (RecDeploy.bdCover rel nextE colsE wE join rowIn final).DE Claims.GS.plainDecodeRD ∧
      decOK (FlockDeploy.deployZk rel) Claims.GS.zkDecodeRD :=
  ⟨FlockDeploy.listWithin_plainTrim rel, FlockDeploy.listWithin_zk rel⟩

/-- `KSPrivBound`'s antecedents at the deployment, as `KSBound_meets`. -/
theorem KSPrivBound_meets (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (nextE : Msg → List Msg → List FlockDeploy.Coin → Claims.OC Claims.Out)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) (join : List Msg → Msg)
    (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × RecDeploy.Salt)) × Msg)) :
    decOK (RecDeploy.bdCover rel nextE colsE wE join rowIn final).DE Claims.GS.plainDecodeRD ∧
      decOK (FlockDeploy.deployZk rel) Claims.GS.zkDecodeRD :=
  KSBound_meets rel nextE colsE wE join rowIn final

/-- `Inner.KS`'s antecedent at the deployment: the inner radius decoder meets the spec at the deployed plain record. -/
theorem Inner.KS_meets (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) :
    decOK (FlockDeploy.deployPlain rel) Claims.GS.plainDecodeRD :=
  FlockDeploy.listWithin_plain rel

/-- `Inner.KS`'s antecedent at the trimmed record: the inner radius decoder meets the spec at the trimmed plain
record. -/
theorem Inner.KS_meets_trim (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) :
    decOK (FlockDeploy.deployPlainTrim rel) Claims.GS.plainDecodeRD :=
  FlockDeploy.listWithin_plainTrim rel

/-- `Inner.KSAbort`'s antecedent at the deployment: the inner radius decoder meets the spec at the deployed plain
record. -/
theorem Inner.KSAbort_meets (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) :
    decOK (FlockDeploy.deployPlain rel) Claims.GS.plainDecodeRD :=
  FlockDeploy.listWithin_plain rel

/-- `Inner.KSAbort`'s antecedent at the trimmed record: the inner radius decoder meets the spec at the trimmed plain
record. -/
theorem Inner.KSAbort_meets_trim (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) :
    decOK (FlockDeploy.deployPlainTrim rel) Claims.GS.plainDecodeRD :=
  FlockDeploy.listWithin_plainTrim rel

/-- `ZK`'s checkable antecedent at the deployment: the deployed outer record is masked. -/
theorem ZK_meets (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) :
    (FlockDeploy.deployZk rel).zk = true :=
  rfl

/-- `ZKPriv`'s checkable antecedent at the deployment: the deployed outer record is masked. -/
theorem ZKPriv_meets (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) :
    (FlockDeploy.deployZk rel).zk = true :=
  rfl

/-- `ZKStop`'s checkable antecedents at the deployment: the deployed outer record is masked; the prover `ZKStop` reads
at the compiled verifier's record with its stop is `refPStop` on the tape `tapeOf x` draws, and `refPStop` sends the
empty message wherever the stop has fired; the stop fires at the first round on every statement out of scope or whose
inner statement the inner verifier can't accept, and, at an inner-ready statement in scope, nowhere before the staging
round where every answer is a coin's bytes and no prefix of the transcript holds a bad inner coin. -/
theorem ZKStop_meets {N Ω : Msg → Type} [∀ x, Fintype (N x)] [∀ x, Nonempty (N x)] [∀ x, Fintype (Ω x)]
    [∀ x, Nonempty (Ω x)] (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (IV : Claims.Verifier FlockDeploy.Coin) (ivScope : Msg → Bool) (b : RecDeploy.Brings)
    (tapeOf : (x : Msg) → N x × Ω x → RecDeploy.Tape b) (SO : ℕ → Msg → ℕ → Recursion.OSim) (nAt : Msg → ℕ → ℕ)
    (innerMsg : Msg → List (Msg × Out) → List Msg → Claims.OC Msg) :
    (FlockDeploy.deployZk rel).zk = true ∧
      (∀ (w : Claims.Flock.Wit) (x : Msg),
        (RecDeploy.deployRecStop IV ivScope b tapeOf SO nAt innerMsg).prover w x =
          ⟨fun x' hist ω => (RecDeploy.refPStop IV ivScope b w).next x' hist (tapeOf x ω)⟩) ∧
      (∀ (H : Msg → Msg) (w : Claims.Flock.Wit) (x : Msg) (hist : List (Msg × Msg)) (ω : RecDeploy.Tape b),
        (RecDeploy.deployRecStop IV ivScope b tapeOf SO nAt innerMsg).halted H (RecDeploy.stopRounds IV x) x
            (RecDeploy.asRounds hist) = true → (RecDeploy.refPStop IV ivScope b w).next x hist ω = .ret []) ∧
      (∀ (H : Msg → Msg) (n : ℕ) (x : Msg) (ms : List (Msg × Out)),
        (RecDeploy.scopeCheck ivScope x = false ∨
            FlockComplete.ready FlockDeploy.sha512 (RecDeploy.xIOf x) = false) →
          (RecDeploy.deployRecStop IV ivScope b tapeOf SO nAt innerMsg).halted H n x ms = true) ∧
      ∀ (H : Msg → Msg) (x : Msg) (ms : List (Msg × Out)), RecDeploy.scopeCheck ivScope x = true →
        FlockComplete.ready FlockDeploy.sha512 (RecDeploy.xIOf x) = true → ms.length < RecDeploy.stopRounds IV x →
          RecDeploy.Coined ms → (∀ i ≤ ms.length, ¬ RecDeploy.badAt x (ms.take i)) →
            (RecDeploy.deployRecStop IV ivScope b tapeOf SO nAt innerMsg).halted H (RecDeploy.stopRounds IV x) x ms =
              false :=
  ⟨rfl, RecDeploy.prover_deployRecStop IV ivScope b tapeOf SO nAt innerMsg,
    RecDeploy.refPStop_halted IV ivScope b tapeOf SO nAt innerMsg,
    RecDeploy.halted_of_statement IV ivScope b tapeOf SO nAt innerMsg,
    RecDeploy.halted_live IV ivScope b tapeOf SO nAt innerMsg⟩

/-- **Where `ZKStop`'s stop leaves the prover live** at the compiled verifier's record with its stop:

* the inner statement that `RecDeploy.PlainLeavesAdmitted` names is in Flock's scope and true, and at every statement
  in scope whose inner statement it is, the reference prover hasn't stopped before the staging round where every
  answer is a coin's bytes and no prefix of the transcript holds a bad inner coin (the inner readiness the stop reads
  holds there, `RecDeploy.ready_of_admitted`; where Flock's honest prover is complete, the inner verifier accepts
  there, `RecDeploy.admitted_acc`);
* at a statement in scope whose inner statement is under `2^64` bytes and whose tables' regions and honest record fit
  (`FlockComplete.RecordFitsAt`), the same: in scope, those three facts are exactly the inner readiness the stop reads
  (`RecDeploy.ready_inScope_iff`).

No statement in scope is exhibited: the scope check and Flock's setup read the statement's JSON with `Lean.Json.parse`,
a `partial` function the kernel can't unfold (as at Flock, `Claims.Flock.scope_meets_plain`), and the honest record's
size (`RecordFitsAt`) is a bound with no proof here. -/
theorem ZKStop_meets_ready {N Ω : Msg → Type} [∀ x, Fintype (N x)] [∀ x, Nonempty (N x)] [∀ x, Fintype (Ω x)]
    [∀ x, Nonempty (Ω x)] (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (IV : Claims.Verifier FlockDeploy.Coin) (ivScope : Msg → Bool) (b : RecDeploy.Brings)
    (tapeOf : (x : Msg) → N x × Ω x → RecDeploy.Tape b) (SO : ℕ → Msg → ℕ → Recursion.OSim) (nAt : Msg → ℕ → ℕ)
    (innerMsg : Msg → List (Msg × Out) → List Msg → Claims.OC Msg) :
    (RecDeploy.PlainLeavesAdmitted rel →
      ∃ inner : Msg, Claims.Flock.Scope (FlockDeploy.deployPlain rel) inner ∧
        (∃ w, Claims.Flock.R (FlockDeploy.deployPlain rel).parse inner w) ∧
        ∀ (H : Msg → Msg) (x : Msg) (ms : List (Msg × Out)), RecDeploy.scopeCheck ivScope x = true →
          RecDeploy.xIOf x = inner → ms.length < RecDeploy.stopRounds IV x → RecDeploy.Coined ms →
            (∀ i ≤ ms.length, ¬ RecDeploy.badAt x (ms.take i)) →
            (RecDeploy.deployRecStop IV ivScope b tapeOf SO nAt innerMsg).halted H (RecDeploy.stopRounds IV x) x ms =
              false) ∧
      ∀ (H : Msg → Msg) (x : Msg) (ms : List (Msg × Out)), RecDeploy.scopeCheck ivScope x = true →
        (RecDeploy.xIOf x).length < 2 ^ 64 →
        (∀ s spec sts, FlockDeploy.setupOf false (RecDeploy.xIOf x) = some (s, spec, sts) →
          (∀ st ∈ sts, FlockComplete.RegionsFit st) ∧
            FlockComplete.RecordFitsAt FlockDeploy.sha512 (RecDeploy.xIOf x) spec sts) →
        ms.length < RecDeploy.stopRounds IV x → RecDeploy.Coined ms →
          (∀ i ≤ ms.length, ¬ RecDeploy.badAt x (ms.take i)) →
          (RecDeploy.deployRecStop IV ivScope b tapeOf SO nAt innerMsg).halted H (RecDeploy.stopRounds IV x) x ms =
            false := by
  refine ⟨fun hA => ?_, fun H x ms hs hl hfit hlen hc hb => ?_⟩
  · obtain ⟨inner, hr, hS, hR⟩ := RecDeploy.ready_of_admitted hA
    refine ⟨inner, hS, hR, fun H x ms hs hx hlen hc hb => ?_⟩
    exact RecDeploy.halted_live IV ivScope b tapeOf SO nAt innerMsg H x ms hs (by rw [hx]; exact hr) hlen hc hb
  obtain ⟨s, spec, sts, hset, -, hiff⟩ := RecDeploy.ready_inScope_iff hs
  exact RecDeploy.halted_live IV ivScope b tapeOf SO nAt innerMsg H x ms hs (hiff.2 ⟨hl, hfit s spec sts hset⟩) hlen hc
    hb

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
