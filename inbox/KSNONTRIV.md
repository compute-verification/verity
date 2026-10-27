# KSNONTRIV (bc-c7d6462d-41a8-5405-83b0-88ef8803b116) inbox

- 21:27Z coordinator: launched for LEAN-66's nontrivial lemma for RecursiveKS (due Tue 08:00Z). Asked lean on LEAN-66 for the witnesses entry shape.
- 21:36Z coordinator: top's reading (21:35Z, lean rules on LEAN-66): a named premise discharged by a deployed run counts for the honest lemma ONLY if the lemma names that run (its run id or art: id) so the check can be rerun. So HonestRunAccepted, IrShaped and HonestWitSatisfies must each cite the PASSING witness run id / art: id in their docstrings (and the lock witnesses entry). Today no passing witness exists yet: #3744's fixture is being restaged (PRTRIAGE, new art via INF-270) and the deployed trace difftest is being profiled (CAPLOWER).
- 22:10Z KSNONTRIV (statement, task 1). Branch `cursor/proofs-ksnontriv-27b2` (from kslock 8811d713c4).
  **Accept-all means**: the compiled recursive verifier `(D.record IV).V`, whose acceptance `PrBad` reads, replaced by the
  one that keeps its rounds `k` and answers `.acc` at its first round with no SHA-512 call. Everything else (the parser,
  `xI`, the inner view, `IVStar`, `Bd`, `DO`, `Cp`, `DenseAt`) stays deployed:
  `def RecAt.acceptAll (D : RecAt Coin) : RecAt Coin := { D with record := fun IV => { D.record IV with V := ⟨(D.record IV).V.k, fun _ _ _ => .ret .acc⟩ } }`
  (`runs` is unchanged, by `rfl`). The right-hand side is re-read at accept-all too (`PrB = 0`: no verifier calls).
  **Full statement** (`¬` of RecursiveKS's conclusion at accept-all):
  `theorem recursiveKS_nontrivial : ¬ ∀ b aux nOf hd recRest, KSLin2On sha512 (recBoundAt relation {deployedB b with tr := recTr aux nOf hd recRest} (layR elemR)).acceptAll (bdDeployed relation (nextR rdR) (deployPlainTrimR relation rdR).cols wER joinR rowInDeployed finalR wR) (deployZkRec relation) CpDep DenseAt`.
  **Finding: this cannot be proved in Lean, for accept-all or for any other verifier.** It needs some `x` with
  `DenseAt x` and right-hand side `< 1`.
  - Where `RecDeploy.readX x = none`: `parse x = none`, `nS x = 0`, `xI = []`, so the inner Flock has no tables
    (`N₀ [] = 0`, `k = 0`). Then `κ ≥ e⁰ = 1 ≥ PrBad`, and the bound holds at every verifier.
  - `readX x = some _` needs `jsonOf "rec-inner" … = .ok _`, which runs `Lean.Json.Parser.anyCore`, a `partial def`,
    which the kernel treats as opaque. So no `x` is provably parsed. In the model where that parser always errors,
    every `x` is `DenseAt` with `κ ≥ 1`, and accept-all meets RecursiveKS. This is the 26-Oct vacuity class: what makes
    the statement non-trivial sits behind an opaque parser.
  **What I'll prove instead**:
  (A) `recursiveKS_acceptAll_of_readX_none : (∀ x, RecDeploy.readX x = none) → ∀ b aux nOf hd recRest, KSLin2On … (…).acceptAll … DenseAt`.
      This is the obstruction, machine-checked.
  (B) The narrowest nontrivial form: `¬ RecursiveKS-at-accept-all` from a named premise about one fixture statement
      `x₀`. The premise: the deployed parser takes `x₀` to a dense statement whose inner claim has no witness, with
      LamDeployed's `PlainIn`/`PartIn` facts there. Per top's 21:35Z reading, it is discharged by a deployed run that
      the lemma names by `art:` id. ExpectedCR (ecr/sha-512, in the lock) is read only at accept-all's four finders,
      whose declared cost is 0 under the silent prover.
      I'm checking that (B) closes, and that a `decOK` decoder exists for `Bd.deploy` and `DO`; if none does, the ∀ is
      itself vacuous. I'll report by 23:30Z. If (B) can't close, the RecursiveKS entry goes on LEAN-66's allowlist with
      (A) as its reason. The fix is then a total JSON reader on the `rec-inner`/partition path (the code side), so that a
      fixture parses in the kernel.
- 22:13Z coordinator: cpu-4 /workspace is at 94% (13 warm copies, 173 GB). NOW: delete every cpu-4 warm copy of yours whose build has finished (via a research run on cpu-4: rm only your own wt-* dirs), and reuse ONE copy per line of work, not one per run. infra removes wt-* untouched for 2 h from 23:30Z; at 97% the proofs slots p1-p4 come off.
- 22:15Z coordinator: ci: lock-only conflicts with next are FINE on main (#4716: merge.py merges build-determined lock records as a candidate; trains regenerate). A mark refused for a lean-audit.json conflict means your research tool is stale: in your checkout run 'git fetch origin main && git merge origin/main' (or pull) then 'uv sync', and re-mark. If it still refuses, send the exact refusal line to me.
- 22:20Z KSNONTRIV: plan (B) closes, with no hash premise. Branch `cursor/proofs-ksnontriv-27b2` @ bdad6a3feb; the build
  is r20261026-221609-ff79 (vy-nebius-cpu-4, proofs pool), waiting for a slot.
  `Flock.Guarantees.recursiveKS_nontrivial : FalseInScope → ¬ RecursiveKSAll`, where `RecursiveKSAll` is RecursiveKS's
  statement with `.acceptAll`. Accept-all also drops the record's recorded calls and V* statements, since it computes
  nothing. `FalseInScope` is one closed Prop: brings, a record, and a dense statement with no witness in the deployed
  scope (`PlainIn`, `PartIn`, sizes). There `λ₂ ≥ 100` gives κ < 1/2, while accept-all has Pr[acc ∧ ¬extract] = 1.
  The decoders aren't vacuous: they are the deployed ones (`listWithin_plainTrim`, `listWithin_zkDecodeRD`).
  The form is `¬ ∀ b`, not `∀ b, ¬`: κ reads `b.st`/`b.zl`.
  NONVAC's draft rule (cursor/lean-nonvac-fc8d) doesn't yet accept it. Its head is not RecursiveKSAt, and the premise
  is not in an `Assumptions` module. I asked lean on LEAN-66 (reply to the 21:27Z thread) to unfold Guarantees heads and
  to take `FalseInScope` as a named assumption that a fixture run discharges; else RecursiveKS goes on `unwitnessed`.
- 00:28Z KSNONTRIV: restated to lean's 23:31Z rule. Branch `cursor/proofs-ksnontriv-27b2` @ 81037941d4; build
  r20261027-002521-4ef9 (vy-nebius-cpu-4, proofs pool; waiting for a slot; the warm copy is built, so about 10-20 min
  once it starts). The 22:56Z build (60cdcb9634) failed: one linarith atom mismatch, two names, one arity, and a
  timeout from a wrong `hO` type. The general lemma, the κ < 1/2 step and the RecDeploy form compiled.
  Statement:
    theorem Flock.Guarantees.recursiveKS_nontrivial (h : RecDeploy.FalseInScope) :
      ∃ aux nOf hd recRest, ∀ b : RecDeploy.Brings,
        ¬ Claims.Recursion.KSLin2On FlockDeploy.sha512
          (RecDeploy.recBoundAt relation {deployedB b with tr := recTr aux nOf hd recRest} (layR elemR)).acceptAll
          (bdDeployed relation (nextR rdR) (deployPlainTrimR relation rdR).cols wER joinR rowInDeployed finalR wR)
          (deployZkRec relation) CpDep DenseAt
  That is RecursiveKS's KSLin2On with `.acceptAll` inserted, false at every brings `b`. The earlier note that κ reads
  `b.st`/`b.zl` was wrong: `tr` is overwritten by `recTr …`, and `Nums.κ` doesn't read `zl`.
  Built in Lean: accept-all, every brings, the deployed decoders, the silent prover, and M and R.
  One hypothesis, a closed named assumption in an Assumptions module:
  `RecDeploy.FalseInScope` (new file `Proofs/Flock/Claims/Recursion/Assumptions/FalseInScope.lean`, beside
  `PlainLeavesAdmitted`). It says some x, with a coin-server record, is DenseAt, has no Flock witness at xIOf x
  (deployed bdDeployed), is PlainIn at xIOf x, has PartIn at each staged part, and has nS ≤ 2^16 and setup ≤ 2^56.
  It's the one unbuildable field: the deployed reader reads only through jsonOf (`partial`, kernel-opaque) and only
  where each part's circuit is V*'s byte for byte (22 MB), so no statement can be built in Lean.
  `recursiveKSAtAll_of_readX_none` proves the bound holds at accept-all wherever nothing reads, so some such fact is
  needed. Its witness would be a fixture: the deployed compiled statement over an inner statement whose claimed output
  isn't its circuit's. No such run exists yet.
  Open for lean (LEAN-66): the head is KSLin2On. RecursiveKSAt hardcodes `recBoundAt …`, so no `¬ RecursiveKSAt …`
  can hold accept-all; `concluded` needs to read through RecursiveKS → RecursiveKSAt → KSLin2On. RecursiveKS's lock
  name: it isn't in a lock on kslock yet.
- 01:03Z KSNONTRIV: narrowed per the 01:08Z order. Head 3a74607b5f; build r20261027-010045-f8f2 (cpu-4, waiting for a proofs slot).
  The inner statement is built and its falsity, density and PlainIn are proved. The one named assumption is
  `RecDeploy.KSFixture := ∃ x aux nOf hd recRest, DeployedParses x ∧ PartsInScope x (recTr …)`, in
  `Assumptions/KSFixture.lean` (replaces FalseInScope.lean). 4ef9 (old head) built clean, with 8 lemmas on
  [propext, Classical.choice, Quot.sound]. Posted on LEAN-66 (comment cf421bc0). Asked RECE about the parameter form.

## 01:05Z from RECE (bc-2bb0c923): `RecursiveKSAt` with its compiled record as a parameter
Yes, and it's a small change, done on `cursor/proofs-kslock-4929` @ d9c1d1ba85 (KSStatement.lean):
- `Flock.Guarantees.RecursiveKSAtRec (D : Claims.Recursion.RecAt Coin) rd final wE w : Prop := KSLin2On sha512 D (bdDeployed relation (nextR rd) (deployPlainTrimR relation rd).cols wE joinR rowInDeployed final w) (deployZkRec relation) CpDep DenseAt`.
- `RecursiveKSAt rd final rest wE w b aux nOf hd recRest := RecursiveKSAtRec (recBoundAt relation {b with tr := recTr aux nOf hd recRest} (layR rest)) rd final wE w`. That is definitionally the old body; every proof of it is term-mode, so none changes.
- Your head can then be `¬ RecursiveKSAtRec (recBoundAt relation {deployedB b with tr := recTr aux nOf hd recRest} (layR elemR)).acceptAll rdR finalR wER wR`, which is your current statement folded.
- Build: r20261027-010308-729e on cpu-3, building KSDeployed and KSV plus axioms and the closure counts. ETA ~01:30–01:45Z (that copy last built tieprem).

- 02:46Z RECE (kslock): kslock-4929 is at c4403584dc: 1d0ce8ab97 (RecursiveKS proved, 8530 green on the standard axioms) merged with rowsfit-3a9d 7682f6090f (honfix, honest_half_fixture), ksnontriv-27b2 53a30ae7bc and kswit-00fe 8b53bc6c71, plus `Flock.SecurityProofs.RecursiveKS` (@proofs) listed under `guarantees`. r20261027-024405-7f27 (cpu-3) builds KSProved, HonestHalfFixture and KSNontrivial and prints their axioms. The one records run (--update) follows once 7f27 is green and KSNONTRIV's own green run id is posted. Please push no further changes to the merged branches without telling me.
- 02:50Z KSNONTRIV: the fixture is redesigned and built clean. Branch `cursor/proofs-ksnontriv-27b2` @ 53a30ae7bc; build
  r20261027-023911-64d4 (cpu-3): BUILD_RC=0, FINAL_RC=0, 11 lemmas on [propext, Classical.choice, Quot.sound].
  The rowView form (01974bba54, also clean in r20261027-022228-aa96) could not be witnessed.
  The probe r20261027-021317-15ee, a compiled lean_exe over core Flock on rec-coins/16-16-8-n64, found 0 of 6144
  fixed positions with a zero A row. The deployed form's fixed bits are hm96 digest bits on the pinned net (ranges
  can't overlap, and checkNets pins the net) or public-input self rows. So falsity of a deployed inner statement is a
  preimage question that the kernel can't settle.
  The new proof needs no false statement:
  - At accept-all the record stages no V* statement, so each session's statement is `[]`.
  - The outer parser refuses `[]` (`RecDeploy.zkRec_parse_nil`, via DeployScope's `insO_nil`).
  - So E_O assembles nothing, the inner view is ⟨[], rej⟩, and E* returns none (`E_none_of_vstar_nil`).
  - `prBad_acc` takes "E returns none" in place of "no witness".
  - The general form is `not_ksLin2On_acceptAll … (hk : 0 < V.k x) (hS : 0 < nS x) (hP : DO.parse [] = none) (hκ)`.
  Statement (unchanged head):
    recursiveKS_nontrivial (h : RecDeploy.KSFixture) : ∃ aux nOf hd recRest, ∀ b, ¬ KSLin2On sha512
      (recBoundAt relation {deployedB b with tr := recTr …} (layR elemR)).acceptAll (bdDeployed …) (deployZkRec relation) CpDep DenseAt
  The one named assumption: `KSFixture := ∃ x aux nOf hd recRest, DeployedParses x ∧ PartsInScope x (recTr …)`, where
  `DeployedParses x := (parse x).map (holo, sparse) = some (false, none) ∧ 0 < nS x ∧ (setupOf false (xIOf x)).map setupView = some true`.
  An honest inner statement now witnesses it: rec-coins plus V* staging, with the inner at m 27, kLog 22, 3 regions,
  mPts 29 and 2 points, which is inside setupView.
  The witness run is next. It needs a compiled exe over Security/Proofs (parse/nS/setupOf are Mathlib-importing
  modules), or Flock as a dynlib under the interpreter. `pick` in PartsInScope is still Classical.choose.
  RECE has merged 53a30ae7bc into kslock-4929 (c4403584dc). I'll push nothing more to ksnontriv-27b2 without telling RECE.

- 03:15Z KSNONTRIV: the KSFixture witness is done.
  - Run r20261027-030140-0e78 (cpu-3, SUCCESS); evidence art:476851730535363877c1ec51d6a6558a939de0842b245a6a4ec8ef2dcde6f71a
    (`summary.json`, `all: true`), refs bytes=art:d872affc… (c2a7's staged V* tree).
  - `flock-verify` built at 53a30ae7bc, with a `dbgTrace` of the setup numbers in a scratch copy (`probe.diff`), read:
    - DeployedParses: `rec-inner` writes no `holo` or `sparse` key (dense); `rec-statements` lists 10 V* statements (nS = 10);
      the plain inner setup is 1 table, 2 points, mPts 27, (m, kLog, regions) = (25, 22, 3), inside setupView, and accepted.
    - PartsInScope: all 10 parts' `--zk` setups are 1 table, 2 points, mPts 27–33, m 26–32, kLog 22–26, regions ≤ 48,
      inside PartIn, and all accepted.
  - Gaps, stated in the docstrings: no Lean term `x` is formed (these are the compiled reader's readings on the run's files);
    the staging is c2a7's at 4331ec922a; the parts are c2a7's, not `pick`'s choice.
  - RECE: I am pushing one docstring-only commit to `cursor/proofs-ksnontriv-27b2`, 5881ef4954 on top of 53a30ae7bc
    (KSFixture.lean, the DeployedParses and PartsInScope docstrings cite the run and art). Merge it into kslock if you
    want the citation in the lock's assumption record; no statement or proof changes, so no lock record changes. A build
    of it follows; I'll post its id.
- 03:17Z KSNONTRIV: 5881ef4954 (docstring-only, KSFixture cites r20261027-030140-0e78 / art:476851730535…f71a) built green in r20261027-031312-c79d (cpu-3): BUILD_RC=0, FINAL_RC=0, all 11 lemmas on [propext, Classical.choice, Quot.sound]. Nothing more from me on ksnontriv-27b2 unless asked.

- 03:26Z RECE (kslock): landprep-345d d84ccfd0cf (and f1asm b5f207e97c) is missing the RecursiveKS lock listing, and its lock merge dropped PRF-212's retirement.
  - kslock-4929 4a57bdfd24 is landprep d84ccfd0cf plus one file, Security/lean-audit.json (+21 −14). It adds:
    - `Flock.SecurityProofs.RecursiveKS`, `RecDeploy.HonestHalfFixture.honest_half_fixture` and `Flock.Guarantees.recursiveKS_nontrivial` under `guarantees` (@proofs);
    - `RecDeploy.HonestFixture` and `RecDeploy.KSFixture` under `assumptions` (keep KSFixture's record; its witness run 0e78 is under a smell check, and nothing in the lock cites it);
    - kswit-00fe's top-level `witnesses` entry.
  - It removes `Flock.SecurityProofs.RecursiveSound` and `RecursiveSoundExec` from `guarantees` and `shape.frozen` again. The code side of ca22793809 survived; its lock lines were taken from main's side.
  - Please merge kslock 4a57bdfd24 into landprep (a fast-forward of one lock file) before the one records run. ff46 at d84ccfd0cf would record without them.
  - 7f27 (cpu-3, c4403584dc): HonestHalfFixture RC=0 and KSNontrivial RC=0. KSProved failed only on VerdictRowEInner:229/:278 (rowsfit's new mem_portOpenings), which landprep already fixes. r20261027-032425-667b rebuilds all three at 4a57bdfd24 and prints the four axioms, ETA ~04:00Z.

- KSNONTRIV 05:10Z: recursiveKS_nontrivial restated per lean's LEAN-66 answer (Slack 1793074723.641809): cursor/proofs-ksnontriv-27b2 f865fe85f8 (merge of 4a57bdfd24 + KSNontrivial.lean only; KSFixture docstring 5881ef4954 rides along). Head: ∃ aux nOf hd recRest, ∀ b, ¬ RecursiveKSAtRec (recBoundAt … {deployedB b with tr := recTr …} (layR elemR)).acceptAll rdR finalR wER wR; proof unchanged (defeq). Name unchanged, so the lock's witness entry is unchanged; KSFixture stays under assumptions (@proofs). Landprep da50a63313 already merges it. Note: 4a57bdfd24 alone fails AccTies.lean (accCommit_eq family, lines 373/391/399; r20261027-042550-d70b), which landprep's 5d4e462f37 fixes. Build + #4871 witness_failures at da50a63313: r20261027-050727-70d0 (cpu-3), ETA ~05:35Z.

- KSNONTRIV 05:46Z: witness_failures (#4871, 7b57a65c85) = 0 at landprep da50a63313 (contains f865fe85f8): r20261027-050727-70d0, BUILD_RC=0 (11262 jobs, KSProved included), recursiveKS_nontrivial on [propext, Classical.choice, Quot.sound]. Nothing more from me on ksnontriv-27b2.
