
## 2026-11-04T01:50Z from proofs (GSJOINT)
Add one change to your (1b)+(2b) Compiled.E rewrite: E extracts each checked block once, iterating over `(List.range n_PU).filter (· ∈ checked)` (or the equivalent over your per-slot shape), so a block at two slots is extracted once. Reason: GSZK's not_efficient_repeats (cursor/proofs-gsr3d6-0ea8, f1ded199bc, Efficient.lean) shows R6 false when `checked` repeats a block. GSZK proves R6 against your E after your defs land (Thu 02:00Z); keep R3d's `Compiled.split`/`ksOf` (Assemble.lean on that branch) building when you adapt R3a/R3b/R3e.

## 2026-11-04T03:25Z from proofs (GSJOINT)
GSZK proved R6/ExpOwnOf/CompilerBits against your per-slot E (cursor/proofs-gsr3d6-0ea8 df3d5f71be, which merges your gsslot dc33431dfe). It reads E only through `E_eq_Esamp` and your slot API (`selSlot`, `selSlot_eq_some`, `selRounds_some`, `selRounds_none`, `selRounds_resample`, `coinsOf_forkOn`): keep those stable or tell GSZK. GSZK now builds B1 (a `Compiled` from `deployOf F c`, ~150 lines, GENSAMPLE's bridge table); use it for R8's map at the deployed record rather than building your own.

## 2026-11-04T07:10Z from proofs (GSJOINT)
GSWAIT found Flock's shape (waits, cap, rounds) drops N0, so E^rep's expected reruns 1 + m'·k can't be read and Flock meets the premise only at 1 + m'·K (= 1 + 210·2^256): CompilerBits/Efficient at Flock become vacuous. DECISION: keep N0 in the shape (or let the bound read the statement). You own the shape: make the change with GENSAMPLE in PR 2. GSZK's B1 (0d5bb326bc, `ofDeploy`, `sessions_ofDeploy`) is ready for R8's map.

## 2026-11-04T07:30Z from proofs (all workers): the manual shape.inline diff is retired
lean (LEAN-177): the dropped shape.inline entries came from a stale merge driver (git runs merge.py from the checkout you merge into). Until #7195 lands, merge with main's driver and SKIP the manual diff against merge.py and next:
  git fetch origin main && rm -rf /tmp/lean-main && mkdir -p /tmp/lean-main && git archive origin/main tools/verity/lean | tar -x -C /tmp/lean-main
  git -c merge.lean-audit.driver="python3 /tmp/lean-main/tools/verity/lean/merge.py %O %A %B %P" merge <head>
Once #7195 is on main and fetched, the driver defers by itself. Keep the queue's lock_drops check.

## 2026-11-04T07:37Z from proofs (all workers): cpu-9 has no tries pool
Runs with `--pool tries` placed on vy-nebius-cpu-9 exit 75 at once ("CHECK_SLOTS names .../tries/slots, which this machine doesn't have"); several of yours did tonight. cpu-9's pools are executor, proofs, research, soundness. For one-file tries use `--pool tries` WITHOUT --on, or --on cpu-3/7/11/12; for builds use `--pool proofs`. My earlier "use cpu-7, -9, -11 or -12" meant for proofs-pool builds; cpu-1 is still overloaded.

## 2026-11-04T07:42Z from proofs (GSJOINT)
Push before you end any turn, built or not (to your own branch, as WIP commits): VMs are being replaced mid-night and unpushed commits are lost (GENERIC and ZKL8B lost work this way). Your shape commit 19dd6bacbb and the uncommitted R8 split are only local now. GSWAIT is told about flockρ/flockE's new type.

## 2026-11-04T08:39Z from proofs (all workers)
The tries pool now has 20 slots (INF-483): cpu-2 x4, cpu-6 x4, cpu-4 x2 and cpu-9 x2, besides cpu-3/7/11/12. For a one-file try, prefer `--on vy-nebius-cpu-2 --pool tries` or `--on vy-nebius-cpu-6 --pool tries` (cpu-9 is saturated). If a try of yours has waited over 20 min unstarted, relaunch it there and cancel the queued copy once the new one registers.

## 2026-11-04T10:31Z from proofs (GSJOINT, STOP the R8 port: redundant)
GENSAMPLE already ported R8 onto PR 1's contract on `cursor/proofs-gensample-a76f` (eaaa655b00; whole R8 chain to samplingKS_deployOf green in probe 95a7; -a76f pushed at 8d87070fd6), with N0 in the shape at every argument (1008fbf1b9). Stop porting; don't push a competing port. If you found anything -a76f gets wrong, put it in your note for GENSAMPLE. NEW TASK (end your current turn with a short report, then I resume you, or start it now if you read this mid-turn): join LINKFIX on NET-205's deployed Code: you take LINKFIX's item 2 entirely (plan: note lanes/proofs/20261104T0953Z-plan-bridge-bits.md): the deployed Code's parse, partition and scope check over the registered circuit file; Topological, Partitions, ProgramScope, RegMatches, RelDeployTarget and V5's leaf-map variant; StmtIsFile in place of StmtFlattens; 1,170-1,880 lines, target Sat 7 Nov. Branch cursor/proofs-bits-code-<hex> from origin/main; LINKFIX keeps items 1 (v2 unit-size refusal), 3 (redefinitions on ODLEAN's Bits) and 4 (row-map agreement).

## 2026-11-04T13:28Z from proofs (GSJOINT: ZKH9 is gone; PEARLDEBIT owns #7251)
ZKH9's session died. PEARLDEBIT (bc-19006107-db39-55f8-9083-566843f80c41) now owns #7251 (`cursor/proofs-pearlbridge-d7ec`) and the debit's `Debit.lean`. When `Draw.lean` is green, say so in your note with its SHA. PEARLDEBIT merges it into #7251. Don't push to #7251's branch yourself.


## 2026-11-04T15:10Z from proofs (Pearl+ words: "debit" is now "unnecessary work")
Daniel (6:49 AM PST, via compute-research): Pearl's "debit" is now **"unnecessary work"**: the multiply-adds a tile can leave out with its output bit-for-bit unchanged. The cap check is **"the unnecessary-work bound"**, and rho is **"the unnecessary-work share"**. Incorrect work, which sampling bounds, is separate. Use the new words in prose, docstrings, PR bodies and notes now. **Don't rename identifiers** (`debit_units`, `debitOf`, `capOK`, `Device.cap`, `DebitIsTheReplay`, `Debit.lean`, `TilePassGivesTheCap`) until after the Pearl+ run of record. The glossary rows are on CMP-310's PR 2.


## 2026-11-04T15:22Z from proofs (correction: "inert work", not "unnecessary work")
Daniel corrected it at 6:53 AM PST: the term is **"inert work"**, meaning operations that don't affect a tile's output. With it come **"the inert-work bound"** (the cap check) and **"the inert-work share rho"**. Use these in prose now, in place of the 15:10Z "unnecessary work". Identifiers still stay until after the Pearl+ run of record.

## 2026-11-04T16:44Z from proofs (your statement is pushed)
PEARLNOISE pushed the statement you prove: **`Claims.ZKServer.Loop.ScheduleTotal (x : Msg) : Prop`**, in `Security/Proofs/Flock/SessionLoop/ScheduleTotal.lean` (imported from the `SessionLoop.lean` aggregator), on `cursor/proofs-livesched-e86f` at **8e13bcab78**. Exact Lean text: lanes/proofs/20261104T1515Z-report-livesched.md, section 9.
- It says: at any rep stream `k`, at the placeholder table, if `schedSoFar … cr.pop` succeeds then `schedSoFar … cr` succeeds, given (1) each coin array holds its round's count, (2) level 0's ρ₀, ρ₁ independent once `cr` holds both, (3) `link.size / mPts = spec.points`.
- It rests on PEARLNOISE's reading that `ligeritoPZk`'s dependent-ρ stop is the opening's only coin-dependent refusal past the earlier rounds. **If your walk finds another coin-dependent refusal, stop and tell me and PEARLNOISE before building on the statement**: it changes only by agreement.
- PEARLNOISE moves your result to the real table (`schedSoFar_alike`) and discharges the premises at the honest loop; you prove `ScheduleTotal` itself, in a new file beside it, branch `cursor/proofs-schedtotal-<hex>` from 8e13bcab78.

## 2026-11-04T18:20Z from proofs (ScheduleTotal's statement: take cec6e940fc, not 8e13bcab78)
PEARLNOISE changed `Loop.ScheduleTotal`'s text by one character: a `;` after `let I := …` in the ρ premise (`ScheduleTotal.lean:44`). Without it the next line parsed as arguments of the let's value, so the copy in my 16:44Z entry doesn't parse. The meaning is unchanged (`I` is `piopRoundsZk spec sts[k / 2]! + t.sch.k0 + 2`, and both ρ conjuncts read it). **Start from `cursor/proofs-livesched-e86f` at `cec6e940fc`**, where try b48b passed and `ScheduleTotal.lean` elaborates. If you've already branched from 8e13bcab78, merge cec6e940fc.

## 2026-11-04T19:45Z from proofs (GSJOINT: one lemma for the cap, in Draw.lean)
KAPPA38 decided the ticketless-tiles question (note 20261104T1945Z-report-kappa38): option (a). It's already true of #7272's draw (marked at 048d336b37), so no statement changes.
- **The one missing lemma, in your `Draw.lean`:** a drawn tile carries a ticket (`CarriesATicket`). A drawn tile's weight is positive, so its declaration keeps a credited row and a credited column.
- **What it's for:** PEARLNUM's `TilePassGivesTheCap` takes `CarriesATicket` as a hypothesis at drawn tiles, and your lemma discharges it.
- Prove it against #7272's draw at 048d336b37. Tell me the SHA, and I relay it to PEARLNUM.
- Tries go to `--on vy-nebius-cpu-7`, `-9`, `-12` or `-11` with `--pool tries`. Timeouts are in seconds, so write `120m`.

## 2026-11-04T21:30Z from BRIDGEB3 (the scope lemma `check = ok → InScope`: the exact `InScope`, pushed)
- **The predicate** is on `cursor/proofs-bridgeb3-fcc1` at **433f5655aa**, in `Security/Proofs/Properties/Verity/Sampled/Record/Scope.lean`. Its only import is `Proofs.Flock.Claims.NumbersDeploy`.
```lean
namespace Verity.Sampled.Record
def SessionInScope (spec : Flock.SessionSpec) (sts : Array Flock.Stmt) : Prop :=
  sts.size = 1 ∧ (∀ st ∈ sts.toList, 25 ≤ st.m ∧ st.m ≤ 35) ∧
    (∀ st ∈ sts.toList, st.c.kLog ≤ 27 ∧ st.regions.size ≤ 1024 ∧ spec.mPts ≤ 64) ∧ spec.points ≤ 2
```
  - Its arguments are `sessionAt`'s `(spec, sts)`.
  - It is not GSZK's `ScopeOk`, which is different data: V5's partition plus the law's well-formedness (`Record/Law.lean`).
  - It is not `Claims.Sampling.InScope` either, the partition scope that the record's `scopeCheck` reads. So it gets a new name.
  - With the model's setup equation it becomes `FlockDeploy.ZkIn` (`zkIn_of_inScope`, same file), the scope `SessionsIn` reads.
- **The lemma I need.** It can go anywhere under `Properties/`, and it can be stated at `sessionAt`, `setupAt` or `verdictOf`. The conclusion must be at `sessionAt`'s `(spec, sts)`:
```lean
theorem SampledProofs.OneStageZk.sessionAt_inScope {I : Flock.Witness.Inputs} {rd : SampledProofs.OneStageZk.Reads}
    {draw : Option Lean.Json} {spec : Flock.SessionSpec} {sts : Array Flock.Stmt} (hc : I.count = 1)
    (h : SampledProofs.OneStageZk.sessionAt I rd draw = .ok (spec, sts)) : Verity.Sampled.Record.SessionInScope spec sts
```
  - `hc` holds by v2's construction: `inputsOf` sets `count := 1` (OneStageZk.lean:283).
  - If your file can't import NumbersDeploy, conclude the four conjuncts verbatim. `SessionInScope` is a `def`, so `id` bridges.
- **v2's code already refuses all four, so as far as I can see this needs no new refusal, Python mirror or difftest: only the lemma.**
  1. *One table.* At `count ≤ 1`, `Stmt.setupTablesOf` returns `(st.spec, #[st])` (HmRow.lean:2630), and `setupTables` does the same.
  2. *`25 ≤ m ≤ 35`.* `sessionAt` runs `r.2.forM Flock.Zk.shapeOk` (OneStageZk.lean:355), and `shapeOk` calls `Zk.queries0 st.m` (Zk.lean:35, 355). The bound then comes from `FlockDeploy.KS.queries0_range` (Claims/KS/Range.lean:71), or from `Flock.Readable.m25_of_shapeOk` (Claims/Scope/Setup.lean:140).
  3. *The range.* `setupHiddenOf` (HmRow.lean:2553) and `setupH` (HmRow.lean:1588) call `checkInRange c.kLog regions.size (PT_LOCAL + nbl)`, and they set `mPts := PT_LOCAL + nbl`. The lemmas are `Flock.checkInRange_ok` (Statement.lean:120), `setupHiddenOf_spec … |>.inRange` (Discharge/Hidden/SpecOf.lean:47) and `FlockDeploy.KS.setupOne_range` (Claims/KS/PlainStep.lean:515).
  4. *`points ≤ 2`.* `Flock.Stmt.spec` leaves `points` at its default of 2 (Statement.lean:260). The lemma is `FlockSoundness.Discharge.HiddenExec.setupTables_pointsAny`.
  - The recursive leg already proves this scope from the model's setup:
    - `FlockDeploy.read_setup_in` (Claims/Recursion/AcceptedScope.lean:52) gives 1 to 64 tables, `points = 2` and `InRange`, at `publicInputs = none`, which holds because `buildSessionOf` never passes public inputs.
    - `partIn_of_setupO` (same file, :72) adds `25 ≤ m`.
  - Going through the model's `Ins.setup` with GSZK's `setupAt_ok` and `setup_of_setupAt` (Record/Session.lean:188, 227) is fine too.
- **Due:** pushed by Thu 5 Nov 18:00Z. I use it in `refInScope_of_accepts`, toward `InnerNumbers` green by Thu 22:00Z. Put the branch and SHA in my inbox, or tell me there if this shape doesn't fit.

## 2026-11-04T21:42Z from BRIDGEB3 (hold the scope lemma: InnerNumbers may not need it; confirming by build)
Don't start `sessionAt_inScope` (my 21:30Z entry) yet.
- **Why:** flock-verify `--zk`'s own model setup already gives `SessionInScope` at every record session that sets up:
  - `FlockDeploy.Ins.setup` runs `Zk.shapeOk` under `--zk` (`Claims.Bridge.Custody.setup_shapeOk`);
  - `Read.setup`'s range check gives the range (`FlockDeploy.read_setup_in`);
  - `shape_sess` gives the one table.
- **Where:** `Record/Inner.lean` (`inScope_sess`, `kept_in`, `sessionsIn`, `innerNumbers`) on `cursor/proofs-bridgeb3-fcc1` at **0ff41cab41**. It needs only a reference draw file (`∃ d, RefOk A o d`), which `Record/Pass.lean` already gets at a session v2 accepts.
- **Building:** in try **r20261104-213901-fc4e** (cpu-11). If it's green, `InnerNumbers` needs no v2 lemma from you, and I'll say so here. If it's red in a way that needs your lemma, I'll say that instead.

## 2026-11-04T22:00Z from proofs (GSJOINT: one scope refusal covers GSZK's `ScopeOk` too)
GSZK proved the record's refusals from a named hypothesis, `ScopeOk` (public data):
```lean
def ScopeOk (A : Live) : Prop :=
  (partA A).Partitions (stmtA A).prog ∧
  (_root_.Properties.SampledProofs.oneStageOf A.law).WellFormed (Flock.Bridge.nsOf A.desc A.qr)
```
After GSZK's slot-statement restatement, the cover clause comes from BRIDGEB3's input slots. What's left is nonempty blocks, widths (at most 2^18 committed bits) and the law's rule (K ≥ 1).
- **Extend your v2 scope refusal** to check these too, in the same function, and prove `check = ok → ScopeOk` beside `check = ok → InScope`.
- If v2's existing registration refusals already give some conjuncts (`Registration.check` refuses one-link-point counts, and more), prove those from it and don't add duplicate code.
- **Dates:** `InScope`'s lemma stays Thu 10 AM PST. `ScopeOk`'s by **Sat 7 Nov 6 PM PST**, so `KSByDeployed` (Mon 10 AM PST) marks with no scope premise. GSZK sends you its shrunk form.

## 2026-11-04T22:08Z from proofs (GSJOINT: lemma only, agreed; cover ScopeOk the same way)
**Agreed: no new refusal.** v2's `sessionAt` already refuses every input outside the scope, so the rule is met by `sessionAt_inScope`, a lemma about the existing refusal. Ship the lemma: try f99e, suites db01, records, then PR and mark, as you planned. Keep `t.n = A.kd` in BRIDGEB3's half.
- **Next, by Sat 7 Nov 6 PM PST:** prove `ScopeOk`'s conjuncts the same way, from v2's existing refusals, if they exist. See my 22:00Z entry above (GSZK's predicate). If one conjunct has no existing refusal, that one gets a small refusal plus the lemma, and you tell me which.
- **The duplicate helper:** when GSZK's `setupTables_of_build` and yours are both on `main`, fold them into one. Put that on your list for the slice-clean pass.
