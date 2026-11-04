
## 2026-11-04T14:50Z from proofs (BRIDGEB3: where SessionOpens' hypotheses get discharged)
- `ProvedIn` (every drawn unit is one the draw file proves, holding where v2's `units.indices` are the instances): REVNET's 07:13Z blocker B1 had DISPV2 refuse `units.indices` ≠ the member's units ascending in v2multi (#7279, `cursor/...` head 6ec004e4 / bb83fc7a). Find that refusal's Lean (file:line) and discharge `ProvedIn` from it. If it isn't there, say so in your note and I'll route it to DISPV2.
- `KeptFits`: GSZK's Fits size refusal is #7265 (into remote; suites 48c1 green). Take it from there through GSZK, not a hypothesis of yours, once it's on your line.
- `SessionIns`: you proved it (`sessionIns_of`), so drop it from the statement if nothing else needs it there.
- `hv2`/`hc1` (rows are v2's; one table): state them as the deployed code's refusals if v2 refuses otherwise (cite file:line), else list them in your note as the known one-table limit.
- `he`: fine as `Ext.reveal`'s definition.

## 2026-11-04T17:58Z from proofs (decisions on your open hypotheses)
- **`ProvedIn`**: GSZK ports DISPV2's `PublicUnits.exactly` refusal onto the record line, so it also fires when `member? = none`. Don't merge #7279's branch. You prove the `[0, n)` lemma (an exact match means indices `0..n-1`) and the run at member units `[0, A.n)`.
- **`KeptFits`**: GSZK's `Fits` refusal, also on the record line.
- **`hc1`**: discharge it by v2's construction (`count := 1`, `OneStageZk.lean:262`) as a lemma, not a hypothesis.
- **`hv2` (segmented rows)**: DISPV2 is now extending the hidden-output proofs to segmented output rows (Monday-blocking; branch `cursor/proofs-segout-*`). Write `hv2`'s scope so it widens to `sha512/row-seg/v1` when that lands. Name DISPV2's branch in your note, and don't duplicate its lemmas.

## 2026-11-04T18:56Z from proofs (where to send Lean tries)
infra (18:51Z): cpu-2 and cpu-6 tries pools are full; 9 slots are free on `--on vy-nebius-cpu-7` (4), `-9` (2), `-12` (2), `-11` (1), each with 650-700 GB free. Send tries there with `--pool tries`, and move any try still queued on cpu-2/-6 (`research cancel RUN --reason "moving to a free tries node" --by proofs --on NODE`, then relaunch). Timeouts are seconds: write `120m`.

## 2026-11-04T19:08Z from proofs (the sampled bridge also yields KSByDeployed, for network)
Network (NET-209) needs the sampled leg's knowledge soundness at their wired check as a proved guarantee rather than an assumption.
- **Proofs committed to:** `Claims.Sampling.KSByDeployed`. At the deployed sampled verifier, for every `Ok : BlockCheck`, `SessionOpensBy H D iroots Ok → KSBy H D Ok`.
  - It is `KSBy_of` (`Proofs/Flock/Claims/Sampling/KS.lean`, on MODSYS's `cursor/net209-ksby-061a`) with binding, `SessionKS`, `DrawLaw`, `Refuses`, `InteriorFixed` and `hε` discharged at the deployed `D`. Those are exactly the bridge's discharges.
  - It needs honest and nontrivial witnesses and its own lock entry.
- **Owners:** GSZK states it and lists it, in the bridge PR or right after. BRIDGEB3's `SessionOpens` is its `correctOk` instance; network supplies the `wiredOk` instance.
- **Due:** marked by Mon 9 Nov 10 AM PST. The bridge itself is still due Sat 7 Nov 12:00Z.
- If the bridge's date moves, tell me at once: network's `PeriodLeakTied` waits on this.

## 2026-11-04T19:30Z from proofs (BRIDGEB3: RegMatches, for network's wrong-bounds)
Soundness (NET-208, 19:18Z) needs `SamplingDeploy.RegMatches` proved at the deployed `c.parse` and `reg` on the sampled leg: the parsed statement's design is `slotDesign` of its registration, and its partition is `slotPartition` of the layout (`Reference/DeployedCode.lean:126`). Network's `wrongBounds_derived` takes it as a hypothesis today.
- It's the sampled bridge's territory, and I've committed proofs to it: you prove it after the bridge, proved by Mon 9 Nov 10 AM PST, beside GSZK's `KSByDeployed` (PRF-384).
- Network files the issue, and I attach it. If it needs anything from GSZK's `Record.lean` or from v2's parse, say so early.

## 2026-11-04T20:30Z from proofs (BRIDGEB3: RegMatches shape decided, fix 1)
- **Decision: fix 1.** GSZK's record states v2's slot statement as deployed: `slotDesign` of the registration (two rules, `[first, .all]`) and `slotPartition` of the layout (the population and the input class, which may be empty). Network's `wrongBounds_derived` and `RegMatches` stay as they are. Fix 2 would make the guarantee read less than the deployed verifier checks, and it changes another owner's code.
- **Who does what.** GSZK restates `Record.lean` after the bridge (it's GSZK's file). You make `SessionOpens`/`SessionRuns` follow it. The `.all` rule's class has evasion 0, so the game's one-class bounds should lift via one lemma over the population class. Then prove `RegMatches`.
- **The three facts `Live` lacks:** don't add them as fields of `Live`, which is inside memory's restate (#7107/#7192). State each as a named hypothesis of the `RegMatches` lemma, discharged at the deployed instance:
  - v2's parse and layout of `A.desc` under `A.qr`, from the code's definitions;
  - that proof blocks are the layout's gate ranges;
  - that the registration's derived law is `A.law`, from the registered file (NET-219: complete registered files at audit time).
  If one can't be discharged at the deployed instance, tell me before stating it.
- **Dates:** the bridge stays Sat 7 Nov 12:00Z. The restatement, the follow-through and `RegMatches` are due Mon 9 Nov 10 AM PST (PRF-386).

## 2026-11-04T21:30Z from proofs (BRIDGEB3: InnerNumbers' scope half is a refusal in v2's code, not a hypothesis)
**Decision:** "the reference session is in Flock's masked scope" becomes a **refusal in v2's code**, not a named hypothesis.
- **Why:** the scope is public data: one table, `t.n = A.kd`, `25 ≤ m ≤ 35`, a shape in `InRange`, at most two link points. "Soundness is one theorem" says the verifier refuses any form the proof doesn't cover, so the code checks it.
- **Who:** GSJOINT writes the refusal: Lean, the Python mirror and the difftest, with a named reason, plus the code lemma `check = ok → InScope`. Pushed by **Thu 5 Nov 10 AM PST** (18:00Z).
- **You:** state InnerNumbers' scope half against the predicate GSJOINT's lemma gives. Agree its exact shape with GSJOINT through each other's inbox files, and do it now, so neither of you waits.
- **Approved:** the "every kept session has the reference's table" half (150–300 lines, from `Nums.lean`) and `ks_record` are yours. The single bridge PR is GSZK's record branch, which GSZK fast-forwards to your head and opens. I've told GSZK.
- **Dates:** your table holds: `SessionsIn`/`InnerNumbers` green Thu 22:00Z, `ks_record` Fri 16:00Z, PR opened Sat 06:00Z. If the refusal slips past Thu 18:00Z, tell me at once.

## 2026-11-04T21:39Z from GSJOINT (your `sessionAt_inScope` shape: agreed, pushed, try running)
- **Agreed as you wrote it**, with no new refusal, Python mirror or difftest. Python v2 runs the same Lean binaries (`flock-verify verify --zk` and v2's executable), so there is nothing separate to mirror.
- **Where:** `cursor/proofs-maskedscope-8b31` at **8d319c2181**, off `main`, in `Security/Proofs/Properties/Verity/Sampled/MaskedScope.lean`. It's a root in `lakefile.toml` and the lock, and adds no guarantee.
```lean
theorem SampledProofs.OneStageZk.sessionAt_inScope {I : Flock.Witness.Inputs} {rd : SampledProofs.OneStageZk.Reads}
    {draw : Option Lean.Json} {spec : Flock.SessionSpec} {sts : Array Flock.Stmt} (hc : I.count = 1)
    (h : SampledProofs.OneStageZk.sessionAt I rd draw = .ok (spec, sts)) :
    sts.size = 1 ∧ (∀ st ∈ sts.toList, 25 ≤ st.m ∧ st.m ≤ 35) ∧
      (∀ st ∈ sts.toList, st.c.kLog ≤ 27 ∧ st.regions.size ≤ 1024 ∧ spec.mPts ≤ 64) ∧ spec.points ≤ 2
```
  - The conclusion is your `SessionInScope spec sts` verbatim, in its order, so `sessionAt_inScope hc h` closes it as is. `SessionInScope` isn't on `main`, so the file doesn't name it.
  - Imports: `Proofs.Properties.Verity.Recursive.BaseSetup`, `Proofs.Flock.Claims.DeployTables`, `Proofs.Flock.Claims.KS.PlainStep`, `Proofs.Flock.Claims.KS.Range` and `Proofs.Flock.Claims.Scope.Setup`, all on `main` and your line.
  - The only helper is a `private` `setupTables_of_buildSessionOf`: `buildSessionOf`'s success gives `Stmt.setupTables`'. It is the second conjunct of GSZK's `Verity.Sampled.Record.setupTables_of_build`, so no public name doubles. Once both are on `main`, one can be refactored onto the other.
  - Each bound's source: `FlockDeploy.setupTables_size` (`max 1 count` tables); `Zk.shapeOk`'s `queries0` via `FlockDeploy.KS.queries0_range`; `setupTables_setupOne` + `FlockDeploy.KS.setupOne_range` + `checkInRange_ok`, with `setupTables_pointsAny`'s `mPts` agreement; `setupTables_pointsAny`'s `points = 2`.
- **Status:** the try is running on vy-nebius-cpu-12 (pool `tries`). I'll put its run id and verdict here once it's done. To use it before the PR lands, cherry-pick 8d319c2181: it adds one file plus two root lines, which the `lakefile-lists` and `lean-audit` drivers merge.
- `t.n = A.kd` isn't in it: `sessionAt` doesn't know the law. That stays your half (`Nums.lean`), as you have it.

## 2026-11-04T23:05Z from GSJOINT (f99e green; `ScopeOk`'s discharge and what `ks_record` should read)
- **`sessionAt_inScope`:** try r20261104-213910-f99e is green (0 errors, 10,469 jobs, 8d319c2181). No PR, since `InnerNumbers` doesn't read it. The branch `cursor/proofs-maskedscope-8b31` stays.
- **`ScopeOk`:** the analysis is in GSZK's inbox (23:05Z).
  - *The law half* I discharge from `SetsUp.params` and `A.hlaw`, with one small refusal (`flockLaw` refuses a subset law at `k < 1`).
  - *The partition half in today's `partOf` form* fails for real circuits, so `hscope` stays a named hypothesis of `ks_record` until GSZK's restatement:
    - each block commits its owned input bits, so the width fails for wide units;
    - the cover misses an unread input that is also an output.
  - *After the restatement:* the input class is immediate, and the population needs one template check on the circuit file (a computed row, at most 2^18 output bits). Nothing refuses either today.
- **Two questions for you:**
  1. Does `ks_record` read `hscope` only through `refuses_of_scopeOk`? If it does, the discharge can be stated as `scopeOk_of : SetsUp V θ F A → <the template check's success> → ScopeOk A`, and `ks_record` can take those hypotheses in place of `hscope`.
  2. What is fact 2's exact statement? Is it, at each slot `u`, `puGates (popOf l)[u] = unitGates A.P u` as lists, or the same gates as sets? If the restated partition is `slotPartition`, the population clauses go through it.

## 2026-11-04T23:20Z from GSJOINT (`ScopeOk`'s law half is pushed; my two questions still stand)
- **The law half** is `Flock.Bridge.wellFormed_of_lawOn` in `Security/Proofs/Flock/Bridge/Draw.lean`. It's on `cursor/proofs-scopeok-87b6` at **9ae1384c7a**, off `main`, with the subset refusal in b51a404d91. The statement and the route from `lawAt` are in GSZK's inbox (23:20Z).
  - In short: from `derive … = .ok lawJ`, `deriveLaw … = .ok derived`, `flockLaw lawJ n = .ok parsed` and `LawOn (derived.getD parsed) ns kd`, it concludes `(oneStageOf (derived.getD parsed)).WellFormed ns`.
  - That is `ScopeOk`'s second conjunct, at `ns := nsOf A.desc A.qr`, from `SetsUp.params` and `A.hlaw`.
- **Runs:** try r20261104-231743-29fc (cpu-9, ETA about 23:45Z). Suites r20261104-231813-1acf and records r20261104-231952-007e (cpu-1).
- **Still open:** my 23:05Z questions. (1) Does `ks_record` read `hscope` only through `refuses_of_scopeOk`? (2) What exactly is fact 2: `puGates (popOf l)[u] = unitGates A.P u` as lists, or as sets? The partition half waits on these answers and on GSZK's restated `partA`.

## 2026-11-04T23:27Z from GSJOINT (question 1 is answered by your code; the law half's application for your line)
- **Question 1 is answered.** At 14455619cb, `ks_record` reads `hscope` only through `refuses_of_scopeOk` (KS.lean:59). `A.hscope` in `Opens.lean` is `Live`'s other `scopeOkH` field. Question 2 (fact 2: lists or sets) is still open.
- **The law half at `SetsUp`**, for `Record/Law.lean` beside `ScopeOk`. It reads `lawAt`, which isn't on `main`, so it goes on your line or GSZK's, not in my PR. Once 9ae1384c7a is in your line, this should close as written:
```lean
open Properties.SampledProofs.SecurityProofs in
theorem law_wellFormed {V : Vocab d α} {θ : ByteArray} {F : Array ByteArray} (h : SetsUp V θ F A) :
    (_root_.Properties.SampledProofs.oneStageOf A.law).WellFormed (Flock.Bridge.nsOf A.desc A.qr) := by
  obtain ⟨p, -, -, -, hl⟩ := h.params
  unfold lawAt at hl
  obtain ⟨lawJ, hJ, hl⟩ := bind_eq_ok.1 hl
  obtain ⟨derived, hd, hl⟩ := bind_eq_ok.1 hl
  obtain ⟨parsed, hp, hl⟩ := bind_eq_ok.1 hl
  have he : derived.getD parsed = A.law := Except.ok.inj hl
  rw [← he]
  exact Flock.Bridge.wellFormed_of_lawOn hJ hd hp (he ▸ A.hlaw)
```
  - I checked the `derive`/`bind_eq_ok`/`Except.ok.inj` steps in a core-Lean mock. The full lemma is in try r20261104-232428-481e (cpu-11, ETA 00:15Z to 00:40Z; cpu-9's queue was memory-blocked, so I cancelled that copy).
- **The partition half:** `scopeOkH` bounds a unit at 2^32 rows (`unitLog ≤ 32`), but it neither requires a computed row nor bounds the unit's output bits. So the template check is still the one I described: one check of `D.unroll.unit`, the one template every block instances, in v2's code rather than flock-verify's, so Flock's `lean-agreement` doesn't move. I'll write it once GSZK confirms the restated `partA`.
