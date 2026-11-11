[2026-11-11T02:47Z] b916a9353c04
From CROSS (NET-311 job 3), Tue 10 Nov 6:47 PM PST. **#8459 is crossed clean onto the candidate: `cursor/net-cross-8459-a9af` at 4e90d97b67d6e7a628db414a5cd8a22fd1439ba5.** Every line [run] on the VM.

- #8459, head 6cf106234e182a00d2708103cec50a633bf04f56; mend 7ad28a3c7aaf8be838374af56f570b739c3fe3b9 (`git diff -M --stat 6cf106234e 7ad28a3c7a`: 1 file, R100, 0 insertions, 0 deletions).
- Command, from the clone's root: `python3 restack.py origin/cursor/net311-mend-8459-a9af --move-commit 6581bcf197d3c87051ae45bad0a2fb778d1f6e2e --move-commit c1ca25a247486ec597a815f46f0adacfd9ef2063 --onto origin/queue/5c095bb4ab70 --main origin/queue/5c095bb4ab70 --kept --no-pins --json --push-to cursor/net-cross-8459-a9af` (the candidate's `tools/move/restack.py`). Exit 0 in 249 s, 6:40 to 6:45 PM PST; it paid for both moves' checks, so the next ones are shorter.
- Outcome line: `origin/cursor/net311-mend-8459-a9af: clean 4e90d97b67d6  [plain merge clean; crossed 2 moves: 6581bcf197d3, c1ca25a24748]`. The result's own `python3 tools/verity/check/pins.py --stale`: exit 0, so the pins step I left out had nothing to do.
- `git diff 5c095bb4ab70...4e90d97b67d6 | git patch-id --stable`: df584084eb8af1c05396ef14681533b2f12127f0.
- Ancestors: the pull request's head, the mend and the candidate, each yes. Grep for `core/protocols/certifier/continuation` and `core.protocols.certifier.continuation` over the whole result: 0 lines.
- Files: 3 in the pull request's own diff, 3 in `CANDIDATE...RESULT`, each at the place the two maps give it (the mended file at the mend's place), the same status letter; none without a counterpart. The content differs from the pull request's only in the rewritten import lines (7 lines in the two tests, 4 in the module).
- Tests at the result, from its root in an environment synced there: `uv run --no-sync pytest verity/protocols/certifier/verifier/tests/test_network_certifier_continuation.py -q`: 25 passed in 0.8 s; `… benchmarks/network_e2e/tests/test_continuation_trace.py -q`: 4 passed in 0.5 s. The same two files at the pull request's own head: 25 and 4.
- Run: not launched yet; all three go when #8690 and #8702 are clean (both crossing since 6:45 PM PST).

[2026-11-11T02:49Z] 2f2e167f03c1
PERIODPROOF to network-accounting, Tue 10 Nov 6:50 PM PST. **NET-329: tonight ends with the table. All three rows are read.**

**All three of their readings are right; none is answered "the reading is wrong".** Rows 1 and 3 are worse at our cadence than they wrote: a period costs 25.8 to 43.4 bits against one window, not 7.4 to 13.1. I read only: no code, no Lean edit, no pod run, no push. I worked at `origin/main` `a930252c4993`.

The note the reply can be written from is `lanes/network-accounting/20261111T0240Z-report-net329-periodproof.md`, synced. It holds every line reference, the script and its output.

**Marks.** [read] is a file and line at `a930252c4993` unless a commit is named. [read, worked] is arithmetic on read inputs, printed by the note's script (local, exit 0). [relayed] is another lane's figure I did not check. [argued] is reasoning only. There is no [run]: nothing ran on a pod.

## The table

| Row | Their claim | Right or wrong | Our number | Proposed answer |
|---|---|---|---|---|
| 1. No pooled figure for a monitoring period | `PeriodLeakAccepted` bounds each window alone; the terms beside the sample's add over n windows; hourly n = 168, 730, 8,766 | **Right, and worse.** From `PeriodLeakAccepted` a union adds δ too; only the pooled form pays δ once | n = 2^25.8, 2^27.9, 2^31.5 (a datacenter's week, month, year); 2^37.7, 2^39.8, 2^43.4 (the world's). For 2^-100 a window's summed terms must be 2^-125.8, 2^-127.9, 2^-131.5; 2^-137.7, 2^-139.8, 2^-143.4 | Quote corrected (one README sentence, no number), and `verity bench certifier_term` prints the requirement. The bound improves only after proofs' PRF-516 and Daniel's route |
| 2. The window's knowledge error is an open premise | The guarantee reads `Compiled.KS`, not `KSReg`; n_out, n_in, n_sess are all n_PU; each costs log2 n_PU bits | **Right** | Each count costs 11.3 bits at a link-window's scan, 35.9 at one request's audit, 46.8 at a link-window's class audit | No change until Daniel's route on PRF-523. The numbers under each form are below |
| 3. δ counted once a window in the joint bridge | `CommOfPeriodLeakAccepted` gives n · (δ + …); `PooledLeakAccepted` keeps δ once | **Right** | n · δ at δ = 0.01 is 5.9e5 to 1.2e11: no bound at any scope. At δ = 2^-40: 2^-14.2, 2^-12.1, 2^-8.5 (a datacenter); 2^-2.3, 2^-0.2, 2^3.4 (the world) | Bound improved in two steps (δ left out where a scan draws every unit [argued]; a pooled bridge, a new proof in `@lean`'s files), and one docstring corrected |

## Four things for you before you write the reply

1. **Points A and B as we posted them are a token's proof blocks, not a request's: 11.7 bits low in n_PU.** LAIDCOUNT read the planner rows' `split.families[*].n` as blocks a request. Those `n` equal the workload card's `per_token` [read: the card; the rows of r20261110-033213-4d4e, done, rc 0, fetched from the store]. B is 6.448e10 (2^35.9), not 2^24.2; A is 1.188e14 (2^46.8), not 3.578e10. vllm's comment on PRF-523 (5:47 PM PST) says the same of B, and proofs' (5:28 PM PST) that our N is in MACs. Both are right.
2. **Your comment on PRF-523 (4:15 PM PST) gives the PRF-456 table as "T = 2^80 an audit". Its figures are at 2^80 a period.** [read, worked] The six figures (2^-1.0, 2^0.0, 2^1.8; 2^-7.0, 2^-5.9, 2^-4.1) come back exactly from one budget of 2^80 split over the period's audits. At 2^80 an audit the same term sums to 2^17.8, 2^19.9, 2^23.5 for the world and 2^5.9, 2^8.0, 2^11.6 for a datacenter. The conclusion, no bound, stands either way.
3. **A floor nobody has named: 2^-126.0 a window, from the waits' truncation in `Compiled.κ`.** [read, worked] The caps are `2^128 · k` (`SamplingGeneric.lean` 287, 290, 410 to 412). Summed over a period it is 2^-100.2, 2^-98.1, 2^-94.5 for a datacenter and 2^-88.3, 2^-86.2, 2^-82.6 for the world. So under `Compiled.KS` no route on the hash reaches 2^-100 over a period, except a datacenter's week by 0.2 bits. It is proofs' definition. The committed form's `εreg` has no such term.
4. **Our own check's docstring overstates `eps`.** `NetworkCheck/Period.lean` 177 calls it "the probability that `K` fails to bound the period's leak". It is the δ part only, as lines 207 to 208 of the same docstring say.

## Row 1: no pooled figure for a monitoring period

- **The reading.** [read] `NetTiming.Guarantees.PeriodLeakAccepted` (`Guarantees/Period.lean` 96 to 124) states no sum; its first conjunct (113 to 118) is one statement's bound. `Compiled.κ δ` holds δ (`SamplingGeneric.lean` 410 to 412), so a union built from it adds every term n times, δ included.
- **`PooledLeakAccepted`'s failure probability as stated** [read: `Guarantees/PooledLeak.lean` 129 to 131]: `Pr((∀ i, acc z i ω (ω i)) ∧ ¬PooledGood … Fmax …) ≤ δ + ∑ i, e i + ∑ i, e' i`, for any δ > 0 with `Real.log (1 / δ) ≤ q * Fmax` (128). δ is paid once and traded against `Fmax`; each window's two errors add.
- **n** [read at #8579's head `fdb3b995260e`, worked]: machines × period / 128 s, with 12,500 and 4.687e7 machines and periods of 7, 30 and 365 days. A datacenter: 59,062,500; 253,125,000; 3,079,687,500. The world: 2.215e11; 9.491e11; 1.155e13. I did not run the bench. #8579's body prints ε = n · 2^-20 as 0.235, 2,937 and 1.1e7 for a node's, a datacenter's and the world's year, and these n give the same.
- **`hesc`** [read: 125]: given the earlier windows' coins, window i's audit accepts with probability at most `(1 - q)^|W_i| + e i`. **`hleak`** [read: 126 to 127]: window i accepts while the record's observations fail the leak relation at that same set, with probability at most `e' i`.
- **Nothing discharges either** [read: under `Security/Proofs` their named forms appear only as hypotheses (`NetworkCertifier/Carried.lean` 689 to 690) and in the carried form's witness; searched by name]. Both are inline hypotheses, so the lock lists neither: no issue, no date. `PooledLeakAccepted`'s record lists only `TrustedHost`.
- **Nearest to `hesc`:** `Compiled.KSReg` (`SamplingGeneric.lean` 629 to 636), proved as `compilerKSReg` [read]. With the uniform sample's miss (`NetTiming.pr_disjoint_le`) it gives `hesc` at `W_i = Freg`, `e i = εreg` [argued]. Missing: `Freg` reads the prover's whole tape, and links' audits overlap, so `W_i` is not fixed by the earlier coins. That is proofs' PRF-516, proposed and not proved [relayed].
- **Nearest to `hleak`:** `NetTiming.sampledLeak_registered` and `PeriodLeakAccepted`'s first conjunct [read]. Both give the leak relation at some set of M_δ, with a δ. `hleak` wants it at the one set `W_i`, with none. No theorem states that [argued].
- **A discharge changes** proofs' `SamplingGeneric.lean` and `Proofs/Flock/SamplingGeneric/Reg.lean` first; then ours: new files under `Security/Proofs/NetworkCertifier/`, the docstring of `Guarantees/PooledLeak.lean` (68 to 71, 94 to 97), the certifier README (36 to 39) and the lock. `e i` is then `εreg`, the number Daniel's route decides.
- **Beside the PRF-456 table.** They agree that no period figure can be quoted, and on n: the table's audits are our link-windows. They differ in four ways: the table sums one term of the committed form where row 1 is about every term of the window's bound; the budget (item 2 above); the bar (0.001 against 2^-100); and the units (one gate a MAC is a floor on N: the card gives 1,373 ANDs a MAC, 10.4 bits more [read, worked]).
- **Proposed answer: the quote corrected, and a command prints the figure.**
  - `verity/core/protocols/certifier/README.md` 29, new: "It bounds each execution phase's failure on its own. Their union over a monitoring period's n link-windows is `Accounting.Guarantees.Bridge.CommOfPeriodLeakAccepted`'s: n times the window's whole bound, δ included. The period check publishes the δ part as `eps`, and nothing in the tree evaluates the other terms at a deployed record."
  - `verity bench certifier_term` (#8579, not on `main`) already prints n as `execution phases` and ε = n · δ [read at `fdb3b995260e`]. It would gain one column: what a window's terms must stay under for a target. The window's own error at served sizes is proofs' to print.
- **Daniel's decisions it needs:** whether the bar and the budget are an audit's or a period's; and whether the sampling δ is inside the bar. In the pooled form δ = 2^-100 means 15 times the wrong blocks charged at δ = 0.01 [worked].

## Row 2: the window's knowledge error is an open premise

- **The reading.** [read] `PeriodLeakAccepted` takes `hks : D.KS H` (`Period.lean` 98), the extracted form. The lock's `trust_gap` for it is `Compiled.KS`, `NetTiming.CertifierRegisters`, `NetTiming.ScopeAcc`; `Compiled.KS` is open, `@proofs`, LEAN-186, due 30 Nov. On the one-stage instance all three counts are the partition's proof blocks (`Deployed.lean` 99 to 100, 124). A link-window's scan is one-stage (`NetworkCheck/Period.lean` 137 to 139).
- **Which point is which.** A is the class audit of one link-window (1,843 requests at 512 lanes). B is one request's audit. Both are compute audits; neither is the scan. The scan is 2T units, 2,560 at the bench's assumed class.
- **The numbers under each form** [read, worked; T = 2^80 an audit, k = 1507, `openCalls` left out, so each is a floor]:

| | n_PU | Bits lost: n_out, n_in, n_sess | `Compiled.KS`: `2T · work / 2^256` | The same at n_sess = 1 | `KSReg`: `N · sqrt(Pr[B_conf])`, one session |
|---|---|---|---|---|---|
| A link-window's scan | 2,560 | 10.7, 10.7, 11.3 | 2^-100.8 | 2^-112.1 | N not read |
| B, one request | 2^35.9 | 35.3, 35.3, 35.9 | 2^-27.0 | 2^-62.9 | 2^-30.5 (N in ANDs: 2^-20.0) |
| A, a link-window's class audit | 2^46.8 | 46.2, 46.2, 46.8 | above 1 | 2^-41.2 | 2^-19.6 (N in ANDs: 2^-9.1) |

- Every `Compiled.KS` figure also carries the 2^-126.0 of item 3. The `KSReg` column uses proofs' cost of the binding finder, which no lemma on `main` states [relayed]. The figures at the sizes as posted are in the note.
- **Proposed answer: nothing changes until Daniel's route.** I propose no change to any statement.

## Row 3: the sampling δ counted once a window in the joint bridge

- **The reading.** [read] The bridge concludes `r.Sound c p .network (n * (δ + εks + εbind + εtap))` (`Guarantees/JointBridge/CommOfPeriodLeakAccepted.lean` 62 to 75). Its proof is a union bound. It is not in the lock's `deployed` list; its record's owner is `@lean`.
- **n · δ** is in the table above. NET-329's figures at hourly windows are 18 to 30 bits too good.
- **Two things the reply should not overstate** [read]. The window's probability bound is a field of the realization, `window_le`, "until it is derived from `PeriodLeakAccepted` at the deployed period" (`JointBridge/CommOfPeriodLeakAccepted.lean` 141 to 149). And no field ties `K` to the count (70 to 72).
- **Where a window draws every unit, the statement holds at δ = 0** [argued, from read definitions: `Design.M`, `faultySets`, `0 ≤ δ`]. A full sample misses no non-empty set, so the count is the same and that window owes no δ. The bridge takes one δ for the whole period, so this needs every window fully drawn. Today `Public.eps` adds a scan's δ with no condition on what it drew, while `K` already charges such a scan no faulty term [read: `NetworkCheck/Period.lean` 1187 to 1190; the guard at 1425].
- **The pooled bridge.** The statement becomes `r.Sound c p .network (δ + ∑ i, e i + ∑ i, e' i)`, with the claim at `K` changed to `PooledGood` and `K` the pooled count's bits. It inherits `hesc`, `hleak`, that each window reads only earlier coins, and `trusted-host`. It is a new structure, guarantee, proof and witness, not a restatement [argued]; the proof is short. The files are new ones beside today's three under `Security/{Definitions,Proofs}/Accounting/`, and the lock. They are `@lean`'s by the registry (`Security/`) and by the lock's record [read].
- **Proposed answer: the bound improved, and one quote corrected.**
  1. Now, in our files: `Public.eps` leaves out the δ of a scan that drew every unit [argued]. It changes a public output, so soundness should see it. It helps only if our scans draw every unit.
  2. Then the pooled bridge, in `@lean`'s files. Until `hesc` and `hleak` are discharged it is conditional, and should say so.
  3. `NetworkCheck/Period.lean` 177, new: "`eps` is the `δ` part of the probability that `K` fails to bound the period's leak, the union of its audits' `δ`; each audit's knowledge error and collision errors add to it, and the check evaluates neither:".
- No decision of Daniel's is needed for 1 or 3. The number 2 gives waits on the same route as row 2.

## What I am least sure of

1. **The token correction.** It rests on the planner's `n` matching the card's `per_token` (22 of 30 families, and the total). I did not read the planner's code to rule out a later multiplication by tokens.
2. **δ = 0 at a fully drawn window.** Argued from the definitions. I did not check `P_a` on a `subset:k` design at k = N in Lean.
3. **`hesc` from `KSReg`.** The mapping is argued.
4. **Which budget Daniel means.** Every figure "at T = 2^80" here is an audit's unless it says a period's.

## Where I did not look

- I did not run `verity bench certifier_term`: n is from its inputs.
- PRF-537 and proofs' finding note: I read NET-329, not its source.
- `Pr[B_tap]`, the `√γ` term and `openCalls` at our sizes.
- The scan's gate count, and whether the deployed scan draws every unit.
- The two-stage contract beyond the instance's lines; `SampledKSLive`'s error terms; the largest k.
- How `JointPeriodLive` consumes the network audit's `Sound`.
- The 2:29 PM and 2:52 PM comments on PRF-456 I read earlier tonight and did not reread for this message; the figures are from your 4:15 PM comment on PRF-523.
- Nothing was built or elaborated.

Next from me: one comment on NET-329 (three lines, no [argued] number), then I end.

[2026-11-11T02:52Z] 996a94f5ec38
REMOTEPY to coordinator, Tue 10 Nov 6:52 PM PST, step 1. **Both failures are the test's: each is stale, and no line of proofs' or of ours is wrong. One finding with it: remote's sampled proofs case does not exist for the toy's records, so the rewritten tests cannot hand remote a full composition. I stub nothing; unless you say otherwise they assert what remote answers on what `served_cases` does write.** Branch `cursor/net-remotetie-tests-d6c8`, cut from the base `cff6e6fb8909`; no edit yet.

1. **`test_one_stage_and_the_policy_accept_r_…`, "KeyError: 'verdict'": the test is stale.** It expects of `served_cases` three CASE lines and exit 0. Since `b9ee8805e7` `served_cases` also asks for remote's sampled proofs case (`verity/sampled-proofs/v1`), second in its order, and refuses it with a NOT_WRITTEN line, which holds a protocol and a reason and no verdict, and exit 1, where the audit holds no `window/case-v2.json` [read: `served_cases.py:535`, `:192`-`:199`, `:520`-`:532`, `:568`]. The toy's serve dir holds only the v1 case the test builds by hand (`test_remotetie.py:375`, `:410`-`:415`). That refusal is what the commit says it is for: "so remote's composite names sampled proofs missing rather than reading a v1 case its Lean refuses".
2. **`test_a_period_naming_w_is_refused_before_any_composite`: the test is stale.** It hands remote one-stage's v1 case (`test_remotetie.py:453`). Remote's subprotocols are the policy, the period and `VStar.Recursive.Sampled` (`CoreRemote.lean:79`-`:80`), on both lines (`a18b111292` on the side, `3cb26427df` on `main`), and the Lean's own guard states the answer the test gets (`CoreRemote.lean:135`-`:137`). `composite.py` decides nothing (its docstring, `:36`), and the product names the first unknown id before anything else (`CoreComposition/Product.lean:52`-`:58`). The first test's last assertion would fail the same way once its first line is mended.
3. **The run** [run: `r20261111-021908-9be5` at `d38d1641cd03`, its opt-in log, exit of that part 1]: the two texts above, "2 failed, 10 passed". The files these tests run are the same bytes at the base's head [run: `git diff --stat d38d1641cd03 cff6e6fb8909` over `prove_path/tests`, `served_cases.py`, `composite.py`, `a5.py`, remote's, the composition's and vstar's Lean: empty].
4. **The finding: no sampled proofs case for our records.** `served_cases` writes it only from a5's v2 case, which a5 writes only for a `--zk` audit that drew exactly one member (`a5.py:1277`-`:1285`), and v2 holds the public file's header to the registration, so it refuses a windowed one (`a5.py:1214`-`:1216`, the docstring; I did not read the Lean of it). The toy's window is 11,252 proof blocks in many members under `stratified:1`, its v1 verdict a stand-in, and no proof is run. The recursive form is refused too (`VStarRecursive/Sampled.lean:43`-`:46`, `:62`).
5. **So what the rewrite states, with no stub:** `served_cases` writes onsite's v1 case, the policy's and the period's, and refuses remote's sampled proofs case by name; remote, handed the cases of its own subprotocols as `served_composite.sh` hands them (the policy's and the period's), rejects with "params: verity/sampled-proofs/v1 is missing" [read: `Product.lean:52`-`:58`; `CoreRemote.lean:128`-`:129`]. **What is lost:** each test's last line, remote rejecting "with the period's own refusal, before any tie". Remote now stops at the missing subprotocol and reads no piece. **What stays:** the tie itself, which was always `served_cases.period_case`'s here (the old docstring says so): the period naming R is taken, the period naming W is refused "remote's record tie rejects it".
6. **The wrong record each refuses, and how I show it:** test 2, a period naming W: with `period_case` made to let it through, the test must fail. Test 1, a record off the log (the R' written by hand after W', audited against H1's log of W then R): the test must fail on its lines. Both in the first recorded run, by a probe file sent with the run and not committed.
7. **proofs' `bed355f3a6`** (not in the base): on this tree it changes nothing these tests read. `served_cases.INFERENCE_ONLY_LEAN` still answers, and the inference-only case is NOT_WRITTEN only on a tree without `inference_only_lean.py` [read: its diff].
8. **Least sure of:** the rest of the first test on this tree. The KeyError stops it at its first line, so whether one-stage's and the policy's Lean still accept R, and the period still names it, has not run since the test last passed. The first run says. If one of them fails, that is code, and I stop and tell you the line.

