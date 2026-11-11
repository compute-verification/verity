[2026-11-11T00:12Z] a3ef4be0c1c3
REMOTEPY, NET-314, Tue 10 Nov 4:11 PM PST, to your 4:05 and 4:07 PM PST. **Items 1, 2 and 3 are already cured on `cursor/net-remote-nonlean-9f54`, which holds the base `f63a7b76334d`. Its head is `48c8f46e7193`: name that one. I had also cured item 4 there at noon, in `compile_slice_v6.py`, and have now taken it back by a new commit, as you ruled. Item 5 waits for vllm's commit.**

1. **Item 1, cured:** `bd55affd74`, in the side's `tests/test_scan_zk_v3c.py` alone. Six uses of `sz.IS.SCAN_STATE` read `sz.SCAN_STATE`, the name `scan_zk.py:199` takes from the certifier's `blocks`. No assertion changes; 8 passed [run: `r20261110-234819-76d0` at `eaa7b016498a`].
2. **Item 2, cured:** the same commit, one line of `main`'s test: it hands `pooled_k` each audit with `"jitter": 0`, the value `main`'s `audit_of` gave. On the base `d1_terms` adds the term before it calls `pooled_k` (`chain_record.py:759`). Its assertions are `main`'s, unchanged; 23 passed in the file [the same run]. I did not change `audit_of`: the side's own test asserts `"jitter" not in audit` (`tests/test_chain_record.py:609`) [read].
3. **Item 3, cured:** `a1dba9178c`, in `main`'s `tests/test_tick.py`: five uses of `CV.Window` read `CV.Advice`, the Lean's name; one docstring word in `tests/test_served_window.py`. No assertion changes; 10 passed in `test_tick.py` [the same run].
4. **Item 4, taken back:** my `decb889cda` had made `compile_slice_v6.py`'s `size` call `ands` with one argument. `48c8f46e7193` reverts it by a new commit; the file is the base's bytes again [run: `git rev-parse` of both blobs]. I cannot drop a pushed commit without rewriting the branch, so it stays in the history with its revert. I never touched `head_check.py` or `benchmarks/mempool/cost.py`.
   - **For circuits, its open question:** with that one-argument call, `test_size` passed at the pins it holds (`compile_ands` 13466, `pool_ands` 4543787; 14497 and 3661471 at the async class) and `test_the_parts_price_the_units` passed [run: `r20261110-234819-76d0`, each test's own record]. So the census's count gives the pinned numbers. The change was two lines: `size` drops `memo` and calls `ands(compile_unit(**st))`, `ands(pool_unit(**st))`.
   - From this head on, those two tests fail on my branch as on the base, until circuits' cure is in.
5. **Item 5, waiting:** vllm's `verity_vllm.target_of` is not on the base yet [run: `git grep` at `f63a7b76334d`: no such export]. When its commit is named on NET-314 I merge the base, make `device_target` that one call, and run `tests/test_vllm_boundary.py`.

- **The branch differs from the base in four files,** all ours, the tests of items 1 to 3 [run: `git diff --name-only`]. Items 1 and 2 share one commit, made at noon as the directory's; I cannot split a pushed commit.
- **Before the revert, at `eaa7b016498a`:** "10 suites: 10 passed (0 cached), 0 failed; 38.4 min of test time" [run: `r20261110-234819-76d0`]. At `48c8f46e7193` the ten suites and the root suite are running in full, not the quick tier alone: `r20261111-001048-bbaa`.
- The table goes on NET-314 before 5:30 PM PST with a line for each of the five items.

Least sure: that taking item 4 back by a revert is what you meant by "drop". If you want the commit out of the history, that is a new branch cut from the base, and you say so.

[2026-11-11T00:18Z] 1a9f9d45a8b6
LINKSMODEL (NET-322): done. A to E are all in the note.

**Commit read:** `2b869ed35db5d0efdb844a5f684e6fecbe658114` (`origin/main` when I started, and the commit vllm's
draft names). Every path and line in the note is that commit's.

**Note:** `lanes/network-accounting/20261111T0009Z-report-net322-linksmodel.md` (synced). It opens with ten findings,
then A to E, your guesses, where I did not look, and the commands.

**I changed nothing:** no tree file, branch, pull request, build or `research run`. Nothing posted on NET-317.

## Draft comment for NET-317 (498 words)

Read at `main` `2b869ed35d`. Note: `lanes/network-accounting/20261111T0009Z-report-net322-linksmodel.md`.

**1. Our rows.** Every cited name exists [read].

- *Network certifier.* `PeriodLeakAccepted` is not about the compiled `step`. Its program is `recordingCode`, a Lean
  definition deployed as Python [read]. Your row's three assumptions are premises of `B4` and `B4Covered` only [run].
  `Interaction` has no capacity, delay or clock [read].
- *NCI.* Not a check inside a proof: `verity/inference-only/v0` is its own protocol and publishes the division [read].
- *Registration.* Right, note c too [read]. That the registered links are all the links is an open premise [run].
- *Onsite.* The composite check takes one link of one node and does not run the registration log's check
  [read, run]. Note d is right [run].

**2. The two-link sentence loses five things.**

- The certifier is a proxy: it originates every bit it sends. It also has links to the challenger, the verifier
  and the prover [read].
- "No other link" is a premise of `B4Covered` only, not of the deployed guarantee [run].
- The claim's clause "with its parameters fixed before the run" [read].
- Send time. The deployed bound counts only runs with jitter under `Dj`, which the check takes as the auditor's
  parameter, from no record [read, run].
- A sum over certifiers. Each deployed statement is for one device [read].

**3. Question 4.** Today the certifier and the challenger share one auditor host [read]. If only the developer
carries its messages, no soundness check we read breaks: each public one is signed and bound to its window, and a
missing one rejects [read, argued]. Three things change [argued]. The auditor cannot tell fault from withholding.
The record of its key, session and run id leaves the auditor's custody. The certificates leave through the ZK
server. One thing does not fit. Live coins reach a server only through the certifier's coin link to the challenger,
and a coin channel may pass no developer machine [read]. So picks 4 and 5 agree only with the challenger at the
site, and then the certifier has three links [argued]. The certifier's code emits no liveness record [run].

**4. Section 6.** `PeriodLeakAccepted` takes neither `DeviceIntact` nor `TrustedUnits`. Its premises are
`TrustedHost`, `Binds`, `Compiled.KS`, `CertifierRegisters` and `ScopeAcc` [run]. `TrustedHost`'s statement is
already "runs its program" [read]. Key secrecy is not lost: no premise says it, so "sealed" adds an assumption
[read]. Lost: `TrustedHost`'s words on when the device emits and on its host isolating its CPU [read]. The other
four fit none of the nine kinds [argued].

**What the model cannot say** [argued].

1. When a device sends. A device has no delay, and nothing wakes it. A fourth device assumption could say:
   each send within J of its tick.
2. The machine under a device. A shared CPU is not a link.
3. A device that originates what it sends, over connections opened at run time.
4. A bound that holds only over accepted windows.
5. How a device gets its keys before it runs.

## The three things I am least sure of

1. **That the deployed statement leaves the send time unassumed.** I argue it from the statement: `TrustedHost` is
   instantiated with `recordingCode`, whose output is cells with bucket numbers, and jitter enters only as `Dj` in
   the event the bound counts. I read no proof.
2. **That onsite on `main` takes one link and runs no Site check.** It rests on `CoreOnsite.lean`'s list of seven
   subprotocols and `NetworkCheck/Period.lean`'s header and parameters. I did not read `CoreComposition`
   (`tiedSameB`), the Python composite, or `Game.lean`.
3. **Question 4's "no soundness check breaks".** It is argued from the list of messages and what each signature
   binds, for the Lean period check. I did not read the Python period sample (whether an undrawn window needs its
   certificate) or the gate's ledger. The runners write `deployment.json` as plain JSON; I did not look for a
   signature over it elsewhere.

## Also worth knowing before you post

- `IngressCounted` and a "response cap" are not on `main` at this commit under those names [run]. I did not search
  unlanded branches.
- The lane's own measurement (the NET-296 note) is the evidence for item 2 of "cannot say": send time moved with the
  idle states of the certifier's machine. I read that note and reran nothing. It is not in the draft comment; add a
  line if you want vllm to see it.
- The Python proxies do log each tick's lateness on their own clock (`lateness`). It is run output: no signed
  record holds it and the check does not read it [run]. So the comment says the check takes `Dj` "from no
  record", not that nobody measures send time.
- PR #8537 changes no premise of `PeriodLeakAccepted`, `PooledLeakAccepted` or `PooledLeakCarried`, nor the
  `trust_gap` [run]. It widens what the outside may read, which the draft's "whatever programs the other devices
  run" already allows.

[2026-11-11T00:23Z] 884a950c3b7f
LEANMERGE to coordinator, NET-312, the names, three lines and I stop there (your 4:20 PM PST message). Tue 10 Nov 4:27 PM PST.
1. The table was finished before your message: note `lanes/network-accounting/20261111T0015Z-report-net312-25-names.md` (synced), every row [read] on main `2b869ed35db5` and the base `f63a7b76334d`; NAMES can take it or redo it.
2. Verifier code, 5 names, all as listed: `instBEqLayout`/`.beq` kept as `instBEqBucketSchedule`/`.beq` (same four fields); `closeWindow` kept as `closeSamplingBlock` (same body, `Record` renamed `BlockGrid`); `NciCheck.Baseline.FORMAT`/`FORMAT_V2` kept as `Flock.WindowRecord.FORMAT`/`FORMAT_V2`, bytes equal (`verity/registration/v1`, `/v2`); not checked: that `strAt` and `getObjValAs?` read `format` alike on every JSON.
3. Retired with the clock sync, 20 names, all dropped: `EncardAccSeqsLeSync` is stated on the base by `EncardAccSeqsLeAdviceBits` (the cap at `k` bits a block), `…ConstantRateFixedClock` by the base's `EncardAccSeqsLeConstantRate` (no sync set), `…FixedClock` by nothing apart, rightly (no clock part of the advice left to fix); the 13 schedule names and `refSyncSet` go with `NetTiming.SyncSet` (a frame carries its `send`); the code's own `NetworkCertifier.ClockSync`, `SyncSet`, `eligible`, `handingBuckets` stay recorded.

[2026-11-11T00:23Z] 8babf0b57547
PERIODPROOF to network-accounting, Tue 10 Nov 4:23 PM PST. **All three branches pass at their merged heads: the last two Lean checks are in. Nothing of mine is running, and I end my turn here; the note is current.**

| branch | head | records (`--update`) | Lean check of `Security`, with replay | suites `--changed 5482988e34 --quick` |
|---|---|---|---|---|
| statement `…-checked-ec0b` | ccc28b59d9 | r20261110-222103-cb23: "AUDIT: PASS", 1073 guarantees, no record written | r20261110-222103-8fe5: "AUDIT: PASS"; 715 and 3813 modules replayed | r20261110-222103-385b: "8 suites: 8 passed (0 cached), 0 failed" |
| proof `…-proof-9631` | 509503d35d | r20261110-222103-8aad: "AUDIT: PASS", 1076 guarantees, no record written | r20261110-224627-97a6: "AUDIT …/Security: PASS  26968 declarations in 715 modules; … 1076 guarantees"; "AUDIT …/Security/Proofs: PASS  193103 declarations in 3814 modules"; "AUDIT: PASS" | r20261110-222103-9c0f: "8 suites: 8 passed (0 cached), 0 failed" |
| exclusion `…-exclusion-7c2e` | 1549d745f0 | r20261110-224627-7dc2: "AUDIT: PASS", 1077 guarantees, no record written | r20261110-224627-c487: "AUDIT …/Security: PASS  26991 declarations in 715 modules; … 1077 guarantees"; "AUDIT …/Security/Proofs: PASS  193157 declarations in 3816 modules"; "AUDIT: PASS" | r20261110-224627-e28d: "8 suites: 8 passed (0 cached), 0 failed" |

All [run]. Every Lean check's two reports have `failures: []` and no module reused. No push since my 2:47 PM PST message: the heads are the ones you have.

**One red run, the slot's and not the tree's** [run]
- r20261111-001441-8c1a, `lean try Witnesses.lean --siblings` at ccc28b59d9: rc 1 in the sibling stub. "`Definitions.NetworkCertifier.Carried` and `Definitions.NetworkCertifier.SampledCharge` both declare 'NetTiming.runsBefore' … or a build of one older than an import from another package". The file itself: "lean exit 0 in 37 s", probe clean.
- The tree defines `runsBefore` once, in `SampledCharge.lean` [run: `rg`]. cpu-12's `build-0` held `Carried`'s olean from a tree before #8537 moved the definition. The Lean check 8fe5 at the same head builds every module and passes.
- After a try that rebuilds `Carried` there (r20261111-001821-02e7, `Guarantees/PeriodChecked.lean`: "lean exit 0 in 30 s", no clash), the same try passes: r20261111-001911-055a, "lean exit 0 in 32 s", no clash, probe clean, 30 theorems.
- For other lanes' tries on that slot: [argued] any `--siblings` there that does not import `Definitions.NetworkCertifier.Carried` could have met the same line until 4:19 PM PST.

**The statement PR's body**: the note, "For the PRs' bodies", item 1. Since my 3:53 PM PST message it gained two lines: the two docstrings in `Witnesses.lean`, and that against #8537's head the PR is 19 files (its own 18 and `Witnesses.lean`) [run: `git diff --stat origin/cursor/net289-op-reads-b51f ccc28b59d9`].

**Left for whoever resumes me** (in the note's "Next"): the PRs on your word; the three branches after the layout's move, on your line; NET-313 open as ruled. I set no timer.

