[2026-11-11T09:13Z] f87cb05306e9
LEANMERGE to the coordinator, Wed 11 Nov 1:13 AM PST. `378a`'s outcome, read from its files (`fetch --all`).

**In one line:** the package's own check passed and kept a record (`regenerated`, four true values, all my change's,
committed and pushed); the run failed on its requirer `Security`, for one lock value of my change and one failure that
is the base's, not mine. Not the node's.

## 1. Verdict lines, the package and step that ended it, `path`

- `path`: `slot` [run: `result.json` `workload_fingerprint.path`; log line 1 "lean-records: taking the slot path"].
- What ended it: the requirer `Security`, in `check.py`'s lock rules (`guarantees`, `trees`), exit 1 after 573 s. The
  build, the axioms and every declaration passed in all three packages.
- Verbatim [run: `r20261111-070615-378a` `stdout.log`, rc 1]:
  - `AUDIT …/Security/Properties: PASS  3290 declarations in 104 modules; axioms propext, Classical.choice, Quot.sound; 533 guarantees`
  - `lean-records: Security/Properties: check.py exit 0 in 5046 s … without the kernel replay or the tests: a lean-fast/v1 result`
  - `lean-records: Security/Properties: the regenerated record is …/lean-audit/record/Security/Properties/lean-audit.json; commit it over Security/Properties/lean-audit.json`
  - `AUDIT …/Security/Proofs: PASS  198848 declarations in 3896 modules; axioms propext, Classical.choice, Quot.sound; 0 guarantees`
  - `AUDIT …/Security: FAIL  27598 declarations in 727 modules; axioms propext, Classical.choice, Quot.sound; 1099 guarantees`
  - `lean-records: Security: check.py exit 1 in 573 s …`
  - `lean-records: Security requires Security/Properties, and its records fail on this build, not only on record differences, which take a person`
  - `OUTCOME`: `Security/Properties` regenerated, `Security/Proofs` pass, `Security` failed; outcome failed.
- Of the 5046 s, 4400 s were `proved-in-wait` and 338 s the build [run: the audit's `seconds`].

## 2. Which of the three

**`Security/Properties`: `regenerated`, true values, mine. Committed.**

- The kept record against the head's lock, as JSON: four values differ, all under `reads`; key order and layout are
  the same (the text diff is those four lines); `format` is 1 in both [run: my comparison on the VM].

| Entry | Was | Is |
|---|---|---|
| `reads["NetworkCheck.Period"].digest` | `58568376bc79…` | `22674ce8e3b4…` |
| `reads["NetworkCheck.Period"].definitions["NetworkCheck.Period.factsOf"]` | `ee751ac31ab2…` | `d6edc3686ad2…` |
| `reads["Properties.NetworkPeriod"].digest` | `90782066c03a…` | `3dbdf9f0d053…` |
| `reads["Properties.NetworkPeriod"].definitions["Properties.NetworkPeriod.RowsTiedChargedAt"]` | `e77972111642…` | `178350d0e4fd…` |

- Both are my change's own: `factsOf` has the three refusals, `RowsTiedChargedAt` the conjunct `f.drawn ≤ f.units`.
  The audit's `review` prints these two definitions as changed and no other (its one further line: "reads: 397
  definitions read in another module or under another name: the same under their old names").
- **`rowsTiedCharged`'s own record does not move**, against what we expected: its `type_hash` (`967b07df…`) and
  `unfolded_hash` (`01fba8c7…`) are as in the base's lock, so its `unwitnessed` entry stands. No guarantee's record
  moves in this lock (0 of 533). The statement's change is carried by `RowsTiedChargedAt` under `reads`.
- Who reads `factsOf` there, by the `review` line: the three `Properties.NetworkPeriod.rows…` and 23 `Verity.*`
  guarantees (`coins_couple`, `remote_soundness_of_subprotocols`, `soundness_at_of_componentBounds`,
  `soundness_at_of_targets`, `soundness_of_subprotocols`, `soundness_of_upFront`, `target_of_periods`, `wipe_target`
  and their witnesses). Their records do not move; the definition they read did.
- **The commit:** `13419dcffc5a04702396bda69d670e874d144183` on `cursor/net325-drawn-le-units-22f3`, pushed as a
  fast-forward at 1:12 AM PST: the kept record over `Security/Properties/lean-audit.json`, byte for byte (sha256
  `ab85ba4ec320…`, the run's `record_sha256`). I did this under your rule of 1:06 AM PST; the brief's "no lock
  commit" said the opposite. If you want the base's lock back, it is one revert commit: say so.

**`Security`: a real failure, two messages. Not fixed.**

- `guarantees: definitions in NetworkCheck.Period that Flock.SecurityProofs.RecursiveAuditC3 read changed (changed: NetworkCheck.Period.factsOf)`
  - Mine: the same `factsOf`. `Security/lean-audit.json`'s `reads["NetworkCheck.Period"]` needs the same two values
    (digest `22674ce8e3b4…`, `factsOf` `d6edc3686ad2…`) [run: the requirer's report's `reads`]. It is the only
    module of that lock whose reads move, and `RecursiveAuditC3` (owner `@proofs`) its only reader there.
  - The run kept no record of `Security`, so I committed none and wrote none by hand.
- `trees: NetTiming.Witness.PooledLeakAccepted_honest read NetTiming.RowsFormat.recordFormat in Proofs.NetworkCertifier.RecordRows, a module of the proofs tree (Proofs, Proofs.Properties): a guarantee's or property's statement reads only the definitions tree …`
  - Not mine. The witness (`Security/Proofs/NetworkCertifier/PeriodWitnesses.lean:30`, owner `@network-accounting` in
    the lock), `recordFormat` (`Security/Proofs/NetworkCertifier/RecordRows.lean:199`) and `Security/lean-audit.json`
    are the base's byte for byte [run: `git diff --quiet f5b870781287 HEAD` on them]. So the base alone fails it
    [argued: I ran nothing at the base].
  - The base's last commits on those files: `0fb03a8fe6` (the witnesses) and `96c2c1a9dc` ("fxQueueSched moves to the
    definitions tree"), which moved one of the two definitions the witness reads and left `recordFormat`.
  - While it stands, no records run of `Security` ends green on this base, whatever my branch holds.

## 3. `d815`: the three suites, all passed [run: `r20261111-070538-d815`, rc 0]

- `passed  verity/core/protocols/network 89 passed in 522 s; 89 outcomes recorded`
- `passed  verity-core-composition      30 passed, 1 skipped in 4560 s; 30 outcomes recorded`
- `passed  verity-network-traces        545 passed in 383 s; 545 outcomes recorded`
- `3 suites: 3 passed (0 cached), 0 failed; 91.1 min of test time`
- The old base's nine reds in `verity-network-traces` are gone: 545 is the 536 and the nine.

## What I did not do

- No second run: nothing here is the node's.
- No run of `Security`'s records, and no hand-written value in its lock. Say if you want the run sent; it will end
  failed on the `trees` line until that is cured.
- I read `378a`'s `stderr.log` tail once on my VM; its slot lines carry other runs' commands, cut at 260 columns, no
  secret in them. I filter those lines now.

Next from me: job 1's report and job 2's draft, by inbox.

[2026-11-11T09:15Z] 4a8d0ac3c9e4
LEANMERGE to the coordinator, Wed 11 Nov 1:15 AM PST. NET-325 on lean's new base: the end report.

**The branch is ready to name, with one caveat that is not this change's: on this base `Security`'s lock check is red
whatever the branch holds (a `trees` failure of the base's own witness), and `Security`'s lock still needs one module's
`reads` re-recorded for this change.**

## What to name on LEAN-249

- **Branch and head:** `cursor/net325-drawn-le-units-22f3` at `13419dcffc5a04702396bda69d670e874d144183`.
  - `13419dcffc` is the lock commit of 1:12 AM PST over the merge commit `0b12ac8cd69f` (parents `750292d4cfe3`, my
    six commits, and the base `f5b8707812871f312448b16f039cccea773388f6`). Every push was a fast-forward.
  - Against the base it differs in six files: my five and `Security/Properties/lean-audit.json`
    [run: `git diff --name-only f5b870781287 HEAD`].
- **`rowsTiedCharged`'s record: it does not move.** `type_hash` `967b07df0522…` and `unfolded_hash` `01fba8c75836…`
  are the base's, so its `unwitnessed` entry (the type hash) stands as lean's worker wrote it [run: `378a`'s kept
  record against the base's lock, as JSON].
- **Its module's `reads`: two values move**, committed in `13419dcffc`:
  - `reads["Properties.NetworkPeriod"].digest`: `90782066c03a…` to `3dbdf9f0d053…`.
  - `…definitions["Properties.NetworkPeriod.RowsTiedChargedAt"]`: `e77972111642…` to `178350d0e4fd…`.
- **And two more in the same lock, which you did not expect**, also mine and committed:
  - `reads["NetworkCheck.Period"].digest`: `58568376bc79…` to `22674ce8e3b4…`.
  - `…definitions["NetworkCheck.Period.factsOf"]`: `ee751ac31ab2…` to `d6edc3686ad2…`.
- **One thing of another lane's reads what moved:** `Security/lean-audit.json` pins `factsOf` too, read there by
  `Flock.SecurityProofs.RecursiveAuditC3` (`@proofs`). Its `reads["NetworkCheck.Period"]` needs the same digest and
  `factsOf` value. Not recorded: `378a` kept no record of `Security`. Only these two locks pin `factsOf`
  [run: `git grep` over `*lean-audit.json`]. In `Security/Properties`, 23 `Verity.*` guarantees read `factsOf` as
  well; their records do not move.

## The merge

- `f5b870781287` merged into `750292d4cfe3`, no rebase, no conflict [run: `git merge`, Tue 10 Nov 11:01 PM PST].
- It touched none of my five files: the base has not changed them since `2f30e1709f1e`.
- The locks were the base's whole until `13419dcffc`, which changes four values of one lock.

## The files

- `verity/core/protocols/network/lean/NetworkCheck/Period.lean`
- `Security/Properties/NetworkPeriod.lean`
- `Security/Proofs/Properties/NetworkPeriod.lean`
- `verity/core/protocols/network/tests/period_case.py`
- `verity/core/protocols/network/tests/test_network_period_protocol.py`
- `Security/Properties/lean-audit.json` (four values under `reads`)

The files naming `factsOf`, `auditCheck`, `auditChecks` or `chargedOf` are the first three [run: `git grep -w`].

## The runs, all at `0b12ac8cd69f`, `--on auto`, one each

| Run | What | Exit | Verdict lines as printed |
|---|---|---|---|
| `r20261111-070505-89cf` | try `NetworkCheck/Period.lean` | 0 | "lean exit 0 in 13 s"; "axiom probe: no theorems in it" |
| `r20261111-070350-1452` | try `Security/Properties/NetworkPeriod.lean` | 0 | "lean exit 0 in 67 s"; "axiom probe: no theorems in it" |
| `r20261111-070432-b967` | try `Security/Proofs/Properties/NetworkPeriod.lean` | 0 | "lean exit 0 in 141 s"; "axiom probe: none of its 39 theorems uses an axiom beyond propext, Classical.choice and Quot.sound" |
| `r20261111-070311-18ac` | the period check's Lake package, its `#guard`s in it | 0 | "✔ [67/70] Built NetworkCheck.Period (2.2s)"; "Build completed successfully (70 jobs)." |
| `r20261111-070538-d815` | the three suites | 0 | "passed  verity/core/protocols/network 89 passed in 522 s"; "passed  verity-core-composition      30 passed, 1 skipped in 4560 s"; "passed  verity-network-traces        545 passed in 383 s"; "3 suites: 3 passed (0 cached), 0 failed" |
| `r20261111-070615-378a` | records of `Security/Properties`, `--update` | 1 | "AUDIT …/Security/Properties: PASS … 533 guarantees"; "AUDIT …/Security/Proofs: PASS  198848 declarations in 3896 modules"; "AUDIT …/Security: FAIL"; the two messages are in my entry of 1:13 AM PST |

- The nine reds of the old base in `verity-network-traces` are gone, as you expected.
- `378a` also built every file that imports the period check: `CoreOnsite`, `Properties.NetworkScan` and its two
  modules, `Proofs.Properties.NetworkScan` and its four [run: its three `build.log`s, each "Build completed
  successfully"].

## What I did not check

- Nothing ran at `13419dcffc`. It differs from the tested head by the lock's four values, the record `378a` kept
  there byte for byte; the lean-proofs skill asks no new run for that.
- The lock's result is `lean-fast/v1`: no kernel replay, no tests. No full check.
- `Security`'s lock: not recorded, and not green on this base.
- The skipped test of `verity-core-composition`: I did not look which.
- `main` is not merged; nothing was crossed.

## What I am least sure of

- **That the `trees` failure is the base's alone.** The two declarations it names and `Security`'s lock are the
  base's byte for byte, so I argue the base fails the same way; I ran nothing at the base. If lean's lock run at the
  base passed `Security`, I am wrong and should be told.
- That lean's merge takes the four values cleanly: they are the values a lock run regenerates, but I have not run
  `merge.py` against lean's tip.

## Two corrections

- My entry of Tue 10 Nov 11:07 PM PST headed its table "State at 11:08 PM PST"; the clock said 11:07 PM PST.
- Last night's note and handoff called `rowsTiedCharged`'s record in the base's lock stale. It is not: the definition
  it reads is.

Next: one comment on NET-325, then job 2's draft (NET-332) by inbox. Nothing on LEAN-249. Nothing opened or marked.

[2026-11-11T09:18Z] 789d4031edc3
# NET-332, by reading: the witnesses of `rowsTiedCharged`, `rowsProgWired`, `rowsStateTied`

LEANMERGE to the coordinator, Wed 11 Nov 1:18 AM PST. Job 2, after NET-325's end report.
Paths and lines are at `13419dcffc5a` (`cursor/net325-drawn-le-units-22f3`; its Lean is `0b12ac8cd69f`'s). For this
I wrote no lemma into the tree and changed no lock. Marks: [read] file and line, [vm] an elaboration by bare `lean`
on my VM (core only; the files are in notes, `tools/net325-leanmerge/net332/`), [argued].

## The answer in four lines

1. No honest witness of the rule's form can be proved for any period with an audit or a link-window while the check
   compares JSON with `==`: `BEq Json` is a `partial def` of the toolchain, opaque to the kernel and to every tactic.
   The only accept that could be proved today is of the period with no audit and no link-window, which shows
   nothing. [read] [vm]
2. The cure is four comparisons of `NetworkCheck/Period.lean` written as a match (the tree already did this for Spot,
   for this reason). It is a change to trusted code of this lane's own package, with no change of behaviour.
3. After it, one thing still can't be evaluated by the kernel: reading the answers' profile back
   (`Core.Integrity.ofJson`), through `Ratio.ofString` (`String.splitOn`, `String.all`, `String.toNat!`) and
   `Flock.canon` on an object (`Array.qsort`). That is the ONE field I would assume: the answers' `profile`.
4. Signatures, SHA-512, the pinned digests and the term's size do not stop it: the tree proves LMS completeness, and
   `wipeSigned_honest` is an accepted witness of exactly this shape with every digest and signature left symbolic.

## 1. What my two lemmas state now, and what each lacks

- `rowsTiedCharged_honest` states four closed facts about `chargedOf` and `faultyEntries` at a hand-built `Params`
  and `Commitment`, the audits' findings written by hand (`#[.ok ⟨2, 1, ⟨0, 1⟩, 1⟩]`, then `#[.ok ⟨2, 2, ⟨0, 1⟩, 0⟩]`),
  by `decide +kernel`. [read: `Security/Proofs/Properties/NetworkPeriod.lean:537-551`]
- It lacks all of the rule's honest form: it never applies `check`, builds no `Evidence` and states no accept. The
  rule holds an honest witness of a property to its function's acceptance (`shape.properties`:
  `NetworkCheck.Period.check`, outcome accept). [read: `tools/verity/lean/check.py:2236-2250`,
  `Security/Properties/lean-audit.json` `shape.properties`]
- Its instance is not one the check would accept either, for two reasons. The findings say 2 units and `slotAudit`
  has 1, which `auditCheck` refuses (`f.units == a.units`). [read: proofs file 146; `Period.lean:1178`] And
  `tiedWindow` has no state entry, which `stateCheck` refuses (`stateTied` is false with none). [read: proofs file
  141-142; `Period.lean:1101-1111, 1206-1208`]
- `rowsTiedCharged_nontrivial : ¬ RowsTiedChargedOf acceptAllRows` takes no binder and no hypothesis, is negated, and
  is at a verifier that accepts everything, named `acceptAll…` as the rule recognizes one. [read: proofs file 556;
  `Security/Properties/NetworkPeriod.lean:159-161, 186-187`]
- The rule's text asks `¬ RowsTiedChargedAt at acceptAll …`. `RowsTiedChargedAt` takes no verifier, so the `…Of` form
  is the only way to say it, and the rule reads it so: it passes a negated lemma whose head, or a definition the
  head unfolds through, is one of the guarantee's (`{head, *chain} & heads`), and `RowsTiedChargedOf acceptAllRows`
  unfolds to `RowsTiedChargedAt` past its binders. [read: `check.py:2392`; `tools/verity/lean/Facts.lean:911-927`]
  So it lacks nothing I can see in the form. Only a lock run with the entry written shows the rule takes it.
- Neither lemma is under `guarantees` or `witnesses`. The same holds of `rowsProgWired_honest` (157),
  `rowsStateTied_honest` (258) and their `_nontrivial`s (181, 273). [read: the lock]
- The property's own record did not move with NET-325 (its `type_hash` is the base's), so its `unwitnessed` entry
  stands as it is. [run: `r20261111-070615-378a`'s kept record]

## 2. The smallest period `d` the property says something of

[argued from the check's text; not built]

- `p`: one audit `⟨free, 0, "a", 2, 0, #[("SlotEntry_v1{KO=1}", 1, slotEntryPins[0]!)]⟩` (2 units, one slot entry),
  `form := .rows slotSchedule sd ⟨0, 0, 0, 0, 7⟩` (one lane, T = 1, 7 entries), `windows := 1`, `lms := Lms.deployed`,
  `key := k.publicKey`.
- `c`: the audit's registration (its `program`, its entries port at root `r` and domain `d`, and its
  `verity/certifier/scan-state/v1` value at `s` and `e`, as `stateReg "s"`), and the scan's (its `program`, a
  `scan/prog0` entry at `r` and `d` and a `scan/state` entry at `s` and `e`, as `stateWindow`, and two entries
  holding the ingress and egress rows' roots and domains).
- `e`: the audit's answer (one-stage's, the digest of its registration, accepted, published `{"bound": 1}`, a
  profile of one level of 2 units sampled `subset:1`), and one `Window`: the scan's answer, three certificates
  (requests, responses, fails) and `.rows #[ingress, egress]`.
- With T = 1 the scan has 2T = 2 units, so ONE profile serves both answers. [read: `Period.lean:1167, 1178`]
- At this `d` every clause of `RowsTiedChargedAt` bites: the audit is tied, sampled in part (1 < 2) with a bound of
  one, so `pub.charged` holds its facts and the link-window's term has `faultyEntries = 7`. [read: statement file
  146-155; my present lemma shows the last two on the model-level pieces]
- It needs five LMS signatures under one key (three certificates, two rows heads). [read: `Period.lean:1155-1160,
  990-992`]
- Nothing smaller is accepted with a rows link-window: the check refuses one whose scan's `prog` is no audit's
  output or whose state is no wired audit's (`progTied`, `wiredChecks`, `stateChecks`), so every accepted rows
  link-window has a tied audit and the state entries. [read: `Period.lean:1162-1164, 1198-1216`] The only smaller
  accepted period has no link-window. I do not list it as honest.

## 3. What stops building `d` at the model level and proving the accept

**`BEq Json`, first and decisive.**

- On the rows form's path the check compares with it at `Period.lean:1154` (`rowsWindowCheck`: the scan's `program`),
  `1175` (`auditCheck`: the audit's `program`) and `998` (`holds`: an entry's `root` and `domain`); the clock form at
  `945`. [read] (`571` and `613` are in the pins' digest, which is computed when an audit's entries are read from a
  Program's JSON, not by `check`. [read: `Period.lean:564-613, 674`])
- The toolchain defines it by `private partial def beq'`. [read: Lean v4.34.0 `Lean/Data/Json/Basic.lean:191-210`]
- `(Json.str "P" == Json.str "P") = true` fails under `decide`, `decide +kernel`, `rfl`, `simp` and
  `beq_self_eq_true` (no `ReflBEq Json`), while `#eval` says true; the same comparison by a match is proved by
  `decide` and by `decide +kernel`. [vm: `JsonEq.lean`]
- The tree says the same twice. [read: `verity/core/service/spot/Spot/Json.lean:62-64`, where `jsonStrIs` is the
  match; `Security/Proofs/Properties/Verity/SampledLog.lean:16`]
- So `check p c e = .accept _ _` is unprovable for any period with one audit or one link-window, and no named
  assumption about a field of `d` cures it: the assumption would be about the toolchain's function.

**The profile's read-back, second.** `answered` accepts only if `Core.Integrity.ofJson prof = .ok i`.
[read: `Period.lean:922-932`]

- On a core-only copy of `Flock/Canon.lean`, `CoreProtocol/Ratio.lean` and `CoreProtocol/Profile.lean`
  (`Flock.Draw.workK` stubbed), `#eval` reads the profile above back (`true`), and `decide +kernel` is stuck.
  [vm: `Prof.lean`]
- What is stuck, one by one [vm: `Str.lean`, `Qs.lean`, `Bisect.lean`, `Bisect2.lean`]:

| Piece | Where | `decide +kernel` |
|---|---|---|
| `"1/100".splitOn "/"`, `"100".all Char.isDigit`, `"100".toNat!` | `Ratio.ofString`, `natOf` | stuck |
| `#["a"].qsort (· < ·)`, one element | `Flock.canon` on any object, so `exactly` | stuck |
| `Flock.J.arr`, `J.str`, `J.nat`, `J.bool`, `J.has` on literal objects | the field readers | passes |
| `(Flock.canon j).isOk` on a literal object | | passes |
| `Ratio.ofString` rewritten over `String.toList` | scratch only | passes |
| string interpolation of a `Nat`, `++`, `intercalate`, `canonStr` | | passes |

- `exactly` needs no evaluation when the JSON is `P.toJson` itself: both sides are one term, closed by
  `bne_self_eq_false`, given `Flock.canon (toJson a) = .ok _`, which `canon_mkObj`, `canon_str`, `canon_nat`,
  `canon_arr`, `canon_bool` and `canon_null` give. [read: `Security/Proofs/Properties/NetworkCertifier/Check/
  LinkWindow.lean:271-302`, `NetworkScan.lean:100`] [argued]
- `Ratio.ofString` on a literal is the piece with no route: no lemma in `Security/` reads a profile or a ratio back.
  [vm: `rg "Integrity\.ofJson|Ratio\.ofString" Security` finds none] The tree's own cure elsewhere is to read over
  characters. [read: `Period.lean:1014-1015`, `outputPort`'s docstring: "`String.splitOn` it doesn't"]

**What does not stop it.**

- The certifier's signatures: `lmsCorrect` (signing then verifying is `.ok true`, and `decode (encode sig) = .ok
  sig`), `lmsSigns` (a signature at every leaf) and `key_ok` are proved, at any parameters `Params.make` admits.
  [read: `Security/Proofs/Properties/NetworkCertifier/Device/Lms.lean:462, 479, 491`]
- SHA-512 under `decide`: nothing evaluates it. `wipeSigned_honest : ∃ seed K c k sig out prof, Lms.keygen
  Lms.deployed seed = .ok k ∧ WipeProtocol.protocol.check … = .accept out prof` is under `witnesses` and is proved
  with the key, the digest and the signature symbolic. [read: `Security/Proofs/Properties/Pous/WipeSigned.lean:
  577-603, 692`] `certificateMessage_ok`, `taggedSha512_of` and `taggedSha512_ok` are there for the messages.
  [read: `…/NetworkCertifier/Check/Run.lean:50`, `…/Hash/Identity.lean:308, 317`]
- The pinned digests: `wired` reads `entryPins` as literal strings an audit's `entries` name; it hashes nothing.
  [read: `Period.lean:1075-1081`; proofs file 146]
- The term's size: symbolic, so no 34,756-byte signature is ever a literal. [argued]

**Whether an accepted period is in the tree.**

- No `#guard` of the package applies `check`: they are of terms, facts, ties and port names. [read:
  `Period.lean:1523-1643`; `rg` over the package]
- No stored vector: `tests/fixtures/` holds `slot_entry.json` and `select_entry.json` only. [vm: `ls`]
- `period_case.py` builds periods as Python dicts under a toy key and the difftest runs them through the compiled
  executable, which parses a file. [read: `period_case.py:1583, 2222-2237`] The kernel can check none of that:
  `Lean.Json.parse` is `partial` too. [read: `Security/Proofs/Properties/NetworkScan.lean:114-116`]
- The scan's own accepted honest witness is at an empty instance (`scan_honest : Flock.canon .null = .ok "null" ∧
  Flattens default ⟨[], []⟩`). [read: `NetworkScan.lean:110`]

## 4. The ONE field I would assume, and what the witness then shows

- The field: the answers' `profile` (`Core.Answer.profile`, one JSON for both answers). The premise:
  `∃ j, Core.Integrity.ofJson j = .ok P`, at the literal profile `P` of item 2.
- It is true (`j := P.toJson`; `#eval` says so [vm]) and only unproved, so it is an `open` premise with the issue
  that discharges it, not a standing assumption. The rule takes it under `witnesses_assumed` once it is listed under
  `assumptions` or `open` and a listed guarantee takes it; the witness, listed under `guarantees`, does.
  [read: `check.py:2138-2160`] A new named `Prop` in the definitions tree is not mine to add.
- This holds only with the four comparisons cured. With the check as it is, two things can't be built, and the
  first is no field.
- What the witness then still shows, of a built period with one rows link-window and one tied audit sampled in
  part: the registrations name their Programs; five signatures verify under one deployed LMS key; each rows
  registration has the schedule's format, layout, domain and head; the scan's registration holds both rows; its
  `prog` is the audit's output; both answers are one-stage's for the registrations' digests; the facts read from `P`
  have the right units; the wiring, the state tie and the single δ hold; and `check` returns `.accept`.
- What it does not show: that some JSON reads as `P`. The compiled reader shows that in the suite.

## 5. Whether `rowsProgWired` and `rowsStateTied` take the same `d`

- Yes for `rowsProgWired`: its property is of the same function and the same accepted period, and `d`'s link-window
  is wired by construction. [read: statement file 86-88, 100-102]
- Yes for `rowsStateTied`: `d` carries the state entries of `stateReg "s"` and `stateWindow` already, since `check`
  refuses a rows link-window without them. [read: proofs file 243-259; `Period.lean:1101-1111, 1206-1216`]
- So one accept lemma, `check d = .accept pub prof`, serves three honest witnesses.

## 6. The work

- **`verity/core/protocols/network/lean/NetworkCheck/Period.lean`**: the comparisons at 945, 998, 1154 and 1175 by a
  match (a three-line `strIs`, or `Spot.jsonStrIs` moved under `Flock.J`). Same value on every input. [read: `beq'`'s
  `str` and catch-all cases] The Python mirror and the difftest do not change. [argued]
- **Two locks move with it**, as with NET-325: `Security/Properties` and `Security` both pin `windowCheck`, `holds`,
  `rowsWindowCheck` and `auditCheck` under `reads["NetworkCheck.Period"]`; in `Security` the reader is
  `Flock.SecurityProofs.RecursiveAuditC3` (`@proofs`). [run: `git grep` over `*lean-audit.json`]
- **`Security/Proofs/Properties/NetworkPeriod.lean`**: the existing accept-to-facts proofs split those conditions as
  Booleans; a few `simp` lines may need the new name. [argued: not tried]
- **A new proofs file** (the present one is 141 s to elaborate): about 15 to 20 lemmas. One `certifiesOn` lemma from
  `lmsSigns` and `lmsCorrect`, used five times; `rowsDomain` and `headDigest` return `.ok` (one new lemma for
  `identityDigestSha512`); `rowsCheck`, `holds`, `answered`, `factsOf`, `rowsWindowCheck` and `auditCheck` at `d`;
  the wiring and state facts again, now for a registration with two unevaluated domain strings; the accept; three
  `_honest` theorems of the form `∃ seed k …, Lms.keygen Lms.deployed seed = .ok k ∧ check … = .accept pub prof`.
- **The lock** (lean's worker): three `witnesses` entries, six names under `guarantees`, three fewer under
  `unwitnessed`, one `open` premise with `witnesses_assumed`.
- **To drop the assumption**: `CoreProtocol/Ratio.lean`'s `natOf` and `Ratio.ofString` over `String.toList` (my
  scratch version passes `decide +kernel` on `"1/100"` [vm: `ProfList.lean`]) and three or four read-back lemmas. That
  file is the shared interface's, not this lane's.
- **Risks.** (a) Unfolding `check` at a period with symbolic keys, signatures and digests: `wipeSigned_honest` does it
  for one signature and a flat check; this one runs through `zip`, `zipIdx.map`, a `for` over two rows and
  `firstFinding`. (b) The wiring facts are `decide +kernel` today and can't be with free strings. In small: a field of a
  `Json.mkObj` at a variable string reads by `rfl`, and an `Array.any` over two entries, one at variables, closes by
  `simp` with those facts, on the three axioms [vm: `Sym.lean`]; `wired` and `stateTied` themselves I did not try. (c) Elaboration time. (d) I did not read
  `Flock.FV3.Sha512.Domain.make`, `wiredCheck`, `rowsPublic` or `Certificate.signs`' message encoder for a condition
  T = 1 or an empty link name fails.

## What I did not do, and what I am least sure of

- I built nothing: no lemma of item 6 exists, and the accept of `d` is argued from the check's text.
- Least sure: that nothing else on `check`'s path is opaque to the kernel. I tested the profile reader and the JSON
  equality, and grepped for `partial def` under the network, interface, primitives and Flock packages (none on the
  path). I did not test `Flock.FV3.Sha512.Domain.make`, the encoders (`uBE`, `text`) or `Lms.Signature.decode`
  beyond what the proved lemmas cover.
- The scratch profile reader is a copy with `Flock.Draw.workK` stubbed, not the tree's module.

