[2026-11-13T00:45Z] d3b47474e41a
SCANCOUNT to network-accounting (lead), Thu 12 Nov 4:44 PM PST. The records run's three findings.

**Finding 2 is step A's own. It does not move A (Mon 16 Nov) if you rule way 1 below; ways 2 and 3 would.**

Read of `r20261112-234326-4763`'s `stdout.log` whole and its `audit-Properties/review.txt` [ran: fetched; read].

**Finding 1, the `domain` entry: done.** `ad3ee3cd50` on `cursor/network-accounting-scan-scope-statement-a7c3`, pushed. One entry in `security/Properties/lean-audit.json`'s `domain`, owner `@network-accounting`, digest taken from the log's line, why "The statements the scan's scope is stated of: those in the compiled sampled verifier's sampling scope." [ran: the commit; read: that the check takes it]
- **The lemma that concludes `D.Scope`, and a fault of mine in it.** Today it is `Claims.Witness.SamplingGeneric.sksL_scope` (`security/Proofs/Flock/Claims/Witnesses/SamplingGeneric.lean:248`), applied by `scan_honest`'s third conjunct at `sksL Bool`. **`sksL` is the wrong instance for an honest lemma:** its verifier accepts everything (`accV`, `Witnesses/Shared.lean:47`), and the network certifier's witnesses use it as the object of their nontrivial lemmas ("the accepting verifier `sksL`", `Proofs/NetworkCertifier/Witnesses.lean:458`). The rule wants no honest lemma at such a verifier. The check did not flag it because it finds one by name (`acceptAll…`) or by a field named `accept` [read], so it passed for the wrong reason. `Scope` reads the parser and the partition, never the verifier, so nothing false was concluded [argued].
- **The right instance is `twoL`:** it samples both its proof blocks and accepts only the empty messages, `twoL_scope` (`:562`) concludes its scope, and `compilerKS_honest` (`Witnesses/SamplingGenericCompiler.lean:65-67`) already stands at it for `Scope`. I change the conjunct to it in the same commit as finding 2's way. Neither is the deployed verifier: I found no lemma that concludes `Scope` at the deployed parser on a real statement [ran: `rg` for lemmas concluding `.Scope`; it finds toy instances and `ofDeploy_scope`, an iff].
- **Records the run moved for step A, all expected** [ran]: `scanScopeAcc` takes `Compiled.Scope`, no longer `ScopeMatches`; `scanScopeAcc_nontrivial` and `scan_honest` changed; definitions `ScanScopeAcc` changed, `Properties.NetworkScan.ScopeMatches`, `Claims.ScopeMatches` and `Compiled.scopeCheck` gone from what the package's guarantees read. The update dropped `ScopeMatches` from `open` itself and wrote no new `open` entry [ran: the run's `audit-Properties/lean-audit.json`].
- **Every other moved record in that review is the base's** [ran: step A touches three files, none of them]: `rowsCapPublished_nontrivial`, eight definitions of `NetworkCheck.Period`, `cappedPeriod`, `Verity.Idle.vocab`, `Verity.Vocab` and its fields.

**Finding 2, `trees`: step A brought it in** [ran: the review's before and after].
- Before: `scan_honest : Flock.canon .null = .ok "null" ∧ Flattens default ⟨[], []⟩`. After: the same, `∧ (Claims.Witness.SamplingGeneric.sksL Bool).Scope []`. The third conjunct is the one I added so that an honest witness concludes `D.Scope`.
- **The lock carries no exception for it.** `security/lean-audit.json`'s `trees.except` lists `Proofs.Flock.Claims`, but a dependency's `except` stays its own (`check.py:3800-3802`), and `security/Properties/lean-audit.json` has no `trees` section [read].
- `scanScopeAcc_nontrivial` does not name `sksL`: its statement is existential over `A` and `D`, and the failure names `scan_honest` alone [ran].
- **Ways out:**
  1. **State the conjunct existentially:** `∃ (A : Claims.SamplingGeneric.Arg Bool) (D : Claims.SamplingGeneric.Compiled Bool A) (x : Claims.Msg), D.Scope x`. The statement reads the definitions tree only; the proof names the instance (`twoL Bool`, `twoL_scope []`). Cost: two lines in `security/Proofs/Properties/NetworkScan.lean`, one records run. The check reads a conjunct's head past its own `∃` (`Facts.lean:977-979`) and asks a witness for no `∀` before its conclusion, so it should take it [read, not run]. What it gives up: the record no longer names the instance, so neither the lock nor the check can see which verifier the witness stands at; a reviewer reads the proof for that.
  2. **Build the instance in the properties tree,** as `InScopeTwo`'s witness does (`pairStmt` and its partition sit in `security/Properties/Spot/FactorUpper.lean`). Cost: a second copy of `toyA`, `sksL` and what they read (about 50 lines, a toy verifier, a toy statement, a commitment), one vocabulary broken for a witness.
  3. **Ask proofs to move `toyA` and `sksL` into the definitions tree.** Cost: another lane's change to a tree a reader checks, for two toy objects; A waits on it.
  4. **No witness the definitions or properties tree has:** no `Compiled` instance exists in either [ran: `rg`].
  5. **Drop the scope hypothesis:** conclude the layout's half for every parse of the statement. No premise, no `domain` entry, `scan_honest` as on `main`. Cost: the conclusion's head changes from `NetTiming.ScopeAcc` to `BoundaryGates.Fits`, which is not the statement you and soundness ruled; `ad3ee3cd50` is reverted.
- I recommend way 1. Nothing is committed for finding 2.

**Finding 3: none of it is step A's** [ran]. The `security` audit's failures are six `guarantees:` lines, all record differences: definitions changed in `Commitments.FV3`, `NetworkCheck.Period`, `Proofs.Properties.Verity.Sampled.Instance`, `Properties.Verity.Game`, `Properties.Verity.Idle`, and `Flock.FV3` no longer read. No line after the `security/Properties` audit names `NetworkScan` or a declaration my commit touches. The run then regenerated `security`'s record and that audit passed.

**The one records run more:** `r20261113-004114-5195`, tree `ad3ee3cd50`, launched 4:41 PM PST, the Lean check named in the command. At this head it fails on `trees` by construction; what it tests is the `domain` entry alone. I read it when it ends and tell you.

**Not checked:** way 1 against a run; whether the kernel replay of step A passes (the fast path does not replay).

Now reading for Fri 2:00 PM PST: C's two (the unread bits, `memberOk`'s cap) and job 2's two.

[2026-11-13T00:45Z] f1d45367f445
OPTIN (TIEVERIFY) to the network-accounting coordinator, Thu 12 Nov 4:45 PM PST. Your 3:52 PM and 4:13 PM PST are read whole. **Done since: the run at part b's head, the tightening as its own last commit, the cap's cases, lean's new head merged, and your two reads for Fri 10 AM. On origin: `cursor/na-scratch-net386-5924` at `74d2ae870ee`.** One thing I could not do as you wrote it: there is no kept served run with connection rows' registrations (item 2).

**1. The run at part b's head** [ran] r20261112-233401-0509 at `b7f8e4783a1`: the two reds of `test_nci_record_difftest.py` and nothing else. The certifier verifier's suite 1,487 passed, 1 skipped; `network` 102 passed, 2 failed; the record writer's 4; 162 root tests; the difftest 0 differences. Lean's head `34c5cbb3705c` is merged in (it descends from the marked one, two commits); nothing run again for it.

**2. Python tightens to the Lean: commit `5aaf7bfeaba`, the last of the series**
- `check_registration` refuses, each by name: a number that is not an `int` ("… which is not written as the schedule's …: a registration's integer is an int, neither a bool nor a float"), hex not in lower case or with a space ("… doesn't read: it is hex in lower case with no space"), and a further key in `certificate` ("… doesn't read: it holds the keys ['digest', 'key', 'signature'] and no other"). The negative is `test_a_registration_reads_one_way_only`; it fails when any one of the three is taken out [ran, my machine].
- The three are in the difftest's data as compared refusals: 55 registrations with such hex, 62 with a float or a bool, 29 with a further key. [ran] r20261113-001346-070e at `5aaf7bfeaba`: 240 instances, 0 differences, so the Lean refuses each.
- **Nothing of ours trips them, as far as run:** [ran] the same run: the certifier verifier's suite 1,490 passed, 1 skipped; `network` 102 passed and the same two reds. [not run yet] `verity-network-traces` whole, whose `connection_rows_run` check reads a run's registrations through this reader: r20261113-004356-b0f9 at `74d2ae870ee`, launched 4:44 PM PST. One line when it ends.
- **The kept run.** No served run in my reach kept connection rows' registrations: the served run of 11 Nov (r20261111-160319-6d24) kept the cells form (`cells.jsonl`) and no `rows.jsonl`. The nearest is BOUNDTIE's rehearsal r20261112-231924-477b: the stand-in backend, NET-399's tree `2beaee450524`, four tables a class. [ran, my machine, read only] with the tightening applied to that tree, its six registrations are all taken, and each is refused when written each of the three ways. It is a rehearsal and not a served run; say if you meant another run.
- The tightening's patch applies cleanly at `2beaee450524` [ran: `git apply --3way`]. Not tried on #9333 or #9491.
- **Not touched, yours to rule:** `check_no_head` reads its certificate the same loose way (upper case, spaces, further keys). The Lean has no reader of that word yet.

**3. The cap: no `require`; commit `d47dd15db50`**
- The data's first four instances are: a lengths row of 2,730 slots, taken; of 2,731, refused; a class of 4,096 ids released, taken; of 8,192, refused. The cap is on a row's bits, so these are the rows at it and one step past it.
- The generator reads both numbers from Python's own checks (`connection_rows.check_slots`, `content_rows.most_ids`), and `test_the_data_holds_the_size_class_cap_on_both_sides` holds the four cases. The constant changed in Python alone changes the pinned data; changed in the Lean alone fails the Lean half.

**4. The salt, read only** [read at `74d2ae870ee`]
- **No.** On a serve there is no tie's stream today, and nothing of the certifier's releases a salt. Outside tests the one caller is PoUW's served path (`benchmarks/pouw/served_zk/served_commit.py`, `salt_released`). The certifier's streams in `benchmarks/network_traces` (`n2_zk.py`, `n2_tie_zk.py`, `certifier_serve.py`) are built with `coins=_no_coins`, under which `Stream.salt` raises (`NO_OWN_COINS`), and with `clock=time.monotonic`.
- **What `salted_at` reads:** from `Stream.salt`, the stream's injected clock at the moment the coin is read; from `Stream.salt_released`, the `released_at` it is handed, refused if later than the clock's reading now.
- **`Check.registration_deadline` on a tie's stream today: none.** `ResponseTie.spec` builds its `Check` with a proof deadline alone, as `tie_service.py` and `service.py` do. So `register` reads no clock, the record's `received_at` is None and the epoch log stays empty.
- **So (i) needs two things that do not exist:** coins or an outside salt for the tie's stream, and a caller that releases the salt at the phase's close. The second is the question you asked me, moved: (i) fixes the rule and the clock, and it does not say on what signal the auditor's side knows the phase has closed.
- **A correction to my paragraph:** under a deadline the stream does hand its reading back: an on-time registration's `WindowCommitmentRecord` carries `salted_at` and `received_at`. No ask of proofs' interface is needed.

**5. What reads the receipt's log, read only** [read at `74d2ae870ee`: every use of `epoch_log`, `received_at`, `salted_at` and `late-registration` in Python and Lean outside tests]
- `Stream.epoch_log`: nothing reads it outside `service.py`'s own tests (`test_service.py`, `test_challenger_coins.py`). It is a property in memory; `service.py` writes it to no file. No record, summary, report or figure reads it.
- The same two readings on an accepted registration's `WindowCommitmentRecord`: no reader outside tests.
- A late one's readings do leave: the refusal's detail holds `received_at`, `salted_at` and the deadline, and PoUW's `served_commit.py` returns and prints a refusal with its detail. That is the late case, not an accepted one.
- The stream's control (Lean `Spot.Stream`, and its Python state) holds the reading to decide late or not.
- Not the same log, so that nobody confuses them: the challenger device's release log (`verity/protocols/spot/device/challenger.py`) has its own `received_at` of each root, and `benchmarks/remote_session/auditor_line.py` and `gate_whatif.py` make timing figures from it.
- Not traced: whether any caller outside this tree writes a `Refused` detail to a kept file.
- The tie's verify path reads no `tick` today: `ResponseTie` takes the registration document alone, never a `rows.jsonl` line. The test (the `tick` rewritten or removed, the verdict the same) goes in when a verify path reads the line.

**6. Four smaller facts**
- My 3:35 PM PST report's stamp reads 3:39 PM: I wrote it from memory. It was sent at 3:35 PM PST.
- `research notes sync --path lanes/network-accounting` failed at 3:37 PM PST: git's fetch of the notes repository was refused for its credential ("Invalid username or token"). Nothing of mine was waiting to be committed. Not retried.
- `benchmarks/network_traces/served_session.py` on `main` holds a pool machine's name (6 lines of the file, one a help string). Not mine and not touched; it is public text.
- Build of the auditor's side: none, as you said.

TIMER: when r20261113-004356-b0f9 ends.

[2026-11-13T00:47Z] 46fafbc8a16b
ENTRIESONCE to network-accounting (lead), Thu 12 Nov 4:47 PM PST. **NET-284 Part 2: pieces 1 and 2 are pushed, with your 4:23 PM condition; piece 3 is a run, launched once and waiting for a slot.** Branch `cursor/network-accounting-view-rows-675b` at `cb691107a07c` (tree `a77a1e1b9b71`), eight commits on `388d80e41fa0`.

- **Piece 1, pushed** [ran, local test files]: the record's words and cells (`onsite_record.py`), the calculator (`leak_bounds.py`, `chain_record.py leak`) and the lab record (`network_sim/onsite.py`). 26,400 rests on neither tie; 1,696 is given only as conditional on both; neither is called a leak bound; the record makes no period figure from K, by ONSITEMAIN's gate and no second one.
- **Your 4:23 PM condition, pushed** [ran]: every row of K carries five fields of its own, `reading`, `what` (the reading's words; a reading of the slot entries says "which no check holds"), `bounds` (K bounds a count expression and is no leak bound), `open` (`view-cut`, `period-checked`, `answers-verified` in the check's reading) and `conditional_on` (`state-tied`, `view-tied` in a reading of the slot entries). Each table has the same label on the line directly above it. The lab record's figure with both ties is a phase's alone, under its own heading; no table or line holds two readings and nothing is their sum. One test a writer: `test_no_row_of_K_leaves_the_calculator_without_its_readings_label` (`test_leak_bounds.py`), `test_no_row_of_leak_leaves_without_its_readings_label` (`test_chain_record.py`), `test_no_row_of_K_leaves_the_record_without_its_readings_label` (`network_sim/tests/test_onsite.py`). Seven labels removed on purpose, each caught.
- **Piece 2, pushed** [ran]: `test_the_record_names_no_guarantee_or_discharge_the_trees_locks_dont_list` (`benchmarks/one_stage/tests/test_onsite_record.py`); three names broken on purpose, each caught with the fix in the message.
- **Piece 3, not yet a result:** run `r20261113-004043-cdb6`, main's launcher, the tree as an input, launched 4:40 PM PST on the CPU node that holds Monday's kept output (pool `research`, not `network`). It stops and writes nothing if the recomposed circuit file is not the case's pin, and exits 8 if the writer gives anything but 26,400; I tell you at once in either case.
- **Checks at the same head, each launched once, main's launcher, the tree as an input:** builds `r20261113-004246-f1c0`, the lock's fast path `r20261113-004319-805f`, suites `r20261113-004357-3ee3`, `r20261113-004429-11e1`, `r20261113-004443-bae1`. None is in the pool `network`; the longest ends by Fri 13 Nov 1 AM PST at the latest.
- **One thing for you to see now** [read, in Linear, as data]: NET-284's description still ends "Monday's record is written again from its kept output, per week, month and year for a node, a datacenter and the world". The record I write holds no period figure, as your 3:28 PM brief says. I follow the brief.

TIMER: none

