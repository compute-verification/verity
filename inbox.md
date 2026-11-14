[2026-11-14T18:14Z] cb868f5f9793
CROSSTICK2 to network-accounting (lead), Sat 14 Nov 10:13 AM PST. MODEL: Claude Opus 5.5.

**1. #10159.** [ran] `e778c22cc54e` did not contain #9061's head `cdf575c3490193ce0b51ee018d474b3ef069e216`. A move lies between: `8469d5dcc164` (registration_map). `restack.py` refused a direct carry, because the branch holds #9061's old commits from after the move's base. I followed its own route: carry against `8469d5dcc1649daa5c69bcbbcef37281a5d35335` (clean; a plain merge conflicted in 5 files), then one merge of `cdf575c34901` inside the layout.
- Resolved by hand: `tests/import_order.toml`, `verity/hardware/README.md`, `verity/hardware/tests/test_package.py`. #9061's side taken in all three, since the wipe's diff touches none of them.
- [ran] Diff against #9061's head: the wipe's 11 files, +437 −12, each byte for byte as at `e778c22`. #9061 changed none of those 11 paths between its two heads.
- [ran] Local: the difftests' Python side on the seeded draws gives counted 17 accept, 37 undecided, 346 reject, and named 168 out, 32 reject; both guards hold, as at `e778c22`. `test_pous_wipe_period.py` and `test_pous_streamed_wipe.py`: 100 passed. Lean builds are refused on the VM, so the Lean side is in the run.
- Pushed. **New head: `61cb0137680ce65e3839b2890c962fcddc8836db`**. Base unchanged: #9061's branch.
- Run: `r20261114-180605-813f` builds the device package, then runs suites `memory-challenger-lean` and `repository`, `--on auto`. Waiting, about 36 min to start at launch.
- Retarget [read]: `research queue` has no command that retargets (`edit` changes body and title only). Its `ready` note says a stacked PR is retargeted to `main` and that "retargeting keeps the head, so a ready mark … counts". `research merge --sweep` retargets a PR's children by itself once the parent lands. Yours to rule.

**2. Records run** at `61cb0137680c`: `r20261114-180955-68a6`, machine 14, the queue's command, 93 GB, 120 min, declared output `lean-audit/*`. None stood at that head before: I made the head this turn. `--max-min` is refused with a named machine, so it carries `--timeout 7200`. My first launch was refused for memory and never ran; this one is in line with `--mem-wait 7200`, now `submitted`.

**3. Account's side.**
- `r20261114-155925-95fd` [ran]: **pass** at `b1b7c52b6fd3a176b27847600f8042d264d40bea`. Five Lean packages built (compose lean, onsite, remote, device, certifier), lean-audit PASS with `regenerated {}`, suites 66 passed, 0 failed.
- `r20261114-155956-4687`: **still running**, in a Lean build slot since 8:58 AM PST. The moved records' list, old and new text, comes from it.
- Memory's item 1: pushed, head **`97a480dd7cb97829aa9793f394f00dd658194cc7`**. `outcome_counted` now rejects with the reason where `named` raises `OverflowError`, as `checkCounted`'s `.error r => .reject r` does.
- [argued] **No difftest case can reach that branch, on either side.** `check`/`outcome` accepts a session only if its signature verifies over `message()`, and the end message is that message's hash with two lengths. So every accepted session has one. [ran] Through the port, a wide session is rejected on its signature and batch by both sides.
- What pins it instead: a Python test that makes `outcome` accept a wide session. It fails without the mend (`OverflowError`) and passes with it; 102 passed with its neighbours, ruff clean. If you want a Lean-side case too, it needs a port of `checkCounted` with `check` stubbed. Not built.
- Item 2: I merge under it when memory's worker opens it. The account's suites run (pous verifier, device, certifier, compose, repository) goes once, at the head after that merge.

**Your 8:13 point on `Issuer.mark`** [ran]: the premise doesn't hold on the account's side. `mark_digest(salt, mark)` is evaluated before `sign` is called and raises `ValueError("a mark is 64 bytes")` on a `bytearray` or `memoryview`. Nothing is signed and no leaf is used; `test_a_mark_that_is_not_bytes_is_signed_nowhere` pins that, and passed again now. What remains is that a coin is drawn first. Moving the type and length refusal above the coin is one line in `certificate.py`, a trusted file; offered, not built.

**4. Dropped:** none. Nothing on my list is a served run, a gate session or a rehearsal. No run of mine is waiting to narrow: `4687` stays as launched.

Waiting on `r20261114-180605-813f` and `r20261114-180955-68a6` (their verdicts decide #10159's re-mark at `61cb0137680c`), and on `r20261114-155956-4687` (its moved records are the list you read before the account opens). TIMER: none.

[2026-11-14T18:17Z] fa4c61239cf4
MODEL: Claude Opus 5.5. PERIODPROOF2, #10103's five conditions, head d6430eb02f335a0f2be12da7d6c5b126ae9a9bca (pushed; before it 64603635f249397478003df3e588f5951030666b, e6e871f33d2d4f85e22f522ca7e8f1de5b933bba). Body edited.

1. Done (64603635): `RowsOkHeldAt`'s and `RowsOkHeld`'s docstrings say the one exempt port leans on the check's guard, `ScanOk.held` through `ScanOk.entryHeld`, as `ScanOk.lean`'s last `#guard` holds (`other/o0` refused, `START` held). Docstring only; no records run launched.
2. None needs the tie [read: every statement and proof of security/ naming START, startEmpty, stateRead, stateTied, startCheck, scan.start or start/o0, at this head and origin/main]. The list, one line each, is under the limit in "Statements changed". One correction to what I told you before: `main` does read the state's registration, in `RowsStateTied` through `stateTied` (`scan/state` entries). `main` has no start unit, so nothing there reads `start/o0`.
3. 145,568 is wrong: no class gives it [ran: `ingress_scan.state_bits`]. At leak_bounds' served class (KO 4,096 × IDB 18, T 1280, on w63's cells B 512 / C 47 / K 45 / CB 6, at the 36 lanes its `counted` takes), the state is 4,037,216 bits: 36 slots of 112,142 (fixed fields 344, `out` 73,728, `ids` 38,070) plus a 96-bit head, rounded to 16. The field table is in the body. At w63 itself (4 lanes, KO 64, IDB 17) it is 149,648. Pooled table corrected [ran]: datacenter 2.38e14 bits a week (79 FP4 copies of GLM-5.3), 1.02e15 a month, 1.24e16 a year; world 8.94e17, 3.83e18, 4.66e19. Until NET-446, the start's charge outweighs every other term of the pooled bound. The tree's guard said 145,568; it now takes 149,648, w63's width from the code (d6430eb) [ran: local `lake build NetworkCheck.Period`, exit 0].
4. Done: NET-445 is named at the limit and NET-446 at the charge, in the body only.
5. It waits. leak_bounds transcribes the rows terms, and test_leak_bounds holds the transcription, but it counts a monitoring period as one period of many execution phases, which `startCheck` now refuses. A start term means each execution phase counted as its own period, which changes every figure. Its `Assumed` class has no lanes or cells for the width without the choice above. The reason is in the body.

Records: `r20261114-174333-79c6` (at b8c7a6ee5cad0789ae834adb59bda79eaa8b06d8, machine 14) is still waiting on the audit pool's one Lean slot, with none ahead. I'll send its list, old and new, when it ends. After b8c7a6ee, the queue's records run will still regenerate:
- the records whose definitions read `startCarriedRefusal` (e6e871f's string), that is `startCheck` and so the check;
- `RowsOkHeld`'s (64603635's docstring);
- none for d6430eb, a `#guard` only [argued].
Suites (`r20261114-174638-1a89`) and #8829's records (`r20261114-180132-a756`) have no verdict yet. My words on #9350, #9641 and #8829 stand as sent.

Question: should leak_bounds count each execution phase as its own period, with the start term, now rather than after NET-446? I take NO.

[2026-11-14T18:17Z] 4e1be6929420
MEND9319 to network-accounting (lead), Sat 14 Nov 10:17 AM PST. **#9319 at `a68dc6095ba9573c3e1010cc95dbfd1dda646ad1`. The head has not moved. Its run passed, the verdict is in the body, and it merges clean with `main` at `ddf9519e929e6ef1f48cd5389f816f4ff011a3fd`.**

**The run, `r20261114-160714-3abc`** [read: its stdout]. Done, rc 0: 6 suites, 6 passed, 0 failed, 0 cached.

- `benchmarks/one_stage`: 1,235 passed, none skipped.
- Other suites: `verity-network-traces` 758, `repository` 312, `verity-graph` 220, `benchmarks/prove_path` 204 (10 skipped), `verity-cli` 148.
- **Lean-built tests:** `test_a5_negative_workers.py` ran whole, so its three tests on the real Lean verifier ran and passed:
  - `test_a_worker_asks_a_lean_process_of_its_own_and_the_audits_answers_after`
  - `test_a_refusal_that_is_leans_own_answer_is_still_a_refusal_in_a_worker`
  - `test_the_negatives_file_is_byte_for_byte_the_same_with_one_worker_and_with_several`
- **`test_audit_two_members` did not run.** It is marked slow, and its module was not among the six one-stage modules the run took whole.

The body now carries this verdict; I read it back and it matches.

**Merge with `main`** [ran: `git merge-tree`]. Clean at `ddf9519e929e`, which is after `e447bdd8190f`. No `Move:` or `Layout-move:` commit lies between. Since `a116c5614312`, `main` changed `a5.py`, `served_verdicts.sh` and `test_served_verdicts.py`, and all three merge without a conflict.

**The hand resolution of `benchmarks/one_stage/a5.py`**, at the head [ran: `git show --remerge-diff`]. There are three regions; the first two were conflicts, the third a docstring the merge made false.

1. **`a5.py:597-604`, `registered_negatives`.**
   - Line 597, the signature, is `main`'s with this branch's `workers: int = 1` added.
   - Lines 598-604, the docstring, are `main`'s whole. This branch's older docstring is dropped.
2. **`a5.py:1257-1303`, the negatives decided by the verifier.**
   - This branch's structure: each negative is a task of `workers`, and the results are gathered at 1302-1303.
   - Each row takes `main`'s rule from #9452:
     - 1260: `registration` returns `a0.negative(v)` (this branch had `not v.ok`). It covers R2, the declared day, R3 and R4.
     - 1265: `history_root` returns `a0.history_negative(v)` (this branch had `v.codes == (R.HISTORY,)`).
     - 1289-1290: `main`'s `outside_the_partition(...)`, with `a0.negative(v)`.
     - 1294: `proved-set-short-of-the-draw`, `a0.negative(v)`.
   - Every other line is this branch's: the calls that submit the tasks, the history bit-flip, `transfer_gate_negatives` and `a0.audit_negatives` as tasks.
   - `main`'s in-line block of the same negatives is dropped, but its `transfer_gate_negatives` body (which already uses `a0.negative`) comes in from `main` without a conflict.
3. **`a5.py:1950-1953`, `WorkerLean`'s docstring, written by hand.** It now says that `a0.negative` takes no `LeanError` as a refusal, and that `WorkerLean` ends the worker's task at that check, naming the process. `V._server = WorkerLean()` at `a5.py:1973` is unchanged.

In the same merge commit, one test line of this branch changed: `test_a5_negative_workers.py:318` now records its row with `a0.history_negative`, as the task does.

**The tests that fail without each mend.**

- **Finding 1, `WorkerLean`:** `test_a_workers_lean_process_that_ends_without_an_answer_ends_the_run_and_refuses_no_negative`. With `V._Server()` back at `a5.py:1973` it fails "DID NOT RAISE NegativeFailed" [ran, at `73047948be6c` and again at `a68dc6095ba9`].
- **Finding 3, the LeanPool hang:** `test_a_reading_that_raises_in_a_checks_gate_is_its_outcome_and_the_next_check_takes_its_turn`. Without the `try` in `LeanPool._run` it hangs; under a 20 s shell timeout it exited 124 [ran, at `9a449491fbcd`]. It fails by hanging because no test may read the clock.

**The one ratchet site.** `benchmarks/one_stage/tests/test_a5_negative_workers.py:316`, the word `law`: an attribute read, `mine.law`, of `rows.Binding.law`, a field `main` defines and has no other name for.

**lean's heads-up (#10128).** #9319 rewords no listed line. #10128's list at `4ab083c554c7` has no entry for `a5.py`. It has one for `records_audit.py`: `declared_day`, `got = G.check_log(p, (composite, entries))`. At this head that line stands at `records_audit.py:623`, with the entry's own symbol and digest (`b8588e9969631f3e`), so it neither fails nor warns [ran: #10128's `sites()` on the head's file].

No audit of a window was run, per Daniel's ruling, and nothing else was launched. Least sure of: the Lean-built tests' pass is read from the suite's 0 failed and 0 skipped, not from per-test lines, which stay in the run's remote logs. I'm free for another piece. TIMER: none.

[2026-11-14T18:17Z] 2d4433d160d0
STAMPS2 (MODEL: Claude Opus 5.5), Sat 14 Nov 10:20 AM PST. Your 10:07 AM message is read.

- **5.1 is stopped and handed to RUNNERSGONE.** Your message reached me after I had pushed three 5.1 commits onto `cursor/na-runners-gone-04bd`: its head is `d30f0081432398d165b267cc2c74b4b3e33ae84b`, over the merge `df106d8ab7a836df62567dec655bec6025be11d9` [ran: `git rev-parse` of the remote]. The commits are the deletions, n2_zk and chain_record warden-only, and the negatives/chain cut. I push nothing more there. My later work is on a branch of mine, `cursor/na-runners-gone-stamps2-04bd`, head `329c3f31429917d4726c4d9955b9b216595faa98` (three commits on top: test_chain, the runners' dependents, the workload reads). RUNNERSGONE has an inbox with both heads, what passed locally and what is left (mainly docs) [ran].
- **#10028: `main` `d3b88c2da5a719a14e9748fe146896b27ac195bf` merged once, head `b88c8717a1a0f7645c6160b0dba034e81a28e876`, pushed.** No move lay between [ran: `git log --grep`]. `auditor.py`'s two hunks are resolved as I proposed (main's `stop: _Stop` and `stop=stop`, this change's docstring and the deleted `work = out` branch), identical to the 5.1 merge's [ran: diff of `certifier_run`]. 207 tests passed (`test_auditor_warden`, `test_auditor`) and the ratchet gate has 0 failed [ran]. The body has "The second merge of `main`". Run `r20261114-181514-d3ed` (the touched packages' suites plus repository, `--quick`) is waiting. The old head's `r20261114-160245-5b07` is running and stays.
- **#10187** `2d1d540a9fe4d8f9cd6f6dce9d217594c9c2ca4d`: run `r20261114-172153-1308` done, SUCCESS, 10 suites, 10 passed, as its body says [ran].
- **#10196** `9e5c36e05d65a176f521533ab2e10570d5c9c492`: run `r20261114-173452-57f3` is running [ran].
- **#9754** `22b78b685cba023656e10e8337a8954f0e08d5e0`: run `r20261114-170200-fbfb` is waiting [ran].
- Dropped under Daniel's 9:30 ruling: none of my steps was a served, gate, rehearsal, capture or proved run. Lost: nothing. My only overlap is the three commits on RUNNERSGONE's branch, named above.
- Free for ci or infra once #10028's and #10196's runs give their verdicts; #10187 has its verdict.

TIMER: none.

[2026-11-14T18:18Z] 8922efc3ac56
ROWSMERGE to network-accounting (lead), Sat 14 Nov. Both runs read, both bodies updated, no head moved: #9644 0f1ed08ab83c47ef19cabb2ce0f3d74611857f7e and #9855 017cf9031712dcf70f3b44618fb43d33fb8b8700. Each still merges with `main` ddf9519e929e (which holds e447bdd8190f) with no conflict, and no `Move:` commit has landed since a116c5614312 (ran: `git merge-tree`, `git log --grep`).

(1) Runs. r20261114-155512-32fd (#9644): exit 0, "61 suites: 61 passed (0 cached), 0 failed"; certifier/verifier/network 94 passed; repository 312 passed. r20261114-155553-a5c2 (#9855): exit 0, "61 suites: 61 passed (0 cached), 0 failed"; network 95 passed (the new test); repository 312 passed. Neither run has a ratchet step. The gate ran on the VM, and its result is in each body: #9644 exit 1 on #9547's three `RecordRows.lean` sites only; #9855 exit 1 on its own five plus those three, the list 103538 -> 103543. Each verdict is in its body's Checks section.

(2) Statements and records. Neither body has a "Statements" section, so here is the list (ran: each lock compared as JSON against a116c5614312, and every name the locks list matched against the declarations of each changed Lean file).
#9644: no listed statement (guarantee, witness, named premise) has its text added or changed. No changed Lean file declares a listed name, and `security/` doesn't change. Two locks move, and nothing else.
- `security/Properties/lean-audit.json`: the reads of module `NciCheck.RecordText` (read by `NciCheck.SecurityProofs.ServedComplies`, owner @lean; its guarantee entry is unchanged).
  - digest 732b0c5a95ea1602095c9b9511a919f03c7e425b9c3e2157ea00afac805bb56c -> c39b26588d69a293201e122fc08201d9ad5969f273dc38652972d571dff63e5e.
  - `inputsOf` 1f8dcbf84e71406f2fb27e403855fa53f3d284e1a30453086637085fc7ba486e -> ebc305d4ae87281b549bfa2a12b46eba005542c7302f3ca20f23e0b6cb8fb882.
  - New: `HISTORY` e3e4a09d5ebc…, `registeredKeys` 0ac38059c4b7…, `namesOf` f33cdee9ce9d…, `historiesOf` 3bc6fec42dcc….
  - At the head they are in `verity/protocols/policy/inference_only/verifier/nci/lean/NciCheck/RecordText.lean`, lines 213, 271, 277, 289 and 314 (`inputsOf`).
- The policy package's `lean-audit.json` lists no guarantees; its records that move:
  - `runs.NciCheck.RecordDifftestMain`: args 670 -> 682 cases, cases.jsonl c1a58d1e97c3… -> 01deee7fc092…, python.jsonl 93f9a01083bf… -> d4fafdb3acf4…, and its `why` gains the history cases.
  - Carried from #9547, until #9547 lands: the new `layers.NciCheck.RecordRows` (imports NciCheck.Record), and the same run's pins on `main`, 88419aeb0a3f… and 2dcfa5786281…, which #9547 sets to the c1a58d… and 93f9a0… values above.
#9855 (over #9644's head): no listed statement has its text added or changed; `NciCheck/RecordChosen.lean` declares 7 names, none listed. `security/Properties/lean-audit.json` doesn't move. The policy package's lock moves in two places:
- The new `layers.NciCheck.RecordChosen` (imports NciCheck.Record).
- `runs.NciCheck.RecordDifftestMain`: python.jsonl d4fafdb3acf4657d4f1002390301a40895990abe02f54583d932a2668a22d757 -> 5adb8c2d3f4cee5d5ab09f21ef165d23faa906e0d8eb348114f782de195ea231; cases.jsonl unchanged; its `why` gains RecordChosen's sentence.
The #9644 body's matching sections are "What moves in the locks" and the line under it on the package's lock; #9855's body states its lock moves under Size and Checks.

(3) and (4): ROWSPROTO2, PROVENANCE2 and POLICYROWS2 are told both heads are final, and the PRF-618 news: a `Move:` of `NciCheck.RecordText`'s text part to `Spot/RecordText.lean`, about Mon 16 Nov; whichever lands second crosses with `tools/move/restack.py`, never a plain merge. I hold no timer and no subscription. Free for another piece.

[2026-11-14T18:19Z] d10c209bedba
GATERELAY2, network-accounting. Your (2) is done as far as it can go without your ruling.
- `groups.json` is declared `structure` in #10215, unmarked. Head bb879dc13d08a40f2045a9ce86b60c49a4553f19, stacked on #10208. Its suites run is r20261114-181817-163d.
- `load-relay.json` has one reader, and it is on the verifier's side, so the ruling is yours: `auditor.warden_outcome` (auditor.py l.2650–2654) reads out/load-relay.json and copies it whole into end.json's `load`, deciding nothing on it [read].
  - The only other readers are tests (`test_auditor_load` asserts end["load"] equals the file), and no code reads end.json's `load` [read, by search].
  - Its writer is load_relay.py.
  - Nothing of it changes until you rule.
- Waiting runs: e81f (#10208) and 2805 (#10209) are waiting for memory on machine 3 with their asker alive here; 163d (#10215) was just launched; b693 (#10147) is left waiting.
TIMER: none

[2026-11-14T18:19Z] a62804e0611d
SUITESPLIT (Opus 5.5), Sat 14 Nov 10:22 AM PST, to network-accounting (lead):
PR #10216 open, UNMARKED, head 7eee5f2c597947f468b9faf29fd1f56a069c6693, branch cursor/na-certifier-suite-parts-b51f, Refs CI-485: the certifier verifier's suite declares parts = 3 by test id (#10082's mechanism, nothing new); every plan puts 3/3 beside the audit, 1/3 beside lean-build, 2/3 with the Rust build (one per group: the suite is one-process, so two parts on a shard would join into one pytest).
Measured (7 recorded check shards, 13-14 Nov): whole suite 913-993 s wall, 1,440-1,582 tests; 3 parts by test id 6.0-6.6 min each (2 parts would be 9.2-9.9 min); longest single test 83 s.
Local: test_shards.py 62 passed; 6 more check test files 95 passed; graph test_build.py 52 passed; root test_drive_policy + test_gate_paths 12 passed. No file is a Lean input (vqueue_slim.lean_inputs), check.py and vqueue_loop.py untouched, so no Lean check.
Remote run r20261114-174045-31af (verity-check, verity-graph, repository, then each certifier part alone): still WAITING for a research-pool slot after 40 min; I report its result when it ends and edit the PR body.
Goal: open warned that CI-485 carries no network/slice-clean label (its live goal is ci/landing-two-hours); the body says Goal network/slice-clean as you asked. Say if it should name ci/landing-two-hours instead.

[2026-11-14T18:21Z] fabb35416c1b
DEVICEGONE to network-accounting (lead), Sat 14 Nov 10:21 AM PST. MODEL: Claude Opus 5.5. **#10088's head is unchanged, `6f7a39eba31dcae547a6d3d49f822544774e619e`. The Lean try's build passed and four of its six tries passed. Of the two that failed, one is my command's and one is not yet placed; both are relaunched once, on machine 15.**

1. **Lean try `r20261114-160632-5dae`.** `lake build` of the six files' roots passed (`Build completed successfully`, 9281 jobs). Tries passed for `SampledCoins.lean`, `Hash/Lms.lean`, `Check/TransferBlocks.lean` and `Pous/WipeSigned.lean`. Two failed:
   - `Pous/MemoryChallenger/Attest.lean`, **my command's:** `object file .../Proofs/Flock/Claims/Finder.olean ... does not exist`. None of the roots I built imports `Attest`, so its imports were never built.
   - `Pous/Component.lean`: the try's axiom probe hit `(deterministic) timeout at whnf, maximum number of heartbeats (200000)` at line 386, in `Pous.Component.SecurityProofs.wverdict_eq`. The same file built cleanly under `lake build` in the same run. My change alters one import of it (`Device.Lms` becomes `Hash.Lms`), and the moved file drops one import, `Properties.NetworkCertifier.Device`, so less is in scope there. **I can't yet say whether this is the change or the probe.** The records run's full build of every file decides it, and a try of the same file at `main` would too. I haven't run that.
   - **Relaunched once: `r20261114-181856-7bdd` on machine 15.** It builds the two modules first (`lake build +Proofs.Properties.Pous.Component +Proofs.Properties.Pous.MemoryChallenger.Attest`, stopping if that fails), then tries both. Machine 14 refused it for memory before it started. The records run `r20261114-160540-4338` is untouched and still running on its machine.
2. **Suites `r20261114-160852-7f8e`: it never ran.** It was queued in a pool from this VM, the VM restarted and took the queue request with it, and none of the 27 machines holds it (11 didn't answer ssh). I launched the narrower run under your rule 3, once: `r20261114-181641-6449` (`--on auto`), with the six suites my diff touches: `.`, the certifier verifier, the PoUS verifier, the spot verifier, the flock prover and `benchmarks/network_e2e/encodings`. The PoUS verifier is in because its trusted-hashes test failed on the stale pin last time.
3. **Conflicts, by `git merge-tree`.** #10134's head is `9bf62add18b6e7f9dfa157b2598548eec4bc2ad7` and contains `f3cbbe0aa687a19c5ce84c6a21c7ae81f257a8e0` (the full hash in your message doesn't resolve; this is the commit). `main` is `ddf9519e929e6ef1f48cd5389f816f4ff011a3fd`, which contains `e447bdd8190f`.
   - **With `main`: none**, and no `Move:` or `Layout-move:` commit lies between, so a plain merge is the carry.
   - **With #10134: the five files lean's worker named.** In each, #10134 renames `EncardProgramRunsLeWindows` to `EncardProgramRunsLeExecutionPhases` (and `encard…` with it) inside lines #10088 deletes.
     - `security/Proofs/NetworkCertifier/Causal.lean` and `security/Properties/NetworkCertifier/DeviceOutput/Runs.lean`: #10088 deletes the files; the carry keeps them deleted.
     - `security/Proofs/NetworkCertifier.lean`: one docstring line naming the theorem, which #10088 deletes; the carry keeps it deleted.
     - `security/Proofs/NetworkCertifier/README.md`, two conflicts: lines #10088 deletes or rewrites so they name no statement. The carry takes #10088's text there and #10134's renames everywhere else (`MonitoringPeriodMissProb`, `ExecutionPhasesPlaced`, `executionPhasesCount`).
     - `verity/protocols/accounting/certifier/verifier/PROTOCOL.md`, three conflicts: the same, #10088's text where it removed the name.
     - **One carry the merge doesn't flag:** `security/Properties/lean-audit.json` merges clean but keeps two entries #10134 renamed and #10088 deletes, `witnesses` and `shape.properties` under `"Proofs.NetworkCertifier.encardProgramRunsLeExecutionPhases"`. Whoever lands second drops those two.
     - The twelve reworded reasons in `security/lean-audit.json` survive the merge (12 of 12).
   - **Which way:** #10134's commit `4261d4e1cb8` carries `Move: tools/move/lean_names_map.toml`. If #10134 lands first, #10088 carries by `tools/move/restack.py`, whose move script finds nothing to rename in the deleted files; the two lock entries still go by hand. If #10088 lands first, #10134 re-runs its map on `main`. Not carried.
4. **The records run `r20261114-160540-4338` is still running.** Its list of records whose `reads` or text move comes when it ends.

I hold no timer and no subscription. CLOCKGONE needs no line; the head did not move.

[2026-11-14T18:23Z] eb1650373fc0
LOWRISKGUARD to network-accounting (lead), for ci. **Condition (1) fails on `main` at `ddf9519e929e6ef1f48cd5389f816f4ff011a3fd`: two `verifier/` directories are not excluded as directories by `lowrisk_excludes`.** I checked with the queue's own matcher, `research.queue.lowrisk`, on `drive.toml`'s list as `DriveFlags.parse` reads it [ran].

- `verity/protocols/accounting/registration/verifier/`: both tracked files are kept off the lane only by name globs (`registration.py`, `__init__.py`). A new file there would be low-risk.
- `verity/protocols/compose/verifier/`: its tracked files are all under `lean/`, `onsite/` and `remote/`, which are each excluded, plus the Lean globs. A new file directly in `compose/verifier/`, or in a new subdirectory, would be low-risk.

The fix for ci would be adding `verity/protocols/accounting/registration/verifier/` and `verity/protocols/compose/verifier/` to `lowrisk_excludes`. The second entry would make the `onsite/`, `remote/` and `lean/` entries redundant, but not dead. I have not edited `drive.toml` and have not weakened the test, so the test fails on `main` until ci adds them.

Condition (2), dead entries: none of the 69 entries is dead; each matches a tracked path [ran]. I'm going on to write the test and open its pull request unmarked.

[2026-11-14T18:26Z] 4f8ba95b3e29
PROVENANCE2 to network-accounting (lead), Sat 14 Nov 10:40 AM PST. MODEL: Claude Opus 5.5. TIMER: none, no subscription. Read in the inbox: Daniel's rulings, the narrower pre-mark run, ci's machine rule, ROWSMERGE's line.
- Mend tries at `2004ebdeaacfd052b18f23598a2db881be428247`, all `--siblings`: `r20261114-160503-1862` RecordDeclared, `r20261114-161511-f98a` RecordRoles, `r20261114-161511-8359` RecordProvenance. Each lean exit 0, no name clash, axiom probe clean (15, 14 and 3 theorems). So the mend is the binder alone.
- 1. `cursor/na-scratch-provenance-7c3e` is now `72240804dc383ab69e0d5692ed772cffbec2e9b1` (pushed): #9644's `0f1ed08ab83c` merged. The one conflict was the pin `tools/pous_grader/TRUSTED.sha256` (main moved the proofs lakefile too), regenerated; `pins.py --stale` clean. Its diff against #9644's head is my six files. Launched: Lean check of `security/Proofs` on machine 15, `r20261114-182325-984e` (machine 3 and machine 14 refused for memory; nothing else launched); suites `repository` and `verity/protocols/accounting/pous/verifier` (`--quick`, `--on auto`), `r20261114-182429-95ae`, about 33 min to a slot. Those two are the suites my diff touches: `security/` and `tools/pous_grader/` are no suite of their own, and the pin's tests are those two suites'.
- 2. The PR opens when both pass. Its body, with every statement verbatim under "Statements for the lead" (none listed in a lock), is at `lanes/network-accounting/provenance2-pr-body.md` in the notes.
- 3. The mend: a commit that waits for whichever of my PR and #10113 lands second, not a second PR. A PR stacked on #10113 would stack on a PR that isn't mine, which the landing rule doesn't allow, and #10113 doesn't need my files. The commit touches only `RecordDeclared.lean` (+10 -9) and should cherry-pick onto either landing. If mine lands first, #10113's owner carries it; if #10113 lands first, I carry it after merging `main`. The branch stays as it is, the commit's reference.
- 4. No lemma of mine names the module path. My three files import only `Proofs.Properties.Nci.*` modules and reach `RecordText` only through `open NciCheck.RecordText (…)`, by declaration names. If the move keeps full names (namespace `NciCheck.RecordText`), nothing of mine changes; only `RecordInputs`, which isn't mine, imports downstream of the move. Not run.
- Waiting on `r20261114-182325-984e` and `r20261114-182429-95ae`: if both pass, I open the PR.

[2026-11-14T18:26Z] c11860f5ed7f
ROWSPROTO2 to network-accounting (lead), Sat 14 Nov 10:25 AM PST. **#10108's new head is `64b676b13595a32c209c46cce30ac10310247532`, pushed. Its suites run is `r20261114-182221-476d` (queued); I send its verdict in one line when it ends.**

**The merge.** I merged #9855's head `017cf9031712dcf70f3b44618fb43d33fb8b8700` into `cceb77c1b052b3842aaa909c857fbabf8aad5ae9`. I resolved one file by hand: `tools/verity/check/public_boundary.toml` (modify/delete), where I took `main`'s deletion, so the widening drops out. Every other file merged clean. #10108's own diff against its base has the same `git patch-id --verbatim` as at `eba54ef543a8`: the same 11 files.

**The run.** It covers the suites of the packages the diff touches, `verity/protocols/accounting/certifier/verifier/network` and `verity/protocols/policy/inference_only/verifier` (the nci package is under it), plus `repository`, slow tests included, launched with `--on auto`. I launched no Lean check, because the diff changes no Lean since `r20261114-084332-1ffe`'s head `2f8b1b9c3682e5386b8ce5e6d70cd965abf0c6b5`: no file of the nci package differs between that head and this one. The merge did change Lean the package requires: `Spot/OneStageZk.lean` and ten other Spot files, plus `Flock/KsScope.lean`. `RowsProtocol` reads `Spot.OneStageZk.STATEMENT` and `tablesOf`, `Spot.FilePart` and `Flock.ProvedScope`. The first two names are unchanged, and the last two files did not change. The rows test builds `inference-only-rows` on the merged tree (`D.build`), so the run will catch a broken build.

**The parts, as the body now states them** (I added the section "The parts" from the module docstring):
- params: canonical JSON with exactly the keys `alpha`, `circuits`, `gamma`, `log`, `pins` and `program`.
  - `gamma` is Γ, the most rows a block holds.
  - `alpha` is `[num, den]`, `den` not 0, the most fresh bits a row.
  - `circuits` is `[[template id, SHA-512 lower-case hex], …]`, one pair for each of the record's templates, in the record's order.
  - `log` is the registration records before the run's own, oldest first.
  - `pins` are the closure digests of the reader Definitions, each a SHA-512 in lower-case hex.
  - `program` is a `verity-ir/descriptor/v1` text.
- commitment: the run's own registration record, read by v0's `Protocol.commitmentOf` (canonical JSON, an object, with its digest).
- evidence: canonical JSON `{record}` or `{order, record}`: the circuit record's text, and the order of its unit ranges where the record states none.
- blobs: the templates' circuit files and the pinned lookup tables. A blob is read only under a digest `circuits` names or a pinned table's own.

**`checkBlobs`' refusals, in the order checked.** In the tests column, "quick" is `test_each_change_takes_its_path_on_both_sides` with the named change. "agreement" is `test_the_rows_protocol_agrees_with_the_reference` (slow). "files[X]" is `test_the_rows_protocol_on_circuit_files[X]` (slow). The first three blocks of rows are the decoders' refusals, before `checkBlobs`, and `Core.Protocol.answer` writes their prefix.

| Refusal, as the reason opens | Condition | Test that reaches it |
|---|---|---|
| `params: the keys of the params are not alpha, circuits, gamma, log, pins, program` | the params' keys are not exactly those six | quick `params-key` |
| `params: the params' program is not a string …` | `program` not a string | quick `program-type` |
| `params: the params' log is not a list of registration records` | `log` not an array | quick `log-type` |
| `params: the params' gamma is not a natural number` | `gamma` not a natural number | quick `gamma-type` |
| `params: the params' alpha is not [num, den], two natural numbers, den not 0` | `alpha` not a pair of naturals, or `den` = 0 | quick `alpha-den-0` |
| `params: the params' pins are not a list of SHA-512s in lower-case hex` | `pins` not an array, or an entry not 128 lower-case hex digits | quick `pins-type` |
| `params: the params' circuits are not a list of [template id, SHA-512 in lower-case hex] pairs` | `circuits` not an array of `[string, digest]` pairs | quick `circuits-type` |
| `commitment: …` | not canonical JSON, or not an object (v0's decoder) | quick `commitment-type`, `commitment-noncanonical` |
| `evidence: …` | not canonical JSON, or not an object (`Core.canonicalJson`) | quick `evidence-noncanonical` |
| `evidence: the keys of the evidence are not record, or order and record` | keys neither `{record}` nor `{order, record}` | quick `evidence-key` |
| `evidence: the evidence's record is not a string …` | `record` not a string | quick `record-type` |
| `Record.read`'s code, as `CODE` or `CODE: message` (`instance`, `program`, `not-topological`, `record`, `trusted`, `run-program`, `run-circuit`, `templates`, `own-rows`, `wires`, `leaf-widths`, `inputs`, `blocks`, `reader-layout`, `reader-roles`; `params` cannot arise) | the record, read once with the params' Program, pins and log and the run's own record, fails a stage of the record form | quick `refused` (each of the record difftest's refused cases, with its code), `record-program` (`run-program`), `record-circuit` (`run-circuit`), `log-repeat` (`trusted`) |
| `circuits: the params' circuits do not name the record's templates, each once, in the record's order` | `circuits`' ids are not the record's template ids, in order | quick `circuits-drop`, `circuits-extra`, `circuits-order`, `circuits-id` |
| `circuit-file: template T: no blob has the SHA-512 D` | for the first template in order, no blob has the digest `circuits` names | quick `no-blob`, `blob-changed`; files[no-blob] |
| `circuit-file: template T: <the reader's refusal>` | `Flock.ProvedScope.circuitOf` under SPOT's one-stage tags refuses the file | quick `honest` (its blobs are not circuit files); files[refused:…] on the six refused fixtures, each with its text |
| `circuit-file: template T: one unit of the file has no computed row` | `Spot.FilePart.ofCircuit`'s `m` is 0 | **no test**: no file the reader takes has m = 0 |
| `circuits: …`, a second time | `RecordRows.ofSummary` gives none: a template missing from the table | **no test**: cannot arise once the `circuits` row above passes |
| `size: a block has more than Γ rows` | a block's rows > Γ | files[size], files[more-rows]; agreement's `file-size` |
| `nci: a block reads more than α fresh bits a row` | fresh × den > num × rows | files[nci]; agreement's `file-nci` |
| accept, public `{}`, profile `{blocks, chosen, fresh, rows}` | none of the above | files[accept], files[zk-sessions]; agreement's `file-accept` |

Two refusals are reached by no test: "no computed row", and `circuits` when the summary gives none. Each stays as a stated invariant, as the body's "Not checked" says. The cut column (which test fails when a check is removed) is in the body's table, unchanged.

**Item 4: the design.** Nothing in it is still open beyond the body's section "Defaults taken that the lead has not confirmed" (nine defaults), which waits on your read. The body as it stands is the design. When proofs' move of `RecordText`'s text part lands, the restack touches one name of mine, `RecordText.Tmpl` (`RowsProtocol.lean`, `namesTemplates`), and with names kept that line stays as it is. I import `RecordText` through `RecordRows` and `Record`, not directly.

**Item 5:** #10108 rewords no listed line. None of its 11 files is in `tests/lint/allowlists/repo_python_verdicts.json` on #10128's branch, and #10128's tests read only listed sites.

I dropped no served run, gate session, rehearsal, capture or proved run, because none was on my list. TIMER: none.

