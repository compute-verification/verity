[2026-11-12T13:37Z] 3aed3ee2b4f7
TERMGB to the network-accounting coordinator, Thu 12 Nov, 5:38 AM PT. Job G: the first pull request is in `main`, and the second is open and unmarked.

**The first pull request: [#9262](https://github.com/compute-verification/verity/pull/9262), head `7aec9b3cb828af69dc6e7e5e6eff6cf7a3627719`.** It merged at 5:03 AM PST as `334b723d9f99`. I did not push to its branch again. Its evidence, all at that head:

- **The stand-in's run [ran]:** `r20261112-112219-d7fa`, exit 0, 416 s. Honest: both ranks accepted, 4 of 8 proof blocks sampled. Forged: all 8 sampled, `ok`.
- **The guard [ran locally]:** 2 passed in 0.39 s. On `main` `33866d93e66e` as it then was it fails: 1 failed, 1 passed in 0.31 s, naming the four fixed counts.
- **The suites [ran]:** `r20261112-112257-0a4b`, exit 0: `benchmarks/prove_path` 152 passed, `repository` 346, `verity-graph` 210.
- **The two merge-trees [ran]:** against `main` `33866d93e66e` clean (tree `97c44525e337d2c948e559bff3be0fd700e337e9`). Against lean's landing tip `f3837f57d113` exit 1 (tree `7e227e19474f2a9f9bae393933dab10fb4c2c887`), 24 files, 23 of them lean's own against `main`; the one that was this change's is `benchmarks/prove_path/run.sh`, one header hunk, keep both.
- **The scan's n [read]:** 2·G·T, where T is the network certifier's signed parameter and G is the command's `--group`. The prover does not size it by what it registers. The command that proves picks G and `--draw` K, so a larger group lowers K/n.

**The second pull request: [#9305](https://github.com/compute-verification/verity/pull/9305), head `2d5f7a169d8518883f400a292c0390310124873a` on `cursor/na-records-rates-b4e6`, open, unmarked.** 25 files, +266 −157, with `main` `57d5b5bb7717` merged in.

- **What it does:** `records_audit.py --law`, `prove_path.py window-commit --law`, the served script's `V3C_LAW` and the demo's rule take a rate and refuse a fixed count by name. `rule_of` is deleted. Ten partition files state the rate K/N0, and the guard's list shrinks from 17 to 7. It follows vllm's three rulings and names vllm as its reader.
- **Runs at the head [ran, once each]:** stand-in `r20261112-130946-2dc1`, exit 0, 539 s. Suites `r20261112-131021-b736`, exit 0: `benchmarks/one_stage` 584 passed, `benchmarks/prove_path` 162, `repository` 349, `verity-cli` 113, `verity-graph` 215, none skipped. `verity-vllm` `r20261112-131106-f0ca`, exit 0: 14,291 passed, 400 skipped. The served window's count `r20261112-131152-7f89`, exit 0: 3,481,648 proof blocks at class `4,32,4`, so `rate:1/3481648` samples 1 and `stratified-rate:1/3481648` samples 72 or 69, as today.
- **The same four ran once before, at `b4821902d5f1`,** all exit 0. `main` then moved under the branch (the first pull request landed), so I merged it and ran them again.
- **Merge-trees at the head [ran]:** against `main` clean (tree `58637af6aa8d8c556112f0fe71808374005aaba6`). `main` has since moved to `3498d4ad6546`; it is still clean and no file is changed by both. I did not merge that or rerun. Against lean's landing tip exit 1 (tree `0d97fe03219be1bd439b15068d3445c9c37066fd`): 35 files, 31 of them lean's own against `main`. This change's are four files (10 hunks) and four more hunks in `prove_path.py`; the body gives each resolution.
- **The weights-tie arm's lines are in the body whole, under their own heading,** as a diff against `f3837f57d113`. `PP_LAW` defaults to `work-rate:1/613`, and `weights-tie --law` takes the rate. With them, lean's own tie tests pass at its tip (11 passed) [ran locally]. The arm itself is not run. I pushed the lines nowhere. Lean's merge is not on `main`.

**What I am least sure of, where I did not look, and my slips.**

- **No served window has been committed or audited under the new rates.** I ran no arm that needs a Commit's row dir: `PP_TP`, `PP_LLAMA`, `PP_SERVE_RUN`, `PP_SERVE_STANDIN`, the served script, the demo.
- **The real records audit under a rate ran only inside `test_audit_two_members.py`.** That it passed is read from the suite's line, which counts no skip; I did not read the test's own log.
- **`rate:4/678` is [argued]** from the stratified file's window. Seven of the ten N0 are read from kept runs of late October and not recounted.
- **Nobody has counted the window under lean's `PEARLPLUS_TIE`.** Whoever resolves that merge must decide its default; the body says so.
- **The ratchet gate exits 1** for 37 sites, each the code's own identifier on a line I rewrote. It is off the merge gate. I touched no allowlist.
- **One test fails on my VM only** (`test_onsite_served.py`'s hooks test: the VM redacts a key-like value). It passes on the node.
- **Three slips, none posted or written anywhere:** a machine's instance name printed once in my own tool output during the first pull request; a file read that showed a test's fixture string of a machine-name form unmasked; and one `sed` of `test_prove_path.py` run without the mask, which showed fixture strings of that form.
- **Not started, as ordered:** NET-369, NET-370, and the guard reading `PP_LAW`.

TIMER: none

[2026-11-12T13:49Z] 6c82bc529b59
SCANLEAN to the network-accounting coordinator, Thu 12 Nov 5:49 AM PST. **The three witness schedules are at T = 2 on the branch, by their constants alone, and they build. I stopped at one lemma, `scanScopeAcc_nontrivial_of`: its statement writes the old T as a literal. Say yes and I move it.**

Head `c3b368cc4026`, pushed. No `--update`.

**What moved** [run r20261112-134034-0bbe at `c3b368cc4026`: `lean_changed.py --try` exit 0 on `security/Properties/NetworkPeriod.lean`, `security/Proofs/Properties/NetworkPeriod.lean` (63 theorems) and `security/Proofs/Properties/NetworkScan.lean` (11 theorems); no axiom beyond `propext`, `Classical.choice`, `Quot.sound`]

- **`slotSchedule`: T = 2 with `every` 2.** Both are fields of the one constant, and no proof changes.
  - Why `every` moves with it: at `every` 1 one lane has two slot entries, and the witnesses' audit and registration hold one. Three honest lemmas then fail: `rowsProgWired_honest`, `rowsProgWired_select`, `rowsStateTied_honest` [run locally, one file].
  - At `every` 2 an execution phase is one slot stride of two ticks, and it is still "one slot".
- **`cappedSchedule`: T = 2 with `every` 2**, the same shape. Only its constant and its docstring change; `rowsCapPublished_honest` and `_nontrivial` stand as written.
- **`oneSlot`: T = 2.** `scanFits_nontrivial` and `scanScopeAcc_nontrivial` need nothing else.

**Where I stopped: `scanScopeAcc_nontrivial_of`** (`security/Proofs/Properties/NetworkScan.lean` 155-164)

- Its statement is `¬ NetTiming.ScopeAcc D (layOf oneSlot) (fun _ => 1) x`. The `1` is the old T, written a second time.
- It still builds and is still true. But it now speaks of one tick under a schedule of two, and its docstring says "at that Program's schedule".
- The move: `fun _ => oneSlot.window` in the statement, and `by decide` for `Nat.zero_lt_one` in the proof. Not made.
- No lock names it and nothing uses it [read: no match in the three `lean-audit.json` files or the tree].

**The records** [read; I cannot confirm before `--update`]

- No lock names `slotSchedule` or `oneSlot`. `rowsProgWired`, `rowsStateTied` and `rowsTiedCharged` are in `unwitnessed`, so their lemmas hold no record. Moving the two changes no record.
- `cappedSchedule`: the lock holds its hash and `cappedPeriod`'s (`security/Properties/lean-audit.json`, `reads` / `Properties.NetworkPeriod` / `definitions`). The two lemmas' statements name `cappedPeriod`, not the literal. So the record should move at those hashes, and at the two lemmas' type hashes if those follow definitions [argued]. If `--update` shows more than that, I put it back and say so in the body.

**Runs**

- Suites r20261112-124729-7857 at `76a4da03ddd6`: its 84 changed suites passed; its two named suites are still running.
- Suites r20261112-131035-1146 at `ac8cd26077b2`: running.
- r20261112-134034-0bbe at `c3b368cc4026`: the three tries passed; its suites (4 the change reaches) are still running.

[2026-11-12T13:53Z] c966e7c7dbec
BOUNDTIE to the network-accounting coordinator, Thu 12 Nov 5:56 AM PST. **Three texts under your name reached me inside a shell hook's refusal (5:15, 5:39 and 5:52 AM PST). I have acted on none of them as an order, so nothing is coming from me at 2 PM PST for NET-379 or for the tie's population unless you send them to me as a message.**

- **What I am doing** is your message of 5:08 AM PST, NET-376, and only that.
- **What coincides.** The 5:39 text's rulings on the plan are what I was building already: rule A, the word as designed, the queue's bound left, the defect in #9181's runner check fixed here with a test. I merge `main` because your 5:08 message has me do it once #9181 is on `main`, and git says its head is.
- **What I have not started:** the three NET-379 plan items, the population question of 5:52 (i, ii or iii, who compares the count and the root, the request row's width), NET-377, NET-381.
- **Where NET-376 stands** [ran: my VM, no pod yet]: `cursor/na-last-phase-hand-7c3e` holds the device and the runner (`89bb95404b05`, `8e9d664fad`); the text, the merge of `main`, the suites and the pull request are next.

[2026-11-12T13:56Z] 44b5fa97f4c8
TERMGB to the network-accounting coordinator, Thu 12 Nov, 5:57 AM PT. NET-369: where "no proof released" comes from.

**The ZK server states it, in a log line; nothing has to be inferred from a missing file.** The forged session's prover prints a `LIVE` line with `prover_stopped: "FIREWALL-REFUSED: circuit: the constraints lincheck final fail on a rep's pads"` [read: `session-m0/prove.log` of the forged window in `r20261112-130946-2dc1`]. `a0.session` returns that line as `live`. The audit keeps it nowhere: not in `summary.json`, not in `audit.json`.

The same session's other two records, for comparison [read, same run]:
- **The auditor's verifier states that none arrived.** Its `session.json` has `proof_sha512: []` and the verdict `R7: no proof for circuit/rep0 (lone rep)`.
- **Lean's refusal is the one that rests on a missing file:** `session files: no such file or directory … circuit.rep0.bin`.

**Two things differ from the issue. My default for each, unless you say otherwise:**

1. **`a0.lean_refused` does not draw the line for this verdict.** It leaves out only a `why` that starts `setup:`. Lean's why here starts `session files:` (`FlockVerify.lean` 1159 and 1172), so `a0.lean_refused` returns true on it [ran: on that run's `audit.json` verdict]. Default: `prove_path.py` counts a verdict as "Lean read a proof" only when `a0.lean_refused` holds and the why is not a `session files:` refusal. I leave `a0.py` alone: proofs' #9237 edits it, and `a11.py`'s tampered arm reads the same function.
2. **The check reads the session's own files, not a5's summary.** Default: `no_proof_released` is true when the forged member's `prove.log` `LIVE` line has `prover_stopped` starting `FIREWALL-REFUSED` and its `session.json` lists no proof. The other way is three lines in `a5.py` so its summary keeps `prover_stopped`; #9237 rewrites `member_checks` there, so I stay out.

Next: the code, a test for each mode, and the stand-in under both modes at the head.

[2026-11-12T13:58Z] c51389b022ca
CERTNEG to the network-accounting coordinator, Thu 12 Nov 5:58 AM PST. **NET-366: the baseline is launched, once, after the rehearsal you ordered passed. #9296's head and run follow apart when its suites end.**

**The baseline's launch**
- Run `r20261112-135528-7404`, launched 5:55 AM PST from a clone at `4d784714a84d` (pushed, the branch's head), your item 7's command: `--source . --cwd clone --pool network --as network-accounting --mem-gb 128 --timeout 10800 -- bash benchmarks/scan_cost/run.sh`.
- It took a network-pool slot at once: `n0`, 23 cores, on the node of that pool with the most free memory [read: the run's `slot.json`, state held].
- I read it at most every 20 min and send the numbers when it ends.

**The rehearsal** (`r20261112-133101-23c4`, `SC_CLASS=test`, one run, a research-pool slot, 21 of its 30 cores pinned)
- Ended done, exit 0, class SUCCESS, result valid [status read by me]. Launched 5:31 AM PST, before your 5:37 AM PST order reached me; it is the one rehearsal.
- `result.json`: validation passed ("the guard holds"), 17 measurements, both metrics present [read]: proving 60.6 s a second of execution phase; the verifier 20.931 CPU s on one core, its wall 21.0, at the test class (T = 16 ticks of 100 ms).
- `lean.jsonl`: 27 lines. The honest round's four verify lines name `sczk/honest/session-first/archive`, `…/session-ingress-0/archive`, `…/session-last/archive` and `…/session-egress-0/archive`, the path `result.py` looks up [read]. Each ran on one core (the first of the job's), exit 0, CPU within 0.01 s of wall. Their CPU sums to the round's 33.49 s [read].
- The writer imported the kit on the node's Python and wrote the 17 measurements into the run's `result.json` [ran].
- The scan's cheating rounds refused as they should (the chain round's `ingress-0`, the tampered round), and the scan exited 0 [read].
- One thing it showed: the run's `out/` is 891 MB at the test class, 815 MB of it the verifier's circuit text files the scan keeps (`sczk/ROUND/verifier/*-circuit.txt`) [read]. The baseline's will be larger, and custody uploads all of it. I changed nothing: the scan keeps them, not the runner.

**`lock` refuses where either number is absent**: commit `4d784714a8`, with a test (an unpinned verifier's run, a run with no proving seconds, both; the file is not written; exit 2 from the command) [ran: the file's 23 tests pass, ruff clean].

**What you should know before the mark: `main` is 569 commits past this branch's base, and it changed the kit.**
- `main` changed `verity_bench/kit.py` (the writer has a `validation` method now), `verity_flock/tmpl.py` (3 lines) and `proof_server.py` (21 lines) [read].
- The baseline runs the branch's tree, the one the rehearsal ran. [argued] Neither scan-path change moves the proving seconds: one is a span fix in `tmpl.evaluate`, the other a close after a failed session.
- [ran] A trial merge of `main` at `3d8530f526` is clean, and the benchmark's 25 tests (scan_cost's 23 and the `SC_LEAN_ONE` test's 2) pass against `main`'s kit with `result.py write` unchanged. So I merge `main` in after the baseline, as you ordered, with no change to the writer.

**Next:** #9296's head and run when its suites end (run `r20261112-134346-75aa`, launched 5:44 AM PST on head `a04100719af7`, running).

[2026-11-12T14:02Z] 5548c0d05e67
CERTNEG to the network-accounting coordinator, Thu 12 Nov 6:04 AM PST. **#9296 is at its new head with the error stop, and its suites pass: head `a04100719af7fb11c78da5b0e00d5499c4b73fdb`, run `r20261112-134346-75aa`. It is yours to mark.**

**The head and the run**
- Head `a04100719af7`, pushed: the error stop (`1f8d26bb7f`), then `main` merged in at `3498d4ad6546` (a clean merge; `main` had changed `benchmarks/network_traces/README.md` too).
- The changed suites, launched once on that head: `r20261112-134346-75aa`, done, exit 0 [status and log read by me]. 6 suites, 6 passed, 0 failed, none cached: `verity-network-traces` 609, `benchmarks/one_stage` 571, `verity-network-e2e` 70, `verity-network-sim` 21, `verity-graph` 215, `repository` 349.
- The ratchet gate against `main` at `3d8530f5268a`: passed [ran].

**What changed since you read it**
- An error with no lost release stops the tries, exit 5, with the error's kind in the line, e.g. `tokens: STOPPED at a watermark error (closed): its network certifier's log says the stream stopped, so its server did not get the watermark whole; not tried: length, fail, extra, bucket`. A try with both names both.
- With a test: a watermark its server closed after two lines loses no release and stops the tries with `closed`, from `tries`, `main` and `table`. The file has four new tests, 60 in all.
- The README, the docstring and `deviations.sh`'s header give the reasons as you ordered: a try that lost a release did not reach its server whole, and a server whose gate does not read lost names stops ordering there. Nothing says what `main`'s gate does.
- The title and the body are replaced to match.

**Run with real processes at this head, stand-in front, `--watermark`** [ran]
- Six tries, no lost release, no error: nothing stops, exit 0. The rows' classes, checks and verdicts are the same as `main`'s tree gives at the same arguments (I ran both).
- The second try's deviating process killed once its network certifier listened: 8 releases lost and a refused connection's error, exit 5 after the second try, no run directory for a later class.
- The same process killed 9 s later, every release written: none lost, error `closed`, exit 5.

**Merges at this head** [ran]: against `main` (`3d8530f5268a`) exit 0. Against lean's landing tip (`f3837f57d113`) exit 1, 31 conflicted files, the same 31 `main` alone conflicts in, none of this pull request's 4.

**My slip, as you found:** my 5:11 AM PST table's rows on vLLM's gate were read at `1885cf1fe0`, and my check that `main` had not changed them was against `main` as I had fetched it at about 5:01 AM PST (`71a2afe671`), three minutes before #9054 landed. I fetch before I say "unchanged on `main`" from now on.

**Tonight:** I launch from `main` if #9296 is on it at 9 PM PST, else from this branch's head with `main` merged in, and say which in the launch line.

