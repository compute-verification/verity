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

