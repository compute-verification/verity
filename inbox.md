[2026-11-13T19:50Z] 884adb3c2d2b
TERMGB to the lead, Fri 13 Nov 11:50 AM PST. **#9329 is carried to today's `main` on the carried #9305 and pushed: `cursor/na-forged-no-proof-b4e6` at `10b60872b98f`** [ran: read from origin after the push], a fast-forward of `84414ed14eec`.

- **The tool's words** (`r20261113-185602-fe0c`, from `84414ed14eec` to `main` at `d119240998be`): `clean`, 24 moves crossed, none skipped; result `0c882bec6a4e`.
- **Then one merge of #9305's carried head `f5bd1f351236`**, both at today's `main`, no move crossed. Git merged every file itself. **The merged tree is the tool's result's, hash for hash** [ran], so the merge changes no line: it only makes #9305's head an ancestor.
- **No file was resolved by hand on this branch**, at either merge.
- **Over #9305's head: its own six files, +213 -44**, the same six and the same counts as before the carry [ran]. Against today's `main`: 26 files, +558 -220.
- **On the VM at this head** [ran]: `test_records_audit.py`, `test_prove_path.py`, `test_run_rules.py`: 143 passed, 2 skipped.
- **#9305's pins step** (`r20261113-193716-3235`): exit 0, no pin stale, nothing changed.
- **Launched at this head, once each:** `r20261113-194903-d5c7`, the suites changed since `d119240998be` and then the tie's arm; `r20261113-194942-a6f2`, the pins step. #9305's job (`r20261113-193631-b4bc`) still waits for memory on the pool's node.

**One thing is open again, and it is why the tie's arm is in #9329's job too.**
- My line of Thu 12 Nov 10:41 AM PST argued that the tie's second arm, `weights-tie-other`, fails under #9329's rule for `--forged`. That window holds no forged unit: every unit is its Definition's and a read opens no root. So a true proof leaves the ZK server, Lean refuses the session on the read, and `no_proof_released` is false [argued, never run].
- You held it for vllm's word, since the arm then stopped before its sample. On `main` today it reaches its sample.
- So at #9329's head `PP_WEIGHTS_TIE=1` may exit 1 on that one tie where it exits 0 at #9305's. The two jobs say which.
- **I build no mode for it.** If it fails there, I send you the arm's own lines and the three choices of that line again (a mode of its own for a refused read; leave it and name it; or pass on either check, which I advised against).

[2026-11-13T19:53Z] a298028196cc
TERMGB to the lead, Fri 13 Nov 11:53 AM PST. **#9501's run of record has ended and its body is set** (`research queue edit 9501`, 41,775 bytes). The head is unchanged, `b2e51a3184bb` [ran: read from origin]. Nothing was pushed.

- **The run, `r20261113-175741-0c1b`, at `b2e51a3184bb`:** `suites.py --changed d119240998be --quick`, exit 1. Of 61 suites 60 passed (one from the cache) and 1 failed.
- **The one failure is `verity-bench`: 619 passed, 8 failed, by the eight tests the body names and no other** [ran: the run's eight names against the body's eight, the same set].
- **Passes that touch the change** [read: the run's log]: the certifier's verifier suite 1,434 passed and 1 skipped; `verity-network-traces` 746; `benchmarks/certifier_term` 15; `benchmarks/prove_path` 170; `benchmarks/one_stage` 1,069; `repository` 357; `verity-vllm` 17,154 passed and 479 skipped.
- **The body says, as you ruled:** nothing lists a format in `FORMATS` until form 1 is closed (under what it does not close); the eight red tests are proofs' to mend with the join of the two readers; the third harness test that fails only inside a research run is named.
- **Next:** the writer's pull request when its suites run ends (52 of 61 suites done, `verity-bench` the only failure so far, the same eight). Then the bodies of #9305 and #9329 when their jobs end. #9329's job is running; #9305's still waits for memory.

