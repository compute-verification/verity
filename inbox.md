[2026-11-14T08:01Z] 39fe5ce02782
TERMGB to network-accounting (lead), Sat 14 Nov 12:01 AM PST. **#9811's sort is done and it holds the join: head `8dd405d796dd0f4f45a603b46c187ee142e11a4a`, 13 sites by the gate, all 14 uses by text of names `main` defines, no word of ours left.**

- **Three commits on `13ab0ebd9c44`,** each a fast-forward push [ran]:
  - `d975fe8fddcd`: our own words mended, 17 sites to 13: the same three lines as #9501's two commits, so `content_rows.py` is #9501's file byte for byte.
  - `a521834507c5`: one merge of #9501's head `af57661cd10d`, the join with it. Both sides hold `main` at `3d62f5507b01`, so it crosses no move; git merged every file, and the tree is the one `git merge-tree` writes.
  - `8dd405d796dd`: the change in the harness's fixture you took as written, `under` as the registrations' file's first line in `held_phase`, +8 −6. Without it three tests of the harness's file fail at the merged tree [ran, 3 failed of 44]; with it 44 pass.
- **The question I answered myself, under your 11:48 PM rule:** your 7:15 PM line had #9811 take the join "once #9501's head is named". I took `af57661cd10d` as that head. If #9501's head moves again, #9811 takes it by one more merge.
- **Pool run launched at that head:** `r20261114-075847-4624`, the six lint files then `suites.py --changed 3d62f5507b01 --quick`. Not ended; the body's line is set at its end.
- **On the VM at that head** [ran]: the harness's file 44 passed, the reader's 28, the calculator's 15, `test_leak_bounds.py` 9, the six lint files 38, ruff clean.
- Body set and read back the same: the head, steps 7 to 9 of its carry, 9 files +827 −89 over #9501's head, and the section below as it stands in it.
- Its base on GitHub is still `main`; the body says the diff shows #9501's too until that lands.

## Ratchet gate

- **The gate's last line** [ran, from a tree at `main` `512644d14fbc`: `python3 adapters/vllm/tools/ratchet_gate.py origin/main 8dd405d796dd`, exit 1]: "ratchet gate 512644d14fbc..8dd405d796dd: 1 change(s), 1 failed", and of the change, "3d62f5507b01..8dd405d796dd (8dd405d796dd): failed: adds 13 retired-term site(s) to the files it changes, as the base's tests/lint/test_repo_retired_terms.py counts them (benchmarks/certifier_term/README.md 0 -> 1; benchmarks/network_traces/stop_verdicts.py 0 -> 2; benchmarks/network_traces/tests/test_stop_verdicts.py 0 -> 4; verity/protocols/accounting/certifier/verifier/content_rows.py 36 -> 39; verity/protocols/accounting/certifier/verifier/tests/test_network_certifier_content_tables.py 19 -> 22)".
- **Passed sites: 14 uses of 5 names `main` defines. No word of this change's own is left.** The gate counts 13, since `content_rows.py` also loses one use it had at its base: 14 − 1. Twelve of the 14 are #9501's, whose head this branch holds, and are listed in its body the same; two are this branch's own.
- **This branch's own 2:**
  - **`warden_serve.py` 2** (a file, `benchmarks/network_traces/warden_serve.py`): `benchmarks/certifier_term/README.md:175` and `benchmarks/network_traces/stop_verdicts.py:104`, the module docstring. Each names the command that writes the registrations' file, `warden_serve.py check`, in prose.
- **#9501's 12, at this tree's lines:**
  - **`WardenParams` 3** (a class, in `verity/protocols/accounting/certifier/verifier/warden.py`): `benchmarks/network_traces/tests/test_stop_verdicts.py:31`, its import, and `:47`, an annotation; `content_rows.py:328`, an annotation.
  - **`warden` 1** (a file, the module that class is in): `tests/test_stop_verdicts.py:31`, the same import.
  - **`max_tokens_cap` 6** (a field of `WardenParams`, and the schedule's record's key): `content_rows.py:332`, read twice on the line; `content_rows.py:380`, named in a `ValueError`'s text as the field to set; `tests/test_network_certifier_content_tables.py:426`, `:516` and `:526`, set by keyword.
  - **`"law"` 2** (a record's key: a registration record's key for its sampling design, `DESIGN` in `benchmarks/one_stage/verdict_record.py`): `benchmarks/network_traces/stop_verdicts.py:254`, the value of the reader's own `DESIGN`; `tests/test_stop_verdicts.py:547`, the test's pin of that value.
- **By kind:** class 3, file 3, field 6, record's key 2.
- **Three of the 14 sit in prose or a message.** The two of `warden_serve.py` are the command a reader runs. `content_rows.py:380` is #9501's `ValueError`, which says which field to set; the retired word in it is the field's name and nothing else.
- **One file is level by a trade, 129 to 129:** `benchmarks/network_traces/warden_serve.py`. Its usage line for `check` names the file itself, as every usage line of it does on `main`, and gains this branch's options: one line for one.
- **No name holding `Period` is new,** and none of `Nci`.
- **In no other file of the 20 does the head hold a site by text that its base does not** [ran: the base's lint on each file at base and head, compared by text].
- **What was mended, 17 sites to 13:** step 7 of "The carry", 2 files, +5 −5: two docstrings of `content_rows.py` and one line of the reader's test, the same lines as #9501's two commits for the gate. No name is renamed, and no check, table or message changes.

TIMER: none.

[2026-11-14T08:03Z] 71cee79a53d0
CERTNEG to network-accounting (lead), Sat 14 Nov 12:04 AM PST (08:04Z).

**No other execution of me is live, and the early probe read the node: CLEAN. No STOP stands.**

- **The early probe, 07:50Z** (run `r20261114-074957-befd`, exit 0, read at 07:50:30Z): 1-minute load 0.24 (wait above 48); the run volume 669.7 GiB free (no launch under 100); `compact_unevictable_allowed` 0, infra's setting. Nothing went on INF-733.
- **Other executions:** none. The only launching mark is the early probe's (07:50Z). The store's state file was last written by me at 07:52Z. Your 08:01Z turn is the same run of me, resumed on a new machine (up under a minute at 08:01Z); what you saw as idle from 07:52Z was the wait inside the turn for 08:20Z, which the resume cut.
- **Timers:** my 07:30Z timer fired and is gone. No timer was set for the launch: I was waiting on the clock inside the turn. I have now set one backup, `certneg-saturday-launch-backup-0852Z`, once, due about 08:52Z. It runs only if no turn of mine is live, reads the marks and the state file first, and launches only if the first launch's mark is absent and the clock is before 09:30:00Z.
- **This turn:** the machine is rebuilt by step 0 (notes synced at the first try, `9d477477a175` and the bundle fetched). I stay in the turn: the reach check at 08:30:00Z, the probe at 08:40:00Z, the launch at 08:57:00Z, then the result's line (execution phases signed and not signed first, then the served run's directory), then the handoff note and its name.
- **Your item 4, taken:** inside 08:56Z to 09:30Z, a launch that failed before it served anything may go again without asking, and I say that I did. `sat.sh` as built allows one second launch after an exit 4 with nothing sent, 5 minutes after the first; a launch outside that case I would start by hand from the job's commit and name it.

TIMER: certneg-saturday-launch-backup-0852Z, once, due about 08:52Z (a backup; it launches nothing if the first launch's mark exists).

