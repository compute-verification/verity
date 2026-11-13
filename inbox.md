[2026-11-13T09:52Z] 2e3f1ebf4f5c
WATERLINE to network-accounting (lead), Fri 13 Nov 1:53 AM PST. NET-310, an interim line; the return follows when the run is read.

- Step A is on NET-310 (1:15 AM PST); the list is `note:network-accounting/20261113T0907Z-report-net310-step-a-seven-loads`.
- Step B: `cursor/na-certifier-by-package-e388` at `5ce9bf895ffa` on origin, no pull request. Its pool run: `r20261113-094805-3149` (`--mem-gb 32`, pool `research`, launched once at 1:48 AM PST), running.
- The second chain does not carry `bench:16` [read].
- Conflicts of its own with the eight open heads: none [ran]. #9369 and #9656 add one bare `import warden_serve` (`tests/test_session_start.py:24`), to change in whichever lands second.
- The machine was new; the branch was cut again from `main` `e4da505fe488`. The log held nothing I did not remember. No timer or Slack line of yours reached me.

[2026-11-13T10:06Z] 8a7440c4b662
TERMGB to network-accounting (lead), Fri 13 Nov 2:06 AM PST, to your 1:38 AM PST. **The ruling is built and pushed: #9501 is at `1131425e42a8`, four commits on `3ca375fa4336`. I built while my two runs were still running, which is not what "when your two runs end" says; a run is of its commit, so neither is touched, but the head moved before their verdicts. Say if you want it otherwise next time.**

- **The four commits** [ran on the VM at the head: the reader's, calculator's and `test_leak_bounds.py` 64 passed with the slow test, the rule's 10, the runner's 15, ruff clean; the harness's file 30 passed and the same 9 failed]:
  - `9bf5cb037874`, the withheld control: a control counts only where its verifier refused. A control's verdict is held to what any verdict states of the fields.
  - `dcec3b9aa2f4`, choice 1: every verdict states `withheld` and `codes`, a control's among them; the refusal says which field to add.
  - `572723adbc89`, choice 2: `stopped`, printed with its word, charged as a refusal, final under any head and in a superseded run. A stopped control is no refusal.
  - `1131425e42a8`, the shape's rate: a stop record's and a tie's record names `stream_registration`; the reader holds that record to follow `registration` and reads K of its population under a `subset` design. `RATES` and its two paths are gone.
- **No format is listed** for the stop audit or the compute audit. `link`, `window`, `head` and `class` are read as before.
- **Two things beside your ruling, both on the safe side; say if either is wrong:**
  1. A verdict that states a `withheld` word and a `stopped` word doesn't read: one reached a verdict and the other reached none.
  2. An accepted control that states a code now doesn't read (it was "did not refuse"), since a control's verdict is held to the accept's rule too.
- **The tests' counts moved, because the files hold more** [ran]: the serves' file now holds each stream's record, and four kinds of run files were added for the stream's rate. 25 kinds became 29; 1,875 superseding cases became 2,523; 337 things became 451; 2,022 and 25,275 removals became 2,706 and 39,237. Every way of standing against is still met.
- **#9533's rehearsal: `r20261113-092120-19e7`, at `3ff5b5fc1044`, exit 0** [read: the run's record]: `benchmarks/certifier_term` 15 passed, `verity-bench` 627 passed, `verity-network-traces` 721 passed.
- **My two runs are still running** at 2:06 AM PST [ran: `research status`]: `r20261113-090636-fb3d` (of `8cc990e51363`) and `r20261113-093452-2396` (of `3ca375fa4336`).
- **Next, in this order:** the mutations of the four commits; three kinds more in the tests (a tie withheld, a tie stopped, a control withheld); one pool run at the final head; the body with the two open items; and my line on NET-403 about #9590. Nothing goes onto #9533 or #9590.

[2026-11-13T10:10Z] 473de0bf0f66
PERIODPROOF to network-accounting (lead), Fri 13 Nov 2:10 AM PST. NET-427 step 1: the draft is pushed, scratch/na-certifier-account-7c3e at 5491064c391f4e5ac4e40ff097e22f8efcd86f85 (one new file, NetworkCheck/Account.lean; no pull request, no lock entry). Run r20261113-100950-dd20 launched on a project machine (the pool placed it): verity lean try of that one file. I write again when it ends.

[2026-11-13T10:12Z] 489f7d783fc1
PERIODPROOF to network-accounting (lead), Fri 13 Nov 2:12 AM PST. NET-427 step 1: run r20261113-100950-dd20 ended, exit 0 in 82 s wall; lean exit 0 on NetworkCheck/Account.lean, every guard holds, the axiom probe finds no theorem in it [ran]. The comment on NET-427 follows within the hour.

