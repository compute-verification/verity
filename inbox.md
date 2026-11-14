[2026-11-14T00:57Z] 31304fd442f2
WATERLINE to network-accounting (lead), Fri 13 Nov 4:56 PM PST. **All three are done: NET-310's comment with the three diffs is posted, the first run's line is sent, the second run is on its slot. NET-418's read is posted too. I woke on a new machine; nothing pushed or synced was lost.**

**NET-310's comment** (4:40 PM PST): the mended cut's change to `a5_warden.py`, `warden_replay.py` and `tests/test_a5_warden.py`, three fenced diffs, 168 lines, starting with no `@`. It is taken on the second run's base; with it `test_a5_warden.py` passes 10 [ran]. A worker's comment rings no one.

**The first run** `r20261113-225609-cef7` ended 4:16 PM PST, exit 1: 60 suites, 59 passed (4 cached), 1 failed, `benchmarks/one_stage` at the 8 tests since mended. Its last eight all passed. 67 minutes from its slot [ran: its output read].

**The second run is `r20261114-004346-e560`**, on its slot since 4:44 PM PST, limit 7,200 s from the slot. I launched `r20261114-004208-fb20` at 4:42 PM PST; the launcher started a twin on another pool, the twin took a slot first, and the first id ended with exit 75 without running [read: its last line]. One run, one slot.
- **Base** (local, never pushed): `main` `3d62f5507b01`, then #9754's head (clean), then `1069a740eb02`. With #9754's head in, four files conflict, not two, all in import blocks:
  1. `connection_rows_run.py`: the branch's `blob_files`, `certifier_key` and `run_clock` clock; `main`'s `id_request`, which `main` now reads there; #9754's path line.
  2. The verifier's `warden.py`: both, the branch's line with `RUN_ID_BYTES` and `main`'s `to_watermark` line.
  3. `load_relay.py`: the branch's `certifier_key`; nothing reads the other side's name.
  4. `warden_serve.py`: the branch's three modules, and the load relay by package name as #9754 loads it.
  The recipe is in the notes (`tools/net310/cut/resolve_base.py`); run again it gives the same tree [ran].
- **The cut on it:** no refusal, script unchanged at `32e7c6136eff`; 47 files changed, 410 reads redirected; 59 files, +1,421 −1,232. The retired terms' list goes 105,541 to 105,548, the tree's count 6 fewer [ran].
- Before launch, here: seven test files of the hand-merged files, 193 passed; the import listing "the same" [ran].

**NET-418's read** (4:49 PM PST, 99 words on the issue; whole in note `20261114T0047Z-read-net418-first-fault-a-lost-release-under-tickfeed`):
- `TickFeed` still passes a lost release's request on, before its line and after it, and answers none (`tick_plugin.py` 157–162, 166, 182–200) [ran: a probe at `main` with a server that waits; read].
- "Nothing ever ends it" no longer holds: vllm's #9708 is on `main`, so the request ends with its connection, if vLLM aborts at a close, which nobody has run.
- The three files are vllm's in the registry; vllm's lead passed this fix to us on VLLM-324. The mend is my next job: about 50 lines of source in three files, 80 of tests.
- One choice I take unless you say otherwise: where the server's 200 head has gone, `TickFeed` ends the stream with no further event.

**`K` and the leak figure:** none of my ten bodies, and no line they add, quotes either [ran: a search of the bodies and the diffs' added lines]. Your 4:08 PM line reached me at 4:22 PM; I use the corrected sentence. No timer or Slack line of yours came.

**The draft body** holds the six rulings (8 TO-FILL lines). Notes synced at `fcc0865116`; no commit is unpushed.

Running: `r20261114-004346-e560`.
TIMER: `waterline-cut-second-run`, once, 6:05 PM PST, to read that run and send you its line.

[2026-11-14T00:57Z] 28ec9bfa4b10
network-accounting/GATERELAY to the lead, Fri 13 Nov 5 PM PST. **Your three questions of 4:08 PM PST, ahead of the return, which follows when two runs that end in the next minutes have ended.**

**1. Is the piece's reject at `accepted on the period check` on `main` today for a rehearsal's fetch too? Yes, by reading; not yet as ran.**

- `decide` compares the chain's report with the bare word: `if doc.get("verdict") != "accepted": raise Rejected(...)` (`certifier_piece.py` line 586 on `origin/main` at `bc142cd03b25`).
- The report's verdict for an honest scan is never the bare word. `chain_record.scan_step` gives `RESTS` whenever the honest round's registration names tied entries, and it always does: the tied names are a function of the scan's Program, query and verifier (`scan_zk.program`, `registration._tied`), not of the load. So the reject reads nothing a session differs in.
- Since `ab09a419f39` ("scan_zk/chain_record: the scan's acceptance rests on the period check", Mon 9 Nov 10:31 PM PST, on `main`). `decide`'s comparison is `f009a59849f` of Tue 27 Oct, unchanged since. Both by `git log -S`.
- No test holds it: no test of the piece or of the served runner hands `decide` a report with that word (`git grep` on `main` finds it in `chain_record.py` alone).
- Ran: sessions only (the three, each rejected there). A rehearsal's front with the real chain is on a node now, `r20261114-005059-1aba`, launched 4:51 PM PST, about 75 minutes; I send its line when it ends.

**2. Which side is right: the chain's report.** The word was made on purpose, to say that the scan's acceptance alone is not an acceptance: it rests on the period check, which ties the lanes' rows to the audit that registered them. `chain_record.py`'s own exit code already takes the word as a pass (its line 1232 on `main`). `decide`'s next steps are that period check, so `decide` is the stale side: it stops at the word instead of going on to the check the word names.

**3. The size of the mend: one comparison.** In `decide`, let `chain_record.RESTS` go on as `accepted` does (two lines), and two cases in `test_certifier_piece.py` (a report with the word goes on to the period check; `refused` still rejects). No trusted file: both files are in `benchmarks/network_traces/`, and `served_session.py`, which the piece imports, already imports `chain_record` inside its `chain`, so the piece reading `chain_record.RESTS` opens no edge the listing lacks (I run `tests/test_boundaries.py` before I say so of a written change).

**What the mend does not give, by reading:** the plain `piece` then reaches the period check, and for an honest run the check still refuses, at the wiring ("UNWIRED": the compute audit is the scan's stand-in block). Only the split around V3c (`chain`, then `period` on V3c's `period.json`) has a case the period check accepts, and the onsite table sets the plain `piece` as `CERTIFIER_CMD`. So the mend moves the reject from line 586 to the period check; an accept in the network cell needs the split in the table too. Two by-hand runs on a node are reaching that period check on a session's first execution phase now (`r20261113-234535-2032`, `r20261113-234608-c908`); their word is in the return.

Not mended in #9837, whose head is `98d6493c0f2b348e0661ae7863d25b7518467a0f`, unmoved.

**Also:** the first wording of soundness's sentence I wrote nowhere. The body of #9837 carries the 4:21 PM PST sentence and proofs' two facts (edited 4:49 PM PST, no push). #9667 is on `main` by git (`6ce9c470081e` is an ancestor of `origin/main`), so I am merging `main` into #9782 now, as your 10:41 AM PST line says, and send its head.

