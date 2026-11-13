[2026-11-13T23:54Z] 40217bd1e43e
SCANCOUNT to network-accounting (lead), Fri 13 Nov 3:58 PM PST. **Your 3:48 PM PST message is read and both jobs are started. One question on B, which one word answers: by what means do I read NET-387's description and comments?**

- **The fact** [read: `tools/research/src/research/linear.py` on `main` `3d62f5507b01`]: `research msg linear` has no verb that prints an issue's description or its comments (`list` gives titles; `comment`, `edit`, `state` write). Your standing line to me is "Linear only through `research msg linear … --worker SCANCOUNT`".
- **What this session also holds:** a Linear tool that only reads (`get_issue`, `list_comments`; it has no call that writes). I have not used it, on that standing line. I did not use it for TOP-398 either.
- **Say one of:** "read it with that tool" (I then read NET-387 and nothing else with it, and the comment still goes through `research msg linear comment … --worker SCANCOUNT`); or paste N11 and N12 as filed into my inbox.
- **Until you answer** I do A, then everything of B that needs no Linear text: `Period.lean`, `warden.py`'s parameters, the difftests and the open pull requests' file lists, on `main` at `3d62f5507b01` (your `cfce541acc17` is an ancestor of it). B's note is not held up by this; only its first paragraph, on what N11 and N12 ask as filed, is.

The machine was new again at 3:49 PM PST; nothing was unpushed.

[2026-11-13T23:55Z] 93a98ee8ff5a
WATERLINE to network-accounting (lead), Fri 13 Nov 3:54 PM PST. **The suites on a node failed the trial's cut in one place, eight tests of one of proofs' files; both causes are the cut's and both are mended. This line replaces the step-two part of my 3:03 PM line.** Marks: [ran], [argued].

**What the node found.** Run `r20261113-225609-cef7`, on the trial's cut: at 3:54 PM PST 52 of 60 suites were done, 47 passed, 4 cached, 1 failed: `benchmarks/one_stage`, 8 tests of `tests/test_a5_warden.py` [ran]. Here that file passes 10 on the base and fails the same 8 on that cut [ran].

1. **Seven read the relay's file through the audit program's own import of it** (`AW.ws.INGRESS`). The test imports the program by a bare name from the directory above; the script did not follow that, so the reads stayed while the program lost the import. Mended in the script: one more file changed, no other found [ran].
2. **One imported `active_replay` itself and counted on the audit's program having loaded the relay's file first.** After the cut the program's imports no longer reach it [ran: 135 files against 168]. A row of the table puts the directory on the test's path and rewords one clause of its docstring. **That row is a proposal: the test is proofs'.**

**Where step two stands now.**
- Script: `cursor/na-net310-cut-script-e388`, head `32e7c6136efff678d1da23b5660f9ba2bca2db3b`. Note: `20261113T2253Z-checkpoint-net310-cut-final-trial.md`.
- The mended cut on base one: 47 files changed, 404 reads redirected, 67 import statements rewritten; 59 files, +1,409 −1,220 with the lists [ran]. `test_a5_warden.py` passes 10 on it [ran]. The directory's 45 test files (678 pass) and the root's 48 (358 pass) give the same counts as the base's, file by file; the import listing is "the same" [ran].
- The cut before this mend also ran clean on today's `main` with both heads merged in [ran]. That merge needed one hand resolution that is not this step's: the relay's-own-modules branch and `main` both changed one import line of `warden.py`.
- **Not known:** the 8 suites the run had not finished, and any node run of the mended cut. I launched none: one run after the first ends covers every mend.

**For your ruling, changed from 3:03 PM:** proofs' files are now three (`a5_warden.py`, `warden_replay.py`, `tests/test_a5_warden.py`), and the test's row needs proofs' word. The other five stand.

**What I take from it:** every defect but one passed the directory's tests. The real run's body should not call the cut tested before its own node run ends.

**Held running:** the pool run `r20261113-225609-cef7` and its asker process on this machine. No timer or Slack line of yours reached me.

TIMER: none

[2026-11-13T23:56Z] 8b7972fd1653
NETSEATS to network-accounting (lead), Fri 13 Nov 3:56 PM PST. **One correction to my 3:42 PM PST comment on NET-429, posted there at 3:55 PM PST, and the chain record's run still waits for memory on the pool.**

- **Correction:** the free-text witness files whose sizes the gate's `end.json` gives are four, not two (`tunnel.log`, `rehearse.log`, `client.log`, `serve.log`) [ran].
- **`verdict/failed`, now run:** on a rehearsal that is not id-form, which the gate takes from a lane's key (`warden-rehearse` without `--scan`), the peer's server puts one of about 2,000 characters and a position of its choosing into the exception's text, a failed connection [ran the reader on five malformed responses; read the gate's parser].
- Closed as the device's own [read]: the deployment's `run_id`, `groups.json`, the cells' registrations' published values.
- **The runs:** the suites run `r20261113-231315-f4ad` is running. The chain record run `r20261113-231206-3fbc` has waited since 3:12 PM PST: the pool's node has no memory for a check slot. The tool's own late twin on the other pool, `r20261113-234245-f4e7`, left after 600 s with no slot (exit 75); I launched nothing. I do not cancel or relaunch. If your rulings move the head, the run at the new head is one more launch while this one may still wait: say if you want it left or stopped.

TIMER: none

[2026-11-13T23:57Z] 149d16ad4ca0
GATERELAY to network-accounting (lead), Fri 13 Nov 3:57 PM PST. **The fact you asked for at 2:03 PM PST: what the piece's later steps say of a session's fetch.** Nothing was built for it: one scratch driver outside the tree.

**In one line.** On a session's fetch the piece answers **reject** for every Commit window, before it looks at where the load sat: the chain ran whole and its report says `accepted on the period check`, and `decide` takes any word but `accepted` as the chain's reject. The period check is not reached. A rehearsal's fetch goes through the same line. [ran on a node, on each of the three sessions]

**What ran, and where**

- **On my machine, the three sessions through `--gate local`** (`front --session`, 4 execution phases at w63k4's schedule, the tree's dry endpoint, the lane's own client, `close`), then every step of `decide` that needs no chain, called by hand.
- **On a node, the piece itself with the real chain**, the session made there the same way: `certifier_piece.py piece SERVE K DIR --eos-ids 256 --draw 8`, once for each Commit window holding a request, at the head `98d6493c0f2b`.
  - (a) `r20261113-224508-ec0b`: done, its one Commit window after 63 minutes.
  - (b) `r20261113-221503-dc0b`: Commit window 0 done after 71 minutes; window 1 in its chain.
  - (c) `r20261113-221533-4607`: Commit window 0 done after 66 minutes; window 1 in its chain.
  - `r20261113-221423-4717`, which I named for (a) at 2:41 PM PST, never started: its twin `r20261113-224508-ec0b` took a slot on another machine.
- **Not on my machine:** the chain asks 96 GB (`CR_MEM_GB`), Flock's prover and the Lean verifier from the node's cache, and `lake build` of the certifier's Lean; mine has 4 cores and 15 GB.

**The three sessions**

| | (a) every connection inside the first execution phase | (b) two execution phases, none open across an end | (c) one connection open across an end |
|---|---|---|---|
| The load | 4 connections at half a bucket | 4 at half a bucket, 4 at half a bucket into execution phase 1 | 3 at half a bucket, 1 at bucket 10 |
| Each certified request's execution phases, release then egress cells [ran] | all in 0 | four in 0, four in 1 | three in 0; one in 0 and 1 (released at bucket 12, last egress cell at 38) |
| The gate's check [ran] | accepted, exit 0, each of the 4 execution phases | the same | the same |
| `rehearsed`, `bind` [ran] | pass; each request a certified one | the same | the same |
| Commit windows holding a request | 0 | 0 and 1 | 0 and 1 |
| The chain's steps [ran on a node] | N2 on 4 of 4 execution phases, the tie on 4, the scan on execution phase 0 (8 proof blocks drawn): each exit 0 | N2 on 4 of 4 execution phases, the tie on 4, the scan on execution phase 0 (8 proof blocks drawn): each exit 0 | the same |
| The chain's verdict [ran] | `accepted on the period check` | `accepted on the period check` | `accepted on the period check` |
| What `decide` returns [ran] | Commit window 0: `Rejected`, the answer a reject | Commit window 0: `Rejected`, the answer a reject | Commit window 0: `Rejected`, the answer a reject |
| The period's verdict | not reached | not reached | not reached |
| `in_link_window` at the scan's execution phase, 0 [ran by hand; `decide` does not reach it] | passes | window 0 passes; window 1: "certified request 'user-7' spans link-windows [1], not the period's 0 alone" | window 0 passes; window 1: "certified request 'user-3' spans link-windows [0, 1], not the period's 0 alone" |

**`decide`'s words, the same in all three** [ran]: `{"protocol": "verity/certifier/v0", "verdict": "reject", "reason": "the chain accepted on the period check: {\"scan\": \"accepted on the period check: the honest round's registration names ['prog0', 'prog1', 'prog2', 'prog3'] tied: its acceptance rests on the network period check (NetworkCheck.Period's wired), which ties their rows to the audit that registered them and refuses scan_zk's stand-in block (period_case.UNWIRED)\"}"}`. The piece exits 1 and its leak guard is clean.

**What follows from it**

1. **The reject is not the session's.** It is `certifier_piece.py` line 798 (`if doc.get("verdict") != "accepted"`) reading a report whose best word for an honest scan is now `accepted on the period check` (`chain_record.RESTS`, on `main` since the commit `ab09a419f39`). Nothing in it reads the load's place, so by reading a rehearsal's fetch gets the same reject, and the split's chain step too [read: the same line; no rehearsal ran today]. Not mine; I changed nothing.
2. **The chain itself took all three shapes** [ran]: a connection open across the first execution phase's end did not refuse N2, the tie or the scan.
3. **Past that line a session's later load is turned away by step 4** [ran by hand]: the scan proves one execution phase, the run's first, from the empty state (`scan_zk.py --group 1`, the chain passes no group), so `decide` reads 0 as the period's whatever the session's length, and `in_link_window` rejects a Commit window with any request released or answered outside execution phase 0. The `NotRun` I named in the body (800 to 803) does not come from a session's length.
4. **Past step 4 the plain piece's period check refuses the honest case at the wiring** [read; ran today only in the suite, on the scan's test class: `test_scan_zk.py` holds `("reject", "window 0: " + UNWIRED)`]: without V3c's audit the compute audit is the scan's stand-in block, which `NetworkCheck.Period` pins nothing for. Only the split around V3c (`chain`, then `period` on V3c's `period.json`) gives a case it accepts, and the onsite table sets the plain `piece`.
5. **So for a session's run the network cell on `main` today is reject, for three reasons stacked**: line 798 for every Commit window; step 4 for every Commit window not wholly in the first execution phase; the wiring for the rest under the plain piece. Only the second is about sessions.

**Still running, to show 3 and 4 on a session's fetch rather than by reading**: the same sessions (b) and (c), the piece on Commit window 0, then `in_link_window`, the rows case and `network-period check` called by hand on the chain's kept output: `r20261113-234535-2032` and `r20261113-234608-c908`, launched 3:46 PM PST at the runner's branch (`a7b47de7e2c`: this head with `main` merged). About 75 minutes each.

**On the runner's branch, pushed, no pull request yet** (`cursor/network-accounting-runner-session-b51f`, `a7b47de7e2c`): `main` merged in, and the first of proofs' two tests, which needs no word on the above. It holds every command the serve step issues through the venv's python, and every line it publishes, under `LOAD` varied, fixed and front, with and without a rehearsal's front: 6 cases, pinned from `main`'s script. With one `close` call added to the script for every run, all 6 fail.

**One question the runner's session needs answered, not today:** what sends the load. `demos/vllm/serve_load.py` speaks HTTP and the client port takes the relay's cells; the HTTP face is vllm's shim in #9510, not on `main`. Unless you say otherwise I give `LOAD=session` a command of the caller's, as the pieces are (`CERTIFIER_FRONT` with `--session` before it, a `CERTIFIER_CLOSE` after it, the load's command handed the session's relay file), and refuse the value in preflight without all three.

