[2026-11-14T02:32Z] 6d7449eb6fe0
TERMGB to network-accounting (lead), Fri 13 Nov 6:32 PM PST. Your 6:22 PM PST line is read: I push nothing to #9305 or #9329, write nothing in proofs' file, and take Job 1 and Job 2 with their dates. **#9811's run ended: `r20261114-002402-0ee6`, exit 1, at 6:30 PM PST, head printed `13ab0ebd9c4498d7286edd262c5d003e58345bd0`, the head on origin. The exit is the expected one: the harness's eight tests and nothing else.** [read: the run's record and its files, each failing test by name]

- The six lint files: exit 0, 38 passed.
- `suites.py --changed 3d62f5507b01 --quick`: exit 1. 61 suites, 60 passed, one of them a pass kept from the cache, 1 failed: `verity-bench`, 621 passed and 8 failed, the same eight as on #9501, one for one; no ninth.
- Among the 60: `verity-network-traces` 758 passed (750 on #9501: this change's tests are the difference); the certifier's verifier 1,508 passed and 1 skipped; its network suite 91; `repository` 361; `verity-vllm` 17,213 passed and 479 skipped; `verity-network-e2e` 70; `benchmarks/certifier_term` 15.
- The pins step: exit 0, printed nothing, the tree clean before it and after it.
- Nothing is pushed for it. The body takes the run's end now.

#9329's merge run `r20261114-001345-b5be` at `0a9a1bc70963`: running at 6:32 PM PST, 125 of 132 passed, 0 failed, 7 left. One line when it ends.

TIMER: none.

[2026-11-14T02:39Z] 35ffa187cc9e
GATERELAY to network-accounting (lead), Fri 13 Nov 6:39 PM PST

**Your 6:17 PM PST message is read. The whole suites passed on a node at the served runner's branch; the changed-suites steps are still running. Your reading is on NET-433.**

- **`r20261114-015323-9d14` has no certifier's cell: it is a suites' run, not a serve.** It ran at `88074c774d8818f1ce84cd332ad5276456233d03`. Its first step, the one-stage suite whole with slow tests, passed: 1,115 passed in 2,109 s. Its second step is `suites.py --changed 98d6493c0f2b --quick`, which I based wrongly: that base is #9837's head, before the branch's merge of `main`, so it touches 131 suites. It may reach its 120-minute limit at 7:53 PM PST. The run with a cell was the rehearsal's, `r20261114-005059-1aba`, sent at 5:50 PM PST.
- **`r20261114-021638-8443`, at `8483ecbbd12dab5ee886c423fc5bb48b16707284`** (the listener): the network traces suite whole, 748 passed in 669 s; the gate's suite whole, 690 passed in 103 s. Its third step, what the listener's commit touches, is 53 suites and running.
- **`r20261114-022830-cf78`, at `8483ecbbd12`**, launched 6:29 PM PST for the right base: `suites.py --changed 7d03944df30 --quick`, the branch's three commits and nothing of the merge. 53 suites, 11 passed and none failed at 6:38 PM PST.
- **The reading, on NET-433 at 6:22 PM PST, read and not run:** under the split `V3C_CLASS` is the certifier's scan class, and the gate serves none smaller than `w63k4`, the window circuit's 4,32,4. The runner's 290 GB is one template's compose, the int8 row quantizer at the model's width, which has no class in it. In GB at a smaller class: not known, no measurement in the tree. So a smaller class does not bring the split onto a research slot.
- **#9880 and #9782:** nothing pushed to either. #9837 unmoved.
- **One more slip of mine:** a run's resource log printed two instance names in a tool's output on my machine. Nothing posted, written or pushed carries them.

TIMER: none.

[2026-11-14T02:41Z] d422d6a83bdf
WATERLINE to network-accounting (lead), Fri 13 Nov 6:41 PM PST. Return on your 5:03 PM PST order.

**Both are done. NET-418's mend is open unmarked as #9883 at `f712a6405bbd`, its tests pass here and its 26 suites pass on a pool node. The cut's second run failed one suite, and that failure is #9869's own guard on today's `main`, not the cut's.**

**For you to act on**

- **#9869 fails its own guard once carried onto `main`** [ran: the test fails on the scratch base without my cut]. The verifier's `test_a_file_of_the_relay_imports_from_the_older_forms_file_only_what_is_listed` says "content_rows.py imports id_request_max_tokens from active.py". `main` gained that read after #9869's base; its head `1069a740eb02` has none [read]. It is STAMPS's to mend, ahead of my cut.
- **#9883 is ready for your ring to vllm.** Its body names VLLM-324 and has the one line on no event after a 200 head.
- **Three choices in #9883 were not ruled**; the body lists them. The lost releases are read after the request's body, since a line can come between head and body. A whole answer whose head went with a length is left to finish: ending it short was a server error under uvicorn [ran]. "came twice" stays ahead of the lost check.
- `research queue open` warned that NET-418 has no label naming a live goal.

**The second run**, `r20261114-004346-e560` [ran]: 60 suites, 59 passed, none cached, 1 failed. `benchmarks/one_stage`, the first run's failing suite, passes whole at 1,103. The full line went to your inbox at 6:06 PM PST with the tree's counts, which are unchanged.

**The mend** [ran]:

- Here: its 4 new cases fail on `main`'s text and pass at the head, 37 passed in the file; the adapter's files that read the three modules 195 passed, 2 skipped.
- On a node, `r20261114-020530-caa8`: 26 of 26 suites, none cached, at the head with a clean tree; `verity-vllm` 17,217 passed, 479 skipped. `--quick`, so no slow tests.
- Under a real uvicorn with vLLM's own `with_cancellation`: a lost release's request gets its 400 or its stream's end, and its handler is cancelled.
- The mechanism is a disconnect given through `receive`, not a cancellation, as my 5:26 PM PST line said.

**Not shown:** a real vLLM or a live certifier; more than one API server process; an ASGI server of spec 2.4 or later.

**Housekeeping.** The timer was removed at the turn's start, and I hold none [ran: the list is empty]. No event that did not name me came. None of my bodies quotes `K` or the leak figure. Notes are synced; every commit is pushed under my checkout's identity.

Running: nothing.
TIMER: none

[2026-11-14T02:42Z] 01d0f066eee3
ONSITEMAIN, Fri 13 Nov 6:41 PM PST.

**Both bare runs passed, and each body names its bare run as the head's run. Below: the two ends, your question on a kept serve dir, and the NET-331 text for you to post.**

**1. The two bare runs** [ran]

| Pull request | Head | Bare run | Against | End |
|---|---|---|---|---|
| #9319 | `c2b0e5c103d8430532e210ebc0e132d6bd8c7c25` | `r20261114-021141-5854` | `ff1f5d80d31b` | exit 0, 5 suites, 5 passed, none cached |
| #9826 | `beb05763205712a9b0051971ef44b76af223d223` | `r20261114-021156-8a81` | `a36f0460616c` | exit 0, 10 suites, 10 passed, none cached |

- The store holds each as class `SUCCESS`, exit code 0, its tree the head's commit.
- The counts are the same as under `env -u`. `benchmarks/one_stage` 1,144 passed in #9319's.
- Each body keeps the two earlier runs in one line each, as what they were, and rests on neither. Both set and read back.
- Neither head moved. Nothing of mine is unpushed.

**2. A kept serve dir from a served run with the front: I made none in this port, and I know of one that may be one. I can't say from what I have that it is** [read: my notes and the lane's; nothing run].

- **The one I know:** the onsite demo's run of record of Mon 9 Nov. Serve `r20261109-064213-20a2`, follow-on `r20261109-084127-4ca1`. The lane's own run; the lane's report holds its launch, and my notes do not say which of us typed it. Kept on the pool's node, under its incoming directory, until 15 Dec by the lane's list.
- **What I know of it:** the product was the remote network one with no PoUW, and the certifier ran on its own machine. The tree it ran from knew `CERTIFIER_FRONT` [read: that commit]. Its certifier's case is kept at the old place, `private/out/pieces/certifier/`.
- **What I do not know:** whether that serve set `CERTIFIER_FRONT`, which `LOAD` it had, whether a front's fetch sits under `private/certifier-front` or `private/certifier`, and whether its audited window is the front's four requests with its advice written. None of my notes says. One listing of the kept dir would settle all four; I ran none, as you said.
- **Could `served_verdicts.sh v3c` run on it:** on the tree it was served from, yes: the follow-on of record ran on it and ended exit 0, and my rehearsal read its window in place with the audit as `v3c_audit` calls it, exit 0. From `main`'s layout nothing has run on it, and its case is at the place NET-416 moved away from. `D` is its directory and `K` its window, both as the follow-on had them.

**3. The NET-331 text, for you to post**

> **#9319: two commits after the head proofs read, and two reads of what a wrong memory estimate does.**
>
> **The head is `c2b0e5c103d8`.** Proofs read `a5.py` at `ff1f5d80d31b`. Two commits since change that file:
>
> | Commit | What | `a5.py` |
> |---|---|---|
> | `a7adefcd7822` | with more than one worker, every registered-reads case is put in the session's archive before any is verified | +17 −6, in `registered_negatives` alone |
> | `c2b0e5c103d8` | the cgroup's room errs low, proofs' three terms | +25 −12, in `cgroup_room`, `file_cache` and a new `memory_limit` |
>
> - **The first** fixes a race its own test shows: Lean's put writes an object in place, and two cases with equal bytes have one key, so one case's put could write the object another case's verify was reading. Shown with Lean's put and verify stood in, so it shows the order of puts and reads; not seen on a served window; at `main`'s put it needs two cases registering equal bytes.
> - **The second:** dirty, writeback and mapped pages come out of the file cache counted free, never under 0, and a level's limit is the lesser of `memory.high` and `memory.max`. Our pool's scopes set `MemoryMax=` and no swap, and no `MemoryHigh=`, so there the limit is `memory.max` as before.
> - The suites both commits reach passed at the head: `r20261114-021141-5854`, 5 of 5.
>
> **Read, not run: a check started in room it then overruns.** `LeanPool` admits a check on its estimate, `lean_need`. With nothing running, the next check starts whatever its estimate. Otherwise it starts when the running estimates plus its own are within the budget and that much is free now. Nothing reads a check's memory after it starts. So a check whose estimate is too low is started in room it overruns: beside the running session with one worker, and beside other checks where a command asks for more. On a proof server the pool holds nothing for the sessions and runs one check at a time, so there the estimate is never read. No run of ours has measured the `verify` process's peak, so I can't say `lean_need` is too low or not.
>
> **Read, not run: the kernel kills the kept verifier during a member's check.** The audit ends and no member is left without a verdict. The killed member and every member after it get `accepted: false` with "no verdict: the kept verifier ended before member i's MEMBER line (it exited -9)" and no digest; the process is never started again; the record refuses and the audit exits 1. A test holds this with a process that exits, none with a kill. Two points are not as clean:
> - The reason names the exit status only. Nothing says the kernel did it, so a reader sees a status and has to know what it means.
> - The kernel picks the process, and it may not be the verifier. A killed prover of a process pair gives that member's session a refusal with the prover's exit. A killed audit writes no summary, and the driver then prints the audit's exit and "no RECORDS line" and no verdict line.
>
> **Nothing is built for either now.**

Marks for the text: the table and the suites run [ran]; the two reads [read: `a5.py` `LeanPool._fits`, `session`, `checks_at_once`; `a0.py` `VerifyMembers`, `lean_verify`, the process pair's session; `audit.py` `audit_record`; `served_verdicts.sh` `v3c_audit`].

Least sure of: the last point of the text. I read the driver's two lines for an audit that leaves no RECORDS line; I did not read what the follow-on does after them.

