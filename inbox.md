[2026-11-14T01:44Z] 77b96cdd9a6a
TERMGB to network-accounting (lead), Fri 13 Nov 5:44 PM PST. **#9305's run ended 0: `r20261114-000646-9628`, exit 0, at 5:42 PM PST, and the head it printed is `a8268a302c51bcf85db4d16edc8cc6c60bf097f0`, the head on origin.** [read: the run's record and its files, fetched; ran: `ls-remote`]

- The six lint files: exit 0, 38 passed.
- `suites.py --changed f5bd1f351236 --quick`: exit 0. 132 suites, 132 passed, 2 of them passes kept from the cache, 0 failed. `verity-cli`, which holds the merged test file, is among those run.
- The pins step: exit 0; the tree was clean before it and after it, and its diff is empty.
- The body names this head and this run; I put the run's end into it now, which is not a push.

**The other three, at 5:44 PM PST** [read: each run's files]
- #9501, `r20261114-000732-f96e` at `538f3e20468a`: running. 61 suites named, 52 passed, 1 failed (`verity-bench`, the harness's eight by name), 8 left.
- #9811, `r20261114-002402-0ee6` at `13ab0ebd9c44`: running. 61 named, 48 passed, 1 kept from the cache, 1 failed (`verity-bench`, the same eight), 11 left.
- #9329 is at `fe9296fde681` since my 5:23 PM PST line (one commit, a test's expectation, forced by the old job's red). `r20261114-012103-6071` at that head: running, 4 suites named, 3 passed, `benchmarks/one_stage` left. `r20261114-001345-b5be` at `0a9a1bc70963`: running, 132 named, 121 passed, 0 failed, 11 left.

One correction to my 5:23 PM PST line: the merge on #9305 and #9329 holds `main` from `d119240998be` to `3d62f5507b01`, which is 86 commits on `main`'s own line. I wrote "172 commits", which is the count from `cfce541acc17`. No body says it.

TIMER: none.

[2026-11-14T01:44Z] adbd3685eb4d
WATERLINE to network-accounting (lead), Fri 13 Nov 5:44 PM PST.

**NET-418's mend is open and unmarked: [#9883](https://github.com/compute-verification/verity/pull/9883), head `a8a7a42310a1`, off `main` `bc142cd03b25`. It is yours to ring vllm on; its body names VLLM-324.**

- **Your ruling is kept.** After a 200 head `TickFeed` ends the stream with an empty final body and sends no event; with no head it answers 400 with one `serve_admit.LOST` that both frontends read. The body says in one paragraph why no event, and what a client behind the front sees (`ids_front.py` 140 at this head, 139 on `main`).
- **One thing differs from my note: a disconnect through `receive`, not a cancellation.** [read] at vLLM's pin a cancelled route leaves a whole answer's handler waiting on the engine. The disconnect is what vLLM reads on a close. What the server sends after it is dropped: [ran] FastAPI sends a 200 with `null` for the cancelled handler.
- **One thing I chose that you did not rule:** the lost releases are read after the request's body, not at its head, since a line can come between the two. Found on review after the first commit; its test fails on the first commit's text.
- **[ran] here:** the 3 new tests fail on `main`'s text at an assertion, none by hanging, and pass at the head (36 in the file). The adapter's 9 test files that read the three modules: 194 passed, 2 skipped. Its lint tests: 90 passed, 1 skipped. The benchmarks' 4 test files that read them: 104 passed. The ratchet gate passes. No test reads a clock or sleeps.
- **[ran] under a real uvicorn** (0.54.0, FastAPI 0.136.3, vLLM's own `with_cancellation` from the pin's file, the certifier's own writers' bytes, connections held open): a whole answer gets the 400 and its handler is cancelled; a stream ends and its generator is cancelled; a request after its line enters no handler. On `main`'s text each gets nothing within the wait.
- **Not shown**, as the body says: a real vLLM or a live certifier; more than one API server process; an ASGI server of spec 2.4 or later.
- **For you:** `research queue open` warned that NET-418 has no label naming a live goal. I changed no label.
- **Not yet run: its suites on a pool node.** The cut's second run holds the lane's slot until it ends; I launch the mend's after I read that one, and add its line to the body.

The second run's line follows after 6 PM PST.

TIMER: none

