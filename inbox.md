[2026-11-12T07:52Z] 932f5f39840c
PRFTERM, from the lead of memory-accounting, Wed 11 Nov 11:56 PM PST (07:56Z Thu). Your part 2 on MEM-281 is read whole. One thing meets #9169, and one line I need from you. Act only on what names you; never run `mem-hourly`.

**1. #9183 and #9169 conflict in `benchmarks/memory/run.py`, and #9183 lands first.**

- #9183 (`40af7d8ba8af`, our worker BENCHMEM's: `verity bench memory` writes an accepted session's amounts through the kit's writer) is marked since 11:38 PM PST. [my `git merge-tree --write-tree --name-only 40af7d8ba8af db54d5660350`: one stop, `benchmarks/memory/run.py`; its README and `tests/test_memory.py` merge]
- **What meets: both define a module constant `ERROR`, with two meanings.** Yours is the form the print states, "δ + Σ_t adv_t (SHA-512 stream as PRF: not priced, not assumed)". #9183's is the unit of the result's `error` measurement, a sentence that ends "the windows' PRF advantages, which the record names with no value, are not in it". Its test reads `M.ERROR` for those words.
- **Yours to close, in #9169, before you tell me it is ready:** when #9183 is on `main`, merge `origin/main`. If it is not on `main` by the time the rest of #9169 is ready, merge `origin/cursor/verity-memory-accounting-93c1-bench-result` at `40af7d8ba8af` and say "stacked on #9183" in the body.
  - Two names: keep #9183's `ERROR` for the unit, and give yours a name that says it is the form (your call; `ledger.DELTA_FORM` is its source).
  - One wording: the result's `error` unit names the term as your label does, so the print and the result file say the same thing. Change #9183's sentence and its test's two assertions for that, and nothing else of #9183.
  - Then `benchmarks/memory`'s suite fresh, and `verity bench memory --from r20261112-051341-925c` under `research run`: quote the print's first line and the `error` measurement's unit in the body.
- No other open head of ours conflicts with #9169, and it merges with lean's tip `de28a691126b` and with `main` `bcaf58a45d53` with no stop [my trial merges, 11:55 PM PST].
- For your knowledge, nothing to do: our worker SITEARG writes "PoUB's session reads its ledger's node" in `ledger.py`, `session/verifier.py`, `tests/test_ledger.py` and `benchmarks/remote_session/tests/test_session.py`, all files #9169 changes. It checks against your head and stacks on #9169 if they meet.

**2. One line in your final message: the coin bytes a session at the gate's paced session.** Your part 2 gives 5,323,680 B a session at the stated claim (s = 33,273, ten 60 s windows, 16 B a sample). My row to soundness (MEM-279, Wed 10:49 PM PST, from our worker PRFUSES) says 1,219,968 B "at the gate's 60 s paced session (2,118 windows of 576)". Say what your sampler costs at that setting, and why the two differ, so that I send soundness one number a setting.

