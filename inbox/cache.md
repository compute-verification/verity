# CACHE's inbox (from the proofs coordinator)

## 02:00Z Thu 12 Nov: Daniel's follow-up question (5:57 PM PST), which your final must answer first

Daniel, in his words: "Is the problem just fundamental or is it in our demo and the fact that our demo is at a weird
scale, is like at a different scale from what we're trying to extrapolate to? I'd really like to hold these two things
separate. Let me know whether this is an extrapolation problem or like a problem with our proof system."

So your note and your message for him keep three things in three separate columns, for prefix caching and for
speculative decoding each, and say which of the three each cost belongs to:

1. **The proof system and the guarantee as stated on `main`.** What Flock, the sampling design and the guarantee's own
   statement cannot do or must charge whatever the traffic is. The one candidate I know: the Glossary's fan-out (a
   served cap of S bits charges each replay block at most F S when replay blocks read each other's values). Decide,
   with file and line: does a pool committed once an execution phase (the cost model's pool unit, `CacheStep_v3` on
   `MemCheck_v1` and `BlockKey_v1`, `POOL_WINDOW_ANDS` in `benchmarks/spot_figures/breakeven/sweep.py`) leave every
   replay block closed, so F stays 1, or is a read of the pool a read of another replay block's value under the
   definition? If the Lean decides it, name the definition. If nothing decides it, say "open" and say what statement
   would. This is the one line Daniel most needs, and "I do not know" with the reason is an answer.
2. **What the demo's code does today,** which is neither of the others: the two-stage serve turns caching off
   (`boundary-rows`; `engine_profile.py:146-156`), the replay refuses a hit, the capture map builds no draft layer, the
   one-stage check does read a hit (`prefix_reads.py`). Note that the serving-cost target for GLM-5.3 was written with
   the cache on (`benchmarks/serve_tax/README.md`, "the snapshot bundle ... with vLLM's prefix cache on") and the
   two-stage arm departs from it.
3. **The extrapolation,** which I told him is one of traffic shape and not of machine count: every price we have for
   caching (off 13.86%; a pool a phase 4.37% at 600 s and 1.10e11 to 3.05e11 ANDs) is at the 128k plan's reuse, about
   12% of prompt tokens, and lab traffic repeats 86% to 98%. For each price say which input is the plan's and what it
   becomes at each of your four traffic shapes, or that it cannot be computed and what is missing.

I told Daniel at 6:00 PM PST: not the proof system; the demo's choice (caching off) is cheap on our traffic and does
not extrapolate to agent traffic; one question in the guarantee's accounting (item 1) is open and you answer it by
10 PM PST. If you find that wrong, say so in the first line of your 03:00Z entry and I correct it to him at once.
