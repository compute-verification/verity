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

## 02:22Z Thu 12 Nov: Daniel's ruling (6:02 and 6:03 PM PST), which changes what your final is for

Daniel: "it really just does seem like we need this to work with prefix caching ... Someone needs to figure out how to
get this to work. Let's make this a tier 1 goal", and then: "I think this tier 1 goal should be like: production vLLM
is reproducible at <1% serving-time overhead. And so that will be an agentic workload simulation, and all of the
prod-vLLM settings (with as few changes as possible)."

It is on the Goals page: `vllm/production-vllm-reproducible` (owner vllm on my recommendation, brief VLLM-377), and
ours under it, `proofs/cached-serving-audit-design`, due Thu 12 Nov 6 PM PST. Your note is that design's first draft.
So, on top of the three columns of my 02:00Z section:

1. **Caching off is no longer a route.** Price it once, as what we do today, and spend the rest on routes with the
   cache ON and speculative decoding ON.
2. **For each route say what it costs at SERVING, against 1% of serving time:** what serving must commit or record
   that stock vLLM does not (per cached block read, per speculative step, per execution phase), in bytes hashed and in
   seconds where the tree has a measurement (`benchmarks/serve_tax`, the commitment's own timing rows, vllm's goal
   says -0.1% [-2.1, +2.0] with the serving state committed every 30 s), and whether that cost grows with the share of
   prompt tokens that repeat. A route whose serving cost you cannot bound says so.
3. **"Reproducible" is the test:** a sampled replay block re-executed from what serving committed gives the served
   tokens bit for bit. For a cache hit say exactly which committed values the replay reads in place of recomputing,
   and for a speculative step what the replay must know (the draft tokens, the accept rule's inputs) to land on the
   same tokens.
4. **The load is agentic:** AgentX's shape is the first column now, our plan the last. More than half of AgentX's
   requests are over our 131,072-token limit; say what each route needs from the context limit.
5. **End with one recommended route and its list of changes by lane** (vllm's serve and replay, the cache step, the
   guarantee's statement, the cost model), each with what you read that makes you think it is possible, and what you
   could not decide.

The hours stand: entry 03:00Z, final 05:15Z. If the wider job does not fit, the order is 1, 3, 2, then 5, then 4.
