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

## 02:20Z Thu 12 Nov: A CORRECTION to my earlier sections, and three more things Daniel asked for (6:10 to 6:11 PM PST)

**The correction.** I wrote that the replay refuses a cache hit. That is true only of the two-stage serve's path
(`boundary_rows.py:602` refuses to install with the cache on; `adapters/vllm/README.md:110`). **#8403 is on `main` since
Wed 11 Nov 15:23Z:** certified serving keeps vLLM's prefix cache on, and a sampled request's replay recomputes the prefix
its hit read from the request's own committed tokens, so the fan-out is 1 and serving is unchanged (`gh pr view 8403`:
`compile_steps(..., recompute_hits=True)`, `check.replay`, `recomputed_prefix`, `rebuild.py --shape-only RID
--recompute-hits`; 13.88% more computed FLOPs on one GLM-5.3 TP8 record; not covered: a 128k hit, and in class G a hit
whose producers ran in an earlier execution phase, whose statement shape was asked of proofs). `PREFIX_OFF = 0.1386` in
the cost model is the same ratio and reaches the rows of record as `reproved`. So your routes with the cache on are at
least two: the hit recomputed in the replay (on `main`, F = 1, the replay pays), and the hit read from a pool committed
once an execution phase (the replay is cheap, serving pays, F undecided). Price both. If your 03:00Z entry already says
this, good.

**Daniel's three additions to the goal:**
1. "It also should require as little advice as possible (ideally ~0 bits per request)." For each route give the advice
   in bits a request by class (hits, admission, release buckets, terminator, aborts, lengths, seeds, routing, the
   speculative accept lengths) and what removes each. Today's certified run, cache off: about 5.6 bits a request, by
   the code the lengths and finish reasons (goal `proofs/spot-advice-certified`).
2. "And we need its AOT circuit!" Say for each feature what the circuit compiled ahead of time must hold that it does
   not today (the draft layer and the accept rule; the hit's recompute chain or the pool's read), and whether the
   route needs a circuit that depends on the served record.
3. On the 131,072-token limit: "is this a hard limit? It sounds like perhaps we should be serving longer context...?"
   I answered: not hard, it is the size we compile and size tables at. Say what you find that contradicts that.

**A second worker, REPLAY** (bc-41f793c1-8f06-58b3-be91-09b2e5f545ee, branch `cursor/proofs-replay-<4 hex>`, entries in
`inbox/replay.md` there, first entry 03:30Z), compares the replay routes by cost (a whole execution phase from a
snapshot; one request with stand-ins that keep the others' shapes; a recomputed hit against a pool read). You keep the
prices of the two features on the four traffic shapes; do not do its table.

## 02:28Z Thu 12 Nov, from the coordinator: Daniel, 6:22 PM PST: no restriction on context length

- **His words:** "I think it would be pretty reasonable to shoot for saying that we have no restrictions on context length, max context length is fine. Does that seem within reach or no?" I answered: yes as a property of the design, not as a result shown this week; the code holds 131,072 in four places, and one cost grows with context.
- **What changes for you:** the context column of your table is the model's own limit, 1,048,576 positions (GLM-5.3's config of record, `max_position_embeddings`, plain RoPE with no scaling; circuits on CIR-186), beside 131,072. For each route say what grows with context: at serving (bytes committed and time a token), in the replay of one sampled span (tokens recomputed), in advice, and in what the guarantee charges.
- **The point I gave Daniel. Confirm or correct it with file and line:** a replay that recomputes everything before the span it replays grows with context, as it does with reuse. One that reads the earlier attention values from the request's table committed while serving does not, and that is what a 128k request's replay blocks already do (`proofs/spot-128k-attention-proved`: "A replay block is a span of positions inside the request, and it reads the earlier attention values from that table"). So a long context and a cache hit are one question: whether a replay block may read values another block computed, and what the guarantee then charges it (the fan-out F, GLOSSARY.md:362-364).
- **What I read tonight. Check each before you use it:**
  - `CLASS_G_LIMITS` and `CLASS_G_SERVED` hold `max_model_len` 131,072 (`adapters/vllm/verity_vllm/engine/profiles/engine_classes.py:33-36`).
  - The long row's mask Program spans 63 classes, 4,096 to 131,072 keys in steps of 2,048; its radix select runs over 32 + 17 bits; `RowsPad`'s pinned ranges end at 131,072 (CIR-186's comments).
  - The index logits are 86,944 + 837,038·N ANDs a row, in 18 of 78 layers (57 share an index); the select is about 193 ANDs a key (CIR-186, circuits' counts).
  - The unit that merges the selected positions exactly fills the largest block of 2^27 bits at 131,072 keys (vLLM's goal `vllm/served-128k-window-committed`; relayed).
  - One rank's compile took 28,360 s and 788 GiB at 131,072 tokens, and 110 to 150 min and 253 GiB at 40,000 (circuits' goals).
- circuits is asked the circuit's side on CIR-244 (Thu 2 PM PST); vLLM the serving side on VLLM-377 (Thu 10 AM PST).

## 03:17Z Thu 12 Nov, from the proofs lead: the route is chosen tonight at 07:00Z, and the serve is vLLM's published line

- **Daniel moved the goal to Fri 13 Nov 9 AM PST** (6:44 PM PST, on VLLM-377). vllm chooses the route with us at 07:00Z tonight. Your final at 05:15Z is what I bring: keep that hour.
- **Assume the serve is vLLM's published line for eight H200s:** FP8 KV cache, MTP speculative decoding with five draft tokens a step, Model Runner V2, prefix caching on (`recipes.vllm.ai/zai-org/GLM-5.3`, "FP8 on 8xH200 (standard)"). Their two-node line for long agentic contexts uses three draft tokens.
- **In your final, add one section:** what in your prices and in the statement for shared readers moves under that serve, and what does not. In particular: a speculative step emits up to six tokens a request a step; does the patterns' form with shared readers read steps or positions anywhere; and does anything in the bound read the page's element type.
- **Lead with one line I can read out at 07:00Z:** the statement to prove, its price against today's bound on your four traffic shapes, and the date you would give for it in Lean.
- REPLAY's final is in (`inbox/replay.md` on `cursor/proofs-replay-136d`, "Entry 2 and final"); it cites your sketch at about 1.5 times for 86 requests sharing a prefix and about 1,900 times for the sets' form. Correct either figure if it is not what you would say now.
