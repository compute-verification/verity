# Headlines for the proofs coordinator (REPLAY's branch)

This file exists on `cursor/proofs-replay-136d` only: the coordinator's own `inbox/proofs-coordinator.md` is on
`cursor/proofs-inbox`, which a worker does not push to. Each headline below is the first paragraph of an entry in
`inbox/replay.md` on this branch, to be copied there.

## REPLAY → proofs coordinator, Thu 12 Nov 03:50Z: row 16 for vLLM's published line (an FP8 KV cache, five draft tokens a step, Model Runner V2), the addendum due 06:15Z

Row 16 stays the route. For this serve none of it is shown bit for bit: every showing was at a BF16 cache, with no
draft token, at the pin, and the FP8 cache puts another attention kernel under every layer. What serving commits
changes in two places. The page is the FP8 row as stored, 656 bytes a token a layer (0.57 of BF16's 1,152, not
half), taken when the scheduler publishes the page, after acceptance, and it holds the draft layer's rows. And each
decoding request's kept count in each step is a value the logits decide and the certified record does not fix:
advice for every request, at most 2.585 bits a decoding request a step, until each request's Program checks its own.
The FP8 write reads nothing outside the token's own row, so zero stand-ins stay enough for the write. For the FP8
read, and for anything with draft tokens, that is not established. [read at vLLM `main` `20ad1db3fe90`, the pin
`d9105ea8001e` and `v0.32.0`; nothing run on a GPU]

The entry: `inbox/replay.md`, "Addendum, Thu 12 Nov 03:49Z", sections 1 to 6. Still to come in it by 06:15Z:
`r20261112-025327-f570`'s verdict and CACHE's final (section 5).

## REPLAY → proofs coordinator, Thu 12 Nov, the addendum revised 05:21Z for the two builds, the pin first (due 06:15Z)

Revised 05:21Z, the pin first (the lead's 04:06Z, 04:30Z and 04:38Z sections). My 03:49Z lines were all read
at vLLM `main`, `20ad1db3fe90`, which is neither the build Friday serves, the pin `d9105ea8001e` (P), nor the tag
`v0.34.0rc1` (T). Section 0 puts each of the 69 cited places at both: at P 34 have the same text, 26 differ and 9
are not there; at T 66 have the same text and 3 differ. The conflict, at each build apart: at P a hit under `mtp`
drops its trailing block; at T it drops none, and a block's key covers the token after it. vllm's reader is right at
T, and my line was wrong where I read it. Three more lines of mine are corrected: when a page enters the prefix
cache; a row's draft count in a step, which is the scheduler's and not only short near a request's end; and vLLM's
own token replay, which refuses a draft at P and takes one at T. Between the builds the route sees six differences:
the FP8 write's scale rule, the drawn token's noise, the hit's rule, when a page is published, vLLM's token replay,
and T's skipping of draft passes nothing would verify. The accept rule and the scheduler's arithmetic for a row's
draft count are the same at both. vllm's two questions: (a) yes at both builds: the request's block computes the
accept test and the kept count is its output, with no draft probabilities under the default; its gate count is not
read. (b) Registered, in round 1's step record, and charged until checked: at most 2.585 bits a decode row a step
for the kept count, and 0 for the scheduled count once the advice code models the scheduler under drafts. At P as
entry 2's showings were served, a BF16 cache and no draft token, entry 2's row 16 stands as written.
`r20261112-025327-f570` ended 05:10:05Z, exit 0: the step record's and the canonical top-k's cost at saturation is
not told from zero, and a cost under about 1% is not resolved by it (section 5).

The entry: `inbox/replay.md`, "Addendum, Thu 12 Nov 03:49Z, revised", section 0 for the two builds, the conflict and
vllm's two questions; sections 1 to 6 as before with the corrections marked. The 03:50Z headline above is superseded
where the two differ: the page's hash is not "taken when the scheduler publishes the page", and its lines were read
at vLLM `main`, not at the pin. [read at the pin `d9105ea8001e`, the tag `v0.34.0rc1` (`52aeb5806585`), vLLM `main`
`20ad1db3fe90` and `v0.32.0`; nothing run on a GPU]

`r20261112-025327-f570` ended 05:10:05Z, exit 0 (section 5): the step record's and the canonical top-k's cost at
saturation is not told from zero (`record` −0.36%, `canonical-topk` +0.20%, `no-prefix-cache` +0.35%, against stock
over stock −1.05% with passes from −6.75% to +0.98%); a cost under about 1% is not resolved by it, so the modeled
0.25% is neither shown nor refuted. It has no committing arm, so the page hash's cost at serving is not in it.

## REPLAY → proofs coordinator, Thu 12 Nov 13:28Z: final (due 16:00Z): row 16 stays the route; the kept drafts are 3 bits a decode row at fixed width (your figure, confirmed, and not the least: never more than 1 bit an output token; 1.84 by entropy and 1.36 by vllm's code on one greedy pass), 293 bits a request on AgentX's median and 188.5 on the 128k plan, 0.66 bits an output token; no route can be certified with a draft today; vllm's first try did not differ; at the pin the FP8 cache's two values part at a prefill row on the MHA path, and under the FP8 cache's backend every row but a full 1 + K decode row is one

The entry is `inbox/replay.md`, "Entry 3 and final", on this branch, and is appended whole to the coordinator inbox on
`cursor/verity-proofs-84b5` as the 12:48Z message ordered.

## REPLAY → proofs coordinator, Thu 12 Nov 13:31Z: revision to my final of 13:28Z, on your 13:18Z: VLLM-392 is named in the kept drafts' row and in the message; what it will charge is the count's own code, 0.30 bits an output token on the try, under the fixed width's 0.66

The revision is the last section of `inbox/replay.md` on this branch and is appended to the coordinator inbox on
`cursor/verity-proofs-84b5`.
