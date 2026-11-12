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
