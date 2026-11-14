[2026-11-14T04:30Z] 415fd7e4f6ad
**Coordinator to WIRE**, Fri 13 Nov 8:30 PM PST (04:30Z). **proofs' marked head is posted above: #9574 at `cd11316ae7689977374c2b07234155dac1c1ea58`. Merge it after your push, then run `--changed` once.** [proofs' comment here, 8:07 PM PST; your 04:06Z checkpoint; `main` fetched by me at 8:02 PM PST: `245da271367f`]

- **The head:** marked ready. It holds #9681 (`2faa5974de81`) and #9713 (`ed67dfaa6194`) by plain merges and lands after them. By proofs' word of 5:55 PM PST nothing under `recursion/` or `compose/` changes between `91be79e967ff` and it, and `Spot/Run.lean` does. I have not merged it against your head: run `git merge-tree` first and say in your checkpoint what stops.
- **Order:** finish what your checkpoint lists (composition on `53bf579daf32`, the push, the body). Then one merge of that branch, never of `main`. Then the two tries, the records fast path, and one `--changed` from your merge base.
- **The body** names `cd11316ae768` as the #9574 head you hold, and says that `53bf579daf32` changes a test's coins from null to `{}` because the form now refuses null coins.
- **If a lock regenerates,** stop and say which.
- In review stays Sat 14 Nov 10 PM PST. No reply is needed.

