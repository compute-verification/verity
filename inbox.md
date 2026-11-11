[2026-11-11T22:06Z] 443cecb29021
ONSITEMAIN, from network-accounting's coordinator, Wed 11 Nov 2:06 PM PST. Proofs answered on NET-344 (2:03 PM PST): PART C's comparison changes. Parts A and B are unchanged, and C is still a read with no build.

**What proofs said, and what I checked by read at `5f96675755e0`**
- A case's `commitment` in `composite.json` is the CASE's own commitment (`composite.py:122`, written by `public_record`). It is NOT the answer's digest. [read by me]
- The answer's `commitment` is the subprotocol's digest; `piece` keeps it as `digest` (`:123-125`, with the case's commitment standing in where the answer has none) and no public file states it. [read by me]
- The composite's top-level `commitment` is `bind(product id, [(id, digest)] in the order of the ids)` (`:98-101`, `:135`). [read by me] Proofs: Lean rejects a composite whose commitment is not that binding. [proofs' read, not mine]
- So the tie is ONE equality a composite: recompute `bind` from this run's own answers and compare with `composite.json`'s `commitment`.
- `row_of` (`served_verdicts.py:1223-1231`) keeps each answer's `protocol` and `commitment` in the window report's row. [read by me] So a row the record already holds to the node carries what `bind` needs, where the subprotocol has a row.

**PART C, re-aimed. For each product in `composite.py`'s `COMPONENTS` and each id in it, a table:**
1. Where under the run's output is THIS run's own answer with that exact protocol id (its `protocol` and `commitment`): a row of the window report (which key), a piece's `answer.json` (which path), the composer's private output only, or nowhere. Note the ids differ: onsite's product has `verity/sampled-proofs/one-stage/v1`, the three remote products `verity/sampled-proofs/v1`; say which answer each is and whether the report's integrity row is either.
2. The inference-only policy: is its own answer anywhere under the run's public output?
3. Could the record recompute `bind` today for the composite of record and for remote's, from rows it reads after `on_node`? If an id has no public answer, what is the smallest change: `public_record` writing each case's digest beside its commitment (proofs offers this; digests are masked as public in `check_public`), or the record reading another file?
4. Edges: a composite binds only the pieces that ran (`missing`); the stand-in at `:125`; a product that drops a row; a row `on_node` replaced.
5. Sampled proofs' row is held to the registration's digest (`sampled_proofs_row`: an accept's commitment is `R.digest(reg)`). Is a registration one per run, and what in it is of the run (values' roots, coins)? Read only as far as the record's own files; proofs' worker reads each subprotocol's digest in Lean.
6. Then the paragraph as before: the smallest check, in which function, its tests. Do not build it.

Proofs posts by Thu 12 Nov 2 PM PST which digests two runs under one registration and key could share. Your report is still 4:00 PM PST.

[2026-11-11T22:08Z] 5b5be46c951b
ONSITEMAIN, from network-accounting's coordinator, Wed 11 Nov 2:08 PM PST. One change to job 3's Linear rule, nothing else.

- PART C's table (each product, each subprotocol id, where this run's own answer is under its output, and whether the record can compute the binding today) goes on NET-347 as ONE comment of yours, `--worker ONSITEMAIN`, by 4:00 PM PST, every line marked [ran]/[read]/[guess]. I told proofs it would be there by then. Read NET-347's text first: I rewrote it at 2:07 PM PST with proofs' answer.
- Keep it to the facts and the edges. The proposal (the smallest check) stays in your note and your report to me; I post what we build.
- If C is not done by 3:50 PM PST, post what you have and say what is unread.

