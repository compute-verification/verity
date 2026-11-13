[2026-11-13T13:04Z] 336e5317c3a7
From proofs, Fri 13 Nov 13:04Z, an addition to order 4 (sent 12:59Z by Task). Soundness has ruled on NET-403 (12:50Z) and PRF-490 (12:53Z). Read it before part 1's entry; it changes part 2 slightly.

- "This is a soundness finding, not a detail: below a rate of 1/1 the prover chooses among samples, which breaks 'no retries, nothing resamples'. A, B and C are all taken."
- B is taken as you sized it. Until B lands the two docstrings in `coins.py` (15 to 16, 351 to 352) are false of these two commands, "so add a sentence saying so": put that sentence in B's draft, in both docstrings.
- "One thing beyond B before the two-stage commands run under a gate: the Lean form holds the family and the root too, not only the Python commands." Lean's `across` says the caller names the family; the compiled two-stage form must derive it from the layout and refuse otherwise, "because the GLM-5.3 goal allows no Python decision on the soundness path".
- "STOPDRIVER's read of the one-stage command's verifier half and the compiled two-stage form says whether they already do. If either takes the family from the hand-off, that is the same finding in Lean: a refusal there, and a named premise until it lands." If the deployed path has it, it goes to Daniel.
- No figure from a two-stage run below 1/1 is cited until B and its Lean counterpart are in.

So part 1 is exactly the read soundness waits for: the one line by 13:30Z says YES or NO for (i) the one-stage command's verifier half and (ii) the compiled two-stage form, nothing else. In part 1's entry add, as Lean text and not as code: where the refusal would sit in the compiled form (which function, which step), what it compares, its refusal word, and which listed records it would move. Part 2 (B's draft in Python, with the docstring sentence) goes ahead after that entry whatever part 1 finds; the stop I wrote ("if part 1 finds the compiled form has the same freedom, stop before part 2") is withdrawn, since soundness wants both.

