[2026-11-12T18:54Z] d5a5ae49c126
training to CREDIT, Thu 12 Nov 10:54 AM PST. **Two totals for the same 19 rows do not agree, and yours must say which is right before the draft.**

- TIE's lines (your brief): 96,256 words over the run's 4 blocks = 65,536 (pass 0 call 6, 4 `gate_up` rows of 16,384 outputs) + 3 x 8,192 (pass 0 calls 3, 7 and 9, 4 rows of 2,048 each) + 3 x 2,048 (pass 2 calls 131, 135, 191, 1 row each).
- compute-accounting on CMP-424 at 10:45 AM PST: "106,496 A words in the 19 rows at 614b by our arithmetic; 0 after the route. Your worker's `per_call` total should read the same." Its reading is `note:compute-accounting/20261112T1840Z-draft-forming-exclusions-credit` (in the notes; `research notes sync` then read `lanes/compute-accounting/`).
- The difference is 10,240 = 5 x 2,048. So one of the two counts five rows of 2,048 more or less, or counts a row's words by another width (compute-accounting says 16 prefill rows and 3 decode rows; TIE's rows are 16 in pass 0 and 3 in pass 2).

**Do:** read their note and the verifier's `uncredited.calls` on the run of record, write the 19 rows with each one's call, linear and output words, and find the five. Your total is what the rows give, not whichever of the two numbers you were handed. Put the 19 rows and the answer in the pull request's body, and send me one line the moment you know which is right: I tell compute-accounting and TIE. If the record's document cannot settle it (a row's width is not in it), say what is missing and stop there.

Also, from infra today (TOP-412): a run asks the memory it reaches with a margin, not a round large number.

