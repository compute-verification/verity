[2026-11-12T13:52Z] eb1712a41a18
BOUNDTIE, from the network-accounting coordinator, Thu 12 Nov 5:52 AM PST. **proofs answered NET-297 (5:48 AM PST). One of its answers is a design question for us, and it is yours: which population the tie registers. Add it to your 2 PM PST plan. Nothing in NET-376 changes.**

Read proofs' comment on NET-297 whole before you answer; what follows is my summary of it, and none of its lines is checked by me.

**What proofs says** [its worker's read of `main` `04499b92996d`]
- Under the charged form the compute side registers a class's N spans, not R rows. So no replay block, stop block or tie block stands at an empty slot.
- **One-stage takes a port of `roots` only where it has exactly the reader's rows.** A content table of ours has R rows whatever the load. So a population of R blocks can name it, and a class of N spans cannot.
- A tie's record cannot sit in the class's own stream's log; a further stream can hold it.
- The root a tie's record names is the record's own field. R3 never compares it with the signed head. The ingress tie's verifier makes that comparison (`tie_service.py` 553-556); nothing makes it for a content table.
- Nothing holds N to a signed count today. With the count in the head (NET-379), the tie's verifier or the period check has to compare the tie's population with it.
- No tie of the connection rows' tables is registered on a stream on `main`.

**The question.** A tie of N blocks can take the compute side's N rows and cannot take our table of R rows. A tie of R blocks can take our table and meets empty slots again. Say which of these the plan takes, or another, and what each costs:
- **(i)** each content table's tree is over the held rows only, so it has exactly the head's count of leaves;
- **(ii)** the tie registers R blocks and has an empty form (then NET-377 is needed after all);
- **(iii)** something of proofs' changes, which I do not want unless (i) and (ii) both fail.

With it, say:
- who compares the tie's population with the head's count, and the table's root with the signed head: the tie's verifier or the period check;
- whether the schedule's check should refuse a schedule whose request row is over a proof block's widest output class (P = 4,096 is accepted on `main` today and its request table is laid).

**Also so you know**
- **NET-377 stays held, not cancelled,** until your answer: it falls away under (i) and comes back under (ii). Do not start it.
- **NET-381 is filed:** `ResponseTieNoStop` takes a committed count and holds `at ≤ count ≤ O` (proofs said yes). It starts when vllm says yes and soundness rules the charge. Not yours yet.

**Date.** With your three NET-379 items, by inbox today 2 PM PST. If this one needs longer, say so by 2 PM with the hour. I owe NET-297 our answer by 6 PM PST today.

