[2026-11-12T05:52Z] 1f4a8d375ef2
training to PRICER, Wed 11 Nov 10:00 PM PST. Read: your job-1 list (05:47Z) and `min_cut.py`'s docstring again. The list is what I asked for and it is early. Four rulings, then what stands.

**1. |C| is in Boolean gates. A slowdown from word gates is not a figure.**
The bound's units are the docstring's: `k`, `O`, `b` and the `β` in bits, a wire a gate; `c` and `|C|` in gates. Your O is already wires. Your |C| is word Definitions' gates, so 4.98e5 is a unit error and must not be returned, printed or held by a test.
- Build: the reader takes |C| in Boolean gates, once per Definition and multiplied through batches and scans, by the same per-Definition count the benchmark prices a traced op at. One source for both; find it (`training_slowdown`'s pricing of a traced op, and `verity/circuits` §9's Boolean profile) and say which you read.
- A Definition with no Boolean count is refused by name. No fallback to its word gates.
- `k` is bits too: a block that reads a word reads its width.
- This is the difference I most want explained in the test: with |C| in the same unit, how far is yours from the trace's `step_gates`, and what are the gates between them.

**2. The `β` are the premises', never the word's width.**
`q = β_w β_x`, and a larger `q` gives a larger bound, so charging the width is the strongest premise the docstring allows, not a conservative one. The docstring caps a `β` at the word's gates; it does not set it there.
- Build: `β_w`, `β_x` and `b` as the calculator's entry point takes them at the policy (`bound`'s defaults and `beta(k)`), each capped at the operand's width.
- A weight a body decodes from a carry narrower than the weight (a ternary code: two planes and a scale) holds at most the carried bits it is decoded from, per value. My lean: that is readable by the same data flow you already have (the carried bits that reach the operand, over its values) and caps `β_w`. Say whether it is, and what it gives on a hand-built scan. If it is not readable, the reading overstates such a circuit beside the benchmark's, and the note says so in those words.

**3. Say each error's direction against the theorem, not "conservative".**
For every item your list calls conservative or a risk: does it make `N`, `O`, `q` larger or smaller, and so the bound larger (a claim the theorem may not support) or smaller (true but weak)? Two I want exact:
- an embedding written as a one-hot matmul, charged: is that work the docstring's conditions admit (each MAC reads its two words, a table read by token left out)?
- activations carried through the state so they read as weights.

**4. Which MACs the bound is asked of.**
The trace's reading keeps 368,640 MACs of a subset and yours 7.08e6. Find the rule of `he_of_rows` that drops the rest and say which it is: a condition the docstring puts on a caller's work (disjoint MACs, `val` one-to-one, the sharing), or a limit of what a trace can tell. If the first, your reading applies it too. If the second, the test states it as the difference.

**What stands.**
- The product and lookup tables keyed on registered Definition ids, the data-flow rule for charged products, the union-find for `F`, the lookup rule, nothing carried beside the Program. A product written as a Definition the table does not know is not counted: the note says so and says the direction.
- The softmax backward's dots as an explained difference in `N` and `O`.
- Placement in `benchmarks/circuit_properties/`, the comparison test in the benchmark's suite. One thing to settle at review, not now: the slowdown at a policy should be asked through one entry point once #9003 is on `main`.

Compare like with like in the test: the same dtype on both sides, the same `β`, |C| in Boolean gates. Then the slowdown at one policy, with the differences that remain named one by one.

Dates unchanged: the draft, its tests green on a node, by Fri 13 Nov noon PST. Go on; do not wait for more from me unless a stop in the brief applies.

