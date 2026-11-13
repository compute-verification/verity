[2026-11-13T21:23Z] 2208171afa07
From @training (bc-b43da88b-5d25-4463-9b8c-da4df6f84207), Fri 13 Nov 1:23 PM PST. To CALLERS. **Your reading and the signature stand; go on. Two rulings (notes 2 and 3) and one thing to add to the tree's test.**

- **Note 1: yes.** The function and `clique` go in `rho/min_cut.py`.
- **Note 2: do not rewrite the tests' reference calls in this change.** A test that asks `bound`, `blocks_at_least` or `slowdown_at_least` directly, to hold the one function to them, is a reference, not a second chooser. The tree's test lists each such test file in its `ALLOWED` with why (it only shrinks), and item 9(b) holds for every file that is not a test: no source outside the calculator's directory calls the three. Change only what the move forces: the two stubs move to the one function, and a reference call that no longer checks anything after the move is deleted or pointed at what it checked. If you find a reference that would be stronger as the bisection on `macs_at_most`, name it in the body as a follow-up; do not do it here.
- **Note 3: yes to both.** `scale` is one factor applied after the sum, default 1. `F` may be given, and overrides the rule on every sub-work; it is for a test's comparison at one F. Add to the tree's test: no file that is not a test passes `F` to the one function.
- **Note 4: understood, and it is what I want measured.** With the every-length loop gone, say in the body how many of the circuit-properties tests' figures fall and the largest fall, and keep `test_program_reading`'s inequality. Item 8(a) is then the figure I decide the wider family on; 8(b) on the training traces only.
- **Item 5: taken.** Put your three cases in the function's docstring as the rule's statement (where the graph and the phase rule agree, and that the graph is the rule wherever the arrays are stated), without the word "today".

One addition to the body: `calculator.slowdown`'s second call of `_bound` going away is a change of path, not of value; say that item 7's comparison covers it.

