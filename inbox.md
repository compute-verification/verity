[2026-11-12T07:27Z] f2ac1db0e9ca
training to PRICER (TRN-135), answering your stop on rule 4 (your 07:07Z message). Read; your reasoning stands, and the result is the finding: from the circuit alone, every product of a model that starts with a table lookup is left out, so the reading gives no slowdown on it.

1. **The weak side is the reading's default and stays as you built it.** `work.read(program)` with nothing beside the Program charges nothing downstream of a gather by the batch. Say so in the docstring and the body as the reading's first limit, with your table (the trace's blocks at each k against the Program's 0).

2. **Build the other side as a separate, named statement, off by default.** One thing stated beside the Program: which external input holds the tokens, and how its words lay out as rows by positions. Not a constant's shape, not a guess from member counts. With it:
   - A batch's members are positions only where data flow shows it: each member reads the gathered rows of one position's index word, one to one. Where the Program's batch structure does not show that correspondence, the members are not positions, statement or no statement. This is what keeps a per-head constant from making anything specific.
   - Rows become specific to a position at an op that combines them, member by member over positions, with a stated constant whose words differ between the positions of a row. That is checked on the constant's words (the mask).
   - An external input whose values the Program does not state (the rope tables) makes nothing specific on the reading's own word. If the statement names it as positional, the reading takes it and lists it in its result as a premise it could not check.
   - Everything else is the weak side.
   Do not go further than these three: no rule from an input's name, shape or size.

3. **Report both readings side by side** on the tiny step at each k: nothing stated, and the layout stated, each against the trace's M, group by group with rates and β. The invariant (nothing charged that the trace does not charge, at no more MACs or bits) holds for both. Your plan to show each product's rate and β beside the trace's row whether or not it is charged is right; keep it.

4. If even with the layout stated the member-to-position correspondence cannot be read off the Program's batch nodes, stop on that alone and tell me what the Program would have to state for it.

What this is for: whether a submission states its token layout beside the circuit is the lead's call, and I take him your two readings, so each must say exactly what it rests on. The order: the default reading with the rate rule, the invariant and the adapter scan first; the stated layout second. If the second does not fit by Fri 13 Nov noon PST, say so in Thursday's 6 PM interim and I re-date it; do not cut the first to fit it.

