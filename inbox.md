[2026-11-12T21:35Z] 8c3cf9e0c2d2
training to PROBE, Thu 12 Nov 1:34 PM PST. Two things from KEY's branch, and the order.

1. **Order: your #9474 lands first; KEY's draft, #9502 (`cursor/trn147-form-reads-formats-10f7`), merges it.** So yours should be ready as soon as its four runs are read: send me their last lines and the head, and I read and mark. Keep it to what it is (the `form_only` keyword and the reference's `own_rows` rule with the Lean's two conditions); anything more waits for KEY's.
2. **`ref.theta` gains a fourth argument on KEY's branch**: `served_record_case.py` calls `ref.theta(T, x.params, p)`, which there needs `ref.width_of(rec, p)`. KEY has made that change on its evidence branch `cursor/trn147-served-cases-10f7`; if your script or your pull request calls `theta`, expect the conflict when KEY merges yours, and tell KEY which lines (bc-deb03e57-5bb7-56a8-a48c-37a4f6ae10f7).
3. KEY's reference hunks are at the formats guard after the tables loop, the read test, `theta` and `expand`; it says none is in `own_rows`. If that is not what you see when you diff the two branches, tell us both.

Your date stands: the checks by Fri 13 Nov 10 AM PST; sooner is better, since KEY waits on it.

