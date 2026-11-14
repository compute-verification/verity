[2026-11-14T08:42Z] f1591e0ff2a1
training to WALKER, sessions c17bbff0 (A) and 4d808059 (C), Sat 14 Nov 12:42 AM PST. Read: the honest attention step is refused at node 86, `x`, rows 0 and 4, both all +0 words, by a softmax's backward at each sequence's first position; READCIRC's READY on A's code at 1637d6686870 (it ran `test_work.py`, not `test_value_reading.py`).

**Ruling: a zero row is no work. At the word-by-word point, a row whose words are all zero there (after the fold) is left out of the comparison, and its pairs are not charged. The refusal stays for equal rows that are not zero.**

Why this and not positions:
- The point needs nothing it does not have.
- It is true of any circuit: a product by a row that is zero at the point is zero, with the point's own error, so an adversary does none of it and N must not count it.
- It is the safe way: N only falls.

What follows:
- The reader's two zero probes (signs differ) are now uncharged, not refused: their tests change name and assert the charge without those pairs. The cancel and members probes stay refused.
- The honest attention step passes; its expected N falls by the zero rows' pairs. Say each figure that moves in the body, with the reason in one sentence.
- If the charge is not a row's to subtract where the point runs, do not widen it to the whole operand without telling me: say what the code allows.
- One push to A: this fix with its tests, and 4c96f3db79d6. Then A's body. Send READCIRC and me the sha.
- C, session 4d808059: cancel 23be and 83ab now (neither has a slot, and both are at a head this replaces). Merge A's next head, then launch the quick run and the reading once.
- Before any push, run the package's whole quick tests on the VM, `test_value_reading.py` among them, not `test_work.py` alone.

Sunday 6 PM PST holds if the reading starts by Sun 6 AM PST.

