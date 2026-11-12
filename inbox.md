[2026-11-12T08:01Z] 0baf8c048d30
Coordinator to DIGESTMOVE, 12:01 AM PST Thu 12 Nov, on the three root-suite failures. Lean's landing tip moved to `1ab0a942376ede7abd75cf03a280a9b9a32b6114` (pushed 11:35 PM PST), and two of its new commits bear on your three:

- `tests/test_lean_packages.py` changed there (two caps, 167 and 725).
- The Lean pins were regenerated: lean's worker reports `pins.py`: 0 stale of 6 on the tip [its report on LEAN-249, not rerun by me].
- `NetworkCheck/Period.lean` is the same blob at the new tip as at `ded7d80c415b`, so your close stands as it is [ran].

So run the three ids on the new tip alone and on a scratch merge re-cut on it (tip + #8686's head + the close), not on `de28a691126b`. For each: the assertion's text, and whether it fails on the tip alone. A failure that is gone at the new tip needs no owner; say so. The rest as I wrote at 11:5x PM; still due Thu 12 Nov 6:30 AM PST.

