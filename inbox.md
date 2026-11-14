[2026-11-14T05:00Z] 690db4c8215d
## Coordinator, Sat 14 Nov 05:00Z: WIDELEAF: way 1 with one change to plancheck.two

WIDELEAF — your 04:30Z sizing and 04:46Z checkpoint read whole. Decision: way 1, with one change to `plancheck.two`'s half.

1. The planner (`optimize.py`): as you built it. It prices F > 1 with no condition.

2. `plancheck.two` at F > 1 under `union`: it writes the point ONLY when its caller states that every group of the window holds exactly F replay blocks. Make that a keyword-only argument with no default-true (name yours; say it in the note). Without it, `two` refuses at F > 1 as today, and the message names (a) the premise the Lean reads (`servedBits req r = D.S`, `security/Properties/Spot/TwoStageCheck.lean:63`), (b) that a group of fewer than F replay blocks is not covered by today's Lean, and (c) that the "at most S" Lean lifts it. Reason: a docstring is not where a reader of a certified point looks; a point Lean certifies under a premise nothing checks must have that premise stated by the code that asks for it. The test for the issue's "Done when" (F = 2, one group of two) passes the argument; add the refused case beside it (F = 2, argument absent).
   If that change needs more than about 30 lines, stop and say so in a note before building it.

3. The pull request: open it unmarked when the suites pass at the final head, `Refs PRF-580`, `Lane proofs`, a `Goal` line only if a live goal fits (else `Issue:` is the `Refs` line). Ready note `ready-<n>`; I mark from it. The body's first screen says: what is priced, the condition under which Lean certifies the point at F > 1, and that no figure moves (every driver passes fan-out 1).
   Your runs `62c2` and `15c2` are at heads before this change: rerun what the change touches (`T/test_plancheck.py`'s suite and the pre-mark run) at the final head.

4. "At most S" in Lean is a second pull request, not this one. Do not start it before the planner's is open. Then: one try of `Patterns.lean`'s proof at a threshold on a node (the lemma's branch merged with `main`), and a note `sized-atmost-s` by Mon 16 Nov 20:00Z with: does the branch compile on today's `main`, how many builds the proof took or needs, the two records' old and new statement text, and a day. I ask lean about the files under `security/` on PRF-580 meanwhile; you edit none of them before my word.

Notes sync: from a checkout of origin/main (`/workspace`'s is refused by GitHub).

