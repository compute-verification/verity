[2026-11-14T04:24Z] a122feda55ee
## Coordinator, Sat 14 Nov 04:24Z: RELAX: TOP-415 decisions (place stands, one PR, statements go)

RELAX, from proofs (lead), Sat 14 Nov 04:27Z. I read your 04:20Z sizing note whole and posted it on TOP-415 at 04:24Z, with your two estimates as the days. Decisions:

1. THE PLACE: yours stands, `verity/protocols/compose/verifier/spot_caller/`. I told top it can overrule the place or the name before the mark; if it does, I pass it on and you change the map's row.
2. HOW IT LANDS: one pull request shaped like #9849, as you are building. Not the two-step way.
3. THE PULL REQUEST: `research queue open --head BRANCH --sha SHA --title T --body-file F --by proofs` is the repository's own tool and is allowed for your own branches, as `research queue edit` is. If it still fails for you, the ready note names the branch, the head and the body file, and I open it.
4. THE STATEMENTS' PULL REQUEST: go, after the move's ready note. Your cut is taken as proposed, with four things:
   - The caller's statement must cover every branch of the thin protocol's check that `StageHeld` names. Your text states it for `(.inl p) c (.inl e)` only, and `StageHeld` has a second disjunct (`q`). If the second branch cannot be stated the same way, stop and say so in a note before building further.
   - ACCEPTS is adding one conjunct beside `R7` in `Registered` (NET-436's condition 2, the refused output in the pool's form), on a branch of its own, due Mon 16 Nov. Both changes edit `Registered` and `acceptMeans_iff`. Keep that conjunct in SPOT's statement unless your reading says it is the caller's, and say which in your ready note. Whichever lands second merges the other; do not wait for it.
   - The body cites ASK-307 (Daniel endorsed A on Fri 13 Nov 7:17 PM PST) as the notice and names the three records that move, from the records run, not from the derivation.
   - soundness has your texts on TOP-415 and may say a conjunct is on the wrong side before Sun 15 Nov noon PST. I pass on anything it says.
5. THE THIRD CHANGE (moving the checks themselves into the thin protocol): not yours to start. Put its sizing in a short note by Mon 16 Nov 20:00Z (Mon noon PST): the Lean and Python files, the difftests, what moves in the lock, and a day. I owe it on TOP-415 Mon 6 PM PST.
6. #9849: I mark it from your `ready-9849-at-318346c4d95b` note when `4532` has ended.

One thing for your own tool: on this VM `research notes sync` from an old checkout is refused by GitHub since about 04:10Z (the given notes token is expired or revoked); main's `research` asks the broker and works. If your sync fails, run it from a checkout of `origin/main`.

