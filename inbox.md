[2026-11-14T03:36Z] 537a30e894fc
## Coordinator, Sat 14 Nov 03:36Z: RELAX: TOP-415 endorsed; size within two hours, then start the move

RELAX, from the proofs coordinator, Sat 14 Nov 03:42Z (Fri 7:42 PM PST). One change to item 3 of the order I resumed you with at 03:22Z (TOP-415's package move). Items 1 and 2 (#9849's and #9892's ready notes) stay first.

Daniel endorsed A on ASK-307 at 7:17 PM PST: SPOT is called on every run through one thin protocol that holds the caller's facts. It is no longer provisional, and top asks proofs for the size and the day now: the package move first (0 lock records), then the two statements, `AcceptMeans` and `AcceptInferenceOnly`, in ONE pull request after it. And by Daniel's ruling of 7:15 PM PST a change to a claim is a notice to him, not a gate: nothing waits for Tue 17 Nov any more.

So item 3 becomes:
(a) A FIRST CUT of the sizing within two hours of reading this, as a note `sizing-top415-package-move`: the files that move and where to (count), the importers that change (count), whether `move_check.py` certifies it as a pure move, the open pull requests that touch the moved paths, and YOUR DAY AND HOUR (PST) for the move's ready note. Then, in the same note, the same for the two statements' pull request: what `AcceptMeans` and `AcceptInferenceOnly` say today and what they will say (both texts), which lock records move (their own records do: say which others), whether any DEPLOYED guarantee's statement changes, the size, and its day. I post both days on TOP-415 from that note. A rough count now beats an exact one at 8 AM.
(b) THEN START THE MOVE without waiting for my word: a new branch `cursor/proofs-relax-top415-move-<hex>` from today's `origin/main`, by the move tool if it is a layout move (so that `tools/verity/check/move_check.py BASE HEAD` certifies it; a certified pure move lands alone, ahead of the queue), the Lean fast path of each package whose files move and `suites.py --changed origin/main --quick` on nodes you name, the pull request opened UNMARKED (`Lane proofs`, `Refs TOP-415`), and a note `ready-<number>`. I mark.
(c) The two statements' pull request after the move is yours too, unless your note says another worker's reading fits it better (name the worker). Do not start (c) before the move's ready note.

STOP and write both texts if the move would change any lock record (lean counted 0) or any statement.

