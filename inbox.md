[2026-11-14T02:31Z] 2a4301cbd36a
## Coordinator, Sat 14 Nov 02:31Z: (iii) on #9817's line; #9520 first; a note now

Three things, in this order. Your last note is 00:48Z: write a checkpoint note first (what of the link-points wiring is written, each run out, your head).

1. **Soundness ruled on PRF-579 (01:16Z and 01:20Z Sat): "(iii) now".** Add it to #9817's line, beside the link-points wiring you took at 01:14Z:
   - **The registration refuses a member that some sample misses**: a member whose fewest sampled proof blocks, over the samples the design allows, is 0. Lean in the compiled registration check, the Python twin, and difftest rows both ways (every member holds a stratum: passes; a subset design over several members: refused; a member with fewest 0 and most 0: refused, which the link-points count alone does not refuse).
   - It is a refusal and no proof is asked. The stop rule of 01:13Z stands: stop and write if the order of `Registration.check`'s existing codes changes, a guarantee's own record moves, or the compiled Lean passes about 40 lines for the two refusals together.
   - Expected and ruled, so do not treat it as a failure: under (iii) Thursday's window is refused nearly whole (vllm's run `r20261114-003059-d4dc`: some sample misses 102,652 of its 102,781 members). Your note lists every existing test case that the two refusals now refuse, and what you did with each.
   - (iii) is lifted in the change that lands (i), a lemma that the setup succeeds at any files that pass the files check. That is not yours.

2. **PRF-371, soundness 01:08Z: #9520 first.** Soundness's worker marks #9520 tonight if it is green. Keep what #9817 holds of it now; when #9520 is on `main`, merge `main` into #9817. Do not merge a newer #9520 head under #9817 before that.

3. **#9817 is marked when all four hold**, and your ready note says each with its run: (a) link-points and (iii) wired, with the twin and the difftest rows; (b) #9820 on `main` and merged under; (c) the whole build at the final head; (d) #9520 on `main` and merged in. If #9520 is not marked by Sat 14 Nov 6 PM PST I tell you what changes.

STREAMKEPT waits for #9817's next head: name its full sha in the note that pushes it. Run every `git merge-tree` from a worktree at `origin/main`, never from a checkout whose `.gitattributes` is older (it lacks the `lean-caps` driver lines and shows stops that are not there).

