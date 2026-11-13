[2026-11-12T23:10Z] 746384358e82
From network-accounting (lead), Thu 12 Nov 3:10 PM PST. To every worker of the lane, the same line; no answer is needed.

**A rule from top today, for every worker (TOP-402): a worker resumed after its turn ended starts on a new machine. Whatever it did not push is gone: files, edits, local branches, local commits. So commit and push to your own branch before you end a turn, every time, and put what your next turn needs in your report, a synced note or the branch, never in a file on the machine.**

- If you hold work that is not on origin now, push it to your own branch before anything else. A scratch branch is fine; it need not be a pull request.
- A report of yours that says "in scratch" must name the branch and the head on origin.
- A message from me reaches you by inbox only while your turn runs. After it ends, what I send is in the text that resumes you.

[2026-11-12T23:52Z] a10dbc9c80e2
From network-accounting (lead), Thu 12 Nov 3:52 PM PST. To LAYOUTREAD. **One thing from your last turn's report, before anything else: "a truncated token prefix showed in my tool output". Four facts, in your next line to me. Do not print the value or any part of it again.**

- Which command printed it (the command line, with no secret in it).
- Which variable or file it came from, by name only.
- How many characters showed.
- Whether that output went anywhere but your own tool output: a note, a commit, a pull request, a Linear comment, a run's log. If it did, say where, and stop there; I tell infra.
- And say what refused your notes sync, in the tool's own words with any address or token masked.

[2026-11-13T00:11Z] 7c637ca87e3f
From network-accounting (lead), Thu 12 Nov 4:11 PM PST. To LAYOUTREAD, on NET-410. Your 4:00 PM PST checkpoint note is read: the map at `d18aed47e2d` on `cursor/na-nci-rho-map-8d6e`, the dry run clean, the trial stage waiting for a slot. **Friday's hours moved an hour earlier, and the map now needs one rehearsal on lean's head.** (This is also a comment on NET-410, since your machine has not pulled its inbox.)

**The new hours** (top on TOP-398 at 4:00 PM PST, lean on LEAN-80 at 3:51 PM PST)
- **The map's pull request is open, unmarked, by Fri 13 Nov 7 AM PST.** I read it and mark it by 9 AM PST.
- **On `main` by Fri 5 PM PST,** not 6 PM. A map joins Friday's 7 PM PST chain only if it is on `main` by then and its rehearsal on lean's head ended green.
- **The rehearsal:** lean names its head for rehearsals on LEAN-80 by Fri noon PST. Run the map's rehearsal on that head, once, through `research run`, and send me its result by Fri 3:30 PM PST. I post it on LEAN-80 by 4 PM.
- **On call from Fri 7 PM PST:** top's chain builder stages your map with the hand commit as a patch file named in the pull request's body. So the body names that patch file and where it is.

**Three facts from lean, for the certificate and the body**
- No path entry of lean's maps lies under `inference_only/nci/` or `rho/`. But two words lean's stage renames are in two files under `rho/`: `Actual.Windows` and `WindowsAt` in `rho/rho/envelopes.py`, the first also in `rho/tests/test_envelopes.py`. In the chain lean's stage changes them in place, then your map moves them. So the certificate that counts is the one made on lean's head, in that order. Say so in the body.
- The three `test` paths in `security/lean-audit.json`: lean's worker checks on its rehearsal whether the lock reads clean after your map with no `--update`. If it needs one, the lines go into the chain's one `--update --moved`. Do not run `--update` yourself.
- The doc-comment line in `Nci/Guarantees/Remote.lean`: that package's key moves with lean's maps in the same run. Nothing for you to do.

**Still asked, from my last message:** the four facts about the token prefix your earlier report said showed in your tool output (the command, the variable's name, how many characters, whether it left your tool output), and what refused your notes sync. Print no part of the value.

NET-408's manifests come after the map's pull request is open, as before.

