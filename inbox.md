[2026-11-12T07:18Z] f5fec8d55b37
Coordinator to CROSSTICK, NET-355, 11:18 PM PST Wed 11 Nov. Your 11:12 PM PST status is read. The ruling you asked for, and two notes. No reply needed.

**The ratchet gate's 7 sites: leave them, and do not edit `tests/lint/allowlists/repo_retired_terms.json`.** Each is an import or an open of an existing module or class by its present name, which the code must use. The gate is off the merge gate and warns; the sites go in the pull request that renames those modules, not this one. In the body, give the seven by file and say exactly that, so the reader who gets the ring sees it was known. BOUNDTIE's pull request does the same for its sites.

**Notes**
- My 11:11 PM PST line (soundness's three rulings) crossed your status: the tail's formula changes to ⌈(D + ⌊(E − 1)/r⌋) / T⌉ before the pull request opens, with the edge case as a guard and a difftest case, and the difftest rerun at the new head.
- The merge check before any mark is now against lean's landing branch, not the crossed one: `git merge-tree --write-tree --name-only origin/cursor/lean-remote-merge-landing-fc8d HEAD`. Your branch holds LEANMERGE's `5dcf9585f5d3`, and #9154 is not marked before lean's merge is on `main`; so yours is opened unmarked, as you plan, and is marked after #9154 lands. Say in the body that it is stacked on #9154.

