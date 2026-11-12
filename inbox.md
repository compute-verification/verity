[2026-11-12T09:41Z] 59e912763f3d
bounds to training, TRN-67, Thu 12 Nov 1:41 AM PST. One part of step 7 I cannot do with the tools the rules allow: clearing GitHub's draft flag on #8930.

- READ (`research queue --help`, `queue.py`): `--draft` exists only for `open`; `edit` changes title and body; nothing clears the flag. Its docstring: "Draft state is ignored: agents open PRs as drafts, and the ready label is what says one is done."
- READ (`gh pr view`, read only): #8593 and #8595 are MERGED with `isDraft` true, and no pull request of ours has a `ready_for_review` event. So the flag stops nothing.
- My `gh` may not write, and I do not try `gh pr ready`.

What I do instead, unless you say otherwise: the body and the title stop calling it a draft (the "must not land before" paragraph goes), it stays unmarked, and my final message says the flag is still set. If you want the flag cleared, it takes `gh pr ready 8930` from an account that may write.

