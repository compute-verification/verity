[2026-11-13T17:13Z] 2250ab10cd37
COUNTS, Fri 13 Nov, 9:12 AM PST. #9071 carry: way (a) is clean; the push is by hand, of the tool's own result, and I am telling you first.

- RAN run r20261113-163332-07cd (node, clone of `main` 5eb1231b2558, branch sent as a bundle, every call `--dry-run --kept --json`):
  - `restack.py cursor/training-digest-form-string-a8e2 --main eb067c7c439`: `clean`, result 6b1f947b61716961b49e05a6f3bfcb0e7e9a0adf, crossed 70395628ec67 and eb067c7c439b, skipped none.
  - `restack.py <that result as a local branch>` (default `--main origin/main`): `clean`, result dbe411610ca12e9db61c7c5718ebe9dce4653b49, crossed 28, skipped none.
  - beside them the plain dry run on the branch: `clean-skipped` (the two early moves), result df262f91e55d. Its tree differs from way (a)'s only in where three lines sit in `tests/lint/allowlists/repo_retired_terms.json`; as values the two lists are equal.
- Why by hand: the node's clone can't push (`tools/move/README.md`), so the tool's `--push-to` had nothing to push to; the run bundled the results and I fetched them. On my VM the tool does not finish in time.
- What I push now, nothing else: `git push origin dbe411610ca1:refs/heads/cursor/training-digest-form-string-carried-a8e2` (what `--push-to` runs; the name is free), then `git push origin cursor/training-digest-form-string-carried-a8e2:cursor/training-digest-form-string-a8e2` (a fast-forward: 2dea0d32fdcc is an ancestor of the result, and so is `main`). No commit of mine is in it.
- Then tests 1 to 6. One thing already READ in the result: the three locks merged without conflict (no `locks-stale`), but the merge left two sorted lists in the old order after a rename (`shape.premises` in `security/lean-audit.json`, one `trust_gap` list in `security/Properties/lean-audit.json`); the records run says whether that needs a commit on top.

If you want the push another way, say so and I stop where I am.

