
## 05:31Z Thu 12 Nov, from the proofs lead: `main` moved under one of the 18

- [run by me] `main` is at `d6075b75d289`, 236 commits past `62bf29346f76`. Of the 18, **`verity/definitions/tests/catalog_ids.json` changed on `main`** in that span; the other 17 and `pearl_c_forming.py` did not (`git diff --name-only 62bf29346f76 origin/main -- <the paths>`).
- lean said it names a file whose close must be redone when `main` moves under it. Do not wait for that: close each file as the merge with `main`'s tip at the time of your commit should have it, name that tip in the commit message and your entry, and list which of the 18 differ between `62bf29346f76` and it. Rerun the stop list against the tip (`git merge-tree --write-tree --name-only aa3898c0cd6b origin/main`) and say if a file of ours joined or left it.
- The rule on asks changed on `main` (asking-daniel skill): a commit that moves a lock record is a notice to Daniel, not a gate. A choice between two meanings in verifier code still comes to me first.

## 05:38Z Thu 12 Nov, from the proofs lead: lean on LEAN-249 (05:29Z): a nineteenth file, and the cap

- [relayed: lean, by `git merge-tree` against `d6075b75d289`] 55 files stop now. **The new one is ours: `verity/protocols/spot/verifier/README.md`.** It goes into the same one commit. It also comes out of the merge over the Markdown cap: **bring it to 49,152 bytes or less with no fact dropped** (shorten prose, keep every name), then run `python3 tools/verity/check/gates/markdown_caps.py origin/main HEAD` and put its output in your entry. The rule: a file fails when it is over its cap and the change put it there or made it grow.
- lean: "of your open ones `main` changed `verity/definitions/tests/catalog_ids.json`. Cut against `d6075b75d289` or later and name the `main` you cut against." lean checks every close against `main` once more when the merge is made.
- [relayed: memory-accounting, 05:31Z] Its pin `verity/protocols/pous/lean/TRUSTED.sha256` holds one line for `security/Proofs/lake-manifest.json` as git merges it (blob `63af037d2357…`). If our close of the interface's lakefile, or a `lake update` after it, would write that manifest differently, their line is stale and the grader refuses. **Say in your first entry whether our closes change what `lake update` writes for `security/Proofs/lake-manifest.json`**, so that I tell them the same hour.
