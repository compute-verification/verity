
## 05:31Z Thu 12 Nov, from the proofs lead: `main` moved under one of the 18

- [run by me] `main` is at `d6075b75d289`, 236 commits past `62bf29346f76`. Of the 18, **`verity/definitions/tests/catalog_ids.json` changed on `main`** in that span; the other 17 and `pearl_c_forming.py` did not (`git diff --name-only 62bf29346f76 origin/main -- <the paths>`).
- lean said it names a file whose close must be redone when `main` moves under it. Do not wait for that: close each file as the merge with `main`'s tip at the time of your commit should have it, name that tip in the commit message and your entry, and list which of the 18 differ between `62bf29346f76` and it. Rerun the stop list against the tip (`git merge-tree --write-tree --name-only aa3898c0cd6b origin/main`) and say if a file of ours joined or left it.
- The rule on asks changed on `main` (asking-daniel skill): a commit that moves a lock record is a notice to Daniel, not a gate. A choice between two meanings in verifier code still comes to me first.
