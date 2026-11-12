[2026-11-12T22:31Z] 98547ac28ed2
SORT, TRN-146, final: one verdict for each of proofs' twenty old open unmarked PRs.

Queue rows and bases: none of the twenty reaches `remote` or the old `next`, and none carries a `Queue:` merge. All twenty are based on `main` and unmarked; #7850, #7952, #8120 and #8177 are stale. RAN (`research queue status --json`, `gh pr view --json baseRefName`, `git merge-base --is-ancestor`).

Every carry is a merge of `main` 30d6fcc92ff7, never a rebase, on a branch of my own. Each carry's head holds its PR's head (RAN, `git merge-base --is-ancestor`).

| PR | Verdict | Evidence |
|---|---|---|
| #7850 | Still wanted; proofs decides; not started | A generated move over 218 files with 75 conflicts: the carry is a re-stage with `stage.py`. Not started, waiting for proofs' yes. RAN (`git merge-tree`). |
| #7952 | On `main` (moot) | 8767136d8f moved its entries to `witnesses`. RAN. |
| #8120 | Superseded by #9209 (open) | The one piece not in #9209 is the `ofRowsFast` presize twin. RAN (diff). |
| #8139 | Still wanted; held by OLDFOUR | Head f51a277c6e. Its quick tier `9dd0` passed 71 of 71; it is unmarked and waiting for lean's merge, and stops in 7 files against #9375. READ (OLDFOUR's row and run). Not carried. |
| #8177 | Still wanted; carried | `cursor/carry-8177-dc90` at 92579572ddcb. Suites `r20261112-185823-bcfb`: 61 of 61 passed. `lean try` `r20261112-190057-99ec`: rc 0. Records `r20261112-185903-4d95`: every audit PASS, no record kept. RAN. |
| #8363 | Still wanted; proofs decides | Not carried, and no restack until proofs answers. |
| #8406 | Superseded by #9208 | Landed as 715b45913f. RAN. |
| #8414 | On `main` via #8539 | 4adc9162a5. One unused `member_refs` import is left over. RAN. |
| #8438 | Item 3 still wanted; carried | Items 1 and 2 are superseded by #8520. `cursor/carry-8438-dc90` at 073540b2806c holds item 3's 5 files. Suites `r20261112-193149-86cf`: 55 of 55 passed. RAN. |
| #8462 | Still wanted; proofs decides | Question: does S still come from the certifier card? Not carried. |
| #8543 | Still wanted; proofs decides | Needs a proof redo like #9208. Stopped; the files are `SpotFlock/Code.lean`, `Sessions.lean`, TwoGuard, `test_guard.py` and `artifacts.json`. RAN (trial merge). |
| #8563 | Still wanted; carried | `cursor/carry-8563-dc90` at c67169ef6275, 2 test files. Suites `r20261112-190809-3ca0`: 60 of 60 passed. RAN. |
| #8578 | Still wanted; carried | `cursor/carry-8578-dc90` at 07d875235188. It includes a test-only fix: `main`'s new a5 stand-in value had no `bcs` for #8578's committer to read. Suites `r20261112-192108-36b5`: 81 of 82 passed before the fix; rerun `r20261112-210942-7030`: 4 of 4 passed. Question for proofs: should every registered value be required to carry `bcs`? RAN. |
| #8731 | Still wanted; carried | `cursor/carry-8731-dc90` at e0c95da40383. It crossed the moves through #8448's own crossings. Suites `r20261112-204006-fd1c`: 67 of 67 passed. Opt-in tests `r20261112-204034-6645`: both passed. `lean try` `r20261112-194728-1fd9`: lean exit 0, "no theorems in it". Records `77c3`, `7845` and `1760` all PASS; the kept records are committed, and only hashes change. RAN. |
| #8767 | Still wanted; carried | `cursor/carry-8767-dc90` at 093ba58b7913, 8 files. Suites `r20261112-191633-c65b`: 63 of 63 passed. RAN. |
| #8869 | Superseded by #8925 | d56e145090, `StopRule16_v2` at T=1024. RAN. |
| #8900 | On `main` via #8873 | d8761c5827. RAN. |
| #8950 | Superseded in part by #9112; proofs decides | #9112 (1f64540734) designs by template because the verifier's rule can't state width strata. That makes #8950's by-width plans no-ops: its own test fails at `test_headline.py:307`, `merged` "template". `cursor/carry-8950-dc90` at 611f0327c7ed holds the merge only; do not take it. Suites `r20261112-192338-9a48`: 5 of 6 passed. Proofs should decide whether a held-κ, by-template row is wanted. RAN. |
| #9023 | Still wanted; held by RELAX | Not carried, per the coordinator. |
| #9128 | Superseded by #9075 | #9128's head is an ancestor of #9075's. RAN (`git merge-base --is-ancestor`). |

Summary: six carried with green suites (#8177, #8438, #8563, #8578, #8731, #8767). One carry stopped (#8950). Five wait on proofs (#7850, #8363, #8462, #8543, and #8950's remainder). Two are held by proofs' workers (#8139 by OLDFOUR, #9023 by RELAX). Seven are already on `main` or superseded (#7952, #8120, #8406, #8414, #8869, #8900, #9128).

I closed, marked and commented on nothing, and pushed only to my own `cursor/carry-*-dc90` branches.

