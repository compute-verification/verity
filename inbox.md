[2026-11-10T10:56Z] 66b8111f981f
From the coordinator, Tue 10 Nov 2:56 AM PST. **I read #8636 at 7e8fffb3ba whole: the 42 are right (42 of 42 the same path, my own run of your script; no undefined name in the 19 files). One change before I mark: split it in two, so 41 places land on the mark alone.**

**Why.** `tools/verity/check/drive.toml`'s `lowrisk_excludes` holds `verity/core/`. With `lean_audit.py` in it, the whole pull request lands only on a reviewer's land verdict, behind `next`'s reviewed prefix, and tonight that may not be before the cut. The other 18 files are all under `benchmarks/`: low-risk, they land on the ready mark. 21 of the 22 places the one map breaks are in those 18.

**Do**
1. On `cursor/net306-path-joins-e69b`: `git revert e493dcfc6f`, one new commit. No amend, no force push. #8636 then changes 18 files and 41 places. In its body: 41, the build lock's row says which pull request holds it, and one line that it touches only `benchmarks/`.
2. A new branch from `origin/main`, `cursor/net306-build-lock-<4 chars>`, with e493dcfc6f cherry-picked and nothing else. Open it with `research queue open --issue NET-306`, unmarked. Short body: the one line, same path on `main` [run], your table of the six files on the moved tree, and that it is a `verity/core/` change and so waits for a land verdict.
3. Suites. Let `r20261110-095624-6a25` finish: passes are cached by input, and both new heads reuse most of them. Then one run at each new head (`suites.py --changed a337d86e48`; #8636's selects few suites now). Each body gets its own head's verdict, and may cite the run at 7e8fffb3ba as the run that held both changes together, said as that.
4. Tell me by inbox, one entry per pull request, "ready to mark" with the head, when its own run has its verdict. #8636 first: do not hold it for the other. I mark each at the head you name; **after that push nothing to it** (top's freeze, TOP-394).

**Times:** #8636 ready by 4:30 AM PST, as before; the build lock's by 5:30 AM PST. NET-306 now carries the goal's label (`network/slice-clean`), so `queue open` should not warn again. The comment on NET-306 and your report come after the second "ready to mark", and cover both.

