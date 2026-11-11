[2026-11-11T04:25Z] 23bb8c906421
From CROSS (NET-311 job 4), Tue 10 Nov 8:25 PM PST. **#7109 is pushed: `cursor/net169-tally-061a` is at `dbf5daaa36e111cefc239b88f6b7715fd1c2f495` since 8:22 PM PST, a fast-forward from `53f2a6bfd010`. Its two runs are launched.**

- Queue status at 8:21 PM PST read "not marked ready at 53f2a6bf" [run].
- Quick tier: `r20261111-042308-64fe` (58 suites). The lock's records, your command with `--source` the worktree at the head: `r20261111-042303-0a73`. Both have `source_sha` `dbf5daaa36e1` and were waiting for a slot at 8:24 PM PST [run: `job.json`]. Two runs for this head, no more.

**Text for #7109's body:**

> Crossed onto the layout move (`main` `5c095bb4ab70`), three steps over `53f2a6bfd010`:
> - `fb8af76df2bf`: a merge of `main` `a930252c4993` on the old paths. One file is resolved by hand, `tools/verity/check/public_boundary.toml`: both sides append a `[[needed]]` entry at the same place, and both are kept, main's two and then this change's one.
> - `68d86350489e`: the result of `tools/move/restack.py` on that merge over both moves (`6581bcf197`, `c1ca25a247`): `clean`.
> - `dbf5daaa36e1`: `security/lean-audit.json` only, 344 lines. The move's script rewrites no file over 4,194,304 bytes, and this change's lock is 4,206,183, so the crossing left the paths from before the move in it. This commit rewrites them by the moves' own maps: 340 lines become `main`'s lines, 4 are this change's frozen reasons. The rule, applied to `main`'s lock from before the first move, gives `main`'s lock after the second byte for byte. Against `main` the lock is +499 −30, this change's own, as before the crossing.
> - `public_equal.py fb8af76df2bf dbf5daaa36e1`: `public_equal: EQUAL: 0 became public, 0 became private`.

Runs in flight, all mine: `0481` (#8499; 12 suites passed, none failed at 8:24 PM PST), `dcf5` (#8708; 28 passed, none failed), `64fe` and `0a73` (#7109). Job 3's three ended exit 0: `857d` 54 suites passed, `9a68` 58, `9862` 58 [run: `research fetch`]. Next: the note's section and the one comment on NET-311, then I end my turn on the four runs.

[2026-11-11T04:29Z] b5087f0ec2a5
CROSSB, NET-311, setup and where `restack.py`'s docstring and --help differ from the brief. Tue 10 Nov, 8:30 PM PST.
1. origin/main is 5c095bb4ab7012ba359e0a85f4ecf19892b99629; all six branches are at the table's heads [run: fetch 8:16 PM]. `uv sync` done; the clone defines every merge driver `.gitattributes` names [run: `restack.missing_drivers()` gives none].
2. Moves: with no `--move-commit` the tool takes every move origin/main has and the branch lacks, by its move line. For five heads that is your two. For #8547 it is four [run: `restack.moves_on`]: c6a38b702d28 and 9776ae330bca (both `Move: tools/move/core_incorrect_map.toml`, rename.py) before the two layout moves. Your command names two, so c6a38b702d28 reaches #8547 by step 1's ordinary merge of b5354b9c2344, not by its script. #8547's own diff names none of that map's words (`RefineWrong`, `mem_wrong`, `exists_wrong_fine`, `card_wrong_le`: 0 lines [run]), so I ran your command as written; say so if you want it crossed by the default instead.
3. `--push-to` pushes a `locks-stale` result too (docstring; the brief lists it under the outcomes to stop on). If one comes it will be on my branch, unresolved, and I report it.
4. `--json` prints no outcome line: each entry gives `restack.line()` of the JSON, the tool's own function.
5. The README's command is `restack.py BRANCH --push-to NEW --kept`: `--onto origin/main` is its default, and `--no-pins` skips its step 4 (I check the pins per head instead).
6. Suites runs take `--changed` as origin/main's sha (the node may not have the ref), as the testing skill's pre-mark run does.

