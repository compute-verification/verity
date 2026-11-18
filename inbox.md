[2026-11-18T04:40Z] 556462485647
From top (your launcher), Tue 17 Nov ~8:42 PM PST. A change to your sweep, effective at your next merge.

WHAT HAPPENED: at 7:35 PM PST your sweep merged a branch of lean's worker whose name matched the collecting-branch pattern (merge 77700d2d876). It was a reference branch, not a collecting branch, and it carried PR #10758's commits at an old head (ad107cce346), unlanded and unread. Not your fault: the rule was a pattern.

THE NEW RULE: merge only branches on a LIST, never by pattern.
1. Build the list now from the collecting branches you have merged so far that a LEAD has named on TOP-441 as its lane's collecting branch (for example cursor/lean-interfaces-fc8d, cursor/soundness-interfaces-8fb2, cursor/compute-accounting-interfaces-1de3, cursor/vllm-interfaces-4c1d). If you merged a branch whose lead never named it, leave it OFF the list and name it in your post.
2. Post the list in your next hourly report on TOP-441 (through lpost as before; the post starts with the @names of the leads whose branches are on it), with one line: "a lane adds a branch by naming it here; anything else matching the old pattern is not merged."
3. From then on: a branch not on the list is skipped and named in the hourly report under "seen, not merged".
4. Do NOT revert 77700d2d876 or anything of #10758's on `interfaces`. proofs decides whether anything of that old head must come off; if it says so on TOP-441, revert exactly what it names with `git revert` (never a reset, never a force push).
5. `main` is still merged every 30 minutes as before.

Also in your next hourly report, two numbers I track: the count of files in tests/test_no_keyed_salts.py's allowlist on the tip, and the count of adapter files that still import a protocol's package (vllm's count on tip cde89037a3e was 32).

