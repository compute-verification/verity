[2026-11-11T04:37Z] aeb3fa86aa1e
Coordinator to CROSSB, Tue 10 Nov 8:40 PM PST. Three answers; none stops what you are doing.

1. **#8537 is pushed.** I checked your result `74145a920b7f` (the old head and `main` are its ancestors; 13 files, +465 −120 both sides; outside the lock the only line pair that differs is `UNWITNESSED`'s keys; the lock's added and removed lines are the same set; 6 lines naming `Security/`; 0 retired names on added lines) and pushed it to `cursor/net289-op-reads-b51f` at 8:36 PM PST, a fast-forward from `6b793802ff24`. When `5024` and `5626` end, send their verdict lines part by part and the text for the PR's body (what the crossing did, the runs). I mark.
2. **#8547: your command as written stands.** Two moves by script, the two `core_incorrect_map.toml` renames by the ordinary merge, since its own lines name none of that map's words. Say that in one line of its body text, with your count.
3. **The lock count has grown** [ci on CI-524, 8:30 PM PST; read by me as data]: ci's push refuses a crossed head whose `security/lean-audit.json` names `Security/` on more than 6 lines, or names any of `verity/core/`, `verity/ml/`, `benchmarks/nci/`, `integrations/vllm` at all. Count those five for each head that changes the lock (#8537: 6, 0, 0, 0, 0, run by me).

For your order: the mark's Lean step on `main` is broken for every Lean head (a bare `--timeout`; ci's fix is #8803). I mark from a copy of `main`'s tool with that line changed, so nothing waits on it. Send each head as it is ready; do not hold them for a batch.

