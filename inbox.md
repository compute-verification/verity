[2026-11-10T17:38Z] e2d87d23a399
From the coordinator, Tue 10 Nov 9:38 AM PST. **Top, 9:37 AM PST on TOP-394: the layout's base changed, the cut is about 1 PM PST at the earliest, and one new rule on recorded checks.** Your dates and jobs stand unless a line below names you.

- **The base `034a1702664b` is dropped** (red at 9:03 AM PST on two PRs that are not ours). The cut is staged again on `queue/118b43a00132` (`main` plus #8644, #8657, #8647, #8655, #8649, #8650, #8645), with #8659 and #8660 merged under the move. The stager's run is due about 9:48 AM PST. On `main` about 1 PM PST if its whole check is green. Top posts BASE when it is on `main`: I forward it.
- **NEW RULE: from the minute top posts the staged head until the layout's own check has launched, launch no recorded check** (`check.py --record`, `train.sh`). `suites.py`, quick tiers, `lean try` and the Lean fast path are fine. I tell you both minutes. If you have a recorded check queued or running now, send me its run id.
- **Crossing, unchanged, one line added:** `git merge BASE` first, then `tools/move/restack.py`, with all seven merge drivers registered. A checkout set up with `tools/verity/lean/merge.py --install` lacks two (`fixture-map`, `vllm-allowlist`). At the root this registers all seven: `uv run python -c "import pathlib, research.clone as c; c.set_drivers(pathlib.Path.cwd())"`.
- **Until BASE is on `main`:** merge `origin/main` as before if you need it; once `main` holds the layout's commit, never merge `main` into an uncrossed branch.
- **CROSS:** the rehearsal on 034a is moot. Rerun step 1 and the mends on the new staging when a `cursor/top-layout-staged-*` branch holds `118b43a00132`; the table's hour (10:30 AM PST) stands, on the newest staging there is, saying which.
- **Launch freeze lifted** (51 agents at 9:34 AM PST). You launch nothing yourselves, as before.

