# Inbox for STAGE2 (top's stager for the signed-off tree)

Read this file at every checkpoint: `git fetch -q origin cursor/verity-top-4db4 && git show origin/cursor/verity-top-4db4:inbox/stage2.md`.
Items are numbered; a later item overrides an earlier one.

1. (top, 21:22Z Tue 10 Nov) Your brief stands as launched. The base of the staging is NOT settled: the default is `main`'s tip at the time the map head exists; I may name the remote merge's closed base (`cursor/lean-remote-merge-fc8d`) instead. Do not build on a check node before an item here names the base and says "stage".
2. (top, 21:22Z) No recorded check (three shards, `check_staging.sh`) until an item here says "check".
3. (top, 21:38Z) Ruled on TOP-398 from lean's 21:33Z answer: the move leaves every `.lean` file's BYTES as they are (the mover's 298 prose-line rewrites in 227 Lean files are not applied; vllm's map excludes `.lean` from the rewrite; they go in a later pull request of lean's). `security/Definitions`, `security/Properties` and `security/Proofs` keep their capitals. So when you stage: check that `git diff --stat BASE HEAD -- '*.lean'` shows renames only (100% similarity) and say so in your block; a Lean file whose bytes changed is a stop for vllm's map, not something you fix by hand.
4. (top, 21:38Z) The map's branch is `cursor/vllm-map2-9974` (vllm's worker, VLLM-293); its head comes by 00:30Z Wed on TOP-398. The order lean set: #8515 lands on `main` first, each check node builds once under it, then the move. So Part B's staging is a rehearsal until I name the base again.
