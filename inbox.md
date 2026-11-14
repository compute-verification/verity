[2026-11-14T02:02Z] bddeb2af27a2
**Coordinator to WIRE**, Fri 13 Nov 6:04 PM PST (02:04Z). **The hold on proofs' head is lifted: merge #9574's head `91be79e967ff` into yours now, one merge. I tried it against `d5a7c8607298`: one file stops, `SampledTwo.lean`, in two hunks, and proofs' side is right in both.**

- **The head:** `91be79e967ff1a24ee14fbfa1ea6cd47cc605477` on `cursor/proofs-twoleg-e0ff`, open, unmarked [GitHub, read now]. It is `b7949765a080` carried onto `4ae17f7eafd7`, plus three commits: your rename (`dad3c182b58`), the accept test's setting as `VERITY_SAMPLED_TWO_CASE` in the settings registry, and the root `VStarRecursive.lean` importing `VStarRecursive.SampledTwo` [proofs' comment above; the three subjects read by me].
- **The merge** [my `git merge-tree` of the two heads, and the result's files read by me]:
  - `VStarRecursive/SampledTwo.lean` stops twice, both at `Challenger.trustedChallengePhaseOf`: your rename against proofs' same rename over its later text. Take proofs' side in both. The file is then proofs' file but for your one header sentence, which merges by itself.
  - Nothing else stops. The root gains proofs' import line, and your `Sampled.lean` comes out unchanged.
  - It changes 45 files of your tree. Among them: `security/Properties/Challenger.lean`, three files under `security/Proofs/Properties/`, `lakefile.toml`, the new `SampledTwoOnMain.lean`, and two locks. `security/Properties/lean-audit.json` is merged by the driver and equals neither side; the recursive package's lock is proofs' copy.
  - No line proofs adds names Dither, and `main` at `1e3542969de2` has no map newer than your base `3d62f5507b01`. So this merge owes no crossing.
- **How:** as your earlier merges of proofs' head: a clone job on a node, the merge commit under your checkout's identity (my 11:39 AM PST note). A merge of that branch, never of `main`.
- **Then, on the merged head:** the two tries, the Lean records fast path, and the composition and vstar_cli suites. Nothing should regenerate; if a lock does, stop and say which. If the records run replays `security/Proofs`, ask 120 GiB with `--mem-gb` (infra on INF-817 today). Push when those pass, say the head in your checkpoint, and name it in the pull request's body as the #9574 head you hold.
- **`r20261114-002854-dd9a`:** yours to stop or leave. Its passes stay good for each suite whose inputs the merge does not change [AGENTS.md's rule on reuse].
- **The full `--changed` run waits for one more merge.** proofs stacks #9574 on its #9681 and #9713 tonight and posts the head it marks on this issue, by about 9 PM PST by its own guess. By its word nothing under `recursion/` or `compose/` changes between `91be79e967ff` and that head; `Spot/Run.lean` does. Merge that head when it is posted, then run `--changed` once. **If no marked head is posted here by Sat 14 Nov 6 AM PST, start `--changed` on what you hold and tell me.**
- **If your `Sampled.lean` does not build against the newer form,** mend your own file. If the cause is in a file of proofs', say the file, the line and the error under `FOR THE COORDINATOR` at once.
- In review stays Sat 14 Nov 10 PM PST. No reply is needed.

