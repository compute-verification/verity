[2026-11-18T05:33Z] d2ab4c3d943e
From the proofs lead, a rule from a fault of ours an hour ago: a job NEVER writes into its shipped source tree (`--cwd source`): runs of the same commit on one machine SHARE that tree, so a patch, a reverse-apply, a `git checkout` of a file or a generated file written there is read by every other lane's run. If a job must modify the tree (to compare with and without a change, to regenerate a file), it first copies the tree to a `mktemp -d` directory outside the run directory and works there (the writing-runs skill), or uses `--cwd clone`.

