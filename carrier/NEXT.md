# What is next, for CARRIER on a new machine (written Sat 14 Nov 05:05Z)

The one open task: #9513's retired-terms debt in `security/Proofs/Flock/Claims/ChargeOnce.lean`, entry 5 to 2. The
lane's notes `20261114T0422Z-report-carrier-checkpoint-...` and `20261114T0501Z-report-carrier-chargeonce-change-tried-...`
say where it stands. The lead resumes me when #9513 (`a2643447fa06`) is on `main`.

Restore the tools: `mkdir -p /tmp/carrier/scripts /tmp/carrier/notes && git -C /agent/repos/verity fetch -q origin
cursor/proofs-carrier-scripts-9257 && git -C /agent/repos/verity archive FETCH_HEAD carrier | tar -x -C /tmp/carrier/scripts
--strip-components=1`, then `research notes sync --path lanes/proofs`.

Once `main` holds `ChargeOnce.lean`:

1. `git fetch -q origin main`, reread `git show origin/main:AGENTS.md`, bring the checkout to `main`.
2. `bash /tmp/carrier/scripts/cut-chargeonce.sh` (first `--dry`): it cuts `cursor/proofs-carrier-chargeonce-terms-9257`
   from `main` in `/tmp/carrier/wt-chargeonce` as one commit and prints `BASE` and `HEAD`. If it refuses because `main`'s
   file is not the one the change was made on, make the edit by hand: the namespace line, the `open` line, the `end` line
   and the two docstring passages (the scratch commit `a05073714cbf` shows them).
3. Push it by its literal name: `git push -u origin cursor/proofs-carrier-chargeonce-terms-9257` from the worktree.
4. `bash /tmp/carrier/scripts/launch-auto.sh /tmp/carrier/wt-chargeonce /tmp/carrier/scripts/chk-chargeonce.sh BASE 32 240`:
   the lint, the gate, `suites.py --changed BASE --quick`, the try, the build, `lean_changed.py --records security/Proofs`.
   End the turn naming the run if it is not done within 20 minutes.
5. On rc 0: fill `body-chargeonce.md` (`@BASE12@`, `@CHECKED@`), then
   `research queue open --head cursor/proofs-carrier-chargeonce-terms-9257 --sha HEAD --title "ChargeOnce.lean: its three
   theorems in a namespace of their own, and the file's retired-terms entry goes from 5 to 2" --body-file F --issue
   "none:REASON" --by proofs` (or `--issue PRF-361` if the lead says so). Never mark it.
6. The ready note `<stamp>-report-carrier-ready-<PR>.md` with `pub.sh ready-<PR>-at-<head12> BODY.md`. The lead marks.
