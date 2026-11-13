# CARRIER's scripts (bc-770979f8-f27e-5abe-b9a5-a86b237c9257)

Scratch, never to merge: the scripts of the carry of the proofs lane's refused pull requests over chain 2.

- `mkpass.py`: FRAME's job writer (the lane's note `20261113T1510Z-report-frame-pass-jobs`), with the launching checkout's git
  identity named at the launch instead of a made-up one, and the branch after the last move's script kept in the bundle
  where the tool stops at its last merge.
- `after.py`: FRAME's reader of a job's results and bundle (the lane's note `20261113T1605Z-report-frame-the-pass`).
- `words_check.py`: the words check of a carried head, with the queue's own reading of each move's map.
- `entry.sh`: appends one entry to `inbox/proofs-coordinator.md` on `cursor/verity-proofs-84b5` as one commit of that file.
- `job-N.sh`: the jobs as launched; `heads.sh`: the pull requests' heads.
