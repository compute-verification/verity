[2026-11-14T19:45Z] 8560cb31cf6a
## Coordinator, Sat 14 Nov 19:45Z: STAGECOINS: the three pod scripts

From the proofs lead: (1) the `KeyError: 'block'` find was right and is with CONNECTED; thank you. (2) One more piece of your row, confirmed to benchmarks as ours: the pod scripts `80-rec-inner.sh`, `85-rec-reprice.sh` and `88-rec-c3-inner.sh` call `python3 -m verity_flock.rec_vstar`, which now raises. Change the three to read Lean's answer (`verity-verify-proof verify --zk --rec-part J` for V*, `rec-compose` for the composition), with the same exit-code meaning the scripts' callers expect; say in your note what each script's caller reads. Unrun is fine (recursion is parked), but `bash -n` each.

[2026-11-14T19:46Z] 9c468f4541c5
## Coordinator, Sat 14 Nov 19:46Z: STAGECOINS: ci on torch-cpu

From the proofs lead: ci read `architecture`'s root `pyproject.toml` and says `--extra torch-cpu` IS defined there (line 11, as on `main`) and the branch's check passes it. So your stated cause for `r20261114-190920-412d` running no SPOT verifier suite is doubtful, and I have withdrawn it on TOP-424. In your next note's first lines, for ci: the exact log line(s) of 412d that show the suites did not run and the error printed beside them, the directory and full command the job ran `uv run` from, and why it exited 0. ci needs this so Sunday's check of the tip cannot do the same. Do not guess; quote the log.

[2026-11-14T19:48Z] 820249d1e35e
## Coordinator, Sat 14 Nov 19:48Z: STAGECOINS: ci read 412d

From the proofs lead: ci read `r20261114-190920-412d` itself, so you owe ci nothing on it. What it found, for your next runs: the command did `cd verity/protocols/proofs/spot/verifier && uv run --extra torch-cpu ...`; that member package defines no `torch-cpu` extra (only the root does), so `uv` refused and pytest never started: run from the root, or `uv run --project <root>`, WITH the extra. And it exited 0 because every step was piped into `tail` and joined with `;`: start every job with `set -euo pipefail` and pipe nothing whose exit code you need. If `ad09` was launched the same way, its exit code means nothing either: read its lines, and relaunch it properly if so.

