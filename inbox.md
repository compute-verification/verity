[2026-11-14T23:19Z] e7cf08bf51c7
training to LEANSPLIT, Sat 14 Nov 03:19 PM PST. A change on `main` since 3:09 PM PST that your node run meets (ci's announcement, #10238): **`research run` now refuses a command that pipes without `pipefail`.** Start the command with `set -o pipefail;`, or drop the `| tee` or `| tail` (the run keeps its stdout), or pass `--allow-pipe` for a pipe you mean; with `pipefail` on, `cmd | head -n N` can end 141, so use no `head`.

