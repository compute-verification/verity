# Mark to `main`

How long a PR waits from its ready mark to `main`: for each PR the verified queue landed on `main` in a window, the hours
from its ready mark at the head that landed to the move of `main` that holds it. The PRs are split as the queue splits its
batches, plain and Lean, and each class gives four measurements:

| Measurement | Unit |
| --- | --- |
| `ci/plain-mark-to-main-p90`, `ci/lean-mark-to-main-p90` | hours, the 90th percentile, with the target of 2 h |
| `ci/plain-mark-to-main-median`, `ci/lean-mark-to-main-median` | hours |
| `ci/plain-landed`, `ci/lean-landed` | the PRs counted |
| `ci/plain-within-2h`, `ci/lean-within-2h` | the share of them on `main` within 2 h of the mark |

The two p90s are the metrics of the goals that a plain PR, and one that changes Lean, lands within two hours of its mark
(`benchmarks/metrics.toml`). `landing.json`, beside the result, holds every landing's row: the PR, its head, the `main` it
landed on, its mark, its hours and its class.

- **The window**: the 7 days before the run (`--days N`, `--until TIME` for another).
- **A landing** is an `advance` of `main` that the loop's decision log on the control pod records as done and not the
  shadow's, at its time; a PR is counted once a head, at its first landing.
- **The mark** is the `ready` label in force when `main` moved, at the head that landed or at a head a restack across a move
  made it from: the rule `research queue stats` counts mark to `main` by.
- **The split** is the loop's own: a PR is Lean when its change is a Lean change that touches a Lean source, not a lock
  alone (`vqueue.Ready.lean`), so it goes in a Lean batch; every other PR is plain, a change to `security/`'s proofs alone
  among them.
- **Out of the count**: a PR landed by hand, or by drive's old path before the loop moved `main`, since neither is an
  advance in the loop's log; and a landed PR with no mark in force when it landed, which `landing.json` lists. Percentiles
  interpolate between the two nearest values, as `research queue stats` does.

Every input is read through the queue's own code (`landing.py`'s docstring names each). A run that can't read one, the
decision log, a label pull, a head's restack lookup or its split, writes no measurement and fails its validation.

## Run it

```sh
research run --project verity --cwd source -- bash benchmarks/landing/run.sh
```

It runs as a local research run, from a checkout on an agent VM with the team's keys: it reads the control pod
(`research pods ssh vy-control-verity`), GitHub (`gh`, for restacks) and the evidence store's labels, and a CPU node reaches
none of the three. It reads for about a minute and computes nothing heavy.

`uv run tools/verity/check/suites.py benchmarks/landing` runs its tests: the computation on a small fixture, with stand-ins
for the pod, GitHub, the store and git.
